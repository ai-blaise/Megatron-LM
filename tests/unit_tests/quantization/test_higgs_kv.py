# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the 2-bit HIGGS dense MLA-latent KV fake-quant.

Covers:
  * codec construction (EDEN2-16 codebook reproducibility, slot bytes)
  * pack / unpack indices round-trip
  * forward parity with the SGLang reference at high cos-sim
  * forward shape preservation across ``[s, b, d]`` vs ``[s*b, d]``
  * backward matches torch.autograd on the STE-detach forward

The tests live in float64 on CPU so they run without GPU access. The CUDA
kernel correctness check lives in the same file but is gated on CUDA
availability.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from megatron.core.quantization.higgs import (  # noqa: E402
    HIGGS_CODEBOOK_SIZE,
    HIGGS_EDEN2_16,
    HIGGS_LATENT_DIM,
    HIGGS_NUM_PAIRS,
    HIGGS_PACKED_BYTES,
    HIGGS_PAIR_DIM,
    HIGGS_SLOT_BYTES,
    apply_higgs_dense_2bit_kv,
    build_higgs_buffers,
    pack_higgs_2bit_indices,
    unpack_higgs_2bit_indices,
)
from megatron.core.quantization.higgs.reference import (  # noqa: E402
    fwht,
    higgs_backward,
    higgs_forward,
    reference_compress,
    reference_decompress,
)


def _make_buffers(dtype=torch.float64, layer_idx=0):
    return build_higgs_buffers(
        latent_dim=HIGGS_LATENT_DIM,
        preset="dense_2bit",
        layer_idx=layer_idx,
        device="cpu",
        dtype=dtype,
    )


def test_buffers_reproducibility():
    a = _make_buffers(layer_idx=0)
    b = _make_buffers(layer_idx=0)
    torch.testing.assert_close(a.codebook, b.codebook, rtol=0, atol=0)
    torch.testing.assert_close(a.codebook_norm_sq, b.codebook_norm_sq, rtol=0, atol=0)


def test_buffers_layer_invariant():
    """EDEN2-16 has no per-layer randomness; buffers are layer-invariant."""
    a = _make_buffers(layer_idx=0)
    b = _make_buffers(layer_idx=11)
    torch.testing.assert_close(a.codebook, b.codebook, rtol=0, atol=0)


def test_codebook_matches_eden2_16():
    buf = _make_buffers(dtype=torch.float32)
    expected = torch.tensor(HIGGS_EDEN2_16, dtype=torch.float32)
    torch.testing.assert_close(buf.codebook, expected, rtol=0, atol=0)
    expected_norm_sq = (expected * expected).sum(dim=-1)
    torch.testing.assert_close(buf.codebook_norm_sq, expected_norm_sq)


def test_slot_layout_constants():
    assert HIGGS_PAIR_DIM == 2
    assert HIGGS_CODEBOOK_SIZE == 16
    assert HIGGS_NUM_PAIRS == HIGGS_LATENT_DIM // HIGGS_PAIR_DIM == 256
    assert HIGGS_PACKED_BYTES == HIGGS_NUM_PAIRS // 2 == 128
    assert HIGGS_SLOT_BYTES == 258


def test_pack_unpack_round_trip():
    indices = torch.randint(0, 16, (5, HIGGS_NUM_PAIRS), dtype=torch.uint8)
    packed = pack_higgs_2bit_indices(indices)
    assert packed.shape == (5, HIGGS_PACKED_BYTES)
    assert torch.equal(unpack_higgs_2bit_indices(packed, HIGGS_NUM_PAIRS), indices)


def test_fwht_self_inverse():
    torch.manual_seed(0)
    x = torch.randn(8, HIGGS_LATENT_DIM, dtype=torch.float64)
    torch.testing.assert_close(fwht(fwht(x)), x, rtol=1e-9, atol=1e-9)


def test_forward_shape_and_finite():
    buf = _make_buffers()
    torch.manual_seed(0)
    x = torch.randn(16, HIGGS_LATENT_DIM, dtype=torch.float64)
    y = higgs_forward(x, buf)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_quantization_cos_sim_above_threshold():
    """Cos sim of unit-norm latent vs HIGGS round-trip is comfortably high.

    Reference acceptance gate from the OP test
    (``test_higgs_dense_2bit_kv.py::test_higgs_codec_round_trip_preserves_rope``)
    is ``min cos_sim > 0.85`` on random bf16 Gaussian inputs at the input
    distribution the EDEN2-16 codebook is calibrated for. We assert the
    same threshold here for fp64 inputs; the mean is typically ~0.94.
    """

    buf = _make_buffers()
    torch.manual_seed(0)
    x = torch.randn(64, HIGGS_LATENT_DIM, dtype=torch.float64)
    x = x / x.norm(dim=-1, keepdim=True)
    y = higgs_forward(x, buf)
    cos = torch.nn.functional.cosine_similarity(y, x, dim=-1)
    assert cos.min().item() > 0.85, f"min cos = {cos.min().item()}"
    assert cos.mean().item() > 0.90, f"mean cos = {cos.mean().item()}"


def test_reference_codec_round_trip_preserves_rope():
    """Mirrors the OP test ``test_higgs_codec_round_trip_preserves_rope``."""

    buf = _make_buffers(dtype=torch.float32)
    torch.manual_seed(0)
    n = 8
    latent = torch.randn(n, 1, HIGGS_LATENT_DIM, dtype=torch.bfloat16)
    rope = torch.randn(n, 1, 64, dtype=torch.bfloat16)
    compressed = reference_compress(latent, rope, buf)
    assert compressed.shape == (n, 1, HIGGS_SLOT_BYTES)
    restored = reference_decompress(compressed, buf, torch.bfloat16)
    # rope must round-trip exactly (bf16 -> bf16 is the identity).
    assert torch.equal(restored[..., HIGGS_LATENT_DIM:], rope)
    cos = torch.nn.functional.cosine_similarity(
        restored[..., :HIGGS_LATENT_DIM].float().reshape(n, HIGGS_LATENT_DIM),
        latent.float().reshape(n, HIGGS_LATENT_DIM),
        dim=-1,
    )
    assert torch.all(cos > 0.85), f"min cos_sim = {cos.min().item()}"


def _ste_autograd_forward(x: torch.Tensor, buf):
    """STE-detach reference forward.

    ``(quantized - normalized * mask).detach() + normalized * mask`` routes
    upstream gradients through ``normalized`` with the saturation mask while
    the forward output equals the codebook value (mirrors the kernel).
    Used as the gradient oracle for the analytic backward.
    """

    rotated = fwht(x)
    rot_norm = torch.linalg.vector_norm(rotated, dim=-1).clamp_min(1e-8)
    scale = rot_norm / math.sqrt(buf.latent_dim)
    normalized = rotated / scale[:, None]
    pairs = normalized.reshape(-1, buf.num_pairs, HIGGS_PAIR_DIM)
    scores = 2.0 * torch.matmul(pairs, buf.codebook.T) - buf.codebook_norm_sq
    indices = torch.argmax(scores, dim=-1)
    values = buf.codebook[indices].reshape(normalized.shape)

    mask = torch.ones_like(normalized)
    ste_recon_unit = (values - normalized * mask).detach() + normalized * mask
    rotated_recon = ste_recon_unit * scale[:, None]
    return fwht(rotated_recon)


def test_backward_matches_torch_autograd():
    """Analytic backward reproduces torch.autograd on the STE-detach forward."""

    buf = _make_buffers()
    torch.manual_seed(0)
    x = torch.randn(4, HIGGS_LATENT_DIM, dtype=torch.float64)
    upstream = torch.randn_like(x)

    x_ad = x.clone().requires_grad_(True)
    y_ad = _ste_autograd_forward(x_ad, buf)
    (y_ad * upstream).sum().backward()
    g_torch = x_ad.grad

    y_ref, intermediates = higgs_forward(x, buf, return_intermediates=True)
    g_analytic = higgs_backward(upstream, intermediates, buf)

    torch.testing.assert_close(y_ad.detach(), y_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(g_torch, g_analytic, rtol=1e-6, atol=1e-6)


def test_autograd_function_shape_preserves():
    buf = _make_buffers(dtype=torch.float32)
    torch.manual_seed(0)
    for shape in [
        (HIGGS_LATENT_DIM,),
        (4, HIGGS_LATENT_DIM),
        (3, 2, HIGGS_LATENT_DIM),
        (2, 5, 7, HIGGS_LATENT_DIM),
    ]:
        x = torch.randn(*shape, requires_grad=True)
        y = apply_higgs_dense_2bit_kv(x, buf)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


def test_autograd_function_grad_flows():
    buf = _make_buffers(dtype=torch.float32)
    torch.manual_seed(0)
    x = torch.randn(8, HIGGS_LATENT_DIM, requires_grad=True)
    y = apply_higgs_dense_2bit_kv(x, buf)
    upstream = torch.randn_like(y)
    (y * upstream).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0


def test_build_rejects_wrong_dim():
    with pytest.raises(ValueError):
        build_higgs_buffers(latent_dim=256, device="cpu")
    with pytest.raises(ValueError):
        build_higgs_buffers(latent_dim=HIGGS_LATENT_DIM, preset="bogus", device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_forward_matches_reference():
    """CUDA kernel forward matches the fp32 reference to cos_sim >= 0.999976.

    Threshold matches the OP test acceptance gate
    (``test_higgs_dense_2bit_kv.py::test_higgs_store_dequant_matches_reference``).
    """

    from megatron.core.quantization.higgs.kernels.build import get_ext

    try:
        ext = get_ext()
    except RuntimeError as e:
        pytest.skip(f"CUDA extension not buildable: {e}")
    assert hasattr(ext, "higgs_kv_fwd")

    buf_cpu = _make_buffers(dtype=torch.float32)
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    n = 32
    x = torch.randn(n, HIGGS_LATENT_DIM, dtype=torch.bfloat16, device=device)

    buf = build_higgs_buffers(
        latent_dim=HIGGS_LATENT_DIM, preset="dense_2bit", device=device, dtype=torch.float32
    )

    y_cuda = apply_higgs_dense_2bit_kv(x, buf).to(torch.float32)
    y_ref = higgs_forward(x.to(torch.float32).cpu(), buf_cpu).to(torch.float32).to(device)
    cos = torch.nn.functional.cosine_similarity(y_cuda, y_ref, dim=-1)
    assert cos.min().item() > 0.999976, f"min cos = {cos.min().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_backward_matches_reference():
    """CUDA backward matches the fp32 reference within fp32 reduction noise."""

    from megatron.core.quantization.higgs.kernels.build import get_ext

    try:
        ext = get_ext()
    except RuntimeError as e:
        pytest.skip(f"CUDA extension not buildable: {e}")
    assert hasattr(ext, "higgs_kv_bwd")

    device = torch.device("cuda:0")
    buf = build_higgs_buffers(
        latent_dim=HIGGS_LATENT_DIM, preset="dense_2bit", device=device, dtype=torch.float32
    )
    buf_cpu = _make_buffers(dtype=torch.float32)

    torch.manual_seed(1)
    n = 8
    x_cuda = torch.randn(
        n, HIGGS_LATENT_DIM, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    upstream = torch.randn_like(x_cuda)
    y_cuda = apply_higgs_dense_2bit_kv(x_cuda, buf)
    (y_cuda * upstream).sum().backward()
    grad_cuda = x_cuda.grad.to(torch.float32).cpu()

    x_ref = x_cuda.detach().to(torch.float32).cpu()
    upstream_ref = upstream.detach().to(torch.float32).cpu()
    _, inter = higgs_forward(x_ref, buf_cpu, return_intermediates=True)
    grad_ref = higgs_backward(upstream_ref, inter, buf_cpu)

    # bf16 round-trip on saved intermediates introduces small drift; allow
    # up to 1e-2 abs / 1e-2 rel, the same tolerance used for the TurboQuant
    # CUDA backward gradient comparison.
    torch.testing.assert_close(grad_cuda, grad_ref, rtol=1e-2, atol=1e-2)
