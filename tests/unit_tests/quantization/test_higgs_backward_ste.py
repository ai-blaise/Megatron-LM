# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Backward STE tests for the HIGGS dense 2-bit fake-quant.

Mirrors the pattern used by the NVFP4 IndexCache backward tests
(``test_indexcache.py`` after commit ``7e78f288``):

1. The analytic backward matches torch.autograd on the STE-detach forward at
   float64 precision (no kernel involved).
2. ``apply_higgs_dense_2bit_kv`` admits gradients on inputs with arbitrary
   leading dimensions, with the gradient shape matching the input shape.
3. The CUDA backward kernel produces gradients within bf16 reduction noise of
   the fp32 reference (gated on CUDA availability).
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
    HIGGS_LATENT_DIM,
    HIGGS_PAIR_DIM,
    apply_higgs_dense_2bit_kv,
    build_higgs_buffers,
)
from megatron.core.quantization.higgs.reference import (  # noqa: E402
    fwht,
    higgs_backward,
    higgs_forward,
)


def _make_buffers(dtype=torch.float64):
    return build_higgs_buffers(
        latent_dim=HIGGS_LATENT_DIM, preset="dense_2bit", device="cpu", dtype=dtype
    )


def _ste_autograd_forward(x: torch.Tensor, buf):
    """STE-detach reference forward (mirrors the kernel)."""

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


def test_ste_backward_matches_autograd_fp64():
    """Analytic closed form matches torch.autograd on the STE-detach forward."""

    buf = _make_buffers()
    torch.manual_seed(0)
    x = torch.randn(8, HIGGS_LATENT_DIM, dtype=torch.float64)
    upstream = torch.randn_like(x)

    x_ad = x.clone().requires_grad_(True)
    y_ad = _ste_autograd_forward(x_ad, buf)
    (y_ad * upstream).sum().backward()
    g_torch = x_ad.grad

    y_ref, intermediates = higgs_forward(x, buf, return_intermediates=True)
    g_analytic = higgs_backward(upstream, intermediates, buf)

    torch.testing.assert_close(y_ad.detach(), y_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(g_torch, g_analytic, rtol=1e-6, atol=1e-6)


def test_apply_backward_arbitrary_leading_dims():
    """Backward flows correctly through arbitrary input shapes."""

    buf = _make_buffers(dtype=torch.float32)
    torch.manual_seed(0)
    for shape in [
        (4, HIGGS_LATENT_DIM),
        (3, 2, HIGGS_LATENT_DIM),
        (2, 3, 5, HIGGS_LATENT_DIM),
    ]:
        x = torch.randn(*shape, requires_grad=True)
        y = apply_higgs_dense_2bit_kv(x, buf)
        upstream = torch.randn_like(y)
        (y * upstream).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()


def test_backward_gradient_is_nonzero():
    """The STE gradient on a random non-degenerate input is non-trivial."""

    buf = _make_buffers(dtype=torch.float32)
    torch.manual_seed(0)
    x = torch.randn(16, HIGGS_LATENT_DIM, requires_grad=True)
    y = apply_higgs_dense_2bit_kv(x, buf)
    upstream = torch.randn_like(y)
    (y * upstream).sum().backward()
    assert x.grad.abs().mean().item() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_backward_matches_reference_under_bf16_noise():
    """CUDA backward agrees with the fp32 reference up to bf16 round-trip noise."""

    from megatron.core.quantization.higgs.kernels.build import get_ext

    try:
        ext = get_ext()
    except RuntimeError as e:
        pytest.skip(f"CUDA extension not buildable: {e}")
    assert hasattr(ext, "higgs_kv_bwd")

    device = torch.device("cuda:0")
    buf = build_higgs_buffers(
        latent_dim=HIGGS_LATENT_DIM, preset="dense_2bit",
        device=device, dtype=torch.float32,
    )
    buf_cpu = _make_buffers(dtype=torch.float32)

    torch.manual_seed(0)
    n = 8
    x_cuda = torch.randn(
        n, HIGGS_LATENT_DIM, dtype=torch.bfloat16, device=device,
        requires_grad=True,
    )
    upstream = torch.randn_like(x_cuda)

    y_cuda = apply_higgs_dense_2bit_kv(x_cuda, buf)
    (y_cuda * upstream).sum().backward()
    grad_cuda = x_cuda.grad.to(torch.float32).cpu()

    x_ref = x_cuda.detach().to(torch.float32).cpu()
    upstream_ref = upstream.detach().to(torch.float32).cpu()
    _, inter = higgs_forward(x_ref, buf_cpu, return_intermediates=True)
    grad_ref = higgs_backward(upstream_ref, inter, buf_cpu)

    torch.testing.assert_close(grad_cuda, grad_ref, rtol=1e-2, atol=1e-2)
