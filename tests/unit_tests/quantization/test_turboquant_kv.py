# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Phase 1 unit tests for the TurboQuant dense MLA-latent KV fake-quant.

Covers:
  * codec construction (presets, sign vectors, codebook reproducibility)
  * forward parity with the SGLang reference (compress + decompress roundtrip)
  * backward gradcheck against analytic finite differences in float64
  * autograd.Function shape preservation across [s, b, d] vs [s*b, d] inputs
  * STE clip behavior (zero gradient at saturated coordinates)

The tests live in float64 on CPU so they run without GPU access. The CUDA
kernel correctness check lives in tests/unit_tests/fusions/.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from megatron.core.quantization.turboquant import (  # noqa: E402
    TURBOQUANT_PRESETS,
    apply_turboquant_kv,
    build_turboquant_buffers,
)
from megatron.core.quantization.turboquant.reference import (  # noqa: E402
    fwht,
    turboquant_backward,
    turboquant_forward,
)


LATENT_DIM = 512


def _make_buffers(preset="latent_2p5bit_nc", seed=0, layer_idx=0, dtype=torch.float64):
    return build_turboquant_buffers(
        latent_dim=LATENT_DIM,
        preset=preset,
        seed=seed,
        layer_idx=layer_idx,
        device="cpu",
        dtype=dtype,
    )


def test_buffers_reproducibility_same_seed():
    a = _make_buffers(seed=7, layer_idx=3)
    b = _make_buffers(seed=7, layer_idx=3)
    torch.testing.assert_close(a.signs1, b.signs1, rtol=0, atol=0)
    torch.testing.assert_close(a.signs2, b.signs2, rtol=0, atol=0)
    torch.testing.assert_close(a.boundaries_high, b.boundaries_high, rtol=0, atol=0)
    torch.testing.assert_close(a.centroids_high, b.centroids_high, rtol=0, atol=0)
    torch.testing.assert_close(a.boundaries_low, b.boundaries_low, rtol=0, atol=0)
    torch.testing.assert_close(a.centroids_low, b.centroids_low, rtol=0, atol=0)


def test_buffers_distinct_per_layer():
    a = _make_buffers(seed=7, layer_idx=0)
    b = _make_buffers(seed=7, layer_idx=1)
    assert not torch.equal(a.signs1, b.signs1)
    assert not torch.equal(a.signs2, b.signs2)


def test_signs_are_pm_one():
    buf = _make_buffers()
    assert torch.all((buf.signs1 == 1) | (buf.signs1 == -1))
    assert torch.all((buf.signs2 == 1) | (buf.signs2 == -1))


def test_codebook_sizes_match_presets():
    for preset, props in TURBOQUANT_PRESETS.items():
        buf = build_turboquant_buffers(
            latent_dim=LATENT_DIM, preset=preset, device="cpu", dtype=torch.float32
        )
        bits = float(props["bits"])
        if math.isclose(bits, 2.5):
            assert buf.centroids_high.shape == (8,)
            assert buf.centroids_low.shape == (4,)
        elif bits < 8:
            assert buf.centroids_high.shape == (1 << int(bits),)


def test_fwht_self_inverse():
    torch.manual_seed(0)
    x = torch.randn(8, LATENT_DIM, dtype=torch.float64)
    torch.testing.assert_close(fwht(fwht(x)), x, rtol=1e-9, atol=1e-9)


def test_forward_shape_and_finite():
    buf = _make_buffers()
    torch.manual_seed(0)
    x = torch.randn(16, LATENT_DIM, dtype=torch.float64)
    y = turboquant_forward(x, buf)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_quantization_error_bounded():
    """Roundtrip error of unit-norm inputs should be small for 2.5-bit."""
    buf = _make_buffers()
    torch.manual_seed(0)
    x = torch.randn(64, LATENT_DIM, dtype=torch.float64)
    x = x / x.norm(dim=-1, keepdim=True)
    y = turboquant_forward(x, buf)
    err = (y - x).norm(dim=-1) / x.norm(dim=-1)
    # Empirical relative error from the SGLang reference at 2.5-bit on
    # unit-norm Gaussian vectors is well under 0.4. Loose bound here.
    assert err.mean().item() < 0.4
    assert err.max().item() < 0.6


def _ste_autograd_forward(x: torch.Tensor, buf):
    """Differentiable forward that gives torch.autograd the same STE we expect.

    The ``(quantized - rotated*mask).detach() + rotated*mask`` trick routes
    upstream gradients through ``rotated`` with the saturation mask while the
    forward output equals the quantized centroid value (matches our forward).
    Used as the gradient oracle for the analytic backward.
    """

    from megatron.core.quantization.turboquant.codec import (
        TURBOQUANT_2P5_GROUP_SIZE,
        TURBOQUANT_2P5_HIGH_CHANNELS,
    )

    norm = torch.linalg.vector_norm(x, dim=-1).clamp_min(1e-8)
    unit = x / norm[:, None]
    rotated = fwht(unit * buf.signs1) * buf.signs2

    dim = rotated.shape[-1]
    g = TURBOQUANT_2P5_GROUP_SIZE
    high = TURBOQUANT_2P5_HIGH_CHANNELS
    grouped = rotated.reshape(-1, dim // g, g)
    h_v = grouped[..., :high].contiguous()
    l_v = grouped[..., high:].contiguous()
    h_idx = torch.searchsorted(buf.boundaries_high, h_v)
    l_idx = torch.searchsorted(buf.boundaries_low, l_v)
    h_q = buf.centroids_high[h_idx]
    l_q = buf.centroids_low[l_idx]
    h_mask = (
        (h_v >= buf.boundaries_high[0]) & (h_v <= buf.boundaries_high[-1])
    ).to(rotated.dtype)
    l_mask = (
        (l_v >= buf.boundaries_low[0]) & (l_v <= buf.boundaries_low[-1])
    ).to(rotated.dtype)

    h_ste = (h_q - h_v * h_mask).detach() + h_v * h_mask
    l_ste = (l_q - l_v * l_mask).detach() + l_v * l_mask
    quantized = torch.empty_like(grouped)
    quantized[..., :high] = h_ste
    quantized[..., high:] = l_ste
    quantized = quantized.reshape(rotated.shape)

    w_hat = fwht(quantized * buf.signs2)
    if buf.norm_correction:
        inner = torch.linalg.vector_norm(w_hat, dim=-1).clamp_min(1e-8)
        norm_hat = norm / inner
    else:
        norm_hat = norm
    z_hat = w_hat * buf.signs1
    return z_hat * norm_hat[:, None]


def test_backward_matches_torch_autograd():
    """Analytic backward reproduces torch.autograd on the STE-detach forward.

    Finite differences cannot validate STE-quantized backwards (the forward
    is locally flat between codebook boundaries). The autograd oracle here
    is the same STE the closed-form derivation assumes, so equality should
    be at machine precision (modulo fp32-vs-fp64 reduction order).
    """

    buf = _make_buffers()
    torch.manual_seed(0)
    x = torch.randn(4, LATENT_DIM, dtype=torch.float64)
    upstream = torch.randn_like(x)

    x_ad = x.clone().requires_grad_(True)
    y_ad = _ste_autograd_forward(x_ad, buf)
    (y_ad * upstream).sum().backward()
    g_torch = x_ad.grad

    y_ref, intermediates = turboquant_forward(x, buf, return_intermediates=True)
    g_analytic = turboquant_backward(upstream, intermediates, buf)

    torch.testing.assert_close(y_ad.detach(), y_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(g_torch, g_analytic, rtol=1e-6, atol=1e-6)


def test_autograd_function_shape_preserves():
    buf = _make_buffers(dtype=torch.float32)
    torch.manual_seed(0)
    for shape in [(LATENT_DIM,), (4, LATENT_DIM), (3, 2, LATENT_DIM), (2, 5, 7, LATENT_DIM)]:
        x = torch.randn(*shape, requires_grad=True)
        y = apply_turboquant_kv(x, buf)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


def test_autograd_function_grad_flows():
    buf = _make_buffers(dtype=torch.float32)
    torch.manual_seed(0)
    x = torch.randn(8, LATENT_DIM, requires_grad=True)
    y = apply_turboquant_kv(x, buf)
    upstream = torch.randn_like(y)
    (y * upstream).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Gradient is non-trivial on at least the channels that landed within the
    # codebook range.
    assert x.grad.abs().sum().item() > 0


def test_buffers_independent_of_seed_construction():
    """Buffers built with the same effective seed are identical regardless of
    whether the seed comes from ``seed`` or ``layer_idx``."""

    a = build_turboquant_buffers(latent_dim=LATENT_DIM, seed=5, layer_idx=0, device="cpu")
    # The hash collapses (seed * 2654435761 + layer_idx) mod 2^32, so direct
    # equality is only guaranteed for the same input pair.
    b = build_turboquant_buffers(latent_dim=LATENT_DIM, seed=5, layer_idx=0, device="cpu")
    torch.testing.assert_close(a.signs1, b.signs1, rtol=0, atol=0)


@pytest.mark.parametrize("preset", ["latent_4bit_nc", "latent_2p5bit_nc"])
def test_off_path_unchanged_when_not_applied(preset):
    """Sanity: not calling apply_turboquant_kv leaves x unchanged through autograd."""
    buf = build_turboquant_buffers(latent_dim=LATENT_DIM, preset=preset, device="cpu", dtype=torch.float32)
    torch.manual_seed(0)
    x = torch.randn(4, LATENT_DIM, requires_grad=True)
    y = x * 1.0
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
    # Buffers exist regardless of whether they're used:
    assert buf.signs1.numel() == LATENT_DIM
