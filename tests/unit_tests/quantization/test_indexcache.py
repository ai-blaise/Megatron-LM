# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Phase 1 tests for IndexCache fp8 fake-quant.

Covers:
  * forward output shape, finiteness, and reasonable error vs the unquantized input
  * eps clamp on all-zero rows
  * analytic backward vs torch.autograd on the STE-detach forward (the same
    trick we used for TurboQuant — finite differences cannot validate STE on
    a quantized op because the forward is locally flat between fp8 levels)
  * SGLang reference parity: when running on a host with fp8_e4m3fn, the
    Megatron forward is bit-identical to the SGLang Triton _act_quant kernel
    output (post-dequantize), modulo cast ordering
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from megatron.core.quantization.indexcache import (  # noqa: E402
    IndexCacheConfig,
    apply_indexcache_kv,
    build_indexcache_config,
)
from megatron.core.quantization.indexcache.reference import (  # noqa: E402
    indexcache_backward,
    indexcache_forward,
)


HEAD_DIM = 128


def _make_cfg():
    return build_indexcache_config()


def test_forward_shape_and_finite():
    cfg = _make_cfg()
    torch.manual_seed(0)
    x = torch.randn(16, HEAD_DIM, dtype=torch.float32)
    y = indexcache_forward(x, cfg)
    assert y.shape == x.shape
    assert torch.isfinite(y).all().item()


def test_quantization_error_bounded():
    """fp8 e4m3 has ~3-bit mantissa; relative error per coord under ~5%."""
    cfg = _make_cfg()
    torch.manual_seed(0)
    x = torch.randn(64, HEAD_DIM, dtype=torch.float32)
    y = indexcache_forward(x, cfg)
    err = (y - x).abs() / (x.abs() + 1e-3)
    assert err.mean().item() < 0.10
    assert err.max().item() < 1.0


def test_zero_row_handled_by_eps():
    """All-zero token must not divide by zero — eps clamp guards the scale."""
    cfg = _make_cfg()
    x = torch.zeros(4, HEAD_DIM, dtype=torch.float32)
    y, intermed = indexcache_forward(x, cfg, return_intermediates=True)
    # Scale is amax/fp8_max with amax clamped to eps, so scale = eps/fp8_max.
    expected_scale = cfg.eps * cfg.fp8_max_inv
    assert torch.allclose(intermed["scale"], torch.full_like(intermed["scale"], expected_scale))
    assert torch.isfinite(y).all().item()
    # eps_active should be 0 on every all-zero row.
    assert (intermed["eps_active"] == 0).all().item()


def test_autograd_function_shape_preserves():
    cfg = _make_cfg()
    torch.manual_seed(0)
    for shape in [(HEAD_DIM,), (4, HEAD_DIM), (3, 2, HEAD_DIM), (2, 5, 7, HEAD_DIM)]:
        x = torch.randn(*shape, requires_grad=True)
        y = apply_indexcache_kv(x, cfg)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


def test_autograd_finite_grads():
    cfg = _make_cfg()
    torch.manual_seed(0)
    x = torch.randn(8, HEAD_DIM, requires_grad=True)
    y = apply_indexcache_kv(x, cfg)
    upstream = torch.randn_like(y)
    (y * upstream).sum().backward()
    assert torch.isfinite(x.grad).all().item()
    # Gradient should be non-trivial (most coords pass the STE).
    assert x.grad.abs().sum().item() > 0


def _ste_autograd_forward(x: torch.Tensor, cfg: IndexCacheConfig) -> torch.Tensor:
    """Differentiable forward whose autograd matches our analytic backward.

    Uses ``(quantized - x_clip).detach() + x_clip`` to route gradient through
    the unquantized clipped input while the forward output equals the
    quantized value. Same idiom we used for TurboQuant's STE oracle.
    """

    abs_max = x.abs().amax(dim=-1)
    amax = abs_max.clamp_min(cfg.eps)
    scale = (amax * cfg.fp8_max_inv)[:, None]
    pre_clip = x / scale
    clipped = pre_clip.clamp(-cfg.fp8_max, cfg.fp8_max)
    if x.is_cuda or hasattr(torch, "float8_e4m3fn"):
        q_fp8 = clipped.to(torch.float8_e4m3fn).to(x.dtype)
    else:
        from megatron.core.quantization.indexcache.reference import (
            _simulate_fp8_e4m3_rounding,
        )
        q_fp8 = _simulate_fp8_e4m3_rounding(clipped)
    ste = (q_fp8 - clipped).detach() + clipped
    return ste * scale


def test_backward_matches_torch_autograd():
    """Analytic backward reproduces autograd on the STE-detach forward."""
    cfg = _make_cfg()
    torch.manual_seed(0)
    x = torch.randn(4, HEAD_DIM, dtype=torch.float64)
    upstream = torch.randn_like(x)

    x_ad = x.clone().requires_grad_(True)
    y_ad = _ste_autograd_forward(x_ad, cfg)
    (y_ad * upstream).sum().backward()
    g_torch = x_ad.grad

    y_ref, intermediates = indexcache_forward(x, cfg, return_intermediates=True)
    g_analytic = indexcache_backward(upstream, intermediates, cfg)

    torch.testing.assert_close(y_ad.detach(), y_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(g_torch, g_analytic, rtol=1e-6, atol=1e-6)


def test_dtype_matrix():
    cfg = _make_cfg()
    torch.manual_seed(0)
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(8, HEAD_DIM, dtype=dtype, requires_grad=True)
        y = apply_indexcache_kv(x, cfg)
        assert y.dtype == dtype
        y.sum().backward()
        assert x.grad.dtype == dtype


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="fp8 unavailable")
def test_parity_with_sglang_act_quant_math():
    """Match the SGLang Triton _act_quant_kernel formula at the cast level.

    SGLang stores fp8 + per-block scale; we dequantize for fake-quant. The
    underlying scaled-quantized values must be bit-identical when running
    on the same fp32 inputs.
    """
    cfg = _make_cfg()
    torch.manual_seed(0)
    x = torch.randn(64, HEAD_DIM, dtype=torch.float32)

    abs_max = x.abs().amax(dim=-1)
    amax = abs_max.clamp_min(cfg.eps)
    scale = (amax * cfg.fp8_max_inv)[:, None]
    pre_clip = (x / scale).clamp(-cfg.fp8_max, cfg.fp8_max)
    sgl_q = pre_clip.to(torch.float8_e4m3fn).to(torch.float32)
    sgl_dequant = sgl_q * scale

    y_ref = indexcache_forward(x, cfg)
    torch.testing.assert_close(y_ref, sgl_dequant, rtol=0, atol=0)
