# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for IndexCache fake-quant.

Covers:
  * FP8 forward output shape, finiteness, and error vs the unquantized input
  * eps clamp on all-zero rows
  * FP8 analytic backward vs torch.autograd on the STE-detach forward (the same
    trick we used for TurboQuant — finite differences cannot validate STE on
    a quantized op because the forward is locally flat between fp8 levels)
  * NVFP4 packed value/scale layout and the four-group STE backward
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
    INDEXCACHE_QUANT_DISABLED,
    INDEXCACHE_QUANT_FP8,
    INDEXCACHE_QUANT_NVFP4,
    IndexCacheConfig,
    apply_indexcache_kv,
    build_indexcache_config,
    resolve_indexcache_quantization,
)
from megatron.core.quantization.indexcache.reference import (  # noqa: E402
    indexcache_backward,
    indexcache_forward,
)


HEAD_DIM = 128


def _make_cfg():
    return build_indexcache_config()


def _make_nvfp4_cfg():
    return build_indexcache_config(quantization=INDEXCACHE_QUANT_NVFP4)


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


def _ceil_to_ue8m0_exp(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF).bool().int())
    return exp.clamp(1, 254).to(torch.uint8)


def _ue8m0_exp_to_float(exp: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (exp.to(torch.int32) << 23).contiguous().view(torch.float32).to(dtype)


def _pack_ue8m0_exp_to_int(exp: torch.Tensor) -> torch.Tensor:
    return exp.contiguous().view(torch.int32).reshape(exp.shape[0])


def _quantize_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    idx = torch.zeros_like(ax, dtype=torch.uint8)
    idx = torch.where(ax > 0.25, torch.ones_like(idx), idx)
    idx = torch.where(ax >= 0.75, torch.full_like(idx, 2), idx)
    idx = torch.where(ax > 1.25, torch.full_like(idx, 3), idx)
    idx = torch.where(ax >= 1.75, torch.full_like(idx, 4), idx)
    idx = torch.where(ax > 2.5, torch.full_like(idx, 5), idx)
    idx = torch.where(ax >= 3.5, torch.full_like(idx, 6), idx)
    idx = torch.where(ax > 5.0, torch.full_like(idx, 7), idx)
    sign = (x < 0) & (idx != 0)
    return idx | (sign.to(torch.uint8) << 3)


def _e2m1_codes_to_values(codes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=dtype,
        device=codes.device,
    )
    values = lut[(codes & 0x7).long()]
    return torch.where((codes & 0x8).bool(), -values, values)


def _ref_indexer_nvfp4(x: torch.Tensor, cfg: IndexCacheConfig):
    compute_dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    xf = x.to(compute_dtype)
    rows, cols = xf.shape
    assert cols == HEAD_DIM
    groups = xf.view(rows, 4, 32)
    exp = _ceil_to_ue8m0_exp(groups.abs().amax(dim=2).clamp_min(cfg.eps) / 6.0)
    scale = _ue8m0_exp_to_float(exp, compute_dtype)
    codes = _quantize_to_e2m1_codes(groups / scale.unsqueeze(-1)).view(rows, cols)
    packed_values = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
    y = (_e2m1_codes_to_values(codes, compute_dtype).view(rows, 4, 32) * scale.unsqueeze(-1))
    return y.reshape_as(xf).to(x.dtype), packed_values.contiguous(), _pack_ue8m0_exp_to_int(exp)


def _nvfp4_backward_oracle(grad_y: torch.Tensor, intermediates: dict, cfg: IndexCacheConfig):
    x = intermediates["x_compute"]
    g = grad_y.to(x.dtype)
    groups = x.view(x.shape[0], 4, 32)
    g_groups = g.view_as(groups)
    q_groups = intermediates["q_e2m1"].view_as(groups)
    mask_groups = intermediates["clip_mask"].view_as(groups)
    scale = intermediates["scale"]
    argmax = intermediates["argmax"].long()
    eps_active = intermediates["eps_active"]

    direct = g_groups * mask_groups
    inner = (g_groups * (q_groups - mask_groups * groups / scale.unsqueeze(-1))).sum(dim=-1)
    sign = torch.gather(groups.sign(), -1, argmax.unsqueeze(-1)).squeeze(-1)
    update = inner * sign * eps_active / 6.0
    rank = torch.zeros_like(direct)
    rank.scatter_(-1, argmax.unsqueeze(-1), update.unsqueeze(-1))
    return (direct + rank).reshape_as(x).to(grad_y.dtype)


def _nvfp4_ste_autograd_forward(x: torch.Tensor, cfg: IndexCacheConfig) -> torch.Tensor:
    groups = x.view(x.shape[0], 4, 32)
    scale_base = groups.abs().amax(dim=-1).clamp_min(cfg.eps) / 6.0
    scale_exp = _ceil_to_ue8m0_exp(scale_base)
    scale_quant = _ue8m0_exp_to_float(scale_exp, x.dtype)
    scale = (scale_quant - scale_base).detach() + scale_base
    pre_clip = groups / scale.unsqueeze(-1)
    clipped = pre_clip.clamp(-6.0, 6.0)
    codes = _quantize_to_e2m1_codes(clipped)
    q = _e2m1_codes_to_values(codes, x.dtype)
    ste = (q - clipped).detach() + clipped
    return (ste * scale.unsqueeze(-1)).reshape_as(x)


def test_disabled_indexcache_is_noop():
    cfg = build_indexcache_config(quantization=INDEXCACHE_QUANT_DISABLED)
    x = torch.randn(2, HEAD_DIM, requires_grad=True)
    y = apply_indexcache_kv(x, cfg)
    assert y is x
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_nvfp4_forward_shape_finite_and_reference_dequant():
    cfg = _make_nvfp4_cfg()
    torch.manual_seed(3)
    x = torch.randn(9, HEAD_DIM, dtype=torch.float32) * 0.75
    y, intermediates = indexcache_forward(x, cfg, return_intermediates=True)
    ref_y, _, _ = _ref_indexer_nvfp4(x, cfg)
    assert y.shape == x.shape
    assert torch.isfinite(y).all().item()
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
    assert intermediates["scale"].shape == (x.shape[0], 4)
    assert intermediates["q_e2m1"].shape == x.shape


def test_nvfp4_packed_value_and_scale_layout_matches_oracle():
    cfg = _make_nvfp4_cfg()
    x = torch.stack(
        [
            torch.linspace(-3.25, 3.25, HEAD_DIM),
            torch.arange(HEAD_DIM, dtype=torch.float32).sub(64).div(11),
        ],
        dim=0,
    )
    _, intermediates = indexcache_forward(x, cfg, return_intermediates=True)
    _, ref_values, ref_scales = _ref_indexer_nvfp4(x, cfg)
    torch.testing.assert_close(intermediates["packed_values"], ref_values)
    torch.testing.assert_close(intermediates["packed_scales"], ref_scales)


@pytest.mark.parametrize("case", ["random", "zero_tiny", "e2m1_max", "tie_argmax"])
def test_nvfp4_backward_matches_ste_oracle(case):
    cfg = _make_nvfp4_cfg()
    torch.manual_seed(5)
    x = torch.randn(3, HEAD_DIM, dtype=torch.float64) * 0.4
    if case == "zero_tiny":
        x[0].zero_()
        x[1].fill_(cfg.eps * 0.25)
    elif case == "e2m1_max":
        x[0].zero_()
        x[0, 0:32] = torch.linspace(-6.0, 6.0, 32, dtype=x.dtype)
        x[1].mul_(32.0)
    elif case == "tie_argmax":
        x[0].zero_()
        x[0, 0] = 2.0
        x[0, 7] = -2.0
        x[0, 32] = -3.0
        x[0, 33] = 3.0

    upstream = torch.randn_like(x)
    _, intermediates = indexcache_forward(x, cfg, return_intermediates=True)
    got = indexcache_backward(upstream, intermediates, cfg)
    expected = _nvfp4_backward_oracle(upstream, intermediates, cfg)
    torch.testing.assert_close(got, expected, rtol=1e-7, atol=1e-7)

    if case == "zero_tiny":
        assert (intermediates["eps_active"][0:2] == 0).all().item()
    if case == "tie_argmax":
        assert intermediates["argmax"][0, 0].item() == 0
        assert intermediates["argmax"][0, 1].item() == 0


def test_nvfp4_backward_matches_torch_autograd_ste():
    cfg = _make_nvfp4_cfg()
    torch.manual_seed(11)
    x = torch.randn(4, HEAD_DIM, dtype=torch.float64) * 0.5
    x[0, 0:32] = torch.linspace(-7.5, 8.0, 32, dtype=x.dtype)
    x[1].fill_(cfg.eps * 0.25)
    upstream = torch.randn_like(x)

    x_ad = x.clone().requires_grad_(True)
    y_ad = _nvfp4_ste_autograd_forward(x_ad, cfg)
    (y_ad * upstream).sum().backward()

    y_ref, intermediates = indexcache_forward(x, cfg, return_intermediates=True)
    g_ref = indexcache_backward(upstream, intermediates, cfg)

    torch.testing.assert_close(y_ref, y_ad.detach(), rtol=0, atol=0)
    torch.testing.assert_close(g_ref, x_ad.grad, rtol=1e-7, atol=1e-7)


def test_nvfp4_autograd_shape_and_dtype_preserves():
    cfg = _make_nvfp4_cfg()
    torch.manual_seed(7)
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(2, 3, HEAD_DIM, dtype=dtype, requires_grad=True)
        y = apply_indexcache_kv(x, cfg)
        assert y.shape == x.shape
        assert y.dtype == dtype
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.dtype == dtype


def test_indexcache_config_and_cli_selection():
    from argparse import ArgumentParser

    from megatron.training.arguments import add_megatron_arguments

    assert resolve_indexcache_quantization(
        quantization=INDEXCACHE_QUANT_DISABLED, quant_enabled=True
    ) == INDEXCACHE_QUANT_FP8
    assert resolve_indexcache_quantization(
        quantization=INDEXCACHE_QUANT_NVFP4, quant_enabled=False
    ) == INDEXCACHE_QUANT_NVFP4

    parser = ArgumentParser(allow_abbrev=False)
    add_megatron_arguments(parser)
    args = parser.parse_args(["--dsa-indexcache-quantization", INDEXCACHE_QUANT_NVFP4])
    assert args.dsa_indexcache_quantization == INDEXCACHE_QUANT_NVFP4
    assert not args.dsa_indexcache_quant_enabled

    args = parser.parse_args(["--dsa-indexcache-quant-enabled"])
    assert args.dsa_indexcache_quant_enabled
    assert resolve_indexcache_quantization(
        quantization=args.dsa_indexcache_quantization,
        quant_enabled=args.dsa_indexcache_quant_enabled,
    ) == INDEXCACHE_QUANT_FP8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_non_blackwell_cuda_uses_reference_fallback(monkeypatch):
    if torch.cuda.get_device_capability() >= (10, 0):
        pytest.skip("Blackwell should exercise the CUDA NVFP4 extension path.")

    import megatron.core.quantization.indexcache.autograd as indexcache_autograd

    def fail_load_ext():
        raise AssertionError("NVFP4 non-Blackwell path must not load the extension")

    monkeypatch.setattr(indexcache_autograd, "_try_load_cuda_ext", fail_load_ext)
    cfg = _make_nvfp4_cfg()
    x = torch.randn(2, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = apply_indexcache_kv(x, cfg)
    y.float().sum().backward()
    assert y.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad.float()).all().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_blackwell_cuda_packed_backward_matches_reference():
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("NVFP4 packed CUDA backward requires Blackwell.")

    from megatron.core.quantization.indexcache.kernels.build import get_ext

    cfg = _make_nvfp4_cfg()
    ext = get_ext()
    for dtype in [torch.float32, torch.bfloat16]:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(20260512)
        x = (
            torch.randn((4, HEAD_DIM), device="cuda", dtype=torch.float32, generator=gen)
            * 0.65
        )
        x[0].zero_()
        x[1, 0] = 2.0
        x[1, 7] = -2.0
        x[1, 32] = -3.0
        x[1, 33] = 3.0
        gy = torch.randn((4, HEAD_DIM), device="cuda", dtype=torch.float32, generator=gen)
        x = x.to(dtype).contiguous()
        gy = gy.to(dtype).contiguous()

        ref_y, ref_inter = indexcache_forward(x, cfg, return_intermediates=True)
        ref_gx = indexcache_backward(gy, ref_inter, cfg)

        out = torch.empty_like(x)
        scale = torch.empty((4, 4), device="cuda", dtype=torch.float32)
        q = torch.empty((4, HEAD_DIM), device="cuda", dtype=torch.float32)
        mask = torch.empty((4, HEAD_DIM), device="cuda", dtype=torch.uint8)
        argmax = torch.empty((4, 4), device="cuda", dtype=torch.int32)
        eps_active = torch.empty((4, 4), device="cuda", dtype=torch.uint8)
        packed_values = torch.empty((4, 64), device="cuda", dtype=torch.uint8)
        packed_scales = torch.empty((4,), device="cuda", dtype=torch.int32)
        gx = torch.empty_like(x)

        ext.indexcache_nvfp4_fwd(
            x, out, scale, q, mask, argmax, eps_active,
            packed_values, packed_scales, cfg.eps,
        )
        ext.indexcache_nvfp4_bwd_packed(
            gy, x, scale, packed_values, mask, argmax, eps_active, gx, cfg.fp4_max,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(out, ref_y, rtol=0, atol=0)
        if dtype is torch.float32:
            torch.testing.assert_close(gx, ref_gx, rtol=1e-6, atol=1e-6)
        else:
            torch.testing.assert_close(gx.float(), ref_gx.float(), rtol=1e-2, atol=1e-3)


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
