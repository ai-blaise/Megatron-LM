# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU correctness tests for NVFP4 activation-ECO bias correction.

Covers:
  * forward shape, finiteness, per-block range
  * STE backward through the activation cast (analytic vs torch.autograd
    on the STE-detach forward — same trick used for IndexCache and
    TurboQuant since FD cannot validate STE on a quantized op)
  * activation-ECO bias correction recovers the unbiased dW exactly
  * statistical bias reduction across many random trials
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from megatron.core.quantization.nvfp4_act_eco import (  # noqa: E402
    apply_nvfp4_act_eco_linear,
    build_nvfp4_act_eco_config,
)
from megatron.core.quantization.nvfp4_act_eco.reference import (  # noqa: E402
    _round_to_nvfp4_grid,
    activation_eco_bias_correction,
    nvfp4_act_quant_forward,
)


HIDDEN = 64  # multiple of NVFP4 block size (16)
OUT = 32


def _cfg():
    return build_nvfp4_act_eco_config()


def _old_grid_round(x, *, fp4_max):
    grid = torch.tensor(
        [
            -fp4_max,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            fp4_max,
        ],
        dtype=x.dtype,
        device=x.device,
    )
    return grid[(x.unsqueeze(-1) - grid).abs().argmin(dim=-1)]


def test_threshold_rounder_matches_previous_grid_rounder():
    cfg = _cfg()
    values = torch.cat(
        [
            torch.linspace(-8.0, 8.0, 4097),
            torch.tensor(
                [
                    -5.0,
                    -3.5,
                    -2.5,
                    -1.75,
                    -1.25,
                    -0.75,
                    -0.25,
                    0.25,
                    0.75,
                    1.25,
                    1.75,
                    2.5,
                    3.5,
                    5.0,
                ]
            ),
        ]
    )
    expected = _old_grid_round(values.clamp(-cfg.fp4_max, cfg.fp4_max), fp4_max=cfg.fp4_max)
    got = _round_to_nvfp4_grid(values, fp4_max=cfg.fp4_max)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_forward_shape_and_finite():
    cfg = _cfg()
    torch.manual_seed(0)
    x = torch.randn(8, HIDDEN, dtype=torch.float32)
    q = nvfp4_act_quant_forward(x, cfg)
    assert q.shape == x.shape
    assert torch.isfinite(q).all().item()


def test_forward_per_block_range():
    """After dequant, each coord stays close to its block_amax envelope.

    The per-block scale is stored at FP8 e4m3 precision (3-bit mantissa),
    so the reconstructed scale can be up to ~12% larger than the true
    block_amax/fp4_max, allowing dequantized values up to ~1.15 *
    block_amax. Using 1.3x as a conservative bound.
    """
    cfg = _cfg()
    torch.manual_seed(0)
    x = torch.randn(16, HIDDEN, dtype=torch.float32)
    q = nvfp4_act_quant_forward(x, cfg)
    block_amax = (
        x.abs()
        .reshape(16, HIDDEN // cfg.block_size, cfg.block_size)
        .amax(dim=-1)
    )
    q_amax = (
        q.abs().reshape(16, HIDDEN // cfg.block_size, cfg.block_size).amax(-1)
    )
    assert (q_amax <= 1.3 * block_amax + 1e-5).all().item()


def test_forward_quant_error_bounded():
    """Per-block max relative error < ~30% (NVFP4's worst-case quantum)."""
    cfg = _cfg()
    torch.manual_seed(0)
    x = torch.randn(64, HIDDEN, dtype=torch.float32) * 5.0
    q = nvfp4_act_quant_forward(x, cfg)
    abs_err = (q - x).abs().reshape(64, -1, cfg.block_size)
    block_amax = x.abs().reshape(64, -1, cfg.block_size).amax(-1, keepdim=True)
    rel = (abs_err / block_amax.clamp_min(1e-6)).amax(-1)
    assert (rel < 0.35).all().item(), f"max rel err {rel.max()}"


def test_zero_input_yields_zero():
    cfg = _cfg()
    x = torch.zeros(4, HIDDEN, dtype=torch.float32)
    q = nvfp4_act_quant_forward(x, cfg)
    torch.testing.assert_close(q, x)


def _ste_detach_forward(x, W, cfg):
    """torch.autograd-friendly STE-detach reference for the fused linear.

    Saturated-zero STE: gradient flows on unsaturated lanes (mask=1)
    and is killed on saturated lanes (mask=0). Encoded for autograd as
    ``q_through = q.detach() + (x - x.detach()) * mask`` so
    ``d q_through / dx = mask`` and the forward equals ``q.detach()``.
    """
    q, intermediates = nvfp4_act_quant_forward(x, cfg, return_intermediates=True)
    mask = intermediates["clip_mask"].to(x.dtype)
    q_through = q.detach() + (x - x.detach()) * mask
    return q_through @ W.T


def _full_bf16_forward(x, W):
    """The unbiased reference: pure BF16 matmul, no quant."""
    return x @ W.T


def test_dW_act_eco_matches_unbiased_gradient():
    """The whole point of activation-ECO: dW with correction == dy.T @ x_pre."""
    cfg = _cfg()
    torch.manual_seed(7)
    x = torch.randn(8, HIDDEN, dtype=torch.float64)
    W = torch.randn(OUT, HIDDEN, dtype=torch.float64)

    x_act = x.clone().requires_grad_(True)
    W_act = W.clone().requires_grad_(True)
    y = apply_nvfp4_act_eco_linear(x_act, W_act, cfg)
    y.sum().backward()
    dw_act_eco = W_act.grad.clone()

    x_full = x.clone().requires_grad_(True)
    W_full = W.clone().requires_grad_(True)
    y_full = _full_bf16_forward(x_full, W_full)
    y_full.sum().backward()
    dw_full = W_full.grad.clone()

    torch.testing.assert_close(dw_act_eco, dw_full, atol=1e-10, rtol=1e-10)


def test_dx_matches_ste_detach_autograd():
    """dx is plain STE through the cast; matches torch.autograd on the STE-detach forward."""
    cfg = _cfg()
    torch.manual_seed(11)
    x = torch.randn(8, HIDDEN, dtype=torch.float64)
    W = torch.randn(OUT, HIDDEN, dtype=torch.float64)

    x_act = x.clone().requires_grad_(True)
    y_act = apply_nvfp4_act_eco_linear(x_act, W.clone(), cfg)
    y_act.sum().backward()
    dx_act = x_act.grad.clone()

    x_ref = x.clone().requires_grad_(True)
    y_ref = _ste_detach_forward(x_ref, W.clone(), cfg)
    y_ref.sum().backward()
    dx_ref = x_ref.grad.clone()

    torch.testing.assert_close(dx_act, dx_ref, atol=1e-10, rtol=1e-10)


def test_bias_reduction_statistical():
    """
    Across many trials the activation-ECO dW is closer to the unbiased dW
    than the naive QAT dW = dy.T @ q(x) is.
    """
    cfg = _cfg()
    torch.manual_seed(31)
    n_trials = 200
    naive_err_sum = 0.0
    eco_err_sum = 0.0

    for _ in range(n_trials):
        x = torch.randn(8, HIDDEN, dtype=torch.float32)
        W = torch.randn(OUT, HIDDEN, dtype=torch.float32)
        dy = torch.randn(8, OUT, dtype=torch.float32)

        # unbiased gradient (reference)
        dw_full = dy.T @ x

        # naive QAT: dW = dy.T @ q(x)
        q = nvfp4_act_quant_forward(x, cfg)
        dw_naive = dy.T @ q

        # activation-ECO corrected
        dw_eco = dw_naive + activation_eco_bias_correction(dy, x, q)

        naive_err_sum += (dw_naive - dw_full).norm().item()
        eco_err_sum += (dw_eco - dw_full).norm().item()

    avg_naive = naive_err_sum / n_trials
    avg_eco = eco_err_sum / n_trials
    # The correction should drive the error to ~zero; allow a tight float
    # tolerance and require the naive baseline to be at least 1e-3 worse
    # so the test fails loudly if the correction silently regresses.
    assert avg_eco < 1e-4, f"act-ECO residual error {avg_eco} too large"
    assert avg_naive > avg_eco * 100, (
        f"naive baseline error {avg_naive} not much larger than "
        f"act-ECO {avg_eco}; correction may be a no-op"
    )


def test_dx_finite_when_some_lanes_saturate():
    """STE on saturated lanes: gradient zeroed on saturated coords, finite overall."""
    cfg = _cfg()
    torch.manual_seed(13)
    x = torch.randn(4, HIDDEN, dtype=torch.float32) * 100.0  # forces saturation
    W = torch.randn(OUT, HIDDEN, dtype=torch.float32)
    x_act = x.clone().requires_grad_(True)
    y = apply_nvfp4_act_eco_linear(x_act, W, cfg)
    y.sum().backward()
    assert torch.isfinite(x_act.grad).all().item()


def test_eco_correction_is_zero_when_all_lanes_round_to_themselves():
    """
    When every block's amax is the FP4 max (6) — so the per-block scale
    is exactly 1.0 and stays at 1.0 after FP8 cast — and every coord
    already lies on the NVFP4 grid, q(x) == x exactly and the correction
    term vanishes. We seed the first lane of each block with ±6 to pin
    the scale, and fill the rest of the block with grid values.
    """
    cfg = _cfg()
    bs = cfg.block_size
    grid = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    )
    torch.manual_seed(0)
    n = 4
    n_blocks = HIDDEN // bs
    # fill with random grid values, then pin each block's first lane to ±6
    idx = torch.randint(0, len(grid), (n, HIDDEN))
    x = grid[idx].to(torch.float32)
    sign = torch.randint(0, 2, (n, n_blocks)).to(torch.float32) * 2 - 1
    pinned = (sign * 6.0).reshape(n, n_blocks, 1).expand(-1, -1, 1)
    x_blocks = x.reshape(n, n_blocks, bs).clone()
    x_blocks[:, :, 0:1] = pinned
    x = x_blocks.reshape(n, HIDDEN)

    q = nvfp4_act_quant_forward(x, cfg)
    torch.testing.assert_close(q, x, atol=1e-5, rtol=1e-5)
    dy = torch.randn(n, OUT, dtype=torch.float32)
    correction = activation_eco_bias_correction(dy, x, q)
    assert correction.abs().max().item() < 1e-5
