#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Diagnose whether ECO reproduces master-weight residual behavior.

This is a toy Monte Carlo diagnostic, not a trainer.  It compares the
parameter-space residual a classical master-weight path preserves against the
parameter-space correction implied by ECO after that correction has been stored
in FlashAdamW's rowwise quantized first-moment state.

The key unit conversion is:

    Adam update ~= -lr * m / ((1 - beta1**t) * denom)

If ECO injects ``delta_m`` into the first moment, the constant-lr/denom
parameter-space correction carried by the momentum tail is approximated by:

    replay ~= -lr / (1 - beta1**t) * beta1 / (1 - beta1) * delta_m / denom

Paper ECO is constructed so this replay is close to the weight quantization
residual before the first-moment state is quantized.  This script checks how
much of that equivalence survives rowwise INT8 moment storage.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eco_stability_sweep import _lr_at_step


@dataclass(frozen=True)
class Method:
    name: str
    alpha_mode: str
    lr_floor_fraction: float = 0.0
    group_budget: float | None = None
    qa_mode: str = "none"
    scale_budget: float = 4.0
    max_boost: float = 8.0
    grid_min_steps: float = 0.5
    residual_quant: str = "linear"
    projection_error_budget: float = 0.25
    projection_steps: int = 16
    projection_metric: str = "rms"


@dataclass
class StepMetrics:
    step: int
    method: str
    lr: float
    master_resid_rms: float
    master_resid_p99: float
    master_resid_p999: float
    eco_resid_rms: float
    eco_resid_p999: float
    intended_replay_rms: float
    intended_replay_p999: float
    stored_replay_rms: float
    stored_replay_p999: float
    intended_energy_gain: float
    stored_energy_gain: float
    intended_abs_ratio_p50: float
    intended_abs_ratio_p99: float
    intended_abs_ratio_p999: float
    stored_abs_ratio_p50: float
    stored_abs_ratio_p99: float
    stored_abs_ratio_p999: float
    material_abs_ratio_p99: float
    material_abs_ratio_p999: float
    p_bad_ratio: float
    p_bad_material_ratio: float
    p_any_bad_345b: float
    p_any_bad_material_345b: float
    mom_abs_p999: float
    mom_abs_max: float
    mom_scale_p99: float
    mom_scale_inflation_p99: float
    mom_saturation_frac: float
    carrier_resid_p999: float
    carrier_resid_scale_p99: float


@dataclass
class FinalDistribution:
    method: str
    stored_ratio: torch.Tensor
    intended_ratio: torch.Tensor
    stored_replay_abs: torch.Tensor
    residual_abs: torch.Tensor


def _quantiles(x: torch.Tensor, qs: Iterable[float]) -> list[float]:
    if x.numel() == 0:
        return [float("nan") for _ in qs]
    q_tensor = torch.tensor(list(qs), device=x.device, dtype=torch.float32)
    values = torch.quantile(x.detach().float(), q_tensor)
    return [float(v.item()) for v in values]


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.detach().float().square())).item())


def _safe_ratio(num: torch.Tensor, denom: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mask = denom.abs() > eps
    out = torch.zeros_like(num, dtype=torch.float32)
    out[mask] = num[mask].float() / denom[mask].float()
    return out


def _energy_gain(replay: torch.Tensor, residual: torch.Tensor) -> float:
    denom = residual.float().square().sum().clamp_min(1e-30)
    return float((replay.float() * residual.float()).sum().div(denom).item())


def _p_any(frac: float, virtual_params: int) -> float:
    if frac <= 0:
        return 0.0
    return 1.0 - math.exp(-min(700.0, frac * virtual_params))


def _eco_alpha_for_method(
    method: Method,
    *,
    bc1: float,
    lr: float,
    base_lr: float,
    beta1: float,
) -> float:
    if method.alpha_mode == "off":
        return 0.0
    factor = 1.0 - 1.0 / beta1
    if method.alpha_mode == "paper":
        lr_floor = base_lr * method.lr_floor_fraction
        effective_lr = max(lr, lr_floor) if lr_floor > 0.0 else lr
        return (bc1 / effective_lr) * factor if effective_lr != 0.0 else 0.0
    if method.alpha_mode == "no_inv_lr":
        return bc1 * factor
    raise ValueError(f"unknown ECO alpha mode: {method.alpha_mode}")


def _quantize_moment_no_stats(
    m: torch.Tensor, group_size: int, moment_quantizer: str = "softsign"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    view = m.view(-1, group_size)
    scale = view.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = (view / scale).clamp(-1.0, 1.0)
    if moment_quantizer == "softsign":
        transformed = 2.0 * normalized / (1.0 + normalized.abs())
        packed = torch.floor(transformed * 127.0 + 0.5).clamp(-127.0, 127.0)
        unpacked = packed / 127.0
        recovered = unpacked / (2.0 - unpacked.abs()).clamp_min(1e-12)
    elif moment_quantizer == "linear":
        packed = torch.floor(normalized * 127.0 + 0.5).clamp(-127.0, 127.0)
        recovered = packed / 127.0
    else:
        raise ValueError(f"unknown moment quantizer: {moment_quantizer}")
    scale_f16 = scale.squeeze(1).to(torch.float16)
    return (
        (recovered * scale_f16.float().view(-1, 1)).reshape_as(m),
        scale_f16.float(),
        packed,
    )


def _quantize_moment_with_stats(
    m: torch.Tensor, group_size: int, moment_quantizer: str = "softsign"
) -> tuple[torch.Tensor, torch.Tensor, float]:
    recovered, scale_f16, packed = _quantize_moment_no_stats(
        m, group_size, moment_quantizer
    )
    saturation_frac = float((packed.abs() >= 127.0).float().mean().item())
    return (
        recovered,
        scale_f16,
        saturation_frac,
    )


def _apply_group_budget(
    delta_m: torch.Tensor,
    group_scale: torch.Tensor,
    *,
    group_size: int,
    budget: float | None,
) -> torch.Tensor:
    if budget is None:
        return delta_m
    if budget <= 0.0:
        return torch.zeros_like(delta_m)

    view = delta_m.view(-1, group_size)
    delta_absmax = view.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    group_budget = (group_scale.view(-1, 1).float().clamp_min(1e-12) * budget)
    shrink = torch.minimum(torch.ones_like(delta_absmax), group_budget / delta_absmax)
    return (view * shrink).reshape_as(delta_m)


def _apply_scale_preserve_budget(
    m: torch.Tensor,
    delta_m: torch.Tensor,
    group_scale: torch.Tensor,
    *,
    group_size: int,
    scale_budget: float,
) -> torch.Tensor:
    if scale_budget <= 1.0:
        return torch.zeros_like(delta_m)

    delta_view = delta_m.view(-1, group_size)
    delta_absmax = delta_view.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    old_scale = group_scale.view(-1, 1).float().clamp_min(1e-12)
    allowance = (scale_budget - 1.0) * old_scale
    shrink = torch.minimum(torch.ones_like(delta_absmax), allowance / delta_absmax)
    return (delta_view * shrink).reshape_as(delta_m)


def _apply_grid_clip(
    delta_m: torch.Tensor,
    group_scale: torch.Tensor,
    *,
    group_size: int,
    min_steps: float,
) -> torch.Tensor:
    if min_steps <= 0.0:
        return delta_m
    # Near zero, the softsign moment quantizer has about scale / 254 spacing.
    step = group_scale.view(-1, 1).float().clamp_min(1e-12) / 254.0
    threshold = step * min_steps
    view = delta_m.view(-1, group_size)
    return torch.where(view.abs() >= threshold, view, torch.zeros_like(view)).reshape_as(delta_m)


def _apply_stored_replay_aware_boost(
    m: torch.Tensor,
    delta_m: torch.Tensor,
    *,
    group_size: int,
    max_boost: float,
    moment_quantizer: str,
) -> torch.Tensor:
    if max_boost <= 1.0:
        return delta_m

    m_old_q, _, _ = _quantize_moment_no_stats(m, group_size, moment_quantizer)
    m_new_q, _, _ = _quantize_moment_no_stats(m + delta_m, group_size, moment_quantizer)
    stored_delta = (m_new_q - m_old_q).view(-1, group_size)
    desired_delta = delta_m.view(-1, group_size)
    denom = desired_delta.square().sum(dim=1, keepdim=True).clamp_min(1e-30)
    gain = (stored_delta * desired_delta).sum(dim=1, keepdim=True) / denom
    boost = torch.where(
        (gain > 1e-3) & (gain < 1.0),
        1.0 / gain.clamp_min(1e-3),
        torch.ones_like(gain),
    )
    boost = boost.clamp(1.0, max_boost)
    return (desired_delta * boost).reshape_as(delta_m)


def _apply_capacity_projection(
    m_before: torch.Tensor,
    delta_m: torch.Tensor,
    residual: torch.Tensor,
    denom: torch.Tensor,
    group_scale: torch.Tensor,
    *,
    group_size: int,
    lr: float,
    bc1: float,
    beta1: float,
    scale_budget: float,
    replay_error_budget: float,
    projection_steps: int,
    projection_metric: str,
    moment_quantizer: str,
) -> torch.Tensor:
    """Project ECO injection onto what the quantized moment carrier can replay.

    This keeps the paper ECO direction, then searches per INT8 moment group for
    the largest fraction whose *stored* replay still matches that fractional
    target after the softsign companding quantizer.  The bound is therefore on
    quantized carrier behavior, not on raw delta magnitude.
    """

    if projection_steps <= 0 or replay_error_budget < 0.0 or scale_budget <= 1.0:
        return torch.zeros_like(delta_m)

    m_view = m_before.view(-1, group_size)
    delta_view = delta_m.view(-1, group_size)
    residual_view = residual.view(-1, group_size)
    denom_view = denom.view(-1, group_size)
    old_scale = group_scale.view(-1, 1).float().clamp_min(1e-12)

    chosen = torch.zeros_like(old_scale)
    found = torch.zeros_like(old_scale, dtype=torch.bool)
    replay_coef = -(lr / bc1) * (beta1 / (1.0 - beta1))

    for idx in range(projection_steps + 1):
        frac = 1.0 - (idx / projection_steps)
        frac_tensor = torch.full_like(old_scale, frac)
        candidate = (m_view + delta_view * frac_tensor).reshape_as(m_before)
        candidate_q, new_scale, _ = _quantize_moment_no_stats(
            candidate, group_size, moment_quantizer
        )

        stored_delta = candidate_q.view(-1, group_size) - m_view
        stored_replay = replay_coef * stored_delta / denom_view
        target_replay = residual_view * frac_tensor

        target_energy = target_replay.float().square().mean(dim=1, keepdim=True)
        if projection_metric == "rms":
            replay_error = (stored_replay - target_replay).float().square().mean(
                dim=1, keepdim=True
            )
            rel_error = torch.sqrt(replay_error / target_energy.clamp_min(1e-30))
            rel_error = torch.where(
                target_energy > 1e-30,
                rel_error,
                torch.zeros_like(rel_error),
            )
        elif projection_metric == "gain":
            target_dot = (stored_replay.float() * target_replay.float()).mean(
                dim=1, keepdim=True
            )
            gain = target_dot / target_energy.clamp_min(1e-30)
            rel_error = (gain - 1.0).abs()
            rel_error = torch.where(
                target_energy > 1e-30,
                rel_error,
                torch.zeros_like(rel_error),
            )
        else:
            raise ValueError(f"unknown projection metric: {projection_metric}")

        scale_inflation = new_scale.view(-1, 1) / old_scale
        ok = (scale_inflation <= scale_budget) & (rel_error <= replay_error_budget)
        take = ok & ~found
        chosen = torch.where(take, frac_tensor, chosen)
        found = found | ok

    return (delta_view * chosen).reshape_as(delta_m)


def _quantize_var_sqrt(v: torch.Tensor, group_size: int) -> torch.Tensor:
    view = v.clamp_min(0.0).sqrt().view(-1, group_size)
    scale = view.amax(dim=1, keepdim=True).clamp_min(1e-30)
    packed = torch.floor((view / scale).clamp(0.0, 1.0) * 255.0 + 0.5).clamp(
        0.0, 255.0
    )
    scale_f16 = scale.squeeze(1).to(torch.float16)
    recovered_sqrt = packed / 255.0 * scale_f16.float().view(-1, 1)
    return recovered_sqrt.square().reshape_as(v)


def _quantize_residual_linear_state(
    residual: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    view = residual.view(-1, group_size)
    scale = view.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    packed = torch.floor((view / scale).clamp(-1.0, 1.0) * 127.0 + 0.5).clamp(
        -127.0, 127.0
    )
    return packed.to(torch.int8).reshape_as(residual), scale.squeeze(1).to(torch.float16)


def _dequantize_residual_linear_state(
    residual_i8: torch.Tensor, scales_f16: torch.Tensor, group_size: int
) -> torch.Tensor:
    view = residual_i8.float().view(-1, group_size)
    scale = scales_f16.float().view(-1, 1)
    return (view / 127.0 * scale).reshape(residual_i8.shape)


def _quantize_residual_softsign_state(
    residual: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    view = residual.view(-1, group_size)
    scale = view.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    normalized = (view / scale).clamp(-1.0, 1.0)
    transformed = 2.0 * normalized / (1.0 + normalized.abs())
    packed = torch.floor(transformed * 127.0 + 0.5).clamp(-127.0, 127.0)
    return packed.to(torch.int8).reshape_as(residual), scale.squeeze(1).to(torch.float16)


def _dequantize_residual_softsign_state(
    residual_i8: torch.Tensor, scales_f16: torch.Tensor, group_size: int
) -> torch.Tensor:
    view = residual_i8.float().view(-1, group_size)
    unpacked = view / 127.0
    recovered = unpacked / (2.0 - unpacked.abs()).clamp_min(1e-12)
    return (recovered * scales_f16.float().view(-1, 1)).reshape(residual_i8.shape)


def _quantize_residual_state(
    residual: torch.Tensor, group_size: int, quant: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if quant == "linear":
        return _quantize_residual_linear_state(residual, group_size)
    if quant == "softsign":
        return _quantize_residual_softsign_state(residual, group_size)
    raise ValueError(f"unknown residual quantizer: {quant}")


def _dequantize_residual_state(
    residual_i8: torch.Tensor, scales_f16: torch.Tensor, group_size: int, quant: str
) -> torch.Tensor:
    if quant == "linear":
        return _dequantize_residual_linear_state(residual_i8, scales_f16, group_size)
    if quant == "softsign":
        return _dequantize_residual_softsign_state(residual_i8, scales_f16, group_size)
    raise ValueError(f"unknown residual quantizer: {quant}")


def _maybe_quantize_state(
    m: torch.Tensor, v: torch.Tensor, group_size: int, moment_quantizer: str
) -> tuple[torch.Tensor, torch.Tensor]:
    m_q, _, _ = _quantize_moment_with_stats(m, group_size, moment_quantizer)
    return m_q, _quantize_var_sqrt(v, group_size)


def _roundtrip_fp8_e4m3fn(x: torch.Tensor) -> torch.Tensor:
    """Round through FP8 E4M3FN when PyTorch exposes the production dtype."""
    if hasattr(torch, "float8_e4m3fn"):
        try:
            return x.to(torch.float8_e4m3fn).float()
        except RuntimeError:
            pass
    return x


def _round_to_nvfp4_e2m1_grid(x: torch.Tensor) -> torch.Tensor:
    """Round normalized values to the NVFP4 E2M1 grid used by FlashOptim."""
    ax = x.abs()
    mag = torch.where(
        ax <= 0.25,
        torch.zeros_like(ax),
        torch.where(
            ax < 0.75,
            torch.full_like(ax, 0.5),
            torch.where(
                ax <= 1.25,
                torch.ones_like(ax),
                torch.where(
                    ax < 1.75,
                    torch.full_like(ax, 1.5),
                    torch.where(
                        ax <= 2.5,
                        torch.full_like(ax, 2.0),
                        torch.where(
                            ax < 3.5,
                            torch.full_like(ax, 3.0),
                            torch.where(
                                ax <= 5.0,
                                torch.full_like(ax, 4.0),
                                torch.full_like(ax, 6.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    return torch.where(x < 0.0, -mag, mag)


def _nvfp4_2d_weight_cast_reference(
    x: torch.Tensor,
    *,
    weight_cols: int,
    block_size: int,
    stochastic: bool,
    dither_coef: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Reference the production rowwise NVFP4 2D master-weight cast.

    This mirrors the FlashAdamW + NVFP4 ECO cast ingredients closely enough for
    diagnostics: 16x16 block amax, tensor-global scale, FP8 E4M3FN block-scale
    roundtrip, optional block-aware stochastic dither, E2M1 grid rounding, then
    dequantization from the packed value.  It intentionally returns only the
    dequantized post-cast tensor, because ECO consumes ``pre_cast - post_cast``.
    """
    if weight_cols <= 0:
        raise ValueError("--weight-cols must be positive")
    if block_size <= 0:
        raise ValueError("--cast-group-size/block-size must be positive")

    original_shape = x.shape
    flat = x.float().reshape(-1)
    original_numel = flat.numel()
    if original_numel == 0:
        return x.clone()

    rows = math.ceil(original_numel / weight_cols)
    padded_numel = rows * weight_cols
    if padded_numel != original_numel:
        flat = torch.nn.functional.pad(flat, (0, padded_numel - original_numel))
    mat = flat.view(rows, weight_cols)

    padded_rows = math.ceil(rows / block_size) * block_size
    padded_cols = math.ceil(weight_cols / block_size) * block_size
    padded = torch.zeros(
        (padded_rows, padded_cols), dtype=torch.float32, device=x.device
    )
    padded[:rows, :weight_cols] = mat

    tile_rows = padded_rows // block_size
    tile_cols = padded_cols // block_size
    tiles = (
        padded.view(tile_rows, block_size, tile_cols, block_size)
        .permute(0, 2, 1, 3)
        .reshape(tile_rows, tile_cols, block_size * block_size)
    )
    block_amax = tiles.abs().amax(dim=-1)
    global_amax = mat.abs().amax()
    global_scale = torch.where(
        global_amax > 0.0,
        torch.tensor(2688.0, dtype=torch.float32, device=x.device) / global_amax,
        torch.ones((), dtype=torch.float32, device=x.device),
    )

    block_scale = block_amax * (global_scale / 6.0)
    block_scale = block_scale.clamp(min=-448.0, max=448.0)
    block_scale = _roundtrip_fp8_e4m3fn(block_scale)
    expanded_scale = block_scale.repeat_interleave(block_size, dim=0).repeat_interleave(
        block_size, dim=1
    )[:rows, :weight_cols]

    cast_input = mat
    if stochastic:
        rand = torch.rand(mat.shape, generator=generator, device=x.device)
        dither = (rand * 2.0 - 1.0) * dither_coef * expanded_scale / global_scale
        cast_input = mat + dither

    encoded = torch.where(
        expanded_scale > 0.0,
        cast_input * global_scale / expanded_scale.clamp_min(1e-30),
        cast_input,
    )
    encoded = encoded.clamp(-6.0, 6.0)
    fp4 = _round_to_nvfp4_e2m1_grid(encoded)
    post = torch.where(
        expanded_scale > 0.0,
        fp4 * expanded_scale / global_scale,
        torch.zeros_like(fp4),
    )
    return post.reshape(-1)[:original_numel].reshape(original_shape)


def _make_initial_theta(
    *,
    sample_elems: int,
    init_std: float,
    group_size: int,
    weight_cols: int,
    dither_coef: float,
    device: str,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    theta = torch.randn(sample_elems, generator=gen, device=device, dtype=torch.float32)
    theta = theta * init_std
    return _nvfp4_2d_weight_cast_reference(
        theta,
        weight_cols=weight_cols,
        block_size=group_size,
        stochastic=False,
        dither_coef=dither_coef,
        generator=gen,
    )


def _simulate_master_path(
    *,
    initial_theta: torch.Tensor,
    steps: int,
    base_lr: float,
    warmup_steps: int,
    beta1: float,
    beta2: float,
    eps: float,
    curvature: float,
    grad_noise_std: float,
    group_size: int,
    cast_group_size: int,
    weight_cols: int,
    dither_coef: float,
    quantized_state: bool,
    stochastic_cast: bool,
    moment_quantizer: str,
    seed: int,
) -> tuple[list[dict[str, float]], list[torch.Tensor]]:
    device = str(initial_theta.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    theta_master = initial_theta.clone()
    theta_model = _nvfp4_2d_weight_cast_reference(
        theta_master,
        weight_cols=weight_cols,
        block_size=cast_group_size,
        stochastic=False,
        dither_coef=dither_coef,
        generator=gen,
    )
    m = torch.zeros_like(theta_master)
    v = torch.zeros_like(theta_master)

    metrics: list[dict[str, float]] = []
    residuals: list[torch.Tensor] = []

    for step in range(1, steps + 1):
        lr = _lr_at_step(step, base_lr=base_lr, warmup_steps=warmup_steps)
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step

        grad = curvature * theta_model
        if grad_noise_std > 0.0:
            grad = grad + torch.randn(
                theta_model.numel(), generator=gen, device=theta_model.device
            ) * grad_noise_std

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad.square()

        denom = torch.sqrt(v / bc2).add_(eps)
        theta_master = theta_master - lr * (m / bc1) / denom
        theta_model = _nvfp4_2d_weight_cast_reference(
            theta_master,
            weight_cols=weight_cols,
            block_size=cast_group_size,
            stochastic=stochastic_cast,
            dither_coef=dither_coef,
            generator=gen,
        )
        residual = theta_master - theta_model
        residual_abs = residual.abs()
        p50, p99, p999 = _quantiles(residual_abs, (0.5, 0.99, 0.999))
        metrics.append(
            {
                "step": float(step),
                "lr": lr,
                "master_resid_rms": _rms(residual),
                "master_resid_p50": p50,
                "master_resid_p99": p99,
                "master_resid_p999": p999,
            }
        )
        residuals.append(residual.detach())

        if quantized_state:
            m, v = _maybe_quantize_state(m, v, group_size, moment_quantizer)

    return metrics, residuals


def _simulate_eco_method(
    *,
    method: Method,
    initial_theta: torch.Tensor,
    master_metrics: list[dict[str, float]],
    steps: int,
    base_lr: float,
    warmup_steps: int,
    beta1: float,
    beta2: float,
    eps: float,
    curvature: float,
    grad_noise_std: float,
    group_size: int,
    cast_group_size: int,
    weight_cols: int,
    dither_coef: float,
    quantized_state: bool,
    stochastic_cast: bool,
    moment_quantizer: str,
    bad_ratio_threshold: float,
    ratio_residual_quantile: float,
    virtual_params: int,
    seed: int,
) -> tuple[list[StepMetrics], FinalDistribution]:
    device = str(initial_theta.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    theta = initial_theta.clone()
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    side_residual = torch.zeros_like(theta)
    carrier_residual_i8 = torch.zeros(theta.shape, device=theta.device, dtype=torch.int8)
    carrier_residual_scales = torch.zeros(
        theta.numel() // group_size, device=theta.device, dtype=torch.float16
    )

    rows: list[StepMetrics] = []
    final_distribution: FinalDistribution | None = None

    for step in range(1, steps + 1):
        lr = _lr_at_step(step, base_lr=base_lr, warmup_steps=warmup_steps)
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step

        grad = curvature * theta
        if grad_noise_std > 0.0:
            grad = grad + torch.randn(
                theta.numel(), generator=gen, device=theta.device
            ) * grad_noise_std

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad.square()

        denom = torch.sqrt(v / bc2).add_(eps)
        update = (m / bc1) / denom
        param_base = theta + side_residual if method.qa_mode == "side_residual" else theta
        pre_cast = param_base - lr * update
        post_cast = _nvfp4_2d_weight_cast_reference(
            pre_cast,
            weight_cols=weight_cols,
            block_size=cast_group_size,
            stochastic=stochastic_cast,
            dither_coef=dither_coef,
            generator=gen,
        )
        residual = pre_cast - post_cast

        if quantized_state:
            m_before, scales_before, adam_saturation_frac = _quantize_moment_with_stats(
                m, group_size, moment_quantizer
            )
            v_stored = _quantize_var_sqrt(v, group_size)
        else:
            m_before = m
            v_stored = v
            adam_saturation_frac = 0.0
            scales_before = torch.ones(
                math.ceil(m_before.numel() / group_size),
                device=m_before.device,
                dtype=torch.float32,
            )
        denom_inject = torch.sqrt(v_stored / bc2).add_(eps)

        alpha = _eco_alpha_for_method(
            method, bc1=bc1, lr=lr, base_lr=base_lr, beta1=beta1
        )
        if method.qa_mode == "side_residual":
            delta_m = torch.zeros_like(residual)
            intended_replay = residual
            stored_replay = residual
            m_stored = m_before
            if quantized_state:
                scales_after = scales_before
                saturation_frac = adam_saturation_frac
            else:
                scales_after = torch.ones_like(scales_before)
                saturation_frac = 0.0
            scale_inflation = torch.ones_like(scales_before)
        else:
            delta_m = alpha * denom_inject * residual
            if quantized_state and alpha != 0.0:
                if method.qa_mode == "carrier_residual":
                    carrier_residual_deq = _dequantize_residual_state(
                        carrier_residual_i8,
                        carrier_residual_scales,
                        group_size,
                        method.residual_quant,
                    )
                    delta_m = delta_m + carrier_residual_deq
                    full_delta_m = delta_m
                    inject_delta_m = _apply_scale_preserve_budget(
                        m,
                        full_delta_m,
                        scales_before,
                        group_size=group_size,
                        scale_budget=method.scale_budget,
                    )
                    deferred_delta_m = full_delta_m - inject_delta_m
                    delta_m = inject_delta_m
                elif method.qa_mode == "scale_preserve":
                    delta_m = _apply_scale_preserve_budget(
                        m,
                        delta_m,
                        scales_before,
                        group_size=group_size,
                        scale_budget=method.scale_budget,
                    )
                elif method.qa_mode == "stored_replay":
                    delta_m = _apply_stored_replay_aware_boost(
                        m,
                        delta_m,
                        group_size=group_size,
                        max_boost=method.max_boost,
                        moment_quantizer=moment_quantizer,
                    )
                    delta_m = _apply_scale_preserve_budget(
                        m,
                        delta_m,
                        scales_before,
                        group_size=group_size,
                        scale_budget=method.scale_budget,
                    )
                elif method.qa_mode == "grid_clip":
                    delta_m = _apply_grid_clip(
                        delta_m,
                        scales_before,
                        group_size=group_size,
                        min_steps=method.grid_min_steps,
                    )
                    delta_m = _apply_scale_preserve_budget(
                        m,
                        delta_m,
                        scales_before,
                        group_size=group_size,
                        scale_budget=method.scale_budget,
                    )
                elif method.qa_mode == "capacity_project":
                    delta_m = _apply_capacity_projection(
                        m_before,
                        delta_m,
                        residual,
                        denom_inject,
                        scales_before,
                        group_size=group_size,
                        lr=lr,
                        bc1=bc1,
                        beta1=beta1,
                        scale_budget=method.scale_budget,
                        replay_error_budget=method.projection_error_budget,
                        projection_steps=method.projection_steps,
                        projection_metric=method.projection_metric,
                        moment_quantizer=moment_quantizer,
                    )

            delta_m = _apply_group_budget(
                delta_m,
                scales_before,
                group_size=group_size,
                budget=method.group_budget,
            )
            intended_replay = (
                -(lr / bc1) * (beta1 / (1.0 - beta1)) * delta_m / denom_inject
                if alpha != 0.0
                else torch.zeros_like(residual)
            )

            m_intended = m_before + delta_m
            if quantized_state and alpha != 0.0:
                m_stored, scales_after, saturation_frac = _quantize_moment_with_stats(
                    m_intended, group_size, moment_quantizer
                )
                scale_inflation = scales_after / scales_before.clamp_min(1e-30)
            elif quantized_state:
                m_stored = m_before
                scales_after = scales_before
                scale_inflation = torch.ones_like(scales_before)
                saturation_frac = adam_saturation_frac
            else:
                m_stored = m_intended
                scales_after = torch.ones_like(scales_before)
                scale_inflation = torch.ones_like(scales_before)
                saturation_frac = 0.0

            stored_delta_m = m_stored - m_before
            stored_replay = (
                -(lr / bc1) * (beta1 / (1.0 - beta1)) * stored_delta_m / denom_inject
                if alpha != 0.0
                else torch.zeros_like(residual)
            )
            if method.qa_mode == "carrier_residual":
                carrier_residual_f32 = deferred_delta_m + (delta_m - stored_delta_m)
                carrier_residual_i8, carrier_residual_scales = _quantize_residual_state(
                    carrier_residual_f32,
                    group_size,
                    method.residual_quant,
                )

        residual_abs = residual.abs()
        intended_abs = intended_replay.abs()
        stored_abs = stored_replay.abs()
        intended_ratio = _safe_ratio(intended_replay, residual)
        stored_ratio = _safe_ratio(stored_replay, residual)
        intended_abs_ratio = intended_ratio.abs()
        stored_abs_ratio = stored_ratio.abs()

        p_bad = float((stored_abs_ratio > bad_ratio_threshold).float().mean().item())
        material_floor = torch.quantile(residual_abs.detach().float(), ratio_residual_quantile)
        material_mask = residual_abs >= material_floor.clamp_min(1e-30)
        material_ratio = stored_abs_ratio[material_mask]
        material_p99, material_p999 = _quantiles(material_ratio, (0.99, 0.999))
        p_bad_material = (
            float((material_ratio > bad_ratio_threshold).float().mean().item())
            if material_ratio.numel()
            else 0.0
        )
        eco_p999 = _quantiles(residual_abs, (0.999,))[0]
        intended_p50, intended_p99, intended_p999 = _quantiles(
            intended_abs_ratio, (0.5, 0.99, 0.999)
        )
        stored_p50, stored_p99, stored_p999 = _quantiles(
            stored_abs_ratio, (0.5, 0.99, 0.999)
        )
        mom_p999, mom_max = _quantiles(m_stored.abs(), (0.999, 1.0))
        scale_p99 = _quantiles(scales_after.detach().float(), (0.99,))[0]
        inflation_p99 = _quantiles(scale_inflation.detach().float(), (0.99,))[0]
        if method.qa_mode == "carrier_residual":
            carrier_residual_for_metrics = _dequantize_residual_state(
                carrier_residual_i8,
                carrier_residual_scales,
                group_size,
                method.residual_quant,
            )
            carrier_resid_p999 = _quantiles(carrier_residual_for_metrics.abs(), (0.999,))[0]
            carrier_resid_scale_p99 = _quantiles(
                carrier_residual_scales.detach().float(), (0.99,)
            )[0]
        else:
            carrier_resid_p999 = 0.0
            carrier_resid_scale_p99 = 0.0

        master = master_metrics[step - 1]
        rows.append(
            StepMetrics(
                step=step,
                method=method.name,
                lr=lr,
                master_resid_rms=master["master_resid_rms"],
                master_resid_p99=master["master_resid_p99"],
                master_resid_p999=master["master_resid_p999"],
                eco_resid_rms=_rms(residual),
                eco_resid_p999=eco_p999,
                intended_replay_rms=_rms(intended_replay),
                intended_replay_p999=_quantiles(intended_abs, (0.999,))[0],
                stored_replay_rms=_rms(stored_replay),
                stored_replay_p999=_quantiles(stored_abs, (0.999,))[0],
                intended_energy_gain=_energy_gain(intended_replay, residual),
                stored_energy_gain=_energy_gain(stored_replay, residual),
                intended_abs_ratio_p50=intended_p50,
                intended_abs_ratio_p99=intended_p99,
                intended_abs_ratio_p999=intended_p999,
                stored_abs_ratio_p50=stored_p50,
                stored_abs_ratio_p99=stored_p99,
                stored_abs_ratio_p999=stored_p999,
                material_abs_ratio_p99=material_p99,
                material_abs_ratio_p999=material_p999,
                p_bad_ratio=p_bad,
                p_bad_material_ratio=p_bad_material,
                p_any_bad_345b=_p_any(p_bad, virtual_params),
                p_any_bad_material_345b=_p_any(p_bad_material, virtual_params),
                mom_abs_p999=mom_p999,
                mom_abs_max=mom_max,
                mom_scale_p99=scale_p99,
                mom_scale_inflation_p99=inflation_p99,
                mom_saturation_frac=saturation_frac,
                carrier_resid_p999=carrier_resid_p999,
                carrier_resid_scale_p99=carrier_resid_scale_p99,
            )
        )

        if step == steps:
            # Keep a bounded number of final samples for plotting.
            keep = min(262_144, residual.numel())
            idx = torch.linspace(
                0, residual.numel() - 1, keep, device=residual.device
            ).long()
            final_distribution = FinalDistribution(
                method=method.name,
                stored_ratio=stored_ratio[idx].detach().cpu(),
                intended_ratio=intended_ratio[idx].detach().cpu(),
                stored_replay_abs=stored_abs[idx].detach().cpu(),
                residual_abs=residual_abs[idx].detach().cpu(),
            )

        m = m_stored
        v = v_stored
        side_residual = residual if method.qa_mode == "side_residual" else side_residual
        theta = post_cast

    assert final_distribution is not None
    return rows, final_distribution


def _write_timeseries(path: Path, rows: list[StepMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(StepMetrics.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def _write_final_summary(path: Path, rows: list[StepMetrics]) -> list[StepMetrics]:
    final_rows: list[StepMetrics] = []
    seen: set[str] = set()
    for row in reversed(rows):
        if row.method in seen:
            continue
        seen.add(row.method)
        final_rows.append(row)
    final_rows.reverse()

    fieldnames = list(StepMetrics.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in final_rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})
    return final_rows


def _plot_outputs(
    *,
    output_dir: Path,
    rows: list[StepMetrics],
    distributions: list[FinalDistribution],
    bad_ratio_threshold: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 18,
        }
    )

    by_method: dict[str, list[StepMetrics]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)

    colors = {
        "off": "#6b7280",
        "paper_1_over_lr": "#dc2626",
        "warmup_floor": "#f97316",
        "warmup_floor_group4": "#7c3aed",
        "warmup_floor_group2": "#2563eb",
        "qa_scale4": "#7c3aed",
        "qa_stored4": "#0891b2",
        "qa_grid4": "#be123c",
        "proj_s2_e25": "#0f766e",
        "proj_s4_e25": "#0284c7",
        "proj_s2_e10": "#4338ca",
        "proj_s2_e100": "#65a30d",
        "proj_s4_e100": "#ca8a04",
        "projg_s2_e25": "#9333ea",
        "projg_s4_e25": "#c026d3",
        "carrier_i8": "#16a34a",
        "side_resid": "#111827",
        "no_inv_lr": "#059669",
    }
    styles = {
        "off": {"linestyle": ":", "linewidth": 2.0, "alpha": 0.75},
        "paper_1_over_lr": {"linestyle": "-", "linewidth": 2.4, "alpha": 0.85},
        "warmup_floor": {"linestyle": "-", "linewidth": 2.4, "alpha": 0.85},
        "warmup_floor_group4": {"linestyle": "--", "linewidth": 3.6, "alpha": 1.0},
        "warmup_floor_group2": {"linestyle": "-.", "linewidth": 3.0, "alpha": 1.0},
        "qa_scale4": {"linestyle": "--", "linewidth": 3.2, "alpha": 1.0},
        "qa_stored4": {"linestyle": "-.", "linewidth": 3.0, "alpha": 1.0},
        "qa_grid4": {"linestyle": (0, (5, 2, 1, 2)), "linewidth": 3.0, "alpha": 1.0},
        "proj_s2_e25": {"linestyle": "-", "linewidth": 3.4, "alpha": 1.0},
        "proj_s4_e25": {"linestyle": "--", "linewidth": 3.2, "alpha": 1.0},
        "proj_s2_e10": {"linestyle": "-.", "linewidth": 3.0, "alpha": 1.0},
        "proj_s2_e100": {"linestyle": (0, (3, 1, 1, 1)), "linewidth": 3.0, "alpha": 1.0},
        "proj_s4_e100": {"linestyle": (0, (7, 2)), "linewidth": 3.0, "alpha": 1.0},
        "projg_s2_e25": {"linestyle": "-", "linewidth": 3.2, "alpha": 1.0},
        "projg_s4_e25": {"linestyle": "--", "linewidth": 3.2, "alpha": 1.0},
        "carrier_i8": {"linestyle": "-", "linewidth": 3.4, "alpha": 1.0},
        "side_resid": {"linestyle": ":", "linewidth": 3.0, "alpha": 0.95},
        "no_inv_lr": {"linestyle": ":", "linewidth": 2.4, "alpha": 0.85},
    }
    method_order = [
        "off",
        "no_inv_lr",
        "paper_1_over_lr",
        "warmup_floor",
        "qa_scale4",
        "qa_stored4",
        "qa_grid4",
        "proj_s2_e25",
        "proj_s4_e25",
        "proj_s2_e10",
        "proj_s2_e100",
        "proj_s4_e100",
        "projg_s2_e25",
        "projg_s4_e25",
        "carrier_i8",
        "side_resid",
        "warmup_floor_group2",
        "warmup_floor_group4",
    ]
    endpoint_marker_methods = {
        "warmup_floor_group4",
        "warmup_floor_group2",
        "qa_scale4",
        "qa_stored4",
        "qa_grid4",
        "proj_s2_e25",
        "proj_s4_e25",
        "proj_s2_e10",
        "proj_s2_e100",
        "proj_s4_e100",
        "projg_s2_e25",
        "projg_s4_e25",
        "carrier_i8",
        "side_resid",
    }

    def line(
        ax,
        y_attr: str,
        *,
        title: str,
        ylabel: str,
        logy: bool = False,
        include_legend: bool = False,
    ) -> None:
        for method in method_order:
            if method not in by_method:
                continue
            method_rows = by_method[method]
            xs = [r.step for r in method_rows]
            ys = [getattr(r, y_attr) for r in method_rows]
            ax.plot(
                xs,
                ys,
                label=method,
                color=colors.get(method),
                marker="o" if method in endpoint_marker_methods else None,
                markevery=[-1] if method in endpoint_marker_methods else None,
                markersize=5,
                **styles.get(method, {}),
            )
        ax.set_title(title, pad=10)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        if include_legend:
            ax.legend(loc="best", frameon=False)

    def save(fig, stem: str, *, top: float = 0.90) -> None:
        handles, labels = [], []
        for ax in fig.axes:
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            handles.extend(ax_handles)
            labels.extend(ax_labels)
        label_to_handle = dict(zip(labels, handles))
        if label_to_handle:
            fig.legend(
                label_to_handle.values(),
                label_to_handle.keys(),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncol=min(5, len(label_to_handle)),
                frameon=False,
            )
        fig.tight_layout(rect=[0.02, 0.02, 0.98, top])
        fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Master residual is method-independent; draw once plus ECO residuals.
    first_method_rows = next(iter(by_method.values()))

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=180)
    axes = axes.flatten()
    axes[0].plot(
        [r.step for r in first_method_rows],
        [r.master_resid_p999 for r in first_method_rows],
        color="black",
        linewidth=2.5,
        label="master residual target",
    )
    for method in method_order:
        if method not in by_method:
            continue
        method_rows = by_method[method]
        axes[0].plot(
            [r.step for r in method_rows],
            [r.eco_resid_p999 for r in method_rows],
            color=colors.get(method),
            marker="o" if method in endpoint_marker_methods else None,
            markevery=[-1] if method in endpoint_marker_methods else None,
            markersize=5,
            label=f"{method} residual",
            **styles.get(method, {}),
        )
    axes[0].set_title("P99.9 Quantization Residual", pad=10)
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("|residual|")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)

    line(
        axes[1],
        "intended_energy_gain",
        title="Intended ECO Replay / Residual Energy",
        ylabel="dot(replay, residual) / ||residual||^2",
    )
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.5, label="master target")
    axes[1].axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    line(
        axes[2],
        "stored_energy_gain",
        title="Stored ECO Replay After Moment Quantization",
        ylabel="dot(replay, residual) / ||residual||^2",
    )
    axes[2].axhline(1.0, color="black", linestyle=":", linewidth=1.5, label="master target")
    axes[2].axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    line(
        axes[3],
        "material_abs_ratio_p999",
        title="P99.9 |Stored Replay / Residual| for Material Residuals",
        ylabel="ratio",
        logy=True,
    )
    axes[3].axhline(1.0, color="black", linestyle=":", linewidth=1.5)
    axes[3].axhline(bad_ratio_threshold, color="red", linestyle=":", linewidth=1.2)
    fig.suptitle("ECO Master-Weight Replay Equivalence", fontweight="bold", y=1.035)
    save(fig, "eco_master_equivalence_dashboard", top=0.88)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=180)
    axes = axes.flatten()
    line(
        axes[0],
        "stored_replay_p999",
        title="P99.9 Stored Replay in Parameter Units",
        ylabel="|parameter replay|",
        logy=True,
    )
    axes[0].plot(
        [r.step for r in first_method_rows],
        [r.master_resid_p999 for r in first_method_rows],
        color="black",
        linewidth=2.0,
        linestyle=":",
        label="master residual target",
    )

    line(
        axes[1],
        "mom_scale_inflation_p99",
        title="P99 Moment-Scale Inflation",
        ylabel="scale_after / scale_before",
        logy=True,
    )
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.5)

    line(
        axes[2],
        "p_any_bad_material_345b",
        title=f"P(any material ratio > {bad_ratio_threshold:g}) at 345B",
        ylabel="probability",
    )
    axes[2].set_ylim(-0.03, 1.03)

    line(
        axes[3],
        "mom_abs_p999",
        title="P99.9 |First Moment|",
        ylabel="|m| after injection",
        logy=True,
    )
    fig.suptitle("ECO Moment-State Health", fontweight="bold", y=1.035)
    save(fig, "eco_master_equivalence_moment_health", top=0.88)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=180)
    line(
        axes[0],
        "mom_saturation_frac",
        title="Moment Quantizer Saturation Fraction",
        ylabel="fraction",
        logy=True,
    )
    line(
        axes[1],
        "stored_abs_ratio_p999",
        title="Raw P99.9 |Stored Replay / Residual|",
        ylabel="ratio, includes tiny residuals",
        logy=True,
    )
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.5)
    axes[1].axhline(bad_ratio_threshold, color="red", linestyle=":", linewidth=1.2)
    fig.suptitle("ECO Tail Diagnostics", fontweight="bold", y=1.05)
    save(fig, "eco_master_equivalence_tail_diagnostics", top=0.82)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=180)
    axes = axes.flatten()
    focused_methods = [
        "paper_1_over_lr",
        "warmup_floor",
        "qa_scale4",
        "qa_stored4",
        "qa_grid4",
        "proj_s2_e25",
        "proj_s4_e25",
        "proj_s2_e10",
        "proj_s2_e100",
        "proj_s4_e100",
        "projg_s2_e25",
        "projg_s4_e25",
        "carrier_i8",
        "side_resid",
    ]
    focused_attrs = [
        (
            "stored_energy_gain",
            "Stored Replay Energy Gain",
            "1.0 means master-weight-equivalent",
            False,
            1.0,
        ),
        (
            "material_abs_ratio_p999",
            "P99.9 Replay / Residual",
            "material residual coords only",
            True,
            1.0,
        ),
        (
            "stored_replay_p999",
            "P99.9 Stored Replay",
            "parameter units",
            True,
            None,
        ),
        (
            "mom_scale_inflation_p99",
            "P99 Moment-Scale Inflation",
            "scale_after / scale_before",
            True,
            1.0,
        ),
    ]
    for ax, (attr, title, ylabel, logy, hline) in zip(axes, focused_attrs):
        for method in focused_methods:
            if method not in by_method:
                continue
            method_rows = by_method[method]
            xs = [r.step for r in method_rows]
            ys = [getattr(r, attr) for r in method_rows]
            ax.plot(
                xs,
                ys,
                label=method,
                color=colors.get(method),
                marker="o"
                if method in {
                    "qa_scale4",
                    "qa_stored4",
                    "qa_grid4",
                    "proj_s2_e25",
                    "proj_s4_e25",
                    "proj_s2_e10",
                    "proj_s2_e100",
                    "proj_s4_e100",
                    "projg_s2_e25",
                    "projg_s4_e25",
                    "carrier_i8",
                    "side_resid",
                }
                else None,
                markevery=[-1]
                if method in {
                    "qa_scale4",
                    "qa_stored4",
                    "qa_grid4",
                    "proj_s2_e25",
                    "proj_s4_e25",
                    "proj_s2_e10",
                    "proj_s2_e100",
                    "proj_s4_e100",
                    "projg_s2_e25",
                    "projg_s4_e25",
                    "carrier_i8",
                    "side_resid",
                }
                else None,
                markersize=6,
                **styles.get(method, {}),
            )
        if attr == "stored_replay_p999":
            ax.plot(
                [r.step for r in first_method_rows],
                [r.master_resid_p999 for r in first_method_rows],
                color="black",
                linestyle=":",
                linewidth=2.0,
                label="master target",
            )
        if hline is not None:
            ax.axhline(hline, color="black", linestyle=":", linewidth=1.3)
        ax.set_title(title, pad=10)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.set_xlim(0, max(r.step for r in first_method_rows))
        ax.grid(True, alpha=0.25)
    fig.suptitle("Focused Full Run Comparison (max step = 1200)", fontweight="bold", y=1.035)
    save(fig, "eco_master_equivalence_full1200_focused", top=0.88)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=180)
    line(
        axes[0],
        "carrier_resid_p999",
        title="P99.9 Carrier Residual Debt",
        ylabel="|dequantized int8 residual debt|",
        logy=True,
    )
    line(
        axes[1],
        "carrier_resid_scale_p99",
        title="P99 Carrier Residual Scale",
        ylabel="fp16 group scale",
        logy=True,
    )
    fig.suptitle("Carrier Residual State (Persistent INT8 + FP16 Scales)", fontweight="bold", y=1.05)
    save(fig, "eco_master_equivalence_carrier_residual", top=0.82)

    fig, axes = plt.subplots(2, 2, figsize=(19, 12), dpi=180)
    axes = axes.flatten()
    bins = torch.logspace(-8, 4, 180).numpy()
    ratio_bins = torch.logspace(-8, 8, 220).numpy()

    def positive_numpy(x: torch.Tensor) -> np.ndarray:
        arr = x.detach().float().cpu().numpy()
        return arr[np.isfinite(arr) & (arr > 0.0)]

    for dist in distributions:
        color = colors.get(dist.method)
        values = [
            (axes[0], positive_numpy(dist.residual_abs), bins),
            (axes[1], positive_numpy(dist.stored_replay_abs), bins),
            (axes[2], positive_numpy(dist.intended_ratio.abs()), ratio_bins),
            (axes[3], positive_numpy(dist.stored_ratio.abs()), ratio_bins),
        ]
        for ax, data, data_bins in values:
            if data.size == 0:
                continue
            ax.hist(
                data,
                bins=data_bins,
                histtype="step",
                density=False,
                linewidth=1.8,
                color=color,
                label=dist.method,
            )

    titles = [
        "Final |Quantization Residual|",
        "Final |Stored ECO Replay|",
        "Final |Intended Replay / Residual|",
        "Final |Stored Replay / Residual|",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, pad=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("sample count")
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        "Final-Step ECO Equivalence Distributions",
        fontweight="bold",
        y=1.035,
    )
    save(fig, "eco_master_equivalence_final_distributions", top=0.88)


def _print_summary(rows: list[StepMetrics]) -> None:
    final = _write_final_summary(Path("/tmp/eco_master_equivalence_summary.csv"), rows)
    header = (
        "method",
        "stored_energy_gain",
        "stored_ratio_p999",
        "stored_replay_p999",
        "master_resid_p999",
        "mom_scale_infl_p99",
        "p_any_bad_345b",
    )
    print("\t".join(header))
    for row in final:
        print(
            "\t".join(
                [
                    row.method,
                    f"{row.stored_energy_gain:.3e}",
                    f"{row.stored_abs_ratio_p999:.3e}",
                    f"{row.stored_replay_p999:.3e}",
                    f"{row.master_resid_p999:.3e}",
                    f"{row.mom_scale_inflation_p99:.3e}",
                    f"{row.p_any_bad_345b:.3e}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--sample-elems", type=int, default=1_048_576)
    parser.add_argument("--base-lr", type=float, default=5e-6)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=980,
        help="31348 warmup samples / GBS32 is about 980 optimizer steps.",
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--curvature", type=float, default=1e-2)
    parser.add_argument("--grad-noise-std", type=float, default=1e-3)
    parser.add_argument("--init-std", type=float, default=2e-2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--cast-group-size",
        type=int,
        default=16,
        help="NVFP4 2D block edge. Production rowwise NVFP4 uses 16.",
    )
    parser.add_argument(
        "--weight-cols",
        type=int,
        default=7168,
        help="Column count used to reshape samples into production-like 2D weight tiles.",
    )
    parser.add_argument(
        "--dither-coef",
        type=float,
        default=float(os.getenv("MEGATRON_NVFP4_SR_DITHER_COEF", "0.25")),
        help="Block-aware NVFP4 stochastic-rounding dither coefficient.",
    )
    parser.add_argument("--deterministic-cast", action="store_true")
    parser.add_argument("--fp32-optimizer-state", action="store_true")
    parser.add_argument(
        "--moment-quantizer",
        choices=("softsign", "linear"),
        default="softsign",
        help="INT8 first-moment carrier companding used by the FlashAdamW state simulation.",
    )
    parser.add_argument("--bad-ratio-threshold", type=float, default=10.0)
    parser.add_argument(
        "--ratio-residual-quantile",
        type=float,
        default=0.5,
        help="Only residuals above this per-step quantile contribute to material ratio charts.",
    )
    parser.add_argument("--virtual-params", type=int, default=345_000_000_000)
    parser.add_argument(
        "--methods",
        default=(
            "off,paper_1_over_lr,warmup_floor,qa_scale4,qa_stored4,"
            "qa_grid4,proj_s2_e25,proj_s4_e25,proj_s2_e10,"
            "proj_s2_e100,proj_s4_e100,projg_s2_e25,projg_s4_e25,"
            "carrier_i8,side_resid,no_inv_lr"
        ),
        help=(
            "Comma list from: off,paper_1_over_lr,warmup_floor,"
            "warmup_floor_group4,warmup_floor_group2,qa_scale4,qa_stored4,"
            "qa_grid4,proj_s2_e25,proj_s4_e25,proj_s2_e10,"
            "proj_s2_e100,proj_s4_e100,projg_s2_e25,projg_s4_e25,"
            "carrier_i8,side_resid,no_inv_lr."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eco_master_equivalence"),
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.sample_elems % args.group_size != 0:
        raise ValueError("--sample-elems must be divisible by --group-size")
    if args.sample_elems % args.cast_group_size != 0:
        raise ValueError("--sample-elems must be divisible by --cast-group-size")
    if args.weight_cols <= 0:
        raise ValueError("--weight-cols must be positive")

    method_map = {
        "off": Method("off", "off"),
        "paper_1_over_lr": Method("paper_1_over_lr", "paper"),
        "warmup_floor": Method("warmup_floor", "paper", lr_floor_fraction=1.0),
        "warmup_floor_group4": Method(
            "warmup_floor_group4", "paper", lr_floor_fraction=1.0, group_budget=4.0
        ),
        "warmup_floor_group2": Method(
            "warmup_floor_group2", "paper", lr_floor_fraction=1.0, group_budget=2.0
        ),
        "qa_scale4": Method(
            "qa_scale4",
            "paper",
            lr_floor_fraction=1.0,
            qa_mode="scale_preserve",
            scale_budget=4.0,
        ),
        "qa_stored4": Method(
            "qa_stored4",
            "paper",
            lr_floor_fraction=1.0,
            qa_mode="stored_replay",
            scale_budget=4.0,
            max_boost=8.0,
        ),
        "qa_grid4": Method(
            "qa_grid4",
            "paper",
            lr_floor_fraction=1.0,
            qa_mode="grid_clip",
            scale_budget=4.0,
            grid_min_steps=0.5,
        ),
        "proj_s2_e25": Method(
            "proj_s2_e25",
            "paper",
            qa_mode="capacity_project",
            scale_budget=2.0,
            projection_error_budget=0.25,
            projection_steps=16,
        ),
        "proj_s4_e25": Method(
            "proj_s4_e25",
            "paper",
            qa_mode="capacity_project",
            scale_budget=4.0,
            projection_error_budget=0.25,
            projection_steps=16,
        ),
        "proj_s2_e10": Method(
            "proj_s2_e10",
            "paper",
            qa_mode="capacity_project",
            scale_budget=2.0,
            projection_error_budget=0.10,
            projection_steps=16,
        ),
        "proj_s2_e100": Method(
            "proj_s2_e100",
            "paper",
            qa_mode="capacity_project",
            scale_budget=2.0,
            projection_error_budget=1.0,
            projection_steps=16,
        ),
        "proj_s4_e100": Method(
            "proj_s4_e100",
            "paper",
            qa_mode="capacity_project",
            scale_budget=4.0,
            projection_error_budget=1.0,
            projection_steps=16,
        ),
        "projg_s2_e25": Method(
            "projg_s2_e25",
            "paper",
            qa_mode="capacity_project",
            scale_budget=2.0,
            projection_error_budget=0.25,
            projection_steps=16,
            projection_metric="gain",
        ),
        "projg_s4_e25": Method(
            "projg_s4_e25",
            "paper",
            qa_mode="capacity_project",
            scale_budget=4.0,
            projection_error_budget=0.25,
            projection_steps=16,
            projection_metric="gain",
        ),
        "carrier_i8": Method(
            "carrier_i8",
            "paper",
            lr_floor_fraction=1.0,
            qa_mode="carrier_residual",
            scale_budget=4.0,
            residual_quant="linear",
        ),
        "side_resid": Method("side_resid", "off", qa_mode="side_residual"),
        "no_inv_lr": Method("no_inv_lr", "no_inv_lr"),
    }
    methods = [method_map[name.strip()] for name in args.methods.split(",") if name.strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_theta = _make_initial_theta(
        sample_elems=args.sample_elems,
        init_std=args.init_std,
        group_size=args.cast_group_size,
        weight_cols=args.weight_cols,
        dither_coef=args.dither_coef,
        device=args.device,
        seed=args.seed,
    )

    quantized_state = not args.fp32_optimizer_state
    master_metrics, _ = _simulate_master_path(
        initial_theta=initial_theta,
        steps=args.steps,
        base_lr=args.base_lr,
        warmup_steps=args.warmup_steps,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        curvature=args.curvature,
        grad_noise_std=args.grad_noise_std,
        group_size=args.group_size,
        cast_group_size=args.cast_group_size,
        weight_cols=args.weight_cols,
        dither_coef=args.dither_coef,
        quantized_state=quantized_state,
        stochastic_cast=not args.deterministic_cast,
        moment_quantizer=args.moment_quantizer,
        seed=args.seed + 10_000,
    )

    all_rows: list[StepMetrics] = []
    distributions: list[FinalDistribution] = []
    for method in methods:
        rows, dist = _simulate_eco_method(
            method=method,
            initial_theta=initial_theta,
            master_metrics=master_metrics,
            steps=args.steps,
            base_lr=args.base_lr,
            warmup_steps=args.warmup_steps,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            curvature=args.curvature,
            grad_noise_std=args.grad_noise_std,
            group_size=args.group_size,
            cast_group_size=args.cast_group_size,
            weight_cols=args.weight_cols,
            dither_coef=args.dither_coef,
            quantized_state=quantized_state,
            stochastic_cast=not args.deterministic_cast,
            moment_quantizer=args.moment_quantizer,
            bad_ratio_threshold=args.bad_ratio_threshold,
            ratio_residual_quantile=args.ratio_residual_quantile,
            virtual_params=args.virtual_params,
            seed=args.seed + 10_000,
        )
        all_rows.extend(rows)
        distributions.append(dist)

    timeseries_path = args.output_dir / "eco_master_equivalence_timeseries.csv"
    final_path = args.output_dir / "eco_master_equivalence_final_summary.csv"
    _write_timeseries(timeseries_path, all_rows)
    final_rows = _write_final_summary(final_path, all_rows)
    _plot_outputs(
        output_dir=args.output_dir,
        rows=all_rows,
        distributions=distributions,
        bad_ratio_threshold=args.bad_ratio_threshold,
    )

    print(f"Wrote {timeseries_path}")
    print(f"Wrote {final_path}")
    print(f"Wrote {args.output_dir / 'eco_master_equivalence_dashboard.png'}")
    print(f"Wrote {args.output_dir / 'eco_master_equivalence_final_distributions.png'}")
    print()
    print(
        "\t".join(
            [
                "method",
                "stored_gain",
                "material_ratio_p999",
                "replay_p999",
                "target_p999",
                "scale_infl_p99",
                "p_any_bad_material_345b",
            ]
        )
    )
    for row in final_rows:
        print(
            "\t".join(
                [
                    row.method,
                    f"{row.stored_energy_gain:.3e}",
                    f"{row.material_abs_ratio_p999:.3e}",
                    f"{row.stored_replay_p999:.3e}",
                    f"{row.master_resid_p999:.3e}",
                    f"{row.mom_scale_inflation_p99:.3e}",
                    f"{row.p_any_bad_material_345b:.3e}",
                ]
            )
        )


if __name__ == "__main__":
    main()
