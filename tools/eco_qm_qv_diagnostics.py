#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""ECO diagnostics for independently quantized Adam moments.

This is a Monte Carlo diagnostic for FlashAdamW ECO, not a trainer.  It asks:

* does ECO still behave like master-weight residual replay once exp_avg is
  stored as softsign INT8 groups?
* does quantizing exp_avg_sq as sqrt(v) UINT8 materially distort the ECO
  denominator?
* did lr flooring/projection preserve the intended carrier behavior or merely
  suppress ECO during warmup?

The simulation mirrors the relevant production order:

1. update Adam m/v from the gradient;
2. apply the Adam parameter update and NVFP4 model-weight cast;
3. store m/v through the FlashAdamW quantizers;
4. inject ECO residual into the stored first moment and store m again.

It extrapolates rare bad-coordinate probabilities to the model scale with
``1 - exp(-N * p)``.  That extrapolation is intentionally conservative: even a
small bad-coordinate rate is unacceptable at hundreds of billions of params.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eco_master_equivalence import (  # noqa: E402
    _lr_at_step,
    _nvfp4_2d_weight_cast_reference,
    _quantize_moment_no_stats,
    _quantize_moment_with_stats,
)


@dataclass(frozen=True)
class EcoCase:
    name: str
    alpha_mode: str = "floor"
    q_m: bool = True
    q_v: bool = True
    projection: bool = True
    qv_min_code: int = 0
    denom_floor_abs: float = 0.0
    scale_budget: float = 2.0
    gain_budget: float = 0.25
    projection_steps: int = 16


def _quantiles(x: torch.Tensor, qs: Iterable[float]) -> list[float]:
    if x.numel() == 0:
        return [float("nan") for _ in qs]
    q_tensor = torch.tensor(list(qs), device=x.device, dtype=torch.float32)
    values = torch.quantile(x.detach().float(), q_tensor)
    return [float(v.item()) for v in values]


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.detach().float().square())).item())


def _p_any(frac: float, virtual_params: int) -> float:
    if frac <= 0.0:
        return 0.0
    return 1.0 - math.exp(-min(700.0, frac * virtual_params))


def _safe_ratio(num: torch.Tensor, denom: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    out = torch.zeros_like(num, dtype=torch.float32)
    mask = denom.abs() > eps
    out[mask] = num[mask].float() / denom[mask].float()
    return out


def _energy_gain(replay: torch.Tensor, residual: torch.Tensor) -> float:
    denom = residual.float().square().sum().clamp_min(1e-30)
    return float((replay.float() * residual.float()).sum().div(denom).item())


def _group_absmax(x: torch.Tensor, group_size: int) -> torch.Tensor:
    return x.view(-1, group_size).abs().amax(dim=1).clamp_min(1e-12)


def _quantize_v_sqrt(
    v: torch.Tensor,
    group_size: int,
    *,
    min_code: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Production-style sqrt(v) UINT8 group quantization.

    ``min_code`` is diagnostic only.  ``0`` matches the current storage.  ``1``
    tests whether preserving a nonzero code for nonzero sqrt(v) fixes denominator
    collapse without adding fp32/bf16 residual storage.
    """

    view = v.clamp_min(0.0).sqrt().view(-1, group_size)
    scale = view.amax(dim=1, keepdim=True).clamp_min(1e-30)
    packed = torch.floor((view / scale).clamp(0.0, 1.0) * 255.0 + 0.5).clamp(
        0.0, 255.0
    )
    if min_code > 0:
        floor = torch.full_like(packed, float(min_code))
        packed = torch.where(view > 0.0, torch.maximum(packed, floor), packed)
    zero_frac = float(((packed == 0.0) & (view > 0.0)).float().mean().item())
    scale_f16 = scale.squeeze(1).to(torch.float16)
    recovered_sqrt = packed / 255.0 * scale_f16.float().view(-1, 1)
    recovered = recovered_sqrt.square().reshape_as(v)
    scale_p99 = _quantiles(scale_f16.float(), (0.99,))[0]
    return recovered, scale_f16.float(), zero_frac, scale_p99


def _eco_alpha(
    case: EcoCase,
    *,
    bc1: float,
    lr: float,
    base_lr: float,
    beta1: float,
) -> float:
    if case.alpha_mode == "off":
        return 0.0
    if case.alpha_mode == "no_inv_lr":
        return bc1 * (1.0 - 1.0 / beta1)
    if case.alpha_mode == "paper":
        effective_lr = lr
    elif case.alpha_mode == "floor":
        effective_lr = max(lr, base_lr)
    else:
        raise ValueError(f"unknown alpha mode: {case.alpha_mode}")
    if effective_lr == 0.0:
        return 0.0
    return (bc1 / effective_lr) * (1.0 - 1.0 / beta1)


def _project_qm_gain(
    m_before: torch.Tensor,
    delta_m: torch.Tensor,
    denom: torch.Tensor,
    group_scale: torch.Tensor,
    *,
    group_size: int,
    scale_budget: float,
    gain_budget: float,
    projection_steps: int,
    moment_quantizer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the Triton projection: bound stored replay gain and scale growth."""

    if projection_steps < 1:
        return torch.zeros_like(delta_m), torch.zeros(
            m_before.numel() // group_size, device=m_before.device
        )

    m_view = m_before.view(-1, group_size)
    delta_view = delta_m.view(-1, group_size)
    denom_view = denom.view(-1, group_size)
    old_scale = group_scale.view(-1, 1).float().clamp_min(1e-12)

    chosen = torch.zeros_like(old_scale)
    found = torch.zeros_like(old_scale, dtype=torch.bool)

    for idx in range(projection_steps):
        frac = 1.0 - (idx / projection_steps)
        candidate_delta = delta_view * frac
        target_replay = candidate_delta / denom_view
        target_energy = target_replay.float().square().sum(dim=1, keepdim=True)
        candidate = (m_view + candidate_delta).reshape_as(m_before)
        candidate_q, new_scale, _ = _quantize_moment_no_stats(
            candidate, group_size, moment_quantizer
        )
        stored_delta = candidate_q.view(-1, group_size) - m_view
        stored_replay = stored_delta / denom_view
        gain = (stored_replay.float() * target_replay.float()).sum(
            dim=1, keepdim=True
        ) / target_energy.clamp_min(1e-30)
        scale_inflation = new_scale.view(-1, 1).float() / old_scale
        ok = (
            (target_energy > 1e-30)
            & (scale_inflation <= scale_budget)
            & ((gain - 1.0).abs() <= gain_budget)
        )
        take = ok & ~found
        chosen = torch.where(take, torch.full_like(chosen, frac), chosen)
        found = found | ok

    return (delta_view * chosen).reshape_as(delta_m), chosen.squeeze(1)


def _make_initial_theta(
    *,
    sample_elems: int,
    init_std: float,
    cast_group_size: int,
    weight_cols: int,
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
        block_size=cast_group_size,
        stochastic=False,
        dither_coef=0.0,
        generator=gen,
    )


def _simulate_case(
    *,
    case: EcoCase,
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
    virtual_params: int,
    bad_ratio_threshold: float,
    material_quantile: float,
    moment_quantizer: str,
    seed: int,
) -> list[dict[str, float | str | int]]:
    gen = torch.Generator(device=str(initial_theta.device))
    gen.manual_seed(seed)

    theta = initial_theta.clone()
    m_state = torch.zeros_like(theta)
    v_state = torch.zeros_like(theta)
    rows: list[dict[str, float | str | int]] = []

    for step in range(1, steps + 1):
        lr = _lr_at_step(step, base_lr=base_lr, warmup_steps=warmup_steps)
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step

        grad = curvature * theta
        if grad_noise_std > 0.0:
            grad = grad + torch.randn(theta.numel(), generator=gen, device=theta.device) * grad_noise_std

        m_work = beta1 * m_state + (1.0 - beta1) * grad
        v_work = beta2 * v_state + (1.0 - beta2) * grad.square()

        denom_update = torch.sqrt(v_work / bc2).add_(eps)
        update = (m_work / bc1) / denom_update
        pre_cast = theta - lr * update
        post_cast = _nvfp4_2d_weight_cast_reference(
            pre_cast,
            weight_cols=weight_cols,
            block_size=cast_group_size,
            stochastic=False,
            dither_coef=0.0,
            generator=gen,
        )
        residual = pre_cast - post_cast
        theta = post_cast

        if case.q_m:
            m_before, m_scales_before, m_sat_before = _quantize_moment_with_stats(
                m_work, group_size, moment_quantizer
            )
        else:
            m_before = m_work
            m_scales_before = _group_absmax(m_work, group_size)
            m_sat_before = 0.0

        if case.q_v:
            v_before, v_scales, qv_zero_frac, qv_scale_p99 = _quantize_v_sqrt(
                v_work, group_size, min_code=case.qv_min_code
            )
        else:
            v_before = v_work
            v_scales = _group_absmax(v_work.clamp_min(0.0).sqrt(), group_size)
            qv_zero_frac = 0.0
            qv_scale_p99 = _quantiles(v_scales, (0.99,))[0]

        denom_inject = torch.sqrt(v_before / bc2).add_(eps)
        if case.denom_floor_abs > 0.0:
            denom_inject = torch.maximum(
                denom_inject,
                torch.tensor(case.denom_floor_abs, device=denom_inject.device),
            )
        denom_ratio = denom_inject / denom_update.clamp_min(1e-30)

        alpha = _eco_alpha(case, bc1=bc1, lr=lr, base_lr=base_lr, beta1=beta1)
        raw_delta_m = alpha * denom_inject * residual
        if case.projection and case.q_m and alpha != 0.0:
            delta_m, chosen_frac = _project_qm_gain(
                m_before,
                raw_delta_m,
                denom_inject,
                m_scales_before,
                group_size=group_size,
                scale_budget=case.scale_budget,
                gain_budget=case.gain_budget,
                projection_steps=case.projection_steps,
                moment_quantizer=moment_quantizer,
            )
        else:
            delta_m = raw_delta_m
            chosen_frac = torch.ones(theta.numel() // group_size, device=theta.device)
            if alpha == 0.0:
                chosen_frac.zero_()

        m_intended = m_before + delta_m
        if case.q_m:
            m_after, m_scales_after, m_sat_after = _quantize_moment_with_stats(
                m_intended, group_size, moment_quantizer
            )
        else:
            m_after = m_intended
            m_scales_after = _group_absmax(m_after, group_size)
            m_sat_after = 0.0

        replay_coef = -(lr / bc1) * (beta1 / (1.0 - beta1))
        intended_replay = replay_coef * delta_m / denom_inject
        stored_replay = replay_coef * (m_after - m_before) / denom_inject
        residual_abs = residual.abs()
        stored_abs_ratio = _safe_ratio(stored_replay, residual).abs()
        intended_abs_ratio = _safe_ratio(intended_replay, residual).abs()
        material_floor = torch.quantile(residual_abs.detach().float(), material_quantile)
        material_mask = residual_abs >= material_floor.clamp_min(1e-30)
        material_ratio = stored_abs_ratio[material_mask]
        p_bad = float((stored_abs_ratio > bad_ratio_threshold).float().mean().item())
        p_bad_material = (
            float((material_ratio > bad_ratio_threshold).float().mean().item())
            if material_ratio.numel()
            else 0.0
        )

        m_scale_inflation = m_scales_after / m_scales_before.clamp_min(1e-30)
        denom_ratio_p001, denom_ratio_p01, denom_ratio_p50, denom_ratio_p99 = _quantiles(
            denom_ratio, (0.001, 0.01, 0.5, 0.99)
        )
        chosen_p01, chosen_p50, chosen_p99 = _quantiles(chosen_frac, (0.01, 0.5, 0.99))
        material_p99, material_p999 = _quantiles(material_ratio, (0.99, 0.999))
        stored_p50, stored_p99, stored_p999 = _quantiles(
            stored_abs_ratio, (0.5, 0.99, 0.999)
        )
        intended_p50, intended_p99, intended_p999 = _quantiles(
            intended_abs_ratio, (0.5, 0.99, 0.999)
        )
        delta_p99, delta_p999, delta_max = _quantiles(raw_delta_m.abs(), (0.99, 0.999, 1.0))
        stored_delta_p99, stored_delta_p999 = _quantiles(
            (m_after - m_before).abs(), (0.99, 0.999)
        )
        scale_inflation_p50, scale_inflation_p99, scale_inflation_max = _quantiles(
            m_scale_inflation, (0.5, 0.99, 1.0)
        )
        denom_p001, denom_p01, denom_p50 = _quantiles(denom_inject, (0.001, 0.01, 0.5))
        update_rms = _rms(update)
        update_p99, update_p999 = _quantiles(update.abs(), (0.99, 0.999))

        rows.append(
            {
                "case": case.name,
                "step": step,
                "lr": lr,
                "alpha": alpha,
                "q_m": int(case.q_m),
                "q_v": int(case.q_v),
                "projection": int(case.projection),
                "qv_min_code": case.qv_min_code,
                "denom_floor_abs": case.denom_floor_abs,
                "loss_proxy": float(theta.float().square().mean().item()),
                "grad_rms": _rms(grad),
                "update_rms": update_rms,
                "update_p99": update_p99,
                "update_p999": update_p999,
                "residual_rms": _rms(residual),
                "residual_p999": _quantiles(residual_abs, (0.999,))[0],
                "intended_replay_rms": _rms(intended_replay),
                "stored_replay_rms": _rms(stored_replay),
                "intended_energy_gain": _energy_gain(intended_replay, residual),
                "stored_energy_gain": _energy_gain(stored_replay, residual),
                "intended_abs_ratio_p50": intended_p50,
                "intended_abs_ratio_p99": intended_p99,
                "intended_abs_ratio_p999": intended_p999,
                "stored_abs_ratio_p50": stored_p50,
                "stored_abs_ratio_p99": stored_p99,
                "stored_abs_ratio_p999": stored_p999,
                "material_abs_ratio_p99": material_p99,
                "material_abs_ratio_p999": material_p999,
                "p_bad_ratio": p_bad,
                "p_bad_material_ratio": p_bad_material,
                "p_any_bad_345b": _p_any(p_bad, virtual_params),
                "p_any_bad_material_345b": _p_any(p_bad_material, virtual_params),
                "raw_delta_m_p99": delta_p99,
                "raw_delta_m_p999": delta_p999,
                "raw_delta_m_max": delta_max,
                "stored_delta_m_p99": stored_delta_p99,
                "stored_delta_m_p999": stored_delta_p999,
                "m_scale_inflation_p50": scale_inflation_p50,
                "m_scale_inflation_p99": scale_inflation_p99,
                "m_scale_inflation_max": scale_inflation_max,
                "m_saturation_frac_before": m_sat_before,
                "m_saturation_frac_after": m_sat_after,
                "qv_zero_frac": qv_zero_frac,
                "qv_scale_p99": qv_scale_p99,
                "denom_p001": denom_p001,
                "denom_p01": denom_p01,
                "denom_p50": denom_p50,
                "denom_ratio_p001": denom_ratio_p001,
                "denom_ratio_p01": denom_ratio_p01,
                "denom_ratio_p50": denom_ratio_p50,
                "denom_ratio_p99": denom_ratio_p99,
                "projection_frac_p01": chosen_p01,
                "projection_frac_p50": chosen_p50,
                "projection_frac_p99": chosen_p99,
            }
        )

        m_state = m_after.detach()
        v_state = v_before.detach()

    return rows


def _default_cases() -> list[EcoCase]:
    return [
        EcoCase("off_qm1_qv1", alpha_mode="off", projection=False),
        EcoCase("paper_qm1_qv1", alpha_mode="paper", projection=False),
        EcoCase("floor_qm1_qv1", alpha_mode="floor", projection=False),
        EcoCase("proj_qm1_qv1", alpha_mode="floor", projection=True),
        EcoCase("proj_qm1_qv0", alpha_mode="floor", q_m=True, q_v=False, projection=True),
        EcoCase("proj_qm0_qv1", alpha_mode="floor", q_m=False, q_v=True, projection=False),
        EcoCase("proj_qm0_qv0", alpha_mode="floor", q_m=False, q_v=False, projection=False),
        EcoCase(
            "proj_qm1_qv1_min_code1",
            alpha_mode="floor",
            q_m=True,
            q_v=True,
            projection=True,
            qv_min_code=1,
        ),
        EcoCase(
            "proj_qm1_qv1_denom_floor",
            alpha_mode="floor",
            q_m=True,
            q_v=True,
            projection=True,
            denom_floor_abs=1e-5,
        ),
        EcoCase("no_inv_lr_qm1_qv1", alpha_mode="no_inv_lr", projection=False),
    ]


def _write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        raise RuntimeError("no rows to write")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, float | str | int]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = list(dict.fromkeys(str(row["case"]) for row in rows))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {case: color_cycle[i % len(color_cycle)] for i, case in enumerate(cases)}

    def series(case: str, key: str) -> tuple[list[int], list[float]]:
        selected = [row for row in rows if row["case"] == case]
        return [int(row["step"]) for row in selected], [float(row[key]) for row in selected]

    panels = [
        ("stored_energy_gain", "stored ECO residual replay gain", "linear"),
        ("material_abs_ratio_p999", "material residual |replay/residual| p999", "log"),
        ("p_any_bad_material_345b", "P(any material coord > bad ratio) @345B", "linear"),
        ("denom_ratio_p001", "denom_quantized / denom_fp32 p0.1%", "log"),
        ("qv_zero_frac", "nonzero sqrt(v) stored as zero", "linear"),
        ("m_scale_inflation_p99", "Qm scale inflation p99", "log"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for ax, (key, title, yscale) in zip(axes.flat, panels, strict=True):
        for case in cases:
            xs, ys = series(case, key)
            ax.plot(xs, ys, label=case, linewidth=1.8, color=colors[case])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        if yscale == "log":
            ax.set_yscale("log")
        if key.startswith("p_any"):
            ax.set_ylim(-0.02, 1.02)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("ECO Qm/Qv Carrier Diagnostics", fontsize=14)
    fig.savefig(out_dir / "eco_qm_qv_dashboard.png", dpi=180)
    plt.close(fig)

    panels2 = [
        ("loss_proxy", "quadratic loss proxy", "log"),
        ("update_rms", "Adam update RMS before quant cast", "log"),
        ("raw_delta_m_p999", "raw ECO delta_m p999", "log"),
        ("stored_delta_m_p999", "stored ECO delta_m p999", "log"),
        ("projection_frac_p50", "projection fraction p50", "linear"),
        ("projection_frac_p01", "projection fraction p01", "linear"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for ax, (key, title, yscale) in zip(axes.flat, panels2, strict=True):
        for case in cases:
            xs, ys = series(case, key)
            ax.plot(xs, ys, label=case, linewidth=1.8, color=colors[case])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        if yscale == "log":
            ax.set_yscale("log")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("ECO Injection Magnitudes and Projection", fontsize=14)
    fig.savefig(out_dir / "eco_qm_qv_magnitudes.png", dpi=180)
    plt.close(fig)


def _write_summary(path: Path, rows: list[dict[str, float | str | int]], final_step: int) -> None:
    final = [row for row in rows if int(row["step"]) == final_step]
    final.sort(key=lambda row: str(row["case"]))
    columns = [
        "case",
        "stored_energy_gain",
        "material_abs_ratio_p999",
        "p_any_bad_material_345b",
        "denom_ratio_p001",
        "qv_zero_frac",
        "m_scale_inflation_p99",
        "projection_frac_p50",
        "loss_proxy",
    ]
    lines = ["# ECO Qm/Qv Diagnostics", ""]
    lines.append(f"Final simulated step: {final_step}")
    lines.append("")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in final:
        vals: list[str] = []
        for col in columns:
            val = row[col]
            if isinstance(val, str):
                vals.append(val)
            elif isinstance(val, int):
                vals.append(str(val))
            else:
                vals.append(f"{float(val):.4g}")
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--sample-elems", type=int, default=524_288)
    parser.add_argument("--base-lr", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=980)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--curvature", type=float, default=1e-2)
    parser.add_argument("--grad-noise-std", type=float, default=1e-3)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--cast-group-size", type=int, default=16)
    parser.add_argument("--weight-cols", type=int, default=512)
    parser.add_argument("--virtual-params", type=int, default=345_000_000_000)
    parser.add_argument("--bad-ratio-threshold", type=float, default=4.0)
    parser.add_argument("--material-quantile", type=float, default=0.95)
    parser.add_argument("--moment-quantizer", choices=("softsign", "linear"), default="softsign")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.sample_elems % args.group_size != 0:
        raise ValueError("--sample-elems must be divisible by --group-size")
    if args.sample_elems % args.weight_cols != 0:
        raise ValueError("--sample-elems must be divisible by --weight-cols")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (REPO_ROOT / "artifacts" / f"eco_qm_qv_diagnostics_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    initial_theta = _make_initial_theta(
        sample_elems=args.sample_elems,
        init_std=args.init_std,
        cast_group_size=args.cast_group_size,
        weight_cols=args.weight_cols,
        device=args.device,
        seed=args.seed,
    )

    rows: list[dict[str, float | str | int]] = []
    cases = _default_cases()
    for idx, case in enumerate(cases):
        print(f"[{idx + 1}/{len(cases)}] simulating {case.name}", flush=True)
        rows.extend(
            _simulate_case(
                case=case,
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
                virtual_params=args.virtual_params,
                bad_ratio_threshold=args.bad_ratio_threshold,
                material_quantile=args.material_quantile,
                moment_quantizer=args.moment_quantizer,
                seed=args.seed + 10_000,
            )
        )

    csv_path = out_dir / "eco_qm_qv_metrics.csv"
    _write_csv(csv_path, rows)
    _plot(rows, out_dir)
    _write_summary(out_dir / "summary.md", rows, args.steps)

    print(f"wrote {csv_path}")
    print(f"wrote {out_dir / 'summary.md'}")
    print(f"wrote {out_dir / 'eco_qm_qv_dashboard.png'}")
    print(f"wrote {out_dir / 'eco_qm_qv_magnitudes.png'}")


if __name__ == "__main__":
    main()
