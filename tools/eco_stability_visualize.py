#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Visualize ECO stability as virtual parameter count scales.

This script extends ``eco_stability_sweep.py`` with two things:

* an optional matched second-moment pseudo-injection for ECO residuals, and
* an SVG dashboard plus CSV outputs for scale extrapolation.

The simulation is still intentionally a toy model.  It models independent
coordinates under AdamW, a block NVFP4-like weight recast, FlashAdamW-style
rowwise INT8 moment storage, and the production ECO injection shape:

    m += alpha_t * denom_t * (pre_cast - post_cast)

For matched-``v`` variants, the same first-moment injection is interpreted as a
pseudo-gradient and its square is added to Adam's second moment:

    g_eco = delta_m / (1 - beta1)
    v += v_match * (1 - beta2) * g_eco**2

This is not a proposed final algorithm; it is a stress test for whether
denominator matching suppresses the rare update tails that appear at scale.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eco_stability_sweep import (
    _eco_alpha,
    _fake_nvfp4_block_cast,
    _lr_at_step,
    _parse_int_list,
    _quantize_optimizer_state,
    _safe_quantile,
)


@dataclass(frozen=True)
class Method:
    name: str
    alpha_mode: str
    alpha_gain: float = 1.0
    v_match: float = 0.0
    pseudo_grad_clip: float | None = None


@dataclass
class MethodHistory:
    method: Method
    step: list[int]
    lr: list[float]
    bad_frac: list[float]
    max_update: list[float]
    p999_update: list[float]
    max_m: list[float]
    p999_m: list[float]
    pseudo_grad_p999: list[float]
    delta_m_p999: list[float]
    p001_denom: list[float]
    min_denom: list[float]
    v_sqrt_rowmax: list[float]
    cast_error_rms: list[float]
    correction_gain: list[float]
    loss_proxy: list[float]

    @property
    def cumulative_hazard(self) -> list[float]:
        total = 0.0
        out = []
        for frac in self.bad_frac:
            frac = min(max(frac, 0.0), 1.0 - 1e-12)
            total += -math.log1p(-frac)
            out.append(total)
        return out


COLORS = {
    "off": "#4b5563",
    "paper": "#dc2626",
    "paper*0.1": "#ef4444",
    "paper*0.01": "#fca5a5",
    "paper+v0.001": "#fb7185",
    "paper+v0.01": "#fb7185",
    "paper+v0.01c": "#f97316",
    "lr_floor": "#2563eb",
    "lr_floor+v0.01": "#60a5fa",
    "lr_floor+v0.01c": "#7c3aed",
    "no_inv_lr": "#059669",
    "no_inv*1e3": "#10b981",
    "no_inv*1e4": "#84cc16",
    "no_inv*1e5": "#a3e635",
    "no_inv_lr+v": "#0d9488",
}


def _default_methods() -> list[Method]:
    return [
        Method("off", "off"),
        Method("paper", "paper"),
        Method("paper*0.1", "paper", 0.1),
        Method("paper*0.01", "paper", 0.01),
        Method("paper+v0.001", "paper", v_match=0.001),
        Method("paper+v0.01", "paper", v_match=0.01),
        Method("paper+v0.01c", "paper", v_match=0.01, pseudo_grad_clip=1e-2),
        Method("lr_floor", "lr_floor"),
        Method("lr_floor+v0.01", "lr_floor", v_match=0.01),
        Method("lr_floor+v0.01c", "lr_floor", v_match=0.01, pseudo_grad_clip=1e-2),
        Method("no_inv_lr", "no_inv_lr"),
        Method("no_inv*1e3", "no_inv_lr", 1_000.0),
        Method("no_inv*1e4", "no_inv_lr", 10_000.0),
        Method("no_inv*1e5", "no_inv_lr", 100_000.0),
    ]


def _simulate_method(
    *,
    method: Method,
    device: str,
    sample_elems: int,
    steps: int,
    base_lr: float,
    warmup_steps: int,
    beta1: float,
    beta2: float,
    eps: float,
    grad_noise_std: float,
    curvature: float,
    init_std: float,
    group_size: int,
    bad_update_threshold: float,
    stochastic_cast: bool,
    quantized_state: bool,
    seed: int,
) -> MethodHistory:
    if sample_elems % group_size != 0:
        raise ValueError("--sample-elems must be divisible by --group-size")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    theta = (
        torch.randn(sample_elems, generator=gen, device=device, dtype=torch.float32)
        * init_std
    )
    theta = _fake_nvfp4_block_cast(
        theta, group_size=group_size, stochastic=False, generator=gen
    )
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    hist = MethodHistory(
        method=method,
        step=[],
        lr=[],
        bad_frac=[],
        max_update=[],
        p999_update=[],
        max_m=[],
        p999_m=[],
        pseudo_grad_p999=[],
        delta_m_p999=[],
        p001_denom=[],
        min_denom=[],
        v_sqrt_rowmax=[],
        cast_error_rms=[],
        correction_gain=[],
        loss_proxy=[],
    )

    for step in range(1, steps + 1):
        lr = _lr_at_step(step, base_lr=base_lr, warmup_steps=warmup_steps)
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step

        grad = curvature * theta
        if grad_noise_std > 0.0:
            grad = (
                grad
                + torch.randn(sample_elems, generator=gen, device=device) * grad_noise_std
            )

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad.square()
        if quantized_state:
            m, v = _quantize_optimizer_state(m, v, group_size)

        denom = torch.sqrt(v / bc2).add_(eps)
        update = m / bc1 / denom
        update_abs = (lr * update).abs()
        pre_cast = theta - lr * update
        post_cast = _fake_nvfp4_block_cast(
            pre_cast, group_size=16, stochastic=stochastic_cast, generator=gen
        )
        error = pre_cast - post_cast

        alpha = _eco_alpha(
            method.alpha_mode, bc1=bc1, lr=lr, base_lr=base_lr, beta1=beta1
        ) * method.alpha_gain
        delta_m = torch.zeros_like(m)
        pseudo_grad = torch.zeros_like(m)
        if alpha != 0.0:
            delta_m = alpha * denom * error
            m = m + delta_m
            if method.v_match > 0.0:
                pseudo_grad = delta_m / (1.0 - beta1)
                if method.pseudo_grad_clip is not None:
                    pseudo_grad = pseudo_grad.clamp(
                        min=-method.pseudo_grad_clip, max=method.pseudo_grad_clip
                    )
                v = v + method.v_match * (1.0 - beta2) * pseudo_grad.square()
            if quantized_state:
                m, v = _quantize_optimizer_state(m, v, group_size)

        theta = post_cast

        bad = (
            ~torch.isfinite(theta)
            | ~torch.isfinite(m)
            | ~torch.isfinite(v)
            | (update_abs > bad_update_threshold)
        )
        finite_denom = denom[torch.isfinite(denom)]
        hist.step.append(step)
        hist.lr.append(float(lr))
        hist.bad_frac.append(float(bad.float().mean().item()))
        hist.max_update.append(float(update_abs.max().item()))
        hist.p999_update.append(_safe_quantile(update_abs, 0.999))
        hist.max_m.append(float(m.abs().max().item()))
        hist.p999_m.append(_safe_quantile(m.abs(), 0.999))
        hist.pseudo_grad_p999.append(_safe_quantile(pseudo_grad.abs(), 0.999))
        hist.delta_m_p999.append(_safe_quantile(delta_m.abs(), 0.999))
        hist.p001_denom.append(_safe_quantile(finite_denom, 0.001))
        hist.min_denom.append(float(finite_denom.min().item()) if finite_denom.numel() else 0.0)
        hist.v_sqrt_rowmax.append(
            float(v.clamp_min(0.0).sqrt().view(-1, group_size).amax(dim=1).max().item())
        )
        hist.cast_error_rms.append(float(error.square().mean().sqrt().item()))
        hist.correction_gain.append(float(abs(lr * alpha / bc1)) if bc1 > 0.0 else 0.0)
        hist.loss_proxy.append(float(0.5 * theta.square().mean().item()))

    return hist


def _p_any(cumulative_hazard: float, virtual_params: int) -> float:
    return 1.0 - math.exp(-min(700.0, virtual_params * cumulative_hazard))


def _first_p50_step(history: MethodHistory, virtual_params: int) -> int | None:
    for step, hazard in zip(history.step, history.cumulative_hazard):
        if _p_any(hazard, virtual_params) >= 0.5:
            return step
    return None


def _write_per_step_csv(path: Path, histories: list[MethodHistory]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "step",
                "lr",
                "bad_frac",
                "max_update",
                "p999_update",
                "max_m",
                "p999_m",
                "pseudo_grad_p999",
                "delta_m_p999",
                "p001_denom",
                "min_denom",
                "v_sqrt_rowmax",
                "cast_error_rms",
                "correction_gain",
                "loss_proxy",
            ]
        )
        for hist in histories:
            for i, step in enumerate(hist.step):
                writer.writerow(
                    [
                        hist.method.name,
                        step,
                        hist.lr[i],
                        hist.bad_frac[i],
                        hist.max_update[i],
                        hist.p999_update[i],
                        hist.max_m[i],
                        hist.p999_m[i],
                        hist.pseudo_grad_p999[i],
                        hist.delta_m_p999[i],
                        hist.p001_denom[i],
                        hist.min_denom[i],
                        hist.v_sqrt_rowmax[i],
                        hist.cast_error_rms[i],
                        hist.correction_gain[i],
                        hist.loss_proxy[i],
                    ]
                )


def _write_scale_csv(
    path: Path, histories: list[MethodHistory], virtual_params: list[int], checkpoints: list[int]
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "virtual_params", "first_p50_step", *[f"p_any_step_{s}" for s in checkpoints]])
        for hist in histories:
            hazards = hist.cumulative_hazard
            for n in virtual_params:
                row = [hist.method.name, n, _first_p50_step(hist, n)]
                for step in checkpoints:
                    idx = min(max(step, 1), len(hazards)) - 1
                    row.append(_p_any(hazards[idx], n))
                writer.writerow(row)


def _nice_num(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:.0e}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 1:
        return f"{x:.1f}"
    return f"{x:.1e}"


def _log10_values(values: list[float]) -> list[float]:
    positive = [v for v in values if v > 0]
    floor = min(positive) * 0.5 if positive else 1e-12
    return [math.log10(max(v, floor)) for v in values]


def _line_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, tuple[list[float], list[float]]],
    x_log: bool,
    y_log: bool,
    x_ticks: list[float] | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    width: int = 360,
    height: int = 255,
) -> str:
    ml, mr, mt, mb = 54, 14, 28, 42
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    all_x = [x for xs, _ in series.values() for x in xs]
    all_y = [y for _, ys in series.values() for y in ys if y is not None and math.isfinite(y)]
    if not all_y:
        all_y = [0.0, 1.0]

    tx = (lambda v: math.log10(max(v, 1e-30))) if x_log else (lambda v: v)
    if y_log:
        ty_values = _log10_values(all_y)
        y0 = math.floor(min(ty_values)) if y_min is None else math.log10(y_min)
        y1 = math.ceil(max(ty_values)) if y_max is None else math.log10(y_max)
        ty = lambda v: math.log10(max(v, 10**y0))
    else:
        y0 = min(all_y) if y_min is None else y_min
        y1 = max(all_y) if y_max is None else y_max
        if y0 == y1:
            y0 -= 1.0
            y1 += 1.0
        pad = 0.08 * (y1 - y0)
        if y_min is None:
            y0 -= pad
        if y_max is None:
            y1 += pad
        ty = lambda v: v

    x0 = min(tx(v) for v in all_x)
    x1 = max(tx(v) for v in all_x)
    if x0 == x1:
        x0 -= 1.0
        x1 += 1.0

    def sx(v: float) -> float:
        return ml + (tx(v) - x0) / (x1 - x0) * plot_w

    def sy(v: float) -> float:
        return mt + (1.0 - (ty(v) - y0) / (y1 - y0)) * plot_h

    parts = [
        f'<g class="panel">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{ml}" y="18" font-size="13" font-weight="700">{escape(title)}</text>',
        f'<line x1="{ml}" y1="{mt + plot_h}" x2="{ml + plot_w}" y2="{mt + plot_h}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + plot_h}" stroke="#374151" stroke-width="1"/>',
    ]

    if x_ticks is None:
        if x_log:
            lo = math.ceil(x0)
            hi = math.floor(x1)
            x_ticks = [10**p for p in range(lo, hi + 1)]
        else:
            x_ticks = [all_x[0], all_x[len(all_x) // 2], all_x[-1]]
    for tick in x_ticks:
        if tick <= 0:
            continue
        x = sx(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{mt + plot_h}" x2="{x:.1f}" y2="{mt + plot_h + 4}" stroke="#374151"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt + plot_h + 17}" text-anchor="middle" font-size="9" fill="#4b5563">{escape(_nice_num(tick))}</text>')

    if y_log:
        ticks = [10**p for p in range(math.ceil(y0), math.floor(y1) + 1)]
    else:
        ticks = [y0 + (y1 - y0) * i / 4.0 for i in range(5)]
    for tick in ticks:
        if y_log and tick <= 0:
            continue
        y = sy(tick)
        parts.append(f'<line x1="{ml - 4}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" stroke="#374151"/>')
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{ml - 7}" y="{y + 3:.1f}" text-anchor="end" font-size="9" fill="#4b5563">{escape(_nice_num(tick))}</text>')

    for name, (xs, ys) in series.items():
        color = COLORS.get(name, "#111827")
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys) if y is not None and math.isfinite(y))
        if points:
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')

    parts.append(f'<text x="{ml + plot_w / 2}" y="{height - 7}" text-anchor="middle" font-size="10" fill="#374151">{escape(x_label)}</text>')
    parts.append(f'<text x="13" y="{mt + plot_h / 2}" transform="rotate(-90 13 {mt + plot_h / 2})" text-anchor="middle" font-size="10" fill="#374151">{escape(y_label)}</text>')
    parts.append("</g>")
    return "\n".join(parts)


def _render_svg(
    path: Path,
    *,
    histories: list[MethodHistory],
    virtual_params: list[int],
    bad_update_threshold: float,
    checkpoints: list[int],
    steps: int,
) -> None:
    panel_w, panel_h = 370, 255
    cols, rows = 3, 4
    legend_h = 92
    width = panel_w * cols
    height = panel_h * rows + legend_h
    x_ticks = [1e6, 1e8, 1e10, 1e11]
    if max(virtual_params) >= 3e11:
        x_ticks.append(3e11)

    def failure_series() -> dict[str, tuple[list[float], list[float]]]:
        out = {}
        for h in histories:
            ys = []
            for n in virtual_params:
                step = _first_p50_step(h, n)
                ys.append(float(step if step is not None else steps + 1))
            out[h.method.name] = ([float(v) for v in virtual_params], ys)
        return out

    def p_any_series(step: int) -> dict[str, tuple[list[float], list[float]]]:
        out = {}
        for h in histories:
            hazards = h.cumulative_hazard
            idx = min(max(step, 1), len(hazards)) - 1
            out[h.method.name] = (
                [float(v) for v in virtual_params],
                [_p_any(hazards[idx], n) for n in virtual_params],
            )
        return out

    def step_series(metric: str) -> dict[str, tuple[list[float], list[float]]]:
        out = {}
        for h in histories:
            out[h.method.name] = ([float(s) for s in h.step], [float(v) for v in getattr(h, metric)])
        return out

    panels = [
        _line_chart(
            title=f"First P(any bad) >= 50% @ update>{bad_update_threshold:g}",
            x_label="virtual params",
            y_label="step (top means no crossing)",
            series=failure_series(),
            x_log=True,
            y_log=False,
            x_ticks=x_ticks,
            y_min=0,
            y_max=steps + 1,
            width=panel_w,
            height=panel_h,
        ),
        _line_chart(
            title=f"P(any bad) by step {checkpoints[-1]}",
            x_label="virtual params",
            y_label="probability",
            series=p_any_series(checkpoints[-1]),
            x_log=True,
            y_log=False,
            x_ticks=x_ticks,
            y_min=0,
            y_max=1,
            width=panel_w,
            height=panel_h,
        ),
        _line_chart(
            title="Sample bad-coordinate fraction",
            x_label="step",
            y_label="fraction",
            series=step_series("bad_frac"),
            x_log=False,
            y_log=True,
            width=panel_w,
            height=panel_h,
        ),
        _line_chart(
            title="Max effective update over sample",
            x_label="step",
            y_label="|lr * mhat / denom|",
            series=step_series("max_update"),
            x_log=False,
            y_log=True,
            width=panel_w,
            height=panel_h,
        ),
        _line_chart(
            title="P99.9 effective update",
            x_label="step",
            y_label="|lr * mhat / denom|",
            series=step_series("p999_update"),
            x_log=False,
            y_log=True,
            width=panel_w,
            height=panel_h,
        ),
        _line_chart(
            title="Max |first moment|",
            x_label="step",
            y_label="|m|",
            series=step_series("max_m"),
            x_log=False,
            y_log=True,
            width=panel_w,
            height=panel_h,
        ),
        _line_chart(
            title="P99.9 |first moment|",
            x_label="step",
            y_label="|m|",
            series=step_series("p999_m"),
            x_log=False,
            y_log=True,
            width=panel_w,
            height=panel_h,
        ),
        _line_chart(
            title="P99.9 ECO pseudo-gradient",
            x_label="step",
            y_label="|delta_m|/(1-beta1)",
            series=step_series("pseudo_grad_p999"),
            x_log=False,
            y_log=True,
            width=panel_w,
            height=panel_h,
        ),
        _line_chart(
            title="P99.9 ECO delta_m",
            x_label="step",
            y_label="|delta_m|",
            series=step_series("delta_m_p999"),
            x_log=False,
            y_log=True,
            width=panel_w,
            height=panel_h,
        ),
        _line_chart(
            title="P0.1 Adam denominator",
            x_label="step",
            y_label="sqrt(vhat)+eps",
            series=step_series("p001_denom"),
            x_log=False,
            y_log=True,
            width=panel_w,
            height=panel_h,
        ),
        _line_chart(
            title="Max row sqrt(v) scale",
            x_label="step",
            y_label="max row sqrt(v)",
            series=step_series("v_sqrt_rowmax"),
            x_log=False,
            y_log=True,
            width=panel_w,
            height=panel_h,
        ),
        _line_chart(
            title="Cast residual RMS",
            x_label="step",
            y_label="rms(pre-post)",
            series=step_series("cast_error_rms"),
            x_log=False,
            y_log=True,
            width=panel_w,
            height=panel_h,
        ),
        _line_chart(
            title="Residual correction gain",
            x_label="step",
            y_label="|lr * alpha / bc1|",
            series=step_series("correction_gain"),
            x_log=False,
            y_log=True,
            width=panel_w,
            height=panel_h,
        ),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f9fafb"/>',
        '<style>text{font-family:Inter,Arial,sans-serif}.panel{filter:none}</style>',
    ]
    for i, panel in enumerate(panels):
        x = (i % cols) * panel_w
        y = (i // cols) * panel_h
        parts.append(f'<g transform="translate({x},{y})">{panel}</g>')

    legend_y = panel_h * rows + 19
    legend_x = 18
    parts.append(f'<text x="{legend_x}" y="{legend_y}" font-size="13" font-weight="700">Methods</text>')
    cursor = legend_x + 70
    line_y = legend_y
    for h in histories:
        color = COLORS.get(h.method.name, "#111827")
        item_w = 92 + 6 * len(h.method.name)
        if cursor + item_w > width - 18:
            cursor = legend_x + 70
            line_y += 18
        parts.append(f'<line x1="{cursor}" y1="{line_y - 4}" x2="{cursor + 22}" y2="{line_y - 4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{cursor + 28}" y="{line_y}" font-size="11">{escape(h.method.name)}</text>')
        cursor += item_w
    parts.append(
        f'<text x="{legend_x}" y="{line_y + 23}" font-size="11" fill="#4b5563">'
        f'Bad means effective update exceeds {bad_update_threshold:g}. Scale curves extrapolate independent-coordinate hazard from the sampled toy model; they are diagnostics, not trainer loss predictions.'
        '</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--sample-elems", type=int, default=262_144)
    parser.add_argument(
        "--virtual-params",
        type=_parse_int_list,
        default=_parse_int_list("1m,10m,100m,800m,8b,30b,70b,202b,345b"),
    )
    parser.add_argument("--base-lr", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=980)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-noise-std", type=float, default=1e-3)
    parser.add_argument("--curvature", type=float, default=1e-2)
    parser.add_argument("--init-std", type=float, default=2e-2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bad-update-threshold", type=float, default=1e-4)
    parser.add_argument("--deterministic-cast", action="store_true")
    parser.add_argument("--fp32-state", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eco_stability"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = _default_methods()
    histories = []
    for method in methods:
        histories.append(
            _simulate_method(
                method=method,
                device=args.device,
                sample_elems=args.sample_elems,
                steps=args.steps,
                base_lr=args.base_lr,
                warmup_steps=args.warmup_steps,
                beta1=args.beta1,
                beta2=args.beta2,
                eps=args.eps,
                grad_noise_std=args.grad_noise_std,
                curvature=args.curvature,
                init_std=args.init_std,
                group_size=args.group_size,
                bad_update_threshold=args.bad_update_threshold,
                stochastic_cast=not args.deterministic_cast,
                quantized_state=not args.fp32_state,
                seed=args.seed,
            )
        )

    checkpoints = [5, 20, 100]
    per_step_csv = args.output_dir / "eco_stability_per_step.csv"
    scale_csv = args.output_dir / "eco_stability_scale_summary.csv"
    svg_path = args.output_dir / "eco_stability_dashboard.svg"
    _write_per_step_csv(per_step_csv, histories)
    _write_scale_csv(scale_csv, histories, args.virtual_params, checkpoints)
    _render_svg(
        svg_path,
        histories=histories,
        virtual_params=args.virtual_params,
        bad_update_threshold=args.bad_update_threshold,
        checkpoints=checkpoints,
        steps=args.steps,
    )

    print(f"Wrote {svg_path}")
    print(f"Wrote {per_step_csv}")
    print(f"Wrote {scale_csv}")
    print("\nFinal-step summary:")
    print(
        "method\tbad_frac\tmax_update\tp999_update\tmax_m\tp999_m\t"
        "pseudo_grad_p999\tdelta_m_p999\tp001_denom\tv_sqrt_rowmax\t"
        "cast_error_rms\tcorrection_gain"
    )
    for h in histories:
        print(
            f"{h.method.name}\t{h.bad_frac[-1]:.3e}\t{h.max_update[-1]:.3e}\t"
            f"{h.p999_update[-1]:.3e}\t{h.max_m[-1]:.3e}\t"
            f"{h.p999_m[-1]:.3e}\t{h.pseudo_grad_p999[-1]:.3e}\t"
            f"{h.delta_m_p999[-1]:.3e}\t{h.p001_denom[-1]:.3e}\t"
            f"{h.v_sqrt_rowmax[-1]:.3e}\t{h.cast_error_rms[-1]:.3e}\t"
            f"{h.correction_gain[-1]:.3e}"
        )


if __name__ == "__main__":
    main()
