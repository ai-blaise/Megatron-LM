#!/usr/bin/env python3
"""CPU simulations for the FlashAdamW -> FSDP NVFP4 update pipeline.

This tool consumes `[update_delta.adam]` logs and simulates:

* fresh-moment Adam update scale from observed post-clip gradient RMS values;
* approximate rowwise 16x16 NVFP4 recast noise;
* ECO-style residual injection pressure;
* simple update/weight ratio cap coverage.

The simulation is intentionally approximate. It is meant to reject obviously
bad next-run hypotheses before spending GPU time, not to replace the instrumented
training diagnostic.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass

import torch


FLOAT_RE = r"[-+0-9.eE]+"
UPDATE_RE = re.compile(
    r"\[update_delta\.adam\].*?"
    r"rank=(?P<rank>\d+) iter=(?P<iter>\d+) step=(?P<step>\d+) "
    r"owner=(?P<owner>\S+) name=(?P<name>\S+) "
    r"numel=(?P<numel>\d+) sample=(?P<sample>\d+) "
    r"shard_offset=(?P<shard_offset>\d+) lr=(?P<lr>" + FLOAT_RE + r") "
    r"grad_rms=(?P<grad_rms>" + FLOAT_RE + r") "
    r"grad_absmax=(?P<grad_absmax>" + FLOAT_RE + r") "
    r"pre_rms=(?P<pre_rms>" + FLOAT_RE + r") "
    r"post_rms=(?P<post_rms>" + FLOAT_RE + r") "
    r"delta_rms=(?P<delta_rms>" + FLOAT_RE + r") "
    r"delta_absmax=(?P<delta_absmax>" + FLOAT_RE + r") "
    r"delta_over_param_norm=(?P<rel>" + FLOAT_RE + r") "
    r"delta_over_lr_grad_norm=(?P<over>" + FLOAT_RE + r") "
    r"delta_grad_cos=(?P<cos>" + FLOAT_RE + r")"
)


@dataclass
class UpdateRow:
    rank: int
    iteration: int
    step: int
    owner: str
    name: str
    numel: int
    sample: int
    lr: float
    grad_rms: float
    grad_absmax: float
    pre_rms: float
    post_rms: float
    delta_rms: float
    delta_absmax: float
    rel: float
    over: float
    cos: float


def _rms(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(x.float() * x.float())).item())


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    values = sorted(xs)
    return values[int(q * (len(values) - 1))]


def _stats(xs: list[float]) -> str:
    if not xs:
        return "n=0"
    return (
        f"n={len(xs)} p50={_median(xs):.3e} "
        f"p90={_quantile(xs, 0.9):.3e} max={max(xs):.3e}"
    )


def parse_update_rows(log_dir: pathlib.Path) -> list[UpdateRow]:
    rows: list[UpdateRow] = []
    files = list(log_dir.rglob("*.log"))
    if not files:
        files = list(log_dir.rglob("*"))
    for path in files:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except UnicodeDecodeError:
            continue
        for line in lines:
            match = UPDATE_RE.search(line)
            if match is None:
                continue
            d = match.groupdict()
            rows.append(
                UpdateRow(
                    rank=int(d["rank"]),
                    iteration=int(d["iter"]),
                    step=int(d["step"]),
                    owner=d["owner"],
                    name=d["name"],
                    numel=int(d["numel"]),
                    sample=int(d["sample"]),
                    lr=float(d["lr"]),
                    grad_rms=float(d["grad_rms"]),
                    grad_absmax=float(d["grad_absmax"]),
                    pre_rms=float(d["pre_rms"]),
                    post_rms=float(d["post_rms"]),
                    delta_rms=float(d["delta_rms"]),
                    delta_absmax=float(d["delta_absmax"]),
                    rel=float(d["rel"]),
                    over=float(d["over"]),
                    cos=float(d["cos"]),
                )
            )
    return rows


def make_pre(numel: int, rms: float, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(numel, generator=gen)
    x = x * torch.exp(0.35 * torch.randn(numel, generator=gen))
    return x / max(_rms(x), 1.0e-30) * rms


def make_grad(numel: int, rms: float, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(numel, generator=gen)
    x = x * torch.exp(1.25 * torch.randn(numel, generator=gen))
    x = x * (torch.rand(numel, generator=gen) > 0.12).float()
    return x / max(_rms(x), 1.0e-30) * max(rms, 1.0e-13)


def adam_delta(
    grad: torch.Tensor,
    lr: float,
    step: int = 1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1.0e-8,
    m0: torch.Tensor | None = None,
    v0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if m0 is None:
        m0 = torch.zeros_like(grad)
    if v0 is None:
        v0 = torch.zeros_like(grad)
    m = beta1 * m0 + (1.0 - beta1) * grad
    v = beta2 * v0 + (1.0 - beta2) * grad * grad
    mh = m / (1.0 - beta1**step)
    vh = v / (1.0 - beta2**step)
    delta = -lr * mh / (torch.sqrt(vh) + eps)
    return delta, m, v


LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def nvfp4_rowwise_approx(x: torch.Tensor, *, sr: bool, seed: int) -> torch.Tensor:
    """Approximate 16x16 rowwise NVFP4 cast on CPU.

    We flatten into 256-element blocks to mimic 16x16 block amax. The real
    implementation has FP8 block-scale quantization, row/column layout, and
    distributed all-reduce effects; this approximation is enough to estimate
    update-vs-recast scale.
    """

    flat = x.flatten().float()
    numel = int(flat.numel())
    pad = (-numel) % 256
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    blocks = flat.view(-1, 256)
    unit = blocks.abs().amax(dim=1).clamp_min(1.0e-30)[:, None] / 6.0
    normalized = blocks / unit
    if sr:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        normalized = normalized + (torch.rand(normalized.shape, generator=gen) - 0.5) * 0.5
    normalized = normalized.clamp(-6.0, 6.0)
    abs_norm = normalized.abs()
    idx = torch.empty_like(abs_norm, dtype=torch.long)
    idx[abs_norm <= 0.25] = 0
    idx[(abs_norm > 0.25) & (abs_norm < 0.75)] = 1
    idx[(abs_norm >= 0.75) & (abs_norm <= 1.25)] = 2
    idx[(abs_norm > 1.25) & (abs_norm < 1.75)] = 3
    idx[(abs_norm >= 1.75) & (abs_norm <= 2.5)] = 4
    idx[(abs_norm > 2.5) & (abs_norm < 3.5)] = 5
    idx[(abs_norm >= 3.5) & (abs_norm <= 5.0)] = 6
    idx[abs_norm > 5.0] = 7
    q = LEVELS.index_select(0, idx.view(-1)).view_as(idx) * unit
    q = torch.where(normalized < 0.0, -q, q)
    return q.reshape(-1)[:numel]


def projection(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sum(a * b).item() / max(float(torch.sum(b * b).item()), 1.0e-30))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = max(float(a.norm().item()) * float(b.norm().item()), 1.0e-30)
    return float(torch.sum(a * b).item() / denom)


def summarize_observed(rows: list[UpdateRow], iteration: int) -> list[str]:
    lines = []
    active = [r for r in rows if r.iteration == iteration and r.lr > 0 and r.delta_rms > 0]
    lines.append(f"observed_iter{iteration}_nonzero_rows={len(active)}")
    by_owner: dict[str, list[UpdateRow]] = defaultdict(list)
    for row in active:
        by_owner[row.owner].append(row)
    for owner in sorted(by_owner):
        vals = by_owner[owner]
        if len(vals) < 2:
            continue
        lines.append(
            f"{owner:24s} rows={len(vals):3d} "
            f"grad_rms_med={_median([v.grad_rms for v in vals]):.3e} "
            f"pre_rms_med={_median([v.pre_rms for v in vals]):.3e} "
            f"delta/lr_med={_median([v.delta_rms / v.lr for v in vals]):.3f} "
            f"rel_max={max(v.rel for v in vals):.3e}"
        )
    return lines


def run_simulation(
    rows: list[UpdateRow],
    *,
    iteration: int,
    owners: list[str],
    rows_per_owner: int,
    sample_elems: int,
) -> list[str]:
    active = [r for r in rows if r.iteration == iteration and r.lr > 0 and r.delta_rms > 0]
    by_owner: dict[str, list[UpdateRow]] = defaultdict(list)
    for row in active:
        by_owner[row.owner].append(row)
    selected: list[UpdateRow] = []
    for owner in owners:
        selected.extend(
            sorted(by_owner.get(owner, []), key=lambda r: r.delta_rms, reverse=True)[
                :rows_per_owner
            ]
        )

    lines = [f"simulated_rows={len(selected)} sample_elems={sample_elems}"]
    if not selected:
        return lines

    eps_values = [1.0e-8, 3.0e-8, 1.0e-7, 3.0e-7, 1.0e-6]
    lines.append("fresh_moment_adam_eps_sweep:")
    for eps in eps_values:
        ratios = []
        for idx, row in enumerate(selected):
            grad = make_grad(sample_elems, row.grad_rms, 10_000 + idx)
            delta, _, _ = adam_delta(grad, lr=row.lr, eps=eps)
            ratios.append(_rms(delta) / max(row.lr, 1.0e-30))
        lines.append(f"  eps={eps:.0e}: delta/lr {_stats(ratios)}")

    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for idx, row in enumerate(selected):
        pre = make_pre(sample_elems, row.pre_rms, 20_000 + idx)
        pre_grid = nvfp4_rowwise_approx(pre, sr=False, seed=30_000 + idx)
        grad = make_grad(sample_elems, row.grad_rms, 40_000 + idx)
        fp32_delta, m1, v1 = adam_delta(grad, lr=row.lr, eps=1.0e-8)
        updated_grid = pre_grid + fp32_delta
        q_sr = nvfp4_rowwise_approx(updated_grid, sr=True, seed=50_000 + idx)
        q_rn = nvfp4_rowwise_approx(updated_grid, sr=False, seed=60_000 + idx)
        cast_error = updated_grid - q_sr
        model_delta_sr = q_sr - pre_grid
        model_delta_rn = q_rn - pre_grid

        beta1 = 0.9
        beta2 = 0.95
        bc1 = 1.0 - beta1
        eco_scalar = (bc1 / max(row.lr, 5.0e-5)) * (1.0 - 1.0 / beta1)
        denom = torch.sqrt(v1 / (1.0 - beta2)) + 1.0e-8
        eco_m = eco_scalar * denom * cast_error

        owner = row.owner
        agg[owner]["sim_delta_over_lr"].append(_rms(fp32_delta) / max(row.lr, 1.0e-30))
        agg[owner]["sr_cast_error_over_delta"].append(
            float(cast_error.norm().item()) / max(float(fp32_delta.norm().item()), 1.0e-30)
        )
        agg[owner]["rn_model_delta_projection"].append(projection(model_delta_rn, fp32_delta))
        agg[owner]["sr_model_delta_projection"].append(projection(model_delta_sr, fp32_delta))
        agg[owner]["sr_model_delta_cos"].append(cosine(model_delta_sr, fp32_delta))
        agg[owner]["eco_m_over_m"].append(
            float(eco_m.norm().item()) / max(float(m1.norm().item()), 1.0e-30)
        )

    lines.append("nvfp4_and_eco_approx_by_owner:")
    for owner in owners:
        if owner not in agg:
            continue
        lines.append(f"  {owner}")
        for key in (
            "sim_delta_over_lr",
            "sr_cast_error_over_delta",
            "rn_model_delta_projection",
            "sr_model_delta_projection",
            "sr_model_delta_cos",
            "eco_m_over_m",
        ):
            lines.append(f"    {key:28s} {_stats(agg[owner][key])}")

    lines.append("update_ratio_cap_coverage:")
    for cap in (1.0e-2, 5.0e-3, 2.0e-3, 1.0e-3, 5.0e-4, 2.0e-4, 1.0e-4):
        affected = [r for r in active if r.rel > cap]
        pct = 100.0 * len(affected) / max(len(active), 1)
        scale = max((r.rel / cap for r in affected), default=1.0)
        lines.append(
            f"  cap={cap:.0e}: affected={len(affected)}/{len(active)} "
            f"({pct:.1f}%) worst_scale_down={scale:.1f}x"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=pathlib.Path, required=True)
    parser.add_argument("--iteration", type=int, default=2)
    parser.add_argument("--sample-elems", type=int, default=8192)
    parser.add_argument("--rows-per-owner", type=int, default=12)
    parser.add_argument(
        "--owners",
        default=(
            "moe_shared_expert,attention_mla_out,attention_mla_kv_up,"
            "attention_mla_q,moe_routed_expert_fc2"
        ),
    )
    args = parser.parse_args()

    rows = parse_update_rows(args.log_dir)
    print(f"parsed_update_delta_rows={len(rows)}")
    for line in summarize_observed(rows, args.iteration):
        print(line)
    owners = [owner.strip() for owner in args.owners.split(",") if owner.strip()]
    for line in run_simulation(
        rows,
        iteration=args.iteration,
        owners=owners,
        rows_per_owner=args.rows_per_owner,
        sample_elems=args.sample_elems,
    ):
        print(line)


if __name__ == "__main__":
    main()
