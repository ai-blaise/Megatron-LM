#!/usr/bin/env python3
"""Stitch Megatron numeric-debug update pipeline logs by parameter.

The training loss diagnostics need one view that follows the same FSDP-local
parameter shard through:

* finalize-model-grads token scaling;
* gradient clipping;
* FlashAdamW's fp32/main-weight update;
* NVFP4 rowwise recast error.

This script is intentionally offline-only. It parses copied stdout logs and
does not import Megatron or touch distributed state.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


FLOAT_RE = r"[-+0-9.eE]+"

TENSOR_RE = re.compile(
    r"\[numeric-debug\]\[rank=(?P<rank>\d+) local=(?P<local>\d+) "
    r"iter=(?P<iter>\d+) phase=(?P<phase>[^\]]+)\]\s+"
    r"(?P<label>\S+): .*?"
    r"numel=(?P<numel>\d+) sample_numel=(?P<sample_numel>\d+) "
    r"all_finite=(?P<all_finite>\w+) .*?"
    r"mean=(?P<mean>" + FLOAT_RE + r") "
    r"rms=(?P<rms>" + FLOAT_RE + r") "
    r"absmax=(?P<absmax>" + FLOAT_RE + r")"
)

ADAM_RE = re.compile(
    r"\[update_delta\.adam\] "
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
    r"delta_over_param_norm=(?P<delta_over_param_norm>" + FLOAT_RE + r") "
    r"delta_over_lr_grad_norm=(?P<delta_over_lr_grad_norm>" + FLOAT_RE + r") "
    r"delta_grad_cos=(?P<delta_grad_cos>" + FLOAT_RE + r")"
)

CAST_RE = re.compile(
    r"\[update_delta\.nvfp4_cast\] "
    r"rank=(?P<rank>\d+) iter=(?P<iter>\d+) step=(?P<step>\d+) "
    r"owner=(?P<owner>\S+) name=(?P<name>\S+) "
    r"numel=(?P<numel>\d+) sample=(?P<sample>\d+) "
    r"shard_offset=(?P<shard_offset>\d+) rowwise_byte_offset=(?P<rowwise_byte_offset>\d+) "
    r"lr=(?P<lr>" + FLOAT_RE + r") "
    r"pre_rms=(?P<pre_rms>" + FLOAT_RE + r") "
    r"post_rms=(?P<post_rms>" + FLOAT_RE + r") "
    r"cast_error_rms=(?P<cast_error_rms>" + FLOAT_RE + r") "
    r"cast_error_absmax=(?P<cast_error_absmax>" + FLOAT_RE + r") "
    r"cast_error_over_param_norm=(?P<cast_error_over_param_norm>" + FLOAT_RE + r")"
)

AFTER_CLIP_LABEL_RE = re.compile(
    r"optimizer_grad\.after_clip_grad_norm_(?P<clip_norm>" + FLOAT_RE + r")\."
    r"(?P<name>.*)\.grad$"
)


@dataclass
class Row:
    stage: str
    rank: int
    iteration: int
    name: str
    owner: str
    metrics: dict[str, float]
    numel: int = 0
    sample: int = 0
    source: str = ""


@dataclass
class Chain:
    rank: int
    iteration: int
    name: str
    owner: str
    rows: dict[str, list[Row]] = field(default_factory=lambda: defaultdict(list))

    def add(self, row: Row) -> None:
        self.rows[row.stage].append(row)
        if self.owner == "unknown" and row.owner != "unknown":
            self.owner = row.owner


def median(values: Iterable[float]) -> float:
    xs = [x for x in values if math.isfinite(x)]
    return statistics.median(xs) if xs else float("nan")


def quantile(values: Iterable[float], q: float) -> float:
    xs = sorted(x for x in values if math.isfinite(x))
    if not xs:
        return float("nan")
    return xs[int(q * (len(xs) - 1))]


def fmt(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.3e}"


def normalize_name(name: str) -> str:
    if name.endswith(".grad"):
        name = name[: -len(".grad")]
    name = re.sub(r"^chunk\d+\.", "", name)
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


def infer_owner(name: str) -> str:
    n = normalize_name(name)
    if "core_attention.indexer" in n or "hisa" in n or "indexcache" in n:
        return "dsa_hisa_indexer"
    if ".shared_experts." in n:
        return "moe_shared_expert"
    if ".experts.linear_fc1" in n:
        return "moe_routed_expert_fc1"
    if ".experts.linear_fc2" in n:
        return "moe_routed_expert_fc2"
    if ".mlp.linear_fc" in n:
        return "dense_mlp"
    if ".self_attention.linear_kv_up_proj" in n:
        return "attention_mla_kv_up"
    if ".self_attention.linear_proj" in n:
        return "attention_mla_out"
    if (
        ".self_attention.linear_q" in n
        or ".self_attention.linear_q_down_proj" in n
        or ".self_attention.linear_q_up_proj" in n
    ):
        return "attention_mla_q"
    if ".self_attention." in n:
        return "attention_mla_other"
    if ".gated_norm" in n or ".norm" in n:
        return "gated_norm"
    if ".router." in n or ".gate." in n:
        return "moe_router"
    if "embedding" in n:
        return "embedding"
    return "other"


def tensor_row_from_match(match: re.Match[str], source: str) -> Row | None:
    label = match.group("label")
    stage: str
    name: str
    metrics: dict[str, float] = {}

    if label.startswith("finalize_grad.before_token_scale."):
        stage = "before_token_scale"
        name = label[len("finalize_grad.before_token_scale.") :]
        if name.endswith(".grad"):
            name = name[: -len(".grad")]
    elif label.startswith("finalize_grad.after_token_scale."):
        stage = "after_token_scale"
        name = label[len("finalize_grad.after_token_scale.") :]
        if name.endswith(".grad"):
            name = name[: -len(".grad")]
    else:
        clip_match = AFTER_CLIP_LABEL_RE.match(label)
        if clip_match is None:
            return None
        stage = "after_clip"
        name = clip_match.group("name")
        metrics["clip_global_norm"] = float(clip_match.group("clip_norm"))

    metrics.update(
        mean=float(match.group("mean")),
        rms=float(match.group("rms")),
        absmax=float(match.group("absmax")),
        all_finite=1.0 if match.group("all_finite") == "True" else 0.0,
    )
    canonical = normalize_name(name)
    return Row(
        stage=stage,
        rank=int(match.group("rank")),
        iteration=int(match.group("iter")),
        name=canonical,
        owner=infer_owner(canonical),
        metrics=metrics,
        numel=int(match.group("numel")),
        sample=int(match.group("sample_numel")),
        source=source,
    )


def adam_row_from_match(match: re.Match[str], source: str) -> Row:
    d = match.groupdict()
    name = normalize_name(d["name"])
    metrics = {
        "step": float(d["step"]),
        "lr": float(d["lr"]),
        "grad_rms": float(d["grad_rms"]),
        "grad_absmax": float(d["grad_absmax"]),
        "pre_rms": float(d["pre_rms"]),
        "post_rms": float(d["post_rms"]),
        "delta_rms": float(d["delta_rms"]),
        "delta_absmax": float(d["delta_absmax"]),
        "delta_over_param_norm": float(d["delta_over_param_norm"]),
        "delta_over_lr_grad_norm": float(d["delta_over_lr_grad_norm"]),
        "delta_grad_cos": float(d["delta_grad_cos"]),
    }
    return Row(
        stage="adam",
        rank=int(d["rank"]),
        iteration=int(d["iter"]),
        name=name,
        owner=d["owner"],
        metrics=metrics,
        numel=int(d["numel"]),
        sample=int(d["sample"]),
        source=source,
    )


def cast_row_from_match(match: re.Match[str], source: str) -> Row:
    d = match.groupdict()
    name = normalize_name(d["name"])
    metrics = {
        "step": float(d["step"]),
        "lr": float(d["lr"]),
        "pre_rms": float(d["pre_rms"]),
        "post_rms": float(d["post_rms"]),
        "cast_error_rms": float(d["cast_error_rms"]),
        "cast_error_absmax": float(d["cast_error_absmax"]),
        "cast_error_over_param_norm": float(d["cast_error_over_param_norm"]),
        "shard_offset": float(d["shard_offset"]),
        "rowwise_byte_offset": float(d["rowwise_byte_offset"]),
    }
    return Row(
        stage="cast",
        rank=int(d["rank"]),
        iteration=int(d["iter"]),
        name=name,
        owner=d["owner"],
        metrics=metrics,
        numel=int(d["numel"]),
        sample=int(d["sample"]),
        source=source,
    )


def parse_logs(log_dir: pathlib.Path) -> dict[tuple[int, int, str], Chain]:
    chains: dict[tuple[int, int, str], Chain] = {}
    files = [p for p in log_dir.rglob("*") if p.is_file()]
    for path in sorted(files):
        source = str(path)
        with path.open("r", errors="ignore") as handle:
            for line in handle:
                row: Row | None = None
                if "finalize_grad." in line or "optimizer_grad.after_clip_grad_norm" in line:
                    match = TENSOR_RE.search(line)
                    if match is not None:
                        row = tensor_row_from_match(match, source)
                elif "[update_delta.adam]" in line:
                    match = ADAM_RE.search(line)
                    if match is not None:
                        row = adam_row_from_match(match, source)
                elif "[update_delta.nvfp4_cast]" in line:
                    match = CAST_RE.search(line)
                    if match is not None:
                        row = cast_row_from_match(match, source)
                if row is None:
                    continue
                key = (row.rank, row.iteration, row.name)
                if key not in chains:
                    chains[key] = Chain(
                        rank=row.rank,
                        iteration=row.iteration,
                        name=row.name,
                        owner=row.owner,
                    )
                chains[key].add(row)
    return chains


def stage_metric(chain: Chain, stage: str, metric: str) -> float:
    return median(row.metrics.get(metric, float("nan")) for row in chain.rows.get(stage, []))


def chain_row(chain: Chain) -> dict[str, float | str | int]:
    after_token = stage_metric(chain, "after_token_scale", "rms")
    after_clip = stage_metric(chain, "after_clip", "rms")
    adam_grad = stage_metric(chain, "adam", "grad_rms")
    delta = stage_metric(chain, "adam", "delta_rms")
    cast_error = stage_metric(chain, "cast", "cast_error_rms")
    return {
        "rank": chain.rank,
        "iteration": chain.iteration,
        "owner": chain.owner if chain.owner != "unknown" else infer_owner(chain.name),
        "name": chain.name,
        "before_token_rms": stage_metric(chain, "before_token_scale", "rms"),
        "after_token_rms": after_token,
        "after_clip_rms": after_clip,
        "clip_scale": after_clip / after_token if after_token and math.isfinite(after_token) else float("nan"),
        "adam_grad_rms": adam_grad,
        "adam_delta_rms": delta,
        "adam_delta_over_param": stage_metric(chain, "adam", "delta_over_param_norm"),
        "adam_delta_over_lr_grad": stage_metric(chain, "adam", "delta_over_lr_grad_norm"),
        "adam_delta_grad_cos": stage_metric(chain, "adam", "delta_grad_cos"),
        "cast_error_rms": cast_error,
        "cast_error_over_param": stage_metric(chain, "cast", "cast_error_over_param_norm"),
        "cast_error_over_update": cast_error / delta if delta and math.isfinite(delta) else float("nan"),
        "has_before_token": int(bool(chain.rows.get("before_token_scale"))),
        "has_after_token": int(bool(chain.rows.get("after_token_scale"))),
        "has_after_clip": int(bool(chain.rows.get("after_clip"))),
        "has_adam": int(bool(chain.rows.get("adam"))),
        "has_cast": int(bool(chain.rows.get("cast"))),
    }


def print_coverage(rows: list[dict[str, float | str | int]]) -> None:
    total = len(rows)
    print(f"chains={total}")
    for stage in ("before_token", "after_token", "after_clip", "adam", "cast"):
        key = f"has_{stage}"
        count = sum(int(row[key]) for row in rows)
        print(f"  {stage:13s} {count:6d} / {total}")
    full = sum(
        int(row["has_after_token"])
        and int(row["has_after_clip"])
        and int(row["has_adam"])
        and int(row["has_cast"])
        for row in rows
    )
    print(f"  full_chain     {full:6d} / {total}")


def print_owner_summary(rows: list[dict[str, float | str | int]]) -> None:
    grouped: dict[str, list[dict[str, float | str | int]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["owner"])].append(row)
    print("\nowner summary:")
    header = (
        "owner                         chains full  "
        "after_tok_rms clip_scale delta/param_p50 delta/param_p90 "
        "cast/update_p50 cast/update_p90 cast/param_p50"
    )
    print(header)
    for owner in sorted(grouped):
        vals = grouped[owner]
        full = [
            row
            for row in vals
            if row["has_after_token"] and row["has_after_clip"] and row["has_adam"] and row["has_cast"]
        ]
        base = full if full else vals
        print(
            f"{owner:29s} {len(vals):6d} {len(full):4d}  "
            f"{fmt(median(float(r['after_token_rms']) for r in base)):>13s} "
            f"{fmt(median(float(r['clip_scale']) for r in base)):>10s} "
            f"{fmt(median(float(r['adam_delta_over_param']) for r in base)):>15s} "
            f"{fmt(quantile((float(r['adam_delta_over_param']) for r in base), 0.9)):>15s} "
            f"{fmt(median(float(r['cast_error_over_update']) for r in base)):>15s} "
            f"{fmt(quantile((float(r['cast_error_over_update']) for r in base), 0.9)):>15s} "
            f"{fmt(median(float(r['cast_error_over_param']) for r in base)):>14s}"
        )


def print_top(
    rows: list[dict[str, float | str | int]],
    *,
    key: str,
    title: str,
    top_n: int,
) -> None:
    def value(row: dict[str, float | str | int]) -> float:
        x = float(row[key])
        return x if math.isfinite(x) else float("-inf")

    print(f"\n{title}:")
    for row in sorted(rows, key=value, reverse=True)[:top_n]:
        print(
            f"rank={int(row['rank']):3d} owner={str(row['owner']):24s} "
            f"{key}={fmt(float(row[key]))} "
            f"clip={fmt(float(row['clip_scale']))} "
            f"after_tok={fmt(float(row['after_token_rms']))} "
            f"after_clip={fmt(float(row['after_clip_rms']))} "
            f"delta_rms={fmt(float(row['adam_delta_rms']))} "
            f"cast/update={fmt(float(row['cast_error_over_update']))} "
            f"name={row['name']}"
        )


def parse_iteration(value: str, chains: dict[tuple[int, int, str], Chain]) -> int:
    iterations = sorted({key[1] for key in chains})
    if not iterations:
        raise SystemExit("No parseable update-pipeline rows found.")
    if value == "latest":
        return iterations[-1]
    iteration = int(value)
    if iteration not in iterations:
        raise SystemExit(f"Requested iter {iteration}, available={iterations}")
    return iteration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=pathlib.Path)
    parser.add_argument("--iteration", default="latest", help="iteration number or latest")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    chains = parse_logs(args.log_dir)
    available = sorted({key[1] for key in chains})
    iteration = parse_iteration(str(args.iteration), chains)
    rows = [chain_row(chain) for chain in chains.values() if chain.iteration == iteration]
    print(f"log_dir={args.log_dir}")
    print(f"available_iterations={available}")
    print(f"selected_iteration={iteration}")
    print_coverage(rows)
    print_owner_summary(rows)
    print_top(
        rows,
        key="adam_delta_over_param",
        title=f"top {args.top_n} Adam update / parameter norm",
        top_n=args.top_n,
    )
    print_top(
        rows,
        key="cast_error_over_update",
        title=f"top {args.top_n} NVFP4 cast error / Adam update",
        top_n=args.top_n,
    )
    print_top(
        rows,
        key="after_token_rms",
        title=f"top {args.top_n} token-scaled gradient RMS",
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
