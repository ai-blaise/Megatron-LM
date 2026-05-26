#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Render compact Markdown from jq-reduced PyTorch trace summaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def fmt_time(us: float) -> str:
    if us >= 1_000_000:
        return f"{us / 1_000_000:.3f}s"
    if us >= 1_000:
        return f"{us / 1_000:.3f}ms"
    return f"{us:.1f}us"


def merge_stat(dst: dict[str, Any], src: dict[str, Any], rank: int) -> None:
    dst["count"] += int(src.get("count", 0))
    dst["total_us"] += float(src.get("total_us", 0.0))
    dst["max_us"] = max(float(dst["max_us"]), float(src.get("max_us", 0.0)))
    dst["ranks"].add(rank)


def empty_stat() -> dict[str, Any]:
    return {"count": 0, "total_us": 0.0, "max_us": 0.0, "ranks": set()}


def stat_avg(stat: dict[str, Any]) -> float:
    return float(stat["total_us"]) / int(stat["count"]) if stat["count"] else 0.0


def rank_text(ranks: set[int]) -> str:
    values = sorted(ranks)
    text = ",".join(str(v) for v in values[:8])
    if len(values) > 8:
        text += ",..."
    return text


def row(name: str, stat: dict[str, Any]) -> str:
    return (
        f"| `{name}` | {stat['count']} | {fmt_time(stat['total_us'])} | "
        f"{fmt_time(stat_avg(stat))} | {fmt_time(stat['max_us'])} | {rank_text(stat['ranks'])} |"
    )


def top_items(stats: dict[str, dict[str, Any]], top: int) -> list[tuple[str, dict[str, Any]]]:
    return sorted(stats.items(), key=lambda kv: (kv[1]["total_us"], kv[1]["count"]), reverse=True)[
        :top
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    summaries = [json.loads(path.read_text()) for path in args.input]
    groups: dict[str, dict[str, Any]] = defaultdict(empty_stat)
    names: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(empty_stat))
    rank_lines = []
    total_events = 0
    for summary in summaries:
        rank = int(summary["rank"])
        total_events += int(summary.get("events", 0))
        first = summary.get("first_ts")
        last = summary.get("last_ts")
        span = max(0.0, float(last) - float(first)) if first is not None and last is not None else 0.0
        rank_lines.append((rank, int(summary.get("events", 0)), span))
        for group, stat in summary.get("groups", {}).items():
            merge_stat(groups[group], stat, rank)
        for group, per_name in summary.get("names", {}).items():
            for name, stat in per_name.items():
                merge_stat(names[group][name], stat, rank)

    lines = [
        "# Profile Trace Reduced Summary",
        "",
        f"reduced_files: {len(args.input)}",
        f"events: {total_events}",
        "",
        "## Rank Trace Spans",
        "",
        "| Rank | Events | Trace span |",
        "| ---: | ---: | ---: |",
    ]
    for rank, events, span in sorted(rank_lines):
        lines.append(f"| {rank} | {events} | {fmt_time(span)} |")

    lines.extend(
        [
            "",
            "## Category Totals",
            "",
            "| Category | Count | Total | Avg | Max | Ranks |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name, stat in top_items(groups, args.top):
        lines.append(row(name, stat))

    for group in (
        "profiler_step",
        "custom_range",
        "hisa",
        "dsa",
        "moe",
        "fsdp",
        "comm",
        "copy_mem",
        "aten",
        "cuda_kernel",
        "cuda_runtime",
    ):
        lines.extend(
            [
                "",
                f"## Top {group}",
                "",
                "| Name | Count | Total | Avg | Max | Ranks |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for name, stat in top_items(names[group], args.top):
            lines.append(row(name, stat))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
