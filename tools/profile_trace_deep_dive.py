#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Targeted reducer for large PyTorch Chrome traces from the A4 DSA/HISA runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from profile_trace_report import discover_traces, fmt_time, iter_trace_events, rank_from_path


@dataclass
class Stat:
    count: int = 0
    total_us: float = 0.0
    max_us: float = 0.0
    ranks: set[int] = field(default_factory=set)

    def add(self, rank: int, dur_us: float) -> None:
        self.count += 1
        self.total_us += dur_us
        self.max_us = max(self.max_us, dur_us)
        self.ranks.add(rank)

    @property
    def avg_us(self) -> float:
        return self.total_us / self.count if self.count else 0.0

    def to_json(self) -> dict[str, object]:
        return {
            "count": self.count,
            "total_us": self.total_us,
            "avg_us": self.avg_us,
            "max_us": self.max_us,
            "ranks": sorted(self.ranks),
        }


def groups_for_event(name: str, cat: str) -> list[str]:
    hay = f"{name} {cat}".lower()
    groups: list[str] = []
    if name.startswith("ProfilerStep"):
        groups.append("profiler_step")
    if name.startswith(("dsa.", "hisa.", "moe.", "fsdp.", "pipeline.")):
        groups.append("custom_range")
    if "hisa" in hay or "_hisa" in hay:
        groups.append("hisa")
    if "dsa" in hay or "dsattention" in hay:
        groups.append("dsa")
    if "moe" in hay or "expert" in hay or "deepep" in hay:
        groups.append("moe")
    if "fsdp" in hay or "all_gather_params" in hay:
        groups.append("fsdp")
    if (
        "nccl" in hay
        or "allreduce" in hay
        or "all_reduce" in hay
        or "alltoall" in hay
        or "all_to_all" in hay
        or "reduce_scatter" in hay
        or "broadcast" in hay
    ):
        groups.append("comm")
    if "memcpy" in hay or "memset" in hay or "copy" in hay or name == "aten::copy_":
        groups.append("copy_mem")
    if name.startswith("aten::"):
        groups.append("aten")
    if "kernel" in hay and not name.startswith("cudaLaunchKernel"):
        groups.append("cuda_kernel")
    if "cuda" in hay or name.startswith("cuda"):
        groups.append("cuda_runtime")
    if not groups:
        groups.append("other")
    return groups


def top_items(stats: dict[str, Stat], top: int) -> list[tuple[str, Stat]]:
    return sorted(stats.items(), key=lambda kv: (kv[1].total_us, kv[1].count), reverse=True)[:top]


def fmt_ranks(ranks: set[int]) -> str:
    values = sorted(ranks)
    out = ",".join(str(x) for x in values[:8])
    if len(values) > 8:
        out += ",..."
    return out


def stat_row(name: str, stat: Stat) -> str:
    return (
        f"| `{name}` | {stat.count} | {fmt_time(stat.total_us)} | "
        f"{fmt_time(stat.avg_us)} | {fmt_time(stat.max_us)} | {fmt_ranks(stat.ranks)} |"
    )


def write_markdown(
    path: Path,
    traces: list[Path],
    events: int,
    by_group: dict[str, Stat],
    by_group_name: dict[str, dict[str, Stat]],
    per_rank_span: dict[int, tuple[float, float]],
    top: int,
) -> None:
    lines: list[str] = [
        "# Profile Trace Deep Dive",
        "",
        f"traces: {len(traces)}",
        f"events: {events}",
        "",
        "## Step And Trace Span",
        "",
        "| Rank | Trace span | ProfilerStep total | ProfilerStep count | ProfilerStep max |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    profiler_stats = by_group_name.get("profiler_step", {})
    for rank in sorted(per_rank_span):
        first, last = per_rank_span[rank]
        step_total = 0.0
        step_count = 0
        step_max = 0.0
        for stat in profiler_stats.values():
            if rank in stat.ranks:
                # The rank-specific total is emitted separately below, so this table is
                # a coarse all-step view. Use the per-name ProfilerStep table for detail.
                step_total += stat.total_us / max(len(stat.ranks), 1)
                step_count += stat.count // max(len(stat.ranks), 1)
                step_max = max(step_max, stat.max_us)
        lines.append(
            f"| {rank} | {fmt_time(max(0.0, last - first))} | "
            f"{fmt_time(step_total)} | {step_count} | {fmt_time(step_max)} |"
        )

    lines.extend(
        [
            "",
            "## Category Totals",
            "",
            "| Category | Count | Total | Avg | Max | Ranks |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name, stat in top_items(by_group, top):
        lines.append(stat_row(name, stat))

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
        for name, stat in top_items(by_group_name.get(group, {}), top):
            lines.append(stat_row(name, stat))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--text", action="store_true")
    args = parser.parse_args()

    traces = discover_traces(args.input)
    by_group: dict[str, Stat] = defaultdict(Stat)
    by_group_name: dict[str, dict[str, Stat]] = defaultdict(lambda: defaultdict(Stat))
    per_rank_span: dict[int, tuple[float, float]] = {}
    events = 0

    for trace in traces:
        rank = rank_from_path(trace)
        first_ts: float | None = None
        last_ts: float | None = None
        for event in iter_trace_events(trace):
            events += 1
            name = str(event.get("name", "<unnamed>"))
            cat = str(event.get("cat", ""))
            dur_us = float(event.get("dur") or 0.0)
            ts = event.get("ts")
            if isinstance(ts, (int, float)):
                first_ts = float(ts) if first_ts is None else min(first_ts, float(ts))
                last_ts = float(ts) if last_ts is None else max(last_ts, float(ts))
            for group in groups_for_event(name, cat):
                by_group[group].add(rank, dur_us)
                by_group_name[group][name].add(rank, dur_us)
        if rank >= 0 and first_ts is not None and last_ts is not None:
            per_rank_span[rank] = (first_ts, last_ts)

    write_markdown(
        args.markdown,
        traces,
        events,
        by_group,
        by_group_name,
        per_rank_span,
        args.top,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "traces": [str(p) for p in traces],
                    "events": events,
                    "groups": {k: v.to_json() for k, v in by_group.items()},
                    "top": {
                        group: {name: stat.to_json() for name, stat in top_items(items, args.top)}
                        for group, items in by_group_name.items()
                    },
                    "per_rank_span_us": {
                        str(rank): max(0.0, last - first)
                        for rank, (first, last) in per_rank_span.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.text:
        print(f"traces={len(traces)} events={events}")
        for group in ("profiler_step", "hisa", "dsa", "moe", "comm", "copy_mem", "aten", "cuda_kernel"):
            print(f"\n== {group} ==")
            for name, stat in top_items(by_group_name.get(group, {}), min(args.top, 12)):
                print(
                    f"{fmt_time(stat.total_us):>10} count={stat.count:<8} "
                    f"avg={fmt_time(stat.avg_us):>10} max={fmt_time(stat.max_us):>10} {name}"
                )
        print(f"\nmarkdown={args.markdown}")
        if args.json:
            print(f"json={args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
