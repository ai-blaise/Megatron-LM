#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Summarize exported PyTorch Chrome traces into lightweight HTML/Markdown reports."""

from __future__ import annotations

import argparse
import gzip
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Stat:
    count: int = 0
    total_us: float = 0.0
    max_us: float = 0.0
    bytes_total: float = 0.0
    bytes_max: float = 0.0
    ranks: set[int] = field(default_factory=set)

    def add(self, rank: int, dur_us: float, byte_value: float = 0.0) -> None:
        self.count += 1
        self.total_us += dur_us
        self.max_us = max(self.max_us, dur_us)
        self.bytes_total += byte_value
        self.bytes_max = max(self.bytes_max, byte_value)
        self.ranks.add(rank)

    @property
    def avg_us(self) -> float:
        return self.total_us / self.count if self.count else 0.0


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def iter_trace_events(path: Path) -> Iterable[dict[str, Any]]:
    """Yield traceEvents without loading multi-GB Chrome traces into memory."""
    decoder = json.JSONDecoder()
    in_events = False
    buf = ""

    with open_text(path) as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk and not buf:
                return
            buf += chunk

            if not in_events:
                stripped = buf.lstrip()
                if stripped.startswith("["):
                    buf = stripped[1:]
                    in_events = True
                else:
                    idx = buf.find('"traceEvents"')
                    if idx < 0:
                        if not chunk:
                            return
                        buf = buf[-64:]
                        continue
                    arr = buf.find("[", idx)
                    if arr < 0:
                        if not chunk:
                            return
                        buf = buf[idx:]
                        continue
                    buf = buf[arr + 1 :]
                    in_events = True

            while in_events:
                buf = buf.lstrip()
                if not buf:
                    break
                if buf[0] == ",":
                    buf = buf[1:]
                    continue
                if buf[0] == "]":
                    return
                try:
                    event, end = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    if not chunk:
                        return
                    if len(buf) > 128 * 1024 * 1024:
                        raise RuntimeError(f"Could not decode trace event in {path}; buffer grew too large")
                    break
                if isinstance(event, dict):
                    yield event
                buf = buf[end:]

            if not chunk:
                return


def rank_from_path(path: Path) -> int:
    match = re.search(r"rank-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def event_category(event: dict[str, Any]) -> str:
    name = str(event.get("name", ""))
    cat = str(event.get("cat", ""))
    hay = f"{name} {cat}".lower()
    if "memory" in hay or "allocation" in hay or "allocator" in hay:
        return "memory"
    if "nccl" in hay or "alltoall" in hay or "all_to_all" in hay or "all-reduce" in hay:
        return "comm"
    if name.startswith(("dsa.", "hisa.", "moe.", "pipeline.", "fsdp.")):
        return "custom_range"
    if name.startswith("aten::"):
        return "aten"
    if "kernel" in hay or "cuda_kernel" in hay:
        return "cuda_kernel"
    if "cuda" in hay or "cuda_runtime" in hay:
        return "cuda_runtime"
    return "other"


def extract_bytes(value: Any, key_hint: str = "") -> float:
    total = 0.0
    if isinstance(value, dict):
        for key, child in value.items():
            total += extract_bytes(child, str(key))
    elif isinstance(value, list):
        for child in value:
            total += extract_bytes(child, key_hint)
    elif isinstance(value, (int, float)) and any(
        token in key_hint.lower() for token in ("byte", "memory", "alloc", "reserved")
    ):
        total += float(value)
    return total


def discover_traces(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    traces = []
    for pattern in ("rank-*.json.gz", "rank-*.json"):
        traces.extend(root.rglob(pattern))
    return sorted(set(traces))


def fmt_time(us: float) -> str:
    if us >= 1_000_000:
        return f"{us / 1_000_000:.3f}s"
    if us >= 1_000:
        return f"{us / 1_000:.3f}ms"
    return f"{us:.1f}us"


def fmt_bytes(num_bytes: float) -> str:
    if num_bytes <= 0:
        return ""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TiB"


def top_items(stats: dict[str, Stat], top: int) -> list[tuple[str, Stat]]:
    return sorted(stats.items(), key=lambda kv: (kv[1].total_us, kv[1].count), reverse=True)[:top]


def table_rows(items: list[tuple[str, Stat]], max_total: float) -> str:
    rows = []
    for name, stat in items:
        width = 0.0 if max_total <= 0 else min(100.0, 100.0 * stat.total_us / max_total)
        ranks = ",".join(str(r) for r in sorted(stat.ranks)[:12])
        if len(stat.ranks) > 12:
            ranks += ",..."
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(name)}</code></td>"
            f"<td class='num'>{stat.count}</td>"
            f"<td class='num'>{fmt_time(stat.total_us)}</td>"
            f"<td class='num'>{fmt_time(stat.avg_us)}</td>"
            f"<td class='num'>{fmt_time(stat.max_us)}</td>"
            f"<td class='num'>{fmt_bytes(stat.bytes_max)}</td>"
            f"<td><div class='bar'><span style='width:{width:.1f}%'></span></div></td>"
            f"<td>{html.escape(ranks)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def html_section(title: str, items: list[tuple[str, Stat]]) -> str:
    max_total = items[0][1].total_us if items else 0.0
    return f"""
<section>
  <h2>{html.escape(title)}</h2>
  <table>
    <thead><tr><th>Name</th><th>Count</th><th>Total</th><th>Avg</th><th>Max</th><th>Max Bytes</th><th>Share</th><th>Ranks</th></tr></thead>
    <tbody>
      {table_rows(items, max_total)}
    </tbody>
  </table>
</section>
"""


def markdown_section(title: str, items: list[tuple[str, Stat]]) -> str:
    lines = [f"## {title}", "", "| Name | Count | Total | Avg | Max | Max Bytes | Ranks |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, stat in items:
        ranks = ",".join(str(r) for r in sorted(stat.ranks)[:12])
        if len(stat.ranks) > 12:
            ranks += ",..."
        lines.append(
            f"| `{name}` | {stat.count} | {fmt_time(stat.total_us)} | {fmt_time(stat.avg_us)} | "
            f"{fmt_time(stat.max_us)} | {fmt_bytes(stat.bytes_max)} | {ranks} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Trace file or directory tree")
    parser.add_argument("--output", required=True, type=Path, help="HTML report path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--text", action="store_true", help="Print a concise terminal summary")
    args = parser.parse_args()

    traces = discover_traces(args.input)
    by_category: dict[str, Stat] = defaultdict(Stat)
    by_name: dict[str, Stat] = defaultdict(Stat)
    by_custom: dict[str, Stat] = defaultdict(Stat)
    by_kernel: dict[str, Stat] = defaultdict(Stat)
    by_aten: dict[str, Stat] = defaultdict(Stat)
    by_comm: dict[str, Stat] = defaultdict(Stat)
    by_memory: dict[str, Stat] = defaultdict(Stat)
    per_trace: dict[str, Stat] = defaultdict(Stat)
    parsed_events = 0

    for trace in traces:
        rank = rank_from_path(trace)
        trace_key = f"rank-{rank}" if rank >= 0 else trace.name
        for event in iter_trace_events(trace):
            parsed_events += 1
            name = str(event.get("name", "<unnamed>"))
            category = event_category(event)
            dur_us = float(event.get("dur") or 0.0)
            byte_value = extract_bytes(event.get("args", {}))
            by_category[category].add(rank, dur_us, byte_value)
            by_name[name].add(rank, dur_us, byte_value)
            per_trace[trace_key].add(rank, dur_us, byte_value)
            if category == "custom_range":
                by_custom[name].add(rank, dur_us, byte_value)
            elif category == "cuda_kernel":
                by_kernel[name].add(rank, dur_us, byte_value)
            elif category == "aten":
                by_aten[name].add(rank, dur_us, byte_value)
            elif category == "comm":
                by_comm[name].add(rank, dur_us, byte_value)
            elif category == "memory":
                by_memory[name].add(rank, dur_us, byte_value)

    sections = [
        ("Category Totals", top_items(by_category, args.top)),
        ("Custom Ranges: DSA/HISA/MoE/Pipeline", top_items(by_custom, args.top)),
        ("CUDA Kernels", top_items(by_kernel, args.top)),
        ("ATen Ops", top_items(by_aten, args.top)),
        ("Communication", top_items(by_comm, args.top)),
        ("Memory Events", top_items(by_memory, args.top)),
        ("All Events", top_items(by_name, args.top)),
        ("Per Trace", top_items(per_trace, args.top)),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(html_section(title, items) for title, items in sections)
    args.output.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Megatron Profile Trace Report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #151515; }}
h1 {{ margin-bottom: 0; }}
.meta {{ color: #555; margin: 8px 0 24px; }}
section {{ margin: 28px 0; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
th {{ text-align: left; background: #f4f4f4; position: sticky; top: 0; }}
.num {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
.bar {{ min-width: 120px; height: 12px; background: #ececec; border-radius: 2px; overflow: hidden; }}
.bar span {{ display: block; height: 100%; background: linear-gradient(90deg, #2b6cb0, #38a169); }}
</style>
</head>
<body>
<h1>Megatron Profile Trace Report</h1>
<div class="meta">traces={len(traces)} events={parsed_events} input={html.escape(str(args.input))}</div>
{body}
</body>
</html>
""",
        encoding="utf-8",
    )

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            "\n".join(
                [
                    "# Megatron Profile Trace Report",
                    "",
                    f"traces: {len(traces)}",
                    f"events: {parsed_events}",
                    f"input: `{args.input}`",
                    "",
                    *(markdown_section(title, items) for title, items in sections),
                ]
            ),
            encoding="utf-8",
        )

    if args.text:
        print(f"traces={len(traces)} events={parsed_events}")
        for title, items in sections[:4]:
            print(f"\n== {title} ==")
            for name, stat in items[: min(args.top, 12)]:
                print(
                    f"{fmt_time(stat.total_us):>10} count={stat.count:<8} "
                    f"avg={fmt_time(stat.avg_us):>10} max={fmt_time(stat.max_us):>10} {name}"
                )
        print(f"\nhtml={args.output}")
        if args.markdown:
            print(f"markdown={args.markdown}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
