#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validate PyTorch profiler artifacts before treating a profile run as usable."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def iter_trace_events(path: Path) -> Iterable[dict[str, Any]]:
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
                            raise RuntimeError("traceEvents array not found")
                        buf = buf[-64:]
                        continue
                    arr = buf.find("[", idx)
                    if arr < 0:
                        if not chunk:
                            raise RuntimeError("traceEvents array start not found")
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
                    # Force gzip footer verification for .gz inputs.
                    for _ in f:
                        pass
                    return
                try:
                    event, end = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    if not chunk:
                        raise RuntimeError("traceEvents ended with incomplete JSON event")
                    if len(buf) > 128 * 1024 * 1024:
                        raise RuntimeError("trace event decoder buffer exceeded 128MiB")
                    break
                if isinstance(event, dict):
                    yield event
                buf = buf[end:]

            if not chunk:
                raise RuntimeError("traceEvents array did not close")


def discover_traces(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    traces: list[Path] = []
    for pattern in ("rank-*.json.gz", "rank-*.json"):
        traces.extend(root.rglob(pattern))
    return sorted(set(traces))


def rank_from_path(path: Path) -> int | None:
    match = re.search(r"rank-(\d+)", path.name)
    return int(match.group(1)) if match else None


def parse_rank_spec(spec: str | None) -> set[int]:
    if not spec:
        return set()
    ranks: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = item.split("-", 1)
            ranks.update(range(int(lo), int(hi) + 1))
        else:
            ranks.add(int(item))
    return ranks


def event_kind(event: dict[str, Any]) -> str:
    name = str(event.get("name", ""))
    cat = str(event.get("cat", ""))
    hay = f"{name} {cat}".lower()
    if "kernel" in hay:
        return "kernel"
    if "cuda" in hay:
        return "cuda"
    if name.startswith(("dsa.", "hisa.", "moe.", "fsdp.", "pipeline.")):
        return "custom"
    if name.startswith("ProfilerStep"):
        return "profiler_step"
    if name.startswith("aten::"):
        return "aten"
    if "memory" in hay or "alloc" in hay:
        return "memory"
    return "other"


def validate_trace(path: Path, min_bytes: int, min_events: int) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    stat = path.stat()
    rank = rank_from_path(path)
    counts: Counter[str] = Counter()
    total_events = 0
    first_ts: float | None = None
    last_ts: float | None = None
    if stat.st_size < min_bytes:
        errors.append(f"file too small: {stat.st_size} bytes < {min_bytes}")
    try:
        for event in iter_trace_events(path):
            total_events += 1
            counts[event_kind(event)] += 1
            ts = event.get("ts")
            if isinstance(ts, (int, float)):
                first_ts = float(ts) if first_ts is None else min(first_ts, float(ts))
                last_ts = float(ts) if last_ts is None else max(last_ts, float(ts))
    except (EOFError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        errors.append(f"unreadable trace: {type(exc).__name__}: {exc}")
    if total_events < min_events:
        errors.append(f"too few events: {total_events} < {min_events}")

    return (
        {
            "path": str(path),
            "rank": rank,
            "bytes": stat.st_size,
            "events": total_events,
            "kinds": dict(counts),
            "first_ts": first_ts,
            "last_ts": last_ts,
        },
        errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Trace file or directory tree")
    parser.add_argument("--require-ranks", default="", help="Comma/range rank list, e.g. 0,7,88,95")
    parser.add_argument("--min-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--min-events", type=int, default=100)
    parser.add_argument("--require-cuda-events", action="store_true")
    parser.add_argument("--json", type=Path, help="Optional JSON summary output")
    args = parser.parse_args()

    traces = discover_traces(args.input)
    required_ranks = parse_rank_spec(args.require_ranks)
    errors: list[str] = []
    summaries = []
    seen_ranks: set[int] = set()
    total_kind_counts: Counter[str] = Counter()

    if not traces:
        errors.append(f"no rank trace files found under {args.input}")

    for trace in traces:
        summary, trace_errors = validate_trace(trace, args.min_bytes, args.min_events)
        summaries.append(summary)
        if summary["rank"] is not None:
            seen_ranks.add(int(summary["rank"]))
        total_kind_counts.update(summary["kinds"])
        errors.extend(f"{trace}: {err}" for err in trace_errors)

    missing = sorted(required_ranks - seen_ranks)
    if missing:
        errors.append(f"missing required ranks: {missing}")
    if args.require_cuda_events and total_kind_counts["kernel"] + total_kind_counts["cuda"] == 0:
        errors.append("no CUDA/kernel events found in parsed traces")

    report = {
        "input": str(args.input),
        "trace_count": len(traces),
        "seen_ranks": sorted(seen_ranks),
        "required_ranks": sorted(required_ranks),
        "kind_counts": dict(total_kind_counts),
        "traces": summaries,
        "errors": errors,
        "ok": not errors,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        "profile_validation "
        f"ok={report['ok']} traces={len(traces)} ranks={report['seen_ranks']} "
        f"kinds={report['kind_counts']}"
    )
    for err in errors:
        print(f"ERROR: {err}", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
