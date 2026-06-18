#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fast line-oriented reducer for pretty-printed PyTorch Chrome traces.

PyTorch's exported traces in this run are multi-line JSON, but each event's
``cat``/``name`` line is followed by a ``ts``/``dur`` line. Full JSON parsing is
too slow for 100GB+ decompressed traces; this reducer extracts the fields we
need directly while streaming stdin.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import BinaryIO


CAT_RE = re.compile(rb'"cat":\s*"([^"]*)"')
NAME_RE = re.compile(rb'"name":\s*"([^"]*)"')
DUR_RE = re.compile(rb'"dur":\s*([-+0-9.eE]+)')
TS_RE = re.compile(rb'"ts":\s*([-+0-9.eE]+)')


def empty_stat() -> dict[str, float | int]:
    return {"count": 0, "total_us": 0.0, "max_us": 0.0}


def add_stat(stats: dict[str, dict[str, float | int]], key: str, dur: float) -> None:
    stat = stats[key]
    stat["count"] = int(stat["count"]) + 1
    stat["total_us"] = float(stat["total_us"]) + dur
    stat["max_us"] = max(float(stat["max_us"]), dur)


def decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def groups_for(name: str, cat: str) -> list[str]:
    hay = f"{name} {cat}".lower()
    out: list[str] = []
    if name.startswith("ProfilerStep"):
        out.append("profiler_step")
    if name.startswith(("dsa.", "hisa.", "moe.", "fsdp.", "pipeline.")):
        out.append("custom_range")
    if "hisa" in hay or "_hisa" in hay:
        out.append("hisa")
    if "dsa" in hay or "dsattention" in hay:
        out.append("dsa")
    if "moe" in hay or "expert" in hay or "deepep" in hay:
        out.append("moe")
    if "fsdp" in hay or "all_gather_params" in hay:
        out.append("fsdp")
    if any(token in hay for token in ("nccl", "allreduce", "all_reduce", "alltoall", "all_to_all", "reduce_scatter", "broadcast")):
        out.append("comm")
    if any(token in hay for token in ("memcpy", "memset", "copy")) or name == "aten::copy_":
        out.append("copy_mem")
    if name.startswith("aten::"):
        out.append("aten")
    if "kernel" in hay and not name.startswith("cudaLaunchKernel"):
        out.append("cuda_kernel")
    if "cuda" in hay or name.startswith("cuda"):
        out.append("cuda_runtime")
    if not out:
        out.append("other")
    return out


def reduce_stream(stream: BinaryIO, rank: int) -> dict[str, object]:
    groups: dict[str, dict[str, float | int]] = defaultdict(empty_stat)
    names: dict[str, dict[str, dict[str, float | int]]] = defaultdict(lambda: defaultdict(empty_stat))
    events = 0
    first_ts: float | None = None
    last_ts: float | None = None
    current_name: str | None = None
    current_cat = ""

    for line in stream:
        if b'"name":' in line:
            name_match = NAME_RE.search(line)
            if name_match:
                current_name = decode(name_match.group(1))
                cat_match = CAT_RE.search(line)
                current_cat = decode(cat_match.group(1)) if cat_match else ""
        if current_name is None or b'"dur":' not in line:
            continue
        dur_match = DUR_RE.search(line)
        if not dur_match:
            continue
        try:
            dur = float(dur_match.group(1))
        except ValueError:
            continue
        ts_match = TS_RE.search(line)
        if ts_match:
            try:
                ts = float(ts_match.group(1))
            except ValueError:
                ts = None
            if ts is not None:
                first_ts = ts if first_ts is None else min(first_ts, ts)
                last_ts = ts if last_ts is None else max(last_ts, ts)
        events += 1
        for group in groups_for(current_name, current_cat):
            add_stat(groups, group, dur)
            add_stat(names[group], current_name, dur)
        current_name = None
        current_cat = ""

    return {
        "rank": rank,
        "events": events,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "groups": groups,
        "names": names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = reduce_stream(sys.stdin.buffer, args.rank)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
