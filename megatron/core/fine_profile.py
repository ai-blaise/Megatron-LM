# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in fine-grained profiler ranges for custom training hot paths."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import nullcontext
from fnmatch import fnmatch

import torch


def _env_flag(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _rank_selected(spec: str, rank: int) -> bool:
    spec = (spec or "0").strip()
    if spec.lower() in {"all", "*"}:
        return True
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = item.split("-", 1)
            if lo.strip().isdigit() and hi.strip().isdigit() and int(lo) <= rank <= int(hi):
                return True
        elif item.isdigit() and int(item) == rank:
            return True
        elif fnmatch(str(rank), item):
            return True
    return False


def fine_profile_enabled() -> bool:
    if not _env_flag("MEGATRON_FINE_PROFILE", "0"):
        return False
    ranks = os.getenv("MEGATRON_FINE_PROFILE_RANKS", os.getenv("PROFILE_RANKS", "0"))
    return _rank_selected(ranks, _distributed_rank())


def _timer_enabled() -> bool:
    return _env_flag("MEGATRON_FINE_PROFILE_TIMERS", "0")


def _range_selected(name: str) -> bool:
    spec = os.getenv("MEGATRON_FINE_PROFILE_TIMER_FILTER", "*")
    for item in spec.split(","):
        item = item.strip()
        if item and fnmatch(name, item):
            return True
    return False


_PENDING_TIMERS = []


class _FineProfileTimer:
    def __init__(self, name: str):
        self.name = name
        self.start_wall = 0.0
        self.end_wall = 0.0
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        self.start_wall = time.perf_counter()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.end_event is not None:
            self.end_event.record()
        self.end_wall = time.perf_counter()
        _PENDING_TIMERS.append(self)
        return False


def fine_profile_range(name: str):
    if not fine_profile_enabled() or not _range_selected(name):
        return nullcontext()
    if _timer_enabled():
        return _FineProfileTimer(name)
    return torch.profiler.record_function(name)


def fine_profile_flush(iteration: int | None = None) -> None:
    if not fine_profile_enabled() or not _timer_enabled() or not _PENDING_TIMERS:
        return

    pending = list(_PENDING_TIMERS)
    _PENDING_TIMERS.clear()
    if any(item.end_event is not None for item in pending):
        torch.cuda.synchronize()

    stats = defaultdict(lambda: [0, 0.0, 0.0])
    for item in pending:
        if item.start_event is not None and item.end_event is not None:
            try:
                elapsed_ms = float(item.start_event.elapsed_time(item.end_event))
            except (RuntimeError, ValueError):
                elapsed_ms = (item.end_wall - item.start_wall) * 1000.0
        else:
            elapsed_ms = (item.end_wall - item.start_wall) * 1000.0
        stat = stats[item.name]
        stat[0] += 1
        stat[1] += elapsed_ms
        stat[2] = max(stat[2], elapsed_ms)

    rank = _distributed_rank()
    top_n = int(os.getenv("MEGATRON_FINE_PROFILE_TIMER_TOP_N", "60"))
    min_total_ms = float(os.getenv("MEGATRON_FINE_PROFILE_TIMER_MIN_TOTAL_MS", "0"))
    total_ms = sum(stat[1] for stat in stats.values())
    iter_text = "none" if iteration is None else str(iteration)
    print(
        f"[fine_profile] rank={rank} iteration={iter_text} "
        f"ranges={len(pending)} unique={len(stats)} total_ms={total_ms:.3f}",
        flush=True,
    )
    sorted_items = sorted(stats.items(), key=lambda item: item[1][1], reverse=True)
    for name, (count, total, max_ms) in sorted_items[:top_n]:
        if total < min_total_ms:
            continue
        print(
            f"[fine_profile] rank={rank} iteration={iter_text} "
            f"total_ms={total:.3f} avg_ms={total / count:.3f} max_ms={max_ms:.3f} "
            f"count={count} name={name}",
            flush=True,
        )
