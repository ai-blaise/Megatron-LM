# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import os
import subprocess


def local_gpu_numa_node(local_rank: int) -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    gpu = f"GPU{local_rank}"
    for line in out.splitlines():
        if not line.startswith(gpu):
            continue
        parts = line.split()
        if parts and parts[-1].lstrip("-").isdigit():
            node = int(parts[-1])
            return node if node >= 0 else None
    return None


def pin_process_to_numa_node(node: int | None) -> bool:
    if node is None or not hasattr(os, "sched_setaffinity"):
        return False
    cpu_dir = f"/sys/devices/system/node/node{node}"
    try:
        cpulist = open(os.path.join(cpu_dir, "cpulist"), encoding="utf-8").read().strip()
    except OSError:
        return False
    cpus = _parse_cpulist(cpulist)
    if not cpus:
        return False
    os.sched_setaffinity(0, cpus)
    return True


def _parse_cpulist(cpulist: str) -> set[int]:
    cpus: set[int] = set()
    for part in cpulist.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        elif part:
            cpus.add(int(part))
    return cpus
