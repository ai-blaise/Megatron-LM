# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Signal-driven CRIU hooks for flashtraining Megatron workers."""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from typing import Any

logger = logging.getLogger(__name__)

STATE_DIR = os.environ.get("MEGATRON_CRIU_STATE_DIR", "/var/run/megatron-criu")
READY_FILE = os.path.join(STATE_DIR, "pre_snapshot.ready")
READY_ERR_FILE = READY_FILE + ".err"
RESUME_FILE = os.path.join(STATE_DIR, "post_restore.done")
PRE_SNAPSHOT_SIGNAL = signal.SIGRTMIN + 5
POST_RESTORE_SIGNAL = signal.SIGRTMIN + 6

_installed = False
_snapshot_state: dict[str, Any] = {}


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _write_json(path: str, payload: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
    os.replace(tmp, path)


def _write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _clear_state_files() -> None:
    for path in (READY_FILE, READY_ERR_FILE, RESUME_FILE):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _torch_cuda_synchronize() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _distributed_state() -> dict[str, Any] | None:
    try:
        import torch.distributed as dist
    except ImportError:
        return None
    if not dist.is_available() or not dist.is_initialized():
        return None
    state = {
        "backend": str(dist.get_backend()),
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "master_addr": os.environ.get("MASTER_ADDR", ""),
        "master_port": os.environ.get("MASTER_PORT", ""),
    }
    dist.destroy_process_group()
    return state


def _restore_distributed(state: dict[str, Any]) -> None:
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return
    if dist.is_available() and dist.is_initialized():
        return
    if state.get("master_addr"):
        os.environ["MASTER_ADDR"] = state["master_addr"]
    if state.get("master_port"):
        os.environ["MASTER_PORT"] = state["master_port"]
    dist.init_process_group(
        backend=state["backend"],
        rank=int(state["rank"]),
        world_size=int(state["world_size"]),
    )
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        tensor = torch.zeros(1, device=device)
        dist.all_reduce(tensor)
    dist.barrier()


def _pre_snapshot_handler(signum: int, frame: Any) -> None:
    del signum, frame
    try:
        _clear_state_files()
        if _env_enabled("MEGATRON_CRIU_SYNC_IN_SIGNAL"):
            _torch_cuda_synchronize()
        dist_state = None
        if _env_enabled("MEGATRON_CRIU_DISTRIBUTED_IN_SIGNAL"):
            dist_state = _distributed_state()
        if dist_state is not None:
            _snapshot_state["distributed"] = dist_state
        if _env_enabled("MEGATRON_CRIU_SYNC_IN_SIGNAL"):
            _torch_cuda_synchronize()
        _write_json(
            READY_FILE,
            {
                "rank": os.environ.get("RANK", ""),
                "ts": time.time(),
                "distributed": dist_state is not None,
            },
        )
        logger.info("Megatron CRIU pre_snapshot complete")
    except Exception as exc:
        logger.exception("Megatron CRIU pre_snapshot failed")
        _write_text(READY_ERR_FILE, repr(exc))


def _post_restore_handler(signum: int, frame: Any) -> None:
    del signum, frame
    try:
        dist_state = _snapshot_state.pop("distributed", None)
        if dist_state is not None:
            _restore_distributed(dist_state)
        if _env_enabled("MEGATRON_CRIU_SYNC_IN_SIGNAL"):
            _torch_cuda_synchronize()
        _write_json(
            RESUME_FILE,
            {
                "rank": os.environ.get("RANK", ""),
                "ts": time.time(),
                "distributed": dist_state is not None,
            },
        )
        logger.info("Megatron CRIU post_restore complete")
    except Exception:
        logger.exception("Megatron CRIU post_restore failed")
        os._exit(1)


def install() -> None:
    global _installed
    if _installed:
        return
    os.makedirs(STATE_DIR, exist_ok=True)
    signal.signal(PRE_SNAPSHOT_SIGNAL, _pre_snapshot_handler)
    signal.signal(POST_RESTORE_SIGNAL, _post_restore_handler)
    _installed = True
    logger.info("Megatron CRIU hooks installed")


if os.environ.get("MEGATRON_CRIU_ENABLE") == "1":
    install()
