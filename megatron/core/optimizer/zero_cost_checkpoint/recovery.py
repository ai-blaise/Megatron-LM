# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import os
from typing import Any

import torch

from megatron.core.utils import to_local_if_dtensor

from .persistence import load_atomic_torch_save
from .snapshot_spec import SnapshotPlanner


def load_zcc_state_dict(
    path: str,
    mode: str = "auto",
    *,
    durable_dir: str | None = None,
) -> dict[str, Any]:
    modes = ("flash", "peer", "durable") if mode == "auto" else (mode,)
    errors = []
    for candidate in modes:
        try:
            if candidate == "peer":
                return _load_from_peer(path)
            return load_atomic_torch_save(
                _resolve_snapshot_path(path, candidate, durable_dir=durable_dir)
            )
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("Unable to load ZCC snapshot; " + "; ".join(errors))


def restore_zcc_state(
    megatron_optimizer: Any,
    payload: dict[str, Any],
    *,
    extra_tensor_attrs: tuple[str, ...] = (),
    restore_rng: bool = True,
    restore_dither: bool = True,
    opt_param_scheduler: Any | None = None,
) -> tuple[list[str], list[str]]:
    if not extra_tensor_attrs:
        manager = getattr(megatron_optimizer, "zero_cost_checkpoint_manager", None)
        config = getattr(manager, "config", None)
        extra_tensor_attrs = getattr(config, "extra_tensor_attrs", ())
    specs = {
        spec.name: spec
        for spec in SnapshotPlanner(extra_tensor_attrs).plan(megatron_optimizer)
    }
    payload_tensors = {item["name"]: item for item in payload.get("tensors", [])}
    missing = [name for name in specs if name not in payload_tensors]
    unexpected = [name for name in payload_tensors if name not in specs]
    with torch.no_grad():
        for name, item in payload_tensors.items():
            spec = specs.get(name)
            if spec is None:
                continue
            local_tensor = to_local_if_dtensor(spec.tensor)
            data = item["data"].to(device=local_tensor.device, dtype=local_tensor.dtype)
            local_tensor.copy_(data.view(local_tensor.shape), non_blocking=local_tensor.is_cuda)

    metadata = payload.get("metadata", {})
    if opt_param_scheduler is not None and "opt_param_scheduler" in metadata:
        opt_param_scheduler.load_state_dict(metadata["opt_param_scheduler"])
    if restore_rng:
        if "torch_rng_state" in metadata:
            torch.set_rng_state(metadata["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in metadata:
            torch.cuda.set_rng_state_all(metadata["cuda_rng_state_all"])
    if restore_dither and "dither_step_counter" in metadata:
        try:
            from megatron.core.optimizer import nvfp4_sr

            nvfp4_sr._DITHER_STEP_COUNTER[0] = int(metadata["dither_step_counter"])
        except (ImportError, AttributeError, IndexError, TypeError, ValueError):
            pass
    return missing, unexpected


def _resolve_snapshot_path(
    path: str,
    tier: str,
    *,
    durable_dir: str | None = None,
) -> str:
    if os.path.isfile(path):
        return path
    if tier == "durable" and durable_dir:
        rank_dir = os.path.basename(os.path.normpath(path))
        step_dir = os.path.basename(os.path.dirname(os.path.normpath(path)))
        return os.path.join(durable_dir, step_dir, rank_dir, "zcc_durable.pt")
    filename = "zcc_snapshot.pt" if tier == "flash" else "zcc_durable.pt"
    return os.path.join(path, filename)


def _load_from_peer(path: str) -> dict[str, Any]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("peer recovery requires an initialized process group")
    raise RuntimeError(
        "peer recovery must be initiated by recover_from_peer() with donor and replacement ranks"
    )


def recover_from_peer(
    *,
    donor_rank: int,
    receiver_rank: int,
    tensor: torch.Tensor,
    group: torch.distributed.ProcessGroup | None = None,
) -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return False
    rank = torch.distributed.get_rank()
    if rank == donor_rank:
        torch.distributed.send(tensor.cpu(), dst=receiver_rank, group=group)
        return True
    if rank == receiver_rank:
        incoming = torch.empty_like(tensor, device="cpu")
        torch.distributed.recv(incoming, src=donor_rank, group=group)
        tensor.copy_(incoming.to(tensor.device), non_blocking=True)
        return True
    return False
