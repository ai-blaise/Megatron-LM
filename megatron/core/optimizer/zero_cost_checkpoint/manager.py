# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import atexit
import os
import shutil
import time
import weakref
from typing import Any

import torch

from .arena import FusedOptimizerStateBuffer, PinnedMirror
from .config import ZeroCostCheckpointConfig
from .cuda_events import CudaEventPool
from .persistence import atomic_torch_save, tensor_payload
from .snapshot_spec import SnapshotPlanner, TensorSnapshotSpec, iter_torch_optimizers
from .worker import AsyncDumpWorker, DumpTask
from megatron.core.utils import to_local_if_dtensor


class ZeroCostCheckpointManager:
    def __init__(
        self,
        megatron_optimizer: Any,
        config: ZeroCostCheckpointConfig,
    ):
        self._optimizer_ref = weakref.ref(megatron_optimizer)
        self.config = config
        self._planner = SnapshotPlanner(config.extra_tensor_attrs)
        self._event_pool = CudaEventPool()
        self._dump_worker = AsyncDumpWorker(config.workers_num)
        self._mirrors: dict[str, PinnedMirror] = {}
        self._last_snapshot_step = 0
        self._initialized = False
        self._stream = self._make_stream()
        atexit.register(self.close)

    def sync_before_step(self) -> None:
        self._dump_worker.check_error()
        if torch.cuda.is_available() and self._stream is not None:
            self._stream.synchronize()
        self._ensure_initialized()

    def snapshot_after_step(
        self,
        step: int,
        *,
        opt_param_scheduler: Any | None = None,
    ) -> None:
        optimizer = self._optimizer_ref()
        if optimizer is None:
            return
        self._ensure_initialized()
        specs = self._planner.plan(optimizer)
        payload = {
            "step": step,
            "created_at": time.time(),
            "metadata": self._planner.metadata(
                optimizer,
                include_dither=self.config.include_dither,
                include_rng=self.config.include_rng,
                opt_param_scheduler=opt_param_scheduler,
            ),
            "tensors": [],
        }
        self._copy_specs_to_mirrors(specs)

        for spec in specs:
            mirror = self._mirrors[spec.name]
            local_tensor = to_local_if_dtensor(spec.tensor)
            restored = mirror.tensor.view(local_tensor.shape).to(dtype=local_tensor.dtype)
            payload["tensors"].append(tensor_payload(spec.name, restored))

        flash_path = self._snapshot_path(self._flash_root(), step, durable=False)
        atomic_torch_save(flash_path, payload, compression="none")
        if self.config.durable_dir and step % self.config.durable_interval == 0:
            durable_path = self._snapshot_path(self.config.durable_dir, step, durable=True)
            self._dump_worker.submit(
                DumpTask(
                    durable_path,
                    payload,
                    compression=self.config.compress,
                )
            )
        self._cleanup_old_snapshots(step)
        self._last_snapshot_step = step

    def snapshot_bucket(self, bucket: Any) -> None:
        optimizer = self._optimizer_ref()
        if optimizer is None:
            return
        self._ensure_initialized()
        self._copy_specs_to_mirrors(self._planner.plan_bucket(optimizer, bucket))

    def finalize(self) -> None:
        self._dump_worker.drain()

    def close(self) -> None:
        self._dump_worker.close()

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        optimizer = self._optimizer_ref()
        if optimizer is None:
            return
        for wrapper in _iter_megatron_optimizers(optimizer):
            init_state_fn = getattr(wrapper, "init_state_fn", None)
            torch_optimizer = getattr(wrapper, "optimizer", None)
            if init_state_fn is not None and isinstance(
                torch_optimizer, torch.optim.Optimizer
            ):
                init_state_fn(torch_optimizer, getattr(wrapper, "config", None))
        for torch_optimizer in iter_torch_optimizers(optimizer):
            setattr(torch_optimizer, "_zcc_manager", self)
            if self.config.fuse_state_buffer:
                FusedOptimizerStateBuffer(torch_optimizer).fuse()
        self._initialized = True

    def _make_stream(self) -> torch.cuda.Stream | None:
        if not torch.cuda.is_available():
            return None
        get_priority_range = getattr(torch.cuda, "get_stream_priority_range", None)
        if get_priority_range is None:
            return torch.cuda.Stream()
        try:
            _, low = get_priority_range()
            return torch.cuda.Stream(priority=low)
        except RuntimeError:
            return torch.cuda.Stream()

    def _copy_specs_to_mirrors(self, specs: list[TensorSnapshotSpec]) -> None:
        event = self._event_pool.acquire()
        for spec in specs:
            source = to_local_if_dtensor(spec.tensor)
            mirror = self._mirrors.get(spec.name)
            if (
                mirror is None
                or mirror.tensor.numel() != source.numel()
                or mirror.tensor.dtype != source.dtype
            ):
                mirror = PinnedMirror(source.detach())
                self._mirrors[spec.name] = mirror
            mirror.copy_from(source.detach(), stream=self._stream)
        if torch.cuda.is_available() and self._stream is not None:
            event.record(self._stream)
            event.synchronize()
            self._event_pool.release(event)

    def _flash_root(self) -> str:
        roots = (self.config.flash_device,) + self.config.flash_stripe
        if len(roots) == 1:
            return roots[0]
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return roots[rank % len(roots)]

    @staticmethod
    def _snapshot_path(root: str, step: int, *, durable: bool) -> str:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        name = "zcc_durable.pt" if durable else "zcc_snapshot.pt"
        return os.path.join(root, f"step_{step:07d}", f"rank_{rank:05d}", name)

    def _cleanup_old_snapshots(self, current_step: int) -> None:
        retain = self.config.retain_latest
        if retain <= 0:
            return
        if _local_rank() != 0:
            return
        roots = [self.config.flash_device, *self.config.flash_stripe]
        if self.config.durable_dir:
            roots.append(self.config.durable_dir)
        for root in dict.fromkeys(roots):
            self._cleanup_root(root, current_step, retain)

    @staticmethod
    def _cleanup_root(root: str, current_step: int, retain: int) -> None:
        if not root or not os.path.isdir(root):
            return
        step_dirs: list[tuple[int, str]] = []
        for name in os.listdir(root):
            if not name.startswith("step_"):
                continue
            try:
                step = int(name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            step_dirs.append((step, os.path.join(root, name)))
        keep_steps = {
            step for step, _ in sorted(step_dirs, key=lambda item: item[0])[-retain:]
        }
        keep_steps.add(current_step)
        for step, path in step_dirs:
            if step in keep_steps:
                continue
            shutil.rmtree(path, ignore_errors=True)


def _local_rank() -> int:
    for name in ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        raw = os.getenv(name)
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                return 0
    return 0


def install_zero_cost_checkpoint(megatron_optimizer: Any, config: Any) -> Any:
    zcc_config = ZeroCostCheckpointConfig.from_optimizer_config(config)
    zcc_config.validate()
    if not zcc_config.enabled:
        return megatron_optimizer
    manager = ZeroCostCheckpointManager(
        megatron_optimizer,
        zcc_config,
    )
    setattr(megatron_optimizer, "zero_cost_checkpoint_manager", manager)
    return megatron_optimizer


def _iter_megatron_optimizers(megatron_optimizer: Any) -> list[Any]:
    if hasattr(megatron_optimizer, "chained_optimizers"):
        result = []
        for child in megatron_optimizer.chained_optimizers:
            result.extend(_iter_megatron_optimizers(child))
        return result
    return [megatron_optimizer]
