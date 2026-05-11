# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


def _wait(work: Any) -> None:
    wait_blocking = getattr(work, "wait_blocking", None)
    if wait_blocking is not None:
        wait_blocking()
        return
    wait = getattr(work, "wait", None)
    if wait is not None:
        wait()


@dataclass
class NCCLXRecoveryCoordinator:
    zcc_manager: Any | None = None
    max_step_retries: int = 1
    init_handles: list[Any] = field(default_factory=list)
    _abort_handles: list[Any] = field(default_factory=list, init=False)
    _step_retries: int = field(default=0, init=False)

    def register_comm(self, comm: Any) -> None:
        get_init_handle = getattr(comm, "get_init_handle", None)
        if get_init_handle is not None:
            try:
                self.init_handles.append(get_init_handle())
            except RuntimeError as exc:
                if "getInitHandle not implemented" not in str(exc):
                    raise
        register_abort_hook = getattr(comm, "register_abort_hook", None)
        if register_abort_hook is not None:
            self._abort_handles.append(register_abort_hook(self._on_abort))

    def register_process_group(
        self,
        group: Any | None = None,
        *,
        device_type: str = "cuda",
    ) -> None:
        from megatron.core import torchcomms_adapter

        self.register_comm(torchcomms_adapter.get_torchcomm(group, device_type=device_type))

    def reset_step(self) -> None:
        self._step_retries = 0

    def try_recover(
        self,
        comm: Any,
        uuid: int,
        *,
        restore_fn: Callable[[], None] | None = None,
        timeout: Any | None = None,
        hints: dict[str, str] | None = None,
    ) -> bool:
        if self._step_retries >= self.max_step_retries:
            return False
        if not self.init_handles:
            return False
        reconfigure = getattr(comm, "reconfigure", None)
        if reconfigure is None:
            return False

        self._step_retries += 1
        work = reconfigure(uuid, self.init_handles, timeout, hints)
        _wait(work)
        if restore_fn is not None:
            restore_fn()
        return True

    def _on_abort(self) -> None:
        manager = self.zcc_manager
        if manager is None:
            return
        force_durable_flush = getattr(manager, "force_durable_flush", None)
        if force_durable_flush is not None:
            force_durable_flush()
