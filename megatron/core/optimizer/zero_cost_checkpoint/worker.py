# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

from .persistence import atomic_torch_save


@dataclass(frozen=True)
class DumpTask:
    path: str
    payload: dict[str, Any]
    compression: str = "none"


class AsyncDumpWorker:
    def __init__(self, workers_num: int = 1):
        self._queue: queue.Queue[DumpTask | None] = queue.Queue()
        self._error: BaseException | None = None
        self._closed = False
        self._threads = [
            threading.Thread(target=self._run, daemon=True)
            for _ in range(max(1, workers_num))
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, task: DumpTask) -> None:
        self.check_error()
        if self._closed:
            raise RuntimeError("ZCC dump worker is closed")
        self._queue.put(task)

    def drain(self) -> None:
        self._queue.join()
        self.check_error()

    def close(self) -> None:
        if self._closed:
            self.check_error()
            return
        self._closed = True
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=5.0)
        self.check_error()

    def check_error(self) -> None:
        if self._error is not None:
            error, self._error = self._error, None
            raise error

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                atomic_torch_save(
                    task.path,
                    task.payload,
                    compression=task.compression,
                )
            except BaseException as exc:
                self._error = exc
            finally:
                self._queue.task_done()
