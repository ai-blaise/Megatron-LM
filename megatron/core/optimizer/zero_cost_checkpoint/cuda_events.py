# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from collections import deque

import torch


class CudaEventPool:
    def __init__(self, capacity: int = 64):
        self._events = deque()
        if torch.cuda.is_available():
            for _ in range(capacity):
                self._events.append(torch.cuda.Event(enable_timing=False))

    def acquire(self) -> torch.cuda.Event | None:
        if not torch.cuda.is_available():
            return None
        if self._events:
            return self._events.popleft()
        return torch.cuda.Event(enable_timing=False)

    def release(self, event: torch.cuda.Event | None) -> None:
        if event is not None:
            self._events.append(event)
