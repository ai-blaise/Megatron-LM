# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from megatron.core.utils import to_local_if_dtensor


@dataclass(frozen=True)
class FusedTensorView:
    owner: Any
    attr: str
    offset: int
    nbytes: int
    shape: torch.Size
    dtype: torch.dtype


class FusedOptimizerStateBuffer:
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.buffer: torch.Tensor | None = None
        self.views: list[FusedTensorView] = []

    @staticmethod
    def _align(offset: int, itemsize: int) -> int:
        if itemsize <= 1:
            return offset
        remainder = offset % itemsize
        return offset if remainder == 0 else offset + itemsize - remainder

    @staticmethod
    def _maybe_quantized_tensors(state: dict[str, Any]) -> list[tuple[Any, str, torch.Tensor]]:
        tensors = []
        for value in state.values():
            if not all(hasattr(value, attr) for attr in ("is_quantized", "numel")):
                continue
            for attr in ("_quantized", "_scales", "_data"):
                tensor = getattr(value, attr, None)
                if isinstance(tensor, torch.Tensor):
                    tensors.append((value, attr, tensor))
        return tensors

    def fuse(self) -> torch.Tensor | None:
        entries: list[tuple[Any, str, torch.Tensor, int]] = []
        total = 0
        device = None
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                for owner, attr, tensor in self._maybe_quantized_tensors(
                    self.optimizer.state.get(param, {})
                ):
                    if tensor.numel() == 0:
                        continue
                    if device is None:
                        device = tensor.device
                    if tensor.device != device:
                        continue
                    total = self._align(total, tensor.element_size())
                    entries.append((owner, attr, tensor, total))
                    total += tensor.numel() * tensor.element_size()

        if not entries:
            self.buffer = None
            self.views = []
            return None

        buffer = torch.empty(total, dtype=torch.uint8, device=device)
        views: list[FusedTensorView] = []
        for owner, attr, tensor, offset in entries:
            nbytes = tensor.numel() * tensor.element_size()
            view = buffer[offset : offset + nbytes].view(tensor.dtype).view(tensor.shape)
            view.copy_(tensor)
            setattr(owner, attr, view)
            views.append(
                FusedTensorView(
                    owner=owner,
                    attr=attr,
                    offset=offset,
                    nbytes=nbytes,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                )
            )

        self.buffer = buffer
        self.views = views
        setattr(self.optimizer, "_zcc_fused_state_buffer", buffer)
        setattr(self.optimizer, "_zcc_fused_state_views", views)
        return buffer


class PinnedMirror:
    def __init__(self, source: torch.Tensor):
        source = to_local_if_dtensor(source).detach()
        pin = source.is_cuda and torch.cuda.is_available()
        try:
            self.tensor = torch.empty(
                source.numel(),
                dtype=source.dtype,
                device="cpu",
                pin_memory=pin,
            )
        except RuntimeError:
            self.tensor = torch.empty(source.numel(), dtype=source.dtype, device="cpu")

    def copy_from(self, source: torch.Tensor, stream: torch.cuda.Stream | None = None) -> None:
        source = to_local_if_dtensor(source).detach()
        source_flat = _flatten_for_copy(source)
        if stream is not None and source.is_cuda:
            with torch.cuda.stream(stream):
                self.tensor.copy_(source_flat, non_blocking=True)
            return
        self.tensor.copy_(source_flat.cpu(), non_blocking=False)


def _flatten_for_copy(source: torch.Tensor) -> torch.Tensor:
    try:
        return source.view(-1)
    except RuntimeError:
        return source.reshape(-1)
