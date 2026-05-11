# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import Any

import torch


class PersistentAllGatherWindow:
    def __init__(
        self,
        comm: Any,
        output: torch.Tensor | tuple[int, ...] | torch.Size,
        output_dtype: torch.dtype | None = None,
        *,
        device: torch.device | str | int | None = None,
        hints: dict[str, str] | None = None,
        timeout: Any | None = None,
    ):
        self.comm = comm
        self.hints = hints
        self.timeout = timeout
        if isinstance(output, torch.Tensor):
            if output_dtype is not None and output.dtype != output_dtype:
                raise ValueError("output_dtype must match output.dtype when output is a tensor")
            self.output = output
        else:
            if output_dtype is None:
                raise ValueError("output_dtype is required when output is a shape")
            if device is None:
                device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            self.output = torch.empty(output, dtype=output_dtype, device=device)
        self.handle = comm.all_gather_p_init(self.output, hints, timeout)

    def exec(self, input_tensor: torch.Tensor, *, async_op: bool = True) -> Any:
        return self.comm.all_gather_p_exec(
            self.handle,
            input_tensor,
            async_op,
            self.hints,
            self.timeout,
        )

    def free(self) -> None:
        if self.handle is None:
            return
        self.comm.all_gather_p_free(self.handle)
        self.handle = None

    def __enter__(self) -> "PersistentAllGatherWindow":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.free()
