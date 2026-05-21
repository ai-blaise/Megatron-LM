# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Temporary forward activation offload helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

import torch


_SUPPRESSED_TEMP_OFFLOAD_MODULES: ContextVar[frozenset[str]] = ContextVar(
    "_SUPPRESSED_TEMP_OFFLOAD_MODULES", default=frozenset()
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "off", "no")


def _enabled_for_name(name: str) -> bool:
    if not _env_flag("MEGATRON_TEMP_ACTIVATION_OFFLOAD", default=True):
        return False
    if name in _SUPPRESSED_TEMP_OFFLOAD_MODULES.get():
        return False
    modules = os.getenv("MEGATRON_TEMP_ACTIVATION_OFFLOAD_MODULES", "")
    return name in set(modules.split())


@contextmanager
def suppress_temp_activation_offload(names: Optional[set[str]] = None):
    """Temporarily bypass selected temp activation offload sites.

    StreamBP replay builds short-lived inner autograd graphs. The normal temp
    offload path is useful for outer forward residency, but inside replay it can
    turn each MoE chunk into CPU copy + reload + allocator trim work.
    """

    if not names:
        yield
        return
    current = _SUPPRESSED_TEMP_OFFLOAD_MODULES.get()
    token = _SUPPRESSED_TEMP_OFFLOAD_MODULES.set(frozenset(set(current).union(names)))
    try:
        yield
    finally:
        _SUPPRESSED_TEMP_OFFLOAD_MODULES.reset(token)


def maybe_temp_cpu_offload(
    tensor: Optional[torch.Tensor],
    name: str,
    *,
    enabled: bool = True,
    training: bool = True,
) -> Optional[torch.Tensor]:
    """Move a large forward-live tensor to CPU until it is needed again."""

    if tensor is None or not enabled or not training or not _enabled_for_name(name):
        return tensor
    if not torch.is_tensor(tensor) or not tensor.is_cuda:
        return tensor
    if torch.cuda.is_current_stream_capturing():
        return tensor
    min_mb = int(os.getenv("MEGATRON_TEMP_ACTIVATION_OFFLOAD_MIN_MB", "16"))
    if tensor.numel() * tensor.element_size() < min_mb * 1024 * 1024:
        return tensor

    non_blocking = _env_flag("MEGATRON_TEMP_ACTIVATION_OFFLOAD_NON_BLOCKING", default=False)
    offloaded = tensor.to("cpu", non_blocking=non_blocking)
    if _env_flag("MEGATRON_TEMP_ACTIVATION_OFFLOAD_EMPTY_CACHE", default=True):
        torch.cuda.empty_cache()
    return offloaded


def maybe_temp_cpu_reload(
    tensor: Optional[torch.Tensor],
    device: torch.device,
    name: str,
) -> Optional[torch.Tensor]:
    """Reload a tensor previously moved by ``maybe_temp_cpu_offload``."""

    if tensor is None or not torch.is_tensor(tensor) or tensor.device.type != "cpu":
        return tensor
    non_blocking = _env_flag("MEGATRON_TEMP_ACTIVATION_OFFLOAD_NON_BLOCKING", default=False)
    return tensor.to(device, non_blocking=non_blocking)
