# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in tensor and CUDA-memory audit helpers for large training-path peaks."""

from __future__ import annotations

import os
from typing import Any

import torch

_TENSOR_AUDIT_COUNTS: dict[str, int] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _current_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.getenv("RANK", "0"))


def _rank_enabled(rank: int) -> bool:
    spec = os.getenv("MEGATRON_TENSOR_AUDIT_RANKS", "all").strip()
    if spec in ("", "all", "*"):
        return True
    ranks = {item.strip() for item in spec.replace(",", " ").split() if item.strip()}
    return str(rank) in ranks


def enabled(tag: str) -> bool:
    if not _env_flag("MEGATRON_TENSOR_AUDIT", default=False):
        return False
    rank = _current_rank()
    if not _rank_enabled(rank):
        return False
    tag_filter = os.getenv("MEGATRON_TENSOR_AUDIT_FILTER", "").strip()
    if tag_filter and tag_filter not in tag:
        return False
    limit = _env_int("MEGATRON_TENSOR_AUDIT_LIMIT", 256)
    count_key = f"{rank}:{tag}"
    count = _TENSOR_AUDIT_COUNTS.get(count_key, 0)
    if limit >= 0 and count >= limit:
        return False
    _TENSOR_AUDIT_COUNTS[count_key] = count + 1
    return True


def _format_tensor(name: str, tensor: torch.Tensor, mib: int) -> str:
    tensor_type = type(tensor).__name__
    try:
        tensor_mib = tensor.numel() * tensor.element_size() / mib
        summary = (
            f"{name}=shape{tuple(tensor.shape)}:{tensor.dtype}:type={tensor_type}:"
            f"{tensor_mib:.1f}MiB:contig={int(tensor.is_contiguous())}"
        )
    except Exception as exc:
        return f"{name}=tensor_meta_error:type={tensor_type}:error={exc}"
    if any(token in tensor_type for token in ("NVFP4Tensor", "Float8Tensor", "MXFP8Tensor")):
        return summary + ":skipped_unsupported_tensor_type=1"
    value_limit = _env_int("MEGATRON_TENSOR_AUDIT_SMALL_TENSOR_VALUES", 32)
    if (
        value_limit > 0
        and tensor.numel() <= value_limit
        and not tensor.is_floating_point()
        and not tensor.is_complex()
    ):
        try:
            values = tensor.detach().cpu().view(-1).tolist()
            summary += f":sum={sum(values)}:max={max(values) if values else 0}:values={values}"
        except Exception as exc:
            summary += f":value_error={exc}"
    return summary


def _format_int_sequence(name: str, values: list[int] | tuple[int, ...]) -> str:
    total = sum(values)
    max_value = max(values) if values else 0
    nonzero = sum(1 for v in values if v)
    return f"{name}=sum{total}:max{max_value}:nonzero{nonzero}:len{len(values)}"


def _format_value(name: str, value: Any, mib: int) -> str:
    if torch.is_tensor(value):
        return _format_tensor(name, value, mib)
    if isinstance(value, (list, tuple)) and all(torch.is_tensor(v) for v in value):
        total_mib = sum(v.numel() * v.element_size() for v in value) / mib
        shapes = ",".join(str(tuple(v.shape)) for v in value[:4])
        suffix = "" if len(value) <= 4 else ",..."
        return f"{name}=tensor_seq{len(value)}:{total_mib:.1f}MiB:[{shapes}{suffix}]"
    if isinstance(value, (list, tuple)) and all(isinstance(v, int) for v in value):
        return _format_int_sequence(name, value)
    return f"{name}={value}"


def tensor_audit(tag: str, **values: Any) -> None:
    """Print one compact tensor/memory audit line when explicitly enabled.

    Controlled by:
      - MEGATRON_TENSOR_AUDIT=1
      - MEGATRON_TENSOR_AUDIT_FILTER=<substring>
      - MEGATRON_TENSOR_AUDIT_RANKS="all" or comma/space-separated global ranks
      - MEGATRON_TENSOR_AUDIT_LIMIT=<per-rank, per-tag max lines>
      - MEGATRON_TENSOR_AUDIT_SYNC=1 to synchronize before measuring memory
    """
    if not enabled(tag):
        return
    if _env_flag("MEGATRON_TENSOR_AUDIT_SYNC", default=False) and torch.cuda.is_available():
        torch.cuda.synchronize()

    rank = _current_rank()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    mib = 1024 * 1024
    parts = [
        f"[tensor_audit][rank{rank}/local{local_rank}] {tag}",
        f"grad={int(torch.is_grad_enabled())}",
    ]
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        parts.extend(
            [
                f"alloc={torch.cuda.memory_allocated() / mib:.1f}MiB",
                f"reserved={torch.cuda.memory_reserved() / mib:.1f}MiB",
                f"max_alloc={torch.cuda.max_memory_allocated() / mib:.1f}MiB",
                f"free={free_bytes / mib:.1f}MiB",
                f"total={total_bytes / mib:.1f}MiB",
            ]
        )
    parts.extend(_format_value(name, value, mib) for name, value in values.items())
    print(" | ".join(parts), flush=True)
