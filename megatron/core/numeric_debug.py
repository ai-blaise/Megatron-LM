# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lightweight numeric diagnostics for large distributed training runs.

This module is intentionally env-gated. It is meant for short probes where we
need to identify the first subsystem that creates NaNs/Infs without changing
the training math.
"""

from __future__ import annotations

import math
import os
import re
import threading
from collections import defaultdict
from typing import Any, Iterable

import torch


_CTX = {"iteration": None, "microbatch": None, "phase": None}
_EVENT_COUNTS: dict[tuple[int, int, str, str], int] = defaultdict(int)
_HOOK_LOCK = threading.Lock()


def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    return _flag("MEGATRON_NUMERIC_DEBUG")


def _rank() -> int:
    return int(os.getenv("RANK", "0"))


def _local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "0")))


def _parse_ranks(value: str) -> set[int] | None:
    value = value.strip().lower()
    if not value or value in ("all", "*"):
        return None
    ranks: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ranks.update(range(int(start), int(end) + 1))
        else:
            ranks.add(int(part))
    return ranks


def rank_allowed() -> bool:
    if not enabled():
        return False
    ranks = _parse_ranks(os.getenv("MEGATRON_NUMERIC_DEBUG_RANKS", "all"))
    return ranks is None or _rank() in ranks


def rank_allowed_for(scope: str) -> bool:
    if not enabled():
        return False
    scoped = os.getenv(f"MEGATRON_NUMERIC_DEBUG_{scope.upper()}_RANKS")
    if scoped is None:
        return rank_allowed()
    ranks = _parse_ranks(scoped)
    return ranks is None or _rank() in ranks


def name_matches(name: str, scope: str, default: str = "") -> bool:
    """Return whether a debug name matches the scoped regex filter."""
    pattern = os.getenv(f"MEGATRON_NUMERIC_DEBUG_{scope.upper()}_REGEX", default)
    if not pattern:
        return True
    try:
        return re.search(pattern, name) is not None
    except re.error:
        return pattern in name


def active_for(scope: str | None = None) -> bool:
    scope_name = "" if scope is None else scope.upper()
    scoped_start = (
        os.getenv(f"MEGATRON_NUMERIC_DEBUG_{scope_name}_START_ITER")
        if scope_name
        else None
    )
    start_value = scoped_start or os.getenv("MEGATRON_NUMERIC_DEBUG_START_ITER", "1")
    try:
        start = int(start_value)
    except ValueError:
        start = 1
    return context_iteration(0) >= start


def set_context(
    *, iteration: int | None = None, microbatch: int | None = None, phase: str | None = None
) -> None:
    if iteration is not None:
        _CTX["iteration"] = int(iteration)
    if microbatch is not None:
        _CTX["microbatch"] = int(microbatch)
    if phase is not None:
        _CTX["phase"] = phase


def context_iteration(default: int = 0) -> int:
    iteration = _CTX.get("iteration")
    return default if iteration is None else int(iteration)


def should_log(*, force: bool = False, iteration: int | None = None) -> bool:
    if force:
        return True
    if not rank_allowed():
        return False
    if not active_for():
        return False
    interval = int(os.getenv("MEGATRON_NUMERIC_DEBUG_INTERVAL", "1"))
    if interval <= 0:
        return False
    current = context_iteration(0) if iteration is None else int(iteration)
    return current <= int(os.getenv("MEGATRON_NUMERIC_DEBUG_FIRST_N", "16")) or current % interval == 0


def event_allowed(name: str, *, limit: int | None = None, force: bool = False) -> bool:
    if force:
        return True
    if not rank_allowed():
        return False
    if not active_for():
        return False
    if limit is None:
        limit = int(os.getenv("MEGATRON_NUMERIC_DEBUG_EVENT_LIMIT", "16"))
    if limit < 0:
        return True
    key = (
        _rank(),
        context_iteration(0),
        str(_CTX.get("phase") or ""),
        name,
    )
    _EVENT_COUNTS[key] += 1
    return _EVENT_COUNTS[key] <= limit


def _to_local_tensor(value: Any) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    tensor = value.detach()
    if hasattr(tensor, "to_local"):
        try:
            tensor = tensor.to_local()
        except Exception:
            pass
    return tensor


def _sample_flat(tensor: torch.Tensor, max_elems: int) -> torch.Tensor:
    flat = tensor.reshape(-1)
    if flat.numel() <= max_elems:
        return flat
    stride = max(1, math.ceil(flat.numel() / max_elems))
    return flat[::stride][:max_elems]


def tensor_stats(
    value: Any,
    *,
    max_elems: int | None = None,
    full_finite: bool | None = None,
) -> dict[str, Any] | None:
    tensor = _to_local_tensor(value)
    if tensor is None:
        return None
    if max_elems is None:
        max_elems = int(os.getenv("MEGATRON_NUMERIC_DEBUG_MAX_ELEMS", "262144"))
    if full_finite is None:
        full_finite = _flag("MEGATRON_NUMERIC_DEBUG_FULL_FINITE")

    stats: dict[str, Any] = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        stats.update({"empty": True, "all_finite": True})
        return stats

    try:
        if full_finite and tensor.is_floating_point():
            finite_mask = torch.isfinite(tensor)
            all_finite = bool(finite_mask.all().item())
            stats["all_finite"] = all_finite
            if not all_finite:
                stats["nan_count"] = int(torch.isnan(tensor).sum().item())
                stats["inf_count"] = int(torch.isinf(tensor).sum().item())
        elif tensor.is_floating_point():
            sample_for_finite = _sample_flat(tensor, max_elems)
            finite_mask = torch.isfinite(sample_for_finite)
            stats["sample_all_finite"] = bool(finite_mask.all().item())
            stats["sample_nan_count"] = int(torch.isnan(sample_for_finite).sum().item())
            stats["sample_inf_count"] = int(torch.isinf(sample_for_finite).sum().item())
        else:
            stats["all_finite"] = True
    except Exception as exc:
        stats["finite_check_error"] = str(exc)

    try:
        sample = _sample_flat(tensor, max_elems)
    except Exception as exc:
        stats["sample_error"] = str(exc)
        return stats
    if sample.dtype == torch.bool:
        sample_float = sample.to(torch.float32)
    elif sample.is_floating_point():
        sample_float = sample.to(torch.float32)
    else:
        sample_float = sample.to(torch.float32)

    if tensor.is_floating_point():
        finite_sample = sample_float[torch.isfinite(sample_float)]
    else:
        finite_sample = sample_float

    stats["sample_numel"] = int(sample.numel())
    if finite_sample.numel() == 0:
        stats["sample_has_no_finite"] = True
        return stats

    abs_sample = finite_sample.abs()
    stats.update(
        {
            "min": float(finite_sample.min().item()),
            "max": float(finite_sample.max().item()),
            "mean": float(finite_sample.mean().item()),
            "rms": float(torch.sqrt(torch.mean(finite_sample * finite_sample)).item()),
            "absmax": float(abs_sample.max().item()),
        }
    )
    if finite_sample.numel() >= 2:
        q = torch.quantile(abs_sample, torch.tensor([0.5, 0.99, 0.999], device=abs_sample.device))
        stats["abs_p50"] = float(q[0].item())
        stats["abs_p99"] = float(q[1].item())
        stats["abs_p999"] = float(q[2].item())
    return stats


def _is_nonfinite_stats(stats: dict[str, Any] | None) -> bool:
    if stats is None:
        return False
    if stats.get("all_finite") is False:
        return True
    if stats.get("sample_all_finite") is False:
        return True
    return int(stats.get("nan_count", 0)) > 0 or int(stats.get("inf_count", 0)) > 0


def _format_stats(stats: dict[str, Any]) -> str:
    ordered = [
        "shape",
        "dtype",
        "numel",
        "sample_numel",
        "all_finite",
        "sample_all_finite",
        "nan_count",
        "inf_count",
        "sample_nan_count",
        "sample_inf_count",
        "min",
        "max",
        "mean",
        "rms",
        "absmax",
        "abs_p50",
        "abs_p99",
        "abs_p999",
    ]
    parts = []
    for key in ordered:
        if key not in stats:
            continue
        value = stats[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.6e}")
        else:
            parts.append(f"{key}={value}")
    for key, value in stats.items():
        if key not in ordered:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def log_line(name: str, message: str, *, force: bool = False, event: str | None = None) -> None:
    if not rank_allowed():
        return
    if not force and event is not None and not event_allowed(event):
        return
    if not force and event is None and not should_log():
        return
    print(
        f"[numeric-debug][rank={_rank()} local={_local_rank()} "
        f"iter={context_iteration(0)} phase={_CTX.get('phase')}] {name}: {message}",
        flush=True,
    )


def log_tensor(
    name: str,
    value: Any,
    *,
    force: bool = False,
    periodic: bool = True,
    event: str | None = None,
    max_elems: int | None = None,
    full_finite: bool | None = None,
) -> bool:
    if not rank_allowed():
        return False
    if not force and not active_for():
        return False
    stats = tensor_stats(value, max_elems=max_elems, full_finite=full_finite)
    is_bad = _is_nonfinite_stats(stats)
    if stats is None:
        return False
    if force or is_bad or (periodic and should_log()):
        if event is None or event_allowed(event, force=force or is_bad):
            log_line(name, _format_stats(stats), force=True)
    return is_bad


def log_tensors(
    prefix: str,
    value: Any,
    *,
    force: bool = False,
    periodic: bool = True,
    max_items: int = 8,
) -> bool:
    is_bad = False
    if torch.is_tensor(value):
        return log_tensor(
            prefix,
            value,
            force=force,
            periodic=periodic,
            event=f"{prefix}.tensor",
        )
    if isinstance(value, dict):
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                break
            is_bad = log_tensors(
                f"{prefix}.{key}", item, force=force, periodic=periodic
            ) or is_bad
        return is_bad
    if isinstance(value, (tuple, list)):
        for idx, item in enumerate(value[:max_items]):
            is_bad = log_tensors(
                f"{prefix}.{idx}", item, force=force, periodic=periodic
            ) or is_bad
        return is_bad
    return False


def log_router_stats(
    *,
    layer_number: int | None,
    logits: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    expert_bias: torch.Tensor | None,
    padding_mask: torch.Tensor | None = None,
) -> None:
    if not _flag("MEGATRON_NUMERIC_DEBUG_ROUTER") or not rank_allowed():
        return
    if not rank_allowed_for("ROUTER") or not active_for("ROUTER"):
        return
    event = f"router.layer{layer_number}"
    if not event_allowed(event, limit=int(os.getenv("MEGATRON_NUMERIC_DEBUG_ROUTER_LIMIT", "64"))):
        return
    force = _flag("MEGATRON_NUMERIC_DEBUG_ROUTER_FORCE")
    log_tensor(f"{event}.logits", logits, force=force, event=f"{event}.logits")
    log_tensor(f"{event}.probs", probs, force=force, event=f"{event}.probs")
    log_tensor(f"{event}.routing_map", routing_map, force=force, event=f"{event}.routing_map")
    if expert_bias is not None:
        log_tensor(f"{event}.expert_bias", expert_bias, force=force, event=f"{event}.expert_bias")

    with torch.no_grad():
        route = routing_map
        if padding_mask is not None:
            pad = padding_mask.reshape(-1)
            if pad.numel() == route.shape[0]:
                route = route & (~pad).unsqueeze(-1)
        tokens = route.to(torch.float32).sum(dim=0)
        mean = tokens.mean()
        std = tokens.std(unbiased=False)
        max_tokens = tokens.max()
        min_tokens = tokens.min()
        zeros = int((tokens == 0).sum().item())
        log_line(
            event,
            "tokens_per_expert "
            f"sum={float(tokens.sum().item()):.3f} "
            f"min={float(min_tokens.item()):.3f} "
            f"max={float(max_tokens.item()):.3f} "
            f"mean={float(mean.item()):.3f} "
            f"std={float(std.item()):.3f} "
            f"max_over_mean={float((max_tokens / mean.clamp_min(1.0)).item()):.6f} "
            f"zeros={zeros}",
            force=True,
        )


def log_param_stats(model_chunks: Iterable[torch.nn.Module], *, phase: str) -> None:
    if (
        not _flag("MEGATRON_NUMERIC_DEBUG_PARAM_STATS")
        or not rank_allowed_for("PARAM")
        or not active_for("PARAM")
    ):
        return
    if not should_log():
        return
    topk = int(os.getenv("MEGATRON_NUMERIC_DEBUG_TOPK", "16"))
    seen: set[int] = set()
    rows: list[tuple[float, str, str]] = []
    bad_rows: list[str] = []

    for chunk_idx, model_chunk in enumerate(model_chunks):
        for name, param in model_chunk.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            prefix = f"chunk{chunk_idx}.{name}"
            for tensor_name, tensor in (
                ("param", param),
                ("grad", param.grad),
                ("main_grad", getattr(param, "main_grad", None)),
            ):
                if tensor is None:
                    continue
                stats = tensor_stats(tensor, full_finite=False)
                if stats is None:
                    continue
                score = float(stats.get("absmax", 0.0))
                rows.append((score, f"{prefix}.{tensor_name}", _format_stats(stats)))
                if _is_nonfinite_stats(stats):
                    bad_rows.append(f"{prefix}.{tensor_name} {_format_stats(stats)}")

    log_line("param_stats", f"{phase} scanned={len(rows)} bad={len(bad_rows)}", force=True)
    for row in bad_rows[:topk]:
        log_line("param_stats.bad", row, force=True)
    for _, name, stats in sorted(rows, key=lambda item: item[0], reverse=True)[:topk]:
        log_line("param_stats.top", f"{phase} {name} {stats}", force=True)


def attach_param_names(model_chunks: Iterable[torch.nn.Module]) -> None:
    """Attach debug-only names to parameter objects for optimizer-side logs."""
    if not (enabled() or _flag("MEGATRON_GRAD_OWNERSHIP")):
        return
    seen: set[int] = set()
    for chunk_idx, model_chunk in enumerate(model_chunks):
        for name, param in model_chunk.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            setattr(param, "_numeric_debug_name", f"chunk{chunk_idx}.{name}")


def register_grad_hooks(model_chunks: Iterable[torch.nn.Module]) -> None:
    if (
        not _flag("MEGATRON_NUMERIC_DEBUG_GRAD_HOOKS")
        or not rank_allowed_for("GRAD")
        or not active_for("GRAD")
    ):
        return
    default_pattern = r"shared_experts\.linear_fc2\.weight"
    abort_on_bad = _flag(
        "MEGATRON_NUMERIC_DEBUG_GRAD_ABORT",
        _flag("MEGATRON_NUMERIC_DEBUG_ABORT_ON_NONFINITE", True),
    )
    registered = 0

    with _HOOK_LOCK:
        for chunk_idx, model_chunk in enumerate(model_chunks):
            handles = getattr(model_chunk, "_numeric_debug_grad_hook_handles", None)
            if handles is not None:
                continue
            handles = []
            for name, param in model_chunk.named_parameters():
                qualified = f"chunk{chunk_idx}.{name}"
                if not name_matches(qualified, "GRAD", default_pattern):
                    continue
                if not param.requires_grad:
                    continue

                def _hook(grad, *, qualified=qualified):
                    bad = log_tensor(
                        f"grad_hook.{qualified}",
                        grad,
                        force=True,
                        periodic=False,
                        full_finite=True,
                    )
                    if bad and abort_on_bad:
                        raise RuntimeError(
                            f"numeric debug found nonfinite autograd grad for {qualified}"
                        )
                    return grad

                handles.append(param.register_hook(_hook))
                registered += 1
            setattr(model_chunk, "_numeric_debug_grad_hook_handles", handles)

    log_line("grad_hooks", f"registered {registered} parameter grad hooks", force=True)


def register_forward_hooks(model_chunks: Iterable[torch.nn.Module]) -> None:
    if (
        not _flag("MEGATRON_NUMERIC_DEBUG_FORWARD_HOOKS")
        or not rank_allowed_for("HOOK")
        or not active_for("HOOK")
    ):
        return
    class_filter = tuple(
        item.strip()
        for item in os.getenv(
            "MEGATRON_NUMERIC_DEBUG_HOOK_CLASSES",
            "TransformerLayer,MoELayer,TopKRouter,DSAttention,DSAIndexer,SelfAttention,MLP",
        ).split(",")
        if item.strip()
    )
    abort_on_bad = _flag("MEGATRON_NUMERIC_DEBUG_ABORT_ON_NONFINITE", True)
    log_inputs = _flag("MEGATRON_NUMERIC_DEBUG_HOOK_INPUTS")
    verbose = _flag("MEGATRON_NUMERIC_DEBUG_VERBOSE_HOOKS")

    with _HOOK_LOCK:
        for chunk_idx, model_chunk in enumerate(model_chunks):
            if getattr(model_chunk, "_numeric_debug_hooks_registered", False):
                continue
            handles = []
            for module_name, module in model_chunk.named_modules():
                class_name = module.__class__.__name__
                if class_filter and not any(item in class_name for item in class_filter):
                    continue
                qualified = f"chunk{chunk_idx}.{module_name or '<root>'}.{class_name}"

                def _hook(mod, inputs, output, *, qualified=qualified):
                    force = verbose and should_log()
                    bad = False
                    if log_inputs:
                        bad = log_tensors(
                            f"hook.{qualified}.input",
                            inputs,
                            force=force,
                            periodic=verbose,
                        ) or bad
                    bad = log_tensors(
                        f"hook.{qualified}.output",
                        output,
                        force=force,
                        periodic=verbose,
                    ) or bad
                    if bad and abort_on_bad:
                        raise RuntimeError(
                            f"numeric debug found nonfinite tensor in forward hook {qualified}"
                        )

                handles.append(module.register_forward_hook(_hook, always_call=False))
            setattr(model_chunk, "_numeric_debug_hooks_registered", True)
            setattr(model_chunk, "_numeric_debug_hook_handles", handles)
            log_line(
                "hooks",
                f"registered {len(handles)} forward hooks on chunk {chunk_idx}",
                force=True,
            )
