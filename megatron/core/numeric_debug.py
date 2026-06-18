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
import inspect
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


def _env_rank_allowed(name: str, default: str = "all") -> bool:
    ranks = _parse_ranks(os.getenv(name, default))
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


def _debug_tensor_type_name(tensor: torch.Tensor) -> str:
    return type(tensor).__name__


def _is_debug_unsupported_tensor(tensor: torch.Tensor) -> bool:
    type_name = _debug_tensor_type_name(tensor)
    # Transformer Engine quantized wrapper tensors do not expose the full
    # reshape/cast semantics used by the generic debug sampler. Sampling them
    # through torch ops can corrupt the CUDA context and hide the real failure.
    return any(token in type_name for token in ("NVFP4Tensor", "Float8Tensor", "MXFP8Tensor"))


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
        "tensor_type": _debug_tensor_type_name(tensor),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }
    if _is_debug_unsupported_tensor(tensor):
        stats["skipped_unsupported_tensor_type"] = True
        return stats
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
    try:
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
    except Exception as exc:
        stats["stats_error"] = str(exc)
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
        "tensor_type",
        "device",
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
        "skipped_unsupported_tensor_type",
        "finite_check_error",
        "sample_error",
        "stats_error",
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
    try:
        stats = tensor_stats(value, max_elems=max_elems, full_finite=full_finite)
    except Exception as exc:
        if force or event_allowed(f"{name}.tensor_stats_exception", limit=64):
            log_line(name, f"tensor_stats_exception={exc}", force=True)
        return False
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


def grad_provenance_enabled() -> bool:
    return enabled() and (
        _flag("MEGATRON_GRAD_PROVENANCE")
        or _flag("MEGATRON_NUMERIC_DEBUG_GRAD_PROVENANCE")
    )


def _grad_provenance_rank_allowed() -> bool:
    if not grad_provenance_enabled():
        return False
    scoped = os.getenv("MEGATRON_NUMERIC_DEBUG_GRAD_PROVENANCE_RANKS")
    if scoped is not None:
        return _env_rank_allowed("MEGATRON_NUMERIC_DEBUG_GRAD_PROVENANCE_RANKS")
    return _env_rank_allowed("MEGATRON_GRAD_PROVENANCE_RANKS", "0")


def _grad_provenance_event_allowed(name: str) -> bool:
    if not _grad_provenance_rank_allowed() or not active_for("GRAD_PROVENANCE"):
        return False
    if not name_matches(name, "GRAD_PROVENANCE", ""):
        return False
    limit = int(os.getenv("MEGATRON_GRAD_PROVENANCE_EVENT_LIMIT", "4096"))
    return event_allowed(f"grad_provenance.{name}", limit=limit)


def _callsite_source(stack_depth: int = 2) -> str:
    frame = inspect.currentframe()
    try:
        for _ in range(stack_depth):
            if frame is None:
                return "<unknown>"
            frame = frame.f_back
        if frame is None:
            return "<unknown>"
        return f"{frame.f_code.co_filename}:{frame.f_lineno}"
    finally:
        del frame


def _callable_source(fn: Any) -> str:
    try:
        path = inspect.getsourcefile(fn) or inspect.getfile(fn)
        _, line = inspect.getsourcelines(fn)
        return f"{path}:{line}"
    except Exception:
        return "<unknown>"


def _iter_named_tensors(value: Any, prefix: str = "", max_items: int = 16):
    if torch.is_tensor(value):
        yield prefix or "tensor", value
        return
    if isinstance(value, dict):
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                break
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_named_tensors(item, next_prefix, max_items=max_items)
        return
    if isinstance(value, (tuple, list)):
        for idx, item in enumerate(value[:max_items]):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _iter_named_tensors(item, next_prefix, max_items=max_items)


def log_grad_function(
    name: str,
    *,
    source: str | None = None,
    tensors: dict[str, Any] | None = None,
    message: str = "",
) -> None:
    """Log an exact custom-autograd/function boundary with gradient tensor stats."""

    if not _grad_provenance_event_allowed(name):
        return
    src = source or _callsite_source(stack_depth=2)
    suffix = f" {message}" if message else ""
    log_line("grad_function", f"function={name} source={src}{suffix}", force=True)
    max_items = int(os.getenv("MEGATRON_GRAD_PROVENANCE_MAX_TENSORS", "16"))
    full_finite = _flag("MEGATRON_GRAD_PROVENANCE_FULL_FINITE", False)
    max_elems = int(os.getenv("MEGATRON_GRAD_PROVENANCE_MAX_ELEMS", "65536"))
    for label, tensor in (tensors or {}).items():
        for tensor_label, item in _iter_named_tensors(tensor, label, max_items=max_items):
            log_tensor(
                f"grad_function.{name}.{tensor_label}",
                item,
                force=True,
                periodic=False,
                event=f"grad_function.{name}.{tensor_label}",
                max_elems=max_elems,
                full_finite=full_finite,
            )


def _module_direct_param_names(module: torch.nn.Module, max_params: int) -> str:
    names: list[str] = []
    for idx, (param_name, param) in enumerate(module.named_parameters(recurse=False)):
        if idx >= max_params:
            names.append("...")
            break
        debug_name = getattr(param, "_numeric_debug_name", None)
        names.append(debug_name or param_name)
    return ",".join(names) if names else "<none>"


def register_grad_provenance_hooks(model_chunks: Iterable[torch.nn.Module]) -> None:
    """Attach module and activation-gradient hooks for short forensic runs.

    This is intentionally expensive. It answers "which exact module/function emitted this
    gradient" by combining full backward hooks with tensor-output hooks and source locations.
    """

    if not _grad_provenance_rank_allowed() or not active_for("GRAD_PROVENANCE"):
        return
    class_regex = os.getenv(
        "MEGATRON_GRAD_PROVENANCE_CLASS_REGEX",
        (
            "TransformerLayer|DSAttention|DSAIndexer|SelfAttention|MoELayer|"
            "TopKRouter|GroupedMLP|MLP|Linear|ColumnParallel|RowParallel|"
            "LayerNorm|RMSNorm|Gated"
        ),
    )
    name_regex = os.getenv(
        "MEGATRON_GRAD_PROVENANCE_MODULE_REGEX",
        "dsa|hisa|index|attention|self_attention|mlp|moe|expert|router|gate|embedding|output|norm|linear|decoder|layers",
    )
    max_modules = int(os.getenv("MEGATRON_GRAD_PROVENANCE_MAX_MODULES", "512"))
    max_tensors = int(os.getenv("MEGATRON_GRAD_PROVENANCE_MAX_TENSORS", "16"))
    max_params = int(os.getenv("MEGATRON_GRAD_PROVENANCE_MAX_PARAM_NAMES", "8"))
    log_module_backward = _flag("MEGATRON_GRAD_PROVENANCE_MODULE_BACKWARD", True)
    log_output_grads = _flag("MEGATRON_GRAD_PROVENANCE_OUTPUT_GRADS", True)
    full_finite = _flag("MEGATRON_GRAD_PROVENANCE_FULL_FINITE", False)
    max_elems = int(os.getenv("MEGATRON_GRAD_PROVENANCE_MAX_ELEMS", "65536"))

    def _regex_match(pattern: str, value: str) -> bool:
        if not pattern:
            return True
        try:
            return re.search(pattern, value, re.IGNORECASE) is not None
        except re.error:
            return pattern.lower() in value.lower()

    registered = 0
    with _HOOK_LOCK:
        for chunk_idx, model_chunk in enumerate(model_chunks):
            if getattr(model_chunk, "_numeric_debug_grad_provenance_registered", False):
                continue
            handles = []
            for module_name, module in model_chunk.named_modules():
                if registered >= max_modules:
                    break
                class_name = module.__class__.__name__
                qualified = f"chunk{chunk_idx}.{module_name or '<root>'}"
                if not _regex_match(class_regex, class_name) and not _regex_match(
                    name_regex, qualified
                ):
                    continue
                source = _callable_source(getattr(module, "forward", None))
                params = _module_direct_param_names(module, max_params)
                label = f"{qualified}.{class_name}"

                if log_output_grads:

                    def _forward_hook(mod, inputs, output, *, label=label, source=source, params=params):
                        if not _grad_provenance_rank_allowed() or not active_for("GRAD_PROVENANCE"):
                            return
                        count = 0
                        for out_name, tensor in _iter_named_tensors(
                            output, "output", max_items=max_tensors
                        ):
                            if count >= max_tensors:
                                break
                            if not torch.is_tensor(tensor) or not tensor.requires_grad:
                                continue
                            count += 1

                            def _tensor_grad_hook(
                                grad,
                                *,
                                label=label,
                                out_name=out_name,
                                source=source,
                                params=params,
                            ):
                                event_name = f"module_tensor.{label}.{out_name}"
                                if not _grad_provenance_event_allowed(event_name):
                                    return grad
                                log_line(
                                    "grad_provenance.tensor",
                                    (
                                        f"module={label} output={out_name} "
                                        f"source={source} direct_params={params}"
                                    ),
                                    force=True,
                                )
                                log_tensor(
                                    f"grad_provenance.tensor.{label}.{out_name}",
                                    grad,
                                    force=True,
                                    periodic=False,
                                    event=f"grad_provenance.tensor.{label}.{out_name}",
                                    max_elems=max_elems,
                                    full_finite=full_finite,
                                )
                                return grad

                            try:
                                tensor.register_hook(_tensor_grad_hook)
                            except RuntimeError:
                                pass

                    handles.append(module.register_forward_hook(_forward_hook, always_call=False))

                if log_module_backward:

                    def _backward_hook(mod, grad_input, grad_output, *, label=label, source=source, params=params):
                        event_name = f"module_backward.{label}"
                        if not _grad_provenance_event_allowed(event_name):
                            return None
                        log_line(
                            "grad_provenance.module",
                            f"module={label} source={source} direct_params={params}",
                            force=True,
                        )
                        for in_name, tensor in _iter_named_tensors(
                            grad_input, "grad_input", max_items=max_tensors
                        ):
                            log_tensor(
                                f"grad_provenance.module.{label}.{in_name}",
                                tensor,
                                force=True,
                                periodic=False,
                                event=f"grad_provenance.module.{label}.{in_name}",
                                max_elems=max_elems,
                                full_finite=full_finite,
                            )
                        for out_name, tensor in _iter_named_tensors(
                            grad_output, "grad_output", max_items=max_tensors
                        ):
                            log_tensor(
                                f"grad_provenance.module.{label}.{out_name}",
                                tensor,
                                force=True,
                                periodic=False,
                                event=f"grad_provenance.module.{label}.{out_name}",
                                max_elems=max_elems,
                                full_finite=full_finite,
                            )
                        return None

                    handles.append(module.register_full_backward_hook(_backward_hook))

                registered += 1
            setattr(model_chunk, "_numeric_debug_grad_provenance_registered", True)
            setattr(model_chunk, "_numeric_debug_grad_provenance_handles", handles)

    log_line("grad_provenance", f"registered {registered} module provenance hooks", force=True)


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
    skipped_quantized = 0

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
                if _is_debug_unsupported_tensor(tensor):
                    # TE NVFP4 tensors do not support generic flatten/sample/cast
                    # semantics. Update-delta diagnostics sample fp32 main shards
                    # and post-cast residuals directly, so avoid corrupting the
                    # CUDA context with a redundant model-param stats read.
                    skipped_quantized += 1
                    continue
                stats = tensor_stats(tensor, full_finite=False)
                if stats is None:
                    continue
                score = float(stats.get("absmax", 0.0))
                rows.append((score, f"{prefix}.{tensor_name}", _format_stats(stats)))
                if _is_nonfinite_stats(stats):
                    bad_rows.append(f"{prefix}.{tensor_name} {_format_stats(stats)}")

    log_line(
        "param_stats",
        f"{phase} scanned={len(rows)} bad={len(bad_rows)} "
        f"skipped_quantized={skipped_quantized}",
        force=True,
    )
    for row in bad_rows[:topk]:
        log_line("param_stats.bad", row, force=True)
    for _, name, stats in sorted(rows, key=lambda item: item[0], reverse=True)[:topk]:
        log_line("param_stats.top", f"{phase} {name} {stats}", force=True)


_GRAD_SUMMARY_CATEGORIES = (
    "embedding",
    "lm_head",
    "router",
    "expert",
    "shared_expert",
    "dsa_hisa_indexer",
    "attention_mla",
    "dense_mlp",
    "gated_norm",
    "norm",
    "other",
)


def _grad_summary_category(name: str) -> str:
    lowered = name.lower()
    if "embedding" in lowered or "word_embeddings" in lowered:
        return "embedding"
    if "output_layer" in lowered or "lm_head" in lowered:
        return "lm_head"
    if "shared_experts" in lowered or "shared_expert" in lowered:
        return "shared_expert"
    if ".experts" in lowered or "experts." in lowered or "grouped" in lowered:
        return "expert"
    if "dsa" in lowered or "hisa" in lowered or "indexer" in lowered or "indexcache" in lowered:
        return "dsa_hisa_indexer"
    if "attention" in lowered or "mla" in lowered or "linear_q" in lowered or "linear_kv" in lowered:
        return "attention_mla"
    if "gated_norm" in lowered or "gatednorm" in lowered or "input_gated_norm" in lowered:
        return "gated_norm"
    if "linear_fc" in lowered or ".mlp." in lowered:
        return "dense_mlp"
    if "router" in lowered:
        return "router"
    if "norm" in lowered:
        return "norm"
    return "other"


def _grad_summary_layer_key(name: str) -> str:
    """Best-effort stable layer key for exact grad provenance logs."""

    lowered = name.lower()
    match = re.search(r"(?:decoder\.)?layers\.(\d+)", lowered)
    if match:
        return f"layer_{int(match.group(1)):03d}"
    match = re.search(r"(?:^|[._])layer[_.]?(\d+)(?:[._]|$)", lowered)
    if match:
        return f"layer_{int(match.group(1)):03d}"
    if "embedding" in lowered or "word_embeddings" in lowered:
        return "embedding"
    if "output_layer" in lowered or "lm_head" in lowered:
        return "lm_head"
    return "global"


def _grad_summary_grad_for_param(param: torch.nn.Parameter, kind: str) -> torch.Tensor | None:
    if kind == "main_grad":
        return getattr(param, "main_grad", None)
    if kind == "grad":
        return getattr(param, "grad", None)
    raise ValueError(f"unsupported grad summary kind: {kind}")


def log_grad_summary(model_chunks: Iterable[torch.nn.Module], *, phase: str) -> None:
    """Print exact local/global gradient summaries grouped by subsystem.

    This intentionally scans full local grad tensors when enabled. It is for short
    diagnostic runs where attribution matters more than throughput.
    """

    if not _flag("MEGATRON_NUMERIC_DEBUG_GRAD_SUMMARY"):
        return
    if not active_for("GRAD_SUMMARY"):
        return

    kinds = [
        item.strip()
        for item in os.getenv("MEGATRON_NUMERIC_DEBUG_GRAD_SUMMARY_KINDS", "main_grad,grad").split(",")
        if item.strip()
    ]
    topk = int(os.getenv("MEGATRON_NUMERIC_DEBUG_GRAD_SUMMARY_TOPK", "32"))
    top_layers = int(os.getenv("MEGATRON_NUMERIC_DEBUG_GRAD_SUMMARY_TOP_LAYERS", "32"))
    top_sort = os.getenv("MEGATRON_NUMERIC_DEBUG_GRAD_SUMMARY_TOP_SORT", "norm").strip().lower()
    if top_sort not in ("norm", "absmax", "rms"):
        top_sort = "norm"
    full_finite = _flag("MEGATRON_NUMERIC_DEBUG_GRAD_SUMMARY_FULL_FINITE", True)
    print_local = _flag("MEGATRON_NUMERIC_DEBUG_GRAD_SUMMARY_LOCAL", True)
    print_global = _flag("MEGATRON_NUMERIC_DEBUG_GRAD_SUMMARY_GLOBAL", True)
    should_print = rank_allowed_for("GRAD_SUMMARY")

    for kind in kinds:
        cat_stats = {
            cat: {
                "params": 0,
                "trainable": 0,
                "grad_present": 0,
                "grad_none": 0,
                "numel": 0,
                "norm2": 0.0,
                "absmax": 0.0,
                "nonfinite": 0,
                "zero": 0,
                "errors": 0,
            }
            for cat in _GRAD_SUMMARY_CATEGORIES
        }
        layer_stats: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
            lambda: {
                "params": 0,
                "trainable": 0,
                "grad_present": 0,
                "grad_none": 0,
                "numel": 0,
                "norm2": 0.0,
                "absmax": 0.0,
                "nonfinite": 0,
                "zero": 0,
                "errors": 0,
            }
        )
        top_rows: list[tuple[float, float, float, str, str, str, int, int, str]] = []

        seen: set[int] = set()
        for chunk_idx, model_chunk in enumerate(model_chunks):
            for name, param in model_chunk.named_parameters():
                if id(param) in seen:
                    continue
                seen.add(id(param))
                qualified = f"chunk{chunk_idx}.{name}"
                cat = _grad_summary_category(qualified)
                layer = _grad_summary_layer_key(qualified)
                stats = cat_stats[cat]
                layer_key = (layer, cat)
                layer_bucket = layer_stats[layer_key]
                stats["params"] += 1
                layer_bucket["params"] += 1
                stats["numel"] += int(param.numel())
                layer_bucket["numel"] += int(param.numel())
                if param.requires_grad:
                    stats["trainable"] += 1
                    layer_bucket["trainable"] += 1
                grad = _grad_summary_grad_for_param(param, kind)
                if grad is None:
                    stats["grad_none"] += 1
                    layer_bucket["grad_none"] += 1
                    continue
                local = _to_local_tensor(grad)
                if local is None:
                    stats["grad_none"] += 1
                    layer_bucket["grad_none"] += 1
                    continue
                stats["grad_present"] += 1
                layer_bucket["grad_present"] += 1
                stats["numel"] += 0
                if local.numel() == 0:
                    continue
                try:
                    work = local.detach()
                    if not work.is_floating_point():
                        work = work.float()
                    else:
                        work = work.float()
                    if full_finite:
                        finite = torch.isfinite(work)
                        nonfinite_count = int((~finite).sum().item())
                        finite_work = torch.where(finite, work, torch.zeros_like(work))
                    else:
                        nonfinite_count = 0
                        finite_work = work
                    abs_work = finite_work.abs()
                    norm2 = float(torch.sum(finite_work * finite_work).item())
                    absmax = float(abs_work.max().item())
                    zero_count = int((finite_work == 0).sum().item())
                    rms = math.sqrt(norm2 / max(int(work.numel()), 1))
                    stats["norm2"] += norm2
                    stats["absmax"] = max(float(stats["absmax"]), absmax)
                    stats["nonfinite"] += nonfinite_count
                    stats["zero"] += zero_count
                    layer_bucket["norm2"] = float(layer_bucket["norm2"]) + norm2
                    layer_bucket["absmax"] = max(float(layer_bucket["absmax"]), absmax)
                    layer_bucket["nonfinite"] = int(layer_bucket["nonfinite"]) + nonfinite_count
                    layer_bucket["zero"] = int(layer_bucket["zero"]) + zero_count
                    if topk > 0:
                        top_rows.append(
                            (
                                norm2,
                                absmax,
                                rms,
                                qualified,
                                cat,
                                layer,
                                int(work.numel()),
                                nonfinite_count,
                                str(local.dtype).replace("torch.", ""),
                            )
                        )
                except Exception as exc:
                    stats["errors"] += 1
                    layer_bucket["errors"] = int(layer_bucket["errors"]) + 1
                    if should_print and event_allowed(f"grad_summary.error.{qualified}", limit=128):
                        log_line(
                            "grad_summary.error",
                            f"phase={phase} kind={kind} category={cat} name={qualified} error={exc}",
                            force=True,
                        )

        if print_global:
            local_rows = []
            for cat in _GRAD_SUMMARY_CATEGORIES:
                stats = cat_stats[cat]
                local_rows.append(
                    [
                        float(stats["params"]),
                        float(stats["trainable"]),
                        float(stats["grad_present"]),
                        float(stats["grad_none"]),
                        float(stats["numel"]),
                        float(stats["norm2"]),
                        float(stats["absmax"]),
                        float(stats["nonfinite"]),
                        float(stats["zero"]),
                        float(stats["errors"]),
                    ]
                )
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            global_sum = torch.tensor(local_rows, dtype=torch.float64, device=device)
            global_max = global_sum.clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX)
            if should_print:
                for idx, cat in enumerate(_GRAD_SUMMARY_CATEGORIES):
                    params, trainable, present, none, numel, norm2, _absmax_sum, nonfinite, zero, errors = [
                        global_sum[idx, j].item() for j in range(10)
                    ]
                    absmax = global_max[idx, 6].item()
                    norm = math.sqrt(max(norm2, 0.0))
                    log_line(
                        "grad_summary.global",
                        (
                            f"phase={phase} kind={kind} category={cat} "
                            f"params={params:.0f} trainable={trainable:.0f} "
                            f"grad_present={present:.0f} grad_none={none:.0f} "
                            f"numel={numel:.0f} norm={norm:.6e} absmax={absmax:.6e} "
                            f"nonfinite={nonfinite:.0f} zero={zero:.0f} errors={errors:.0f}"
                        ),
                        force=True,
                    )

        if should_print and print_local:
            local_total_norm2 = sum(float(stats["norm2"]) for stats in cat_stats.values())
            for cat in _GRAD_SUMMARY_CATEGORIES:
                stats = cat_stats[cat]
                norm = math.sqrt(max(float(stats["norm2"]), 0.0))
                log_line(
                    "grad_summary.local",
                    (
                        f"phase={phase} kind={kind} category={cat} "
                        f"params={stats['params']} trainable={stats['trainable']} "
                        f"grad_present={stats['grad_present']} grad_none={stats['grad_none']} "
                        f"numel={stats['numel']} norm={norm:.6e} "
                        f"absmax={float(stats['absmax']):.6e} nonfinite={stats['nonfinite']} "
                        f"zero={stats['zero']} errors={stats['errors']}"
                    ),
                    force=True,
                )
            if top_layers > 0:
                for (layer, cat), stats in sorted(
                    layer_stats.items(), key=lambda item: float(item[1]["norm2"]), reverse=True
                )[:top_layers]:
                    norm2 = float(stats["norm2"])
                    norm = math.sqrt(max(norm2, 0.0))
                    pct = 100.0 * norm2 / local_total_norm2 if local_total_norm2 > 0 else 0.0
                    log_line(
                        "grad_summary.top_layer",
                        (
                            f"phase={phase} kind={kind} layer={layer} category={cat} "
                            f"params={int(stats['params'])} trainable={int(stats['trainable'])} "
                            f"grad_present={int(stats['grad_present'])} grad_none={int(stats['grad_none'])} "
                            f"numel={int(stats['numel'])} norm={norm:.6e} "
                            f"pct_local_norm2={pct:.4f} absmax={float(stats['absmax']):.6e} "
                            f"nonfinite={int(stats['nonfinite'])} zero={int(stats['zero'])} "
                            f"errors={int(stats['errors'])}"
                        ),
                        force=True,
                    )
            if top_sort == "absmax":
                sort_key = lambda item: item[1]
            elif top_sort == "rms":
                sort_key = lambda item: item[2]
            else:
                sort_key = lambda item: item[0]
            for norm2, absmax, rms, qualified, cat, layer, numel, nonfinite, dtype in sorted(
                top_rows, key=sort_key, reverse=True
            )[:topk]:
                norm = math.sqrt(max(norm2, 0.0))
                pct = 100.0 * norm2 / local_total_norm2 if local_total_norm2 > 0 else 0.0
                log_line(
                    "grad_summary.top_param",
                    (
                        f"phase={phase} kind={kind} layer={layer} category={cat} "
                        f"name={qualified} dtype={dtype} numel={numel} norm={norm:.6e} "
                        f"pct_local_norm2={pct:.4f} absmax={absmax:.6e} "
                        f"rms={rms:.6e} nonfinite={nonfinite}"
                    ),
                    force=True,
                )
                log_line(
                    "grad_summary.top",
                    (
                        f"phase={phase} kind={kind} layer={layer} category={cat} "
                        f"name={qualified} dtype={dtype} numel={numel} norm={norm:.6e} "
                        f"pct_local_norm2={pct:.4f} absmax={absmax:.6e} "
                        f"rms={rms:.6e} nonfinite={nonfinite}"
                    ),
                    force=True,
                )


def attach_param_names(model_chunks: Iterable[torch.nn.Module]) -> None:
    """Attach debug-only names to parameter objects for optimizer-side logs."""
    if not (
        enabled()
        or _flag("MEGATRON_GRAD_OWNERSHIP")
        or _flag("MEGATRON_NONFINITE_GRAD_OWNERSHIP")
        or _flag("MEGATRON_UPDATE_DELTA_DEBUG")
    ):
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
    ):
        return
    default_pattern = r"shared_experts\.linear_fc2\.weight"
    abort_on_bad = _flag(
        "MEGATRON_NUMERIC_DEBUG_GRAD_ABORT",
        _flag("MEGATRON_NUMERIC_DEBUG_ABORT_ON_NONFINITE", True),
    )
    log_all = _flag("MEGATRON_NUMERIC_DEBUG_GRAD_LOG_ALL", True)
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

                def _hook(grad, *, qualified=qualified, param=param):
                    if not active_for("GRAD"):
                        return grad
                    grad_already_added = getattr(param, "grad_added_to_main_grad", False)
                    main_grad = getattr(param, "main_grad", None)
                    debug_value = main_grad if grad_already_added and main_grad is not None else grad
                    debug_suffix = "main_grad" if debug_value is main_grad else "grad"
                    bad = log_tensor(
                        f"grad_hook.{qualified}.{debug_suffix}",
                        debug_value,
                        force=log_all,
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
