# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import math
import os
from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.fine_profile import fine_profile_range
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.models.common.embeddings.rope_utils import fused_apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_triton import (
    _bf16_grad_atomics_enabled,
    _dsa_split_qk_backward_row_cuda,
    _hisa_block_reps_batched_cuda,
    _hisa_dsa_split_qk_fused_forward_cuda,
    _hisa_dsa_split_qk_persistent_forward_cuda,
    dsa_indexer_scores_triton,
    hisa_kl_loss_and_grad_triton,
    hisa_attention_target_probs_triton,
    is_hisa_kl_grad_triton_supported,
    is_hisa_attention_target_probs_triton_supported,
    is_dsa_indexer_scores_triton_supported,
    is_sparse_dsa_split_qk_triton_supported,
    is_sparse_dsa_teacher_triton_supported,
    is_sparse_dsa_triton_supported,
    sparse_dsa_attention_split_qk_triton,
    sparse_dsa_attention_split_qk_with_teacher_triton,
    sparse_dsa_attention_triton,
    sparse_dsa_attention_with_teacher_triton,
)
from megatron.core.quantization.indexcache import (
    INDEXCACHE_QUANT_NVFP4,
    IndexCacheHISAConfig,
    describe_indexcache_hisa_select_with_scores,
    indexcache_hisa_megakernel_batched_select_with_scores,
    indexcache_hisa_select_with_scores,
    indexcache_hisa_selector_backend_name,
    indexcache_hisa_topk,
    indexcache_hisa_topk_with_scores,
)
from megatron.core.extensions.hisa_indexer import (
    apply_hisa_score_backward,
    hisa_selector_forward_and_save,
)
from megatron.core.extensions.hisa_indexer.reference import (
    hisa_unselect_grad_from_topk_to_scores,
)

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None

_TE_DEQUANTIZABLE_TENSOR_TYPES = ()
try:
    from transformer_engine.pytorch.tensor import QuantizedTensor as _TEQuantizedTensor

    _TE_DEQUANTIZABLE_TENSOR_TYPES += (_TEQuantizedTensor,)
except (ImportError, ModuleNotFoundError):
    pass

try:
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor as _TENVFP4Tensor

    _TE_DEQUANTIZABLE_TENSOR_TYPES += (_TENVFP4Tensor,)
except (ImportError, ModuleNotFoundError):
    pass

try:
    from transformer_engine.pytorch.float8_tensor import Float8Tensor as _TEFloat8Tensor

    _TE_DEQUANTIZABLE_TENSOR_TYPES += (_TEFloat8Tensor,)
except (ImportError, ModuleNotFoundError):
    pass


_DSA_STREAMING_INDEXER_TOPK_ENV = "MEGATRON_DSA_STREAMING_INDEXER_TOPK"
_DSA_INDEXER_KEY_BLOCK_SIZE_ENV = "MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE"
_DSA_SORT_TOPK_INDICES_ENV = "MEGATRON_DSA_SORT_TOPK_INDICES"
_DSA_COMPACT_TOPK_INDICES_ENV = "MEGATRON_DSA_COMPACT_TOPK_INDICES"
_DSA_STREAM_TRITON_ATTENTION_CHUNKS_ENV = "MEGATRON_DSA_STREAM_TRITON_ATTENTION_CHUNKS"
_DSA_VALIDATE_TOPK_INDICES_ENV = "MEGATRON_DSA_VALIDATE_TOPK_INDICES"
_HISA_TARGET_ROW_CHUNK_ENV = "MEGATRON_HISA_TARGET_ROW_CHUNK"
_HISA_FUSED_INDEXER_LOSS_ENV = "MEGATRON_HISA_FUSED_INDEXER_LOSS"
_HISA_DSA_FUSED_FORWARD_ENV = "MEGATRON_HISA_DSA_FUSED_FORWARD"
_HISA_DSA_PERSISTENT_FORWARD_ENV = "MEGATRON_HISA_DSA_PERSISTENT_FORWARD"
_HISA_ASSUME_SORTED_POSITIONS_ENV = "MEGATRON_HISA_ASSUME_SORTED_POSITIONS"
_HISA_FALLBACK_DENSE_IF_SHORT_ENV = "MEGATRON_HISA_FALLBACK_DENSE_IF_SHORT"
_HISA_SELECTED_SCORE_BWD_BATCHED_CUDA_ENV = "MEGATRON_HISA_SELECTED_SCORE_BWD_BATCHED_CUDA"
_HISA_INDEXER_LOSS_DEBUG_ENV = "MEGATRON_HISA_INDEXER_LOSS_DEBUG"
_HISA_INDEXER_LOSS_DEBUG_RANKS_ENV = "MEGATRON_HISA_INDEXER_LOSS_DEBUG_RANKS"
_HISA_INDEXER_LOSS_DEBUG_EVENTS_ENV = "MEGATRON_HISA_INDEXER_LOSS_DEBUG_EVENTS"
_HISA_INDEXER_LOSS_DEBUG_MAX_LINES_ENV = "MEGATRON_HISA_INDEXER_LOSS_DEBUG_MAX_LINES"
_HISA_INDEXER_LOSS_DEBUG_EXACT_NUMEL_ENV = "MEGATRON_HISA_INDEXER_LOSS_DEBUG_EXACT_NUMEL"
_DSA_CHUNK_INDEXER_PROJ_ENV = "MEGATRON_DSA_CHUNK_INDEXER_PROJ"
_DSA_SP_PROJECT_BEFORE_GATHER_ENV = "MEGATRON_DSA_SP_PROJECT_BEFORE_GATHER"
_DSA_INDEXER_ROPE_FUSION_ENV = "MEGATRON_DSA_INDEXER_ROPE_FUSION"
_DSA_INDEXER_ROPE_INPLACE_ENV = "MEGATRON_DSA_INDEXER_ROPE_INPLACE"
_DSA_INDEXER_TORCH_K_NORM_ENV = "MEGATRON_DSA_INDEXER_TORCH_K_NORM"
_DSA_INDEXER_AUX_LOSS_AUTOSCALE_ENV = "MEGATRON_DSA_INDEXER_AUX_LOSS_AUTOSCALE"
_DSA_CP_GATHER_SPLIT_KV_REF_ENV = "MEGATRON_DSA_CP_GATHER_SPLIT_KV_REF"
_DSA_CP_SORTED_ZIGZAG_GATHER_ENV = "MEGATRON_DSA_CP_SORTED_ZIGZAG_GATHER"
_DSA_SP_Q_PROJ_TRIM_CACHE_ENV = "MEGATRON_DSA_SP_Q_PROJ_TRIM_CACHE"
_DSA_SP_Q_PROJ_TRIM_SAFETY_MB_ENV = "MEGATRON_DSA_SP_Q_PROJ_TRIM_SAFETY_MB"
_DSA_SP_Q_PROJ_TRIM_CACHED_MB_ENV = "MEGATRON_DSA_SP_Q_PROJ_TRIM_CACHED_MB"
_DSA_SP_Q_PROJ_TRIM_SYNC_ENV = "MEGATRON_DSA_SP_Q_PROJ_TRIM_SYNC"
_DSA_DEBUG_SYNC_ENV = "MEGATRON_DSA_DEBUG_SYNC"
_DSA_RUNTIME_AUDIT_ENV = "MEGATRON_DSA_RUNTIME_AUDIT"
_DSA_RUNTIME_AUDIT_RANKS_ENV = "MEGATRON_DSA_RUNTIME_AUDIT_RANKS"
_DSA_INDEXER_K_GRAD_DEBUG_ENV = "MEGATRON_DSA_INDEXER_K_GRAD_DEBUG"
_DSA_INDEXER_K_GRAD_DEBUG_RANKS_ENV = "MEGATRON_DSA_INDEXER_K_GRAD_DEBUG_RANKS"
_DSA_INDEXER_K_GRAD_DEBUG_LAYERS_ENV = "MEGATRON_DSA_INDEXER_K_GRAD_DEBUG_LAYERS"
_DSA_INDEXER_K_GRAD_DEBUG_MAX_LINES_ENV = "MEGATRON_DSA_INDEXER_K_GRAD_DEBUG_MAX_LINES"
_HISA_RUNTIME_LOGGED: set[tuple[int, str, str]] = set()
_DSA_RUNTIME_LOGGED: set[tuple[int, str, str]] = set()
_HISA_INDEXER_LOSS_DEBUG_LINES = 0
_HISA_INDEXER_LOSS_DEBUG_CALLS = 0
_DSA_INDEXER_K_GRAD_DEBUG_LINES = 0


def _env_flag_enabled(name: str, default: str = "1") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _dsa_indexer_aux_loss_autoscale_enabled() -> bool:
    return _env_flag_enabled(_DSA_INDEXER_AUX_LOSS_AUTOSCALE_ENV, "0")


def _dsa_distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _dsa_runtime_audit_enabled(rank: int) -> bool:
    if not _env_flag_enabled(_DSA_RUNTIME_AUDIT_ENV, "0"):
        return False
    raw_ranks = os.getenv(_DSA_RUNTIME_AUDIT_RANKS_ENV, "0").strip().lower()
    if raw_ranks in {"*", "all"}:
        return True
    for item in raw_ranks.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit() and int(item) == rank:
            return True
    return False


def _rank_in_csv_env(rank: int, env_name: str, default: str = "0") -> bool:
    raw_ranks = os.getenv(env_name, default).strip().lower()
    if raw_ranks in {"*", "all"}:
        return True
    for item in raw_ranks.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit() and int(item) == rank:
            return True
    return False


def _int_in_csv_env(value: int, env_name: str, default: str = "*") -> bool:
    raw_values = os.getenv(env_name, default).strip().lower()
    if raw_values in {"*", "all"}:
        return True
    for item in raw_values.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            if start.strip().isdigit() and end.strip().isdigit() and int(start) <= value <= int(end):
                return True
        elif item.isdigit() and int(item) == value:
            return True
    return False


def _hisa_indexer_loss_debug_enabled(rank: int) -> bool:
    return _env_flag_enabled(_HISA_INDEXER_LOSS_DEBUG_ENV, "0") and _rank_in_csv_env(
        rank, _HISA_INDEXER_LOSS_DEBUG_RANKS_ENV, "0,7,88,95"
    )


def _hisa_indexer_loss_debug_stats(label: str, tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return f"{label}=None"
    with torch.no_grad():
        t = tensor.detach()
        if t.numel() == 0:
            return f"{label}: shape={tuple(t.shape)} dtype={t.dtype} numel=0"
        if not t.is_floating_point():
            t_i64 = t.to(torch.int64)
            return (
                f"{label}: shape={tuple(t.shape)} dtype={t.dtype} numel={t.numel()} "
                f"min={int(t_i64.min().item())} max={int(t_i64.max().item())}"
            )
        t_f = t.float()
        finite = torch.isfinite(t_f)
        finite_count = int(finite.sum().item())
        nonfinite_count = int(t_f.numel() - finite_count)
        nan_count = int(torch.isnan(t_f).sum().item())
        inf_count = int(torch.isinf(t_f).sum().item())
        exact_numel = int(os.getenv(_HISA_INDEXER_LOSS_DEBUG_EXACT_NUMEL_ENV, str(16 * 1024 * 1024)))
        if t_f.numel() > exact_numel:
            absmax_raw = float(t_f.abs().max().item())
            return (
                f"{label}: shape={tuple(t.shape)} dtype={t.dtype} numel={t.numel()} "
                f"finite={finite_count} nonfinite={nonfinite_count} nan={nan_count} inf={inf_count} "
                f"absmax_raw={absmax_raw:.6e} stats=skipped_large"
            )
        if finite_count:
            finite_vals = t_f[finite]
            min_val = float(finite_vals.min().item())
            max_val = float(finite_vals.max().item())
            absmax = float(finite_vals.abs().max().item())
            mean_val = float(finite_vals.mean().item())
        else:
            min_val = max_val = absmax = mean_val = float("nan")
        return (
            f"{label}: shape={tuple(t.shape)} dtype={t.dtype} numel={t.numel()} "
            f"finite={finite_count} nonfinite={nonfinite_count} nan={nan_count} inf={inf_count} "
            f"min={min_val:.6e} max={max_val:.6e} absmax={absmax:.6e} mean={mean_val:.6e}"
        )


def _log_hisa_indexer_loss_debug(event: str, call_id: int, **tensors) -> None:
    global _HISA_INDEXER_LOSS_DEBUG_LINES
    rank = _dsa_distributed_rank()
    if not _hisa_indexer_loss_debug_enabled(rank):
        return
    raw_events = os.getenv(_HISA_INDEXER_LOSS_DEBUG_EVENTS_ENV, "*").strip().lower()
    if raw_events not in {"*", "all"}:
        allowed = [item.strip() for item in raw_events.split(",") if item.strip()]
        if allowed and not any(item in event.lower() for item in allowed):
            return
    max_lines = int(os.getenv(_HISA_INDEXER_LOSS_DEBUG_MAX_LINES_ENV, "256"))
    force = False
    for tensor in tensors.values():
        if tensor is None or not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        with torch.no_grad():
            force = force or bool((~torch.isfinite(tensor.detach().float())).any().item())
        if force:
            break
    if _HISA_INDEXER_LOSS_DEBUG_LINES >= max_lines and not force:
        return
    _HISA_INDEXER_LOSS_DEBUG_LINES += 1
    parts = [
        f"[hisa_indexer_loss_debug] rank={rank}",
        f"call={call_id}",
        f"event={event}",
    ]
    for label, tensor in tensors.items():
        parts.append(_hisa_indexer_loss_debug_stats(label, tensor))
    print(" | ".join(parts), flush=True)


def _dsa_register_indexer_k_grad_debug(
    label: str,
    layer_number: Optional[int],
    tensor: torch.Tensor,
) -> None:
    if not _env_flag_enabled(_DSA_INDEXER_K_GRAD_DEBUG_ENV, "0"):
        return
    if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
        return
    rank = _dsa_distributed_rank()
    if not _rank_in_csv_env(rank, _DSA_INDEXER_K_GRAD_DEBUG_RANKS_ENV, "0"):
        return
    layer = int(layer_number or -1)
    if not _int_in_csv_env(layer, _DSA_INDEXER_K_GRAD_DEBUG_LAYERS_ENV, "*"):
        return

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        global _DSA_INDEXER_K_GRAD_DEBUG_LINES
        max_lines = int(os.getenv(_DSA_INDEXER_K_GRAD_DEBUG_MAX_LINES_ENV, "128"))
        force = False
        if isinstance(grad, torch.Tensor) and grad.is_floating_point():
            with torch.no_grad():
                force = bool((~torch.isfinite(grad.detach().float())).any().item())
        if _DSA_INDEXER_K_GRAD_DEBUG_LINES >= max_lines and not force:
            return grad
        _DSA_INDEXER_K_GRAD_DEBUG_LINES += 1
        try:
            from megatron.core import numeric_debug as _numeric_debug

            iteration = _numeric_debug.context_iteration(-1)
        except Exception:
            iteration = -1
        print(
            " | ".join(
                [
                    f"[dsa_indexer_k_grad_debug] rank={rank}",
                    f"iter={iteration}",
                    f"layer={layer}",
                    f"label={label}",
                    _hisa_indexer_loss_debug_stats("grad", grad),
                ]
            ),
            flush=True,
        )
        return grad

    tensor.register_hook(_hook)


def _log_hisa_runtime_once(label: str, message: str, mode: str = "default") -> None:
    rank = _dsa_distributed_rank()
    if not _dsa_runtime_audit_enabled(rank):
        return
    key = (rank, label, mode)
    if key in _HISA_RUNTIME_LOGGED:
        return
    _HISA_RUNTIME_LOGGED.add(key)
    print(f"[hisa_runtime][rank{rank}] layer={label} mode={mode} {message}", flush=True)


def _log_dsa_runtime_once(label: str, message: str, mode: str = "default") -> None:
    rank = _dsa_distributed_rank()
    if not _dsa_runtime_audit_enabled(rank):
        return
    key = (rank, label, mode)
    if key in _DSA_RUNTIME_LOGGED:
        return
    _DSA_RUNTIME_LOGGED.add(key)
    print(f"[dsa_runtime][rank{rank}] layer={label} mode={mode} {message}", flush=True)


def _dsa_debug_sync(label: str, tensor: Optional[torch.Tensor] = None) -> None:
    if not _env_flag_enabled(_DSA_DEBUG_SYNC_ENV, "0"):
        return
    if not torch.cuda.is_available():
        return
    try:
        if tensor is not None and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        else:
            torch.cuda.synchronize()
    except Exception as exc:
        device = None if tensor is None else tensor.device
        shape = None if tensor is None else tuple(tensor.shape)
        dtype = None if tensor is None else tensor.dtype
        raise RuntimeError(
            f"DSA debug sync failed after {label}; device={device}, shape={shape}, dtype={dtype}"
        ) from exc


def _dsa_next_owner_segment_range_end(q_start: int, q_end: int, owner_segment_len: int) -> int:
    """Limit a query range so it does not cross a rank-major SP owner segment."""

    if owner_segment_len <= 0:
        raise ValueError(f"owner_segment_len must be positive, got {owner_segment_len}")
    if q_end <= q_start:
        return q_end
    next_owner_boundary = ((q_start // owner_segment_len) + 1) * owner_segment_len
    return min(q_end, next_owner_boundary)


def _maybe_trim_cuda_cache_for_dsa_sp_q_projection(
    device: torch.device, required_bytes: int, *, force: bool = False
) -> None:
    """Release allocator cache before compact SP owner-projected DSA Q allocation."""

    if not torch.cuda.is_available():
        return
    raw = os.getenv(_DSA_SP_Q_PROJ_TRIM_CACHE_ENV, "1").strip().lower()
    if raw in {"0", "false", "off", "no"}:
        return
    if device.type != "cuda":
        return

    with torch.cuda.device(device):
        free_bytes, _ = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
    cached = max(0, reserved - allocated)

    mib = 1024 * 1024
    safety_mb = int(os.getenv(_DSA_SP_Q_PROJ_TRIM_SAFETY_MB_ENV, "1024"))
    cached_threshold_mb = int(os.getenv(_DSA_SP_Q_PROJ_TRIM_CACHED_MB_ENV, "512"))
    needed_bytes = max(0, required_bytes) + safety_mb * mib
    if (force and cached > 0) or (
        free_bytes < needed_bytes and cached > cached_threshold_mb * mib
    ):
        sync_raw = os.getenv(_DSA_SP_Q_PROJ_TRIM_SYNC_ENV, "1").strip().lower()
        if sync_raw not in {"0", "false", "off", "no"}:
            torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _tensor_safe_for_custom_inplace(x: torch.Tensor) -> bool:
    """Return whether a custom autograd Function can legally mutate ``x``."""
    if x.requires_grad and x.is_leaf:
        return False
    if getattr(x, "_base", None) is not None:
        return False
    is_view = getattr(x, "_is_view", None)
    if callable(is_view) and is_view():
        return False
    return True


def _should_dequantize_tensor(x: torch.Tensor) -> bool:
    """Return whether ``x.dequantize()`` is a real quantized-tensor materialization.

    Plain dense torch tensors expose ``Tensor.dequantize`` too, but calling it on
    a differentiable tensor installs a NotImplemented autograd node. Only
    dequantize actual torch quantized tensors or TE quantized tensor wrappers.
    """

    if bool(getattr(x, "is_quantized", False)):
        return True
    return bool(_TE_DEQUANTIZABLE_TENSOR_TYPES) and isinstance(
        x, _TE_DEQUANTIZABLE_TENSOR_TYPES
    )


def _dequantize_tensor_if_needed(
    x: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    if not _should_dequantize_tensor(x):
        return x
    if dtype is None:
        return x.dequantize()
    try:
        return x.dequantize(dtype=dtype)
    except TypeError:
        return x.dequantize()


class _DSAIndexerRopeFunction(torch.autograd.Function):
    """Apply indexer RoPE to the PE prefix while writing the full head directly."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        pe_dim: int,
        mscale: float,
        interleaved: bool,
    ) -> torch.Tensor:
        ext = _try_load_hisa_cuda_ext()
        if ext is None or not hasattr(ext, "dsa_indexer_rope_fwd"):
            raise RuntimeError("DSA indexer RoPE CUDA extension is unavailable")

        x_contig = x.contiguous()
        aliases_input = x_contig.data_ptr() == x.data_ptr()
        use_inplace = (
            _env_flag_enabled(_DSA_INDEXER_ROPE_INPLACE_ENV)
            and hasattr(ext, "dsa_indexer_rope_fwd_inplace")
            and (not aliases_input or _tensor_safe_for_custom_inplace(x))
        )
        if use_inplace:
            if aliases_input:
                ctx.mark_dirty(x)
            ext.dsa_indexer_rope_fwd_inplace(
                x_contig,
                rotary_pos_emb,
                int(pe_dim),
                float(mscale),
                bool(interleaved),
            )
            out = x_contig
        else:
            out = torch.empty_like(x_contig)
            ext.dsa_indexer_rope_fwd(
                x_contig,
                rotary_pos_emb,
                out,
                int(pe_dim),
                float(mscale),
                bool(interleaved),
            )
        ctx.save_for_backward(rotary_pos_emb)
        ctx.pe_dim = int(pe_dim)
        ctx.mscale = float(mscale)
        ctx.interleaved = bool(interleaved)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (rotary_pos_emb,) = ctx.saved_tensors
        ext = _try_load_hisa_cuda_ext()
        if ext is None or not hasattr(ext, "dsa_indexer_rope_bwd"):
            raise RuntimeError("DSA indexer RoPE CUDA extension is unavailable")

        grad_out_contig = grad_out.contiguous()
        grad_x = torch.empty_like(grad_out_contig)
        ext.dsa_indexer_rope_bwd(
            grad_out_contig,
            rotary_pos_emb,
            grad_x,
            ctx.pe_dim,
            ctx.mscale,
            ctx.interleaved,
        )
        return grad_x, None, None, None, None


class _DSAIndexerFlatRopeInplaceFunction(torch.autograd.Function):
    """Apply indexer RoPE in-place before the projection tensor is viewed as heads."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        heads: int,
        head_dim: int,
        pe_dim: int,
        mscale: float,
        interleaved: bool,
    ) -> torch.Tensor:
        ext = _try_load_hisa_cuda_ext()
        if ext is None or not hasattr(ext, "dsa_indexer_rope_fwd_inplace_flat"):
            raise RuntimeError("DSA indexer flat RoPE CUDA extension is unavailable")

        if not x.is_contiguous():
            raise RuntimeError("DSA indexer flat RoPE in-place input must be contiguous")
        if not _tensor_safe_for_custom_inplace(x):
            raise RuntimeError("DSA indexer flat RoPE in-place input is not autograd-safe")
        ctx.mark_dirty(x)
        ext.dsa_indexer_rope_fwd_inplace_flat(
            x,
            rotary_pos_emb,
            int(heads),
            int(head_dim),
            int(pe_dim),
            float(mscale),
            bool(interleaved),
        )
        ctx.save_for_backward(rotary_pos_emb)
        ctx.heads = int(heads)
        ctx.head_dim = int(head_dim)
        ctx.pe_dim = int(pe_dim)
        ctx.mscale = float(mscale)
        ctx.interleaved = bool(interleaved)
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (rotary_pos_emb,) = ctx.saved_tensors
        ext = _try_load_hisa_cuda_ext()
        if ext is None or not hasattr(ext, "dsa_indexer_rope_bwd_flat"):
            raise RuntimeError("DSA indexer flat RoPE CUDA extension is unavailable")

        grad_out_contig = grad_out.contiguous()
        grad_x = torch.empty_like(grad_out_contig)
        ext.dsa_indexer_rope_bwd_flat(
            grad_out_contig,
            rotary_pos_emb,
            grad_x,
            ctx.heads,
            ctx.head_dim,
            ctx.pe_dim,
            ctx.mscale,
            ctx.interleaved,
        )
        return grad_x, None, None, None, None, None, None


def _dsa_indexer_key_block_size(topk: int) -> int:
    raw = os.getenv(_DSA_INDEXER_KEY_BLOCK_SIZE_ENV)
    if raw:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{_DSA_INDEXER_KEY_BLOCK_SIZE_ENV} must be positive, got {value}")
        return value
    return max(2048, min(4096, max(1, topk)))


def _hisa_target_row_chunk(q_len: int) -> int:
    raw = os.getenv(_HISA_TARGET_ROW_CHUNK_ENV)
    if raw:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{_HISA_TARGET_ROW_CHUNK_ENV} must be positive, got {value}")
        return min(value, q_len)
    return min(128, q_len)


def _dsa_topk_buffer_dtype(sk: int) -> torch.dtype:
    if _dsa_topk_indices_can_use_int16(sk):
        return torch.int16
    return torch.int32


def _dsa_topk_indices_can_use_int16(sk: int) -> bool:
    # Valid selected-token ids are [-1, sk - 1]. 32k context therefore fits in
    # signed int16 because the largest real id is 32767.
    return (
        _env_flag_enabled(_DSA_COMPACT_TOPK_INDICES_ENV, "0")
        and sk <= torch.iinfo(torch.int16).max + 1
    )


def _maybe_narrow_dsa_topk_indices(topk_indices: torch.Tensor, sk: int) -> torch.Tensor:
    """Keep selected-token buffers off int64 in CUDA hot paths.

    PyTorch `topk` emits `long` indices, but the DSA Triton/CUDA kernels and
    current sequence lengths only need compact selected-token IDs. Narrowing here
    also covers dense/streaming fallback selectors before they feed the shared
    loss and attention code. Do not narrow below int32 if a future caller exceeds
    the signed int16 id range.
    """

    if _dsa_topk_indices_can_use_int16(sk) and topk_indices.dtype != torch.int16:
        return topk_indices.contiguous().to(torch.int16)
    if topk_indices.dtype == torch.long and sk <= torch.iinfo(torch.int32).max:
        return topk_indices.contiguous().to(torch.int32)
    return topk_indices


def _maybe_compact_hisa_saved_topk_indices(topk_indices: torch.Tensor, sk: int) -> torch.Tensor:
    """Store HISA selected-score backward indices in the smallest supported dtype."""

    if _dsa_topk_indices_can_use_int16(sk):
        return topk_indices.contiguous().to(torch.int16)
    if topk_indices.dtype != torch.int32:
        return topk_indices.contiguous().to(torch.int32)
    return topk_indices.contiguous()


def _maybe_sort_dsa_topk_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    if not _env_flag_enabled(_DSA_SORT_TOPK_INDICES_ENV, "0"):
        return topk_indices
    return topk_indices.sort(dim=-1).values


def _maybe_sort_dsa_topk_indices_and_scores(
    topk_indices: torch.Tensor,
    selected_scores: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not _env_flag_enabled(_DSA_SORT_TOPK_INDICES_ENV, "0"):
        return topk_indices, selected_scores
    sorted_indices, order = topk_indices.sort(dim=-1)
    if selected_scores is None:
        return sorted_indices, None
    original_shape = topk_indices.shape
    sorted_scores = selected_scores.reshape(original_shape).gather(-1, order)
    return sorted_indices, sorted_scores.reshape_as(selected_scores)


def _dsa_process_group_size(group: Optional[torch.distributed.ProcessGroup]) -> int:
    return group.size() if group is not None else 1


def _dsa_process_group_rank(group: Optional[torch.distributed.ProcessGroup]) -> int:
    return group.rank() if group is not None else 0


def _in_te_no_grad_activation_recompute_forward() -> bool:
    """Return True in TE's checkpoint forward phase, before backward replay."""

    try:
        from transformer_engine.pytorch.distributed import (
            in_fp8_activation_recompute_phase,
            is_fp8_activation_recompute_enabled,
        )
    except (ImportError, ModuleNotFoundError):
        return False
    return (
        is_fp8_activation_recompute_enabled()
        and not in_fp8_activation_recompute_phase()
        and not torch.is_grad_enabled()
    )


class _BroadcastFromTensorParallelOwner(torch.autograd.Function):
    """Broadcast a compact sequence-parallel chunk and reduce its gradient to the owner."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, owner_global_rank: int, group):
        ctx.owner_global_rank = int(owner_global_rank)
        ctx.group = group
        output = torch.empty_like(input_)
        if torch.distributed.get_rank() == ctx.owner_global_rank:
            output.copy_(input_)
        torch.distributed.broadcast(output, src=ctx.owner_global_rank, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.contiguous()
        torch.distributed.reduce(
            grad_input,
            dst=ctx.owner_global_rank,
            op=torch.distributed.ReduceOp.SUM,
            group=ctx.group,
        )
        if torch.distributed.get_rank() != ctx.owner_global_rank:
            grad_input.zero_()
        return grad_input, None, None


def _torch_layer_norm_like_te(layer_norm: torch.nn.Module, x: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply a torch LayerNorm equivalent for TE LayerNorm modules."""

    output_dtype = x.dtype
    weight = getattr(layer_norm, "weight", None)
    bias = getattr(layer_norm, "bias", None)
    if weight is None:
        raise AttributeError("LayerNorm fallback requires a weight parameter")

    x = _dequantize_tensor_if_needed(x)

    zero_centered_gamma = bool(getattr(layer_norm, "zero_centered_gamma", False))
    if zero_centered_gamma:
        weight = weight + 1
    weight = weight.to(dtype=x.dtype)
    if bias is not None:
        bias = bias.to(dtype=x.dtype)

    output = F.layer_norm(
        x,
        (x.size(-1),),
        weight=weight,
        bias=bias,
        eps=eps,
    )
    return output.to(dtype=output_dtype)


def _dsa_cp_position_ids_for_rank(
    local_seq_len: int, cp_size: int, cp_rank: int, device: torch.device
) -> torch.Tensor:
    """Return absolute sequence positions for one CP rank's zigzag-local tokens."""

    if cp_size <= 1:
        return torch.arange(local_seq_len, device=device, dtype=torch.long)
    if local_seq_len % 2 != 0:
        raise ValueError(
            f"DSA CP position mapping expects an even local sequence length, got {local_seq_len}"
        )

    chunk_len = local_seq_len // 2
    first = torch.arange(
        cp_rank * chunk_len, (cp_rank + 1) * chunk_len, device=device, dtype=torch.long
    )
    second_chunk = 2 * cp_size - cp_rank - 1
    second = torch.arange(
        second_chunk * chunk_len,
        (second_chunk + 1) * chunk_len,
        device=device,
        dtype=torch.long,
    )
    return torch.cat((first, second), dim=0)


def _dsa_cp_local_position_ids(
    local_seq_len: int, cp_group: Optional[torch.distributed.ProcessGroup], device: torch.device
) -> torch.Tensor:
    return _dsa_cp_position_ids_for_rank(
        local_seq_len,
        _dsa_process_group_size(cp_group),
        _dsa_process_group_rank(cp_group),
        device,
    )


def _dsa_cp_gathered_position_ids(
    local_seq_len: int, cp_group: Optional[torch.distributed.ProcessGroup], device: torch.device
) -> torch.Tensor:
    cp_size = _dsa_process_group_size(cp_group)
    if cp_size <= 1:
        return torch.arange(local_seq_len, device=device, dtype=torch.long)
    return torch.cat(
        [
            _dsa_cp_position_ids_for_rank(local_seq_len, cp_size, cp_rank, device)
            for cp_rank in range(cp_size)
        ],
        dim=0,
    )


class _DSACPSortedZigzagGather(torch.autograd.Function):
    """Gather CP zigzag-local tensors directly into absolute-position order."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup):
        cp_size = _dsa_process_group_size(cp_group)
        cp_rank = _dsa_process_group_rank(cp_group)
        if cp_size <= 1:
            ctx.cp_group = None
            return input_
        local_seq_len = input_.size(0)
        if local_seq_len % 2 != 0:
            raise ValueError(
                "DSA CP sorted gather expects an even local sequence length, got "
                f"{local_seq_len}"
            )

        chunk_len = local_seq_len // 2
        output_shape = (local_seq_len * cp_size, *input_.shape[1:])
        output = torch.empty(output_shape, dtype=input_.dtype, device=input_.device)

        first_views = [
            output[rank * chunk_len : (rank + 1) * chunk_len] for rank in range(cp_size)
        ]
        torch.distributed.all_gather(
            first_views,
            input_[:chunk_len].contiguous(),
            group=cp_group,
        )

        second_views = []
        for rank in range(cp_size):
            seq_chunk = 2 * cp_size - rank - 1
            second_views.append(output[seq_chunk * chunk_len : (seq_chunk + 1) * chunk_len])
        torch.distributed.all_gather(
            second_views,
            input_[chunk_len:].contiguous(),
            group=cp_group,
        )

        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.chunk_len = chunk_len
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cp_group = ctx.cp_group
        if cp_group is None:
            return grad_output, None

        chunk_len = ctx.chunk_len
        cp_rank = ctx.cp_rank
        cp_size = ctx.cp_size
        local_grad = grad_output.new_empty((chunk_len * 2, *grad_output.shape[1:]))
        local_grad[:chunk_len].copy_(
            grad_output[cp_rank * chunk_len : (cp_rank + 1) * chunk_len]
        )
        seq_chunk = 2 * cp_size - cp_rank - 1
        local_grad[chunk_len:].copy_(
            grad_output[seq_chunk * chunk_len : (seq_chunk + 1) * chunk_len]
        )
        torch.distributed.all_reduce(local_grad, group=cp_group)
        return local_grad, None


def _dsa_cp_sorted_zigzag_gather(
    input_: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    if _dsa_process_group_size(cp_group) <= 1:
        return input_
    if not _env_flag_enabled(_DSA_CP_SORTED_ZIGZAG_GATHER_ENV, "0"):
        return gather_from_sequence_parallel_region(input_, group=cp_group)
    return _DSACPSortedZigzagGather.apply(input_, cp_group)


def _dsa_thd_local_sequence_offsets(
    cu_seqlens: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
) -> list[tuple[int, int]]:
    """Return local packed-THD offsets for each sequence on this CP rank."""

    cp_size = _dsa_process_group_size(cp_group)
    offsets = cu_seqlens.detach().cpu().tolist()
    local_offsets = []
    local_cursor = 0
    for start, end in zip(offsets[:-1], offsets[1:]):
        seq_len = int(end) - int(start)
        if seq_len <= 0:
            local_offsets.append((local_cursor, local_cursor))
            continue
        if seq_len % cp_size != 0:
            raise ValueError(
                f"DSA packed THD CP expects each padded sequence length to be divisible by "
                f"context_parallel_size; got sequence length {seq_len} and CP {cp_size}"
            )
        local_len = seq_len // cp_size
        local_offsets.append((local_cursor, local_cursor + local_len))
        local_cursor += local_len
    return local_offsets


def _dsa_thd_tp_local_sequence_offsets(
    cu_seqlens: torch.Tensor,
    local_seq_len: int,
    tp_group: Optional[torch.distributed.ProcessGroup],
) -> list[tuple[int, int]]:
    """Return local packed-THD offsets for TP sequence-parallel shards.

    Packed THD cu_seqlens describe the global packed token stream. With TP
    sequence parallelism, each rank owns a contiguous token slice from every
    packed sequence, not a whole-sequence slice selected by TP rank. Rebase the
    global per-sequence lengths onto the local packed stream so recursion never
    slices past the rank-local key/value tensors.
    """

    tp_size = _dsa_process_group_size(tp_group)
    if tp_size <= 1:
        offsets = cu_seqlens.detach().cpu().tolist()
        return [(int(start), int(end)) for start, end in zip(offsets[:-1], offsets[1:])]

    global_offsets = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    local_offsets = []
    local_cursor = 0
    for start, end in zip(global_offsets[:-1], global_offsets[1:]):
        seq_len = int(end) - int(start)
        if seq_len <= 0:
            local_offsets.append((local_cursor, local_cursor))
            continue
        if seq_len % tp_size != 0:
            raise ValueError(
                f"DSA packed THD TP expects each padded sequence length to be divisible by "
                f"tensor_model_parallel_size; got sequence length {seq_len} and TP {tp_size}"
            )
        local_len = seq_len // tp_size
        local_offsets.append((local_cursor, local_cursor + local_len))
        local_cursor += local_len
    if local_seq_len >= 0 and local_cursor > local_seq_len:
        raise ValueError(
            f"DSA packed THD TP local offsets exceed local tensor length: "
            f"offsets end at {local_cursor}, local length {local_seq_len}"
        )
    return local_offsets


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    x = _dequantize_tensor_if_needed(x, dtype=torch.bfloat16)
    if x.dtype != torch.bfloat16:
        x = x.to(dtype=torch.bfloat16)
    if x.numel() == 0:
        return x
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    hidden_size = x.size(-1)
    return hadamard_transform(x.contiguous(), scale=hidden_size**-0.5)


class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the indexer loss for logging.

        Args:
            loss: The loss tensor.
            layer_number: Layer index of the loss, 1-indexed.
            num_layers: The number of total layers.
            reduce_group: The group for reducing the loss.
            avg_group: The group for averaging the loss.
        """
        # Skip indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker():
        """Collect and reduce the indexer losses across ranks."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]

        torch.distributed.all_reduce(
            values, group=parallel_state.get_pipeline_model_parallel_group()
        )
        # Reduce indexer losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )
        torch.distributed.all_reduce(
            values,
            group=parallel_state.get_data_parallel_group(with_context_parallel=False),
            op=torch.distributed.ReduceOp.AVG,
        )

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer,
        wandb_writer=None,
        total_loss_dict=None,
        per_layer_logging: bool = False,
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            wandb_writer: Weights & Biases writer.
            total_loss_dict: Dictionary to accumulate total losses.
            per_layer_logging: Whether to log per-layer losses.
        """
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        indexer_loss_values = tracker["values"] * loss_scale
        num_layers = indexer_loss_values.shape[0]

        # Average across all layers (assuming all layers have sparse attention)
        avg_indexer_loss = indexer_loss_values.sum() / num_layers

        # Log average loss
        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss

        if writer is not None:
            writer.add_scalar("indexer loss", avg_indexer_loss, iteration)

        if wandb_writer is not None:
            wandb_writer.log({"indexer loss": avg_indexer_loss}, iteration)

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


class DSAIndexerAuxLossState:
    """Track per-microbatch DSA auxiliary losses for explicit loss composition."""

    losses = []

    @classmethod
    def add(cls, loss: torch.Tensor):
        cls.losses.append(loss)

    @classmethod
    def total(cls) -> Optional[torch.Tensor]:
        if not cls.losses:
            return None
        return torch.stack(cls.losses).sum()

    @classmethod
    def clear(cls):
        cls.losses.clear()


def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    sq, b, np, hn = query.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += causal_mask.view(1, 1, sq, sk)
    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    # [b, sq, sk] -> [b, sq, sk]
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    # L1 normalize target on the last dimension. Doesn't use abs() because attention_scores are
    # obtained from softmax so they are already non-negative.
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

    # Compute KL divergence: KL(target || index) = target(x) * log(target(x) / index(x))
    # kl_per_element [b, sq, sk]
    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )

    # [b, sq, sk] -> [b, sq] -> [1]
    # Each token has same weight in the loss.
    kl_div = kl_per_element.sum(dim=-1).mean()

    # Scale by coefficient.
    indexer_loss = kl_div * loss_coeff

    return indexer_loss


def _compute_index_scores(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Perform index score using BF16 precision.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
    This is a BF16 implementation of the `fp8_index` logic:
        1. Compute attention scores: q @ k^T;
        2. Apply ReLU activation;
        3. Weight by attention weights;
        4. Sum across attention heads.

    Args:
        q: BF16 [seqlen_q, batch, index_n_heads, index_head_dim], the query tensor.
        weights: BF16 [seqlen_q, batch, index_n_heads], the attention weights.
        k: BF16 [seqlen_k, batch, index_head_dim], the key tensor.

    Returns:
        index_scores: FP32 [batch, seqlen_q, seqlen_k], the index scores.
    """
    # Compute attention scores: q @ k^T
    # [seqlen_q, batch, index_n_heads, index_head_dim] @ [seqlen_k, batch, index_head_dim]^T
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

    # Keep this out-of-place: when the DSA KL loss is enabled, autograd needs
    # the ReLU output version to remain stable through the weight multiply.
    index_scores = torch.relu(index_scores)

    # Weight each head by attention weights.
    # [seqlen_q, batch, index_n_heads, seqlen_k] * [seqlen_q, batch, index_n_heads, 1]
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = index_scores * weights.unsqueeze(-1)

    # Sum across attention heads.
    # [seqlen_q, batch, index_n_heads, seqlen_k] -> [seqlen_q, batch, seqlen_k]
    index_scores = index_scores.sum(dim=2)

    # Transpose to [batch, seqlen_q, seqlen_k].
    index_scores = index_scores.transpose(0, 1)

    return index_scores


def fused_qk_topk_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    mask: Optional[torch.Tensor] = None,
):
    """Naive implementation of QK Topk."""
    seqlen = q.size(0)
    # =========================================
    # Compute index scores
    # =========================================
    # [batch, seqlen, seqlen]
    index_scores = _compute_index_scores(q, weights, k)
    if mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    # =========================================
    # Select top-k indices
    # =========================================
    topk_k = min(index_topk, seqlen)
    # [batch, seqlen, index_topk]
    # Keep this legacy fused-loss helper deterministic and aligned with the
    # autograd reference tests. The production chunked/HISA path keeps its own
    # sorted=False top-k where order is not part of the sparse attention semantics.
    topk_indices = index_scores.topk(topk_k, dim=-1, sorted=True)[1]

    return index_scores, topk_indices


def fwd_fused_indexer_loss_naive(
    q, weights, k, query, key, topk, softmax_scale, loss_coeff, mask, sparse_loss, pg_collection
):
    """Naive implementation of forward pass for indexer loss."""
    index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, topk, mask)

    indexer_loss = compute_dsa_indexer_loss(
        index_scores,
        topk_indices,
        query,
        key,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        pg_collection,
    )

    return topk_indices, indexer_loss


def bwd_fused_indexer_loss_naive(
    q,
    weights,
    k,
    query,
    key,
    topk_indices,
    softmax_scale,
    loss_coeff,
    sparse_loss,
    grad_loss,
    pg_collection,
):
    """Naive implementation of backward pass for indexer loss."""
    index_scores = _compute_index_scores(q, weights, k)  # [B, Sq, Sk]

    sq, b, np, hn = query.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query_reshaped = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key_reshaped = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query_reshaped.float(), key_reshaped.float()) * softmax_scale
    # Free reshaped tensors - no longer needed after bmm
    del query_reshaped, key_reshaped

    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # Apply causal mask to both attention and index scores
    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores = attention_scores + causal_mask.view(1, 1, sq, sk)
    # [b, sq, sk] + [1, sq, sk] -> [b, sq, sk]
    index_scores = index_scores + causal_mask.unsqueeze(0)
    # Free causal_mask - no longer needed
    del causal_mask

    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_mask

    # Compute softmax for both
    attention_scores_softmax = torch.nn.functional.softmax(
        attention_scores, dim=-1, dtype=torch.float32
    )
    # Free attention_scores immediately
    del attention_scores

    index_scores_softmax = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)
    # Free index_scores - no longer needed after softmax
    del index_scores

    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    # Free attention_scores_softmax
    del attention_scores_softmax

    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores_sum.contiguous(), group=pg_collection.tp)

    # L1 normalize
    attention_scores_normalized = attention_scores_sum / attention_scores_sum.sum(
        dim=-1, keepdim=True
    )
    # Free attention_scores_sum - no longer needed after normalization
    del attention_scores_sum

    # Backward through loss = kl_div * loss_coeff
    # where kl_div = kl_per_element.sum(dim=-1).mean()
    grad_kl_div = grad_loss * loss_coeff  # scalar

    # Backward through mean: distribute gradient equally
    grad_kl_per_row = grad_kl_div / (b * sq)  # scalar value for each row

    # Backward through sum(dim=-1): broadcast back to [b, sq, sk]
    # Each element in a row contributes to the sum, so gradient is same for all
    grad_kl_per_element = grad_kl_per_row.view(1, 1, 1).expand(b, sq, sk)

    # Backward through kl_per_element = target * (log(target) - log(index))
    # ∂kl/∂index_softmax = -target / index_softmax
    grad_index_scores_softmax = (
        -attention_scores_normalized / (index_scores_softmax + 1e-10) * grad_kl_per_element
    )
    # Free attention_scores_normalized - no longer needed
    del attention_scores_normalized

    # Backward through softmax: ∂L/∂x = softmax * (∂L/∂softmax - sum(∂L/∂softmax * softmax))
    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(dim=-1, keepdim=True)
    grad_index_scores_logits = index_scores_softmax * (grad_index_scores_softmax - sum_grad)
    # Free intermediate tensors
    del index_scores_softmax, grad_index_scores_softmax, sum_grad

    # Zero out gradients for masked positions
    # Create a mask for valid (non-masked) positions
    # Causal mask: position (i, j) is valid if j <= i
    causal_valid_mask = torch.tril(
        torch.ones((sq, sk), device=q.device, dtype=torch.bool)
    )  # [sq, sk]
    if sparse_loss:
        # Also apply index mask - only topk positions are valid
        index_valid_mask = index_mask == 0  # [b, sq, sk]
        del index_mask  # Free index_mask immediately after use
        valid_mask = causal_valid_mask.unsqueeze(0) & index_valid_mask  # [b, sq, sk]
        del index_valid_mask
    else:
        del index_mask  # Free index_mask even if not used for sparse_loss
        valid_mask = causal_valid_mask.unsqueeze(0).expand(b, sq, sk)  # [b, sq, sk]
    del causal_valid_mask

    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    del valid_mask

    # Transpose from [b, sq, sk] to [sq, b, sk]
    grad_index_scores = grad_index_scores_logits.transpose(0, 1)  # [sq, b, sk]
    del grad_index_scores_logits

    # Backward through sum over heads: expand gradient
    grad_weighted_scores = grad_index_scores.unsqueeze(2)  # [sq, b, 1, sk]
    del grad_index_scores

    # Compute forward values needed for backward
    scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())  # [sq, b, h, sk]
    # Compute relu_mask before relu (saves memory vs keeping both scores and relu output)
    relu_mask = scores > 0
    scores_after_relu = torch.relu(scores)
    del scores

    # Backward through multiplication by weights: index_scores_per_head * weights
    # ∂L/∂weights = grad * relu_scores (sum over sk)
    grad_weights = (grad_weighted_scores * scores_after_relu).sum(dim=-1)  # [sq, b, h]

    # ∂L/∂relu_scores = grad * weights
    grad_scores_after_relu = grad_weighted_scores * weights.unsqueeze(-1)  # [sq, b, h, sk]
    del grad_weighted_scores, scores_after_relu

    # Backward through ReLU
    grad_scores = grad_scores_after_relu * relu_mask.float()  # [sq, b, h, sk]
    del grad_scores_after_relu, relu_mask

    # Backward through einsum 'sbhd,tbd->sbht'
    # ∂L/∂q = einsum('sbht,tbd->sbhd', grad_scores, k)
    grad_q = torch.einsum('sbht,tbd->sbhd', grad_scores, k.float())  # [sq, b, h, d]
    # ∂L/∂k = einsum('sbht,sbhd->tbd', grad_scores, q)
    grad_k = torch.einsum('sbht,sbhd->tbd', grad_scores, q.float())  # [sk, b, d]
    del grad_scores

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


class FusedDSAIndexerLoss(torch.autograd.Function):
    """Fused implementation of DSA Indexer Loss."""

    @staticmethod
    def forward(
        ctx,
        q,
        weights,
        k,
        query,
        key,
        softmax_scale,
        topk,
        loss_coeff,
        mask,
        sparse_loss,
        pg_collection,
    ):
        """
        Fused forward: index_scores never materialized in full.
        """
        topk_indices, loss = fwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk,
            softmax_scale,
            loss_coeff,
            mask,
            sparse_loss,
            pg_collection,
        )

        # Save for backward (recomputation strategy)
        ctx.save_for_backward(q, weights, k, query, key, topk_indices)
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        ctx.pg_collection = pg_collection

        return topk_indices, loss

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss):
        """
        Backward: Recompute what we need.
        """
        q, weights, k, query, key, topk_indices = ctx.saved_tensors

        grad_q, grad_weights, grad_k = bwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            ctx.softmax_scale,
            ctx.loss_coeff,
            ctx.sparse_loss,
            grad_loss,
            ctx.pg_collection,
        )

        # query and key are detached in forward, so return None for their gradients
        return grad_q, grad_weights, grad_k, None, None, None, None, None, None, None, None


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for indexer loss.

    This custom autograd function attaches a KL divergence loss to the activation
    to train the indexer to predict attention scores without affecting the forward pass.
    """

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output: The output tensor (activation).
            indexer_loss: The indexer KL divergence loss tensor.

        Returns:
            torch.Tensor: The output tensor unchanged.
        """
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output: The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                gradient.
        """
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale: The scale value to set.
        """
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


@dataclass
class DSAIndexerSubmodules:
    """
    Configuration class for specifying the submodules of an DSA Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_wk: Linear projection for key.
        k_norm: Layer normalization for key.
        linear_weights_proj: Linear projection for attention weights.
    """

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


@dataclass
class DSAttentionSubmodules:
    """
    Configuration class for specifying the submodules of DSAttention.

    Args:
        indexer: DSA Indexer module for computing sparse attention indices.
    """

    indexer: Union[ModuleSpec, type] = None


class DSAIndexer(MegatronModule):
    """
    DSA Lightning Indexer for DeepSeek Sparse Attention.

    Computes index scores to identify the top-k most relevant key-value pairs for each query in
    sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            submodules (DSAIndexerSubmodules): Indexer submodules specification.
            pg_collection (ProcessGroupCollection, optional): Process groups for the indexer.
        """
        super().__init__(config=config)
        self._indexer_rope_config = copy.copy(self.config)
        self._indexer_rope_config.apply_rope_fusion = False
        self.hidden_size = self.config.hidden_size
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            self.config.q_lora_rank
            if self.config.q_lora_rank is not None
            else self.config.hidden_size
        )

        self.index_n_heads = self.config.dsa_indexer_n_heads
        self.index_head_dim = self.config.dsa_indexer_head_dim
        self.index_topk = self.config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        self.indexcache_config = None
        self.indexcache_hisa_config = None
        indexcache_quantization = getattr(
            self.config, "dsa_indexcache_quantization", "disabled"
        )
        indexcache_quantization_enabled = getattr(
            self.config, "dsa_indexcache_quant_enabled", False
        )
        if indexcache_quantization_enabled or indexcache_quantization != "disabled":
            from megatron.core.quantization.indexcache import (
                build_indexcache_config,
                resolve_indexcache_quantization,
            )

            self.indexcache_config = build_indexcache_config(
                eps=getattr(self.config, "dsa_indexcache_quant_eps", 1e-4),
                quantization=resolve_indexcache_quantization(
                    quantization=indexcache_quantization,
                    quant_enabled=indexcache_quantization_enabled,
                ),
            )
        if getattr(self.config, "dsa_indexcache_hisa_enabled", False):
            if (
                self.indexcache_config is None
                or self.indexcache_config.quantization != INDEXCACHE_QUANT_NVFP4
            ):
                raise ValueError(
                    "dsa_indexcache_hisa_enabled requires "
                    "dsa_indexcache_quantization='nvfp4_e2m1_ue8m0'."
                )
            self.indexcache_hisa_config = IndexCacheHISAConfig(
                enabled=True,
                block_size=int(getattr(self.config, "dsa_indexcache_hisa_block_size", 128)),
                block_topk=int(getattr(self.config, "dsa_indexcache_hisa_block_topk", 64)),
                compression_ratio=float(
                    getattr(self.config, "dsa_indexcache_hisa_compression_ratio", 4.0)
                ),
                fallback_to_dense_if_short=_env_flag_enabled(
                    _HISA_FALLBACK_DENSE_IF_SHORT_ENV, "0"
                ),
            )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        # Initialize Position Embedding.
        if self.config.rope_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f'Unsupported RoPE type: {self.config.rope_type}, supported types are "rope" and '
                f'"yarn"'
            )

        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_wk = build_module(
            submodules.linear_wk,
            self.hidden_size,
            self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        k_norm_config = copy.copy(self.config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            config=k_norm_config,
            hidden_size=self.index_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _apply_rope(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor, mscale: float):
        """Apply RoPE to the input tensor."""
        if self._use_fused_indexer_rope_writer(x, rotary_pos_emb):
            return _DSAIndexerRopeFunction.apply(
                x,
                rotary_pos_emb,
                self.qk_pos_emb_head_dim,
                float(mscale),
                bool(self._indexer_rope_config.rotary_interleaved),
            )

        # x_pe   [seqlen, batch, *, qk_pos_emb_head_dim]
        # x_nope [seqlen, batch, *, index_head_dim - qk_pos_emb_head_dim]
        # To align with DeepSeek's implementation,
        # x_pe is placed at the front, and x_nope is placed at the back.
        x_pe, x_nope = torch.split(
            x, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )
        if self._use_fused_indexer_rope(x_pe, rotary_pos_emb):
            x_pe = fused_apply_rotary_pos_emb(
                x_pe,
                rotary_pos_emb,
                interleaved=self._indexer_rope_config.rotary_interleaved,
            )
            if mscale != 1.0:
                x_pe = x_pe * mscale
        else:
            x_pe = apply_rotary_pos_emb(
                x_pe,
                rotary_pos_emb,
                config=self._indexer_rope_config,
                cu_seqlens=None,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
                # This flag is for the MLA-style interleaving in RoPE.
                # Set it to False, as indexer does not apply interleaved RoPE.
                mla_rotary_interleaved=False,
            )
        # [seqlen, batch, *, index_head_dim]
        x = torch.cat([x_pe, x_nope], dim=-1)
        return x

    def _use_fused_indexer_flat_rope_writer(
        self,
        x: torch.Tensor,
        rotary_pos_emb,
        heads: int,
        head_dim: int,
    ) -> bool:
        """Return whether flat projected indexer RoPE can run before reshape."""
        if not _env_flag_enabled(_DSA_INDEXER_ROPE_FUSION_ENV, "1"):
            return False
        if not _env_flag_enabled(_DSA_INDEXER_ROPE_INPLACE_ENV, "1"):
            return False
        if isinstance(rotary_pos_emb, tuple) or rotary_pos_emb is None:
            return False
        if x.dim() != 3 or rotary_pos_emb.dim() != 4:
            return False
        if not x.is_contiguous():
            return False
        if not _tensor_safe_for_custom_inplace(x):
            return False
        if not x.is_cuda or not rotary_pos_emb.is_cuda:
            return False
        if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return False
        if rotary_pos_emb.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return False
        if x.is_leaf and x.requires_grad:
            return False
        if self._indexer_rope_config.mrope_section is not None and rotary_pos_emb.shape[1] > 1:
            return False
        if heads <= 0 or head_dim != self.index_head_dim:
            return False
        if x.size(-1) != heads * head_dim:
            return False
        if self.qk_pos_emb_head_dim <= 0 or self.qk_pos_emb_head_dim > head_dim:
            return False
        if self.qk_pos_emb_head_dim % 2 != 0:
            return False
        if rotary_pos_emb.size(-1) != self.qk_pos_emb_head_dim:
            return False
        if rotary_pos_emb.size(0) not in (1, x.size(0)):
            return False
        if rotary_pos_emb.size(1) not in (1, x.size(1)):
            return False
        if rotary_pos_emb.size(2) not in (1, heads):
            return False
        ext = _try_load_hisa_cuda_ext()
        return ext is not None and hasattr(ext, "dsa_indexer_rope_fwd_inplace_flat")

    def _use_fused_indexer_rope_writer(self, x: torch.Tensor, rotary_pos_emb) -> bool:
        """Return whether DSA indexer RoPE can write the full head in one CUDA call."""
        if not _env_flag_enabled(_DSA_INDEXER_ROPE_FUSION_ENV, "1"):
            return False
        if isinstance(rotary_pos_emb, tuple) or rotary_pos_emb is None:
            return False
        if x.dim() != 4 or rotary_pos_emb.dim() != 4:
            return False
        if not x.is_cuda or not rotary_pos_emb.is_cuda:
            return False
        if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return False
        if rotary_pos_emb.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return False
        if self._indexer_rope_config.mrope_section is not None and rotary_pos_emb.shape[1] > 1:
            return False
        if x.size(-1) != self.index_head_dim:
            return False
        if self.qk_pos_emb_head_dim <= 0 or self.qk_pos_emb_head_dim > self.index_head_dim:
            return False
        if self.qk_pos_emb_head_dim % 2 != 0:
            return False
        if rotary_pos_emb.size(-1) != self.qk_pos_emb_head_dim:
            return False
        if rotary_pos_emb.size(0) not in (1, x.size(0)):
            return False
        if rotary_pos_emb.size(1) not in (1, x.size(1)):
            return False
        if rotary_pos_emb.size(2) not in (1, x.size(2)):
            return False
        ext = _try_load_hisa_cuda_ext()
        return ext is not None and hasattr(ext, "dsa_indexer_rope_fwd")

    def _use_fused_indexer_rope(self, x_pe: torch.Tensor, rotary_pos_emb) -> bool:
        """Return whether DSA indexer RoPE can use TE's fused BSHD RoPE kernel."""
        if not _env_flag_enabled(_DSA_INDEXER_ROPE_FUSION_ENV, "1"):
            return False
        if not self.config.apply_rope_fusion:
            return False
        if fused_apply_rotary_pos_emb is None:
            return False
        if isinstance(rotary_pos_emb, tuple) or rotary_pos_emb is None:
            return False
        if x_pe.dim() != 4 or rotary_pos_emb.dim() != 4:
            return False
        if not x_pe.is_cuda or not rotary_pos_emb.is_cuda:
            return False
        if self._indexer_rope_config.mrope_section is not None and rotary_pos_emb.shape[1] > 1:
            return False
        return x_pe.size(-1) == rotary_pos_emb.size(-1)

    @staticmethod
    def _slice_rotary_pos_emb(rotary_pos_emb, start: int, end: int):
        if isinstance(rotary_pos_emb, tuple):
            return tuple(None if item is None else item[start:end] for item in rotary_pos_emb)
        return rotary_pos_emb[start:end]

    @staticmethod
    def _index_rotary_pos_emb(rotary_pos_emb, indices: torch.Tensor):
        if isinstance(rotary_pos_emb, tuple):
            return tuple(
                None if item is None else item.index_select(0, indices.to(item.device))
                for item in rotary_pos_emb
            )
        return rotary_pos_emb.index_select(0, indices.to(rotary_pos_emb.device))

    def _prepare_inputs_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
        gather_sequence_parallel: bool = True,
    ):
        """Prepare shared DSA indexer inputs before per-query projection/top-k work."""
        orig_seqlen = x.size(0)
        pad_len = 0
        if self.config.fp4 and packed_seq_params is None:
            # TE NVFP4 kernels require the flattened token dimension to be divisible
            # by 16. Packed THD recursion can hand the indexer arbitrary per-sample
            # lengths, so pad the private indexer projections and trim before use.
            bsz = x.size(1)
            while ((orig_seqlen + pad_len) * bsz) % 16 != 0:
                pad_len += 1
            if pad_len:
                x = torch.cat(
                    (x, x.new_zeros((pad_len, bsz, x.size(2)))),
                    dim=0,
                )
                qr = torch.cat(
                    (qr, qr.new_zeros((pad_len, bsz, qr.size(2)))),
                    dim=0,
                )

        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

        if (
            gather_sequence_parallel
            and self.config.sequence_parallel
            and self.pg_collection.tp.size() > 1
        ):
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        return x, qr, rotary_pos_emb, mscale, orig_seqlen, pad_len

    def _project_query_before_topk(
        self,
        query_x: torch.Tensor,
        query_qr: torch.Tensor,
        query_rotary_pos_emb,
        mscale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project one query span into DSA indexer q and score weights."""
        query_seqlen, bsz, _ = query_qr.size()
        q, _ = self.linear_wq_b(query_qr)
        _dsa_debug_sync("indexer query linear_wq_b", q)
        if self._use_fused_indexer_flat_rope_writer(
            q, query_rotary_pos_emb, self.index_n_heads, self.index_head_dim
        ):
            q = _DSAIndexerFlatRopeInplaceFunction.apply(
                q,
                query_rotary_pos_emb,
                self.index_n_heads,
                self.index_head_dim,
                self.qk_pos_emb_head_dim,
                float(mscale),
                bool(self._indexer_rope_config.rotary_interleaved),
            )
            q = q.reshape(query_seqlen, bsz, self.index_n_heads, self.index_head_dim)
            _dsa_debug_sync("indexer query fused rope", q)
        else:
            q = q.reshape(query_seqlen, bsz, self.index_n_heads, self.index_head_dim)
            q = self._apply_rope(q, query_rotary_pos_emb, mscale)
            _dsa_debug_sync("indexer query torch rope", q)
        q = rotate_activation(q)
        _dsa_debug_sync("indexer query hadamard", q)

        weights, _ = self.linear_weights_proj(query_x)
        _dsa_debug_sync("indexer query weights", weights)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale
        return q, weights

    def _project_query_chunk_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        rotary_pos_emb,
        mscale: float,
        q_start: int,
        q_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._project_query_before_topk(
            x[q_start:q_end],
            qr[q_start:q_end],
            self._slice_rotary_pos_emb(rotary_pos_emb, q_start, q_end),
            mscale,
        )

    def _project_key_before_topk(
        self,
        x: torch.Tensor,
        rotary_pos_emb,
        mscale: float,
        orig_seqlen: int,
        pad_len: int,
        apply_indexcache: bool = True,
    ) -> torch.Tensor:
        """Project the full DSA indexer key prefix once."""
        seqlen, bsz, _ = x.size()
        k, _ = self.linear_wk(x)
        _dsa_debug_sync("indexer key linear_wk", k)
        debug_layer_number = getattr(self, "layer_number", None)
        _dsa_register_indexer_k_grad_debug("linear_wk_output", debug_layer_number, k)
        if self.config.fp4 and (
            not torch.is_grad_enabled() or _env_flag_enabled(_DSA_INDEXER_TORCH_K_NORM_ENV, "1")
        ):
            # TE LayerNorm can assert on saved-stat outputs in FP4 activation
            # recompute/StreamBP paths. Keep this fallback scoped to the DSA
            # indexer key norm; the HISA selector and selected-attention kernels
            # still use their fused implementations. The fallback is parameterized
            # with the TE module weights, so grad-enabled replay still trains it.
            k = _torch_layer_norm_like_te(self.k_norm, k, self.config.layernorm_epsilon)
        else:
            k = self.k_norm(k)
        _dsa_debug_sync("indexer key norm", k)
        if self._use_fused_indexer_flat_rope_writer(k, rotary_pos_emb, 1, self.index_head_dim):
            k = _DSAIndexerFlatRopeInplaceFunction.apply(
                k,
                rotary_pos_emb,
                1,
                self.index_head_dim,
                self.qk_pos_emb_head_dim,
                float(mscale),
                bool(self._indexer_rope_config.rotary_interleaved),
            )
            _dsa_debug_sync("indexer key fused rope", k)
        else:
            k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
            k = self._apply_rope(k, rotary_pos_emb, mscale)
            k = k.reshape(seqlen, bsz, self.index_head_dim)
            _dsa_debug_sync("indexer key torch rope", k)
        k = rotate_activation(k)
        _dsa_debug_sync("indexer key hadamard", k)
        _dsa_register_indexer_k_grad_debug("post_norm_rope_hadamard", debug_layer_number, k)
        if pad_len:
            k = k[:orig_seqlen]

        if apply_indexcache and self.indexcache_config is not None:
            from megatron.core.quantization.indexcache import apply_indexcache_kv

            k = apply_indexcache_kv(k, self.indexcache_config)
            _dsa_debug_sync("indexer key indexcache", k)
            _dsa_register_indexer_k_grad_debug("post_indexcache", debug_layer_number, k)
        return k

    def _apply_indexcache_to_key(self, k: torch.Tensor) -> torch.Tensor:
        if self.indexcache_config is None:
            return k
        from megatron.core.quantization.indexcache import apply_indexcache_kv

        return apply_indexcache_kv(k, self.indexcache_config)

    def _project_local_key_then_gather_before_topk(
        self,
        x: torch.Tensor,
        rotary_pos_emb,
        mscale: float,
        orig_seqlen: int,
        pad_len: int,
        apply_indexcache: bool = True,
    ) -> torch.Tensor:
        """Project the local SP hidden shard before gathering compact DSA index keys."""
        local_rank = self.pg_collection.tp.rank()
        local_start = local_rank * orig_seqlen
        local_rotary_pos_emb = self._slice_rotary_pos_emb(
            rotary_pos_emb, local_start, local_start + x.size(0)
        )
        k_local = self._project_key_before_topk(
            x,
            local_rotary_pos_emb,
            mscale,
            orig_seqlen,
            pad_len,
            apply_indexcache=False,
        )
        k = gather_from_sequence_parallel_region(k_local, group=self.pg_collection.tp)
        debug_layer_number = getattr(self, "layer_number", None)
        _dsa_register_indexer_k_grad_debug("post_sp_gather", debug_layer_number, k)
        if apply_indexcache:
            k = self._apply_indexcache_to_key(k)
            _dsa_register_indexer_k_grad_debug("post_sp_gather_indexcache", debug_layer_number, k)
        return k

    def _project_query_chunk_sp_owner_broadcast_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        rotary_pos_emb,
        mscale: float,
        q_start: int,
        q_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project one global query chunk from its SP owner, then broadcast compact tensors.

        The current training shape uses DSA chunks that divide the TP-local sequence
        shard. That lets us avoid gathering the full hidden state while preserving
        exact per-token projections and autograd via reduce-to-owner backward.
        """
        tp_group = self.pg_collection.tp
        tp_size = tp_group.size()
        local_seq_len = x.size(0)
        q_len = q_end - q_start
        if tp_size <= 1:
            return self._project_query_chunk_before_topk(
                x, qr, rotary_pos_emb, mscale, q_start, q_end
            )
        if local_seq_len <= 0:
            raise ValueError("DSA sequence-parallel query projection received empty local shard")

        owner_rank = q_start // local_seq_len
        if owner_rank >= tp_size or q_end > (owner_rank + 1) * local_seq_len:
            raise RuntimeError(
                "MEGATRON_DSA_SP_PROJECT_BEFORE_GATHER requires DSA chunks to stay within "
                f"one TP sequence shard; got q_start={q_start}, q_end={q_end}, "
                f"local_seq_len={local_seq_len}, tp_size={tp_size}"
            )

        local_offset = q_start - owner_rank * local_seq_len
        if tp_group.rank() == owner_rank:
            q, weights = self._project_query_before_topk(
                x[local_offset : local_offset + q_len],
                qr[local_offset : local_offset + q_len],
                self._slice_rotary_pos_emb(rotary_pos_emb, q_start, q_end),
                mscale,
            )
        else:
            q = torch.empty(
                q_len,
                x.size(1),
                self.index_n_heads,
                self.index_head_dim,
                device=x.device,
                dtype=x.dtype,
                requires_grad=True,
            )
            weights = torch.empty(
                q_len,
                x.size(1),
                self.index_n_heads,
                device=x.device,
                dtype=x.dtype,
                requires_grad=True,
            )

        owner_global_rank = torch.distributed.get_global_rank(tp_group, int(owner_rank))
        q = _BroadcastFromTensorParallelOwner.apply(q, owner_global_rank, tp_group)
        weights = _BroadcastFromTensorParallelOwner.apply(weights, owner_global_rank, tp_group)
        return q, weights

    def _project_query_positions_sp_owner_broadcast_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        rotary_pos_emb,
        mscale: float,
        query_positions: torch.Tensor,
        query_base: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project gathered-query rows from SP owners without gathering hidden states.

        StreamBP DSA replay passes query rows ordered by global sequence position,
        typically grouped as one local chunk per TP rank.  Projecting from the
        owning rank and broadcasting the compact indexer tensors preserves the
        trainable-indexer path while avoiding a full hidden/QR all-gather before
        HISA and selected-attention kernels run.
        """

        tp_group = self.pg_collection.tp
        tp_size = tp_group.size()
        local_seq_len = x.size(0)
        if tp_size <= 1:
            local_indices = query_positions.to(device=x.device, dtype=torch.long)
            q, weights = self._project_query_before_topk(
                x.index_select(0, local_indices),
                qr.index_select(0, local_indices)
                if qr.size(0) == x.size(0)
                else qr.index_select(0, local_indices - int(local_indices.min().item())),
                self._index_rotary_pos_emb(rotary_pos_emb, local_indices),
                mscale,
            )
            return q, weights
        if local_seq_len <= 0:
            raise ValueError("DSA StreamBP SP query projection received empty local shard")

        query_positions = query_positions.to(device=x.device, dtype=torch.long).contiguous()
        if query_positions.dim() != 1 or query_positions.numel() == 0:
            raise ValueError(
                "DSA StreamBP query positions must be a non-empty 1D tensor, got "
                f"{tuple(query_positions.shape)}"
            )
        max_position = local_seq_len * tp_size
        if bool((query_positions < 0).any().item()) or bool(
            (query_positions >= max_position).any().item()
        ):
            raise ValueError(
                "DSA StreamBP query positions exceed the TP-gathered prefix: "
                f"max_position={max_position}, positions_shape={tuple(query_positions.shape)}"
            )

        owners = torch.div(query_positions, local_seq_len, rounding_mode="floor")
        local_offsets = query_positions - owners * local_seq_len
        if query_base is None:
            query_base = int(local_offsets.min().item())
        else:
            query_base = int(query_base)

        def select_qr(offsets: torch.Tensor, segment_start: int, segment_end: int) -> torch.Tensor:
            if qr.size(0) == x.size(0):
                return qr.index_select(0, offsets)
            if qr.size(0) == query_positions.numel():
                row_indices = torch.arange(
                    segment_start,
                    segment_end,
                    device=qr.device,
                    dtype=torch.long,
                )
                return qr.index_select(0, row_indices)
            relative_offsets = offsets - query_base
            if bool((relative_offsets < 0).any().item()) or bool(
                (relative_offsets >= qr.size(0)).any().item()
            ):
                raise ValueError(
                    "Cannot align StreamBP DSA QR rows with local owner offsets: "
                    f"qr_len={qr.size(0)}, query_base={query_base}, "
                    f"offset_min={int(offsets.min().item())}, "
                    f"offset_max={int(offsets.max().item())}"
                )
            return qr.index_select(0, relative_offsets.to(device=qr.device))

        q_segments = []
        weight_segments = []
        start = 0
        num_positions = query_positions.numel()
        while start < num_positions:
            owner_rank = int(owners[start].item())
            end = start + 1
            while end < num_positions and int(owners[end].item()) == owner_rank:
                end += 1

            offsets = local_offsets[start:end]
            positions = query_positions[start:end]
            if tp_group.rank() == owner_rank:
                local_x = x.index_select(0, offsets.to(device=x.device))
                local_qr = select_qr(offsets, start, end)
                if positions.numel() > 1 and bool(
                    (positions[1:] != positions[:-1] + 1).any().item()
                ):
                    query_rotary_pos_emb = self._index_rotary_pos_emb(rotary_pos_emb, positions)
                else:
                    query_rotary_pos_emb = self._slice_rotary_pos_emb(
                        rotary_pos_emb,
                        int(positions[0].item()),
                        int(positions[-1].item()) + 1,
                    )
                q_segment, weights_segment = self._project_query_before_topk(
                    local_x,
                    local_qr,
                    query_rotary_pos_emb,
                    mscale,
                )
            else:
                segment_len = end - start
                q_segment = torch.empty(
                    segment_len,
                    x.size(1),
                    self.index_n_heads,
                    self.index_head_dim,
                    device=x.device,
                    dtype=x.dtype,
                    requires_grad=True,
                )
                weights_segment = torch.empty(
                    segment_len,
                    x.size(1),
                    self.index_n_heads,
                    device=x.device,
                    dtype=x.dtype,
                    requires_grad=True,
                )

            owner_global_rank = torch.distributed.get_global_rank(tp_group, owner_rank)
            q_segments.append(
                _BroadcastFromTensorParallelOwner.apply(
                    q_segment, owner_global_rank, tp_group
                )
            )
            weight_segments.append(
                _BroadcastFromTensorParallelOwner.apply(
                    weights_segment, owner_global_rank, tp_group
                )
            )
            start = end

        if len(q_segments) == 1:
            return q_segments[0], weight_segments[0]
        return torch.cat(q_segments, dim=0), torch.cat(weight_segments, dim=0)

    def forward_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
        query_indices: Optional[torch.Tensor] = None,
        apply_indexcache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All computations before topk."""
        x, qr, rotary_pos_emb, mscale, orig_seqlen, pad_len = self._prepare_inputs_before_topk(
            x, qr, packed_seq_params
        )
        query_x = x
        query_qr = qr
        query_rotary_pos_emb = rotary_pos_emb
        if query_indices is not None:
            if packed_seq_params is not None:
                raise ValueError("DSA query_indices are only supported for unpacked StreamBP chunks")
            query_indices = query_indices.to(device=x.device, dtype=torch.long)
            if query_indices.dim() != 1:
                raise ValueError(
                    f"DSA query_indices must be 1D, got shape {tuple(query_indices.shape)}"
                )
            if query_indices.numel() == 0:
                raise ValueError("DSA query_indices must not be empty")
            if bool((query_indices < 0).any().item()) or bool(
                (query_indices >= x.size(0)).any().item()
            ):
                raise ValueError(
                    f"DSA query_indices out of range for sequence length {x.size(0)}"
                )
            query_x = x.index_select(0, query_indices)
            if qr.size(0) == query_indices.numel():
                query_qr = qr
            else:
                query_qr = qr.index_select(0, query_indices)
            query_rotary_pos_emb = self._index_rotary_pos_emb(rotary_pos_emb, query_indices)

        q, weights = self._project_query_before_topk(
            query_x, query_qr, query_rotary_pos_emb, mscale
        )
        k = self._project_key_before_topk(
            x,
            rotary_pos_emb,
            mscale,
            orig_seqlen,
            pad_len,
            apply_indexcache=apply_indexcache,
        )
        if pad_len and query_indices is None:
            q = q[:orig_seqlen]
        if pad_len and query_indices is None:
            weights = weights[:orig_seqlen]

        return q, k, weights

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for DSA Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen].
            topk_indices: Top-k indices [batch, seqlen, index_topk].
        """
        assert packed_seq_params is None, "Packed sequence is not supported for DSAttention"

        # [seqlen, batch, index_n_heads * index_head_dim]
        # [seqlen, batch, index_head_dim]
        # [seqlen, batch, index_n_heads]
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        # [batch, seqlen, seqlen], [batch, seqlen, index_topk]
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, self.index_topk, mask)

        return index_scores, topk_indices

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass for DSA Indexer.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
        """
        _, topk_indices = self.forward_with_scores(x, qr, mask, packed_seq_params)
        return topk_indices


def unfused_dsa_fn(query, key, value, topk_indices, softmax_scale):
    """
    Unfused sparse attention implementation.
    """
    sq, b, np, hn = query.size()
    skv = key.size(0)
    hnv = value.size(3)

    # ===================================
    # Raw attention scores [b, np, sq, skv]
    # ===================================
    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [skv, b, np, hn] -> [b, np, hn, skv] -> [b * np, hn, skv]
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, skv)
    # Compute attention scores [b * np, sq, skv]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, skv]
    attention_scores = attention_scores.reshape(b, np, sq, skv)

    # ===================================
    # Apply sparse mask from indexer
    # ===================================
    # index_mask [b, sq, skv]
    index_mask = torch.full((b, sq, skv), float("-inf"), device=attention_scores.device)
    index_mask.scatter_(-1, topk_indices, 0)
    # causal_mask [sq, skv]
    causal_mask = torch.triu(
        torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=index_mask.device),
        diagonal=1,
    )
    # [b, sq, skv] + [1, sq, skv] -> [b, sq, skv]
    index_mask += causal_mask.view(1, sq, skv)
    # [b, np, sq, skv] + [b, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += index_mask.unsqueeze(1)
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)

    # ===================================
    # Output
    # ===================================
    # [skv, b, np, hnv] -> [b, np, skv, hnv] -> [b * np, skv, hnv]
    value = value.permute(1, 2, 0, 3).reshape(b * np, skv, hnv)
    # Reshape attention_scores: [b, np, sq, skv] -> [b * np, sq, skv]
    attention_scores = attention_scores.reshape(b * np, sq, skv)
    # Compute output: [b * np, sq, hnv]
    output = torch.bmm(attention_scores.to(value.dtype), value)
    # Reshape output: [b * np, sq, hnv] -> [b, np, sq, hnv] -> [sq, b, np, hnv]
    output = output.reshape(b, np, sq, hnv).permute(2, 0, 1, 3).contiguous()
    # Flatten: [sq, b, np, hnv] -> [sq, b, np * hnv]
    output = output.reshape(sq, b, np * hnv)
    return output


def _apply_dsa_score_mask(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    q_start: int,
    q_end: int,
    sk: int,
    is_causal: bool,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if is_causal:
        if query_positions is not None or key_positions is not None:
            if query_positions is None or key_positions is None:
                raise ValueError("query_positions and key_positions must be provided together")
            q_pos = query_positions.to(device=scores.device)
            k_pos = key_positions.to(device=scores.device)
            if q_pos.numel() != q_end - q_start:
                raise ValueError(
                    f"query_positions length {q_pos.numel()} does not match chunk length "
                    f"{q_end - q_start}"
                )
            if k_pos.numel() != sk:
                raise ValueError(f"key_positions length {k_pos.numel()} does not match sk {sk}")
        else:
            q_pos = torch.arange(q_start, q_end, device=scores.device)
            k_pos = torch.arange(sk, device=scores.device)
        if scores.dim() == 4:
            q_pos = q_pos.view(1, 1, -1, 1)
            k_pos = k_pos.view(1, 1, 1, -1)
        else:
            q_pos = q_pos.view(1, -1, 1)
            k_pos = k_pos.view(1, 1, -1)
        return scores.masked_fill(k_pos > q_pos, float("-inf"))
    if mask is None:
        return scores
    if mask.dim() == 2:
        return scores + mask[q_start:q_end, :].unsqueeze(0)
    if mask.dim() == 3:
        mask_slice = mask[:, q_start:q_end, :]
        if scores.dim() == 4:
            mask_slice = mask_slice.unsqueeze(1)
        return scores + mask_slice
    raise ValueError(f"DSA mask must be 2D or 3D, got shape {tuple(mask.shape)}")


def _streaming_qk_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk: int,
    mask: Optional[torch.Tensor],
    q_start: int,
    q_end: int,
    sk: int,
    is_causal: bool,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute exact DSA indexer top-k without materializing all per-head scores."""

    topk_k = min(topk, sk)
    key_block_size = _dsa_indexer_key_block_size(topk_k)
    running_scores = None
    running_indices = None
    candidate_scores_buffer = None
    candidate_indices_buffer = None

    with torch.no_grad():
        # The streaming path is used only when the DSA indexer loss is disabled,
        # so top-k selection should not build or retain an autograd graph.
        q_float = q.float()
        k_float = k if k.dtype == torch.float32 else k.float()
        weights_view = weights.unsqueeze(-1)
        if is_causal:
            if query_positions is not None or key_positions is not None:
                if query_positions is None or key_positions is None:
                    raise ValueError("query_positions and key_positions must be provided together")
                q_pos = query_positions.to(device=q.device).view(1, -1, 1)
            else:
                q_pos = torch.arange(q_start, q_end, device=q.device).view(1, -1, 1)

        for key_start in range(0, sk, key_block_size):
            key_end = min(key_start + key_block_size, sk)
            key_block = k_float[key_start:key_end]

            block_scores = torch.einsum("sbhd,tbd->sbht", q_float, key_block)
            block_scores = torch.relu_(block_scores)
            block_scores.mul_(weights_view)
            block_scores = block_scores.sum(dim=2).transpose(0, 1)

            if is_causal:
                if key_positions is not None:
                    k_pos = key_positions[key_start:key_end].to(device=block_scores.device).view(
                        1, 1, -1
                    )
                else:
                    k_pos = torch.arange(
                        key_start, key_end, device=block_scores.device
                    ).view(1, 1, -1)
                block_scores.masked_fill_(k_pos > q_pos, float("-inf"))
            elif mask is not None:
                if mask.dim() == 2:
                    block_scores.add_(
                        mask[q_start:q_end, key_start:key_end].unsqueeze(0)
                    )
                elif mask.dim() == 3:
                    block_scores.add_(mask[:, q_start:q_end, key_start:key_end])
                else:
                    raise ValueError(f"DSA mask must be 2D or 3D, got shape {tuple(mask.shape)}")

            block_topk = min(topk_k, key_end - key_start)
            block_scores, block_indices = block_scores.topk(
                block_topk, dim=-1, sorted=False
            )
            block_indices.add_(key_start)

            if running_scores is None:
                running_scores = block_scores
                running_indices = block_indices
            else:
                candidate_len = running_scores.size(-1) + block_scores.size(-1)
                if (
                    candidate_scores_buffer is None
                    or candidate_scores_buffer.shape[:-1] != running_scores.shape[:-1]
                    or candidate_scores_buffer.size(-1) < candidate_len
                ):
                    candidate_shape = (
                        *running_scores.shape[:-1],
                        topk_k + min(topk_k, key_block_size),
                    )
                    candidate_scores_buffer = torch.empty(
                        candidate_shape, device=running_scores.device, dtype=running_scores.dtype
                    )
                    candidate_indices_buffer = torch.empty(
                        candidate_shape, device=running_indices.device, dtype=running_indices.dtype
                    )
                candidate_scores = candidate_scores_buffer.narrow(-1, 0, candidate_len)
                candidate_indices = candidate_indices_buffer.narrow(-1, 0, candidate_len)
                torch.cat((running_scores, block_scores), dim=-1, out=candidate_scores)
                torch.cat((running_indices, block_indices), dim=-1, out=candidate_indices)
                keep_k = min(topk_k, candidate_len)
                running_scores, selected = candidate_scores.topk(
                    keep_k, dim=-1, sorted=False
                )
                running_indices = candidate_indices.gather(-1, selected)

    return running_indices


def _sparse_dsa_attention_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    mask: Optional[torch.Tensor],
    q_start: int,
    is_causal: bool,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q_len, bsz, num_heads, head_dim = query.size()
    value_head_dim = value.size(3)
    topk_k = topk_indices.size(-1)
    topk_block_size = 64
    batch_outputs = []

    for batch_idx in range(bsz):
        selected_positions = topk_indices[batch_idx]
        valid_selected = selected_positions >= 0
        gather_positions = selected_positions.clamp_min(0)
        query_batch = query[:, batch_idx]
        key_batch = key[:, batch_idx]
        score_blocks = []
        for topk_start in range(0, topk_k, topk_block_size):
            topk_end = min(topk_start + topk_block_size, topk_k)
            block_positions = gather_positions[:, topk_start:topk_end]
            block_index = block_positions.reshape(-1)
            key_block = key_batch.index_select(0, block_index)
            key_block = key_block.view(q_len, topk_end - topk_start, num_heads, head_dim)
            score_blocks.append(
                torch.einsum("qhd,qkhd->qkh", query_batch.float(), key_block.float())
            )
        attention_scores = torch.cat(score_blocks, dim=1) * softmax_scale

        if is_causal:
            if query_positions is not None or key_positions is not None:
                if query_positions is None or key_positions is None:
                    raise ValueError("query_positions and key_positions must be provided together")
                q_pos = query_positions.to(device=query.device).unsqueeze(1)
                selected_abs_positions = key_positions.to(device=query.device).index_select(
                    0, gather_positions.reshape(-1).long()
                )
                selected_abs_positions = selected_abs_positions.view_as(selected_positions)
                invalid = selected_abs_positions > q_pos
            else:
                q_pos = torch.arange(q_start, q_start + q_len, device=query.device).unsqueeze(1)
                invalid = selected_positions > q_pos
            attention_scores = attention_scores.masked_fill(invalid.unsqueeze(-1), float("-inf"))
        elif mask is not None:
            if mask.dim() == 2:
                selected_mask = mask[q_start : q_start + q_len, :].gather(1, gather_positions)
            elif mask.dim() == 3:
                selected_mask = mask[batch_idx, q_start : q_start + q_len, :].gather(
                    1, gather_positions
                )
            else:
                raise ValueError(f"DSA mask must be 2D or 3D, got shape {tuple(mask.shape)}")
            attention_scores = attention_scores + selected_mask.unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(
            (~valid_selected).unsqueeze(-1), float("-inf")
        )

        attention_probs = torch.softmax(attention_scores, dim=1, dtype=torch.float32).to(
            value.dtype
        )
        del attention_scores

        value_batch = value[:, batch_idx]
        output_batch = None
        for topk_start in range(0, topk_k, topk_block_size):
            topk_end = min(topk_start + topk_block_size, topk_k)
            block_positions = gather_positions[:, topk_start:topk_end]
            block_index = block_positions.reshape(-1)
            selected_value = value_batch.index_select(0, block_index)
            selected_value = selected_value.view(
                q_len, topk_end - topk_start, num_heads, value_head_dim
            )
            block_output = torch.einsum(
                "qkh,qkhd->qhd", attention_probs[:, topk_start:topk_end, :], selected_value
            )
            output_batch = block_output if output_batch is None else output_batch + block_output
        del attention_probs
        batch_outputs.append(output_batch)

    output = torch.stack(batch_outputs, dim=1)
    return output.reshape(q_len, bsz, num_heads * value_head_dim)


def _hisa_config_for_topk(config: IndexCacheHISAConfig, topk: int) -> IndexCacheHISAConfig:
    if config.topk_tokens == topk:
        return config
    return replace(config, topk_tokens=topk)


_HISA_CAUSAL_PREFIX_LENS_CACHE: dict[tuple[str, Optional[int], int, int, int], torch.Tensor] = {}


def _hisa_causal_prefix_lens_from_offsets(
    q_len: int,
    sk: int,
    *,
    q_start: int,
    device: torch.device,
) -> torch.Tensor:
    """Build causal prefix lengths without launching CUDA arange on the hot path."""

    q_len = int(q_len)
    sk = int(sk)
    q_start = int(q_start)
    if q_len <= 0:
        return torch.empty((0,), device=device, dtype=torch.int32)

    if device.type != "cuda":
        return torch.arange(q_start + 1, q_start + q_len + 1, device=device, dtype=torch.int32).clamp_(
            0, sk
        )

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
        device = torch.device("cuda", device_index)

    cache_key = (device.type, device_index, q_start, q_len, sk)
    cached = _HISA_CAUSAL_PREFIX_LENS_CACHE.get(cache_key)
    if cached is not None and cached.device == device:
        return cached

    # Some full-stack B200/packed-DSA runs hit cudaErrorInvalidValue in
    # torch.arange(..., device=cuda:N) on pipeline ranks 12-15. Materializing
    # this tiny metadata vector on CPU and copying it to the target device keeps
    # the exact causal-prefix semantics while avoiding that fragile CUDA arange
    # launch. Cache it because the same chunk offsets repeat every layer/step.
    host_prefix_lens = torch.arange(q_start + 1, q_start + q_len + 1, dtype=torch.int32).clamp_(
        0, sk
    )
    with torch.cuda.device(device):
        prefix_lens = host_prefix_lens.to(device=device, non_blocking=False)
    _HISA_CAUSAL_PREFIX_LENS_CACHE[cache_key] = prefix_lens
    return prefix_lens


def _hisa_prefix_lens_for_chunk(
    q_len: int,
    bsz: int,
    sk: int,
    *,
    q_start: int,
    is_causal: bool,
    mask: Optional[torch.Tensor],
    query_positions: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mask is not None:
        return None

    prefix_lens_is_final = False
    if is_causal:
        if query_positions is not None or key_positions is not None:
            if query_positions is None or key_positions is None:
                return None
            if query_positions.dim() != 1 or key_positions.dim() != 1:
                return None
            if query_positions.numel() != q_len or key_positions.numel() != sk:
                return None
            key_positions = key_positions.to(device=device, dtype=torch.long).contiguous()
            query_positions = query_positions.to(device=device, dtype=torch.long).contiguous()
            if (
                not _env_flag_enabled(_HISA_ASSUME_SORTED_POSITIONS_ENV, "1")
                and key_positions.numel() > 1
                and bool((key_positions[1:] < key_positions[:-1]).any().item())
            ):
                return None
            prefix_lens = torch.searchsorted(key_positions, query_positions, right=True)
        else:
            prefix_lens = _hisa_causal_prefix_lens_from_offsets(
                q_len, sk, q_start=q_start, device=device
            )
            prefix_lens_is_final = True
    else:
        prefix_lens = torch.full((q_len,), sk, device=device, dtype=torch.int32)
        prefix_lens_is_final = True

    if not prefix_lens_is_final:
        prefix_lens = prefix_lens.clamp(0, sk).to(torch.int32)

    if bsz == 1:
        return prefix_lens
    return prefix_lens.repeat(bsz)


def _flatten_hisa_indexer_rows(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_len, bsz, num_index_heads, index_head_dim = q.shape
    sk = k.shape[0]
    q_flat = q.permute(1, 0, 2, 3).reshape(bsz * q_len, num_index_heads, index_head_dim)
    weights_flat = weights.permute(1, 0, 2).reshape(bsz * q_len, num_index_heads)
    k_concat = k.permute(1, 0, 2).reshape(bsz * sk, k.shape[-1])
    k_offsets = torch.arange(
        0,
        (bsz + 1) * sk,
        sk,
        device=k.device,
        dtype=torch.long,
    )
    token_to_batch_idx = torch.arange(bsz, device=q.device, dtype=torch.long).repeat_interleave(
        q_len
    )
    return q_flat, weights_flat, k_concat, k_offsets, token_to_batch_idx


def _hisa_attention_target_probs(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    tp_group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    q_len, bsz, num_heads, head_dim = query.shape
    topk_k = topk_indices.shape[-1]
    if is_hisa_attention_target_probs_triton_supported(query, key, topk_indices):
        attention_probs = hisa_attention_target_probs_triton(
            query, key, topk_indices, float(softmax_scale)
        )
        if _dsa_process_group_size(tp_group) > 1:
            torch.distributed.all_reduce(attention_probs.contiguous(), group=tp_group)
        attention_probs = attention_probs / attention_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-20)
        return attention_probs

    row_chunk = _hisa_target_row_chunk(q_len)
    attention_probs = torch.empty(
        (bsz * q_len, topk_k),
        device=query.device,
        dtype=torch.float32,
    )

    for batch_idx in range(bsz):
        query_b = query[:, batch_idx].detach().float()
        key_b = key[:, batch_idx].detach().float()
        selected = topk_indices[batch_idx].to(torch.int32)
        valid = selected >= 0

        for row_start in range(0, q_len, row_chunk):
            row_end = min(row_start + row_chunk, q_len)
            selected_rows = selected[row_start:row_end]
            valid_rows = valid[row_start:row_end]
            safe_selected = selected_rows.clamp_min(0)
            rows = row_end - row_start
            selected_key = key_b.index_select(0, safe_selected.reshape(-1))
            selected_key = selected_key.view(rows, topk_k, num_heads, head_dim)
            scores = (
                torch.einsum(
                    "qhd,qkhd->qkh",
                    query_b[row_start:row_end],
                    selected_key,
                )
                * softmax_scale
            )
            scores = scores.masked_fill(~valid_rows.unsqueeze(-1), float("-inf"))
            probs = torch.softmax(scores, dim=1, dtype=torch.float32).sum(dim=2)
            out_start = batch_idx * q_len + row_start
            attention_probs[out_start : out_start + rows] = probs

    if _dsa_process_group_size(tp_group) > 1:
        torch.distributed.all_reduce(attention_probs.contiguous(), group=tp_group)
    attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True).clamp_min(
        1e-20
    )
    return attention_probs


class _HISAIndexerLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        prefix_lens: torch.Tensor,
        softmax_scale: float,
        topk: int,
        config: IndexCacheHISAConfig,
        tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len, bsz, _, _ = q.shape
        q_flat, weights_flat, k_concat, k_offsets, token_to_batch_idx = _flatten_hisa_indexer_rows(
            q, weights, k
        )
        config = _hisa_config_for_topk(config, int(topk))
        topk_flat, cache = hisa_selector_forward_and_save(
            q_flat,
            k_concat,
            k_offsets,
            weights_flat,
            prefix_lens,
            token_to_batch_idx,
            config,
        )
        if cache is None:
            raise RuntimeError("HISA indexer loss was called for a dense-fallback chunk.")

        topk_indices = topk_flat.reshape(bsz, q_len, int(topk)).to(torch.int32)
        candidate_scores = (
            torch.relu(cache.candidate_dot) * cache.weights.unsqueeze(1)
        ).sum(dim=-1)
        valid = cache.topk_positions >= 0
        selected_scores = candidate_scores.gather(1, cache.topk_positions.clamp_min(0))
        selected_scores = selected_scores.masked_fill(~valid, float("-inf"))

        attention_probs = _hisa_attention_target_probs(
            query, key, topk_indices, float(softmax_scale), tp_group
        )
        index_probs = torch.softmax(selected_scores, dim=-1, dtype=torch.float32)
        index_probs = torch.where(valid, index_probs, torch.zeros_like(index_probs))
        kl = attention_probs * (
            torch.log(attention_probs + 1e-10) - torch.log(index_probs + 1e-10)
        )
        loss_sum = kl.masked_fill(~valid, 0).sum()

        ctx.cache = cache
        ctx.config = config
        ctx.q_shape = tuple(q.shape)
        ctx.k_shape = tuple(k.shape)
        ctx.q_dtype = q.dtype
        ctx.weights_dtype = weights.dtype
        ctx.k_dtype = k.dtype
        ctx.save_for_backward(index_probs, attention_probs, valid)
        return topk_indices, loss_sum

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss_sum):
        index_probs, attention_probs, valid = ctx.saved_tensors
        cache = ctx.cache
        grad_topk_logit = (index_probs - attention_probs) * grad_loss_sum
        grad_topk_logit = grad_topk_logit.masked_fill(~valid, 0)
        grad_candidate_scores, grad_block_scores = hisa_unselect_grad_from_topk_to_scores(
            grad_topk_logit, cache
        )
        grad_q_flat, grad_k_by_batch, grad_weights_flat = apply_hisa_score_backward(
            grad_candidate_scores,
            grad_block_scores,
            cache,
            config=ctx.config,
        )

        q_len, bsz, num_index_heads, index_head_dim = ctx.q_shape
        sk, _, index_k_dim = ctx.k_shape
        grad_q = (
            grad_q_flat.view(bsz, q_len, num_index_heads, index_head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
            .to(ctx.q_dtype)
        )
        grad_weights = (
            grad_weights_flat.view(bsz, q_len, num_index_heads)
            .permute(1, 0, 2)
            .contiguous()
            .to(ctx.weights_dtype)
        )
        grad_k = torch.stack(grad_k_by_batch, dim=1).reshape(sk, bsz, index_k_dim).to(ctx.k_dtype)

        return (
            grad_q,
            grad_weights,
            grad_k,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _try_load_hisa_cuda_ext():
    try:
        from megatron.core.extensions.hisa_indexer.kernels.build import get_ext

        return get_ext()
    except Exception:
        return None


def _hisa_fused_indexer_loss_enabled() -> bool:
    return _env_flag_enabled(_HISA_FUSED_INDEXER_LOSS_ENV, "1")


def _hisa_dsa_fused_forward_enabled() -> bool:
    if _env_flag_enabled(_HISA_DSA_FUSED_FORWARD_ENV, "0"):
        raise RuntimeError(
            f"{_HISA_DSA_FUSED_FORWARD_ENV}=1 requested, but the experimental full "
            "HISA+DSA selected-attention fused forward path is disabled for production. "
            "Whole-path benchmarks were slower than the HISA selector megakernel plus "
            "the existing selected-attention path."
        )
    return False


def _hisa_dsa_persistent_forward_enabled() -> bool:
    if _env_flag_enabled(_HISA_DSA_PERSISTENT_FORWARD_ENV, "0"):
        raise RuntimeError(
            f"{_HISA_DSA_PERSISTENT_FORWARD_ENV}=1 requested, but this experimental "
            "persistent HISA/DSA forward path was removed from the production extension "
            "build after whole-path benchmarks showed it was slower than the existing "
            "HISA selector plus selected-attention path."
        )
    return False


def _hisa_fused_effective_block_topk(sk: int, config: IndexCacheHISAConfig) -> int:
    block_size = int(config.block_size)
    block_count = int(math.ceil(int(sk) / block_size))
    if block_count <= 0:
        return 0
    if float(config.compression_ratio) > 0:
        ratio_f = float(config.compression_ratio)
        if abs(ratio_f - round(ratio_f)) < 1e-6:
            ratio_i = int(round(ratio_f))
            effective = (block_count + ratio_i - 1) // ratio_i
        else:
            effective = int(math.ceil(block_count / ratio_f))
        effective = max(1, min(int(effective), block_count))
    else:
        effective = min(int(config.block_topk), block_count)
    forced = tuple(config.forced_boundary_blocks or ())
    if forced:
        forced_static = 0
        if "first" in forced and block_count >= 1:
            forced_static += 1
        if "last" in forced and block_count >= 1:
            forced_static += 1
            if "first" in forced:
                forced_static -= int(block_count == 1)
        if "last_minus_one" in forced and block_count >= 2:
            forced_static += 1
            if "first" in forced:
                forced_static -= int(block_count == 2)
        effective = max(int(effective), int(forced_static))
    return max(1, min(int(effective), block_count))


def _hisa_selected_score_backward_cuda(
    grad_scores: torch.Tensor,
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    topk_indices_i32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _try_load_hisa_cuda_ext()
    if ext is None or not hasattr(ext, "hisa_selected_score_bwd"):
        raise RuntimeError("HISA selected-score backward CUDA extension is unavailable")

    with fine_profile_range("hisa.selected_score_bwd.single.prepare"):
        q_f = q_rows.contiguous().float()
        weights_f = weights_rows.contiguous().float()
        k_f = k_rows.contiguous().float()
        topk_i32 = topk_indices_i32.contiguous().to(torch.int32)
        grad_f = grad_scores.contiguous().float()
        grad_q = torch.zeros_like(q_f)
        grad_k = torch.zeros_like(k_f)
        grad_w = torch.zeros_like(weights_f)
    with fine_profile_range("hisa.selected_score_bwd.single.cuda"):
        ext.hisa_selected_score_bwd(
            grad_f,
            q_f,
            k_f,
            weights_f,
            topk_i32,
            grad_q,
            grad_k,
            grad_w,
        )
    return grad_q, grad_w, grad_k


def _hisa_selected_score_backward_cuda_batched(
    grad_scores: torch.Tensor,
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk_indices_i32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run selected-score backward for a full microbatch in one CUDA call.

    The underlying CUDA kernel accepts one contiguous key table. For MBS>1 we
    concatenate the per-batch indexer K rows and add a per-batch offset to
    selected top-k indices. This keeps the math identical to one call per batch
    item while avoiding repeated extension launches in the trainable HISA path.
    """

    q_len, bsz, num_heads, head_dim = q.shape
    sk = k.shape[0]
    topk_k = topk_indices_i32.shape[-1]
    ext = _try_load_hisa_cuda_ext()
    use_batched_cuda = _env_flag_enabled(_HISA_SELECTED_SCORE_BWD_BATCHED_CUDA_ENV, "1")
    if (
        use_batched_cuda
        and
        ext is not None
        and hasattr(ext, "hisa_selected_score_bwd_batched")
        and num_heads == 64
        and head_dim == 128
        and q.dtype == k.dtype
        and q.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and weights.dtype in (torch.float32, torch.bfloat16, torch.float16)
    ):
        with fine_profile_range("hisa.selected_score_bwd.batched.prepare"):
            q_c = q.contiguous()
            weights_c = weights.contiguous()
            k_c = k.contiguous()
            topk_i32 = topk_indices_i32.contiguous().to(torch.int32)
            grad_f = grad_scores.reshape(bsz * q_len, topk_k).contiguous().float()
            grad_q = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
            grad_k = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
            grad_w = torch.zeros(weights.shape, device=weights.device, dtype=torch.float32)
        with fine_profile_range("hisa.selected_score_bwd.batched.cuda"):
            ext.hisa_selected_score_bwd_batched(
                grad_f,
                q_c,
                k_c,
                weights_c,
                topk_i32,
                grad_q,
                grad_k,
                grad_w,
            )
        return grad_q, grad_w, grad_k

    with fine_profile_range("hisa.selected_score_bwd.batched.fallback_prepare"):
        q_flat = q.permute(1, 0, 2, 3).reshape(bsz * q_len, num_heads, head_dim)
        weights_flat = weights.permute(1, 0, 2).reshape(bsz * q_len, num_heads)
        k_concat = k.permute(1, 0, 2).reshape(bsz * sk, k.shape[-1])
        topk_batched = topk_indices_i32.contiguous().to(torch.int32)
        batch_offsets = (
            torch.arange(bsz, device=topk_batched.device, dtype=torch.int32).view(bsz, 1, 1)
            * int(sk)
        )
        topk_global = torch.where(topk_batched >= 0, topk_batched + batch_offsets, topk_batched)
    grad_q_flat, grad_w_flat, grad_k_concat = _hisa_selected_score_backward_cuda(
        grad_scores.reshape(bsz * q_len, topk_k),
        q_flat,
        weights_flat,
        k_concat,
        topk_global.reshape(bsz * q_len, topk_k),
    )
    with fine_profile_range("hisa.selected_score_bwd.batched.fallback_finalize"):
        grad_q = grad_q_flat.reshape(bsz, q_len, num_heads, head_dim).permute(1, 0, 2, 3)
        grad_weights = grad_w_flat.reshape(bsz, q_len, num_heads).permute(1, 0, 2)
        grad_k = grad_k_concat.reshape(bsz, sk, k.shape[-1]).permute(1, 0, 2)
    return grad_q.contiguous(), grad_weights.contiguous(), grad_k.contiguous()


class _HISASelectWithScores(torch.autograd.Function):
    """HISA selector with custom selected-logit gradients.

    The selector/top-k itself is non-differentiable. The selected logits remain
    differentiable via ``_hisa_selected_score_backward_cuda`` so the indexer can
    train without keeping the BMM/gather selector graph alive through StreamBP.
    """

    @staticmethod
    def forward(
        ctx,
        q_rows: torch.Tensor,
        weights_rows: torch.Tensor,
        k_rows: torch.Tensor,
        prefix_lens: torch.Tensor,
        topk: int,
        config: IndexCacheHISAConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _HISA_INDEXER_LOSS_DEBUG_CALLS
        _HISA_INDEXER_LOSS_DEBUG_CALLS += 1
        debug_call_id = _HISA_INDEXER_LOSS_DEBUG_CALLS
        if q_rows.size(-1) != 128 or k_rows.size(-1) != 128:
            raise RuntimeError(
                "HISA selected-score autograd path cannot handle this chunk: "
                f"selected-score backward requires head_dim=128, got q_dim={q_rows.size(-1)} "
                f"k_dim={k_rows.size(-1)}"
            )
        result = indexcache_hisa_select_with_scores(
            q_rows,
            weights_rows,
            k_rows,
            int(topk),
            config=config,
            prefix_lens=prefix_lens,
        )
        if result is None:
            reason = describe_indexcache_hisa_select_with_scores(
                q_rows,
                weights_rows,
                k_rows,
                int(topk),
                config=config,
                prefix_lens=prefix_lens,
            )
            raise RuntimeError(
                "HISA selected-score autograd path cannot handle this chunk: " + reason
            )
        topk_i32, selected_scores = result
        topk_compact = _maybe_compact_hisa_saved_topk_indices(topk_i32, k_rows.shape[0])
        selected_scores = selected_scores.contiguous().float()
        ctx.mark_non_differentiable(topk_compact)
        _log_hisa_indexer_loss_debug(
            "select_scores.single.forward",
            debug_call_id,
            selected_scores=selected_scores,
            topk=topk_i32,
        )
        ctx.save_for_backward(
            q_rows,
            weights_rows,
            k_rows,
            topk_compact,
        )
        ctx.debug_call_id = debug_call_id
        return topk_compact, selected_scores

    @staticmethod
    def backward(ctx, grad_topk: torch.Tensor | None, grad_selected_scores: torch.Tensor | None):
        q_rows, weights_rows, k_rows, topk_i32 = ctx.saved_tensors
        if grad_selected_scores is None:
            return (
                torch.zeros_like(q_rows),
                torch.zeros_like(weights_rows),
                torch.zeros_like(k_rows),
                None,
                None,
                None,
            )
        _log_hisa_indexer_loss_debug(
            "select_scores.single.backward_input",
            ctx.debug_call_id,
            grad_selected_scores=grad_selected_scores,
        )
        grad_q, grad_w, grad_k = _hisa_selected_score_backward_cuda(
            grad_selected_scores,
            q_rows,
            weights_rows,
            k_rows,
            topk_i32,
        )
        _log_hisa_indexer_loss_debug(
            "select_scores.single.backward_output",
            ctx.debug_call_id,
            grad_q=grad_q,
            grad_w=grad_w,
            grad_k=grad_k,
        )
        grad_q = grad_q.to(q_rows.dtype)
        grad_w = grad_w.to(weights_rows.dtype)
        grad_k = grad_k.to(k_rows.dtype)
        _log_hisa_indexer_loss_debug(
            "select_scores.single.backward_return",
            ctx.debug_call_id,
            grad_q=grad_q,
            grad_w=grad_w,
            grad_k=grad_k,
        )
        return (
            grad_q,
            grad_w,
            grad_k,
            None,
            None,
            None,
        )


class _HISASelectWithScoresBatched(torch.autograd.Function):
    """Batched HISA selector with one selected-score CUDA backward call."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_reps: Optional[torch.Tensor],
        topk: int,
        config: IndexCacheHISAConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _HISA_INDEXER_LOSS_DEBUG_CALLS
        _HISA_INDEXER_LOSS_DEBUG_CALLS += 1
        debug_call_id = _HISA_INDEXER_LOSS_DEBUG_CALLS
        if q.size(-1) != 128 or k.size(-1) != 128:
            raise RuntimeError(
                "HISA selected-score autograd path cannot handle this chunk: "
                f"selected-score backward requires head_dim=128, got q_dim={q.size(-1)} "
                f"k_dim={k.size(-1)}"
            )
        q_len, bsz, _, _ = q.shape
        if indexcache_hisa_selector_backend_name() == "megakernel":
            result = indexcache_hisa_megakernel_batched_select_with_scores(
                q,
                weights,
                k,
                int(topk),
                config=config,
                prefix_lens=prefix_lens,
                block_reps_precomputed=block_reps,
            )
            if result is None:
                reason = describe_indexcache_hisa_select_with_scores(
                    q[:, 0],
                    weights[:, 0],
                    k[:, 0],
                    int(topk),
                    config=config,
                    prefix_lens=(
                        prefix_lens
                        if prefix_lens.numel() == q_len
                        else prefix_lens[:q_len]
                    ),
                )
                raise RuntimeError(
                    "HISA selected-score autograd path cannot handle this chunk: " + reason
                )
            topk_i32, selected_scores = result
            topk_compact = _maybe_compact_hisa_saved_topk_indices(topk_i32, k.shape[0])
            ctx.mark_non_differentiable(topk_compact)
            ctx.save_for_backward(
                q,
                weights,
                k,
                topk_compact,
            )
            selected_scores = selected_scores.contiguous().float()
            _log_hisa_indexer_loss_debug(
                "select_scores.batched.forward",
                debug_call_id,
                selected_scores=selected_scores,
                topk=topk_i32,
            )
            ctx.debug_call_id = debug_call_id
            return topk_compact, selected_scores

        topk_batches = []
        selected_score_batches = []
        for batch_idx in range(bsz):
            if prefix_lens.numel() == q_len:
                prefix_b = prefix_lens
            else:
                prefix_b = prefix_lens[batch_idx * q_len : (batch_idx + 1) * q_len]
            result = indexcache_hisa_select_with_scores(
                q[:, batch_idx],
                weights[:, batch_idx],
                k[:, batch_idx],
                int(topk),
                config=config,
                prefix_lens=prefix_b,
            )
            if result is None:
                reason = describe_indexcache_hisa_select_with_scores(
                    q[:, batch_idx],
                    weights[:, batch_idx],
                    k[:, batch_idx],
                    int(topk),
                    config=config,
                    prefix_lens=prefix_b,
                )
                raise RuntimeError(
                    "HISA selected-score autograd path cannot handle this chunk: " + reason
                )
            topk_i32, selected_scores = result
            topk_batches.append(topk_i32.contiguous().to(torch.int32))
            selected_score_batches.append(selected_scores.contiguous().float())

        topk_i32 = torch.stack(topk_batches, dim=0).contiguous()
        selected_scores = torch.stack(selected_score_batches, dim=0).reshape(
            bsz * q_len, int(topk)
        )
        topk_compact = _maybe_compact_hisa_saved_topk_indices(topk_i32, k.shape[0])
        ctx.mark_non_differentiable(topk_compact)
        ctx.save_for_backward(
            q,
            weights,
            k,
            topk_compact,
        )
        _log_hisa_indexer_loss_debug(
            "select_scores.batched.forward",
            debug_call_id,
            selected_scores=selected_scores,
            topk=topk_i32,
        )
        ctx.debug_call_id = debug_call_id
        return topk_compact, selected_scores

    @staticmethod
    def backward(ctx, grad_topk: torch.Tensor | None, grad_selected_scores: torch.Tensor | None):
        q, weights, k, topk_i32 = ctx.saved_tensors
        if grad_selected_scores is None:
            return (
                torch.zeros_like(q),
                torch.zeros_like(weights),
                torch.zeros_like(k),
                None,
                None,
                None,
            )
        _log_hisa_indexer_loss_debug(
            "select_scores.batched.backward_input",
            ctx.debug_call_id,
            grad_selected_scores=grad_selected_scores,
        )
        grad_q, grad_w, grad_k = _hisa_selected_score_backward_cuda_batched(
            grad_selected_scores,
            q,
            weights,
            k,
            topk_i32,
        )
        _log_hisa_indexer_loss_debug(
            "select_scores.batched.backward_output",
            ctx.debug_call_id,
            grad_q=grad_q,
            grad_w=grad_w,
            grad_k=grad_k,
        )
        grad_q = grad_q.to(q.dtype)
        grad_w = grad_w.to(weights.dtype)
        grad_k = grad_k.to(k.dtype)
        _log_hisa_indexer_loss_debug(
            "select_scores.batched.backward_return",
            ctx.debug_call_id,
            grad_q=grad_q,
            grad_w=grad_w,
            grad_k=grad_k,
        )
        return (
            grad_q,
            grad_w,
            grad_k,
            None,
            None,
            None,
            None,
        )


class _HISADSASplitQKFusedForward(torch.autograd.Function):
    """Single-launch local HISA selector + split-QK selected attention forward."""

    @staticmethod
    def forward(
        ctx,
        q_indexer: torch.Tensor,
        weights: torch.Tensor,
        k_indexer: torch.Tensor,
        block_reps: torch.Tensor,
        prefix_lens: torch.Tensor,
        query_nope: torch.Tensor,
        query_pe: torch.Tensor,
        key_nope: torch.Tensor,
        key_pe: torch.Tensor,
        value: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        softmax_scale: float,
        q_start: int,
        block_size: int,
        block_topk: int,
        compression_ratio: float,
        effective_block_topk: int,
        topk_count: int,
        has_positions: bool,
        force_first: bool,
        force_last: bool,
        force_last_minus_one: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global _HISA_INDEXER_LOSS_DEBUG_CALLS
        _HISA_INDEXER_LOSS_DEBUG_CALLS += 1
        debug_call_id = _HISA_INDEXER_LOSS_DEBUG_CALLS
        q_pos_arg = query_positions if has_positions else None
        k_pos_arg = key_positions if has_positions else None
        fused_forward_impl = (
            _hisa_dsa_split_qk_persistent_forward_cuda
            if _hisa_dsa_persistent_forward_enabled()
            else _hisa_dsa_split_qk_fused_forward_cuda
        )
        (
            output,
            topk_indices,
            selected_scores,
            teacher_probs,
            lse,
        ) = fused_forward_impl(
            q_indexer,
            weights,
            k_indexer,
            block_reps,
            prefix_lens,
            query_nope,
            query_pe,
            key_nope,
            key_pe,
            value,
            q_pos_arg,
            k_pos_arg,
            float(softmax_scale),
            int(q_start),
            int(block_size),
            int(block_topk),
            float(compression_ratio),
            int(effective_block_topk),
            int(topk_count),
            bool(force_first),
            bool(force_last),
            bool(force_last_minus_one),
        )
        query_nope_c = query_nope if query_nope.stride(-1) == 1 else query_nope.contiguous()
        query_pe_c = query_pe.contiguous()
        key_nope_c = key_nope if key_nope.stride(-1) == 1 else key_nope.contiguous()
        key_pe_c = key_pe.contiguous()
        value_c = value if value.stride(-1) == 1 else value.contiguous()
        ctx.mark_non_differentiable(teacher_probs, topk_indices)
        ctx.save_for_backward(
            q_indexer,
            weights,
            k_indexer,
            query_nope_c,
            query_pe_c,
            key_nope_c,
            key_pe_c,
            value_c,
            topk_indices,
            query_positions,
            key_positions,
            output,
            lse,
        )
        ctx.softmax_scale = float(softmax_scale)
        ctx.q_start = int(q_start)
        ctx.has_positions = bool(has_positions)
        ctx.debug_call_id = debug_call_id
        _log_hisa_indexer_loss_debug(
            "hisa_dsa_fused.forward",
            debug_call_id,
            selected_scores=selected_scores,
            teacher_probs=teacher_probs,
            topk=topk_indices,
            output=output,
        )
        q_len, bsz, num_heads, value_dim = output.shape
        return (
            output.reshape(q_len, bsz, num_heads * value_dim),
            selected_scores.contiguous().float(),
            teacher_probs,
            topk_indices,
        )

    @staticmethod
    def backward(
        ctx,
        grad_output: Optional[torch.Tensor],
        grad_selected_scores: Optional[torch.Tensor],
        grad_teacher_probs: Optional[torch.Tensor] = None,
        grad_topk: Optional[torch.Tensor] = None,
    ):
        (
            q_indexer,
            weights,
            k_indexer,
            query_nope,
            query_pe,
            key_nope,
            key_pe,
            value,
            topk_indices,
            query_positions,
            key_positions,
            output,
            lse,
        ) = ctx.saved_tensors
        q_len, bsz, num_heads, value_dim = output.shape
        if grad_output is None:
            grad_output_4d = torch.zeros_like(output)
        else:
            grad_output_4d = grad_output.reshape(q_len, bsz, num_heads, value_dim).contiguous()
        _log_hisa_indexer_loss_debug(
            "hisa_dsa_fused.backward_input",
            ctx.debug_call_id,
            grad_output=grad_output_4d,
            grad_selected_scores=grad_selected_scores,
        )

        grad_dtype = (
            query_nope.dtype
            if _bf16_grad_atomics_enabled()
            and query_nope.dtype in (torch.bfloat16, torch.float16)
            else torch.float32
        )
        grad_query_nope = torch.empty_like(query_nope, dtype=grad_dtype)
        grad_query_pe = torch.empty_like(query_pe, dtype=grad_dtype)
        grad_key_nope = torch.zeros_like(key_nope, dtype=grad_dtype)
        grad_key_pe = torch.zeros_like(key_pe, dtype=grad_dtype)
        grad_value = torch.zeros_like(value, dtype=grad_dtype)
        with fine_profile_range("dsa.split_qk.backward.row_cuda"):
            _dsa_split_qk_backward_row_cuda(
                query_nope,
                query_pe,
                key_nope,
                key_pe,
                value,
                topk_indices,
                query_positions if ctx.has_positions else None,
                key_positions if ctx.has_positions else None,
                output,
                lse,
                grad_output_4d,
                grad_query_nope,
                grad_query_pe,
                grad_key_nope,
                grad_key_pe,
                grad_value,
                float(ctx.softmax_scale),
                int(ctx.q_start),
                0,
                key_nope.shape[0],
                True,
                True,
                True,
                True,
            )

        if grad_selected_scores is None:
            grad_q_indexer = torch.zeros_like(q_indexer)
            grad_weights = torch.zeros_like(weights)
            grad_k_indexer = torch.zeros_like(k_indexer)
        else:
            grad_q_indexer, grad_weights, grad_k_indexer = _hisa_selected_score_backward_cuda_batched(
                grad_selected_scores,
                q_indexer,
                weights,
                k_indexer,
                topk_indices,
            )
        _log_hisa_indexer_loss_debug(
            "hisa_dsa_fused.backward_output",
            ctx.debug_call_id,
            grad_q_indexer=grad_q_indexer,
            grad_weights=grad_weights,
            grad_k_indexer=grad_k_indexer,
            grad_query_nope=grad_query_nope,
            grad_query_pe=grad_query_pe,
            grad_key_nope=grad_key_nope,
            grad_key_pe=grad_key_pe,
            grad_value=grad_value,
        )
        grad_q_indexer = grad_q_indexer.to(q_indexer.dtype)
        grad_weights = grad_weights.to(weights.dtype)
        grad_k_indexer = grad_k_indexer.to(k_indexer.dtype)
        grad_query_nope = grad_query_nope.to(query_nope.dtype)
        grad_query_pe = grad_query_pe.to(query_pe.dtype)
        grad_key_nope = grad_key_nope.to(key_nope.dtype)
        grad_key_pe = grad_key_pe.to(key_pe.dtype)
        grad_value = grad_value.to(value.dtype)
        _log_hisa_indexer_loss_debug(
            "hisa_dsa_fused.backward_return",
            ctx.debug_call_id,
            grad_q_indexer=grad_q_indexer,
            grad_weights=grad_weights,
            grad_k_indexer=grad_k_indexer,
            grad_query_nope=grad_query_nope,
            grad_query_pe=grad_query_pe,
            grad_key_nope=grad_key_nope,
            grad_key_pe=grad_key_pe,
            grad_value=grad_value,
        )

        return (
            grad_q_indexer,
            grad_weights,
            grad_k_indexer,
            None,
            None,
            grad_query_nope,
            grad_query_pe,
            grad_key_nope,
            grad_key_pe,
            grad_value,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _SelectedScoresKLLoss(torch.autograd.Function):
    """KL loss with a compact custom gradient for selected indexer logits."""

    @staticmethod
    def forward(
        ctx,
        selected_scores: torch.Tensor,
        teacher_probs: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        global _HISA_INDEXER_LOSS_DEBUG_CALLS
        _HISA_INDEXER_LOSS_DEBUG_CALLS += 1
        debug_call_id = _HISA_INDEXER_LOSS_DEBUG_CALLS
        selected_scores_f = selected_scores.float()
        teacher_probs_f = teacher_probs.float()
        valid = valid.to(device=selected_scores.device, dtype=torch.bool)
        _log_hisa_indexer_loss_debug(
            "selected_kl.forward_input",
            debug_call_id,
            selected_scores=selected_scores_f,
            teacher_probs=teacher_probs_f,
            valid=valid,
        )
        teacher_probs_f = torch.where(
            torch.isfinite(teacher_probs_f) & (teacher_probs_f >= 0.0),
            teacher_probs_f,
            torch.zeros_like(teacher_probs_f),
        )
        masked_scores = selected_scores_f.masked_fill(~valid, float("-inf"))
        teacher_probs_f = teacher_probs_f.masked_fill(~valid, 0.0)
        if is_hisa_kl_grad_triton_supported(masked_scores, teacher_probs_f):
            loss_sum, grad_selected_scores = hisa_kl_loss_and_grad_triton(
                masked_scores.contiguous(), teacher_probs_f.contiguous()
            )
        else:
            has_valid = valid.any(dim=-1, keepdim=True)
            safe_scores = torch.where(has_valid, masked_scores, torch.zeros_like(masked_scores))
            index_probs = torch.softmax(safe_scores, dim=-1, dtype=torch.float32)
            index_probs = torch.where(valid & has_valid, index_probs, torch.zeros_like(index_probs))
            teacher_probs_f = torch.where(
                valid & has_valid,
                teacher_probs_f,
                torch.zeros_like(teacher_probs_f),
            )
            teacher_sum = teacher_probs_f.sum(dim=-1, keepdim=True)
            teacher_probs_f = torch.where(
                teacher_sum > 0.0,
                teacher_probs_f / teacher_sum.clamp_min(1e-20),
                torch.zeros_like(teacher_probs_f),
            )
            kl = teacher_probs_f * (
                torch.log(teacher_probs_f + 1e-10) - torch.log(index_probs + 1e-10)
            )
            loss_sum = kl.masked_fill(~valid, 0).sum()
            prob_ratio = index_probs / (index_probs + 1e-10)
            teacher_prob_ratio = teacher_probs_f * prob_ratio
            row_scale = teacher_prob_ratio.sum(dim=-1, keepdim=True)
            grad_selected_scores = (
                index_probs * row_scale - teacher_prob_ratio
            ).masked_fill(~valid, 0)
        _log_hisa_indexer_loss_debug(
            "selected_kl.forward_output",
            debug_call_id,
            loss_sum=loss_sum,
            grad_selected_scores=grad_selected_scores,
        )
        ctx.save_for_backward(grad_selected_scores.to(selected_scores.dtype))
        ctx.debug_call_id = debug_call_id
        return loss_sum

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (grad_selected_scores,) = ctx.saved_tensors
        grad = grad_selected_scores
        if grad_output is not None:
            grad = grad * grad_output.to(dtype=grad.dtype)
        _log_hisa_indexer_loss_debug(
            "selected_kl.backward",
            ctx.debug_call_id,
            grad_output=grad_output,
            grad_selected_scores=grad,
        )
        return grad, None, None


class _HISAFusedIndexerLoss(torch.autograd.Function):
    """Fused HISA indexer-loss path.

    Forward:
      - CUDA HISA selector emits top-k indices and selected indexer logits.
      - Triton selected-attention teacher emits compact teacher probabilities.
      - Triton KL kernel emits the scalar loss and compact dL/d(selected_logits).

    Backward:
      - CUDA selected-score backward scatters gradients into indexer q/k/weights.

    The HISA top-k selection remains non-differentiable, matching the existing
    trainable indexer path.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        prefix_lens: torch.Tensor,
        softmax_scale: float,
        topk: int,
        config: IndexCacheHISAConfig,
        tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _HISA_INDEXER_LOSS_DEBUG_CALLS
        _HISA_INDEXER_LOSS_DEBUG_CALLS += 1
        debug_call_id = _HISA_INDEXER_LOSS_DEBUG_CALLS
        q_len, bsz, _, _ = q.shape
        topk_k = min(int(topk), int(k.shape[0]))
        config = _hisa_config_for_topk(config, topk_k)

        topk_batches = []
        selected_score_batches = []
        teacher_prob_batches = []
        for batch_idx in range(bsz):
            if prefix_lens.numel() == q_len:
                prefix_b = prefix_lens
            else:
                prefix_b = prefix_lens[batch_idx * q_len : (batch_idx + 1) * q_len]
            query_b = query[:, batch_idx : batch_idx + 1]
            key_b = key[:, batch_idx : batch_idx + 1]
            selector_result = indexcache_hisa_select_with_scores(
                q[:, batch_idx],
                weights[:, batch_idx],
                k[:, batch_idx],
                topk_k,
                config=config,
                prefix_lens=prefix_b,
            )
            if selector_result is None:
                reason = describe_indexcache_hisa_select_with_scores(
                    q[:, batch_idx],
                    weights[:, batch_idx],
                    k[:, batch_idx],
                    topk_k,
                    config=config,
                    prefix_lens=prefix_b,
                )
                raise RuntimeError(
                    "fused HISA indexer loss cannot produce selected scores without "
                    f"dense fallback: {reason}"
                )
            topk_i32, selected_scores = selector_result
            if not is_hisa_attention_target_probs_triton_supported(
                query_b, key_b, topk_i32.unsqueeze(0)
            ):
                raise RuntimeError(
                    "fused HISA indexer loss requires compact Triton teacher; "
                    f"unsupported shapes query={tuple(query_b.shape)} key={tuple(key_b.shape)} "
                    f"topk={tuple(topk_i32.unsqueeze(0).shape)}"
                )
            teacher_probs = hisa_attention_target_probs_triton(
                query_b,
                key_b,
                topk_i32.unsqueeze(0),
                float(softmax_scale),
            )
            topk_batches.append(topk_i32)
            selected_score_batches.append(selected_scores)
            teacher_prob_batches.append(teacher_probs)

        topk_i32 = torch.stack(topk_batches, dim=0).contiguous()
        selected_scores = torch.stack(selected_score_batches, dim=0).reshape(
            bsz * q_len, topk_k
        )
        teacher_probs = torch.stack(teacher_prob_batches, dim=0).reshape(bsz * q_len, topk_k)
        topk_compact = _maybe_compact_hisa_saved_topk_indices(topk_i32, k.shape[0])
        ctx.mark_non_differentiable(topk_compact)
        _log_hisa_indexer_loss_debug(
            "forward_selected",
            debug_call_id,
            selected_scores=selected_scores,
            teacher_probs_pre_reduce=teacher_probs,
            topk=topk_i32,
        )

        if _dsa_process_group_size(tp_group) > 1:
            torch.distributed.all_reduce(teacher_probs.contiguous(), group=tp_group)
        teacher_row_sum = teacher_probs.sum(dim=-1, keepdim=True)
        teacher_probs = teacher_probs / teacher_row_sum.clamp_min(1e-20)
        _log_hisa_indexer_loss_debug(
            "forward_teacher_normalized",
            debug_call_id,
            teacher_probs=teacher_probs,
            teacher_row_sum=teacher_row_sum,
        )
        if is_hisa_kl_grad_triton_supported(selected_scores, teacher_probs):
            loss_sum, grad_selected_scores = hisa_kl_loss_and_grad_triton(
                selected_scores, teacher_probs
            )
        else:
            index_probs = torch.softmax(selected_scores, dim=-1, dtype=torch.float32)
            valid = topk_i32.reshape(bsz * q_len, topk_k) >= 0
            index_probs = torch.where(valid, index_probs, torch.zeros_like(index_probs))
            kl = teacher_probs * (
                torch.log(teacher_probs + 1e-10) - torch.log(index_probs + 1e-10)
            )
            loss_sum = kl.masked_fill(~valid, 0).sum()
            prob_ratio = index_probs / (index_probs + 1e-10)
            teacher_prob_ratio = teacher_probs * prob_ratio
            row_scale = teacher_prob_ratio.sum(dim=-1, keepdim=True)
            grad_selected_scores = (
                index_probs * row_scale - teacher_prob_ratio
            ).masked_fill(~valid, 0)
        _log_hisa_indexer_loss_debug(
            "forward_kl",
            debug_call_id,
            loss_sum=loss_sum,
            grad_selected_scores=grad_selected_scores,
        )

        ctx.save_for_backward(
            q,
            weights,
            k,
            topk_compact,
            grad_selected_scores,
        )
        ctx.q_len = q_len
        ctx.bsz = bsz
        ctx.debug_call_id = debug_call_id
        return topk_compact, loss_sum

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss_sum):
        q, weights, k, topk_i32, grad_selected_scores = ctx.saved_tensors
        q_len = ctx.q_len
        bsz = ctx.bsz
        if grad_loss_sum is None:
            return (
                torch.zeros_like(q),
                torch.zeros_like(weights),
                torch.zeros_like(k),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        scale = grad_loss_sum.float() if torch.is_tensor(grad_loss_sum) else 1.0
        grad_selected_scores = grad_selected_scores * scale
        _log_hisa_indexer_loss_debug(
            "backward_input",
            ctx.debug_call_id,
            grad_loss_sum=grad_loss_sum if torch.is_tensor(grad_loss_sum) else None,
            grad_selected_scores=grad_selected_scores,
        )

        grad_q, grad_weights, grad_k = _hisa_selected_score_backward_cuda_batched(
            grad_selected_scores,
            q,
            weights,
            k,
            topk_i32,
        )
        _log_hisa_indexer_loss_debug(
            "backward_output",
            ctx.debug_call_id,
            grad_q=grad_q,
            grad_weights=grad_weights,
            grad_k=grad_k,
        )
        grad_q = grad_q.to(q.dtype)
        grad_weights = grad_weights.to(weights.dtype)
        grad_k = grad_k.to(k.dtype)
        return (
            grad_q,
            grad_weights,
            grad_k,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def chunked_dsa_forward(
    q: Union[torch.Tensor, Callable[[int, int], Tuple[torch.Tensor, torch.Tensor]]],
    k: torch.Tensor,
    weights: Optional[torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    topk: int,
    mask: Optional[torch.Tensor],
    is_causal: bool,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
    chunk_size: int,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    indexcache_hisa_config: Optional[IndexCacheHISAConfig] = None,
    dsa_split_qk: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    hisa_log_label: str = "DSAttention",
    dsa_runtime_context: str = "",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    sq, bsz, num_heads, head_dim = query.size()
    sk = key.size(0)
    if topk <= 0 or sk <= 0:
        raise ValueError(
            "chunked_dsa_forward received non-positive sparse selection extent: "
            f"topk={topk} sk={sk} query_shape={tuple(query.shape)} key_shape={tuple(key.shape)} "
            f"value_shape={tuple(value.shape)} q_shape={'callable' if callable(q) else tuple(q.shape)} "
            f"k_shape={tuple(k.shape)} weights_shape={None if weights is None else tuple(weights.shape)} "
            f"chunk_size={chunk_size} loss_coeff={loss_coeff} "
            f"query_positions_shape={None if query_positions is None else tuple(query_positions.shape)} "
            f"key_positions_shape={None if key_positions is None else tuple(key_positions.shape)}"
        )
    split_query_pe = None
    split_key_pe = None
    split_kv_nope_value_ref = None
    if dsa_split_qk is not None:
        split_query_pe, split_key_pe, *split_extra = dsa_split_qk
        split_kv_nope_value_ref = split_extra[0] if split_extra else None
        if split_query_pe.size(0) != sq or split_query_pe.shape[:3] != query.shape[:3]:
            raise ValueError(
                "DSA split query positional component must match query prefix shape, got "
                f"{tuple(split_query_pe.shape)} for query {tuple(query.shape)}"
            )
        if split_key_pe.size(0) != sk or split_key_pe.size(1) != bsz:
            raise ValueError(
                "DSA split key positional component must match key sequence/batch, got "
                f"{tuple(split_key_pe.shape)} for key {tuple(key.shape)}"
            )

    def materialize_attention_qk(query_part: torch.Tensor, query_pe_part: torch.Tensor):
        if query_pe_part is None or split_key_pe is None:
            return query_part, key
        key_pe = split_key_pe
        if key_pe.size(2) == 1:
            key_pe = key_pe.expand(-1, -1, key.size(2), -1)
        return torch.cat([query_part, query_pe_part], dim=-1), torch.cat([key, key_pe], dim=-1)

    q_weights_provider = q if callable(q) else None
    q_range_limiter = (
        getattr(q_weights_provider, "_dsa_limit_range", None)
        if q_weights_provider is not None
        else None
    )

    def iter_query_ranges():
        for base_q_start in range(0, sq, chunk_size):
            base_q_end = min(base_q_start + chunk_size, sq)
            if q_range_limiter is None:
                yield base_q_start, base_q_end
                continue
            q_start = base_q_start
            while q_start < base_q_end:
                q_end = int(q_range_limiter(q_start, base_q_end))
                if q_end <= q_start or q_end > base_q_end:
                    raise ValueError(
                        "DSA q_weights_provider range limiter returned an invalid range: "
                        f"base=({base_q_start}, {base_q_end}) current_start={q_start} "
                        f"limited_end={q_end}"
                    )
                yield q_start, q_end
                q_start = q_end

    work_device = query.device
    outputs = []
    output_buffer = None
    topk_buffer = None
    use_triton_attention = None
    use_streaming_indexer_topk = loss_coeff <= 0 and _env_flag_enabled(
        _DSA_STREAMING_INDEXER_TOPK_ENV, "1"
    )
    use_indexcache_hisa_topk = indexcache_hisa_config is not None
    index_scores_buffer = None
    topk_values_buffer = None
    topk_indices_buffer = None
    loss_sum = None
    loss_count = 0
    attention_teacher_score_chunks = []
    attention_teacher_row_indices = []
    stream_triton_attention_chunks = _env_flag_enabled(_DSA_STREAM_TRITON_ATTENTION_CHUNKS_ENV, "0")
    hisa_selector_name = (
        indexcache_hisa_selector_backend_name() if use_indexcache_hisa_topk else "disabled"
    )
    runtime_mode = "grad_loss" if loss_coeff > 0 else "grad_no_loss" if torch.is_grad_enabled() else "no_grad"
    use_hisa_dsa_fused_forward = (
        _hisa_dsa_fused_forward_enabled() or _hisa_dsa_persistent_forward_enabled()
    )
    fused_hisa_block_reps = None
    fused_hisa_effective_block_topk = None
    shared_hisa_block_reps = None

    def get_fused_hisa_block_reps() -> tuple[torch.Tensor, int]:
        nonlocal fused_hisa_block_reps, fused_hisa_effective_block_topk
        if indexcache_hisa_config is None:
            raise RuntimeError("fused HISA/DSA forward requested without HISA config")
        if fused_hisa_block_reps is None:
            with fine_profile_range(f"dsa.{hisa_log_label}.hisa_dsa_fused.block_reps"):
                fused_hisa_block_reps = _hisa_block_reps_batched_cuda(
                    k,
                    int(indexcache_hisa_config.block_size),
                )
            fused_hisa_effective_block_topk = _hisa_fused_effective_block_topk(
                sk,
                indexcache_hisa_config,
            )
        return fused_hisa_block_reps, int(fused_hisa_effective_block_topk)

    def get_shared_hisa_block_reps() -> Optional[torch.Tensor]:
        nonlocal shared_hisa_block_reps
        if indexcache_hisa_config is None:
            return None
        if indexcache_hisa_selector_backend_name() != "megakernel":
            return None
        if shared_hisa_block_reps is None:
            with fine_profile_range(f"dsa.{hisa_log_label}.hisa.block_reps"):
                shared_hisa_block_reps = _hisa_block_reps_batched_cuda(
                    k,
                    int(indexcache_hisa_config.block_size),
                )
        return shared_hisa_block_reps

    def hisa_fail_closed(
        reason: str,
        *,
        dense_indexer_fallback: int = 0,
        dense_teacher_fallback: int = 0,
    ) -> None:
        raise RuntimeError(
            "HISA fail-closed violation: "
            f"{reason}; hisa_selector={hisa_selector_name} hisa_deferred_teacher=0 "
            f"dense_indexer_fallback={dense_indexer_fallback} "
            f"dense_teacher_fallback={dense_teacher_fallback} "
            f"query_shape={tuple(query.shape)} key_shape={tuple(key.shape)} "
            f"k_shape={tuple(k.shape)} topk={topk} chunk_size={chunk_size}"
        )

    def log_hisa_runtime(deferred_teacher: bool) -> None:
        if not use_indexcache_hisa_topk:
            return
        _log_hisa_runtime_once(
            hisa_log_label,
            f"hisa_selector={hisa_selector_name} "
            f"hisa_deferred_teacher={int(deferred_teacher)} "
            "dense_indexer_fallback=0 dense_teacher_fallback=0 "
            f"grad_enabled={int(torch.is_grad_enabled())} "
            f"loss_enabled={int(loss_coeff > 0)} "
            f"topk={topk} chunk_size={chunk_size} split_qk={int(split_query_pe is not None)}",
            mode=runtime_mode,
        )

    def log_dsa_runtime(
        *,
        topk_source: str,
        attention_backend: str,
        teacher_backend: str,
        hisa_deferred_teacher: bool,
    ) -> None:
        _log_dsa_runtime_once(
            hisa_log_label,
            f"attention_backend={attention_backend} topk_source={topk_source} "
            f"teacher_backend={teacher_backend} "
            f"hisa_selector={hisa_selector_name} "
            f"hisa_deferred_teacher={int(hisa_deferred_teacher)} "
            "dense_indexer_fallback=0 dense_teacher_fallback=0 "
            f"split_qk={int(split_query_pe is not None)} "
            f"stream_triton_chunks={int(stream_triton_attention_chunks)} "
            f"loss_coeff={loss_coeff:g} sparse_loss={int(sparse_loss)} "
            f"grad_enabled={int(torch.is_grad_enabled())} "
            f"loss_enabled={int(loss_coeff > 0)} "
            f"topk={topk} chunk_size={chunk_size} sq={sq} sk={sk} bsz={bsz} "
            f"q_provider={int(q_weights_provider is not None)} "
            f"positions={int(query_positions is not None or key_positions is not None)} "
            f"{dsa_runtime_context}".strip(),
            mode=runtime_mode,
        )

    def append_output_chunk(chunk_output: torch.Tensor, q_start: int, q_end: int):
        nonlocal output_buffer
        if output_buffer is None:
            output_buffer = chunk_output.new_empty((sq, *chunk_output.shape[1:]))
        expected_q = q_end - q_start
        if chunk_output.size(0) != expected_q:
            raise ValueError(
                "DSA chunk output sequence dimension does not match chunk range: "
                f"output_shape={tuple(chunk_output.shape)} q_range=({q_start}, {q_end})"
            )
        output_buffer[q_start:q_end].copy_(chunk_output)

    for q_start, q_end in iter_query_ranges():
        chunk_profile = f"dsa.{hisa_log_label}.chunk.{q_start}_{q_end}"
        with fine_profile_range(f"{chunk_profile}.prepare_q_weights"):
            if q_weights_provider is not None:
                q_chunk, weights_chunk = q_weights_provider(q_start, q_end)
            else:
                q_chunk = q[q_start:q_end]
                weights_chunk = weights[q_start:q_end]
            query_chunk = query[q_start:q_end]
            query_pe_chunk = None if split_query_pe is None else split_query_pe[q_start:q_end]
            query_positions_chunk = (
                None if query_positions is None else query_positions[q_start:q_end]
            )

        topk_indices = None
        index_scores = None
        hisa_selected_scores = None
        hisa_loss_already_accumulated = False
        hisa_loss_deferred_to_attention = False
        topk_source = "unset"
        if use_indexcache_hisa_topk:
            if loss_coeff > 0:
                topk_k = min(topk, sk)
                if not _hisa_fused_indexer_loss_enabled():
                    hisa_fail_closed("MEGATRON_HISA_FUSED_INDEXER_LOSS is disabled")
                if not (q_chunk.is_cuda and k.is_cuda and query_chunk.is_cuda and key.is_cuda):
                    hisa_fail_closed("HISA training selected-score path requires CUDA tensors")
                with fine_profile_range(f"{chunk_profile}.hisa.prefix_lens"):
                    prefix_lens = _hisa_prefix_lens_for_chunk(
                        q_end - q_start,
                        bsz,
                        sk,
                        q_start=q_start,
                        is_causal=is_causal,
                        mask=mask,
                        query_positions=query_positions_chunk,
                        key_positions=key_positions,
                        device=work_device,
                    )
                if prefix_lens is None:
                    hisa_fail_closed(
                        "could not construct causal prefix lengths for HISA selected-score path"
                    )
                if (
                    indexcache_hisa_config.fallback_to_dense_if_short
                    and int(prefix_lens.max().item()) <= topk_k
                ):
                    hisa_fail_closed("HISA config requested dense short-context fallback")

                with fine_profile_range(f"{chunk_profile}.attention.teacher_support_check"):
                    dummy_topk = torch.empty(
                        (bsz, q_end - q_start, topk_k),
                        device=work_device,
                        dtype=torch.int32,
                    )
                    teacher_supported = (
                        query_pe_chunk is not None
                        and split_key_pe is not None
                        and is_sparse_dsa_split_qk_triton_supported(
                            query_chunk,
                            query_pe_chunk,
                            key,
                            split_key_pe,
                            value,
                            dummy_topk,
                            mask,
                            is_causal,
                            query_positions=query_positions_chunk,
                            key_positions=key_positions,
                            kv_nope_value_ref=split_kv_nope_value_ref,
                        )
                    ) or (
                        query_pe_chunk is None
                        and is_sparse_dsa_teacher_triton_supported(
                            query_chunk,
                            key,
                            value,
                            dummy_topk,
                            mask,
                            is_causal,
                            query_positions=query_positions_chunk,
                            key_positions=key_positions,
                        )
                    )
                if not teacher_supported:
                    hisa_fail_closed(
                        "selected sparse-attention teacher path is unsupported for this chunk"
                    )
                if use_hisa_dsa_fused_forward:
                    if query_pe_chunk is None or split_key_pe is None:
                        hisa_fail_closed(
                            "fused HISA/DSA forward currently requires split-QK MLA tensors"
                        )
                    if indexcache_hisa_config.fallback_to_dense_if_short:
                        hisa_fail_closed(
                            "fused HISA/DSA forward refuses dense short-context fallback"
                        )
                    fused_block_reps, fused_effective_block_topk = get_fused_hisa_block_reps()
                    forced = tuple(indexcache_hisa_config.forced_boundary_blocks or ())
                    has_positions = query_positions_chunk is not None or key_positions is not None
                    empty_positions = torch.empty(0, device=work_device, dtype=torch.long)
                    with fine_profile_range(f"{chunk_profile}.hisa_dsa_fused.forward"):
                        (
                            chunk_output,
                            hisa_selected_scores,
                            attention_teacher_probs,
                            topk_indices,
                        ) = _HISADSASplitQKFusedForward.apply(
                            q_chunk,
                            weights_chunk,
                            k,
                            fused_block_reps,
                            prefix_lens,
                            query_chunk,
                            query_pe_chunk,
                            key,
                            split_key_pe,
                            value,
                            query_positions_chunk if has_positions else empty_positions,
                            key_positions if has_positions else empty_positions,
                            float(softmax_scale),
                            int(q_start),
                            int(indexcache_hisa_config.block_size),
                            int(indexcache_hisa_config.block_topk),
                            float(indexcache_hisa_config.compression_ratio),
                            int(fused_effective_block_topk),
                            int(topk_k),
                            bool(has_positions),
                            "first" in forced,
                            "last" in forced,
                            "last_minus_one" in forced,
                        )
                    if pg_collection is not None and _dsa_process_group_size(pg_collection.tp) > 1:
                        with fine_profile_range(f"{chunk_profile}.attention.teacher_tp_all_reduce"):
                            torch.distributed.all_reduce(
                                attention_teacher_probs.contiguous(), group=pg_collection.tp
                            )
                    with fine_profile_range(f"{chunk_profile}.indexer_loss.selected_scores_kl"):
                        attention_teacher_probs = attention_teacher_probs / attention_teacher_probs.sum(
                            dim=-1, keepdim=True
                        ).clamp_min(1e-20)
                        flat_topk = topk_indices.reshape(bsz * (q_end - q_start), -1)
                        valid = flat_topk >= 0
                        chunk_loss_sum = _SelectedScoresKLLoss.apply(
                            hisa_selected_scores,
                            attention_teacher_probs.reshape_as(hisa_selected_scores),
                            valid,
                        )
                    loss_sum = chunk_loss_sum if loss_sum is None else loss_sum + chunk_loss_sum
                    loss_count += hisa_selected_scores.size(0)
                    topk_source = "hisa_dsa_fused"
                    log_hisa_runtime(deferred_teacher=True)
                    log_dsa_runtime(
                        topk_source=topk_source,
                        attention_backend="fused_hisa_dsa_split_qk",
                        teacher_backend="fused_hisa_dsa_local_teacher",
                        hisa_deferred_teacher=True,
                    )
                    append_output_chunk(chunk_output, q_start, q_end)
                    del chunk_output
                    continue
                try:
                    with fine_profile_range(f"{chunk_profile}.hisa.select_with_scores"):
                        topk_i32, selected_scores = _HISASelectWithScoresBatched.apply(
                            q_chunk,
                            weights_chunk,
                            k,
                            prefix_lens,
                            get_shared_hisa_block_reps(),
                            topk_k,
                            indexcache_hisa_config,
                        )
                except RuntimeError as exc:
                    if "cannot handle this chunk" not in str(exc):
                        raise
                    hisa_fail_closed(str(exc))
                topk_indices = topk_i32
                hisa_selected_scores = selected_scores
                hisa_loss_deferred_to_attention = True
                topk_source = "hisa_selected_scores"
                log_hisa_runtime(deferred_teacher=True)
            else:
                with fine_profile_range(f"{chunk_profile}.hisa.topk"):
                    topk_indices = indexcache_hisa_topk(
                        q_chunk,
                        weights_chunk,
                        k,
                        topk,
                        config=indexcache_hisa_config,
                        q_start=q_start,
                        is_causal=is_causal,
                        mask=mask,
                        query_positions=query_positions_chunk,
                        key_positions=key_positions,
                    )
                if topk_indices is None:
                    if q_chunk.is_cuda or k.is_cuda:
                        hisa_fail_closed("HISA top-k selection returned None")
                    with fine_profile_range(f"{chunk_profile}.dense_indexer_fallback.topk"):
                        index_scores = _compute_index_scores(q_chunk, weights_chunk, k)
                        index_scores = _apply_dsa_score_mask(
                            index_scores,
                            mask,
                            q_start,
                            q_end,
                            sk,
                            is_causal,
                            query_positions=query_positions_chunk,
                            key_positions=key_positions,
                        )
                        topk_k = min(topk, sk)
                        topk_indices = index_scores.topk(topk_k, dim=-1, sorted=False)[1]
                    topk_source = "dense_index_scores_topk_cpu_hisa_short"
                else:
                    topk_source = "hisa_topk"
                    log_hisa_runtime(deferred_teacher=False)

        if topk_indices is not None:
            if (
                loss_coeff > 0
                and hisa_selected_scores is None
                and not hisa_loss_already_accumulated
            ):
                if use_indexcache_hisa_topk:
                    hisa_fail_closed(
                        "HISA selected scores are missing; refusing full-prefix DSA index-score KL",
                        dense_indexer_fallback=1,
                    )
                index_scores = _compute_index_scores(q_chunk, weights_chunk, k)
                index_scores = _apply_dsa_score_mask(
                    index_scores,
                    mask,
                    q_start,
                    q_end,
                    sk,
                    is_causal,
                    query_positions=query_positions_chunk,
                    key_positions=key_positions,
                )
            else:
                index_scores = None
        elif use_streaming_indexer_topk:
            if is_dsa_indexer_scores_triton_supported(
                q_chunk,
                weights_chunk,
                k,
                mask,
                is_causal,
                query_positions=query_positions_chunk,
                key_positions=key_positions,
            ):
                score_shape = (bsz, q_end - q_start, sk)
                if index_scores_buffer is None or tuple(index_scores_buffer.shape) != score_shape:
                    index_scores_buffer = torch.empty(
                        score_shape, device=work_device, dtype=torch.float32
                    )
                index_scores = dsa_indexer_scores_triton(
                    q_chunk,
                    weights_chunk,
                    k,
                    q_start,
                    query_positions=query_positions_chunk,
                    key_positions=key_positions,
                    out=index_scores_buffer,
                )
                topk_k = min(topk, sk)
                topk_shape = (*score_shape[:-1], topk_k)
                if topk_values_buffer is None or tuple(topk_values_buffer.shape) != topk_shape:
                    topk_values_buffer = torch.empty(
                        topk_shape, device=index_scores.device, dtype=index_scores.dtype
                    )
                    topk_indices_buffer = torch.empty(
                        topk_shape, device=index_scores.device, dtype=torch.long
                    )
                _, topk_indices = torch.topk(
                    index_scores,
                    topk_k,
                    dim=-1,
                    sorted=False,
                    out=(topk_values_buffer, topk_indices_buffer),
                )
                topk_source = "triton_indexer_scores_topk"
            else:
                index_scores = None
                topk_indices = _streaming_qk_topk(
                    q_chunk,
                    weights_chunk,
                    k,
                    topk,
                    mask,
                    q_start,
                    q_end,
                    sk,
                    is_causal,
                    query_positions=query_positions_chunk,
                    key_positions=key_positions,
                )
                topk_source = "streaming_qk_topk"
        else:
            if use_indexcache_hisa_topk:
                hisa_fail_closed(
                    "HISA top-k is unavailable; refusing full-prefix DSA indexer fallback",
                    dense_indexer_fallback=1,
                )
            index_scores = _compute_index_scores(q_chunk, weights_chunk, k)
            index_scores = _apply_dsa_score_mask(
                index_scores,
                mask,
                q_start,
                q_end,
                sk,
                is_causal,
                query_positions=query_positions_chunk,
                key_positions=key_positions,
            )
            topk_k = min(topk, sk)
            topk_indices = index_scores.topk(topk_k, dim=-1, sorted=False)[1]
            topk_source = "dense_index_scores_topk"

        if topk_indices is not None:
            with fine_profile_range(f"{chunk_profile}.topk.compact_dtype"):
                topk_indices = _maybe_narrow_dsa_topk_indices(topk_indices, sk)

        if use_triton_attention is None:
            with fine_profile_range(f"{chunk_profile}.attention.support_check"):
                if query_pe_chunk is not None and split_key_pe is not None:
                    use_triton_attention = is_sparse_dsa_split_qk_triton_supported(
                        query_chunk,
                        query_pe_chunk,
                        key,
                        split_key_pe,
                        value,
                        topk_indices,
                        mask,
                        is_causal,
                        query_positions=query_positions_chunk,
                        key_positions=key_positions,
                    )
                else:
                    use_triton_attention = is_sparse_dsa_triton_supported(
                        query_chunk,
                        key,
                        value,
                        topk_indices,
                        mask,
                        is_causal,
                        query_positions=query_positions_chunk,
                        key_positions=key_positions,
                    )
        if _env_flag_enabled(_DSA_VALIDATE_TOPK_INDICES_ENV, "0") and bool(
            (topk_indices < -1).any().item()
        ):
            use_triton_attention = False
        if (
            use_indexcache_hisa_topk
            and not use_triton_attention
            and (query_chunk.is_cuda or key.is_cuda or value.is_cuda)
        ):
            hisa_fail_closed(
                "selected Sparse MLA/DSA Triton attention path is unsupported for HISA top-k"
            )
        if use_triton_attention:
            attention_backend = (
                "triton_split_qk" if query_pe_chunk is not None and split_key_pe is not None else "triton"
            )
            if loss_coeff > 0 and hisa_loss_deferred_to_attention:
                teacher_backend = (
                    "compact_triton_split_qk"
                    if query_pe_chunk is not None and split_key_pe is not None
                    else "compact_triton"
                )
            else:
                teacher_backend = "none"
        else:
            attention_backend = "torch_sparse_chunk"
            teacher_backend = "dense_teacher" if loss_coeff > 0 else "none"
        log_dsa_runtime(
            topk_source=topk_source,
            attention_backend=attention_backend,
            teacher_backend=teacher_backend,
            hisa_deferred_teacher=hisa_loss_deferred_to_attention,
        )

        if loss_coeff > 0 and hisa_loss_already_accumulated:
            pass
        elif (
            loss_coeff > 0
            and hisa_selected_scores is not None
            and not hisa_loss_deferred_to_attention
        ):
            if use_indexcache_hisa_topk:
                hisa_fail_closed(
                    "HISA selected scores were not deferred to sparse-attention teacher",
                    dense_teacher_fallback=1,
                )
            flat_topk = topk_indices.reshape(bsz * (q_end - q_start), -1)
            valid = flat_topk >= 0
            selected_scores = hisa_selected_scores.masked_fill(~valid, float("-inf"))
            attention_query_chunk, attention_key = materialize_attention_qk(
                query_chunk, query_pe_chunk
            )
            attention_probs = _hisa_attention_target_probs(
                attention_query_chunk,
                attention_key,
                topk_indices,
                float(softmax_scale),
                pg_collection.tp if pg_collection is not None else None,
            )
            index_probs = torch.softmax(selected_scores, dim=-1, dtype=torch.float32)
            index_probs = torch.where(valid, index_probs, torch.zeros_like(index_probs))
            kl = attention_probs * (
                torch.log(attention_probs + 1e-10) - torch.log(index_probs + 1e-10)
            )
            chunk_loss_sum = kl.masked_fill(~valid, 0).sum()
            loss_sum = chunk_loss_sum if loss_sum is None else loss_sum + chunk_loss_sum
            loss_count += bsz * (q_end - q_start)
        elif loss_coeff > 0 and not hisa_loss_deferred_to_attention:
            if use_indexcache_hisa_topk:
                hisa_fail_closed(
                    "refusing dense teacher score construction for HISA indexer loss",
                    dense_teacher_fallback=1,
                )
            loss_index_scores = index_scores
            attention_query_chunk, attention_key_full = materialize_attention_qk(
                query_chunk, query_pe_chunk
            )
            attention_query = attention_query_chunk.detach().permute(1, 2, 0, 3).reshape(
                bsz * num_heads, q_end - q_start, attention_query_chunk.size(-1)
            )
            attention_key = attention_key_full.detach().permute(1, 2, 3, 0).reshape(
                bsz * num_heads, attention_key_full.size(-1), sk
            )
            attention_scores = torch.bmm(attention_query.float(), attention_key.float())
            attention_scores = attention_scores.reshape(bsz, num_heads, q_end - q_start, sk)
            attention_scores = attention_scores * softmax_scale
            attention_scores = _apply_dsa_score_mask(
                attention_scores,
                mask,
                q_start,
                q_end,
                sk,
                is_causal,
                query_positions=query_positions_chunk,
                key_positions=key_positions,
            )

            if sparse_loss:
                valid_topk = topk_indices >= 0
                selected_counts = torch.zeros_like(loss_index_scores, dtype=torch.int32)
                selected_counts.scatter_add_(
                    -1,
                    topk_indices.clamp_min(0),
                    valid_topk.to(dtype=selected_counts.dtype),
                )
                index_mask = torch.full_like(loss_index_scores, float("-inf"))
                index_mask = index_mask.masked_fill(selected_counts > 0, 0)
                loss_index_scores = loss_index_scores + index_mask
                attention_scores = attention_scores + index_mask.unsqueeze(1)

            attention_probs = torch.softmax(attention_scores, dim=-1, dtype=torch.float32)
            index_probs = torch.softmax(loss_index_scores, dim=-1, dtype=torch.float32)

            attention_probs = attention_probs.sum(dim=1)
            if pg_collection.tp.size() > 1:
                torch.distributed.all_reduce(attention_probs.contiguous(), group=pg_collection.tp)
            attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)

            kl = attention_probs * (
                torch.log(attention_probs + 1e-10) - torch.log(index_probs + 1e-10)
            )
            chunk_loss_sum = kl.sum(dim=-1).sum()
            loss_sum = chunk_loss_sum if loss_sum is None else loss_sum + chunk_loss_sum
            loss_count += bsz * (q_end - q_start)
        elif index_scores is not None:
            del index_scores

        if use_triton_attention:
            with fine_profile_range(f"{chunk_profile}.topk.sort_for_triton"):
                topk_indices, hisa_selected_scores = _maybe_sort_dsa_topk_indices_and_scores(
                    topk_indices, hisa_selected_scores
                )
            if stream_triton_attention_chunks:
                if hisa_loss_deferred_to_attention:
                    if query_pe_chunk is not None and split_key_pe is not None:
                        with fine_profile_range(
                            f"{chunk_profile}.attention.triton_split_qk_with_teacher"
                        ):
                            (
                                chunk_output,
                                attention_teacher_probs,
                            ) = sparse_dsa_attention_split_qk_with_teacher_triton(
                                query_chunk,
                                query_pe_chunk,
                                key,
                                split_key_pe,
                                value,
                                topk_indices,
                                softmax_scale,
                                q_start,
                                query_positions=query_positions_chunk,
                                key_positions=key_positions,
                                kv_nope_value_ref=split_kv_nope_value_ref,
                            )
                    else:
                        with fine_profile_range(f"{chunk_profile}.attention.triton_with_teacher"):
                            (
                                chunk_output,
                                attention_teacher_probs,
                            ) = sparse_dsa_attention_with_teacher_triton(
                                query_chunk,
                                key,
                                value,
                                topk_indices,
                                softmax_scale,
                                q_start,
                                query_positions=query_positions_chunk,
                                key_positions=key_positions,
                            )
                    if pg_collection is not None and _dsa_process_group_size(pg_collection.tp) > 1:
                        with fine_profile_range(f"{chunk_profile}.attention.teacher_tp_all_reduce"):
                            torch.distributed.all_reduce(
                                attention_teacher_probs.contiguous(), group=pg_collection.tp
                            )
                    with fine_profile_range(f"{chunk_profile}.indexer_loss.selected_scores_kl"):
                        attention_teacher_probs = attention_teacher_probs / attention_teacher_probs.sum(
                            dim=-1, keepdim=True
                        ).clamp_min(1e-20)
                        flat_topk = topk_indices.reshape(bsz * (q_end - q_start), -1)
                        valid = flat_topk >= 0
                        chunk_loss_sum = _SelectedScoresKLLoss.apply(
                            hisa_selected_scores,
                            attention_teacher_probs.reshape_as(hisa_selected_scores),
                            valid,
                        )
                    loss_sum = chunk_loss_sum if loss_sum is None else loss_sum + chunk_loss_sum
                    loss_count += hisa_selected_scores.size(0)
                    append_output_chunk(chunk_output, q_start, q_end)
                    del chunk_output
                else:
                    if query_pe_chunk is not None and split_key_pe is not None:
                        with fine_profile_range(f"{chunk_profile}.attention.triton_split_qk"):
                            chunk_output = sparse_dsa_attention_split_qk_triton(
                                query_chunk,
                                query_pe_chunk,
                                key,
                                split_key_pe,
                                value,
                                topk_indices,
                                softmax_scale,
                                q_start,
                                query_positions=query_positions_chunk,
                                key_positions=key_positions,
                                kv_nope_value_ref=split_kv_nope_value_ref,
                            )
                    else:
                        with fine_profile_range(f"{chunk_profile}.attention.triton"):
                            chunk_output = sparse_dsa_attention_triton(
                                query_chunk,
                                key,
                                value,
                                topk_indices,
                                softmax_scale,
                                q_start,
                                query_positions=query_positions_chunk,
                                key_positions=key_positions,
                            )
                    append_output_chunk(chunk_output, q_start, q_end)
                    del chunk_output
                continue
            if topk_buffer is None:
                topk_buffer = torch.empty(
                    (bsz, sq, topk_indices.size(-1)),
                    device=topk_indices.device,
                    dtype=_dsa_topk_buffer_dtype(sk),
                )
            topk_buffer[:, q_start:q_end, :].copy_(topk_indices)
            if hisa_loss_deferred_to_attention:
                attention_teacher_score_chunks.append(hisa_selected_scores)
                attention_teacher_row_indices.append(
                    torch.cat(
                        [
                            torch.arange(
                                batch_idx * sq + q_start,
                                batch_idx * sq + q_end,
                                device=topk_buffer.device,
                                dtype=torch.int32,
                            )
                            for batch_idx in range(bsz)
                        ]
                    )
                )
        else:
            fallback_query_chunk, fallback_key = materialize_attention_qk(
                query_chunk, query_pe_chunk
            )
            chunk_output = _sparse_dsa_attention_chunk(
                fallback_query_chunk,
                fallback_key,
                value,
                topk_indices,
                softmax_scale,
                mask,
                q_start,
                is_causal,
                query_positions=query_positions_chunk,
                key_positions=key_positions,
            )
            append_output_chunk(chunk_output, q_start, q_end)
            del chunk_output

    if use_triton_attention and stream_triton_attention_chunks:
        if output_buffer is None:
            raise RuntimeError("DSA streamed attention produced no output chunks")
        output = output_buffer
    elif use_triton_attention:
        # Keep top-k generation chunked to bound the indexer score tensor, then run
        # the selected-token attention as one autograd op so K/V gradients are
        # accumulated once per layer instead of once per query chunk.
        if attention_teacher_score_chunks:
            if split_query_pe is not None and split_key_pe is not None:
                output, attention_teacher_probs = sparse_dsa_attention_split_qk_with_teacher_triton(
                    query,
                    split_query_pe,
                    key,
                    split_key_pe,
                    value,
                    topk_buffer,
                    softmax_scale,
                    0,
                    query_positions=query_positions,
                    key_positions=key_positions,
                    kv_nope_value_ref=split_kv_nope_value_ref,
                )
            else:
                output, attention_teacher_probs = sparse_dsa_attention_with_teacher_triton(
                    query,
                    key,
                    value,
                    topk_buffer,
                    softmax_scale,
                    0,
                    query_positions=query_positions,
                    key_positions=key_positions,
                )
            if pg_collection is not None and _dsa_process_group_size(pg_collection.tp) > 1:
                torch.distributed.all_reduce(
                    attention_teacher_probs.contiguous(), group=pg_collection.tp
                )
            attention_teacher_probs = attention_teacher_probs / attention_teacher_probs.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-20)
            selected_scores = torch.cat(attention_teacher_score_chunks, dim=0)
            flat_topk_buffer = topk_buffer.reshape(bsz * sq, topk_buffer.size(-1))
            attention_teacher_probs = torch.cat(
                [
                    attention_teacher_probs.index_select(0, row_indices)
                    for row_indices in attention_teacher_row_indices
                ],
                dim=0,
            )
            flat_topk = torch.cat(
                [
                    flat_topk_buffer.index_select(0, row_indices)
                    for row_indices in attention_teacher_row_indices
                ],
                dim=0,
            )
            valid = flat_topk >= 0
            chunk_loss_sum = _SelectedScoresKLLoss.apply(
                selected_scores,
                attention_teacher_probs.reshape_as(selected_scores),
                valid,
            )
            loss_sum = chunk_loss_sum if loss_sum is None else loss_sum + chunk_loss_sum
            loss_count += selected_scores.size(0)
        else:
            if split_query_pe is not None and split_key_pe is not None:
                output = sparse_dsa_attention_split_qk_triton(
                    query,
                    split_query_pe,
                    key,
                    split_key_pe,
                    value,
                    topk_buffer,
                    softmax_scale,
                    0,
                    query_positions=query_positions,
                    key_positions=key_positions,
                    kv_nope_value_ref=split_kv_nope_value_ref,
                )
            else:
                output = sparse_dsa_attention_triton(
                    query,
                    key,
                    value,
                    topk_buffer,
                    softmax_scale,
                    0,
                    query_positions=query_positions,
                    key_positions=key_positions,
                )
    else:
        if output_buffer is not None:
            output = output_buffer
        else:
            output = torch.cat(outputs, dim=0)

    indexer_loss = None
    if loss_sum is not None:
        indexer_loss = loss_sum * (loss_coeff / loss_count)
    return output, indexer_loss


class DSAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an DSA Indexer to compute top-k
    attention indices for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.layer_number = layer_number

        self.indexer = build_module(
            submodules.indexer, config=self.config, pg_collection=pg_collection
        )
        self.indexer.layer_number = layer_number
        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0) or 0.0
        if indexer_loss_coeff <= 0:
            for param in self.indexer.parameters():
                param.requires_grad_(False)

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        streambp_positions: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dsa_split_qk: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor [sq, b, np, hn].
            key: Key tensor [skv, b, np, hn].
            value: Value tensor [skv, b, np, hnv].
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, hidden_size]
        """
        if (
            self.training
            and _env_flag_enabled("MEGATRON_DSA_REQUIRE_SPLIT_QK", "0")
            and dsa_split_qk is None
        ):
            raise RuntimeError(
                "MEGATRON_DSA_REQUIRE_SPLIT_QK=1 but DSAttention did not receive "
                "split-Q/K MLA tensors. Disable MLA RoPE fusion or unset the guard "
                "for non-split DSA runs."
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            cp_group = self.indexer.pg_collection.cp
            cp_size = _dsa_process_group_size(cp_group)
            cu_seqlens_q = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            cu_seqlens_kv = (
                packed_seq_params.cu_seqlens_kv_padded
                if packed_seq_params.cu_seqlens_kv_padded is not None
                else packed_seq_params.cu_seqlens_kv
            )
            if cu_seqlens_q is None or cu_seqlens_kv is None:
                raise ValueError("DSAttention THD path requires query and KV cu_seqlens")

            def normalize_cu(cu_seqlens: torch.Tensor, name: str) -> torch.Tensor:
                if cu_seqlens.dim() == 2:
                    if cu_seqlens.size(0) != 1:
                        raise ValueError(
                            f"DSAttention THD path expects micro-batch-size 1, got "
                            f"{name} shape {tuple(cu_seqlens.shape)}"
                        )
                    cu_seqlens = cu_seqlens[0]
                if cu_seqlens.dim() != 1:
                    raise ValueError(
                        f"{name} must be 1D, got shape {tuple(cu_seqlens.shape)}"
                    )
                return cu_seqlens

            cu_seqlens_q = normalize_cu(cu_seqlens_q, "cu_seqlens_q")
            cu_seqlens_kv = normalize_cu(cu_seqlens_kv, "cu_seqlens_kv")
            if cu_seqlens_q.numel() != cu_seqlens_kv.numel():
                raise ValueError(
                    "DSAttention THD StreamBP path requires query and KV cu_seqlens "
                    f"with the same number of sequences, got {cu_seqlens_q.numel()} and "
                    f"{cu_seqlens_kv.numel()}"
                )

            if query.dim() == 4:
                if query.size(1) != 1 or key.size(1) != 1 or value.size(1) != 1:
                    raise ValueError("DSAttention THD path only supports a dummy batch dimension")
                query = query.squeeze(1)
                key = key.squeeze(1)
                value = value.squeeze(1)
            split_query_pe = split_key_pe = split_kv_nope_value_ref = None
            if dsa_split_qk is not None:
                split_query_pe, split_key_pe, *split_extra = dsa_split_qk
                split_kv_nope_value_ref = split_extra[0] if split_extra else None
                if split_query_pe.dim() == 4:
                    if split_query_pe.size(1) != 1:
                        raise ValueError("DSAttention THD split query PE expects dummy batch")
                    split_query_pe = split_query_pe.squeeze(1)
                if split_key_pe.dim() == 4:
                    if split_key_pe.size(1) != 1:
                        raise ValueError("DSAttention THD split key PE expects dummy batch")
                    split_key_pe = split_key_pe.squeeze(1)
                if split_kv_nope_value_ref is not None and split_kv_nope_value_ref.dim() == 4:
                    if split_kv_nope_value_ref.size(1) != 1:
                        raise ValueError("DSAttention THD split KV ref expects dummy batch")
                    split_kv_nope_value_ref = split_kv_nope_value_ref.squeeze(1)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            if qr.dim() == 2:
                qr = qr.unsqueeze(1)

            if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
                raise ValueError(
                    "DSAttention THD path expects query/key/value as [tokens, heads, dim]"
                )
            if dsa_split_qk is not None:
                if (
                    split_query_pe.dim() != 3
                    or split_query_pe.shape[:2] != query.shape[:2]
                    or split_key_pe.dim() != 3
                    or split_key_pe.size(0) != key.size(0)
                ):
                    raise ValueError(
                        "DSAttention THD split-QK tensors must align with query/key, got "
                        f"query_pe={tuple(split_query_pe.shape)} key_pe={tuple(split_key_pe.shape)}"
                    )
                if split_kv_nope_value_ref is not None and (
                    split_kv_nope_value_ref.dim() != 3
                    or split_kv_nope_value_ref.shape[:2] != key.shape[:2]
                    or split_kv_nope_value_ref.size(-1) != key.size(-1) + value.size(-1)
                ):
                    raise ValueError(
                        "DSAttention THD split KV ref must align with key/value, got "
                        f"kv_ref={tuple(split_kv_nope_value_ref.shape)} key={tuple(key.shape)} "
                        f"value={tuple(value.shape)}"
                    )
            if x.dim() != 3 or qr.dim() != 3:
                raise ValueError("DSAttention THD path expects x/qr as [tokens, batch, dim]")

            streambp_query_positions = None
            streambp_key_positions = None
            if streambp_positions is not None:
                if cp_size > 1:
                    raise ValueError("StreamBP DSA currently requires context_parallel_size == 1")
                streambp_query_positions, streambp_key_positions = streambp_positions
                if streambp_query_positions is None or streambp_key_positions is None:
                    raise ValueError("StreamBP DSA requires both query and key positions")
                streambp_query_positions = streambp_query_positions.to(
                    device=query.device, dtype=torch.long
                )
                streambp_key_positions = streambp_key_positions.to(
                    device=key.device, dtype=torch.long
                )
                if streambp_query_positions.numel() != query.size(0):
                    raise ValueError(
                        f"StreamBP DSA query position length "
                        f"{streambp_query_positions.numel()} does not match query length "
                        f"{query.size(0)}"
                    )
                if streambp_key_positions.numel() != key.size(0):
                    raise ValueError(
                        f"StreamBP DSA key position length {streambp_key_positions.numel()} "
                        f"does not match key length {key.size(0)}"
                    )

            tp_group = self.indexer.pg_collection.tp
            tp_size = _dsa_process_group_size(tp_group)
            if tp_size > 1 and (x.size(0) != key.size(0) or qr.size(0) != key.size(0)):
                # TP sequence-parallel packed replay can carry THD metadata while
                # attention K/V are gathered differently from the indexer x/qr
                # inputs. The generic THD recursion slices x/qr by attention-KV
                # offsets first, which can make the trainable indexer see an
                # empty K. Route the whole local packed tensor through the
                # non-packed path so the SP-aware indexer projection can gather
                # or owner-broadcast compact q/k before top-k.
                return self.forward(
                    query.unsqueeze(1),
                    key.unsqueeze(1),
                    value.unsqueeze(1),
                    attention_mask,
                    x,
                    qr,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=None,
                    streambp_positions=(
                        None
                        if streambp_positions is None
                        else (streambp_query_positions, streambp_key_positions)
                    ),
                    dsa_split_qk=(
                        None
                        if dsa_split_qk is None
                        else (
                            split_query_pe.unsqueeze(1),
                            split_key_pe.unsqueeze(1),
                            *(
                                ()
                                if split_kv_nope_value_ref is None
                                else (split_kv_nope_value_ref.unsqueeze(1),)
                            ),
                        )
                    ),
                )

            outputs = []
            q_cu_total = int(cu_seqlens_q[-1].item())
            kv_cu_total = int(cu_seqlens_kv[-1].item())
            if (
                tp_size > 1
                and cp_size == 1
                and (q_cu_total > query.size(0) or kv_cu_total > key.size(0))
            ):
                local_q_offsets = _dsa_thd_tp_local_sequence_offsets(
                    cu_seqlens_q, query.size(0), tp_group
                )
                local_kv_offsets = _dsa_thd_tp_local_sequence_offsets(
                    cu_seqlens_kv, key.size(0), tp_group
                )
            elif cp_size > 1:
                local_q_offsets = _dsa_thd_local_sequence_offsets(cu_seqlens_q, cp_group)
                local_kv_offsets = _dsa_thd_local_sequence_offsets(cu_seqlens_kv, cp_group)
            else:
                q_offsets = cu_seqlens_q.detach().cpu().tolist()
                kv_offsets = cu_seqlens_kv.detach().cpu().tolist()
                local_q_offsets = [
                    (int(start), int(end)) for start, end in zip(q_offsets[:-1], q_offsets[1:])
                ]
                local_kv_offsets = [
                    (int(start), int(end)) for start, end in zip(kv_offsets[:-1], kv_offsets[1:])
                ]
            for (q_start, q_end), (kv_start, kv_end) in zip(local_q_offsets, local_kv_offsets):
                if q_end <= q_start:
                    continue
                if kv_end <= kv_start:
                    raise ValueError("DSAttention THD path received query tokens with empty KV span")
                if kv_start >= key.size(0) or kv_end > key.size(0):
                    raise ValueError(
                        "DSAttention THD path sliced outside local KV tensor: "
                        f"q_span=({q_start},{q_end}) kv_span=({kv_start},{kv_end}) "
                        f"query_len={query.size(0)} key_len={key.size(0)} value_len={value.size(0)} "
                        f"x_len={x.size(0)} qr_len={qr.size(0)} "
                        f"q_cu_total={q_cu_total} kv_cu_total={kv_cu_total} "
                        f"tp_rank={_dsa_process_group_rank(tp_group)} tp_size={tp_size} "
                        f"cp_size={cp_size}"
                    )
                sequence_streambp_positions = None
                if streambp_positions is not None:
                    key_origin = streambp_key_positions[kv_start]
                    sequence_streambp_positions = (
                        streambp_query_positions[q_start:q_end] - key_origin,
                        streambp_key_positions[kv_start:kv_end] - key_origin,
                    )
                outputs.append(
                    self.forward(
                        query[q_start:q_end].unsqueeze(1),
                        key[kv_start:kv_end].unsqueeze(1),
                        value[kv_start:kv_end].unsqueeze(1),
                        attention_mask,
                        x[kv_start:kv_end],
                        qr[kv_start:kv_end],
                        attn_mask_type=attn_mask_type,
                        attention_bias=attention_bias,
                        packed_seq_params=None,
                        streambp_positions=sequence_streambp_positions,
                        dsa_split_qk=(
                            None
                            if dsa_split_qk is None
                            else (
                                split_query_pe[q_start:q_end].unsqueeze(1),
                                split_key_pe[kv_start:kv_end].unsqueeze(1),
                                *(
                                    ()
                                    if split_kv_nope_value_ref is None
                                    else (
                                        split_kv_nope_value_ref[kv_start:kv_end].unsqueeze(1),
                                    )
                                ),
                            )
                        ),
                    )
                )

            if not outputs:
                return query.new_empty((0, 1, value.size(1) * value.size(2)))
            return torch.cat(outputs, dim=0)

        sq, b, np, hn = query.size()
        skv = key.size(0)
        hnv = value.size(3)

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        x = x.detach()
        qr = qr.detach()

        streambp_query_positions = None
        streambp_key_positions = None
        streambp_query_indices = None
        if streambp_positions is not None:
            streambp_query_positions, streambp_key_positions = streambp_positions
            if streambp_query_positions is None or streambp_key_positions is None:
                raise ValueError("StreamBP DSA requires both query and key positions")
            streambp_query_indices = streambp_query_positions.to(device=x.device, dtype=torch.long)
            if streambp_query_indices.numel() != sq:
                raise ValueError(
                    f"StreamBP DSA query position length {streambp_query_indices.numel()} "
                    f"does not match query length {sq}"
                )
            if streambp_key_positions.numel() != skv:
                raise ValueError(
                    f"StreamBP DSA key position length {streambp_key_positions.numel()} "
                    f"does not match key length {skv}"
                )

        is_causal = False
        # Get a FP32 mask with -inf for masked positions.
        if attn_mask_type is not None:
            assert attn_mask_type == AttnMaskType.causal, 'Only causal mask is supported for now'
            is_causal = True
            float_mask = None
        else:
            assert attention_mask.shape == (b, 1, sq, skv), 'attention_mask shape mismatch'
            # [b, 1, sq, skv] -> [b, sq, skv]
            mask = attention_mask.squeeze()
            # float_mask [b, sq, skv]
            float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                mask, float('-inf')
            )

        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)
        chunk_size = int(self.config.dsa_chunk_size)
        cp_group = self.indexer.pg_collection.cp
        cp_size = _dsa_process_group_size(cp_group)
        defer_indexcache_for_cp = cp_size > 1 and self.indexer.indexcache_config is not None
        use_sp_project_before_gather = (
            _env_flag_enabled(_DSA_SP_PROJECT_BEFORE_GATHER_ENV, "1")
            and self.config.sequence_parallel
            and self.indexer.pg_collection.tp.size() > 1
            and packed_seq_params is None
            and chunk_size > 0
            and _env_flag_enabled(_DSA_CHUNK_INDEXER_PROJ_ENV, "1")
        )
        streambp_sp_project_before_gather = (
            streambp_positions is not None
            and use_sp_project_before_gather
            and streambp_query_positions is not None
            and streambp_key_positions is not None
            and x.size(0) * self.indexer.pg_collection.tp.size() == skv
            and streambp_key_positions.numel() == skv
        )
        chunk_indexer_q = (
            packed_seq_params is None
            and chunk_size > 0
            and _env_flag_enabled(_DSA_CHUNK_INDEXER_PROJ_ENV, "1")
            and (
                (streambp_positions is None and sq > chunk_size)
                or streambp_sp_project_before_gather
            )
        )
        sp_project_before_gather = False
        if chunk_indexer_q:
            sp_project_before_gather = (
                use_sp_project_before_gather
                and (
                    streambp_sp_project_before_gather
                    or (
                        streambp_positions is None
                        and x.size(0) * self.indexer.pg_collection.tp.size() == sq
                        and x.size(0) % chunk_size == 0
                    )
                )
            )
            if sp_project_before_gather:
                (
                    indexer_x,
                    indexer_qr,
                    indexer_rotary_pos_emb,
                    indexer_mscale,
                    indexer_orig_seqlen,
                    indexer_pad_len,
                ) = self.indexer._prepare_inputs_before_topk(
                    x, qr, packed_seq_params, gather_sequence_parallel=False
                )
                k = self.indexer._project_local_key_then_gather_before_topk(
                    indexer_x,
                    indexer_rotary_pos_emb,
                    indexer_mscale,
                    indexer_orig_seqlen,
                    indexer_pad_len,
                    apply_indexcache=not defer_indexcache_for_cp,
                )

                def q_weights_provider(q_start: int, q_end: int):
                    return self.indexer._project_query_chunk_sp_owner_broadcast_before_topk(
                        indexer_x,
                        indexer_qr,
                        indexer_rotary_pos_emb,
                        indexer_mscale,
                        q_start,
                        q_end,
                    )
                if streambp_sp_project_before_gather:
                    streambp_query_positions_for_provider = streambp_query_positions
                    local_seq_len = indexer_x.size(0)
                    tp_size = self.indexer.pg_collection.tp.size()
                    owner_segment_len = (
                        streambp_query_positions_for_provider.numel() // tp_size
                        if tp_size > 1
                        and streambp_query_positions_for_provider.numel() % tp_size == 0
                        else None
                    )
                    streambp_query_base = int(
                        (
                            streambp_query_positions_for_provider
                            % (local_seq_len * tp_size)
                            % local_seq_len
                        )
                        .min()
                        .item()
                    )

                    def q_weights_provider(q_start: int, q_end: int):
                        q_len = q_end - q_start
                        element_size = indexer_x.element_size()
                        required_bytes = (
                            q_len
                            * indexer_x.size(1)
                            * self.indexer.index_n_heads
                            * (self.indexer.index_head_dim + 1)
                            * element_size
                        )
                        _maybe_trim_cuda_cache_for_dsa_sp_q_projection(
                            indexer_x.device, required_bytes
                        )
                        return self.indexer._project_query_positions_sp_owner_broadcast_before_topk(
                            indexer_x,
                            indexer_qr,
                            indexer_rotary_pos_emb,
                            indexer_mscale,
                            streambp_query_positions_for_provider[q_start:q_end],
                            query_base=streambp_query_base,
                        )

                    if owner_segment_len is not None:

                        def limit_streambp_sp_owner_range(q_start: int, q_end: int) -> int:
                            return _dsa_next_owner_segment_range_end(
                                q_start, q_end, owner_segment_len
                            )

                        q_weights_provider._dsa_limit_range = limit_streambp_sp_owner_range

            else:
                (
                    indexer_x,
                    indexer_qr,
                    indexer_rotary_pos_emb,
                    indexer_mscale,
                    indexer_orig_seqlen,
                    indexer_pad_len,
                ) = self.indexer._prepare_inputs_before_topk(x, qr, packed_seq_params)
                k = self.indexer._project_key_before_topk(
                    indexer_x,
                    indexer_rotary_pos_emb,
                    indexer_mscale,
                    indexer_orig_seqlen,
                    indexer_pad_len,
                    apply_indexcache=not defer_indexcache_for_cp,
                )

                def q_weights_provider(q_start: int, q_end: int):
                    return self.indexer._project_query_chunk_before_topk(
                        indexer_x,
                        indexer_qr,
                        indexer_rotary_pos_emb,
                        indexer_mscale,
                        q_start,
                        q_end,
                    )

            q = q_weights_provider
            weights = None
        else:
            q, k, weights = self.indexer.forward_before_topk(
                x,
                qr,
                packed_seq_params,
                query_indices=streambp_query_indices,
                apply_indexcache=not defer_indexcache_for_cp,
            )
        numeric_debug = None
        log_dsa_debug = False
        force_debug = False
        if os.getenv("MEGATRON_NUMERIC_DEBUG_DSA", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            from megatron.core import numeric_debug as _numeric_debug

            numeric_debug = _numeric_debug
            log_dsa_debug = numeric_debug.rank_allowed_for("DSA") and numeric_debug.event_allowed(
                f"dsa.layer{self.layer_number}",
                limit=int(os.getenv("MEGATRON_NUMERIC_DEBUG_DSA_LIMIT", "64")),
            )
            force_debug = os.getenv("MEGATRON_NUMERIC_DEBUG_DSA_FORCE", "").lower() in (
                "1",
                "true",
                "yes",
            )
            if log_dsa_debug:
                numeric_debug.log_tensor(
                    f"dsa.layer{self.layer_number}.query", query, force=force_debug
                )
                numeric_debug.log_tensor(f"dsa.layer{self.layer_number}.key", key, force=force_debug)
                numeric_debug.log_tensor(
                    f"dsa.layer{self.layer_number}.value", value, force=force_debug
                )
                if not callable(q):
                    numeric_debug.log_tensor(
                        f"dsa.layer{self.layer_number}.q_index", q, force=force_debug
                    )
                numeric_debug.log_tensor(f"dsa.layer{self.layer_number}.k_index", k, force=force_debug)
                if weights is not None:
                    numeric_debug.log_tensor(
                        f"dsa.layer{self.layer_number}.weights", weights, force=force_debug
                    )
        query_positions = streambp_query_positions
        key_positions = streambp_key_positions
        if cp_size > 1:
            if streambp_positions is not None:
                raise ValueError("StreamBP DSA currently requires context_parallel_size == 1")
            if not is_causal:
                raise NotImplementedError("DSAttention CP path currently supports causal masks only")
            local_skv = key.size(0)
            cp_sorted_gather = _env_flag_enabled(_DSA_CP_SORTED_ZIGZAG_GATHER_ENV, "0")
            query_positions = _dsa_cp_local_position_ids(sq, cp_group, query.device)
            if cp_sorted_gather:
                key_positions = torch.arange(
                    local_skv * _dsa_process_group_size(cp_group),
                    device=key.device,
                    dtype=torch.long,
                )
                key_order = None
            else:
                key_positions = _dsa_cp_gathered_position_ids(local_skv, cp_group, key.device)
                cp_sort_cache_key = (
                    int(local_skv),
                    _dsa_process_group_size(cp_group),
                    _dsa_process_group_rank(cp_group),
                    str(key.device),
                )
                cached_sort = getattr(self, "_dsa_cp_key_sort_cache", None)
                if cached_sort is not None and cached_sort[0] == cp_sort_cache_key:
                    key_positions, key_order = cached_sort[1], cached_sort[2]
                else:
                    key_order = None
                    if key_positions.numel() > 1 and bool(
                        (key_positions[1:] < key_positions[:-1]).any().item()
                    ):
                        key_order = torch.argsort(key_positions)
                        key_positions = key_positions.index_select(0, key_order)
                    setattr(
                        self,
                        "_dsa_cp_key_sort_cache",
                        (cp_sort_cache_key, key_positions, key_order),
                    )
            k = _dsa_cp_sorted_zigzag_gather(k, cp_group)
            key = _dsa_cp_sorted_zigzag_gather(key, cp_group)
            value = _dsa_cp_sorted_zigzag_gather(value, cp_group)
            if key_order is not None:
                k = k.index_select(0, key_order)
                key = key.index_select(0, key_order)
                value = value.index_select(0, key_order)
            if defer_indexcache_for_cp:
                k = self.indexer._apply_indexcache_to_key(k)
            if dsa_split_qk is not None:
                split_query_pe, split_key_pe, *split_extra = dsa_split_qk
                gathered_extra = []
                if _env_flag_enabled(_DSA_CP_GATHER_SPLIT_KV_REF_ENV, "0"):
                    for extra_tensor in split_extra:
                        if extra_tensor is None:
                            gathered_extra.append(None)
                            continue
                        gathered = _dsa_cp_sorted_zigzag_gather(extra_tensor, cp_group)
                        if key_order is not None:
                            gathered = gathered.index_select(0, key_order)
                        gathered_extra.append(gathered)
                else:
                    # The packed kv reference is only a backward optimization.  Under CP it
                    # requires another full-sequence gather and sort on the key side; the
                    # split-Q/K Triton path can instead emit separate key/value grads.
                    gathered_extra = [None for _ in split_extra]
                gathered_split_key_pe = _dsa_cp_sorted_zigzag_gather(split_key_pe, cp_group)
                if key_order is not None:
                    gathered_split_key_pe = gathered_split_key_pe.index_select(0, key_order)
                dsa_split_qk = (
                    split_query_pe,
                    gathered_split_key_pe,
                    *gathered_extra,
                )
        indexcache_quantization = (
            self.indexer.indexcache_config.quantization
            if self.indexer.indexcache_config is not None
            else "disabled"
        )
        dsa_runtime_context = (
            f"cp_size={cp_size} defer_indexcache_for_cp={int(defer_indexcache_for_cp)} "
            f"sequence_parallel={int(self.config.sequence_parallel)} "
            f"chunk_indexer_q={int(chunk_indexer_q)} "
            f"sp_project_before_gather={int(sp_project_before_gather)} "
            f"streambp={int(streambp_positions is not None)} "
            f"indexcache_quant={indexcache_quantization} "
            f"hisa_enabled={int(self.indexer.indexcache_hisa_config is not None)}"
            f" cp_sorted_zigzag_gather={int(cp_sorted_gather) if cp_size > 1 else 0}"
        )
        output, indexer_loss = chunked_dsa_forward(
            q,
            k,
            weights,
            query,
            key,
            value,
            self.softmax_scale,
            self.indexer.index_topk,
            float_mask,
            is_causal,
            indexer_loss_coeff if self.training and torch.is_grad_enabled() else 0.0,
            getattr(self.config, "dsa_indexer_use_sparse_loss", False),
            self.indexer.pg_collection,
            chunk_size,
            query_positions=query_positions,
            key_positions=key_positions,
            indexcache_hisa_config=self.indexer.indexcache_hisa_config,
            dsa_split_qk=dsa_split_qk,
            hisa_log_label=f"DSAttention:{self.layer_number}",
            dsa_runtime_context=dsa_runtime_context,
        )
        if indexer_loss is not None:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss, layer_number=self.layer_number, num_layers=self.config.num_layers
            )
            if _dsa_indexer_aux_loss_autoscale_enabled():
                output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
            else:
                DSAIndexerAuxLossState.add(indexer_loss)
        if numeric_debug is not None and log_dsa_debug:
            numeric_debug.log_tensor(
                f"dsa.layer{self.layer_number}.output", output, force=force_debug
            )
            if indexer_loss is not None:
                numeric_debug.log_tensor(
                    f"dsa.layer{self.layer_number}.indexer_loss",
                    indexer_loss,
                    force=force_debug,
                )

        return output
