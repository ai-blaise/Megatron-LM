# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import os
from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = version.parse(triton.__version__) >= version.parse("2.0.0")
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


_DSA_TRITON_ENV = "MEGATRON_DSA_TRITON"
_DSA_TRITON_BLOCK_K_ENV = "MEGATRON_DSA_TRITON_BLOCK_K"
_DSA_TRITON_BLOCK_K_BWD_ENV = "MEGATRON_DSA_TRITON_BLOCK_K_BWD"
_DSA_TRITON_BLOCK_Q_ENV = "MEGATRON_DSA_TRITON_BLOCK_Q"
_DSA_TRITON_BLOCK_Q_BWD_ENV = "MEGATRON_DSA_TRITON_BLOCK_Q_BWD"
_DSA_TRITON_INDEXER_ENV = "MEGATRON_DSA_TRITON_INDEXER"
_DSA_TRITON_INDEXER_BLOCK_Q_ENV = "MEGATRON_DSA_TRITON_INDEXER_BLOCK_Q"
_DSA_TRITON_INDEXER_BLOCK_K_ENV = "MEGATRON_DSA_TRITON_INDEXER_BLOCK_K"
_DSA_TRITON_BF16_GRAD_ATOMICS_ENV = "MEGATRON_DSA_TRITON_BF16_GRAD_ATOMICS"
_DSA_TRITON_BWD_NUM_WARPS_ENV = "MEGATRON_DSA_TRITON_BWD_NUM_WARPS"
_DSA_TRITON_GROUPED_KV_BWD_ENV = "MEGATRON_DSA_TRITON_GROUPED_KV_BWD"
_DSA_TRITON_KEY_BLOCK_KV_BWD_ENV = "MEGATRON_DSA_TRITON_KEY_BLOCK_KV_BWD"
_DSA_TRITON_KEY_BLOCK_KV_BWD_KEYS_ENV = "MEGATRON_DSA_TRITON_KEY_BLOCK_KV_BWD_KEYS"
_DSA_TRITON_KEY_BLOCK_KV_BWD_QUERIES_ENV = "MEGATRON_DSA_TRITON_KEY_BLOCK_KV_BWD_QUERIES"
_DSA_TRITON_QTILE_UNIQUE_KV_BWD_ENV = "MEGATRON_DSA_TRITON_QTILE_UNIQUE_KV_BWD"
_DSA_TRITON_QTILE_UNIQUE_KV_BWD_Q_ENV = "MEGATRON_DSA_TRITON_QTILE_UNIQUE_KV_BWD_Q"
_DSA_TRITON_QTILE_UNIQUE_KV_BWD_K_ENV = "MEGATRON_DSA_TRITON_QTILE_UNIQUE_KV_BWD_K"
_DSA_CUDA_KV_BWD_ENV = "MEGATRON_DSA_CUDA_KV_BWD"
_DSA_CUDA_KV_BWD_TILE_Q_ENV = "MEGATRON_DSA_CUDA_KV_BWD_TILE_Q"
_DSA_CUDA_KV_BWD_TILE_K_ENV = "MEGATRON_DSA_CUDA_KV_BWD_TILE_K"
_DSA_CUDA_BWD_FROM_SCORES_ENV = "MEGATRON_DSA_CUDA_BWD_FROM_SCORES"
_DSA_CUDA_ROW_BWD_FROM_SCORES_ENV = "MEGATRON_DSA_CUDA_ROW_BWD_FROM_SCORES"
_DSA_CUDA_SORTED_KV_BWD_ENV = "MEGATRON_DSA_CUDA_SORTED_KV_BWD"
_HISA_TARGET_TRITON_ENV = "MEGATRON_HISA_TARGET_TRITON"
_HISA_TARGET_BLOCK_K_ENV = "MEGATRON_HISA_TARGET_BLOCK_K"
_HISA_KL_GRAD_TRITON_ENV = "MEGATRON_HISA_KL_GRAD_TRITON"
_DSA_TEACHER_FROM_LSE_ENV = "MEGATRON_DSA_TEACHER_FROM_LSE"
_DSA_TEACHER_SCORE_SCRATCH_ENV = "MEGATRON_DSA_TEACHER_SCORE_SCRATCH"
_DSA_TRITON_BWD_SCORE_SCRATCH_ENV = "MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH"


def _env_enabled() -> bool:
    raw = os.getenv(_DSA_TRITON_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _indexer_env_enabled() -> bool:
    raw = os.getenv(_DSA_TRITON_INDEXER_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _hisa_target_env_enabled() -> bool:
    raw = os.getenv(_HISA_TARGET_TRITON_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _hisa_kl_grad_env_enabled() -> bool:
    raw = os.getenv(_HISA_KL_GRAD_TRITON_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _teacher_from_lse_enabled() -> bool:
    raw = os.getenv(_DSA_TEACHER_FROM_LSE_ENV, "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _teacher_score_scratch_enabled() -> bool:
    raw = os.getenv(_DSA_TEACHER_SCORE_SCRATCH_ENV, "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _triton_bwd_score_scratch_enabled() -> bool:
    raw = os.getenv(_DSA_TRITON_BWD_SCORE_SCRATCH_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _maybe_narrow_topk_indices(topk_indices: torch.Tensor, sk: int) -> torch.Tensor:
    if topk_indices.dtype == torch.long and sk <= torch.iinfo(torch.int32).max:
        return topk_indices.to(torch.int32)
    return topk_indices


def _topk_sorted_ascending(topk_indices: torch.Tensor) -> bool:
    if topk_indices.shape[-1] <= 1:
        return True
    # Only used for an opt-in debug/perf path that requires binary search over
    # selected tokens. The sync is intentionally kept out of the default path.
    return bool((topk_indices[..., 1:] >= topk_indices[..., :-1]).all().item())


def _bf16_grad_atomics_enabled() -> bool:
    # Experimental bandwidth knob only. The default path keeps K/V gradient
    # accumulation in fp32 so W4A4KV4 + IndexCache FP8 semantics are unchanged.
    raw = os.getenv(_DSA_TRITON_BF16_GRAD_ATOMICS_ENV, "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _key_block_kv_backward_enabled() -> bool:
    raw = os.getenv(_DSA_TRITON_KEY_BLOCK_KV_BWD_ENV, "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _grouped_kv_backward_enabled() -> bool:
    raw = os.getenv(_DSA_TRITON_GROUPED_KV_BWD_ENV, "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _qtile_unique_kv_backward_enabled() -> bool:
    raw = os.getenv(_DSA_TRITON_QTILE_UNIQUE_KV_BWD_ENV, "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _cuda_kv_backward_requested() -> bool:
    raw = os.getenv(_DSA_CUDA_KV_BWD_ENV, "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _cuda_bwd_from_scores_requested() -> bool:
    raw = os.getenv(_DSA_CUDA_BWD_FROM_SCORES_ENV, "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _cuda_row_bwd_from_scores_requested() -> bool:
    raw = os.getenv(_DSA_CUDA_ROW_BWD_FROM_SCORES_ENV, "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _cuda_sorted_kv_backward_requested() -> bool:
    raw = os.getenv(_DSA_CUDA_SORTED_KV_BWD_ENV, "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _cuda_kv_backward_tile_q() -> int:
    raw = os.getenv(_DSA_CUDA_KV_BWD_TILE_Q_ENV)
    value = int(raw) if raw else 2
    if value <= 0:
        raise ValueError(f"{_DSA_CUDA_KV_BWD_TILE_Q_ENV} must be positive, got {value}")
    return value


def _cuda_kv_backward_tile_k() -> int:
    raw = os.getenv(_DSA_CUDA_KV_BWD_TILE_K_ENV)
    value = int(raw) if raw else 4
    if value <= 0:
        raise ValueError(f"{_DSA_CUDA_KV_BWD_TILE_K_ENV} must be positive, got {value}")
    return value


def _try_load_dsa_cuda_ext():
    try:
        from megatron.core.extensions.hisa_indexer.kernels.build import get_ext

        return get_ext()
    except Exception:
        return None


def _cuda_kv_backward_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    query_positions: torch.Tensor | None,
    key_positions: torch.Tensor | None,
) -> bool:
    if not _cuda_kv_backward_requested():
        return False
    if query_positions is not None or key_positions is not None:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda and topk_indices.is_cuda):
        return False
    if query.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    if not (key.dtype == query.dtype and value.dtype == query.dtype):
        return False
    if topk_indices.dtype not in (torch.int16, torch.int32, torch.int64):
        return False
    tile_q = _cuda_kv_backward_tile_q()
    tile_k = _cuda_kv_backward_tile_k()
    if tile_q * tile_k > 8:
        return False
    ext = _try_load_dsa_cuda_ext()
    return ext is not None and hasattr(ext, "dsa_sparse_kv_bwd")


def _cuda_bwd_from_scores_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    teacher_score_scratch: torch.Tensor,
    emit_teacher: bool,
) -> bool:
    if not (_cuda_bwd_from_scores_requested() and emit_teacher and _teacher_score_scratch_enabled()):
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda and topk_indices.is_cuda):
        return False
    if not teacher_score_scratch.is_cuda or teacher_score_scratch.dtype != torch.float32:
        return False
    if query.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    if not (key.dtype == query.dtype and value.dtype == query.dtype):
        return False
    if topk_indices.dtype not in (torch.int16, torch.int32, torch.int64):
        return False
    tile_q = _cuda_kv_backward_tile_q()
    tile_k = _cuda_kv_backward_tile_k()
    if tile_q * tile_k > 8:
        return False
    ext = _try_load_dsa_cuda_ext()
    return ext is not None and hasattr(ext, "dsa_sparse_bwd_from_scores")


def _cuda_row_bwd_from_scores_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    teacher_score_scratch: torch.Tensor,
    emit_teacher: bool,
) -> bool:
    if not (
        _cuda_row_bwd_from_scores_requested()
        and emit_teacher
        and _teacher_score_scratch_enabled()
    ):
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda and topk_indices.is_cuda):
        return False
    if not teacher_score_scratch.is_cuda or teacher_score_scratch.dtype != torch.float32:
        return False
    if query.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    if not (key.dtype == query.dtype and value.dtype == query.dtype):
        return False
    if topk_indices.dtype not in (torch.int16, torch.int32, torch.int64):
        return False
    ext = _try_load_dsa_cuda_ext()
    return ext is not None and hasattr(ext, "dsa_sparse_bwd_from_scores_row")


def _cuda_sorted_kv_backward_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    teacher_score_scratch: torch.Tensor,
    emit_teacher: bool,
    query_positions: torch.Tensor | None,
    key_positions: torch.Tensor | None,
) -> bool:
    if not (
        _cuda_sorted_kv_backward_requested()
        and emit_teacher
        and _teacher_score_scratch_enabled()
    ):
        return False
    if query_positions is not None or key_positions is not None:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda and topk_indices.is_cuda):
        return False
    if not teacher_score_scratch.is_cuda or teacher_score_scratch.dtype != torch.float32:
        return False
    if query.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    if not (key.dtype == query.dtype and value.dtype == query.dtype):
        return False
    if topk_indices.dtype not in (torch.int16, torch.int32, torch.int64):
        return False
    ext = _try_load_dsa_cuda_ext()
    return ext is not None and hasattr(ext, "dsa_sparse_kv_bwd_sorted_from_scores")


def _dsa_sparse_kv_backward_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    softmax_scale: float,
    q_start: int,
    tile_q: int,
    tile_k: int,
) -> None:
    ext = _try_load_dsa_cuda_ext()
    if ext is None or not hasattr(ext, "dsa_sparse_kv_bwd"):
        raise RuntimeError("DSA sparse K/V CUDA backward extension is unavailable")
    ext.dsa_sparse_kv_bwd(
        query,
        key,
        value,
        topk_indices,
        output,
        lse,
        grad_output,
        grad_key,
        grad_value,
        float(softmax_scale),
        int(q_start),
        int(tile_q),
        int(tile_k),
    )


def _dsa_sparse_backward_from_scores_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    selected_scores: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    softmax_scale: float,
    tile_q: int,
    tile_k: int,
) -> None:
    ext = _try_load_dsa_cuda_ext()
    if ext is None or not hasattr(ext, "dsa_sparse_bwd_from_scores"):
        raise RuntimeError("DSA sparse backward-from-scores CUDA extension is unavailable")
    ext.dsa_sparse_bwd_from_scores(
        query,
        key,
        value,
        topk_indices,
        selected_scores,
        output,
        lse,
        grad_output,
        grad_query,
        grad_key,
        grad_value,
        float(softmax_scale),
        int(tile_q),
        int(tile_k),
    )


def _dsa_sparse_backward_from_scores_row_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    selected_scores: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    softmax_scale: float,
) -> None:
    ext = _try_load_dsa_cuda_ext()
    if ext is None or not hasattr(ext, "dsa_sparse_bwd_from_scores_row"):
        raise RuntimeError("DSA row-owned backward-from-scores CUDA extension is unavailable")
    ext.dsa_sparse_bwd_from_scores_row(
        query,
        key,
        value,
        topk_indices,
        selected_scores,
        output,
        lse,
        grad_output,
        grad_query,
        grad_key,
        grad_value,
        float(softmax_scale),
    )


def _dsa_sparse_kv_backward_sorted_from_scores_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    selected_scores: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    softmax_scale: float,
) -> None:
    ext = _try_load_dsa_cuda_ext()
    if ext is None or not hasattr(ext, "dsa_sparse_kv_bwd_sorted_from_scores"):
        raise RuntimeError("DSA sorted-segment K/V backward CUDA extension is unavailable")
    ext.dsa_sparse_kv_bwd_sorted_from_scores(
        query,
        key,
        value,
        topk_indices,
        selected_scores,
        output,
        lse,
        grad_output,
        grad_key,
        grad_value,
        float(softmax_scale),
    )


def _block_k_from_env(env_name: str, topk: int, large_topk_default: int) -> int:
    raw = os.getenv(env_name)
    if raw:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{env_name} must be positive, got {value}")
        return min(triton.next_power_of_2(value), 256)
    if topk >= 1024:
        return large_topk_default
    return min(triton.next_power_of_2(topk), 128)


def _forward_block_k(topk: int) -> int:
    return _block_k_from_env(_DSA_TRITON_BLOCK_K_ENV, topk, large_topk_default=128)


def _backward_block_k(topk: int) -> int:
    return _block_k_from_env(_DSA_TRITON_BLOCK_K_BWD_ENV, topk, large_topk_default=64)


def _hisa_target_block_k(topk: int) -> int:
    return _block_k_from_env(_HISA_TARGET_BLOCK_K_ENV, topk, large_topk_default=64)


def _hisa_kl_block_k(topk: int) -> int:
    return min(triton.next_power_of_2(topk), 2048)


def _sparse_block_q() -> int:
    raw = os.getenv(_DSA_TRITON_BLOCK_Q_ENV)
    if not raw:
        return 1
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{_DSA_TRITON_BLOCK_Q_ENV} must be positive, got {value}")
    return min(triton.next_power_of_2(value), 4)


def _sparse_backward_block_q() -> int:
    raw = os.getenv(_DSA_TRITON_BLOCK_Q_BWD_ENV)
    if not raw:
        return _sparse_block_q()
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{_DSA_TRITON_BLOCK_Q_BWD_ENV} must be positive, got {value}")
    return min(triton.next_power_of_2(value), 4)


def _key_block_kv_bwd_keys() -> int:
    raw = os.getenv(_DSA_TRITON_KEY_BLOCK_KV_BWD_KEYS_ENV)
    value = int(raw) if raw else 4
    if value <= 0:
        raise ValueError(f"{_DSA_TRITON_KEY_BLOCK_KV_BWD_KEYS_ENV} must be positive, got {value}")
    return min(triton.next_power_of_2(value), 16)


def _key_block_kv_bwd_queries() -> int:
    raw = os.getenv(_DSA_TRITON_KEY_BLOCK_KV_BWD_QUERIES_ENV)
    value = int(raw) if raw else 16
    if value <= 0:
        raise ValueError(
            f"{_DSA_TRITON_KEY_BLOCK_KV_BWD_QUERIES_ENV} must be positive, got {value}"
        )
    return min(triton.next_power_of_2(value), 32)


def _qtile_unique_kv_bwd_queries() -> int:
    raw = os.getenv(_DSA_TRITON_QTILE_UNIQUE_KV_BWD_Q_ENV)
    value = int(raw) if raw else 4
    if value <= 0:
        raise ValueError(
            f"{_DSA_TRITON_QTILE_UNIQUE_KV_BWD_Q_ENV} must be positive, got {value}"
        )
    return min(triton.next_power_of_2(value), 4)


def _qtile_unique_kv_bwd_topk() -> int:
    raw = os.getenv(_DSA_TRITON_QTILE_UNIQUE_KV_BWD_K_ENV)
    value = int(raw) if raw else 8
    if value <= 0:
        raise ValueError(
            f"{_DSA_TRITON_QTILE_UNIQUE_KV_BWD_K_ENV} must be positive, got {value}"
        )
    return min(triton.next_power_of_2(value), 16)


def _key_block_kv_bwd_search_steps(topk: int) -> int:
    if topk <= 1:
        return 1
    # The Triton lower-bound loop uses a fixed iteration count. Keep one guard
    # step beyond ceil(log2(topk)) so edge cases at the inclusive/exclusive
    # boundary converge exactly while still avoiding the old hard-coded 16
    # iterations for common topk values.
    return int(topk).bit_length()


def _indexer_block_size(env_name: str, default: int, maximum: int) -> int:
    raw = os.getenv(env_name)
    value = int(raw) if raw else default
    if value <= 0:
        raise ValueError(f"{env_name} must be positive, got {value}")
    return min(triton.next_power_of_2(value), maximum)


def _num_warps_from_env(env_name: str, default: int) -> int:
    raw = os.getenv(env_name)
    value = int(raw) if raw else default
    if value not in (1, 2, 4, 8):
        raise ValueError(f"{env_name} must be one of 1, 2, 4, or 8, got {value}")
    return value


def is_dsa_indexer_scores_triton_supported(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
) -> bool:
    """Return whether the fused DSA indexer score kernel can handle this call."""

    if not (_env_enabled() and _indexer_env_enabled()) or not HAVE_TRITON:
        return False
    if not (q.is_cuda and weights.is_cuda and k.is_cuda):
        return False
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if k.dtype != q.dtype or weights.dtype != q.dtype:
        return False
    if mask is not None or not is_causal:
        return False
    if (query_positions is None) != (key_positions is None):
        return False
    if query_positions is not None:
        if not (query_positions.is_cuda and key_positions.is_cuda):
            return False
        if query_positions.dim() != 1 or key_positions.dim() != 1:
            return False
        if query_positions.size(0) != q.size(0) or key_positions.size(0) != k.size(0):
            return False
    if q.dim() != 4 or weights.dim() != 3 or k.dim() != 3:
        return False
    if q.size(0) <= 0 or k.size(0) <= 0:
        return False
    if q.size(1) != k.size(1) or q.size(1) != weights.size(1):
        return False
    if q.size(2) != weights.size(2):
        return False
    if q.size(3) != k.size(2):
        return False
    if q.size(2) <= 0 or q.size(2) > 128:
        return False
    if q.size(3) <= 0 or q.size(3) > 256:
        return False
    return True


@triton.jit
def _dsa_indexer_scores_kernel(
    q_ptr,
    weights_ptr,
    k_ptr,
    query_pos_ptr,
    key_pos_ptr,
    scores_ptr,
    q_len: tl.constexpr,
    sk: tl.constexpr,
    num_index_heads: tl.constexpr,
    head_dim: tl.constexpr,
    q_start,
    q_stride_s: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    w_stride_s: tl.constexpr,
    w_stride_b: tl.constexpr,
    w_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_d: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_s: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_block = tl.program_id(0)
    k_block = tl.program_id(1)
    batch_idx = tl.program_id(2)

    q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    d_offsets = tl.arange(0, BLOCK_D)
    q_valid = q_offsets < q_len
    k_valid = k_offsets < sk

    scores = tl.zeros((BLOCK_Q, BLOCK_K), tl.float32)
    for head_idx in tl.range(0, num_index_heads):
        q_ptrs = (
            q_ptr
            + q_offsets[:, None] * q_stride_s
            + batch_idx * q_stride_b
            + head_idx * q_stride_h
            + d_offsets[None, :] * q_stride_d
        )
        k_ptrs = (
            k_ptr
            + k_offsets[None, :] * k_stride_s
            + batch_idx * k_stride_b
            + d_offsets[:, None] * k_stride_d
        )
        q_tile = tl.load(
            q_ptrs,
            mask=q_valid[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        k_tile = tl.load(
            k_ptrs,
            mask=k_valid[None, :] & (d_offsets[:, None] < head_dim),
            other=0.0,
        )
        head_scores = tl.dot(
            q_tile,
            k_tile,
            input_precision="tf32",
            out_dtype=tl.float32,
        )
        head_scores = tl.maximum(head_scores, 0.0)
        head_weights = tl.load(
            weights_ptr
            + q_offsets * w_stride_s
            + batch_idx * w_stride_b
            + head_idx * w_stride_h,
            mask=q_valid,
            other=0.0,
        ).to(tl.float32)
        scores += head_scores * head_weights[:, None]

    if HAS_POSITIONS:
        query_abs = tl.load(query_pos_ptr + q_offsets, mask=q_valid, other=0)
        key_abs = tl.load(key_pos_ptr + k_offsets, mask=k_valid, other=0)
    else:
        query_abs = q_start + q_offsets
        key_abs = k_offsets
    valid = q_valid[:, None] & k_valid[None, :] & (key_abs[None, :] <= query_abs[:, None])
    scores = tl.where(valid, scores, -float("inf"))

    out_ptrs = (
        scores_ptr
        + batch_idx * out_stride_b
        + q_offsets[:, None] * out_stride_s
        + k_offsets[None, :]
    )
    tl.store(out_ptrs, scores, mask=q_valid[:, None] & k_valid[None, :])


def dsa_indexer_scores_triton(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    q_start: int = 0,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused DSA indexer score construction for the no-indexer-loss path."""

    q_len, batch_size, num_index_heads, head_dim = q.shape
    sk = k.shape[0]
    if out is None:
        scores = torch.empty((batch_size, q_len, sk), device=q.device, dtype=torch.float32)
    else:
        expected_shape = (batch_size, q_len, sk)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"DSA indexer score output must have shape {expected_shape}, "
                f"got {tuple(out.shape)}"
            )
        if out.device != q.device or out.dtype != torch.float32:
            raise ValueError("DSA indexer score output must be a float32 tensor on q.device")
        scores = out
    has_positions = query_positions is not None
    if has_positions:
        query_positions = query_positions.contiguous()
        key_positions = key_positions.contiguous()
    else:
        query_positions = scores
        key_positions = scores

    block_q = _indexer_block_size(_DSA_TRITON_INDEXER_BLOCK_Q_ENV, 8, 16)
    block_k = _indexer_block_size(_DSA_TRITON_INDEXER_BLOCK_K_ENV, 64, 128)
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(q_len, block_q), triton.cdiv(sk, block_k), batch_size)

    _dsa_indexer_scores_kernel[grid](
        q,
        weights,
        k,
        query_positions,
        key_positions,
        scores,
        q_len,
        sk,
        num_index_heads,
        head_dim,
        int(q_start),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        scores.stride(0),
        scores.stride(1),
        HAS_POSITIONS=has_positions,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return scores


def is_hisa_attention_target_probs_triton_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    """Return whether the fused HISA target-probability kernel can handle this call."""

    if not (_env_enabled() and _hisa_target_env_enabled()) or not HAVE_TRITON:
        return False
    if not (query.is_cuda and key.is_cuda and topk_indices.is_cuda):
        return False
    if query.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if key.dtype != query.dtype:
        return False
    if topk_indices.dtype not in (torch.int16, torch.int32, torch.int64):
        return False
    if query.dim() != 4 or key.dim() != 4 or topk_indices.dim() != 3:
        return False
    if query.size(1) != key.size(1) or query.size(2) != key.size(2):
        return False
    if query.size(3) != key.size(3):
        return False
    if topk_indices.size(0) != query.size(1) or topk_indices.size(1) != query.size(0):
        return False
    if query.size(0) <= 0 or key.size(0) <= 0 or topk_indices.size(-1) <= 0:
        return False
    if query.size(2) <= 0 or query.size(2) > 256:
        return False
    if query.size(3) <= 0 or query.size(3) > 256:
        return False
    if topk_indices.size(-1) > key.size(0):
        return False
    return True


@triton.jit
def _hisa_target_lse_kernel(
    query_ptr,
    key_ptr,
    topk_ptr,
    lse_ptr,
    softmax_scale,
    topk_count: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    q_stride_s: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    topk_stride_b: tl.constexpr,
    topk_stride_s: tl.constexpr,
    topk_stride_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    d_offsets = tl.arange(0, BLOCK_D)
    topk_offsets = tl.arange(0, BLOCK_K)
    query = tl.load(
        query_ptr
        + q_idx * q_stride_s
        + batch_idx * q_stride_b
        + head_idx * q_stride_h
        + d_offsets * q_stride_d,
        mask=d_offsets < head_dim,
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    for topk_start in tl.range(0, topk_count, BLOCK_K):
        k_offsets = topk_start + topk_offsets
        valid_topk = k_offsets < topk_count
        selected = tl.load(
            topk_ptr
            + batch_idx * topk_stride_b
            + q_idx * topk_stride_s
            + k_offsets * topk_stride_k,
            mask=valid_topk,
            other=0,
        ).to(tl.int32)
        selected_valid = selected >= 0
        safe_selected = tl.maximum(selected, 0)
        key = tl.load(
            key_ptr
            + safe_selected[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + head_idx * k_stride_h
            + d_offsets[None, :] * k_stride_d,
            mask=valid_topk[:, None] & selected_valid[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(key * query[None, :], axis=1) * softmax_scale
        scores = tl.where(valid_topk & selected_valid, scores, -float("inf"))

        block_m = tl.max(scores, axis=0)
        has_valid_block = block_m > -float("inf")
        m_new = tl.where(has_valid_block, tl.maximum(m_i, block_m), m_i)
        safe_m_i = tl.where(has_valid_block, m_i, 0.0)
        safe_m_new = tl.where(has_valid_block, m_new, 0.0)
        safe_scores = tl.where(valid_topk & selected_valid, scores, safe_m_new)
        block_probs = tl.exp(safe_scores - safe_m_new)
        block_probs = tl.where(valid_topk & selected_valid, block_probs, 0.0)
        l_i = l_i * tl.where(has_valid_block, tl.exp(safe_m_i - safe_m_new), 1.0) + tl.sum(
            block_probs, axis=0
        )
        m_i = m_new

    row = batch_idx * tl.num_programs(0) + q_idx
    tl.store(lse_ptr + row * num_heads + head_idx, m_i + tl.log(l_i))


@triton.jit
def _hisa_target_probs_kernel(
    query_ptr,
    key_ptr,
    topk_ptr,
    lse_ptr,
    out_ptr,
    softmax_scale,
    topk_count: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    q_stride_s: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    topk_stride_b: tl.constexpr,
    topk_stride_s: tl.constexpr,
    topk_stride_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    topk_block = tl.program_id(2)

    d_offsets = tl.arange(0, BLOCK_D)
    topk_offsets = topk_block * BLOCK_K + tl.arange(0, BLOCK_K)
    valid_topk = topk_offsets < topk_count
    selected = tl.load(
        topk_ptr
        + batch_idx * topk_stride_b
        + q_idx * topk_stride_s
        + topk_offsets * topk_stride_k,
        mask=valid_topk,
        other=0,
    ).to(tl.int32)
    selected_valid = selected >= 0
    safe_selected = tl.maximum(selected, 0)

    row = batch_idx * tl.num_programs(0) + q_idx
    prob_sum = tl.zeros((BLOCK_K,), tl.float32)
    for head_idx in tl.range(0, num_heads):
        query = tl.load(
            query_ptr
            + q_idx * q_stride_s
            + batch_idx * q_stride_b
            + head_idx * q_stride_h
            + d_offsets * q_stride_d,
            mask=d_offsets < head_dim,
            other=0.0,
        ).to(tl.float32)
        key = tl.load(
            key_ptr
            + safe_selected[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + head_idx * k_stride_h
            + d_offsets[None, :] * k_stride_d,
            mask=valid_topk[:, None] & selected_valid[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(key * query[None, :], axis=1) * softmax_scale
        scores = tl.where(valid_topk & selected_valid, scores, -float("inf"))
        row_lse = tl.load(lse_ptr + row * num_heads + head_idx).to(tl.float32)
        probs = tl.exp(scores - row_lse)
        prob_sum += tl.where(valid_topk & selected_valid, probs, 0.0)

    tl.store(out_ptr + row * topk_count + topk_offsets, prob_sum, mask=valid_topk)


def hisa_attention_target_probs_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Fused HISA target probability construction over selected top-k attention keys."""

    q_len, bsz, num_heads, head_dim = query.shape
    topk_indices = _maybe_narrow_topk_indices(topk_indices, key.shape[0])
    topk_count = topk_indices.shape[-1]
    lse = torch.empty((bsz * q_len, num_heads), device=query.device, dtype=torch.float32)
    out = torch.empty((bsz * q_len, topk_count), device=query.device, dtype=torch.float32)
    block_k = _hisa_target_block_k(topk_count)
    block_d = triton.next_power_of_2(head_dim)

    _hisa_target_lse_kernel[(q_len, bsz, num_heads)](
        query,
        key,
        topk_indices,
        lse,
        float(softmax_scale),
        topk_count,
        head_dim,
        num_heads,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4,
    )
    _hisa_target_probs_kernel[(q_len, bsz, triton.cdiv(topk_count, block_k))](
        query,
        key,
        topk_indices,
        lse,
        out,
        float(softmax_scale),
        topk_count,
        head_dim,
        num_heads,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def hisa_attention_target_probs_from_lse_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Construct selected-teacher mass using LSE emitted by sparse DSA attention."""

    q_len, bsz, num_heads, head_dim = query.shape
    topk_indices = _maybe_narrow_topk_indices(topk_indices, key.shape[0])
    topk_count = topk_indices.shape[-1]
    expected_lse_shape = (bsz * q_len, num_heads)
    lse = lse.reshape(expected_lse_shape).contiguous()
    out = torch.empty((bsz * q_len, topk_count), device=query.device, dtype=torch.float32)
    block_k = _hisa_target_block_k(topk_count)
    block_d = triton.next_power_of_2(head_dim)

    _hisa_target_probs_kernel[(q_len, bsz, triton.cdiv(topk_count, block_k))](
        query,
        key,
        topk_indices,
        lse,
        out,
        float(softmax_scale),
        topk_count,
        head_dim,
        num_heads,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def is_hisa_kl_grad_triton_supported(
    selected_scores: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> bool:
    if not (_env_enabled() and _hisa_kl_grad_env_enabled()) or not HAVE_TRITON:
        return False
    if not (selected_scores.is_cuda and teacher_probs.is_cuda):
        return False
    if selected_scores.dtype != torch.float32 or teacher_probs.dtype != torch.float32:
        return False
    if selected_scores.dim() != 2 or teacher_probs.dim() != 2:
        return False
    if selected_scores.shape != teacher_probs.shape:
        return False
    if selected_scores.size(1) <= 0 or selected_scores.size(1) > 2048:
        return False
    return True


@triton.jit
def _hisa_kl_grad_kernel(
    selected_scores_ptr,
    teacher_probs_ptr,
    grad_scores_ptr,
    loss_ptr,
    num_rows: tl.constexpr,
    topk_count: tl.constexpr,
    stride_s_row: tl.constexpr,
    stride_t_row: tl.constexpr,
    stride_g_row: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    valid_k = offsets < topk_count

    scores = tl.load(
        selected_scores_ptr + row * stride_s_row + offsets,
        mask=valid_k,
        other=-float("inf"),
    ).to(tl.float32)
    teacher = tl.load(
        teacher_probs_ptr + row * stride_t_row + offsets,
        mask=valid_k,
        other=0.0,
    ).to(tl.float32)
    valid = valid_k & (scores > -3.0e38)
    scores = tl.where(valid, scores, -float("inf"))

    row_max = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - row_max)
    exp_scores = tl.where(valid, exp_scores, 0.0)
    denom = tl.sum(exp_scores, axis=0)
    index_probs = exp_scores / denom
    index_probs = tl.where(valid, index_probs, 0.0)
    teacher = tl.where(valid, teacher, 0.0)

    prob_ratio = index_probs / (index_probs + 1.0e-10)
    teacher_prob_ratio = teacher * prob_ratio
    row_scale = tl.sum(teacher_prob_ratio, axis=0)
    grad = index_probs * row_scale - teacher_prob_ratio
    grad = tl.where(valid, grad, 0.0)
    tl.store(grad_scores_ptr + row * stride_g_row + offsets, grad, mask=valid_k)

    kl = teacher * (tl.log(teacher + 1.0e-10) - tl.log(index_probs + 1.0e-10))
    kl = tl.where(valid, kl, 0.0)
    loss = tl.sum(kl, axis=0)
    tl.atomic_add(loss_ptr, loss, sem="relaxed")


def hisa_kl_loss_and_grad_triton(
    selected_scores: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute HISA indexer KL loss and dL/d(selected_scores) in one Triton launch."""

    if not is_hisa_kl_grad_triton_supported(selected_scores, teacher_probs):
        raise RuntimeError("hisa_kl_loss_and_grad_triton called for unsupported tensors")
    num_rows, topk_count = selected_scores.shape
    grad_scores = torch.empty_like(selected_scores)
    loss = torch.zeros((), device=selected_scores.device, dtype=torch.float32)
    block_k = _hisa_kl_block_k(topk_count)
    _hisa_kl_grad_kernel[(num_rows,)](
        selected_scores,
        teacher_probs,
        grad_scores,
        loss,
        num_rows,
        topk_count,
        selected_scores.stride(0),
        teacher_probs.stride(0),
        grad_scores.stride(0),
        BLOCK_K=block_k,
        num_warps=8,
    )
    return loss, grad_scores


def is_sparse_dsa_triton_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
) -> bool:
    """Return whether the fused sparse DSA attention kernel can handle this call."""

    if not _env_enabled() or not HAVE_TRITON:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda and topk_indices.is_cuda):
        return False
    if query.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if topk_indices.dtype not in (torch.int16, torch.int32, torch.int64):
        return False
    if (query_positions is None) != (key_positions is None):
        return False
    if query_positions is not None:
        if not (query_positions.is_cuda and key_positions.is_cuda):
            return False
        if query_positions.dim() != 1 or key_positions.dim() != 1:
            return False
        if query_positions.size(0) != query.size(0) or key_positions.size(0) != key.size(0):
            return False
    if mask is not None or not is_causal:
        return False
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False
    if topk_indices.dim() != 3:
        return False
    if query.size(1) <= 0 or key.size(1) != query.size(1) or value.size(1) != query.size(1):
        return False
    if topk_indices.size(0) != query.size(1) or topk_indices.size(1) != query.size(0):
        return False
    if query.size(2) != key.size(2) or query.size(2) != value.size(2):
        return False
    if query.size(3) != key.size(3):
        return False
    # DeepSeek-V3.2 MLA passes Q/K as qk_head_dim + qk_pos_emb_head_dim and V as
    # v_head_dim. Keep this first kernel bounded to the model shapes we need.
    if query.size(3) <= 0 or query.size(3) > 256:
        return False
    if value.size(3) <= 0 or value.size(3) > 256:
        return False
    if topk_indices.size(-1) <= 0 or topk_indices.size(-1) > key.size(0):
        return False
    return True


def is_sparse_dsa_teacher_triton_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
) -> bool:
    """Return whether selected attention can also emit indexer teacher mass."""

    return is_sparse_dsa_triton_supported(
        query,
        key,
        value,
        topk_indices,
        mask,
        is_causal,
        query_positions=query_positions,
        key_positions=key_positions,
    )


@triton.jit
def _sparse_dsa_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    topk_ptr,
    query_pos_ptr,
    key_pos_ptr,
    output_ptr,
    lse_ptr,
    teacher_ptr,
    teacher_score_ptr,
    softmax_scale,
    q_len: tl.constexpr,
    bsz: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    v_stride_s: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    topk_count: tl.constexpr,
    q_start,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_QD: tl.constexpr,
    BLOCK_VD: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
    EMIT_TEACHER: tl.constexpr,
    USE_TEACHER_SCORE_SCRATCH: tl.constexpr,
):
    q_block = tl.program_id(0)
    head_batch_idx = tl.program_id(1)
    head_idx = head_batch_idx % num_heads
    batch_idx = head_batch_idx // num_heads

    q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qd_offsets = tl.arange(0, BLOCK_QD)
    vd_offsets = tl.arange(0, BLOCK_VD)
    topk_offsets = tl.arange(0, BLOCK_K)
    q_valid = q_offsets < q_len

    q_ptrs = (
        query_ptr
        + ((q_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * head_dim
        + qd_offsets[None, :]
    )
    query = tl.load(q_ptrs, mask=q_valid[:, None] & (qd_offsets[None, :] < head_dim), other=0.0)
    query = query.to(tl.float32)

    m_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_Q,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_VD), tl.float32)
    if HAS_POSITIONS:
        q_abs = tl.load(query_pos_ptr + q_offsets, mask=q_valid, other=0)
    else:
        q_abs = q_start + q_offsets

    for topk_start in tl.range(0, topk_count, BLOCK_K):
        k_offsets = topk_start + topk_offsets
        valid_topk = k_offsets < topk_count
        selected = tl.load(
            topk_ptr
            + (batch_idx * q_len + q_offsets[:, None]) * topk_count
            + k_offsets[None, :],
            mask=q_valid[:, None] & valid_topk[None, :],
            other=0,
        ).to(tl.int32)
        selected_valid = selected >= 0
        safe_selected = tl.maximum(selected, 0)
        if HAS_POSITIONS:
            selected_abs = tl.load(
                key_pos_ptr + safe_selected,
                mask=q_valid[:, None] & valid_topk[None, :] & selected_valid,
                other=0,
            )
            valid = (
                q_valid[:, None]
                & valid_topk[None, :]
                & selected_valid
                & (selected_abs <= q_abs[:, None])
            )
        else:
            valid = (
                q_valid[:, None]
                & valid_topk[None, :]
                & selected_valid
                & (selected <= q_abs[:, None])
            )

        key_ptrs = (
            key_ptr
            + ((safe_selected[:, :, None] * bsz + batch_idx) * num_heads + head_idx) * head_dim
            + qd_offsets[None, None, :]
        )
        key = tl.load(
            key_ptrs,
            mask=valid[:, :, None] & (qd_offsets[None, None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(key * query[:, None, :], axis=2) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))
        if EMIT_TEACHER and USE_TEACHER_SCORE_SCRATCH:
            scratch_ptrs = (
                teacher_score_ptr
                + ((batch_idx * q_len + q_offsets[:, None]) * num_heads + head_idx)
                * topk_count
                + k_offsets[None, :]
            )
            tl.store(scratch_ptrs, scores, mask=q_valid[:, None] & valid_topk[None, :])

        block_m = tl.max(scores, axis=1)
        has_valid_block = block_m > -float("inf")
        m_new = tl.where(has_valid_block, tl.maximum(m_i, block_m), m_i)
        safe_m_i = tl.where(has_valid_block, m_i, 0.0)
        safe_m_new = tl.where(has_valid_block, m_new, 0.0)
        alpha = tl.where(has_valid_block, tl.exp(safe_m_i - safe_m_new), 1.0)
        safe_scores = tl.where(valid, scores, safe_m_new[:, None])
        probs = tl.exp(safe_scores - safe_m_new[:, None])
        probs = tl.where(valid, probs, 0.0)

        value_ptrs = (
            value_ptr
            + safe_selected[:, :, None] * v_stride_s
            + batch_idx * v_stride_b
            + head_idx * v_stride_h
            + vd_offsets[None, None, :] * v_stride_d
        )
        value = tl.load(
            value_ptrs,
            mask=valid[:, :, None] & (vd_offsets[None, None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha[:, None] + tl.sum(probs[:, :, None] * value, axis=1)
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        m_i = m_new

    output = acc / l_i[:, None]
    row_lse = m_i + tl.log(l_i)
    if EMIT_TEACHER:
        for topk_start in tl.range(0, topk_count, BLOCK_K):
            k_offsets = topk_start + topk_offsets
            valid_topk = k_offsets < topk_count
            if USE_TEACHER_SCORE_SCRATCH:
                scores = tl.load(
                    teacher_score_ptr
                    + ((batch_idx * q_len + q_offsets[:, None]) * num_heads + head_idx)
                    * topk_count
                    + k_offsets[None, :],
                    mask=q_valid[:, None] & valid_topk[None, :],
                    other=-float("inf"),
                )
                valid = q_valid[:, None] & valid_topk[None, :] & (scores > -3.0e38)
            else:
                selected = tl.load(
                    topk_ptr
                    + (batch_idx * q_len + q_offsets[:, None]) * topk_count
                    + k_offsets[None, :],
                    mask=q_valid[:, None] & valid_topk[None, :],
                    other=0,
                ).to(tl.int32)
                selected_valid = selected >= 0
                safe_selected = tl.maximum(selected, 0)
                if HAS_POSITIONS:
                    selected_abs = tl.load(
                        key_pos_ptr + safe_selected,
                        mask=q_valid[:, None] & valid_topk[None, :] & selected_valid,
                        other=0,
                    )
                    valid = (
                        q_valid[:, None]
                        & valid_topk[None, :]
                        & selected_valid
                        & (selected_abs <= q_abs[:, None])
                    )
                else:
                    valid = (
                        q_valid[:, None]
                        & valid_topk[None, :]
                        & selected_valid
                        & (selected <= q_abs[:, None])
                    )

                key_ptrs = (
                    key_ptr
                    + ((safe_selected[:, :, None] * bsz + batch_idx) * num_heads + head_idx)
                    * head_dim
                    + qd_offsets[None, None, :]
                )
                key = tl.load(
                    key_ptrs,
                    mask=valid[:, :, None] & (qd_offsets[None, None, :] < head_dim),
                    other=0.0,
                ).to(tl.float32)
                scores = tl.sum(key * query[:, None, :], axis=2) * softmax_scale
                scores = tl.where(valid, scores, -float("inf"))
            final_probs = tl.exp(scores - row_lse[:, None])
            final_probs = tl.where(valid, final_probs, 0.0)
            tl.atomic_add(
                teacher_ptr
                + (batch_idx * q_len + q_offsets[:, None]) * topk_count
                + k_offsets[None, :],
                final_probs,
                sem="relaxed",
                mask=valid,
            )

    output_ptrs = (
        output_ptr
        + ((q_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * value_dim
        + vd_offsets[None, :]
    )
    tl.store(output_ptrs, output, mask=q_valid[:, None] & (vd_offsets[None, :] < value_dim))
    tl.store(lse_ptr + (batch_idx * q_len + q_offsets) * num_heads + head_idx, row_lse, mask=q_valid)


@triton.jit
def _sparse_dsa_backward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    topk_ptr,
    query_pos_ptr,
    key_pos_ptr,
    output_ptr,
    lse_ptr,
    teacher_score_ptr,
    grad_output_ptr,
    grad_query_ptr,
    grad_key_ptr,
    grad_value_ptr,
    softmax_scale,
    q_len: tl.constexpr,
    bsz: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    v_stride_s: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    topk_count: tl.constexpr,
    q_start,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_QD: tl.constexpr,
    BLOCK_VD: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
    EMIT_KV_GRADS: tl.constexpr,
    GROUP_KV_GRADS: tl.constexpr,
    USE_SCORE_SCRATCH: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    q_block = tl.program_id(0)
    head_batch_idx = tl.program_id(1)
    head_idx = head_batch_idx % num_heads
    batch_idx = head_batch_idx // num_heads

    q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qd_offsets = tl.arange(0, BLOCK_QD)
    vd_offsets = tl.arange(0, BLOCK_VD)
    topk_offsets = tl.arange(0, BLOCK_K)
    q_valid = q_offsets < q_len

    q_ptrs = (
        query_ptr
        + ((q_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * head_dim
        + qd_offsets[None, :]
    )
    query = tl.load(q_ptrs, mask=q_valid[:, None] & (qd_offsets[None, :] < head_dim), other=0.0)
    query = query.to(tl.float32)

    grad_output_ptrs = (
        grad_output_ptr
        + ((q_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * value_dim
        + vd_offsets[None, :]
    )
    grad_output = tl.load(
        grad_output_ptrs,
        mask=q_valid[:, None] & (vd_offsets[None, :] < value_dim),
        other=0.0,
    ).to(tl.float32)

    output_ptrs = (
        output_ptr
        + ((q_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * value_dim
        + vd_offsets[None, :]
    )
    output = tl.load(
        output_ptrs,
        mask=q_valid[:, None] & (vd_offsets[None, :] < value_dim),
        other=0.0,
    ).to(tl.float32)
    row_lse = tl.load(
        lse_ptr + (batch_idx * q_len + q_offsets) * num_heads + head_idx,
        mask=q_valid,
        other=0.0,
    )
    row_lse = row_lse.to(tl.float32)
    delta = tl.sum(grad_output * output, axis=1)

    grad_query = tl.zeros((BLOCK_Q, BLOCK_QD), tl.float32)
    if HAS_POSITIONS:
        q_abs = tl.load(query_pos_ptr + q_offsets, mask=q_valid, other=0)
    else:
        q_abs = q_start + q_offsets

    for topk_start in tl.range(0, topk_count, BLOCK_K):
        k_offsets = topk_start + topk_offsets
        valid_topk = k_offsets < topk_count
        selected = tl.load(
            topk_ptr
            + (batch_idx * q_len + q_offsets[:, None]) * topk_count
            + k_offsets[None, :],
            mask=q_valid[:, None] & valid_topk[None, :],
            other=0,
        ).to(tl.int32)
        selected_valid = selected >= 0
        safe_selected = tl.maximum(selected, 0)
        if HAS_POSITIONS:
            selected_abs = tl.load(
                key_pos_ptr + safe_selected,
                mask=q_valid[:, None] & valid_topk[None, :] & selected_valid,
                other=0,
            )
            valid = (
                q_valid[:, None]
                & valid_topk[None, :]
                & selected_valid
                & (selected_abs <= q_abs[:, None])
            )
        else:
            valid = (
                q_valid[:, None]
                & valid_topk[None, :]
                & selected_valid
                & (selected <= q_abs[:, None])
            )

        key_ptrs = (
            key_ptr
            + ((safe_selected[:, :, None] * bsz + batch_idx) * num_heads + head_idx) * head_dim
            + qd_offsets[None, None, :]
        )
        key = tl.load(
            key_ptrs,
            mask=valid[:, :, None] & (qd_offsets[None, None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        if USE_SCORE_SCRATCH:
            scores = tl.load(
                teacher_score_ptr
                + ((batch_idx * q_len + q_offsets[:, None]) * num_heads + head_idx)
                * topk_count
                + k_offsets[None, :],
                mask=q_valid[:, None] & valid_topk[None, :],
                other=-float("inf"),
            ).to(tl.float32)
            valid = valid & (scores > -3.0e38)
        else:
            scores = tl.sum(key * query[:, None, :], axis=2) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))
        probs = tl.exp(scores - row_lse[:, None])
        probs = tl.where(valid, probs, 0.0)

        value_ptrs = (
            value_ptr
            + safe_selected[:, :, None] * v_stride_s
            + batch_idx * v_stride_b
            + head_idx * v_stride_h
            + vd_offsets[None, None, :] * v_stride_d
        )
        value = tl.load(
            value_ptrs,
            mask=valid[:, :, None] & (vd_offsets[None, None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)

        dp = tl.sum(value * grad_output[:, None, :], axis=2)
        ds = probs * (dp - delta[:, None]) * softmax_scale
        grad_query += tl.sum(ds[:, :, None] * key, axis=1)

        if EMIT_KV_GRADS and GROUP_KV_GRADS:
            edge_offsets = tl.arange(0, BLOCK_E)
            edge_q = edge_offsets // BLOCK_K
            edge_k = edge_offsets - edge_q * BLOCK_K
            selected_flat = tl.reshape(selected, (BLOCK_E,))
            safe_selected_flat = tl.reshape(safe_selected, (BLOCK_E,))
            valid_flat = tl.reshape(valid, (BLOCK_E,))
            ds_flat = tl.reshape(ds, (BLOCK_E,))
            probs_flat = tl.reshape(probs, (BLOCK_E,))
            same_key = (
                (selected_flat[:, None] == selected_flat[None, :])
                & valid_flat[:, None]
                & valid_flat[None, :]
            )
            has_prev = tl.max(
                tl.where(same_key & (edge_offsets[None, :] < edge_offsets[:, None]), 1, 0),
                axis=1,
            ) > 0
            representative = valid_flat & ~has_prev

            query_edges = tl.load(
                query_ptr
                + (
                    (
                        (q_block * BLOCK_Q + edge_q[:, None]) * bsz
                        + batch_idx
                    )
                    * num_heads
                    + head_idx
                )
                * head_dim
                + qd_offsets[None, :],
                mask=valid_flat[:, None] & (qd_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            grad_output_edges = tl.load(
                grad_output_ptr
                + (
                    (
                        (q_block * BLOCK_Q + edge_q[:, None]) * bsz
                        + batch_idx
                    )
                    * num_heads
                    + head_idx
                )
                * value_dim
                + vd_offsets[None, :],
                mask=valid_flat[:, None] & (vd_offsets[None, :] < value_dim),
                other=0.0,
            ).to(tl.float32)
            same_key_f = same_key.to(tl.float32)
            grad_key_acc = tl.dot(
                same_key_f,
                ds_flat[:, None] * query_edges,
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            grad_value_acc = tl.dot(
                same_key_f,
                probs_flat[:, None] * grad_output_edges,
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            tl.atomic_add(
                grad_key_ptr
                + ((safe_selected_flat[:, None] * bsz + batch_idx) * num_heads + head_idx)
                * head_dim
                + qd_offsets[None, :],
                grad_key_acc,
                sem="relaxed",
                mask=representative[:, None] & (qd_offsets[None, :] < head_dim),
            )
            tl.atomic_add(
                grad_value_ptr
                + ((safe_selected_flat[:, None] * bsz + batch_idx) * num_heads + head_idx)
                * value_dim
                + vd_offsets[None, :],
                grad_value_acc,
                sem="relaxed",
                mask=representative[:, None] & (vd_offsets[None, :] < value_dim),
            )
        elif EMIT_KV_GRADS:
            tl.atomic_add(
                grad_key_ptr
                + ((safe_selected[:, :, None] * bsz + batch_idx) * num_heads + head_idx)
                * head_dim
                + qd_offsets[None, None, :],
                ds[:, :, None] * query[:, None, :],
                sem="relaxed",
                mask=valid[:, :, None] & (qd_offsets[None, None, :] < head_dim),
            )
            tl.atomic_add(
                grad_value_ptr
                + ((safe_selected[:, :, None] * bsz + batch_idx) * num_heads + head_idx)
                * value_dim
                + vd_offsets[None, None, :],
                probs[:, :, None] * grad_output[:, None, :],
                sem="relaxed",
                mask=valid[:, :, None] & (vd_offsets[None, None, :] < value_dim),
            )

    grad_query_ptrs = (
        grad_query_ptr
        + ((q_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * head_dim
        + qd_offsets[None, :]
    )
    tl.store(grad_query_ptrs, grad_query, mask=q_valid[:, None] & (qd_offsets[None, :] < head_dim))


@triton.jit
def _sparse_dsa_backward_key_block_kv_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    topk_ptr,
    query_pos_ptr,
    key_pos_ptr,
    output_ptr,
    lse_ptr,
    grad_output_ptr,
    grad_key_ptr,
    grad_value_ptr,
    softmax_scale,
    q_len: tl.constexpr,
    bsz: tl.constexpr,
    sk: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    v_stride_s: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    topk_count: tl.constexpr,
    q_start,
    BLOCK_KEYS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_QD: tl.constexpr,
    BLOCK_VD: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
):
    key_block = tl.program_id(0)
    q_block = tl.program_id(1)
    head_batch_idx = tl.program_id(2)
    head_idx = head_batch_idx % num_heads
    batch_idx = head_batch_idx // num_heads

    key_offsets = key_block * BLOCK_KEYS + tl.arange(0, BLOCK_KEYS)
    q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qd_offsets = tl.arange(0, BLOCK_QD)
    vd_offsets = tl.arange(0, BLOCK_VD)

    key_valid = key_offsets < sk
    q_valid = q_offsets < q_len

    query = tl.load(
        query_ptr
        + ((q_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * head_dim
        + qd_offsets[None, :],
        mask=q_valid[:, None] & (qd_offsets[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    grad_output = tl.load(
        grad_output_ptr
        + ((q_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * value_dim
        + vd_offsets[None, :],
        mask=q_valid[:, None] & (vd_offsets[None, :] < value_dim),
        other=0.0,
    ).to(tl.float32)
    output = tl.load(
        output_ptr
        + ((q_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * value_dim
        + vd_offsets[None, :],
        mask=q_valid[:, None] & (vd_offsets[None, :] < value_dim),
        other=0.0,
    ).to(tl.float32)
    row_lse = tl.load(
        lse_ptr + (batch_idx * q_len + q_offsets) * num_heads + head_idx,
        mask=q_valid,
        other=0.0,
    )
    row_lse = row_lse.to(tl.float32)
    delta = tl.sum(grad_output * output, axis=1)

    grad_key_acc = tl.zeros((BLOCK_KEYS, BLOCK_QD), tl.float32)
    grad_value_acc = tl.zeros((BLOCK_KEYS, BLOCK_VD), tl.float32)
    local_keys = tl.arange(0, BLOCK_KEYS)

    if HAS_POSITIONS:
        q_abs = tl.load(query_pos_ptr + q_offsets, mask=q_valid, other=0)
    else:
        q_abs = q_start + q_offsets

    for local_key in tl.static_range(0, BLOCK_KEYS):
        selected_key = key_block * BLOCK_KEYS + local_key
        key_ok = selected_key < sk

        # The SFT launcher sorts selected top-k indices. Use that property to
        # avoid scanning every top-k chunk for every key block in this opt-in
        # key-block K/V backward path.
        lo = tl.full((BLOCK_Q,), 0, tl.int32)
        hi = tl.full((BLOCK_Q,), topk_count, tl.int32)
        for _ in tl.static_range(0, SEARCH_STEPS):
            mid = (lo + hi) // 2
            mid_value = tl.load(
                topk_ptr + (batch_idx * q_len + q_offsets) * topk_count + mid,
                mask=q_valid & (mid < topk_count),
                other=2147483647,
            ).to(tl.int32)
            go_right = mid_value < selected_key
            lo = tl.where(go_right, mid + 1, lo)
            hi = tl.where(go_right, hi, mid)
        found_value = tl.load(
            topk_ptr + (batch_idx * q_len + q_offsets) * topk_count + lo,
            mask=q_valid & (lo < topk_count),
            other=-1,
        ).to(tl.int32)
        present = found_value == selected_key
        if HAS_POSITIONS:
            key_abs = tl.load(key_pos_ptr + selected_key, mask=key_ok, other=-1)
        else:
            key_abs = selected_key
        causal = key_abs <= q_abs
        edge_valid = q_valid & key_ok & present & causal

        key_vec = tl.load(
            key_ptr + ((selected_key * bsz + batch_idx) * num_heads + head_idx) * head_dim + qd_offsets,
            mask=key_ok & (qd_offsets < head_dim),
            other=0.0,
        ).to(tl.float32)
        value_vec = tl.load(
            value_ptr
            + selected_key * v_stride_s
            + batch_idx * v_stride_b
            + head_idx * v_stride_h
            + vd_offsets * v_stride_d,
            mask=key_ok & (vd_offsets < value_dim),
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(query * key_vec[None, :], axis=1) * softmax_scale
        probs = tl.exp(scores - row_lse)
        probs = tl.where(edge_valid, probs, 0.0)
        dp = tl.sum(grad_output * value_vec[None, :], axis=1)
        ds = probs * (dp - delta) * softmax_scale
        ds = tl.where(edge_valid, ds, 0.0)

        grad_key_update = tl.sum(ds[:, None] * query, axis=0)
        grad_value_update = tl.sum(probs[:, None] * grad_output, axis=0)
        grad_key_acc += tl.where(
            local_keys[:, None] == local_key, grad_key_update[None, :], 0.0
        )
        grad_value_acc += tl.where(
            local_keys[:, None] == local_key, grad_value_update[None, :], 0.0
        )

    tl.atomic_add(
        grad_key_ptr
        + ((key_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * head_dim
        + qd_offsets[None, :],
        grad_key_acc,
        sem="relaxed",
        mask=key_valid[:, None]
        & (qd_offsets[None, :] < head_dim)
        & (grad_key_acc != 0.0),
    )
    tl.atomic_add(
        grad_value_ptr
        + ((key_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * value_dim
        + vd_offsets[None, :],
        grad_value_acc,
        sem="relaxed",
        mask=key_valid[:, None]
        & (vd_offsets[None, :] < value_dim)
        & (grad_value_acc != 0.0),
    )


@triton.jit
def _sparse_dsa_backward_qtile_unique_kv_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    topk_ptr,
    query_pos_ptr,
    key_pos_ptr,
    output_ptr,
    lse_ptr,
    grad_output_ptr,
    grad_key_ptr,
    grad_value_ptr,
    softmax_scale,
    q_len: tl.constexpr,
    bsz: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    v_stride_s: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    topk_count: tl.constexpr,
    q_start,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_QD: tl.constexpr,
    BLOCK_VD: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
):
    q_block = tl.program_id(0)
    topk_block = tl.program_id(1)
    head_batch_idx = tl.program_id(2)
    head_idx = head_batch_idx % num_heads
    batch_idx = head_batch_idx // num_heads

    edge_offsets = tl.arange(0, BLOCK_E)
    edge_q = edge_offsets // BLOCK_K
    edge_k = edge_offsets - edge_q * BLOCK_K
    q_offsets = q_block * BLOCK_Q + edge_q
    topk_offsets = topk_block * BLOCK_K + edge_k
    qd_offsets = tl.arange(0, BLOCK_QD)
    vd_offsets = tl.arange(0, BLOCK_VD)

    q_valid = q_offsets < q_len
    topk_valid = topk_offsets < topk_count
    selected = tl.load(
        topk_ptr + (batch_idx * q_len + q_offsets) * topk_count + topk_offsets,
        mask=q_valid & topk_valid,
        other=-1,
    ).to(tl.int32)
    selected_valid = selected >= 0
    safe_selected = tl.maximum(selected, 0)

    if HAS_POSITIONS:
        q_abs = tl.load(query_pos_ptr + q_offsets, mask=q_valid, other=0)
        selected_abs = tl.load(
            key_pos_ptr + safe_selected,
            mask=q_valid & topk_valid & selected_valid,
            other=0,
        )
        valid = q_valid & topk_valid & selected_valid & (selected_abs <= q_abs)
    else:
        q_abs = q_start + q_offsets
        valid = q_valid & topk_valid & selected_valid & (selected <= q_abs)

    query = tl.load(
        query_ptr
        + ((q_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * head_dim
        + qd_offsets[None, :],
        mask=valid[:, None] & (qd_offsets[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        key_ptr
        + ((safe_selected[:, None] * bsz + batch_idx) * num_heads + head_idx) * head_dim
        + qd_offsets[None, :],
        mask=valid[:, None] & (qd_offsets[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    value = tl.load(
        value_ptr
        + safe_selected[:, None] * v_stride_s
        + batch_idx * v_stride_b
        + head_idx * v_stride_h
        + vd_offsets[None, :] * v_stride_d,
        mask=valid[:, None] & (vd_offsets[None, :] < value_dim),
        other=0.0,
    ).to(tl.float32)
    grad_output = tl.load(
        grad_output_ptr
        + ((q_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * value_dim
        + vd_offsets[None, :],
        mask=valid[:, None] & (vd_offsets[None, :] < value_dim),
        other=0.0,
    ).to(tl.float32)
    output = tl.load(
        output_ptr
        + ((q_offsets[:, None] * bsz + batch_idx) * num_heads + head_idx) * value_dim
        + vd_offsets[None, :],
        mask=valid[:, None] & (vd_offsets[None, :] < value_dim),
        other=0.0,
    ).to(tl.float32)
    row_lse = tl.load(
        lse_ptr + (batch_idx * q_len + q_offsets) * num_heads + head_idx,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    scores = tl.sum(query * key, axis=1) * softmax_scale
    probs = tl.exp(scores - row_lse)
    probs = tl.where(valid, probs, 0.0)
    delta = tl.sum(grad_output * output, axis=1)
    dp = tl.sum(value * grad_output, axis=1)
    ds = probs * (dp - delta) * softmax_scale
    ds = tl.where(valid, ds, 0.0)

    edge_ids = tl.arange(0, BLOCK_E)
    same_key = (
        (selected[:, None] == selected[None, :])
        & valid[:, None]
        & valid[None, :]
    )
    has_prev = tl.max(
        tl.where(same_key & (edge_ids[None, :] < edge_ids[:, None]), 1, 0),
        axis=1,
    ) > 0
    representative = valid & ~has_prev
    same_key_f = same_key.to(tl.float32)

    grad_key_edges = ds[:, None] * query
    grad_value_edges = probs[:, None] * grad_output
    grad_key_acc = tl.dot(
        same_key_f,
        grad_key_edges,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    grad_value_acc = tl.dot(
        same_key_f,
        grad_value_edges,
        input_precision="ieee",
        out_dtype=tl.float32,
    )

    tl.atomic_add(
        grad_key_ptr
        + ((safe_selected[:, None] * bsz + batch_idx) * num_heads + head_idx) * head_dim
        + qd_offsets[None, :],
        grad_key_acc,
        sem="relaxed",
        mask=representative[:, None] & (qd_offsets[None, :] < head_dim),
    )
    tl.atomic_add(
        grad_value_ptr
        + ((safe_selected[:, None] * bsz + batch_idx) * num_heads + head_idx) * value_dim
        + vd_offsets[None, :],
        grad_value_acc,
        sem="relaxed",
        mask=representative[:, None] & (vd_offsets[None, :] < value_dim),
    )


class SparseDSAAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        topk_indices: torch.Tensor,
        softmax_scale: float,
        q_start: int,
        query_positions: torch.Tensor | None,
        key_positions: torch.Tensor | None,
        emit_teacher: bool,
        return_lse: bool,
    ) -> torch.Tensor:
        q_len, bsz, num_heads, head_dim = query.shape
        _, _, _, value_dim = value.shape
        topk_indices = _maybe_narrow_topk_indices(topk_indices, key.shape[0])
        topk_count = topk_indices.shape[-1]

        query_flat = query.contiguous()
        key_flat = key.contiguous()
        value_flat = value if value.stride(-1) == 1 else value.contiguous()
        topk_flat = topk_indices.contiguous()
        has_positions = query_positions is not None
        if has_positions:
            query_positions = query_positions.contiguous()
            key_positions = key_positions.contiguous()
        else:
            query_positions = topk_flat
            key_positions = topk_flat

        output = torch.empty(
            (q_len, bsz, num_heads, value_dim), device=query.device, dtype=query.dtype
        )
        lse = torch.empty((bsz * q_len, num_heads), device=query.device, dtype=torch.float32)
        teacher_probs = (
            torch.zeros((bsz * q_len, topk_count), device=query.device, dtype=torch.float32)
            if emit_teacher
            else output
        )
        teacher_score_scratch = (
            torch.empty(
                (bsz * q_len, num_heads, topk_count),
                device=query.device,
                dtype=torch.float32,
            )
            if emit_teacher and _teacher_score_scratch_enabled()
            else output
        )
        block_q = _sparse_block_q()
        block_k = _forward_block_k(topk_count)
        backward_block_k = _backward_block_k(topk_count)
        backward_num_warps = _num_warps_from_env(_DSA_TRITON_BWD_NUM_WARPS_ENV, 4)
        key_block_kv_backward = _key_block_kv_backward_enabled()
        if key_block_kv_backward and not _topk_sorted_ascending(topk_flat):
            key_block_kv_backward = False
        cuda_row_bwd_from_scores = _cuda_row_bwd_from_scores_supported(
            query_flat,
            key_flat,
            value_flat,
            topk_flat,
            teacher_score_scratch,
            bool(emit_teacher),
        )
        cuda_sorted_kv_backward = (
            not cuda_row_bwd_from_scores
            and _cuda_sorted_kv_backward_supported(
                query_flat,
                key_flat,
                value_flat,
                topk_flat,
                teacher_score_scratch,
                bool(emit_teacher),
                query_positions if has_positions else None,
                key_positions if has_positions else None,
            )
        )
        cuda_bwd_from_scores = (
            not cuda_row_bwd_from_scores
            and not cuda_sorted_kv_backward
        ) and _cuda_bwd_from_scores_supported(
            query_flat,
            key_flat,
            value_flat,
            topk_flat,
            teacher_score_scratch,
            bool(emit_teacher),
        )
        cuda_kv_backward = (
            not cuda_bwd_from_scores
            and
            _cuda_kv_backward_supported(
                query_flat,
                key_flat,
                value_flat,
                topk_flat,
                query_positions if has_positions else None,
                key_positions if has_positions else None,
            )
            and not key_block_kv_backward
            and not cuda_row_bwd_from_scores
            and not cuda_sorted_kv_backward
        )
        block_qd = triton.next_power_of_2(head_dim)
        block_vd = triton.next_power_of_2(value_dim)
        grid = (triton.cdiv(q_len, block_q), num_heads * bsz)

        _sparse_dsa_forward_kernel[grid](
            query_flat,
            key_flat,
            value_flat,
            topk_flat,
            query_positions,
            key_positions,
            output,
            lse,
            teacher_probs,
            teacher_score_scratch,
            float(softmax_scale),
            q_len,
            bsz,
            num_heads,
            head_dim,
            value_dim,
            value_flat.stride(0),
            value_flat.stride(1),
            value_flat.stride(2),
            value_flat.stride(3),
            topk_count,
            int(q_start),
            BLOCK_Q=block_q,
            BLOCK_K=block_k,
            BLOCK_QD=block_qd,
            BLOCK_VD=block_vd,
            HAS_POSITIONS=has_positions,
            EMIT_TEACHER=bool(emit_teacher),
            USE_TEACHER_SCORE_SCRATCH=bool(
                emit_teacher and _teacher_score_scratch_enabled()
            ),
            num_warps=4,
        )

        ctx.save_for_backward(
            query_flat,
            key_flat,
            value_flat,
            topk_flat,
            query_positions,
            key_positions,
            output,
            lse,
            teacher_score_scratch,
        )
        ctx.softmax_scale = float(softmax_scale)
        ctx.q_start = int(q_start)
        ctx.has_positions = has_positions
        ctx.num_heads = num_heads
        ctx.bsz = bsz
        ctx.head_dim = head_dim
        ctx.value_dim = value_dim
        ctx.topk_count = topk_count
        ctx.block_q = block_q
        ctx.block_k = block_k
        ctx.backward_block_q = _sparse_backward_block_q()
        ctx.backward_block_k = backward_block_k
        ctx.backward_num_warps = backward_num_warps
        ctx.key_block_kv_backward = (
            key_block_kv_backward
            and not cuda_bwd_from_scores
            and not cuda_row_bwd_from_scores
            and not cuda_sorted_kv_backward
        )
        ctx.qtile_unique_kv_backward = (
            _qtile_unique_kv_backward_enabled()
            and not key_block_kv_backward
            and not cuda_kv_backward
            and not cuda_bwd_from_scores
            and not cuda_row_bwd_from_scores
            and not cuda_sorted_kv_backward
        )
        ctx.cuda_kv_backward = cuda_kv_backward
        ctx.cuda_bwd_from_scores = cuda_bwd_from_scores
        ctx.cuda_row_bwd_from_scores = cuda_row_bwd_from_scores
        ctx.cuda_sorted_kv_backward = cuda_sorted_kv_backward
        ctx.backward_score_scratch = bool(
            emit_teacher
            and _teacher_score_scratch_enabled()
            and _triton_bwd_score_scratch_enabled()
            and not cuda_bwd_from_scores
            and not cuda_row_bwd_from_scores
        )
        ctx.grouped_kv_backward = (
            _grouped_kv_backward_enabled()
            and not key_block_kv_backward
            and not ctx.qtile_unique_kv_backward
            and not cuda_kv_backward
            and not cuda_bwd_from_scores
            and not cuda_row_bwd_from_scores
            and not cuda_sorted_kv_backward
        )
        ctx.cuda_kv_backward_tile_q = _cuda_kv_backward_tile_q()
        ctx.cuda_kv_backward_tile_k = _cuda_kv_backward_tile_k()
        ctx.qtile_unique_kv_backward_queries = _qtile_unique_kv_bwd_queries()
        ctx.qtile_unique_kv_backward_topk = _qtile_unique_kv_bwd_topk()
        ctx.key_block_kv_backward_keys = _key_block_kv_bwd_keys()
        ctx.key_block_kv_backward_queries = _key_block_kv_bwd_queries()
        ctx.key_block_kv_backward_search_steps = _key_block_kv_bwd_search_steps(topk_count)
        ctx.block_qd = block_qd
        ctx.block_vd = block_vd

        output = output.reshape(q_len, bsz, num_heads * value_dim)
        if emit_teacher:
            ctx.mark_non_differentiable(teacher_probs)
            return output, teacher_probs
        if return_lse:
            ctx.mark_non_differentiable(lse)
            return output, lse
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_teacher: torch.Tensor | None = None):
        (
            query,
            key,
            value,
            topk_indices,
            query_positions,
            key_positions,
            output,
            lse,
            teacher_score_scratch,
        ) = ctx.saved_tensors
        q_len = query.shape[0]
        sk = key.shape[0]
        num_heads = ctx.num_heads
        head_dim = ctx.head_dim
        value_dim = ctx.value_dim

        bsz = ctx.bsz
        grad_output = grad_output.reshape(q_len, bsz, num_heads, value_dim).contiguous()
        grad_query = torch.empty(
            (q_len, bsz, num_heads, head_dim), device=query.device, dtype=torch.float32
        )
        kv_grad_dtype = (
            key.dtype
            if _bf16_grad_atomics_enabled() and key.dtype in (torch.bfloat16, torch.float16)
            else torch.float32
        )
        grad_key = torch.zeros(
            (sk, bsz, num_heads, head_dim), device=key.device, dtype=kv_grad_dtype
        )
        grad_value = torch.zeros(
            (sk, bsz, num_heads, value_dim), device=value.device, dtype=kv_grad_dtype
        )
        if ctx.cuda_row_bwd_from_scores:
            grad_query.zero_()
            _dsa_sparse_backward_from_scores_row_cuda(
                query,
                key,
                value,
                topk_indices,
                teacher_score_scratch,
                output,
                lse,
                grad_output,
                grad_query,
                grad_key,
                grad_value,
                ctx.softmax_scale,
            )
        elif ctx.cuda_bwd_from_scores:
            grad_query.zero_()
            _dsa_sparse_backward_from_scores_cuda(
                query,
                key,
                value,
                topk_indices,
                teacher_score_scratch,
                output,
                lse,
                grad_output,
                grad_query,
                grad_key,
                grad_value,
                ctx.softmax_scale,
                ctx.cuda_kv_backward_tile_q,
                ctx.cuda_kv_backward_tile_k,
            )
        else:
            grid = (triton.cdiv(q_len, ctx.backward_block_q), num_heads * bsz)

            _sparse_dsa_backward_kernel[grid](
                query,
                key,
                value,
                topk_indices,
                query_positions,
                key_positions,
                output,
                lse,
                teacher_score_scratch,
                grad_output,
                grad_query,
                grad_key,
                grad_value,
                ctx.softmax_scale,
                q_len,
                bsz,
                num_heads,
                head_dim,
                value_dim,
                value.stride(0),
                value.stride(1),
                value.stride(2),
                value.stride(3),
                ctx.topk_count,
                ctx.q_start,
                BLOCK_Q=ctx.backward_block_q,
                BLOCK_K=ctx.backward_block_k,
                BLOCK_QD=ctx.block_qd,
                BLOCK_VD=ctx.block_vd,
                HAS_POSITIONS=ctx.has_positions,
                EMIT_KV_GRADS=not (
                    ctx.key_block_kv_backward
                    or ctx.qtile_unique_kv_backward
                    or ctx.cuda_kv_backward
                    or ctx.cuda_sorted_kv_backward
                ),
                GROUP_KV_GRADS=ctx.grouped_kv_backward,
                USE_SCORE_SCRATCH=ctx.backward_score_scratch,
                BLOCK_E=ctx.backward_block_q * ctx.backward_block_k,
                num_warps=ctx.backward_num_warps,
            )
        if ctx.cuda_kv_backward:
            _dsa_sparse_kv_backward_cuda(
                query,
                key,
                value,
                topk_indices,
                output,
                lse,
                grad_output,
                grad_key,
                grad_value,
                ctx.softmax_scale,
                ctx.q_start,
                ctx.cuda_kv_backward_tile_q,
                ctx.cuda_kv_backward_tile_k,
            )
        if ctx.cuda_sorted_kv_backward:
            _dsa_sparse_kv_backward_sorted_from_scores_cuda(
                query,
                key,
                value,
                topk_indices,
                teacher_score_scratch,
                output,
                lse,
                grad_output,
                grad_key,
                grad_value,
                ctx.softmax_scale,
            )
        if ctx.key_block_kv_backward:
            key_block_grid = (
                triton.cdiv(sk, ctx.key_block_kv_backward_keys),
                triton.cdiv(q_len, ctx.key_block_kv_backward_queries),
                num_heads * bsz,
            )
            _sparse_dsa_backward_key_block_kv_kernel[key_block_grid](
                query,
                key,
                value,
                topk_indices,
                query_positions,
                key_positions,
                output,
                lse,
                grad_output,
                grad_key,
                grad_value,
                ctx.softmax_scale,
                q_len,
                bsz,
                sk,
                num_heads,
                head_dim,
                value_dim,
                value.stride(0),
                value.stride(1),
                value.stride(2),
                value.stride(3),
                ctx.topk_count,
                ctx.q_start,
                BLOCK_KEYS=ctx.key_block_kv_backward_keys,
                BLOCK_Q=ctx.key_block_kv_backward_queries,
                BLOCK_QD=ctx.block_qd,
                BLOCK_VD=ctx.block_vd,
                SEARCH_STEPS=ctx.key_block_kv_backward_search_steps,
                HAS_POSITIONS=ctx.has_positions,
                num_warps=ctx.backward_num_warps,
            )
        if ctx.qtile_unique_kv_backward:
            qtile_grid = (
                triton.cdiv(q_len, ctx.qtile_unique_kv_backward_queries),
                triton.cdiv(ctx.topk_count, ctx.qtile_unique_kv_backward_topk),
                num_heads * bsz,
            )
            _sparse_dsa_backward_qtile_unique_kv_kernel[qtile_grid](
                query,
                key,
                value,
                topk_indices,
                query_positions,
                key_positions,
                output,
                lse,
                grad_output,
                grad_key,
                grad_value,
                ctx.softmax_scale,
                q_len,
                bsz,
                num_heads,
                head_dim,
                value_dim,
                value.stride(0),
                value.stride(1),
                value.stride(2),
                value.stride(3),
                ctx.topk_count,
                ctx.q_start,
                BLOCK_Q=ctx.qtile_unique_kv_backward_queries,
                BLOCK_K=ctx.qtile_unique_kv_backward_topk,
                BLOCK_E=ctx.qtile_unique_kv_backward_queries
                * ctx.qtile_unique_kv_backward_topk,
                BLOCK_QD=ctx.block_qd,
                BLOCK_VD=ctx.block_vd,
                HAS_POSITIONS=ctx.has_positions,
                num_warps=ctx.backward_num_warps,
            )

        return (
            grad_query.to(query.dtype),
            grad_key.to(key.dtype),
            grad_value.to(value.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def sparse_dsa_attention_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    q_start: int = 0,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused sparse DSA attention over already-selected top-k key/value positions."""

    return SparseDSAAttentionTriton.apply(
        query,
        key,
        value,
        topk_indices,
        softmax_scale,
        q_start,
        query_positions,
        key_positions,
        False,
        False,
    )


def sparse_dsa_attention_with_lse_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    q_start: int = 0,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused selected attention that also returns non-differentiable per-head LSE."""

    return SparseDSAAttentionTriton.apply(
        query,
        key,
        value,
        topk_indices,
        softmax_scale,
        q_start,
        query_positions,
        key_positions,
        False,
        True,
    )


def sparse_dsa_attention_with_teacher_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    q_start: int = 0,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused selected attention plus local teacher mass over selected top-k.

    The returned teacher tensor is the per-TP-rank sum over attention heads of
    softmax probabilities on the selected candidates. It is intentionally
    non-differentiable: the teacher is a target distribution for the DSA
    indexer, while the selected attention output keeps the normal Q/K/V
    backward path.
    """

    if _teacher_from_lse_enabled() and is_hisa_attention_target_probs_triton_supported(
        query, key, topk_indices
    ):
        output, lse = sparse_dsa_attention_with_lse_triton(
            query,
            key,
            value,
            topk_indices,
            softmax_scale,
            q_start,
            query_positions,
            key_positions,
        )
        teacher = hisa_attention_target_probs_from_lse_triton(
            query.detach(),
            key.detach(),
            topk_indices,
            lse,
            float(softmax_scale),
        )
        return output, teacher

    return SparseDSAAttentionTriton.apply(
        query,
        key,
        value,
        topk_indices,
        softmax_scale,
        q_start,
        query_positions,
        key_positions,
        True,
        False,
    )
