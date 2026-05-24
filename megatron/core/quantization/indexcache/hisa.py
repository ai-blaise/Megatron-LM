# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""HISA selector helpers for NVFP4 IndexCache DSA indexer.

The IndexCache quantizer remains responsible for producing the NVFP4
fake-quantized K tensor; HISA chooses a block-compressed candidate set before
the usual sparse DSA attention consumes token indices. The training helpers in
``megatron.core.extensions.hisa_indexer`` reuse this config surface for the
score-formula backward.
"""

from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core.fine_profile import fine_profile_range
from megatron.core.quantization.indexcache.autograd import (
    get_indexcache_nvfp4_packed_tensors,
)


_HISA_SELECTOR_CUDA_ENV = "MEGATRON_HISA_SELECTOR_CUDA"
_HISA_SELECTOR_BACKEND_ENV = "MEGATRON_HISA_SELECTOR_BACKEND"
_HISA_CANDIDATE_SLOT_GROUP_ENV = "MEGATRON_HISA_CANDIDATE_SLOT_GROUP"
_HISA_CANDIDATE_MAX_TEMP_MB_ENV = "MEGATRON_HISA_CANDIDATE_MAX_TEMP_MB"
_HISA_CANDIDATE_FREE_MEM_RESERVE_MB_ENV = "MEGATRON_HISA_CANDIDATE_FREE_MEM_RESERVE_MB"
_HISA_CANDIDATE_TRIM_CACHE_ENV = "MEGATRON_HISA_CANDIDATE_TRIM_CACHE"
_HISA_COMPACT_CANDIDATE_TOPK_ENV = "MEGATRON_HISA_COMPACT_CANDIDATE_TOPK"
_HISA_SELECTOR_ROW_CHUNK_ENV = "MEGATRON_HISA_SELECTOR_ROW_CHUNK"
_HISA_MEGAKERNEL_PARALLEL_REFINE_ENV = "MEGATRON_HISA_MEGAKERNEL_PARALLEL_REFINE"
_HISA_MEGAKERNEL_STREAMING_SCRATCH_ENV = "MEGATRON_HISA_MEGAKERNEL_STREAMING_SCRATCH"
_HISA_MEGAKERNEL_SERIAL_MAX_CANDIDATES = 8192
_HISA_MEGAKERNEL_PARALLEL_MAX_CANDIDATES = 8192
_HISA_ASSUME_SORTED_POSITIONS_ENV = "MEGATRON_HISA_ASSUME_SORTED_POSITIONS"
_HISA_BMM_FP32_ACCUM_TENSORCORES_ENV = "MEGATRON_HISA_BMM_FP32_ACCUM_TENSORCORES"
_HISA_BMM_CUBLASDX_REFINE_ENV = "MEGATRON_HISA_BMM_CUBLASDX_REFINE"


@dataclass(frozen=True)
class IndexCacheHISAConfig:
    """Configuration for the opt-in HISA IndexCache selector."""

    enabled: bool = False
    block_size: int = 128
    block_topk: int = 64
    compression_ratio: float = 4.0
    topk_tokens: int = 2048
    execution_mode: str = "optimized"
    fallback_to_dense_if_short: bool = True
    forced_boundary_blocks: tuple[str, ...] = ("first", "last")

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"HISA block_size must be positive, got {self.block_size}")
        if self.block_topk <= 0:
            raise ValueError(f"HISA block_topk must be positive, got {self.block_topk}")
        if self.compression_ratio < 0:
            raise ValueError(
                f"HISA compression_ratio must be non-negative, got {self.compression_ratio}"
            )
        if self.topk_tokens <= 0:
            raise ValueError(f"HISA topk_tokens must be positive, got {self.topk_tokens}")
        if self.execution_mode not in ("optimized", "reference", "compute_only"):
            raise ValueError(
                "HISA execution_mode must be one of "
                f"('optimized', 'reference', 'compute_only'), got {self.execution_mode!r}"
            )

    @property
    def is_enabled(self) -> bool:
        return self.enabled

    @property
    def is_optimized(self) -> bool:
        return self.execution_mode == "optimized"


def hisa_block_topk_counts(
    block_counts: torch.Tensor,
    *,
    block_size: int,
    topk_tokens: int,
    compression_ratio: float,
) -> tuple[torch.Tensor, int]:
    """Return the per-row dynamic HISA block budget.

    For compression-ratio mode this matches the accepted OP contract:
    ``m=ceil(M/compression_ratio)`` capped by ``M`` for ``t > k``. The
    candidate pool may be smaller than ``topk_tokens`` and is padded by the
    caller; ordinary dense selection is used only when the context fits top-k.
    """

    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive for dynamic HISA budgets.")
    if abs(compression_ratio - round(compression_ratio)) < 1e-6:
        ratio = int(round(compression_ratio))
        selected = torch.div(block_counts + ratio - 1, ratio, rounding_mode="floor")
    else:
        selected = torch.ceil(block_counts.float() / compression_ratio).to(torch.int32)
    selected = torch.minimum(selected, block_counts)
    selected = torch.where(block_counts > 0, selected, torch.zeros_like(selected))
    max_selected = int(selected.max().item()) if selected.numel() else 0
    return selected.to(torch.int32), max(1, max_selected)


def _forced_boundary_block_counts(
    block_counts: torch.Tensor, forced_boundary_blocks: tuple[str, ...]
) -> torch.Tensor:
    """Return the per-row number of unique forced HISA boundary blocks."""

    if not forced_boundary_blocks:
        return torch.zeros_like(block_counts, dtype=torch.int32)
    forced = set(forced_boundary_blocks)
    unknown = forced.difference({"first", "last", "last_minus_one"})
    if unknown:
        raise ValueError(f"Unknown HISA forced boundary block {next(iter(unknown))!r}.")

    counts = torch.zeros_like(block_counts, dtype=torch.int32)
    if "first" in forced:
        counts = counts + (block_counts >= 1).to(dtype=torch.int32)
    if "last" in forced:
        counts = counts + (block_counts >= 1).to(dtype=torch.int32)
        if "first" in forced:
            counts = counts - (block_counts == 1).to(dtype=torch.int32)
    if "last_minus_one" in forced:
        counts = counts + (block_counts >= 2).to(dtype=torch.int32)
        if "first" in forced:
            counts = counts - (block_counts == 2).to(dtype=torch.int32)
    return counts.clamp_min(0)


def _include_forced_boundary_budget(
    block_topk_counts: Optional[torch.Tensor],
    effective_block_topk: int,
    row_block_counts: torch.Tensor,
    forced_boundary_blocks: tuple[str, ...],
) -> tuple[Optional[torch.Tensor], int]:
    """Ensure the HISA block budget can actually contain forced boundary blocks."""

    if not forced_boundary_blocks:
        return block_topk_counts, effective_block_topk
    forced_counts = _forced_boundary_block_counts(row_block_counts, forced_boundary_blocks)
    if block_topk_counts is None:
        effective = max(int(effective_block_topk), int(forced_counts.max().item()))
        return None, effective
    adjusted = torch.maximum(block_topk_counts.to(torch.int32), forced_counts)
    adjusted = torch.minimum(adjusted, row_block_counts.to(torch.int32))
    effective = max(int(effective_block_topk), int(adjusted.max().item()))
    return adjusted, effective


def _weighted_relu_dsa_score(
    q_rows: torch.Tensor, k_rows: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    logits = torch.einsum("qhd,kd->qkh", q_rows.float(), k_rows.float())
    logits = torch.relu_(logits) * weights.float().unsqueeze(1)
    return logits.sum(dim=-1)


def _mean_pool_blocks(k_rows: torch.Tensor, block_size: int) -> torch.Tensor:
    if k_rows.numel() == 0:
        return k_rows.new_zeros((0, k_rows.shape[-1]))
    pad_len = (-k_rows.shape[0]) % block_size
    if pad_len:
        padded = torch.cat((k_rows, k_rows.new_zeros((pad_len, k_rows.shape[-1]))), dim=0)
        valid = torch.ones((k_rows.shape[0],), device=k_rows.device, dtype=k_rows.dtype)
        valid = torch.cat((valid, valid.new_zeros((pad_len,))), dim=0)
    else:
        padded = k_rows
        valid = torch.ones((k_rows.shape[0],), device=k_rows.device, dtype=k_rows.dtype)
    reps = padded.reshape(-1, block_size, k_rows.shape[-1]).sum(dim=1)
    counts = valid.reshape(-1, block_size).sum(dim=1).clamp_min_(1.0)
    return reps / counts[:, None]


def _forced_block_indices(
    block_count: int, names: tuple[str, ...], device: torch.device
) -> torch.Tensor:
    indices: set[int] = set()
    for name in names:
        if name == "first":
            indices.add(0)
        elif name == "last":
            indices.add(block_count - 1)
        elif name == "last_minus_one":
            if block_count >= 2:
                indices.add(block_count - 2)
        else:
            raise ValueError(f"Unknown HISA forced boundary block {name!r}.")
    return torch.tensor(
        sorted(i for i in indices if 0 <= i < block_count),
        device=device,
        dtype=torch.long,
    )


def _select_hisa_blocks(
    block_scores: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    block_topk: int,
    block_topk_counts: Optional[torch.Tensor],
    forced_boundary_blocks: tuple[str, ...],
) -> torch.Tensor:
    selected = torch.full(
        (block_scores.shape[0], block_topk),
        -1,
        device=block_scores.device,
        dtype=torch.long,
    )
    for row in range(block_scores.shape[0]):
        block_count = int(block_counts[row].item())
        if block_count <= 0:
            continue
        scores = block_scores[row, :block_count].clone()
        forced = _forced_block_indices(block_count, forced_boundary_blocks, scores.device)
        if forced.numel() > 0:
            scores[forced] = float("inf")
        row_topk = (
            int(block_topk_counts[row].item())
            if block_topk_counts is not None
            else block_topk
        )
        keep = min(row_topk, block_topk, block_count)
        selected[row, :keep] = torch.topk(scores, k=keep, sorted=False).indices
    return selected


def _dense_qk_topk_for_chunk(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk: int,
    *,
    is_causal: bool,
    q_start: int,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sq, bsz, _, _ = q.shape
    sk = k.shape[0]
    topk_k = min(topk, sk)
    scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    scores = torch.relu_(scores)
    scores.mul_(weights.unsqueeze(-1).float())
    scores = scores.sum(dim=2).transpose(0, 1)
    if is_causal:
        if query_positions is not None or key_positions is not None:
            if query_positions is None or key_positions is None:
                return None
            q_pos = query_positions.to(device=q.device, dtype=torch.long).view(1, sq, 1)
            k_pos = key_positions.to(device=q.device, dtype=torch.long).view(1, 1, sk)
        else:
            q_pos = torch.arange(q_start, q_start + sq, device=q.device).view(1, sq, 1)
            k_pos = torch.arange(sk, device=q.device).view(1, 1, sk)
        scores.masked_fill_(k_pos > q_pos, float("-inf"))
    return scores.topk(topk_k, dim=-1, sorted=False).indices.to(torch.int32)


def _mean_pool_all_blocks(k_rows: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool all contiguous K blocks for one batch item."""

    block_count = int(math.ceil(k_rows.shape[0] / block_size))
    pad_len = block_count * block_size - k_rows.shape[0]
    if pad_len:
        padded = torch.cat(
            (k_rows, k_rows.new_zeros((pad_len, k_rows.shape[-1]))), dim=0
        )
        counts = k_rows.new_full((block_count,), float(block_size))
        counts[-1] = float(block_size - pad_len)
    else:
        padded = k_rows
        counts = k_rows.new_full((block_count,), float(block_size))
    reps = padded.reshape(block_count, block_size, k_rows.shape[-1]).sum(dim=1)
    return reps / counts[:, None].clamp_min_(1.0)


def _prefix_lens_for_hisa_chunk(
    sq: int,
    sk: int,
    *,
    q_start: int,
    is_causal: bool,
    query_positions: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if is_causal:
        if query_positions is not None or key_positions is not None:
            if query_positions is None or key_positions is None:
                return None
            if query_positions.dim() != 1 or key_positions.dim() != 1:
                return None
            if query_positions.numel() != sq or key_positions.numel() != sk:
                return None
            query_positions = query_positions.to(device=device, dtype=torch.long).contiguous()
            key_positions = key_positions.to(device=device, dtype=torch.long).contiguous()
            if (
                os.getenv(_HISA_ASSUME_SORTED_POSITIONS_ENV, "1").strip().lower()
                in {"0", "false", "off", "no"}
                and key_positions.numel() > 1
                and bool((key_positions[1:] < key_positions[:-1]).any().item())
            ):
                return None
            prefix_lens = torch.searchsorted(key_positions, query_positions, right=True)
        else:
            prefix_lens = torch.arange(q_start + 1, q_start + sq + 1, device=device)
        return prefix_lens.clamp_(0, sk).to(torch.long)
    return torch.full((sq,), sk, device=device, dtype=torch.long)


def _hisa_selector_cuda_enabled() -> bool:
    backend = _hisa_selector_backend()
    if backend == "bmm":
        return False
    return os.getenv(_HISA_SELECTOR_CUDA_ENV, "0").lower() not in ("0", "false", "no")


def _hisa_selector_backend() -> str:
    raw = os.getenv(_HISA_SELECTOR_BACKEND_ENV, "auto").strip().lower()
    aliases = {
        "": "auto",
        "default": "auto",
        "tensorcore": "bmm",
        "tensor_core": "bmm",
        "tc": "bmm",
        "torch": "bmm",
        "cublas": "bmm",
        "blackwell": "megakernel",
        "sm100": "megakernel",
        "cute": "megakernel",
        "cublasdx_mega": "megakernel",
        "mega": "megakernel",
        "batched": "megakernel",
    }
    raw = aliases.get(raw, raw)
    if raw in ("packed", "nvfp4", "packed_nvfp4", "nvfp4_cuda"):
        raw = "packed_cuda"
    if raw in ("cublasdx", "packed_cublasdx", "nvfp4_cublasdx"):
        raw = "packed_cublasdx"
    if raw in (
        "tiled_cublasdx",
        "packed_tiled_cublasdx",
        "packed_cublasdx_tiled",
        "nvfp4_cublasdx_tiled",
    ):
        raw = "packed_cublasdx_tiled"
    if raw in (
        "fp8_cublasdx",
        "packed_fp8_cublasdx",
        "nvfp4_fp8_cublasdx",
    ):
        raw = "packed_cublasdx_fp8"
    if raw in (
        "deepgemm",
        "packed_deepgemm",
        "nvfp4_deepgemm",
        "fp4_deepgemm",
    ):
        raw = "deepgemm"
    if raw not in (
        "auto",
        "cuda",
        "bmm",
        "packed_cuda",
        "packed_cublasdx",
        "packed_cublasdx_tiled",
        "packed_cublasdx_fp8",
        "deepgemm",
        "megakernel",
    ):
        raise ValueError(
            f"{_HISA_SELECTOR_BACKEND_ENV} must be one of auto, cuda, bmm, packed_cuda, "
            f"packed_cublasdx, packed_cublasdx_tiled, packed_cublasdx_fp8, deepgemm, "
            f"megakernel; "
            f"got {raw!r}"
        )
    return raw


def indexcache_hisa_selector_backend_name() -> str:
    """Return the normalized HISA selector backend name used by this process."""

    return _hisa_selector_backend()


def _hisa_candidate_slot_group() -> int:
    raw = os.getenv(_HISA_CANDIDATE_SLOT_GROUP_ENV, "8")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{_HISA_CANDIDATE_SLOT_GROUP_ENV} must be positive, got {value}")
    return value


def _hisa_candidate_max_temp_bytes() -> int | None:
    raw = os.getenv(_HISA_CANDIDATE_MAX_TEMP_MB_ENV, "32").strip().lower()
    if raw in ("0", "false", "off", "no", ""):
        return None
    value_mb = int(raw)
    if value_mb < 0:
        raise ValueError(f"{_HISA_CANDIDATE_MAX_TEMP_MB_ENV} must be non-negative, got {value_mb}")
    return value_mb * 1024 * 1024


def _hisa_candidate_free_mem_reserve_bytes() -> int:
    raw = os.getenv(_HISA_CANDIDATE_FREE_MEM_RESERVE_MB_ENV, "64").strip().lower()
    if raw in ("0", "false", "off", "no", ""):
        return 0
    value_mb = int(raw)
    if value_mb < 0:
        raise ValueError(
            f"{_HISA_CANDIDATE_FREE_MEM_RESERVE_MB_ENV} must be non-negative, got {value_mb}"
        )
    return value_mb * 1024 * 1024


def _hisa_maybe_trim_cuda_cache(required_bytes: int, device: torch.device) -> None:
    """Release cached CUDA blocks before tight HISA scratch allocations.

    HISA runs inside already-large DSA/PP forward waves. When the allocator has
    enough cached-but-unallocated memory, a small candidate scratch can still
    fail if the driver-visible free pool is nearly empty. Only trim when the
    expected scratch plus the configured reserve will not fit.
    """

    raw = os.getenv(_HISA_CANDIDATE_TRIM_CACHE_ENV, "0").strip().lower()
    if raw in ("0", "false", "off", "no", ""):
        return
    if required_bytes <= 0 or not torch.cuda.is_available() or device.type != "cuda":
        return
    try:
        with torch.cuda.device(device):
            free_bytes, _ = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
    except (RuntimeError, AssertionError):
        return
    reserve_bytes = _hisa_candidate_free_mem_reserve_bytes()
    if int(free_bytes) >= int(required_bytes) + int(reserve_bytes):
        return
    cached_bytes = max(0, int(reserved) - int(allocated))
    if cached_bytes <= 0:
        return
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def _hisa_compact_candidate_topk_enabled() -> bool:
    raw = os.getenv(_HISA_COMPACT_CANDIDATE_TOPK_ENV, "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _hisa_effective_candidate_slot_group(
    requested_slot_group: int,
    *,
    rows: int,
    block_size: int,
    num_heads: int,
    head_dim: int,
    k_dtype: torch.dtype,
) -> int:
    """Return the HISA BMM slot group after any explicit scratch cap.

    ``MEGATRON_HISA_CANDIDATE_MAX_TEMP_MB=0`` disables the cap. The GCP A4
    training profile uses that no-cap mode so candidate refinement is not
    artificially serialized by memory-safety defaults from smaller runs.
    """

    slot_group = max(1, int(requested_slot_group))
    max_temp_bytes = _hisa_candidate_max_temp_bytes()
    if max_temp_bytes is None:
        return slot_group
    if torch.cuda.is_available():
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except (RuntimeError, AssertionError):
            free_bytes = 0
        if free_bytes > 0:
            reserve_bytes = _hisa_candidate_free_mem_reserve_bytes()
            free_cap = max(0, int(free_bytes) - int(reserve_bytes))
            if max_temp_bytes > 0:
                max_temp_bytes = min(max_temp_bytes, free_cap)
            else:
                max_temp_bytes = free_cap
    if max_temp_bytes <= 0:
        return 1
    try:
        k_element_size = torch.empty((), dtype=k_dtype).element_size()
    except TypeError:
        k_element_size = 4
    # Candidate refinement materializes gathered candidate K plus the BMM output:
    #   candidate_k: [rows, slot_group * block_size, head_dim] in k dtype
    #   cand_dot:    [rows, num_heads, slot_group * block_size] in fp32
    bytes_per_slot = int(rows) * int(block_size) * (
        int(head_dim) * int(k_element_size) + int(num_heads) * 4
    )
    if bytes_per_slot <= 0:
        return slot_group
    return max(1, min(slot_group, max_temp_bytes // bytes_per_slot))


def _hisa_selector_row_chunk(num_rows: int) -> int:
    raw = os.getenv(_HISA_SELECTOR_ROW_CHUNK_ENV, "256")
    value = int(raw)
    if value <= 0:
        return max(1, num_rows)
    return min(max(1, value), max(1, num_rows))


def _next_power_of_two_int(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _hisa_cublasdx_refine_scratch_bytes(rows: int, candidate_capacity: int, topk_k: int) -> int:
    # candidate_scores, candidate_indices, topk_indices, and selected_scores.
    return int(rows) * (
        int(candidate_capacity) * 8
        + int(topk_k) * 8
    )


def _hisa_cublasdx_refine_live_row_chunk(
    rows: int,
    candidate_capacity: int,
    topk_k: int,
    device: torch.device,
) -> int:
    if rows <= 1 or not torch.cuda.is_available() or device.type != "cuda":
        return max(1, int(rows))
    try:
        with torch.cuda.device(device):
            free_bytes, _ = torch.cuda.mem_get_info()
    except (RuntimeError, AssertionError):
        return max(1, int(rows))
    budget = max(0, int(free_bytes) - _hisa_candidate_free_mem_reserve_bytes())
    required = _hisa_cublasdx_refine_scratch_bytes(rows, candidate_capacity, topk_k)
    if required <= budget:
        return int(rows)
    bytes_per_row = _hisa_cublasdx_refine_scratch_bytes(1, candidate_capacity, topk_k)
    if bytes_per_row <= 0:
        return int(rows)
    # Leave allocator and library workspaces breathing room; this split is only
    # active under tight live-memory pressure.
    return max(1, min(int(rows), int((budget * 3 // 4) // bytes_per_row)))


def _hisa_bmm_fp32_accum_tensorcores_enabled() -> bool:
    raw = os.getenv(_HISA_BMM_FP32_ACCUM_TENSORCORES_ENV, "0").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _hisa_bmm_cublasdx_refine_enabled() -> bool:
    raw = os.getenv(_HISA_BMM_CUBLASDX_REFINE_ENV, "0").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _hisa_megakernel_parallel_refine_enabled() -> bool:
    raw = os.getenv(_HISA_MEGAKERNEL_PARALLEL_REFINE_ENV, "0").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _hisa_megakernel_streaming_scratch(_topk_tokens: int) -> Optional[int]:
    """Optional HISA candidate scratch cap for the streaming megakernel refine path."""

    raw = os.getenv(_HISA_MEGAKERNEL_STREAMING_SCRATCH_ENV, "0").strip().lower()
    if raw in ("", "0", "false", "off", "no"):
        return None
    raise RuntimeError(
        f"{_HISA_MEGAKERNEL_STREAMING_SCRATCH_ENV} is disabled for the production "
        "HISA selector path; the validated megakernel is hard-capped at 8192 candidates"
    )


def _hisa_bmm_fp32_accum(batch_a: torch.Tensor, batch_b: torch.Tensor) -> torch.Tensor:
    """Batched matmul with FP32 output, optionally using BF16/FP16 inputs.

    The default path casts operands to FP32 before matmul to preserve the exact
    selector behavior we have already validated. The opt-in path keeps BF16/FP16
    operands and asks PyTorch/cuBLAS for FP32 output accumulation, which can map
    to tensor cores on Blackwell. It is gated because BF16 multiplication can
    perturb near-tie top-k decisions even when accumulation and output are FP32.
    """

    if (
        _hisa_bmm_fp32_accum_tensorcores_enabled()
        and batch_a.is_cuda
        and batch_b.is_cuda
        and batch_a.dtype == batch_b.dtype
        and batch_a.dtype in (torch.bfloat16, torch.float16)
    ):
        return torch.bmm(batch_a.contiguous(), batch_b.contiguous(), out_dtype=torch.float32)
    return torch.bmm(batch_a.float(), batch_b.float())


def _try_load_hisa_cuda_ext():
    try:
        from megatron.core.extensions.hisa_indexer.kernels.build import get_ext

        return get_ext()
    except (ImportError, RuntimeError, OSError):
        return None


def _is_blackwell_or_newer(device: torch.device | int | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return major >= 10


def _selected_hisa_scores(
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    sq, _, head_dim = q_rows.shape
    topk_k = topk_indices.shape[-1]
    row_chunk = _hisa_selector_row_chunk(sq)
    scores_out = q_rows.new_empty((sq, topk_k), dtype=torch.float32)
    for row_start in range(0, sq, row_chunk):
        row_end = min(row_start + row_chunk, sq)
        q_chunk = q_rows[row_start:row_end]
        weights_chunk = weights_rows[row_start:row_end]
        indices_chunk = topk_indices[row_start:row_end]
        valid = indices_chunk >= 0
        safe_idx = indices_chunk.clamp_min(0).reshape(-1)
        selected_k = k_rows.index_select(0, safe_idx).view(row_end - row_start, topk_k, head_dim)
        dot = torch.einsum("qhd,qkd->qkh", q_chunk, selected_k)
        scores = (torch.relu(dot) * weights_chunk.unsqueeze(1)).sum(dim=-1)
        scores_out[row_start:row_end] = scores.masked_fill(~valid, float("-inf"))
    return scores_out


def _hisa_dense_cublasdx_candidate_topk(
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    top_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    topk_k: int,
    block_size: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Refine HISA BMM block candidates without materializing gathered K rows."""

    if not _hisa_bmm_cublasdx_refine_enabled():
        return None
    if not (q_rows.is_cuda and k_rows.is_cuda and weights_rows.is_cuda):
        return None
    if q_rows.dtype != torch.float32 or k_rows.dtype != torch.float32:
        return None
    sq, heads, head_dim = q_rows.shape
    if heads != 64 or head_dim != 128:
        return None
    if top_blocks.numel() == 0:
        return None
    ext = _try_load_hisa_cuda_ext()
    if ext is None or not hasattr(ext, "hisa_selector_dense_cublasdx_refine_fwd"):
        return None

    top_blocks_i32 = top_blocks.to(device=q_rows.device, dtype=torch.int32).contiguous()
    prefix_i32 = prefix_lens.to(device=q_rows.device, dtype=torch.int32).contiguous()
    candidate_count = int(top_blocks_i32.shape[1]) * int(block_size)
    candidate_capacity = _next_power_of_two_int(max(int(topk_k), candidate_count))
    live_row_chunk = _hisa_cublasdx_refine_live_row_chunk(
        int(sq), int(candidate_capacity), int(topk_k), q_rows.device
    )
    if live_row_chunk < int(sq):
        topk_indices = torch.empty((sq, int(topk_k)), device=q_rows.device, dtype=torch.int32)
        selected_scores = torch.empty((sq, int(topk_k)), device=q_rows.device, dtype=torch.float32)
        for row_start in range(0, int(sq), live_row_chunk):
            row_end = min(row_start + live_row_chunk, int(sq))
            chunk_result = _hisa_dense_cublasdx_candidate_topk(
                q_rows[row_start:row_end],
                weights_rows[row_start:row_end],
                k_rows,
                top_blocks[row_start:row_end],
                prefix_lens[row_start:row_end],
                topk_k,
                block_size,
            )
            if chunk_result is None:
                return None
            chunk_indices, chunk_scores = chunk_result
            topk_indices[row_start:row_end] = chunk_indices
            selected_scores[row_start:row_end] = chunk_scores
        return topk_indices, selected_scores

    candidate_scores = torch.empty(
        (sq, candidate_capacity), device=q_rows.device, dtype=torch.float32
    )
    candidate_indices = torch.empty(
        (sq, candidate_capacity), device=q_rows.device, dtype=torch.int32
    )
    topk_indices = torch.empty((sq, int(topk_k)), device=q_rows.device, dtype=torch.int32)
    selected_scores = torch.empty((sq, int(topk_k)), device=q_rows.device, dtype=torch.float32)

    ext.hisa_selector_dense_cublasdx_refine_fwd(
        q_rows.contiguous(),
        k_rows.contiguous(),
        weights_rows.contiguous(),
        prefix_i32,
        top_blocks_i32,
        candidate_scores,
        candidate_indices,
        topk_indices,
        selected_scores,
        int(block_size),
        int(topk_k),
    )
    return topk_indices, selected_scores


def _hisa_grouped_candidate_topk(
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    top_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    topk_k: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score HISA candidate blocks in tensor-core-friendly block-slot groups."""

    sq, _, head_dim = q_rows.shape
    row_chunk = _hisa_selector_row_chunk(sq)
    if row_chunk < sq:
        indices_out = torch.empty((sq, int(topk_k)), device=q_rows.device, dtype=torch.int32)
        scores_out = q_rows.new_empty((sq, int(topk_k)), dtype=torch.float32)
        for row_start in range(0, sq, row_chunk):
            row_end = min(row_start + row_chunk, sq)
            indices, scores = _hisa_grouped_candidate_topk(
                q_rows[row_start:row_end],
                weights_rows[row_start:row_end],
                k_rows,
                top_blocks[row_start:row_end],
                prefix_lens[row_start:row_end],
                topk_k,
                block_size,
            )
            indices_out[row_start:row_end] = indices
            scores_out[row_start:row_end] = scores
        return indices_out, scores_out

    sk = k_rows.shape[0]
    with fine_profile_range("hisa.candidate_refine.cublasdx_try"):
        cublasdx_result = _hisa_dense_cublasdx_candidate_topk(
            q_rows,
            weights_rows,
            k_rows,
            top_blocks,
            prefix_lens,
            topk_k,
            block_size,
        )
    if cublasdx_result is not None:
        return cublasdx_result

    offsets = torch.arange(block_size, device=q_rows.device, dtype=torch.int32)
    requested_slot_group = min(_hisa_candidate_slot_group(), max(1, top_blocks.shape[1]))
    slot_group = _hisa_effective_candidate_slot_group(
        requested_slot_group,
        rows=sq,
        block_size=block_size,
        num_heads=q_rows.shape[1],
        head_dim=head_dim,
        k_dtype=k_rows.dtype,
    )

    if _hisa_compact_candidate_topk_enabled():
        candidate_width = int(top_blocks.shape[1]) * int(block_size)
        candidate_capacity = max(int(topk_k), candidate_width)
        slot_width = max(1, int(slot_group)) * int(block_size)
        try:
            k_element_size = torch.empty((), dtype=k_rows.dtype).element_size()
        except TypeError:
            k_element_size = 4
        score_scratch_bytes = int(sq) * candidate_capacity * 4
        # Per slot group: candidate K, QK scores, candidate score vector,
        # candidate indices/safe indices, and validity masks. This is an
        # intentionally conservative estimate for deciding whether to return
        # cached CUDA blocks before entering the HISA refinement loop.
        group_scratch_bytes = int(sq) * slot_width * (
            int(head_dim) * int(k_element_size)
            + int(q_rows.shape[1]) * 4
            + 4
            + 8
            + 2
        )
        _hisa_maybe_trim_cuda_cache(
            score_scratch_bytes + group_scratch_bytes,
            q_rows.device,
        )
        with fine_profile_range("hisa.candidate_refine.compact_alloc_scores"):
            all_scores = q_rows.new_full((sq, candidate_capacity), float("-inf"))
        for slot_start in range(0, top_blocks.shape[1], slot_group):
            slot_end = min(slot_start + slot_group, top_blocks.shape[1])
            with fine_profile_range(f"hisa.candidate_refine.compact_slot.{slot_start}_{slot_end}"):
                block_ids = top_blocks[:, slot_start:slot_end].to(torch.int32)
                valid_block = block_ids >= 0
                cand_idx = block_ids.clamp_min(0).unsqueeze(-1) * block_size + offsets.view(
                    1, 1, -1
                )
                valid = (
                    valid_block.unsqueeze(-1)
                    & (cand_idx < prefix_lens.view(-1, 1, 1))
                    & (cand_idx < sk)
                )
                candidate_indices = cand_idx.reshape(sq, -1)
                safe_idx = candidate_indices.clamp(0, max(sk - 1, 0)).to(torch.long).reshape(-1)
                candidate_k = k_rows.index_select(0, safe_idx).view(sq, -1, head_dim)

                cand_dot = _hisa_bmm_fp32_accum(q_rows, candidate_k.transpose(1, 2))
                cand_dot.relu_()
                cand_scores = torch.bmm(weights_rows.unsqueeze(1), cand_dot).squeeze(1)
                valid_flat = valid.reshape(sq, -1)
                cand_scores = cand_scores.masked_fill(~valid_flat, float("-inf"))

                dst_start = slot_start * block_size
                dst_end = dst_start + candidate_indices.shape[1]
                all_scores[:, dst_start:dst_end] = cand_scores

        with fine_profile_range("hisa.candidate_refine.compact_final_topk"):
            running_scores, gather_pos = torch.topk(
                all_scores, k=topk_k, dim=-1, sorted=False
            )
            slot_ids = torch.div(gather_pos, int(block_size), rounding_mode="floor")
            offsets = (gather_pos - slot_ids * int(block_size)).to(torch.int32)
            safe_slot_ids = slot_ids.clamp(0, max(int(top_blocks.shape[1]) - 1, 0))
            selected_blocks = top_blocks.to(torch.int32).gather(1, safe_slot_ids)
            running_indices = selected_blocks * int(block_size) + offsets
            valid_final = (
                torch.isfinite(running_scores)
                & (gather_pos < candidate_width)
                & (selected_blocks >= 0)
                & (running_indices < prefix_lens.view(-1, 1))
                & (running_indices < sk)
            )
        return running_indices.masked_fill(~valid_final, -1), running_scores

    running_scores = q_rows.new_full((sq, topk_k), float("-inf"))
    running_indices = torch.full((sq, topk_k), -1, device=q_rows.device, dtype=torch.int32)
    for slot_start in range(0, top_blocks.shape[1], slot_group):
        slot_end = min(slot_start + slot_group, top_blocks.shape[1])
        with fine_profile_range(f"hisa.candidate_refine.streaming_slot.{slot_start}_{slot_end}"):
            block_ids = top_blocks[:, slot_start : slot_start + slot_group].to(torch.int32)
            valid_block = block_ids >= 0
            cand_idx = block_ids.clamp_min(0).unsqueeze(-1) * block_size + offsets.view(
                1, 1, -1
            )
            valid = (
                valid_block.unsqueeze(-1)
                & (cand_idx < prefix_lens.view(-1, 1, 1))
                & (cand_idx < sk)
            )
            candidate_indices = cand_idx.reshape(sq, -1)
            safe_idx = candidate_indices.clamp(0, max(sk - 1, 0)).to(torch.long).reshape(-1)
            candidate_k = k_rows.index_select(0, safe_idx).view(sq, -1, head_dim)

            # [Q, H, D] x [Q, D, G*B] -> [Q, H, G*B]. The default helper preserves
            # exact FP32 selector semantics; the opt-in tensor-core mode keeps
            # BF16/FP16 inputs and requests FP32 output accumulation from cuBLAS.
            cand_dot = _hisa_bmm_fp32_accum(q_rows, candidate_k.transpose(1, 2))
            # Same memory discipline as the block selector: avoid materializing the
            # [rows, heads, candidates] broadcast product before reducing heads.
            cand_dot.relu_()
            cand_scores = torch.bmm(weights_rows.unsqueeze(1), cand_dot).squeeze(1)
            cand_scores = cand_scores.masked_fill(~valid.reshape(sq, -1), float("-inf"))

            merged_scores = torch.cat((running_scores, cand_scores), dim=-1)
            merged_indices = torch.cat((running_indices, candidate_indices), dim=-1)
            running_scores, gather_pos = torch.topk(
                merged_scores, k=topk_k, dim=-1, sorted=False
            )
            running_indices = merged_indices.gather(1, gather_pos)

    valid_final = torch.isfinite(running_scores)
    return running_indices.masked_fill(~valid_final, -1), running_scores


def _indexcache_hisa_topk_bmm_for_batch(
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    topk_k: int,
    *,
    config: IndexCacheHISAConfig,
    prefix_lens: torch.Tensor,
    block_topk_counts: Optional[torch.Tensor],
    effective_block_topk: int,
    k_b_precomputed: Optional[torch.Tensor] = None,
    block_reps_precomputed: Optional[torch.Tensor] = None,
    block_prefix_precomputed: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor-core-friendly HISA selector for one batch item.

    This keeps the same two-stage HISA contract as the extension selector:
    select compressed blocks first, then score only those candidate tokens.
    The hot candidate scoring path is batched matmul, which maps much better to
    Blackwell than scalar per-token CUDA loops while preserving the selected set.
    """

    sq_total = q_rows.shape[0]
    row_chunk = _hisa_selector_row_chunk(sq_total)
    use_tc_inputs = (
        _hisa_bmm_fp32_accum_tensorcores_enabled()
        and q_rows.is_cuda
        and k_rows.is_cuda
        and q_rows.dtype == k_rows.dtype
        and q_rows.dtype in (torch.bfloat16, torch.float16)
    )
    if sq_total > row_chunk:
        with fine_profile_range("hisa.bmm.precompute_reps_and_prefix"):
            k_b_once = (
                k_rows.contiguous()
                if use_tc_inputs
                else (k_b_precomputed if k_b_precomputed is not None else k_rows.float())
            )
            block_size = int(config.block_size)
            sk = k_b_once.shape[0]
            block_count = int(math.ceil(sk / block_size))
            reps_once = (
                block_reps_precomputed
                if block_reps_precomputed is not None
                else _mean_pool_all_blocks(k_b_once, block_size)
            )
            if block_prefix_precomputed is None:
                pad_len = block_count * block_size - sk
                if pad_len:
                    padded_k = torch.cat((k_b_once, k_b_once.new_zeros((pad_len, k_b_once.shape[-1]))), dim=0)
                else:
                    padded_k = k_b_once
                block_prefix_once = padded_k.reshape(block_count, block_size, k_b_once.shape[-1]).cumsum(dim=1)
                if pad_len:
                    del padded_k
            else:
                block_prefix_once = block_prefix_precomputed
        indices_out = torch.empty((sq_total, int(topk_k)), device=q_rows.device, dtype=torch.int32)
        scores_out = q_rows.new_empty((sq_total, int(topk_k)), dtype=torch.float32)
        for row_start in range(0, sq_total, row_chunk):
            row_end = min(row_start + row_chunk, sq_total)
            block_counts_chunk = (
                block_topk_counts[row_start:row_end]
                if block_topk_counts is not None
                else None
            )
            with fine_profile_range(f"hisa.bmm.row_chunk.{row_start}_{row_end}"):
                indices, scores = _indexcache_hisa_topk_bmm_for_batch(
                    q_rows[row_start:row_end],
                    weights_rows[row_start:row_end],
                    k_rows,
                    topk_k,
                    config=config,
                    prefix_lens=prefix_lens[row_start:row_end],
                    block_topk_counts=block_counts_chunk,
                    effective_block_topk=effective_block_topk,
                    k_b_precomputed=k_b_once,
                    block_reps_precomputed=reps_once,
                    block_prefix_precomputed=block_prefix_once,
                )
            indices_out[row_start:row_end] = indices
            scores_out[row_start:row_end] = scores
        return indices_out, scores_out

    with fine_profile_range("hisa.bmm.cast_inputs"):
        q_b = q_rows.contiguous() if use_tc_inputs else q_rows.float()
        w_b = weights_rows.float()
        k_b = (
            k_rows.contiguous()
            if use_tc_inputs and k_b_precomputed is None
            else (k_b_precomputed if k_b_precomputed is not None else k_rows.float())
        )
    sq, _, head_dim = q_b.shape
    sk = k_b.shape[0]
    block_size = int(config.block_size)
    block_count = int(math.ceil(sk / block_size))
    arange_blocks = torch.arange(block_count, device=q_b.device, dtype=torch.int32)

    # The block selector is non-differentiable. Gradients for indexer training
    # flow only through the selected logits, matching the CUDA selector path.
    with torch.no_grad():
        row_block_counts = torch.div(
            prefix_lens + block_size - 1, block_size, rounding_mode="floor"
        ).to(torch.int32)
        with fine_profile_range("hisa.bmm.block_reps"):
            reps = (
                block_reps_precomputed
                if block_reps_precomputed is not None
                else _mean_pool_all_blocks(k_b, block_size)
            )
        with fine_profile_range("hisa.bmm.block_scores"):
            block_dot = _hisa_bmm_fp32_accum(
                q_b.reshape(1, sq * q_b.shape[1], head_dim),
                reps.t().unsqueeze(0),
            ).reshape(sq, q_b.shape[1], block_count)
            # Keep the block selector workspace flat. The previous transpose +
            # broadcast multiply materialized [rows, blocks, heads], which is a
            # large temporary at 32k context. ReLU is selector-local/no-grad, and
            # the batched 1xH @ HxB reduction writes only [rows, blocks].
            block_dot.relu_()
            block_scores = torch.bmm(w_b.unsqueeze(1), block_dot).squeeze(1)
        block_scores = block_scores.masked_fill(
            arange_blocks.view(1, -1) >= row_block_counts.view(-1, 1),
            float("-inf"),
        )

        # Correct the per-row partial final block. A global block mean would
        # leak future tokens inside the current causal block.
        with fine_profile_range("hisa.bmm.partial_block_correction"):
            last_blocks = (row_block_counts - 1).clamp_min(0)
            starts = last_blocks * block_size
            safe_prefix = prefix_lens.clamp(0, sk)
            raw_counts = safe_prefix - starts
            offsets = (raw_counts - 1).clamp_min(0)
            if block_prefix_precomputed is not None:
                block_prefix = block_prefix_precomputed
            else:
                pad_len = block_count * block_size - sk
                if pad_len:
                    padded_k = torch.cat((k_b, k_b.new_zeros((pad_len, head_dim))), dim=0)
                else:
                    padded_k = k_b
                # Prefix only within each HISA block. This avoids rebuilding a full
                # sequence-length cumsum for every query chunk while preserving the
                # exact causal partial-block correction.
                block_prefix = padded_k.reshape(block_count, block_size, head_dim).cumsum(dim=1)
            sums = block_prefix[last_blocks, offsets]
            sums = torch.where(raw_counts.unsqueeze(-1) > 0, sums, torch.zeros_like(sums))
            counts = raw_counts.clamp_min(1).to(k_b.dtype).unsqueeze(-1)
            partial_reps = sums / counts
            partial_dot = _hisa_bmm_fp32_accum(q_b, partial_reps.unsqueeze(-1)).squeeze(-1)
            partial_scores = (torch.relu(partial_dot) * w_b).sum(dim=-1)
            block_scores.scatter_(1, last_blocks.view(-1, 1), partial_scores.view(-1, 1))

        valid_rows = row_block_counts > 0
        if "first" in config.forced_boundary_blocks:
            block_scores[valid_rows, 0] = float("inf")
        if "last" in config.forced_boundary_blocks:
            block_scores.scatter_(
                1,
                last_blocks.view(-1, 1),
                torch.full((sq, 1), float("inf"), device=q_b.device),
            )
        if "last_minus_one" in config.forced_boundary_blocks:
            prev_blocks = (row_block_counts - 2).clamp_min(0)
            has_prev = row_block_counts > 1
            block_scores[has_prev] = block_scores[has_prev].scatter(
                1,
                prev_blocks[has_prev].view(-1, 1),
                torch.full((int(has_prev.sum().item()), 1), float("inf"), device=q_b.device),
            )

        with fine_profile_range("hisa.bmm.block_topk"):
            block_keep = min(effective_block_topk, block_count)
            top_block_values, top_blocks = torch.topk(
                block_scores, k=block_keep, dim=-1, sorted=False
            )
            top_blocks = top_blocks.masked_fill(
                torch.isnan(top_block_values) | (top_block_values == float("-inf")), -1
            )
            if block_topk_counts is not None and block_keep > 0:
                slot_ids = torch.arange(block_keep, device=q_b.device).view(1, -1)
                top_blocks = top_blocks.masked_fill(
                    slot_ids >= block_topk_counts.view(-1, 1), -1
                )

    with fine_profile_range("hisa.bmm.candidate_refine"):
        return _hisa_grouped_candidate_topk(
            q_b,
            w_b,
            k_b,
            top_blocks,
            prefix_lens,
            topk_k,
            block_size,
        )


def _indexcache_hisa_topk_cuda_for_batch(
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    topk_k: int,
    *,
    config: IndexCacheHISAConfig,
    prefix_lens: torch.Tensor,
    block_topk_counts: Optional[torch.Tensor],
    effective_block_topk: int,
    return_scores: bool,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    if (
        not _hisa_selector_cuda_enabled()
        or not q_rows.is_cuda
        or not config.is_optimized
        or not _is_blackwell_or_newer(q_rows.device)
    ):
        return None
    ext = _try_load_hisa_cuda_ext()
    if ext is None or not hasattr(ext, "hisa_selector_fwd"):
        return None

    sq, _, head_dim = q_rows.shape
    sk = k_rows.shape[0]
    block_size = int(config.block_size)
    block_count = int(math.ceil(sk / block_size))
    if block_count <= 0:
        return None
    if block_topk_counts is None:
        block_topk_counts = torch.full(
            (sq,),
            min(int(config.block_topk), block_count),
            device=q_rows.device,
            dtype=torch.int32,
        )
    else:
        block_topk_counts = block_topk_counts.to(device=q_rows.device, dtype=torch.int32)

    q_f = q_rows.contiguous().float()
    w_f = weights_rows.contiguous().float()
    k_f = k_rows.contiguous().float()
    reps = _mean_pool_all_blocks(k_f, block_size).contiguous()
    prefix_i32 = prefix_lens.to(device=q_rows.device, dtype=torch.int32).contiguous()
    indices_i32 = torch.empty((sq, topk_k), device=q_rows.device, dtype=torch.int32)
    kernel_scores = torch.empty((sq, topk_k), device=q_rows.device, dtype=torch.float32)
    forced = tuple(config.forced_boundary_blocks or ())

    ext.hisa_selector_fwd(
        q_f,
        k_f,
        reps,
        w_f,
        prefix_i32,
        block_topk_counts.contiguous(),
        indices_i32,
        kernel_scores,
        block_size,
        int(effective_block_topk),
        int(topk_k),
        "first" in forced,
        "last" in forced,
        "last_minus_one" in forced,
    )
    topk_indices = indices_i32
    if return_scores:
        # Keep training gradients in PyTorch and avoid saving the full HISA
        # candidate cache. The selector itself is non-differentiable top-k;
        # selected logits are the differentiable carrier for the KL loss.
        scores = _selected_hisa_scores(
            q_rows.float(), weights_rows.float(), k_rows.float(), topk_indices
        )
    else:
        scores = None
    return topk_indices, scores


def indexcache_hisa_cuda_select_with_scores(
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    topk: int,
    *,
    config: IndexCacheHISAConfig,
    prefix_lens: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """CUDA HISA selector returning compact selected logits for custom autograd.

    This helper intentionally returns the selector kernel's ``selected_scores``
    directly instead of rebuilding them with PyTorch tensor ops. The selector
    itself remains non-differentiable; callers that train through the selected
    logits should pair this with the fused selected-score backward kernel.
    """

    if (
        not _hisa_selector_cuda_enabled()
        or not q_rows.is_cuda
        or not config.is_optimized
        or not _is_blackwell_or_newer(q_rows.device)
    ):
        return None
    ext = _try_load_hisa_cuda_ext()
    if ext is None or not hasattr(ext, "hisa_selector_fwd"):
        return None

    sq, _, head_dim = q_rows.shape
    sk = k_rows.shape[0]
    topk_k = min(int(topk), sk)
    if topk_k <= 0:
        return None
    prefix_lens = prefix_lens.to(device=q_rows.device, dtype=torch.long).clamp(0, sk)
    if config.fallback_to_dense_if_short and int(prefix_lens.max().item()) <= topk_k:
        return None
    block_size = int(config.block_size)
    block_count = int(math.ceil(sk / block_size))
    if block_count <= 0:
        return None

    row_block_counts = torch.div(
        prefix_lens + block_size - 1, block_size, rounding_mode="floor"
    ).to(torch.int32)
    if config.compression_ratio > 0:
        block_topk_counts, effective_block_topk = hisa_block_topk_counts(
            row_block_counts,
            block_size=block_size,
            topk_tokens=topk_k,
            compression_ratio=config.compression_ratio,
        )
    else:
        block_topk_counts = torch.full(
            (sq,),
            min(int(config.block_topk), block_count),
            device=q_rows.device,
            dtype=torch.int32,
        )
        effective_block_topk = min(int(config.block_topk), block_count)
    block_topk_counts, effective_block_topk = _include_forced_boundary_budget(
        block_topk_counts,
        effective_block_topk,
        row_block_counts,
        tuple(config.forced_boundary_blocks or ()),
    )

    q_f = q_rows.contiguous().float()
    w_f = weights_rows.contiguous().float()
    k_f = k_rows.contiguous().float()
    reps = _mean_pool_all_blocks(k_f, block_size).contiguous()
    prefix_i32 = prefix_lens.to(dtype=torch.int32).contiguous()
    indices_i32 = torch.empty((sq, topk_k), device=q_rows.device, dtype=torch.int32)
    selected_scores = torch.empty((sq, topk_k), device=q_rows.device, dtype=torch.float32)
    forced = tuple(config.forced_boundary_blocks or ())

    ext.hisa_selector_fwd(
        q_f,
        k_f,
        reps,
        w_f,
        prefix_i32,
        block_topk_counts.contiguous(),
        indices_i32,
        selected_scores,
        block_size,
        int(effective_block_topk),
        int(topk_k),
        "first" in forced,
        "last" in forced,
        "last_minus_one" in forced,
    )
    return indices_i32, selected_scores


def indexcache_hisa_packed_nvfp4_select_with_scores(
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    topk: int,
    *,
    config: IndexCacheHISAConfig,
    prefix_lens: torch.Tensor,
    ext_op: str = "hisa_selector_nvfp4_fwd",
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """CUDA HISA selector that scores candidate K rows from packed NVFP4.

    ``apply_indexcache_kv`` attaches packed NVFP4 values/scales to its
    fake-dequantized result. This path consumes that sidecar directly for the
    token-refine stage, avoiding an additional dense K read/contiguous copy in
    the selector. It is intentionally opt-in until full-model timing proves it
    beats the BMM/cuBLAS path at production shapes.
    """

    if not (q_rows.is_cuda and config.is_optimized and _is_blackwell_or_newer(q_rows.device)):
        return None
    packed = get_indexcache_nvfp4_packed_tensors(k_rows)
    if packed is None:
        return None
    ext = _try_load_hisa_cuda_ext()
    if ext is None or not hasattr(ext, ext_op):
        return None

    packed_values, packed_scales, packed_row_offset, packed_row_stride = packed
    sq, heads, head_dim = q_rows.shape
    if ext_op in (
        "hisa_selector_nvfp4_cublasdx_fwd",
        "hisa_selector_nvfp4_cublasdx_tiled_fwd",
        "hisa_selector_nvfp4_cublasdx_fp8_fwd",
    ) and (heads != 64 or head_dim != 128):
        return None
    sk = k_rows.shape[0]
    topk_k = min(int(topk), sk)
    if topk_k <= 0:
        return None
    prefix_lens = prefix_lens.to(device=q_rows.device, dtype=torch.long).clamp(0, sk)
    if config.fallback_to_dense_if_short and int(prefix_lens.max().item()) <= topk_k:
        return None
    block_size = int(config.block_size)
    block_count = int(math.ceil(sk / block_size))
    if block_count <= 0:
        return None

    row_block_counts = torch.div(
        prefix_lens + block_size - 1, block_size, rounding_mode="floor"
    ).to(torch.int32)
    if config.compression_ratio > 0:
        block_topk_counts, effective_block_topk = hisa_block_topk_counts(
            row_block_counts,
            block_size=block_size,
            topk_tokens=topk_k,
            compression_ratio=config.compression_ratio,
        )
    else:
        block_topk_counts = torch.full(
            (sq,),
            min(int(config.block_topk), block_count),
            device=q_rows.device,
            dtype=torch.int32,
        )
        effective_block_topk = min(int(config.block_topk), block_count)
    block_topk_counts, effective_block_topk = _include_forced_boundary_budget(
        block_topk_counts,
        effective_block_topk,
        row_block_counts,
        tuple(config.forced_boundary_blocks or ()),
    )

    q_f = q_rows.contiguous().float()
    w_f = weights_rows.contiguous().float()
    # Stage-1 block representatives are still dense means of the
    # fake-dequantized K. Stage-2 candidate-token scoring loads packed NVFP4.
    reps = _mean_pool_all_blocks(k_rows.float(), block_size).contiguous()
    prefix_i32 = prefix_lens.to(dtype=torch.int32).contiguous()
    indices_i32 = torch.empty((sq, topk_k), device=q_rows.device, dtype=torch.int32)
    selected_scores = torch.empty((sq, topk_k), device=q_rows.device, dtype=torch.float32)
    forced = tuple(config.forced_boundary_blocks or ())

    packed_values_c = packed_values.contiguous()
    packed_scales_c = packed_scales.contiguous()
    block_topk_counts_c = block_topk_counts.contiguous()
    if ext_op == "hisa_selector_nvfp4_cublasdx_tiled_fwd":
        candidate_capacity = _next_power_of_two_int(
            max(int(topk_k), int(effective_block_topk) * block_size)
        )
        selected_blocks = torch.empty(
            (sq, int(effective_block_topk)), device=q_rows.device, dtype=torch.int32
        )
        candidate_scores = torch.empty(
            (sq, candidate_capacity), device=q_rows.device, dtype=torch.float32
        )
        candidate_indices = torch.empty(
            (sq, candidate_capacity), device=q_rows.device, dtype=torch.int32
        )
        ext.hisa_selector_nvfp4_cublasdx_tiled_fwd(
            q_f,
            packed_values_c,
            packed_scales_c,
            reps,
            w_f,
            prefix_i32,
            block_topk_counts_c,
            selected_blocks,
            candidate_scores,
            candidate_indices,
            indices_i32,
            selected_scores,
            int(sk),
            int(packed_row_offset),
            int(packed_row_stride),
            block_size,
            int(effective_block_topk),
            int(topk_k),
            "first" in forced,
            "last" in forced,
            "last_minus_one" in forced,
        )
    else:
        getattr(ext, ext_op)(
            q_f,
            packed_values_c,
            packed_scales_c,
            reps,
            w_f,
            prefix_i32,
            block_topk_counts_c,
            indices_i32,
            selected_scores,
            int(sk),
            int(packed_row_offset),
            int(packed_row_stride),
            block_size,
            int(effective_block_topk),
            int(topk_k),
            "first" in forced,
            "last" in forced,
            "last_minus_one" in forced,
        )
    return indices_i32, selected_scores


def _deepgemm_hisa_build_block_prefix(k_rows: torch.Tensor, block_size: int) -> torch.Tensor:
    block_count = int(math.ceil(k_rows.shape[0] / block_size))
    pad_len = block_count * block_size - k_rows.shape[0]
    if pad_len:
        padded = torch.cat((k_rows, k_rows.new_zeros((pad_len, k_rows.shape[-1]))), dim=0)
    else:
        padded = k_rows
    return padded.reshape(block_count, block_size, k_rows.shape[-1]).cumsum(dim=1)


def _deepgemm_hisa_build_fused_cache(
    packed: tuple[torch.Tensor, torch.Tensor, int, int],
    *,
    sk: int,
    head_dim: int,
) -> torch.Tensor:
    packed_values, packed_scales, packed_row_offset, packed_row_stride = packed
    k_values = packed_values[
        packed_row_offset : packed_row_offset + sk * packed_row_stride : packed_row_stride
    ].contiguous()
    k_scales = packed_scales[
        packed_row_offset : packed_row_offset + sk * packed_row_stride : packed_row_stride
    ].contiguous()
    page_size = 64
    pages = int(math.ceil(sk / page_size))
    fused_cache = torch.empty(
        (pages, page_size, 1, head_dim // 2 + 4),
        device=packed_values.device,
        dtype=torch.uint8,
    )
    fused_page_flat = fused_cache.view(pages, page_size * (head_dim // 2 + 4))
    padded_values = torch.zeros(
        (pages * page_size, head_dim // 2), device=packed_values.device, dtype=torch.uint8
    )
    padded_scales = torch.zeros((pages * page_size,), device=packed_values.device, dtype=torch.int32)
    padded_values[:sk] = k_values.view(sk, head_dim // 2).view(torch.uint8)
    padded_scales[:sk] = k_scales.view(sk)
    fused_page_flat[:, : page_size * (head_dim // 2)] = padded_values.view(
        pages, page_size * (head_dim // 2)
    )
    fused_page_flat[:, page_size * (head_dim // 2) :] = padded_scales.view(
        pages, page_size
    ).view(torch.uint8)
    return fused_cache


def _indexcache_hisa_deepgemm_select_with_scores(
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    topk: int,
    *,
    config: IndexCacheHISAConfig,
    prefix_lens: torch.Tensor,
    block_topk_counts: torch.Tensor,
    effective_block_topk: int,
    k_b_precomputed: Optional[torch.Tensor] = None,
    block_reps_precomputed: Optional[torch.Tensor] = None,
    block_prefix_precomputed: Optional[torch.Tensor] = None,
    fused_cache_precomputed: Optional[torch.Tensor] = None,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Select HISA candidates with DeepGEMM FP4 MQA scoring.

    This backend intentionally uses the accepted OP-style selector math for the
    candidate-refine stage: Q is quantized to FP4 and K is read from the NVFP4
    IndexCache sidecar. The selected-score custom backward still treats top-k
    as non-differentiable and propagates through the selected score formula for
    the chosen token ids.
    """

    if not (q_rows.is_cuda and weights_rows.is_cuda and k_rows.is_cuda):
        return None
    if not (config.is_optimized and _is_blackwell_or_newer(q_rows.device)):
        return None
    if int(config.block_size) != 128:
        return None
    packed = get_indexcache_nvfp4_packed_tensors(k_rows)
    if packed is None:
        return None
    try:
        import deep_gemm  # type: ignore
        from deep_gemm.utils import per_token_cast_to_fp4  # type: ignore
    except (AssertionError, ImportError, RuntimeError, OSError):
        return None

    sq, heads, head_dim = q_rows.shape
    if heads != 64 or head_dim != 128:
        return None
    sk = k_rows.shape[0]
    topk_k = min(int(topk), sk)
    if topk_k <= 0:
        return None
    block_size = int(config.block_size)
    block_count = int(math.ceil(sk / block_size))
    block_keep = min(int(effective_block_topk), block_count)
    if block_keep <= 0 or block_keep * block_size < topk_k:
        return None

    sq_total = q_rows.shape[0]
    row_chunk = _hisa_selector_row_chunk(sq_total)
    if sq_total > row_chunk:
        k_b_once = k_b_precomputed if k_b_precomputed is not None else k_rows.float()
        reps_once = (
            block_reps_precomputed
            if block_reps_precomputed is not None
            else _mean_pool_all_blocks(k_b_once, block_size)
        )
        block_prefix_once = (
            block_prefix_precomputed
            if block_prefix_precomputed is not None
            else _deepgemm_hisa_build_block_prefix(k_b_once, block_size)
        )
        fused_cache_once = (
            fused_cache_precomputed
            if fused_cache_precomputed is not None
            else _deepgemm_hisa_build_fused_cache(packed, sk=sk, head_dim=head_dim)
        )
        index_chunks = []
        score_chunks = []
        for row_start in range(0, sq_total, row_chunk):
            row_end = min(row_start + row_chunk, sq_total)
            result = _indexcache_hisa_deepgemm_select_with_scores(
                q_rows[row_start:row_end],
                weights_rows[row_start:row_end],
                k_rows,
                topk,
                config=config,
                prefix_lens=prefix_lens[row_start:row_end],
                block_topk_counts=block_topk_counts[row_start:row_end],
                effective_block_topk=effective_block_topk,
                k_b_precomputed=k_b_once,
                block_reps_precomputed=reps_once,
                block_prefix_precomputed=block_prefix_once,
                fused_cache_precomputed=fused_cache_once,
            )
            if result is None:
                return None
            indices, scores = result
            index_chunks.append(indices)
            score_chunks.append(scores)
        return torch.cat(index_chunks, dim=0), torch.cat(score_chunks, dim=0)

    q_b = q_rows.float()
    w_b = weights_rows.float()
    k_b = k_b_precomputed if k_b_precomputed is not None else k_rows.float()
    prefix_lens = prefix_lens.to(device=q_rows.device, dtype=torch.long).clamp(0, sk)
    row_block_counts = torch.div(
        prefix_lens + block_size - 1, block_size, rounding_mode="floor"
    ).to(torch.int32)
    if bool((row_block_counts <= 0).any().item()):
        return None
    partial_rows = (prefix_lens % block_size) != 0
    if bool(partial_rows.any().item()) and "last" not in tuple(config.forced_boundary_blocks or ()):
        return None

    arange_blocks = torch.arange(block_count, device=q_rows.device, dtype=torch.int32)
    with torch.no_grad():
        reps = (
            block_reps_precomputed
            if block_reps_precomputed is not None
            else _mean_pool_all_blocks(k_b, block_size)
        )
        block_dot = _hisa_bmm_fp32_accum(
            q_b.reshape(1, sq * heads, head_dim),
            reps.t().unsqueeze(0),
        ).reshape(sq, heads, block_count)
        block_dot.relu_()
        block_scores = torch.bmm(w_b.unsqueeze(1), block_dot).squeeze(1)
        block_scores = block_scores.masked_fill(
            arange_blocks.view(1, -1) >= row_block_counts.view(-1, 1),
            float("-inf"),
        )

        last_blocks = (row_block_counts - 1).clamp_min(0)
        starts = last_blocks * block_size
        raw_counts = prefix_lens.to(torch.int32) - starts
        offsets = (raw_counts - 1).clamp_min(0)
        block_prefix = (
            block_prefix_precomputed
            if block_prefix_precomputed is not None
            else _deepgemm_hisa_build_block_prefix(k_b, block_size)
        )
        sums = block_prefix[last_blocks.long(), offsets.long()]
        sums = torch.where(raw_counts.unsqueeze(-1) > 0, sums, torch.zeros_like(sums))
        counts = raw_counts.clamp_min(1).to(k_b.dtype).unsqueeze(-1)
        partial_reps = sums / counts
        partial_dot = _hisa_bmm_fp32_accum(q_b, partial_reps.unsqueeze(-1)).squeeze(-1)
        partial_scores = (torch.relu(partial_dot) * w_b).sum(dim=-1)
        block_scores.scatter_(1, last_blocks.view(-1, 1).long(), partial_scores.view(-1, 1))

        valid_rows = row_block_counts > 0
        if "first" in config.forced_boundary_blocks:
            block_scores[valid_rows, 0] = float("inf")
        if "last" in config.forced_boundary_blocks:
            block_scores.scatter_(
                1,
                last_blocks.view(-1, 1).long(),
                torch.full((sq, 1), float("inf"), device=q_rows.device),
            )
        if "last_minus_one" in config.forced_boundary_blocks:
            prev_blocks = (row_block_counts - 2).clamp_min(0)
            has_prev = row_block_counts > 1
            block_scores[has_prev] = block_scores[has_prev].scatter(
                1,
                prev_blocks[has_prev].view(-1, 1).long(),
                torch.full((int(has_prev.sum().item()), 1), float("inf"), device=q_rows.device),
            )

        top_block_values, top_blocks = torch.topk(
            block_scores, k=block_keep, dim=-1, sorted=False
        )
        top_blocks = top_blocks.masked_fill(
            torch.isnan(top_block_values) | (top_block_values == float("-inf")), -1
        ).to(torch.int32)

        # DeepGEMM has a contiguous candidate-length mask per row. Keep the
        # selected partial causal block as the final valid slot so that its
        # future tokens are masked without a separate candidate mask tensor.
        valid_counts = block_topk_counts.clamp(0, block_keep).to(torch.int32)
        if bool(partial_rows.any().item()):
            top_blocks = top_blocks.clone()
            for row in range(sq):
                count = int(valid_counts[row].item())
                if count <= 0:
                    continue
                last_block = int(last_blocks[row].item())
                match = (top_blocks[row] == last_block).nonzero(as_tuple=False)
                if match.numel() == 0:
                    return None
                src = int(match[0, 0].item())
                dst = count - 1
                if src != dst:
                    tmp = top_blocks[row, dst].clone()
                    top_blocks[row, dst] = top_blocks[row, src]
                    top_blocks[row, src] = tmp
        slot_ids = torch.arange(block_keep, device=q_rows.device).view(1, -1)
        top_blocks = top_blocks.masked_fill(slot_ids >= valid_counts.view(-1, 1), -1)

    page_size = 64
    pages = int(math.ceil(sk / page_size))
    fused_cache = (
        fused_cache_precomputed
        if fused_cache_precomputed is not None
        else _deepgemm_hisa_build_fused_cache(packed, sk=sk, head_dim=head_dim)
    )

    safe_blocks = top_blocks.clamp_min(0)
    page_table = torch.empty((sq, block_keep * 2), device=q_rows.device, dtype=torch.int32)
    page_table[:, 0::2] = safe_blocks * 2
    page_table[:, 1::2] = safe_blocks * 2 + 1
    raw_counts_i32 = (prefix_lens.to(torch.int32) - last_blocks * block_size).clamp(1, block_size)
    context_lens = valid_counts * block_size
    if bool(partial_rows.any().item()):
        context_lens = torch.where(
            partial_rows,
            (valid_counts - 1).clamp_min(0) * block_size + raw_counts_i32,
            context_lens,
        )
    context_lens = context_lens.clamp_min(0).view(sq, 1).contiguous().to(torch.int32)

    q_values, q_scales = per_token_cast_to_fp4(
        q_rows.reshape(-1, head_dim),
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    q_in = (
        q_values.view(torch.int8).view(sq, 1, heads, head_dim // 2).contiguous(),
        q_scales.view(sq, 1, heads).contiguous(),
    )
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, page_size, deep_gemm.get_num_sms()
    )
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        q=q_in,
        kv_cache=fused_cache,
        weights=w_b.contiguous(),
        context_lens=context_lens,
        block_table=page_table.contiguous(),
        schedule_meta=schedule_meta,
        max_context_len=block_keep * block_size,
        clean_logits=False,
        logits_dtype=torch.float32,
        indices=None,
    )
    selected_scores, candidate_pos = torch.topk(logits, k=topk_k, dim=-1, sorted=False)
    block_slot = torch.div(candidate_pos, block_size, rounding_mode="floor")
    token_offset = candidate_pos - block_slot * block_size
    topk_indices = top_blocks.gather(1, block_slot.to(torch.long)) * block_size + token_offset.to(
        torch.int32
    )
    valid = torch.isfinite(selected_scores) & (candidate_pos < context_lens)
    topk_indices = topk_indices.masked_fill(~valid, -1).to(torch.int32)
    selected_scores = selected_scores.masked_fill(~valid, float("-inf")).float()
    return topk_indices, selected_scores


def indexcache_hisa_megakernel_batched_select_with_scores(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk: int,
    *,
    config: IndexCacheHISAConfig,
    prefix_lens: torch.Tensor,
    block_reps_precomputed: Optional[torch.Tensor] = None,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Run the fused batched HISA selector extension.

    This is the training fast path intended to replace the Python loop over
    batch items plus the ATen BMM/topk/gather chain. It returns top-k indices as
    ``[B, Q, K]`` and selected scores flattened as ``[B * Q, K]`` to match the
    selected-score autograd contract in DSA.
    """

    if not (q.is_cuda and weights.is_cuda and k.is_cuda):
        return None
    if not (config.is_optimized and _is_blackwell_or_newer(q.device)):
        return None
    if q.dim() != 4 or weights.dim() != 3 or k.dim() != 3:
        return None
    q_len, bsz, heads, head_dim = q.shape
    sk = k.shape[0]
    if k.shape[1] != bsz or k.shape[2] != head_dim:
        return None
    if weights.shape != (q_len, bsz, heads):
        return None
    if heads != 64 or head_dim != 128:
        return None
    if q.dtype != k.dtype:
        return None
    if q.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return None
    if weights.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return None
    topk_k = min(int(topk), int(sk))
    if topk_k <= 0:
        return None

    if not prefix_lens.is_cuda:
        return None
    if prefix_lens.dtype not in (torch.int32, torch.int64):
        return None
    if config.fallback_to_dense_if_short:
        return None
    if prefix_lens.numel() == q_len:
        prefix_arg = prefix_lens.reshape(q_len)
    elif prefix_lens.numel() == bsz * q_len:
        prefix_arg = prefix_lens.reshape(bsz, q_len)
    else:
        return None
    if prefix_arg.device != q.device:
        return None

    block_size = int(config.block_size)
    block_count = int(math.ceil(sk / block_size))
    if block_count <= 0:
        return None
    if config.compression_ratio > 0:
        ratio_f = float(config.compression_ratio)
        if abs(ratio_f - round(ratio_f)) < 1e-6:
            ratio_i = int(round(ratio_f))
            effective_block_topk = (block_count + ratio_i - 1) // ratio_i
        else:
            effective_block_topk = int(math.ceil(block_count / ratio_f))
        effective_block_topk = max(1, min(int(effective_block_topk), block_count))
    else:
        effective_block_topk = min(int(config.block_topk), block_count)
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
        effective_block_topk = max(int(effective_block_topk), int(forced_static))
    use_parallel_refine = _hisa_megakernel_parallel_refine_enabled()
    candidate_capacity_needed = _next_power_of_two_int(
        max(int(topk_k), int(effective_block_topk) * block_size)
    )
    max_candidates = (
        _HISA_MEGAKERNEL_PARALLEL_MAX_CANDIDATES
        if use_parallel_refine
        else _HISA_MEGAKERNEL_SERIAL_MAX_CANDIDATES
    )
    if candidate_capacity_needed > max_candidates:
        raise RuntimeError(
            "HISA megakernel candidate capacity exceeds the validated 8192-candidate "
            f"path: capacity={candidate_capacity_needed} max={max_candidates} "
            f"parallel_refine={int(use_parallel_refine)} block_keep={effective_block_topk} "
            f"block_size={block_size} topk={topk_k}"
        )

    ext = _try_load_hisa_cuda_ext()
    streaming_scratch = (
        _hisa_megakernel_streaming_scratch(topk_k) if use_parallel_refine else None
    )
    streaming_refine = use_parallel_refine and (
        candidate_capacity_needed > _HISA_MEGAKERNEL_SERIAL_MAX_CANDIDATES
        or (
            streaming_scratch is not None
            and int(streaming_scratch) < int(candidate_capacity_needed)
        )
    )
    if (
        ext is None
        or not hasattr(ext, "hisa_block_reps_batched_fwd")
        or (
            streaming_refine
            and not hasattr(ext, "hisa_selector_megakernel_parallel_streaming_batched_fwd")
        )
        or (
            use_parallel_refine
            and not streaming_refine
            and not hasattr(ext, "hisa_selector_megakernel_parallel_batched_fwd")
        )
        or (
            not use_parallel_refine
            and not hasattr(ext, "hisa_selector_megakernel_batched_fwd")
        )
    ):
        return None

    q_c = q.contiguous()
    k_c = k.contiguous()
    weights_c = weights.contiguous()
    prefix_c = prefix_arg.contiguous()
    if block_reps_precomputed is not None:
        if block_reps_precomputed.shape != (bsz, block_count, head_dim):
            return None
        if block_reps_precomputed.device != k.device:
            return None
        if block_reps_precomputed.dtype != k.dtype:
            return None
        block_reps = block_reps_precomputed.contiguous()
    else:
        block_reps = torch.empty(
            (bsz, block_count, head_dim), device=k.device, dtype=k.dtype
        )
        ext.hisa_block_reps_batched_fwd(k_c, block_reps, block_size)
    topk_indices = torch.empty(
        (bsz, q_len, topk_k), device=q.device, dtype=torch.int32
    )
    selected_scores = torch.empty(
        (bsz, q_len, topk_k), device=q.device, dtype=torch.float32
    )
    if use_parallel_refine:
        candidate_capacity = (
            int(streaming_scratch or _HISA_MEGAKERNEL_SERIAL_MAX_CANDIDATES)
            if streaming_refine
            else candidate_capacity_needed
        )
        selected_blocks = torch.empty(
            (bsz, q_len, int(effective_block_topk)), device=q.device, dtype=torch.int32
        )
        candidate_keys = torch.empty(
            (bsz, q_len, candidate_capacity), device=q.device, dtype=torch.int64
        )
        if streaming_refine:
            topk_ordinals = torch.empty(
                (bsz, q_len, int(topk_k)), device=q.device, dtype=torch.int32
            )
            ext.hisa_selector_megakernel_parallel_streaming_batched_fwd(
                q_c,
                k_c,
                block_reps,
                weights_c,
                prefix_c,
                selected_blocks,
                candidate_keys,
                topk_indices,
                selected_scores,
                topk_ordinals,
                block_size,
                int(config.block_topk),
                float(config.compression_ratio),
                int(effective_block_topk),
                int(topk_k),
                int(candidate_capacity_needed),
                "first" in forced,
                "last" in forced,
                "last_minus_one" in forced,
            )
        else:
            ext.hisa_selector_megakernel_parallel_batched_fwd(
                q_c,
                k_c,
                block_reps,
                weights_c,
                prefix_c,
                selected_blocks,
                candidate_keys,
                topk_indices,
                selected_scores,
                block_size,
                int(config.block_topk),
                float(config.compression_ratio),
                int(effective_block_topk),
                int(topk_k),
                "first" in forced,
                "last" in forced,
                "last_minus_one" in forced,
            )
    else:
        ext.hisa_selector_megakernel_batched_fwd(
            q_c,
            k_c,
            block_reps,
            weights_c,
            prefix_c,
            topk_indices,
            selected_scores,
            block_size,
            int(config.block_topk),
            float(config.compression_ratio),
            int(effective_block_topk),
            int(topk_k),
            "first" in forced,
            "last" in forced,
            "last_minus_one" in forced,
        )
    return topk_indices, selected_scores.reshape(bsz * q_len, topk_k)


def indexcache_hisa_select_with_scores(
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    topk: int,
    *,
    config: IndexCacheHISAConfig,
    prefix_lens: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Select HISA top-k and selected logits using the configured backend.

    ``MEGATRON_HISA_SELECTOR_BACKEND=deepgemm`` uses FP4-quantized Q and the
    packed NVFP4 IndexCache sidecar for the candidate-refine GEMM. ``bmm`` keeps
    the exact BF16/FP32 candidate path for parity checks, and ``cuda`` preserves
    the extension-only selector for debugging.
    """

    if not (q_rows.is_cuda and config.is_optimized and _is_blackwell_or_newer(q_rows.device)):
        return None

    sq, _, _ = q_rows.shape
    sk = k_rows.shape[0]
    topk_k = min(int(topk), sk)
    if topk_k <= 0:
        return None
    prefix_lens = prefix_lens.to(device=q_rows.device, dtype=torch.long).clamp(0, sk)
    if config.fallback_to_dense_if_short and int(prefix_lens.max().item()) <= topk_k:
        return None
    block_size = int(config.block_size)
    block_count = int(math.ceil(sk / block_size))
    if block_count <= 0:
        return None

    row_block_counts = torch.div(
        prefix_lens + block_size - 1, block_size, rounding_mode="floor"
    ).to(torch.int32)
    if config.compression_ratio > 0:
        block_topk_counts, effective_block_topk = hisa_block_topk_counts(
            row_block_counts,
            block_size=block_size,
            topk_tokens=topk_k,
            compression_ratio=config.compression_ratio,
        )
    else:
        block_topk_counts = torch.full(
            (sq,),
            min(int(config.block_topk), block_count),
            device=q_rows.device,
            dtype=torch.int32,
        )
        effective_block_topk = min(int(config.block_topk), block_count)
    block_topk_counts, effective_block_topk = _include_forced_boundary_budget(
        block_topk_counts,
        effective_block_topk,
        row_block_counts,
        tuple(config.forced_boundary_blocks or ()),
    )

    backend = _hisa_selector_backend()
    if backend == "megakernel":
        result = indexcache_hisa_megakernel_batched_select_with_scores(
            q_rows.unsqueeze(1),
            weights_rows.unsqueeze(1),
            k_rows.unsqueeze(1),
            topk_k,
            config=config,
            prefix_lens=prefix_lens,
        )
        if result is None:
            return None
        indices_bqk, scores_bq = result
        return indices_bqk[0], scores_bq

    if backend in (
        "packed_cuda",
        "packed_cublasdx",
        "packed_cublasdx_tiled",
        "packed_cublasdx_fp8",
    ):
        packed_ext_ops = {
            "packed_cuda": "hisa_selector_nvfp4_fwd",
            "packed_cublasdx": "hisa_selector_nvfp4_cublasdx_fwd",
            "packed_cublasdx_tiled": "hisa_selector_nvfp4_cublasdx_tiled_fwd",
            "packed_cublasdx_fp8": "hisa_selector_nvfp4_cublasdx_fp8_fwd",
        }
        packed_result = indexcache_hisa_packed_nvfp4_select_with_scores(
            q_rows,
            weights_rows,
            k_rows,
            topk_k,
            config=config,
            prefix_lens=prefix_lens,
            ext_op=packed_ext_ops[backend],
        )
        if packed_result is not None:
            return packed_result
        return None

    if backend == "deepgemm":
        return _indexcache_hisa_deepgemm_select_with_scores(
            q_rows,
            weights_rows,
            k_rows,
            topk_k,
            config=config,
            prefix_lens=prefix_lens,
            block_topk_counts=block_topk_counts,
            effective_block_topk=effective_block_topk,
        )

    if backend in ("auto", "bmm"):
        indices, scores = _indexcache_hisa_topk_bmm_for_batch(
            q_rows,
            weights_rows,
            k_rows,
            topk_k,
            config=config,
            prefix_lens=prefix_lens,
            block_topk_counts=block_topk_counts,
            effective_block_topk=effective_block_topk,
        )
        return indices.to(torch.int32), scores

    return indexcache_hisa_cuda_select_with_scores(
        q_rows,
        weights_rows,
        k_rows,
        topk_k,
        config=config,
        prefix_lens=prefix_lens,
    )


def describe_indexcache_hisa_select_with_scores(
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    topk: int,
    *,
    config: IndexCacheHISAConfig,
    prefix_lens: torch.Tensor,
) -> str:
    """Describe why the optimized HISA selected-score path is or is not eligible."""

    backend = _hisa_selector_backend()
    pieces = [f"backend={backend}"]
    if not config.enabled:
        return "; ".join((*pieces, "config.enabled=0"))
    if not q_rows.is_cuda:
        return "; ".join((*pieces, "q_rows is not CUDA"))
    if not weights_rows.is_cuda:
        return "; ".join((*pieces, "weights_rows is not CUDA"))
    if not k_rows.is_cuda:
        return "; ".join((*pieces, "k_rows is not CUDA"))
    if not config.is_optimized:
        return "; ".join((*pieces, f"execution_mode={config.execution_mode!r}"))
    if not _is_blackwell_or_newer(q_rows.device):
        return "; ".join((*pieces, "device is not Blackwell or newer"))

    sq, heads, head_dim = q_rows.shape
    sk = k_rows.shape[0]
    topk_k = min(int(topk), sk)
    if topk_k <= 0:
        return "; ".join((*pieces, f"topk_k={topk_k} sk={sk}"))
    prefix_lens = prefix_lens.to(device=q_rows.device, dtype=torch.long).clamp(0, sk)
    if config.fallback_to_dense_if_short and int(prefix_lens.max().item()) <= topk_k:
        return "; ".join(
            (*pieces, "fallback_to_dense_if_short would choose dense short-context path")
        )
    block_size = int(config.block_size)
    block_count = int(math.ceil(sk / block_size)) if block_size > 0 else 0
    if block_count <= 0:
        return "; ".join((*pieces, f"block_count={block_count} block_size={block_size} sk={sk}"))

    row_block_counts = torch.div(
        prefix_lens + block_size - 1, block_size, rounding_mode="floor"
    ).to(torch.int32)
    if config.compression_ratio > 0:
        block_topk_counts, effective_block_topk = hisa_block_topk_counts(
            row_block_counts,
            block_size=block_size,
            topk_tokens=topk_k,
            compression_ratio=config.compression_ratio,
        )
    else:
        block_topk_counts = torch.full(
            (sq,),
            min(int(config.block_topk), block_count),
            device=q_rows.device,
            dtype=torch.int32,
        )
        effective_block_topk = min(int(config.block_topk), block_count)
    block_topk_counts, effective_block_topk = _include_forced_boundary_budget(
        block_topk_counts,
        effective_block_topk,
        row_block_counts,
        tuple(config.forced_boundary_blocks or ()),
    )

    if backend == "megakernel":
        ext = _try_load_hisa_cuda_ext()
        if ext is None:
            return "; ".join((*pieces, "HISA CUDA extension unavailable"))
        if not hasattr(ext, "hisa_block_reps_batched_fwd") or not hasattr(
            ext, "hisa_selector_megakernel_batched_fwd"
        ):
            return "; ".join((*pieces, "HISA CUDA extension lacks megakernel entry points"))
        if heads != 64 or head_dim != 128:
            return "; ".join(
                (*pieces, f"megakernel requires q shape [Q,64,128], got heads={heads} dim={head_dim}")
            )
        if q_rows.dtype != k_rows.dtype:
            return "; ".join((*pieces, f"megakernel requires q/k dtype match: {q_rows.dtype} vs {k_rows.dtype}"))
        if q_rows.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return "; ".join((*pieces, f"megakernel unsupported q dtype {q_rows.dtype}"))
        use_parallel_refine = _hisa_megakernel_parallel_refine_enabled()
        candidate_capacity_needed = _next_power_of_two_int(
            max(int(topk_k), int(effective_block_topk) * block_size)
        )
        max_candidates = (
            _HISA_MEGAKERNEL_PARALLEL_MAX_CANDIDATES
            if use_parallel_refine
            else _HISA_MEGAKERNEL_SERIAL_MAX_CANDIDATES
        )
        if candidate_capacity_needed > max_candidates:
            return "; ".join(
                (
                    *pieces,
                    f"megakernel candidate capacity too large: capacity={candidate_capacity_needed} "
                    f"max={max_candidates} parallel_refine={int(use_parallel_refine)} "
                    f"block_keep={effective_block_topk} block_size={block_size} topk={topk_k}",
                )
            )
        if (
            use_parallel_refine
            and candidate_capacity_needed > _HISA_MEGAKERNEL_SERIAL_MAX_CANDIDATES
            and not hasattr(ext, "hisa_selector_megakernel_parallel_streaming_batched_fwd")
        ):
            return "; ".join(
                (
                    *pieces,
                    "HISA CUDA extension lacks streaming parallel megakernel entry point "
                    f"needed for capacity={candidate_capacity_needed}",
                )
            )
        return "; ".join((*pieces, "eligible by static checks; megakernel returned None"))

    if backend == "deepgemm":
        if block_size != 128:
            return "; ".join((*pieces, f"deepgemm requires block_size=128, got {block_size}"))
        if get_indexcache_nvfp4_packed_tensors(k_rows) is None:
            return "; ".join((*pieces, "missing NVFP4 IndexCache packed sidecar on k_rows"))
        try:
            import deep_gemm  # noqa: F401  # type: ignore
            from deep_gemm.utils import per_token_cast_to_fp4  # noqa: F401  # type: ignore
        except (AssertionError, ImportError, RuntimeError, OSError) as exc:
            return "; ".join((*pieces, f"deep_gemm import failed: {type(exc).__name__}: {exc}"))
        if heads != 64 or head_dim != 128:
            return "; ".join(
                (*pieces, f"deepgemm requires q shape [Q,64,128], got heads={heads} dim={head_dim}")
            )
        block_keep = min(int(effective_block_topk), block_count)
        if block_keep <= 0 or block_keep * block_size < topk_k:
            return "; ".join(
                (
                    *pieces,
                    f"candidate capacity too small: block_keep={block_keep} "
                    f"block_size={block_size} topk={topk_k}",
                )
            )
        if bool((row_block_counts <= 0).any().item()):
            return "; ".join((*pieces, "one or more rows have zero valid HISA blocks"))
        partial_rows = (prefix_lens % block_size) != 0
        if bool(partial_rows.any().item()) and "last" not in tuple(
            config.forced_boundary_blocks or ()
        ):
            return "; ".join((*pieces, "partial causal rows require forced last block"))
        return "; ".join(
            (
                *pieces,
                "eligible by static checks; selector returned None during candidate planning",
            )
        )

    if backend in ("packed_cuda", "packed_cublasdx", "packed_cublasdx_tiled", "packed_cublasdx_fp8"):
        if get_indexcache_nvfp4_packed_tensors(k_rows) is None:
            return "; ".join((*pieces, "missing NVFP4 IndexCache packed sidecar on k_rows"))
        ext = _try_load_hisa_cuda_ext()
        if ext is None:
            return "; ".join((*pieces, "HISA CUDA extension unavailable"))
        if backend == "packed_cuda":
            ext_op = "hisa_selector_nvfp4_fwd"
        elif backend == "packed_cublasdx":
            ext_op = "hisa_selector_nvfp4_cublasdx_fwd"
        elif backend == "packed_cublasdx_tiled":
            ext_op = "hisa_selector_nvfp4_cublasdx_tiled_fwd"
        else:
            ext_op = "hisa_selector_nvfp4_cublasdx_fp8_fwd"
        if not hasattr(ext, ext_op):
            return "; ".join((*pieces, f"HISA CUDA extension lacks {ext_op}"))
        if backend != "packed_cuda" and (heads != 64 or head_dim != 128):
            return "; ".join(
                (*pieces, f"{backend} requires q shape [Q,64,128], got heads={heads} dim={head_dim}")
            )
        return "; ".join((*pieces, "eligible by static checks; packed selector returned None"))

    if backend == "cuda":
        ext = _try_load_hisa_cuda_ext()
        if ext is None:
            return "; ".join((*pieces, "HISA CUDA extension unavailable"))
        if not hasattr(ext, "hisa_selector_fwd"):
            return "; ".join((*pieces, "HISA CUDA extension lacks hisa_selector_fwd"))
        return "; ".join((*pieces, "eligible by static checks; CUDA selector returned None"))

    if backend in ("auto", "bmm"):
        return "; ".join((*pieces, "BMM HISA selected-score path should be available"))

    return "; ".join((*pieces, "unknown backend state"))


def indexcache_hisa_cuda_select_scores_teacher(
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    attn_query_rows: torch.Tensor,
    attn_key_rows: torch.Tensor,
    topk: int,
    *,
    config: IndexCacheHISAConfig,
    prefix_lens: torch.Tensor,
    softmax_scale: float,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """CUDA HISA selector fused with local teacher probability construction."""

    if (
        not _hisa_selector_cuda_enabled()
        or not q_rows.is_cuda
        or not attn_query_rows.is_cuda
        or not config.is_optimized
        or not _is_blackwell_or_newer(q_rows.device)
    ):
        return None
    ext = _try_load_hisa_cuda_ext()
    if ext is None or not hasattr(ext, "hisa_selector_teacher_fwd"):
        return None

    sq, _, head_dim = q_rows.shape
    sk = k_rows.shape[0]
    topk_k = min(int(topk), sk)
    if topk_k <= 0:
        return None
    if attn_query_rows.dim() != 3 or attn_key_rows.dim() != 3:
        return None
    if attn_query_rows.shape[0] != sq or attn_key_rows.shape[0] != sk:
        return None
    if attn_query_rows.shape[1:] != attn_key_rows.shape[1:]:
        return None
    if attn_query_rows.shape[-1] <= 0 or attn_query_rows.shape[-1] > 256:
        return None
    prefix_lens = prefix_lens.to(device=q_rows.device, dtype=torch.long).clamp(0, sk)
    if config.fallback_to_dense_if_short and int(prefix_lens.max().item()) <= topk_k:
        return None

    block_size = int(config.block_size)
    block_count = int(math.ceil(sk / block_size))
    if block_count <= 0:
        return None

    row_block_counts = torch.div(
        prefix_lens + block_size - 1, block_size, rounding_mode="floor"
    ).to(torch.int32)
    if config.compression_ratio > 0:
        block_topk_counts, effective_block_topk = hisa_block_topk_counts(
            row_block_counts,
            block_size=block_size,
            topk_tokens=topk_k,
            compression_ratio=config.compression_ratio,
        )
    else:
        block_topk_counts = torch.full(
            (sq,),
            min(int(config.block_topk), block_count),
            device=q_rows.device,
            dtype=torch.int32,
        )
        effective_block_topk = min(int(config.block_topk), block_count)
    block_topk_counts, effective_block_topk = _include_forced_boundary_budget(
        block_topk_counts,
        effective_block_topk,
        row_block_counts,
        tuple(config.forced_boundary_blocks or ()),
    )

    q_f = q_rows.contiguous().float()
    w_f = weights_rows.contiguous().float()
    k_f = k_rows.contiguous().float()
    aq_f = attn_query_rows.contiguous().float()
    ak_f = attn_key_rows.contiguous().float()
    reps = _mean_pool_all_blocks(k_f, block_size).contiguous()
    prefix_i32 = prefix_lens.to(dtype=torch.int32).contiguous()
    indices_i32 = torch.empty((sq, topk_k), device=q_rows.device, dtype=torch.int32)
    selected_scores = torch.empty((sq, topk_k), device=q_rows.device, dtype=torch.float32)
    teacher_probs = torch.empty((sq, topk_k), device=q_rows.device, dtype=torch.float32)
    forced = tuple(config.forced_boundary_blocks or ())

    ext.hisa_selector_teacher_fwd(
        q_f,
        k_f,
        reps,
        w_f,
        aq_f,
        ak_f,
        prefix_i32,
        block_topk_counts.contiguous(),
        indices_i32,
        selected_scores,
        teacher_probs,
        block_size,
        int(effective_block_topk),
        int(topk_k),
        float(softmax_scale),
        "first" in forced,
        "last" in forced,
        "last_minus_one" in forced,
    )
    return indices_i32, selected_scores, teacher_probs


def indexcache_hisa_topk_with_scores(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk: int,
    *,
    config: IndexCacheHISAConfig,
    q_start: int,
    is_causal: bool,
    mask: Optional[torch.Tensor],
    query_positions: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    return_scores: bool = False,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Batched HISA top-k selection for one DSA query chunk.

    The original reference helper walks rows, blocks, and candidates in Python.
    This path keeps the same HISA contract but moves the hot work to batched
    GPU tensor ops. It returns ``topk_indices [B, Q, K]`` and, when requested,
    the corresponding selected indexer logits flattened as ``[B*Q, K]`` so the
    DSA indexer KL can train through the selected candidate scores.
    """

    if not config.enabled or mask is not None:
        return None
    if q.dim() != 4 or k.dim() != 3 or weights.dim() != 3:
        return None

    sq, bsz, _, head_dim = q.shape
    sk = k.shape[0]
    if head_dim != k.shape[-1]:
        return None
    topk_k = min(topk, sk)
    if topk_k <= 0:
        return None

    prefix_lens = _prefix_lens_for_hisa_chunk(
        sq,
        sk,
        q_start=q_start,
        is_causal=is_causal,
        query_positions=query_positions,
        key_positions=key_positions,
        device=q.device,
    )
    if prefix_lens is None:
        return None
    if config.fallback_to_dense_if_short and int(prefix_lens.max().item()) <= topk_k:
        return None

    block_size = int(config.block_size)
    block_count = int(math.ceil(sk / block_size))
    if block_count <= 0:
        return None

    topk_out = torch.full((bsz, sq, topk_k), -1, device=q.device, dtype=torch.int32)
    score_rows = [] if return_scores else None

    # HISA's block stage is a selector. Keep it out of the autograd graph; the
    # selected candidate logits below are what carry indexer-loss gradients.
    with torch.no_grad():
        row_block_counts = torch.div(
            prefix_lens + block_size - 1, block_size, rounding_mode="floor"
        ).to(torch.int32)
        if config.compression_ratio > 0:
            block_topk_counts, effective_block_topk = hisa_block_topk_counts(
                row_block_counts,
                block_size=block_size,
                topk_tokens=topk_k,
                compression_ratio=config.compression_ratio,
            )
        else:
            block_topk_counts = None
            effective_block_topk = min(config.block_topk, block_count)
        block_topk_counts, effective_block_topk = _include_forced_boundary_budget(
            block_topk_counts,
            effective_block_topk,
            row_block_counts,
            tuple(config.forced_boundary_blocks or ()),
        )

    for batch_idx in range(bsz):
        q_b = q[:, batch_idx].float()
        w_b = weights[:, batch_idx].float()
        k_b = k[:, batch_idx].float()

        cuda_result = _indexcache_hisa_topk_cuda_for_batch(
            q_b,
            w_b,
            k_b,
            topk_k,
            config=config,
            prefix_lens=prefix_lens,
            block_topk_counts=block_topk_counts,
            effective_block_topk=effective_block_topk,
            return_scores=return_scores,
        )
        if cuda_result is not None:
            cuda_indices, cuda_scores = cuda_result
            topk_out[batch_idx] = cuda_indices
            if return_scores:
                score_rows.append(cuda_scores)
            continue

        running_indices, running_scores = _indexcache_hisa_topk_bmm_for_batch(
            q_b,
            w_b,
            k_b,
            topk_k,
            config=config,
            prefix_lens=prefix_lens,
            block_topk_counts=block_topk_counts,
            effective_block_topk=effective_block_topk,
        )
        topk_out[batch_idx] = running_indices
        if return_scores:
            score_rows.append(running_scores)

    if not bool((topk_out >= 0).any().item()):
        return None
    selected_scores = torch.cat(score_rows, dim=0) if return_scores else None
    return topk_out.contiguous(), selected_scores


def indexcache_hisa_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk: int,
    *,
    config: IndexCacheHISAConfig,
    q_start: int,
    is_causal: bool,
    mask: Optional[torch.Tensor],
    query_positions: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return HISA top-k indices for one DSA query chunk, or ``None`` to fallback."""

    fast = indexcache_hisa_topk_with_scores(
        q,
        weights,
        k,
        topk,
        config=config,
        q_start=q_start,
        is_causal=is_causal,
        mask=mask,
        query_positions=query_positions,
        key_positions=key_positions,
        return_scores=False,
    )
    if fast is not None:
        return fast[0]

    if not config.enabled:
        return None
    if mask is not None:
        return None
    if q.dim() != 4 or k.dim() != 3 or weights.dim() != 3:
        return None

    sq, bsz, _, _ = q.shape
    sk = k.shape[0]
    topk_k = min(topk, sk)
    if topk_k <= 0:
        return None

    if is_causal:
        if query_positions is not None or key_positions is not None:
            if query_positions is None or key_positions is None:
                return None
            if query_positions.dim() != 1 or key_positions.dim() != 1:
                return None
            if query_positions.numel() != sq or key_positions.numel() != sk:
                return None
            query_positions = query_positions.to(device=q.device, dtype=torch.long).contiguous()
            key_positions = key_positions.to(device=q.device, dtype=torch.long).contiguous()
            if (
                os.getenv(_HISA_ASSUME_SORTED_POSITIONS_ENV, "1").strip().lower()
                in {"0", "false", "off", "no"}
                and key_positions.numel() > 1
                and bool((key_positions[1:] < key_positions[:-1]).any().item())
            ):
                return None
            prefix_lens = torch.searchsorted(key_positions, query_positions, right=True)
        else:
            prefix_lens = torch.arange(q_start + 1, q_start + sq + 1, device=q.device)
        prefix_lens = prefix_lens.clamp_(0, sk)
        if config.fallback_to_dense_if_short and int(prefix_lens.max().item()) <= topk_k:
            return None
        if config.fallback_to_dense_if_short and int(prefix_lens.min().item()) <= topk_k:
            return _dense_qk_topk_for_chunk(
                q,
                weights,
                k,
                topk,
                is_causal=is_causal,
                q_start=q_start,
                query_positions=query_positions,
                key_positions=key_positions,
            )
    else:
        if config.fallback_to_dense_if_short and sk <= topk_k:
            return None
        prefix_lens = torch.full((sq,), sk, device=q.device, dtype=torch.long)

    with torch.no_grad():
        qf = q.float()
        wf = weights.float()
        kf = k.float()
        block_size = config.block_size
        block_count = int(math.ceil(sk / block_size))
        reps = torch.stack(
            [_mean_pool_blocks(kf[:, batch_idx, :], block_size) for batch_idx in range(bsz)],
            dim=1,
        )
        topk_indices = torch.empty(
            (bsz, sq, topk_k), device=q.device, dtype=torch.int32
        )
        topk_indices.fill_(-1)
        for batch_idx in range(bsz):
            reps_b = reps[:, batch_idx, :]
            block_scores = _weighted_relu_dsa_score(
                qf[:, batch_idx, :, :], reps_b, wf[:, batch_idx, :]
            )
            row_block_counts = torch.div(
                prefix_lens + block_size - 1, block_size, rounding_mode="floor"
            ).to(torch.int32)
            arange_blocks = torch.arange(block_count, device=q.device, dtype=torch.int32).view(
                1, -1
            )
            block_scores = block_scores.masked_fill(
                arange_blocks >= row_block_counts.view(-1, 1),
                float("-inf"),
            )
            if config.compression_ratio > 0:
                block_topk_counts, effective_block_topk = hisa_block_topk_counts(
                    row_block_counts,
                    block_size=block_size,
                    topk_tokens=topk_k,
                    compression_ratio=config.compression_ratio,
                )
            else:
                block_topk_counts = None
                effective_block_topk = min(config.block_topk, block_count)
            block_topk_counts, effective_block_topk = _include_forced_boundary_budget(
                block_topk_counts,
                effective_block_topk,
                row_block_counts,
                tuple(config.forced_boundary_blocks or ()),
            )
            top_blocks = _select_hisa_blocks(
                block_scores,
                row_block_counts,
                block_topk=effective_block_topk,
                block_topk_counts=block_topk_counts,
                forced_boundary_blocks=config.forced_boundary_blocks,
            )

            for row in range(sq):
                prefix_len = int(prefix_lens[row].item())
                valid_blocks = top_blocks[row][top_blocks[row] >= 0]
                ranges = []
                for block_id in valid_blocks.tolist():
                    start = block_id * block_size
                    end = min(start + block_size, prefix_len)
                    if start < end:
                        ranges.append(torch.arange(start, end, device=q.device, dtype=torch.int32))
                if not ranges:
                    return None
                candidate_indices = torch.cat(ranges).unique(sorted=False)
                if candidate_indices.numel() <= topk_k:
                    topk_indices[batch_idx, row, : candidate_indices.numel()] = candidate_indices
                    continue
                candidate_k = kf[:prefix_len, batch_idx, :].index_select(0, candidate_indices)
                candidate_scores = _weighted_relu_dsa_score(
                    qf[row : row + 1, batch_idx, :, :],
                    candidate_k,
                    wf[row : row + 1, batch_idx, :],
                ).squeeze(0)
                rel = torch.topk(candidate_scores, k=topk_k, sorted=False).indices
                topk_indices[batch_idx, row, :] = candidate_indices.index_select(0, rel)

    if not bool((topk_indices >= 0).any().item()):
        return None
    return topk_indices.contiguous()
