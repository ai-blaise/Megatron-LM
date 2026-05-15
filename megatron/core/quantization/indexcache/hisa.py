# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""HISA selector helpers for NVFP4 IndexCache DSA indexer.

The IndexCache quantizer remains responsible for producing the NVFP4
fake-quantized K tensor; HISA chooses a block-compressed candidate set before
the usual sparse DSA attention consumes token indices. The training helpers in
``megatron.core.extensions.hisa_indexer`` reuse this config surface for the
score-formula backward.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch


_HISA_SELECTOR_CUDA_ENV = "MEGATRON_HISA_SELECTOR_CUDA"
_HISA_CANDIDATE_SLOT_GROUP_ENV = "MEGATRON_HISA_CANDIDATE_SLOT_GROUP"


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
    return scores.topk(topk_k, dim=-1, sorted=False).indices


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
            if key_positions.numel() > 1 and bool(
                (key_positions[1:] < key_positions[:-1]).any().item()
            ):
                return None
            prefix_lens = torch.searchsorted(key_positions, query_positions, right=True)
        else:
            prefix_lens = torch.arange(q_start + 1, q_start + sq + 1, device=device)
        return prefix_lens.clamp_(0, sk).to(torch.long)
    return torch.full((sq,), sk, device=device, dtype=torch.long)


def _hisa_selector_cuda_enabled() -> bool:
    return os.getenv(_HISA_SELECTOR_CUDA_ENV, "0").lower() not in ("0", "false", "no")


def _hisa_candidate_slot_group() -> int:
    raw = os.getenv(_HISA_CANDIDATE_SLOT_GROUP_ENV, "8")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{_HISA_CANDIDATE_SLOT_GROUP_ENV} must be positive, got {value}")
    return value


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
    valid = topk_indices >= 0
    safe_idx = topk_indices.clamp_min(0).reshape(-1)
    selected_k = k_rows.index_select(0, safe_idx).view(sq, topk_k, head_dim)
    dot = torch.einsum("qhd,qkd->qkh", q_rows, selected_k)
    scores = (torch.relu(dot) * weights_rows.unsqueeze(1)).sum(dim=-1)
    return scores.masked_fill(~valid, float("-inf"))


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
    sk = k_rows.shape[0]
    offsets = torch.arange(block_size, device=q_rows.device, dtype=torch.long)
    running_scores = q_rows.new_full((sq, topk_k), float("-inf"))
    running_indices = torch.full((sq, topk_k), -1, device=q_rows.device, dtype=torch.long)
    slot_group = min(_hisa_candidate_slot_group(), max(1, top_blocks.shape[1]))

    for slot_start in range(0, top_blocks.shape[1], slot_group):
        block_ids = top_blocks[:, slot_start : slot_start + slot_group]
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
        safe_idx = candidate_indices.clamp(0, max(sk - 1, 0)).reshape(-1)
        candidate_k = k_rows.index_select(0, safe_idx).view(sq, -1, head_dim)

        # [Q, H, D] x [Q, D, G*B] -> [Q, H, G*B]. This keeps exact fp32
        # reference semantics while replacing many per-slot score kernels with
        # fewer batched GEMMs. If the run enables TF32 matmul globally, cuBLAS
        # can map this same path onto tensor cores without changing dispatch.
        cand_dot = torch.bmm(q_rows, candidate_k.transpose(1, 2))
        cand_scores = (torch.relu(cand_dot) * weights_rows.unsqueeze(-1)).sum(dim=1)
        cand_scores = cand_scores.masked_fill(~valid.reshape(sq, -1), float("-inf"))

        merged_scores = torch.cat((running_scores, cand_scores), dim=-1)
        merged_indices = torch.cat((running_indices, candidate_indices), dim=-1)
        running_scores, gather_pos = torch.topk(
            merged_scores, k=topk_k, dim=-1, sorted=False
        )
        running_indices = merged_indices.gather(1, gather_pos)

    valid_final = torch.isfinite(running_scores)
    return running_indices.masked_fill(~valid_final, -1), running_scores


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
    topk_indices = indices_i32.to(torch.long)
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

    topk_out = torch.full((bsz, sq, topk_k), -1, device=q.device, dtype=torch.long)
    score_rows = [] if return_scores else None
    arange_blocks = torch.arange(block_count, device=q.device, dtype=torch.long)

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

        with torch.no_grad():
            reps = _mean_pool_all_blocks(k_b, block_size)
            block_dot = torch.einsum("qhd,md->qmh", q_b, reps)
            block_scores = (torch.relu(block_dot) * w_b.unsqueeze(1)).sum(dim=-1)
            block_scores = block_scores.masked_fill(
                arange_blocks.view(1, -1) >= row_block_counts.long().view(-1, 1),
                float("-inf"),
            )

            # Correct the per-row partial final block. A global block mean would
            # leak future tokens inside the current causal block.
            if is_causal:
                last_blocks = (row_block_counts.long() - 1).clamp_min(0)
                starts = last_blocks * block_size
                csum = torch.cat(
                    (k_b.new_zeros((1, head_dim)), k_b.cumsum(dim=0)), dim=0
                )
                safe_prefix = prefix_lens.clamp(0, sk)
                sums = csum.index_select(0, safe_prefix) - csum.index_select(0, starts)
                counts = (safe_prefix - starts).clamp_min(1).to(k_b.dtype).unsqueeze(-1)
                partial_reps = sums / counts
                partial_dot = torch.einsum("qhd,qd->qh", q_b, partial_reps)
                partial_scores = (torch.relu(partial_dot) * w_b).sum(dim=-1)
                block_scores.scatter_(1, last_blocks.view(-1, 1), partial_scores.view(-1, 1))

            valid_rows = row_block_counts > 0
            if "first" in config.forced_boundary_blocks:
                block_scores[valid_rows, 0] = float("inf")
            if "last" in config.forced_boundary_blocks:
                last_blocks = (row_block_counts.long() - 1).clamp_min(0)
                block_scores.scatter_(1, last_blocks.view(-1, 1), torch.full((sq, 1), float("inf"), device=q.device))
            if "last_minus_one" in config.forced_boundary_blocks:
                prev_blocks = (row_block_counts.long() - 2).clamp_min(0)
                has_prev = row_block_counts > 1
                block_scores[has_prev] = block_scores[has_prev].scatter(
                    1,
                    prev_blocks[has_prev].view(-1, 1),
                    torch.full((int(has_prev.sum().item()), 1), float("inf"), device=q.device),
                )

            block_keep = min(effective_block_topk, block_count)
            top_block_values, top_blocks = torch.topk(
                block_scores, k=block_keep, dim=-1, sorted=False
            )
            top_blocks = top_blocks.masked_fill(
                torch.isnan(top_block_values) | (top_block_values == float("-inf")), -1
            )
            if block_topk_counts is not None and block_keep > 0:
                slot_ids = torch.arange(block_keep, device=q.device).view(1, -1)
                top_blocks = top_blocks.masked_fill(
                    slot_ids >= block_topk_counts.long().view(-1, 1), -1
                )

        running_indices, running_scores = _hisa_grouped_candidate_topk(
            q_b,
            w_b,
            k_b,
            top_blocks,
            prefix_lens,
            topk_k,
            block_size,
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
            if key_positions.numel() > 1 and bool(
                (key_positions[1:] < key_positions[:-1]).any().item()
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
            (bsz, sq, topk_k), device=q.device, dtype=torch.long
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
            arange_blocks = torch.arange(block_count, device=q.device).view(1, -1)
            block_scores = block_scores.masked_fill(
                arange_blocks >= row_block_counts.to(torch.long).view(-1, 1),
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
                        ranges.append(torch.arange(start, end, device=q.device))
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
