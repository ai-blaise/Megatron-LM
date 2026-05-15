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
from dataclasses import dataclass
from typing import Optional

import torch


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
) -> torch.Tensor:
    sq, bsz, _, _ = q.shape
    sk = k.shape[0]
    topk_k = min(topk, sk)
    scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    scores = torch.relu_(scores)
    scores.mul_(weights.unsqueeze(-1).float())
    scores = scores.sum(dim=2).transpose(0, 1)
    if is_causal:
        q_pos = torch.arange(q_start, q_start + sq, device=q.device).view(1, sq, 1)
        k_pos = torch.arange(sk, device=q.device).view(1, 1, sk)
        scores.masked_fill_(k_pos > q_pos, float("-inf"))
    return scores.topk(topk_k, dim=-1, sorted=False).indices


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

    if not config.enabled:
        return None
    if mask is not None or query_positions is not None or key_positions is not None:
        return None
    if q.dim() != 4 or k.dim() != 3 or weights.dim() != 3:
        return None

    sq, bsz, _, _ = q.shape
    sk = k.shape[0]
    topk_k = min(topk, sk)
    if topk_k <= 0:
        return None

    if is_causal:
        prefix_lens = torch.arange(q_start + 1, q_start + sq + 1, device=q.device)
        if config.fallback_to_dense_if_short and int(prefix_lens.max().item()) <= topk_k:
            return None
        if config.fallback_to_dense_if_short and int(prefix_lens.min().item()) <= topk_k:
            return _dense_qk_topk_for_chunk(
                q, weights, k, topk, is_causal=is_causal, q_start=q_start
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
