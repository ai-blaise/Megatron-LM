# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Runtime dispatch for the HISA 4:1 score backward."""

from __future__ import annotations

from typing import Optional

import torch

from megatron.core.quantization.indexcache import IndexCacheHISAConfig
from megatron.core.extensions.hisa_indexer.reference import (
    HISAForwardCache,
    hisa_forward_reference,
    hisa_score_backward_reference,
)


def _try_load_cuda_ext():
    """Lazy-load the CUDA extension. Returns ``None`` on non-CUDA hosts."""

    try:
        from megatron.core.extensions.hisa_indexer.kernels.build import get_ext

        return get_ext()
    except Exception:
        return None


def _is_blackwell_or_newer(device: torch.device | int | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return major >= 10


def hisa_selector_forward_and_save(
    q: torch.Tensor,
    k_concat: torch.Tensor,
    k_offsets: torch.Tensor,
    weights: torch.Tensor,
    prefix_lens: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    config: IndexCacheHISAConfig,
) -> tuple[torch.Tensor, Optional[HISAForwardCache]]:
    """Forward returning ``(topk_indices, HISAForwardCache)`` for callers that
    integrate the STE themselves and want explicit access to the saved
    intermediates without going through autograd save_for_backward."""

    k_offsets_cpu = k_offsets.detach().cpu().tolist()
    k_by_batch = [
        k_concat[k_offsets_cpu[i] : k_offsets_cpu[i + 1]]
        for i in range(len(k_offsets_cpu) - 1)
    ]
    topk_indices, cache = hisa_forward_reference(
        q.detach(),
        [rows.detach() for rows in k_by_batch],
        weights.detach(),
        prefix_lens,
        token_to_batch_idx,
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=config.topk_tokens,
        fallback_to_dense_if_short=config.fallback_to_dense_if_short,
        forced_boundary_blocks=tuple(config.forced_boundary_blocks or ()),
    )
    if topk_indices is None:
        empty = torch.full(
            (q.shape[0], config.topk_tokens),
            -1,
            device=q.device,
            dtype=torch.int32,
        )
        return empty, None
    return topk_indices, cache


def apply_hisa_score_backward(
    grad_candidate_scores: torch.Tensor,
    grad_block_scores: torch.Tensor,
    cache: HISAForwardCache,
    *,
    config: IndexCacheHISAConfig,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Direct backward-only entry over the saved cache. Returns dense
    ``(grad_Q [Q,H,D], grad_K per batch list, grad_w [Q,H])``.
    """

    use_cuda = (
        cache.q.is_cuda
        and config.is_optimized
        and _is_blackwell_or_newer(cache.q.device)
        and _try_load_cuda_ext() is not None
    )
    ext = _try_load_cuda_ext() if use_cuda else None
    return _dispatch_score_backward(
        grad_candidate_scores, grad_block_scores, cache, cuda_path=use_cuda, cuda_ext=ext
    )


def _dispatch_score_backward(
    grad_candidate_scores: torch.Tensor,
    grad_block_scores: torch.Tensor,
    cache: HISAForwardCache,
    *,
    cuda_path: bool,
    cuda_ext,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Dispatch to CUDA when available, fall back to reference otherwise."""

    if grad_candidate_scores.numel() == 0 and grad_block_scores.numel() == 0:
        return (
            torch.zeros_like(cache.q),
            [torch.zeros_like(rows) for rows in cache.k_by_batch],
            torch.zeros_like(cache.weights),
        )
    if cuda_path and cuda_ext is not None and hasattr(cuda_ext, "hisa_score_bwd"):
        return _cuda_score_backward(
            grad_candidate_scores, grad_block_scores, cache, cuda_ext
        )
    return hisa_score_backward_reference(
        grad_candidate_scores, grad_block_scores, cache
    )


def _cuda_score_backward(
    grad_candidate_scores: torch.Tensor,
    grad_block_scores: torch.Tensor,
    cache: HISAForwardCache,
    cuda_ext,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Pack the cache into the CUDA-extension layout and dispatch.

    Layout invariants enforced here:
      - ``q`` is contiguous fp32 ``[Q, H, D]``.
      - ``weights`` is contiguous fp32 ``[Q, H]``.
      - ``k_concat`` is contiguous fp32 ``[sum_b L_b, D]`` with cumulative
        offsets ``k_offsets [B+1]`` int32.
      - ``candidate_indices`` is int32 with -1 padding for unfilled slots.
      - ``candidate_dot`` is fp32 ``[Q, candidate_len, H]``.
      - ``block_dot`` is fp32 ``[Q, max_blocks, H]``.
      - ``selected_block_mask``, ``selected_candidate_mask`` are uint8.
    """

    q = cache.q.contiguous().float()
    weights = cache.weights.contiguous().float()
    Q, H, D = q.shape
    candidate_len = cache.candidate_dot.shape[1]
    max_blocks = cache.block_dot.shape[1]

    # Concatenate K and build offsets.
    k_concat = torch.cat([rows.contiguous().float() for rows in cache.k_by_batch], dim=0)
    L_list = [rows.shape[0] for rows in cache.k_by_batch]
    k_offsets = torch.tensor(
        [0] + list(_cumulative(L_list)), device=q.device, dtype=torch.int32
    )

    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k_concat)
    grad_w = torch.zeros_like(weights)

    cuda_ext.hisa_score_bwd(
        grad_candidate_scores.contiguous().float(),
        grad_block_scores.contiguous().float(),
        q,
        k_concat,
        k_offsets,
        weights,
        cache.token_to_batch_idx.to(torch.int32).contiguous(),
        cache.prefix_lens.to(torch.int32).contiguous(),
        cache.top_blocks.to(torch.int32).contiguous(),
        cache.candidate_indices.to(torch.int32).contiguous(),
        cache.candidate_dot.contiguous().float(),
        cache.block_dot.contiguous().float(),
        cache.selected_block_mask.to(torch.uint8).contiguous(),
        cache.selected_candidate_mask.to(torch.uint8).contiguous(),
        grad_q,
        grad_k,
        grad_w,
        int(cache.block_size),
    )
    grad_k_by_batch: list[torch.Tensor] = []
    cumulative = 0
    for L in L_list:
        grad_k_by_batch.append(grad_k[cumulative : cumulative + L])
        cumulative += L
    return grad_q, grad_k_by_batch, grad_w


def _cumulative(values: list[int]) -> list[int]:
    out: list[int] = []
    total = 0
    for v in values:
        total += int(v)
        out.append(total)
    return out
