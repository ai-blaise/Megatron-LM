# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""HISA 4:1 indexer reference implementation (FP64 autograd-parity oracle).

This module implements the forward and analytical backward in pure PyTorch.
It is used as the oracle for the FP64 cos-sim parity gate in Phase A and as
the deterministic fallback when the CUDA kernel is unavailable.

The backward derivation matches the per-stage Jacobians described in
``megatron/core/extensions/hisa_indexer/__init__.py``:

  - mean_pool: ``d L / d k_s += (1/N_b) * d L / d k_block_b`` for s in block b.
  - block_score / candidate_score: a weighted-ReLU dot product whose Jacobian
    is straightforward; the ReLU non-smooth point uses the standard
    subgradient ``H(x) = 1 iff x > 0``.
  - top-k selection: pass-through (STE) — the selection mask is treated as
    constant during the backward.
  - NVFP4 dequant: STE-detach on the per-group scale, gradient flows through
    ``dequant(quant(x))`` ≡ ``x`` for purposes of computing ``d L / d x``.

The reference path keeps the IndexCache K in dense FP32; the CUDA path
operates on the packed NVFP4 + UE8M0 layout the inference forward kernel
already lays down in the IndexCache.

Rule 6 references: HIGGS-Megatron backward
(``megatron/core/quantization/higgs/kernels/csrc/higgs_kv_bwd.cu``) for the
STE-detach pattern; TurboQuant Megatron backward for the module layout;
vLLM ``csrc/moe/topk_softmax_kernels.cu::topkGating`` for the XOR-butterfly
reduction we mirror in the CUDA path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HISAForwardCache:
    """Saved intermediates for the analytical backward."""

    q: torch.Tensor                 # [Q, H, D] fp32 — recomputed Q dense
    k_by_batch: list[torch.Tensor]  # length B; each [L_b, D] fp32
    weights: torch.Tensor           # [Q, H]
    token_to_batch_idx: torch.Tensor  # [Q] int64
    prefix_lens: torch.Tensor       # [Q] int64
    block_size: int
    block_topk_per_row: torch.Tensor  # [Q] int32
    top_blocks: torch.Tensor          # [Q, max_block_topk] int32
    candidate_indices: torch.Tensor   # [Q, candidate_len] int32 (-1 padding)
    candidate_dot: torch.Tensor       # [Q, candidate_len, H] fp32 (pre-ReLU)
    block_dot: torch.Tensor           # [Q, max_blocks, H] fp32 (pre-ReLU)
    selected_block_mask: torch.Tensor  # [Q, max_blocks] bool
    selected_candidate_mask: torch.Tensor  # [Q, candidate_len] bool
    topk_positions: torch.Tensor      # [Q, topk_tokens] int64 in {0..candidate_len-1}
    topk_indices: torch.Tensor        # [Q, topk_tokens] int32 (-1 padding)


def mean_pool_blocks(k_rows: torch.Tensor, block_size: int) -> torch.Tensor:
    """``k_block_b = mean({k_s : s in block b})`` — linear, has Jacobian."""

    n = k_rows.shape[0]
    block_count = (n + block_size - 1) // block_size
    pad = block_count * block_size - n
    if pad > 0:
        padded = torch.cat(
            [k_rows, k_rows.new_zeros((pad, k_rows.shape[-1]))], dim=0
        )
        counts = k_rows.new_full((block_count,), float(block_size))
        counts[-1] = float(block_size - pad)
    else:
        padded = k_rows
        counts = k_rows.new_full((block_count,), float(block_size))
    pooled = padded.reshape(block_count, block_size, -1).sum(dim=1)
    return pooled / counts.unsqueeze(-1)


def weighted_relu_dsa_score(
    q: torch.Tensor, k: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(score, raw_dot)`` for the indexer formula.

    ``score_{t,b} = sum_h w_{t,h} * ReLU(q_{t,h} . k_b)``
    ``raw_dot_{t,b,h} = q_{t,h} . k_b``

    We save ``raw_dot`` to recover the ReLU mask in the backward.
    """

    raw_dot = torch.einsum("qhd,kd->qkh", q, k)
    score = (torch.relu(raw_dot) * w.unsqueeze(1)).sum(dim=-1)
    return score, raw_dot


def hisa_forward_reference(
    q: torch.Tensor,
    k_by_batch: list[torch.Tensor],
    weights: torch.Tensor,
    prefix_lens: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    *,
    block_size: int = 128,
    compression_ratio: float = 4.0,
    topk_tokens: int = 2048,
    block_topk: Optional[int] = None,
    fallback_to_dense_if_short: bool = True,
    forced_boundary_blocks: tuple[str, ...] = ("first", "last"),
) -> tuple[torch.Tensor, Optional[HISAForwardCache]]:
    """FP32 reference forward producing token indices + saved cache.

    Returns ``(topk_indices [Q, topk_tokens] int32, cache)`` or
    ``(None, None)`` when every prefix is short enough for dense (matches
    the SGLang OP fallback path).
    """

    device = q.device
    Q, H, D = q.shape
    q = q.float()
    weights = weights.float()
    prefix_lens = prefix_lens.reshape(-1).to(device=device, dtype=torch.int64)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=device, dtype=torch.int64
    )

    if fallback_to_dense_if_short and bool(torch.all(prefix_lens <= topk_tokens).item()):
        return None, None

    batch_block_counts = [
        (rows.shape[0] + block_size - 1) // block_size for rows in k_by_batch
    ]
    max_blocks = max(batch_block_counts) if batch_block_counts else 0
    row_block_counts = torch.empty((Q,), device=device, dtype=torch.int32)
    for row in range(Q):
        batch_idx = int(token_to_batch_idx[row].item())
        prefix_len = min(int(prefix_lens[row].item()), k_by_batch[batch_idx].shape[0])
        row_block_counts[row] = (prefix_len + block_size - 1) // block_size

    if block_topk is None:
        # 4:1 -> ceil(M / 4), where M is the per-row eligible prefix block
        # count. Do not lift this to topk_tokens / block_size: for small
        # t > k, HISA keeps the requested compression ratio and the candidate
        # top-k returns the whole compressed candidate set.
        ratio = float(compression_ratio)
        block_topk_counts = torch.ceil(row_block_counts.float() / ratio).to(torch.int32)
        block_topk_counts = torch.minimum(block_topk_counts, row_block_counts)
        block_topk_eff = max(1, int(block_topk_counts.max().item()))
    else:
        block_topk_counts = torch.full(
            (Q,), block_topk, device=device, dtype=torch.int32
        )
        block_topk_counts = torch.minimum(block_topk_counts, row_block_counts)
        block_topk_eff = max(1, block_topk)

    # Stage 1: mean_pool + block_score.
    block_dot = q.new_zeros((Q, max_blocks, H))
    block_scores = q.new_full((Q, max_blocks), float("-inf"))
    for row in range(Q):
        batch_idx = int(token_to_batch_idx[row].item())
        prefix_len = min(int(prefix_lens[row].item()), k_by_batch[batch_idx].shape[0])
        n_blocks = int(row_block_counts[row].item())
        if prefix_len <= 0 or n_blocks <= 0:
            continue
        reps = mean_pool_blocks(
            k_by_batch[batch_idx][:prefix_len].to(device).float(), block_size
        )
        score, raw_dot = weighted_relu_dsa_score(
            q[row : row + 1], reps, weights[row : row + 1]
        )
        block_scores[row, : score.shape[1]] = score[0]
        block_dot[row, : raw_dot.shape[1], :] = raw_dot[0]

    # Stage 1c: block top-k (with boundary forcing).
    top_blocks = torch.full(
        (Q, block_topk_eff), -1, device=device, dtype=torch.int32
    )
    selected_block_mask = torch.zeros((Q, max_blocks), device=device, dtype=torch.bool)
    for row in range(Q):
        n_blocks = int(row_block_counts[row].item())
        if n_blocks == 0:
            continue
        scores = block_scores[row, :n_blocks].clone()
        forced = _forced_block_indices(n_blocks, forced_boundary_blocks, device)
        if forced.numel() > 0:
            scores[forced] = float("inf")
        keep = min(int(block_topk_counts[row].item()), block_topk_eff, n_blocks)
        if keep <= 0:
            continue
        idx = torch.topk(scores, k=keep, sorted=False).indices.to(torch.int32)
        top_blocks[row, :keep] = idx
        selected_block_mask[row].scatter_(0, idx.long(), True)

    # Stage 2: candidate enumeration + score.
    candidate_len = block_topk_eff * block_size
    candidate_indices = torch.full(
        (Q, candidate_len), -1, device=device, dtype=torch.int32
    )
    candidate_dot = q.new_zeros((Q, candidate_len, H))
    selected_candidate_mask = torch.zeros(
        (Q, candidate_len), device=device, dtype=torch.bool
    )
    candidate_scores = q.new_full((Q, candidate_len), float("-inf"))
    for row in range(Q):
        prefix_len = int(prefix_lens[row].item())
        batch_idx = int(token_to_batch_idx[row].item())
        rows_k = k_by_batch[batch_idx].to(device).float()
        for slot in range(block_topk_eff):
            block_id = int(top_blocks[row, slot].item())
            if block_id < 0:
                continue
            start = block_id * block_size
            end = min(start + block_size, prefix_len)
            for j, tok in enumerate(range(start, end)):
                cand_slot = slot * block_size + j
                candidate_indices[row, cand_slot] = tok
                k_row = rows_k[tok]
                dot = (q[row] * k_row.unsqueeze(0)).sum(dim=-1)  # [H]
                candidate_dot[row, cand_slot, :] = dot
                score = (torch.relu(dot) * weights[row]).sum()
                candidate_scores[row, cand_slot] = score
                selected_candidate_mask[row, cand_slot] = True

    # Stage 2c: candidate top-k.
    keep = min(topk_tokens, candidate_len)
    topk_positions = torch.full(
        (Q, topk_tokens), -1, device=device, dtype=torch.int64
    )
    topk_indices = torch.full(
        (Q, topk_tokens), -1, device=device, dtype=torch.int32
    )
    for row in range(Q):
        valid = selected_candidate_mask[row].nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        scores = candidate_scores[row, valid]
        k = min(keep, valid.numel())
        rel = torch.topk(scores, k=k, sorted=False).indices
        positions = valid[rel]
        topk_positions[row, :k] = positions.to(torch.int64)
        topk_indices[row, :k] = candidate_indices[row, positions]

    cache = HISAForwardCache(
        q=q,
        k_by_batch=[rows.to(device).float() for rows in k_by_batch],
        weights=weights,
        token_to_batch_idx=token_to_batch_idx,
        prefix_lens=prefix_lens,
        block_size=block_size,
        block_topk_per_row=block_topk_counts,
        top_blocks=top_blocks,
        candidate_indices=candidate_indices,
        candidate_dot=candidate_dot,
        block_dot=block_dot,
        selected_block_mask=selected_block_mask,
        selected_candidate_mask=selected_candidate_mask,
        topk_positions=topk_positions,
        topk_indices=topk_indices,
    )
    return topk_indices, cache


def _forced_block_indices(
    n_blocks: int, names: tuple[str, ...], device: torch.device
) -> torch.Tensor:
    idx_set: set[int] = set()
    for name in names:
        if name == "first":
            idx_set.add(0)
        elif name == "last":
            idx_set.add(n_blocks - 1)
        elif name == "last_minus_one":
            if n_blocks >= 2:
                idx_set.add(n_blocks - 2)
        else:
            raise ValueError(f"Unknown forced-block name {name!r}.")
    idx_set = {i for i in idx_set if 0 <= i < n_blocks}
    return torch.tensor(sorted(idx_set), device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# Analytical backward
# ---------------------------------------------------------------------------


def hisa_score_backward_reference(
    grad_candidate_scores: torch.Tensor,
    grad_block_scores: torch.Tensor,
    cache: HISAForwardCache,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Analytical backward through both score formulas.

    Args:
        grad_candidate_scores: ``[Q, candidate_len]`` upstream gradient on the
            per-candidate score (after the ReLU + weighted sum, before the
            top-k STE).
        grad_block_scores: ``[Q, max_blocks]`` upstream gradient on the
            block-level score (after the ReLU + weighted sum, before the
            block top-k STE).
        cache: saved forward intermediates.

    Returns:
        ``(grad_Q [Q, H, D] fp32, grad_K_by_batch list of [L_b, D] fp32,
           grad_w_indexer [Q, H] fp32)``.

    Notes:
        * The selection masks are treated as constant (STE pass-through at the
          two top-k boundaries).
        * The ReLU non-smooth point uses the standard subgradient.
        * NVFP4 STE is applied at the consumer level — the reference path
          works on dense FP32 K.
    """

    q = cache.q
    weights = cache.weights
    Q, H, D = q.shape
    grad_q = torch.zeros_like(q)
    grad_w = torch.zeros_like(weights)
    grad_k_by_batch = [torch.zeros_like(rows) for rows in cache.k_by_batch]

    block_size = cache.block_size

    # ------------------------------------------------------------------
    # Backward through stage-2 candidate_score:
    #   I_{t,s} = sum_h w_{t,h} * ReLU(q_{t,h} . k_s)
    # d I / d q_{t,h} = w_{t,h} * H(dot_{t,s,h}) * k_s
    # d I / d k_s     = sum_h w_{t,h} * H(dot_{t,s,h}) * q_{t,h}
    # d I / d w_{t,h} = ReLU(dot_{t,s,h})
    # ------------------------------------------------------------------
    cand_mask = cache.selected_candidate_mask  # [Q, candidate_len]
    cand_idx = cache.candidate_indices         # [Q, candidate_len]
    cand_dot = cache.candidate_dot             # [Q, candidate_len, H]
    relu_mask_c = (cand_dot > 0).float()
    # grad on `I` -> per-head pre-ReLU contribution
    g = grad_candidate_scores.unsqueeze(-1) * weights.unsqueeze(1) * relu_mask_c
    g = g * cand_mask.float().unsqueeze(-1)  # zero out masked candidates

    for row in range(Q):
        batch_idx = int(cache.token_to_batch_idx[row].item())
        prefix_len = int(cache.prefix_lens[row].item())
        valid = cand_mask[row].nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        tokens = cand_idx[row, valid].long()
        # Filter against prefix in case of stale padding.
        ok = (tokens >= 0) & (tokens < prefix_len)
        valid = valid[ok]
        tokens = tokens[ok]
        if valid.numel() == 0:
            continue

        k_rows = cache.k_by_batch[batch_idx].index_select(0, tokens)  # [S, D]
        g_row = g[row, valid]  # [S, H]

        # d L / d q_{row, h} += sum_s g_row[s, h] * k_rows[s]
        grad_q[row] += torch.einsum("sh,sd->hd", g_row, k_rows)
        # d L / d k_rows[s] += sum_h g_row[s, h] * q[row, h]
        contrib = torch.einsum("sh,hd->sd", g_row, q[row])  # [S, D]
        grad_k_by_batch[batch_idx].index_add_(0, tokens, contrib)
        # d L / d w_{row, h} += sum_s grad_candidate_scores[row, valid] * ReLU(dot)[s, h]
        grad_w[row] += (
            grad_candidate_scores[row, valid].unsqueeze(-1)
            * torch.relu(cand_dot[row, valid])
        ).sum(dim=0)

    # ------------------------------------------------------------------
    # Backward through stage-1 block_score + mean_pool:
    #   J_{t,b} = sum_h w_{t,h} * ReLU(q_{t,h} . k_block_b)
    #   k_block_b = mean({k_s : s in block b})
    # d J / d q_{t,h} = w_{t,h} * H(block_dot_{t,b,h}) * k_block_b
    # d J / d k_block_b = sum_h w_{t,h} * H(block_dot_{t,b,h}) * q_{t,h}
    # d k_block_b / d k_s = 1 / N_b   for s in block b
    # ------------------------------------------------------------------
    block_dot = cache.block_dot  # [Q, max_blocks, H]
    selected_block_mask = cache.selected_block_mask.float()
    relu_mask_b = (block_dot > 0).float()
    gb = grad_block_scores.unsqueeze(-1) * weights.unsqueeze(1) * relu_mask_b
    gb = gb * selected_block_mask.unsqueeze(-1)

    for row in range(Q):
        batch_idx = int(cache.token_to_batch_idx[row].item())
        prefix_len = int(cache.prefix_lens[row].item())
        rows_k = cache.k_by_batch[batch_idx]
        L = min(prefix_len, rows_k.shape[0])
        if L <= 0:
            continue
        n_blocks = (L + block_size - 1) // block_size
        # Recompute the per-row pooled reps and per-block lengths.
        reps = mean_pool_blocks(rows_k[:L].float(), block_size)
        N_b = torch.full((n_blocks,), float(block_size), device=q.device)
        if L % block_size != 0:
            N_b[-1] = float(L - (n_blocks - 1) * block_size)
        active_blocks = selected_block_mask[row, :n_blocks].nonzero(as_tuple=False).flatten()
        if active_blocks.numel() == 0:
            continue
        gb_row = gb[row, active_blocks]  # [B_act, H]
        reps_active = reps[active_blocks]  # [B_act, D]

        # grad_Q += sum_b gb[b, h] * reps[b]
        grad_q[row] += torch.einsum("bh,bd->hd", gb_row, reps_active)
        # grad on block reps
        d_rep = torch.einsum("bh,hd->bd", gb_row, q[row])  # [B_act, D]
        # propagate mean-pool: 1/N_b distributed to each row in the block
        for j, b in enumerate(active_blocks.tolist()):
            start = b * block_size
            end = min(start + block_size, L)
            if end <= start:
                continue
            n_tokens = float(end - start)
            grad_k_by_batch[batch_idx][start:end] += d_rep[j].unsqueeze(0) / n_tokens
        # grad_w += sum_b grad_block_scores[b] * ReLU(block_dot[b])
        grad_w[row] += (
            grad_block_scores[row, active_blocks].unsqueeze(-1)
            * torch.relu(block_dot[row, active_blocks])
        ).sum(dim=0)

    return grad_q, grad_k_by_batch, grad_w


def hisa_unselect_grad_from_topk_to_scores(
    grad_topk_logit: torch.Tensor,
    cache: HISAForwardCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    """STE pass-through: route ``d L / d (selected-score)`` back into the
    candidate-score and block-score score tensors.

    The downstream sparse-MLA produces ``grad_topk_logit`` of shape
    ``[Q, topk_tokens]`` (one scalar per selected token). When HISA ran the
    candidate-score refine stage, we route that gradient back into
    ``grad_candidate_scores[Q, candidate_len]`` at the slot the token came
    from. When the compressed candidate pool is no larger than ``topk_tokens``,
    the OP inference kernel maps the selected blocks directly and never
    materializes candidate scores; in that map-all case there is no
    candidate-score gradient to route. The block-topk boundary remains pure
    STE, so ``grad_block_scores`` is zero in both cases.

    Callers that want a non-zero block-score gradient can pass it directly
    to :func:`hisa_score_backward_reference`.
    """

    Q = grad_topk_logit.shape[0]
    if cache.candidate_indices.shape[1] <= grad_topk_logit.shape[1]:
        empty_candidate = cache.candidate_dot.new_empty((Q, 0))
        empty_block = cache.block_dot.new_empty((Q, 0))
        return empty_candidate, empty_block

    grad_candidate_scores = torch.zeros_like(cache.candidate_dot[:, :, 0])
    valid = cache.topk_positions >= 0
    rows, slots = valid.nonzero(as_tuple=True)
    pos = cache.topk_positions[rows, slots]
    grad_candidate_scores[rows, pos] = grad_topk_logit[rows, slots]
    grad_block_scores = torch.zeros(
        (Q, cache.block_dot.shape[1]),
        dtype=cache.block_dot.dtype,
        device=cache.block_dot.device,
    )
    return grad_candidate_scores, grad_block_scores
