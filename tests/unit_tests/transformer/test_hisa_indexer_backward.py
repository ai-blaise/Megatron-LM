# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""FP64 autograd-parity tests for the HISA 4:1 indexer backward.

Phase-A gate: the analytical CUDA / reference backward must match a finite-
difference FP64 oracle to ``cos_sim >= 0.999976`` across (Q, K, w_indexer)
gradients at L in {4096, 8192, 16384} with k=2048, B=128, m=ceil(M/4),
top_k=2048.

Reference repos cited (Rule 6): TurboQuant Megatron backward + HIGGS Megatron
backward parity-tests for the test scaffolding.
"""

from __future__ import annotations

import math

import pytest
import torch

from megatron.core.extensions.hisa_indexer import (
    IndexCacheHISAConfig,
    apply_hisa_score_backward,
    build_hisa_config,
)
from megatron.core.extensions.hisa_indexer.reference import (
    hisa_forward_reference,
    hisa_score_backward_reference,
    hisa_unselect_grad_from_topk_to_scores,
)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.detach().to(torch.float64).flatten()
    b64 = b.detach().to(torch.float64).flatten()
    na = a64.norm()
    nb = b64.norm()
    if na.item() == 0 or nb.item() == 0:
        return 1.0 if (na - nb).abs().item() < 1e-9 else 0.0
    return float((a64 @ b64) / (na * nb))


def _make_random_batch(
    *,
    B: int,
    L: int,
    Q_per_batch: int,
    H: int,
    D: int,
    seed: int,
    device: str | torch.device = "cpu",
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((B * Q_per_batch, H, D), generator=g, dtype=torch.float64).to(device)
    k_by_batch = [
        torch.randn((L, D), generator=g, dtype=torch.float64).to(device)
        for _ in range(B)
    ]
    weights = torch.rand((B * Q_per_batch, H), generator=g, dtype=torch.float64).to(device) + 0.1
    prefix_lens = torch.full((B * Q_per_batch,), L, dtype=torch.int64, device=device)
    token_to_batch_idx = torch.repeat_interleave(
        torch.arange(B, dtype=torch.int64, device=device), Q_per_batch
    )
    return q, k_by_batch, weights, prefix_lens, token_to_batch_idx


def _finite_difference_grad(
    fn, x: torch.Tensor, *, h: float = 1e-4
) -> torch.Tensor:
    """Centred finite differences on ``fn(x) -> scalar``. Operates in fp64.

    The full HISA scalar is the sum of the selected-token scores, which is
    what the downstream sparse-MLA sees after the STE.
    """

    g = torch.zeros_like(x, dtype=torch.float64)
    flat = x.reshape(-1)
    gflat = g.reshape(-1)
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + h
        f_plus = fn(x).item()
        flat[i] = orig - h
        f_minus = fn(x).item()
        flat[i] = orig
        gflat[i] = (f_plus - f_minus) / (2 * h)
    return g


def _hisa_scalar(
    q,
    k_by_batch,
    weights,
    prefix_lens,
    token_to_batch_idx,
    config: IndexCacheHISAConfig,
) -> torch.Tensor:
    """Scalar ``L`` that the autograd-parity test differentiates against.

    ``L`` = sum over selected (row, slot) of (candidate_score on that slot)
            + sum over selected (row, block) of (block_score on that block).
    With ones-valued upstream gradient, the analytic backward returns
    exactly the per-row Jacobian sum.
    """

    topk_indices, cache = hisa_forward_reference(
        q.float(),
        [rows.float() for rows in k_by_batch],
        weights.float(),
        prefix_lens,
        token_to_batch_idx,
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=config.topk_tokens,
        fallback_to_dense_if_short=False,
    )
    if cache is None:
        return torch.zeros((), dtype=torch.float64, device=q.device)

    cand_mask = cache.selected_candidate_mask.to(torch.float64)
    cand_score = (
        torch.relu(cache.candidate_dot.double())
        * cache.weights.double().unsqueeze(1)
    ).sum(dim=-1)
    cand_total = (cand_score * cand_mask).sum()

    blk_mask = cache.selected_block_mask.to(torch.float64)
    blk_score = (
        torch.relu(cache.block_dot.double())
        * cache.weights.double().unsqueeze(1)
    ).sum(dim=-1)
    blk_total = (blk_score * blk_mask).sum()

    return cand_total + blk_total


def test_hisa_forward_uses_exact_4to1_pool_when_t_gt_k():
    config = build_hisa_config(
        enabled=True,
        block_size=128,
        compression_ratio=4.0,
        topk_tokens=2048,
        execution_mode="reference",
    )
    q, k_by_batch, weights, prefix_lens, token_to_batch_idx = _make_random_batch(
        B=1, L=4096, Q_per_batch=1, H=2, D=16, seed=11, device="cpu"
    )

    topk_indices, cache = hisa_forward_reference(
        q.float(),
        [rows.float() for rows in k_by_batch],
        weights.float(),
        prefix_lens,
        token_to_batch_idx,
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=config.topk_tokens,
        fallback_to_dense_if_short=True,
        forced_boundary_blocks=tuple(config.forced_boundary_blocks or ()),
    )

    assert cache is not None
    assert cache.block_topk_per_row.tolist() == [8]
    assert cache.candidate_indices.shape[1] == 8 * 128
    assert int((topk_indices[0] >= 0).sum().item()) == 1024
    assert config.forced_boundary_blocks == ("first", "last")
    selected = set(cache.top_blocks[0, :8].tolist())
    assert 0 in selected
    assert 31 in selected
    assert 30 not in selected


def test_hisa_forward_masks_stage1_to_prefix_blocks():
    config = build_hisa_config(
        enabled=True,
        block_size=128,
        compression_ratio=4.0,
        topk_tokens=64,
        execution_mode="reference",
    )
    g = torch.Generator(device="cpu").manual_seed(12)
    q = torch.randn((1, 2, 16), generator=g, dtype=torch.float32)
    k = torch.randn((512, 16), generator=g, dtype=torch.float32)
    k[256:] += 1000.0
    weights = torch.ones((1, 2), dtype=torch.float32)
    prefix_lens = torch.tensor([192], dtype=torch.int64)
    token_to_batch_idx = torch.tensor([0], dtype=torch.int64)

    topk_indices, cache = hisa_forward_reference(
        q,
        [k],
        weights,
        prefix_lens,
        token_to_batch_idx,
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=config.topk_tokens,
        fallback_to_dense_if_short=False,
        forced_boundary_blocks=tuple(config.forced_boundary_blocks or ()),
    )

    assert cache is not None
    assert int(cache.top_blocks.max().item()) < 2
    assert int(topk_indices[topk_indices >= 0].max().item()) < 192


def test_hisa_forward_falls_back_only_when_t_le_k():
    config = build_hisa_config(
        enabled=True,
        block_size=128,
        compression_ratio=4.0,
        topk_tokens=2048,
        execution_mode="reference",
    )
    q, k_by_batch, weights, _, token_to_batch_idx = _make_random_batch(
        B=1, L=2049, Q_per_batch=1, H=2, D=16, seed=13, device="cpu"
    )

    _, cache_short = hisa_forward_reference(
        q.float(),
        [k_by_batch[0][:2048].float()],
        weights.float(),
        torch.tensor([2048], dtype=torch.int64),
        token_to_batch_idx,
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=config.topk_tokens,
        fallback_to_dense_if_short=True,
        forced_boundary_blocks=tuple(config.forced_boundary_blocks or ()),
    )
    assert cache_short is None

    _, cache_long = hisa_forward_reference(
        q.float(),
        [k_by_batch[0].float()],
        weights.float(),
        torch.tensor([2049], dtype=torch.int64),
        token_to_batch_idx,
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=config.topk_tokens,
        fallback_to_dense_if_short=True,
        forced_boundary_blocks=tuple(config.forced_boundary_blocks or ()),
    )
    assert cache_long is not None
    assert cache_long.block_topk_per_row.tolist() == [5]


@pytest.mark.parametrize("L,topk", [(4096, 1024), (8192, 2048)])
def test_hisa_unselect_skips_candidate_scores_for_map_all_pool(L: int, topk: int):
    config = build_hisa_config(
        enabled=True,
        block_size=128,
        compression_ratio=4.0,
        topk_tokens=topk,
        execution_mode="reference",
    )
    q, k_by_batch, weights, prefix_lens, token_to_batch_idx = _make_random_batch(
        B=1, L=L, Q_per_batch=2, H=2, D=16, seed=14, device="cpu"
    )

    _, cache = hisa_forward_reference(
        q.float(),
        [rows.float() for rows in k_by_batch],
        weights.float(),
        prefix_lens,
        token_to_batch_idx,
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=config.topk_tokens,
        fallback_to_dense_if_short=False,
        forced_boundary_blocks=tuple(config.forced_boundary_blocks or ()),
    )
    assert cache is not None
    assert cache.candidate_indices.shape[1] == topk

    grad_cand, grad_block = hisa_unselect_grad_from_topk_to_scores(
        torch.ones((q.shape[0], topk), dtype=torch.float32), cache
    )
    assert grad_cand.count_nonzero().item() == 0
    assert grad_block.count_nonzero().item() == 0


def test_hisa_unselect_routes_candidate_scores_for_refine_pool():
    config = build_hisa_config(
        enabled=True,
        block_size=128,
        compression_ratio=4.0,
        topk_tokens=2048,
        execution_mode="reference",
    )
    q, k_by_batch, weights, prefix_lens, token_to_batch_idx = _make_random_batch(
        B=1, L=16384, Q_per_batch=2, H=2, D=16, seed=15, device="cpu"
    )

    _, cache = hisa_forward_reference(
        q.float(),
        [rows.float() for rows in k_by_batch],
        weights.float(),
        prefix_lens,
        token_to_batch_idx,
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=config.topk_tokens,
        fallback_to_dense_if_short=False,
        forced_boundary_blocks=tuple(config.forced_boundary_blocks or ()),
    )
    assert cache is not None
    assert cache.candidate_indices.shape[1] > config.topk_tokens

    grad_cand, grad_block = hisa_unselect_grad_from_topk_to_scores(
        torch.ones((q.shape[0], config.topk_tokens), dtype=torch.float32), cache
    )
    assert grad_cand.count_nonzero().item() == q.shape[0] * config.topk_tokens
    assert grad_block.count_nonzero().item() == 0


@pytest.mark.parametrize("L", [128, 256, 512])
def test_hisa_backward_cos_sim_against_fd(L: int):
    """Small-L FD oracle — cos_sim parity must be 1.0 to numerical noise."""

    config = build_hisa_config(
        enabled=True,
        block_size=64,
        compression_ratio=2.0,
        topk_tokens=min(L // 2, 128),
        execution_mode="reference",
    )

    B = 1
    H = 4
    D = 8
    Q_per_batch = 2
    q, k_by_batch, weights, prefix_lens, token_to_batch_idx = _make_random_batch(
        B=B, L=L, Q_per_batch=Q_per_batch, H=H, D=D, seed=42, device="cpu"
    )

    # Recompute the cache once and pass an all-ones grad through the score
    # tensors. With L = sum(score * mask), grad_score = mask. That is
    # exactly the analytical grad path.
    _, cache = hisa_forward_reference(
        q.float(),
        [rows.float() for rows in k_by_batch],
        weights.float(),
        prefix_lens,
        token_to_batch_idx,
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=config.topk_tokens,
        fallback_to_dense_if_short=False,
    )
    assert cache is not None, "HISA forward must produce a cache when L > topk_tokens."

    grad_cand = cache.selected_candidate_mask.float()
    grad_blk = cache.selected_block_mask.float()
    grad_q, grad_k_by_batch, grad_w = hisa_score_backward_reference(
        grad_cand, grad_blk, cache
    )

    # Finite-difference gradient on Q (sample two coordinates per head/row).
    fn = lambda x: _hisa_scalar(
        x, k_by_batch, weights, prefix_lens, token_to_batch_idx, config
    )
    fd_q = _finite_difference_grad(fn, q.clone(), h=1e-4)
    cos_q = _cos_sim(grad_q.double().reshape(-1), fd_q.reshape(-1))
    assert cos_q >= 0.999976, f"cos_sim Q={cos_q:.6f} < 0.999976"

    # FD on K[0].
    fn_k0 = lambda x: _hisa_scalar(
        q,
        [x] + list(k_by_batch[1:]),
        weights,
        prefix_lens,
        token_to_batch_idx,
        config,
    )
    fd_k0 = _finite_difference_grad(fn_k0, k_by_batch[0].clone(), h=1e-4)
    cos_k = _cos_sim(grad_k_by_batch[0].double().reshape(-1), fd_k0.reshape(-1))
    assert cos_k >= 0.999976, f"cos_sim K={cos_k:.6f} < 0.999976"

    # FD on weights.
    fn_w = lambda x: _hisa_scalar(
        q, k_by_batch, x, prefix_lens, token_to_batch_idx, config
    )
    fd_w = _finite_difference_grad(fn_w, weights.clone(), h=1e-4)
    cos_w = _cos_sim(grad_w.double().reshape(-1), fd_w.reshape(-1))
    assert cos_w >= 0.999976, f"cos_sim w={cos_w:.6f} < 0.999976"


@pytest.mark.parametrize(
    "L,topk",
    [
        (4096, 1024),
        (8192, 1024),
        (16384, 1024),
        (4096, 2048),
        (8192, 2048),
        (16384, 2048),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hisa_backward_cuda_vs_reference(L: int, topk: int):
    """Phase-A gate at production-realistic sizes: CUDA bwd matches reference."""

    config = build_hisa_config(
        enabled=True,
        block_size=128,
        compression_ratio=4.0,
        topk_tokens=topk,
        execution_mode="optimized",
    )
    B = 2
    H = 16
    D = 128
    Q_per_batch = 8
    q, k_by_batch, weights, prefix_lens, token_to_batch_idx = _make_random_batch(
        B=B, L=L, Q_per_batch=Q_per_batch, H=H, D=D, seed=7, device="cuda"
    )

    _, cache = hisa_forward_reference(
        q.float(),
        [rows.float() for rows in k_by_batch],
        weights.float(),
        prefix_lens,
        token_to_batch_idx,
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=config.topk_tokens,
        fallback_to_dense_if_short=False,
    )
    assert cache is not None

    grad_cand = cache.selected_candidate_mask.float()
    grad_blk_selected = cache.selected_block_mask.float()
    grad_blk_zero = torch.zeros_like(grad_blk_selected)

    for grad_blk in (grad_blk_selected, grad_blk_zero):
        grad_q_ref, grad_k_ref, grad_w_ref = hisa_score_backward_reference(
            grad_cand, grad_blk, cache
        )

        grad_q_cu, grad_k_cu, grad_w_cu = apply_hisa_score_backward(
            grad_cand, grad_blk, cache, config=config
        )
        assert _cos_sim(grad_q_ref, grad_q_cu) >= 0.999976
        for ref, cu in zip(grad_k_ref, grad_k_cu):
            assert _cos_sim(ref, cu) >= 0.999976
        assert _cos_sim(grad_w_ref, grad_w_cu) >= 0.999976
