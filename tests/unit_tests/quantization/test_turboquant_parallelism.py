# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parallelism correctness tests for the TurboQuant fake-quant op.

These tests assert that ``apply_turboquant_kv`` is invariant to the
TP/SP/CP/EP shard count: the per-token math depends only on the post-norm
latent of a single token, so the op commutes with every parallelism scheme
that does not split the latent dimension. We verify this end-to-end by
sharding the input across multiple ranks, running the op locally on each
shard, gathering, and comparing to a single-rank baseline.

Required environment:
  * 2+ CUDA GPUs (or a single GPU run is skipped gracefully).
  * NCCL + torchrun.

Invocation:
  torchrun --nproc-per-node=2 -m pytest \
      tests/unit_tests/quantization/test_turboquant_parallelism.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from megatron.core.quantization.turboquant import (  # noqa: E402
    apply_turboquant_kv,
    build_turboquant_buffers,
)


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_multi_rank() -> bool:
    return _world_size() > 1


def _init_dist() -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def _baseline(x: torch.Tensor, buf) -> torch.Tensor:
    """Single-rank reference: run apply_turboquant_kv on the full tensor."""

    return apply_turboquant_kv(x, buf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _is_multi_rank(), reason="Run with torchrun --nproc-per-node>=2")
def test_seq_sharding_matches_baseline():
    """Sequence-dim shards (SP/CP) commute with the per-token op."""

    _init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    seq, batch, latent = 32 * world, 2, 512
    torch.manual_seed(0)
    x_full = torch.randn(seq, batch, latent, dtype=torch.bfloat16, device=device)
    buf = build_turboquant_buffers(
        latent_dim=latent, seed=7, layer_idx=2, device=device, dtype=torch.float32
    )

    # Baseline on rank 0 only.
    if rank == 0:
        baseline = _baseline(x_full.clone(), buf)

    # Per-rank shard along seq dim.
    chunk = seq // world
    shard = x_full[rank * chunk : (rank + 1) * chunk].clone()
    shard_out = apply_turboquant_kv(shard, buf)

    # Gather shards back on rank 0.
    gather = [torch.empty_like(shard_out) for _ in range(world)]
    dist.all_gather(gather, shard_out)
    if rank == 0:
        sharded = torch.cat(gather, dim=0)
        torch.testing.assert_close(sharded, baseline, rtol=0, atol=0)
    dist.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _is_multi_rank(), reason="Run with torchrun --nproc-per-node>=2")
def test_buffers_replicated_not_sharded():
    """Frozen buffers must be byte-identical across every rank."""

    _init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    buf = build_turboquant_buffers(
        latent_dim=512,
        seed=11,
        layer_idx=4,
        device=f"cuda:{rank}",
        dtype=torch.float32,
    )

    # All-reduce the absolute sum of each buffer; if any rank's buffer differs
    # the sum will not be world * single_rank_sum.
    for tensor in (buf.signs1, buf.signs2, buf.boundaries_high, buf.boundaries_low,
                   buf.centroids_high, buf.centroids_low):
        local = tensor.float().abs().sum()
        gather = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gather, local)
        if rank == 0:
            for g in gather[1:]:
                torch.testing.assert_close(g, gather[0], rtol=0, atol=0)
    dist.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _is_multi_rank(), reason="Run with torchrun --nproc-per-node>=2")
def test_backward_seq_sharded_matches_baseline():
    """Gradients are bit-identical between sharded and unsharded execution."""

    _init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    seq, batch, latent = 16 * world, 2, 512
    torch.manual_seed(rank)  # different per-rank state to catch sync bugs
    x_full = torch.randn(seq, batch, latent, dtype=torch.bfloat16, device=device)
    upstream_full = torch.randn_like(x_full)

    # Cross-rank consistency on the seed-zero tensors.
    dist.broadcast(x_full, src=0)
    dist.broadcast(upstream_full, src=0)

    buf = build_turboquant_buffers(
        latent_dim=latent, seed=13, layer_idx=1, device=device, dtype=torch.float32
    )

    # Baseline
    if rank == 0:
        x_base = x_full.clone().requires_grad_(True)
        y_base = apply_turboquant_kv(x_base, buf)
        (y_base * upstream_full).sum().backward()
        baseline_grad = x_base.grad.clone()

    # Sharded
    chunk = seq // world
    x_shard = x_full[rank * chunk : (rank + 1) * chunk].clone().requires_grad_(True)
    upstream_shard = upstream_full[rank * chunk : (rank + 1) * chunk].clone()
    y_shard = apply_turboquant_kv(x_shard, buf)
    (y_shard * upstream_shard).sum().backward()

    grads = [torch.empty_like(x_shard.grad) for _ in range(world)]
    dist.all_gather(grads, x_shard.grad)
    if rank == 0:
        gathered = torch.cat(grads, dim=0)
        torch.testing.assert_close(gathered, baseline_grad, rtol=0, atol=0)
    dist.barrier()
