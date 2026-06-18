#!/usr/bin/env python3
"""Validate the strict DeepEP dispatch-to-expert-major path on one NVLink node."""

from __future__ import annotations

import argparse
import inspect
import os

import torch
import torch.distributed as dist

from megatron.core.fusions.fused_deepep_permute import (
    deepep_indices_permute,
    deepep_indices_unpermute,
)
from megatron.core.transformer.moe.fused_a2a import (
    fused_combine,
    fused_combine_expert_major,
    fused_dispatch,
    fused_dispatch_expert_major,
    set_deepep_num_sms,
)


def _init_dist(local_rank: int, world_size: int):
    torch.cuda.set_device(local_rank)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{os.getenv('MASTER_ADDR', '127.0.0.1')}:{os.getenv('MASTER_PORT', '29573')}",
        "world_size": world_size,
        "rank": local_rank,
    }
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    group = dist.new_group(list(range(world_size)))
    return group


def _require_patched_deepep():
    from deep_ep import Buffer

    missing = [
        name
        for name in ("dispatch_expert_major", "combine_expert_major")
        if not hasattr(Buffer, name)
    ]
    if missing:
        raise RuntimeError(f"patched DeepEP APIs missing: {', '.join(missing)}")


def _configure_deepep_num_sms():
    value = os.getenv("MEGATRON_DEEPEP_NUM_SMS", "").strip()
    if not value:
        return
    num_sms = int(value)
    set_deepep_num_sms(num_sms)
    combine_value = os.getenv("MEGATRON_DEEPEP_COMBINE_NUM_SMS", "").strip()
    if combine_value and int(combine_value) != num_sms:
        raise RuntimeError(
            "MEGATRON_DEEPEP_COMBINE_NUM_SMS must match MEGATRON_DEEPEP_NUM_SMS"
        )


def _make_routing(num_tokens: int, num_experts: int, topk: int):
    scores = torch.randn((num_tokens, num_experts), device="cuda", dtype=torch.float32)
    topk_idx = torch.topk(scores, topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.randn((num_tokens, topk), device="cuda", dtype=torch.float32)
    return topk_idx.contiguous(), topk_weights.contiguous()


def _bench_ms(fn, iters: int, warmup: int):
    for _ in range(warmup):
        out = fn()
        if isinstance(out, tuple):
            del out
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
        if isinstance(out, tuple):
            del out
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(1, iters)


def _worker(local_rank: int, args):
    group = _init_dist(local_rank, args.num_processes)
    _require_patched_deepep()
    _configure_deepep_num_sms()
    torch.manual_seed(args.seed + local_rank)

    num_experts = args.num_processes * args.num_local_experts
    hidden = torch.randn(
        (args.num_tokens, args.hidden), device="cuda", dtype=torch.bfloat16
    )
    topk_idx, topk_weights = _make_routing(args.num_tokens, num_experts, args.num_topk)

    recv_hidden, recv_topk_idx, recv_topk_weights, counts, handle = fused_dispatch(
        hidden, topk_idx, topk_weights, num_experts, group
    )
    old_expert_hidden, _, old_row_map, old_edge_to_row, _ = deepep_indices_permute(
        recv_hidden, recv_topk_idx, recv_topk_weights, counts, 1
    )

    (
        new_expert_hidden,
        new_recv_topk_idx,
        _new_probs,
        new_row_map,
        new_edge_to_row,
        new_counts,
        new_handle,
    ) = fused_dispatch_expert_major(
        hidden,
        topk_idx,
        topk_weights,
        num_experts,
        args.num_local_experts,
        group,
    )

    old_recv_order = deepep_indices_unpermute(
        old_expert_hidden,
        old_row_map,
        recv_hidden.shape,
        recv_topk_idx,
        old_edge_to_row,
        args.num_local_experts,
    )
    new_recv_order = deepep_indices_unpermute(
        new_expert_hidden,
        new_row_map,
        recv_hidden.shape,
        new_recv_topk_idx,
        new_edge_to_row,
        args.num_local_experts,
    )
    old_combined, _ = fused_combine(old_recv_order, group, handle)
    new_combined, _ = fused_combine(new_recv_order, group, new_handle)
    strict_combined, _ = fused_combine_expert_major(
        new_expert_hidden,
        group,
        new_handle,
        new_recv_topk_idx,
        new_edge_to_row,
        new_row_map,
        args.num_local_experts,
    )
    torch.cuda.synchronize()

    diff = (old_combined.float() - new_combined.float()).abs()
    max_abs = diff.max()
    rel = max_abs / old_combined.float().abs().max().clamp_min(1.0)
    strict_diff = (old_combined.float() - strict_combined.float()).abs()
    strict_max_abs = strict_diff.max()
    strict_rel = strict_max_abs / old_combined.float().abs().max().clamp_min(1.0)
    count_match = torch.equal(counts.cpu(), new_counts.cpu())
    if local_rank == 0:
        print(
            "megatron_dispatch_expert_major parity "
            f"max_abs={max_abs.item():.6g} rel={rel.item():.6g} "
            f"strict_combine_max_abs={strict_max_abs.item():.6g} "
            f"strict_combine_rel={strict_rel.item():.6g} "
            f"old_rows={old_expert_hidden.shape[0]} "
            f"new_rows={new_expert_hidden.shape[0]} count_match={count_match}",
            flush=True,
        )
    if not count_match:
        raise AssertionError("expert token counts differ")
    if rel.item() >= args.rtol and max_abs.item() >= args.atol:
        raise AssertionError(f"parity failed: max_abs={max_abs.item()} rel={rel.item()}")
    if strict_rel.item() >= args.rtol and strict_max_abs.item() >= args.atol:
        raise AssertionError(
            "strict combine parity failed: "
            f"max_abs={strict_max_abs.item()} rel={strict_rel.item()}"
        )

    if args.bench_iters > 0:

        def old_path():
            r_hidden, r_idx, r_probs, r_counts, _handle = fused_dispatch(
                hidden, topk_idx, topk_weights, num_experts, group
            )
            return deepep_indices_permute(r_hidden, r_idx, r_probs, r_counts, 1)

        def new_path():
            return fused_dispatch_expert_major(
                hidden,
                topk_idx,
                topk_weights,
                num_experts,
                args.num_local_experts,
                group,
            )

        old_ms = _bench_ms(old_path, args.bench_iters, args.bench_warmup)
        new_ms = _bench_ms(new_path, args.bench_iters, args.bench_warmup)
        values = torch.tensor([old_ms, new_ms], device="cuda")
        dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
        if local_rank == 0:
            speedup = values[0].item() / max(values[1].item(), 1e-9)
            print(
                "dispatch_expert_major bench max_rank "
                f"old_dispatch_plus_compact_permute={values[0].item():.3f}ms "
                f"new_dispatch_expert_major={values[1].item():.3f}ms "
                f"speedup={speedup:.3f}x",
                flush=True,
            )

        def old_output_path():
            recv_order = deepep_indices_unpermute(
                new_expert_hidden,
                new_row_map,
                recv_hidden.shape,
                new_recv_topk_idx,
                new_edge_to_row,
                args.num_local_experts,
            )
            return fused_combine(recv_order, group, new_handle)

        def strict_output_path():
            return fused_combine_expert_major(
                new_expert_hidden,
                group,
                new_handle,
                new_recv_topk_idx,
                new_edge_to_row,
                new_row_map,
                args.num_local_experts,
            )

        old_combine_ms = _bench_ms(old_output_path, args.bench_iters, args.bench_warmup)
        strict_combine_ms = _bench_ms(strict_output_path, args.bench_iters, args.bench_warmup)
        values = torch.tensor([old_combine_ms, strict_combine_ms], device="cuda")
        dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
        if local_rank == 0:
            speedup = values[0].item() / max(values[1].item(), 1e-9)
            print(
                "combine_expert_major bench max_rank "
                f"old_unpermute_plus_combine={values[0].item():.3f}ms "
                f"strict_combine_expert_major={values[1].item():.3f}ms "
                f"speedup={speedup:.3f}x",
                flush=True,
            )

        recv_order_static = deepep_indices_unpermute(
            new_expert_hidden,
            new_row_map,
            recv_hidden.shape,
            new_recv_topk_idx,
            new_edge_to_row,
            args.num_local_experts,
        )
        torch.cuda.synchronize()

        def old_unpermute_only():
            return deepep_indices_unpermute(
                new_expert_hidden,
                new_row_map,
                recv_hidden.shape,
                new_recv_topk_idx,
                new_edge_to_row,
                args.num_local_experts,
            )

        def old_combine_only():
            return fused_combine(recv_order_static, group, new_handle)

        old_unpermute_ms = _bench_ms(old_unpermute_only, args.bench_iters, args.bench_warmup)
        old_combine_only_ms = _bench_ms(old_combine_only, args.bench_iters, args.bench_warmup)
        values = torch.tensor([old_unpermute_ms, old_combine_only_ms], device="cuda")
        dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
        if local_rank == 0:
            print(
                "combine_old_path_breakdown max_rank "
                f"compact_unpermute={values[0].item():.3f}ms "
                f"deepep_combine_only={values[1].item():.3f}ms",
                flush=True,
            )

    dist.barrier(group)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-local-experts", type=int, default=16)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--bench-iters", type=int, default=0)
    parser.add_argument("--bench-warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
