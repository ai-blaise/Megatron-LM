#!/usr/bin/env python3
"""NCCL and Megatron process-group smoke for the GCP A4 fleet."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Iterable


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


def _apply_local_rank_nccl_hca() -> None:
    if not _env_flag("MEGATRON_LOCAL_RANK_NCCL_HCA"):
        return

    local_rank = int(os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "0")))
    hcas = [
        item.strip()
        for item in os.getenv(
            "MEGATRON_NCCL_LOCAL_RANK_HCAS",
            "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
        ).split(",")
        if item.strip()
    ]
    if hcas:
        hca = hcas[local_rank % len(hcas)]
        if not hca.startswith(("=", "^")):
            hca = f"={hca}"
        os.environ["NCCL_IB_HCA"] = hca


_apply_local_rank_nccl_hca()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist


def log(message: str) -> None:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    now = time.strftime("%H:%M:%S")
    print(
        f"[{now} rank={rank} local={local_rank} host={socket.gethostname()}] {message}",
        flush=True,
    )


def parse_sizes(raw: str) -> list[int]:
    return [int(item) for item in raw.replace(" ", "").split(",") if item]


def env_summary() -> str:
    keys = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_MAX_NCHANNELS",
        "NCCL_CROSS_NIC",
        "NCCL_IB_QPS_PER_CONNECTION",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_RUNTIME_CONNECT",
        "GLOO_SOCKET_IFNAME",
        "MEGATRON_LOCAL_RANK_NCCL_HCA",
        "MEGATRON_NCCL_LOCAL_RANK_HCAS",
    )
    return " ".join(f"{key}={os.environ.get(key, '')}" for key in keys)


def all_reduce_sizes(sizes: Iterable[int], iters: int) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    expected = world_size * (world_size + 1) / 2
    for numel in sizes:
        for idx in range(iters):
            value = torch.full((numel,), float(rank + 1), device=device)
            dist.all_reduce(value)
            torch.cuda.synchronize()
            if not torch.allclose(value, torch.full_like(value, expected)):
                raise RuntimeError(
                    f"all_reduce mismatch numel={numel} iter={idx} "
                    f"got={value.flatten()[0].item()} expected={expected}"
                )
        log(f"world all_reduce passed numel={numel} iters={iters}")


def p2p_ring(numel: int) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    send_peer = (rank + 1) % world_size
    recv_peer = (rank - 1) % world_size
    send_tensor = torch.full((numel,), float(rank), device=device)
    recv_tensor = torch.full((numel,), -1.0, device=device)
    ops = [
        dist.P2POp(dist.isend, send_tensor, peer=send_peer),
        dist.P2POp(dist.irecv, recv_tensor, peer=recv_peer),
    ]
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    torch.cuda.synchronize()
    expected = torch.full_like(recv_tensor, float(recv_peer))
    if not torch.equal(recv_tensor, expected):
        raise RuntimeError(f"p2p mismatch rank={rank} got={recv_tensor} expected={expected}")
    log(f"world p2p ring passed numel={numel}")


def group_all_reduce(name: str, group: dist.ProcessGroup) -> None:
    group_rank = dist.get_rank(group=group)
    group_size = dist.get_world_size(group=group)
    device = torch.device("cuda", torch.cuda.current_device())
    value = torch.full((1,), float(group_rank + 1), device=device)
    dist.all_reduce(value, group=group)
    torch.cuda.synchronize()
    expected = group_size * (group_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"{name} all_reduce mismatch got={value.item()} expected={expected}")
    log(f"{name} all_reduce passed group_rank={group_rank} group_size={group_size}")


def group_all_to_all(name: str, group: dist.ProcessGroup) -> None:
    group_rank = dist.get_rank(group=group)
    group_size = dist.get_world_size(group=group)
    global_rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    member_tensor = torch.tensor([global_rank], dtype=torch.int64, device=device)
    gathered_members = [torch.empty_like(member_tensor) for _ in range(group_size)]
    dist.all_gather(gathered_members, member_tensor, group=group)
    members = [int(item.item()) for item in gathered_members]
    input_tensor = torch.tensor(
        [global_rank * 1000 + peer for peer in range(group_size)],
        dtype=torch.float32,
        device=device,
    )
    output_tensor = torch.empty_like(input_tensor)
    dist.all_to_all_single(output_tensor, input_tensor, group=group)
    torch.cuda.synchronize()
    expected = torch.tensor(
        [int(member) * 1000 + group_rank for member in members],
        dtype=torch.float32,
        device=device,
    )
    if not torch.equal(output_tensor, expected):
        raise RuntimeError(f"{name} all_to_all mismatch got={output_tensor} expected={expected}")
    log(f"{name} all_to_all passed group_rank={group_rank} group_size={group_size}")


def group_object_smoke(name: str, group: dist.ProcessGroup) -> None:
    group_rank = dist.get_rank(group=group)
    group_size = dist.get_world_size(group=group)
    payload = {"global_rank": dist.get_rank(), "group_rank": group_rank}
    gathered: list[dict[str, int] | None] = [None] * group_size
    dist.all_gather_object(gathered, payload, group=group)
    if any(item is None for item in gathered):
        raise RuntimeError(f"{name} object gather returned None entries")
    log(f"{name} all_gather_object passed group_rank={group_rank} group_size={group_size}")


def pipeline_p2p_smoke() -> None:
    from megatron.core import parallel_state

    group = parallel_state.get_pipeline_model_parallel_group()
    group_size = dist.get_world_size(group=group)
    if group_size <= 1:
        log("pipeline p2p skipped group_size=1")
        return

    global_rank = dist.get_rank()
    prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
    next_rank = parallel_state.get_pipeline_model_parallel_next_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    send_prev = torch.tensor([global_rank], dtype=torch.int64, device=device)
    send_next = torch.tensor([global_rank], dtype=torch.int64, device=device)
    recv_prev = torch.empty_like(send_prev)
    recv_next = torch.empty_like(send_next)
    ops = [
        dist.P2POp(dist.isend, send_prev, peer=prev_rank, group=group),
        dist.P2POp(dist.irecv, recv_prev, peer=prev_rank, group=group),
        dist.P2POp(dist.isend, send_next, peer=next_rank, group=group),
        dist.P2POp(dist.irecv, recv_next, peer=next_rank, group=group),
    ]
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    torch.cuda.synchronize()
    expected_prev = torch.tensor([prev_rank], dtype=torch.int64, device=device)
    expected_next = torch.tensor([next_rank], dtype=torch.int64, device=device)
    if not torch.equal(recv_prev, expected_prev):
        raise RuntimeError(
            f"pipeline p2p recv_prev mismatch got={recv_prev.item()} expected={prev_rank}"
        )
    if not torch.equal(recv_next, expected_next):
        raise RuntimeError(
            f"pipeline p2p recv_next mismatch got={recv_next.item()} expected={next_rank}"
        )
    log(
        "pipeline p2p passed "
        f"group_rank={dist.get_rank(group=group)} group_size={group_size} "
        f"prev={prev_rank} next={next_rank}"
    )


def megatron_group_smoke(args: argparse.Namespace) -> None:
    from megatron.core import parallel_state

    order = "tp-cp-ep-pp-dp" if args.use_tp_pp_dp_mapping else "tp-cp-ep-dp-pp"
    log(
        "initializing Megatron groups "
        f"tp={args.tp} pp={args.pp} cp={args.cp} ep={args.ep} etp={args.etp} order={order}"
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=args.cp,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=args.etp,
        distributed_timeout_minutes=max(1, args.timeout // 60),
        order=order,
        create_gloo_process_groups=True,
    )
    try:
        group_all_reduce("tp", parallel_state.get_tensor_model_parallel_group())
        group_all_reduce("pp", parallel_state.get_pipeline_model_parallel_group())
        group_all_reduce(
            "dp_cp",
            parallel_state.get_data_parallel_group(with_context_parallel=True),
        )
        group_all_reduce("expert_dp", parallel_state.get_expert_data_parallel_group())
        group_all_reduce("expert_model", parallel_state.get_expert_model_parallel_group())
        if args.group_all_to_all:
            group_all_to_all("expert_model", parallel_state.get_expert_model_parallel_group())
        if args.pipeline_p2p_smoke:
            pipeline_p2p_smoke()
        if args.object_smoke:
            group_object_smoke(
                "dp_cp_gloo",
                parallel_state.get_data_parallel_group_gloo(with_context_parallel=True),
            )
            group_object_smoke(
                "expert_dp_gloo",
                parallel_state.get_expert_data_parallel_group_gloo(),
            )
        dist.barrier()
        log("Megatron group smoke passed")
    finally:
        parallel_state.destroy_model_parallel()
        log("Megatron groups destroyed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--sizes", default="1,1024,1048576")
    parser.add_argument("--p2p-numel", type=int, default=4)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--pp", type=int, default=5)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--skip-megatron-groups", action="store_true")
    parser.add_argument("--skip-world-all-reduce", action="store_true")
    parser.add_argument("--skip-world-p2p", action="store_true")
    parser.add_argument("--no-init-device-id", action="store_true")
    parser.add_argument("--pipeline-p2p-smoke", action="store_true")
    parser.add_argument("--use-tp-pp-dp-mapping", action="store_true")
    parser.add_argument("--group-all-to-all", action="store_true")
    parser.add_argument("--object-smoke", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    log(env_summary())
    log(f"torch={torch.__version__} cuda={torch.cuda.get_device_name(torch.cuda.current_device())}")
    init_kwargs = {
        "backend": "nccl",
        "timeout": timedelta(seconds=args.timeout),
    }
    if not args.no_init_device_id:
        init_kwargs["device_id"] = device
    dist.init_process_group(**init_kwargs)
    try:
        log(f"process group initialized world={dist.get_world_size()}")
        if not args.skip_world_all_reduce:
            all_reduce_sizes(parse_sizes(args.sizes), args.iters)
        if not args.skip_world_p2p:
            p2p_ring(args.p2p_numel)
        if not args.skip_megatron_groups:
            megatron_group_smoke(args)
        dist.barrier()
        log("NCCL fleet smoke passed")
    finally:
        dist.destroy_process_group()
        log("process group destroyed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
