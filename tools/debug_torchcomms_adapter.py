#!/usr/bin/env python3
"""Smoke Megatron's TorchComms adapter with CUDA collectives and group setup."""

from __future__ import annotations

import argparse
import os
import socket
import time
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import torch

from megatron.core import torchcomms_adapter


def log(message: str) -> None:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    now = time.strftime("%H:%M:%S")
    print(f"[{now} rank={rank} local={local_rank} host={socket.gethostname()}] {message}", flush=True)


def smoke_parallel_state(args: argparse.Namespace, *, destroy: bool = True) -> None:
    from megatron.core import parallel_state

    log(
        "initializing model-parallel groups "
        f"tp={args.tp} pp={args.pp} cp={args.cp} ep={args.ep} etp={args.etp} "
        f"gloo={not args.disable_gloo_process_groups}"
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=args.cp,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=args.etp,
        distributed_timeout_minutes=max(1, args.timeout // 60),
        create_gloo_process_groups=not args.disable_gloo_process_groups,
    )
    log(
        "model-parallel groups initialized "
        f"tp_rank={parallel_state.get_tensor_model_parallel_rank()} "
        f"pp_rank={parallel_state.get_pipeline_model_parallel_rank()} "
        f"dp_rank={parallel_state.get_data_parallel_rank()} "
        f"ep_rank={parallel_state.get_expert_model_parallel_rank()}"
    )
    torch.distributed.barrier()
    if destroy:
        parallel_state.destroy_model_parallel()
        log("model-parallel groups destroyed")


def smoke_checkpoint_object_collective() -> None:
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_metadata = [
        ShardedTensor(
            key=f"debug.tensor.{rank}",
            data=None,
            dtype=torch.float32,
            local_shape=(1,),
            global_shape=(world_size,),
            global_offset=(rank,),
            axis_fragmentations=(world_size,),
        )
    ]
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, local_metadata)
    assert all(isinstance(entry, list) for entry in gathered), gathered
    assert all(isinstance(entry[0], ShardedTensor) for entry in gathered), gathered
    log(
        "checkpoint object collective gathered "
        f"{len(gathered)} ShardedTensor metadata entries"
    )


def smoke_p2p() -> None:
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert world_size % 2 == 0, "P2P smoke expects an even world size"

    tensor = torch.full((1,), rank, dtype=torch.float32, device="cuda")
    if rank % 2 == 0:
        peer = rank + 1
        op = torch.distributed.P2POp(torch.distributed.isend, tensor, peer=peer)
    else:
        peer = rank - 1
        tensor.zero_()
        op = torch.distributed.P2POp(torch.distributed.irecv, tensor, peer=peer)
    works = torch.distributed.batch_isend_irecv([op])
    for work in works:
        work.wait()
    torch.cuda.synchronize()
    if rank % 2 == 1:
        assert tensor.item() == peer, (rank, peer, tensor.item())
    log("P2POp/batch_isend_irecv smoke passed")


def smoke_pp_group_p2p(use_direct_torchcomm: bool, numel: int) -> None:
    from megatron.core import parallel_state
    from torchcomms.distwrap.pginfo import pg_info_get_global_ranks

    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    pp_ranks = pg_info_get_global_ranks(pp_group)
    device = torch.device("cuda", torch.cuda.current_device())

    next_group_rank = (pp_rank + 1) % pp_size
    prev_group_rank = (pp_rank - 1) % pp_size
    next_global_rank = pp_ranks[next_group_rank]
    prev_global_rank = pp_ranks[prev_group_rank]

    log(
        "pp p2p setup "
        f"pp_rank={pp_rank}/{pp_size} pp_ranks={pp_ranks} "
        f"next_global={next_global_rank} prev_global={prev_global_rank} "
        f"direct_torchcomm={use_direct_torchcomm} numel={numel}"
    )

    send_tensor = torch.full((numel,), float(torch.distributed.get_rank()), device=device)
    recv_tensor = torch.full((numel,), -1.0, device=device)

    if use_direct_torchcomm:
        tc = torchcomms_adapter.get_torchcomm(pp_group, tensor=send_tensor)
        batch_op = tc.batch_op_create()
        batch_op.send(send_tensor, next_group_rank)
        batch_op.recv(recv_tensor, prev_group_rank)
        works = [batch_op.issue(True)]
    else:
        ops = [
            torch.distributed.P2POp(
                torch.distributed.isend,
                send_tensor,
                peer=next_global_rank,
                group=pp_group,
            ),
            torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_tensor,
                peer=prev_global_rank,
                group=pp_group,
            ),
        ]
        works = torch.distributed.batch_isend_irecv(ops)

    for work in works:
        work.wait()
    torch.cuda.synchronize()
    expected = torch.full_like(recv_tensor, float(prev_global_rank))
    assert torch.equal(recv_tensor, expected), (
        torch.distributed.get_rank(),
        recv_tensor,
        expected,
    )
    log("pp-group p2p smoke passed")


def smoke_all_to_all_v() -> None:
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    input_tensor = torch.tensor(
        [rank * 1000 + peer for peer in range(world_size)],
        dtype=torch.float32,
        device="cuda",
    )
    output = torch.empty_like(input_tensor)
    split_sizes = np.ones(world_size, dtype=np.int64)
    torch.distributed.all_to_all_single(
        output,
        input_tensor,
        output_split_sizes=split_sizes,
        input_split_sizes=split_sizes,
    )
    torch.cuda.synchronize()
    expected = torch.tensor(
        [peer * 1000 + rank for peer in range(world_size)],
        dtype=torch.float32,
        device="cuda",
    )
    assert torch.equal(output, expected), (rank, output, expected)
    log("all_to_all_single numpy split-size smoke passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--parallel-state-smoke", action="store_true")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--disable-gloo-process-groups", action="store_true")
    parser.add_argument("--checkpoint-object-smoke", action="store_true")
    parser.add_argument("--p2p-smoke", action="store_true")
    parser.add_argument("--pp-p2p-smoke", action="store_true")
    parser.add_argument("--pp-direct-torchcomm-p2p-smoke", action="store_true")
    parser.add_argument("--pp-p2p-numel", type=int, default=4)
    parser.add_argument("--all-to-all-v-smoke", action="store_true")
    args_cli = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    args = SimpleNamespace(
        distributed_backend="ncclx",
        use_torchcomms=True,
        ncclx_rdma=True,
        ncclx_rdma_backends=os.getenv("MEGATRON_NCCLX_RDMA_BACKENDS", "ib,nvl,socket"),
        ncclx_rdma_profile=os.getenv("MEGATRON_NCCLX_RDMA_PROFILE", "roce"),
        ncclx_roce_gid_index=None,
        ncclx_roce_addr_family=None,
        ncclx_ib_hca=None,
    )
    torchcomms_adapter.init_process_group(
        args,
        backend="nccl",
        timeout=timedelta(seconds=args_cli.timeout),
    )
    try:
        log(f"fd_limit={os.popen('ulimit -Sn').read().strip()}")
        value = torch.tensor([float(rank + 1)], device="cuda")
        torch.distributed.all_reduce(value)
        torch.cuda.synchronize()
        expected = torch.distributed.get_world_size() * (torch.distributed.get_world_size() + 1) / 2
        log(
            f"rank={rank} value={value.item()} expected={expected} "
            f"active={torchcomms_adapter.is_torchcomms_active()} "
            f"backend={torchcomms_adapter.ncclx_backend_string()}"
        )
        keep_parallel_state = args_cli.pp_p2p_smoke or args_cli.pp_direct_torchcomm_p2p_smoke
        if args_cli.parallel_state_smoke:
            smoke_parallel_state(args_cli, destroy=not keep_parallel_state)
        if args_cli.checkpoint_object_smoke:
            smoke_checkpoint_object_collective()
        if args_cli.p2p_smoke:
            smoke_p2p()
        if keep_parallel_state:
            try:
                if not args_cli.parallel_state_smoke:
                    smoke_parallel_state(args_cli, destroy=False)
                if args_cli.pp_direct_torchcomm_p2p_smoke:
                    smoke_pp_group_p2p(
                        use_direct_torchcomm=True,
                        numel=args_cli.pp_p2p_numel,
                    )
                if args_cli.pp_p2p_smoke:
                    smoke_pp_group_p2p(
                        use_direct_torchcomm=False,
                        numel=args_cli.pp_p2p_numel,
                    )
            finally:
                from megatron.core import parallel_state

                parallel_state.destroy_model_parallel()
                log("model-parallel groups destroyed")
        if args_cli.all_to_all_v_smoke:
            smoke_all_to_all_v()
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
