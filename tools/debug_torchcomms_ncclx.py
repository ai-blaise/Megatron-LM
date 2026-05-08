#!/usr/bin/env python3
"""Small staged TorchComms/NCCLX smoke for multinode debugging."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import TCPStore
from torchcomms import distwrap


def log(message: str) -> None:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    now = time.strftime("%H:%M:%S")
    print(f"[{now} rank={rank} local={local_rank} host={socket.gethostname()}] {message}", flush=True)


def env_summary() -> str:
    keys = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_ADDR_FAMILY",
        "NCCL_CTRAN_ENABLE",
        "NCCL_CTRAN_BACKENDS",
        "NCCL_IGNORE_TOPO_LOAD_FAILURE",
        "TC_STORE_PORT",
    ]
    return " ".join(f"{key}={os.environ.get(key, '')}" for key in keys)


def make_store(args: argparse.Namespace) -> TCPStore | None:
    if not args.explicit_store:
        return None
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    port = int(os.environ["TC_STORE_PORT"])
    log(f"creating explicit TCPStore on {args.store_addr}:{port}")
    store = TCPStore(
        host_name=args.store_addr,
        port=port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=timedelta(seconds=args.timeout),
        wait_for_workers=True,
    )
    log("explicit TCPStore created")
    return store


def exercise_all_reduce(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    value = torch.tensor([float(rank + 1)], device=device)
    log("starting distwrap all_reduce")
    work = distwrap.all_reduce(value, async_op=args.async_op)
    if work is not None:
        log("waiting async all_reduce")
        wait_blocking = getattr(work, "wait_blocking", None)
        if wait_blocking is not None:
            wait_blocking()
        else:
            work.wait()
    torch.cuda.synchronize(device)
    expected = int(os.environ["WORLD_SIZE"]) * (int(os.environ["WORLD_SIZE"]) + 1) / 2
    log(f"finished distwrap all_reduce value={value.item()} expected={expected}")


def exercise_split(args: argparse.Namespace) -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    if world_size < 2:
        return
    even = [r for r in range(world_size) if r % 2 == 0]
    odd = [r for r in range(world_size) if r % 2 == 1]
    split_ranks = [even, odd]
    log(f"starting split_group split_ranks={split_ranks}")
    group = distwrap.split_group(split_ranks=split_ranks, backend=args.backend, group_desc="even_odd")
    log("split_group returned")
    value = torch.tensor([float(rank + 1)], device=device)
    distwrap.all_reduce(value, group=group)
    torch.cuda.synchronize(device)
    log(f"finished split all_reduce value={value.item()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cuda:ncclx,cpu:gloo")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--explicit-store", action="store_true")
    parser.add_argument("--store-addr", default=os.environ.get("MASTER_ADDR", "localhost"))
    parser.add_argument("--async-op", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--torch-only", action="store_true")
    args = parser.parse_args()

    log("starting")
    log(env_summary())
    log(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    log(f"cuda device set name={torch.cuda.get_device_name(torch.cuda.current_device())}")

    if args.torch_only:
        dist_backend = args.backend.replace("ncclx", "nccl")
        log(f"calling torch.distributed.init_process_group backend={dist_backend}")
        dist.init_process_group(backend=dist_backend, timeout=timedelta(seconds=args.timeout))
        log(f"torch.distributed.init_process_group returned world={dist.get_world_size()}")
    else:
        store = make_store(args)
        log(f"calling distwrap.init_process_group backend={args.backend} explicit_store={store is not None}")
        distwrap.init_process_group(
            backend=args.backend,
            timeout=timedelta(seconds=args.timeout),
            store=store,
            use_torchcomms=True,
        )
        log(
            "distwrap.init_process_group returned "
            f"dist_initialized={dist.is_initialized()} world={dist.get_world_size()}"
        )
    try:
        if args.torch_only:
            rank = int(os.environ["RANK"])
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            value = torch.tensor([float(rank + 1)], device=device)
            log("starting torch.distributed all_reduce")
            dist.all_reduce(value)
            torch.cuda.synchronize(device)
            log(f"finished torch.distributed all_reduce value={value.item()}")
        else:
            exercise_all_reduce(args)
            if args.split:
                exercise_split(args)
    finally:
        log("destroying process group")
        if args.torch_only:
            dist.destroy_process_group()
        else:
            distwrap.destroy_process_group()
        log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
