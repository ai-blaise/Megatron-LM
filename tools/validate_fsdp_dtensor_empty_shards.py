#!/usr/bin/env python3
"""Validate Megatron-FSDP uneven DTensor save handling with empty local shards.

Run with:
  torchrun --standalone --nproc_per_node=6 tools/validate_fsdp_dtensor_empty_shards.py

This reproduces the checkpoint-shape class that appears when a flattened
SwiGLU linear_fc1 FSDP shard is split into W/V halves. Some ranks can hold a
legitimate empty local shard for one half while other ranks cover the global
boundaries.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, default_planner, load_state_dict, save
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    _get_flattened_mesh_group,
    preprocess_state_dict_for_uneven_dtensor,
    update_uneven_dtensor_chunk_metadata,
    validate_uneven_dtensor,
)


def _build_mesh(dp_size: int, tp_size: int, from_groups: bool) -> DeviceMesh:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected_world_size = dp_size * tp_size
    if world_size != expected_world_size:
        raise RuntimeError(f"Expected {expected_world_size} ranks for this repro, got {world_size}")

    mesh_ranks = torch.arange(world_size).reshape(dp_size, tp_size)
    if from_groups:
        dp_rank = rank // tp_size
        tp_rank = rank % tp_size
        dp_groups = [
            dist.new_group(ranks=[dp * tp_size + tp for dp in range(dp_size)])
            for tp in range(tp_size)
        ]
        tp_groups = [
            dist.new_group(ranks=[dp * tp_size + tp for tp in range(tp_size)])
            for dp in range(dp_size)
        ]
        mesh = DeviceMesh.from_group(
            [dp_groups[tp_rank], tp_groups[dp_rank]],
            device_type="cpu",
            mesh=mesh_ranks.tolist(),
            mesh_dim_names=("dp_cp", "tp"),
        )
    else:
        mesh = DeviceMesh(
            "cpu",
            mesh_ranks,
            mesh_dim_names=("dp_cp", "tp"),
        )
    # Match Megatron-FSDP's TP-then-FSDP shard interpretation for tensors whose
    # FSDP and TP placements both shard dimension 0.
    setattr(mesh, "_shard_order", [1, 0])
    if rank == 0:
        print(f"mesh={mesh}", flush=True)
    return mesh


def _local_rows_for_rank(scenario: str) -> int:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_size = int(os.environ["REPRO_TP_SIZE"])
    dp_rank = rank // tp_size
    tp_rank = rank % tp_size
    dp_size = world_size // tp_size
    if scenario == "single-empty":
        rows_by_coord = {
            (0, 0): 1,
            (1, 0): 1,
            (2, 0): 1,
            (0, 1): 0,
            (1, 1): 2,
            (2, 1): 1,
        }
        return rows_by_coord[(dp_rank, tp_rank)]
    if scenario == "swiglu-flat-half":
        # Model the real save-time SWiGLU V-half split. For each TP rank, the
        # first half of DP/CP shards belong to W and are empty for V. The next
        # DP shard may straddle the W/V split when dp_size is odd.
        rows_per_tp = 6
        full_rows_per_tp = rows_per_tp * 2
        start = dp_rank * full_rows_per_tp // dp_size
        end = (dp_rank + 1) * full_rows_per_tp // dp_size
        v_start = rows_per_tp
        v_end = full_rows_per_tp
        overlap = max(0, min(end, v_end) - max(start, v_start))
        return overlap
    raise ValueError(f"Unknown scenario: {scenario}")


def _make_repro_dtensor(mesh: DeviceMesh, fill_value: float, scenario: str) -> DTensor:
    local_rows = _local_rows_for_rank(scenario)
    local = torch.full((local_rows, 4), fill_value, dtype=torch.float32)
    tp_size = int(os.environ["REPRO_TP_SIZE"])
    global_rows = 6 if scenario == "single-empty" else 6 * tp_size
    return DTensor.from_local(
        local_tensor=local,
        device_mesh=mesh,
        placements=[Shard(0), Shard(0)],
        run_check=False,
        shape=(global_rows, 4),
        stride=torch.empty(global_rows, 4).stride(),
    )


def _make_state_dict(mesh: DeviceMesh, fill_value: float, scenario: str) -> dict:
    dtensor = _make_repro_dtensor(mesh, fill_value, scenario)
    return {
        "model": {
            "decoder.layers.0.mlp.linear_fc1.weight_w": dtensor,
            "_rank0_checkpoint_anchor": torch.tensor([fill_value], dtype=torch.float32),
        },
        "optimizer": {
            "state": {
                "module.decoder.layers.0.mlp.linear_fc1.weight_w": {
                    "exp_avg": _make_repro_dtensor(mesh, fill_value, scenario),
                    "exp_avg_sq": _make_repro_dtensor(mesh, fill_value, scenario),
                },
            },
            "param_groups": [],
        },
    }


def _verify_loaded_state_dict(state_dict: dict, expected_value: float) -> None:
    tensors = [
        state_dict["model"]["decoder.layers.0.mlp.linear_fc1.weight_w"],
        state_dict["optimizer"]["state"]["module.decoder.layers.0.mlp.linear_fc1.weight_w"][
            "exp_avg"
        ],
        state_dict["optimizer"]["state"]["module.decoder.layers.0.mlp.linear_fc1.weight_w"][
            "exp_avg_sq"
        ],
    ]
    for tensor in tensors:
        local = tensor.to_local()
        if local.numel() == 0:
            continue
        expected = torch.full_like(local, expected_value)
        if not torch.equal(local, expected):
            raise AssertionError(
                f"Loaded local tensor mismatch on rank {dist.get_rank()}: "
                f"expected={expected_value}, actual={local}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, default=3)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument(
        "--scenario", choices=("single-empty", "swiglu-flat-half"), default="single-empty"
    )
    parser.add_argument(
        "--from-groups",
        action="store_true",
        help="Build DeviceMesh via DeviceMesh.from_group, matching Megatron-FSDP.",
    )
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    os.environ["REPRO_TP_SIZE"] = str(args.tp_size)

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    mesh = _build_mesh(args.dp_size, args.tp_size, args.from_groups)

    dtensor = _make_repro_dtensor(mesh, float(rank), args.scenario)
    update_uneven_dtensor_chunk_metadata(dtensor)
    validate_uneven_dtensor(dtensor)
    chunk = dtensor.__create_chunk_list__()[0]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(
        gathered,
        {"rank": rank, "offsets": tuple(chunk.offsets), "sizes": tuple(chunk.sizes)},
    )
    if rank == 0:
        print(f"chunks={gathered}", flush=True)
    flat_group = _get_flattened_mesh_group(mesh)
    group_info = [None] * dist.get_world_size()
    dist.all_gather_object(
        group_info,
        {
            "rank": rank,
            "flat_rank": dist.get_rank(flat_group),
            "flat_world": dist.get_world_size(flat_group),
        },
    )
    if rank == 0:
        print(f"flat_group={group_info}", flush=True)

    if args.metadata_only:
        dist.destroy_process_group()
        return

    ckpt_dir = (
        Path(tempfile.gettempdir())
        / f"megatron_fsdp_empty_dtensor_repro_{args.scenario}_{args.dp_size}x{args.tp_size}"
    )
    if rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    state_dict = _make_state_dict(mesh, float(rank), args.scenario)
    preprocess_state_dict_for_uneven_dtensor(state_dict)
    save(state_dict=state_dict, storage_writer=FileSystemWriter(str(ckpt_dir)))
    dist.barrier()

    load_state = _make_state_dict(mesh, -1.0, args.scenario)
    preprocess_state_dict_for_uneven_dtensor(load_state)
    load_state_dict(
        state_dict=load_state,
        storage_reader=FileSystemReader(str(ckpt_dir)),
        planner=default_planner.DefaultLoadPlanner(allow_partial_load=False),
    )
    _verify_loaded_state_dict(load_state, float(rank))
    dist.barrier()

    if rank == 0:
        print(f"saved={ckpt_dir}", flush=True)
        print("round_trip_load=ok", flush=True)
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
