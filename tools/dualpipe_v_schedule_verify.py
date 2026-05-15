# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DualPipeV schedule verifier.

The static mode validates the DualPipeV rank mapping without importing torch.
The distributed mode is intended for torchrun and checks the conservative
sequential V-shaped forward/backward data path with the selected process-group
backend. Memory and throughput fields are emitted only when ``--measure`` is set.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any


@dataclass(frozen=True)
class RankMapping:
    rank: int
    phase0_stage: int
    phase1_stage: int
    prev_phase0_rank: int | None
    next_phase0_rank: int | None
    prev_phase1_rank: int | None
    next_phase1_rank: int | None
    is_entry_rank: bool
    is_bridge_rank: bool
    is_loss_rank: bool


@dataclass(frozen=True)
class StepCounts:
    rank: int
    step_1_f0_warmup: int
    step_2_f0_f1_warmup: int
    step_3_b1_w1_f1: int
    step_4_main_f0_b1_f1_b0: int
    step_5_b1_f1_b0: int
    step_6_b1_b0: int
    step_7_w_b0: int
    step_8_w: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "distributed-smoke"), default="static")
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--pipeline-model-parallel-size", type=int, required=True)
    parser.add_argument("--virtual-pipeline-model-parallel-size", type=int, default=2)
    parser.add_argument("--num-microbatches", "--microbatches", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> int:
    pp_size = args.pipeline_model_parallel_size
    if pp_size < 2:
        raise ValueError("DualPipeV requires --pipeline-model-parallel-size >= 2")
    if args.virtual_pipeline_model_parallel_size != 2:
        raise ValueError("DualPipeV requires --virtual-pipeline-model-parallel-size 2")
    min_microbatches = 2 * pp_size
    if args.num_microbatches is None:
        args.num_microbatches = min_microbatches
    if args.num_microbatches < min_microbatches:
        raise ValueError(
            "DualPipeV requires --num-microbatches >= "
            f"{min_microbatches} for PP size {pp_size}"
        )
    for name in ("micro_batch_size", "seq_length", "hidden_size", "iterations"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    return min_microbatches


def _rank_mapping(rank: int, pp_size: int) -> RankMapping:
    return RankMapping(
        rank=rank,
        phase0_stage=rank,
        phase1_stage=(2 * pp_size) - 1 - rank,
        prev_phase0_rank=rank - 1 if rank > 0 else None,
        next_phase0_rank=rank + 1 if rank < pp_size - 1 else None,
        prev_phase1_rank=rank + 1 if rank < pp_size - 1 else None,
        next_phase1_rank=rank - 1 if rank > 0 else None,
        is_entry_rank=rank == 0,
        is_bridge_rank=rank == pp_size - 1,
        is_loss_rank=rank == 0,
    )


def _step_counts(rank: int, pp_size: int, num_microbatches: int) -> StepCounts:
    return StepCounts(
        rank=rank,
        step_1_f0_warmup=(pp_size - rank - 1) * 2,
        step_2_f0_f1_warmup=rank + 1,
        step_3_b1_w1_f1=pp_size - rank - 1,
        step_4_main_f0_b1_f1_b0=num_microbatches - (2 * pp_size) + rank + 1,
        step_5_b1_f1_b0=pp_size - rank - 1,
        step_6_b1_b0=rank + 1,
        step_7_w_b0=pp_size - rank - 1,
        step_8_w=rank + 1,
    )


def _static_summary(args: argparse.Namespace) -> dict[str, Any]:
    pp_size = args.pipeline_model_parallel_size
    min_microbatches = _validate_args(args)
    rank_mappings = [_rank_mapping(rank, pp_size) for rank in range(pp_size)]
    step_counts = [_step_counts(rank, pp_size, args.num_microbatches) for rank in range(pp_size)]
    return {
        "schedule": "dualpipe_v",
        "support_level": "verification_harness",
        "execution_baseline": "sequential_v_topology",
        "runtime_measurements": False,
        "pipeline_model_parallel_size": pp_size,
        "virtual_pipeline_model_parallel_size": args.virtual_pipeline_model_parallel_size,
        "logical_pipeline_stages": 2 * pp_size,
        "num_microbatches": args.num_microbatches,
        "minimum_microbatches": min_microbatches,
        "mapping": [asdict(mapping) for mapping in rank_mappings],
        "step_counts": [asdict(counts) for counts in step_counts],
        "invariants": {
            "phase0_direction": "rank0_to_rankN",
            "phase1_direction": "rankN_to_rank0",
            "entry_rank": 0,
            "bridge_rank": pp_size - 1,
            "loss_rank": 0,
            "local_modules_per_rank": 2,
        },
    }


def _dist_env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _closed_form_value(input_tensor: "Any", pp_size: int) -> "Any":
    output = input_tensor
    for stage in range(2 * pp_size):
        output = (output * float(stage + 2)) + float(stage)
    return output


def _closed_form_grads(input_tensor: "Any", pp_size: int) -> tuple["Any", list["Any"], list["Any"]]:
    values = [input_tensor]
    current = input_tensor
    for stage in range(2 * pp_size):
        current = (current * float(stage + 2)) + float(stage)
        values.append(current)

    grad = None
    grad_weights: list[Any] = [None] * (2 * pp_size)
    grad_biases: list[Any] = [None] * (2 * pp_size)
    for stage in reversed(range(2 * pp_size)):
        if grad is None:
            grad = values[-1].new_ones(values[-1].shape)
        grad_weights[stage] = (grad * values[stage]).sum()
        grad_biases[stage] = grad.sum()
        grad = grad * float(stage + 2)
    return grad, grad_weights, grad_biases


def _send_recv(
    dist: "Any",
    tensor: "Any",
    *,
    src: int | None,
    dst: int | None,
    group: "Any",
) -> "Any":
    ops = []
    recv_tensor = tensor.new_empty(tensor.shape) if src is not None else None
    if src is not None:
        ops.append(dist.P2POp(dist.irecv, recv_tensor, src, group=group))
    if dst is not None:
        ops.append(dist.P2POp(dist.isend, tensor, dst, group=group))
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()
    return recv_tensor


def _run_distributed_microbatch(
    args: argparse.Namespace, *, measure: bool, microbatch_id: int
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    pp_size = args.pipeline_model_parallel_size
    mapping = _rank_mapping(rank, pp_size)
    use_cuda_tensors = args.backend == "nccl"
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if use_cuda_tensors
        else torch.device("cpu")
    )

    shape = (args.micro_batch_size, args.seq_length, args.hidden_size)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + microbatch_id)
    input_tensor = torch.randn(shape, generator=generator, device=device)

    phase0_input = input_tensor if mapping.is_entry_rank else input_tensor.new_empty(shape)
    if not mapping.is_entry_rank:
        phase0_input = _send_recv(
            dist,
            phase0_input,
            src=mapping.prev_phase0_rank,
            dst=None,
            group=dist.group.WORLD,
        )
    phase0_output = (phase0_input * float(mapping.phase0_stage + 2)) + float(mapping.phase0_stage)
    sent_or_received = _send_recv(
        dist,
        phase0_output,
        src=None,
        dst=mapping.next_phase0_rank,
        group=dist.group.WORLD,
    )
    if sent_or_received is not None:
        raise AssertionError("phase0 send-only path unexpectedly returned a tensor")

    phase1_input = phase0_output if mapping.is_bridge_rank else input_tensor.new_empty(shape)
    if not mapping.is_bridge_rank:
        phase1_input = _send_recv(
            dist,
            phase1_input,
            src=mapping.prev_phase1_rank,
            dst=None,
            group=dist.group.WORLD,
        )
    phase1_output = (phase1_input * float(mapping.phase1_stage + 2)) + float(mapping.phase1_stage)
    _send_recv(
        dist,
        phase1_output,
        src=None,
        dst=mapping.next_phase1_rank,
        group=dist.group.WORLD,
    )

    if mapping.is_loss_rank:
        expected = _closed_form_value(input_tensor, pp_size)
        torch.testing.assert_close(phase1_output, expected, rtol=1e-5, atol=1e-5)

    if mapping.is_loss_rank:
        phase1_grad = phase1_output.new_ones(phase1_output.shape)
    else:
        phase1_grad = input_tensor.new_empty(shape)
        phase1_grad = _send_recv(
            dist,
            phase1_grad,
            src=mapping.next_phase1_rank,
            dst=None,
            group=dist.group.WORLD,
        )
    phase1_grad_weight = (phase1_grad * phase1_input).sum()
    phase1_grad_bias = phase1_grad.sum()
    phase0_grad_from_phase1 = phase1_grad * float(mapping.phase1_stage + 2)
    if not mapping.is_bridge_rank:
        _send_recv(
            dist,
            phase0_grad_from_phase1,
            src=None,
            dst=mapping.prev_phase1_rank,
            group=dist.group.WORLD,
        )

    if mapping.is_bridge_rank:
        phase0_grad = phase0_grad_from_phase1
    else:
        phase0_grad = input_tensor.new_empty(shape)
        phase0_grad = _send_recv(
            dist,
            phase0_grad,
            src=mapping.next_phase0_rank,
            dst=None,
            group=dist.group.WORLD,
        )
    phase0_grad_weight = (phase0_grad * phase0_input).sum()
    phase0_grad_bias = phase0_grad.sum()
    phase0_input_grad = phase0_grad * float(mapping.phase0_stage + 2)
    if not mapping.is_entry_rank:
        _send_recv(
            dist,
            phase0_input_grad,
            src=None,
            dst=mapping.prev_phase0_rank,
            group=dist.group.WORLD,
        )

    _, expected_grad_weights, expected_grad_biases = _closed_form_grads(input_tensor, pp_size)
    torch.testing.assert_close(
        phase0_grad_weight, expected_grad_weights[mapping.phase0_stage], rtol=1e-5, atol=1e-4
    )
    torch.testing.assert_close(
        phase0_grad_bias, expected_grad_biases[mapping.phase0_stage], rtol=1e-5, atol=1e-4
    )
    torch.testing.assert_close(
        phase1_grad_weight, expected_grad_weights[mapping.phase1_stage], rtol=1e-5, atol=1e-4
    )
    torch.testing.assert_close(
        phase1_grad_bias, expected_grad_biases[mapping.phase1_stage], rtol=1e-5, atol=1e-4
    )

    result = {"correctness": "passed"}
    if measure and args.backend == "nccl" and torch.cuda.is_available():
        result["rank_peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    return result


def _run_distributed_iteration(args: argparse.Namespace, *, measure: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for microbatch_id in range(args.num_microbatches):
        result = _run_distributed_microbatch(
            args, measure=measure, microbatch_id=microbatch_id
        )
    return result


def _distributed_summary(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    _validate_args(args)
    if _dist_env_world_size() != args.pipeline_model_parallel_size:
        raise ValueError(
            "torchrun WORLD_SIZE must equal --pipeline-model-parallel-size for this verifier"
        )
    if args.backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("--backend nccl requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(args.backend, timeout=timedelta(minutes=20))

    rank = dist.get_rank()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup_steps):
        _run_distributed_iteration(args, measure=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result: dict[str, Any] = {}
    for _ in range(args.iterations):
        result = _run_distributed_iteration(args, measure=args.measure)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64)
    if args.backend == "nccl" and torch.cuda.is_available():
        elapsed_tensor = elapsed_tensor.cuda()
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)

    summary = _static_summary(args)
    summary["mode"] = "distributed-smoke"
    summary["backend"] = args.backend
    summary["world_size"] = dist.get_world_size()
    summary["correctness"] = result["correctness"]
    if args.measure:
        avg_step_seconds = float(elapsed_tensor.item()) / args.iterations
        tokens_per_step = args.num_microbatches * args.micro_batch_size * args.seq_length
        peak_memory = torch.tensor([result.get("rank_peak_memory_bytes", 0)], dtype=torch.int64)
        if args.backend == "nccl" and torch.cuda.is_available():
            peak_memory = peak_memory.cuda()
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
        summary["runtime_measurements"] = True
        summary["measurements"] = {
            "iterations": args.iterations,
            "warmup_steps": args.warmup_steps,
            "avg_step_seconds": avg_step_seconds,
            "tokens_per_second": tokens_per_step / avg_step_seconds,
            "max_rank_peak_memory_bytes": int(peak_memory.item()),
        }

    dist.barrier()
    should_print = rank == 0
    dist.destroy_process_group()
    if not should_print:
        return {}
    return summary


def main() -> None:
    args = _parse_args()
    if args.mode == "static":
        summary = _static_summary(args)
    else:
        summary = _distributed_summary(args)
    if summary:
        payload = json.dumps(summary, indent=2 if args.pretty else None, sort_keys=True)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as output_file:
                output_file.write(payload)
                output_file.write("\n")
        else:
            print(payload)


if __name__ == "__main__":
    main()
