# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Production A/B verifier for native StreamBP.

Run with torchrun. The verifier builds matching Megatron-Core GPT models,
wraps them in MCore DDP with overlapped gradient reduction, then compares:

* baseline vs StreamBP loss and DDP main_grad tensors,
* peak CUDA memory,
* end-to-end tokens/second over identical measured steps.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

import megatron.core.parallel_state as parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.transformer_config import TransformerConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--ffn-hidden-size", type=int, default=2048)
    parser.add_argument("--num-moe-experts", type=int, default=None)
    parser.add_argument("--moe-layer-freq", type=int, default=1)
    parser.add_argument("--moe-ffn-hidden-size", type=int, default=None)
    parser.add_argument("--moe-router-topk", type=int, default=2)
    parser.add_argument("--moe-aux-loss-coeff", type=float, default=0.0)
    parser.add_argument(
        "--moe-token-dispatcher-type",
        choices=("allgather", "alltoall", "flex"),
        default="allgather",
    )
    parser.add_argument("--moe-grouped-gemm", action="store_true")
    parser.add_argument("--streambp-chunk-size", type=int, default=1024)
    parser.add_argument("--streambp-logits-chunk-size", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--layer-spec", choices=("local", "te"), default="local")
    parser.add_argument(
        "--baseline-full-recompute",
        action="store_true",
        help=(
            "Compare StreamBP against Megatron full activation recompute instead of "
            "a no-recompute baseline. This is useful when the production comparison "
            "target is the existing long-sequence memory-saving mode."
        ),
    )
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--bucket-size", type=int, default=25_000_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--loss-atol", type=float, default=2e-2)
    parser.add_argument("--loss-rtol", type=float, default=2e-2)
    parser.add_argument("--grad-atol", type=float, default=4e-2)
    parser.add_argument("--grad-rtol", type=float, default=4e-2)
    parser.add_argument("--max-memory-ratio", type=float, default=0.98)
    parser.add_argument("--min-throughput-ratio", type=float, default=1.0)
    return parser.parse_args()


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank0_print(message: str) -> None:
    if _rank() == 0:
        print(message, flush=True)


def _init_distributed(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("tools/streambp_prod_verify.py requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=20))
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        create_gloo_process_groups=False,
    )


def _destroy_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
    parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def _dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if args.dtype == "bf16" else torch.float32


def _make_config(
    args: argparse.Namespace, *, use_streambp: bool, baseline_full_recompute: bool = False
) -> TransformerConfig:
    dtype = _dtype(args)
    recompute_kwargs = {}
    if baseline_full_recompute:
        recompute_kwargs = {
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": args.num_layers,
        }
    moe_kwargs = {}
    if args.num_moe_experts is not None:
        moe_kwargs = {
            "num_moe_experts": args.num_moe_experts,
            "moe_layer_freq": args.moe_layer_freq,
            "moe_ffn_hidden_size": args.moe_ffn_hidden_size or args.ffn_hidden_size,
            "moe_router_topk": args.moe_router_topk,
            "moe_aux_loss_coeff": args.moe_aux_loss_coeff,
            "moe_grouped_gemm": args.moe_grouped_gemm,
            "moe_token_dispatcher_type": args.moe_token_dispatcher_type,
        }
    return TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=dtype is torch.bfloat16,
        params_dtype=dtype,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        use_streambp=use_streambp,
        streambp_chunk_size=args.streambp_chunk_size,
        streambp_logits_chunk_size=args.streambp_logits_chunk_size,
        streambp_skip_moe=False,
        **moe_kwargs,
        **recompute_kwargs,
    )


def _make_gpt(
    args: argparse.Namespace, *, use_streambp: bool, baseline_full_recompute: bool = False
) -> GPTModel:
    config = _make_config(
        args, use_streambp=use_streambp, baseline_full_recompute=baseline_full_recompute
    )
    use_te = args.layer_spec == "te"
    if args.num_moe_experts is not None:
        layer_spec = get_gpt_decoder_block_spec(config, use_te)
    else:
        layer_spec = (
            get_gpt_layer_with_transformer_engine_spec()
            if use_te
            else get_gpt_layer_local_spec()
        )
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=args.vocab_size,
        max_sequence_length=args.seq_len,
        pre_process=True,
        post_process=True,
    ).cuda()
    return model.to(dtype=_dtype(args)).train()


def _wrap_ddp(args: argparse.Namespace, model: GPTModel) -> DistributedDataParallel:
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        use_distributed_optimizer=False,
        bucket_size=args.bucket_size,
    )
    ddp = DistributedDataParallel(model.config, ddp_config, model)
    ddp.broadcast_params()
    return ddp


def _make_batch(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 17 * _rank())
    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.micro_batch_size, args.seq_len),
        device="cuda",
        generator=generator,
    )
    labels = torch.randint(
        0,
        args.vocab_size,
        (args.micro_batch_size, args.seq_len),
        device="cuda",
        generator=generator,
    )
    position_ids = torch.arange(args.seq_len, device="cuda", dtype=torch.long).unsqueeze(0)
    position_ids = position_ids.expand(args.micro_batch_size, -1).contiguous()
    attention_mask = torch.triu(
        torch.ones((1, 1, args.seq_len, args.seq_len), dtype=torch.bool, device="cuda"),
        diagonal=1,
    )
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _clear_grad(ddp: DistributedDataParallel) -> None:
    ddp.zero_grad_buffer()
    for param in ddp.parameters():
        param.grad = None


def _step(ddp: DistributedDataParallel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    _clear_grad(ddp)
    get_moe_metrics_tracker().clear()
    loss = ddp(**batch)
    scalar_loss = loss.float().mean()
    scalar_loss.backward()
    ddp.finish_grad_sync()
    get_moe_metrics_tracker().clear()
    return scalar_loss.detach()


def _main_grads(ddp: DistributedDataParallel) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, param in ddp.module.named_parameters():
        main_grad = getattr(param, "main_grad", None)
        if main_grad is not None:
            grads[name] = main_grad.detach().float().cpu()
    return grads


def _max_float(value: float) -> float:
    tensor = torch.tensor(value, device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _max_int(value: int) -> int:
    tensor = torch.tensor(value, device="cuda", dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _check_correctness(
    args: argparse.Namespace,
    baseline: DistributedDataParallel,
    streambp: DistributedDataParallel,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    baseline_loss = _step(baseline, batch)
    streambp_loss = _step(streambp, batch)
    torch.cuda.synchronize()

    loss_diff = float((streambp_loss.float() - baseline_loss.float()).abs().item())
    if not torch.allclose(
        streambp_loss.float(),
        baseline_loss.float(),
        atol=args.loss_atol,
        rtol=args.loss_rtol,
    ):
        raise AssertionError(
            f"StreamBP loss mismatch: baseline={baseline_loss.item()} "
            f"streambp={streambp_loss.item()} diff={loss_diff}"
        )

    baseline_grads = _main_grads(baseline)
    streambp_grads = _main_grads(streambp)
    if baseline_grads.keys() != streambp_grads.keys():
        missing = sorted(set(baseline_grads) ^ set(streambp_grads))
        raise AssertionError(f"StreamBP main_grad key mismatch: {missing[:8]}")

    max_abs = 0.0
    max_rel = 0.0
    worst_name = ""
    for name, baseline_grad in baseline_grads.items():
        streambp_grad = streambp_grads[name]
        abs_diff_tensor = (streambp_grad - baseline_grad).abs()
        denom = baseline_grad.abs().clamp_min(1e-8)
        rel_diff_tensor = abs_diff_tensor / denom
        abs_diff = float(abs_diff_tensor.max().item())
        rel_diff = float(rel_diff_tensor.max().item())
        if abs_diff > max_abs:
            max_abs = abs_diff
            max_rel = rel_diff
            worst_name = name
        if not torch.allclose(
            streambp_grad, baseline_grad, atol=args.grad_atol, rtol=args.grad_rtol
        ):
            raise AssertionError(
                f"StreamBP grad mismatch for {name}: max_abs={abs_diff} max_rel={rel_diff}"
            )

    return {
        "baseline_loss": float(baseline_loss.item()),
        "streambp_loss": float(streambp_loss.item()),
        "loss_abs_diff": loss_diff,
        "max_grad_abs_diff": max_abs,
        "max_grad_rel_diff": max_rel,
        "worst_grad": worst_name,
    }


def _benchmark(
    args: argparse.Namespace,
    label: str,
    ddp: DistributedDataParallel,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    for _ in range(args.warmup_steps):
        _step(ddp, batch)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.measure_steps):
        _step(ddp, batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_bytes = torch.cuda.max_memory_allocated()

    elapsed = _max_float(elapsed)
    peak_bytes = _max_int(peak_bytes)
    tokens = (
        args.measure_steps
        * args.micro_batch_size
        * args.seq_len
        * _world_size()
        // (args.tensor_model_parallel_size * args.pipeline_model_parallel_size)
    )
    tokens_per_second = tokens / elapsed
    return {
        "label": label,
        "elapsed_seconds": elapsed,
        "peak_bytes": peak_bytes,
        "tokens_per_second": tokens_per_second,
    }


def main() -> None:
    args = _parse_args()
    _init_distributed(args)
    try:
        torch.manual_seed(args.seed)
        model_parallel_cuda_manual_seed(args.seed)
        baseline_model = _make_gpt(
            args, use_streambp=False, baseline_full_recompute=args.baseline_full_recompute
        )
        streambp_model = _make_gpt(args, use_streambp=True)
        streambp_model.load_state_dict(baseline_model.state_dict())
        baseline = _wrap_ddp(args, baseline_model)
        streambp = _wrap_ddp(args, streambp_model)
        batch = _make_batch(args)

        correctness = _check_correctness(args, baseline, streambp, batch)
        baseline_perf = _benchmark(args, "baseline", baseline, batch)
        streambp_perf = _benchmark(args, "streambp", streambp, batch)

        memory_ratio = streambp_perf["peak_bytes"] / baseline_perf["peak_bytes"]
        throughput_ratio = (
            streambp_perf["tokens_per_second"] / baseline_perf["tokens_per_second"]
        )
        result = {
            "world_size": _world_size(),
            "args": vars(args),
            "correctness": correctness,
            "baseline": baseline_perf,
            "streambp": streambp_perf,
            "memory_ratio": memory_ratio,
            "throughput_ratio": throughput_ratio,
        }

        if memory_ratio >= args.max_memory_ratio:
            raise AssertionError(
                "StreamBP memory gate failed: "
                f"ratio={memory_ratio:.4f} max_allowed={args.max_memory_ratio:.4f} "
                f"result={json.dumps(result, sort_keys=True)}"
            )
        if throughput_ratio < args.min_throughput_ratio:
            raise AssertionError(
                "StreamBP throughput gate failed: "
                f"ratio={throughput_ratio:.4f} min_required={args.min_throughput_ratio:.4f} "
                f"result={json.dumps(result, sort_keys=True)}"
            )

        _rank0_print("STREAMBP_PROD_VERIFY_OK " + json.dumps(result, sort_keys=True))
    finally:
        _destroy_distributed()


if __name__ == "__main__":
    main()
