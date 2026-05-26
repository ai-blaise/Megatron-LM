#!/usr/bin/env python3
"""Export the Blaise DeepSeek REAP Megatron checkpoint to Hugging Face safetensors."""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_builders import gpt_builder
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.training import get_args, get_model, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from model_provider import model_provider

from tools.convert_blaise_deepseek_v32_reap_to_megatron import (
    BlaiseDeepSeekV32ReapBridge,
    ConfiguredAutoBridge,
    install_bridge_model_type_compat,
    install_state_source,
)


def add_export_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Blaise HF export")
    group.add_argument("--hf-output-path", required=True)
    group.add_argument(
        "--hf-source-model-id",
        default="BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4",
    )
    group.add_argument("--hf-max-shard-size-gb", type=float, default=4.0)
    group.add_argument("--hf-export-load-non-strict", action="store_true")
    group.add_argument("--hf-export-load-source-first", action="store_true")
    group.add_argument("--hf-export-trust-remote-code", action="store_true", default=True)
    return parser


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def configure_bridge_from_megatron_args(model_bridge: BlaiseDeepSeekV32ReapBridge, args) -> None:
    model_bridge.seq_length = args.seq_length
    model_bridge.tensor_model_parallel_size = args.tensor_model_parallel_size
    model_bridge.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    model_bridge.context_parallel_size = args.context_parallel_size
    model_bridge.expert_model_parallel_size = args.expert_model_parallel_size
    model_bridge.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    model_bridge.sequence_parallel = bool(args.sequence_parallel)
    model_bridge.dsa_indexer_loss_coeff = float(getattr(args, "dsa_indexer_loss_coeff", 0.01))
    model_bridge.num_layers_in_first_pipeline_stage = getattr(
        args, "decoder_first_pipeline_num_layers", None
    )
    model_bridge.num_layers_in_last_pipeline_stage = getattr(
        args, "decoder_last_pipeline_num_layers", None
    )


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def save_streaming_safetensors(generator, output_path: Path, max_shard_bytes: int) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_size = 0
    shard_idx = 1
    buffered: dict[str, torch.Tensor] = {}
    buffered_bytes = 0

    def flush() -> None:
        nonlocal shard_idx, buffered, buffered_bytes
        if not buffered:
            return
        filename = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(buffered, output_path / filename)
        for key in buffered:
            weight_map[key] = filename
        shard_idx += 1
        buffered = {}
        buffered_bytes = 0

    for item in generator:
        name = item.param_name
        weight = item.weight.detach().cpu().contiguous()
        size = tensor_nbytes(weight)
        if buffered and buffered_bytes + size > max_shard_bytes:
            flush()
        buffered[name] = weight
        buffered_bytes += size
        total_size += size

    flush()

    # Replace placeholder shard counts in filenames and weight map.
    shard_count = shard_idx - 1
    if shard_count > 0:
        for old_idx in range(1, shard_count + 1):
            old = output_path / f"model-{old_idx:05d}-of-XXXXX.safetensors"
            new_name = f"model-{old_idx:05d}-of-{shard_count:05d}.safetensors"
            old.rename(output_path / new_name)
            for key, filename in list(weight_map.items()):
                if filename == old.name:
                    weight_map[key] = new_name

    with open(output_path / "model.safetensors.index.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {"total_size": total_size},
                "weight_map": dict(sorted(weight_map.items())),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


def main() -> None:
    initialize_megatron(
        extra_args_provider=add_export_args,
        args_defaults={
            "no_load_optim": True,
            "no_load_rng": True,
            "exit_on_missing_checkpoint": True,
        },
    )
    args = get_args()

    install_bridge_model_type_compat()

    output_path = Path(args.hf_output_path)
    max_shard_bytes = int(args.hf_max_shard_size_gb * 1024**3)

    hf_pretrained = PreTrainedCausalLM.from_pretrained(
        args.hf_source_model_id,
        trust_remote_code=args.hf_export_trust_remote_code,
        torch_dtype=torch.bfloat16,
    )
    # The original HF checkpoint stores compressed NVFP4 triples. The custom
    # source exposes virtual `.weight` keys so Bridge builds the right tasks.
    install_state_source(
        hf_pretrained,
        dequant_dtype=torch.bfloat16,
        dequant_device="cpu",
        chunk_rows=512,
    )

    model_bridge = BlaiseDeepSeekV32ReapBridge()
    model_bridge.hf_config = hf_pretrained.config
    configure_bridge_from_megatron_args(model_bridge, args)
    bridge = ConfiguredAutoBridge(hf_pretrained, model_bridge)

    print_rank_0(f"[hf-export] loading Megatron checkpoint from {args.load}")
    model = get_model(
        partial(model_provider, gpt_builder),
        wrap_with_ddp=bool(getattr(args, "use_megatron_fsdp", False)),
    )
    if args.hf_export_load_source_first:
        print_rank_0(f"[hf-export] initializing missing/base weights from {args.hf_source_model_id}")
        bridge.load_hf_weights(model)
    load_checkpoint(
        ddp_model=model,
        optimizer=None,
        opt_param_scheduler=None,
        strict=not args.hf_export_load_non_strict,
    )
    for chunk in model:
        chunk.eval()

    barrier()
    if rank() == 0:
        print(f"[hf-export] saving HF artifacts to {output_path}", flush=True)
        hf_pretrained.save_artifacts(output_path, original_source_path=args.hf_source_model_id)
    barrier()

    print_rank_0("[hf-export] streaming HF weights")
    generator = bridge.export_hf_weights(model, cpu=True, show_progress=(rank() == 0))
    if rank() == 0:
        save_streaming_safetensors(generator, output_path, max_shard_bytes)
        print(f"[hf-export] wrote HF checkpoint to {output_path}", flush=True)
    else:
        for _ in generator:
            pass
    barrier()


if __name__ == "__main__":
    main()
