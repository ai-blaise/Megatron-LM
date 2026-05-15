#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validate a converted Blaise DeepSeek-V3.2 REAP Megatron checkpoint.

This is a targeted conversion smoke.  It loads selected tensors directly from
the Megatron ``torch_dist`` checkpoint on CPU, then compares them against the HF
source tensors exposed by the NVFP4-aware state source.  It intentionally avoids
constructing the full model because Bridge's generic checkpoint loader
materializes dense BF16 parameters and can OOM for this model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

from tools.convert_blaise_deepseek_v32_reap_to_megatron import (
    DEFAULT_MODEL_ID,
    install_state_source,
    parse_dtype,
)


DEFAULT_VALIDATE_PAIRS = [
    (
        "model.layers.0.input_layernorm.weight",
        "decoder.layers.0.input_layernorm.weight",
    ),
    (
        "model.layers.0.input_gated_norm_up.weight",
        "decoder.layers.0.input_gated_norm_up.weight",
    ),
    (
        "model.layers.0.self_attn.indexer.k_norm.weight",
        "decoder.layers.0.self_attention.core_attention.indexer.k_norm.weight",
    ),
    (
        "model.layers.0.self_attn.indexer.weights_proj.weight",
        "decoder.layers.0.self_attention.core_attention.indexer.linear_weights_proj.weight",
    ),
    (
        "model.layers.0.self_attn.indexer.wk.weight",
        "decoder.layers.0.self_attention.core_attention.indexer.linear_wk.weight",
    ),
    (
        "model.layers.0.self_attn.q_a_proj.weight",
        "decoder.layers.0.self_attention.linear_q_down_proj.weight",
    ),
    (
        "model.layers.3.mlp.gate.weight",
        "decoder.layers.3.mlp.router.weight",
    ),
    (
        "model.layers.3.mlp.gate.e_score_correction_bias",
        "decoder.layers.3.mlp.router.expert_bias",
    ),
    (
        "model.layers.60.input_layernorm.weight",
        "decoder.layers.60.input_layernorm.weight",
    ),
    (
        "model.layers.60.self_attn.indexer.weights_proj.weight",
        "decoder.layers.60.self_attention.core_attention.indexer.linear_weights_proj.weight",
    ),
]


def rank0_print(*parts: object) -> None:
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return
    print(*parts, flush=True)


def maybe_destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def maybe_init_process_group() -> None:
    if not dist.is_available() or dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    dist.init_process_group(backend="gloo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-id", default=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "LOAD_CKPT",
            str(Path.home() / "checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron"),
        ),
    )
    parser.add_argument("--dequant-dtype", default=os.environ.get("DEQUANT_DTYPE", "bf16"))
    parser.add_argument("--dequant-device", default=os.environ.get("DEQUANT_DEVICE", "cpu"))
    parser.add_argument("--dequant-chunk-rows", type=int, default=int(os.environ.get("DEQUANT_CHUNK_ROWS", "512")))
    parser.add_argument("--atol", type=float, default=float(os.environ.get("VALIDATE_ATOL", "0.0")))
    parser.add_argument("--rtol", type=float, default=float(os.environ.get("VALIDATE_RTOL", "0.0")))
    parser.add_argument(
        "--pair",
        action="append",
        help="Validation pair in HF_KEY=MCORE_KEY form. Can be repeated.",
    )
    return parser.parse_args()


def parse_pairs(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return DEFAULT_VALIDATE_PAIRS
    pairs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected HF_KEY=MCORE_KEY pair, got {value!r}")
        hf_key, mcore_key = value.split("=", 1)
        pairs.append((hf_key, mcore_key))
    return pairs


def checkpoint_iteration_dir(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.name.startswith("iter_"):
        return path
    latest_file = path / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        iteration = int(latest_file.read_text().strip())
        return path / f"iter_{iteration:07d}"
    iter_dirs = [child for child in path.iterdir() if child.is_dir() and child.name.startswith("iter_")]
    if not iter_dirs:
        raise FileNotFoundError(f"No iter_* checkpoint directory found under {path}")
    return max(iter_dirs, key=lambda child: int(child.name.removeprefix("iter_")))


def load_mcore_tensors(checkpoint_dir: Path, mcore_keys: Iterable[str]) -> dict[str, torch.Tensor]:
    import pickle

    metadata_path = checkpoint_dir / ".metadata"
    with open(metadata_path, "rb") as handle:
        metadata = pickle.load(handle)

    state = {}
    for key in mcore_keys:
        tensor_metadata = metadata.state_dict_metadata.get(key)
        if tensor_metadata is None:
            raise KeyError(f"MCore key {key!r} not found in checkpoint metadata")
        state[key] = torch.empty(
            tuple(tensor_metadata.size),
            dtype=tensor_metadata.properties.dtype,
            device="cpu",
        )

    dcp.load(state, checkpoint_id=str(checkpoint_dir))
    return state


def main() -> None:
    maybe_init_process_group()
    args = parse_args()
    validate_pairs = parse_pairs(args.pair)
    dequant_dtype = parse_dtype(args.dequant_dtype)
    checkpoint_dir = checkpoint_iteration_dir(args.checkpoint)

    rank0_print(f"Loading selected tensors from: {checkpoint_dir}")
    mcore_tensors = load_mcore_tensors(checkpoint_dir, [mcore_key for _, mcore_key in validate_pairs])

    records = []
    is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
    if is_rank0:
        hf_pretrained = PreTrainedCausalLM.from_pretrained(
            args.hf_model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        install_state_source(
            hf_pretrained,
            dequant_dtype=dequant_dtype,
            dequant_device=args.dequant_device,
            chunk_rows=args.dequant_chunk_rows,
        )

        for hf_key, mcore_key in validate_pairs:
            source = hf_pretrained.state[hf_key]
            converted = mcore_tensors[mcore_key]
            converted_cpu = converted.detach().to(device="cpu", dtype=torch.float32)
            source_cpu = source.detach().to(device="cpu", dtype=torch.float32)
            if converted_cpu.shape != source_cpu.shape:
                records.append(
                    {
                        "hf_key": hf_key,
                        "mcore_key": mcore_key,
                        "status": "shape_mismatch",
                        "converted_shape": tuple(converted_cpu.shape),
                        "source_shape": tuple(source_cpu.shape),
                    }
                )
                continue
            diff = (converted_cpu - source_cpu).abs()
            allclose = torch.allclose(converted_cpu, source_cpu, atol=args.atol, rtol=args.rtol)
            records.append(
                {
                    "hf_key": hf_key,
                    "mcore_key": mcore_key,
                    "status": "ok" if allclose else "value_mismatch",
                    "shape": tuple(converted_cpu.shape),
                    "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
                    "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
                    "source_mean_abs": float(source_cpu.abs().mean().item()) if source_cpu.numel() else 0.0,
                }
            )

        failures = [record for record in records if record["status"] != "ok"]
        print("Validation results:", flush=True)
        for record in records:
            print(f"  {record}", flush=True)
        if failures:
            maybe_destroy_process_group()
            raise SystemExit(1)
        print(f"Validated {len(records)} HF tensors against the Megatron checkpoint.", flush=True)

    maybe_destroy_process_group()


if __name__ == "__main__":
    main()
