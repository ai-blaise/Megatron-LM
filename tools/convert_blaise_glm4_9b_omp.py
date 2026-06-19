#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Import/export the Blaise GLM-4-9B FP8 OMP checkpoint through Megatron Bridge.

This tool is intentionally self-contained under ``Megatron-LM/tools`` while
using the user's Megatron-Bridge checkout as the source of truth for GLM4
conversion mappings.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from pathlib import Path
from typing import Any

import torch


MEGATRON_LM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE_ROOT = Path.home() / "Megatron-Bridge"

DEFAULT_HF_MODEL = "BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP"
DEFAULT_IMPORT_OUTPUT = str(Path.home() / "checkpoints/glm4_9b_omp_init")
DEFAULT_EXPORT_INPUT = str(Path.home() / "checkpoints/glm4_9b_omp_trained")
DEFAULT_HF_OUTPUT_PATH = str(Path.home() / "models/GLM-4-9B-0414-OMP-Finetuned")
DEFAULT_TORCH_DTYPE = "bfloat16"
DEFAULT_TP_SIZE = 1
DEFAULT_PP_SIZE = 1

CRITICAL_CONFIG_FIELDS = (
    "model_type",
    "architectures",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "tie_word_embeddings",
    "attention_bias",
    "partial_rotary_factor",
    "max_position_embeddings",
    "rms_norm_eps",
)

EXPECTED_CONFIG_VALUES = {
    "model_type": "glm4",
    "architectures": ["Glm4ForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "vocab_size": 128815,
    "tie_word_embeddings": False,
    "attention_bias": True,
    "partial_rotary_factor": 0.5,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-5,
}


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def resolve_bridge_root() -> Path:
    bridge_root = os.environ.get("MEGATRON_BRIDGE_ROOT")
    if bridge_root:
        return expand_path(bridge_root).resolve()
    return DEFAULT_BRIDGE_ROOT.resolve()


def bootstrap_bridge() -> Path:
    bridge_root = resolve_bridge_root()
    bridge_src = bridge_root / "src"
    if not (bridge_src / "megatron" / "bridge" / "__init__.py").exists():
        raise FileNotFoundError(
            "Megatron Bridge source tree was not found. Set MEGATRON_BRIDGE_ROOT "
            f"or place Megatron-Bridge at {DEFAULT_BRIDGE_ROOT}. Checked: {bridge_src}"
        )

    for path in (str(bridge_src), str(MEGATRON_LM_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return bridge_root


BRIDGE_ROOT = bootstrap_bridge()


def parse_dtype(value: str) -> torch.dtype:
    import torch

    normalized = value.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def config_value(config: Any, field: str) -> Any:
    if hasattr(config, field):
        return getattr(config, field)
    if field == "partial_rotary_factor":
        rope = getattr(config, "rope_parameters", None)
        if isinstance(rope, dict):
            return rope.get("partial_rotary_factor")
    return None


def config_contract(config: Any) -> dict[str, Any]:
    return {field: config_value(config, field) for field in CRITICAL_CONFIG_FIELDS}


def values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return actual is not None and abs(float(actual) - expected) < 1e-12
    return actual == expected


def validate_config_contract(config: Any, *, allow_mismatch: bool) -> list[str]:
    records = []
    for field, expected in EXPECTED_CONFIG_VALUES.items():
        actual = config_value(config, field)
        if not values_match(actual, expected):
            records.append(f"{field}: expected {expected!r}, got {actual!r}")

    quant_config = getattr(config, "quantization_config", None)
    if not isinstance(quant_config, dict):
        records.append("quantization_config: missing or not a dict")
    else:
        if quant_config.get("quant_method") != "fp8":
            records.append(f"quantization_config.quant_method: expected 'fp8', got {quant_config.get('quant_method')!r}")
        if quant_config.get("activation_scheme") != "dynamic":
            records.append(
                "quantization_config.activation_scheme: "
                f"expected 'dynamic', got {quant_config.get('activation_scheme')!r}"
            )
        if quant_config.get("weight_block_size") != [128, 128]:
            records.append(
                "quantization_config.weight_block_size: "
                f"expected [128, 128], got {quant_config.get('weight_block_size')!r}"
            )
        kv_cache_scheme = quant_config.get("kv_cache_scheme")
        if not isinstance(kv_cache_scheme, dict):
            records.append("quantization_config.kv_cache_scheme: missing or not a dict")
        elif kv_cache_scheme.get("quant_method") != "higgs_mha_2bit":
            records.append(
                "quantization_config.kv_cache_scheme.quant_method: "
                f"expected 'higgs_mha_2bit', got {kv_cache_scheme.get('quant_method')!r}"
            )

    if records and not allow_mismatch:
        detail = "\n  - ".join(records)
        raise RuntimeError(f"HF config does not match the GLM4 OMP contract:\n  - {detail}")
    return records


def load_hf_config(hf_model: str, *, trust_remote_code: bool) -> Any:
    from transformers import AutoConfig

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kwargs = {"trust_remote_code": trust_remote_code}
    if token:
        kwargs["token"] = token
    return AutoConfig.from_pretrained(hf_model, **kwargs)


def load_hf_pretrained(hf_model: str, *, trust_remote_code: bool, torch_dtype: torch.dtype) -> PreTrainedCausalLM:
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if token:
        kwargs["token"] = token
    return PreTrainedCausalLM.from_pretrained(hf_model, **kwargs)


def get_auto_bridge_class():
    import megatron.bridge.models.glm.glm4_bridge  # noqa: F401 - registers GLM4Bridge
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge

    return AutoBridge


def configure_fp8_provider(provider: Any) -> None:
    provider.perform_initialization = False
    provider.bf16 = True
    provider.fp16 = False
    provider.params_dtype = torch.bfloat16
    provider.autocast_dtype = torch.bfloat16
    provider.fp8 = "e4m3"
    provider.fp8_recipe = "blockwise"
    provider.fp8_param = True


def is_blockwise_fp8_param(param_weight: torch.Tensor | None) -> bool:
    return bool(
        param_weight is not None
        and hasattr(param_weight, "_rowwise_data")
        and hasattr(param_weight, "_rowwise_scale_inv")
    )


def copy_tensor_with_padding(destination: torch.Tensor, source: torch.Tensor, *, fill_value: float | int) -> None:
    if destination.shape == source.shape:
        destination.copy_(source.to(dtype=destination.dtype, device=destination.device))
        return

    if destination.dim() != source.dim():
        raise ValueError(
            f"Cannot copy tensor with rank mismatch: dest={tuple(destination.shape)} src={tuple(source.shape)}"
        )
    for dest_size, src_size in zip(destination.shape, source.shape):
        if src_size > dest_size:
            raise ValueError(
                f"Cannot copy tensor with larger source shape: dest={tuple(destination.shape)} src={tuple(source.shape)}"
            )

    destination.fill_(fill_value)
    slices = tuple(slice(0, size) for size in source.shape)
    destination[slices].copy_(source.to(dtype=destination.dtype, device=destination.device))


def copy_blockwise_fp8_param(param_weight: torch.Tensor, converted_weight: torch.Tensor, converted_scale: torch.Tensor) -> None:
    raw_data = getattr(param_weight, "_rowwise_data")
    rowwise_scale_inv = getattr(param_weight, "_rowwise_scale_inv")
    raw_weight = converted_weight.contiguous()
    if raw_data.dtype != raw_weight.dtype:
        raw_weight = raw_weight.view(raw_data.dtype)
    copy_tensor_with_padding(raw_data, raw_weight, fill_value=0)
    copy_tensor_with_padding(rowwise_scale_inv, converted_scale, fill_value=1.0)


def merge_qkv_scale_inv(config: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Merge Q/K/V block-scale tensors using Megatron's interleaved QKV order."""
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups
    heads_per_group = head_num // num_query_groups
    q_head_size = getattr(config, "kv_channels", None) or (config.hidden_size // head_num)

    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise RuntimeError(f"Expected 2D QKV scale tensors, got q={q.shape}, k={k.shape}, v={v.shape}")
    if k.shape != v.shape:
        raise RuntimeError(f"Expected K/V scale shapes to match, got k={k.shape}, v={v.shape}")
    if q.shape[1] != k.shape[1]:
        raise RuntimeError(f"Expected Q/K/V scale tile columns to match, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[0] % head_num != 0 or k.shape[0] % num_query_groups != 0:
        raise RuntimeError(f"Cannot map QKV scale rows to heads: q={q.shape}, k={k.shape}, v={v.shape}")

    q_scale_rows_per_head = q.shape[0] // head_num
    kv_scale_rows_per_group = k.shape[0] // num_query_groups
    expected_q_rows = max(1, q_head_size // 128)
    if q_scale_rows_per_head != expected_q_rows or kv_scale_rows_per_group != expected_q_rows:
        raise RuntimeError(
            "Unexpected QKV scale row layout: "
            f"q_rows_per_head={q_scale_rows_per_head}, kv_rows_per_group={kv_scale_rows_per_group}, "
            f"expected={expected_q_rows}, q={q.shape}, k={k.shape}, v={v.shape}"
        )

    q_reshaped = q.view(head_num, q_scale_rows_per_head, q.shape[1])
    k_reshaped = k.view(num_query_groups, kv_scale_rows_per_group, k.shape[1])
    v_reshaped = v.view(num_query_groups, kv_scale_rows_per_group, v.shape[1])

    qkv_scales = []
    for i in range(num_query_groups):
        q_group = q_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
        k_group = k_reshaped[i : i + 1]
        v_group = v_reshaped[i : i + 1]
        qkv_scales.extend([q_group, k_group, v_group])

    return torch.cat(qkv_scales, dim=0).reshape(-1, q.shape[1])


def convert_hf_scale_to_megatron(mapping: Any, hf_scales: Any, megatron_module: Any) -> torch.Tensor:
    if isinstance(hf_scales, dict) and {"q", "k", "v"} <= set(hf_scales):
        config = mapping._get_config(megatron_module)
        merged_scales = merge_qkv_scale_inv(config, hf_scales["q"], hf_scales["k"], hf_scales["v"])
        tp_mapping = getattr(mapping, "_tp_mapping", None)
        if tp_mapping is not None:
            return tp_mapping.hf_to_megatron(merged_scales, megatron_module)
        return merged_scales
    return mapping.hf_to_megatron(hf_scales, megatron_module)


def describe_hf_quantized_source(hf_param: Any, hf_state_dict: Any, quantization_utils: Any) -> str:
    if isinstance(hf_param, dict):
        return "; ".join(
            f"{key}: {describe_hf_quantized_source(value, hf_state_dict, quantization_utils)}"
            for key, value in hf_param.items()
        )

    try:
        weight = hf_state_dict[hf_param]
    except Exception as exc:
        return f"{hf_param}: missing weight ({exc})"

    existing_scale_keys = [
        key for key in quantization_utils.hf_quantized_scale_key_candidates(hf_param) if key in hf_state_dict
    ]
    return f"{hf_param}: dtype={weight.dtype}, shape={tuple(weight.shape)}, scale_keys={existing_scale_keys}"


def load_glm4_omp_fp8_weights(bridge: Any, megatron_model: Any) -> None:
    from megatron.bridge.models.conversion import quantization_utils

    model_bridge = bridge._model_bridge
    pre_trained = bridge.hf_pretrained
    tasks = bridge.get_conversion_tasks(megatron_model)
    hf_state_dict = pre_trained.state if hasattr(pre_trained, "state") else {}

    for task in model_bridge._with_progress_tracking(
        tasks,
        f"Loading from {pre_trained.model_name_or_path}",
    ):
        if task is None or task.megatron_module is None:
            continue

        is_fp8_target = is_blockwise_fp8_param(task.param_weight)
        raw_quantized_pair = quantization_utils.load_hf_quantized_weight_scale_pair(
            task.mapping.hf_param,
            hf_state_dict,
        )

        if raw_quantized_pair is not None:
            if not is_fp8_target:
                raise RuntimeError(
                    "HF FP8 weight maps to a non-FP8 Megatron parameter: "
                    f"megatron={task.mapping.megatron_param}, hf={task.mapping.hf_param}"
                )
            hf_weights_raw, hf_scales_raw = raw_quantized_pair
            converted_weights = task.mapping.hf_to_megatron(hf_weights_raw, task.megatron_module)
            converted_scales = convert_hf_scale_to_megatron(task.mapping, hf_scales_raw, task.megatron_module)
            if converted_weights is not None:
                assert task.param_weight is not None, "param_weight is required for HF->Megatron conversion"
                with torch.no_grad():
                    copy_blockwise_fp8_param(task.param_weight, converted_weights, converted_scales)
            continue
        if is_fp8_target:
            raise RuntimeError(
                "Megatron FP8 parameter did not find a raw HF FP8 weight+scale pair: "
                f"megatron={task.mapping.megatron_param}, hf={task.mapping.hf_param}; "
                f"{describe_hf_quantized_source(task.mapping.hf_param, hf_state_dict, quantization_utils)}"
            )

        hf_weights = model_bridge.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)
        converted_weights = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)
        if converted_weights is None:
            continue
        assert task.param_weight is not None, "param_weight is required for HF->Megatron conversion"
        with torch.no_grad():
            task.param_weight.copy_(converted_weights)

    model_bridge._broadcast_shared_embeddings(megatron_model)


def has_any_glob(hf_pretrained: PreTrainedCausalLM, patterns: tuple[str, ...]) -> bool:
    state = hf_pretrained.state
    for pattern in patterns:
        if state.has_glob(pattern):
            return True
    return False


def validate_hf_state(hf_pretrained: PreTrainedCausalLM, *, allow_mismatch: bool) -> list[str]:
    records = []
    omp_patterns = ("*.mlp.gate_up_proj.weight", "*.mlp.gate_up_proj.weight*")
    weight_patterns = ("*.weight", "*.weight*")
    scale_patterns = (
        "*.scale",
        "*.scale*",
        "*_scale",
        "*_scale*",
        "*scale_inv",
        "*scale_inv*",
        "*.weight_scale",
        "*.weight_scale*",
    )
    if not has_any_glob(hf_pretrained, omp_patterns):
        records.append("OMP fused gate_up_proj weights were not found")
    if not has_any_glob(hf_pretrained, weight_patterns):
        records.append("HF weight tensors were not found")
    if not has_any_glob(hf_pretrained, scale_patterns):
        records.append(f"FP8 scale tensors were not found with patterns: {', '.join(scale_patterns)}")

    if records and not allow_mismatch:
        detail = "\n  - ".join(records)
        raise RuntimeError(f"HF checkpoint state does not match GLM4 OMP expectations:\n  - {detail}")
    return records


def bridge_class_name(bridge: AutoBridge) -> str:
    return type(bridge._model_bridge).__name__


def derived_export_overrides(config: Any) -> dict[str, Any]:
    return {
        "num_query_groups": int(config_value(config, "num_key_value_heads")),
        "add_qkv_bias": bool(config_value(config, "attention_bias")),
    }


def print_provenance(hf_model: str, args: argparse.Namespace) -> None:
    import megatron.core
    import torch
    import transformers

    print("=== GLM4 OMP conversion provenance ===")
    print(f"python:             {sys.executable}")
    print(f"torch:              {torch.__version__}")
    print(f"transformers:       {transformers.__version__}")
    print(f"megatron.bridge:    {BRIDGE_ROOT / 'src' / 'megatron' / 'bridge'}")
    print(f"megatron.core:      {pathlib.Path(megatron.core.__file__).resolve()}")
    print(f"Megatron-LM root:   {MEGATRON_LM_ROOT}")
    print(f"Megatron-Bridge:    {BRIDGE_ROOT}")
    print(f"HF model:           {hf_model}")
    print(f"trust remote code:  {args.trust_remote_code}")
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("HF auth token:      present")
    else:
        print("HF auth token:      not set")
    print()


def run_preflight(args: argparse.Namespace) -> tuple[Any, PreTrainedCausalLM, AutoBridge]:
    AutoBridge = get_auto_bridge_class()

    torch_dtype = parse_dtype(args.torch_dtype)
    print_provenance(args.hf_model, args)
    config = load_hf_config(args.hf_model, trust_remote_code=args.trust_remote_code)
    config_mismatches = validate_config_contract(config, allow_mismatch=args.allow_config_mismatch)

    hf_pretrained = load_hf_pretrained(
        args.hf_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    state_mismatches = validate_hf_state(hf_pretrained, allow_mismatch=args.allow_config_mismatch)
    bridge = AutoBridge(hf_pretrained)
    resolved_bridge = bridge_class_name(bridge)
    if resolved_bridge != "GLM4Bridge" and not args.allow_config_mismatch:
        raise RuntimeError(f"Expected AutoBridge to resolve GLM4Bridge, got {resolved_bridge}")

    print("=== HF config contract ===")
    print(json.dumps(config_contract(config), indent=2, sort_keys=True, default=str))
    print()
    print("=== Derived export overrides ===")
    print(json.dumps(derived_export_overrides(config), indent=2, sort_keys=True))
    print()
    print("=== Bridge resolution ===")
    print(f"AutoBridge model bridge: {resolved_bridge}")
    if config_mismatches or state_mismatches:
        print("Allowed mismatches:")
        for record in config_mismatches + state_mismatches:
            print(f"  - {record}")
    else:
        print("GLM4 OMP source checks passed.")
    print()
    return config, hf_pretrained, bridge


def latest_iteration_dir(path: Path) -> Path | None:
    if path.name.startswith("iter_"):
        return path if path.exists() else None
    latest_file = path / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        raw = latest_file.read_text(encoding="utf-8").strip()
        try:
            iteration = int(raw)
        except ValueError:
            return None
        candidate = path / f"iter_{iteration:07d}"
        if candidate.exists():
            return candidate
    if not path.exists():
        return None
    iter_dirs = [child for child in path.iterdir() if child.is_dir() and child.name.startswith("iter_")]
    if not iter_dirs:
        return None
    return max(iter_dirs, key=lambda child: int(child.name.removeprefix("iter_")))


def disable_conversion_only_offload(provider: Any) -> None:
    """Disable training offload features that allocate CUDA streams during import."""

    overrides = {
        "cpu_offloading": False,
        "cpu_offloading_activations": False,
        "cpu_offloading_weights": False,
        "cpu_offloading_double_buffering": False,
        "cpu_offloading_num_layers": 0,
        "fine_grained_activation_offloading": False,
        "offload_modules": [],
    }
    for key, value in overrides.items():
        if hasattr(provider, key):
            setattr(provider, key, value)


def assert_conversion_offload_disabled(provider: Any) -> None:
    enabled = [
        key
        for key in (
            "cpu_offloading",
            "cpu_offloading_activations",
            "cpu_offloading_weights",
            "cpu_offloading_double_buffering",
            "fine_grained_activation_offloading",
        )
        if bool(getattr(provider, key, False))
    ]
    if enabled:
        raise RuntimeError(f"Conversion import requires offload disabled; still enabled: {', '.join(enabled)}")


def run_import(args: argparse.Namespace) -> int:
    run_preflight(args)
    AutoBridge = get_auto_bridge_class()
    output = expand_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch_dtype = parse_dtype(args.torch_dtype)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if token:
        kwargs["token"] = token

    print("=== HF -> Megatron import ===")
    print(f"Output checkpoint root: {output}")
    print("Import precision policy: preserve HF FP8 weights/scales and build Megatron blockwise FP8 params.")

    bridge = AutoBridge.from_hf_pretrained(args.hf_model, **kwargs)
    bridge.export_weight_dtype = "fp8"
    provider = bridge.to_megatron_provider(load_weights=False)
    configure_fp8_provider(provider)
    disable_conversion_only_offload(provider)
    if hasattr(provider, "finalize"):
        provider.finalize()
    disable_conversion_only_offload(provider)
    assert_conversion_offload_disabled(provider)
    megatron_model = provider.provide_distributed_model(
        wrap_with_ddp=False,
        use_cpu_initialization=not args.use_gpu_initialization,
        mixed_precision_wrapper=None,
    )
    load_glm4_omp_fp8_weights(bridge, megatron_model)

    hf_tokenizer_kwargs = {}
    if hasattr(bridge._model_bridge, "get_hf_tokenizer_kwargs"):
        hf_tokenizer_kwargs = bridge._model_bridge.get_hf_tokenizer_kwargs()
    if args.trust_remote_code:
        if hf_tokenizer_kwargs is None:
            hf_tokenizer_kwargs = {}
        hf_tokenizer_kwargs.setdefault("trust_remote_code", True)

    bridge.save_megatron_model(
        megatron_model,
        output,
        hf_tokenizer_path=args.hf_model,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        low_memory_save=True,
    )

    iteration_dir = latest_iteration_dir(output)
    if iteration_dir is None:
        raise RuntimeError(f"Import completed but no iter_* checkpoint directory was found under {output}")

    print()
    print("=== Import complete ===")
    print(f"Megatron checkpoint root: {output}")
    print(f"Initial SFT load path:    {iteration_dir}")
    print()
    print("SFT handoff:")
    print("MODEL_PROFILE=glm4_9b_omp \\")
    print(f'MEGATRON_CKPT="{output}" \\')
    print('DATA_ROOT="$HOME/data/my_sft_jsonl" \\')
    print("bash examples/sft/sft.sh")
    return 0


def has_run_config(path: Path) -> bool:
    if (path / "run_config.yaml").exists():
        return True
    iteration_dir = latest_iteration_dir(path)
    return bool(iteration_dir and (iteration_dir / "run_config.yaml").exists())


def configure_export_provider(bridge: AutoBridge, config: Any, args: argparse.Namespace) -> Any:
    provider = bridge.to_megatron_provider(load_weights=False)
    overrides = derived_export_overrides(config)
    overrides.update(args.model_override or {})
    for key, value in overrides.items():
        if hasattr(provider, key):
            setattr(provider, key, value)
        elif not args.allow_config_mismatch:
            raise RuntimeError(f"Provider has no export override attribute {key!r}")
    provider.tensor_model_parallel_size = args.tp_size
    provider.pipeline_model_parallel_size = args.pp_size
    return provider


def parse_model_overrides(values: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not values:
        return result
    for value in values:
        key, sep, raw = value.partition("=")
        if not sep or not key:
            raise ValueError(f"Invalid --model-override {value!r}; expected key=value")
        lowered = raw.lower()
        if lowered == "true":
            parsed: Any = True
        elif lowered == "false":
            parsed = False
        else:
            try:
                parsed = int(raw)
            except ValueError:
                try:
                    parsed = float(raw)
                except ValueError:
                    parsed = raw
        result[key] = parsed
    return result


def load_output_config(output_path: Path, *, trust_remote_code: bool) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(output_path), trust_remote_code=trust_remote_code)


def verify_hf_output(source_config: Any, output_path: Path, *, trust_remote_code: bool, allow_mismatch: bool) -> None:
    missing = []
    if not (output_path / "config.json").exists():
        missing.append("config.json")
    if not (output_path / "model.safetensors.index.json").exists():
        missing.append("model.safetensors.index.json")
    if not list(output_path.glob("model-*.safetensors")):
        missing.append("model-*.safetensors")
    if missing:
        raise RuntimeError(f"HF export is missing expected files: {', '.join(missing)}")

    output_config = load_output_config(output_path, trust_remote_code=trust_remote_code)
    mismatches = []
    for field in CRITICAL_CONFIG_FIELDS:
        source_value = config_value(source_config, field)
        output_value = config_value(output_config, field)
        if source_value != output_value:
            mismatches.append(f"{field}: source {source_value!r}, output {output_value!r}")
    if mismatches and not allow_mismatch:
        detail = "\n  - ".join(mismatches)
        raise RuntimeError(f"HF export config shape mismatch:\n  - {detail}")
    if mismatches:
        print("Allowed output config mismatches:")
        for mismatch in mismatches:
            print(f"  - {mismatch}")


def run_export(args: argparse.Namespace) -> int:
    config, _, bridge = run_preflight(args)
    megatron_path = expand_path(args.megatron_path)
    hf_output_path = expand_path(args.hf_output_path)
    if not megatron_path.exists():
        raise FileNotFoundError(f"Megatron checkpoint path does not exist: {megatron_path}")
    hf_output_path.parent.mkdir(parents=True, exist_ok=True)

    model_cfg = None
    if not has_run_config(megatron_path):
        print("No run_config.yaml found; using HF-config-derived provider overrides.")
        model_cfg = configure_export_provider(bridge, config, args)

    print("=== Megatron -> HF export ===")
    print(f"Megatron checkpoint: {megatron_path}")
    print(f"HF output path:      {hf_output_path}")
    bridge.export_weight_dtype = "fp8"
    bridge.export_ckpt(
        megatron_path=megatron_path,
        hf_path=hf_output_path,
        show_progress=not args.no_progress,
        strict=not args.not_strict,
        source_path=args.hf_model,
        model_cfg=model_cfg,
    )
    verify_hf_output(
        config,
        hf_output_path,
        trust_remote_code=args.trust_remote_code,
        allow_mismatch=args.allow_config_mismatch,
    )
    print()
    print("=== Export complete ===")
    print(f"HF model directory: {hf_output_path}")
    print("Push to HuggingFace Hub:")
    print(f'  hf upload your-org/your-repo "{hf_output_path}" .')
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hf-model", default=env_default("HF_MODEL", DEFAULT_HF_MODEL))
    parser.add_argument("--torch-dtype", default=env_default("TORCH_DTYPE", DEFAULT_TORCH_DTYPE))
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=env_bool("TRUST_REMOTE_CODE", True),
    )
    parser.add_argument("--allow-config-mismatch", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate Bridge, HF config, and HF checkpoint state")
    add_common_args(preflight)

    import_parser = subparsers.add_parser("import", help="Import HF GLM4 FP8 OMP to Megatron torch_dist")
    add_common_args(import_parser)
    import_parser.add_argument("--output", default=env_default("IMPORT_OUTPUT", DEFAULT_IMPORT_OUTPUT))
    import_parser.add_argument(
        "--use-gpu-initialization",
        dest="use_gpu_initialization",
        action=argparse.BooleanOptionalAction,
        default=env_bool("USE_GPU_INITIALIZATION", False),
    )

    export_parser = subparsers.add_parser("export", help="Export trained Megatron checkpoint to HF safetensors")
    add_common_args(export_parser)
    export_parser.add_argument("--megatron-path", default=env_default("MEGATRON_CKPT", DEFAULT_EXPORT_INPUT))
    export_parser.add_argument("--hf-output-path", default=env_default("HF_OUTPUT_PATH", env_default("HF_PATH", DEFAULT_HF_OUTPUT_PATH)))
    export_parser.add_argument("--tp-size", type=int, default=int(env_default("TP_SIZE", str(DEFAULT_TP_SIZE))))
    export_parser.add_argument("--pp-size", type=int, default=int(env_default("PP_SIZE", str(DEFAULT_PP_SIZE))))
    export_parser.add_argument("--no-progress", action="store_true")
    export_parser.add_argument("--not-strict", action="store_true")
    export_parser.add_argument(
        "--model-override",
        action="append",
        default=[],
        help="Provider override as key=value. Derived config values are used by default.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "model_override"):
        args.model_override = parse_model_overrides(args.model_override)

    if args.command == "preflight":
        run_preflight(args)
        return 0
    if args.command == "import":
        return run_import(args)
    if args.command == "export":
        return run_export(args)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
