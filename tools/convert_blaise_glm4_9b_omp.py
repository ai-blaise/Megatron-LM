#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Import/export the Blaise GLM-4-9B FP8 OMP checkpoint through Megatron Bridge.

This tool is intentionally self-contained under ``Megatron-LM/tools`` while
using the sibling Megatron-Bridge checkout as the source of truth for GLM4
conversion mappings.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import types
from pathlib import Path
from typing import Any


MEGATRON_LM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = MEGATRON_LM_ROOT.parent

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
    return (WORKSPACE_ROOT / "Megatron-Bridge").resolve()


def bootstrap_bridge() -> Path:
    bridge_root = resolve_bridge_root()
    bridge_src = bridge_root / "src"
    if not (bridge_src / "megatron" / "bridge" / "__init__.py").exists():
        raise FileNotFoundError(
            "Megatron Bridge source tree was not found. Set MEGATRON_BRIDGE_ROOT "
            f"or place Megatron-Bridge next to Megatron-LM. Checked: {bridge_src}"
        )

    for path in (str(bridge_src), str(MEGATRON_LM_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    install_bridge_import_shims(bridge_src)
    return bridge_root


def install_bridge_import_shims(bridge_src: Path) -> None:
    """Avoid importing Bridge's top-level registry when this tool only needs GLM4.

    ``megatron.bridge.__init__`` eagerly imports all Bridge model families,
    including optional diffusion/config dependencies.  The GLM4 converter only
    needs the conversion modules and GLM4 bridge registration, so we install
    lightweight package objects with the correct ``__path__`` and import the
    needed submodules directly.
    """

    package_paths = {
        "megatron.bridge": bridge_src / "megatron" / "bridge",
        "megatron.bridge.models": bridge_src / "megatron" / "bridge" / "models",
        "megatron.bridge.models.conversion": bridge_src / "megatron" / "bridge" / "models" / "conversion",
        "megatron.bridge.models.glm": bridge_src / "megatron" / "bridge" / "models" / "glm",
        "megatron.bridge.models.hf_pretrained": bridge_src / "megatron" / "bridge" / "models" / "hf_pretrained",
    }
    for name, path in package_paths.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        module.__package__ = name
        sys.modules[name] = module


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
    print("Import precision policy: FP8 HF source is dequantized by Bridge into normal Megatron tensors.")
    AutoBridge.import_ckpt(
        hf_model_id=args.hf_model,
        megatron_path=output,
        use_cpu_initialization=not args.use_gpu_initialization,
        **kwargs,
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
    provider = bridge.provider_bridge(bridge.hf_pretrained)
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
        default=env_bool("USE_GPU_INITIALIZATION", True),
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
