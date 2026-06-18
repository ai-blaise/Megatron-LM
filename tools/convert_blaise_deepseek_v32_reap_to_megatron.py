r#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Convert the Blaise DeepSeek-V3.2 REAP NVFP4 checkpoint to Megatron.

The Hugging Face checkpoint stores most linear weights as compressed
NVFP4 triples:

    <name>.weight_packed
    <name>.weight_scale
    <name>.weight_global_scale

Megatron Bridge expects normal ``<name>.weight`` tensors.  This converter
exposes those compressed triples as virtual BF16 weights and lets Bridge's
DeepSeek mappings shard them into a native Megatron ``torch_dist`` checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from safetensors import safe_open

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, ReplicatedMapping
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge, HAVE_TE
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import StateDict, StateSource
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.quantization.indexcache import INDEXCACHE_QUANT_NVFP4


DEFAULT_MODEL_ID = "BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4"
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def rank0_print(*parts: object) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        should_print = torch.distributed.get_rank() == 0
    else:
        should_print = int(os.environ.get("RANK", "0")) == 0
    if should_print:
        print(*parts, flush=True)


def parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def resolve_dequant_device(value: str) -> torch.device:
    normalized = value.lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("--dequant-device=cuda was requested but CUDA is not available")
        if normalized == "cuda":
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device(normalized)
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported dequant device: {value}")


def dequantize_nvfp4_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    block_size: int,
    chunk_rows: int,
) -> torch.Tensor:
    """Decode ModelOpt/compressed-tensors NVFP4 packed weights."""

    if packed.dtype != torch.uint8:
        raise TypeError(f"Expected packed uint8 tensor, got {packed.dtype}")
    if packed.shape[-1] * 2 % block_size != 0:
        raise ValueError(
            f"Unpacked dim {packed.shape[-1] * 2} is not divisible by block size {block_size}"
        )

    unpacked_last_dim = packed.shape[-1] * 2
    expected_scale_last_dim = unpacked_last_dim // block_size
    if scale.shape[-1] != expected_scale_last_dim:
        raise ValueError(
            f"Scale shape {tuple(scale.shape)} does not match packed shape {tuple(packed.shape)} "
            f"for block size {block_size}"
        )

    rows = packed.numel() // packed.shape[-1]
    flat_packed = packed.reshape(rows, packed.shape[-1])
    flat_scale = scale.reshape(rows, scale.shape[-1])
    flat_out = torch.empty((rows, unpacked_last_dim), dtype=dtype, device=device)

    lookup = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=device)
    # The HF compressed-tensors export stores the per-tensor factor as the inverse
    # global scale.  ModelOpt's in-memory dequant path multiplies by
    # ``weights_scaling_factor_2``; here that value is ``1 / weight_global_scale``.
    inv_global_scale = global_scale.reshape(-1)[0].to(device=device, dtype=torch.float32)

    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        packed_chunk = flat_packed[start:end].to(device=device, non_blocking=True)
        scale_chunk = flat_scale[start:end].to(device=device, dtype=torch.float32, non_blocking=True)

        values = torch.empty((end - start, unpacked_last_dim), dtype=torch.float32, device=device)
        values[:, 0::2] = lookup[(packed_chunk & 0x0F).long()]
        values[:, 1::2] = lookup[(packed_chunk >> 4).long()]

        values = values.view(end - start, -1, block_size)
        values = values * scale_chunk.unsqueeze(-1) / inv_global_scale
        flat_out[start:end].copy_(values.reshape(end - start, unpacked_last_dim).to(dtype))

    return flat_out.reshape(*packed.shape[:-1], unpacked_last_dim)


class CompressedNvfp4StateSource(StateSource):
    """Safetensors source that exposes compressed FP4 triples as virtual weights."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        dequant_dtype: torch.dtype,
        dequant_device: str,
        block_size: int = 16,
        chunk_rows: int = 512,
    ) -> None:
        self.model_name_or_path = path
        self.dequant_dtype = dequant_dtype
        self.dequant_device = dequant_device
        self.block_size = block_size
        self.chunk_rows = chunk_rows
        self._resolved_path_cache: Optional[Path] = None
        self._key_to_filename_map_cache: Optional[Dict[str, str]] = None
        self._real_keys_cache: Optional[List[str]] = None
        self._keys_cache: Optional[List[str]] = None
        self._virtual_to_packed_cache: Optional[Dict[str, str]] = None

    @property
    def path(self) -> Path:
        if self._resolved_path_cache is None:
            self._load_key_to_filename_map()
        assert self._resolved_path_cache is not None
        return self._resolved_path_cache

    def _is_local_dir(self) -> bool:
        return Path(self.model_name_or_path).is_dir()

    def _load_key_to_filename_map(self) -> Dict[str, str]:
        if self._key_to_filename_map_cache is not None:
            return self._key_to_filename_map_cache

        local_path = Path(self.model_name_or_path)
        if local_path.is_dir():
            index_path = local_path / "model.safetensors.index.json"
            self._resolved_path_cache = local_path
        else:
            from huggingface_hub import hf_hub_download

            index_path = Path(
                hf_hub_download(
                    repo_id=str(self.model_name_or_path),
                    filename="model.safetensors.index.json",
                )
            )
            self._resolved_path_cache = index_path.parent

        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            self._key_to_filename_map_cache = dict(index["weight_map"])
            return self._key_to_filename_map_cache

        if not local_path.is_dir():
            raise FileNotFoundError(f"No model.safetensors.index.json found for {self.model_name_or_path}")

        key_to_filename: Dict[str, str] = {}
        for safetensor_path in sorted(local_path.glob("*.safetensors")):
            with safe_open(safetensor_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    key_to_filename[key] = safetensor_path.name
        if not key_to_filename:
            raise FileNotFoundError(f"No safetensors files found in {local_path}")
        self._key_to_filename_map_cache = key_to_filename
        return key_to_filename

    def _real_keys(self) -> List[str]:
        if self._real_keys_cache is None:
            self._real_keys_cache = sorted(self._load_key_to_filename_map().keys())
        return self._real_keys_cache

    def _filename_path(self, filename: str) -> Path:
        if self._is_local_dir():
            return Path(self.model_name_or_path) / filename

        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=str(self.model_name_or_path), filename=filename))

    def _load_real_tensors(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        key_to_filename = self._load_key_to_filename_map()
        file_to_keys: Dict[str, List[str]] = defaultdict(list)
        for key in keys:
            if key not in key_to_filename:
                raise KeyError(f"Key not found in safetensors index: {key}")
            file_to_keys[key_to_filename[key]].append(key)

        loaded: Dict[str, torch.Tensor] = {}
        for filename, keys_in_file in file_to_keys.items():
            file_path = self._filename_path(filename)
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                for key in keys_in_file:
                    loaded[key] = handle.get_tensor(key)
        return loaded

    def _virtual_to_packed(self) -> Dict[str, str]:
        if self._virtual_to_packed_cache is not None:
            return self._virtual_to_packed_cache

        real_keys = set(self._real_keys())
        virtual_to_packed: Dict[str, str] = {}
        for key in real_keys:
            if not key.endswith(".weight_packed"):
                continue
            virtual_key = key[: -len("_packed")]
            scale_key = key[: -len("_packed")] + "_scale"
            global_scale_key = key[: -len("_packed")] + "_global_scale"
            if virtual_key in real_keys:
                continue
            if scale_key in real_keys and global_scale_key in real_keys:
                virtual_to_packed[virtual_key] = key

        self._virtual_to_packed_cache = virtual_to_packed
        return virtual_to_packed

    def get_all_keys(self) -> List[str]:
        if self._keys_cache is None:
            keys = set(self._real_keys())
            keys.update(self._virtual_to_packed().keys())
            self._keys_cache = sorted(keys)
        return self._keys_cache

    def load_tensors(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        if not keys:
            return {}

        real_keys: List[str] = []
        virtual_keys: List[str] = []
        real_key_set = set(self._real_keys())
        virtual_to_packed = self._virtual_to_packed()

        for key in keys:
            if key in real_key_set:
                real_keys.append(key)
            elif key in virtual_to_packed:
                virtual_keys.append(key)
            else:
                raise KeyError(f"Key not found in compressed NVFP4 source: {key}")

        loaded: Dict[str, torch.Tensor] = {}
        if real_keys:
            loaded.update(self._load_real_tensors(real_keys))

        for key in virtual_keys:
            packed_key = virtual_to_packed[key]
            scale_key = packed_key[: -len("_packed")] + "_scale"
            global_scale_key = packed_key[: -len("_packed")] + "_global_scale"
            tensors = self._load_real_tensors([packed_key, scale_key, global_scale_key])
            loaded[key] = dequantize_nvfp4_weight(
                tensors[packed_key],
                tensors[scale_key],
                tensors[global_scale_key],
                dtype=self.dequant_dtype,
                device=resolve_dequant_device(self.dequant_device),
                block_size=self.block_size,
                chunk_rows=self.chunk_rows,
            )

        return loaded

    def has_glob(self, pattern: str) -> bool:
        import fnmatch

        return any(fnmatch.fnmatch(key, pattern) for key in self.get_all_keys())


@MegatronModelBridge.register_bridge(
    source="DeepseekV3ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v3",
)
class BlaiseDeepSeekV32ReapBridge(DeepSeekV3Bridge):
    """Bridge for the Blaise REAP DeepSeek-V3.2 DSA/G1/GatedNorm checkpoint."""

    @staticmethod
    def _get_hf_indexer_quantization(hf_config):
        quant_config = getattr(hf_config, "quantization_config", None)
        if not isinstance(quant_config, dict):
            return None
        indexer_config = quant_config.get("indexer_quantization")
        return indexer_config if isinstance(indexer_config, dict) else None

    @staticmethod
    def _get_hf_kv_cache_scheme(hf_config):
        quant_config = getattr(hf_config, "quantization_config", None)
        if not isinstance(quant_config, dict):
            return None
        kv_cache_scheme = quant_config.get("kv_cache_scheme")
        return kv_cache_scheme if isinstance(kv_cache_scheme, dict) else None

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config
        hf_indexer_quantization = self._get_hf_indexer_quantization(hf_config)
        hf_kv_cache_scheme = self._get_hf_kv_cache_scheme(hf_config)

        provider.transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec
        provider.normalization = "RMSNorm"
        provider.transformer_impl = "transformer_engine"
        provider.multi_latent_attention = True
        provider.qk_layernorm = True
        provider.apply_rope_fusion = False

        provider.experimental_attention_variant = "dsa"
        provider.dsa_indexer_n_heads = getattr(hf_config, "index_n_heads", 64)
        provider.dsa_indexer_head_dim = getattr(hf_config, "index_head_dim", 128)
        provider.dsa_indexer_topk = getattr(hf_config, "index_topk", 2048)
        provider.dsa_indexer_loss_coeff = getattr(self, "dsa_indexer_loss_coeff", 0.01)
        provider.dsa_indexer_use_sparse_loss = getattr(self, "dsa_indexer_use_sparse_loss", False)
        if hf_indexer_quantization is not None:
            quant_method = hf_indexer_quantization.get("quant_method")
            if quant_method:
                provider.dsa_indexcache_quantization = str(quant_method)
            hisa_config = hf_indexer_quantization.get("hisa")
            if isinstance(hisa_config, dict) and bool(hisa_config.get("enabled", False)):
                hisa_mode = hisa_config.get("mode", "indexcache-hisa")
                if hisa_mode != "indexcache-hisa":
                    raise ValueError(f"Unsupported IndexCache HISA mode {hisa_mode!r}.")
                if provider.dsa_indexcache_quantization != INDEXCACHE_QUANT_NVFP4:
                    raise ValueError(
                        "HF IndexCache HISA config requires "
                        "quant_method='nvfp4_e2m1_ue8m0'."
                    )
                provider.dsa_indexcache_hisa_enabled = True
                provider.dsa_indexcache_hisa_block_size = int(
                    hisa_config.get("block_size", 128)
                )
                provider.dsa_indexcache_hisa_block_topk = int(
                    hisa_config.get("block_topk", 64)
                )
                provider.dsa_indexcache_hisa_compression_ratio = float(
                    hisa_config.get("compression_ratio", 4.0)
                )
        if hf_kv_cache_scheme is not None:
            quant_method = hf_kv_cache_scheme.get("quant_method")
            if quant_method == "higgs_dense_2bit":
                provider.enable_higgs_dense_2bit_kv_cache = True
                provider.higgs_kv_preset = "dense_2bit"
                provider.turboquant_kv_enabled = False
            elif quant_method:
                raise ValueError(f"Unsupported HF KV cache quantization method {quant_method!r}.")

        provider.attention_output_gate = bool(getattr(hf_config, "attention_output_gate", True))
        provider.gated_norm = bool(getattr(hf_config, "gated_norm", True))
        provider.gated_norm_rank = int(getattr(hf_config, "gated_norm_rank", 16))

        provider.mtp_num_layers = None
        provider.mtp_enabled = False

        provider.seq_length = getattr(self, "seq_length", 32768)
        provider.num_layers_in_first_pipeline_stage = getattr(
            self, "num_layers_in_first_pipeline_stage", None
        )
        provider.num_layers_in_last_pipeline_stage = getattr(
            self, "num_layers_in_last_pipeline_stage", None
        )

        provider.tensor_model_parallel_size = getattr(self, "tensor_model_parallel_size", 1)
        provider.pipeline_model_parallel_size = getattr(self, "pipeline_model_parallel_size", 1)
        provider.context_parallel_size = getattr(self, "context_parallel_size", 1)
        provider.expert_model_parallel_size = getattr(self, "expert_model_parallel_size", 1)
        provider.expert_tensor_parallel_size = getattr(self, "expert_tensor_parallel_size", 1)
        provider.sequence_parallel = getattr(self, "sequence_parallel", True)

        provider.params_dtype = torch.bfloat16
        provider.bf16 = True
        provider.fp16 = False
        provider.attention_softmax_in_fp32 = True
        provider.gradient_accumulation_fusion = False

        if getattr(hf_config, "rope_scaling", None):
            rope_scaling = hf_config.rope_scaling
            provider.rope_type = "yarn"
            provider.rotary_scaling_factor = rope_scaling.get("factor", 40.0)
            provider.original_max_position_embeddings = rope_scaling.get(
                "original_max_position_embeddings", 4096
            )
            provider.beta_fast = rope_scaling.get("beta_fast", 32.0)
            provider.beta_slow = rope_scaling.get("beta_slow", 1.0)
            provider.mscale = rope_scaling.get("mscale", 1.0)
            provider.mscale_all_dim = rope_scaling.get("mscale_all_dim", 1.0)

        provider.moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )
        provider.moe_shared_expert_intermediate_size = (
            hf_config.moe_intermediate_size * hf_config.n_shared_experts
        )

        if not HAVE_TE:
            raise RuntimeError("Transformer Engine is required for this conversion")

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf_config = copy.copy(getattr(self, "_hf_config", getattr(self, "hf_config", None)))
        if hf_config is not None:
            setattr(hf_config, "num_nextn_predict_layers", 0)

        mapping_list = get_deepseek_common_mapping_list(hf_config)
        mapping_list.extend(
            [
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.router.expert_bias",
                    hf_param="model.layers.*.mlp.gate.e_score_correction_bias",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.self_attention.core_attention.indexer.linear_wq_b.weight",
                    hf_param="model.layers.*.self_attn.indexer.wq_b.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.self_attention.core_attention.indexer.linear_wk.weight",
                    hf_param="model.layers.*.self_attn.indexer.wk.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.self_attention.core_attention.indexer.k_norm.weight",
                    hf_param="model.layers.*.self_attn.indexer.k_norm.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.self_attention.core_attention.indexer.k_norm.bias",
                    hf_param="model.layers.*.self_attn.indexer.k_norm.bias",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.self_attention.core_attention.indexer.linear_weights_proj.weight",
                    hf_param="model.layers.*.self_attn.indexer.weights_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_gate_proj.weight",
                    hf_param="model.layers.*.self_attn.gate_proj.weight",
                ),
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.input_gated_norm_down.weight",
                    hf_param="model.layers.*.input_gated_norm_down.weight",
                ),
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.input_gated_norm_up.weight",
                    hf_param="model.layers.*.input_gated_norm_up.weight",
                ),
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.pre_mlp_gated_norm_down.weight",
                    hf_param="model.layers.*.post_attention_gated_norm_down.weight",
                ),
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.pre_mlp_gated_norm_up.weight",
                    hf_param="model.layers.*.post_attention_gated_norm_up.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)


def install_state_source(
    hf_pretrained: PreTrainedCausalLM,
    *,
    dequant_dtype: torch.dtype,
    dequant_device: str,
    chunk_rows: int,
) -> CompressedNvfp4StateSource:
    source = CompressedNvfp4StateSource(
        hf_pretrained.model_name_or_path,
        dequant_dtype=dequant_dtype,
        dequant_device=dequant_device,
        chunk_rows=chunk_rows,
    )
    hf_pretrained._state_dict_accessor = StateDict(source)
    return source


class ConfiguredAutoBridge(AutoBridge):
    """AutoBridge variant that reuses a preconfigured model bridge instance."""

    def __init__(self, hf_pretrained: PreTrainedCausalLM, model_bridge: MegatronModelBridge) -> None:
        super().__init__(hf_pretrained)
        self._configured_model_bridge = model_bridge

    @property
    def _model_bridge(self) -> MegatronModelBridge:
        return self._configured_model_bridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-id", default=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument(
        "--output",
        default=os.environ.get(
            "LOAD_CKPT",
            str(Path.home() / "checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron"),
        ),
    )
    parser.add_argument("--seq-length", type=int, default=int(os.environ.get("SEQ_LENGTH", "32768")))
    parser.add_argument("--tp", type=int, default=int(os.environ.get("TP", "8")))
    parser.add_argument("--pp", type=int, default=int(os.environ.get("PP", "5")))
    parser.add_argument("--cp", type=int, default=int(os.environ.get("CP", "1")))
    parser.add_argument("--ep", type=int, default=int(os.environ.get("EP", "8")))
    parser.add_argument("--etp", type=int, default=int(os.environ.get("ETP", "1")))
    parser.add_argument(
        "--decoder-first-pipeline-num-layers",
        type=int,
        default=int(os.environ.get("DECODER_FIRST_PIPELINE_NUM_LAYERS", "13")),
    )
    parser.add_argument(
        "--decoder-last-pipeline-num-layers",
        type=int,
        default=int(os.environ.get("DECODER_LAST_PIPELINE_NUM_LAYERS", "12")),
    )
    parser.add_argument("--dsa-indexer-loss-coeff", type=float, default=float(os.environ.get("DSA_INDEXER_LOSS_COEFF", "0.01")))
    parser.add_argument("--dequant-dtype", default=os.environ.get("DEQUANT_DTYPE", "bf16"))
    parser.add_argument("--dequant-device", default=os.environ.get("DEQUANT_DEVICE", "auto"))
    parser.add_argument("--dequant-chunk-rows", type=int, default=int(os.environ.get("DEQUANT_CHUNK_ROWS", "512")))
    parser.add_argument("--validate-source-key", default=None)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--no-low-memory-save", action="store_true")
    return parser.parse_args()


def configure_model_bridge(model_bridge: MegatronModelBridge, args: argparse.Namespace) -> None:
    model_bridge.seq_length = args.seq_length
    model_bridge.tensor_model_parallel_size = args.tp
    model_bridge.pipeline_model_parallel_size = args.pp
    model_bridge.context_parallel_size = args.cp
    model_bridge.expert_model_parallel_size = args.ep
    model_bridge.expert_tensor_parallel_size = args.etp
    model_bridge.sequence_parallel = args.tp > 1
    model_bridge.dsa_indexer_loss_coeff = args.dsa_indexer_loss_coeff
    model_bridge.num_layers_in_first_pipeline_stage = (
        args.decoder_first_pipeline_num_layers if args.pp > 1 else None
    )
    model_bridge.num_layers_in_last_pipeline_stage = (
        args.decoder_last_pipeline_num_layers if args.pp > 1 else None
    )


def get_deepseek_common_mapping_list(hf_config) -> list:
    """Call Bridge's DeepSeek mapping helper across supported signatures."""

    import inspect

    if "hf_config" in inspect.signature(get_common_mapping_list).parameters:
        return get_common_mapping_list(hf_config=hf_config)
    return get_common_mapping_list()


def install_bridge_model_type_compat() -> None:
    """Adapt Bridge's model/type probes to this repo's local modules."""

    from megatron.core.enums import ModelType

    if not hasattr(ModelType, "encoder_and_decoder") and hasattr(ModelType, "encoder_or_decoder"):
        ModelType.encoder_and_decoder = ModelType.encoder_or_decoder

    AutoMapping.register_module_type("LinearCrossEntropyModule", "column")


def install_bridge_checkpoint_compat() -> None:
    """Adapt Bridge's save call to this repo's MCore checkpointing signature."""

    import inspect

    from megatron.core import dist_checkpointing

    install_bridge_tokenizer_compat()
    install_bridge_save_config_compat()

    if "async_strategy" in inspect.signature(dist_checkpointing.save).parameters:
        return

    original_save = dist_checkpointing.save

    def save_without_async_strategy(*save_args, async_strategy=None, **save_kwargs):
        del async_strategy
        return original_save(*save_args, **save_kwargs)

    dist_checkpointing.save = save_without_async_strategy


def install_bridge_tokenizer_compat() -> None:
    """Provide Bridge's legacy tokenizer import path for this Megatron branch."""

    module_name = "megatron.core.datasets.megatron_tokenizer"
    if module_name in sys.modules:
        return

    tokenizer_module = types.ModuleType(module_name)

    class MegatronTokenizer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class MegatronLegacyTokenizer(MegatronTokenizer):
        pass

    tokenizer_module.MegatronTokenizer = MegatronTokenizer
    tokenizer_module.MegatronLegacyTokenizer = MegatronLegacyTokenizer
    sys.modules[module_name] = tokenizer_module


def install_bridge_save_config_compat() -> None:
    """Avoid Bridge's fully-parallel save wrapper during one-shot conversion."""

    from megatron.bridge.training import model_load_save

    checkpoint_config = model_load_save.CheckpointConfig
    if getattr(checkpoint_config, "_blaise_non_parallel_save", False):
        return

    def checkpoint_config_without_fully_parallel(*args, **kwargs):
        kwargs.setdefault("fully_parallel_save", False)
        return checkpoint_config(*args, **kwargs)

    checkpoint_config_without_fully_parallel._blaise_non_parallel_save = True
    model_load_save.CheckpointConfig = checkpoint_config_without_fully_parallel


def main() -> None:
    args = parse_args()
    dequant_dtype = parse_dtype(args.dequant_dtype)
    install_bridge_model_type_compat()

    hf_pretrained = PreTrainedCausalLM.from_pretrained(
        args.hf_model_id,
        trust_remote_code=not args.no_trust_remote_code,
        torch_dtype=torch.bfloat16,
    )
    source = install_state_source(
        hf_pretrained,
        dequant_dtype=dequant_dtype,
        dequant_device=args.dequant_device,
        chunk_rows=args.dequant_chunk_rows,
    )

    keys = source.get_all_keys()
    virtual_count = len(source._virtual_to_packed())
    rank0_print(f"HF source: {args.hf_model_id}")
    rank0_print(f"Resolved snapshot: {source.path}")
    rank0_print(f"State keys: {len(keys)} total, {virtual_count} virtual NVFP4 weights")

    if args.validate_source_key:
        tensor = hf_pretrained.state[args.validate_source_key]
        rank0_print(
            f"{args.validate_source_key}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} mean_abs={tensor.float().abs().mean().item():.6e}"
        )
        return

    if args.metadata_only:
        return

    model_bridge = BlaiseDeepSeekV32ReapBridge()
    model_bridge.hf_config = hf_pretrained.config
    configure_model_bridge(model_bridge, args)
    bridge = ConfiguredAutoBridge(hf_pretrained, model_bridge)

    rank0_print(
        "Converting with "
        f"TP={args.tp} PP={args.pp} CP={args.cp} EP={args.ep} ETP={args.etp} "
        f"seq={args.seq_length}"
    )
    if args.structure_only:
        model = bridge.to_megatron_model(
            load_weights=False,
            wrap_with_ddp=False,
            init_model_with_meta_device=True,
            mixed_precision_wrapper=None,
        )
        tasks = bridge.get_conversion_tasks(model)
        missing = [task for task in tasks if task is None]
        present = [task for task in tasks if task is not None]
        rank0_print(f"Structure tasks: {len(present)} mapped, {len(missing)} missing/skipped")
        rank0_print("First mapped parameters:")
        for task in present[:20]:
            rank0_print(f"  {task.global_param_name} <- {task.mapping.hf_param}")
        return

    model = bridge.to_megatron_model(
        load_weights=False,
        wrap_with_ddp=False,
        use_cpu_initialization=False,
        mixed_precision_wrapper=None,
    )
    bridge.load_hf_weights(model)
    install_bridge_checkpoint_compat()
    bridge.save_megatron_model(
        model,
        args.output,
        hf_tokenizer_path=args.hf_model_id,
        hf_tokenizer_kwargs={"trust_remote_code": not args.no_trust_remote_code},
        low_memory_save=not args.no_low_memory_save,
    )
    rank0_print(f"Saved Megatron checkpoint to {args.output}")


if __name__ == "__main__":
    main()
