# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import NoReturn, Optional, Union

import torch

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


def _env_flag_enabled(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _maybe_trim_cuda_cache_before_mla_key_cat(
    k_no_pe: torch.Tensor,
    k_pos_emb: torch.Tensor,
) -> None:
    """Release cached CUDA blocks before the large MLA key materialization.

    StreamBP DSA replay builds the full-context MLA key for every attention
    replay chunk. The allocator can be sitting on multiple GiB of reserved but
    unallocated blocks at this boundary, which is enough to make the full key
    cat fail even though the raw working set would fit. Keep this conditional
    so steady-state runs only pay the synchronization cost near the memory cliff.
    """

    if not _env_flag_enabled("MEGATRON_MLA_TRIM_CACHE_BEFORE_KEY_CAT", "1"):
        return
    if not (torch.cuda.is_available() and k_no_pe.is_cuda and k_pos_emb.is_cuda):
        return

    key_bytes = (
        k_no_pe.numel() * k_no_pe.element_size()
        + k_pos_emb.numel() * k_pos_emb.element_size()
    )
    mib = 1024 * 1024
    margin_bytes = _env_int("MEGATRON_MLA_KEY_CAT_TRIM_MARGIN_MB", 1024) * mib
    cached_threshold = _env_int("MEGATRON_MLA_KEY_CAT_TRIM_CACHED_MB", 512) * mib

    free_bytes, _ = torch.cuda.mem_get_info()
    cached_bytes = max(
        0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    )
    if free_bytes < key_bytes + margin_bytes and cached_bytes > cached_threshold:
        tensor_audit(
            "mla/key_cat_empty_cache",
            k_no_pe=k_no_pe,
            k_pos_emb=k_pos_emb,
            key_bytes_mb=key_bytes // mib,
            free_mb=free_bytes // mib,
            cached_mb=cached_bytes // mib,
        )
        torch.cuda.empty_cache()


from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import split_te_layernorm_column_parallel_linear
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    _yarn_get_mscale,
    apply_rotary_pos_emb,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_audit import tensor_audit
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.streambp import (
    ChunkRange,
    disable_streambp_causal_softmax_fusion,
    make_streambp_packed_seq_params,
    make_streambp_single_sequence_packed_seq_params,
    slice_streambp_attention_mask,
    validate_chunk_range,
)
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params, get_pg_size, is_te_min_version

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        fused_apply_mla_rope_for_kv,
        fused_apply_mla_rope_for_q,
    )
except:
    fused_apply_mla_rope_for_kv = None
    fused_apply_mla_rope_for_q = None


try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELinear,
        set_save_original_input,
    )
    from megatron.core.post_training.modelopt.layers import Linear

    HAVE_TE = True
except ImportError:
    TEColumnParallelLinear, TELinear, Linear, set_save_original_input = None, None, None, None
    HAVE_TE = False


@dataclass
class MLASelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""

    # TODO(nschank): Move layernorms back to the bottom once all other layers have defaults removed.
    q_layernorm: LayerNormBuilder
    kv_layernorm: LayerNormBuilder

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    linear_gate_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None


class MultiLatentAttention(Attention):
    """Multi-Latent Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            pg_collection=pg_collection,
        )
        self.config: MLATransformerConfig

        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads

        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim

        # Overwrite the base class kv shape to support MLA inference
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim

        self.recompute_up_proj = (
            self.config.recompute_granularity == 'selective'
            and "mla_up_proj" in self.config.recompute_modules
        )
        self.qkv_up_checkpoint = None

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale_all_dim)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)
        self.cache_mla_latents = self.config.cache_mla_latents

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == "yarn":

            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are "
                "'rope' and 'yarn'"
            )

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=self.softmax_scale,
            k_channels=self.q_head_dim,
            v_channels=self.config.v_head_dim,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
        )

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
            tp_group=self.pg_collection.tp,
        )

        if (
            HAVE_TE
            and isinstance(self.linear_proj, TELinear)
            and (
                (
                    self.config.fp8
                    and self.config.fp8_recipe != 'delayed'
                    and is_te_min_version("2.6.0dev0")
                )
                or (self.config.fp4 and is_te_min_version("2.7.0.dev0"))
            )
        ):
            # For fp8/fp4 training, the output of the fused core_attn is saved by itself, and
            # linear_proj also saves the quantized tensor of this output. Here we set the
            # linear_proj to save the original input tensors to avoid the extra memory usage of
            # the quantized tensor.
            set_save_original_input(self.linear_proj)

    def _checkpointed_attention_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        rotary_pos_emb=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        dsa_x=None,
        dsa_qr=None,
        dsa_split_qk=None,
        streambp_positions=None,
    ):
        """Forward method with selective activation checkpointing."""

        split_query_pe = None
        split_key_pe = None
        split_kv = None
        if dsa_split_qk is not None:
            split_query_pe, split_key_pe, *split_extra = dsa_split_qk
            split_kv = split_extra[0] if split_extra else None

        streambp_query_positions = None
        streambp_key_positions = None
        if streambp_positions is not None:
            streambp_query_positions, streambp_key_positions = streambp_positions

        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            attention_mask = inputs[3]
            attn_mask_type = inputs[5]
            dsa_x = inputs[6]
            dsa_qr = inputs[7]
            split_query_pe = inputs[8]
            split_key_pe = inputs[9]
            split_kv = inputs[10]
            streambp_query_positions = inputs[11]
            streambp_key_positions = inputs[12]

            attn_mask_type = AttnMaskType(attn_mask_type.item())
            extra_kwargs = {}
            if self.config.experimental_attention_variant == "dsa":
                extra_kwargs["x"] = dsa_x
                extra_kwargs["qr"] = dsa_qr
                if split_query_pe is not None and split_key_pe is not None:
                    dsa_split_qk = (split_query_pe, split_key_pe)
                    if split_kv is not None:
                        dsa_split_qk = (*dsa_split_qk, split_kv)
                    extra_kwargs["dsa_split_qk"] = dsa_split_qk
                if streambp_query_positions is not None and streambp_key_positions is not None:
                    extra_kwargs["streambp_positions"] = (
                        streambp_query_positions,
                        streambp_key_positions,
                    )

            output_ = apply_module(self.core_attention)(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                **extra_kwargs,
            )
            return output_

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb,
            attn_mask_type,
            dsa_x,
            dsa_qr,
            split_query_pe,
            split_key_pe,
            split_kv,
            streambp_query_positions,
            streambp_key_positions,
        )

        return hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        chunk_range: Optional[ChunkRange] = None,
    ):
        """Forward pass for multi-latent attention"""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"
        assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."
        assert not (
            self.training and self.cache_mla_latents
        ), "cache_mla_latents conflicts with training."

        # hidden_states: [sq, b, h]

        inference_context = deprecate_inference_params(inference_context, inference_params)
        streambp_start = streambp_end = streambp_prefix_end = None
        streambp_prefix_packed_seq_params = packed_seq_params
        streambp_core_packed_seq_params = packed_seq_params
        streambp_use_sequence_parallel = False
        streambp_use_sequence_parallel_packed = False
        if chunk_range is not None:
            if key_value_states is not None:
                raise ValueError("StreamBP chunk_range only supports MLA self-attention")
            if inference_context is not None:
                raise ValueError("StreamBP chunk_range is training-only")
            streambp_start, streambp_end = validate_chunk_range(
                chunk_range, hidden_states.size(0)
            )
            streambp_use_sequence_parallel = (
                self.config.sequence_parallel and get_pg_size(self.tp_group) > 1
            )
            streambp_use_sequence_parallel_packed = (
                packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd"
                and streambp_use_sequence_parallel
            )
            if streambp_use_sequence_parallel_packed:
                streambp_prefix_end = hidden_states.size(0)
                streambp_prefix_packed_seq_params = make_streambp_single_sequence_packed_seq_params(
                    packed_seq_params,
                    streambp_prefix_end,
                    streambp_prefix_end,
                )
                streambp_core_packed_seq_params = make_streambp_single_sequence_packed_seq_params(
                    packed_seq_params,
                    streambp_end - streambp_start,
                    streambp_prefix_end,
                )
            else:
                streambp_prefix_end = streambp_end
            if packed_seq_params is not None and not streambp_use_sequence_parallel_packed:
                (
                    streambp_prefix_packed_seq_params,
                    streambp_core_packed_seq_params,
                ) = make_streambp_packed_seq_params(
                    packed_seq_params,
                    streambp_start,
                    streambp_end,
                    kv_end=streambp_prefix_end,
                )
            if not streambp_use_sequence_parallel_packed:
                hidden_states = hidden_states[:streambp_prefix_end]
        if inference_context and not inference_context.is_static_batching():
            assert (
                self.config.cache_mla_latents
            ), "currently to use dynamic backend for MLA cache mla latents must be true"

        if self.config.cache_mla_latents:
            self.prepare_for_absorption()

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        def select_streambp_sequence_parallel_chunk(tensor, start, end, tp_size):
            """Select each TP rank's local StreamBP chunk from an SP-gathered tensor."""

            if tensor.size(0) % tp_size != 0:
                raise ValueError(
                    f"StreamBP sequence-parallel tensor length {tensor.size(0)} is not "
                    f"divisible by TP size {tp_size}"
                )
            prefix_per_rank = tensor.size(0) // tp_size
            if end > prefix_per_rank:
                raise ValueError(
                    f"StreamBP chunk [{start}, {end}) exceeds per-rank prefix "
                    f"length {prefix_per_rank}"
                )
            shape = (tp_size, prefix_per_rank, *tensor.shape[1:])
            return tensor.reshape(shape)[:, start:end].reshape(
                tp_size * (end - start), *tensor.shape[1:]
            )

        def make_streambp_sequence_parallel_positions(start, end, prefix_len, tp_size, device):
            if prefix_len % tp_size != 0:
                raise ValueError(
                    f"StreamBP sequence-parallel prefix length {prefix_len} is not "
                    f"divisible by TP size {tp_size}"
                )
            prefix_per_rank = prefix_len // tp_size
            rank_offsets = torch.arange(tp_size, device=device, dtype=torch.long) * prefix_per_rank
            local_positions = torch.arange(start, end, device=device, dtype=torch.long)
            return (rank_offsets[:, None] + local_positions[None, :]).reshape(-1)

        streambp_query_indices = None
        streambp_sequence_parallel_query_indices = None
        streambp_qkv_query_indices = None
        if chunk_range is not None:
            assert (
                streambp_start is not None
                and streambp_end is not None
                and streambp_prefix_end is not None
            )
            streambp_query_indices = torch.arange(
                streambp_start, streambp_end, device=hidden_states.device, dtype=torch.long
            )
            if streambp_use_sequence_parallel:
                streambp_sequence_parallel_query_indices = make_streambp_sequence_parallel_positions(
                    streambp_start,
                    streambp_end,
                    streambp_prefix_end * get_pg_size(self.tp_group),
                    get_pg_size(self.tp_group),
                    hidden_states.device,
                )
            streambp_qkv_query_indices = streambp_query_indices

        self._dsa_split_qk_parts = None
        with off_interface(self.offload_qkv_linear, hidden_states, "qkv_linear") as hidden_states:
            query, key, value, q_compressed, kv_compressed, gate = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                streambp_prefix_packed_seq_params,
                inference_context=inference_context,
                chunk_range=chunk_range,
                streambp_query_indices=streambp_qkv_query_indices,
            )
            q_compressed_for_dsa = q_compressed
            hidden_states_for_dsa = hidden_states
            streambp_query_positions = None
            streambp_key_positions = None
            if chunk_range is not None:
                assert (
                    streambp_start is not None
                    and streambp_end is not None
                    and streambp_prefix_end is not None
                )
                if streambp_use_sequence_parallel:
                    tp_size = get_pg_size(self.tp_group)

                    def select_streambp_query_tensor(tensor):
                        sequence_parallel_positions = (
                            streambp_sequence_parallel_query_indices.to(device=tensor.device)
                            if streambp_sequence_parallel_query_indices is not None
                            else None
                        )
                        if tensor.size(0) == streambp_query_indices.numel():
                            return (
                                gather_from_sequence_parallel_region(
                                    tensor, group=self.tp_group
                                ),
                                sequence_parallel_positions,
                            )
                        if (
                            sequence_parallel_positions is not None
                            and tensor.size(0) == sequence_parallel_positions.numel()
                        ):
                            return tensor, sequence_parallel_positions
                        if streambp_end <= tensor.size(0):
                            return (
                                gather_from_sequence_parallel_region(
                                    tensor[streambp_start:streambp_end],
                                    group=self.tp_group,
                                ),
                                sequence_parallel_positions,
                            )
                        if tensor.size(0) % tp_size == 0 and streambp_end <= tensor.size(0) // tp_size:
                            selected = select_streambp_sequence_parallel_chunk(
                                tensor, streambp_start, streambp_end, tp_size
                            )
                            positions = make_streambp_sequence_parallel_positions(
                                streambp_start,
                                streambp_end,
                                tensor.size(0),
                                tp_size,
                                tensor.device,
                            )
                            return selected, positions
                        raise ValueError(
                            f"Cannot select StreamBP query chunk [{streambp_start}, {streambp_end}) "
                            f"from tensor with sequence length {tensor.size(0)}"
                        )

                    query, streambp_query_positions = select_streambp_query_tensor(query)
                    if self._dsa_split_qk_parts is not None:
                        split_q_pe, split_k_pe, *split_extra = self._dsa_split_qk_parts
                        split_q_pe, _ = select_streambp_query_tensor(split_q_pe)
                        self._dsa_split_qk_parts = (split_q_pe, split_k_pe, *split_extra)

                    if gate is not None:
                        if gate.size(0) != query.size(0):
                            if gate.size(0) == streambp_query_indices.numel():
                                gate = gather_from_sequence_parallel_region(
                                    gate, group=self.tp_group
                                )
                            elif streambp_end <= gate.size(0):
                                gate = gather_from_sequence_parallel_region(
                                    gate[streambp_start:streambp_end],
                                    group=self.tp_group,
                                )
                            elif gate.size(0) * tp_size == key.size(0):
                                gate = gather_from_sequence_parallel_region(
                                    gate, group=self.tp_group
                                )
                                if gate.size(0) != query.size(0):
                                    gate = select_streambp_sequence_parallel_chunk(
                                        gate, streambp_start, streambp_end, tp_size
                                    )
                            elif (
                                gate.size(0) % tp_size == 0
                                and streambp_end <= gate.size(0) // tp_size
                            ):
                                gate = select_streambp_sequence_parallel_chunk(
                                    gate, streambp_start, streambp_end, tp_size
                                )
                            else:
                                raise ValueError(
                                    f"Cannot align StreamBP gate chunk [{streambp_start}, "
                                    f"{streambp_end}) from tensor with sequence length "
                                    f"{gate.size(0)} to query length {query.size(0)}"
                                )
                    streambp_key_positions = torch.arange(
                        key.size(0), device=key.device, dtype=torch.long
                    )
                    if streambp_use_sequence_parallel_packed:
                        streambp_core_packed_seq_params = (
                            make_streambp_single_sequence_packed_seq_params(
                                packed_seq_params,
                                query.size(0),
                                key.size(0),
                            )
                        )
                else:
                    if query.size(0) != streambp_query_indices.numel():
                        query = query[streambp_start:streambp_end]
                    if self._dsa_split_qk_parts is not None:
                        split_q_pe, split_k_pe, *split_extra = self._dsa_split_qk_parts
                        if split_q_pe.size(0) != query.size(0):
                            split_q_pe = split_q_pe[streambp_start:streambp_end]
                        self._dsa_split_qk_parts = (split_q_pe, split_k_pe, *split_extra)
                    streambp_query_positions = streambp_query_indices.to(device=query.device)
                    streambp_key_positions = torch.arange(
                        streambp_prefix_end,
                        device=query.device,
                        dtype=torch.long,
                    )
                    if gate is not None:
                        if gate.size(0) != query.size(0):
                            gate = gate[streambp_start:streambp_end]
        if self.offload_qkv_linear:
            forced_released_tensors = [hidden_states]
            if self.config.experimental_attention_variant == "dsa":
                # DSA consumes hidden_states_for_dsa after qkv_linear has committed, so
                # releasing this storage here can invalidate the indexer key projection.
                forced_released_tensors = []
            query = off_interface.group_commit(
                query, name="qkv_linear", forced_released_tensors=forced_released_tensors
            )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        query, key, value, _, attn_mask_type, block_table = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        if chunk_range is not None:
            assert (
                streambp_start is not None
                and streambp_end is not None
                and streambp_prefix_end is not None
            )
            attention_mask = slice_streambp_attention_mask(
                attention_mask,
                streambp_start,
                streambp_end,
                streambp_prefix_end,
                device=query.device,
                causal=attn_mask_type == AttnMaskType.causal,
            )

        # TODO: Currently, TE can only accept contiguous tensors for MLA.
        # Split-QK DSA can consume strided Q/K-noPE directly from packed MLA
        # projections, which avoids retaining per-replay contiguous copies in
        # StreamBP attention graphs.
        if self.config.experimental_attention_variant != "dsa" or self._dsa_split_qk_parts is None:
            query = query.contiguous()
        if self.config.experimental_attention_variant != "dsa" or self._dsa_split_qk_parts is None:
            key = key.contiguous()

        # Value is none during decode for absorption
        if value is not None:
            if self.config.experimental_attention_variant != "dsa":
                value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        extra_kwargs = {}
        if self.config.experimental_attention_variant == "dsa":
            # DSA needs the original hidden states and compressed query representation.
            extra_kwargs["x"] = hidden_states_for_dsa
            extra_kwargs["qr"] = q_compressed_for_dsa
            if self._dsa_split_qk_parts is not None:
                extra_kwargs["dsa_split_qk"] = self._dsa_split_qk_parts
            if chunk_range is not None:
                extra_kwargs["streambp_positions"] = (
                    streambp_query_positions,
                    streambp_key_positions,
                )
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=streambp_core_packed_seq_params,
                dsa_x=extra_kwargs.get("x"),
                dsa_qr=extra_kwargs.get("qr"),
                dsa_split_qk=extra_kwargs.get("dsa_split_qk"),
                streambp_positions=extra_kwargs.get("streambp_positions"),
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                with off_interface(
                    self.offload_core_attention and self.training, query, "core_attn"
                ) as query:
                    core_attention_context = (
                        disable_streambp_causal_softmax_fusion(self.core_attention)
                        if chunk_range is not None
                        else nullcontext()
                    )
                    with core_attention_context:
                        core_attn_out = self.core_attention(
                            query,
                            key,
                            value,
                            attention_mask,
                            packed_seq_params=streambp_core_packed_seq_params,
                            attn_mask_type=attn_mask_type,
                            **extra_kwargs,
                        )
            elif self.cache_mla_latents:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                )
                # Only rearrange if not in absorption mode (Flash MLA handles format correctly)
                if not inference_context.is_decode_only():
                    core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')
            dsa_release_tensors = []
            if self.config.experimental_attention_variant == "dsa":
                dsa_release_tensors.extend(
                    [
                        hidden_states_for_dsa,
                        q_compressed_for_dsa,
                        q_compressed,
                        kv_compressed,
                    ]
                )
                if self._dsa_split_qk_parts is not None:
                    dsa_release_tensors.extend(self._dsa_split_qk_parts)
            if self.offload_core_attention and self.training:
                core_attn_out = off_interface.group_commit(
                    core_attn_out,
                    name="core_attn",
                    forced_released_tensors=[query, key, value] + dsa_release_tensors,
                )
            if self.config.experimental_attention_variant == "dsa":
                hidden_states_for_dsa = None
                q_compressed_for_dsa = None
                q_compressed = None
                kv_compressed = None
                self._dsa_split_qk_parts = None
            query = None
            key = None
            value = None

        # We are doing absorption with cache mla latents and decode mode.
        if self.cache_mla_latents and inference_context.is_decode_only():
            # core_attn_out = self.self.up_v_layer(core_attn_out)
            core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
            core_attn_out = core_attn_out.contiguous()

            # Flatten back: [seq, batch, num_heads * v_head_dim]
            core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        #NOTE: Paper-faithful G1 placement for MLA-family attention. Standard SelfAttention
        # applies this in attention.py, but MLA owns its own forward path. Placing the gate
        # here covers normal MLA, DSA (DSAttention is MLA's core_attention), and FlashMLA
        # after any decode-only up-projection/flattening, while still gating before Wo.
        # Old behavior was:
        #   core_attn_out -> linear_proj
        # New behavior when attention_output_gate is enabled:
        #   core_attn_out -> G1 gate -> linear_proj
        if gate is not None:
            core_attn_out = self._apply_output_gate(core_attn_out, gate)

        # =================
        # Output. [sq, b, h]
        # =================
        with off_interface(self.offload_attn_proj, core_attn_out, "attn_proj") as core_attn_out:
            output, bias = self.linear_proj(core_attn_out)
        if self.offload_attn_proj:
            output = off_interface.group_commit(
                output, name="attn_proj", forced_released_tensors=[core_attn_out]
            )

        return output, bias


class MLASelfAttention(MultiLatentAttention):
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        if self.config.q_lora_rank is None:
            # Not projecting query
            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_proj',
            )

        else:
            q_down_proj_kwargs = {}
            if submodules.linear_q_down_proj in [TELinear]:
                q_down_proj_kwargs['parallel_mode'] = 'duplicated'
            elif submodules.linear_q_down_proj in [
                Linear,
                TEColumnParallelLinear,
                ColumnParallelLinear,
            ]:
                q_down_proj_kwargs['gather_output'] = False
            else:
                raise ValueError(f"Unsupported linear_q_down_proj: {submodules.linear_q_down_proj}")

            self.linear_q_down_proj = build_module(
                submodules.linear_q_down_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_down_proj',
                skip_weight_param_allocation=False,
                tp_group=(
                    pg_collection.tp
                    if q_down_proj_kwargs.get('parallel_mode') != 'duplicated'
                    else None
                ),
                **q_down_proj_kwargs,
            )

            self.linear_q_up_proj = build_module(
                submodules.linear_q_up_proj,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='q_up_proj',
                tp_group=pg_collection.tp,
            )

        kv_down_proj_kwargs = {}
        if submodules.linear_kv_down_proj in [TELinear]:
            kv_down_proj_kwargs['parallel_mode'] = 'duplicated'
        elif submodules.linear_kv_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ]:
            kv_down_proj_kwargs['gather_output'] = False
        else:
            raise ValueError(f"Unsupported linear_kv_down_proj: {submodules.linear_kv_down_proj}")

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_down_proj',
            skip_weight_param_allocation=False,
            tp_group=(
                pg_collection.tp
                if kv_down_proj_kwargs.get('parallel_mode') != 'duplicated'
                else None
            ),
            **kv_down_proj_kwargs,
        )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_up_proj',
            tp_group=pg_collection.tp,
        )
        if HAVE_TE and self.config.fp4 and is_te_min_version("2.7.0.dev0"):
            # StreamBP + split-QK DSA can re-enter this TE Linear backward for
            # value/key chunks. Reusing TE's saved NVFP4 input object anywhere
            # on the retained K/V projection graph can leave a later WGRAD GEMM
            # without a valid amax pointer, so keep BF16 inputs for the MLA
            # K/V projections that sit behind the reentrant DSA key/value refs.
            set_save_original_input(self.linear_kv_down_proj)
            set_save_original_input(self.linear_kv_up_proj)

        #NOTE: MLA does not use Attention.forward(), so the standard G1 gate in attention.py
        # does not cover MLA, DSA, or FlashMLA. Build a separate gate projection here so
        # MultiLatentAttention.forward() can apply the same paper placement:
        # attention output -> G1 gate -> Wo.
        if self.config.attention_output_gate:
            self.linear_gate_proj = build_module(
                submodules.linear_gate_proj,
                self.config.hidden_size,
                self.query_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='gate_proj',
                tp_group=pg_collection.tp,
            )
        else:
            self.linear_gate_proj = None

        if self.config.q_lora_rank is not None:
            self.q_layernorm = submodules.q_layernorm(
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_layernorm = submodules.kv_layernorm(
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.turboquant_kv_buffers = None
        if getattr(self.config, "turboquant_kv_enabled", False):
            from megatron.core.quantization.turboquant import build_turboquant_buffers

            buffers = build_turboquant_buffers(
                latent_dim=self.config.kv_lora_rank,
                preset=getattr(self.config, "turboquant_kv_preset", "latent_2p5bit_nc"),
                seed=getattr(self.config, "turboquant_kv_seed", 0),
                layer_idx=layer_number,
                device="cpu",
                dtype=torch.float32,
            )
            self.register_buffer("_turboquant_signs1", buffers.signs1, persistent=False)
            self.register_buffer("_turboquant_signs2", buffers.signs2, persistent=False)
            self.register_buffer(
                "_turboquant_boundaries_high", buffers.boundaries_high, persistent=False
            )
            self.register_buffer(
                "_turboquant_boundaries_low", buffers.boundaries_low, persistent=False
            )
            self.register_buffer(
                "_turboquant_centroids_high", buffers.centroids_high, persistent=False
            )
            self.register_buffer(
                "_turboquant_centroids_low", buffers.centroids_low, persistent=False
            )
            self.turboquant_kv_buffers = buffers

        self.higgs_kv_buffers = None
        if getattr(self.config, "enable_higgs_dense_2bit_kv_cache", False):
            from megatron.core.quantization.higgs import build_higgs_buffers

            higgs_buffers = build_higgs_buffers(
                latent_dim=self.config.kv_lora_rank,
                preset=getattr(self.config, "higgs_kv_preset", "dense_2bit"),
                layer_idx=layer_number,
                device="cpu",
                dtype=torch.float32,
            )
            self.register_buffer(
                "_higgs_codebook", higgs_buffers.codebook, persistent=False
            )
            self.register_buffer(
                "_higgs_codebook_norm_sq",
                higgs_buffers.codebook_norm_sq,
                persistent=False,
            )
            self.higgs_kv_buffers = higgs_buffers

    def _refresh_turboquant_buffers_device(self) -> None:
        """Sync the cached buffer dataclass with the registered tensors."""

        if self.turboquant_kv_buffers is None:
            return
        from megatron.core.quantization.turboquant import TurboQuantBuffers

        self.turboquant_kv_buffers = TurboQuantBuffers(
            latent_dim=self.turboquant_kv_buffers.latent_dim,
            bits=self.turboquant_kv_buffers.bits,
            norm_correction=self.turboquant_kv_buffers.norm_correction,
            signs1=self._turboquant_signs1,
            signs2=self._turboquant_signs2,
            boundaries_high=self._turboquant_boundaries_high,
            boundaries_low=self._turboquant_boundaries_low,
            centroids_high=self._turboquant_centroids_high,
            centroids_low=self._turboquant_centroids_low,
        )

    def _refresh_higgs_buffers_device(self) -> None:
        """Sync the cached HIGGS buffer dataclass with the registered tensors."""

        if self.higgs_kv_buffers is None:
            return
        from megatron.core.quantization.higgs import HiggsBuffers

        self.higgs_kv_buffers = HiggsBuffers(
            latent_dim=self.higgs_kv_buffers.latent_dim,
            pair_dim=self.higgs_kv_buffers.pair_dim,
            codebook_size=self.higgs_kv_buffers.codebook_size,
            bits_per_scalar=self.higgs_kv_buffers.bits_per_scalar,
            codebook=self._higgs_codebook,
            codebook_norm_sq=self._higgs_codebook_norm_sq,
        )

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
        chunk_range: Optional[ChunkRange] = None,
        streambp_query_indices: Optional[torch.Tensor] = None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"
        if packed_seq_params is not None:
            assert (
                packed_seq_params.local_cp_size is None
            ), "hybrid_context_parallel is not supported with MLA yet and is planned for future. \
            Please disable hybrid_context_parallel."

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if streambp_query_indices is not None:
            streambp_query_indices = streambp_query_indices.to(
                device=hidden_states.device, dtype=torch.long
            )
            if streambp_query_indices.dim() != 1:
                raise ValueError(
                    "streambp_query_indices must be 1D, got "
                    f"{tuple(streambp_query_indices.shape)}"
                )
            if bool((streambp_query_indices < 0).any().item()) or bool(
                (streambp_query_indices >= hidden_states.size(0)).any().item()
            ):
                raise ValueError(
                    f"streambp_query_indices out of range for sequence length {hidden_states.size(0)}"
                )

        gate = None
        if self.config.attention_output_gate:
            #NOTE: G1 needs a query-shaped, head-specific gate for the final attention output.
            # Keep it outside MLA's Q/KV low-rank projections so the old MLA projection flow
            # remains easy to restore: remove this block and the later gate application.
            gate_input = hidden_states
            if streambp_query_indices is not None:
                gate_input = hidden_states.index_select(0, streambp_query_indices)
            gate, _ = self.linear_gate_proj(gate_input)
            gate = gate.view(
                *gate.size()[:-1],
                self.num_attention_heads_per_partition,
                self.config.v_head_dim,
            )

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb:[s, b, 1, 64]
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        else:
            if self.config.apply_rope_fusion and chunk_range is None:
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                    rotary_seq_len, dtype=hidden_states.dtype, packed_seq=packed_seq
                )
                rotary_pos_emb = None
                assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
                assert (
                    fused_apply_mla_rope_for_q is not None
                    and fused_apply_mla_rope_for_kv is not None
                ), "Fused MLA RoPE apply is not imported successfully"
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            # if linear_q_down_proj is ColumnParallelLinear:
            #     q_compressed: [s, b, q_lora_rank / TP]
            # elif linear_q_down_proj is Linear:
            #     q_compressed: [s / TP, b, q_lora_rank]
            q_down_input = hidden_states
            if streambp_query_indices is not None:
                q_down_input = hidden_states.index_select(
                    0, streambp_query_indices.to(device=hidden_states.device)
                )
            q_compressed, _ = self.linear_q_down_proj(q_down_input)

            # When output is sharded (ColumnParallelLinear), two things are needed to be
            # identical to a normal Linear.
            #   1. Manually gather output to restore output dim q_lora_rank;
            #   2. Scatter sequence back to s / TP if sequence-parallel since it was
            #      gathered by ColumnParallelLinear.
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)
        else:
            q_compressed = hidden_states

        # if linear_kv_down_proj is ColumnParallelLinear:
        #     kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim) / TP]
        # elif linear_kv_down_proj is Linear:
        #     kv_combined: [s / TP, b, (kv_lora_rank + qk_pos_emb_head_dim)]
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[s, b, kv_lora_rank], k_pos_emb: [s, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if self.config.sequence_parallel:
                # kv_compressed:[s / TP, b, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if get_pg_size(self.tp_group) > 1 and self.config.sequence_parallel:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb, group=self.tp_group)

        if packed_seq_params is not None:
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = apply_module(self.q_layernorm)(q_compressed)

        kv_compressed = apply_module(self.kv_layernorm)(kv_compressed)

        if self.turboquant_kv_buffers is not None:
            from megatron.core.quantization.turboquant import apply_turboquant_kv

            self._refresh_turboquant_buffers_device()
            kv_compressed = apply_turboquant_kv(kv_compressed, self.turboquant_kv_buffers)

        if self.higgs_kv_buffers is not None:
            from megatron.core.quantization.higgs import apply_higgs_dense_2bit_kv

            self._refresh_higgs_buffers_device()
            kv_compressed = apply_higgs_dense_2bit_kv(
                kv_compressed, self.higgs_kv_buffers
            )

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply_for_cached_latent_kv(
            q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
        ):
            if self.config.q_lora_rank is not None:
                # q_compressed: [num_tokens, q_lora_rank]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                # q_compressed: [num_tokens, hidden_size]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            q_no_pe, q_pos_emb = torch.split(
                q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
            )

            # Dynamic batching: use inference context methods
            q_pos_emb = inference_context.apply_rotary_emb_query(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens_q=cu_seqlens_q,
                cp_group=self.pg_collection.cp,
                mscale=mscale,
            )
            # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = inference_context.apply_rotary_emb_key(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cp_group=self.pg_collection.cp,
                mscale=mscale,
            )

            # Create KV cache entry. It will the be the key vector in cache mla latents path
            k_pos_emb_squeezed = k_pos_emb.squeeze(1)
            kv_cached = torch.cat([kv_compressed, k_pos_emb_squeezed], dim=-1)

            # Flag for whether to use absorption. We only use absorption
            # when caching the latents and in decode-only mode
            use_absorption = (
                self.config.cache_mla_latents
                and inference_context
                and inference_context.is_decode_only()
            )
            # Compute query components. Multiply by up k if absorbing
            q_content = (
                torch.einsum("sbhd,hdk->sbhk", q_no_pe, self.up_k_weight)
                if use_absorption
                else q_no_pe
            )
            # Query: content + original positional (latent_dim + pos_dim)
            query = torch.cat([q_content, q_pos_emb], dim=-1)

            key = kv_cached
            value = None

            query = query.contiguous()
            key = key.contiguous()

            return query, key, value

        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [s, b, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [s, b, ...] or [t, ...] for two cases.
            """
            rope_mscale = mscale
            q_streambp_indices = None
            q_streambp_slice = None
            q_up_input = q_compressed
            if streambp_query_indices is not None:
                q_indices = streambp_query_indices.to(device=q_up_input.device, dtype=torch.long)
                if q_indices.numel() == 0:
                    raise ValueError("streambp_query_indices must not be empty")
                if q_up_input.size(0) == q_indices.numel():
                    q_streambp_indices = q_indices
                elif not bool((q_indices < 0).any().item()) and not bool(
                    (q_indices >= q_up_input.size(0)).any().item()
                ):
                    q_up_input = q_up_input.index_select(0, q_indices)
                    q_streambp_indices = q_indices
                else:
                    if chunk_range is None:
                        raise ValueError("StreamBP q projection selection requires chunk_range")
                    start, end = chunk_range
                    if end > q_up_input.size(0):
                        raise ValueError(
                            f"StreamBP q slice [{start}, {end}) exceeds q length "
                            f"{q_up_input.size(0)}"
                        )
                    q_up_input = q_up_input[start:end]
                    q_streambp_slice = (start, end)

            if self.config.q_lora_rank is not None:
                # q_compressed: [num_tokens, q_lora_rank]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_up_proj(q_up_input)
            else:
                # q_compressed: [num_tokens, hidden_size]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_proj(q_up_input)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
            if streambp_query_indices is not None and q.size(0) != streambp_query_indices.numel():
                local_query_len = streambp_query_indices.numel()
                if local_query_len == 0 or q.size(0) % local_query_len != 0:
                    raise ValueError(
                        f"Cannot map StreamBP q length {q.size(0)} to local query length "
                        f"{local_query_len}"
                    )
                if chunk_range is None:
                    raise ValueError("StreamBP gathered q projection requires chunk_range")
                start, end = chunk_range
                gathered_tp_size = q.size(0) // local_query_len
                prefix_per_rank = hidden_states.size(0)
                if end > prefix_per_rank:
                    raise ValueError(
                        f"StreamBP chunk [{start}, {end}) exceeds prefix length "
                        f"{prefix_per_rank}"
                    )
                rank_offsets = (
                    torch.arange(gathered_tp_size, device=q.device, dtype=torch.long)
                    * prefix_per_rank
                )
                local_positions = torch.arange(start, end, device=q.device, dtype=torch.long)
                q_streambp_indices = (
                    rank_offsets[:, None] + local_positions[None, :]
                ).reshape(-1)
                q_streambp_slice = None
            if (
                streambp_query_indices is not None
                and q_streambp_indices is None
                and q_streambp_slice is None
            ):
                q_indices = streambp_query_indices.to(device=q.device, dtype=torch.long)
                if bool((q_indices < 0).any().item()) or bool(
                    (q_indices >= q.size(0)).any().item()
                ):
                    if chunk_range is None:
                        raise ValueError("StreamBP q projection selection requires chunk_range")
                    start, end = chunk_range
                    if end > q.size(0):
                        raise ValueError(
                            f"StreamBP q slice [{start}, {end}) exceeds q length {q.size(0)}"
                        )
                    q = q[start:end]
                    q_streambp_slice = (start, end)
                else:
                    q = q.index_select(0, q_indices)
                    q_streambp_indices = q_indices

            # kv: [num_tokens, n * (qk_head_dim + v_head_dim)]
            kv, _ = self.linear_kv_up_proj(kv_compressed)

            # kv: [num_tokens, n, (qk_head_dim + v_head_dim)]
            kv = kv.view(
                *kv.size()[:-1],
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            def slice_rotary_pos_emb(pos_emb, start, end):
                if pos_emb is None:
                    return None
                if isinstance(pos_emb, tuple):
                    return tuple(None if item is None else item[start:end] for item in pos_emb)
                return pos_emb[start:end]

            def index_rotary_pos_emb(pos_emb, indices):
                if pos_emb is None:
                    return None
                if isinstance(pos_emb, tuple):
                    return tuple(
                        None if item is None else item.index_select(0, indices.to(item.device))
                        for item in pos_emb
                    )
                return pos_emb.index_select(0, indices.to(pos_emb.device))

            # The fused MLA RoPE kernels consume cached cos/sin tensors. StreamBP
            # chunked replay intentionally uses the generic rotary embedding path
            # so query chunks can use chunk-shaped RoPE while keys keep prefix RoPE.
            use_mla_rope_fusion = (
                self.config.apply_rope_fusion
                and rotary_pos_cos is not None
                and rotary_pos_sin is not None
            )
            if use_mla_rope_fusion:
                cp_rank = self.pg_collection.cp.rank()
                cp_size = self.pg_collection.cp.size()
                query = fused_apply_mla_rope_for_q(
                    q,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_head_dim,
                    self.config.qk_pos_emb_head_dim,
                    cu_seqlens_q,
                    cp_rank,
                    cp_size,
                )
                key, value = fused_apply_mla_rope_for_kv(
                    kv,
                    k_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_pos_emb_head_dim,
                    self.config.qk_head_dim,
                    self.config.v_head_dim,
                    cu_seqlens_kv,
                    cp_rank,
                    cp_size,
                )
            else:
                q_len = q.size()[0]
                if inference_context is not None:
                    # add offset to the sequence start for inference
                    sequence_start = inference_context.sequence_len_offset
                    sequence_end = sequence_start + q_len
                    rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]

                # q_no_pe: [num_tokens, n, qk_head_dim]
                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_no_pe, q_pos_emb = torch.split(
                    q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                )

                # k_no_pe: [num_tokens, n, qk_head_dim]
                # value: [num_tokens, n, v_head_dim]
                k_no_pe, value = torch.split(
                    kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
                )

                rope_cu_seqlens_q = cu_seqlens_q
                rope_cu_seqlens_kv = cu_seqlens_kv
                if (
                    chunk_range is not None
                    and packed_seq_params is not None
                    and packed_seq_params.qkv_format == "thd"
                ):

                    def reconcile_rope_cu(cu_seqlens, target_len):
                        if cu_seqlens is None:
                            return None
                        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
                        if int(seqlens.sum().item()) == target_len:
                            return cu_seqlens
                        return torch.tensor(
                            [0, target_len],
                            dtype=cu_seqlens.dtype,
                            device=cu_seqlens.device,
                        )

                    rope_cu_seqlens_q = reconcile_rope_cu(
                        rope_cu_seqlens_q, q_pos_emb.size(0)
                    )
                    rope_cu_seqlens_kv = reconcile_rope_cu(
                        rope_cu_seqlens_kv, k_pos_emb.size(0)
                    )
                    required_rotary_seq_len = 0
                    if rope_cu_seqlens_q is not None:
                        required_rotary_seq_len = max(
                            required_rotary_seq_len, int(rope_cu_seqlens_q[-1].item())
                        )
                    if rope_cu_seqlens_kv is not None:
                        required_rotary_seq_len = max(
                            required_rotary_seq_len, int(rope_cu_seqlens_kv[-1].item())
                        )
                    if required_rotary_seq_len:
                        current_rotary_seq_len = (
                            rotary_pos_emb[0].size(0)
                            if isinstance(rotary_pos_emb, tuple)
                            else rotary_pos_emb.size(0)
                        )
                        if current_rotary_seq_len < required_rotary_seq_len:
                            if self.config.rope_type == "rope":
                                rotary_pos_emb = self.rotary_pos_emb(
                                    required_rotary_seq_len, packed_seq=True
                                )
                            else:
                                rotary_pos_emb, rope_mscale = self.rotary_pos_emb(
                                    required_rotary_seq_len, packed_seq=True
                                )

                query_rotary_pos_emb = rotary_pos_emb
                key_rotary_pos_emb = rotary_pos_emb
                if inference_context is None and (
                    packed_seq_params is None or self.config.context_parallel_size == 1
                ):
                    # StreamBP replays a query chunk against the full causal prefix. Keep
                    # query RoPE chunk-shaped, but keep key RoPE prefix-shaped.
                    if q_streambp_indices is not None:
                        query_rotary_pos_emb = index_rotary_pos_emb(
                            rotary_pos_emb, q_streambp_indices
                        )
                        key_rotary_pos_emb = slice_rotary_pos_emb(
                            rotary_pos_emb, 0, k_pos_emb.size(0)
                        )
                    elif q_streambp_slice is not None:
                        start, end = q_streambp_slice
                        query_rotary_pos_emb = slice_rotary_pos_emb(rotary_pos_emb, start, end)
                        key_rotary_pos_emb = slice_rotary_pos_emb(
                            rotary_pos_emb, 0, k_pos_emb.size(0)
                        )
                    else:
                        rotary_pos_emb = slice_rotary_pos_emb(rotary_pos_emb, 0, q_len)
                        query_rotary_pos_emb = rotary_pos_emb
                        key_rotary_pos_emb = rotary_pos_emb

                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_pos_emb = apply_rotary_pos_emb(
                    q_pos_emb,
                    query_rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=rope_cu_seqlens_q,
                    mscale=rope_mscale,
                    cp_group=self.pg_collection.cp,
                    mla_rotary_interleaved=True,
                )
                # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
                k_pos_emb = apply_rotary_pos_emb(
                    k_pos_emb,
                    key_rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=rope_cu_seqlens_kv,
                    mscale=rope_mscale,
                    cp_group=self.pg_collection.cp,
                    mla_rotary_interleaved=True,
                )

                use_split_dsa_qk = (
                    self.training
                    and inference_context is None
                    and self.config.experimental_attention_variant == "dsa"
                    and _env_flag_enabled("MEGATRON_DSA_SPLIT_QK", "1")
                )
                if use_split_dsa_qk:
                    self._dsa_split_qk_parts = (q_pos_emb, k_pos_emb, kv)
                    query = q_no_pe
                    key = k_no_pe
                    tensor_audit(
                        "mla/qkv_split_inputs",
                        q_no_pe=q_no_pe,
                        q_pos_emb=q_pos_emb,
                        k_no_pe=k_no_pe,
                        k_pos_emb=k_pos_emb,
                        streambp_slice=q_streambp_slice,
                        streambp_indices=(
                            None
                            if q_streambp_indices is None
                            else tuple(q_streambp_indices.shape)
                        ),
                    )
                else:
                    # query: [num_tokens, n, (qk_head_dim + v_head_dim)]
                    query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

                    # key: [num_tokens, n, (qk_head_dim + v_head_dim)]
                    if k_pos_emb.ndim == 4:
                        k_pos_emb = k_pos_emb.expand(
                            -1, -1, self.num_attention_heads_per_partition, -1
                        )
                    else:
                        assert k_pos_emb.ndim == 3
                        k_pos_emb = k_pos_emb.expand(
                            -1, self.num_attention_heads_per_partition, -1
                        )
                    tensor_audit(
                        "mla/qkv_cat_inputs",
                        q_no_pe=q_no_pe,
                        q_pos_emb=q_pos_emb,
                        k_no_pe=k_no_pe,
                        k_pos_emb=k_pos_emb,
                        streambp_slice=q_streambp_slice,
                        streambp_indices=(
                            None
                            if q_streambp_indices is None
                            else tuple(q_streambp_indices.shape)
                        ),
                    )
                    _maybe_trim_cuda_cache_before_mla_key_cat(k_no_pe, k_pos_emb)
                    key = torch.cat([k_no_pe, k_pos_emb], dim=-1)
                    tensor_audit("mla/key_cat_output", key=key)

            # Split-QK DSA kernels consume strided noPE views directly as long as
            # the innermost dimension is contiguous. Avoid materializing full
            # query/key noPE copies on the DSA path; those copies are especially
            # expensive at 32k context and during replay/profile passes.
            if (
                self.config.experimental_attention_variant != "dsa"
                or self._dsa_split_qk_parts is None
            ):
                query = query.contiguous()
                key = key.contiguous()
            else:
                if query.stride(-1) != 1:
                    query = query.contiguous()
                if key.stride(-1) != 1:
                    key = key.contiguous()
            if self.config.experimental_attention_variant != "dsa":
                value = value.contiguous()

            return query, key, value

        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )
        else:
            if self.cache_mla_latents:
                assert (
                    inference_context and not inference_context.is_static_batching()
                ), "Caching MLA latents only works with dynamic backend inference"
                query, key, value = qkv_up_proj_and_rope_apply_for_cached_latent_kv(
                    q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
                )
            else:
                query, key, value = qkv_up_proj_and_rope_apply(
                    q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
                )

        return query, key, value, q_compressed, kv_compressed, gate

    def uncompress_kv_from_cache(self, kv_cached):
        """
        Take a compressed kv and uncompress them
        """
        kv_compressed, k_pos_emb = torch.split(
            kv_cached, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )

        # Seperated out the norm and linear
        kv, _ = self.linear_kv_up_proj_linear(kv_compressed)

        kv = kv.view(
            *kv.size()[:-1],
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
        )

        k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

        # Add head dimension
        k_pos_emb = k_pos_emb.unsqueeze(-2)
        k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)

        key = torch.cat([k_no_pe, k_pos_emb], dim=-1)
        return key, value

    def prepare_for_absorption(self):
        """Prepare the model for absorption optimization in MLA (Multi-Latent Attention).

        This method sets up the necessary components for the absorption technique, which
        optimizes memory during inference by caching compressed KV latents instead
        of full KV states. The absorption technique allows efficient decode-only operations
        by pre-computing certain matrix multiplications.

        Note (Peter): Right now we are not doing true absorption. We will add this support
        at a later time.

        The method performs the following operations:
        1. Splits the fused layernorm + linear layer (linear_kv_up_proj) into separate
        components.
        2. Extracts and stores the up-projection weights for K and V separately, which
        are used during the absorption process
        3. Replaces the identity kv_layernorm with the actual layernorm from the split
        4. Stores the linear component separately for uncompressing KV cache during
        prefill/mixed stages

        This is a one-time setup that should only be called once at initialization when
        cache_mla_latents is enabled.
        """
        # We should only have to call to set once at start
        if not hasattr(self, "up_k_weight"):
            with torch.no_grad():
                linear_kv_up_proj_norm, linear_kv_up_proj_linear = (
                    split_te_layernorm_column_parallel_linear(
                        self.linear_kv_up_proj, self.config, None, self.linear_kv_up_proj.tp_group
                    )
                )

                # Note: When caching latents we overide the kv_layernorm
                # which was an identity before because in the is path
                # we unfused the linear_kv_up_proj
                self.kv_layernorm = linear_kv_up_proj_norm

                # This is used in absorption when we are
                # uncompressing the KV cache in prefill/mixed stages
                self.linear_kv_up_proj_linear = linear_kv_up_proj_linear

                kv_up_weight = (
                    self.linear_kv_up_proj.weight
                )  # [num_heads * (qk_head_dim + v_head_dim), kv_lora_rank]
                kv_up_weight = kv_up_weight.view(
                    self.num_attention_heads_per_partition,
                    self.config.qk_head_dim + self.config.v_head_dim,
                    self.config.kv_lora_rank,
                )
                # Split into K and V up-projection weights. These are used for absorption
                self.up_k_weight = kv_up_weight[
                    :, : self.config.qk_head_dim, :
                ]  # [num_heads, qk_head_dim, kv_lora_rank]
                self.up_v_weight = kv_up_weight[
                    :, self.config.qk_head_dim :, :
                ]  # [num_heads, v_head_dim, kv_lora_rank]

                # We delete the original linear_kv_up_proj as we do not
                # need it for the absorbed path.
                del self.linear_kv_up_proj

    def backward_dw(self) -> NoReturn:
        """Execute weight gradient computation"""
        self._backward_kv_proj()
        self._backward_q_proj()
        self._backward_gate_proj()
        self._backward_output_proj()

    def _backward_kv_proj(self):
        """Computes weight gradients of KV projection layers"""
        self.linear_kv_up_proj.backward_dw()
        self.linear_kv_down_proj.backward_dw()

    def _backward_q_proj(self):
        """Computes weight gradients of Q projection layers"""
        if self.config.q_lora_rank is None:
            self.linear_q_proj.backward_dw()
        else:
            self.linear_q_down_proj.backward_dw()
            self.linear_q_up_proj.backward_dw()

    def _backward_gate_proj(self):
        """Computes weight gradients of the optional MLA G1 gate projection."""
        #NOTE: The old MLA path had no gate projection. Keep this isolated so rollback is
        # just removing the G1 gate build/apply path without touching Q/KV/WO gradients.
        if self.linear_gate_proj is not None:
            self.linear_gate_proj.backward_dw()

    def _backward_output_proj(self):
        """Computes weight gradients of output projection layer"""
        self.linear_proj.backward_dw()

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8/fp4."""
        from megatron.core.extensions.transformer_engine import set_save_original_input

        if self.config.q_lora_rank is not None:
            set_save_original_input(self.linear_q_down_proj)
        set_save_original_input(self.linear_kv_down_proj)

    def clip_qk(self):
        """
        QK Clipping is a technique to clip the query and key attention logits to prevent the
        attention logits from exploding. Per MuonClip usage, we update the weight by calling this
        function after Muon optimizer step.
        """

        if not self.config.qk_clip:
            raise ValueError("qk_clip option needs to be enabled")

        if self.core_attention.current_max_attn_logits is None:
            raise ValueError("current_max_attn_logits is None")

        # Check if we're in absorption mode
        if self.cache_mla_latents and not hasattr(self, 'linear_kv_up_proj'):
            raise ValueError(
                "qk_clip is not supported when cache_mla_latents is enabled and absorption is "
                "active. The linear_kv_up_proj layer has been deleted during absorption "
                "preparation."
            )

        assert self.core_attention.current_max_attn_logits.shape == (
            self.num_attention_heads_per_partition,
        ), f"current_max_attn_logits shape is not ({self.num_attention_heads_per_partition}, ) \
                    but {self.core_attention.current_max_attn_logits.shape}"

        # only update the weight if any head has
        # current_max_attn_logits > qk_clip_threshold
        if torch.any(self.core_attention.current_max_attn_logits > self.config.qk_clip_threshold):
            # Use num_attention_heads_per_partition for tensor parallel scenarios

            # qk_clip_balancing_eta (n, 1, 1)
            assert self.core_attention.current_max_attn_logits.shape == (
                self.num_attention_heads_per_partition,
            ), f"current_max_attn_logits shape is not ({self.num_attention_heads_per_partition},) \
                but {self.core_attention.current_max_attn_logits.shape}"
            self.qk_clip_balancing_eta = torch.clamp(
                self.config.qk_clip_threshold / self.core_attention.current_max_attn_logits, max=1.0
            ).view(self.num_attention_heads_per_partition, 1, 1)
            assert torch.all(self.qk_clip_balancing_eta <= 1.0)

            # Update q side weight, keep qk_pos_emb_head_dim side weight unchanged
            if self.config.q_lora_rank is None:
                q_proj_weight = self.linear_q_proj.weight
            else:
                q_proj_weight = self.linear_q_up_proj.weight

            # Handle different weight access patterns (main_param vs direct access)
            if hasattr(q_proj_weight, 'main_param'):
                q_proj_weight.main_param.data.copy_(
                    self._clip_q_proj_weight(q_proj_weight.main_param.data)
                )
            q_proj_weight.data.copy_(self._clip_q_proj_weight(q_proj_weight.data))

            # Update k side weight, keep v side weight unchanged
            kv_proj_weight = self.linear_kv_up_proj.weight

            # Handle different weight access patterns
            if hasattr(kv_proj_weight, 'main_param'):
                kv_proj_weight.main_param.data.copy_(
                    self._clip_kv_proj_weight(kv_proj_weight.main_param.data)
                )
            kv_proj_weight.data.copy_(self._clip_kv_proj_weight(kv_proj_weight.data))

        # reset current_max_attn_logits
        self.core_attention.current_max_attn_logits = None

    def _clip_q_proj_weight(self, weight):
        """Clip q_proj_weight"""
        # Reshape to (n, a + b, -1)
        weight_reshaped = weight.view(
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.qk_pos_emb_head_dim,
            -1,
        )

        # Split into qk_head_dim and qk_pos_emb_head_dim parts: (n, a, -1) and (n, b, -1)
        weight_q_nope = weight_reshaped[:, : self.config.qk_head_dim, :]
        weight_q_pe = weight_reshaped[:, self.config.qk_head_dim :, :]

        # Clipping
        weight_q_nope.mul_(torch.pow(self.qk_clip_balancing_eta, self.config.qk_clip_alpha))
        weight_q_pe.mul_(self.qk_clip_balancing_eta)

        # Concatenate back and reshape to original shape
        weight_q_updated = torch.cat([weight_q_nope, weight_q_pe], dim=1)
        weight_q_updated = weight_q_updated.view(
            self.num_attention_heads_per_partition
            * (self.config.qk_head_dim + self.config.qk_pos_emb_head_dim),
            -1,
        )

        return weight_q_updated

    def _clip_kv_proj_weight(self, weight):
        """Clip kv_proj_weight"""
        # shape: (n, qk_head_dim + v_head_dim, kv_lora_rank)
        weight_reshaped = weight.view(
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
            -1,
        )

        # Split into qk_head_dim and v_head_dim parts: (n, a, -1) and (n, b, -1)
        weight_k = weight_reshaped[:, : self.config.qk_head_dim, :]
        weight_v = weight_reshaped[:, self.config.qk_head_dim :, :]

        # Clipping
        weight_k.mul_(torch.pow(self.qk_clip_balancing_eta, 1 - self.config.qk_clip_alpha))

        # Concatenate back and reshape to original shape
        weight_kv_updated = torch.cat([weight_k, weight_v], dim=1)
        weight_kv_updated = weight_kv_updated.view(
            self.num_attention_heads_per_partition
            * (self.config.qk_head_dim + self.config.v_head_dim),
            -1,
        )

        return weight_kv_updated
