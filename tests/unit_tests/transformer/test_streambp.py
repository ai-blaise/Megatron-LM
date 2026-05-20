# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager
import math
import os
from typing import Optional
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.nvfp4_act_eco.codec import Nvfp4ActEcoConfig
import megatron.core.quantization.nvfp4_act_eco.te_hook as act_eco_te_hook
import megatron.core.extensions.transformer_engine as te_extension_module
from megatron.core.quantization.nvfp4_act_eco.te_hook import (
    install_act_eco_on_te_grouped_linear,
    install_act_eco_on_te_linear,
    pop_act_eco_grad_correction,
)
from megatron.core.quantization.nvfp4_act_eco.reference import (
    activation_eco_bias_correction,
    nvfp4_act_quant_forward,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
import megatron.core.transformer.streambp as streambp_module
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.streambp import (
    StreamBPMoeAuxState,
    StreamBPMoeAuxStats,
    chunked_lm_head_loss,
    current_streambp_moe_aux_replay,
    iter_streambp_chunks,
    make_streambp_packed_seq_params,
    make_streambp_single_sequence_packed_seq_params,
    mark_streambp_pending_chunks,
    moe_streambp_requires_full_replay,
    replay_streambp_moe_aux_stats,
    should_streambp_register_grad_ready,
    slice_streambp_attention_mask,
    slice_streambp_padding_mask,
    streambp_checkpoint_layer,
    streambp_lm_head_loss,
    supports_streambp_moe_hybrid_replay,
)
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.test_utilities import Utils


class ToyCausalLayer(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, hidden_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden_states, attention_mask=None, chunk_range=None, **_kwargs):
        del attention_mask
        if chunk_range is None:
            start, end = 0, hidden_states.size(0)
        else:
            start, end = chunk_range

        prefix = hidden_states[:end]
        q = self.q(prefix)[start:end]
        k = self.k(prefix)
        v = self.v(prefix)
        mask = slice_streambp_attention_mask(
            None, start, end, end, device=hidden_states.device, causal=True
        )[0, 0]

        scores = torch.einsum("cbh,pbh->bcp", q, k) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        context = torch.einsum("bcp,pbh->cbh", probs, v)
        hidden_chunk = self.proj(context) + hidden_states[start:end]
        return self.mlp(hidden_chunk) + hidden_chunk, None


class RecordingToyCausalLayer(ToyCausalLayer):
    def __init__(self, hidden_size: int):
        super().__init__(hidden_size)
        self.chunk_ranges = []

    def forward(self, hidden_states, attention_mask=None, chunk_range=None, **kwargs):
        self.chunk_ranges.append(chunk_range)
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            chunk_range=chunk_range,
            **kwargs,
        )


class ToyMoeAttentionSplitLayer(torch.nn.Module):
    is_moe_layer = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.config = SimpleNamespace(
            moe_router_load_balancing_type="seq_aux_loss",
            moe_expert_capacity_factor=None,
            moe_z_loss_coeff=None,
        )
        self.no_grad_attention_chunks = []
        self.no_grad_mlp_shapes = []
        self.no_grad_events = []
        self.grad_attention_chunks = []
        self.grad_mlp_shapes = []
        self.grad_events = []

    def _forward_attention(self, hidden_states, chunk_range=None, context=None, **_kwargs):
        assert context is None
        if chunk_range is None:
            chunk = hidden_states
        else:
            start, end = chunk_range
            if not torch.is_grad_enabled():
                self.no_grad_attention_chunks.append((start, end))
                self.no_grad_events.append(("attention", start, end))
            else:
                self.grad_attention_chunks.append((start, end))
                self.grad_events.append(("attention", start, end))
            chunk = hidden_states[start:end]
        return chunk * self.weight, None

    def _forward_mlp(self, hidden_states, inference_context=None, padding_mask=None):
        del inference_context, padding_mask
        if not torch.is_grad_enabled():
            self.no_grad_mlp_shapes.append(tuple(hidden_states.shape))
            self.no_grad_events.append(("mlp", hidden_states.size(0)))
        else:
            self.grad_mlp_shapes.append(tuple(hidden_states.shape))
            self.grad_events.append(("mlp", hidden_states.size(0)))
        return hidden_states + 1.0

    def forward(self, hidden_states, context=None, chunk_range=None, **kwargs):
        hidden_states, context = self._forward_attention(
            hidden_states, context=context, chunk_range=chunk_range, **kwargs
        )
        return self._forward_mlp(hidden_states), context


class ToyOutputLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(vocab_size, hidden_size))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.2)

    def forward(self, input_, weight=None, runtime_gather_output=None):
        del runtime_gather_output
        weight = self.weight if weight is None else weight
        return F.linear(input_, weight), None


def _toy_lm_loss(labels, logits):
    seq_len, batch_size, vocab_size = logits.shape
    return F.cross_entropy(
        logits.reshape(seq_len * batch_size, vocab_size),
        labels.transpose(0, 1).reshape(seq_len * batch_size),
        reduction="none",
    ).view(seq_len, batch_size).transpose(0, 1).contiguous()


def _mock_hadamard_transform(x, scale=1.0):
    return x * scale


def test_streambp_chunk_helpers_build_rectangular_causal_mask():
    assert iter_streambp_chunks(10, 4) == [(0, 4), (4, 8), (8, 10)]
    mask = slice_streambp_attention_mask(None, 2, 5, 5, device=torch.device("cpu"), causal=True)
    expected = torch.tensor(
        [[False, False, False, True, True], [False, False, False, False, True], [False] * 5]
    )
    assert torch.equal(mask[0, 0], expected)


def test_streambp_packed_seq_params_keep_sequence_parallel_rope_offset():
    cu = torch.tensor([0, 32768], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=32768,
        max_seqlen_kv=32768,
    )

    prefix, core = make_streambp_packed_seq_params(
        packed, 2048, 4096, kv_end=4096, sequence_offset=8192
    )

    assert prefix.cu_seqlens_q.tolist() == [8192, 12288]
    assert prefix.cu_seqlens_kv.tolist() == [8192, 12288]
    assert prefix.max_seqlen_q == 12288
    assert prefix.max_seqlen_kv == 12288
    assert core.cu_seqlens_q.tolist() == [0, 2048]
    assert core.cu_seqlens_kv.tolist() == [0, 4096]


def test_streambp_single_sequence_packed_seq_params_describe_local_working_tensor():
    cu = torch.tensor([0, 32768], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=32768,
        max_seqlen_kv=32768,
    )

    local = make_streambp_single_sequence_packed_seq_params(packed, 2048, 8192)

    assert local.cu_seqlens_q.tolist() == [0, 2048]
    assert local.cu_seqlens_kv.tolist() == [0, 8192]
    assert local.max_seqlen_q == 2048
    assert local.max_seqlen_kv == 8192


def test_streambp_padding_mask_slices_batch_and_sequence_major_masks():
    batch_major = torch.arange(2 * 7 * 3).view(2, 7, 3)
    assert torch.equal(
        slice_streambp_padding_mask(batch_major, 2, 5),
        batch_major[:, 2:5, :],
    )

    sequence_major = torch.arange(7 * 2).view(7, 2)
    assert torch.equal(
        slice_streambp_padding_mask(sequence_major, 2, 5),
        sequence_major[2:5, :],
    )


def test_streambp_layer_checkpoint_matches_full_causal_gradients():
    torch.manual_seed(1234)
    full_layer = ToyCausalLayer(hidden_size=8)
    streambp_layer = ToyCausalLayer(hidden_size=8)
    streambp_layer.load_state_dict(full_layer.state_dict())

    full_input = torch.randn(7, 2, 8, requires_grad=True)
    streambp_input = full_input.detach().clone().requires_grad_(True)

    full_output, _ = full_layer(full_input)
    streambp_output, _ = streambp_checkpoint_layer(
        streambp_layer,
        streambp_input,
        chunk_size=3,
        attention_mask=None,
        context=None,
    )

    assert torch.allclose(streambp_output, full_output, atol=1e-6, rtol=1e-6)

    full_output.square().mean().backward()
    streambp_output.square().mean().backward()

    assert torch.allclose(streambp_input.grad, full_input.grad, atol=2e-5, rtol=2e-5)
    for full_param, streambp_param in zip(full_layer.parameters(), streambp_layer.parameters()):
        assert torch.allclose(streambp_param.grad, full_param.grad, atol=2e-5, rtol=2e-5)


def test_streambp_reference_style_forward_chunks_only_backward_replay():
    torch.manual_seed(2460)
    layer = RecordingToyCausalLayer(hidden_size=8)
    hidden_states = torch.randn(7, 2, 8, requires_grad=True)

    output, _ = streambp_checkpoint_layer(
        layer,
        hidden_states,
        chunk_size=3,
        chunk_forward=False,
        attention_mask=None,
        context=None,
    )

    assert layer.chunk_ranges == [None]
    output.float().square().mean().backward()
    assert layer.chunk_ranges == [None, (0, 3), (3, 6), (6, 7)]


def test_streambp_reference_style_forward_matches_full_causal_gradients():
    torch.manual_seed(8642)
    full_layer = ToyCausalLayer(hidden_size=8)
    streambp_layer = ToyCausalLayer(hidden_size=8)
    streambp_layer.load_state_dict(full_layer.state_dict())

    full_input = torch.randn(7, 2, 8, requires_grad=True)
    streambp_input = full_input.detach().clone().requires_grad_(True)

    full_output, _ = full_layer(full_input)
    streambp_output, _ = streambp_checkpoint_layer(
        streambp_layer,
        streambp_input,
        chunk_size=3,
        chunk_forward=False,
        attention_mask=None,
        context=None,
    )

    assert torch.allclose(streambp_output, full_output, atol=1e-6, rtol=1e-6)

    full_output.square().mean().backward()
    streambp_output.square().mean().backward()

    assert torch.allclose(streambp_input.grad, full_input.grad, atol=2e-5, rtol=2e-5)
    for full_param, streambp_param in zip(full_layer.parameters(), streambp_layer.parameters()):
        assert torch.allclose(streambp_param.grad, full_param.grad, atol=2e-5, rtol=2e-5)


def test_streambp_moe_chunk_forward_override_inherits_global_default():
    block = TransformerBlock.__new__(TransformerBlock)
    block.config = SimpleNamespace(
        streambp_chunk_forward=True,
        streambp_moe_chunk_forward=None,
    )

    assert block._streambp_chunk_forward_for_mode("chunked_moe") is True
    assert block._streambp_chunk_forward_for_mode("chunked") is True


def test_streambp_moe_chunk_forward_override_only_affects_moe():
    block = TransformerBlock.__new__(TransformerBlock)
    block.config = SimpleNamespace(
        streambp_chunk_forward=True,
        streambp_moe_chunk_forward=False,
    )

    assert block._streambp_chunk_forward_for_mode("chunked_moe") is False
    assert block._streambp_chunk_forward_for_mode("chunked_packed_dsa") is True


def test_streambp_replay_quantized_te_input_context_restores(monkeypatch):
    class Child(torch.nn.Module):
        def __init__(self, enabled):
            super().__init__()
            self.save_original_input = enabled

    layer = torch.nn.Module()
    layer.enabled_child = Child(True)
    layer.disabled_child = Child(False)

    monkeypatch.setenv("MEGATRON_STREAMBP_REPLAY_SAVE_QUANTIZED_TE_INPUTS", "1")
    with streambp_module._streambp_replay_save_quantized_te_inputs(layer):
        assert layer.enabled_child.save_original_input is False
        assert layer.disabled_child.save_original_input is False

    assert layer.enabled_child.save_original_input is True
    assert layer.disabled_child.save_original_input is False


def test_streambp_replay_quantized_te_input_context_can_be_disabled(monkeypatch):
    class Child(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.save_original_input = True

    layer = torch.nn.Module()
    layer.child = Child()

    monkeypatch.setenv("MEGATRON_STREAMBP_REPLAY_SAVE_QUANTIZED_TE_INPUTS", "0")
    with streambp_module._streambp_replay_save_quantized_te_inputs(layer):
        assert layer.child.save_original_input is True

    assert layer.child.save_original_input is True


def test_te_grouped_wgrad_accum_env_only_affects_experts(monkeypatch):
    cfg = SimpleNamespace(gradient_accumulation_fusion=False)

    monkeypatch.setenv("MEGATRON_TE_GROUPED_LINEAR_FUSE_WGRAD_ACCUM", "1")
    assert te_extension_module._te_grouped_linear_fuse_wgrad_accumulation(
        cfg, is_expert=True
    )
    assert not te_extension_module._te_grouped_linear_fuse_wgrad_accumulation(
        cfg, is_expert=False
    )

    monkeypatch.setenv("MEGATRON_TE_GROUPED_LINEAR_FUSE_WGRAD_ACCUM", "0")
    assert not te_extension_module._te_grouped_linear_fuse_wgrad_accumulation(
        cfg, is_expert=True
    )


def test_streambp_routes_full_replay_required_moe_to_hybrid_when_nonchunk_forward():
    block = TransformerBlock.__new__(TransformerBlock)
    block.training = True
    block.config = SimpleNamespace(
        use_streambp=True,
        streambp_chunk_forward=True,
        streambp_moe_chunk_forward=False,
        streambp_skip_moe=False,
        streambp_skip_dsa=False,
    )
    layer = ToyMoeAttentionSplitLayer()
    layer.config.moe_expert_capacity_factor = 1.0

    assert moe_streambp_requires_full_replay(layer)
    assert supports_streambp_moe_hybrid_replay(layer)
    assert block._streambp_layer_decision(
        layer,
        inference_context=None,
        packed_seq_params=None,
        mhc_manager=None,
    ) == (True, "chunked_moe")

    block.config.streambp_moe_chunk_forward = True
    assert block._streambp_layer_decision(
        layer,
        inference_context=None,
        packed_seq_params=None,
        mhc_manager=None,
    ) == (True, "full_replay_moe")


def test_streambp_moe_nonchunk_forward_and_backward_chunk_attention_once_for_mlp():
    full_layer = ToyMoeAttentionSplitLayer()
    layer = ToyMoeAttentionSplitLayer()
    layer.load_state_dict(full_layer.state_dict())
    full_hidden_states = torch.randn(10, 2, 3, requires_grad=True)
    hidden_states = full_hidden_states.detach().clone().requires_grad_(True)
    recompute_phases = []

    @contextmanager
    def fake_te_recompute_context(*, recompute_phase):
        recompute_phases.append(recompute_phase)
        yield

    full_output, _ = full_layer(full_hidden_states)
    with patch(
        "megatron.core.transformer.streambp._te_activation_recompute_context",
        fake_te_recompute_context,
    ):
        output, _ = streambp_checkpoint_layer(
            layer,
            hidden_states,
            chunk_size=4,
            chunk_forward=False,
            mhc_recompute_manager=None,
        )

        grad_output = torch.randn_like(output)
        assert layer.no_grad_attention_chunks == [(0, 4), (4, 8), (8, 10)]
        assert layer.no_grad_mlp_shapes == [(10, 2, 3)]
        assert torch.allclose(output, full_output)
        full_output.backward(grad_output)
        output.backward(grad_output)

    assert layer.grad_attention_chunks == [(0, 4), (4, 8), (8, 10)]
    assert layer.grad_mlp_shapes == [(10, 2, 3)]
    assert recompute_phases == [False, False, False, False, True, True, True, True]
    assert torch.allclose(hidden_states.grad, full_hidden_states.grad)
    assert torch.allclose(layer.weight.grad, full_layer.weight.grad)


def test_streambp_moe_hybrid_can_split_mlp_replay_into_large_chunks():
    full_layer = ToyMoeAttentionSplitLayer()
    layer = ToyMoeAttentionSplitLayer()
    layer.load_state_dict(full_layer.state_dict())
    full_hidden_states = torch.randn(10, 2, 3, requires_grad=True)
    hidden_states = full_hidden_states.detach().clone().requires_grad_(True)
    recompute_phases = []

    @contextmanager
    def fake_te_recompute_context(*, recompute_phase):
        recompute_phases.append(recompute_phase)
        yield

    full_output, _ = full_layer(full_hidden_states)
    with patch(
        "megatron.core.transformer.streambp._te_activation_recompute_context",
        fake_te_recompute_context,
    ):
        output, _ = streambp_checkpoint_layer(
            layer,
            hidden_states,
            chunk_size=4,
            chunk_forward=False,
            moe_mlp_chunks=2,
            moe_mlp_backward_chunks=4,
            mhc_recompute_manager=None,
        )

        grad_output = torch.randn_like(output)
        assert layer.no_grad_attention_chunks == [(0, 4), (4, 5), (5, 8), (8, 10)]
        assert layer.no_grad_mlp_shapes == [(5, 2, 3), (5, 2, 3)]
        assert layer.no_grad_events == [
            ("attention", 0, 4),
            ("attention", 4, 5),
            ("mlp", 5),
            ("attention", 5, 8),
            ("attention", 8, 10),
            ("mlp", 5),
        ]
        assert torch.allclose(output, full_output)
        full_output.backward(grad_output)
        output.backward(grad_output)

    assert layer.grad_attention_chunks == [
        (0, 3),
        (3, 4),
        (4, 6),
        (6, 8),
        (8, 9),
        (9, 10),
    ]
    assert layer.grad_mlp_shapes == [(3, 2, 3), (3, 2, 3), (3, 2, 3), (1, 2, 3)]
    assert layer.grad_events == [
        ("attention", 0, 3),
        ("mlp", 3),
        ("attention", 3, 4),
        ("attention", 4, 6),
        ("mlp", 3),
        ("attention", 6, 8),
        ("attention", 8, 9),
        ("mlp", 3),
        ("attention", 9, 10),
        ("mlp", 1),
    ]
    assert recompute_phases == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert torch.allclose(hidden_states.grad, full_hidden_states.grad)
    assert torch.allclose(layer.weight.grad, full_layer.weight.grad)


def test_streambp_moe_hybrid_can_split_attention_backward_chunks(monkeypatch):
    monkeypatch.setenv("MEGATRON_STREAMBP_MOE_ATTENTION_BACKWARD_CHUNK_SIZE", "2")

    full_layer = ToyMoeAttentionSplitLayer()
    layer = ToyMoeAttentionSplitLayer()
    layer.load_state_dict(full_layer.state_dict())
    full_hidden_states = torch.randn(10, 2, 3, requires_grad=True)
    hidden_states = full_hidden_states.detach().clone().requires_grad_(True)

    full_output, _ = full_layer(full_hidden_states)
    output, _ = streambp_checkpoint_layer(
        layer,
        hidden_states,
        chunk_size=4,
        chunk_forward=False,
        moe_mlp_chunks=2,
        moe_mlp_backward_chunks=2,
        mhc_recompute_manager=None,
    )

    grad_output = torch.randn_like(output)
    assert layer.no_grad_attention_chunks == [(0, 4), (4, 5), (5, 8), (8, 10)]
    assert layer.no_grad_mlp_shapes == [(5, 2, 3), (5, 2, 3)]
    assert torch.allclose(output, full_output)
    full_output.backward(grad_output)
    output.backward(grad_output)

    assert layer.grad_attention_chunks == [
        (0, 2),
        (2, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (8, 10),
    ]
    assert layer.grad_mlp_shapes == [(5, 2, 3), (5, 2, 3)]
    assert layer.grad_events == [
        ("attention", 0, 2),
        ("attention", 2, 4),
        ("attention", 4, 5),
        ("mlp", 5),
        ("attention", 5, 6),
        ("attention", 6, 8),
        ("attention", 8, 10),
        ("mlp", 5),
    ]
    assert torch.allclose(hidden_states.grad, full_hidden_states.grad)
    assert torch.allclose(layer.weight.grad, full_layer.weight.grad)


def test_streambp_moe_aux_replay_combines_full_counts_with_chunk_token_scale():
    state = StreamBPMoeAuxState()
    router = object()
    first = StreamBPMoeAuxStats(
        tokens_per_expert=torch.tensor([1.0, 2.0]),
        local_num_tokens=torch.tensor(3.0),
        total_num_tokens=torch.tensor(3.0),
        seq_length=4,
        bsz=1,
        with_padding_mask=False,
    )
    second = StreamBPMoeAuxStats(
        tokens_per_expert=torch.tensor([3.0, 5.0]),
        local_num_tokens=torch.tensor(8.0),
        total_num_tokens=torch.tensor(8.0),
        seq_length=6,
        bsz=1,
        with_padding_mask=False,
    )

    state.record(router, "seq_aux_loss", first)
    state.record(router, "seq_aux_loss", second)

    summed = state.get(router, "seq_aux_loss")
    assert torch.equal(summed.tokens_per_expert, torch.tensor([4.0, 7.0]))
    assert summed.seq_length == 10

    with replay_streambp_moe_aux_stats(state):
        replay = current_streambp_moe_aux_replay()
        replay_stats = replay.get(router, "seq_aux_loss")
        assert torch.equal(replay_stats.tokens_per_expert, torch.tensor([4.0, 7.0]))
        assert replay_stats.local_num_tokens.item() == 11.0
        assert replay_stats.total_num_tokens.item() == 11.0
        assert replay_stats.seq_length == 10

    with replay_streambp_moe_aux_stats(state, chunk_index=0):
        replay = current_streambp_moe_aux_replay()
        replay_stats = replay.get(router, "seq_aux_loss")
        assert torch.equal(replay_stats.tokens_per_expert, torch.tensor([4.0, 7.0]))
        assert replay_stats.local_num_tokens.item() == 3.0
        assert replay_stats.total_num_tokens.item() == 11.0
        assert replay_stats.seq_length == 4

    with replay_streambp_moe_aux_stats(state, chunk_index=1):
        replay = current_streambp_moe_aux_replay()
        replay_stats = replay.get(router, "seq_aux_loss")
        assert torch.equal(replay_stats.tokens_per_expert, torch.tensor([4.0, 7.0]))
        assert replay_stats.local_num_tokens.item() == 8.0
        assert replay_stats.total_num_tokens.item() == 11.0
        assert replay_stats.seq_length == 6


def test_streambp_moe_hybrid_handles_full_replay_required_modes():
    full_layer = ToyMoeAttentionSplitLayer()
    layer = ToyMoeAttentionSplitLayer()
    full_layer.config.moe_expert_capacity_factor = 1.0
    layer.config.moe_expert_capacity_factor = 1.0
    layer.load_state_dict(full_layer.state_dict())
    full_hidden_states = torch.randn(10, 2, 3, requires_grad=True)
    hidden_states = full_hidden_states.detach().clone().requires_grad_(True)

    assert moe_streambp_requires_full_replay(layer)

    full_output, _ = full_layer(full_hidden_states)
    output, _ = streambp_checkpoint_layer(
        layer,
        hidden_states,
        chunk_size=4,
        chunk_forward=False,
        mhc_recompute_manager=None,
    )

    grad_output = torch.randn_like(output)
    assert layer.no_grad_attention_chunks == [(0, 4), (4, 8), (8, 10)]
    assert layer.no_grad_mlp_shapes == [(10, 2, 3)]
    assert torch.allclose(output, full_output)

    full_output.backward(grad_output)
    output.backward(grad_output)

    assert layer.grad_attention_chunks == [(0, 4), (4, 8), (8, 10)]
    assert layer.grad_mlp_shapes == [(10, 2, 3)]
    assert torch.allclose(hidden_states.grad, full_hidden_states.grad)
    assert torch.allclose(layer.weight.grad, full_layer.weight.grad)


def test_streambp_layer_checkpoint_tracks_param_grads_for_gradless_input():
    torch.manual_seed(4321)
    layer = ToyCausalLayer(hidden_size=8)
    hidden_states = torch.randn(7, 2, 8)

    output, _ = streambp_checkpoint_layer(
        layer,
        hidden_states,
        chunk_size=3,
        attention_mask=None,
        context=None,
    )

    assert output.requires_grad
    output.float().square().mean().backward()
    assert hidden_states.grad is None
    for param in layer.parameters():
        assert param.grad is not None


def test_streambp_lm_head_loss_matches_full_logits_gradients():
    torch.manual_seed(5678)
    full_head = ToyOutputLayer(hidden_size=6, vocab_size=13)
    streambp_head = ToyOutputLayer(hidden_size=6, vocab_size=13)
    streambp_head.load_state_dict(full_head.state_dict())

    labels = torch.randint(0, 13, (2, 9))
    full_hidden = torch.randn(9, 2, 6, requires_grad=True)
    streambp_hidden = full_hidden.detach().clone().requires_grad_(True)

    full_logits, _ = full_head(input_=full_hidden)
    full_loss = _toy_lm_loss(labels, full_logits)
    streambp_loss = streambp_lm_head_loss(
        streambp_head,
        streambp_hidden,
        labels,
        loss_func=_toy_lm_loss,
        chunk_size=4,
    )

    assert torch.allclose(streambp_loss, full_loss, atol=1e-6, rtol=1e-6)

    full_loss.mean().backward()
    streambp_loss.mean().backward()

    assert torch.allclose(streambp_hidden.grad, full_hidden.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(streambp_head.weight.grad, full_head.weight.grad, atol=2e-5, rtol=2e-5)


def test_chunked_lm_head_loss_works_without_layer_streambp():
    torch.manual_seed(5679)
    full_head = ToyOutputLayer(hidden_size=6, vocab_size=13)
    chunked_head = ToyOutputLayer(hidden_size=6, vocab_size=13)
    chunked_head.load_state_dict(full_head.state_dict())

    labels = torch.randint(0, 13, (2, 9))
    full_hidden = torch.randn(9, 2, 6, requires_grad=True)
    chunked_hidden = full_hidden.detach().clone().requires_grad_(True)

    full_logits, _ = full_head(input_=full_hidden)
    full_loss = _toy_lm_loss(labels, full_logits)
    chunked_loss = chunked_lm_head_loss(
        chunked_head,
        chunked_hidden,
        labels,
        loss_func=_toy_lm_loss,
        chunk_size=4,
    )

    assert torch.allclose(chunked_loss, full_loss, atol=1e-6, rtol=1e-6)

    full_loss.mean().backward()
    chunked_loss.mean().backward()

    assert torch.allclose(chunked_hidden.grad, full_hidden.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(chunked_head.weight.grad, full_head.weight.grad, atol=2e-5, rtol=2e-5)


def test_streambp_lm_head_fused_lce_path_matches_full_logits_gradients():
    torch.manual_seed(5680)
    full_head = ToyOutputLayer(hidden_size=6, vocab_size=13)
    streambp_head = ToyOutputLayer(hidden_size=6, vocab_size=13)
    streambp_head.load_state_dict(full_head.state_dict())

    labels = torch.randint(0, 13, (2, 9))
    full_hidden = torch.randn(9, 2, 6, requires_grad=True)
    streambp_hidden = full_hidden.detach().clone().requires_grad_(True)
    fused_calls = []

    def reference_fused_lce(
        output_layer,
        hidden_states,
        labels_chunk,
        output_layer_kwargs,
        *,
        sequence_parallel_output,
    ):
        assert not sequence_parallel_output
        fused_calls.append((tuple(hidden_states.shape), tuple(labels_chunk.shape)))
        weight = output_layer_kwargs.get("weight")
        weight = output_layer.weight if weight is None else weight
        logits = F.linear(hidden_states, weight)
        return _toy_lm_loss(labels_chunk, logits)

    full_logits, _ = full_head(input_=full_hidden)
    full_loss = _toy_lm_loss(labels, full_logits)
    with patch.object(
        streambp_module, "_streambp_fused_lce_available", return_value=True
    ), patch.object(streambp_module, "_streambp_fused_lce_loss", side_effect=reference_fused_lce):
        streambp_loss = streambp_module.streambp_lm_head_loss(
            streambp_head,
            streambp_hidden,
            labels,
            loss_func=_toy_lm_loss,
            chunk_size=4,
        )

        assert fused_calls == [
            ((4, 2, 6), (2, 4)),
            ((4, 2, 6), (2, 4)),
            ((1, 2, 6), (2, 1)),
        ]
        assert torch.allclose(streambp_loss, full_loss, atol=1e-6, rtol=1e-6)

        full_loss.mean().backward()
        streambp_loss.mean().backward()

    assert torch.allclose(streambp_hidden.grad, full_hidden.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(streambp_head.weight.grad, full_head.weight.grad, atol=2e-5, rtol=2e-5)


def test_streambp_lm_head_single_chunk_fused_lce_avoids_logits_path():
    torch.manual_seed(5682)
    full_head = ToyOutputLayer(hidden_size=6, vocab_size=13)
    streambp_head = ToyOutputLayer(hidden_size=6, vocab_size=13)
    streambp_head.load_state_dict(full_head.state_dict())

    labels = torch.randint(0, 13, (2, 9))
    full_hidden = torch.randn(9, 2, 6, requires_grad=True)
    streambp_hidden = full_hidden.detach().clone().requires_grad_(True)
    fused_calls = []

    def reference_fused_lce(
        output_layer,
        hidden_states,
        labels_chunk,
        output_layer_kwargs,
        *,
        sequence_parallel_output,
    ):
        assert not sequence_parallel_output
        fused_calls.append((tuple(hidden_states.shape), tuple(labels_chunk.shape)))
        weight = output_layer_kwargs.get("weight")
        weight = output_layer.weight if weight is None else weight
        return _toy_lm_loss(labels_chunk, F.linear(hidden_states, weight))

    full_logits, _ = full_head(input_=full_hidden)
    full_loss = _toy_lm_loss(labels, full_logits)
    with patch.object(
        streambp_module, "_streambp_fused_lce_available", return_value=True
    ), patch.object(
        streambp_module, "_streambp_fused_lce_loss", side_effect=reference_fused_lce
    ), patch.object(
        streambp_module, "_call_output_layer", side_effect=AssertionError("logits path used")
    ):
        streambp_loss = streambp_module.streambp_lm_head_loss(
            streambp_head,
            streambp_hidden,
            labels,
            loss_func=_toy_lm_loss,
            chunk_size=9,
        )

        assert fused_calls == [((9, 2, 6), (2, 9))]
        assert torch.allclose(streambp_loss, full_loss, atol=1e-6, rtol=1e-6)

        full_loss.mean().backward()
        streambp_loss.mean().backward()

    assert torch.allclose(streambp_hidden.grad, full_hidden.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(streambp_head.weight.grad, full_head.weight.grad, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streambp_lm_head_real_fused_lce_cuda_matches_unfused_gradients():
    if torch.cuda.get_device_capability(torch.cuda.current_device())[0] != 10:
        pytest.skip("Blackwell fused linear cross entropy requires compute capability 10.x")
    if streambp_module._load_streambp_fused_lce() is None:
        pytest.skip("Blackwell fused linear cross entropy extension is not available")

    torch.manual_seed(5681)
    full_head = ToyOutputLayer(hidden_size=64, vocab_size=31).cuda().bfloat16()
    fused_head = ToyOutputLayer(hidden_size=64, vocab_size=31).cuda().bfloat16()
    fused_head.load_state_dict(full_head.state_dict())

    labels = torch.randint(0, 31, (2, 12), device="cuda")
    full_hidden = torch.randn(12, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fused_hidden = full_hidden.detach().clone().requires_grad_(True)

    with patch.dict(os.environ, {"MEGATRON_STREAMBP_FUSED_LCE": "0"}):
        full_loss = streambp_module.streambp_lm_head_loss(
            full_head,
            full_hidden,
            labels,
            loss_func=_toy_lm_loss,
            chunk_size=4,
        )
    with patch.dict(os.environ, {"MEGATRON_STREAMBP_FUSED_LCE": "1"}):
        fused_loss = streambp_module.streambp_lm_head_loss(
            fused_head,
            fused_hidden,
            labels,
            loss_func=_toy_lm_loss,
            chunk_size=4,
        )

    torch.testing.assert_close(fused_loss.float(), full_loss.float(), rtol=2e-2, atol=2e-2)

    full_loss.float().mean().backward()
    fused_loss.float().mean().backward()

    torch.testing.assert_close(
        fused_hidden.grad.float(), full_hidden.grad.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        fused_head.weight.grad.float(), full_head.weight.grad.float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streambp_lm_head_streaming_sp_fused_lce_matches_original_sp_fused_lce():
    if torch.cuda.get_device_capability(torch.cuda.current_device())[0] != 10:
        pytest.skip("Blackwell fused linear cross entropy requires compute capability 10.x")
    if "RANK" not in os.environ:
        pytest.skip("Run with torchrun --nproc-per-node>=2")
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(backend="nccl")
    if dist.get_world_size() < 2:
        pytest.skip("streaming SP fused LCE requires at least 2 ranks")
    if streambp_module._load_streambp_fused_lce() is None:
        pytest.skip("Blackwell fused linear cross entropy extension is not available")

    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.manual_seed(6000 + rank)
    hidden_size = 64
    local_vocab = 17
    local_seq = 6
    batch = 2
    total_vocab = local_vocab * world

    original_head = ToyOutputLayer(hidden_size=hidden_size, vocab_size=local_vocab).cuda().bfloat16()
    streaming_head = ToyOutputLayer(hidden_size=hidden_size, vocab_size=local_vocab).cuda().bfloat16()
    streaming_head.load_state_dict(original_head.state_dict())
    for head in (original_head, streaming_head):
        head.sequence_parallel = True
        head.tp_group = dist.group.WORLD

    hidden = torch.randn(
        local_seq, batch, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    hidden_stream = hidden.detach().clone().requires_grad_(True)
    labels = torch.empty((batch, local_seq * world), device="cuda", dtype=torch.long)
    if rank == 0:
        labels.random_(0, total_vocab)
    dist.broadcast(labels, src=0)

    with patch.dict(os.environ, {"MEGATRON_STREAMBP_FUSED_LCE_SP_STREAMING": "0"}):
        original_loss = streambp_module.streambp_lm_head_loss(
            original_head,
            hidden,
            labels,
            loss_func=_toy_lm_loss,
            chunk_size=local_seq,
        )
    with patch.dict(
        os.environ,
        {
            "MEGATRON_STREAMBP_FUSED_LCE_SP_STREAMING": "1",
            "MEGATRON_STREAMBP_FUSED_LCE_SP_TILE_SIZE": "3",
        },
    ):
        streaming_loss = streambp_module.streambp_lm_head_loss(
            streaming_head,
            hidden_stream,
            labels,
            loss_func=_toy_lm_loss,
            chunk_size=local_seq,
        )

    torch.testing.assert_close(
        streaming_loss.float(), original_loss.float(), rtol=2e-2, atol=2e-2
    )
    original_loss.float().mean().backward()
    streaming_loss.float().mean().backward()
    torch.testing.assert_close(hidden_stream.grad.float(), hidden.grad.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        streaming_head.weight.grad.float(), original_head.weight.grad.float(), rtol=2e-2, atol=2e-2
    )
    dist.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streambp_mcore_transformer_block_matches_baseline_gradients():
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(2468)
        model_parallel_cuda_manual_seed(2468)
        config_kwargs = dict(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            ffn_hidden_size=64,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        baseline_config = TransformerConfig(**config_kwargs)
        streambp_config = TransformerConfig(
            **config_kwargs,
            use_streambp=True,
            streambp_chunk_size=4,
            streambp_logits_chunk_size=4,
        )
        baseline = TransformerBlock(baseline_config, get_gpt_layer_local_spec()).cuda()
        streambp = TransformerBlock(streambp_config, get_gpt_layer_local_spec()).cuda()
        streambp.load_state_dict(baseline.state_dict())
        baseline.train()
        streambp.train()

        seq_len, batch_size = 12, 2
        attention_mask = torch.triu(
            torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device="cuda"), diagonal=1
        )
        baseline_input = torch.randn(
            seq_len, batch_size, baseline_config.hidden_size, device="cuda", requires_grad=True
        )
        streambp_input = baseline_input.detach().clone().requires_grad_(True)

        baseline_output = baseline(baseline_input, attention_mask)
        streambp_output = streambp(streambp_input, attention_mask)
        assert torch.allclose(streambp_output, baseline_output, atol=3e-4, rtol=3e-4)

        baseline_output.float().square().mean().backward()
        streambp_output.float().square().mean().backward()

        assert torch.allclose(streambp_input.grad, baseline_input.grad, atol=3e-4, rtol=3e-4)
        for baseline_param, streambp_param in zip(baseline.parameters(), streambp.parameters()):
            assert torch.allclose(
                streambp_param.grad, baseline_param.grad, atol=3e-4, rtol=3e-4
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streambp_dsa_rectangular_chunk_matches_full_attention(monkeypatch):
    monkeypatch.setenv("MEGATRON_DSA_STREAMING_INDEXER_TOPK", "0")
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(1357)
        model_parallel_cuda_manual_seed(1357)
        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            use_cpu_initialization=True,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type="rope",
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=16,
            dsa_indexer_loss_coeff=0.0,
            dsa_indexer_use_sparse_loss=False,
        )
        indexer_spec = ModuleSpec(
            module=DSAIndexer,
            submodules=DSAIndexerSubmodules(
                linear_wq_b=ModuleSpec(module=TELinear),
                linear_wk=ModuleSpec(module=TELinear),
                k_norm=ModuleSpec(module=TENorm),
                linear_weights_proj=ModuleSpec(module=TELinear),
            ),
        )
        attn = (
            DSAttention(
                config=config,
                submodules=DSAttentionSubmodules(indexer=indexer_spec),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                    required_pgs=["tp", "cp"]
                ),
            )
            .cuda()
            .bfloat16()
            .train()
        )

        seq_len, batch_size = 24, 1
        start, end = 8, 16
        heads = config.num_attention_heads
        query = torch.randn(
            seq_len, batch_size, heads, config.qk_head_dim, device="cuda", dtype=torch.bfloat16
        )
        key = torch.randn(
            seq_len, batch_size, heads, config.qk_head_dim, device="cuda", dtype=torch.bfloat16
        )
        value = torch.randn(
            seq_len, batch_size, heads, config.v_head_dim, device="cuda", dtype=torch.bfloat16
        )
        x = torch.randn(
            seq_len, batch_size, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        qr = torch.randn(
            seq_len, batch_size, config.q_lora_rank, device="cuda", dtype=torch.bfloat16
        )
        attention_mask = torch.triu(
            torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool), diagonal=1
        ).view(1, 1, seq_len, seq_len)

        with patch(
            "megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform",
            _mock_hadamard_transform,
        ):
            full = attn(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                x=x,
                qr=qr,
                attn_mask_type=AttnMaskType.causal,
            )
            chunk = attn(
                query=query[start:end],
                key=key[:end],
                value=value[:end],
                attention_mask=attention_mask[..., start:end, :end],
                x=x[:end],
                qr=qr[:end],
                attn_mask_type=AttnMaskType.causal,
                streambp_positions=(
                    torch.arange(start, end, device="cuda"),
                    torch.arange(end, device="cuda"),
                ),
            )

            torch.testing.assert_close(chunk, full[start:end], rtol=2e-2, atol=2e-2)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streambp_dsa_sequence_parallel_replay_uses_compact_hisa_path(monkeypatch):
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("Run with torchrun --nproc-per-node=2")
    monkeypatch.setenv("MEGATRON_DSA_STREAMING_INDEXER_TOPK", "0")
    monkeypatch.setenv("MEGATRON_DSA_CHUNK_INDEXER_PROJ", "1")
    monkeypatch.setenv("MEGATRON_HISA_SELECTOR_BACKEND", "bmm")
    monkeypatch.setenv("MEGATRON_HISA_FUSED_INDEXER_LOSS", "1")
    monkeypatch.setenv("MEGATRON_DSA_STREAM_TRITON_ATTENTION_CHUNKS", "1")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    try:
        rank = dist.get_rank()
        tp_size = dist.get_world_size()
        torch.manual_seed(97531)
        model_parallel_cuda_manual_seed(97531)
        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            use_cpu_initialization=True,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type="rope",
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=4,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=4,
            dsa_indexer_loss_coeff=0.01,
            dsa_indexer_use_sparse_loss=False,
            dsa_chunk_size=8,
            dsa_indexcache_quantization="nvfp4_e2m1_ue8m0",
            dsa_indexcache_hisa_enabled=True,
            dsa_indexcache_hisa_block_size=4,
            dsa_indexcache_hisa_block_topk=2,
            dsa_indexcache_hisa_compression_ratio=2.0,
        )
        indexer_spec = ModuleSpec(
            module=DSAIndexer,
            submodules=DSAIndexerSubmodules(
                linear_wq_b=ModuleSpec(module=TELinear),
                linear_wk=ModuleSpec(module=TELinear),
                k_norm=ModuleSpec(module=TENorm),
                linear_weights_proj=ModuleSpec(module=TELinear),
            ),
        )

        def build_attn():
            return (
                DSAttention(
                    config=config,
                    submodules=DSAttentionSubmodules(indexer=indexer_spec),
                    layer_number=1,
                    attn_mask_type=AttnMaskType.causal,
                    attention_type="self",
                    pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                        required_pgs=["tp", "cp"]
                    ),
                )
                .cuda()
                .bfloat16()
                .train()
            )

        baseline = build_attn()
        optimized = build_attn()
        for param in baseline.parameters():
            dist.broadcast(param.data, src=0)
        optimized.load_state_dict(baseline.state_dict())

        local_prefix = 6
        start, end = 2, 6
        query_len = (end - start) * tp_size
        key_len = local_prefix * tp_size
        batch_size = 1
        heads = config.num_attention_heads // tp_size

        torch.manual_seed(314159 + rank)
        local_x = torch.randn(
            local_prefix,
            batch_size,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        local_qr = torch.randn(
            end - start,
            batch_size,
            config.q_lora_rank,
            device="cuda",
            dtype=torch.bfloat16,
        )
        torch.manual_seed(271828)
        query = torch.randn(
            query_len,
            batch_size,
            heads,
            config.qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.randn(
            key_len,
            batch_size,
            heads,
            config.qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = torch.randn(
            key_len,
            batch_size,
            heads,
            config.v_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        rank_offsets = torch.arange(tp_size, device="cuda", dtype=torch.long) * local_prefix
        local_positions = torch.arange(start, end, device="cuda", dtype=torch.long)
        query_positions = (rank_offsets[:, None] + local_positions[None, :]).reshape(-1)
        key_positions = torch.arange(key_len, device="cuda", dtype=torch.long)

        def run(attn, *, sp_project_before_gather: str):
            from megatron.core.transformer.experimental_attention_variant.dsa import (
                DSAIndexerAuxLossState,
            )

            monkeypatch.setenv("MEGATRON_DSA_SP_PROJECT_BEFORE_GATHER", sp_project_before_gather)
            DSAIndexerAuxLossState.clear()
            out = attn(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                x=local_x,
                qr=local_qr,
                attn_mask_type=AttnMaskType.causal,
                streambp_positions=(query_positions, key_positions),
            )
            aux_loss = DSAIndexerAuxLossState.total()
            assert aux_loss is not None
            (out.float().square().mean() + aux_loss.float()).backward()
            grads = [param.grad.detach().clone() for param in attn.indexer.parameters()]
            return out.detach(), aux_loss.detach(), grads

        with patch(
            "megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform",
            _mock_hadamard_transform,
        ):
            baseline_out, baseline_aux, baseline_grads = run(
                baseline, sp_project_before_gather="0"
            )
            optimized_out, optimized_aux, optimized_grads = run(
                optimized, sp_project_before_gather="1"
            )

        torch.testing.assert_close(optimized_out, baseline_out, rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(optimized_aux, baseline_aux, rtol=3e-3, atol=3e-3)
        for optimized_grad, baseline_grad in zip(optimized_grads, baseline_grads):
            torch.testing.assert_close(
                optimized_grad.float(), baseline_grad.float(), rtol=5e-3, atol=5e-3
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streambp_packed_dsa_chunk_matches_full_attention(monkeypatch):
    monkeypatch.setenv("MEGATRON_DSA_STREAMING_INDEXER_TOPK", "0")
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(2469)
        model_parallel_cuda_manual_seed(2469)
        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            use_cpu_initialization=True,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type="rope",
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=16,
            dsa_indexer_loss_coeff=0.0,
            dsa_indexer_use_sparse_loss=False,
        )
        indexer_spec = ModuleSpec(
            module=DSAIndexer,
            submodules=DSAIndexerSubmodules(
                linear_wq_b=ModuleSpec(module=TELinear),
                linear_wk=ModuleSpec(module=TELinear),
                k_norm=ModuleSpec(module=TENorm),
                linear_weights_proj=ModuleSpec(module=TELinear),
            ),
        )
        attn = (
            DSAttention(
                config=config,
                submodules=DSAttentionSubmodules(indexer=indexer_spec),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                    required_pgs=["tp", "cp"]
                ),
            )
            .cuda()
            .bfloat16()
            .train()
        )

        lengths = [10, 14]
        seq_len = sum(lengths)
        start, end = 6, 18
        heads = config.num_attention_heads
        query = torch.randn(
            seq_len, heads, config.qk_head_dim, device="cuda", dtype=torch.bfloat16
        )
        key = torch.randn(
            seq_len, heads, config.qk_head_dim, device="cuda", dtype=torch.bfloat16
        )
        value = torch.randn(
            seq_len, heads, config.v_head_dim, device="cuda", dtype=torch.bfloat16
        )
        x = torch.randn(seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        qr = torch.randn(seq_len, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda")
        cu_seqlens = cu_seqlens.to(torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        )
        _, core_packed = make_streambp_packed_seq_params(packed, start, end)

        with patch(
            "megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform",
            _mock_hadamard_transform,
        ):
            full = attn(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                x=x,
                qr=qr,
                attn_mask_type=AttnMaskType.causal,
                packed_seq_params=packed,
            )
            chunk = attn(
                query=query[start:end],
                key=key[:end],
                value=value[:end],
                attention_mask=None,
                x=x[:end],
                qr=qr[:end],
                attn_mask_type=AttnMaskType.causal,
                packed_seq_params=core_packed,
                streambp_positions=(
                    torch.arange(start, end, device="cuda"),
                    torch.arange(end, device="cuda"),
                ),
            )

        torch.testing.assert_close(chunk, full[start:end], rtol=2e-2, atol=2e-2)

        later_start, later_end = 10, 14
        _, full_kv_core_packed = make_streambp_packed_seq_params(
            packed, later_start, later_end, kv_end=seq_len
        )
        with patch(
            "megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform",
            _mock_hadamard_transform,
        ):
            later_chunk = attn(
                query=query[later_start:later_end],
                key=key,
                value=value,
                attention_mask=None,
                x=x,
                qr=qr,
                attn_mask_type=AttnMaskType.causal,
                packed_seq_params=full_kv_core_packed,
                streambp_positions=(
                    torch.arange(later_start, later_end, device="cuda"),
                    torch.arange(seq_len, device="cuda"),
                ),
            )
        torch.testing.assert_close(
            later_chunk, full[later_start:later_end], rtol=2e-2, atol=2e-2
        )
    finally:
        Utils.destroy_model_parallel()


def _build_moe_transformer_block_pair(
    moe_aux_loss_coeff: float = 0.0,
    moe_router_load_balancing_type: str = "aux_loss",
    streambp_moe_chunk_forward: bool = True,
    streambp_moe_mlp_chunks: int = 1,
    streambp_moe_mlp_backward_chunks: Optional[int] = None,
):
    config_kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        ffn_hidden_size=64,
        moe_ffn_hidden_size=64,
        num_moe_experts=4,
        moe_layer_freq=1,
        moe_router_load_balancing_type=moe_router_load_balancing_type,
        moe_router_topk=2,
        moe_aux_loss_coeff=moe_aux_loss_coeff,
        moe_grouped_gemm=False,
        moe_token_dispatcher_type="allgather",
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
    )
    baseline_config = TransformerConfig(**config_kwargs)
    streambp_config = TransformerConfig(
        **config_kwargs,
        use_streambp=True,
        streambp_chunk_size=4,
        streambp_logits_chunk_size=4,
        streambp_moe_chunk_forward=streambp_moe_chunk_forward,
        streambp_moe_mlp_chunks=streambp_moe_mlp_chunks,
        streambp_moe_mlp_backward_chunks=streambp_moe_mlp_backward_chunks,
    )
    baseline = TransformerBlock(
        baseline_config, get_gpt_decoder_block_spec(baseline_config, False)
    ).cuda()
    streambp = TransformerBlock(
        streambp_config, get_gpt_decoder_block_spec(streambp_config, False)
    ).cuda()
    streambp.load_state_dict(baseline.state_dict())
    baseline.train()
    streambp.train()
    assert any(getattr(layer, "is_moe_layer", False) for layer in streambp.layers)
    assert not any(moe_streambp_requires_full_replay(layer) for layer in streambp.layers)
    return baseline, streambp


def _assert_matching_block_backward(baseline, streambp, *, with_padding_mask: bool):
    seq_len, batch_size, hidden_size = 12, 2, baseline.config.hidden_size
    attention_mask = torch.triu(
        torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device="cuda"), diagonal=1
    )
    padding_mask = None
    if with_padding_mask:
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device="cuda")
        padding_mask[:, -2:] = True

    baseline_input = torch.randn(
        seq_len, batch_size, hidden_size, device="cuda", requires_grad=True
    )
    streambp_input = baseline_input.detach().clone().requires_grad_(True)

    get_moe_metrics_tracker().clear()
    baseline_output = baseline(baseline_input, attention_mask, padding_mask=padding_mask)
    baseline_output.float().square().mean().backward()

    get_moe_metrics_tracker().clear()
    streambp_output = streambp(streambp_input, attention_mask, padding_mask=padding_mask)
    streambp_output.float().square().mean().backward()
    get_moe_metrics_tracker().clear()

    assert torch.allclose(streambp_output, baseline_output, atol=4e-4, rtol=4e-4)
    assert torch.allclose(streambp_input.grad, baseline_input.grad, atol=5e-4, rtol=5e-4)
    for (baseline_name, baseline_param), (streambp_name, streambp_param) in zip(
        baseline.named_parameters(), streambp.named_parameters()
    ):
        assert streambp_name == baseline_name
        if baseline_param.grad is None or streambp_param.grad is None:
            assert baseline_param.grad is None and streambp_param.grad is None
            continue
        assert torch.allclose(
            streambp_param.grad, baseline_param.grad, atol=5e-4, rtol=5e-4
        ), baseline_name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streambp_moe_transformer_block_matches_baseline_gradients():
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(3579)
        model_parallel_cuda_manual_seed(3579)
        baseline, streambp = _build_moe_transformer_block_pair()
        _assert_matching_block_backward(baseline, streambp, with_padding_mask=True)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streambp_moe_aux_loss_full_replay_matches_baseline_gradients():
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(4680)
        model_parallel_cuda_manual_seed(4680)
        baseline, streambp = _build_moe_transformer_block_pair(moe_aux_loss_coeff=0.01)
        _assert_matching_block_backward(baseline, streambp, with_padding_mask=False)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streambp_moe_seq_aux_loss_chunked_matches_baseline_gradients():
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(5791)
        model_parallel_cuda_manual_seed(5791)
        baseline, streambp = _build_moe_transformer_block_pair(
            moe_aux_loss_coeff=0.01,
            moe_router_load_balancing_type="seq_aux_loss",
        )
        _assert_matching_block_backward(baseline, streambp, with_padding_mask=False)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streambp_moe_seq_aux_loss_split_mlp_replay_matches_baseline_gradients():
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(6812)
        model_parallel_cuda_manual_seed(6812)
        baseline, streambp = _build_moe_transformer_block_pair(
            moe_aux_loss_coeff=0.01,
            moe_router_load_balancing_type="seq_aux_loss",
            streambp_moe_chunk_forward=False,
            streambp_moe_mlp_chunks=2,
        )
        _assert_matching_block_backward(baseline, streambp, with_padding_mask=False)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streambp_moe_seq_aux_loss_asymmetric_mlp_replay_matches_baseline_gradients():
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(6813)
        model_parallel_cuda_manual_seed(6813)
        baseline, streambp = _build_moe_transformer_block_pair(
            moe_aux_loss_coeff=0.01,
            moe_router_load_balancing_type="seq_aux_loss",
            streambp_moe_chunk_forward=False,
            streambp_moe_mlp_chunks=2,
            streambp_moe_mlp_backward_chunks=4,
        )
        _assert_matching_block_backward(baseline, streambp, with_padding_mask=False)
    finally:
        Utils.destroy_model_parallel()


def test_streambp_ddp_readiness_counter_waits_until_final_chunk():
    param = torch.nn.Parameter(torch.ones(2))
    marked = mark_streambp_pending_chunks([param], 3)
    assert marked == [param]
    assert should_streambp_register_grad_ready(param) is False
    assert should_streambp_register_grad_ready(param) is False
    assert should_streambp_register_grad_ready(param) is True
    assert not hasattr(param, "_streambp_pending_chunks")


def test_streambp_moe_hybrid_marks_attention_and_mlp_ddp_chunks_separately(monkeypatch):
    class _HybridLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attention = torch.nn.Linear(2, 2, bias=False)
            self.mlp = torch.nn.Linear(2, 2, bias=False)

    layer = _HybridLayer()
    chunks = iter_streambp_chunks(10, 4)
    monkeypatch.setenv("MEGATRON_STREAMBP_SPLIT_MOE_MLP_ATTENTION_BACKWARD", "1")
    monkeypatch.setenv("MEGATRON_STREAMBP_MOE_ATTENTION_BACKWARD_CHUNK_SIZE", "4")

    marked = streambp_module._streambp_moe_hybrid_pending_marks(
        layer,
        seq_len=10,
        chunks=chunks,
        moe_mlp_chunks=2,
        moe_mlp_backward_chunks=2,
    )

    assert set(marked) == {layer.self_attention.weight, layer.mlp.weight}
    assert should_streambp_register_grad_ready(layer.self_attention.weight) is False
    assert should_streambp_register_grad_ready(layer.self_attention.weight) is False
    assert should_streambp_register_grad_ready(layer.self_attention.weight) is True
    assert should_streambp_register_grad_ready(layer.mlp.weight) is False
    assert should_streambp_register_grad_ready(layer.mlp.weight) is True


def test_streambp_rejects_non_overlap_flashadamw_gradient_release():
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel as MCoreDDP,
    )
    from megatron.core.optimizer.flash_optimizers import enable_gradient_release_mcore_ddp

    class _Optimizer:
        def __init__(self, params):
            self.param_groups = [{"params": list(params)}]

    class _Stub(MCoreDDP):
        def __init__(self, module):
            torch.nn.Module.__init__(self)
            self.module = module

    module = torch.nn.Linear(2, 2)
    module.config = type("_Config", (), {"use_streambp": True})()
    ddp = _Stub(module)

    with pytest.raises(ValueError, match="requires overlap_grad_reduce=True"):
        enable_gradient_release_mcore_ddp(ddp, _Optimizer(module.parameters()))


def test_streambp_config_rejects_inexact_dropout():
    with pytest.raises(ValueError, match="requires hidden_dropout=0.0"):
        TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            use_streambp=True,
            attention_dropout=0.0,
        )


def test_act_eco_te_hook_queues_multiple_pending_forwards():
    torch.manual_seed(9012)
    cfg = Nvfp4ActEcoConfig(block_size=16)
    linear = torch.nn.Linear(16, 4, bias=False)
    install_act_eco_on_te_linear(linear, cfg, quantizer_backend="reference")

    x0 = torch.randn(2, 16, requires_grad=True)
    x1 = torch.randn(2, 16, requires_grad=True)
    loss = 2.0 * linear(x0).sum() + 3.0 * linear(x1).sum()
    loss.backward()

    dy0 = torch.full((2, 4), 2.0)
    dy1 = torch.full((2, 4), 3.0)
    q0 = nvfp4_act_quant_forward(x0.detach(), cfg)
    q1 = nvfp4_act_quant_forward(x1.detach(), cfg)
    expected = (
        dy0.T @ x0.detach()
        + dy1.T @ x1.detach()
        + activation_eco_bias_correction(dy0, x0.detach(), q0)
        + activation_eco_bias_correction(dy1, x1.detach(), q1)
    )
    torch.testing.assert_close(linear.weight.grad, expected, rtol=1e-6, atol=1e-6)


def test_act_eco_te_hook_adds_correction_to_ddp_main_grad():
    torch.manual_seed(9020)
    cfg = Nvfp4ActEcoConfig(block_size=16)
    linear = torch.nn.Linear(16, 4, bias=False)
    linear.weight.main_grad = torch.zeros_like(linear.weight)
    install_act_eco_on_te_linear(linear, cfg, quantizer_backend="reference")

    x0 = torch.randn(2, 16, requires_grad=True)
    x1 = torch.randn(2, 16, requires_grad=True)
    loss = 2.0 * linear(x0).sum() + 3.0 * linear(x1).sum()
    loss.backward()

    dy0 = torch.full((2, 4), 2.0)
    dy1 = torch.full((2, 4), 3.0)
    q0 = nvfp4_act_quant_forward(x0.detach(), cfg)
    q1 = nvfp4_act_quant_forward(x1.detach(), cfg)
    base_grad = dy0.T @ x0.detach() + dy1.T @ x1.detach()
    correction = activation_eco_bias_correction(dy0, x0.detach(), q0)
    correction = correction + activation_eco_bias_correction(dy1, x1.detach(), q1)

    torch.testing.assert_close(linear.weight.grad, base_grad, rtol=1e-6, atol=1e-6)
    pending = pop_act_eco_grad_correction(linear.weight)
    assert pending is None
    torch.testing.assert_close(
        linear.weight.main_grad,
        correction,
        rtol=1e-6,
        atol=1e-6,
    )


def test_act_eco_te_hook_handles_tuple_outputs_and_skips_no_grad_capture():
    class TupleLinear(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    torch.manual_seed(9013)
    linear = TupleLinear(16, 4, bias=False)
    install_act_eco_on_te_linear(
        linear,
        Nvfp4ActEcoConfig(block_size=16),
        capture_recompute_only=True,
        quantizer_backend="reference",
    )

    with torch.no_grad():
        out, bias = linear(torch.randn(2, 16))
    assert bias is None
    assert out.shape == (2, 4)

    x = torch.randn(2, 16, requires_grad=True)
    out, bias = linear(x)
    assert bias is None
    out.sum().backward()

    assert linear.weight.grad is not None
    assert x.grad is not None


def test_act_eco_te_hook_survives_released_input_storage_when_cloning():
    class CloneInputLinear(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x.clone()), None

    torch.manual_seed(9015)
    linear = CloneInputLinear(16, 4, bias=False)
    install_act_eco_on_te_linear(
        linear,
        Nvfp4ActEcoConfig(block_size=16),
        clone_captured_input=True,
        quantizer_backend="reference",
    )

    x = torch.randn(2, 16, requires_grad=True)
    out, _ = linear(x)
    x.untyped_storage().resize_(0)
    out.sum().backward()

    assert linear.weight.grad is not None


def test_act_eco_te_hook_slices_gathered_sequence_parallel_activation(monkeypatch):
    class SequenceParallelLinear(torch.nn.Linear):
        def forward(self, x):
            local = x[2:4]
            return super().forward(local), None

    torch.manual_seed(9017)
    cfg = Nvfp4ActEcoConfig(block_size=16)
    linear = SequenceParallelLinear(16, 4, bias=False)
    install_act_eco_on_te_linear(linear, cfg, quantizer_backend="reference")
    monkeypatch.setattr(act_eco_te_hook, "_tensor_model_parallel_rank_size", lambda: (1, 4))

    x = torch.randn(8, 16, requires_grad=True)
    out, _ = linear(x)
    out.sum().backward()

    x_local = x.detach()[2:4]
    dy = torch.ones(2, 4)
    q_local = nvfp4_act_quant_forward(x_local, cfg)
    expected = dy.T @ x_local + activation_eco_bias_correction(dy, x_local, q_local)
    torch.testing.assert_close(linear.weight.grad, expected, rtol=1e-6, atol=1e-6)


def test_act_eco_row_alignment_slices_gathered_sequence_parallel_grad(monkeypatch):
    monkeypatch.setattr(act_eco_te_hook, "_tensor_model_parallel_rank_size", lambda: (2, 4))

    x = torch.randn(2, 16)
    dy = torch.randn(8, 4)
    aligned_x, aligned_dy = act_eco_te_hook._align_activation_and_grad_rows(
        x,
        dy,
        module_name="ToyColumnParallelLinear",
    )

    assert aligned_x is x
    torch.testing.assert_close(aligned_dy, dy[4:6])


def test_act_eco_grouped_hook_applies_per_expert_correction():
    class ToyGroupedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight0 = torch.nn.Parameter(torch.randn(4, 16))
            self.weight1 = torch.nn.Parameter(torch.randn(4, 16))

        def forward(self, x, m_splits):
            chunks = torch.split(x, m_splits)
            out0 = chunks[0] @ self.weight0.T
            out1 = chunks[1] @ self.weight1.T
            return torch.cat([out0, out1], dim=0), None

    torch.manual_seed(9014)
    grouped = ToyGroupedLinear()
    install_act_eco_on_te_grouped_linear(
        grouped,
        Nvfp4ActEcoConfig(block_size=16),
        num_gemms=2,
        quantizer_backend="reference",
    )

    x = torch.randn(6, 16, requires_grad=True)
    y, bias = grouped(x, [2, 4])
    assert bias is None
    y.sum().backward()

    assert grouped.weight0.grad is not None
    assert grouped.weight1.grad is not None
    assert grouped.weight0.grad.shape == grouped.weight0.shape
    assert grouped.weight1.grad.shape == grouped.weight1.shape


def test_act_eco_grouped_hook_adds_correction_to_ddp_main_grad():
    class ToyGroupedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight0 = torch.nn.Parameter(torch.randn(4, 16))
            self.weight1 = torch.nn.Parameter(torch.randn(4, 16))

        def forward(self, x, m_splits):
            chunks = torch.split(x, m_splits)
            out0 = chunks[0] @ self.weight0.T
            out1 = chunks[1] @ self.weight1.T
            return torch.cat([out0, out1], dim=0), None

    torch.manual_seed(9021)
    cfg = Nvfp4ActEcoConfig(block_size=16)
    grouped = ToyGroupedLinear()
    grouped.weight0.main_grad = torch.zeros_like(grouped.weight0)
    grouped.weight1.main_grad = torch.zeros_like(grouped.weight1)
    install_act_eco_on_te_grouped_linear(
        grouped,
        cfg,
        num_gemms=2,
        quantizer_backend="reference",
    )

    x = torch.randn(6, 16, requires_grad=True)
    y, _ = grouped(x, [2, 4])
    y.sum().backward()

    x0, x1 = torch.split(x.detach(), [2, 4])
    dy0 = torch.ones(2, 4)
    dy1 = torch.ones(4, 4)
    base0 = dy0.T @ x0
    base1 = dy1.T @ x1
    corr0 = activation_eco_bias_correction(dy0, x0, nvfp4_act_quant_forward(x0, cfg))
    corr1 = activation_eco_bias_correction(dy1, x1, nvfp4_act_quant_forward(x1, cfg))

    torch.testing.assert_close(grouped.weight0.grad, base0, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(grouped.weight1.grad, base1, rtol=1e-6, atol=1e-6)
    assert pop_act_eco_grad_correction(grouped.weight0) is None
    assert pop_act_eco_grad_correction(grouped.weight1) is None
    torch.testing.assert_close(grouped.weight0.main_grad, corr0, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(grouped.weight1.main_grad, corr1, rtol=1e-6, atol=1e-6)


def test_act_eco_grouped_hook_survives_released_input_storage_when_cloning():
    class CloneInputGroupedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight0 = torch.nn.Parameter(torch.randn(4, 16))
            self.weight1 = torch.nn.Parameter(torch.randn(4, 16))

        def forward(self, x, m_splits):
            x = x.clone()
            chunks = torch.split(x, m_splits)
            out0 = chunks[0] @ self.weight0.T
            out1 = chunks[1] @ self.weight1.T
            return torch.cat([out0, out1], dim=0), None

    torch.manual_seed(9016)
    grouped = CloneInputGroupedLinear()
    install_act_eco_on_te_grouped_linear(
        grouped,
        Nvfp4ActEcoConfig(block_size=16),
        num_gemms=2,
        clone_captured_input=True,
        quantizer_backend="reference",
    )

    x = torch.randn(6, 16, requires_grad=True)
    y, _ = grouped(x, [2, 4])
    x.untyped_storage().resize_(0)
    y.sum().backward()

    assert grouped.weight0.grad is not None
    assert grouped.weight1.grad is not None
