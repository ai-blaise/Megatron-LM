# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.nvfp4_act_eco.codec import Nvfp4ActEcoConfig
from megatron.core.quantization.nvfp4_act_eco.te_hook import install_act_eco_on_te_linear
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
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
    iter_streambp_chunks,
    make_streambp_packed_seq_params,
    make_streambp_single_sequence_packed_seq_params,
    mark_streambp_pending_chunks,
    moe_streambp_requires_full_replay,
    should_streambp_register_grad_ready,
    slice_streambp_attention_mask,
    slice_streambp_padding_mask,
    streambp_checkpoint_layer,
    streambp_lm_head_loss,
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


def test_streambp_ddp_readiness_counter_waits_until_final_chunk():
    param = torch.nn.Parameter(torch.ones(2))
    marked = mark_streambp_pending_chunks([param], 3)
    assert marked == [param]
    assert should_streambp_register_grad_ready(param) is False
    assert should_streambp_register_grad_ready(param) is False
    assert should_streambp_register_grad_ready(param) is True
    assert not hasattr(param, "_streambp_pending_chunks")


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
    linear = torch.nn.Linear(16, 4, bias=False)
    install_act_eco_on_te_linear(linear, Nvfp4ActEcoConfig(block_size=16))

    x0 = torch.randn(2, 16, requires_grad=True)
    x1 = torch.randn(2, 16, requires_grad=True)
    loss = linear(x0).sum() + linear(x1).sum()
    loss.backward()

    assert linear.weight.grad is not None
