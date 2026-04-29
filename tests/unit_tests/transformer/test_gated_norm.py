# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core.fusions.gated_norm import apply_gated_norm
from megatron.core.models.gpt.gpt_layer_specs import (
    HAVE_TE,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
    )
except Exception:  # pragma: no cover - TE is optional in local dev environments.
    TEColumnParallelLinear = TELayerNormColumnParallelLinear = None


class SimpleRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def simple_rms_norm_builder(*, config, hidden_size, eps):
    del config
    return SimpleRMSNorm(hidden_size, eps)


class CaptureAttention(nn.Module):
    def __init__(self, config, layer_number=1, **kwargs):
        super().__init__()
        self.last_input = None

    def forward(self, hidden_states, **kwargs):
        self.last_input = hidden_states.detach().clone()
        return hidden_states, None


class CaptureMLP(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.last_input = None

    def forward(self, hidden_states, padding_mask=None, **kwargs):
        del padding_mask
        self.last_input = hidden_states.detach().clone()
        return hidden_states, None


def passthrough_bda(*args, **kwargs):
    del args, kwargs

    def _apply(outputs, residual, hidden_dropout):
        del residual, hidden_dropout
        return outputs[0]

    return _apply


class TestGatedNormTransformerCpu:
    def _make_layer(self, gated_norm=True, sequence_parallel=False):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            ffn_hidden_size=16,
            normalization="RMSNorm",
            gated_norm=gated_norm,
            gated_norm_rank=4,
            sequence_parallel=sequence_parallel,
            tensor_model_parallel_size=2 if sequence_parallel else 1,
            use_cpu_initialization=True,
            transformer_impl="local",
        )
        submodules = TransformerLayerSubmodules(
            input_layernorm=simple_rms_norm_builder,
            self_attention=ModuleSpec(module=CaptureAttention, params={}),
            self_attn_bda=passthrough_bda,
            pre_cross_attn_layernorm=IdentityOp,
            cross_attention=ModuleSpec(module=CaptureAttention, params={}),
            cross_attn_bda=passthrough_bda,
            pre_mlp_layernorm=simple_rms_norm_builder,
            mlp=ModuleSpec(module=CaptureMLP, params={}),
            mlp_bda=passthrough_bda,
        )
        return TransformerLayer(
            config=config,
            submodules=submodules,
            layer_number=1,
            pg_collection=ProcessGroupCollection(tp=None, pp=None),
        )

    def test_gated_norm_requires_rmsnorm(self):
        with pytest.raises(ValueError, match="gated_norm requires normalization == 'RMSNorm'"):
            TransformerConfig(
                num_layers=1,
                hidden_size=8,
                num_attention_heads=2,
                ffn_hidden_size=16,
                normalization="LayerNorm",
                gated_norm=True,
            )

    def test_gated_norm_rank_validation(self):
        with pytest.raises(ValueError, match="gated_norm_rank must be positive"):
            TransformerConfig(
                num_layers=1,
                hidden_size=8,
                num_attention_heads=2,
                ffn_hidden_size=16,
                normalization="RMSNorm",
                gated_norm=True,
                gated_norm_rank=0,
            )

        with pytest.raises(ValueError, match="gated_norm_rank must be less than or equal"):
            TransformerConfig(
                num_layers=1,
                hidden_size=8,
                num_attention_heads=2,
                ffn_hidden_size=16,
                normalization="RMSNorm",
                gated_norm=True,
                gated_norm_rank=16,
            )

    def test_gated_norm_modules_created_when_enabled(self):
        layer = self._make_layer(gated_norm=True)

        assert layer.input_gated_norm_down is not None
        assert layer.input_gated_norm_up is not None
        assert layer.pre_mlp_gated_norm_down is not None
        assert layer.pre_mlp_gated_norm_up is not None
        assert layer.input_gated_norm_down.weight.shape == (4, 8)
        assert layer.input_gated_norm_up.weight.shape == (8, 4)

    def test_gated_norm_pair_marks_sequence_parallel_weights(self):
        layer = object.__new__(TransformerLayer)
        layer.config = SimpleNamespace(
            hidden_size=8,
            gated_norm_rank=4,
            perform_initialization=False,
            params_dtype=torch.float32,
            sequence_parallel=True,
        )

        gate_down, gate_up = TransformerLayer._build_gated_norm_pair(layer)

        assert getattr(gate_down.weight, "sequence_parallel", False) is True
        assert getattr(gate_up.weight, "sequence_parallel", False) is True

    def test_gated_norm_state_dict_keys_when_enabled(self):
        layer = self._make_layer(gated_norm=True)

        keys = set(layer.state_dict().keys())
        assert "input_gated_norm_down.weight" in keys
        assert "input_gated_norm_up.weight" in keys
        assert "pre_mlp_gated_norm_down.weight" in keys
        assert "pre_mlp_gated_norm_up.weight" in keys
        assert "input_layernorm.weight" in keys
        assert "pre_mlp_layernorm.weight" in keys

    def test_gated_norm_modules_absent_when_disabled(self):
        layer = self._make_layer(gated_norm=False)

        assert layer.input_gated_norm_down is None
        assert layer.input_gated_norm_up is None
        assert layer.pre_mlp_gated_norm_down is None
        assert layer.pre_mlp_gated_norm_up is None

    def test_qk_layernorm_not_gated(self):
        layer = self._make_layer(gated_norm=True)

        assert hasattr(layer, "input_gated_norm_down")
        assert not hasattr(layer.self_attention, "q_gated_norm_down")
        assert not hasattr(layer.self_attention, "k_gated_norm_down")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for transformer gated_norm tests"
)
class TestGatedNormTransformer:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _make_layer(self, gated_norm=True, sequence_parallel=False, recompute=False):
        config_kwargs = dict(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            ffn_hidden_size=16,
            normalization="RMSNorm",
            gated_norm=gated_norm,
            gated_norm_rank=4,
            sequence_parallel=sequence_parallel,
            use_cpu_initialization=True,
            transformer_impl="local",
        )
        if recompute:
            config_kwargs.update(
                recompute_granularity="selective",
                recompute_modules=["layernorm"],
            )

        config = TransformerConfig(**config_kwargs)
        submodules = TransformerLayerSubmodules(
            input_layernorm=simple_rms_norm_builder,
            self_attention=ModuleSpec(module=CaptureAttention, params={}),
            self_attn_bda=passthrough_bda,
            pre_cross_attn_layernorm=IdentityOp,
            cross_attention=ModuleSpec(module=CaptureAttention, params={}),
            cross_attn_bda=passthrough_bda,
            pre_mlp_layernorm=simple_rms_norm_builder,
            mlp=ModuleSpec(module=CaptureMLP, params={}),
            mlp_bda=passthrough_bda,
        )
        return TransformerLayer(config=config, submodules=submodules, layer_number=1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for forward tests")
    def test_gated_norm_called_after_input_layernorm(self):
        layer = self._make_layer(gated_norm=True).cuda()
        x = torch.randn(4, 2, 8, device="cuda")

        layer._forward_attention(x)

        expected = apply_gated_norm(
            layer.input_layernorm(x),
            layer.input_gated_norm_down.weight,
            layer.input_gated_norm_up.weight,
        )
        assert torch.allclose(layer.self_attention.last_input, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for forward tests")
    def test_gated_norm_called_after_pre_mlp_layernorm(self):
        layer = self._make_layer(gated_norm=True).cuda()
        x = torch.randn(4, 2, 8, device="cuda")

        attn_out, _ = layer._forward_attention(x)
        layer._forward_mlp(attn_out)

        expected = apply_gated_norm(
            layer.pre_mlp_layernorm(attn_out),
            layer.pre_mlp_gated_norm_down.weight,
            layer.pre_mlp_gated_norm_up.weight,
        )
        assert torch.allclose(layer.mlp.last_input, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for forward tests")
    def test_layernorm_recompute_with_gated_norm(self):
        layer = self._make_layer(gated_norm=True, recompute=True).cuda()
        x = torch.randn(4, 2, 8, device="cuda", requires_grad=True)

        attn_out, _ = layer._forward_attention(x)
        out = layer._forward_mlp(attn_out)
        loss = out.sum()
        loss.backward()

        assert layer.recompute_input_layernorm is True
        assert layer.recompute_pre_mlp_layernorm is True
        assert layer.input_gated_norm_down.weight.grad is not None
        assert layer.pre_mlp_gated_norm_up.weight.grad is not None

    @pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required for TE spec checks")
    def test_te_fused_layernorm_linear_disabled_with_gated_norm(self):
        spec = get_gpt_layer_with_transformer_engine_submodules(
            gated_norm=True,
            use_te_op_fuser=True,
        )

        assert spec.self_attention.submodules.linear_qkv is TEColumnParallelLinear
        assert spec.self_attention.submodules.linear_qkv is not TELayerNormColumnParallelLinear
        assert spec.mlp.submodules.linear_fc1 is TEColumnParallelLinear
        assert spec.mlp.submodules.linear_fc1 is not TELayerNormColumnParallelLinear
        assert spec.pre_mlp_layernorm is not IdentityOp
