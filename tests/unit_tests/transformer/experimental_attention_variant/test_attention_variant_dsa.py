# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerAuxLossState,
    DSAIndexerLossAutoScaler,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
    FusedDSAIndexerLoss,
    _compute_index_scores,
    _hisa_selected_score_backward_cuda,
    _hisa_selected_score_backward_cuda_batched,
    _sparse_dsa_attention_chunk,
    _streaming_qk_topk,
    chunked_dsa_forward,
    compute_dsa_indexer_loss,
    fused_qk_topk_naive,
    rotate_activation,
)
from megatron.core.transformer.experimental_attention_variant.dsa_triton import (
    dsa_indexer_scores_triton,
    is_dsa_indexer_scores_triton_supported,
    is_sparse_dsa_triton_supported,
    sparse_dsa_attention_with_teacher_triton,
    sparse_dsa_attention_triton,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.quantization.indexcache import IndexCacheHISAConfig, indexcache_hisa_topk
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False
    _hadamard_transform = None


def mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Mock implementation of hadamard_transform for testing without the library installed.

    This is a simple identity-like transformation that preserves shape and applies scaling.
    """
    return x * scale


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    """Automatically patch hadamard_transform in dsa module if not installed."""
    if not HAVE_HADAMARD:
        with patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
            mock_hadamard_transform,
        ):
            yield
    else:
        yield


class TestRotateActivation:
    """Test rotate_activation function."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_rotate_activation_shape(self):
        """Test that rotate_activation preserves shape."""
        batch_size = 2
        seq_len = 16
        hidden_size = 128

        x = torch.randn(seq_len, batch_size, hidden_size, dtype=torch.bfloat16).cuda()
        output = rotate_activation(x)

        assert output.shape == x.shape
        assert output.dtype == torch.bfloat16

    def test_rotate_activation_dtype_check(self):
        """Test that rotate_activation only accepts bfloat16."""
        x = torch.randn(16, 2, 128, dtype=torch.float32).cuda()

        with pytest.raises(AssertionError, match="only support bf16"):
            rotate_activation(x)


@pytest.mark.parametrize("seqlen_and_topk", [[16, 32], [64, 32]])
class TestComputeDSAIndexerLoss:
    """Test compute_dsa_indexer_loss function."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_loss_shape(self, seqlen_and_topk):
        """Test that indexer loss returns a scalar."""
        batch_size = 2
        seqlen = seqlen_and_topk[0]
        num_heads = 4
        head_dim = 128
        index_topk = seqlen_and_topk[1]

        # Create dummy index scores
        index_scores = torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32).cuda()

        # Apply causal mask to index_scores before computing topk
        causal_mask = torch.triu(
            torch.full(
                (seqlen, seqlen), float('-inf'), dtype=torch.float32, device=index_scores.device
            ),
            diagonal=1,
        )
        # [batch_size, seqlen, seqlen] + [seqlen, seqlen] -> [batch_size, seqlen, seqlen]
        masked_index_scores = index_scores + causal_mask

        # Get topk indices from masked index_scores
        topk_k = min(index_topk, seqlen)
        topk_indices = masked_index_scores.topk(topk_k, dim=-1)[1]

        query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        key = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        softmax_scale = head_dim**-0.5

        loss = compute_dsa_indexer_loss(
            index_scores=index_scores,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=softmax_scale,
            loss_coeff=1.0,
            sparse_loss=False,
            pg_collection=self.pg_collection,
        )

        assert loss.shape == torch.Size([])
        assert loss.dtype == torch.float32
        assert loss >= 0  # KL divergence should be non-negative

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_loss_sparse(self, seqlen_and_topk):
        """Test sparse indexer loss computation."""
        batch_size = 2
        seqlen = seqlen_and_topk[0]
        num_heads = 4
        head_dim = 128
        index_topk = seqlen_and_topk[1]

        # Create dummy index scores
        index_scores = torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32).cuda()

        # Apply causal mask to index_scores before computing topk
        causal_mask = torch.triu(
            torch.full(
                (seqlen, seqlen), float('-inf'), dtype=torch.float32, device=index_scores.device
            ),
            diagonal=1,
        )
        # [batch_size, seqlen, seqlen] + [seqlen, seqlen] -> [batch_size, seqlen, seqlen]
        masked_index_scores = index_scores + causal_mask

        # Get topk indices from masked index_scores
        topk_k = min(index_topk, seqlen)
        topk_indices = masked_index_scores.topk(topk_k, dim=-1)[1]

        query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        key = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        softmax_scale = head_dim**-0.5

        loss_sparse = compute_dsa_indexer_loss(
            index_scores=index_scores,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=softmax_scale,
            loss_coeff=1.0,
            sparse_loss=True,
            pg_collection=self.pg_collection,
        )

        loss_dense = compute_dsa_indexer_loss(
            index_scores=index_scores,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=softmax_scale,
            loss_coeff=1.0,
            sparse_loss=False,
            pg_collection=self.pg_collection,
        )

        # Sparse loss should be different from dense loss
        if seqlen > index_topk:
            assert loss_sparse != loss_dense
        else:
            assert loss_sparse == loss_dense
        assert loss_sparse >= 0
        assert loss_dense >= 0


class TestDSAIndexerLossAutoScaler:
    """Test DSAIndexerLossAutoScaler autograd function."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_pass(self):
        """Test that forward pass preserves output."""
        output = torch.randn(16, 2, 128).cuda()
        output.requires_grad_(True)
        indexer_loss = torch.tensor(0.5).cuda()
        indexer_loss.requires_grad_(True)

        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        assert torch.allclose(result, output, atol=0, rtol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_pass(self):
        """Test that backward pass triggers indexer loss backward and scales gradient correctly."""
        output = torch.randn(16, 2, 128).cuda()
        output.requires_grad_(True)

        # Create indexer_loss with computation graph
        # This simulates compute_dsa_indexer_loss which computes KL divergence
        dummy_input = torch.randn(10).cuda()
        dummy_input.requires_grad_(True)
        indexer_loss = dummy_input.mean()

        # Set loss scale
        scale = torch.tensor(2.0).cuda()
        DSAIndexerLossAutoScaler.set_loss_scale(scale)

        # Apply the autograd function
        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        # Trigger backward
        main_loss = result.sum()
        main_loss.backward()

        # Check that gradients flow back to output
        assert output.grad is not None, "Gradient should flow back to parameters"

        # Check that indexer_loss backward was triggered
        assert dummy_input.grad is not None, "Indexer loss backward should be triggered"

        # Verify the gradient is scaled correctly
        expected_grad_per_element = scale.item() / len(dummy_input)
        assert torch.allclose(
            dummy_input.grad,
            torch.full_like(dummy_input, expected_grad_per_element),
            rtol=0,
            atol=0,
        ), f"Gradient should be scaled by loss scale, expected {expected_grad_per_element}, got {dummy_input.grad[0].item()}"


class TestSparseDSATritonAttention:
    """Test the fused sparse DSA attention kernel."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_indexer_scores_match_dense_scores(self, monkeypatch):
        torch.manual_seed(1357)
        seqlen = 29
        q_len = 13
        batch_size = 2
        index_n_heads = 4
        index_head_dim = 32
        q_start = 7

        q = torch.randn(
            q_len, batch_size, index_n_heads, index_head_dim, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(seqlen, batch_size, index_head_dim, device="cuda", dtype=torch.bfloat16)
        weights = torch.rand(q_len, batch_size, index_n_heads, device="cuda", dtype=torch.bfloat16)

        monkeypatch.setenv("MEGATRON_DSA_TRITON_INDEXER", "1")
        monkeypatch.setenv("MEGATRON_DSA_TRITON_INDEXER_BLOCK_Q", "4")
        monkeypatch.setenv("MEGATRON_DSA_TRITON_INDEXER_BLOCK_K", "8")
        assert is_dsa_indexer_scores_triton_supported(q, weights, k, None, True)

        fused_scores = dsa_indexer_scores_triton(q, weights, k, q_start=q_start)
        fused_scores_out = torch.empty_like(fused_scores)
        returned_scores = dsa_indexer_scores_triton(
            q, weights, k, q_start=q_start, out=fused_scores_out
        )
        assert returned_scores is fused_scores_out
        dense_scores = _compute_index_scores(q, weights, k)
        q_pos = torch.arange(q_start, q_start + q_len, device="cuda").view(1, -1, 1)
        k_pos = torch.arange(seqlen, device="cuda").view(1, 1, -1)
        dense_scores = dense_scores.masked_fill(k_pos > q_pos, float("-inf"))

        finite = torch.isfinite(dense_scores)
        assert torch.equal(torch.isfinite(fused_scores), finite)
        assert torch.allclose(fused_scores[finite], dense_scores[finite], atol=2e-1, rtol=8e-2)
        assert torch.equal(torch.isfinite(fused_scores_out), finite)
        assert torch.allclose(
            fused_scores_out[finite], dense_scores[finite], atol=2e-1, rtol=8e-2
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_indexer_scores_honor_positions(self, monkeypatch):
        torch.manual_seed(9753)
        seqlen = 16
        q_len = 8
        batch_size = 1
        index_n_heads = 3
        index_head_dim = 32
        query_positions = torch.tensor(
            [0, 1, 2, 3, 12, 13, 14, 15], device="cuda", dtype=torch.long
        )
        key_positions = torch.tensor(
            [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11],
            device="cuda",
            dtype=torch.long,
        )

        q = torch.randn(
            q_len, batch_size, index_n_heads, index_head_dim, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(seqlen, batch_size, index_head_dim, device="cuda", dtype=torch.bfloat16)
        weights = torch.rand(q_len, batch_size, index_n_heads, device="cuda", dtype=torch.bfloat16)

        monkeypatch.setenv("MEGATRON_DSA_TRITON_INDEXER", "1")
        monkeypatch.setenv("MEGATRON_DSA_TRITON_INDEXER_BLOCK_Q", "4")
        monkeypatch.setenv("MEGATRON_DSA_TRITON_INDEXER_BLOCK_K", "8")
        assert is_dsa_indexer_scores_triton_supported(
            q, weights, k, None, True, query_positions=query_positions, key_positions=key_positions
        )

        fused_scores = dsa_indexer_scores_triton(
            q,
            weights,
            k,
            q_start=0,
            query_positions=query_positions,
            key_positions=key_positions,
        )
        dense_scores = _compute_index_scores(q, weights, k)
        dense_scores = dense_scores.masked_fill(
            key_positions.view(1, 1, -1) > query_positions.view(1, -1, 1),
            float("-inf"),
        )

        finite = torch.isfinite(dense_scores)
        assert torch.equal(torch.isfinite(fused_scores), finite)
        assert torch.allclose(fused_scores[finite], dense_scores[finite], atol=2e-1, rtol=8e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_streaming_indexer_topk_matches_dense_scores(self, monkeypatch):
        torch.manual_seed(4321)
        seqlen = 37
        q_len = 11
        batch_size = 1
        index_n_heads = 4
        index_head_dim = 128
        topk = 8
        q_start = 9
        q_end = q_start + q_len

        q = torch.randn(
            q_len, batch_size, index_n_heads, index_head_dim, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(seqlen, batch_size, index_head_dim, device="cuda", dtype=torch.bfloat16)
        weights = torch.rand(q_len, batch_size, index_n_heads, device="cuda", dtype=torch.bfloat16)

        monkeypatch.setenv("MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE", "7")
        streaming_indices = _streaming_qk_topk(
            q, weights, k, topk, mask=None, q_start=q_start, q_end=q_end, sk=seqlen, is_causal=True
        )

        dense_scores = _compute_index_scores(q, weights, k)
        q_pos = torch.arange(q_start, q_end, device="cuda").view(1, -1, 1)
        k_pos = torch.arange(seqlen, device="cuda").view(1, 1, -1)
        dense_scores = dense_scores.masked_fill(k_pos > q_pos, float("-inf"))
        dense_scores, dense_indices = dense_scores.topk(topk, dim=-1)

        streaming_scores = dense_scores.new_empty(dense_scores.shape)
        full_dense_scores = _compute_index_scores(q, weights, k).masked_fill(
            k_pos > q_pos, float("-inf")
        )
        torch.gather(full_dense_scores, -1, streaming_indices, out=streaming_scores)

        assert torch.allclose(
            streaming_scores.sort(dim=-1).values,
            dense_scores.sort(dim=-1).values,
            atol=0,
            rtol=0,
        )
        assert torch.equal(streaming_indices.sort(dim=-1).values, dense_indices.sort(dim=-1).values)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_chunked_forward_honors_cp_zigzag_positions(self, monkeypatch):
        torch.manual_seed(2468)
        seqlen = 16
        num_heads = 2
        qk_head_dim = 128
        value_head_dim = 128
        index_n_heads = 4
        index_head_dim = 128
        topk = 8
        chunk_size = 4
        softmax_scale = qk_head_dim**-0.5

        rank0_positions = torch.tensor(
            [0, 1, 2, 3, 12, 13, 14, 15], device="cuda", dtype=torch.long
        )
        rank1_positions = torch.tensor(
            [4, 5, 6, 7, 8, 9, 10, 11], device="cuda", dtype=torch.long
        )
        gathered_positions = torch.cat((rank0_positions, rank1_positions), dim=0)
        natural_positions = torch.arange(seqlen, device="cuda", dtype=torch.long)

        q_full = torch.randn(
            seqlen, 1, index_n_heads, index_head_dim, device="cuda", dtype=torch.bfloat16
        )
        k_full = torch.randn(seqlen, 1, index_head_dim, device="cuda", dtype=torch.bfloat16)
        weights_full = torch.rand(
            seqlen, 1, index_n_heads, device="cuda", dtype=torch.bfloat16
        )
        query_full = torch.randn(
            seqlen, 1, num_heads, qk_head_dim, device="cuda", dtype=torch.bfloat16
        )
        key_full = torch.randn(
            seqlen, 1, num_heads, qk_head_dim, device="cuda", dtype=torch.bfloat16
        )
        value_full = torch.randn(
            seqlen, 1, num_heads, value_head_dim, device="cuda", dtype=torch.bfloat16
        )

        q_local = q_full.index_select(0, rank0_positions)
        weights_local = weights_full.index_select(0, rank0_positions)
        query_local = query_full.index_select(0, rank0_positions)

        monkeypatch.setenv("MEGATRON_DSA_TRITON", "0")
        monkeypatch.setenv("MEGATRON_DSA_STREAMING_INDEXER_TOPK", "1")
        monkeypatch.setenv("MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE", "5")
        reference, _ = chunked_dsa_forward(
            q_local,
            k_full,
            weights_local,
            query_local,
            key_full,
            value_full,
            softmax_scale,
            topk,
            mask=None,
            is_causal=True,
            loss_coeff=0.0,
            sparse_loss=False,
            pg_collection=None,
            chunk_size=chunk_size,
            query_positions=rank0_positions,
            key_positions=natural_positions,
        )

        gathered, _ = chunked_dsa_forward(
            q_local,
            k_full.index_select(0, gathered_positions),
            weights_local,
            query_local,
            key_full.index_select(0, gathered_positions),
            value_full.index_select(0, gathered_positions),
            softmax_scale,
            topk,
            mask=None,
            is_causal=True,
            loss_coeff=0.0,
            sparse_loss=False,
            pg_collection=None,
            chunk_size=chunk_size,
            query_positions=rank0_positions,
            key_positions=gathered_positions,
        )

        assert torch.allclose(gathered, reference, atol=8e-2, rtol=8e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_path_with_cp_positions_matches_fallback_backward(self, monkeypatch):
        torch.manual_seed(1357)
        seqlen = 16
        q_len = 8
        num_heads = 2
        qk_head_dim = 128
        value_head_dim = 128
        topk = 8
        softmax_scale = qk_head_dim**-0.5

        query_positions = torch.tensor(
            [0, 1, 2, 3, 12, 13, 14, 15], device="cuda", dtype=torch.long
        )
        key_positions = torch.tensor(
            [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11],
            device="cuda",
            dtype=torch.long,
        )

        query = torch.randn(
            q_len,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        index_scores = torch.randn(1, q_len, seqlen, device="cuda", dtype=torch.float32)
        index_scores = index_scores.masked_fill(
            key_positions.view(1, 1, -1) > query_positions.view(1, -1, 1), float("-inf")
        )
        topk_indices = index_scores.topk(topk, dim=-1).indices
        grad_output = torch.randn(
            q_len, 1, num_heads * value_head_dim, device="cuda", dtype=torch.bfloat16
        )

        monkeypatch.setenv("MEGATRON_DSA_TRITON", "0")
        reference = _sparse_dsa_attention_chunk(
            query,
            key,
            value,
            topk_indices,
            softmax_scale,
            mask=None,
            q_start=0,
            is_causal=True,
            query_positions=query_positions,
            key_positions=key_positions,
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        monkeypatch.setenv("MEGATRON_DSA_TRITON", "1")
        assert is_sparse_dsa_triton_supported(
            query_fused,
            key_fused,
            value_fused,
            topk_indices,
            mask=None,
            is_causal=True,
            query_positions=query_positions,
            key_positions=key_positions,
        )
        fused = sparse_dsa_attention_triton(
            query_fused,
            key_fused,
            value_fused,
            topk_indices,
            softmax_scale,
            query_positions=query_positions,
            key_positions=key_positions,
        )
        (fused * grad_output).sum().backward()

        assert torch.allclose(fused, reference, atol=8e-2, rtol=8e-2)
        assert torch.allclose(query_fused.grad, reference_grads[0], atol=8e-2, rtol=8e-2)
        assert torch.allclose(key_fused.grad, reference_grads[1], atol=8e-2, rtol=8e-2)
        assert torch.allclose(value_fused.grad, reference_grads[2], atol=8e-2, rtol=8e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_path_emits_teacher_probs_from_selected_attention(self, monkeypatch):
        import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

        torch.manual_seed(20260520)
        seqlen = 33
        q_len = 9
        num_heads = 3
        qk_head_dim = 64
        value_head_dim = 64
        topk = 11
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            q_len,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        index_scores = torch.randn(1, q_len, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(q_len, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        index_scores = index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        topk_indices = index_scores.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        query_positions = torch.arange(q_len, device="cuda").view(1, q_len, 1)
        topk_indices = topk_indices.masked_fill(topk_indices > query_positions, -1)

        monkeypatch.setenv("MEGATRON_DSA_TRITON", "1")
        monkeypatch.setenv("MEGATRON_DSA_TEACHER_FROM_LSE", "0")
        monkeypatch.setenv("MEGATRON_DSA_TEACHER_SCORE_SCRATCH", "0")
        fused, teacher = sparse_dsa_attention_with_teacher_triton(
            query,
            key,
            value,
            topk_indices,
            softmax_scale,
        )
        monkeypatch.setenv("MEGATRON_DSA_TEACHER_SCORE_SCRATCH", "1")
        fused_scratch, teacher_scratch = sparse_dsa_attention_with_teacher_triton(
            query,
            key,
            value,
            topk_indices,
            softmax_scale,
        )
        monkeypatch.setenv("MEGATRON_DSA_TEACHER_SCORE_SCRATCH", "0")
        monkeypatch.setenv("MEGATRON_DSA_TEACHER_FROM_LSE", "1")
        fused_lse, teacher_lse = sparse_dsa_attention_with_teacher_triton(
            query,
            key,
            value,
            topk_indices,
            softmax_scale,
        )
        reference = sparse_dsa_attention_triton(query, key, value, topk_indices, softmax_scale)
        teacher_ref = dsa_module._hisa_attention_target_probs(
            query.detach(),
            key.detach(),
            topk_indices,
            softmax_scale,
            tp_group=None,
        )

        teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        teacher_scratch = teacher_scratch / teacher_scratch.sum(dim=-1, keepdim=True).clamp_min(
            1e-20
        )
        teacher_lse = teacher_lse / teacher_lse.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        assert teacher.requires_grad is False
        assert teacher_scratch.requires_grad is False
        assert teacher_lse.requires_grad is False
        torch.testing.assert_close(fused, reference, rtol=0, atol=0)
        torch.testing.assert_close(fused_scratch, reference, rtol=0, atol=0)
        torch.testing.assert_close(fused_lse, reference, rtol=0, atol=0)
        # These two teacher paths use the same selected logits and LSE values,
        # but reduce/normalize fp32 terms in a different order. Keep this
        # tolerance at fp32 roundoff scale so the test still catches semantic
        # drift.
        # The scratch path reuses stored fp32 selected logits instead of
        # recomputing QK, so it can differ from the recompute path at fp32
        # roundoff scale after per-head atomic accumulation.
        torch.testing.assert_close(teacher_scratch, teacher, rtol=1e-6, atol=2e-7)
        torch.testing.assert_close(teacher_lse, teacher, rtol=1e-6, atol=2e-7)
        torch.testing.assert_close(teacher, teacher_ref, rtol=5e-3, atol=5e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_teacher_score_scratch_backward_matches_recompute(self, monkeypatch):
        torch.manual_seed(20260521)
        seqlen = 32
        q_len = 16
        num_heads = 2
        qk_head_dim = 128
        value_head_dim = 128
        topk = 12
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            q_len,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        index_scores = torch.randn(1, q_len, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(q_len, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        topk_indices = index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf")).topk(
            topk, dim=-1, sorted=False
        ).indices
        grad_output = torch.randn(
            q_len, 1, num_heads * value_head_dim, device="cuda", dtype=torch.bfloat16
        )

        def run(use_score_scratch: bool):
            q = query.detach().clone().requires_grad_(True)
            k = key.detach().clone().requires_grad_(True)
            v = value.detach().clone().requires_grad_(True)
            monkeypatch.setenv("MEGATRON_DSA_TRITON", "1")
            monkeypatch.setenv("MEGATRON_DSA_TEACHER_FROM_LSE", "0")
            monkeypatch.setenv(
                "MEGATRON_DSA_TEACHER_SCORE_SCRATCH", "1" if use_score_scratch else "0"
            )
            monkeypatch.setenv("MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH", "1")
            out, teacher = sparse_dsa_attention_with_teacher_triton(
                q,
                k,
                v,
                topk_indices,
                softmax_scale,
            )
            (out * grad_output).sum().backward()
            return out.detach(), teacher.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()

        ref = run(False)
        got = run(True)
        torch.testing.assert_close(got[0], ref[0], rtol=0, atol=0)
        torch.testing.assert_close(got[1], ref[1], rtol=1e-6, atol=2e-7)
        for actual_grad, expected_grad in zip(got[2:], ref[2:]):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=8e-2, atol=8e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_path_accepts_compact_topk_indices(self, monkeypatch):
        torch.manual_seed(24601)
        seqlen = 32
        q_len = 16
        num_heads = 2
        qk_head_dim = 128
        value_head_dim = 128
        topk = 12
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            q_len,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        index_scores = torch.randn(1, q_len, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(q_len, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        index_scores = index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        topk_indices = index_scores.topk(topk, dim=-1, sorted=False).indices
        compact_topk_indices = topk_indices.to(torch.int16)
        grad_output = torch.randn(
            q_len, 1, num_heads * value_head_dim, device="cuda", dtype=torch.bfloat16
        )

        reference = _sparse_dsa_attention_chunk(
            query,
            key,
            value,
            topk_indices,
            softmax_scale,
            mask=None,
            q_start=0,
            is_causal=True,
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        monkeypatch.setenv("MEGATRON_DSA_TRITON", "1")
        assert is_sparse_dsa_triton_supported(
            query_fused,
            key_fused,
            value_fused,
            compact_topk_indices,
            mask=None,
            is_causal=True,
        )
        fused = sparse_dsa_attention_triton(
            query_fused,
            key_fused,
            value_fused,
            compact_topk_indices,
            softmax_scale,
        )
        (fused * grad_output).sum().backward()

        assert torch.allclose(fused, reference, atol=8e-2, rtol=8e-2)
        assert torch.allclose(query_fused.grad, reference_grads[0], atol=8e-2, rtol=8e-2)
        assert torch.allclose(key_fused.grad, reference_grads[1], atol=8e-2, rtol=8e-2)
        assert torch.allclose(value_fused.grad, reference_grads[2], atol=8e-2, rtol=8e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_path_narrows_long_topk_before_saving(self, monkeypatch):
        torch.manual_seed(20260516)
        seqlen = 32
        q_len = 8
        num_heads = 2
        qk_head_dim = 64
        value_head_dim = 64
        topk = 8
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            q_len,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        topk_indices = torch.randint(
            0, seqlen, (1, q_len, topk), device="cuda", dtype=torch.long
        )
        grad_output = torch.randn(
            q_len, 1, num_heads * value_head_dim, device="cuda", dtype=torch.bfloat16
        )
        saved_topk_dtypes = []

        def pack_hook(tensor):
            if tensor.shape == topk_indices.shape:
                saved_topk_dtypes.append(tensor.dtype)
            return tensor

        monkeypatch.setenv("MEGATRON_DSA_TRITON", "1")
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
            fused = sparse_dsa_attention_triton(
                query,
                key,
                value,
                topk_indices,
                softmax_scale,
            )
            (fused * grad_output).sum().backward()

        assert topk_indices.dtype == torch.long
        assert saved_topk_dtypes
        assert torch.long not in saved_topk_dtypes
        assert set(saved_topk_dtypes) == {torch.int32}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_path_masks_padded_topk_indices(self, monkeypatch):
        torch.manual_seed(20260516)
        seqlen = 16
        q_len = 4
        num_heads = 2
        qk_head_dim = 64
        value_head_dim = 64
        topk = 8
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            q_len,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        topk_indices = torch.tensor(
            [
                [
                    [0, -1, -1, -1, -1, -1, -1, -1],
                    [0, 1, -1, -1, -1, -1, -1, -1],
                    [0, 2, 1, -1, -1, -1, -1, -1],
                    [3, 2, 1, 0, -1, -1, -1, -1],
                ]
            ],
            device="cuda",
            dtype=torch.long,
        )
        grad_output = torch.randn(
            q_len, 1, num_heads * value_head_dim, device="cuda", dtype=torch.bfloat16
        )

        reference = _sparse_dsa_attention_chunk(
            query,
            key,
            value,
            topk_indices,
            softmax_scale,
            mask=None,
            q_start=0,
            is_causal=True,
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        monkeypatch.setenv("MEGATRON_DSA_TRITON", "1")
        assert is_sparse_dsa_triton_supported(
            query_fused, key_fused, value_fused, topk_indices, mask=None, is_causal=True
        )
        fused = sparse_dsa_attention_triton(
            query_fused,
            key_fused,
            value_fused,
            topk_indices,
            softmax_scale,
        )
        (fused * grad_output).sum().backward()

        assert torch.allclose(fused, reference, atol=8e-2, rtol=8e-2)
        assert torch.allclose(query_fused.grad, reference_grads[0], atol=8e-2, rtol=8e-2)
        assert torch.allclose(key_fused.grad, reference_grads[1], atol=8e-2, rtol=8e-2)
        assert torch.allclose(value_fused.grad, reference_grads[2], atol=8e-2, rtol=8e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hisa_dispatches_to_triton_with_padded_candidates_and_indexer_loss(
        self, monkeypatch
    ):
        import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

        class _ProcessGroup:
            def size(self):
                return 1

        class _ProcessGroups:
            tp = _ProcessGroup()

        torch.manual_seed(20260515)
        seqlen = 64
        q_len = 8
        index_heads = 2
        index_dim = 32
        num_heads = 2
        qk_head_dim = 64
        value_head_dim = 64
        topk = 48
        softmax_scale = qk_head_dim**-0.5
        hisa_config = IndexCacheHISAConfig(
            enabled=True,
            block_size=16,
            compression_ratio=4.0,
            fallback_to_dense_if_short=False,
        )

        q = torch.randn(
            q_len,
            1,
            index_heads,
            index_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        index_k = torch.randn(
            seqlen, 1, index_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        weights = (
            torch.rand(q_len, 1, index_heads, device="cuda", dtype=torch.bfloat16) + 0.1
        ).requires_grad_()
        query = torch.randn(
            q_len,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        selected = indexcache_hisa_topk(
            q.detach(),
            weights.detach(),
            index_k.detach(),
            topk,
            config=hisa_config,
            q_start=0,
            is_causal=True,
            mask=None,
            query_positions=None,
            key_positions=None,
        )
        assert selected is not None
        assert bool((selected < 0).any().item())
        assert is_sparse_dsa_triton_supported(query, key, value, selected, None, True)

        def fail_fallback(*_args, **_kwargs):
            raise AssertionError("HISA DSA must use the fused Triton selected-attention path")

        def fail_dense_indexer(*_args, **_kwargs):
            raise AssertionError("HISA indexer training must not use dense index-score KL")

        monkeypatch.setattr(dsa_module, "_sparse_dsa_attention_chunk", fail_fallback)
        monkeypatch.setattr(dsa_module, "_compute_index_scores", fail_dense_indexer)
        monkeypatch.setenv("MEGATRON_DSA_TRITON", "1")

        output, indexer_loss = dsa_module.chunked_dsa_forward(
            q,
            index_k,
            weights,
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            topk=topk,
            mask=None,
            is_causal=True,
            loss_coeff=0.1,
            sparse_loss=True,
            pg_collection=_ProcessGroups(),
            chunk_size=q_len,
            indexcache_hisa_config=hisa_config,
        )
        assert indexer_loss is not None
        assert torch.isfinite(output.float()).all().item()
        (output.float().square().mean() + indexer_loss).backward()
        for tensor in (q, index_k, weights):
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad.float()).all().item()
        for tensor in (query, key, value):
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad.float()).all().item()
            assert tensor.grad.float().abs().sum().item() > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hisa_deferred_teacher_path_supports_microbatch_greater_than_one(
        self, monkeypatch
    ):
        import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

        class _ProcessGroup:
            def size(self):
                return 1

        class _ProcessGroups:
            tp = _ProcessGroup()

        class _FailFusedIndexerLoss:
            @staticmethod
            def apply(*_args, **_kwargs):
                raise AssertionError("MBS>1 HISA must defer teacher loss to fused DSA attention")

        torch.manual_seed(20260526)
        seqlen = 96
        q_len = 8
        batch = 3
        index_heads = 4
        index_dim = 128
        num_heads = 2
        qk_head_dim = 64
        value_head_dim = 64
        topk = 48
        softmax_scale = qk_head_dim**-0.5
        hisa_config = IndexCacheHISAConfig(
            enabled=True,
            block_size=16,
            compression_ratio=4.0,
            fallback_to_dense_if_short=False,
        )

        q = torch.randn(
            q_len,
            batch,
            index_heads,
            index_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        index_k = torch.randn(
            seqlen,
            batch,
            index_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weights = (
            torch.rand(q_len, batch, index_heads, device="cuda", dtype=torch.bfloat16) + 0.1
        ).requires_grad_()
        query = torch.randn(
            q_len,
            batch,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            batch,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            batch,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        def fail_fallback(*_args, **_kwargs):
            raise AssertionError("MBS>1 HISA DSA must use fused Triton selected attention")

        def fail_dense_indexer(*_args, **_kwargs):
            raise AssertionError("MBS>1 HISA indexer training must not use dense KL")

        monkeypatch.setattr(dsa_module, "_sparse_dsa_attention_chunk", fail_fallback)
        monkeypatch.setattr(dsa_module, "_compute_index_scores", fail_dense_indexer)
        monkeypatch.setattr(dsa_module, "_HISAFusedIndexerLoss", _FailFusedIndexerLoss)
        monkeypatch.setenv("MEGATRON_DSA_TRITON", "1")

        output, indexer_loss = dsa_module.chunked_dsa_forward(
            q,
            index_k,
            weights,
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            topk=topk,
            mask=None,
            is_causal=True,
            loss_coeff=0.1,
            sparse_loss=True,
            pg_collection=_ProcessGroups(),
            chunk_size=q_len,
            indexcache_hisa_config=hisa_config,
        )
        assert indexer_loss is not None
        assert torch.isfinite(output.float()).all().item()
        (output.float().square().mean() + indexer_loss).backward()
        for tensor in (q, index_k, weights, query, key, value):
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad.float()).all().item()
            assert tensor.grad.float().abs().sum().item() > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hisa_selected_score_batched_backward_matches_per_batch(self, monkeypatch):
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("HISA selected-score CUDA extension is exercised on Blackwell.")

        torch.manual_seed(20260527)
        q_len = 7
        batch = 3
        heads = 4
        head_dim = 128
        seqlen = 64
        topk = 13

        q = torch.randn(q_len, batch, heads, head_dim, device="cuda", dtype=torch.float32)
        weights = torch.randn(q_len, batch, heads, device="cuda", dtype=torch.float32)
        k = torch.randn(seqlen, batch, head_dim, device="cuda", dtype=torch.float32)
        topk_i32 = torch.randint(0, seqlen, (batch, q_len, topk), device="cuda", dtype=torch.int32)
        topk_i32[1, 2, 3] = -1
        grad_scores = torch.randn(batch * q_len, topk, device="cuda", dtype=torch.float32)
        grad_scores[1 * q_len + 2, 3] = 0.0

        monkeypatch.setenv("MEGATRON_HISA_SELECTED_SCORE_BWD_HEAD_GROUP", "2")
        monkeypatch.setenv("MEGATRON_HISA_SELECTED_SCORE_BWD_WARP_GROUPED", "1")

        grad_q_batches = []
        grad_w_batches = []
        grad_k_batches = []
        for batch_idx in range(batch):
            start = batch_idx * q_len
            stop = start + q_len
            grad_q_b, grad_w_b, grad_k_b = _hisa_selected_score_backward_cuda(
                grad_scores[start:stop],
                q[:, batch_idx],
                weights[:, batch_idx],
                k[:, batch_idx],
                topk_i32[batch_idx],
            )
            grad_q_batches.append(grad_q_b)
            grad_w_batches.append(grad_w_b)
            grad_k_batches.append(grad_k_b)
        expected_q = torch.stack(grad_q_batches, dim=1)
        expected_w = torch.stack(grad_w_batches, dim=1)
        expected_k = torch.stack(grad_k_batches, dim=1)

        got_q, got_w, got_k = _hisa_selected_score_backward_cuda_batched(
            grad_scores,
            q,
            weights,
            k,
            topk_i32,
        )

        torch.testing.assert_close(got_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(got_w, expected_w, rtol=0, atol=0)
        torch.testing.assert_close(got_k, expected_k, rtol=1e-6, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hisa_indexer_loss_supports_streambp_positions(self, monkeypatch):
        import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

        class _ProcessGroup:
            def size(self):
                return 1

        class _ProcessGroups:
            tp = _ProcessGroup()

        torch.manual_seed(20260517)
        seqlen = 256
        q_len = 8
        index_heads = 2
        index_dim = 32
        num_heads = 2
        qk_head_dim = 64
        value_head_dim = 64
        topk = 48
        q_offset = 248
        softmax_scale = qk_head_dim**-0.5
        hisa_config = IndexCacheHISAConfig(
            enabled=True,
            block_size=16,
            compression_ratio=4.0,
            fallback_to_dense_if_short=False,
        )

        q = torch.randn(
            q_len,
            1,
            index_heads,
            index_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        index_k = torch.randn(
            seqlen, 1, index_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        weights = (
            torch.rand(q_len, 1, index_heads, device="cuda", dtype=torch.bfloat16) + 0.1
        ).requires_grad_()
        query = torch.randn(
            q_len,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        query_positions = torch.arange(q_offset, q_offset + q_len, device="cuda")
        key_positions = torch.arange(seqlen, device="cuda")

        def fail_fallback(*_args, **_kwargs):
            raise AssertionError("StreamBP-position HISA must use fused Triton DSA")

        def fail_dense_indexer(*_args, **_kwargs):
            raise AssertionError("StreamBP-position HISA must not use dense index-score KL")

        monkeypatch.setattr(dsa_module, "_sparse_dsa_attention_chunk", fail_fallback)
        monkeypatch.setattr(dsa_module, "_compute_index_scores", fail_dense_indexer)
        monkeypatch.setenv("MEGATRON_DSA_TRITON", "1")

        output, indexer_loss = dsa_module.chunked_dsa_forward(
            q,
            index_k,
            weights,
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            topk=topk,
            mask=None,
            is_causal=True,
            loss_coeff=0.1,
            sparse_loss=True,
            pg_collection=_ProcessGroups(),
            chunk_size=q_len,
            query_positions=query_positions,
            key_positions=key_positions,
            indexcache_hisa_config=hisa_config,
        )
        assert indexer_loss is not None
        assert torch.isfinite(output.float()).all().item()
        (output.float().square().mean() + indexer_loss).backward()
        for tensor in (q, index_k, weights, query, key, value):
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad.float()).all().item()
            assert tensor.grad.float().abs().sum().item() > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "dtype,atol,rtol", [(torch.float32, 2e-4, 2e-4), (torch.bfloat16, 8e-2, 8e-2)]
    )
    @pytest.mark.parametrize("qk_head_dim,value_head_dim", [(128, 128), (192, 128)])
    def test_forward_backward_matches_torch_sparse_reference(
        self, dtype, atol, rtol, qk_head_dim, value_head_dim
    ):
        torch.manual_seed(1234)
        seqlen = 32
        num_heads = 2
        topk = 16
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            seqlen, 1, num_heads, qk_head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        key = torch.randn(
            seqlen, 1, num_heads, qk_head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        value = torch.randn(
            seqlen, 1, num_heads, value_head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        index_scores = torch.randn(1, seqlen, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        index_scores = index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        topk_indices = index_scores.topk(topk, dim=-1).indices
        grad_output = torch.randn(seqlen, 1, num_heads * value_head_dim, device="cuda", dtype=dtype)

        reference = _sparse_dsa_attention_chunk(
            query, key, value, topk_indices, softmax_scale, mask=None, q_start=0, is_causal=True
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        assert is_sparse_dsa_triton_supported(
            query_fused, key_fused, value_fused, topk_indices, mask=None, is_causal=True
        )
        fused = sparse_dsa_attention_triton(
            query_fused, key_fused, value_fused, topk_indices, softmax_scale
        )
        (fused * grad_output).sum().backward()

        assert torch.allclose(fused, reference, atol=atol, rtol=rtol)
        assert torch.allclose(query_fused.grad, reference_grads[0], atol=atol, rtol=rtol)
        assert torch.allclose(key_fused.grad, reference_grads[1], atol=atol, rtol=rtol)
        assert torch.allclose(value_fused.grad, reference_grads[2], atol=atol, rtol=rtol)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_backward_supports_microbatch_greater_than_one(self):
        torch.manual_seed(20260520)
        seqlen = 24
        batch = 3
        num_heads = 2
        qk_head_dim = 192
        value_head_dim = 128
        topk = 12
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            seqlen,
            batch,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            batch,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            batch,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        index_scores = torch.randn(batch, seqlen, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        topk_indices = (
            index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            .topk(topk, dim=-1)
            .indices
        )
        grad_output = torch.randn(
            seqlen,
            batch,
            num_heads * value_head_dim,
            device="cuda",
            dtype=torch.float32,
        )

        reference = _sparse_dsa_attention_chunk(
            query, key, value, topk_indices, softmax_scale, mask=None, q_start=0, is_causal=True
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        assert is_sparse_dsa_triton_supported(
            query_fused, key_fused, value_fused, topk_indices, mask=None, is_causal=True
        )
        fused = sparse_dsa_attention_triton(
            query_fused, key_fused, value_fused, topk_indices, softmax_scale
        )
        (fused * grad_output).sum().backward()

        torch.testing.assert_close(fused, reference, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(query_fused.grad, reference_grads[0], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(key_fused.grad, reference_grads[1], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(value_fused.grad, reference_grads[2], rtol=2e-4, atol=2e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_key_block_kv_backward_matches_torch_sparse_reference(self, monkeypatch):
        torch.manual_seed(4321)
        seqlen = 32
        num_heads = 2
        qk_head_dim = 192
        value_head_dim = 128
        topk = 16
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        index_scores = torch.randn(1, seqlen, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        topk_indices = (
            index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            .topk(topk, dim=-1)
            .indices.sort(dim=-1)
            .values
        )
        grad_output = torch.randn(
            seqlen, 1, num_heads * value_head_dim, device="cuda", dtype=torch.float32
        )

        reference = _sparse_dsa_attention_chunk(
            query, key, value, topk_indices, softmax_scale, mask=None, q_start=0, is_causal=True
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        monkeypatch.setenv("MEGATRON_DSA_TRITON_KEY_BLOCK_KV_BWD", "1")
        fused = sparse_dsa_attention_triton(
            query_fused, key_fused, value_fused, topk_indices, softmax_scale
        )
        (fused * grad_output).sum().backward()

        torch.testing.assert_close(fused, reference, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(query_fused.grad, reference_grads[0], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(key_fused.grad, reference_grads[1], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(value_fused.grad, reference_grads[2], rtol=2e-4, atol=2e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_key_block_kv_backward_falls_back_for_unsorted_topk(self, monkeypatch):
        torch.manual_seed(4322)
        seqlen = 32
        num_heads = 2
        qk_head_dim = 192
        value_head_dim = 128
        topk = 16
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        index_scores = torch.randn(1, seqlen, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        topk_indices = (
            index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            .topk(topk, dim=-1)
            .indices
        )
        assert not bool((topk_indices[..., 1:] >= topk_indices[..., :-1]).all().item())
        grad_output = torch.randn(
            seqlen, 1, num_heads * value_head_dim, device="cuda", dtype=torch.float32
        )

        reference = _sparse_dsa_attention_chunk(
            query, key, value, topk_indices, softmax_scale, mask=None, q_start=0, is_causal=True
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        monkeypatch.setenv("MEGATRON_DSA_TRITON_KEY_BLOCK_KV_BWD", "1")
        fused = sparse_dsa_attention_triton(
            query_fused, key_fused, value_fused, topk_indices, softmax_scale
        )
        (fused * grad_output).sum().backward()

        torch.testing.assert_close(fused, reference, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(query_fused.grad, reference_grads[0], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(key_fused.grad, reference_grads[1], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(value_fused.grad, reference_grads[2], rtol=2e-4, atol=2e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_qtile_unique_kv_backward_matches_torch_sparse_reference(self, monkeypatch):
        torch.manual_seed(8765)
        seqlen = 32
        num_heads = 2
        qk_head_dim = 192
        value_head_dim = 128
        topk = 16
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        index_scores = torch.randn(1, seqlen, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        topk_indices = (
            index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            .topk(topk, dim=-1)
            .indices.sort(dim=-1)
            .values
        )
        grad_output = torch.randn(
            seqlen, 1, num_heads * value_head_dim, device="cuda", dtype=torch.float32
        )

        reference = _sparse_dsa_attention_chunk(
            query, key, value, topk_indices, softmax_scale, mask=None, q_start=0, is_causal=True
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        monkeypatch.setenv("MEGATRON_DSA_TRITON_QTILE_UNIQUE_KV_BWD", "1")
        fused = sparse_dsa_attention_triton(
            query_fused, key_fused, value_fused, topk_indices, softmax_scale
        )
        (fused * grad_output).sum().backward()

        torch.testing.assert_close(fused, reference, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(query_fused.grad, reference_grads[0], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(key_fused.grad, reference_grads[1], rtol=4e-4, atol=4e-4)
        torch.testing.assert_close(value_fused.grad, reference_grads[2], rtol=4e-4, atol=4e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_grouped_kv_backward_matches_torch_sparse_reference(self, monkeypatch):
        torch.manual_seed(2468)
        seqlen = 32
        num_heads = 2
        qk_head_dim = 192
        value_head_dim = 128
        topk = 16
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        index_scores = torch.randn(1, seqlen, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        topk_indices = (
            index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            .topk(topk, dim=-1)
            .indices.sort(dim=-1)
            .values
        )
        grad_output = torch.randn(
            seqlen, 1, num_heads * value_head_dim, device="cuda", dtype=torch.float32
        )

        reference = _sparse_dsa_attention_chunk(
            query, key, value, topk_indices, softmax_scale, mask=None, q_start=0, is_causal=True
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        monkeypatch.setenv("MEGATRON_DSA_TRITON_GROUPED_KV_BWD", "1")
        monkeypatch.setenv("MEGATRON_DSA_TRITON_BLOCK_Q_BWD", "4")
        monkeypatch.setenv("MEGATRON_DSA_TRITON_BLOCK_K_BWD", "8")
        fused = sparse_dsa_attention_triton(
            query_fused, key_fused, value_fused, topk_indices, softmax_scale
        )
        (fused * grad_output).sum().backward()

        torch.testing.assert_close(fused, reference, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(query_fused.grad, reference_grads[0], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(key_fused.grad, reference_grads[1], rtol=4e-4, atol=4e-4)
        torch.testing.assert_close(value_fused.grad, reference_grads[2], rtol=4e-4, atol=4e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_kv_backward_matches_torch_sparse_reference_int16_topk(self, monkeypatch):
        import megatron.core.transformer.experimental_attention_variant.dsa_triton as dsa_triton_mod

        torch.manual_seed(97531)
        seqlen = 32
        bsz = 2
        num_heads = 2
        qk_head_dim = 128
        value_head_dim = 64
        topk = 16
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            seqlen,
            bsz,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            bsz,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            bsz,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        index_scores = torch.randn(bsz, seqlen, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        topk_indices = (
            index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            .topk(topk, dim=-1)
            .indices.to(torch.int16)
            .contiguous()
        )
        reference_topk_indices = topk_indices.to(torch.long)
        grad_output = torch.randn(
            seqlen, bsz, num_heads * value_head_dim, device="cuda", dtype=torch.float32
        )

        reference = _sparse_dsa_attention_chunk(
            query,
            key,
            value,
            reference_topk_indices,
            softmax_scale,
            mask=None,
            q_start=0,
            is_causal=True,
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        called = False
        original = dsa_triton_mod._dsa_sparse_kv_backward_cuda

        def wrapped_cuda_kv_backward(*args, **kwargs):
            nonlocal called
            called = True
            return original(*args, **kwargs)

        monkeypatch.setattr(
            dsa_triton_mod, "_dsa_sparse_kv_backward_cuda", wrapped_cuda_kv_backward
        )
        monkeypatch.setenv("MEGATRON_DSA_CUDA_KV_BWD", "1")
        monkeypatch.setenv("MEGATRON_DSA_CUDA_KV_BWD_TILE_Q", "2")
        monkeypatch.setenv("MEGATRON_DSA_CUDA_KV_BWD_TILE_K", "4")

        fused = sparse_dsa_attention_triton(
            query_fused, key_fused, value_fused, topk_indices, softmax_scale
        )
        (fused * grad_output).sum().backward()

        assert called, "CUDA DSA K/V backward path was not exercised"
        torch.testing.assert_close(fused, reference, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(query_fused.grad, reference_grads[0], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(key_fused.grad, reference_grads[1], rtol=4e-4, atol=4e-4)
        torch.testing.assert_close(value_fused.grad, reference_grads[2], rtol=4e-4, atol=4e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_backward_from_scores_matches_torch_sparse_reference(self, monkeypatch):
        import megatron.core.transformer.experimental_attention_variant.dsa_triton as dsa_triton_mod

        torch.manual_seed(86420)
        seqlen = 32
        bsz = 2
        num_heads = 2
        qk_head_dim = 128
        value_head_dim = 64
        topk = 16
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            seqlen,
            bsz,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            bsz,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            bsz,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        index_scores = torch.randn(bsz, seqlen, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        topk_indices = (
            index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            .topk(topk, dim=-1)
            .indices.to(torch.int16)
            .contiguous()
        )
        reference_topk_indices = topk_indices.to(torch.long)
        grad_output = torch.randn(
            seqlen, bsz, num_heads * value_head_dim, device="cuda", dtype=torch.float32
        )

        reference = _sparse_dsa_attention_chunk(
            query,
            key,
            value,
            reference_topk_indices,
            softmax_scale,
            mask=None,
            q_start=0,
            is_causal=True,
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        called = False
        original = dsa_triton_mod._dsa_sparse_backward_from_scores_cuda

        def wrapped_cuda_backward_from_scores(*args, **kwargs):
            nonlocal called
            called = True
            return original(*args, **kwargs)

        monkeypatch.setattr(
            dsa_triton_mod,
            "_dsa_sparse_backward_from_scores_cuda",
            wrapped_cuda_backward_from_scores,
        )
        monkeypatch.setenv("MEGATRON_DSA_TEACHER_SCORE_SCRATCH", "1")
        monkeypatch.setenv("MEGATRON_DSA_CUDA_BWD_FROM_SCORES", "1")
        monkeypatch.setenv("MEGATRON_DSA_CUDA_KV_BWD_TILE_Q", "2")
        monkeypatch.setenv("MEGATRON_DSA_CUDA_KV_BWD_TILE_K", "4")

        fused, teacher = sparse_dsa_attention_with_teacher_triton(
            query_fused, key_fused, value_fused, topk_indices, softmax_scale
        )
        (fused * grad_output).sum().backward()

        assert called, "CUDA DSA backward-from-scores path was not exercised"
        assert teacher.shape == (bsz * seqlen, topk)
        torch.testing.assert_close(fused, reference, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(query_fused.grad, reference_grads[0], rtol=4e-4, atol=4e-4)
        torch.testing.assert_close(key_fused.grad, reference_grads[1], rtol=4e-4, atol=4e-4)
        torch.testing.assert_close(value_fused.grad, reference_grads[2], rtol=4e-4, atol=4e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_row_backward_from_scores_matches_torch_sparse_reference(self, monkeypatch):
        import megatron.core.transformer.experimental_attention_variant.dsa_triton as dsa_triton_mod

        torch.manual_seed(97531)
        seqlen = 32
        bsz = 2
        num_heads = 2
        qk_head_dim = 128
        value_head_dim = 64
        topk = 16
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            seqlen,
            bsz,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            bsz,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            bsz,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        index_scores = torch.randn(bsz, seqlen, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        topk_indices = (
            index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            .topk(topk, dim=-1)
            .indices.to(torch.int16)
            .contiguous()
        )
        reference_topk_indices = topk_indices.to(torch.long)
        grad_output = torch.randn(
            seqlen, bsz, num_heads * value_head_dim, device="cuda", dtype=torch.float32
        )

        reference = _sparse_dsa_attention_chunk(
            query,
            key,
            value,
            reference_topk_indices,
            softmax_scale,
            mask=None,
            q_start=0,
            is_causal=True,
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        called = False
        original = dsa_triton_mod._dsa_sparse_backward_from_scores_row_cuda

        def wrapped_cuda_row_backward_from_scores(*args, **kwargs):
            nonlocal called
            called = True
            return original(*args, **kwargs)

        monkeypatch.setattr(
            dsa_triton_mod,
            "_dsa_sparse_backward_from_scores_row_cuda",
            wrapped_cuda_row_backward_from_scores,
        )
        monkeypatch.setenv("MEGATRON_DSA_TEACHER_SCORE_SCRATCH", "1")
        monkeypatch.setenv("MEGATRON_DSA_CUDA_ROW_BWD_FROM_SCORES", "1")

        fused, teacher = sparse_dsa_attention_with_teacher_triton(
            query_fused, key_fused, value_fused, topk_indices, softmax_scale
        )
        (fused * grad_output).sum().backward()

        assert called, "CUDA row-owned DSA backward-from-scores path was not exercised"
        assert teacher.shape == (bsz * seqlen, topk)
        torch.testing.assert_close(fused, reference, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(query_fused.grad, reference_grads[0], rtol=4e-4, atol=4e-4)
        torch.testing.assert_close(key_fused.grad, reference_grads[1], rtol=4e-4, atol=4e-4)
        torch.testing.assert_close(value_fused.grad, reference_grads[2], rtol=4e-4, atol=4e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_sorted_kv_backward_matches_torch_sparse_reference(self, monkeypatch):
        import megatron.core.transformer.experimental_attention_variant.dsa_triton as dsa_triton_mod

        torch.manual_seed(24680)
        seqlen = 32
        bsz = 2
        num_heads = 2
        qk_head_dim = 128
        value_head_dim = 64
        topk = 16
        softmax_scale = qk_head_dim**-0.5

        query = torch.randn(
            seqlen,
            bsz,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            bsz,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            bsz,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        index_scores = torch.randn(bsz, seqlen, seqlen, device="cuda", dtype=torch.float32)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device="cuda", dtype=torch.bool), diagonal=1
        )
        topk_indices = (
            index_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            .topk(topk, dim=-1)
            .indices.to(torch.int16)
            .contiguous()
        )
        reference_topk_indices = topk_indices.to(torch.long)
        grad_output = torch.randn(
            seqlen, bsz, num_heads * value_head_dim, device="cuda", dtype=torch.float32
        )

        reference = _sparse_dsa_attention_chunk(
            query,
            key,
            value,
            reference_topk_indices,
            softmax_scale,
            mask=None,
            q_start=0,
            is_causal=True,
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        called = False
        original = dsa_triton_mod._dsa_sparse_kv_backward_sorted_from_scores_cuda

        def wrapped_cuda_sorted_kv_backward(*args, **kwargs):
            nonlocal called
            called = True
            return original(*args, **kwargs)

        monkeypatch.setattr(
            dsa_triton_mod,
            "_dsa_sparse_kv_backward_sorted_from_scores_cuda",
            wrapped_cuda_sorted_kv_backward,
        )
        monkeypatch.setenv("MEGATRON_DSA_TEACHER_SCORE_SCRATCH", "1")
        monkeypatch.setenv("MEGATRON_DSA_CUDA_SORTED_KV_BWD", "1")

        fused, teacher = sparse_dsa_attention_with_teacher_triton(
            query_fused, key_fused, value_fused, topk_indices, softmax_scale
        )
        (fused * grad_output).sum().backward()

        assert called, "CUDA sorted-segment DSA K/V backward path was not exercised"
        assert teacher.shape == (bsz * seqlen, topk)
        torch.testing.assert_close(fused, reference, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(query_fused.grad, reference_grads[0], rtol=4e-4, atol=4e-4)
        torch.testing.assert_close(key_fused.grad, reference_grads[1], rtol=4e-4, atol=4e-4)
        torch.testing.assert_close(value_fused.grad, reference_grads[2], rtol=4e-4, atol=4e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_chunked_forward_triton_path_matches_fallback(self, monkeypatch):
        torch.manual_seed(5678)
        seqlen = 48
        num_heads = 2
        qk_head_dim = 192
        value_head_dim = 128
        index_n_heads = 4
        index_head_dim = 128
        topk = 16
        chunk_size = 12
        softmax_scale = qk_head_dim**-0.5

        q = torch.randn(
            seqlen, 1, index_n_heads, index_head_dim, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(seqlen, 1, index_head_dim, device="cuda", dtype=torch.bfloat16)
        weights = torch.rand(seqlen, 1, index_n_heads, device="cuda", dtype=torch.bfloat16)
        query = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn(
            seqlen,
            1,
            num_heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.randn(
            seqlen,
            1,
            num_heads,
            value_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        grad_output = torch.randn(
            seqlen, 1, num_heads * value_head_dim, device="cuda", dtype=torch.bfloat16
        )

        monkeypatch.setenv("MEGATRON_DSA_TRITON", "0")
        monkeypatch.setenv("MEGATRON_DSA_STREAMING_INDEXER_TOPK", "0")
        reference, _ = chunked_dsa_forward(
            q,
            k,
            weights,
            query,
            key,
            value,
            softmax_scale,
            topk,
            mask=None,
            is_causal=True,
            loss_coeff=0.0,
            sparse_loss=False,
            pg_collection=None,
            chunk_size=chunk_size,
        )
        (reference * grad_output).sum().backward()
        reference_grads = (
            query.grad.detach().clone(),
            key.grad.detach().clone(),
            value.grad.detach().clone(),
        )

        query_fused = query.detach().clone().requires_grad_(True)
        key_fused = key.detach().clone().requires_grad_(True)
        value_fused = value.detach().clone().requires_grad_(True)

        monkeypatch.setenv("MEGATRON_DSA_TRITON", "1")
        monkeypatch.setenv("MEGATRON_DSA_STREAMING_INDEXER_TOPK", "1")
        monkeypatch.setenv("MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE", "7")
        monkeypatch.setenv("MEGATRON_DSA_SORT_TOPK_INDICES", "1")
        monkeypatch.setenv("MEGATRON_DSA_COMPACT_TOPK_INDICES", "1")
        fused, _ = chunked_dsa_forward(
            q,
            k,
            weights,
            query_fused,
            key_fused,
            value_fused,
            softmax_scale,
            topk,
            mask=None,
            is_causal=True,
            loss_coeff=0.0,
            sparse_loss=False,
            pg_collection=None,
            chunk_size=chunk_size,
        )
        (fused * grad_output).sum().backward()

        assert torch.allclose(fused, reference, atol=8e-2, rtol=8e-2)
        assert torch.allclose(query_fused.grad, reference_grads[0], atol=8e-2, rtol=8e-2)
        assert torch.allclose(key_fused.grad, reference_grads[1], atol=8e-2, rtol=8e-2)
        assert torch.allclose(value_fused.grad, reference_grads[2], atol=8e-2, rtol=8e-2)


@pytest.mark.parametrize("seqlen_and_topk", [[16, 8], [32, 16], [64, 32]])
@pytest.mark.parametrize("sparse_loss", [False, True])
class TestFusedDSAIndexerLossGradient:
    """Test that FusedDSAIndexerLoss manual backward matches autograd backward."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fused_indexer_loss_gradient_matches_autograd(self, seqlen_and_topk, sparse_loss):
        """
        Test that the manually written backward in FusedDSAIndexerLoss produces
        the same gradients as PyTorch autograd on the unfused implementation.
        """
        seqlen = seqlen_and_topk[0]
        index_topk = seqlen_and_topk[1]
        batch_size = 2
        num_heads = 4
        head_dim = 64
        index_n_heads = 8
        index_head_dim = 64
        softmax_scale = head_dim**-0.5
        loss_coeff = 1.0

        torch.manual_seed(42)

        # Create inputs for indexer
        # q: [seqlen, batch, index_n_heads, index_head_dim]
        q_ref = (
            torch.randn(seqlen, batch_size, index_n_heads, index_head_dim, dtype=torch.float32)
            .cuda()
            .requires_grad_(True)
        )
        # weights: [seqlen, batch, index_n_heads]
        weights_ref = (
            torch.randn(seqlen, batch_size, index_n_heads, dtype=torch.float32)
            .cuda()
            .requires_grad_(True)
        )
        # k: [seqlen, batch, index_head_dim]
        k_ref = (
            torch.randn(seqlen, batch_size, index_head_dim, dtype=torch.float32)
            .cuda()
            .requires_grad_(True)
        )
        # query: [seqlen, batch, num_heads, head_dim] - detached, not trained
        query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        # key: [seqlen, batch, num_heads, head_dim] - detached, not trained
        key = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()

        # Create causal mask
        mask = torch.triu(
            torch.full((seqlen, seqlen), float('-inf'), dtype=torch.float32).cuda(), diagonal=1
        )

        # =============================================
        # Method 1: Autograd (reference)
        # =============================================
        # Compute index scores and apply mask (matches fused_qk_topk_naive behavior)
        index_scores_ref = _compute_index_scores(q_ref, weights_ref, k_ref)
        # Apply mask
        index_scores_masked = index_scores_ref + mask.unsqueeze(0)
        # Get topk indices from masked scores
        topk_k = min(index_topk, seqlen)
        topk_indices = index_scores_masked.topk(topk_k, dim=-1)[1]

        # Compute loss using autograd
        loss_ref = compute_dsa_indexer_loss(
            index_scores=index_scores_masked,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=softmax_scale,
            loss_coeff=loss_coeff,
            sparse_loss=sparse_loss,
            pg_collection=self.pg_collection,
        )

        # Backward with autograd
        loss_ref.backward()

        # Save reference gradients
        grad_q_ref = q_ref.grad.clone()
        grad_weights_ref = weights_ref.grad.clone()
        grad_k_ref = k_ref.grad.clone()

        # =============================================
        # Method 2: FusedDSAIndexerLoss (manual backward)
        # =============================================
        # Clone tensors from ref (detach and require grad again)
        q_fused = q_ref.detach().clone().requires_grad_(True)
        weights_fused = weights_ref.detach().clone().requires_grad_(True)
        k_fused = k_ref.detach().clone().requires_grad_(True)

        # Use FusedDSAIndexerLoss
        topk_indices_fused, loss_fused = FusedDSAIndexerLoss.apply(
            q_fused,
            weights_fused,
            k_fused,
            query.detach(),
            key.detach(),
            softmax_scale,
            index_topk,
            loss_coeff,
            mask,
            sparse_loss,
            self.pg_collection,
        )

        # Backward with manual implementation
        loss_fused.backward()

        # Get fused gradients
        grad_q_fused = q_fused.grad
        grad_weights_fused = weights_fused.grad
        grad_k_fused = k_fused.grad

        # =============================================
        # Compare gradients
        # =============================================
        # Check loss values match
        assert torch.allclose(
            loss_fused, loss_ref, rtol=1e-5, atol=1e-5
        ), f"Loss mismatch: fused={loss_fused.item()}, ref={loss_ref.item()}"

        # Check topk indices match
        assert torch.equal(
            topk_indices_fused, topk_indices
        ), "Top-k indices mismatch between fused and reference"

        # Check gradients match
        assert torch.allclose(
            grad_q_fused, grad_q_ref, rtol=1e-5, atol=1e-5
        ), f"grad_q mismatch: max diff = {(grad_q_fused - grad_q_ref).abs().max().item()}"

        assert torch.allclose(
            grad_weights_fused, grad_weights_ref, rtol=1e-5, atol=1e-5
        ), f"grad_weights mismatch: max diff = {(grad_weights_fused - grad_weights_ref).abs().max().item()}"

        assert torch.allclose(
            grad_k_fused, grad_k_ref, rtol=1e-5, atol=1e-5
        ), f"grad_k mismatch: max diff = {(grad_k_fused - grad_k_ref).abs().max().item()}"


@pytest.mark.parametrize("tensor_model_parallel_size", [2, 4])
@pytest.mark.parametrize("sparse_loss", [False, True])
class TestFusedDSAIndexerLossGradientTP:
    """Test FusedDSAIndexerLoss gradient consistency across different TP sizes."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fused_indexer_loss_gradient_tp_consistency(
        self, tensor_model_parallel_size, sparse_loss
    ):
        """
        Test that FusedDSAIndexerLoss produces consistent gradients across TP ranks
        and matches TP=1 baseline.
        """
        seqlen = 64
        index_topk = 32
        batch_size = 2
        num_heads = 8
        head_dim = 64
        index_n_heads = 8
        index_head_dim = 64
        softmax_scale = head_dim**-0.5
        loss_coeff = 1.0

        # =============================================
        # First run with TP=1 to get baseline
        # =============================================
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        pg_collection_tp1 = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])

        # Create inputs
        q_input = torch.randn(
            seqlen, batch_size, index_n_heads, index_head_dim, dtype=torch.float32
        ).cuda()
        weights_input = torch.randn(seqlen, batch_size, index_n_heads, dtype=torch.float32).cuda()
        k_input = torch.randn(seqlen, batch_size, index_head_dim, dtype=torch.float32).cuda()
        query_input = torch.randn(
            seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        key_input = torch.randn(
            seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        mask = torch.triu(
            torch.full((seqlen, seqlen), float('-inf'), dtype=torch.float32).cuda(), diagonal=1
        )

        # Clone for TP=1
        q_tp1 = q_input.clone().requires_grad_(True)
        weights_tp1 = weights_input.clone().requires_grad_(True)
        k_tp1 = k_input.clone().requires_grad_(True)

        # Forward and backward with TP=1
        topk_indices_tp1, loss_tp1 = FusedDSAIndexerLoss.apply(
            q_tp1,
            weights_tp1,
            k_tp1,
            query_input.detach(),
            key_input.detach(),
            softmax_scale,
            index_topk,
            loss_coeff,
            mask,
            sparse_loss,
            pg_collection_tp1,
        )
        loss_tp1.backward()

        # Save TP=1 results
        grad_q_tp1 = q_tp1.grad.clone()
        grad_weights_tp1 = weights_tp1.grad.clone()
        grad_k_tp1 = k_tp1.grad.clone()
        loss_tp1_value = loss_tp1.detach().clone()

        Utils.destroy_model_parallel()

        # =============================================
        # Run with target TP size
        # =============================================
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        pg_collection_tpn = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        # Clone inputs for TP=N (same values as TP=1)
        q_tpn = q_input.clone().requires_grad_(True)
        weights_tpn = weights_input.clone().requires_grad_(True)
        k_tpn = k_input.clone().requires_grad_(True)

        # query and key need to be split along heads for TP
        head_per_rank = num_heads // tensor_model_parallel_size
        start_head = tp_rank * head_per_rank
        end_head = (tp_rank + 1) * head_per_rank
        query_tpn = query_input[:, :, start_head:end_head, :].clone()
        key_tpn = key_input[:, :, start_head:end_head, :].clone()

        # Forward and backward with TP=N
        topk_indices_tpn, loss_tpn = FusedDSAIndexerLoss.apply(
            q_tpn,
            weights_tpn,
            k_tpn,
            query_tpn.detach(),
            key_tpn.detach(),
            softmax_scale,
            index_topk,
            loss_coeff,
            mask,
            sparse_loss,
            pg_collection_tpn,
        )
        loss_tpn.backward()

        # =============================================
        # Compare results
        # =============================================
        # Loss should be the same
        assert torch.allclose(
            loss_tpn, loss_tp1_value, rtol=1e-5, atol=1e-5
        ), f"Loss mismatch: TP={tensor_model_parallel_size} got {loss_tpn.item()}, TP=1 got {loss_tp1_value.item()}"

        # Top-k indices should be the same
        assert torch.equal(
            topk_indices_tpn, topk_indices_tp1
        ), "Top-k indices mismatch between TP=1 and TP=N"

        # Gradients should match exactly (indexer params are duplicated across TP)
        assert torch.allclose(
            q_tpn.grad, grad_q_tp1, rtol=1e-5, atol=1e-5
        ), f"grad_q mismatch: max diff = {(q_tpn.grad - grad_q_tp1).abs().max().item()}"

        assert torch.allclose(
            weights_tpn.grad, grad_weights_tp1, rtol=1e-5, atol=1e-5
        ), f"grad_weights mismatch: max diff = {(weights_tpn.grad - grad_weights_tp1).abs().max().item()}"

        assert torch.allclose(
            k_tpn.grad, grad_k_tp1, rtol=1e-5, atol=1e-5
        ), f"grad_k mismatch: max diff = {(k_tpn.grad - grad_k_tp1).abs().max().item()}"

        # Check gradients are identical across all TP ranks
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            for grad_tensor, name in [
                (q_tpn.grad, "grad_q"),
                (weights_tpn.grad, "grad_weights"),
                (k_tpn.grad, "grad_k"),
            ]:
                grad_list = [torch.zeros_like(grad_tensor) for _ in range(tp_size)]
                torch.distributed.all_gather(grad_list, grad_tensor, group=pg_collection_tpn.tp)

                for i in range(1, tp_size):
                    assert torch.allclose(
                        grad_list[0], grad_list[i], rtol=0, atol=0
                    ), f"{name} differs between TP rank 0 and rank {i}"

        Utils.destroy_model_parallel()


@pytest.mark.parametrize("seqlen", [16, 64])
class TestDSAIndexer:
    """Test DSA Indexer module basic functionality with TP=1."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        # Create MLA config with sparse attention parameters
        self.index_topk = 32
        self.config = MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            # Sparse attention specific configs
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=self.index_topk,
        )

        # Create indexer submodules spec
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )

        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'cp']
        )
        self.indexer = DSAIndexer(self.config, indexer_submodules, self.pg_collection)

        yield
        Utils.destroy_model_parallel()

    def test_dsa_indexer_constructor(self, seqlen):
        """Test indexer initialization."""
        assert isinstance(self.indexer, DSAIndexer)
        assert self.indexer.hidden_size == 256
        assert self.indexer.index_n_heads == 8
        assert self.indexer.index_head_dim == 64
        assert self.indexer.index_topk == 32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_forward(self, seqlen):
        """Test indexer forward pass."""
        batch_size = 2

        self.indexer.cuda()

        # Create input tensors
        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Forward pass
        topk_indices = self.indexer(x, qr)

        # Check output shape
        assert topk_indices.shape == (batch_size, seqlen, min(self.config.dsa_indexer_topk, seqlen))
        assert topk_indices.dtype == torch.long
        assert torch.all((topk_indices >= 0) & (topk_indices < seqlen))
        # Make sure no duplicate indices are selected
        assert torch.all(
            torch.sort(topk_indices, dim=-1).values[:, :, 1:]
            != torch.sort(topk_indices, dim=-1).values[:, :, :-1]
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_forward_with_scores(self, seqlen):
        """Test indexer forward pass with scores."""
        batch_size = 2

        self.indexer.cuda()

        # Create input tensors
        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Forward pass with scores
        index_scores, topk_indices = self.indexer.forward_with_scores(x, qr)

        # Check output shapes
        assert index_scores.shape == (batch_size, seqlen, seqlen)
        assert topk_indices.shape == (batch_size, seqlen, min(self.config.dsa_indexer_topk, seqlen))
        assert index_scores.dtype == torch.float32
        assert topk_indices.dtype == torch.long
        assert torch.all((topk_indices >= 0) & (topk_indices < seqlen))
        # Make sure no duplicate indices are selected
        assert torch.all(
            torch.sort(topk_indices, dim=-1).values[:, :, 1:]
            != torch.sort(topk_indices, dim=-1).values[:, :, :-1]
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_with_mask(self, seqlen):
        """Test indexer with attention mask."""
        batch_size = 2

        self.indexer.cuda()

        # Create input tensors
        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()
        mask = torch.triu(
            torch.full((batch_size, seqlen, seqlen), float('-inf'), dtype=torch.float32).cuda(),
            diagonal=1,
        )

        # Forward pass with mask
        index_scores, topk_indices = self.indexer.forward_with_scores(x, qr, mask=mask)

        # Check that masked positions are not selected
        # For causal mask, topk_indices[b, i, :] should all be <= i (except for the case that
        # i < index_topk).
        for b in range(batch_size):
            for i in range(seqlen):
                assert torch.all(topk_indices[b, i] <= max(self.index_topk, i))


class TestDSAttention:
    """Test DSAttention module basic functionality with TP=1."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        # Create MLA config with sparse attention parameters
        self.config = MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            # Sparse attention specific configs
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=False,
        )

        # Create sparse attention submodules spec
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )
        indexer_spec = ModuleSpec(module=DSAIndexer, submodules=indexer_submodules)
        sparse_attention_submodules = DSAttentionSubmodules(indexer=indexer_spec)

        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'cp']
        )

        self.sparse_attention = DSAttention(
            config=self.config,
            submodules=sparse_attention_submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
        )

        yield
        Utils.destroy_model_parallel()

    def test_dsa_constructor(self):
        """Test sparse attention initialization."""
        assert isinstance(self.sparse_attention, DSAttention)
        assert hasattr(self.sparse_attention, 'indexer')
        assert isinstance(self.sparse_attention.indexer, DSAIndexer)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_forward(self):
        """Test sparse attention forward pass."""
        seq_len = 16
        batch_size = 2
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads

        self.sparse_attention.cuda()

        # Create input tensors [seq_len, batch, num_heads, head_dim]
        query = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        key = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        value = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        # Original hidden states and low-rank query
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Create causal attention mask
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        DSAIndexerAuxLossState.clear()

        # Forward pass
        output = self.sparse_attention(
            query=query,
            key=key,
            value=value,
            x=x,
            qr=qr,
            attention_mask=attention_mask,
            attn_mask_type=AttnMaskType.causal,
        )

        # Check output shape
        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert output.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_backward(self):
        """Test sparse attention backward pass with indexer loss."""
        seq_len = 16
        batch_size = 2
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads

        self.sparse_attention.train()
        self.sparse_attention.cuda()

        # Create input tensors
        query = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        key = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        value = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        # Original hidden states and low-rank query
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Create causal attention mask
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        # Forward pass
        output = self.sparse_attention(
            query=query,
            key=key,
            value=value,
            x=x,
            qr=qr,
            attention_mask=attention_mask,
            attn_mask_type=AttnMaskType.causal,
        )

        # Backward pass
        indexer_loss = DSAIndexerAuxLossState.total()
        assert indexer_loss is not None
        DSAIndexerAuxLossState.clear()
        loss = output.sum() + indexer_loss
        loss.backward()

        # Check that gradients are computed for inputs
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None

        # Check that indexer parameters have gradients
        for name, param in self.sparse_attention.indexer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Indexer parameter {name} has no gradient"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_topk_selection(self):
        """Test that sparse attention correctly selects top-k indices."""
        seq_len = 16
        batch_size = 2
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads

        self.sparse_attention.eval()
        self.sparse_attention.cuda()

        # Create input tensors
        query = torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        value = torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()

        # Original hidden states and low-rank query
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Create causal attention mask
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        with torch.no_grad():
            # Get topk indices from indexer
            _, topk_indices = self.sparse_attention.indexer.forward_with_scores(x, qr)

            # Forward pass
            output = self.sparse_attention(
                query=query,
                key=key,
                value=value,
                x=x,
                qr=qr,
                attention_mask=attention_mask,
                attn_mask_type=AttnMaskType.causal,
            )

        # Check that topk_indices are valid
        assert torch.all(topk_indices >= 0)
        assert torch.all(topk_indices < seq_len)
        assert topk_indices.shape[2] == min(self.config.dsa_indexer_topk, seq_len)


# ======================================================================================
# Tensor Parallel Consistency Tests
# ======================================================================================


@pytest.mark.parametrize("tensor_model_parallel_size", [2, 4, 8])
@pytest.mark.parametrize("sequence_parallel", [False, True])
class TestIndexerTensorParallel:
    """Test DSA Indexer with different TP sizes and SP settings, compare with TP=1 baseline."""

    def _create_config(self, sequence_parallel=False):
        """Helper to create MLA config."""
        # Get TP size from parallel_state
        tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

        return MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=tensor_model_parallel_size,
            sequence_parallel=sequence_parallel,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            # Sparse attention specific configs
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
        )

    def _create_indexer(self, config, pg_collection):
        """Helper to create indexer."""
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )

        return DSAIndexer(config, indexer_submodules, pg_collection)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_weight_consistency(self, tensor_model_parallel_size, sequence_parallel):
        """Test that indexer weights are identical across ALL GPUs."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config = self._create_config(sequence_parallel=sequence_parallel)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        indexer = self._create_indexer(config, pg_collection).cuda()

        # Check that all weights are identical across ALL ranks (not just TP group)
        world_size = torch.distributed.get_world_size()
        world_rank = torch.distributed.get_rank()

        if world_size > 1:
            for name, param in indexer.named_parameters():
                # Gather weights from ALL ranks in WORLD group
                param_list = [torch.zeros_like(param.data) for _ in range(world_size)]
                torch.distributed.all_gather(param_list, param.data)

                # All weights should be identical across all GPUs
                for i in range(1, world_size):
                    assert torch.allclose(
                        param_list[0], param_list[i], rtol=0, atol=0
                    ), f"Parameter {name} differs between rank 0 and rank {i} (world)"

        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_forward_consistency(self, tensor_model_parallel_size, sequence_parallel):
        """Test that indexer gives consistent results across different TP sizes and SP settings."""
        # First run with TP=1 to get baseline
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config_tp1 = self._create_config(sequence_parallel=False)  # TP=1 doesn't use SP
        pg_collection_tp1 = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        indexer_tp1 = self._create_indexer(config_tp1, pg_collection_tp1).cuda()

        seq_len = 64
        batch_size = 2

        # Create one common input (all ranks create same input with same seed)
        x_input = torch.randn(
            seq_len, batch_size, config_tp1.hidden_size, dtype=torch.bfloat16
        ).cuda()
        qr_input = torch.randn(
            seq_len, batch_size, config_tp1.q_lora_rank, dtype=torch.bfloat16
        ).cuda()

        # Forward pass with gradients enabled
        index_scores_tp1, topk_indices_tp1 = indexer_tp1.forward_with_scores(x_input, qr_input)

        # Backward pass
        loss_tp1 = index_scores_tp1.sum()
        loss_tp1.backward()

        # Save gradients from TP=1
        indexer_tp1_grads = {
            name: param.grad.clone().cpu()
            for name, param in indexer_tp1.named_parameters()
            if param.grad is not None
        }

        Utils.destroy_model_parallel()

        # Now run with target TP size
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config_tpn = self._create_config(sequence_parallel=sequence_parallel)
        pg_collection_tpn = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        indexer_tpn = self._create_indexer(config_tpn, pg_collection_tpn).cuda()

        # Prepare input: split along seqlen if SP is enabled
        if sequence_parallel:
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            seq_per_rank = seq_len // tensor_model_parallel_size
            start_idx = tp_rank * seq_per_rank
            end_idx = (tp_rank + 1) * seq_per_rank
            x_tpn = x_input[start_idx:end_idx]
            qr_tpn = qr_input[start_idx:end_idx]
        else:
            # No SP: all TP ranks see full input
            x_tpn = x_input
            qr_tpn = qr_input

        # Forward pass with gradients enabled
        index_scores_tpn, topk_indices_tpn = indexer_tpn.forward_with_scores(x_tpn, qr_tpn)

        # Backward pass
        loss_tpn = index_scores_tpn.sum()
        loss_tpn.backward()

        # Compare forward outputs
        assert index_scores_tpn.shape == index_scores_tp1.shape
        assert topk_indices_tpn.shape == topk_indices_tp1.shape

        # Check that index scores are close (allow for floating point accumulation errors)
        assert torch.allclose(
            index_scores_tpn, index_scores_tp1, rtol=0, atol=0
        ), f"Index scores mismatch between TP=1 and TP={tensor_model_parallel_size}, SP={sequence_parallel}"

        # Check that topk indices are exactly the same
        assert torch.equal(
            topk_indices_tpn, topk_indices_tp1
        ), f"Top-k indices mismatch between TP=1 and TP={tensor_model_parallel_size}, SP={sequence_parallel}"

        # Compare gradients - indexer grads should be identical (duplicated weights)
        for name, param in indexer_tpn.named_parameters():
            if param.grad is not None and name in indexer_tp1_grads:
                assert torch.allclose(
                    param.grad.cpu(), indexer_tp1_grads[name], rtol=0, atol=0
                ), f"Indexer gradient {name} mismatch between TP=1 and TP={tensor_model_parallel_size}, SP={sequence_parallel}"

        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_gradient_sync(self, tensor_model_parallel_size, sequence_parallel):
        """Test that gradients are properly synchronized within TP group."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config = self._create_config(sequence_parallel=sequence_parallel)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        indexer = self._create_indexer(config, pg_collection).cuda()

        seq_len = 64
        batch_size = 2

        # Create one common input (all ranks create same input with same seed)
        x_input = torch.randn(seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16).cuda()
        qr_input = torch.randn(seq_len, batch_size, config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Prepare input: split along seqlen if SP is enabled
        if sequence_parallel:
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            seq_per_rank = seq_len // tp_size
            start_idx = tp_rank * seq_per_rank
            end_idx = (tp_rank + 1) * seq_per_rank
            x = x_input[start_idx:end_idx]
            qr = qr_input[start_idx:end_idx]
        else:
            # No SP: all TP ranks see full input
            x = x_input
            qr = qr_input

        # Forward and backward
        index_scores, topk_indices = indexer.forward_with_scores(x, qr)
        loss = index_scores.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in indexer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

        # After TP sync, check that gradients are identical within TP group
        # Note: We only check TP group because DDP sync happens separately
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            for name, param in indexer.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Gather gradients from all ranks in TP group only
                    grad_list = [torch.zeros_like(param.grad) for _ in range(tp_size)]
                    torch.distributed.all_gather(grad_list, param.grad, group=pg_collection.tp)

                    # All gradients should be identical within TP group after sync
                    for i in range(1, tp_size):
                        assert torch.allclose(
                            grad_list[0], grad_list[i], rtol=0, atol=0
                        ), f"Gradient for {name} differs between TP rank 0 and rank {i} after TP sync"

        Utils.destroy_model_parallel()


@pytest.mark.parametrize("tensor_model_parallel_size", [2, 4])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("use_sparse_indexer_loss", [False, True])
class TestDSAttentionTensorParallel:
    """Test DSAttention with different TP sizes, SP settings, and sparse indexer loss."""

    def _create_config(self, sequence_parallel=False, use_sparse_indexer_loss=False):
        """Helper to create MLA config."""
        # Get TP size from parallel_state
        tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

        return MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=tensor_model_parallel_size,
            sequence_parallel=sequence_parallel,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            # Sparse attention specific configs
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=use_sparse_indexer_loss,
        )

    def _create_sparse_attention(self, config, pg_collection):
        """Helper to create sparse attention."""
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )
        indexer_spec = ModuleSpec(module=DSAIndexer, submodules=indexer_submodules)
        sparse_attention_submodules = DSAttentionSubmodules(indexer=indexer_spec)

        return DSAttention(
            config=config,
            submodules=sparse_attention_submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=pg_collection,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_weight_consistency(
        self, tensor_model_parallel_size, sequence_parallel, use_sparse_indexer_loss
    ):
        """Test that sparse attention indexer weights are identical across ALL GPUs."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config = self._create_config(
            sequence_parallel=sequence_parallel, use_sparse_indexer_loss=use_sparse_indexer_loss
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        sparse_attention = self._create_sparse_attention(config, pg_collection).cuda()

        # Check that all indexer weights are identical across ALL ranks
        world_size = torch.distributed.get_world_size()
        world_rank = torch.distributed.get_rank()

        if world_size > 1:
            for name, param in sparse_attention.indexer.named_parameters():
                # Gather weights from ALL ranks in WORLD group
                param_list = [torch.zeros_like(param.data) for _ in range(world_size)]
                torch.distributed.all_gather(param_list, param.data)

                # All weights should be identical across all GPUs
                for i in range(1, world_size):
                    torch.testing.assert_close(param_list[0], param_list[i], rtol=0, atol=0)

        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_forward_consistency(
        self, tensor_model_parallel_size, sequence_parallel, use_sparse_indexer_loss
    ):
        """Test that sparse attention gives consistent results across different TP, SP, and sparse loss settings."""
        # First run with TP=1 to get baseline
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config_tp1 = self._create_config(
            sequence_parallel=False, use_sparse_indexer_loss=use_sparse_indexer_loss
        )  # TP=1 doesn't use SP
        pg_collection_tp1 = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        sparse_attention_tp1 = self._create_sparse_attention(config_tp1, pg_collection_tp1).cuda()

        seq_len = 64
        batch_size = 2
        num_heads = config_tp1.num_attention_heads
        head_dim = config_tp1.hidden_size // num_heads

        # Create one common input (all ranks create same input with same seed)
        query_input = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        key_input = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        value_input = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        x_input = torch.randn(
            seq_len, batch_size, config_tp1.hidden_size, dtype=torch.bfloat16
        ).cuda()
        qr_input = torch.randn(
            seq_len, batch_size, config_tp1.q_lora_rank, dtype=torch.bfloat16
        ).cuda()
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        # Forward pass with gradients enabled
        sparse_attention_tp1.train()
        output_tp1 = sparse_attention_tp1(
            query=query_input,
            key=key_input,
            value=value_input,
            x=x_input,
            qr=qr_input,
            attention_mask=attention_mask,
            attn_mask_type=AttnMaskType.causal,
        )

        # Backward pass
        loss_tp1 = output_tp1.sum()
        loss_tp1.backward()

        # Save gradients from TP=1
        indexer_tp1_grads = {
            name: param.grad.clone()
            for name, param in sparse_attention_tp1.indexer.named_parameters()
            if param.grad is not None
        }
        query_tp1_grad = query_input.grad.clone().cpu()
        key_tp1_grad = key_input.grad.clone().cpu()
        value_tp1_grad = value_input.grad.clone().cpu()

        Utils.destroy_model_parallel()

        # Now run with target TP size
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config_tpn = self._create_config(
            sequence_parallel=sequence_parallel, use_sparse_indexer_loss=use_sparse_indexer_loss
        )
        pg_collection_tpn = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        sparse_attention_tpn = self._create_sparse_attention(config_tpn, pg_collection_tpn).cuda()

        # Create one common input (all ranks create same input with same seed)
        query_input = torch.randn(
            seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        key_input = torch.randn(
            seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        value_input = torch.randn(
            seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        x_input = torch.randn(
            seq_len, batch_size, config_tp1.hidden_size, dtype=torch.bfloat16
        ).cuda()
        qr_input = torch.randn(
            seq_len, batch_size, config_tp1.q_lora_rank, dtype=torch.bfloat16
        ).cuda()
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        # Prepare input: split along seqlen if SP is enabled
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        if sequence_parallel:
            seq_per_rank = seq_len // tensor_model_parallel_size
            start_idx = tp_rank * seq_per_rank
            end_idx = (tp_rank + 1) * seq_per_rank
            x_tpn = x_input[start_idx:end_idx]
            qr_tpn = qr_input[start_idx:end_idx]
        else:
            x_tpn = x_input
            qr_tpn = qr_input

        query_input = query_input.detach()
        key_input = key_input.detach()
        value_input = value_input.detach()
        head_per_rank = num_heads // tensor_model_parallel_size
        start_head = tp_rank * head_per_rank
        end_head = (tp_rank + 1) * head_per_rank
        query_tpn = query_input[:, :, start_head:end_head, :].clone().requires_grad_(True)
        key_tpn = key_input[:, :, start_head:end_head, :].clone().requires_grad_(True)
        value_tpn = value_input[:, :, start_head:end_head, :].clone().requires_grad_(True)
        attention_mask_tpn = attention_mask

        # Forward pass with gradients enabled
        sparse_attention_tpn.train()
        output_tpn = sparse_attention_tpn(
            query=query_tpn,
            key=key_tpn,
            value=value_tpn,
            x=x_tpn,
            qr=qr_tpn,
            attention_mask=attention_mask_tpn,
            attn_mask_type=AttnMaskType.causal,
        )

        # Backward pass
        loss_tpn = output_tpn.sum()
        loss_tpn.backward()

        from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

        output_tpn_gathered = gather_from_tensor_model_parallel_region(
            output_tpn, group=pg_collection_tpn.tp
        )
        assert output_tpn_gathered.shape == output_tp1.shape
        assert torch.allclose(
            output_tpn_gathered.detach(), output_tp1.detach(), rtol=0, atol=0
        ), f"Sparse attention outputs mismatch between TP=1 and TP={tensor_model_parallel_size}, SP={sequence_parallel}, sparse_loss={use_sparse_indexer_loss}"

        # 1. Check indexer gradients.
        for name, param in sparse_attention_tpn.indexer.named_parameters():
            if param.grad is not None and name in indexer_tp1_grads:
                torch.testing.assert_close(
                    param.grad, indexer_tp1_grads[name], rtol=1e-5, atol=1e-5
                )

        # 2. Query/Key/Value gradients need to be gathered along num_heads dim (dim 2) if SP is enabled
        # Flatten last two dims: [seq_len, batch, num_heads, head_dim] -> [seq_len, batch, num_heads * head_dim]
        sq, b, nh, hd = query_tpn.grad.shape
        query_grad_flat = query_tpn.grad.reshape(sq, b, nh * hd)
        key_grad_flat = key_tpn.grad.reshape(sq, b, nh * hd)
        value_grad_flat = value_tpn.grad.reshape(sq, b, nh * hd)

        # Gather along last dim
        query_grad_gathered_flat = gather_from_tensor_model_parallel_region(
            query_grad_flat, group=pg_collection_tpn.tp
        )
        key_grad_gathered_flat = gather_from_tensor_model_parallel_region(
            key_grad_flat, group=pg_collection_tpn.tp
        )
        value_grad_gathered_flat = gather_from_tensor_model_parallel_region(
            value_grad_flat, group=pg_collection_tpn.tp
        )

        # Reshape back: [seq_len, batch, num_heads * head_dim] -> [seq_len, batch, num_heads, head_dim]
        query_tpn_grad_gathered = query_grad_gathered_flat.reshape(sq, b, num_heads, hd)
        key_tpn_grad_gathered = key_grad_gathered_flat.reshape(sq, b, num_heads, hd)
        value_tpn_grad_gathered = value_grad_gathered_flat.reshape(sq, b, num_heads, hd)

        assert torch.allclose(
            query_tpn_grad_gathered.cpu(), query_tp1_grad, rtol=0, atol=0
        ), f"Query gradient mismatch between TP=1 and TP={tensor_model_parallel_size}"
        assert torch.allclose(
            key_tpn_grad_gathered.cpu(), key_tp1_grad, rtol=0, atol=0
        ), f"Key gradient mismatch between TP=1 and TP={tensor_model_parallel_size}"
        assert torch.allclose(
            value_tpn_grad_gathered.cpu(), value_tp1_grad, rtol=0, atol=0
        ), f"Value gradient mismatch between TP=1 and TP={tensor_model_parallel_size}"

        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_gradient_sync(
        self, tensor_model_parallel_size, sequence_parallel, use_sparse_indexer_loss
    ):
        """Test that indexer gradients are properly synchronized within TP group."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config = self._create_config(
            sequence_parallel=sequence_parallel, use_sparse_indexer_loss=use_sparse_indexer_loss
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        sparse_attention = self._create_sparse_attention(config, pg_collection).cuda()
        sparse_attention.train()

        seq_len = 64
        batch_size = 2
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        # Create one common input (all ranks create same input with same seed)
        query_input = torch.randn(
            seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        key_input = torch.randn(
            seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        value_input = torch.randn(
            seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        x_input = torch.randn(seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16).cuda()
        qr_input = torch.randn(seq_len, batch_size, config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Prepare input: split along seqlen if SP is enabled
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        if sequence_parallel:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            seq_per_rank = seq_len // tp_size
            start_idx = tp_rank * seq_per_rank
            end_idx = (tp_rank + 1) * seq_per_rank
            x = x_input[start_idx:end_idx]
            qr = qr_input[start_idx:end_idx]
        else:
            x = x_input
            qr = qr_input

        # query, key, value should be split along num_heads dim
        head_per_rank = num_heads // tensor_model_parallel_size
        start_head = tp_rank * head_per_rank
        end_head = (tp_rank + 1) * head_per_rank
        query = query_input[:, :, start_head:end_head, :]
        key = key_input[:, :, start_head:end_head, :]
        value = value_input[:, :, start_head:end_head, :]

        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

        # Forward and backward
        output = sparse_attention(
            query=query,
            key=key,
            value=value,
            x=x,
            qr=qr,
            attention_mask=attention_mask,
            attn_mask_type=AttnMaskType.causal,
        )

        loss = output.sum()
        loss.backward()

        # Check that gradients exist before sync
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None

        # Check that indexer parameters have gradients
        for name, param in sparse_attention.indexer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Indexer parameter {name} has no gradient"

        # Check that indexer gradients are identical within TP group
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            for name, param in sparse_attention.indexer.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Gather gradients from all ranks in TP group only
                    grad_list = [torch.zeros_like(param.grad) for _ in range(tp_size)]
                    torch.distributed.all_gather(grad_list, param.grad, group=pg_collection.tp)

                    # All gradients should be identical within TP group after sync
                    for i in range(1, tp_size):
                        assert torch.allclose(
                            grad_list[0], grad_list[i], rtol=0, atol=0
                        ), f"Indexer gradient for {name} differs between TP rank 0 and rank {i} after TP sync"

        Utils.destroy_model_parallel()
