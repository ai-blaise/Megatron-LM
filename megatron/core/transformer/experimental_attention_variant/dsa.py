# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_triton import (
    is_sparse_dsa_triton_supported,
    sparse_dsa_attention_triton,
)

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


_DSA_STREAMING_INDEXER_TOPK_ENV = "MEGATRON_DSA_STREAMING_INDEXER_TOPK"
_DSA_INDEXER_KEY_BLOCK_SIZE_ENV = "MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE"


def _env_flag_enabled(name: str, default: str = "1") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _dsa_indexer_key_block_size(topk: int) -> int:
    raw = os.getenv(_DSA_INDEXER_KEY_BLOCK_SIZE_ENV)
    if raw:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{_DSA_INDEXER_KEY_BLOCK_SIZE_ENV} must be positive, got {value}")
        return value
    return max(2048, min(4096, max(1, topk)))


def _dsa_process_group_size(group: Optional[torch.distributed.ProcessGroup]) -> int:
    return group.size() if group is not None else 1


def _dsa_process_group_rank(group: Optional[torch.distributed.ProcessGroup]) -> int:
    return group.rank() if group is not None else 0


def _dsa_cp_position_ids_for_rank(
    local_seq_len: int, cp_size: int, cp_rank: int, device: torch.device
) -> torch.Tensor:
    """Return absolute sequence positions for one CP rank's zigzag-local tokens."""

    if cp_size <= 1:
        return torch.arange(local_seq_len, device=device, dtype=torch.long)
    if local_seq_len % 2 != 0:
        raise ValueError(
            f"DSA CP position mapping expects an even local sequence length, got {local_seq_len}"
        )

    chunk_len = local_seq_len // 2
    first = torch.arange(
        cp_rank * chunk_len, (cp_rank + 1) * chunk_len, device=device, dtype=torch.long
    )
    second_chunk = 2 * cp_size - cp_rank - 1
    second = torch.arange(
        second_chunk * chunk_len,
        (second_chunk + 1) * chunk_len,
        device=device,
        dtype=torch.long,
    )
    return torch.cat((first, second), dim=0)


def _dsa_cp_local_position_ids(
    local_seq_len: int, cp_group: Optional[torch.distributed.ProcessGroup], device: torch.device
) -> torch.Tensor:
    return _dsa_cp_position_ids_for_rank(
        local_seq_len,
        _dsa_process_group_size(cp_group),
        _dsa_process_group_rank(cp_group),
        device,
    )


def _dsa_cp_gathered_position_ids(
    local_seq_len: int, cp_group: Optional[torch.distributed.ProcessGroup], device: torch.device
) -> torch.Tensor:
    cp_size = _dsa_process_group_size(cp_group)
    if cp_size <= 1:
        return torch.arange(local_seq_len, device=device, dtype=torch.long)
    return torch.cat(
        [
            _dsa_cp_position_ids_for_rank(local_seq_len, cp_size, cp_rank, device)
            for cp_rank in range(cp_size)
        ],
        dim=0,
    )


def _dsa_thd_local_sequence_offsets(
    cu_seqlens: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
) -> list[tuple[int, int]]:
    """Return local packed-THD offsets for each sequence on this CP rank."""

    cp_size = _dsa_process_group_size(cp_group)
    offsets = cu_seqlens.detach().cpu().tolist()
    local_offsets = []
    local_cursor = 0
    for start, end in zip(offsets[:-1], offsets[1:]):
        seq_len = int(end) - int(start)
        if seq_len <= 0:
            local_offsets.append((local_cursor, local_cursor))
            continue
        if seq_len % cp_size != 0:
            raise ValueError(
                f"DSA packed THD CP expects each padded sequence length to be divisible by "
                f"context_parallel_size; got sequence length {seq_len} and CP {cp_size}"
            )
        local_len = seq_len // cp_size
        local_offsets.append((local_cursor, local_cursor + local_len))
        local_cursor += local_len
    return local_offsets


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    assert (
        x.dtype == torch.bfloat16
    ), f"rotate_activation only support bf16 input, but got {x.dtype}"
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the indexer loss for logging.

        Args:
            loss: The loss tensor.
            layer_number: Layer index of the loss, 1-indexed.
            num_layers: The number of total layers.
            reduce_group: The group for reducing the loss.
            avg_group: The group for averaging the loss.
        """
        # Skip indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker():
        """Collect and reduce the indexer losses across ranks."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]

        torch.distributed.all_reduce(
            values, group=parallel_state.get_pipeline_model_parallel_group()
        )
        # Reduce indexer losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )
        torch.distributed.all_reduce(
            values,
            group=parallel_state.get_data_parallel_group(with_context_parallel=False),
            op=torch.distributed.ReduceOp.AVG,
        )

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer,
        wandb_writer=None,
        total_loss_dict=None,
        per_layer_logging: bool = False,
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            wandb_writer: Weights & Biases writer.
            total_loss_dict: Dictionary to accumulate total losses.
            per_layer_logging: Whether to log per-layer losses.
        """
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        indexer_loss_values = tracker["values"] * loss_scale
        num_layers = indexer_loss_values.shape[0]

        # Average across all layers (assuming all layers have sparse attention)
        avg_indexer_loss = indexer_loss_values.sum() / num_layers

        # Log average loss
        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss

        if writer is not None:
            writer.add_scalar("indexer loss", avg_indexer_loss, iteration)

        if wandb_writer is not None:
            wandb_writer.log({"indexer loss": avg_indexer_loss}, iteration)

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


class DSAIndexerAuxLossState:
    """Track per-microbatch DSA auxiliary losses for explicit loss composition."""

    losses = []

    @classmethod
    def add(cls, loss: torch.Tensor):
        cls.losses.append(loss)

    @classmethod
    def total(cls) -> Optional[torch.Tensor]:
        if not cls.losses:
            return None
        return torch.stack(cls.losses).sum()

    @classmethod
    def clear(cls):
        cls.losses.clear()


def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    sq, b, np, hn = query.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += causal_mask.view(1, 1, sq, sk)
    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    # [b, sq, sk] -> [b, sq, sk]
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    # L1 normalize target on the last dimension. Doesn't use abs() because attention_scores are
    # obtained from softmax so they are already non-negative.
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

    # Compute KL divergence: KL(target || index) = target(x) * log(target(x) / index(x))
    # kl_per_element [b, sq, sk]
    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )

    # [b, sq, sk] -> [b, sq] -> [1]
    # Each token has same weight in the loss.
    kl_div = kl_per_element.sum(dim=-1).mean()

    # Scale by coefficient.
    indexer_loss = kl_div * loss_coeff

    return indexer_loss


def _compute_index_scores(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Perform index score using BF16 precision.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
    This is a BF16 implementation of the `fp8_index` logic:
        1. Compute attention scores: q @ k^T;
        2. Apply ReLU activation;
        3. Weight by attention weights;
        4. Sum across attention heads.

    Args:
        q: BF16 [seqlen_q, batch, index_n_heads, index_head_dim], the query tensor.
        weights: BF16 [seqlen_q, batch, index_n_heads], the attention weights.
        k: BF16 [seqlen_k, batch, index_head_dim], the key tensor.

    Returns:
        index_scores: FP32 [batch, seqlen_q, seqlen_k], the index scores.
    """
    # Compute attention scores: q @ k^T
    # [seqlen_q, batch, index_n_heads, index_head_dim] @ [seqlen_k, batch, index_head_dim]^T
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

    # Keep this out-of-place: when the DSA KL loss is enabled, autograd needs
    # the ReLU output version to remain stable through the weight multiply.
    index_scores = torch.relu(index_scores)

    # Weight each head by attention weights.
    # [seqlen_q, batch, index_n_heads, seqlen_k] * [seqlen_q, batch, index_n_heads, 1]
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = index_scores * weights.unsqueeze(-1)

    # Sum across attention heads.
    # [seqlen_q, batch, index_n_heads, seqlen_k] -> [seqlen_q, batch, seqlen_k]
    index_scores = index_scores.sum(dim=2)

    # Transpose to [batch, seqlen_q, seqlen_k].
    index_scores = index_scores.transpose(0, 1)

    return index_scores


def fused_qk_topk_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    mask: Optional[torch.Tensor] = None,
):
    """Naive implementation of QK Topk."""
    seqlen = q.size(0)
    # =========================================
    # Compute index scores
    # =========================================
    # [batch, seqlen, seqlen]
    index_scores = _compute_index_scores(q, weights, k)
    if mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    # =========================================
    # Select top-k indices
    # =========================================
    topk_k = min(index_topk, seqlen)
    # [batch, seqlen, index_topk]
    topk_indices = index_scores.topk(topk_k, dim=-1)[1]

    return index_scores, topk_indices


def fwd_fused_indexer_loss_naive(
    q, weights, k, query, key, topk, softmax_scale, loss_coeff, mask, sparse_loss, pg_collection
):
    """Naive implementation of forward pass for indexer loss."""
    index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, topk, mask)

    indexer_loss = compute_dsa_indexer_loss(
        index_scores,
        topk_indices,
        query,
        key,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        pg_collection,
    )

    return topk_indices, indexer_loss


def bwd_fused_indexer_loss_naive(
    q,
    weights,
    k,
    query,
    key,
    topk_indices,
    softmax_scale,
    loss_coeff,
    sparse_loss,
    grad_loss,
    pg_collection,
):
    """Naive implementation of backward pass for indexer loss."""
    index_scores = _compute_index_scores(q, weights, k)  # [B, Sq, Sk]

    sq, b, np, hn = query.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query_reshaped = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key_reshaped = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query_reshaped.float(), key_reshaped.float()) * softmax_scale
    # Free reshaped tensors - no longer needed after bmm
    del query_reshaped, key_reshaped

    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # Apply causal mask to both attention and index scores
    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores = attention_scores + causal_mask.view(1, 1, sq, sk)
    # [b, sq, sk] + [1, sq, sk] -> [b, sq, sk]
    index_scores = index_scores + causal_mask.unsqueeze(0)
    # Free causal_mask - no longer needed
    del causal_mask

    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_mask

    # Compute softmax for both
    attention_scores_softmax = torch.nn.functional.softmax(
        attention_scores, dim=-1, dtype=torch.float32
    )
    # Free attention_scores immediately
    del attention_scores

    index_scores_softmax = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)
    # Free index_scores - no longer needed after softmax
    del index_scores

    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    # Free attention_scores_softmax
    del attention_scores_softmax

    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores_sum.contiguous(), group=pg_collection.tp)

    # L1 normalize
    attention_scores_normalized = attention_scores_sum / attention_scores_sum.sum(
        dim=-1, keepdim=True
    )
    # Free attention_scores_sum - no longer needed after normalization
    del attention_scores_sum

    # Backward through loss = kl_div * loss_coeff
    # where kl_div = kl_per_element.sum(dim=-1).mean()
    grad_kl_div = grad_loss * loss_coeff  # scalar

    # Backward through mean: distribute gradient equally
    grad_kl_per_row = grad_kl_div / (b * sq)  # scalar value for each row

    # Backward through sum(dim=-1): broadcast back to [b, sq, sk]
    # Each element in a row contributes to the sum, so gradient is same for all
    grad_kl_per_element = grad_kl_per_row.view(1, 1, 1).expand(b, sq, sk)

    # Backward through kl_per_element = target * (log(target) - log(index))
    # ∂kl/∂index_softmax = -target / index_softmax
    grad_index_scores_softmax = (
        -attention_scores_normalized / (index_scores_softmax + 1e-10) * grad_kl_per_element
    )
    # Free attention_scores_normalized - no longer needed
    del attention_scores_normalized

    # Backward through softmax: ∂L/∂x = softmax * (∂L/∂softmax - sum(∂L/∂softmax * softmax))
    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(dim=-1, keepdim=True)
    grad_index_scores_logits = index_scores_softmax * (grad_index_scores_softmax - sum_grad)
    # Free intermediate tensors
    del index_scores_softmax, grad_index_scores_softmax, sum_grad

    # Zero out gradients for masked positions
    # Create a mask for valid (non-masked) positions
    # Causal mask: position (i, j) is valid if j <= i
    causal_valid_mask = torch.tril(
        torch.ones((sq, sk), device=q.device, dtype=torch.bool)
    )  # [sq, sk]
    if sparse_loss:
        # Also apply index mask - only topk positions are valid
        index_valid_mask = index_mask == 0  # [b, sq, sk]
        del index_mask  # Free index_mask immediately after use
        valid_mask = causal_valid_mask.unsqueeze(0) & index_valid_mask  # [b, sq, sk]
        del index_valid_mask
    else:
        del index_mask  # Free index_mask even if not used for sparse_loss
        valid_mask = causal_valid_mask.unsqueeze(0).expand(b, sq, sk)  # [b, sq, sk]
    del causal_valid_mask

    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    del valid_mask

    # Transpose from [b, sq, sk] to [sq, b, sk]
    grad_index_scores = grad_index_scores_logits.transpose(0, 1)  # [sq, b, sk]
    del grad_index_scores_logits

    # Backward through sum over heads: expand gradient
    grad_weighted_scores = grad_index_scores.unsqueeze(2)  # [sq, b, 1, sk]
    del grad_index_scores

    # Compute forward values needed for backward
    scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())  # [sq, b, h, sk]
    # Compute relu_mask before relu (saves memory vs keeping both scores and relu output)
    relu_mask = scores > 0
    scores_after_relu = torch.relu(scores)
    del scores

    # Backward through multiplication by weights: index_scores_per_head * weights
    # ∂L/∂weights = grad * relu_scores (sum over sk)
    grad_weights = (grad_weighted_scores * scores_after_relu).sum(dim=-1)  # [sq, b, h]

    # ∂L/∂relu_scores = grad * weights
    grad_scores_after_relu = grad_weighted_scores * weights.unsqueeze(-1)  # [sq, b, h, sk]
    del grad_weighted_scores, scores_after_relu

    # Backward through ReLU
    grad_scores = grad_scores_after_relu * relu_mask.float()  # [sq, b, h, sk]
    del grad_scores_after_relu, relu_mask

    # Backward through einsum 'sbhd,tbd->sbht'
    # ∂L/∂q = einsum('sbht,tbd->sbhd', grad_scores, k)
    grad_q = torch.einsum('sbht,tbd->sbhd', grad_scores, k.float())  # [sq, b, h, d]
    # ∂L/∂k = einsum('sbht,sbhd->tbd', grad_scores, q)
    grad_k = torch.einsum('sbht,sbhd->tbd', grad_scores, q.float())  # [sk, b, d]
    del grad_scores

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


class FusedDSAIndexerLoss(torch.autograd.Function):
    """Fused implementation of DSA Indexer Loss."""

    @staticmethod
    def forward(
        ctx,
        q,
        weights,
        k,
        query,
        key,
        softmax_scale,
        topk,
        loss_coeff,
        mask,
        sparse_loss,
        pg_collection,
    ):
        """
        Fused forward: index_scores never materialized in full.
        """
        topk_indices, loss = fwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk,
            softmax_scale,
            loss_coeff,
            mask,
            sparse_loss,
            pg_collection,
        )

        # Save for backward (recomputation strategy)
        ctx.save_for_backward(q, weights, k, query, key, topk_indices)
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        ctx.pg_collection = pg_collection

        return topk_indices, loss

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss):
        """
        Backward: Recompute what we need.
        """
        q, weights, k, query, key, topk_indices = ctx.saved_tensors

        grad_q, grad_weights, grad_k = bwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            ctx.softmax_scale,
            ctx.loss_coeff,
            ctx.sparse_loss,
            grad_loss,
            ctx.pg_collection,
        )

        # query and key are detached in forward, so return None for their gradients
        return grad_q, grad_weights, grad_k, None, None, None, None, None, None, None, None


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for indexer loss.

    This custom autograd function attaches a KL divergence loss to the activation
    to train the indexer to predict attention scores without affecting the forward pass.
    """

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output: The output tensor (activation).
            indexer_loss: The indexer KL divergence loss tensor.

        Returns:
            torch.Tensor: The output tensor unchanged.
        """
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output: The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                gradient.
        """
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale: The scale value to set.
        """
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


@dataclass
class DSAIndexerSubmodules:
    """
    Configuration class for specifying the submodules of an DSA Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_wk: Linear projection for key.
        k_norm: Layer normalization for key.
        linear_weights_proj: Linear projection for attention weights.
    """

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


@dataclass
class DSAttentionSubmodules:
    """
    Configuration class for specifying the submodules of DSAttention.

    Args:
        indexer: DSA Indexer module for computing sparse attention indices.
    """

    indexer: Union[ModuleSpec, type] = None


class DSAIndexer(MegatronModule):
    """
    DSA Lightning Indexer for DeepSeek Sparse Attention.

    Computes index scores to identify the top-k most relevant key-value pairs for each query in
    sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            submodules (DSAIndexerSubmodules): Indexer submodules specification.
            pg_collection (ProcessGroupCollection, optional): Process groups for the indexer.
        """
        super().__init__(config=config)
        self.hidden_size = self.config.hidden_size
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            self.config.q_lora_rank
            if self.config.q_lora_rank is not None
            else self.config.hidden_size
        )

        self.index_n_heads = self.config.dsa_indexer_n_heads
        self.index_head_dim = self.config.dsa_indexer_head_dim
        self.index_topk = self.config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        self.indexcache_config = None
        if getattr(self.config, "dsa_indexcache_quant_enabled", False):
            from megatron.core.quantization.indexcache import build_indexcache_config

            self.indexcache_config = build_indexcache_config(
                eps=getattr(self.config, "dsa_indexcache_quant_eps", 1e-4)
            )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        # Initialize Position Embedding.
        if self.config.rope_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
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
                f'Unsupported RoPE type: {self.config.rope_type}, supported types are "rope" and '
                f'"yarn"'
            )

        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_wk = build_module(
            submodules.linear_wk,
            self.hidden_size,
            self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        k_norm_config = copy.copy(self.config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            config=k_norm_config,
            hidden_size=self.index_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _apply_rope(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor, mscale: float):
        """Apply RoPE to the input tensor."""
        # x_pe   [seqlen, batch, *, qk_pos_emb_head_dim]
        # x_nope [seqlen, batch, *, index_head_dim - qk_pos_emb_head_dim]
        # To align with DeepSeek's implementation,
        # x_pe is placed at the front, and x_nope is placed at the back.
        x_pe, x_nope = torch.split(
            x, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
            # This flag is for the MLA-style interleaving in RoPE.
            # Set it to False, as indexer does not apply interleaved RoPE.
            mla_rotary_interleaved=False,
        )
        # [seqlen, batch, *, index_head_dim]
        x = torch.cat([x_pe, x_nope], dim=-1)
        return x

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All computations before topk."""
        # =========================================
        # Prepare RoPE params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

        # =========================================
        # Gather inputs if sp is enabled
        # =========================================
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        # =========================================
        # Get sequence length and batch size
        # =========================================
        seqlen, bsz, _ = x.size()

        # =========================================
        # q linear and apply rope to q
        # =========================================
        # [seqlen, batch, q_lora_rank] -> [seqlen, batch, index_n_heads * index_head_dim]
        q, _ = self.linear_wq_b(qr)
        # [seqlen, batch, index_n_heads * index_head_dim]
        #   -> [seqlen, batch, index_n_heads, index_head_dim]
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, rotary_pos_emb, mscale)

        # =========================================
        # k linear and apply rope to k
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_head_dim]
        k, _ = self.linear_wk(x)
        k = self.k_norm(k)
        # [seqlen, batch, index_head_dim] -> [seqlen, batch, 1, index_head_dim]
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        k = self._apply_rope(k, rotary_pos_emb, mscale)
        # [seqlen, batch, 1, index_head_dim] -> [seqlen, batch, index_head_dim]
        k = k.reshape(seqlen, bsz, self.index_head_dim)

        # =========================================
        # Rotate activation
        # =========================================
        q = rotate_activation(q)
        k = rotate_activation(k)

        # IndexCache fp8 fake-quant on the post-rotation indexer K. K only —
        # SGLang's reference quantizes the indexer key cache, not the query.
        if self.indexcache_config is not None:
            from megatron.core.quantization.indexcache import apply_indexcache_kv

            k = apply_indexcache_kv(k, self.indexcache_config)

        # =========================================
        # Prepare weights for index scores
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

        return q, k, weights

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for DSA Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen].
            topk_indices: Top-k indices [batch, seqlen, index_topk].
        """
        assert packed_seq_params is None, "Packed sequence is not supported for DSAttention"

        # [seqlen, batch, index_n_heads * index_head_dim]
        # [seqlen, batch, index_head_dim]
        # [seqlen, batch, index_n_heads]
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        # [batch, seqlen, seqlen], [batch, seqlen, index_topk]
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, self.index_topk, mask)

        return index_scores, topk_indices

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass for DSA Indexer.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
        """
        _, topk_indices = self.forward_with_scores(x, qr, mask, packed_seq_params)
        return topk_indices


def unfused_dsa_fn(query, key, value, topk_indices, softmax_scale):
    """
    Unfused sparse attention implementation.
    """
    sq, b, np, hn = query.size()
    skv = key.size(0)
    hnv = value.size(3)

    # ===================================
    # Raw attention scores [b, np, sq, skv]
    # ===================================
    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [skv, b, np, hn] -> [b, np, hn, skv] -> [b * np, hn, skv]
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, skv)
    # Compute attention scores [b * np, sq, skv]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, skv]
    attention_scores = attention_scores.reshape(b, np, sq, skv)

    # ===================================
    # Apply sparse mask from indexer
    # ===================================
    # index_mask [b, sq, skv]
    index_mask = torch.full((b, sq, skv), float("-inf"), device=attention_scores.device)
    index_mask.scatter_(-1, topk_indices, 0)
    # causal_mask [sq, skv]
    causal_mask = torch.triu(
        torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=index_mask.device),
        diagonal=1,
    )
    # [b, sq, skv] + [1, sq, skv] -> [b, sq, skv]
    index_mask += causal_mask.view(1, sq, skv)
    # [b, np, sq, skv] + [b, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += index_mask.unsqueeze(1)
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)

    # ===================================
    # Output
    # ===================================
    # [skv, b, np, hnv] -> [b, np, skv, hnv] -> [b * np, skv, hnv]
    value = value.permute(1, 2, 0, 3).reshape(b * np, skv, hnv)
    # Reshape attention_scores: [b, np, sq, skv] -> [b * np, sq, skv]
    attention_scores = attention_scores.reshape(b * np, sq, skv)
    # Compute output: [b * np, sq, hnv]
    output = torch.bmm(attention_scores.to(value.dtype), value)
    # Reshape output: [b * np, sq, hnv] -> [b, np, sq, hnv] -> [sq, b, np, hnv]
    output = output.reshape(b, np, sq, hnv).permute(2, 0, 1, 3).contiguous()
    # Flatten: [sq, b, np, hnv] -> [sq, b, np * hnv]
    output = output.reshape(sq, b, np * hnv)
    return output


def _apply_dsa_score_mask(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    q_start: int,
    q_end: int,
    sk: int,
    is_causal: bool,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if is_causal:
        if query_positions is not None or key_positions is not None:
            if query_positions is None or key_positions is None:
                raise ValueError("query_positions and key_positions must be provided together")
            q_pos = query_positions.to(device=scores.device)
            k_pos = key_positions.to(device=scores.device)
            if q_pos.numel() != q_end - q_start:
                raise ValueError(
                    f"query_positions length {q_pos.numel()} does not match chunk length "
                    f"{q_end - q_start}"
                )
            if k_pos.numel() != sk:
                raise ValueError(f"key_positions length {k_pos.numel()} does not match sk {sk}")
        else:
            q_pos = torch.arange(q_start, q_end, device=scores.device)
            k_pos = torch.arange(sk, device=scores.device)
        if scores.dim() == 4:
            q_pos = q_pos.view(1, 1, -1, 1)
            k_pos = k_pos.view(1, 1, 1, -1)
        else:
            q_pos = q_pos.view(1, -1, 1)
            k_pos = k_pos.view(1, 1, -1)
        return scores.masked_fill(k_pos > q_pos, float("-inf"))
    if mask is None:
        return scores
    if mask.dim() == 2:
        return scores + mask[q_start:q_end, :].unsqueeze(0)
    if mask.dim() == 3:
        mask_slice = mask[:, q_start:q_end, :]
        if scores.dim() == 4:
            mask_slice = mask_slice.unsqueeze(1)
        return scores + mask_slice
    raise ValueError(f"DSA mask must be 2D or 3D, got shape {tuple(mask.shape)}")


def _streaming_qk_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk: int,
    mask: Optional[torch.Tensor],
    q_start: int,
    q_end: int,
    sk: int,
    is_causal: bool,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute exact DSA indexer top-k without materializing all per-head scores."""

    topk_k = min(topk, sk)
    key_block_size = _dsa_indexer_key_block_size(topk_k)
    running_scores = None
    running_indices = None

    for key_start in range(0, sk, key_block_size):
        key_end = min(key_start + key_block_size, sk)
        key_block = k[key_start:key_end]

        block_scores = torch.einsum("sbhd,tbd->sbht", q.float(), key_block.float())
        block_scores = torch.relu_(block_scores)
        block_scores.mul_(weights.unsqueeze(-1))
        block_scores = block_scores.sum(dim=2).transpose(0, 1)

        if is_causal:
            if query_positions is not None or key_positions is not None:
                if query_positions is None or key_positions is None:
                    raise ValueError("query_positions and key_positions must be provided together")
                q_pos = query_positions.to(device=block_scores.device).view(1, -1, 1)
                k_pos = key_positions[key_start:key_end].to(device=block_scores.device).view(
                    1, 1, -1
                )
            else:
                q_pos = torch.arange(q_start, q_end, device=block_scores.device).view(1, -1, 1)
                k_pos = torch.arange(key_start, key_end, device=block_scores.device).view(1, 1, -1)
            block_scores = block_scores.masked_fill(k_pos > q_pos, float("-inf"))
        elif mask is not None:
            if mask.dim() == 2:
                block_scores = block_scores + mask[q_start:q_end, key_start:key_end].unsqueeze(0)
            elif mask.dim() == 3:
                block_scores = block_scores + mask[:, q_start:q_end, key_start:key_end]
            else:
                raise ValueError(f"DSA mask must be 2D or 3D, got shape {tuple(mask.shape)}")

        block_topk = min(topk_k, key_end - key_start)
        block_scores, block_indices = block_scores.topk(block_topk, dim=-1)
        block_indices = block_indices + key_start

        if running_scores is None:
            running_scores = block_scores
            running_indices = block_indices
        else:
            candidate_scores = torch.cat((running_scores, block_scores), dim=-1)
            candidate_indices = torch.cat((running_indices, block_indices), dim=-1)
            keep_k = min(topk_k, candidate_scores.size(-1))
            running_scores, selected = candidate_scores.topk(keep_k, dim=-1)
            running_indices = candidate_indices.gather(-1, selected)

    return running_indices


def _sparse_dsa_attention_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    mask: Optional[torch.Tensor],
    q_start: int,
    is_causal: bool,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q_len, bsz, num_heads, head_dim = query.size()
    value_head_dim = value.size(3)
    topk_k = topk_indices.size(-1)
    topk_block_size = 64
    batch_outputs = []

    for batch_idx in range(bsz):
        selected_positions = topk_indices[batch_idx]
        query_batch = query[:, batch_idx]
        key_batch = key[:, batch_idx]
        score_blocks = []
        for topk_start in range(0, topk_k, topk_block_size):
            topk_end = min(topk_start + topk_block_size, topk_k)
            block_positions = selected_positions[:, topk_start:topk_end]
            block_index = block_positions.reshape(-1)
            key_block = key_batch.index_select(0, block_index)
            key_block = key_block.view(q_len, topk_end - topk_start, num_heads, head_dim)
            score_blocks.append(
                torch.einsum("qhd,qkhd->qkh", query_batch.float(), key_block.float())
            )
        attention_scores = torch.cat(score_blocks, dim=1) * softmax_scale

        if is_causal:
            if query_positions is not None or key_positions is not None:
                if query_positions is None or key_positions is None:
                    raise ValueError("query_positions and key_positions must be provided together")
                q_pos = query_positions.to(device=query.device).unsqueeze(1)
                selected_abs_positions = key_positions.to(device=query.device).index_select(
                    0, selected_positions.reshape(-1).long()
                )
                selected_abs_positions = selected_abs_positions.view_as(selected_positions)
                invalid = selected_abs_positions > q_pos
            else:
                q_pos = torch.arange(q_start, q_start + q_len, device=query.device).unsqueeze(1)
                invalid = selected_positions > q_pos
            attention_scores = attention_scores.masked_fill(invalid.unsqueeze(-1), float("-inf"))
        elif mask is not None:
            if mask.dim() == 2:
                selected_mask = mask[q_start : q_start + q_len, :].gather(1, selected_positions)
            elif mask.dim() == 3:
                selected_mask = mask[batch_idx, q_start : q_start + q_len, :].gather(
                    1, selected_positions
                )
            else:
                raise ValueError(f"DSA mask must be 2D or 3D, got shape {tuple(mask.shape)}")
            attention_scores = attention_scores + selected_mask.unsqueeze(-1)

        attention_probs = torch.softmax(attention_scores, dim=1, dtype=torch.float32).to(
            value.dtype
        )
        del attention_scores

        value_batch = value[:, batch_idx]
        output_batch = None
        for topk_start in range(0, topk_k, topk_block_size):
            topk_end = min(topk_start + topk_block_size, topk_k)
            block_positions = selected_positions[:, topk_start:topk_end]
            block_index = block_positions.reshape(-1)
            selected_value = value_batch.index_select(0, block_index)
            selected_value = selected_value.view(
                q_len, topk_end - topk_start, num_heads, value_head_dim
            )
            block_output = torch.einsum(
                "qkh,qkhd->qhd", attention_probs[:, topk_start:topk_end, :], selected_value
            )
            output_batch = block_output if output_batch is None else output_batch + block_output
        del attention_probs
        batch_outputs.append(output_batch)

    output = torch.stack(batch_outputs, dim=1)
    return output.reshape(q_len, bsz, num_heads * value_head_dim)


def chunked_dsa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    topk: int,
    mask: Optional[torch.Tensor],
    is_causal: bool,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
    chunk_size: int,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    sq, bsz, num_heads, head_dim = query.size()
    sk = key.size(0)
    outputs = []
    topk_chunks = []
    use_triton_attention = None
    use_streaming_indexer_topk = loss_coeff <= 0 and _env_flag_enabled(
        _DSA_STREAMING_INDEXER_TOPK_ENV, "1"
    )
    loss_sum = None
    loss_count = 0

    for q_start in range(0, sq, chunk_size):
        q_end = min(q_start + chunk_size, sq)
        q_chunk = q[q_start:q_end]
        weights_chunk = weights[q_start:q_end]
        query_chunk = query[q_start:q_end]
        query_positions_chunk = (
            None if query_positions is None else query_positions[q_start:q_end]
        )

        if use_streaming_indexer_topk:
            index_scores = None
            topk_indices = _streaming_qk_topk(
                q_chunk,
                weights_chunk,
                k,
                topk,
                mask,
                q_start,
                q_end,
                sk,
                is_causal,
                query_positions=query_positions_chunk,
                key_positions=key_positions,
            )
        else:
            index_scores = _compute_index_scores(q_chunk, weights_chunk, k)
            index_scores = _apply_dsa_score_mask(
                index_scores,
                mask,
                q_start,
                q_end,
                sk,
                is_causal,
                query_positions=query_positions_chunk,
                key_positions=key_positions,
            )
            topk_k = min(topk, sk)
            topk_indices = index_scores.topk(topk_k, dim=-1)[1]

        if use_triton_attention is None:
            use_triton_attention = is_sparse_dsa_triton_supported(
                query_chunk,
                key,
                value,
                topk_indices,
                mask,
                is_causal,
                query_positions=query_positions_chunk,
                key_positions=key_positions,
            )

        if loss_coeff > 0:
            loss_index_scores = index_scores
            attention_query = query_chunk.detach().permute(1, 2, 0, 3).reshape(
                bsz * num_heads, q_end - q_start, head_dim
            )
            attention_key = key.detach().permute(1, 2, 3, 0).reshape(
                bsz * num_heads, head_dim, sk
            )
            attention_scores = torch.bmm(attention_query.float(), attention_key.float())
            attention_scores = attention_scores.reshape(bsz, num_heads, q_end - q_start, sk)
            attention_scores = attention_scores * softmax_scale
            attention_scores = _apply_dsa_score_mask(
                attention_scores,
                mask,
                q_start,
                q_end,
                sk,
                is_causal,
                query_positions=query_positions_chunk,
                key_positions=key_positions,
            )

            if sparse_loss:
                index_mask = torch.full_like(loss_index_scores, float("-inf"))
                index_mask.scatter_(-1, topk_indices, 0)
                loss_index_scores = loss_index_scores + index_mask
                attention_scores = attention_scores + index_mask.unsqueeze(1)

            attention_probs = torch.softmax(attention_scores, dim=-1, dtype=torch.float32)
            index_probs = torch.softmax(loss_index_scores, dim=-1, dtype=torch.float32)

            attention_probs = attention_probs.sum(dim=1)
            if pg_collection.tp.size() > 1:
                torch.distributed.all_reduce(attention_probs.contiguous(), group=pg_collection.tp)
            attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)

            kl = attention_probs * (
                torch.log(attention_probs + 1e-10) - torch.log(index_probs + 1e-10)
            )
            chunk_loss_sum = kl.sum(dim=-1).sum()
            loss_sum = chunk_loss_sum if loss_sum is None else loss_sum + chunk_loss_sum
            loss_count += bsz * (q_end - q_start)
        elif index_scores is not None:
            del index_scores

        if use_triton_attention:
            topk_chunks.append(topk_indices.to(torch.int32))
        else:
            outputs.append(
                _sparse_dsa_attention_chunk(
                    query_chunk,
                    key,
                    value,
                    topk_indices,
                    softmax_scale,
                    mask,
                    q_start,
                    is_causal,
                    query_positions=query_positions_chunk,
                    key_positions=key_positions,
                )
            )

    if use_triton_attention:
        # Keep top-k generation chunked to bound the indexer score tensor, then run
        # the selected-token attention as one autograd op so K/V gradients are
        # accumulated once per layer instead of once per query chunk.
        topk_indices = torch.cat(topk_chunks, dim=1)
        topk_chunks.clear()
        output = sparse_dsa_attention_triton(
            query,
            key,
            value,
            topk_indices,
            softmax_scale,
            0,
            query_positions=query_positions,
            key_positions=key_positions,
        )
    else:
        output = torch.cat(outputs, dim=0)

    indexer_loss = None
    if loss_sum is not None:
        indexer_loss = loss_sum * (loss_coeff / loss_count)
    return output, indexer_loss


class DSAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an DSA Indexer to compute top-k
    attention indices for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.layer_number = layer_number

        self.indexer = build_module(
            submodules.indexer, config=self.config, pg_collection=pg_collection
        )
        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0) or 0.0
        if indexer_loss_coeff <= 0:
            for param in self.indexer.parameters():
                param.requires_grad_(False)

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor [sq, b, np, hn].
            key: Key tensor [skv, b, np, hn].
            value: Value tensor [skv, b, np, hnv].
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, hidden_size]
        """
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            cp_group = self.indexer.pg_collection.cp
            cp_size = _dsa_process_group_size(cp_group)
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            if cu_seqlens is None:
                raise ValueError("DSAttention THD path requires cu_seqlens")
            if cu_seqlens.dim() == 2:
                if cu_seqlens.size(0) != 1:
                    raise ValueError(
                        f"DSAttention THD path expects micro-batch-size 1, got "
                        f"cu_seqlens shape {tuple(cu_seqlens.shape)}"
                    )
                cu_seqlens = cu_seqlens[0]
            if cu_seqlens.dim() != 1:
                raise ValueError(f"cu_seqlens must be 1D, got shape {tuple(cu_seqlens.shape)}")

            if query.dim() == 4:
                if query.size(1) != 1 or key.size(1) != 1 or value.size(1) != 1:
                    raise ValueError("DSAttention THD path only supports a dummy batch dimension")
                query = query.squeeze(1)
                key = key.squeeze(1)
                value = value.squeeze(1)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            if qr.dim() == 2:
                qr = qr.unsqueeze(1)

            if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
                raise ValueError(
                    "DSAttention THD path expects query/key/value as [tokens, heads, dim]"
                )
            if x.dim() != 3 or qr.dim() != 3:
                raise ValueError("DSAttention THD path expects x/qr as [tokens, batch, dim]")

            outputs = []
            if cp_size > 1:
                local_offsets = _dsa_thd_local_sequence_offsets(cu_seqlens, cp_group)
            else:
                offsets = cu_seqlens.detach().cpu().tolist()
                local_offsets = [
                    (int(start), int(end)) for start, end in zip(offsets[:-1], offsets[1:])
                ]
            for local_start, local_end in local_offsets:
                if local_end <= local_start:
                    continue
                outputs.append(
                    self.forward(
                        query[local_start:local_end].unsqueeze(1),
                        key[local_start:local_end].unsqueeze(1),
                        value[local_start:local_end].unsqueeze(1),
                        attention_mask,
                        x[local_start:local_end],
                        qr[local_start:local_end],
                        attn_mask_type=attn_mask_type,
                        attention_bias=attention_bias,
                        packed_seq_params=None,
                    )
                )

            if not outputs:
                return query.new_empty((0, 1, value.size(1) * value.size(2)))
            return torch.cat(outputs, dim=0)

        sq, b, np, hn = query.size()
        skv = key.size(0)
        hnv = value.size(3)

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        x = x.detach()
        qr = qr.detach()

        is_causal = False
        # Get a FP32 mask with -inf for masked positions.
        if attn_mask_type is not None:
            assert attn_mask_type == AttnMaskType.causal, 'Only causal mask is supported for now'
            is_causal = True
            float_mask = None
        else:
            assert attention_mask.shape == (b, 1, sq, skv), 'attention_mask shape mismatch'
            # [b, 1, sq, skv] -> [b, sq, skv]
            mask = attention_mask.squeeze()
            # float_mask [b, sq, skv]
            float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                mask, float('-inf')
            )

        q, k, weights = self.indexer.forward_before_topk(x, qr, packed_seq_params)
        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)
        chunk_size = int(getattr(self.config, "dsa_chunk_size", 128))
        query_positions = None
        key_positions = None
        cp_group = self.indexer.pg_collection.cp
        cp_size = _dsa_process_group_size(cp_group)
        if cp_size > 1:
            if not is_causal:
                raise NotImplementedError("DSAttention CP path currently supports causal masks only")
            local_skv = key.size(0)
            query_positions = _dsa_cp_local_position_ids(sq, cp_group, query.device)
            key_positions = _dsa_cp_gathered_position_ids(local_skv, cp_group, key.device)
            k = gather_from_sequence_parallel_region(k, group=cp_group)
            key = gather_from_sequence_parallel_region(key, group=cp_group)
            value = gather_from_sequence_parallel_region(value, group=cp_group)
        output, indexer_loss = chunked_dsa_forward(
            q,
            k,
            weights,
            query,
            key,
            value,
            self.softmax_scale,
            self.indexer.index_topk,
            float_mask,
            is_causal,
            indexer_loss_coeff if self.training and torch.is_grad_enabled() else 0.0,
            getattr(self.config, "dsa_indexer_use_sparse_loss", False),
            self.indexer.pg_collection,
            chunk_size,
            query_positions=query_positions,
            key_positions=key_positions,
        )
        if indexer_loss is not None:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss, layer_number=self.layer_number, num_layers=self.config.num_layers
            )
            DSAIndexerAuxLossState.add(indexer_loss)

        return output
