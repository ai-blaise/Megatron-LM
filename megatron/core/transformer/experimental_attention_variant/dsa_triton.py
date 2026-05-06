# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import os
from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = version.parse(triton.__version__) >= version.parse("2.0.0")
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


_DSA_TRITON_ENV = "MEGATRON_DSA_TRITON"
_DSA_TRITON_BLOCK_K_ENV = "MEGATRON_DSA_TRITON_BLOCK_K"


def _env_enabled() -> bool:
    raw = os.getenv(_DSA_TRITON_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _block_k(topk: int) -> int:
    raw = os.getenv(_DSA_TRITON_BLOCK_K_ENV)
    if raw:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{_DSA_TRITON_BLOCK_K_ENV} must be positive, got {value}")
        return min(triton.next_power_of_2(value), 256)
    if topk >= 1024:
        return 64
    return min(triton.next_power_of_2(topk), 128)


def is_sparse_dsa_triton_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
) -> bool:
    """Return whether the fused sparse DSA attention kernel can handle this call."""

    if not _env_enabled() or not HAVE_TRITON:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda and topk_indices.is_cuda):
        return False
    if query.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if topk_indices.dtype not in (torch.int32, torch.int64):
        return False
    if (query_positions is None) != (key_positions is None):
        return False
    if query_positions is not None:
        if not (query_positions.is_cuda and key_positions.is_cuda):
            return False
        if query_positions.dim() != 1 or key_positions.dim() != 1:
            return False
        if query_positions.size(0) != query.size(0) or key_positions.size(0) != key.size(0):
            return False
    if mask is not None or not is_causal:
        return False
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False
    if topk_indices.dim() != 3:
        return False
    if query.size(1) != 1 or key.size(1) != 1 or value.size(1) != 1:
        return False
    if topk_indices.size(0) != 1 or topk_indices.size(1) != query.size(0):
        return False
    if query.size(2) != key.size(2) or query.size(2) != value.size(2):
        return False
    if query.size(3) != key.size(3):
        return False
    # DeepSeek-V3.2 MLA passes Q/K as qk_head_dim + qk_pos_emb_head_dim and V as
    # v_head_dim. Keep this first kernel bounded to the model shapes we need.
    if query.size(3) <= 0 or query.size(3) > 256:
        return False
    if value.size(3) <= 0 or value.size(3) > 256:
        return False
    if topk_indices.size(-1) <= 0 or topk_indices.size(-1) > key.size(0):
        return False
    return True


@triton.jit
def _sparse_dsa_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    topk_ptr,
    query_pos_ptr,
    key_pos_ptr,
    output_ptr,
    lse_ptr,
    softmax_scale,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    topk_count: tl.constexpr,
    q_start,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
):
    q_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    d_offsets = tl.arange(0, BLOCK_D)
    topk_offsets = tl.arange(0, BLOCK_K)

    q_ptrs = query_ptr + (q_idx * num_heads + head_idx) * head_dim + d_offsets
    query = tl.load(q_ptrs, mask=d_offsets < head_dim, other=0.0).to(tl.float32)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    if HAS_POSITIONS:
        q_abs = tl.load(query_pos_ptr + q_idx)
    else:
        q_abs = q_start + q_idx

    for topk_start in tl.range(0, topk_count, BLOCK_K):
        k_offsets = topk_start + topk_offsets
        valid_topk = k_offsets < topk_count
        selected = tl.load(topk_ptr + q_idx * topk_count + k_offsets, mask=valid_topk, other=0)
        if HAS_POSITIONS:
            selected_abs = tl.load(key_pos_ptr + selected, mask=valid_topk, other=0)
            valid = valid_topk & (selected_abs <= q_abs)
        else:
            valid = valid_topk & (selected <= q_abs)

        key_ptrs = (
            key_ptr + (selected[:, None] * num_heads + head_idx) * head_dim + d_offsets[None, :]
        )
        key = tl.load(
            key_ptrs, mask=valid[:, None] & (d_offsets[None, :] < head_dim), other=0.0
        ).to(tl.float32)
        scores = tl.sum(key * query[None, :], axis=1) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))

        block_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_m)
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(scores - m_new)

        value_ptrs = (
            value_ptr + (selected[:, None] * num_heads + head_idx) * value_dim + d_offsets[None, :]
        )
        value = tl.load(
            value_ptrs, mask=valid[:, None] & (d_offsets[None, :] < value_dim), other=0.0
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * value, axis=0)
        l_i = l_i * alpha + tl.sum(probs, axis=0)
        m_i = m_new

    output = acc / l_i
    output_ptrs = output_ptr + (q_idx * num_heads + head_idx) * value_dim + d_offsets
    tl.store(output_ptrs, output, mask=d_offsets < value_dim)
    tl.store(lse_ptr + q_idx * num_heads + head_idx, m_i + tl.log(l_i))


@triton.jit
def _sparse_dsa_backward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    topk_ptr,
    query_pos_ptr,
    key_pos_ptr,
    output_ptr,
    lse_ptr,
    grad_output_ptr,
    grad_query_ptr,
    grad_key_ptr,
    grad_value_ptr,
    softmax_scale,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    value_dim: tl.constexpr,
    topk_count: tl.constexpr,
    q_start,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
):
    q_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    d_offsets = tl.arange(0, BLOCK_D)
    topk_offsets = tl.arange(0, BLOCK_K)

    q_ptrs = query_ptr + (q_idx * num_heads + head_idx) * head_dim + d_offsets
    query = tl.load(q_ptrs, mask=d_offsets < head_dim, other=0.0).to(tl.float32)

    grad_output_ptrs = grad_output_ptr + (q_idx * num_heads + head_idx) * value_dim + d_offsets
    grad_output = tl.load(grad_output_ptrs, mask=d_offsets < value_dim, other=0.0).to(tl.float32)

    output_ptrs = output_ptr + (q_idx * num_heads + head_idx) * value_dim + d_offsets
    output = tl.load(output_ptrs, mask=d_offsets < value_dim, other=0.0).to(tl.float32)
    row_lse = tl.load(lse_ptr + q_idx * num_heads + head_idx).to(tl.float32)
    delta = tl.sum(grad_output * output, axis=0)

    grad_query = tl.zeros((BLOCK_D,), tl.float32)
    if HAS_POSITIONS:
        q_abs = tl.load(query_pos_ptr + q_idx)
    else:
        q_abs = q_start + q_idx

    for topk_start in tl.range(0, topk_count, BLOCK_K):
        k_offsets = topk_start + topk_offsets
        valid_topk = k_offsets < topk_count
        selected = tl.load(topk_ptr + q_idx * topk_count + k_offsets, mask=valid_topk, other=0)
        if HAS_POSITIONS:
            selected_abs = tl.load(key_pos_ptr + selected, mask=valid_topk, other=0)
            valid = valid_topk & (selected_abs <= q_abs)
        else:
            valid = valid_topk & (selected <= q_abs)

        key_ptrs = (
            key_ptr + (selected[:, None] * num_heads + head_idx) * head_dim + d_offsets[None, :]
        )
        key = tl.load(
            key_ptrs, mask=valid[:, None] & (d_offsets[None, :] < head_dim), other=0.0
        ).to(tl.float32)
        scores = tl.sum(key * query[None, :], axis=1) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))
        probs = tl.exp(scores - row_lse)
        probs = tl.where(valid, probs, 0.0)

        value_ptrs = (
            value_ptr + (selected[:, None] * num_heads + head_idx) * value_dim + d_offsets[None, :]
        )
        value = tl.load(
            value_ptrs, mask=valid[:, None] & (d_offsets[None, :] < value_dim), other=0.0
        ).to(tl.float32)

        dp = tl.sum(value * grad_output[None, :], axis=1)
        ds = probs * (dp - delta) * softmax_scale
        grad_query += tl.sum(ds[:, None] * key, axis=0)

        tl.atomic_add(
            grad_key_ptr
            + (selected[:, None] * num_heads + head_idx) * head_dim
            + d_offsets[None, :],
            ds[:, None] * query[None, :],
            sem="relaxed",
            mask=valid[:, None] & (d_offsets[None, :] < head_dim),
        )
        tl.atomic_add(
            grad_value_ptr
            + (selected[:, None] * num_heads + head_idx) * value_dim
            + d_offsets[None, :],
            probs[:, None] * grad_output[None, :],
            sem="relaxed",
            mask=valid[:, None] & (d_offsets[None, :] < value_dim),
        )

    grad_query_ptrs = grad_query_ptr + (q_idx * num_heads + head_idx) * head_dim + d_offsets
    tl.store(grad_query_ptrs, grad_query, mask=d_offsets < head_dim)


class SparseDSAAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        topk_indices: torch.Tensor,
        softmax_scale: float,
        q_start: int,
        query_positions: torch.Tensor | None,
        key_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        q_len, _, num_heads, head_dim = query.shape
        _, _, _, value_dim = value.shape
        topk_count = topk_indices.shape[-1]

        query_flat = query.squeeze(1).contiguous()
        key_flat = key.squeeze(1).contiguous()
        value_flat = value.squeeze(1).contiguous()
        topk_flat = topk_indices.squeeze(0).contiguous()
        has_positions = query_positions is not None
        if has_positions:
            query_positions = query_positions.contiguous()
            key_positions = key_positions.contiguous()
        else:
            query_positions = topk_flat
            key_positions = topk_flat

        output = torch.empty((q_len, num_heads, value_dim), device=query.device, dtype=query.dtype)
        lse = torch.empty((q_len, num_heads), device=query.device, dtype=torch.float32)
        block_k = _block_k(topk_count)
        block_d = triton.next_power_of_2(max(head_dim, value_dim))
        grid = (q_len, num_heads)

        _sparse_dsa_forward_kernel[grid](
            query_flat,
            key_flat,
            value_flat,
            topk_flat,
            query_positions,
            key_positions,
            output,
            lse,
            float(softmax_scale),
            num_heads,
            head_dim,
            value_dim,
            topk_count,
            int(q_start),
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            HAS_POSITIONS=has_positions,
            num_warps=4,
        )

        ctx.save_for_backward(
            query_flat, key_flat, value_flat, topk_flat, query_positions, key_positions, output, lse
        )
        ctx.softmax_scale = float(softmax_scale)
        ctx.q_start = int(q_start)
        ctx.has_positions = has_positions
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.value_dim = value_dim
        ctx.topk_count = topk_count
        ctx.block_k = block_k
        ctx.block_d = block_d

        return output.reshape(q_len, 1, num_heads * value_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value, topk_indices, query_positions, key_positions, output, lse = (
            ctx.saved_tensors
        )
        q_len = query.shape[0]
        sk = key.shape[0]
        num_heads = ctx.num_heads
        head_dim = ctx.head_dim
        value_dim = ctx.value_dim

        grad_output = grad_output.reshape(q_len, num_heads, value_dim).contiguous()
        grad_query = torch.empty(
            (q_len, num_heads, head_dim), device=query.device, dtype=torch.float32
        )
        grad_key = torch.zeros((sk, num_heads, head_dim), device=key.device, dtype=torch.float32)
        grad_value = torch.zeros(
            (sk, num_heads, value_dim), device=value.device, dtype=torch.float32
        )
        grid = (q_len, num_heads)

        _sparse_dsa_backward_kernel[grid](
            query,
            key,
            value,
            topk_indices,
            query_positions,
            key_positions,
            output,
            lse,
            grad_output,
            grad_query,
            grad_key,
            grad_value,
            ctx.softmax_scale,
            num_heads,
            head_dim,
            value_dim,
            ctx.topk_count,
            ctx.q_start,
            BLOCK_K=ctx.block_k,
            BLOCK_D=ctx.block_d,
            HAS_POSITIONS=ctx.has_positions,
            num_warps=4,
        )

        return (
            grad_query.to(query.dtype).unsqueeze(1),
            grad_key.to(key.dtype).unsqueeze(1),
            grad_value.to(value.dtype).unsqueeze(1),
            None,
            None,
            None,
            None,
            None,
        )


def sparse_dsa_attention_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    q_start: int = 0,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused sparse DSA attention over already-selected top-k key/value positions."""

    return SparseDSAAttentionTriton.apply(
        query,
        key,
        value,
        topk_indices,
        softmax_scale,
        q_start,
        query_positions,
        key_positions,
    )
