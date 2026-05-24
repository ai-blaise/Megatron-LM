# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compact-index local permutation helpers for DeepEP MoE dispatch.

DeepEP returns compact per-token local expert indices/probabilities after the
cross-rank dispatch. The generic Megatron postprocess path expands those compact
indices back into dense multihot tensors before immediately sorting them into
per-expert order. These helpers keep the compact representation and materialize
only the tensors needed by TEGroupedMLP and the inverse combine.
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
    triton = None
    tl = None


if not HAVE_TRITON:
    triton_jit = null_decorator
else:
    triton_jit = triton.jit


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


def _row_chunk_size() -> int:
    value = int(os.getenv("MEGATRON_DEEPEP_COMPACT_ROW_CHUNK", "131072"))
    return max(1, value)


def _stable_compact_map_enabled() -> bool:
    value = os.getenv("MEGATRON_DEEPEP_COMPACT_STABLE_MAP", "0")
    return value.strip().lower() not in ("0", "false", "off", "no")


def _compact_backend() -> str:
    return os.getenv("MEGATRON_DEEPEP_COMPACT_BACKEND", "triton").strip().lower()


def _cuda_backend_requested() -> bool:
    return _compact_backend() in {"cuda", "cpp", "megakernel", "rowcuda", "cuda_rows"}


def _cuda_edge_backend_requested() -> bool:
    return _compact_backend() in {"edgecuda", "cuda_edge"}


def _cuda_row_backend_requested() -> bool:
    return _compact_backend() in {"cuda", "cpp", "megakernel", "rowcuda", "cuda_rows"}


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"0", "false", "off", "no"}


def _cuda_backward_requested() -> bool:
    return _env_flag("MEGATRON_DEEPEP_COMPACT_CUDA_BACKWARD", "0")


def _cuda_unpermute_requested() -> bool:
    return _env_flag("MEGATRON_DEEPEP_COMPACT_CUDA_UNPERMUTE", "0")


def _get_cuda_ext():
    from megatron.core.extensions.hisa_indexer.kernels.build import get_ext

    return get_ext()


def _cpu_counts_offsets(tokens_per_expert: torch.Tensor, align_size: int):
    counts_cpu = tokens_per_expert.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if align_size and align_size > 1:
        padded_counts_cpu = torch.div(
            counts_cpu + align_size - 1, align_size, rounding_mode="floor"
        )
        padded_counts_cpu = padded_counts_cpu * align_size
    else:
        padded_counts_cpu = counts_cpu
    offsets_cpu = torch.empty_like(padded_counts_cpu)
    if padded_counts_cpu.numel():
        cumulative_cpu = torch.cumsum(padded_counts_cpu, dim=0)
        offsets_cpu[0] = 0
        offsets_cpu[1:] = cumulative_cpu[:-1]
        num_out_tokens = int(cumulative_cpu[-1])
    else:
        num_out_tokens = 0
    return counts_cpu, padded_counts_cpu.contiguous(), offsets_cpu.contiguous(), num_out_tokens


@triton_jit
def _build_deepep_permute_maps_kernel(
    indices,
    probs,
    offsets,
    actual_counts,
    counters,
    row_map,
    edge_map,
    permuted_probs,
    num_edges: tl.constexpr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
):
    edge = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = edge < num_edges

    expert = tl.load(indices + edge, mask=mask, other=-1)
    valid = mask & (expert >= 0) & (expert < num_experts)
    safe_expert = tl.minimum(tl.maximum(expert, 0), num_experts - 1)
    pos = tl.atomic_add(counters + safe_expert, 1, sem="relaxed", mask=valid)
    actual = tl.load(actual_counts + safe_expert, mask=valid, other=0)
    valid = valid & (pos < actual)

    out = tl.load(offsets + safe_expert, mask=valid, other=0) + pos
    row = edge // topk
    prob = tl.load(probs + edge, mask=valid, other=0.0)

    tl.store(row_map + out, row, mask=valid)
    tl.store(edge_map + out, edge, mask=valid)
    tl.store(permuted_probs + out, prob, mask=valid)


@triton_jit
def _gather_rows_kernel(
    hidden,
    row_map,
    output,
    num_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    src = tl.load(row_map + rows, mask=rows < num_rows, other=-1)
    valid = (rows < num_rows) & (src >= 0)
    safe_src = tl.maximum(src, 0)

    vals = tl.load(
        hidden + safe_src[:, None] * hidden_size + cols[None, :],
        mask=valid[:, None] & (cols[None, :] < hidden_size),
        other=0.0,
    )
    tl.store(
        output + rows[:, None] * hidden_size + cols[None, :],
        vals,
        mask=(rows[:, None] < num_rows) & (cols[None, :] < hidden_size),
    )


@triton_jit
def _scatter_rows_kernel(
    grad_output,
    row_map,
    grad_hidden,
    num_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    dst = tl.load(row_map + rows, mask=rows < num_rows, other=-1)
    valid = (rows < num_rows) & (dst >= 0)
    safe_dst = tl.maximum(dst, 0)

    vals = tl.load(
        grad_output + rows[:, None] * hidden_size + cols[None, :],
        mask=valid[:, None] & (cols[None, :] < hidden_size),
        other=0.0,
    )
    tl.atomic_add(
        grad_hidden + safe_dst[:, None] * hidden_size + cols[None, :],
        vals,
        sem="relaxed",
        mask=valid[:, None] & (cols[None, :] < hidden_size),
    )


@triton_jit
def _scatter_probs_kernel(
    grad_permuted_probs,
    edge_map,
    grad_probs,
    num_rows: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    edge = tl.load(edge_map + rows, mask=rows < num_rows, other=-1)
    valid = (rows < num_rows) & (edge >= 0)
    safe_edge = tl.maximum(edge, 0)
    vals = tl.load(grad_permuted_probs + rows, mask=valid, other=0.0)
    tl.store(grad_probs + safe_edge, vals, mask=valid)


class _DeepEPIndicesPermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, indices, probs, tokens_per_expert, align_size: int):
        if _cuda_backend_requested() or _cuda_edge_backend_requested():
            if not hidden.is_cuda:
                raise RuntimeError("CUDA DeepEP compact permute requires CUDA tensors")
            if align_size and align_size > 1:
                raise RuntimeError(
                    "CUDA DeepEP compact permute currently requires align_size=1; "
                    "disable MEGATRON_DEEPEP_COMPACT_BACKEND=cuda for padded routing"
                )
            if hidden.dim() != 2 or indices.dim() != 2 or probs.shape != indices.shape:
                raise ValueError("Expected hidden [N,H] and matching indices/probs [N,K]")
            if not indices.is_contiguous():
                indices = indices.contiguous()
            if not probs.is_contiguous():
                probs = probs.contiguous()
            if probs.dtype != torch.float32:
                raise RuntimeError("CUDA DeepEP compact permute expects fp32 probabilities")

            _, padded_counts_cpu, offsets_cpu, num_out_tokens = _cpu_counts_offsets(
                tokens_per_expert, align_size
            )
            hidden_size = hidden.shape[1]
            output = torch.empty(
                (num_out_tokens, hidden_size), device=hidden.device, dtype=hidden.dtype
            )
            permuted_probs = torch.empty((num_out_tokens,), device=hidden.device, dtype=probs.dtype)
            row_map = torch.empty((num_out_tokens,), device=hidden.device, dtype=torch.int64)
            edge_map = torch.empty((num_out_tokens,), device=hidden.device, dtype=torch.int64)
            counts = padded_counts_cpu.to(device=hidden.device, non_blocking=True)
            offsets = offsets_cpu.to(device=hidden.device, non_blocking=True)
            counters = torch.zeros((counts.numel(),), device=hidden.device, dtype=torch.int32)

            if _cuda_row_backend_requested():
                edge_to_row = torch.empty(
                    (indices.numel(),), device=hidden.device, dtype=torch.int32
                )
                _get_cuda_ext().moe_deepep_compact_permute_rows_fwd(
                    hidden,
                    indices,
                    probs,
                    offsets,
                    counts,
                    output,
                    permuted_probs,
                    row_map,
                    edge_map,
                    edge_to_row,
                    counters,
                )
            else:
                edge_to_row = torch.empty((0,), device=hidden.device, dtype=torch.int32)
                _get_cuda_ext().moe_deepep_compact_permute_fwd(
                    hidden,
                    indices,
                    probs,
                    offsets,
                    counts,
                    output,
                    permuted_probs,
                    row_map,
                    edge_map,
                    counters,
                )
            ctx.save_for_backward(row_map, edge_map)
            ctx.hidden_shape = hidden.shape
            ctx.probs_shape = probs.shape
            ctx.hidden_size = hidden_size
            ctx.cuda_compact = _cuda_backward_requested()
            return output, permuted_probs, row_map, edge_to_row, counts

        if not HAVE_TRITON or not hidden.is_cuda:
            raise RuntimeError("DeepEP compact-index permute requires Triton CUDA support")
        if hidden.dim() != 2 or indices.dim() != 2 or probs.shape != indices.shape:
            raise ValueError("Expected hidden [N,H] and matching indices/probs [N,K]")
        if not indices.is_contiguous():
            indices = indices.contiguous()
        if not probs.is_contiguous():
            probs = probs.contiguous()

        num_tokens, hidden_size = hidden.shape
        topk = indices.shape[1]
        counts = tokens_per_expert.to(device=hidden.device, dtype=torch.int64)
        if align_size and align_size > 1:
            padded_counts = torch.div(counts + align_size - 1, align_size, rounding_mode="floor")
            padded_counts = padded_counts * align_size
        else:
            padded_counts = counts
        offsets = torch.empty_like(padded_counts)
        cumulative = torch.cumsum(padded_counts, dim=0)
        offsets[0] = 0
        offsets[1:] = cumulative[:-1]
        num_out_tokens = int(cumulative[-1].item()) if cumulative.numel() else 0

        output = torch.empty((num_out_tokens, hidden_size), device=hidden.device, dtype=hidden.dtype)
        permuted_probs = torch.zeros((num_out_tokens,), device=hidden.device, dtype=probs.dtype)
        row_map = torch.full((num_out_tokens,), -1, device=hidden.device, dtype=torch.int64)
        edge_map = torch.full((num_out_tokens,), -1, device=hidden.device, dtype=torch.int64)
        counters = torch.zeros_like(counts, dtype=torch.int32)

        num_edges = num_tokens * topk
        if _stable_compact_map_enabled():
            flat_indices = indices.reshape(-1)
            flat_probs = probs.reshape(-1)
            edge_ids = torch.arange(num_edges, device=hidden.device, dtype=torch.int64)
            valid = (flat_indices >= 0) & (flat_indices < counts.numel())
            valid_edges = edge_ids[valid]
            if valid_edges.numel() > 0:
                valid_experts = flat_indices[valid].to(torch.int64)
                rows = torch.div(valid_edges, topk, rounding_mode="floor")
                # Match routing_map.T masked-select order: expert-major, token-row-minor.
                order = torch.argsort(valid_experts * num_tokens + rows, stable=True)
                valid_edges = valid_edges[order]
                valid_experts = valid_experts[order]
                rows = rows[order]

                actual_edge_counts = torch.bincount(valid_experts, minlength=counts.numel())
                expert_starts = torch.cumsum(actual_edge_counts, dim=0) - actual_edge_counts
                ranks = torch.arange(valid_edges.numel(), device=hidden.device, dtype=torch.int64)
                ranks = ranks - expert_starts[valid_experts]
                keep = ranks < counts[valid_experts]
                if keep.any():
                    valid_edges = valid_edges[keep]
                    valid_experts = valid_experts[keep]
                    rows = rows[keep]
                    ranks = ranks[keep]
                    out = offsets[valid_experts] + ranks
                    row_map[out] = rows
                    edge_map[out] = valid_edges
                    permuted_probs[out] = flat_probs[valid_edges]
        else:
            block = 256
            _build_deepep_permute_maps_kernel[(triton.cdiv(num_edges, block),)](
                indices,
                probs,
                offsets,
                counts,
                counters,
                row_map,
                edge_map,
                permuted_probs,
                num_edges,
                counts.numel(),
                topk,
                BLOCK=block,
                num_warps=4,
            )

        block_m = 4
        block_h = min(256, _next_power_of_2(hidden_size))
        row_chunk = _row_chunk_size()
        for row_start in range(0, num_out_tokens, row_chunk):
            row_end = min(row_start + row_chunk, num_out_tokens)
            rows_this_chunk = row_end - row_start
            _gather_rows_kernel[
                (triton.cdiv(rows_this_chunk, block_m), triton.cdiv(hidden_size, block_h))
            ](
                hidden,
                row_map[row_start:row_end],
                output[row_start:row_end],
                rows_this_chunk,
                hidden_size,
                BLOCK_M=block_m,
                BLOCK_H=block_h,
                num_warps=8,
            )

        ctx.save_for_backward(row_map, edge_map)
        ctx.hidden_shape = hidden.shape
        ctx.probs_shape = probs.shape
        ctx.hidden_size = hidden_size
        ctx.cuda_compact = False
        edge_to_row = torch.empty((0,), device=hidden.device, dtype=torch.int32)
        return output, permuted_probs, row_map, edge_to_row, padded_counts

    @staticmethod
    def backward(
        ctx, grad_output, grad_permuted_probs, grad_row_map, grad_edge_to_row, grad_padded_counts
    ):
        row_map, edge_map = ctx.saved_tensors
        num_tokens, hidden_size = ctx.hidden_shape
        num_rows = row_map.numel()

        if getattr(ctx, "cuda_compact", False):
            ext = _get_cuda_ext()
            grad_output = grad_output.contiguous()
            grad_hidden = torch.zeros(
                (num_tokens, hidden_size), device=grad_output.device, dtype=grad_output.dtype
            )
            ext.moe_deepep_compact_scatter_add(grad_output, row_map, grad_hidden)
            grad_probs = None
            if grad_permuted_probs is not None:
                grad_probs = torch.zeros(
                    ctx.probs_shape,
                    device=grad_permuted_probs.device,
                    dtype=grad_permuted_probs.dtype,
                )
                ext.moe_deepep_compact_scatter_probs(
                    grad_permuted_probs.contiguous(), edge_map, grad_probs
                )
            return grad_hidden, None, grad_probs, None, None

        grad_hidden = torch.zeros(
            (num_tokens, hidden_size), device=grad_output.device, dtype=grad_output.dtype
        )
        block_m = 4
        block_h = min(256, _next_power_of_2(hidden_size))
        row_chunk = _row_chunk_size()
        grad_output = grad_output.contiguous()
        for row_start in range(0, num_rows, row_chunk):
            row_end = min(row_start + row_chunk, num_rows)
            rows_this_chunk = row_end - row_start
            _scatter_rows_kernel[
                (triton.cdiv(rows_this_chunk, block_m), triton.cdiv(hidden_size, block_h))
            ](
                grad_output[row_start:row_end],
                row_map[row_start:row_end],
                grad_hidden,
                rows_this_chunk,
                hidden_size,
                BLOCK_M=block_m,
                BLOCK_H=block_h,
                num_warps=8,
            )

        grad_probs = None
        if grad_permuted_probs is not None:
            grad_probs = torch.zeros(
                ctx.probs_shape,
                device=grad_permuted_probs.device,
                dtype=grad_permuted_probs.dtype,
            )
            block = 256
            row_chunk = _row_chunk_size()
            grad_permuted_probs = grad_permuted_probs.contiguous()
            for row_start in range(0, num_rows, row_chunk):
                row_end = min(row_start + row_chunk, num_rows)
                rows_this_chunk = row_end - row_start
                _scatter_probs_kernel[(triton.cdiv(rows_this_chunk, block),)](
                    grad_permuted_probs[row_start:row_end],
                    edge_map[row_start:row_end],
                    grad_probs,
                    rows_this_chunk,
                    BLOCK=block,
                    num_warps=4,
                )
        return grad_hidden, None, grad_probs, None, None


class _DeepEPIndicesUnpermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permuted_hidden, row_map, restore_shape, indices, edge_to_row, num_experts: int):
        if (
            _cuda_unpermute_requested()
            and _cuda_row_backend_requested()
            and indices is not None
            and edge_to_row is not None
            and edge_to_row.numel() > 0
        ):
            if not permuted_hidden.is_cuda:
                raise RuntimeError("CUDA DeepEP compact unpermute requires CUDA tensors")
            if not indices.is_contiguous():
                indices = indices.contiguous()
            restore_tokens = int(restore_shape[0])
            hidden_size = int(restore_shape[1])
            output = torch.empty(
                (restore_tokens, hidden_size),
                device=permuted_hidden.device,
                dtype=permuted_hidden.dtype,
            )
            _get_cuda_ext().moe_deepep_compact_unpermute_rows(
                permuted_hidden.contiguous(),
                indices,
                edge_to_row,
                output,
                int(num_experts),
            )
            ctx.save_for_backward(row_map)
            ctx.hidden_size = hidden_size
            ctx.cuda_compact = True
            return output

        if (_cuda_backend_requested() or _cuda_edge_backend_requested()) and _cuda_unpermute_requested():
            if not permuted_hidden.is_cuda:
                raise RuntimeError("CUDA DeepEP compact unpermute requires CUDA tensors")
            restore_tokens = int(restore_shape[0])
            hidden_size = int(restore_shape[1])
            output = torch.zeros(
                (restore_tokens, hidden_size),
                device=permuted_hidden.device,
                dtype=permuted_hidden.dtype,
            )
            _get_cuda_ext().moe_deepep_compact_scatter_add(
                permuted_hidden.contiguous(), row_map, output
            )
            ctx.save_for_backward(row_map)
            ctx.hidden_size = hidden_size
            ctx.cuda_compact = True
            return output

        if not HAVE_TRITON or not permuted_hidden.is_cuda:
            raise RuntimeError("DeepEP compact-index unpermute requires Triton CUDA support")
        restore_tokens = int(restore_shape[0])
        hidden_size = int(restore_shape[1])
        num_rows = row_map.numel()
        output = torch.zeros(
            (restore_tokens, hidden_size),
            device=permuted_hidden.device,
            dtype=permuted_hidden.dtype,
        )
        block_m = 4
        block_h = min(256, _next_power_of_2(hidden_size))
        row_chunk = _row_chunk_size()
        permuted_hidden = permuted_hidden.contiguous()
        for row_start in range(0, num_rows, row_chunk):
            row_end = min(row_start + row_chunk, num_rows)
            rows_this_chunk = row_end - row_start
            _scatter_rows_kernel[
                (triton.cdiv(rows_this_chunk, block_m), triton.cdiv(hidden_size, block_h))
            ](
                permuted_hidden[row_start:row_end],
                row_map[row_start:row_end],
                output,
                rows_this_chunk,
                hidden_size,
                BLOCK_M=block_m,
                BLOCK_H=block_h,
                num_warps=8,
            )
        ctx.save_for_backward(row_map)
        ctx.hidden_size = hidden_size
        ctx.cuda_compact = False
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (row_map,) = ctx.saved_tensors
        num_rows = row_map.numel()
        hidden_size = ctx.hidden_size
        grad_permuted = torch.empty(
            (num_rows, hidden_size), device=grad_output.device, dtype=grad_output.dtype
        )
        if getattr(ctx, "cuda_compact", False):
            _get_cuda_ext().moe_deepep_compact_gather(
                grad_output.contiguous(), row_map, grad_permuted
            )
            return grad_permuted, None, None, None, None, None

        block_m = 4
        block_h = min(256, _next_power_of_2(hidden_size))
        row_chunk = _row_chunk_size()
        grad_output = grad_output.contiguous()
        for row_start in range(0, num_rows, row_chunk):
            row_end = min(row_start + row_chunk, num_rows)
            rows_this_chunk = row_end - row_start
            _gather_rows_kernel[
                (triton.cdiv(rows_this_chunk, block_m), triton.cdiv(hidden_size, block_h))
            ](
                grad_output,
                row_map[row_start:row_end],
                grad_permuted[row_start:row_end],
                rows_this_chunk,
                hidden_size,
                BLOCK_M=block_m,
                BLOCK_H=block_h,
                num_warps=8,
            )
        return grad_permuted, None, None, None, None, None


def deepep_indices_permute(
    hidden: torch.Tensor,
    indices: torch.Tensor,
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    align_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute DeepEP compact local expert indices directly into per-expert order."""

    return _DeepEPIndicesPermute.apply(hidden, indices, probs, tokens_per_expert, align_size)


def deepep_indices_unpermute(
    permuted_hidden: torch.Tensor,
    row_map: torch.Tensor,
    restore_shape: torch.Size,
    indices: torch.Tensor | None = None,
    edge_to_row: torch.Tensor | None = None,
    num_experts: int = 0,
) -> torch.Tensor:
    """Restore compact-index permuted expert outputs to DeepEP combine order."""

    return _DeepEPIndicesUnpermute.apply(
        permuted_hidden, row_map, restore_shape, indices, edge_to_row, num_experts
    )
