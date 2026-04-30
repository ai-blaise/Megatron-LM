# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import os
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F
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


_NEVER_USE_TORCH_MM = 1 << 60
_TORCH_MM_MIN_TOKENS_ENV = "MEGATRON_GATED_NORM_TORCH_MM_MIN_TOKENS"
_TORCH_MM_RANK_MIN_TOKENS_ENV = {
    1: "MEGATRON_GATED_NORM_TORCH_MM_R1_MIN_TOKENS",
    8: "MEGATRON_GATED_NORM_TORCH_MM_R8_MIN_TOKENS",
    32: "MEGATRON_GATED_NORM_TORCH_MM_R32_MIN_TOKENS",
    64: "MEGATRON_GATED_NORM_TORCH_MM_R64_MIN_TOKENS",
}


def _parse_min_tokens(raw: str | None, default: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "GatedNorm torch-MM token thresholds must be integers; "
            f"got {raw!r}"
        ) from exc
    if value < 0:
        return _NEVER_USE_TORCH_MM
    return value


def _default_torch_mm_min_tokens(rank: int) -> int:
    if rank >= 64:
        return 256
    if rank >= 32:
        return 512
    if rank >= 8:
        return 2048
    if rank >= 1:
        return 4096
    return _NEVER_USE_TORCH_MM


def _torch_mm_min_tokens(rank: int) -> int:
    global_override = os.getenv(_TORCH_MM_MIN_TOKENS_ENV)
    if global_override is not None:
        return _parse_min_tokens(global_override, _default_torch_mm_min_tokens(rank))

    default = _default_torch_mm_min_tokens(rank)
    for rank_floor in (64, 32, 8, 1):
        if rank >= rank_floor:
            return _parse_min_tokens(
                os.getenv(_TORCH_MM_RANK_MIN_TOKENS_ENV[rank_floor]),
                default,
            )
    return default


def _should_use_torch_mm(num_tokens: int, rank: int, dtype: torch.dtype) -> bool:
    if dtype != torch.bfloat16:
        return False
    return num_tokens >= _torch_mm_min_tokens(rank)


def _validate_gated_norm_inputs(
    normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> tuple[int, int]:
    if not normed.is_cuda or not w_down.is_cuda or not w_up.is_cuda:
        raise RuntimeError("apply_gated_norm requires CUDA tensors")

    if normed.dim() < 2:
        raise ValueError(f"normed must have at least 2 dimensions, got shape {tuple(normed.shape)}")

    allowed_dtypes = (torch.bfloat16, torch.float16, torch.float32)
    if normed.dtype not in allowed_dtypes:
        raise TypeError(f"normed must have dtype bf16, fp16, or fp32; got {normed.dtype}")
    if w_down.dtype != normed.dtype or w_up.dtype != normed.dtype:
        raise TypeError(
            "w_down and w_up must have the same dtype as normed: "
            f"got normed={normed.dtype}, w_down={w_down.dtype}, w_up={w_up.dtype}"
        )

    hidden_size = normed.shape[-1]
    if w_down.dim() != 2 or w_up.dim() != 2:
        raise ValueError("w_down and w_up must be rank-2 tensors")
    if w_down.shape[1] != hidden_size:
        raise ValueError(
            f"w_down must have shape [rank, hidden_size]; got {tuple(w_down.shape)} "
            f"for hidden_size={hidden_size}"
        )
    if w_up.shape[0] != hidden_size:
        raise ValueError(
            f"w_up must have shape [hidden_size, rank]; got {tuple(w_up.shape)} "
            f"for hidden_size={hidden_size}"
        )
    if w_down.shape[0] != w_up.shape[1]:
        raise ValueError(
            "w_down and w_up must agree on the gating rank: "
            f"got {w_down.shape[0]} and {w_up.shape[1]}"
        )

    return hidden_size, w_down.shape[0]


def _require_triton() -> None:
    if not HAVE_TRITON:
        raise RuntimeError("apply_gated_norm requires Triton")


def _next_power_of_2(value: int, maximum: int) -> int:
    return min(triton.next_power_of_2(value), maximum)


@triton.jit
def _gated_norm_forward_kernel(
    y_ptr,
    w_down_ptr,
    w_up_ptr,
    z_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    rank: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    token_idx = tl.program_id(0)
    h_offsets = tl.arange(0, BLOCK_H)
    r_offsets = tl.arange(0, BLOCK_R)
    r_mask = r_offsets < rank

    z = tl.zeros((BLOCK_R,), tl.float32)
    for h_start in tl.range(0, hidden_size, BLOCK_H):
        h = h_start + h_offsets
        h_mask = h < hidden_size
        y = tl.load(y_ptr + token_idx * hidden_size + h, mask=h_mask, other=0.0)
        w_down = tl.load(
            w_down_ptr + r_offsets[:, None] * hidden_size + h[None, :],
            mask=r_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        z += tl.sum(w_down.to(tl.float32) * y[None, :].to(tl.float32), axis=1)

    tl.store(z_ptr + token_idx * rank + r_offsets, z, mask=r_mask)

    sigmoid_z = 1.0 / (1.0 + tl.exp(-z))
    a = tl.where(r_mask, z * sigmoid_z, 0.0)
    for h_start in tl.range(0, hidden_size, BLOCK_H):
        h = h_start + h_offsets
        h_mask = h < hidden_size
        w_up = tl.load(
            w_up_ptr + h[:, None] * rank + r_offsets[None, :],
            mask=h_mask[:, None] & r_mask[None, :],
            other=0.0,
        )
        logits = tl.sum(w_up.to(tl.float32) * a[None, :], axis=1)
        gate = 1.0 / (1.0 + tl.exp(-logits))
        y = tl.load(y_ptr + token_idx * hidden_size + h, mask=h_mask, other=0.0)
        tl.store(output_ptr + token_idx * hidden_size + h, y * gate, mask=h_mask)


@triton.jit
def _gated_norm_backward_kernel(
    y_ptr,
    w_down_ptr,
    w_up_ptr,
    z_ptr,
    grad_output_ptr,
    grad_y_ptr,
    grad_w_down_ptr,
    grad_w_up_ptr,
    hidden_size: tl.constexpr,
    rank: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    token_idx = tl.program_id(0)
    h_offsets = tl.arange(0, BLOCK_H)
    r_offsets = tl.arange(0, BLOCK_R)
    r_mask = r_offsets < rank

    z = tl.load(z_ptr + token_idx * rank + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
    sigmoid_z = 1.0 / (1.0 + tl.exp(-z))
    a = tl.where(r_mask, z * sigmoid_z, 0.0)

    da = tl.zeros((BLOCK_R,), tl.float32)

    # First pass: one dlogits stream feeds both da and dW_up.
    for h_start in tl.range(0, hidden_size, BLOCK_H):
        h = h_start + h_offsets
        h_mask = h < hidden_size
        w_up = tl.load(
            w_up_ptr + h[:, None] * rank + r_offsets[None, :],
            mask=h_mask[:, None] & r_mask[None, :],
            other=0.0,
        )
        logits = tl.sum(w_up.to(tl.float32) * a[None, :], axis=1)
        gate = 1.0 / (1.0 + tl.exp(-logits))
        y = tl.load(y_ptr + token_idx * hidden_size + h, mask=h_mask, other=0.0).to(tl.float32)
        grad_output = tl.load(
            grad_output_ptr + token_idx * hidden_size + h, mask=h_mask, other=0.0
        ).to(tl.float32)
        dlogits = grad_output * y * gate * (1.0 - gate)
        da += tl.sum(dlogits[:, None] * w_up.to(tl.float32), axis=0)
        tl.atomic_add(
            grad_w_up_ptr + h[:, None] * rank + r_offsets[None, :],
            dlogits[:, None] * a[None, :],
            sem="relaxed",
            mask=h_mask[:, None] & r_mask[None, :],
        )

    silu_grad = sigmoid_z * (1.0 + z * (1.0 - sigmoid_z))
    dz = tl.where(r_mask, da * silu_grad, 0.0)

    # Second pass: dz is now available, so compute dy and dW_down in the same launch.
    for h_start in tl.range(0, hidden_size, BLOCK_H):
        h = h_start + h_offsets
        h_mask = h < hidden_size
        w_up = tl.load(
            w_up_ptr + h[:, None] * rank + r_offsets[None, :],
            mask=h_mask[:, None] & r_mask[None, :],
            other=0.0,
        )
        logits = tl.sum(w_up.to(tl.float32) * a[None, :], axis=1)
        gate = 1.0 / (1.0 + tl.exp(-logits))
        y = tl.load(y_ptr + token_idx * hidden_size + h, mask=h_mask, other=0.0).to(tl.float32)
        grad_output = tl.load(
            grad_output_ptr + token_idx * hidden_size + h, mask=h_mask, other=0.0
        ).to(tl.float32)
        w_down = tl.load(
            w_down_ptr + r_offsets[:, None] * hidden_size + h[None, :],
            mask=r_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        dz_w_down = tl.sum(dz[:, None] * w_down.to(tl.float32), axis=0)
        grad_y = grad_output * gate + dz_w_down
        tl.store(grad_y_ptr + token_idx * hidden_size + h, grad_y, mask=h_mask)
        tl.atomic_add(
            grad_w_down_ptr + r_offsets[:, None] * hidden_size + h[None, :],
            dz[:, None] * y[None, :],
            sem="relaxed",
            mask=r_mask[:, None] & h_mask[None, :],
        )


def _gated_norm_torch_mm_forward(
    flat_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    output: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    z = torch.mm(flat_normed, w_down.t()).float()
    activation = F.silu(z).to(w_up.dtype)
    logits = torch.mm(activation, w_up.t())
    torch.sigmoid(logits, out=logits)
    torch.mul(flat_normed, logits, out=output.reshape(-1, hidden_size))
    return z


def _gated_norm_forward(
    normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor, hidden_size: int, rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton()

    input_shape = normed.shape
    flat_normed = normed.reshape(-1, hidden_size).contiguous()
    w_down = w_down.contiguous()
    w_up = w_up.contiguous()

    num_tokens = flat_normed.shape[0]
    z = torch.empty((num_tokens, rank), device=normed.device, dtype=torch.float32)
    output = torch.empty_like(flat_normed)

    if num_tokens == 0:
        return output.reshape(input_shape), flat_normed, w_down, w_up, z

    if _should_use_torch_mm(num_tokens, rank, normed.dtype):
        z = _gated_norm_torch_mm_forward(flat_normed, w_down, w_up, output, hidden_size)
        return output.reshape(input_shape), flat_normed, w_down, w_up, z

    block_h = _next_power_of_2(hidden_size, 128)
    block_r = _next_power_of_2(rank, 64)
    _gated_norm_forward_kernel[(num_tokens,)](
        flat_normed,
        w_down,
        w_up,
        z,
        output,
        hidden_size,
        rank,
        BLOCK_H=block_h,
        BLOCK_R=block_r,
        num_warps=4,
    )

    return output.reshape(input_shape), flat_normed, w_down, w_up, z


def _gated_norm_backward(
    grad_output: torch.Tensor,
    flat_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    z: torch.Tensor,
    input_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton()

    hidden_size = flat_normed.shape[-1]
    num_tokens = flat_normed.shape[0]
    rank = w_down.shape[0]
    flat_grad_output = grad_output.reshape(-1, hidden_size).contiguous()

    grad_normed = torch.empty_like(flat_normed)
    grad_w_down = torch.zeros_like(w_down)
    grad_w_up = torch.zeros_like(w_up)

    block_h = _next_power_of_2(hidden_size, 128)
    block_r = _next_power_of_2(rank, 64)
    _gated_norm_backward_kernel[(num_tokens,)](
        flat_normed,
        w_down,
        w_up,
        z,
        flat_grad_output,
        grad_normed,
        grad_w_down,
        grad_w_up,
        hidden_size,
        rank,
        BLOCK_H=block_h,
        BLOCK_R=block_r,
        num_warps=4,
    )

    return grad_normed.reshape(input_shape), grad_w_down, grad_w_up


class GatedNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor):
        hidden_size, rank = _validate_gated_norm_inputs(normed, w_down, w_up)

        output, flat_normed, w_down, w_up, z = _gated_norm_forward(
            normed, w_down, w_up, hidden_size, rank
        )

        ctx.input_shape = normed.shape
        ctx.save_for_backward(flat_normed, w_down, w_up, z)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        flat_normed, w_down, w_up, z = ctx.saved_tensors
        return _gated_norm_backward(grad_output, flat_normed, w_down, w_up, z, ctx.input_shape)


def apply_gated_norm(
    normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> torch.Tensor:
    """Apply learned GatedNorm after an existing normed tensor."""

    return GatedNormFunction.apply(normed, w_down, w_up)
