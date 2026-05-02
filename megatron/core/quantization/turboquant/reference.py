# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-PyTorch reference forward and backward for the 2.5-bit TurboQuant fake-quant.

The functions here are the gradcheck oracle. They use only differentiable or
explicitly-derived primitives and exactly mirror the math the CUDA kernel will
implement, so a CUDA-vs-reference numerical match is the soundness check for
the kernel.

The forward signature is ``y = turboquant_forward(x, buffers, ste)`` operating
on a 2-D tensor of shape ``[N, latent_dim]``. The backward
``turboquant_backward(grad_y, x, buffers, mask)`` returns ``grad_x`` of the
same shape. Both are written without ``torch.autograd`` so they can be
composed directly inside an ``autograd.Function`` and so that the math is
visible without any framework magic in the way.
"""

from __future__ import annotations

import math

import torch

from megatron.core.quantization.turboquant.codec import (
    TURBOQUANT_2P5_GROUP_SIZE,
    TURBOQUANT_2P5_HIGH_CHANNELS,
    TurboQuantBuffers,
)


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard transform with ``1/sqrt(n)`` normalization.

    Self-inverse and self-adjoint: ``fwht(fwht(x)) == x`` (modulo finite
    precision). The implementation is the standard butterfly written
    functionally so the input is never mutated.
    """

    n = x.shape[-1]
    *batch_shape, _ = x.shape
    y = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        grouped = y.view(-1, n // (2 * h), 2, h)
        a = grouped[:, :, 0, :]
        b = grouped[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    return y.view(*batch_shape, n) / math.sqrt(n)


def _quantize_2p5(
    rotated: torch.Tensor, buffers: TurboQuantBuffers
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-tier searchsorted quantizer for the 2.5-bit preset.

    Returns ``(quantized, ste_mask)`` where ``quantized`` is the centroid
    value selected per channel and ``ste_mask`` is a 0/1 float tensor of the
    same shape, marking 1 where the rotated value was inside the codebook
    range. The mask is what the backward pass uses to gate gradients
    (straight-through with saturating clip).
    """

    dim = rotated.shape[-1]
    groups = dim // TURBOQUANT_2P5_GROUP_SIZE
    high_lanes = TURBOQUANT_2P5_HIGH_CHANNELS
    low_lanes = TURBOQUANT_2P5_GROUP_SIZE - TURBOQUANT_2P5_HIGH_CHANNELS
    grouped = rotated.reshape(-1, groups, TURBOQUANT_2P5_GROUP_SIZE)

    high = grouped[..., :high_lanes].contiguous()
    low = grouped[..., high_lanes:].contiguous()
    high_idx = torch.searchsorted(buffers.boundaries_high, high)
    low_idx = torch.searchsorted(buffers.boundaries_low, low)
    high_q = buffers.centroids_high[high_idx]
    low_q = buffers.centroids_low[low_idx]

    high_lo = buffers.boundaries_high[0]
    high_hi = buffers.boundaries_high[-1]
    low_lo = buffers.boundaries_low[0]
    low_hi = buffers.boundaries_low[-1]
    high_mask = ((high >= high_lo) & (high <= high_hi)).to(rotated.dtype)
    low_mask = ((low >= low_lo) & (low <= low_hi)).to(rotated.dtype)

    quantized = torch.empty_like(grouped)
    quantized[..., :high_lanes] = high_q.to(rotated.dtype)
    quantized[..., high_lanes:] = low_q.to(rotated.dtype)

    mask = torch.empty_like(grouped)
    mask[..., :high_lanes] = high_mask
    mask[..., high_lanes:] = low_mask

    return quantized.reshape(rotated.shape), mask.reshape(rotated.shape)


def _quantize_uniform(
    rotated: torch.Tensor, buffers: TurboQuantBuffers
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-codebook quantizer for 3/4/8-bit presets."""

    if buffers.bits >= 8:
        return rotated, torch.ones_like(rotated)

    idx = torch.searchsorted(buffers.boundaries_high, rotated.contiguous())
    quantized = buffers.centroids_high[idx].to(rotated.dtype)
    lo = buffers.boundaries_high[0]
    hi = buffers.boundaries_high[-1]
    mask = ((rotated >= lo) & (rotated <= hi)).to(rotated.dtype)
    return quantized, mask


def turboquant_forward(
    x: torch.Tensor,
    buffers: TurboQuantBuffers,
    *,
    return_intermediates: bool = False,
):
    """Apply the TurboQuant fake-quant round-trip to ``x``.

    ``x`` has shape ``[N, latent_dim]`` and any floating dtype. The output has
    the same shape and dtype as ``x``. When ``return_intermediates`` is True a
    dict of internal tensors is returned alongside the output for use by the
    backward pass — this is the contract the autograd.Function uses to avoid
    recomputing things in backward.
    """

    if x.dim() != 2 or x.shape[-1] != buffers.latent_dim:
        raise ValueError(
            f"turboquant_forward expects x of shape [N, {buffers.latent_dim}]; "
            f"got {tuple(x.shape)}."
        )

    orig_dtype = x.dtype
    compute_dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    xf = x.to(compute_dtype)
    norm = torch.linalg.vector_norm(xf, dim=-1).clamp_min(1e-8)
    unit = xf / norm[:, None]
    rotated = fwht(unit * buffers.signs1) * buffers.signs2

    if buffers.is_2p5bit:
        quantized, ste_mask = _quantize_2p5(rotated, buffers)
    else:
        quantized, ste_mask = _quantize_uniform(rotated, buffers)

    w_hat = fwht(quantized * buffers.signs2)
    if buffers.norm_correction:
        inner_norm = torch.linalg.vector_norm(w_hat, dim=-1).clamp_min(1e-8)
        norm_hat = norm / inner_norm
    else:
        inner_norm = torch.ones_like(norm)
        norm_hat = norm

    z_hat = w_hat * buffers.signs1
    x_hat = (z_hat * norm_hat[:, None]).to(orig_dtype)

    if return_intermediates:
        return x_hat, {
            "norm": norm,
            "inner_norm": inner_norm,
            "norm_hat": norm_hat,
            "w_hat": w_hat,
            "z_hat": z_hat,
            "ste_mask": ste_mask,
            "x_fp32": xf,
        }
    return x_hat


def turboquant_backward(
    grad_x_hat: torch.Tensor,
    intermediates: dict,
    buffers: TurboQuantBuffers,
) -> torch.Tensor:
    """Closed-form gradient through the fake-quant.

    Given ``∂L/∂x_hat`` and the intermediates saved during the forward pass,
    return ``∂L/∂x``. Quantization is approximated with a saturating
    straight-through estimator: ``∂q/∂r = mask`` where mask is 1 inside the
    codebook range and 0 outside. Every other operator is orthogonal or
    diagonal and contributes its exact analytic Jacobian.
    """

    g = grad_x_hat.to(torch.float32)
    norm = intermediates["norm"]
    inner_norm = intermediates["inner_norm"]
    norm_hat = intermediates["norm_hat"]
    w_hat = intermediates["w_hat"]
    z_hat = intermediates["z_hat"]
    ste_mask = intermediates["ste_mask"]
    x_fp32 = intermediates["x_fp32"]

    grad_z_hat = g * norm_hat[:, None]
    grad_norm_hat = (g * z_hat).sum(dim=-1)

    grad_w_hat = grad_z_hat * buffers.signs1
    if buffers.norm_correction:
        grad_norm_external = grad_norm_hat / inner_norm
        grad_inner_norm = -grad_norm_hat * norm / (inner_norm * inner_norm)
        grad_w_hat = grad_w_hat + (grad_inner_norm / inner_norm)[:, None] * w_hat
    else:
        grad_norm_external = grad_norm_hat

    grad_quantized = fwht(grad_w_hat) * buffers.signs2
    grad_rotated = grad_quantized * ste_mask
    grad_unit = fwht(grad_rotated * buffers.signs2) * buffers.signs1

    inv_norm = 1.0 / norm
    inv_norm3 = inv_norm * inv_norm * inv_norm
    proj = (grad_unit * x_fp32).sum(dim=-1)
    grad_x_from_unit = grad_unit * inv_norm[:, None] - (proj * inv_norm3)[:, None] * x_fp32
    grad_x_from_norm = (grad_norm_external * inv_norm)[:, None] * x_fp32

    return (grad_x_from_unit + grad_x_from_norm).to(grad_x_hat.dtype)
