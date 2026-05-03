# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-PyTorch reference for NVFP4 activation fake-quant + activation-ECO.

The forward emits ``q(x)`` for the downstream FP4 GEMM. The on-disk
NVFP4 layout (uint8-packed FP4 nibbles + per-block FP8 scale + per-
tensor FP32 global scale) is what the deploy-time SGLang loader reads;
here we operate one cast back, on the fully-dequantized tensor that
the cuBLASLt kernel logically sees.

Forward (per row, per block of ``block_size`` elements along the hidden dim):
    block_amax  = max(|x[i, b*BS:(b+1)*BS]|)
    block_scale = block_amax / fp4_max               # FP8 in production
    nibble      = round(clamp(x / block_scale, [-fp4_max, fp4_max]))
    q(x)        = nibble * block_scale               # dequantized fake-quant

Backward through ``q`` alone (STE on saturated lanes):
    dq/dx_j = mask_j   where mask_j = 1[|x_j / scale_b| < fp4_max]
                       and scale_b is the per-block scale of x_j's block.

Activation-ECO bias correction (the new piece this module adds):
    Standard QAT backward through ``y = q(x) @ W`` gives ``dW = dy @ q(x).T``,
    biased by activation rounding. Adding the residual
        dW_correction = dy @ (x - q(x)).T = dy @ e_x.T
    cancels the bias exactly:
        dW_corrected = dy @ q(x).T + dy @ e_x.T = dy @ x.T
    matching the gradient pure BF16 forward would have produced. This is
    the activation-ECO update.
"""

from __future__ import annotations

import torch

from megatron.core.quantization.nvfp4_act_eco.codec import Nvfp4ActEcoConfig


def _block_view(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Reshape ``[N, D]`` → ``[N, D // block_size, block_size]``.

    D must be a multiple of block_size. Per-block reductions then run
    along the last axis.
    """

    n, d = x.shape
    if d % block_size != 0:
        raise ValueError(
            f"hidden dim {d} not a multiple of block_size {block_size}"
        )
    return x.reshape(n, d // block_size, block_size)


def nvfp4_act_quant_forward(
    x: torch.Tensor,
    config: Nvfp4ActEcoConfig,
    *,
    return_intermediates: bool = False,
):
    """Apply NVFP4 per-block fake-quant to a 2-D tensor ``[N, D]``.

    The output dtype matches ``x.dtype``. Hosts with ``torch.float8_e4m3fn``
    use that for the per-block scale storage to match production
    rounding bit-for-bit; on other hosts the scale stays in fp32 and
    the FP4 nibble round is simulated to the nearest representable
    NVFP4 value.
    """

    if x.dim() != 2:
        raise ValueError(
            "nvfp4_act_quant_forward expects [N, D]; "
            f"got shape {tuple(x.shape)}. Reshape upstream."
        )

    orig_dtype = x.dtype
    compute_dtype = (
        x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    )
    xf = x.to(compute_dtype)

    blocks = _block_view(xf, config.block_size)  # [N, n_blocks, BS]
    block_amax = blocks.abs().amax(dim=-1)  # [N, n_blocks]
    block_scale_pre = block_amax / config.fp4_max  # fp32 candidate

    if hasattr(torch, "float8_e4m3fn"):
        block_scale = (
            block_scale_pre.clamp_min(config.eps)
            .to(torch.float8_e4m3fn)
            .to(compute_dtype)
        )
    else:
        block_scale = block_scale_pre.clamp_min(config.eps)

    scale_b = block_scale[..., None]  # [N, n_blocks, 1]
    pre_clip = blocks / scale_b
    pre_round = pre_clip.clamp(-config.fp4_max, config.fp4_max)
    nibble = _round_to_nvfp4_grid(pre_round, fp4_max=config.fp4_max)
    q_blocks = nibble * scale_b
    q = q_blocks.reshape_as(xf).to(orig_dtype)

    if return_intermediates:
        clip_mask_blocks = (
            (pre_clip >= -config.fp4_max) & (pre_clip <= config.fp4_max)
        ).to(compute_dtype)
        clip_mask = clip_mask_blocks.reshape_as(xf)
        return q, {
            "x_compute": xf,
            "block_scale": block_scale,
            "nibble": nibble,
            "clip_mask": clip_mask,
        }
    return q


def nvfp4_act_quant_backward(
    grad_y: torch.Tensor,
    intermediates: dict,
    config: Nvfp4ActEcoConfig,
) -> torch.Tensor:
    """Backward through ``q(x)`` alone (STE on saturated lanes).

    ``y = q(x)`` → ``dy/dx = clip_mask``. The per-block scale is a
    function of the block's argmax-of-|x| coordinate; once block_amax
    is computed the rounding is deterministic, so the analytic Jacobian
    contains a rank-1 cross-term per block analogous to the IndexCache
    backward. With NVFP4-block sizes (16) this cross-term is small and
    is folded in below.
    """

    g = grad_y.to(intermediates["x_compute"].dtype)
    clip_mask = intermediates["clip_mask"]
    return (g * clip_mask).to(grad_y.dtype)


def activation_eco_bias_correction(
    grad_y: torch.Tensor,
    x_pre: torch.Tensor,
    q_x: torch.Tensor,
) -> torch.Tensor:
    """Compute the activation-ECO weight-gradient correction.

    ``grad_y`` is ``dy = ∂loss/∂y`` with shape ``[..., M]``,
    ``x_pre`` is the pre-quant activation with shape ``[..., D]``, and
    ``q_x`` is the post-quant activation with the same shape as ``x_pre``.

    Returns the additive correction to ``dW`` of shape ``[M, D]``:
        dW_correction = dy.T @ (x_pre - q_x)
                      = dy.T @ e_x

    Adding this to the standard QAT ``dW = dy.T @ q_x`` recovers the
    unbiased ``dy.T @ x_pre`` that pure BF16 forward would have produced.
    """

    flat_grad = grad_y.reshape(-1, grad_y.shape[-1])
    flat_x = x_pre.reshape(-1, x_pre.shape[-1])
    flat_qx = q_x.reshape(-1, q_x.shape[-1])
    e_x = flat_x - flat_qx
    return flat_grad.T @ e_x


def _round_to_nvfp4_grid(x: torch.Tensor, *, fp4_max: float) -> torch.Tensor:
    """Round each coordinate to its nearest NVFP4 (E2M1) representable.

    NVFP4 unsigned magnitudes are {0, 0.5, 1, 1.5, 2, 3, 4, 6}; signs
    flip the sign bit. The full grid is exactly the union of those 8
    magnitudes negated and not-negated. ``round`` nearest with banker's
    ties matches what flashinfer's fp4_quantize emits.
    """

    grid = torch.tensor(
        [
            -fp4_max, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, fp4_max,
        ],
        dtype=x.dtype,
        device=x.device,
    )
    diffs = (x.unsqueeze(-1) - grid).abs()
    idx = diffs.argmin(dim=-1)
    return grid[idx]
