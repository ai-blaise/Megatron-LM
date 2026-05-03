# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``torch.autograd.Function`` for activation-ECO fused with a Linear op.

The fused op covers ``y = q(x) @ W.T`` (the canonical Linear forward
shape used by Megatron's ``ColumnParallelLinear`` / ``RowParallelLinear``
and by ``te.Linear``). Forward emits ``q(x) @ W.T``; backward returns:

  * ``dx``: standard STE through the activation cast.
  * ``dW``: the activation-ECO bias-corrected weight gradient
    ``dy.T @ x_pre`` instead of the biased ``dy.T @ q(x)``.

In production the matmul itself runs in NVFP4 via cuBLASLt; this
Function only owns the bias-correction term added to ``dW``. Wiring
this Function in front of ``te.Linear`` (or replacing its autograd
path) is the single hook needed to activate activation-ECO.
"""

from __future__ import annotations

import torch

from megatron.core.quantization.nvfp4_act_eco.codec import Nvfp4ActEcoConfig
from megatron.core.quantization.nvfp4_act_eco.reference import (
    activation_eco_bias_correction,
    nvfp4_act_quant_backward,
    nvfp4_act_quant_forward,
)


class Nvfp4ActEcoLinearFn(torch.autograd.Function):
    """NVFP4 activation fake-quant fused with a Linear, with activation-ECO.

    Forward shape contract:
        x:  [..., D]   (any leading dims; only the last is quantized)
        W:  [M, D]
        y:  [..., M]
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        config: Nvfp4ActEcoConfig,
    ) -> torch.Tensor:
        original_shape = x.shape
        d = original_shape[-1]
        flat = x.reshape(-1, d).contiguous()

        q_x, intermediates = nvfp4_act_quant_forward(
            flat, config, return_intermediates=True
        )
        y = q_x @ weight.T

        ctx.save_for_backward(
            flat,
            q_x,
            weight,
            intermediates["clip_mask"],
            intermediates["block_scale"],
        )
        ctx.config = config
        ctx.original_shape = original_shape
        ctx.out_features = weight.shape[0]
        return y.reshape(*original_shape[:-1], ctx.out_features)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x_pre, q_x, weight, clip_mask, block_scale = ctx.saved_tensors
        config = ctx.config
        d = ctx.original_shape[-1]
        flat_g = grad_y.reshape(-1, ctx.out_features)

        # Standard activation backward through STE: dq(x)/dx = clip_mask
        dx_pre = (flat_g @ weight) * clip_mask
        dx = dx_pre.reshape(ctx.original_shape).to(grad_y.dtype)

        # Standard QAT weight gradient: dW = dy.T @ q(x).
        dw_naive = flat_g.T @ q_x

        # Activation-ECO correction: dW += dy.T @ (x_pre - q(x)).
        dw_correction = activation_eco_bias_correction(
            flat_g, x_pre, q_x
        )
        dw = (dw_naive + dw_correction).to(weight.dtype)
        return dx, dw, None


def apply_nvfp4_act_eco_linear(
    x: torch.Tensor, weight: torch.Tensor, config: Nvfp4ActEcoConfig
) -> torch.Tensor:
    """Public entry point for the fused activation-ECO Linear forward."""
    return Nvfp4ActEcoLinearFn.apply(x, weight, config)
