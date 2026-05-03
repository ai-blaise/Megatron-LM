# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional Transformer Engine `te.Linear` hook for activation-ECO.

Production NVFP4 training uses `te.Linear` for the matmul; activation-ECO
needs to observe the same activation cast TE applies. The hook in this
module wraps a `te.Linear` instance so that:

  * forward: capture the pre-cast BF16 activation, run TE's NVFP4 forward
    as today, then return TE's output unchanged.
  * backward: in addition to TE's standard STE-through-cast gradient,
    add the activation-ECO correction
    ``dW += dy.T @ (x_pre - q(x_pre))``
    to TE's accumulated weight gradient.

TE is GPU-only and must be installed separately (Blackwell + matching
cuBLAS). This module imports lazily so non-TE environments (CPU CI,
fallback BF16 backends) can still import the rest of the
``nvfp4_act_eco`` package.
"""

from __future__ import annotations

import torch

from megatron.core.quantization.nvfp4_act_eco.codec import Nvfp4ActEcoConfig
from megatron.core.quantization.nvfp4_act_eco.reference import (
    activation_eco_bias_correction,
    nvfp4_act_quant_forward,
)


def _try_import_te():
    try:
        import transformer_engine.pytorch as te

        return te
    except Exception:
        return None


def is_te_available() -> bool:
    """Return True if Transformer Engine is importable in this environment."""
    return _try_import_te() is not None


def install_act_eco_on_te_linear(
    te_linear: torch.nn.Module, config: Nvfp4ActEcoConfig
) -> None:
    """Attach activation-ECO bias correction to a `te.Linear` instance.

    Implementation strategy: register a forward_pre_hook that stashes
    `x_pre` on the module, then a full backward hook on the module's
    weight that adds ``dy.T @ e_x`` to the accumulated gradient. We
    reconstruct ``q(x_pre)`` from the same reference fake-quant the
    rest of this module uses; on production hardware the TE cast
    produces a bit-identical (or near-identical) result, so the
    correction term ``e_x = x_pre - q(x_pre)`` matches what TE would
    have computed internally.

    No-op when te_linear has no ``weight`` attribute.
    """
    if not hasattr(te_linear, "weight"):
        return
    if getattr(te_linear, "_act_eco_installed", False):
        return

    cell = {"x_pre": None, "dy": None}

    def _pre(_module, args, _kwargs):
        if not args:
            return None
        x = args[0]
        cell["x_pre"] = x.detach()
        return None

    def _post(_module, _args, output):
        # Tap into the output's grad_fn so we can capture dy in backward.
        if not isinstance(output, torch.Tensor) or not output.requires_grad:
            return output

        class _CaptureGrad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y):
                return y

            @staticmethod
            def backward(ctx, dy):
                cell["dy"] = dy.detach()
                return dy

        return _CaptureGrad.apply(output)

    def _weight_hook(grad):
        x_pre = cell["x_pre"]
        dy = cell["dy"]
        if x_pre is None or dy is None:
            return grad
        q_x = nvfp4_act_quant_forward(
            x_pre.reshape(-1, x_pre.shape[-1]).to(torch.float32),
            config,
        )
        correction = activation_eco_bias_correction(
            dy.reshape(-1, dy.shape[-1]).to(torch.float32),
            x_pre.reshape(-1, x_pre.shape[-1]).to(torch.float32),
            q_x,
        )
        cell["x_pre"] = None
        cell["dy"] = None
        return grad + correction.to(grad.dtype).reshape(grad.shape)

    te_linear.register_forward_pre_hook(_pre, with_kwargs=True)
    te_linear.register_forward_hook(_post)
    te_linear.weight.register_hook(_weight_hook)
    te_linear._act_eco_installed = True
