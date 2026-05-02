# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``torch.autograd.Function`` wrapper for the TurboQuant dense-KV fake-quant.

The forward path runs the reference implementation; this is the rung on the
ladder that lets us validate gradcheck end-to-end before any CUDA work
happens. Phase 2 swaps the forward call site for the fused CUDA kernel and
Phase 3 swaps backward for its own kernel — the autograd.Function shell stays
the same.

The op is per-token (per-row) local: ``forward`` and ``backward`` reshape any
input to ``[N, latent_dim]``, run, and reshape back. This is what makes the
op safe under TP/SP/CP/EP — see ``docs/turboquant/04_mla_integration.md`` for
the parallelism analysis.
"""

from __future__ import annotations

import torch

from megatron.core.quantization.turboquant.codec import TurboQuantBuffers
from megatron.core.quantization.turboquant.reference import (
    turboquant_backward,
    turboquant_forward,
)


class TurboQuantKVFn(torch.autograd.Function):
    """Fake-quant the 512-dim MLA latent with a saturating-STE backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, buffers: TurboQuantBuffers) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, buffers.latent_dim)
        out, intermediates = turboquant_forward(flat, buffers, return_intermediates=True)

        ctx.save_for_backward(
            intermediates["x_fp32"],
            intermediates["norm"],
            intermediates["inner_norm"],
            intermediates["norm_hat"],
            intermediates["w_hat"],
            intermediates["z_hat"],
            intermediates["ste_mask"],
        )
        ctx.buffers = buffers
        ctx.original_shape = original_shape
        return out.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_x_hat: torch.Tensor):
        (x_fp32, norm, inner_norm, norm_hat, w_hat, z_hat, ste_mask) = ctx.saved_tensors
        grad_flat = grad_x_hat.reshape(-1, ctx.buffers.latent_dim)
        intermediates = {
            "x_fp32": x_fp32,
            "norm": norm,
            "inner_norm": inner_norm,
            "norm_hat": norm_hat,
            "w_hat": w_hat,
            "z_hat": z_hat,
            "ste_mask": ste_mask,
        }
        grad_x = turboquant_backward(grad_flat, intermediates, ctx.buffers)
        return grad_x.reshape(ctx.original_shape), None


def apply_turboquant_kv(
    latent: torch.Tensor, buffers: TurboQuantBuffers
) -> torch.Tensor:
    """Public entry point. Accepts any shape whose last dim is ``latent_dim``."""

    return TurboQuantKVFn.apply(latent, buffers)
