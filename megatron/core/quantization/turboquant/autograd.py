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


def _try_load_cuda_ext():
    try:
        from megatron.core.quantization.turboquant.kernels.build import get_ext

        return get_ext()
    except (ImportError, RuntimeError):
        return None


class TurboQuantKVFn(torch.autograd.Function):
    """Fake-quant the 512-dim MLA latent with a saturating-STE backward.

    Forward dispatch picks the fused CUDA kernel when ``x`` is a CUDA tensor
    *and* the build has succeeded; otherwise it falls back to the pure-PyTorch
    reference. Both paths are mathematically identical.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, buffers: TurboQuantBuffers) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, buffers.latent_dim).contiguous()

        ext = _try_load_cuda_ext() if x.is_cuda else None
        if ext is not None:
            out = torch.empty_like(flat)
            n = flat.shape[0]
            indices = torch.empty(n, buffers.latent_dim, dtype=torch.uint8, device=x.device)
            ste_mask = torch.empty(n, buffers.latent_dim, dtype=torch.uint8, device=x.device)
            norm = torch.empty(n, dtype=torch.float32, device=x.device)
            inner_norm = torch.empty(n, dtype=torch.float32, device=x.device)
            # Save w_hat in fp32 to skip the recompute_w_hat region of the
            # backward kernel (~20% of bwd time per region-split profile).
            # bf16 was tried first but the round-trip error compounded through
            # the cross-coordinate sum reduction in chain_outputs to ~1e-3
            # absolute (vs the 1e-6 fp32 baseline); fp32 keeps the gradient
            # at machine-precision parity at the cost of 2 KB/token.
            w_hat_save = torch.empty(
                n, buffers.latent_dim, dtype=torch.float32, device=x.device
            )
            ext.turboquant_kv_fwd(
                flat,
                out,
                indices,
                ste_mask,
                norm,
                inner_norm,
                w_hat_save,
                buffers.signs1.to(x.device).float(),
                buffers.signs2.to(x.device).float(),
                buffers.boundaries_high.to(x.device).float(),
                buffers.boundaries_low.to(x.device).float(),
                buffers.centroids_high.to(x.device).float(),
                buffers.centroids_low.to(x.device).float(),
                buffers.norm_correction,
            )
            ctx.cuda_path = True
            ctx.save_for_backward(flat, indices, ste_mask, norm, inner_norm, w_hat_save)
        else:
            out_ref, intermediates = turboquant_forward(
                flat, buffers, return_intermediates=True
            )
            out = out_ref
            ctx.cuda_path = False
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
        buffers = ctx.buffers
        grad_flat = grad_x_hat.reshape(-1, buffers.latent_dim).contiguous()

        if getattr(ctx, "cuda_path", False):
            (x_flat, indices, ste_mask, norm, inner_norm, w_hat_saved
             ) = ctx.saved_tensors
            ext = _try_load_cuda_ext()
            grad_x = torch.empty_like(x_flat)
            ext.turboquant_kv_bwd(
                grad_flat,
                x_flat,
                indices,
                ste_mask,
                norm,
                inner_norm,
                w_hat_saved,
                buffers.signs1.to(x_flat.device).float(),
                buffers.signs2.to(x_flat.device).float(),
                buffers.centroids_high.to(x_flat.device).float(),
                buffers.centroids_low.to(x_flat.device).float(),
                grad_x,
                buffers.norm_correction,
            )
        else:
            (x_fp32, norm, inner_norm, norm_hat, w_hat, z_hat, ste_mask) = ctx.saved_tensors
            intermediates = {
                "x_fp32": x_fp32,
                "norm": norm,
                "inner_norm": inner_norm,
                "norm_hat": norm_hat,
                "w_hat": w_hat,
                "z_hat": z_hat,
                "ste_mask": ste_mask,
            }
            grad_x = turboquant_backward(grad_flat, intermediates, buffers)

        return grad_x.reshape(ctx.original_shape), None


def apply_turboquant_kv(
    latent: torch.Tensor, buffers: TurboQuantBuffers
) -> torch.Tensor:
    """Public entry point. Accepts any shape whose last dim is ``latent_dim``."""

    return TurboQuantKVFn.apply(latent, buffers)
