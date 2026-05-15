# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``torch.autograd.Function`` wrapper for the 2-bit HIGGS dense-KV fake-quant.

Forward dispatch picks the fused CUDA kernel when ``x`` is a CUDA tensor and
the build has succeeded; otherwise it falls back to the pure-PyTorch
reference. Both paths are mathematically identical modulo fp32 reduction
order. The op is per-token-local on the latent dim; ``forward`` and
``backward`` reshape any input to ``[N, latent_dim]``, run, and reshape back.
This is what makes the op safe under TP/SP/CP/EP --- the same parallelism
analysis that applies to TurboQuant applies here unchanged.

The backward path uses a saturating straight-through estimator on the
codebook step. See ``reference.higgs_backward`` for the closed form.
"""

from __future__ import annotations

import torch

from megatron.core.quantization.higgs.codec import HiggsBuffers
from megatron.core.quantization.higgs.reference import (
    higgs_backward,
    higgs_forward,
)


def _try_load_cuda_ext():
    try:
        from megatron.core.quantization.higgs.kernels.build import get_ext

        return get_ext()
    except (ImportError, RuntimeError):
        return None


class HiggsDenseKVFn(torch.autograd.Function):
    """Fake-quant the 512-dim MLA latent with an STE backward.

    Forward stores either CUDA-saved intermediates (indices + scale + mask)
    or reference intermediates (full dict) on ``ctx``. Backward dispatches
    to the matching path.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, buffers: HiggsBuffers) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, buffers.latent_dim).contiguous()

        ext = _try_load_cuda_ext() if x.is_cuda else None
        if ext is not None and hasattr(ext, "higgs_kv_fwd"):
            n = flat.shape[0]
            out = torch.empty_like(flat)
            indices = torch.empty(
                n, buffers.num_pairs, dtype=torch.uint8, device=x.device
            )
            ste_mask = torch.empty(
                n, buffers.latent_dim, dtype=torch.uint8, device=x.device
            )
            rot_norm = torch.empty(n, dtype=torch.float32, device=x.device)
            scale = torch.empty(n, dtype=torch.float32, device=x.device)
            # rotated and rotated_recon_unit are bf16-saved to keep activation
            # memory bounded (256 B/token each) while skipping the recompute
            # FWHTs in the backward kernel. The same pattern is used by the
            # TurboQuant w_hat_save buffer (turboquant_kv_fwd.cu:66-80) and
            # validated bit-equivalent at <2e-6 vs the recompute path.
            rotated_save = torch.empty(
                n, buffers.latent_dim, dtype=torch.bfloat16, device=x.device
            )
            recon_unit_save = torch.empty(
                n, buffers.latent_dim, dtype=torch.bfloat16, device=x.device
            )
            ext.higgs_kv_fwd(
                flat,
                out,
                indices,
                ste_mask,
                rot_norm,
                scale,
                rotated_save,
                recon_unit_save,
                buffers.codebook.to(x.device).float(),
                buffers.codebook_norm_sq.to(x.device).float(),
            )
            ctx.cuda_path = True
            ctx.save_for_backward(
                flat, indices, ste_mask, rot_norm, scale,
                rotated_save, recon_unit_save,
            )
        else:
            out_ref, intermediates = higgs_forward(
                flat, buffers, return_intermediates=True
            )
            out = out_ref
            ctx.cuda_path = False
            ctx.save_for_backward(
                intermediates["x_compute"],
                intermediates["rotated"],
                intermediates["rot_norm"],
                intermediates["scale"],
                intermediates["normalized"],
                intermediates["indices"],
                intermediates["rotated_recon_unit"],
                intermediates["rotated_recon"],
                intermediates["ste_mask"],
            )

        ctx.buffers = buffers
        ctx.original_shape = original_shape
        return out.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        buffers = ctx.buffers
        grad_flat = grad_y.reshape(-1, buffers.latent_dim).contiguous()

        if getattr(ctx, "cuda_path", False):
            (
                x_flat,
                indices,
                ste_mask,
                rot_norm,
                scale,
                rotated_save,
                recon_unit_save,
            ) = ctx.saved_tensors
            ext = _try_load_cuda_ext()
            grad_x = torch.empty_like(x_flat)
            ext.higgs_kv_bwd(
                grad_flat,
                x_flat,
                indices,
                ste_mask,
                rot_norm,
                scale,
                rotated_save,
                recon_unit_save,
                buffers.codebook.to(x_flat.device).float(),
                buffers.codebook_norm_sq.to(x_flat.device).float(),
                grad_x,
            )
        else:
            (
                x_compute,
                rotated,
                rot_norm,
                scale,
                normalized,
                indices,
                rotated_recon_unit,
                rotated_recon,
                ste_mask,
            ) = ctx.saved_tensors
            intermediates = {
                "x_compute": x_compute,
                "rotated": rotated,
                "rot_norm": rot_norm,
                "scale": scale,
                "normalized": normalized,
                "indices": indices,
                "rotated_recon_unit": rotated_recon_unit,
                "rotated_recon": rotated_recon,
                "ste_mask": ste_mask,
            }
            grad_x = higgs_backward(grad_flat, intermediates, buffers)

        return grad_x.reshape(ctx.original_shape), None


def apply_higgs_dense_2bit_kv(
    latent: torch.Tensor, buffers: HiggsBuffers
) -> torch.Tensor:
    """Public entry point. Accepts any shape whose last dim is ``latent_dim``."""

    return HiggsDenseKVFn.apply(latent, buffers)
