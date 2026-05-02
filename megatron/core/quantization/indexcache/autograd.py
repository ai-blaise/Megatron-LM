# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``torch.autograd.Function`` wrapper for the IndexCache fake-quant.

Phase 1 dispatches to the pure-PyTorch reference. Phase 2 swaps the forward
in to the fused CUDA kernel and Phase 3 swaps the backward; the autograd
wrapper itself does not change shape — same save-for-backward signature,
same per-token semantics.

The op is per-token-local on the last dim. Inputs of any shape ``[..., D]``
are flattened to ``[N, D]`` for the kernel and reshaped back on output.
"""

from __future__ import annotations

import torch

from megatron.core.quantization.indexcache.codec import IndexCacheConfig
from megatron.core.quantization.indexcache.reference import (
    indexcache_backward,
    indexcache_forward,
)


def _try_load_cuda_ext():
    try:
        from megatron.core.quantization.indexcache.kernels.build import get_ext

        return get_ext()
    except (ImportError, RuntimeError):
        return None


class IndexCacheKVFn(torch.autograd.Function):
    """fp8 e4m3 fake-quant for the indexer K tensor with STE-based backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, config: IndexCacheConfig) -> torch.Tensor:
        original_shape = x.shape
        head_dim = original_shape[-1]
        flat = x.reshape(-1, head_dim).contiguous()

        ext = _try_load_cuda_ext() if x.is_cuda else None
        if ext is not None and hasattr(ext, "indexcache_kv_fwd"):
            n = flat.shape[0]
            out = torch.empty_like(flat)
            scale = torch.empty(n, dtype=torch.float32, device=x.device)
            q_fp8 = torch.empty(
                n, head_dim, dtype=torch.float8_e4m3fn, device=x.device
            )
            clip_mask = torch.empty(n, head_dim, dtype=torch.uint8, device=x.device)
            argmax = torch.empty(n, dtype=torch.int32, device=x.device)
            eps_active = torch.empty(n, dtype=torch.uint8, device=x.device)
            ext.indexcache_kv_fwd(
                flat, out, scale, q_fp8, clip_mask, argmax, eps_active,
                config.eps, config.fp8_max,
            )
            ctx.cuda_path = True
            ctx.save_for_backward(flat, scale, q_fp8, clip_mask, argmax, eps_active)
        else:
            out, intermediates = indexcache_forward(
                flat, config, return_intermediates=True
            )
            ctx.cuda_path = False
            ctx.save_for_backward(
                intermediates["x_compute"],
                intermediates["scale"],
                intermediates["q_fp8"],
                intermediates["clip_mask"],
                intermediates["eps_active"],
            )

        ctx.config = config
        ctx.original_shape = original_shape
        return out.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        config = ctx.config
        head_dim = ctx.original_shape[-1]
        grad_flat = grad_y.reshape(-1, head_dim).contiguous()

        if getattr(ctx, "cuda_path", False):
            x_flat, scale, q_fp8, clip_mask, argmax, eps_active = ctx.saved_tensors
            ext = _try_load_cuda_ext()
            grad_x = torch.empty_like(x_flat)
            ext.indexcache_kv_bwd(
                grad_flat, x_flat, scale, q_fp8, clip_mask, argmax, eps_active,
                grad_x, config.fp8_max,
            )
        else:
            x_compute, scale, q_fp8, clip_mask, eps_active = ctx.saved_tensors
            intermediates = {
                "x_compute": x_compute,
                "scale": scale,
                "q_fp8": q_fp8,
                "clip_mask": clip_mask,
                "eps_active": eps_active,
            }
            grad_x = indexcache_backward(grad_flat, intermediates, config)

        return grad_x.reshape(ctx.original_shape), None


def apply_indexcache_kv(
    k: torch.Tensor, config: IndexCacheConfig
) -> torch.Tensor:
    """Public entry. Accepts any shape whose last dim is the head dimension."""

    return IndexCacheKVFn.apply(k, config)
