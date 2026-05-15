# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``torch.autograd.Function`` wrapper for the IndexCache fake-quant.

FP8 can use the CUDA extension on CUDA hosts. NVFP4 uses the Python reference
unless running on Blackwell or newer, where the guarded CUDA path is eligible.

The op is per-token-local on the last dim. Inputs of any shape ``[..., D]``
are flattened to ``[N, D]`` for the kernel and reshaped back on output.
"""

from __future__ import annotations

import torch

from megatron.core.quantization.indexcache.codec import (
    INDEXCACHE_QUANT_FP8,
    INDEXCACHE_QUANT_NVFP4,
    IndexCacheConfig,
)
from megatron.core.quantization.indexcache.reference import (
    indexcache_backward,
    indexcache_forward,
)


def _try_load_cuda_ext():
    try:
        from megatron.core.quantization.indexcache.kernels.build import get_ext

        return get_ext()
    except (ImportError, RuntimeError, OSError):
        return None


def _is_blackwell_or_newer(device: torch.device | int | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 10


class IndexCacheKVFn(torch.autograd.Function):
    """IndexCache fake-quant for the indexer K tensor with STE-based backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, config: IndexCacheConfig) -> torch.Tensor:
        original_shape = x.shape
        head_dim = original_shape[-1]
        flat = x.reshape(-1, head_dim).contiguous()

        ext = None
        if x.is_cuda and config.quantization == INDEXCACHE_QUANT_FP8:
            ext = _try_load_cuda_ext()
        elif (
            x.is_cuda
            and config.quantization == INDEXCACHE_QUANT_NVFP4
            and _is_blackwell_or_newer(x.device)
        ):
            ext = _try_load_cuda_ext()
        if (
            config.quantization == INDEXCACHE_QUANT_FP8
            and ext is not None
            and hasattr(ext, "indexcache_kv_fwd")
        ):
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
            ctx.cuda_quantization = INDEXCACHE_QUANT_FP8
            ctx.save_for_backward(flat, scale, q_fp8, clip_mask, argmax, eps_active)
        elif (
            config.quantization == INDEXCACHE_QUANT_NVFP4
            and x.is_cuda
            and _is_blackwell_or_newer(x.device)
            and ext is not None
            and hasattr(ext, "indexcache_nvfp4_fwd")
        ):
            n = flat.shape[0]
            out = torch.empty_like(flat)
            scale = torch.empty(n, 4, dtype=torch.float32, device=x.device)
            q_e2m1 = torch.empty(n, head_dim, dtype=torch.float32, device=x.device)
            clip_mask = torch.empty(n, head_dim, dtype=torch.uint8, device=x.device)
            argmax = torch.empty(n, 4, dtype=torch.int32, device=x.device)
            eps_active = torch.empty(n, 4, dtype=torch.uint8, device=x.device)
            packed_values = torch.empty(n, 64, dtype=torch.uint8, device=x.device)
            packed_scales = torch.empty(n, dtype=torch.int32, device=x.device)
            ext.indexcache_nvfp4_fwd(
                flat, out, scale, q_e2m1, clip_mask, argmax, eps_active,
                packed_values, packed_scales, config.eps,
            )
            ctx.cuda_path = True
            ctx.cuda_quantization = INDEXCACHE_QUANT_NVFP4
            ctx.save_for_backward(
                flat, scale, packed_values, clip_mask, argmax, eps_active
            )
        else:
            out, intermediates = indexcache_forward(
                flat, config, return_intermediates=True
            )
            ctx.cuda_path = False
            if config.quantization == INDEXCACHE_QUANT_NVFP4:
                ctx.save_for_backward(
                    intermediates["x_compute"],
                    intermediates["scale"],
                    intermediates["q_e2m1"],
                    intermediates["clip_mask"],
                    intermediates["argmax"],
                    intermediates["eps_active"],
                )
            else:
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
            ext = _try_load_cuda_ext()
            if ctx.cuda_quantization == INDEXCACHE_QUANT_FP8:
                x_flat, scale, q_fp8, clip_mask, argmax, eps_active = ctx.saved_tensors
                grad_x = torch.empty_like(x_flat)
                ext.indexcache_kv_bwd(
                    grad_flat, x_flat, scale, q_fp8, clip_mask, argmax, eps_active,
                    grad_x, config.fp8_max,
                )
            elif ctx.cuda_quantization == INDEXCACHE_QUANT_NVFP4:
                x_flat, scale, packed_values, clip_mask, argmax, eps_active = (
                    ctx.saved_tensors
                )
                grad_x = torch.empty_like(x_flat)
                ext.indexcache_nvfp4_bwd_packed(
                    grad_flat,
                    x_flat,
                    scale,
                    packed_values,
                    clip_mask,
                    argmax,
                    eps_active,
                    grad_x,
                    config.fp4_max,
                )
            else:
                raise RuntimeError(
                    f"Unsupported CUDA IndexCache quantization {ctx.cuda_quantization!r}."
                )
        else:
            if config.quantization == INDEXCACHE_QUANT_NVFP4:
                x_compute, scale, q_e2m1, clip_mask, argmax, eps_active = ctx.saved_tensors
                intermediates = {
                    "x_compute": x_compute,
                    "scale": scale,
                    "q_e2m1": q_e2m1,
                    "clip_mask": clip_mask,
                    "argmax": argmax,
                    "eps_active": eps_active,
                }
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

    if not config.is_enabled:
        return k
    return IndexCacheKVFn.apply(k, config)
