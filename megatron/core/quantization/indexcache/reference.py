# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-PyTorch reference forward and backward for IndexCache fp8 fake-quant.

The forward is a faithful port of SGLang's ``act_quant``
(triton_kernel.py:_act_quant_kernel) followed by an immediate dequantize so
the op acts as a fake-quant suitable for training. The backward is new.

Forward pseudocode (SGLang lines 30–73):
    abs_max = max(|x|, dim=-1)
    amax    = max(abs_max, eps)
    scale   = amax / fp8_max
    q       = clamp(x / scale, -fp8_max, +fp8_max)
    q_fp8   = cast<fp8_e4m3>(q)         # introduces rounding noise
    y       = q_fp8 * scale              # dequantized fake-quant output

Backward (closed form):
    Let mask_i = 1[|x_i / scale| <= fp8_max] (saturating-STE on the clip).
    Let q_i be the post-cast fp8 value (a constant per coord at backward time).
    Let amax_active = 1[amax_pre_clamp >= eps] (1 if the eps clamp is inactive).

    Then per coordinate:
        dy_i / dx_i (direct, STE)         = mask_i
        dscale / dx_j (per row)           = sign(x_argmax) / fp8_max
                                            * 1[j == argmax] * amax_active
        dy_i / dscale                     = q_i

    Total grad_x_j = grad_y_j * mask_j
                   + (sum_i grad_y_i * q_i) * dscale/dx_j

The dscale/dx_j contribution is rank-1 per row: it lives only on the argmax
coordinate. We compute it explicitly so the backward matches torch.autograd
on the STE-detach forward to within fp32 reduction noise.
"""

from __future__ import annotations

import torch

from megatron.core.quantization.indexcache.codec import IndexCacheConfig


def indexcache_forward(
    x: torch.Tensor,
    config: IndexCacheConfig,
    *,
    return_intermediates: bool = False,
):
    """Apply fp8 e4m3 fake-quant per-token to a 2-D tensor [N, D].

    The output dtype matches ``x.dtype``. When the host has fp8 support this
    routes through ``torch.float8_e4m3fn``; otherwise the rounding is
    simulated in fp32 to keep CPU-only tests viable.
    """

    if x.dim() != 2:
        raise ValueError(
            "indexcache_forward expects [N, D]; got shape "
            f"{tuple(x.shape)}. Reshape upstream."
        )

    orig_dtype = x.dtype
    compute_dtype = (
        x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    )
    xf = x.to(compute_dtype)
    abs_x = xf.abs()
    abs_max = abs_x.amax(dim=-1)
    amax = abs_max.clamp_min(config.eps)
    scale = amax * config.fp8_max_inv
    scale_b = scale[:, None]

    # Quantize → fp8 → dequantize. We use torch.float8_e4m3fn when available
    # to get exact-cast rounding; otherwise simulate the same effective
    # precision (3 mantissa bits exponent-biased) using compositional ops so
    # the reference can run on CPU without fp8 support.
    pre_cast = (xf / scale_b).clamp_(-config.fp8_max, config.fp8_max)
    if pre_cast.is_cuda or hasattr(torch, "float8_e4m3fn"):
        q_fp8 = pre_cast.to(torch.float8_e4m3fn).to(compute_dtype)
    else:
        q_fp8 = _simulate_fp8_e4m3_rounding(pre_cast)

    y = (q_fp8 * scale_b).to(orig_dtype)

    if return_intermediates:
        clip_mask = (
            (xf / scale_b >= -config.fp8_max) & (xf / scale_b <= config.fp8_max)
        ).to(compute_dtype)
        # eps clamp inactive iff abs_max >= eps; when active the scale is
        # fixed, so its derivative wrt x is zero on the argmax lane too.
        eps_active = (abs_max >= config.eps).to(compute_dtype)
        return y, {
            "x_compute": xf,
            "scale": scale,
            "q_fp8": q_fp8,
            "clip_mask": clip_mask,
            "eps_active": eps_active,
        }
    return y


def indexcache_backward(
    grad_y: torch.Tensor,
    intermediates: dict,
    config: IndexCacheConfig,
) -> torch.Tensor:
    """Closed-form gradient through the fp8 fake-quant.

    See the module docstring for the derivation.
    """

    g = grad_y.to(intermediates["x_compute"].dtype)
    xf = intermediates["x_compute"]
    q_fp8 = intermediates["q_fp8"]
    clip_mask = intermediates["clip_mask"]
    eps_active = intermediates["eps_active"]

    # Derivation (matches the STE-detach forward used as the autograd oracle):
    #   y_i = scale * q_fp8_i  with q_fp8_i routed via STE through the clipped
    #   pre_clip = x/scale, so the autograd path is:
    #       dy_i/dx_j = mask_i * delta_ij
    #                 + dscale/dx_j * (q_fp8_i - mask_i * x_i / scale)
    #   The first term is the direct STE; the second is the rank-1 scale-
    #   path contribution restricted to the row's argmax-of-|x| coordinate.
    grad_direct = g * clip_mask

    scale = intermediates["scale"]  # [N]
    scale_b = scale[:, None]
    inner = (g * (q_fp8 - clip_mask * xf / scale_b)).sum(dim=-1)  # [N]

    abs_xf = xf.abs()
    argmax = abs_xf.argmax(dim=-1)
    sign_at_argmax = torch.gather(xf.sign(), -1, argmax[:, None]).squeeze(-1)
    scale_grad_factor = (sign_at_argmax * eps_active) * config.fp8_max_inv  # [N]
    contrib = inner * scale_grad_factor  # [N]
    rank1 = torch.zeros_like(grad_direct)
    rank1.scatter_(-1, argmax[:, None], contrib[:, None])

    return (grad_direct + rank1).to(grad_y.dtype)


def _simulate_fp8_e4m3_rounding(x: torch.Tensor) -> torch.Tensor:
    """Approximate fp8_e4m3 rounding for CPU paths without native fp8.

    fp8_e4m3 has 4 exponent bits, 3 mantissa bits, no infinities, finite
    range [-448, 448]. We round to the nearest representable value by
    re-encoding via float-to-int truncation on the mantissa. This is a
    *test-only* convenience — production paths always run on CUDA where
    ``torch.float8_e4m3fn`` is the source of truth.
    """

    # Round mantissa to 3 bits per IEEE-style round-to-nearest-even via the
    # exponent normalization trick: subtract the exponent, round, restore.
    abs_x = x.abs().clamp_min(1e-30)
    exponents = torch.floor(torch.log2(abs_x))
    mantissa_scale = torch.pow(2.0, exponents - 3)
    rounded = torch.round(x / mantissa_scale) * mantissa_scale
    return rounded.clamp(-448.0, 448.0)
