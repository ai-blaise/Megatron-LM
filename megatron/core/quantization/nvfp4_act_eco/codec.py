# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Activation-ECO config — NVFP4 per-block fake-quant for layer inputs.

The activation cast that TE applies inside ``te.Linear`` rounds each
input tensor to NVFP4 (per-block FP8 scale + per-tensor FP32 global
scale) before the cuBLASLt FP4 GEMM. With straight-through estimation
on the saturated lanes the gradient flows back as if the cast were
the identity — but the gradient ``dW = dy @ q(x).T`` produced by the
GEMM is biased by the activation rounding. This module implements
the bias-correction term ``dW += dy @ (x - q(x)).T`` that lifts the
weight gradient back to the unbiased ``dy @ x.T`` it would have been
in pure BF16.
"""

from __future__ import annotations

from dataclasses import dataclass


# NVFP4 (E2M1) representable max; matches FP4_E2M1_MAX in flashinfer
# and the value used by FlashOptim's NVFP4 master cast in
# ``megatron/core/optimizer/nvfp4_sr.py``.
NVFP4_E2M1_MAX = 6.0

# FP8 E4M3 representable max; per-block scales are stored at this
# precision in the on-disk compressed-tensors format.
FP8_E4M3_MAX = 448.0

# NVFP4 per-block size; one FP8 scale governs each contiguous group
# of 16 elements along the hidden dim.
NVFP4_BLOCK_SIZE = 16


@dataclass(frozen=True)
class Nvfp4ActEcoConfig:
    """Stateless config for activation-ECO bias correction."""

    block_size: int = NVFP4_BLOCK_SIZE
    fp4_max: float = NVFP4_E2M1_MAX
    fp8_max: float = FP8_E4M3_MAX
    eps: float = 1e-6


def build_nvfp4_act_eco_config(
    *, block_size: int = NVFP4_BLOCK_SIZE, eps: float = 1e-6
) -> Nvfp4ActEcoConfig:
    """Public constructor.

    Mirrors ``build_indexcache_config`` so a model can hold one
    long-lived config object covering both fake-quant ops.
    """

    if block_size <= 0:
        raise ValueError(f"block_size must be positive; got {block_size}")
    return Nvfp4ActEcoConfig(block_size=block_size, eps=eps)
