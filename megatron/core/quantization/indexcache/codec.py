# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""IndexCache config and quantization method selection.

Unlike TurboQuant the IndexCache op carries no frozen randomness or
codebook: the per-token scale is computed online from the input. The only
state is the eps that prevents divide-by-zero on all-zero tokens, which we
keep configurable to track upstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass


# fp8 e4m3 max representable value; matches SGLang's
# fused_store_index_cache.cuh and triton_kernel.py.
INDEXCACHE_FP8_MAX = 448.0

# NVFP4 (E2M1) max representable magnitude; the indexer uses four groups of
# 32 dims per 128-dim row and packs their UE8M0 scale exponents into one word.
INDEXCACHE_NVFP4_E2M1_MAX = 6.0
INDEXCACHE_NVFP4_GROUP_SIZE = 32
INDEXCACHE_NVFP4_HEAD_DIM = 128

INDEXCACHE_QUANT_DISABLED = "disabled"
INDEXCACHE_QUANT_FP8 = "fp8_e4m3"
INDEXCACHE_QUANT_NVFP4 = "nvfp4_e2m1_ue8m0"
INDEXCACHE_QUANTIZATION_CHOICES = (
    INDEXCACHE_QUANT_DISABLED,
    INDEXCACHE_QUANT_FP8,
    INDEXCACHE_QUANT_NVFP4,
)


@dataclass(frozen=True)
class IndexCacheConfig:
    """Stateless config for fake-quant on the DSA indexer K tensor."""

    quantization: str = INDEXCACHE_QUANT_FP8
    eps: float = 1e-4
    fp8_max: float = INDEXCACHE_FP8_MAX
    fp4_max: float = INDEXCACHE_NVFP4_E2M1_MAX
    nvfp4_group_size: int = INDEXCACHE_NVFP4_GROUP_SIZE
    nvfp4_head_dim: int = INDEXCACHE_NVFP4_HEAD_DIM

    def __post_init__(self) -> None:
        if self.quantization not in INDEXCACHE_QUANTIZATION_CHOICES:
            raise ValueError(
                f"Unsupported IndexCache quantization {self.quantization!r}; "
                f"expected one of {INDEXCACHE_QUANTIZATION_CHOICES}."
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be positive; got {self.eps}")

    @property
    def fp8_max_inv(self) -> float:
        return 1.0 / self.fp8_max

    @property
    def fp4_max_inv(self) -> float:
        return 1.0 / self.fp4_max

    @property
    def is_enabled(self) -> bool:
        return self.quantization != INDEXCACHE_QUANT_DISABLED

    @property
    def is_fp8(self) -> bool:
        return self.quantization == INDEXCACHE_QUANT_FP8

    @property
    def is_nvfp4(self) -> bool:
        return self.quantization == INDEXCACHE_QUANT_NVFP4


def resolve_indexcache_quantization(
    *,
    quantization: str | None = None,
    quant_enabled: bool = False,
) -> str:
    """Resolve new explicit method plus legacy boolean alias.

    ``--dsa-indexcache-quant-enabled`` predates method selection and must keep
    meaning fp8 e4m3. An explicit non-disabled method enables quantization even
    if the legacy boolean is false.
    """

    method = quantization or INDEXCACHE_QUANT_DISABLED
    if method not in INDEXCACHE_QUANTIZATION_CHOICES:
        raise ValueError(
            f"Unsupported IndexCache quantization {method!r}; "
            f"expected one of {INDEXCACHE_QUANTIZATION_CHOICES}."
        )
    if quant_enabled and method == INDEXCACHE_QUANT_DISABLED:
        return INDEXCACHE_QUANT_FP8
    return method


def build_indexcache_config(
    *, eps: float = 1e-4, quantization: str = INDEXCACHE_QUANT_FP8
) -> IndexCacheConfig:
    """Public constructor.

    Mirrors the ``build_turboquant_buffers`` shape so callers can hold a
    long-lived config object beside the TurboQuant buffer dataclass.
    """

    return IndexCacheConfig(eps=eps, quantization=quantization)
