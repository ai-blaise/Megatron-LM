# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""IndexCache config — fp8 e4m3 quantization parameters.

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


@dataclass(frozen=True)
class IndexCacheConfig:
    """Stateless config for fp8 e4m3 fake-quant on the indexer K tensor."""

    eps: float = 1e-4
    fp8_max: float = INDEXCACHE_FP8_MAX

    @property
    def fp8_max_inv(self) -> float:
        return 1.0 / self.fp8_max


def build_indexcache_config(*, eps: float = 1e-4) -> IndexCacheConfig:
    """Public constructor.

    Mirrors the ``build_turboquant_buffers`` shape so callers can hold a
    long-lived config object beside the TurboQuant buffer dataclass.
    """

    return IndexCacheConfig(eps=eps)
