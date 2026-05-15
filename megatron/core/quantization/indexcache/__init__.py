# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""IndexCache fake-quant for the DSA indexer's K tensor.

Public surface:
    apply_indexcache_kv(k, config) — selected fake-quant per-token, with
        analytic STE-based backward.
    IndexCacheConfig — dataclass holding the method and quantization constants.
    IndexCacheKVFn — torch.autograd.Function used by ``apply_indexcache_kv``.

This is a direct port of the forward kernel in
``optimization-playground/python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh``
(plus its Triton fallback at
``optimization-playground/python/sglang/srt/layers/attention/nsa/triton_kernel.py``).
The backward is new — the SGLang reference is forward-only.
"""

from megatron.core.quantization.indexcache.codec import (
    INDEXCACHE_FP8_MAX,
    INDEXCACHE_NVFP4_E2M1_MAX,
    INDEXCACHE_NVFP4_GROUP_SIZE,
    INDEXCACHE_NVFP4_HEAD_DIM,
    INDEXCACHE_QUANT_DISABLED,
    INDEXCACHE_QUANT_FP8,
    INDEXCACHE_QUANT_NVFP4,
    INDEXCACHE_QUANTIZATION_CHOICES,
    IndexCacheConfig,
    build_indexcache_config,
    resolve_indexcache_quantization,
)
from megatron.core.quantization.indexcache.autograd import (
    IndexCacheKVFn,
    apply_indexcache_kv,
)
from megatron.core.quantization.indexcache.hisa import (
    IndexCacheHISAConfig,
    hisa_block_topk_counts,
    indexcache_hisa_topk,
)

__all__ = [
    "INDEXCACHE_FP8_MAX",
    "INDEXCACHE_NVFP4_E2M1_MAX",
    "INDEXCACHE_NVFP4_GROUP_SIZE",
    "INDEXCACHE_NVFP4_HEAD_DIM",
    "INDEXCACHE_QUANT_DISABLED",
    "INDEXCACHE_QUANT_FP8",
    "INDEXCACHE_QUANT_NVFP4",
    "INDEXCACHE_QUANTIZATION_CHOICES",
    "IndexCacheConfig",
    "IndexCacheHISAConfig",
    "IndexCacheKVFn",
    "apply_indexcache_kv",
    "build_indexcache_config",
    "hisa_block_topk_counts",
    "indexcache_hisa_topk",
    "resolve_indexcache_quantization",
]
