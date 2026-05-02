# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""IndexCache fp8 fake-quant for the DSA indexer's K tensor.

Public surface:
    apply_indexcache_kv(k, config) — fp8 e4m3 fake-quant per-token, with
        analytic STE-based backward.
    IndexCacheConfig — dataclass holding eps and the (forthcoming) fp8 max.
    IndexCacheKVFn — torch.autograd.Function used by ``apply_indexcache_kv``.

This is a direct port of the forward kernel in
``optimization-playground/python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh``
(plus its Triton fallback at
``optimization-playground/python/sglang/srt/layers/attention/nsa/triton_kernel.py``).
The backward is new — the SGLang reference is forward-only.
"""

from megatron.core.quantization.indexcache.codec import (
    INDEXCACHE_FP8_MAX,
    IndexCacheConfig,
    build_indexcache_config,
)
from megatron.core.quantization.indexcache.autograd import (
    IndexCacheKVFn,
    apply_indexcache_kv,
)

__all__ = [
    "INDEXCACHE_FP8_MAX",
    "IndexCacheConfig",
    "IndexCacheKVFn",
    "apply_indexcache_kv",
    "build_indexcache_config",
]
