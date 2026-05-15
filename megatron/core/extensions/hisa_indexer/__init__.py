# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""HISA 4:1 hierarchical indexer helpers for NSA training.

This module ports the forward kernel published in
``optimization-playground/python/sglang/jit_kernel/csrc/nsa/nvfp4_indexer_quant.cuh``
(SHA ``07ead85c5``) and adds a training-time backward. The selector pipeline is:

  1. mean_pool: ``k_block_b = mean({k_s : s in block b})``
  2. block_score: ``J_{t,b} = sum_h w_{t,h} * ReLU(q_{t,h} . k_block_b)``
  3. block_topk: top-m blocks per query (with forced first + last valid
     boundary blocks, matching the TileLang HISA reference).
  4. candidate_dequant: NVFP4 -> FP32 on the selected blocks.
  5. candidate_score: ``I_{t,s} = sum_h w_{t,h} * ReLU(q_{t,h} . k_s)`` for
     ``s`` in the selected blocks.
  6. candidate_topk: top-k tokens within the candidate pool.
  7. map/store: pure permutation into per-row index tensors.

See ``docs/indexcache/hisa_indexer_backward.md`` for the math, config contract,
and Rule-6 reference notes.
"""

from megatron.core.extensions.hisa_indexer.config import (
    HISA_INDEXER_DEFAULT_BLOCK_SIZE,
    HISA_INDEXER_DEFAULT_COMPRESSION_RATIO,
    HISA_INDEXER_DEFAULT_TOPK_TOKENS,
    HISA_INDEXER_EXECUTION_MODES,
    build_hisa_config,
    parse_hf_hisa_block,
    resolve_hisa_from_hf_config,
)
from megatron.core.extensions.hisa_indexer.autograd import (
    apply_hisa_score_backward,
    hisa_selector_forward_and_save,
)
from megatron.core.quantization.indexcache import IndexCacheHISAConfig

__all__ = [
    "HISA_INDEXER_DEFAULT_BLOCK_SIZE",
    "HISA_INDEXER_DEFAULT_COMPRESSION_RATIO",
    "HISA_INDEXER_DEFAULT_TOPK_TOKENS",
    "HISA_INDEXER_EXECUTION_MODES",
    "IndexCacheHISAConfig",
    "apply_hisa_score_backward",
    "build_hisa_config",
    "hisa_selector_forward_and_save",
    "parse_hf_hisa_block",
    "resolve_hisa_from_hf_config",
]
