# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility constructors for the IndexCache HISA config surface."""

from __future__ import annotations

from typing import Any, Mapping

from megatron.core.quantization.indexcache import IndexCacheHISAConfig


HISA_INDEXER_DEFAULT_BLOCK_SIZE = 128
HISA_INDEXER_DEFAULT_COMPRESSION_RATIO = 4.0
HISA_INDEXER_DEFAULT_TOPK_TOKENS = 2048

# Execution mode picks the kernel path.
#   ``optimized``  : CUDA backward + CUDA forward when Blackwell+ is available,
#                    auto-fallback to ``reference`` otherwise.
#   ``reference``  : pure-PyTorch FP32 path used as the autograd-parity oracle.
#   ``compute_only``: forward only (selector indices) — used for inference-side
#                     consumers that do not need a backward.
HISA_INDEXER_EXECUTION_MODES = ("optimized", "reference", "compute_only")


def parse_hf_hisa_block(block: Mapping[str, Any] | None) -> IndexCacheHISAConfig:
    """Parse ``model.config.json -> quantization_config.indexer_quantization.hisa``.

    Returns the default-disabled config when ``block`` is missing or empty.
    """

    if not block:
        return IndexCacheHISAConfig()
    kwargs: dict[str, Any] = {}
    if "enabled" in block:
        kwargs["enabled"] = bool(block["enabled"])
    if "block_size" in block:
        kwargs["block_size"] = int(block["block_size"])
    if "compression_ratio" in block:
        kwargs["compression_ratio"] = float(block["compression_ratio"])
    if "topk_tokens" in block:
        kwargs["topk_tokens"] = int(block["topk_tokens"])
    if "execution_mode" in block:
        kwargs["execution_mode"] = str(block["execution_mode"])
    if "fallback_to_dense_if_short" in block:
        kwargs["fallback_to_dense_if_short"] = bool(
            block["fallback_to_dense_if_short"]
        )
    if "forced_boundary_blocks" in block:
        kwargs["forced_boundary_blocks"] = tuple(block["forced_boundary_blocks"])
    return IndexCacheHISAConfig(**kwargs)


def resolve_hisa_from_hf_config(
    hf_quant_config: Mapping[str, Any] | None,
) -> IndexCacheHISAConfig:
    """Extract the HISA sub-block from a HuggingFace ``quantization_config``."""

    if not hf_quant_config:
        return IndexCacheHISAConfig()
    indexer_block = hf_quant_config.get("indexer_quantization")
    if not indexer_block:
        return IndexCacheHISAConfig()
    return parse_hf_hisa_block(indexer_block.get("hisa"))


def build_hisa_config(
    *,
    enabled: bool = False,
    block_size: int = HISA_INDEXER_DEFAULT_BLOCK_SIZE,
    compression_ratio: float = HISA_INDEXER_DEFAULT_COMPRESSION_RATIO,
    topk_tokens: int = HISA_INDEXER_DEFAULT_TOPK_TOKENS,
    execution_mode: str = "optimized",
) -> IndexCacheHISAConfig:
    """Public constructor for the common subset of fields."""

    return IndexCacheHISAConfig(
        enabled=enabled,
        block_size=block_size,
        compression_ratio=compression_ratio,
        topk_tokens=topk_tokens,
        execution_mode=execution_mode,
    )
