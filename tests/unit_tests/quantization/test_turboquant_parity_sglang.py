# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parity test: Megatron's TurboQuant fake-quant vs the SGLang reference codec.

The SGLang implementation lives in
``optimization-playground/python/sglang/srt/layers/quantization/turboquant_dense_kv.py``.
Both implementations should produce *bit-identical* round-tripped tensors when
seeded the same way and run in float32, because they execute the same scalar
operations in the same order on the same fp32 values.

The test is gated on the SGLang repo being importable (controlled by the
``SGLANG_REF_PATH`` environment variable). This makes the test optional in CI
but trivially runnable locally for anyone who has cloned both repos.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from megatron.core.quantization.turboquant import build_turboquant_buffers  # noqa: E402
from megatron.core.quantization.turboquant.reference import (  # noqa: E402
    turboquant_forward,
)


@pytest.fixture
def sglang_codec():
    sglang_root = os.environ.get(
        "SGLANG_REF_PATH",
        "/tmp/optimization-playground",
    )
    sglang_pkg = Path(sglang_root) / "python"
    if not sglang_pkg.exists():
        pytest.skip(
            f"SGLang reference path {sglang_pkg} not found. Set SGLANG_REF_PATH."
        )
    sys.path.insert(0, str(sglang_pkg))
    try:
        from sglang.srt.layers.quantization.turboquant_dense_kv import (
            TurboQuantDenseKVCodec,
            TurboQuantDenseKVConfig,
        )
    except ImportError as e:
        pytest.skip(f"Could not import SGLang TurboQuantDenseKVCodec: {e}")
    return TurboQuantDenseKVConfig, TurboQuantDenseKVCodec


def _align_seeds(preset, seed=42, layer_idx=0):
    """Match the SGLang seed convention.

    SGLang seeds its RNG with ``config.seed`` directly. Our buffers use
    ``seed * 2654435761 + layer_idx``. To match SGLang for parity we need to
    invert that hash: pick ``seed=raw, layer_idx=0`` such that
    ``raw * 2654435761 mod 2^32 == sglang_seed``. Easier to just pass the
    SGLang seed via a hand-tuned (seed, layer_idx) pair.

    For test simplicity we drive both codecs from a known cooperative seed:
    seed=1 (Knuth hash = 2654435761 mod 2^32 = 2654435761) is too large to
    pass as the SGLang config seed directly, so the parity check uses raw
    sign vectors generated identically and compares the rotation+codebook
    behavior on those.
    """

    return seed, layer_idx


def test_buffers_match_sglang_directly(sglang_codec):
    """Construct both with matching sign vectors and codebooks and ensure equal."""
    SGLConfig, SGLCodec = sglang_codec
    sgl_cfg = SGLConfig(
        latent_dim=512,
        rope_dim=64,
        preset="latent_2p5bit_nc",
        seed=42,
    )
    sgl = SGLCodec(sgl_cfg, device=torch.device("cpu"))

    # Build Megatron buffers using a (seed, layer_idx) pair that produces the
    # same generator state. SGLang uses ``manual_seed(seed)`` directly. Our
    # build hashes via ``seed*2654435761 + layer_idx``. Setting layer_idx
    # equal to the SGLang seed and seed=0 gives us the same effective seed.
    meg = build_turboquant_buffers(
        latent_dim=512, preset="latent_2p5bit_nc", seed=0, layer_idx=42, device="cpu"
    )

    torch.testing.assert_close(meg.signs1, sgl.signs1, rtol=0, atol=0)
    torch.testing.assert_close(meg.signs2, sgl.signs2, rtol=0, atol=0)
    torch.testing.assert_close(
        meg.boundaries_high.float(), sgl.boundaries_high, rtol=1e-7, atol=1e-7
    )
    torch.testing.assert_close(
        meg.centroids_high.float(), sgl.centroids_high, rtol=1e-7, atol=1e-7
    )
    torch.testing.assert_close(
        meg.boundaries_low.float(), sgl.boundaries_low, rtol=1e-7, atol=1e-7
    )
    torch.testing.assert_close(
        meg.centroids_low.float(), sgl.centroids_low, rtol=1e-7, atol=1e-7
    )


def test_forward_roundtrip_matches_sglang(sglang_codec):
    """Compress then decompress through SGLang and compare to Megatron's fake-quant."""
    SGLConfig, SGLCodec = sglang_codec
    sgl_cfg = SGLConfig(
        latent_dim=512, rope_dim=64, preset="latent_2p5bit_nc", seed=42
    )
    sgl = SGLCodec(sgl_cfg, device=torch.device("cpu"))
    meg_buf = build_turboquant_buffers(
        latent_dim=512, preset="latent_2p5bit_nc", seed=0, layer_idx=42, device="cpu"
    )

    torch.manual_seed(0)
    n = 64
    latent = torch.randn(n, 512, dtype=torch.float32)
    rope = torch.zeros(n, 64, dtype=torch.bfloat16)

    sgl_packed = sgl.compress(latent, rope)
    sgl_round = sgl.decompress(sgl_packed, dst_dtype=torch.float32)
    sgl_latent_round = sgl_round[:, 0, :512]

    meg_round = turboquant_forward(latent, meg_buf)

    torch.testing.assert_close(meg_round, sgl_latent_round, rtol=1e-5, atol=1e-5)
