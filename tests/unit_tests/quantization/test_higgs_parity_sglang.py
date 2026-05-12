# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parity: Megatron's HIGGS fake-quant vs the SGLang reference codec.

The SGLang implementation lives in
``optimization-playground/python/sglang/srt/layers/quantization/higgs_dense_2bit_kv.py``
(main HEAD ``2e2f51717``). Both implementations should produce *bit-identical*
packed byte buffers when run on the same fp32 inputs, because the algorithm
is purely a sequence of public constants (EDEN2-16 codebook + orthonormal
FWHT + fp16 scale) and uses no per-layer randomness.

The test is gated on the SGLang repo being importable (controlled by the
``SGLANG_REF_PATH`` environment variable). This makes the test optional in
CI but trivially runnable locally for anyone who has cloned both repos.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from megatron.core.quantization.higgs import (  # noqa: E402
    HIGGS_LATENT_DIM,
    HIGGS_SLOT_BYTES,
    build_higgs_buffers,
)
from megatron.core.quantization.higgs.reference import (  # noqa: E402
    reference_compress,
    reference_decompress,
)


@pytest.fixture
def sglang_codec():
    """Load the SGLang HIGGS reference codec module directly.

    Importing via ``sglang.srt.layers.quantization.higgs_dense_2bit_kv``
    pulls a heavy transitive import chain (orjson, msgspec, etc.) that we
    don't need for the codec module itself --- it has zero external deps
    beyond torch. We side-load the module from its file path via
    ``importlib.util`` so the parity test can run in a minimal environment.
    """

    import importlib.util

    sglang_root = os.environ.get(
        "SGLANG_REF_PATH",
        "/tmp/optimization-playground",
    )
    codec_path = (
        Path(sglang_root)
        / "python"
        / "sglang"
        / "srt"
        / "layers"
        / "quantization"
        / "higgs_dense_2bit_kv.py"
    )
    if not codec_path.exists():
        pytest.skip(
            f"SGLang HIGGS codec not found at {codec_path}. Set SGLANG_REF_PATH."
        )
    module_name = "sglang_higgs_dense_2bit_kv_ref"
    spec = importlib.util.spec_from_file_location(module_name, codec_path)
    module = importlib.util.module_from_spec(spec)
    # ``@dataclass`` resolves type-annotation strings against
    # ``sys.modules[cls.__module__]``; the codec defines
    # ``HiggsDense2BitConfig`` with type-annotated fields so we must
    # register the module before exec_module fires the dataclass decorator.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        pytest.skip(f"Could not load SGLang HIGGS codec module: {e}")
    return module.HiggsDense2BitConfig, module.HiggsDense2BitCodec


def test_codebook_matches_sglang(sglang_codec):
    """Megatron buffers and SGLang codec carry identical codebook tensors."""

    SGLConfig, SGLCodec = sglang_codec
    cfg = SGLConfig(latent_dim=HIGGS_LATENT_DIM, rope_dim=64)
    sgl = SGLCodec(cfg, device=torch.device("cpu"))
    meg = build_higgs_buffers(
        latent_dim=HIGGS_LATENT_DIM, preset="dense_2bit", device="cpu",
        dtype=torch.float32,
    )

    torch.testing.assert_close(meg.codebook, sgl.codebook, rtol=0, atol=0)
    torch.testing.assert_close(
        meg.codebook_norm_sq, sgl.codebook_norm_sq, rtol=0, atol=0
    )


def test_packed_bytes_match_sglang(sglang_codec):
    """Compress on both implementations and compare packed slot bytes exactly.

    The packed byte layout is determined by deterministic fp32 ops on public
    constants; both implementations must produce identical bytes.
    """

    SGLConfig, SGLCodec = sglang_codec
    cfg = SGLConfig(latent_dim=HIGGS_LATENT_DIM, rope_dim=64)
    sgl = SGLCodec(cfg, device=torch.device("cpu"))
    meg_buf = build_higgs_buffers(
        latent_dim=HIGGS_LATENT_DIM, preset="dense_2bit", device="cpu",
        dtype=torch.float32,
    )

    torch.manual_seed(0)
    n = 16
    latent = torch.randn(n, 1, HIGGS_LATENT_DIM, dtype=torch.bfloat16)
    rope = torch.randn(n, 1, 64, dtype=torch.bfloat16)

    sgl_packed = sgl.compress(latent, rope)
    meg_packed = reference_compress(latent, rope, meg_buf)

    assert sgl_packed.shape == (n, 1, HIGGS_SLOT_BYTES)
    assert meg_packed.shape == sgl_packed.shape

    # Compare byte-for-byte. Both implementations should produce the same
    # codebook indices (deterministic argmax), the same fp16 scale bytes,
    # and the same bf16 rope passthrough. If a sign-of-zero or rounding
    # ambiguity ever causes a one-bit drift on the scale fp16 encoding,
    # the assert below would flag it and we'd document the relaxation.
    assert torch.equal(meg_packed, sgl_packed), (
        f"packed bytes differ: meg vs sgl mismatch in "
        f"{(meg_packed != sgl_packed).sum().item()} / {meg_packed.numel()} bytes"
    )


def test_round_trip_matches_sglang(sglang_codec):
    """Decompressing through SGLang and through Megatron returns the same tensor."""

    SGLConfig, SGLCodec = sglang_codec
    cfg = SGLConfig(latent_dim=HIGGS_LATENT_DIM, rope_dim=64)
    sgl = SGLCodec(cfg, device=torch.device("cpu"))
    meg_buf = build_higgs_buffers(
        latent_dim=HIGGS_LATENT_DIM, preset="dense_2bit", device="cpu",
        dtype=torch.float32,
    )

    torch.manual_seed(1)
    n = 16
    latent = torch.randn(n, 1, HIGGS_LATENT_DIM, dtype=torch.bfloat16)
    rope = torch.randn(n, 1, 64, dtype=torch.bfloat16)

    sgl_packed = sgl.compress(latent, rope)
    sgl_round = sgl.decompress(sgl_packed, dst_dtype=torch.float32)
    meg_round = reference_decompress(sgl_packed, meg_buf, torch.float32)

    torch.testing.assert_close(meg_round, sgl_round, rtol=1e-6, atol=1e-6)
