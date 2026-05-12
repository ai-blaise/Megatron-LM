# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""2-bit HIGGS dense MLA-latent KV fake-quant.

Implements the HIGGS scheme of Pletka et al. (``arXiv:2501.19392``) on the
DeepSeek V3.2-style 512-dim MLA latent KV. Uses a single 512-wide
orthonormal block-Hadamard rotation, one fp16 per-token block scale, and a
4-bit index per pair into the public AquaKV EDEN2-16 codebook (2 bits per
scalar, 128 B / token + 2 B scale + 128 B rope passthrough = 258 B / slot).
Replaces TurboQuant on the dense MLA KV when the user opts in; the two
paths are mutually exclusive at config-validate time.

Public surface mirrors ``megatron.core.quantization.turboquant``:

    apply_higgs_dense_2bit_kv(latent, buffers)
    HiggsBuffers
    build_higgs_buffers(latent_dim, preset, layer_idx, device, dtype)
    HIGGS_PRESETS
"""

from megatron.core.quantization.higgs.codec import (
    HIGGS_BITS_PER_INDEX,
    HIGGS_CODEBOOK_SIZE,
    HIGGS_EDEN2_16,
    HIGGS_INV_SQRT_LATENT_DIM,
    HIGGS_LATENT_DIM,
    HIGGS_NORM_BYTES,
    HIGGS_NUM_PAIRS,
    HIGGS_PACKED_BYTES,
    HIGGS_PAIR_DIM,
    HIGGS_PRESETS,
    HIGGS_ROPE_BYTES,
    HIGGS_ROPE_DIM,
    HIGGS_SLOT_BYTES,
    HiggsBuffers,
    build_higgs_buffers,
    pack_higgs_2bit_indices,
    unpack_higgs_2bit_indices,
)
from megatron.core.quantization.higgs.autograd import (
    HiggsDenseKVFn,
    apply_higgs_dense_2bit_kv,
)

__all__ = [
    "HIGGS_BITS_PER_INDEX",
    "HIGGS_CODEBOOK_SIZE",
    "HIGGS_EDEN2_16",
    "HIGGS_INV_SQRT_LATENT_DIM",
    "HIGGS_LATENT_DIM",
    "HIGGS_NORM_BYTES",
    "HIGGS_NUM_PAIRS",
    "HIGGS_PACKED_BYTES",
    "HIGGS_PAIR_DIM",
    "HIGGS_PRESETS",
    "HIGGS_ROPE_BYTES",
    "HIGGS_ROPE_DIM",
    "HIGGS_SLOT_BYTES",
    "HiggsBuffers",
    "HiggsDenseKVFn",
    "apply_higgs_dense_2bit_kv",
    "build_higgs_buffers",
    "pack_higgs_2bit_indices",
    "unpack_higgs_2bit_indices",
]
