# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""2.5-bit TurboQuant fake-quant for the dense MLA latent KV.

Public surface:
    apply_turboquant_kv(latent, buffers, *, ste="hard")
    TurboQuantBuffers
    build_turboquant_buffers(latent_dim, preset, seed, device, dtype)
    TURBOQUANT_PRESETS
"""

from megatron.core.quantization.turboquant.codec import (
    TURBOQUANT_2P5_GROUP_SIZE,
    TURBOQUANT_2P5_HIGH_CHANNELS,
    TURBOQUANT_PRESETS,
    TurboQuantBuffers,
    build_turboquant_buffers,
)
from megatron.core.quantization.turboquant.autograd import (
    TurboQuantKVFn,
    apply_turboquant_kv,
)

__all__ = [
    "TURBOQUANT_2P5_GROUP_SIZE",
    "TURBOQUANT_2P5_HIGH_CHANNELS",
    "TURBOQUANT_PRESETS",
    "TurboQuantBuffers",
    "TurboQuantKVFn",
    "apply_turboquant_kv",
    "build_turboquant_buffers",
]
