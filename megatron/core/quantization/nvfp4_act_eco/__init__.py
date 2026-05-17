# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NVFP4 activation-ECO bias correction.

Public surface:
    apply_nvfp4_act_eco_linear(x, W, config) — fused activation
        fake-quant + Linear with bias-corrected weight gradient.
    Nvfp4ActEcoConfig — dataclass holding block size, FP4/FP8 maxes,
        and a small eps for the per-block scale clamp.
    Nvfp4ActEcoLinearFn — the underlying torch.autograd.Function.

The math, motivation, and composition with the rest of the FlashOptim
ECO + TurboQuant + IndexCache stack are in
``docs/optimizer/nvfp4_act_eco.md``.
"""

from megatron.core.quantization.nvfp4_act_eco.autograd import (
    Nvfp4ActEcoLinearFn,
    apply_nvfp4_act_eco_linear,
)
from megatron.core.quantization.nvfp4_act_eco.codec import (
    FP8_E4M3_MAX,
    NVFP4_BLOCK_SIZE,
    NVFP4_E2M1_MAX,
    Nvfp4ActEcoConfig,
    build_nvfp4_act_eco_config,
)
from megatron.core.quantization.nvfp4_act_eco.te_hook import (
    install_act_eco_on_te_grouped_linear,
    install_act_eco_on_te_linear,
    is_te_available,
)

__all__ = [
    "FP8_E4M3_MAX",
    "NVFP4_BLOCK_SIZE",
    "NVFP4_E2M1_MAX",
    "Nvfp4ActEcoConfig",
    "Nvfp4ActEcoLinearFn",
    "apply_nvfp4_act_eco_linear",
    "build_nvfp4_act_eco_config",
    "install_act_eco_on_te_grouped_linear",
    "install_act_eco_on_te_linear",
    "is_te_available",
]
