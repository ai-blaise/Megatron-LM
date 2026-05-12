# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen buffers and EDEN2-16 codebook for the 2-bit HIGGS dense MLA KV.

This is a bit-for-bit port of the SGLang reference at
``optimization-playground/python/sglang/srt/layers/quantization/higgs_dense_2bit_kv.py``
(merged on main as ``2e2f51717``). The codebook is the public AquaKV
``EDEN2-16`` grid (``arXiv:2501.19392``, Pletka et al., *Cache Me If You
Must*); the rotation is a single orthonormal block-Hadamard of width
``kv_lora_rank=512``; the per-token block scale is fp16. Slot layout
matches the SGLang store kernel exactly:

    [packed 4-bit pair indices: 128 B] [fp16 scale: 2 B] [rope: 128 B] -> 258 B

The frozen state held by ``HiggsBuffers`` is the codebook itself and its
per-codeword squared norm. Unlike TurboQuant there is no per-layer random
sign vector and no Lloyd-Max construction --- the EDEN2-16 lattice is a
public constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple

import torch


HIGGS_LATENT_DIM = 512
HIGGS_PAIR_DIM = 2
HIGGS_CODEBOOK_SIZE = 16
HIGGS_BITS_PER_INDEX = 4  # log2(16); two scalars per index => 2 bits / scalar
HIGGS_NUM_PAIRS = HIGGS_LATENT_DIM // HIGGS_PAIR_DIM   # 256
HIGGS_PACKED_BYTES = HIGGS_NUM_PAIRS // 2              # 128 bytes
HIGGS_NORM_BYTES = 2                                   # fp16 per-token scale
HIGGS_ROPE_DIM = 64                                    # qk_pos_emb_head_dim
HIGGS_ROPE_BYTES = HIGGS_ROPE_DIM * 2                  # bf16 passthrough
HIGGS_SLOT_BYTES = HIGGS_PACKED_BYTES + HIGGS_NORM_BYTES + HIGGS_ROPE_BYTES  # 258


# EDEN2-16 lattice from the AquaKV repository (16 entries, 2-D codewords).
# Source: optimization-playground HEAD ``2e2f51717``,
# python/sglang/srt/layers/quantization/higgs_dense_2bit_kv.py:HIGGS_EDEN2_16.
HIGGS_EDEN2_16: Tuple[Tuple[float, float], ...] = (
    (-0.8996632695198059, -1.6360418796539307),
    (-0.961183488368988, 1.5999565124511719),
    (-1.882026195526123, 0.678778350353241),
    (0.36300793290138245, -1.9667866230010986),
    (-0.6814072728157043, -0.576818585395813),
    (0.7270012497901917, 0.6186859607696533),
    (0.3359416127204895, 1.8371193408966064),
    (1.859930396080017, 0.036668598651885986),
    (0.17208248376846313, -0.9401724338531494),
    (-1.7599700689315796, -0.6244229674339294),
    (-0.8993809223175049, 0.32267823815345764),
    (0.839488685131073, -0.3017036020755768),
    (1.5314953327178955, 1.2942044734954834),
    (-0.0011779458727687597, 0.00022069070837460458),
    (1.4274526834487915, -1.207889199256897),
    (-0.16123905777931213, 0.8787511587142944),
)


HIGGS_PRESETS: Mapping[str, Mapping[str, object]] = {
    "dense_2bit": {
        "latent_dim": HIGGS_LATENT_DIM,
        "codebook_size": HIGGS_CODEBOOK_SIZE,
        "pair_dim": HIGGS_PAIR_DIM,
        "bits_per_scalar": 2.0,
    },
}


def _check_dim(dim: int) -> None:
    if dim != HIGGS_LATENT_DIM:
        raise ValueError(
            f"2-bit HIGGS dense KV is fixed to latent_dim={HIGGS_LATENT_DIM}; got {dim}."
        )
    if dim % HIGGS_PAIR_DIM:
        raise ValueError(
            f"latent_dim ({dim}) must be a multiple of HIGGS pair dim ({HIGGS_PAIR_DIM})."
        )


@dataclass(frozen=True)
class HiggsBuffers:
    """Frozen, layer-scoped tensors that drive the HIGGS fake-quant.

    All entries live on the same device and float32 dtype. They are produced
    once per ``(layer_idx, latent_dim, preset)`` --- since the EDEN2-16
    codebook does not depend on layer or seed, every layer ends up with the
    same codebook tensor, but ``HiggsBuffers`` mirrors TurboQuant's per-layer
    structure so the integration site can register and refresh buffers in
    exactly the same way. Buffers must be replicated (not sharded) across
    TP/SP/CP/EP ranks; ``build_higgs_buffers`` is deterministic so every rank
    constructs identical state without collective ops.
    """

    latent_dim: int
    pair_dim: int
    codebook_size: int
    bits_per_scalar: float
    codebook: torch.Tensor          # (16, 2) float32 EDEN2-16
    codebook_norm_sq: torch.Tensor  # (16,) float32 ||G_i||^2

    @property
    def num_pairs(self) -> int:
        return self.latent_dim // self.pair_dim

    @property
    def packed_bytes(self) -> int:
        return self.num_pairs // 2

    @property
    def latent_bytes(self) -> int:
        return self.packed_bytes + HIGGS_NORM_BYTES


def build_higgs_buffers(
    *,
    latent_dim: int = HIGGS_LATENT_DIM,
    preset: str = "dense_2bit",
    layer_idx: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> HiggsBuffers:
    """Construct frozen buffers for a single MLA layer.

    Args:
      latent_dim: must equal ``HIGGS_LATENT_DIM=512`` (the only supported
        configuration for 2-bit HIGGS dense MLA KV).
      preset: codebook preset key; only ``"dense_2bit"`` is currently
        supported.
      layer_idx: included for API symmetry with TurboQuant; the EDEN2-16
        codebook is layer-invariant so this argument has no effect on the
        produced buffers.
      device: device the codebook tensors are materialised on.
      dtype: float dtype used for the codebook tensors.
    """

    del layer_idx  # API symmetry only; EDEN2-16 has no per-layer randomness.

    if preset not in HIGGS_PRESETS:
        valid = ", ".join(sorted(HIGGS_PRESETS))
        raise ValueError(f"Unknown HIGGS preset {preset!r}; choices: {valid}.")
    _check_dim(latent_dim)

    device = torch.device(device)
    codebook = torch.tensor(HIGGS_EDEN2_16, dtype=dtype, device=device).contiguous()
    codebook_norm_sq = (codebook * codebook).sum(dim=-1).to(dtype).contiguous()

    return HiggsBuffers(
        latent_dim=latent_dim,
        pair_dim=HIGGS_PAIR_DIM,
        codebook_size=HIGGS_CODEBOOK_SIZE,
        bits_per_scalar=float(HIGGS_PRESETS[preset]["bits_per_scalar"]),
        codebook=codebook,
        codebook_norm_sq=codebook_norm_sq,
    )


def pack_higgs_2bit_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codebook indices, two per byte.

    Args:
      indices: ``(..., num_pairs)`` ``uint8`` tensor with values in [0, 15].

    Returns:
      ``(..., num_pairs // 2)`` ``uint8`` packed tensor.
    """

    indices = indices.to(torch.uint8)
    if indices.shape[-1] % 2:
        indices = torch.nn.functional.pad(indices, (0, 1))
    lo = indices[..., 0::2] & 0x0F
    hi = (indices[..., 1::2] & 0x0F) << 4
    return (lo | hi).contiguous()


def unpack_higgs_2bit_indices(
    packed: torch.Tensor, num_pairs: int
) -> torch.Tensor:
    """Inverse of :func:`pack_higgs_2bit_indices`."""

    packed = packed.to(torch.uint8)
    needed = (num_pairs + 1) // 2
    p = packed[..., :needed]
    out = torch.empty(*p.shape[:-1], needed * 2, dtype=torch.uint8, device=p.device)
    out[..., 0::2] = p & 0x0F
    out[..., 1::2] = (p >> 4) & 0x0F
    return out[..., :num_pairs].contiguous()


# Pre-derived for convenience: 1 / sqrt(latent_dim), used as the orthonormal
# Hadamard prefactor. Matches the kernel constant exactly.
HIGGS_INV_SQRT_LATENT_DIM = 1.0 / math.sqrt(HIGGS_LATENT_DIM)
