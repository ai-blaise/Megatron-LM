# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen buffers and Lloyd-Max codebooks for TurboQuant dense KV.

The constants and codebook construction here are bit-for-bit faithful ports of
the SGLang reference at
``optimization-playground/python/sglang/srt/layers/quantization/turboquant_dense_kv.py``.
We reproduce the seed-derived sign vectors and the Gaussian Lloyd-Max iteration
verbatim so that a tensor compressed by Megatron's training-time fake-quant is
indistinguishable from one round-tripped through the SGLang inference codec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch


TURBOQUANT_2P5_GROUP_SIZE = 128
TURBOQUANT_2P5_HIGH_CHANNELS = 32


TURBOQUANT_PRESETS: Mapping[str, Mapping[str, object]] = {
    "latent_2p5bit_nc": {"bits": 2.5, "norm_correction": True},
    "latent_4bit_nc": {"bits": 4, "norm_correction": True},
    "latent_k3_nc": {"bits": 3, "norm_correction": True},
    "latent_k8": {"bits": 8, "norm_correction": False},
}


def _is_2p5_bits(bits: float) -> bool:
    return math.isclose(float(bits), 2.5)


def _check_2p5_dim(dim: int) -> None:
    if dim % TURBOQUANT_2P5_GROUP_SIZE != 0:
        raise ValueError(
            "2.5-bit TurboQuant requires the latent dimension to be a multiple of "
            f"{TURBOQUANT_2P5_GROUP_SIZE}; got {dim}."
        )


def _lloyd_max_normal(
    bits: int, dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (centroids, boundaries) for the post-rotation Gaussian.

    After the random sign flip + Walsh-Hadamard rotation, each coordinate of a
    unit-norm vector is approximately N(0, 1/dim). We design a Lloyd-Max
    quantizer for that distribution by alternating between centroid and
    boundary updates for 100 iterations, matching the SGLang reference.
    """

    n = 1 << bits
    sigma = 1.0 / math.sqrt(dim)
    dtype = torch.float32
    q = torch.arange(1, n, dtype=dtype, device=device) / n
    boundaries = math.sqrt(2.0) * sigma * torch.erfinv(2.0 * q - 1.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    for _ in range(100):
        lo = torch.cat(
            (torch.tensor([-torch.inf], dtype=dtype, device=device), boundaries)
        )
        hi = torch.cat(
            (boundaries, torch.tensor([torch.inf], dtype=dtype, device=device))
        )
        lo_z = lo / sigma
        hi_z = hi / sigma
        lo_pdf = torch.where(
            torch.isfinite(lo_z), torch.exp(-0.5 * lo_z * lo_z) * inv_sqrt_2pi, 0
        )
        hi_pdf = torch.where(
            torch.isfinite(hi_z), torch.exp(-0.5 * hi_z * hi_z) * inv_sqrt_2pi, 0
        )
        lo_cdf = torch.where(
            torch.isfinite(lo_z), 0.5 * (1.0 + torch.erf(lo_z / math.sqrt(2.0))), 0
        )
        hi_cdf = torch.where(
            torch.isfinite(hi_z), 0.5 * (1.0 + torch.erf(hi_z / math.sqrt(2.0))), 1
        )
        centroids = sigma * (lo_pdf - hi_pdf) / (hi_cdf - lo_cdf).clamp_min(1e-12)
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])

    return centroids.contiguous(), boundaries.contiguous()


@dataclass(frozen=True)
class TurboQuantBuffers:
    """Frozen, layer-scoped tensors that drive the fake-quant.

    All entries live on the same device and float32 dtype. They are produced
    once per ``(layer_idx, latent_dim, preset)`` and never updated during
    training. Buffers must be replicated (not sharded) across TP/SP/CP/EP
    ranks; ``build_turboquant_buffers`` is deterministic for a given
    ``(seed, layer_idx, latent_dim, preset)`` so every rank constructs the
    same tensors locally without needing collective ops.
    """

    latent_dim: int
    bits: float
    norm_correction: bool
    signs1: torch.Tensor
    signs2: torch.Tensor
    boundaries_high: torch.Tensor
    boundaries_low: torch.Tensor
    centroids_high: torch.Tensor
    centroids_low: torch.Tensor

    @property
    def is_2p5bit(self) -> bool:
        return _is_2p5_bits(self.bits)


def build_turboquant_buffers(
    *,
    latent_dim: int,
    preset: str = "latent_2p5bit_nc",
    seed: int = 0,
    layer_idx: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TurboQuantBuffers:
    """Construct frozen buffers for a single MLA layer.

    The seed entering the deterministic generator is ``seed * 2654435761 +
    layer_idx`` (a Knuth multiplicative hash) so two layers do not share sign
    vectors, while a fixed ``(seed, layer_idx)`` pair is reproducible across
    runs and across ranks.
    """

    if preset not in TURBOQUANT_PRESETS:
        valid = ", ".join(sorted(TURBOQUANT_PRESETS))
        raise ValueError(f"Unknown TurboQuant preset {preset!r}; choices: {valid}.")
    bits = float(TURBOQUANT_PRESETS[preset]["bits"])
    norm_correction = bool(TURBOQUANT_PRESETS[preset]["norm_correction"])

    if _is_2p5_bits(bits):
        _check_2p5_dim(latent_dim)

    device = torch.device(device)
    layer_seed = (seed * 2654435761 + layer_idx) & 0xFFFFFFFF
    generator = torch.Generator(device="cpu").manual_seed(layer_seed)

    signs1_raw = torch.randint(
        0, 2, (latent_dim,), generator=generator, dtype=torch.int8
    )
    signs2_raw = torch.randint(
        0, 2, (latent_dim,), generator=generator, dtype=torch.int8
    )
    signs1 = (signs1_raw.to(device=device, dtype=dtype) * 2 - 1).contiguous()
    signs2 = (signs2_raw.to(device=device, dtype=dtype) * 2 - 1).contiguous()

    if _is_2p5_bits(bits):
        centroids_high, boundaries_high = _lloyd_max_normal(3, latent_dim, device)
        centroids_low, boundaries_low = _lloyd_max_normal(2, latent_dim, device)
    elif bits < 8:
        centroids_high, boundaries_high = _lloyd_max_normal(int(bits), latent_dim, device)
        centroids_low = centroids_high
        boundaries_low = boundaries_high
    else:
        empty = torch.empty(0, dtype=dtype, device=device)
        centroids_high = boundaries_high = centroids_low = boundaries_low = empty

    return TurboQuantBuffers(
        latent_dim=latent_dim,
        bits=bits,
        norm_correction=norm_correction,
        signs1=signs1,
        signs2=signs2,
        boundaries_high=boundaries_high.to(dtype),
        boundaries_low=boundaries_low.to(dtype),
        centroids_high=centroids_high.to(dtype),
        centroids_low=centroids_low.to(dtype),
    )
