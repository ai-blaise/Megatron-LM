# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""SpinQuant helpers for rotation-aware low-bit validation.

This module starts with the runtime pieces that can be validated inside
Megatron training. The token-wise K/V quantizer mirrors SpinQuant's per-token
activation/KV contract and uses Transformer Engine NVFP4 pack/dequant on CUDA,
with a straight-through fake-quant fallback for non-CUDA or non-4-bit modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import transformer_engine.pytorch  # noqa: F401
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    HAVE_TE_NVFP4 = True
except (ImportError, ModuleNotFoundError):
    NVFP4Quantizer = None
    HAVE_TE_NVFP4 = False


@dataclass
class SpinQuantRotationSet:
    """Container for SpinQuant's learned/global rotations.

    R1 is the model-wide hidden-state rotation. R2 is keyed by layer index and
    applies to per-head attention value/KV-sensitive paths.
    """

    r1: torch.Tensor | None
    r2_by_layer: dict[int, torch.Tensor]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def hadamard_matrix(size: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Construct a normalized Hadamard matrix for power-of-two sizes."""

    if not _is_power_of_two(size):
        raise ValueError(f"Hadamard rotation requires power-of-two size, got {size}.")
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < size:
        h = torch.cat(
            (
                torch.cat((h, h), dim=1),
                torch.cat((h, -h), dim=1),
            ),
            dim=0,
        )
    return h / (size**0.5)


def random_hadamard_rotation(
    size: int,
    *,
    device=None,
    dtype=torch.float32,
    seed: int = 1234,
) -> torch.Tensor:
    """Generate SpinQuant-style random signed Hadamard initialization."""

    generator = torch.Generator(device=device if device is not None else "cpu")
    generator.manual_seed(seed)
    h = hadamard_matrix(size, device=device, dtype=dtype)
    signs = torch.randint(
        0,
        2,
        (size,),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    signs = signs.to(dtype).mul_(2).sub_(1)
    perm = torch.randperm(size, generator=generator, device=device)
    return h[:, perm] * signs


def load_rotation_set(path: str | Path, *, map_location="cpu") -> SpinQuantRotationSet:
    """Load a SpinQuant rotation checkpoint.

    The facebookresearch/SpinQuant repo saves keys named ``R1`` and
    ``model.layers.{i}.self_attn.R2``. This loader also accepts shorter keys
    containing ``R1`` or ``R2`` so conversion scripts can normalize later.
    """

    state = torch.load(Path(path), map_location=map_location)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError(f"Expected a dict rotation checkpoint at {path}.")

    r1 = None
    r2_by_layer: dict[int, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.endswith("R1") or key == "R1" or "R1.weight" in key:
            r1 = value
            continue
        if "R2" in key:
            layer_idx = None
            parts = key.split(".")
            for idx, part in enumerate(parts):
                if part == "layers" and idx + 1 < len(parts):
                    try:
                        layer_idx = int(parts[idx + 1])
                    except ValueError:
                        layer_idx = None
                    break
            if layer_idx is not None:
                r2_by_layer[layer_idx] = value
    return SpinQuantRotationSet(r1=r1, r2_by_layer=r2_by_layer)


def _resolve_group_size(size: int, group_size: int) -> int:
    if group_size is None or group_size <= 0:
        return size
    if size % group_size != 0:
        return size
    return group_size


def _fake_quantize_tokenwise(
    x: torch.Tensor,
    *,
    bits: int,
    group_size: int = -1,
    symmetric: bool = True,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Apply token-wise fake quantization over the final dimension.

    All leading dimensions are preserved, so an attention tensor shaped
    ``[seq, batch, heads, head_dim]`` gets independent scales for every
    token/batch/head and optional group over ``head_dim``.
    """

    if bits >= 16:
        return x
    if bits <= 0:
        raise ValueError(f"SpinQuant bit width must be positive, got {bits}.")

    x_dtype = x.dtype
    x_float = x.float()
    last_dim = x_float.shape[-1]
    group_size = _resolve_group_size(last_dim, group_size)
    grouped = x_float.reshape(*x_float.shape[:-1], last_dim // group_size, group_size)

    if symmetric:
        maxq = float(2 ** (bits - 1) - 1)
        amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        scale = amax / maxq
        quantized = torch.round(grouped / scale).clamp(-(maxq + 1), maxq)
        dequantized = quantized * scale
    else:
        maxq = float(2**bits - 1)
        xmin = grouped.amin(dim=-1, keepdim=True)
        xmax = grouped.amax(dim=-1, keepdim=True)
        scale = (xmax - xmin).clamp_min(eps) / maxq
        zero = torch.round(-xmin / scale).clamp(0, maxq)
        quantized = torch.round(grouped / scale + zero).clamp(0, maxq)
        dequantized = (quantized - zero) * scale

    dequantized = dequantized.reshape_as(x_float).to(x_dtype)
    return x + (dequantized - x).detach()


def _nvfp4_quantize_dequantize_rows(x: torch.Tensor) -> torch.Tensor:
    """Pack rows to TE NVFP4 and immediately dequantize for attention compute."""

    if not HAVE_TE_NVFP4 or not x.is_cuda:
        return _fake_quantize_tokenwise(x, bits=4)

    original_shape = x.shape
    rows = x.reshape(-1, original_shape[-1])
    row_count = rows.shape[0]
    pad_rows = (-row_count) % 16
    if pad_rows:
        rows = torch.nn.functional.pad(rows, (0, 0, 0, pad_rows))

    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
    )
    dequantized = quantizer.quantize(rows).dequantize()
    if pad_rows:
        dequantized = dequantized[:row_count]
    return dequantized.reshape(original_shape)


def _quantize_tokenwise_nvfp4_or_fake(
    x: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
) -> torch.Tensor:
    if bits == 4:
        return _nvfp4_quantize_dequantize_rows(x)
    return _fake_quantize_tokenwise(
        x,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )


def _quantize_kv_tensor_tokenwise(
    x: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
) -> torch.Tensor:
    """Quantize K/V in SpinQuant's token-wise layout.

    For attention tensors shaped ``[seq, batch, heads, head_dim]`` and
    ``group_size == -1``, one quantization row is one token containing all
    heads. When ``group_size`` is positive, rows are split over the final
    dimension, which gives per-head/per-group behavior.
    """

    if bits >= 16:
        return x

    if x.dim() >= 4 and group_size <= 0:
        original_shape = x.shape
        token_rows = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
        token_rows = _quantize_tokenwise_nvfp4_or_fake(
            token_rows,
            bits=bits,
            group_size=-1,
            symmetric=symmetric,
        )
        return token_rows.reshape(original_shape)

    return _quantize_tokenwise_nvfp4_or_fake(
        x,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )


def quantize_kv_cache_tokenwise(
    key: torch.Tensor,
    value: torch.Tensor,
    config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K/V tensors according to SpinQuant token-wise settings."""

    key = _quantize_kv_tensor_tokenwise(
        key,
        bits=getattr(config, "spinquant_k_bits", 16),
        group_size=getattr(config, "spinquant_k_groupsize", -1),
        symmetric=getattr(config, "spinquant_kv_sym", True),
    )
    value = _quantize_kv_tensor_tokenwise(
        value,
        bits=getattr(config, "spinquant_v_bits", 16),
        group_size=getattr(config, "spinquant_v_groupsize", -1),
        symmetric=getattr(config, "spinquant_kv_sym", True),
    )
    return key, value
