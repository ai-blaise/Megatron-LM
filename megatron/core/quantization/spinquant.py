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


_ROTATION_CACHE: dict[tuple, object] = {}


@dataclass
class SpinQuantRotationSet:
    """Container for SpinQuant's learned/global rotations.

    R1 is the model-wide hidden-state rotation. R2 is keyed by layer index and
    applies to per-head attention value/KV-sensitive paths.
    """

    r1: torch.Tensor | None
    r2_by_layer: dict[int, torch.Tensor]


@dataclass
class SpinQuantFusionStats:
    """Counts of modules rotated by SpinQuant weight fusion."""

    attention_qkv: int = 0
    attention_out: int = 0
    mlp_fc1: int = 0
    mlp_fc2: int = 0
    skipped: int = 0


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


def _rotation_cache_key(config, name: str, layer_idx: int, size: int, device, dtype) -> tuple:
    path = getattr(config, "spinquant_rotation_path", None)
    mode = getattr(config, "spinquant_mode", "random")
    return (
        name,
        mode,
        str(path) if path is not None else None,
        layer_idx,
        size,
        str(device),
        str(dtype),
    )


def _get_rotation_set(config) -> SpinQuantRotationSet:
    path = getattr(config, "spinquant_rotation_path", None)
    if path is None:
        raise ValueError("SpinQuant loaded rotation mode requires spinquant_rotation_path.")
    key = ("rotation_set", str(path))
    cached = _ROTATION_CACHE.get(key)
    if cached is None:
        cached = load_rotation_set(path)
        _ROTATION_CACHE[key] = cached
    return cached


def get_spinquant_r1(config, device, dtype) -> torch.Tensor | None:
    """Return the global R1 rotation for SpinQuant weight fusion."""

    mode = getattr(config, "spinquant_mode", "random")
    if mode == "identity":
        return None

    hidden_size = getattr(config, "hidden_size")
    key = _rotation_cache_key(config, "r1", -1, hidden_size, device, dtype)
    cached = _ROTATION_CACHE.get(key)
    if cached is not None:
        return cached

    if mode == "loaded":
        rotation_set = _get_rotation_set(config)
        if rotation_set.r1 is None:
            raise ValueError("Missing SpinQuant R1 in loaded rotation checkpoint.")
        rotation = rotation_set.r1.to(device=device, dtype=dtype)
    elif mode == "random":
        rotation = random_hadamard_rotation(hidden_size, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown SpinQuant mode {mode}.")

    _ROTATION_CACHE[key] = rotation
    return rotation


def get_spinquant_r2(config, layer_number: int, head_dim: int, device, dtype) -> torch.Tensor | None:
    """Return the per-head R2 rotation for an attention layer."""

    mode = getattr(config, "spinquant_mode", "random")
    if mode == "identity":
        return None

    layer_idx = max(layer_number - 1, 0)
    key = _rotation_cache_key(config, "r2", layer_idx, head_dim, device, dtype)
    cached = _ROTATION_CACHE.get(key)
    if cached is not None:
        return cached

    if mode == "loaded":
        rotation_set = _get_rotation_set(config)
        if layer_idx not in rotation_set.r2_by_layer:
            raise ValueError(f"Missing SpinQuant R2 for layer {layer_idx}.")
        rotation = rotation_set.r2_by_layer[layer_idx].to(device=device, dtype=dtype)
    elif mode == "random":
        rotation = random_hadamard_rotation(
            head_dim,
            device=device,
            dtype=dtype,
            seed=1234 + layer_idx,
        )
    else:
        raise ValueError(f"Unknown SpinQuant mode {mode}.")

    _ROTATION_CACHE[key] = rotation
    return rotation


def _read_linear_weight(linear) -> torch.Tensor | None:
    weight = getattr(linear, "weight", None)
    if weight is None:
        return None
    if hasattr(weight, "dequantize"):
        return weight.dequantize()
    return weight.detach()


def _write_linear_weight(linear, value: torch.Tensor) -> None:
    weight = getattr(linear, "weight", None)
    if weight is None:
        return
    value = value.to(device=weight.device, dtype=weight.dtype)
    if hasattr(weight, "quantize_"):
        weight.quantize_(value)
    else:
        weight.data.copy_(value)


def _matmul_right(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.matmul(weight.float(), rotation.float()).to(weight.dtype)


def _matmul_left(rotation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(rotation.float(), weight.float()).to(weight.dtype)


def _rotate_value_projection_rows(weight: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    head_dim = r2.shape[0]
    if weight.shape[0] % head_dim != 0:
        raise ValueError(
            f"V projection rows {weight.shape[0]} are not divisible by head_dim {head_dim}."
        )
    original_shape = weight.shape
    value_blocks = weight.reshape(-1, head_dim, original_shape[-1])
    value_blocks = torch.matmul(r2.transpose(0, 1).float(), value_blocks.float()).to(weight.dtype)
    return value_blocks.reshape(original_shape)


def _rotate_output_projection_columns(weight: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    head_dim = r2.shape[0]
    if weight.shape[1] % head_dim != 0:
        raise ValueError(
            f"Output projection columns {weight.shape[1]} are not divisible by head_dim {head_dim}."
        )
    original_shape = weight.shape
    blocks = weight.reshape(original_shape[0], -1, head_dim)
    blocks = torch.matmul(blocks.float(), r2.float()).to(weight.dtype)
    return blocks.reshape(original_shape)


def _rotate_interleaved_qkv_weight(attention, r1: torch.Tensor | None) -> bool:
    weight = _read_linear_weight(attention.linear_qkv)
    if weight is None:
        return False
    rotated = weight
    if r1 is not None:
        rotated = _matmul_right(rotated, r1)

    r2 = get_spinquant_r2(
        attention.config,
        attention.layer_number,
        attention.hidden_size_per_attention_head,
        rotated.device,
        rotated.dtype,
    )
    if r2 is not None:
        num_query_heads_per_group = (
            attention.num_attention_heads_per_partition
            // attention.num_query_groups_per_partition
        )
        block_heads = num_query_heads_per_group + 2
        if getattr(attention.config, "attention_output_gate", False):
            block_heads += num_query_heads_per_group
        block = block_heads * attention.hidden_size_per_attention_head
        if getattr(attention.config, "attention_output_gate", False):
            value_offset = (2 * num_query_heads_per_group + 1) * attention.hidden_size_per_attention_head
        else:
            value_offset = (num_query_heads_per_group + 1) * attention.hidden_size_per_attention_head
        if rotated.shape[0] % block != 0:
            raise ValueError(
                f"linear_qkv rows {rotated.shape[0]} are not divisible by QKV block {block}."
            )
        rotated = rotated.clone()
        for start in range(0, rotated.shape[0], block):
            value_slice = slice(start + value_offset, start + value_offset + r2.shape[0])
            rotated[value_slice, :] = _rotate_value_projection_rows(rotated[value_slice, :], r2)

    _write_linear_weight(attention.linear_qkv, rotated)
    return True


def _rotate_attention_output_weight(attention, r1: torch.Tensor | None) -> bool:
    weight = _read_linear_weight(attention.linear_proj)
    if weight is None:
        return False
    rotated = weight
    if r1 is not None:
        rotated = _matmul_left(r1.transpose(0, 1), rotated)
    r2 = get_spinquant_r2(
        attention.config,
        attention.layer_number,
        attention.hidden_size_per_attention_head,
        rotated.device,
        rotated.dtype,
    )
    if r2 is not None:
        rotated = _rotate_output_projection_columns(rotated, r2)
    _write_linear_weight(attention.linear_proj, rotated)
    return True


def _rotate_mlp_weights(mlp, r1: torch.Tensor | None) -> tuple[bool, bool]:
    fc1_done = False
    fc2_done = False
    if r1 is not None and hasattr(mlp, "linear_fc1"):
        weight = _read_linear_weight(mlp.linear_fc1)
        if weight is not None and weight.shape[1] == r1.shape[0]:
            _write_linear_weight(mlp.linear_fc1, _matmul_right(weight, r1))
            fc1_done = True
    if r1 is not None and hasattr(mlp, "linear_fc2"):
        weight = _read_linear_weight(mlp.linear_fc2)
        if weight is not None and weight.shape[0] == r1.shape[0]:
            _write_linear_weight(mlp.linear_fc2, _matmul_left(r1.transpose(0, 1), weight))
            fc2_done = True
    return fc1_done, fc2_done


def fuse_spinquant_weights(model_or_models, config) -> SpinQuantFusionStats:
    """Fuse SpinQuant R1/R2 rotations into Megatron linear weights.

    This mirrors facebookresearch/SpinQuant's QuantizeLinear transformations:
    Q/K/V and MLP input projections get ``W @ R1``; attention/MLP output
    projections get ``R1.T @ W``; V and attention-output per-head paths get
    the layer's R2 rotation. The forward pass stays on Megatron/TE kernels.
    """

    stats = SpinQuantFusionStats()
    if not getattr(config, "spinquant", False) or not getattr(config, "spinquant_fuse_weights", False):
        return stats

    models = model_or_models if isinstance(model_or_models, (list, tuple)) else [model_or_models]
    with torch.no_grad():
        for model in models:
            for module in model.modules():
                if hasattr(module, "linear_qkv") and hasattr(module, "linear_proj"):
                    sample_weight = _read_linear_weight(module.linear_qkv)
                    if sample_weight is None:
                        stats.skipped += 1
                        continue
                    r1 = get_spinquant_r1(config, sample_weight.device, sample_weight.dtype)
                    if _rotate_interleaved_qkv_weight(module, r1):
                        stats.attention_qkv += 1
                    if _rotate_attention_output_weight(module, r1):
                        stats.attention_out += 1
                if hasattr(module, "linear_fc1") and hasattr(module, "linear_fc2"):
                    sample_weight = _read_linear_weight(module.linear_fc1)
                    if sample_weight is None:
                        stats.skipped += 1
                        continue
                    r1 = get_spinquant_r1(config, sample_weight.device, sample_weight.dtype)
                    fc1_done, fc2_done = _rotate_mlp_weights(module, r1)
                    stats.mlp_fc1 += int(fc1_done)
                    stats.mlp_fc2 += int(fc2_done)
    return stats


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
