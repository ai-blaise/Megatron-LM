# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stochastic-rounded NVFP4 master-weight cast for FlashAdamW + ECO.

The FlashOptim ECO paper (arXiv:2601.22101) requires stochastic rounding on
the master-weight quantization so that sub-ULP updates accumulate in
expectation. TE's batched master-weight cast does round-to-nearest only.

This module reproduces TE's ``_cast_master_weights_to_nvfp4_2d`` pipeline
(per-block amax, cross-rank all-reduce, global scale, per-block decode
scale, final 4-bit pack) but inserts a block-aware uniform dither on the
master shards before the final pack. In the dominant |x_scaled| <= 2
regime the NVFP4 grid is uniform (gap = 0.5), so pre-dither + round-to-
nearest is statistically identical to SR.

The dither is strictly in-place on the master shard, which is acceptable
because the ECO path uses a transient master (``_fa_updated_shard``) that
is discarded after the cast.
"""

from typing import List, Optional, Tuple, Any

import torch
import triton
import triton.language as tl

import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.constants import NVFP4_BLOCK_SCALING_SIZE


# Step counter for RNG seeding; each optimizer step uses a new seed so
# consecutive steps get independent noise.
_DITHER_STEP_COUNTER = [0]


def _next_dither_seed() -> int:
    """Monotonically increasing seed shared across ranks + tensors within a step.

    Per-element independence is achieved by combining this seed with the
    element offset inside the Triton kernel via ``tl.rand``.
    """
    _DITHER_STEP_COUNTER[0] += 1
    # Mix in the rank so ranks draw independent noise (required for unbiased
    # SR when gradients are averaged across DP).
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    return (_DITHER_STEP_COUNTER[0] * 2654435761 + rank * 40503) & 0x7FFFFFFF


@triton.jit
def _triton_block_dither_kernel(
    master_ptr,
    decode_scale_ptr,     # FP32, shape (tile_h, tile_w)
    numel: int,
    shard_start_offset: int,
    full_w: int,
    scale_row_stride: int,
    inv_global_scale: float,
    seed: int,
    BLOCK_SIZE: tl.constexpr,
    DITHER_COEF: tl.constexpr,
):
    """Add block-aware uniform dither to master in-place.

    For each element e at flat position p = shard_start_offset + offs:
      tile = (p // full_w // 16, p % full_w // 16)
      gap  = decode_scale[tile] / global_scale
      e   += U(-1, 1) * DITHER_COEF * gap
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    flat_idx = offs + shard_start_offset
    h_idx = flat_idx // full_w
    w_idx = flat_idx % full_w
    tile_h = h_idx // 16
    tile_w = w_idx // 16
    scale_offset = tile_h * scale_row_stride + tile_w

    decode_scale = tl.load(decode_scale_ptr + scale_offset, mask=mask, other=0.0)

    # U(-1, 1) per element. tl.rand uses the (seed, offset) pair → independent.
    rand01 = tl.rand(seed, offs)
    rand_sym = rand01 * 2.0 - 1.0

    # Dither magnitude = DITHER_COEF * gap (gap = decode_scale / global_scale).
    dither = rand_sym * DITHER_COEF * decode_scale * inv_global_scale

    master = tl.load(master_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    noisy = master + dither
    tl.store(master_ptr + offs, noisy, mask=mask)


def _apply_block_dither(
    master: torch.Tensor,          # 1D shard
    decode_scale: torch.Tensor,    # FP32, (tile_h, tile_w)
    inv_global_scale: float,
    full_w: int,
    start_offset: int,
    seed: int,
    dither_coef: float = 0.25,
) -> None:
    """Dispatch the Triton dither kernel on one master shard.

    Dither coefficient 0.25 corresponds to ± half-ULP uniform noise in the
    dominant small-value NVFP4 regime (grid spacing 0.5 in normalized
    coordinates → spacing 0.5 * decode_scale/global_scale in master
    coordinates → half = 0.25 * decode_scale/global_scale).
    """
    assert master.is_cuda and decode_scale.is_cuda
    assert master.dtype == torch.float32, (
        "SR dither expects FP32 master (the transient ECO shard). "
        f"Got {master.dtype}."
    )
    numel = master.numel()
    if numel == 0:
        return

    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _triton_block_dither_kernel[grid](
        master,
        decode_scale,
        numel,
        int(start_offset),
        int(full_w),
        int(decode_scale.shape[1]),
        float(inv_global_scale),
        int(seed),
        BLOCK_SIZE=BLOCK,
        DITHER_COEF=dither_coef,
    )


def cast_master_weights_to_nvfp4_2d_sr(
    params: List[Tuple[Any, Optional[torch.Tensor], Optional[int], Any]],
    group,
    use_fsdp_shard_model_weights: bool = False,
    manual_post_all_gather_processing: bool = False,
) -> None:
    """NVFP4 2D master-weight cast with stochastic rounding.

    Mirrors TE's ``_cast_master_weights_to_nvfp4_2d`` (amax → all-reduce →
    global scale → per-block decode scale → 4-bit pack), inserting a
    block-aware uniform dither on each master shard between the decode-
    scale computation and the final pack. The dither makes the
    (deterministic) round-to-nearest pack statistically equivalent to SR
    in the dominant NVFP4 grid regime.

    Args mirror TE's function exactly.
    """
    if len(params) == 0:
        return

    device = params[0][0].device
    block_len = NVFP4_BLOCK_SCALING_SIZE  # 16

    # ---- Per-tensor bookkeeping (matches TE's layout) ---------------------
    cu_amax_sizes = [0]
    tile_shapes: List[Tuple[int, int]] = []
    tile_widths: List[int] = []
    scale_targets: List[torch.Tensor] = []
    amax_targets: List[Optional[torch.Tensor]] = []
    for model_weight, _, _, _ in params:
        quantizer = model_weight._get_quantizer()
        if not isinstance(quantizer, NVFP4Quantizer):
            raise TypeError(
                f"Expected NVFP4Quantizer, got {type(quantizer).__name__}"
            )
        if not quantizer.with_2d_quantization:
            raise ValueError("NVFP4 2D quantization must be enabled.")
        if len(model_weight.shape) != 2:
            raise ValueError(
                f"Expected 2D model weight, got {len(model_weight.shape)}D"
            )
        h, w = model_weight.shape
        tile_h = (h + block_len - 1) // block_len
        tile_w = (w + block_len - 1) // block_len
        tile_shapes.append((tile_h, tile_w))
        tile_widths.append(tile_w)
        scale_targets.append(model_weight._rowwise_scale_inv)
        amax_targets.append(model_weight._amax_rowwise)
        cu_amax_sizes.append(cu_amax_sizes[-1] + tile_h * tile_w)

    packed_amaxes = torch.zeros(cu_amax_sizes[-1], dtype=torch.float32, device=device)
    packed_scales = torch.zeros(cu_amax_sizes[-1], dtype=torch.float32, device=device)

    global_amaxes = torch.zeros(len(params), dtype=torch.float32, device=device)
    global_amax_views: List[torch.Tensor] = [
        global_amaxes[i : i + 1] for i in range(len(params))
    ]

    amaxes: List[torch.Tensor] = []
    scales: List[torch.Tensor] = []

    master_weight_list: List[torch.Tensor] = []
    partial_amax_list: List[torch.Tensor] = []
    global_amax_list: List[torch.Tensor] = []
    h_list: List[int] = []
    w_list: List[int] = []
    start_offset_list: List[int] = []

    for i, (model_weight, master_weight, start_offset, _) in enumerate(params):
        scale_shape = tile_shapes[i]
        amax = packed_amaxes[cu_amax_sizes[i] : cu_amax_sizes[i + 1]].reshape(scale_shape)
        scale = packed_scales[cu_amax_sizes[i] : cu_amax_sizes[i + 1]].reshape(scale_shape)
        amaxes.append(amax)
        scales.append(scale)

        if master_weight is not None and master_weight.numel() > 0:
            h, w = model_weight.shape
            master_weight_list.append(master_weight)
            partial_amax_list.append(amax)
            global_amax_list.append(global_amax_views[i])
            h_list.append(h)
            w_list.append(w)
            start_offset_list.append(start_offset)

    # ---- Partial amax (per-block) + global amax --------------------------
    if master_weight_list:
        tex.nvfp4_multi_tensor_compute_partial_amax(
            master_weight_list,
            partial_amax_list,
            global_amax_list,
            h_list,
            w_list,
            start_offset_list,
            block_len,
        )

    if packed_amaxes.numel() > 0:
        torch.distributed.all_reduce(
            packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=group
        )
    if global_amaxes.numel() > 0:
        torch.distributed.all_reduce(
            global_amaxes, op=torch.distributed.ReduceOp.MAX, group=group
        )

    # ---- Global scale (per-tensor) ---------------------------------------
    global_scale_tensor = torch.empty_like(global_amaxes)
    tex.nvfp4_compute_global_scale(global_amaxes, global_scale_tensor)
    global_scale_views = [global_scale_tensor[i : i + 1] for i in range(len(params))]

    # ---- Per-block decode scale (FP8 target) + FP32 helper ---------------
    fused_scale_block_amax_list: List[torch.Tensor] = []
    fused_scale_global_amax_list: List[torch.Tensor] = []
    fused_scale_per_block_scale_list: List[torch.Tensor] = []
    fused_scale_target_scale_list: List[torch.Tensor] = []
    fused_scale_target_amax_list: List[torch.Tensor] = []
    fused_scale_tile_rows_list: List[int] = []
    fused_scale_tile_cols_list: List[int] = []
    fused_scale_rows_padded_list: List[int] = []

    partial_cast_inp_list: List[torch.Tensor] = []
    partial_cast_out_list: List[torch.Tensor] = []
    partial_cast_scale_list: List[torch.Tensor] = []
    partial_cast_global_scale_list: List[torch.Tensor] = []
    partial_cast_h_list: List[int] = []
    partial_cast_w_list: List[int] = []
    partial_cast_start_offset_list: List[int] = []

    # Also remember per-tensor ingredients needed by the dither kernel.
    dither_master_list: List[torch.Tensor] = []
    dither_decode_scale_list: List[torch.Tensor] = []
    dither_inv_global_scale_list: List[float] = []
    dither_full_w_list: List[int] = []
    dither_start_offset_list: List[int] = []

    for idx, (
        (model_weight, master_weight, start_offset, model_weight_fragment),
        tile_shape,
        tile_col_cnt,
        target_scale,
        target_amax,
        block_amax,
        per_block_decode_scale,
        global_scale,
    ) in enumerate(
        zip(
            params,
            tile_shapes,
            tile_widths,
            scale_targets,
            amax_targets,
            amaxes,
            scales,
            global_scale_views,
        )
    ):
        if not manual_post_all_gather_processing:
            model_weight.update_usage(rowwise_usage=True, columnwise_usage=False)

        tile_rows = tile_shape[0]
        rows_padded = target_scale.shape[0]
        global_amax_view = global_amaxes[idx : idx + 1]

        if target_amax is not None:
            fused_scale_block_amax_list.append(block_amax)
            fused_scale_global_amax_list.append(global_amax_view)
            fused_scale_per_block_scale_list.append(per_block_decode_scale)
            fused_scale_target_scale_list.append(target_scale)
            fused_scale_target_amax_list.append(target_amax)
            fused_scale_tile_rows_list.append(tile_rows)
            fused_scale_tile_cols_list.append(tile_col_cnt)
            fused_scale_rows_padded_list.append(rows_padded)

        if master_weight is not None and master_weight.numel() > 0:
            end_offset = start_offset + master_weight.numel()
            if not use_fsdp_shard_model_weights:
                rowwise_bytes = model_weight._rowwise_data.view(-1)
                byte_start = start_offset // 2
                byte_end = (end_offset + 1) // 2
                model_weight_fragment = rowwise_bytes[byte_start:byte_end]
            h, w = model_weight.shape

            partial_cast_inp_list.append(master_weight)
            partial_cast_out_list.append(model_weight_fragment)
            partial_cast_scale_list.append(per_block_decode_scale)
            partial_cast_global_scale_list.append(global_scale)
            partial_cast_h_list.append(h)
            partial_cast_w_list.append(w)
            partial_cast_start_offset_list.append(start_offset)

            dither_master_list.append(master_weight)
            dither_decode_scale_list.append(per_block_decode_scale)
            dither_inv_global_scale_list.append(1.0 / float(global_scale.item()))
            dither_full_w_list.append(w)
            dither_start_offset_list.append(start_offset)

    # Convert FP32 per-block decode scale → E4M3 target scale (and update target_amax).
    if fused_scale_block_amax_list:
        tex.nvfp4_multi_tensor_fused_scale(
            fused_scale_block_amax_list,
            fused_scale_global_amax_list,
            fused_scale_per_block_scale_list,
            fused_scale_target_scale_list,
            fused_scale_target_amax_list,
            fused_scale_tile_rows_list,
            fused_scale_tile_cols_list,
            fused_scale_rows_padded_list,
            block_len,
        )

    # ---- Insert stochastic-rounding dither on master shards --------------
    base_seed = _next_dither_seed()
    for i, (master, dscale, inv_gs, full_w, offset) in enumerate(
        zip(
            dither_master_list,
            dither_decode_scale_list,
            dither_inv_global_scale_list,
            dither_full_w_list,
            dither_start_offset_list,
        )
    ):
        _apply_block_dither(
            master=master,
            decode_scale=dscale,
            inv_global_scale=inv_gs,
            full_w=full_w,
            start_offset=offset,
            seed=(base_seed + i) & 0x7FFFFFFF,
        )

    # ---- Final deterministic round-to-nearest pack (now SR in expectation).
    if partial_cast_inp_list:
        tex.nvfp4_multi_tensor_2d_partial_cast(
            partial_cast_inp_list,
            partial_cast_out_list,
            partial_cast_scale_list,
            partial_cast_global_scale_list,
            partial_cast_h_list,
            partial_cast_w_list,
            partial_cast_start_offset_list,
            block_len,
        )
