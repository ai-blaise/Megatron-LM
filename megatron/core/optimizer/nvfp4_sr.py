# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stochastic-rounded NVFP4 master-weight cast for FlashAdamW + ECO.

The FlashOptim ECO paper (arXiv:2601.22101) requires stochastic rounding on
the master-weight quantization so that sub-ULP updates accumulate in
expectation. TE's batched master-weight cast does round-to-nearest only.

This module reproduces TE's ``_cast_master_weights_to_nvfp4_2d`` pipeline
(per-block amax, cross-rank all-reduce, global scale, per-block decode
scale, final 4-bit pack) but inserts a block-aware uniform dither on transient
cast shards before the final pack. In the dominant |x_scaled| <= 2
regime the NVFP4 grid is uniform (gap = 0.5), so pre-dither + round-to-
nearest is statistically identical to SR.

The original transient master shard is preserved so ECO can inject the true
error ``theta - q_sr(theta)`` rather than including the sampled dither in the
error term.
"""

import os
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
    """Add block-aware uniform dither to a cast shard in-place.

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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in (256, 512, 1024, 2048, 4096)
        for nw in (2, 4, 8)
        for ns in (1, 2, 3)
    ],
    key=["numel"],
    # In-place RMW on the master shard: snapshot+restore around each
    # autotune timing trial so the dither isn't applied 45 extra times
    # before the real call (matches the eco_inject autotune contract).
    restore_value=("master_ptr",),
)
@triton.jit
def _triton_block_dither_kernel_autotuned(
    master_ptr,
    decode_scale_ptr,
    numel: int,
    shard_start_offset: int,
    full_w: int,
    scale_row_stride: int,
    inv_global_scale: float,
    seed: int,
    BLOCK_SIZE: tl.constexpr,
    DITHER_COEF: tl.constexpr,
):
    """Autotuned variant of the dither kernel.

    Math is bit-identical to ``_triton_block_dither_kernel`` for any fixed
    BLOCK_SIZE, but Triton's ``tl.rand`` distributes its PRNG state across
    SIMD lanes per-block, so two configs that pick different BLOCK_SIZE
    produce different *realized* random sequences at identical
    (seed, offset) inputs. The per-element dither distribution
    (U(-DITHER_COEF, +DITHER_COEF), zero mean, same variance) and ECO's
    expected-value contract on the cast result are preserved; what is
    lost is bit-for-bit reproducibility across runs that select different
    autotune configs. See ``_apply_block_dither`` for the size-threshold
    dispatch that opts large shards into this autotuned variant.
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
    rand_sym = tl.rand(seed, offs) * 2.0 - 1.0
    dither = rand_sym * DITHER_COEF * decode_scale * inv_global_scale

    master = tl.load(master_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(master_ptr + offs, master + dither, mask=mask)


# Crossover threshold for the autotuned variant. Below this, the fixed
# BLOCK_SIZE=1024 baseline kernel runs ~8-10 us faster (the autotuned
# variant pays autotune-dispatch overhead that exceeds the work for
# small shards). Above this, autotune unlocks larger BLOCK_SIZE +
# num_warps configs that scale toward the HBM bandwidth ceiling and
# beat the baseline by 60-73% on B200. Bench-derived crossover is
# between 4M and 16M elements; threshold set conservatively at the
# midpoint so the baseline path covers the regression zone with margin.
_DITHER_AUTOTUNE_THRESHOLD = 8 * 1024 * 1024  # 8 Mi elements


@triton.jit
def _triton_nvfp4_partial_amax_kernel(
    master_ptr,
    amax_ptr,
    numel: int,
    shard_start_offset: int,
    full_h: int,
    full_w: int,
    tile_w: int,
    BLOCK_ELEMS: tl.constexpr,
):
    """Compute one 16x16-block amax for the local shard."""
    tile_r = tl.program_id(0)
    tile_c = tl.program_id(1)
    offs = tl.arange(0, BLOCK_ELEMS)
    rows = tile_r * 16 + offs // 16
    cols = tile_c * 16 + offs % 16
    flat = rows * full_w + cols
    local = flat - shard_start_offset
    mask = (
        (rows < full_h)
        & (cols < full_w)
        & (local >= 0)
        & (local < numel)
    )
    vals = tl.load(master_ptr + local, mask=mask, other=0.0).to(tl.float32)
    max_abs = tl.max(tl.abs(vals), axis=0)
    tl.store(amax_ptr + tile_r * tile_w + tile_c, max_abs)


@triton.jit
def _triton_nvfp4_partial_cast_kernel(
    master_ptr,
    out_ptr,
    scale_ptr,
    global_scale_ptr,
    num_bytes: int,
    master_numel: int,
    shard_start_offset: int,
    full_h: int,
    full_w: int,
    tile_w: int,
    byte_start: int,
    BLOCK_BYTES: tl.constexpr,
):
    """Pack a shard of FP32 master data into rowwise NVFP4 bytes."""
    offs = tl.program_id(0) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    byte_mask = offs < num_bytes
    logical_byte = byte_start + offs
    flat0 = logical_byte * 2
    flat1 = flat0 + 1

    local0 = flat0 - shard_start_offset
    local1 = flat1 - shard_start_offset
    valid0 = byte_mask & (local0 >= 0) & (local0 < master_numel) & (flat0 < full_h * full_w)
    valid1 = byte_mask & (local1 >= 0) & (local1 < master_numel) & (flat1 < full_h * full_w)

    global_scale = tl.load(global_scale_ptr).to(tl.float32)

    row0 = flat0 // full_w
    col0 = flat0 - row0 * full_w
    scale_off0 = (row0 // 16) * tile_w + (col0 // 16)
    block_scale0 = tl.load(scale_ptr + scale_off0, mask=valid0, other=1.0).to(tl.float32)
    enc0 = tl.where(block_scale0 > 0.0, global_scale / block_scale0, 1.0)
    x0 = tl.load(master_ptr + local0, mask=valid0, other=0.0).to(tl.float32) * enc0
    x0 = tl.minimum(tl.maximum(x0, -6.0), 6.0)

    row1 = flat1 // full_w
    col1 = flat1 - row1 * full_w
    scale_off1 = (row1 // 16) * tile_w + (col1 // 16)
    block_scale1 = tl.load(scale_ptr + scale_off1, mask=valid1, other=1.0).to(tl.float32)
    enc1 = tl.where(block_scale1 > 0.0, global_scale / block_scale1, 1.0)
    x1 = tl.load(master_ptr + local1, mask=valid1, other=0.0).to(tl.float32) * enc1
    x1 = tl.minimum(tl.maximum(x1, -6.0), 6.0)

    ax0 = tl.abs(x0)
    pos0 = tl.where(
        ax0 <= 0.25,
        0,
        tl.where(
            ax0 < 0.75,
            1,
            tl.where(
                ax0 <= 1.25,
                2,
                tl.where(
                    ax0 < 1.75,
                    3,
                    tl.where(
                        ax0 <= 2.5,
                        4,
                        tl.where(ax0 < 3.5, 5, tl.where(ax0 <= 5.0, 6, 7)),
                    ),
                ),
            ),
        ),
    )
    neg0 = tl.where(
        ax0 <= 0.25,
        8,
        tl.where(
            ax0 < 0.75,
            9,
            tl.where(
                ax0 <= 1.25,
                10,
                tl.where(
                    ax0 < 1.75,
                    11,
                    tl.where(
                        ax0 <= 2.5,
                        12,
                        tl.where(ax0 < 3.5, 13, tl.where(ax0 <= 5.0, 14, 15)),
                    ),
                ),
            ),
        ),
    )
    code0 = tl.where(x0 < 0.0, neg0, pos0).to(tl.int32)

    ax1 = tl.abs(x1)
    pos1 = tl.where(
        ax1 <= 0.25,
        0,
        tl.where(
            ax1 < 0.75,
            1,
            tl.where(
                ax1 <= 1.25,
                2,
                tl.where(
                    ax1 < 1.75,
                    3,
                    tl.where(
                        ax1 <= 2.5,
                        4,
                        tl.where(ax1 < 3.5, 5, tl.where(ax1 <= 5.0, 6, 7)),
                    ),
                ),
            ),
        ),
    )
    neg1 = tl.where(
        ax1 <= 0.25,
        8,
        tl.where(
            ax1 < 0.75,
            9,
            tl.where(
                ax1 <= 1.25,
                10,
                tl.where(
                    ax1 < 1.75,
                    11,
                    tl.where(
                        ax1 <= 2.5,
                        12,
                        tl.where(ax1 < 3.5, 13, tl.where(ax1 <= 5.0, 14, 15)),
                    ),
                ),
            ),
        ),
    )
    code1 = tl.where(x1 < 0.0, neg1, pos1).to(tl.int32)

    old = tl.load(out_ptr + offs, mask=byte_mask, other=0).to(tl.int32)
    new = old
    new = tl.where(valid0, (new & 0xF0) | code0, new)
    new = tl.where(valid1, (new & 0x0F) | (code1 << 4), new)
    tl.store(out_ptr + offs, new.to(tl.uint8), mask=byte_mask)


def _has_te_nvfp4_sr_kernels() -> bool:
    """Return whether the installed TE exposes the batched NVFP4 helpers."""
    required = (
        "nvfp4_multi_tensor_compute_partial_amax",
        "nvfp4_compute_global_scale",
        "nvfp4_multi_tensor_fused_scale",
        "nvfp4_multi_tensor_2d_partial_cast",
    )
    return all(hasattr(tex, name) for name in required)


def _compute_partial_amax_fallback(
    master: torch.Tensor,
    amax: torch.Tensor,
    h: int,
    w: int,
    start_offset: int,
) -> None:
    """Fallback for TE's per-block partial amax helper."""
    if master is None or master.numel() == 0:
        return
    grid = (amax.shape[0], amax.shape[1])
    _triton_nvfp4_partial_amax_kernel[grid](
        master,
        amax,
        master.numel(),
        int(start_offset),
        int(h),
        int(w),
        int(amax.shape[1]),
        BLOCK_ELEMS=256,
    )


def _compute_global_scale_fallback(
    global_amaxes: torch.Tensor,
    global_scale: torch.Tensor,
) -> None:
    """Compute NVFP4's per-tensor encode scale."""
    max_fp4 = 6.0
    max_fp8_e4m3 = 448.0
    scale = torch.where(
        global_amaxes > 0,
        (max_fp4 * max_fp8_e4m3) / global_amaxes,
        torch.ones_like(global_amaxes),
    )
    scale = torch.minimum(scale, torch.full_like(scale, torch.finfo(torch.float32).max))
    global_scale.copy_(scale)


def _fused_scale_fallback(
    block_amax: torch.Tensor,
    global_amax: torch.Tensor,
    per_block_scale: torch.Tensor,
    target_scale: torch.Tensor,
    target_amax: torch.Tensor,
    tile_rows: int,
    tile_cols: int,
    rows_padded: int,
) -> None:
    """Fallback for TE's NVFP4 block-scale packing helper."""
    max_fp4 = 6.0
    max_fp8_e4m3 = 448.0
    global_encode_scale = torch.where(
        global_amax > 0,
        (max_fp4 * max_fp8_e4m3) / global_amax,
        torch.ones_like(global_amax),
    )
    scale = block_amax * (global_encode_scale / max_fp4)
    scale = torch.clamp(scale, min=-max_fp8_e4m3, max=max_fp8_e4m3)
    scale_f8 = scale.to(torch.float8_e4m3fn)
    per_block_scale.copy_(scale_f8.float())

    if target_amax is not None:
        target_amax.copy_(global_amax)

    # TE stores the 16x16 2D scale repeated once per logical row. The
    # allocation is padded to [multiple-of-128 rows, multiple-of-4 scale cols].
    scale_u8 = scale_f8.view(torch.uint8)
    rows_to_fill = min(tile_rows * 16, rows_padded, target_scale.shape[0])
    cols_to_fill = min(tile_cols, target_scale.shape[1])
    if rows_to_fill > 0 and cols_to_fill > 0:
        expanded = scale_u8.repeat_interleave(16, dim=0)
        target_scale[:rows_to_fill, :cols_to_fill].copy_(
            expanded[:rows_to_fill, :cols_to_fill]
        )
    if target_scale.shape[1] > cols_to_fill:
        target_scale[:rows_to_fill, cols_to_fill:].zero_()


def _partial_cast_fallback(
    master: torch.Tensor,
    out: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    h: int,
    w: int,
    start_offset: int,
) -> None:
    """Fallback for TE's shard-aware NVFP4 2D partial cast helper."""
    if master.numel() == 0 or out.numel() == 0:
        return
    block = 1024
    grid = (triton.cdiv(out.numel(), block),)
    _triton_nvfp4_partial_cast_kernel[grid](
        master,
        out,
        scale,
        global_scale,
        out.numel(),
        master.numel(),
        int(start_offset),
        int(h),
        int(w),
        int(scale.shape[1]),
        int(start_offset // 2),
        BLOCK_BYTES=block,
    )


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

    Two kernel variants are dispatched based on shard size:
      - ``numel < _DITHER_AUTOTUNE_THRESHOLD`` (8 Mi): the fixed
        BLOCK_SIZE=1024 baseline kernel. Avoids autotune dispatch
        overhead that would slow small shards by 8-10 us per call.
        Common for MLA LoRA / DSA Indexer / per-expert MoE shards under
        moderate DP.
      - ``numel >= _DITHER_AUTOTUNE_THRESHOLD``: the autotuned variant.
        Larger BLOCK_SIZE + num_warps configs scale toward HBM bandwidth
        and reduce per-call time by 60-73% on B200 for FFN / embedding
        shards. Pays a one-time per-shape autotune sweep on first call;
        cached forever after.
    """
    assert master.is_cuda and decode_scale.is_cuda
    assert master.dtype == torch.float32, (
        "SR dither expects FP32 master (the transient ECO shard). "
        f"Got {master.dtype}."
    )
    numel = master.numel()
    if numel == 0:
        return

    if numel < _DITHER_AUTOTUNE_THRESHOLD:
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
    else:
        # Meta-aware grid: the autotune-chosen BLOCK_SIZE feeds back into
        # the CTA count, capped at the standard 2*SM heuristic.
        sm_count = torch.cuda.get_device_properties(master.device).multi_processor_count
        grid = lambda meta: (
            min(2 * sm_count, triton.cdiv(numel, meta["BLOCK_SIZE"])),
        )
        _triton_block_dither_kernel_autotuned[grid](
            master,
            decode_scale,
            numel,
            int(start_offset),
            int(full_w),
            int(decode_scale.shape[1]),
            float(inv_global_scale),
            int(seed),
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
    has_te_nvfp4_sr = _has_te_nvfp4_sr_kernels()

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
        if has_te_nvfp4_sr:
            tex.nvfp4_multi_tensor_compute_partial_amax(
                master_weight_list,
                partial_amax_list,
                global_amax_list,
                h_list,
                w_list,
                start_offset_list,
                block_len,
            )
        else:
            for master, partial_amax, global_amax, h, w, start_offset in zip(
                master_weight_list,
                partial_amax_list,
                global_amax_list,
                h_list,
                w_list,
                start_offset_list,
            ):
                _compute_partial_amax_fallback(
                    master, partial_amax, h, w, start_offset
                )
                global_amax.copy_(master.abs().max().view(1))

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
    if has_te_nvfp4_sr:
        tex.nvfp4_compute_global_scale(global_amaxes, global_scale_tensor)
    else:
        _compute_global_scale_fallback(global_amaxes, global_scale_tensor)
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

    # Per-tensor ingredients needed by the dither + cast path. We keep the
    # original master shard undithered for ECO, and use a one-at-a-time scratch
    # shard for SR so peak memory only includes the largest cast shard.
    dither_master_list: List[torch.Tensor] = []
    dither_out_list: List[torch.Tensor] = []
    dither_scale_list: List[torch.Tensor] = []
    dither_global_scale_list: List[torch.Tensor] = []
    dither_decode_scale_list: List[torch.Tensor] = []
    dither_inv_global_scale_list: List[float] = []
    dither_h_list: List[int] = []
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

            dither_master_list.append(master_weight)
            dither_out_list.append(model_weight_fragment)
            dither_scale_list.append(per_block_decode_scale)
            dither_global_scale_list.append(global_scale)
            dither_decode_scale_list.append(per_block_decode_scale)
            dither_inv_global_scale_list.append(1.0 / float(global_scale.item()))
            dither_h_list.append(h)
            dither_full_w_list.append(w)
            dither_start_offset_list.append(start_offset)

    # Convert FP32 per-block decode scale → E4M3 target scale (and update target_amax).
    if fused_scale_block_amax_list:
        if has_te_nvfp4_sr:
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
        else:
            for (
                block_amax,
                global_amax,
                per_block_scale,
                target_scale,
                target_amax,
                tile_rows,
                tile_cols,
                rows_padded,
            ) in zip(
                fused_scale_block_amax_list,
                fused_scale_global_amax_list,
                fused_scale_per_block_scale_list,
                fused_scale_target_scale_list,
                fused_scale_target_amax_list,
                fused_scale_tile_rows_list,
                fused_scale_tile_cols_list,
                fused_scale_rows_padded_list,
            ):
                _fused_scale_fallback(
                    block_amax,
                    global_amax,
                    per_block_scale,
                    target_scale,
                    target_amax,
                    tile_rows,
                    tile_cols,
                    rows_padded,
                )

    # ---- Insert stochastic-rounding dither on scratch shards and cast them.
    base_seed = _next_dither_seed()
    dither_coef = float(os.getenv("MEGATRON_NVFP4_SR_DITHER_COEF", "0.25"))
    for i, (
        master,
        out,
        scale,
        global_scale,
        dscale,
        inv_gs,
        h,
        full_w,
        offset,
    ) in enumerate(
        zip(
            dither_master_list,
            dither_out_list,
            dither_scale_list,
            dither_global_scale_list,
            dither_decode_scale_list,
            dither_inv_global_scale_list,
            dither_h_list,
            dither_full_w_list,
            dither_start_offset_list,
        )
    ):
        # ECO needs the undithered updated weight for error injection:
        #   error = theta - q_sr(theta)
        # Apply dither to a transient cast shard only, then discard it after TE
        # packs it into the NVFP4 model weight.
        sr_cast_weight = master.clone()
        _apply_block_dither(
            master=sr_cast_weight,
            decode_scale=dscale,
            inv_global_scale=inv_gs,
            full_w=full_w,
            start_offset=offset,
            seed=(base_seed + i) & 0x7FFFFFFF,
            dither_coef=dither_coef,
        )
        if has_te_nvfp4_sr:
            tex.nvfp4_multi_tensor_2d_partial_cast(
                [sr_cast_weight],
                [out],
                [scale],
                [global_scale],
                [h],
                [full_w],
                [offset],
                block_len,
            )
        else:
            _partial_cast_fallback(
                sr_cast_weight,
                out,
                scale,
                global_scale,
                h,
                full_w,
                offset,
            )
        del sr_cast_weight
