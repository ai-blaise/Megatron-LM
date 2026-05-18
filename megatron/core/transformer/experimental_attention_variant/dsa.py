# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import math
import os
from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.models.common.embeddings.rope_utils import fused_apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_triton import (
    dsa_indexer_scores_triton,
    hisa_kl_loss_and_grad_triton,
    hisa_attention_target_probs_triton,
    is_hisa_kl_grad_triton_supported,
    is_hisa_attention_target_probs_triton_supported,
    is_dsa_indexer_scores_triton_supported,
    is_sparse_dsa_split_qk_triton_supported,
    is_sparse_dsa_teacher_triton_supported,
    is_sparse_dsa_triton_supported,
    sparse_dsa_attention_split_qk_triton,
    sparse_dsa_attention_split_qk_with_teacher_triton,
    sparse_dsa_attention_triton,
    sparse_dsa_attention_with_teacher_triton,
)
from megatron.core.quantization.indexcache import (
    INDEXCACHE_QUANT_NVFP4,
    IndexCacheHISAConfig,
    indexcache_hisa_cuda_select_scores_teacher,
    indexcache_hisa_select_with_scores,
    indexcache_hisa_topk,
    indexcache_hisa_topk_with_scores,
)
from megatron.core.extensions.hisa_indexer import (
    apply_hisa_score_backward,
    hisa_selector_forward_and_save,
)
from megatron.core.extensions.hisa_indexer.reference import (
    hisa_unselect_grad_from_topk_to_scores,
)

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None

_TE_DEQUANTIZABLE_TENSOR_TYPES = ()
try:
    from transformer_engine.pytorch.tensor import QuantizedTensor as _TEQuantizedTensor

    _TE_DEQUANTIZABLE_TENSOR_TYPES += (_TEQuantizedTensor,)
except (ImportError, ModuleNotFoundError):
    pass

try:
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor as _TENVFP4Tensor

    _TE_DEQUANTIZABLE_TENSOR_TYPES += (_TENVFP4Tensor,)
except (ImportError, ModuleNotFoundError):
    pass

try:
    from transformer_engine.pytorch.float8_tensor import Float8Tensor as _TEFloat8Tensor

    _TE_DEQUANTIZABLE_TENSOR_TYPES += (_TEFloat8Tensor,)
except (ImportError, ModuleNotFoundError):
    pass


_DSA_STREAMING_INDEXER_TOPK_ENV = "MEGATRON_DSA_STREAMING_INDEXER_TOPK"
_DSA_INDEXER_KEY_BLOCK_SIZE_ENV = "MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE"
_DSA_SORT_TOPK_INDICES_ENV = "MEGATRON_DSA_SORT_TOPK_INDICES"
_DSA_COMPACT_TOPK_INDICES_ENV = "MEGATRON_DSA_COMPACT_TOPK_INDICES"
_DSA_STREAM_TRITON_ATTENTION_CHUNKS_ENV = "MEGATRON_DSA_STREAM_TRITON_ATTENTION_CHUNKS"
_DSA_VALIDATE_TOPK_INDICES_ENV = "MEGATRON_DSA_VALIDATE_TOPK_INDICES"
_HISA_TARGET_ROW_CHUNK_ENV = "MEGATRON_HISA_TARGET_ROW_CHUNK"
_HISA_FUSED_INDEXER_LOSS_ENV = "MEGATRON_HISA_FUSED_INDEXER_LOSS"
_HISA_ASSUME_SORTED_POSITIONS_ENV = "MEGATRON_HISA_ASSUME_SORTED_POSITIONS"
_HISA_FALLBACK_DENSE_IF_SHORT_ENV = "MEGATRON_HISA_FALLBACK_DENSE_IF_SHORT"
_DSA_CHUNK_INDEXER_PROJ_ENV = "MEGATRON_DSA_CHUNK_INDEXER_PROJ"
_DSA_SP_PROJECT_BEFORE_GATHER_ENV = "MEGATRON_DSA_SP_PROJECT_BEFORE_GATHER"
_DSA_INDEXER_ROPE_FUSION_ENV = "MEGATRON_DSA_INDEXER_ROPE_FUSION"
_DSA_INDEXER_ROPE_INPLACE_ENV = "MEGATRON_DSA_INDEXER_ROPE_INPLACE"
_DSA_INDEXER_TORCH_K_NORM_ENV = "MEGATRON_DSA_INDEXER_TORCH_K_NORM"
_DSA_DEBUG_SYNC_ENV = "MEGATRON_DSA_DEBUG_SYNC"


def _env_flag_enabled(name: str, default: str = "1") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _dsa_debug_sync(label: str, tensor: Optional[torch.Tensor] = None) -> None:
    if not _env_flag_enabled(_DSA_DEBUG_SYNC_ENV, "0"):
        return
    if not torch.cuda.is_available():
        return
    try:
        if tensor is not None and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        else:
            torch.cuda.synchronize()
    except Exception as exc:
        device = None if tensor is None else tensor.device
        shape = None if tensor is None else tuple(tensor.shape)
        dtype = None if tensor is None else tensor.dtype
        raise RuntimeError(
            f"DSA debug sync failed after {label}; device={device}, shape={shape}, dtype={dtype}"
        ) from exc


def _tensor_safe_for_custom_inplace(x: torch.Tensor) -> bool:
    """Return whether a custom autograd Function can legally mutate ``x``."""
    if x.requires_grad and x.is_leaf:
        return False
    if getattr(x, "_base", None) is not None:
        return False
    is_view = getattr(x, "_is_view", None)
    if callable(is_view) and is_view():
        return False
    return True


def _should_dequantize_tensor(x: torch.Tensor) -> bool:
    """Return whether ``x.dequantize()`` is a real quantized-tensor materialization.

    Plain dense torch tensors expose ``Tensor.dequantize`` too, but calling it on
    a differentiable tensor installs a NotImplemented autograd node. Only
    dequantize actual torch quantized tensors or TE quantized tensor wrappers.
    """

    if bool(getattr(x, "is_quantized", False)):
        return True
    return bool(_TE_DEQUANTIZABLE_TENSOR_TYPES) and isinstance(
        x, _TE_DEQUANTIZABLE_TENSOR_TYPES
    )


def _dequantize_tensor_if_needed(
    x: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    if not _should_dequantize_tensor(x):
        return x
    if dtype is None:
        return x.dequantize()
    try:
        return x.dequantize(dtype=dtype)
    except TypeError:
        return x.dequantize()


class _DSAIndexerRopeFunction(torch.autograd.Function):
    """Apply indexer RoPE to the PE prefix while writing the full head directly."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        pe_dim: int,
        mscale: float,
        interleaved: bool,
    ) -> torch.Tensor:
        ext = _try_load_hisa_cuda_ext()
        if ext is None or not hasattr(ext, "dsa_indexer_rope_fwd"):
            raise RuntimeError("DSA indexer RoPE CUDA extension is unavailable")

        x_contig = x.contiguous()
        aliases_input = x_contig.data_ptr() == x.data_ptr()
        use_inplace = (
            _env_flag_enabled(_DSA_INDEXER_ROPE_INPLACE_ENV)
            and hasattr(ext, "dsa_indexer_rope_fwd_inplace")
            and (not aliases_input or _tensor_safe_for_custom_inplace(x))
        )
        if use_inplace:
            if aliases_input:
                ctx.mark_dirty(x)
            ext.dsa_indexer_rope_fwd_inplace(
                x_contig,
                rotary_pos_emb,
                int(pe_dim),
                float(mscale),
                bool(interleaved),
            )
            out = x_contig
        else:
            out = torch.empty_like(x_contig)
            ext.dsa_indexer_rope_fwd(
                x_contig,
                rotary_pos_emb,
                out,
                int(pe_dim),
                float(mscale),
                bool(interleaved),
            )
        ctx.save_for_backward(rotary_pos_emb)
        ctx.pe_dim = int(pe_dim)
        ctx.mscale = float(mscale)
        ctx.interleaved = bool(interleaved)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (rotary_pos_emb,) = ctx.saved_tensors
        ext = _try_load_hisa_cuda_ext()
        if ext is None or not hasattr(ext, "dsa_indexer_rope_bwd"):
            raise RuntimeError("DSA indexer RoPE CUDA extension is unavailable")

        grad_out_contig = grad_out.contiguous()
        grad_x = torch.empty_like(grad_out_contig)
        ext.dsa_indexer_rope_bwd(
            grad_out_contig,
            rotary_pos_emb,
            grad_x,
            ctx.pe_dim,
            ctx.mscale,
            ctx.interleaved,
        )
        return grad_x, None, None, None, None


class _DSAIndexerFlatRopeInplaceFunction(torch.autograd.Function):
    """Apply indexer RoPE in-place before the projection tensor is viewed as heads."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        heads: int,
        head_dim: int,
        pe_dim: int,
        mscale: float,
        interleaved: bool,
    ) -> torch.Tensor:
        ext = _try_load_hisa_cuda_ext()
        if ext is None or not hasattr(ext, "dsa_indexer_rope_fwd_inplace_flat"):
            raise RuntimeError("DSA indexer flat RoPE CUDA extension is unavailable")

        if not x.is_contiguous():
            raise RuntimeError("DSA indexer flat RoPE in-place input must be contiguous")
        if not _tensor_safe_for_custom_inplace(x):
            raise RuntimeError("DSA indexer flat RoPE in-place input is not autograd-safe")
        ctx.mark_dirty(x)
        ext.dsa_indexer_rope_fwd_inplace_flat(
            x,
            rotary_pos_emb,
            int(heads),
            int(head_dim),
            int(pe_dim),
            float(mscale),
            bool(interleaved),
        )
        ctx.save_for_backward(rotary_pos_emb)
        ctx.heads = int(heads)
        ctx.head_dim = int(head_dim)
        ctx.pe_dim = int(pe_dim)
        ctx.mscale = float(mscale)
        ctx.interleaved = bool(interleaved)
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (rotary_pos_emb,) = ctx.saved_tensors
        ext = _try_load_hisa_cuda_ext()
        if ext is None or not hasattr(ext, "dsa_indexer_rope_bwd_flat"):
            raise RuntimeError("DSA indexer flat RoPE CUDA extension is unavailable")

        grad_out_contig = grad_out.contiguous()
        grad_x = torch.empty_like(grad_out_contig)
        ext.dsa_indexer_rope_bwd_flat(
            grad_out_contig,
            rotary_pos_emb,
            grad_x,
            ctx.heads,
            ctx.head_dim,
            ctx.pe_dim,
            ctx.mscale,
            ctx.interleaved,
        )
        return grad_x, None, None, None, None, None, None


def _dsa_indexer_key_block_size(topk: int) -> int:
    raw = os.getenv(_DSA_INDEXER_KEY_BLOCK_SIZE_ENV)
    if raw:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{_DSA_INDEXER_KEY_BLOCK_SIZE_ENV} must be positive, got {value}")
        return value
    return max(2048, min(4096, max(1, topk)))


def _hisa_target_row_chunk(q_len: int) -> int:
    raw = os.getenv(_HISA_TARGET_ROW_CHUNK_ENV)
    if raw:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{_HISA_TARGET_ROW_CHUNK_ENV} must be positive, got {value}")
        return min(value, q_len)
    return min(128, q_len)


def _dsa_topk_buffer_dtype(sk: int) -> torch.dtype:
    if _env_flag_enabled(_DSA_COMPACT_TOPK_INDICES_ENV, "0") and sk <= 32768:
        return torch.int16
    return torch.int32


def _maybe_narrow_dsa_topk_indices(topk_indices: torch.Tensor, sk: int) -> torch.Tensor:
    """Keep selected-token buffers off int64 in CUDA hot paths.

    PyTorch `topk` emits `long` indices, but the DSA Triton/CUDA kernels and
    current sequence lengths only need 32-bit selected-token IDs. Narrowing here
    also covers dense/streaming fallback selectors before they feed the shared
    loss and attention code. Do not narrow if a future caller actually exceeds
    int32 addressable key positions.
    """

    if topk_indices.dtype == torch.long and sk <= torch.iinfo(torch.int32).max:
        return topk_indices.to(torch.int32)
    return topk_indices


def _maybe_sort_dsa_topk_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    if not _env_flag_enabled(_DSA_SORT_TOPK_INDICES_ENV, "0"):
        return topk_indices
    return topk_indices.sort(dim=-1).values


def _maybe_sort_dsa_topk_indices_and_scores(
    topk_indices: torch.Tensor,
    selected_scores: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not _env_flag_enabled(_DSA_SORT_TOPK_INDICES_ENV, "0"):
        return topk_indices, selected_scores
    sorted_indices, order = topk_indices.sort(dim=-1)
    if selected_scores is None:
        return sorted_indices, None
    original_shape = topk_indices.shape
    sorted_scores = selected_scores.reshape(original_shape).gather(-1, order)
    return sorted_indices, sorted_scores.reshape_as(selected_scores)


def _dsa_process_group_size(group: Optional[torch.distributed.ProcessGroup]) -> int:
    return group.size() if group is not None else 1


def _dsa_process_group_rank(group: Optional[torch.distributed.ProcessGroup]) -> int:
    return group.rank() if group is not None else 0


def _in_te_no_grad_activation_recompute_forward() -> bool:
    """Return True in TE's checkpoint forward phase, before backward replay."""

    try:
        from transformer_engine.pytorch.distributed import (
            in_fp8_activation_recompute_phase,
            is_fp8_activation_recompute_enabled,
        )
    except (ImportError, ModuleNotFoundError):
        return False
    return (
        is_fp8_activation_recompute_enabled()
        and not in_fp8_activation_recompute_phase()
        and not torch.is_grad_enabled()
    )


class _BroadcastFromTensorParallelOwner(torch.autograd.Function):
    """Broadcast a compact sequence-parallel chunk and reduce its gradient to the owner."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, owner_global_rank: int, group):
        ctx.owner_global_rank = int(owner_global_rank)
        ctx.group = group
        output = torch.empty_like(input_)
        if torch.distributed.get_rank() == ctx.owner_global_rank:
            output.copy_(input_)
        torch.distributed.broadcast(output, src=ctx.owner_global_rank, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.contiguous()
        torch.distributed.reduce(
            grad_input,
            dst=ctx.owner_global_rank,
            op=torch.distributed.ReduceOp.SUM,
            group=ctx.group,
        )
        if torch.distributed.get_rank() != ctx.owner_global_rank:
            grad_input.zero_()
        return grad_input, None, None


def _torch_layer_norm_like_te(layer_norm: torch.nn.Module, x: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply a torch LayerNorm equivalent for TE LayerNorm modules."""

    output_dtype = x.dtype
    weight = getattr(layer_norm, "weight", None)
    bias = getattr(layer_norm, "bias", None)
    if weight is None:
        raise AttributeError("LayerNorm fallback requires a weight parameter")

    x = _dequantize_tensor_if_needed(x)

    zero_centered_gamma = bool(getattr(layer_norm, "zero_centered_gamma", False))
    if zero_centered_gamma:
        weight = weight + 1
    weight = weight.to(dtype=x.dtype)
    if bias is not None:
        bias = bias.to(dtype=x.dtype)

    output = F.layer_norm(
        x,
        (x.size(-1),),
        weight=weight,
        bias=bias,
        eps=eps,
    )
    return output.to(dtype=output_dtype)


def _dsa_cp_position_ids_for_rank(
    local_seq_len: int, cp_size: int, cp_rank: int, device: torch.device
) -> torch.Tensor:
    """Return absolute sequence positions for one CP rank's zigzag-local tokens."""

    if cp_size <= 1:
        return torch.arange(local_seq_len, device=device, dtype=torch.long)
    if local_seq_len % 2 != 0:
        raise ValueError(
            f"DSA CP position mapping expects an even local sequence length, got {local_seq_len}"
        )

    chunk_len = local_seq_len // 2
    first = torch.arange(
        cp_rank * chunk_len, (cp_rank + 1) * chunk_len, device=device, dtype=torch.long
    )
    second_chunk = 2 * cp_size - cp_rank - 1
    second = torch.arange(
        second_chunk * chunk_len,
        (second_chunk + 1) * chunk_len,
        device=device,
        dtype=torch.long,
    )
    return torch.cat((first, second), dim=0)


def _dsa_cp_local_position_ids(
    local_seq_len: int, cp_group: Optional[torch.distributed.ProcessGroup], device: torch.device
) -> torch.Tensor:
    return _dsa_cp_position_ids_for_rank(
        local_seq_len,
        _dsa_process_group_size(cp_group),
        _dsa_process_group_rank(cp_group),
        device,
    )


def _dsa_cp_gathered_position_ids(
    local_seq_len: int, cp_group: Optional[torch.distributed.ProcessGroup], device: torch.device
) -> torch.Tensor:
    cp_size = _dsa_process_group_size(cp_group)
    if cp_size <= 1:
        return torch.arange(local_seq_len, device=device, dtype=torch.long)
    return torch.cat(
        [
            _dsa_cp_position_ids_for_rank(local_seq_len, cp_size, cp_rank, device)
            for cp_rank in range(cp_size)
        ],
        dim=0,
    )


def _dsa_thd_local_sequence_offsets(
    cu_seqlens: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
) -> list[tuple[int, int]]:
    """Return local packed-THD offsets for each sequence on this CP rank."""

    cp_size = _dsa_process_group_size(cp_group)
    offsets = cu_seqlens.detach().cpu().tolist()
    local_offsets = []
    local_cursor = 0
    for start, end in zip(offsets[:-1], offsets[1:]):
        seq_len = int(end) - int(start)
        if seq_len <= 0:
            local_offsets.append((local_cursor, local_cursor))
            continue
        if seq_len % cp_size != 0:
            raise ValueError(
                f"DSA packed THD CP expects each padded sequence length to be divisible by "
                f"context_parallel_size; got sequence length {seq_len} and CP {cp_size}"
            )
        local_len = seq_len // cp_size
        local_offsets.append((local_cursor, local_cursor + local_len))
        local_cursor += local_len
    return local_offsets


def _dsa_thd_tp_local_sequence_offsets(
    cu_seqlens: torch.Tensor,
    local_seq_len: int,
    tp_group: Optional[torch.distributed.ProcessGroup],
) -> list[tuple[int, int]]:
    """Return local packed-THD offsets for TP sequence-parallel shards.

    Packed THD cu_seqlens describe the global packed token stream. With TP
    sequence parallelism, each rank owns a contiguous token slice from every
    packed sequence, not a whole-sequence slice selected by TP rank. Rebase the
    global per-sequence lengths onto the local packed stream so recursion never
    slices past the rank-local key/value tensors.
    """

    tp_size = _dsa_process_group_size(tp_group)
    if tp_size <= 1:
        offsets = cu_seqlens.detach().cpu().tolist()
        return [(int(start), int(end)) for start, end in zip(offsets[:-1], offsets[1:])]

    global_offsets = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    local_offsets = []
    local_cursor = 0
    for start, end in zip(global_offsets[:-1], global_offsets[1:]):
        seq_len = int(end) - int(start)
        if seq_len <= 0:
            local_offsets.append((local_cursor, local_cursor))
            continue
        if seq_len % tp_size != 0:
            raise ValueError(
                f"DSA packed THD TP expects each padded sequence length to be divisible by "
                f"tensor_model_parallel_size; got sequence length {seq_len} and TP {tp_size}"
            )
        local_len = seq_len // tp_size
        local_offsets.append((local_cursor, local_cursor + local_len))
        local_cursor += local_len
    if local_seq_len >= 0 and local_cursor > local_seq_len:
        raise ValueError(
            f"DSA packed THD TP local offsets exceed local tensor length: "
            f"offsets end at {local_cursor}, local length {local_seq_len}"
        )
    return local_offsets


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    x = _dequantize_tensor_if_needed(x, dtype=torch.bfloat16)
    if x.dtype != torch.bfloat16:
        x = x.to(dtype=torch.bfloat16)
    if x.numel() == 0:
        return x
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    hidden_size = x.size(-1)
    return hadamard_transform(x.contiguous(), scale=hidden_size**-0.5)


class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the indexer loss for logging.

        Args:
            loss: The loss tensor.
            layer_number: Layer index of the loss, 1-indexed.
            num_layers: The number of total layers.
            reduce_group: The group for reducing the loss.
            avg_group: The group for averaging the loss.
        """
        # Skip indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker():
        """Collect and reduce the indexer losses across ranks."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]

        torch.distributed.all_reduce(
            values, group=parallel_state.get_pipeline_model_parallel_group()
        )
        # Reduce indexer losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )
        torch.distributed.all_reduce(
            values,
            group=parallel_state.get_data_parallel_group(with_context_parallel=False),
            op=torch.distributed.ReduceOp.AVG,
        )

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer,
        wandb_writer=None,
        total_loss_dict=None,
        per_layer_logging: bool = False,
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            wandb_writer: Weights & Biases writer.
            total_loss_dict: Dictionary to accumulate total losses.
            per_layer_logging: Whether to log per-layer losses.
        """
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        indexer_loss_values = tracker["values"] * loss_scale
        num_layers = indexer_loss_values.shape[0]

        # Average across all layers (assuming all layers have sparse attention)
        avg_indexer_loss = indexer_loss_values.sum() / num_layers

        # Log average loss
        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss

        if writer is not None:
            writer.add_scalar("indexer loss", avg_indexer_loss, iteration)

        if wandb_writer is not None:
            wandb_writer.log({"indexer loss": avg_indexer_loss}, iteration)

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


class DSAIndexerAuxLossState:
    """Track per-microbatch DSA auxiliary losses for explicit loss composition."""

    losses = []

    @classmethod
    def add(cls, loss: torch.Tensor):
        cls.losses.append(loss)

    @classmethod
    def total(cls) -> Optional[torch.Tensor]:
        if not cls.losses:
            return None
        return torch.stack(cls.losses).sum()

    @classmethod
    def clear(cls):
        cls.losses.clear()


def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    sq, b, np, hn = query.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += causal_mask.view(1, 1, sq, sk)
    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    # [b, sq, sk] -> [b, sq, sk]
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    # L1 normalize target on the last dimension. Doesn't use abs() because attention_scores are
    # obtained from softmax so they are already non-negative.
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

    # Compute KL divergence: KL(target || index) = target(x) * log(target(x) / index(x))
    # kl_per_element [b, sq, sk]
    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )

    # [b, sq, sk] -> [b, sq] -> [1]
    # Each token has same weight in the loss.
    kl_div = kl_per_element.sum(dim=-1).mean()

    # Scale by coefficient.
    indexer_loss = kl_div * loss_coeff

    return indexer_loss


def _compute_index_scores(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Perform index score using BF16 precision.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
    This is a BF16 implementation of the `fp8_index` logic:
        1. Compute attention scores: q @ k^T;
        2. Apply ReLU activation;
        3. Weight by attention weights;
        4. Sum across attention heads.

    Args:
        q: BF16 [seqlen_q, batch, index_n_heads, index_head_dim], the query tensor.
        weights: BF16 [seqlen_q, batch, index_n_heads], the attention weights.
        k: BF16 [seqlen_k, batch, index_head_dim], the key tensor.

    Returns:
        index_scores: FP32 [batch, seqlen_q, seqlen_k], the index scores.
    """
    # Compute attention scores: q @ k^T
    # [seqlen_q, batch, index_n_heads, index_head_dim] @ [seqlen_k, batch, index_head_dim]^T
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

    # Keep this out-of-place: when the DSA KL loss is enabled, autograd needs
    # the ReLU output version to remain stable through the weight multiply.
    index_scores = torch.relu(index_scores)

    # Weight each head by attention weights.
    # [seqlen_q, batch, index_n_heads, seqlen_k] * [seqlen_q, batch, index_n_heads, 1]
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = index_scores * weights.unsqueeze(-1)

    # Sum across attention heads.
    # [seqlen_q, batch, index_n_heads, seqlen_k] -> [seqlen_q, batch, seqlen_k]
    index_scores = index_scores.sum(dim=2)

    # Transpose to [batch, seqlen_q, seqlen_k].
    index_scores = index_scores.transpose(0, 1)

    return index_scores


def fused_qk_topk_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    mask: Optional[torch.Tensor] = None,
):
    """Naive implementation of QK Topk."""
    seqlen = q.size(0)
    # =========================================
    # Compute index scores
    # =========================================
    # [batch, seqlen, seqlen]
    index_scores = _compute_index_scores(q, weights, k)
    if mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    # =========================================
    # Select top-k indices
    # =========================================
    topk_k = min(index_topk, seqlen)
    # [batch, seqlen, index_topk]
    # Keep this legacy fused-loss helper deterministic and aligned with the
    # autograd reference tests. The production chunked/HISA path keeps its own
    # sorted=False top-k where order is not part of the sparse attention semantics.
    topk_indices = index_scores.topk(topk_k, dim=-1, sorted=True)[1]

    return index_scores, topk_indices


def fwd_fused_indexer_loss_naive(
    q, weights, k, query, key, topk, softmax_scale, loss_coeff, mask, sparse_loss, pg_collection
):
    """Naive implementation of forward pass for indexer loss."""
    index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, topk, mask)

    indexer_loss = compute_dsa_indexer_loss(
        index_scores,
        topk_indices,
        query,
        key,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        pg_collection,
    )

    return topk_indices, indexer_loss


def bwd_fused_indexer_loss_naive(
    q,
    weights,
    k,
    query,
    key,
    topk_indices,
    softmax_scale,
    loss_coeff,
    sparse_loss,
    grad_loss,
    pg_collection,
):
    """Naive implementation of backward pass for indexer loss."""
    index_scores = _compute_index_scores(q, weights, k)  # [B, Sq, Sk]

    sq, b, np, hn = query.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query_reshaped = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key_reshaped = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query_reshaped.float(), key_reshaped.float()) * softmax_scale
    # Free reshaped tensors - no longer needed after bmm
    del query_reshaped, key_reshaped

    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # Apply causal mask to both attention and index scores
    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores = attention_scores + causal_mask.view(1, 1, sq, sk)
    # [b, sq, sk] + [1, sq, sk] -> [b, sq, sk]
    index_scores = index_scores + causal_mask.unsqueeze(0)
    # Free causal_mask - no longer needed
    del causal_mask

    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_mask

    # Compute softmax for both
    attention_scores_softmax = torch.nn.functional.softmax(
        attention_scores, dim=-1, dtype=torch.float32
    )
    # Free attention_scores immediately
    del attention_scores

    index_scores_softmax = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)
    # Free index_scores - no longer needed after softmax
    del index_scores

    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    # Free attention_scores_softmax
    del attention_scores_softmax

    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores_sum.contiguous(), group=pg_collection.tp)

    # L1 normalize
    attention_scores_normalized = attention_scores_sum / attention_scores_sum.sum(
        dim=-1, keepdim=True
    )
    # Free attention_scores_sum - no longer needed after normalization
    del attention_scores_sum

    # Backward through loss = kl_div * loss_coeff
    # where kl_div = kl_per_element.sum(dim=-1).mean()
    grad_kl_div = grad_loss * loss_coeff  # scalar

    # Backward through mean: distribute gradient equally
    grad_kl_per_row = grad_kl_div / (b * sq)  # scalar value for each row

    # Backward through sum(dim=-1): broadcast back to [b, sq, sk]
    # Each element in a row contributes to the sum, so gradient is same for all
    grad_kl_per_element = grad_kl_per_row.view(1, 1, 1).expand(b, sq, sk)

    # Backward through kl_per_element = target * (log(target) - log(index))
    # ∂kl/∂index_softmax = -target / index_softmax
    grad_index_scores_softmax = (
        -attention_scores_normalized / (index_scores_softmax + 1e-10) * grad_kl_per_element
    )
    # Free attention_scores_normalized - no longer needed
    del attention_scores_normalized

    # Backward through softmax: ∂L/∂x = softmax * (∂L/∂softmax - sum(∂L/∂softmax * softmax))
    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(dim=-1, keepdim=True)
    grad_index_scores_logits = index_scores_softmax * (grad_index_scores_softmax - sum_grad)
    # Free intermediate tensors
    del index_scores_softmax, grad_index_scores_softmax, sum_grad

    # Zero out gradients for masked positions
    # Create a mask for valid (non-masked) positions
    # Causal mask: position (i, j) is valid if j <= i
    causal_valid_mask = torch.tril(
        torch.ones((sq, sk), device=q.device, dtype=torch.bool)
    )  # [sq, sk]
    if sparse_loss:
        # Also apply index mask - only topk positions are valid
        index_valid_mask = index_mask == 0  # [b, sq, sk]
        del index_mask  # Free index_mask immediately after use
        valid_mask = causal_valid_mask.unsqueeze(0) & index_valid_mask  # [b, sq, sk]
        del index_valid_mask
    else:
        del index_mask  # Free index_mask even if not used for sparse_loss
        valid_mask = causal_valid_mask.unsqueeze(0).expand(b, sq, sk)  # [b, sq, sk]
    del causal_valid_mask

    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    del valid_mask

    # Transpose from [b, sq, sk] to [sq, b, sk]
    grad_index_scores = grad_index_scores_logits.transpose(0, 1)  # [sq, b, sk]
    del grad_index_scores_logits

    # Backward through sum over heads: expand gradient
    grad_weighted_scores = grad_index_scores.unsqueeze(2)  # [sq, b, 1, sk]
    del grad_index_scores

    # Compute forward values needed for backward
    scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())  # [sq, b, h, sk]
    # Compute relu_mask before relu (saves memory vs keeping both scores and relu output)
    relu_mask = scores > 0
    scores_after_relu = torch.relu(scores)
    del scores

    # Backward through multiplication by weights: index_scores_per_head * weights
    # ∂L/∂weights = grad * relu_scores (sum over sk)
    grad_weights = (grad_weighted_scores * scores_after_relu).sum(dim=-1)  # [sq, b, h]

    # ∂L/∂relu_scores = grad * weights
    grad_scores_after_relu = grad_weighted_scores * weights.unsqueeze(-1)  # [sq, b, h, sk]
    del grad_weighted_scores, scores_after_relu

    # Backward through ReLU
    grad_scores = grad_scores_after_relu * relu_mask.float()  # [sq, b, h, sk]
    del grad_scores_after_relu, relu_mask

    # Backward through einsum 'sbhd,tbd->sbht'
    # ∂L/∂q = einsum('sbht,tbd->sbhd', grad_scores, k)
    grad_q = torch.einsum('sbht,tbd->sbhd', grad_scores, k.float())  # [sq, b, h, d]
    # ∂L/∂k = einsum('sbht,sbhd->tbd', grad_scores, q)
    grad_k = torch.einsum('sbht,sbhd->tbd', grad_scores, q.float())  # [sk, b, d]
    del grad_scores

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


class FusedDSAIndexerLoss(torch.autograd.Function):
    """Fused implementation of DSA Indexer Loss."""

    @staticmethod
    def forward(
        ctx,
        q,
        weights,
        k,
        query,
        key,
        softmax_scale,
        topk,
        loss_coeff,
        mask,
        sparse_loss,
        pg_collection,
    ):
        """
        Fused forward: index_scores never materialized in full.
        """
        topk_indices, loss = fwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk,
            softmax_scale,
            loss_coeff,
            mask,
            sparse_loss,
            pg_collection,
        )

        # Save for backward (recomputation strategy)
        ctx.save_for_backward(q, weights, k, query, key, topk_indices)
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        ctx.pg_collection = pg_collection

        return topk_indices, loss

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss):
        """
        Backward: Recompute what we need.
        """
        q, weights, k, query, key, topk_indices = ctx.saved_tensors

        grad_q, grad_weights, grad_k = bwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            ctx.softmax_scale,
            ctx.loss_coeff,
            ctx.sparse_loss,
            grad_loss,
            ctx.pg_collection,
        )

        # query and key are detached in forward, so return None for their gradients
        return grad_q, grad_weights, grad_k, None, None, None, None, None, None, None, None


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for indexer loss.

    This custom autograd function attaches a KL divergence loss to the activation
    to train the indexer to predict attention scores without affecting the forward pass.
    """

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output: The output tensor (activation).
            indexer_loss: The indexer KL divergence loss tensor.

        Returns:
            torch.Tensor: The output tensor unchanged.
        """
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output: The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                gradient.
        """
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale: The scale value to set.
        """
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


@dataclass
class DSAIndexerSubmodules:
    """
    Configuration class for specifying the submodules of an DSA Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_wk: Linear projection for key.
        k_norm: Layer normalization for key.
        linear_weights_proj: Linear projection for attention weights.
    """

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


@dataclass
class DSAttentionSubmodules:
    """
    Configuration class for specifying the submodules of DSAttention.

    Args:
        indexer: DSA Indexer module for computing sparse attention indices.
    """

    indexer: Union[ModuleSpec, type] = None


class DSAIndexer(MegatronModule):
    """
    DSA Lightning Indexer for DeepSeek Sparse Attention.

    Computes index scores to identify the top-k most relevant key-value pairs for each query in
    sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            submodules (DSAIndexerSubmodules): Indexer submodules specification.
            pg_collection (ProcessGroupCollection, optional): Process groups for the indexer.
        """
        super().__init__(config=config)
        self._indexer_rope_config = copy.copy(self.config)
        self._indexer_rope_config.apply_rope_fusion = False
        self.hidden_size = self.config.hidden_size
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            self.config.q_lora_rank
            if self.config.q_lora_rank is not None
            else self.config.hidden_size
        )

        self.index_n_heads = self.config.dsa_indexer_n_heads
        self.index_head_dim = self.config.dsa_indexer_head_dim
        self.index_topk = self.config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        self.indexcache_config = None
        self.indexcache_hisa_config = None
        indexcache_quantization = getattr(
            self.config, "dsa_indexcache_quantization", "disabled"
        )
        indexcache_quantization_enabled = getattr(
            self.config, "dsa_indexcache_quant_enabled", False
        )
        if indexcache_quantization_enabled or indexcache_quantization != "disabled":
            from megatron.core.quantization.indexcache import (
                build_indexcache_config,
                resolve_indexcache_quantization,
            )

            self.indexcache_config = build_indexcache_config(
                eps=getattr(self.config, "dsa_indexcache_quant_eps", 1e-4),
                quantization=resolve_indexcache_quantization(
                    quantization=indexcache_quantization,
                    quant_enabled=indexcache_quantization_enabled,
                ),
            )
        if getattr(self.config, "dsa_indexcache_hisa_enabled", False):
            if (
                self.indexcache_config is None
                or self.indexcache_config.quantization != INDEXCACHE_QUANT_NVFP4
            ):
                raise ValueError(
                    "dsa_indexcache_hisa_enabled requires "
                    "dsa_indexcache_quantization='nvfp4_e2m1_ue8m0'."
                )
            self.indexcache_hisa_config = IndexCacheHISAConfig(
                enabled=True,
                block_size=int(getattr(self.config, "dsa_indexcache_hisa_block_size", 128)),
                block_topk=int(getattr(self.config, "dsa_indexcache_hisa_block_topk", 64)),
                compression_ratio=float(
                    getattr(self.config, "dsa_indexcache_hisa_compression_ratio", 4.0)
                ),
                fallback_to_dense_if_short=_env_flag_enabled(
                    _HISA_FALLBACK_DENSE_IF_SHORT_ENV, "0"
                ),
            )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        # Initialize Position Embedding.
        if self.config.rope_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f'Unsupported RoPE type: {self.config.rope_type}, supported types are "rope" and '
                f'"yarn"'
            )

        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_wk = build_module(
            submodules.linear_wk,
            self.hidden_size,
            self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        k_norm_config = copy.copy(self.config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            config=k_norm_config,
            hidden_size=self.index_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _apply_rope(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor, mscale: float):
        """Apply RoPE to the input tensor."""
        if self._use_fused_indexer_rope_writer(x, rotary_pos_emb):
            return _DSAIndexerRopeFunction.apply(
                x,
                rotary_pos_emb,
                self.qk_pos_emb_head_dim,
                float(mscale),
                bool(self._indexer_rope_config.rotary_interleaved),
            )

        # x_pe   [seqlen, batch, *, qk_pos_emb_head_dim]
        # x_nope [seqlen, batch, *, index_head_dim - qk_pos_emb_head_dim]
        # To align with DeepSeek's implementation,
        # x_pe is placed at the front, and x_nope is placed at the back.
        x_pe, x_nope = torch.split(
            x, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )
        if self._use_fused_indexer_rope(x_pe, rotary_pos_emb):
            x_pe = fused_apply_rotary_pos_emb(
                x_pe,
                rotary_pos_emb,
                interleaved=self._indexer_rope_config.rotary_interleaved,
            )
            if mscale != 1.0:
                x_pe = x_pe * mscale
        else:
            x_pe = apply_rotary_pos_emb(
                x_pe,
                rotary_pos_emb,
                config=self._indexer_rope_config,
                cu_seqlens=None,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
                # This flag is for the MLA-style interleaving in RoPE.
                # Set it to False, as indexer does not apply interleaved RoPE.
                mla_rotary_interleaved=False,
            )
        # [seqlen, batch, *, index_head_dim]
        x = torch.cat([x_pe, x_nope], dim=-1)
        return x

    def _use_fused_indexer_flat_rope_writer(
        self,
        x: torch.Tensor,
        rotary_pos_emb,
        heads: int,
        head_dim: int,
    ) -> bool:
        """Return whether flat projected indexer RoPE can run before reshape."""
        if not _env_flag_enabled(_DSA_INDEXER_ROPE_FUSION_ENV, "1"):
            return False
        if not _env_flag_enabled(_DSA_INDEXER_ROPE_INPLACE_ENV, "1"):
            return False
        if isinstance(rotary_pos_emb, tuple) or rotary_pos_emb is None:
            return False
        if x.dim() != 3 or rotary_pos_emb.dim() != 4:
            return False
        if not x.is_contiguous():
            return False
        if not _tensor_safe_for_custom_inplace(x):
            return False
        if not x.is_cuda or not rotary_pos_emb.is_cuda:
            return False
        if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return False
        if rotary_pos_emb.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return False
        if x.is_leaf and x.requires_grad:
            return False
        if self._indexer_rope_config.mrope_section is not None and rotary_pos_emb.shape[1] > 1:
            return False
        if heads <= 0 or head_dim != self.index_head_dim:
            return False
        if x.size(-1) != heads * head_dim:
            return False
        if self.qk_pos_emb_head_dim <= 0 or self.qk_pos_emb_head_dim > head_dim:
            return False
        if self.qk_pos_emb_head_dim % 2 != 0:
            return False
        if rotary_pos_emb.size(-1) != self.qk_pos_emb_head_dim:
            return False
        if rotary_pos_emb.size(0) not in (1, x.size(0)):
            return False
        if rotary_pos_emb.size(1) not in (1, x.size(1)):
            return False
        if rotary_pos_emb.size(2) not in (1, heads):
            return False
        ext = _try_load_hisa_cuda_ext()
        return ext is not None and hasattr(ext, "dsa_indexer_rope_fwd_inplace_flat")

    def _use_fused_indexer_rope_writer(self, x: torch.Tensor, rotary_pos_emb) -> bool:
        """Return whether DSA indexer RoPE can write the full head in one CUDA call."""
        if not _env_flag_enabled(_DSA_INDEXER_ROPE_FUSION_ENV, "1"):
            return False
        if isinstance(rotary_pos_emb, tuple) or rotary_pos_emb is None:
            return False
        if x.dim() != 4 or rotary_pos_emb.dim() != 4:
            return False
        if not x.is_cuda or not rotary_pos_emb.is_cuda:
            return False
        if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return False
        if rotary_pos_emb.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            return False
        if self._indexer_rope_config.mrope_section is not None and rotary_pos_emb.shape[1] > 1:
            return False
        if x.size(-1) != self.index_head_dim:
            return False
        if self.qk_pos_emb_head_dim <= 0 or self.qk_pos_emb_head_dim > self.index_head_dim:
            return False
        if self.qk_pos_emb_head_dim % 2 != 0:
            return False
        if rotary_pos_emb.size(-1) != self.qk_pos_emb_head_dim:
            return False
        if rotary_pos_emb.size(0) not in (1, x.size(0)):
            return False
        if rotary_pos_emb.size(1) not in (1, x.size(1)):
            return False
        if rotary_pos_emb.size(2) not in (1, x.size(2)):
            return False
        ext = _try_load_hisa_cuda_ext()
        return ext is not None and hasattr(ext, "dsa_indexer_rope_fwd")

    def _use_fused_indexer_rope(self, x_pe: torch.Tensor, rotary_pos_emb) -> bool:
        """Return whether DSA indexer RoPE can use TE's fused BSHD RoPE kernel."""
        if not _env_flag_enabled(_DSA_INDEXER_ROPE_FUSION_ENV, "1"):
            return False
        if not self.config.apply_rope_fusion:
            return False
        if fused_apply_rotary_pos_emb is None:
            return False
        if isinstance(rotary_pos_emb, tuple) or rotary_pos_emb is None:
            return False
        if x_pe.dim() != 4 or rotary_pos_emb.dim() != 4:
            return False
        if not x_pe.is_cuda or not rotary_pos_emb.is_cuda:
            return False
        if self._indexer_rope_config.mrope_section is not None and rotary_pos_emb.shape[1] > 1:
            return False
        return x_pe.size(-1) == rotary_pos_emb.size(-1)

    @staticmethod
    def _slice_rotary_pos_emb(rotary_pos_emb, start: int, end: int):
        if isinstance(rotary_pos_emb, tuple):
            return tuple(None if item is None else item[start:end] for item in rotary_pos_emb)
        return rotary_pos_emb[start:end]

    @staticmethod
    def _index_rotary_pos_emb(rotary_pos_emb, indices: torch.Tensor):
        if isinstance(rotary_pos_emb, tuple):
            return tuple(
                None if item is None else item.index_select(0, indices.to(item.device))
                for item in rotary_pos_emb
            )
        return rotary_pos_emb.index_select(0, indices.to(rotary_pos_emb.device))

    def _prepare_inputs_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
        gather_sequence_parallel: bool = True,
    ):
        """Prepare shared DSA indexer inputs before per-query projection/top-k work."""
        orig_seqlen = x.size(0)
        pad_len = 0
        if self.config.fp4 and packed_seq_params is None:
            # TE NVFP4 kernels require the flattened token dimension to be divisible
            # by 16. Packed THD recursion can hand the indexer arbitrary per-sample
            # lengths, so pad the private indexer projections and trim before use.
            bsz = x.size(1)
            while ((orig_seqlen + pad_len) * bsz) % 16 != 0:
                pad_len += 1
            if pad_len:
                x = torch.cat(
                    (x, x.new_zeros((pad_len, bsz, x.size(2)))),
                    dim=0,
                )
                qr = torch.cat(
                    (qr, qr.new_zeros((pad_len, bsz, qr.size(2)))),
                    dim=0,
                )

        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

        if (
            gather_sequence_parallel
            and self.config.sequence_parallel
            and self.pg_collection.tp.size() > 1
        ):
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        return x, qr, rotary_pos_emb, mscale, orig_seqlen, pad_len

    def _project_query_before_topk(
        self,
        query_x: torch.Tensor,
        query_qr: torch.Tensor,
        query_rotary_pos_emb,
        mscale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project one query span into DSA indexer q and score weights."""
        query_seqlen, bsz, _ = query_qr.size()
        q, _ = self.linear_wq_b(query_qr)
        _dsa_debug_sync("indexer query linear_wq_b", q)
        if self._use_fused_indexer_flat_rope_writer(
            q, query_rotary_pos_emb, self.index_n_heads, self.index_head_dim
        ):
            q = _DSAIndexerFlatRopeInplaceFunction.apply(
                q,
                query_rotary_pos_emb,
                self.index_n_heads,
                self.index_head_dim,
                self.qk_pos_emb_head_dim,
                float(mscale),
                bool(self._indexer_rope_config.rotary_interleaved),
            )
            q = q.reshape(query_seqlen, bsz, self.index_n_heads, self.index_head_dim)
            _dsa_debug_sync("indexer query fused rope", q)
        else:
            q = q.reshape(query_seqlen, bsz, self.index_n_heads, self.index_head_dim)
            q = self._apply_rope(q, query_rotary_pos_emb, mscale)
            _dsa_debug_sync("indexer query torch rope", q)
        q = rotate_activation(q)
        _dsa_debug_sync("indexer query hadamard", q)

        weights, _ = self.linear_weights_proj(query_x)
        _dsa_debug_sync("indexer query weights", weights)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale
        return q, weights

    def _project_query_chunk_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        rotary_pos_emb,
        mscale: float,
        q_start: int,
        q_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._project_query_before_topk(
            x[q_start:q_end],
            qr[q_start:q_end],
            self._slice_rotary_pos_emb(rotary_pos_emb, q_start, q_end),
            mscale,
        )

    def _project_key_before_topk(
        self,
        x: torch.Tensor,
        rotary_pos_emb,
        mscale: float,
        orig_seqlen: int,
        pad_len: int,
        apply_indexcache: bool = True,
    ) -> torch.Tensor:
        """Project the full DSA indexer key prefix once."""
        seqlen, bsz, _ = x.size()
        k, _ = self.linear_wk(x)
        _dsa_debug_sync("indexer key linear_wk", k)
        if self.config.fp4 and (
            not torch.is_grad_enabled() or _env_flag_enabled(_DSA_INDEXER_TORCH_K_NORM_ENV, "1")
        ):
            # TE LayerNorm can assert on saved-stat outputs in FP4 activation
            # recompute/StreamBP paths. Keep this fallback scoped to the DSA
            # indexer key norm; the HISA selector and selected-attention kernels
            # still use their fused implementations. The fallback is parameterized
            # with the TE module weights, so grad-enabled replay still trains it.
            k = _torch_layer_norm_like_te(self.k_norm, k, self.config.layernorm_epsilon)
        else:
            k = self.k_norm(k)
        _dsa_debug_sync("indexer key norm", k)
        if self._use_fused_indexer_flat_rope_writer(k, rotary_pos_emb, 1, self.index_head_dim):
            k = _DSAIndexerFlatRopeInplaceFunction.apply(
                k,
                rotary_pos_emb,
                1,
                self.index_head_dim,
                self.qk_pos_emb_head_dim,
                float(mscale),
                bool(self._indexer_rope_config.rotary_interleaved),
            )
            _dsa_debug_sync("indexer key fused rope", k)
        else:
            k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
            k = self._apply_rope(k, rotary_pos_emb, mscale)
            k = k.reshape(seqlen, bsz, self.index_head_dim)
            _dsa_debug_sync("indexer key torch rope", k)
        k = rotate_activation(k)
        _dsa_debug_sync("indexer key hadamard", k)
        if pad_len:
            k = k[:orig_seqlen]

        if apply_indexcache and self.indexcache_config is not None:
            from megatron.core.quantization.indexcache import apply_indexcache_kv

            k = apply_indexcache_kv(k, self.indexcache_config)
            _dsa_debug_sync("indexer key indexcache", k)
        return k

    def _apply_indexcache_to_key(self, k: torch.Tensor) -> torch.Tensor:
        if self.indexcache_config is None:
            return k
        from megatron.core.quantization.indexcache import apply_indexcache_kv

        return apply_indexcache_kv(k, self.indexcache_config)

    def _project_local_key_then_gather_before_topk(
        self,
        x: torch.Tensor,
        rotary_pos_emb,
        mscale: float,
        orig_seqlen: int,
        pad_len: int,
    ) -> torch.Tensor:
        """Project the local SP hidden shard before gathering compact DSA index keys."""
        local_rank = self.pg_collection.tp.rank()
        local_start = local_rank * orig_seqlen
        local_rotary_pos_emb = self._slice_rotary_pos_emb(
            rotary_pos_emb, local_start, local_start + x.size(0)
        )
        k_local = self._project_key_before_topk(
            x,
            local_rotary_pos_emb,
            mscale,
            orig_seqlen,
            pad_len,
            apply_indexcache=False,
        )
        k = gather_from_sequence_parallel_region(k_local, group=self.pg_collection.tp)
        return self._apply_indexcache_to_key(k)

    def _project_query_chunk_sp_owner_broadcast_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        rotary_pos_emb,
        mscale: float,
        q_start: int,
        q_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project one global query chunk from its SP owner, then broadcast compact tensors.

        The current training shape uses DSA chunks that divide the TP-local sequence
        shard. That lets us avoid gathering the full hidden state while preserving
        exact per-token projections and autograd via reduce-to-owner backward.
        """
        tp_group = self.pg_collection.tp
        tp_size = tp_group.size()
        local_seq_len = x.size(0)
        q_len = q_end - q_start
        if tp_size <= 1:
            return self._project_query_chunk_before_topk(
                x, qr, rotary_pos_emb, mscale, q_start, q_end
            )
        if local_seq_len <= 0:
            raise ValueError("DSA sequence-parallel query projection received empty local shard")

        owner_rank = q_start // local_seq_len
        if owner_rank >= tp_size or q_end > (owner_rank + 1) * local_seq_len:
            raise RuntimeError(
                "MEGATRON_DSA_SP_PROJECT_BEFORE_GATHER requires DSA chunks to stay within "
                f"one TP sequence shard; got q_start={q_start}, q_end={q_end}, "
                f"local_seq_len={local_seq_len}, tp_size={tp_size}"
            )

        local_offset = q_start - owner_rank * local_seq_len
        if tp_group.rank() == owner_rank:
            q, weights = self._project_query_before_topk(
                x[local_offset : local_offset + q_len],
                qr[local_offset : local_offset + q_len],
                self._slice_rotary_pos_emb(rotary_pos_emb, q_start, q_end),
                mscale,
            )
        else:
            q = torch.empty(
                q_len,
                x.size(1),
                self.index_n_heads,
                self.index_head_dim,
                device=x.device,
                dtype=x.dtype,
                requires_grad=True,
            )
            weights = torch.empty(
                q_len,
                x.size(1),
                self.index_n_heads,
                device=x.device,
                dtype=x.dtype,
                requires_grad=True,
            )

        owner_global_rank = torch.distributed.get_global_rank(tp_group, int(owner_rank))
        q = _BroadcastFromTensorParallelOwner.apply(q, owner_global_rank, tp_group)
        weights = _BroadcastFromTensorParallelOwner.apply(weights, owner_global_rank, tp_group)
        return q, weights

    def forward_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
        query_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All computations before topk."""
        x, qr, rotary_pos_emb, mscale, orig_seqlen, pad_len = self._prepare_inputs_before_topk(
            x, qr, packed_seq_params
        )
        query_x = x
        query_qr = qr
        query_rotary_pos_emb = rotary_pos_emb
        if query_indices is not None:
            if packed_seq_params is not None:
                raise ValueError("DSA query_indices are only supported for unpacked StreamBP chunks")
            query_indices = query_indices.to(device=x.device, dtype=torch.long)
            if query_indices.dim() != 1:
                raise ValueError(
                    f"DSA query_indices must be 1D, got shape {tuple(query_indices.shape)}"
                )
            if query_indices.numel() == 0:
                raise ValueError("DSA query_indices must not be empty")
            if bool((query_indices < 0).any().item()) or bool(
                (query_indices >= x.size(0)).any().item()
            ):
                raise ValueError(
                    f"DSA query_indices out of range for sequence length {x.size(0)}"
                )
            query_x = x.index_select(0, query_indices)
            if qr.size(0) == query_indices.numel():
                query_qr = qr
            else:
                query_qr = qr.index_select(0, query_indices)
            query_rotary_pos_emb = self._index_rotary_pos_emb(rotary_pos_emb, query_indices)

        q, weights = self._project_query_before_topk(
            query_x, query_qr, query_rotary_pos_emb, mscale
        )
        k = self._project_key_before_topk(x, rotary_pos_emb, mscale, orig_seqlen, pad_len)
        if pad_len and query_indices is None:
            q = q[:orig_seqlen]
        if pad_len and query_indices is None:
            weights = weights[:orig_seqlen]

        return q, k, weights

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for DSA Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen].
            topk_indices: Top-k indices [batch, seqlen, index_topk].
        """
        assert packed_seq_params is None, "Packed sequence is not supported for DSAttention"

        # [seqlen, batch, index_n_heads * index_head_dim]
        # [seqlen, batch, index_head_dim]
        # [seqlen, batch, index_n_heads]
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        # [batch, seqlen, seqlen], [batch, seqlen, index_topk]
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, self.index_topk, mask)

        return index_scores, topk_indices

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass for DSA Indexer.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
        """
        _, topk_indices = self.forward_with_scores(x, qr, mask, packed_seq_params)
        return topk_indices


def unfused_dsa_fn(query, key, value, topk_indices, softmax_scale):
    """
    Unfused sparse attention implementation.
    """
    sq, b, np, hn = query.size()
    skv = key.size(0)
    hnv = value.size(3)

    # ===================================
    # Raw attention scores [b, np, sq, skv]
    # ===================================
    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [skv, b, np, hn] -> [b, np, hn, skv] -> [b * np, hn, skv]
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, skv)
    # Compute attention scores [b * np, sq, skv]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, skv]
    attention_scores = attention_scores.reshape(b, np, sq, skv)

    # ===================================
    # Apply sparse mask from indexer
    # ===================================
    # index_mask [b, sq, skv]
    index_mask = torch.full((b, sq, skv), float("-inf"), device=attention_scores.device)
    index_mask.scatter_(-1, topk_indices, 0)
    # causal_mask [sq, skv]
    causal_mask = torch.triu(
        torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=index_mask.device),
        diagonal=1,
    )
    # [b, sq, skv] + [1, sq, skv] -> [b, sq, skv]
    index_mask += causal_mask.view(1, sq, skv)
    # [b, np, sq, skv] + [b, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += index_mask.unsqueeze(1)
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)

    # ===================================
    # Output
    # ===================================
    # [skv, b, np, hnv] -> [b, np, skv, hnv] -> [b * np, skv, hnv]
    value = value.permute(1, 2, 0, 3).reshape(b * np, skv, hnv)
    # Reshape attention_scores: [b, np, sq, skv] -> [b * np, sq, skv]
    attention_scores = attention_scores.reshape(b * np, sq, skv)
    # Compute output: [b * np, sq, hnv]
    output = torch.bmm(attention_scores.to(value.dtype), value)
    # Reshape output: [b * np, sq, hnv] -> [b, np, sq, hnv] -> [sq, b, np, hnv]
    output = output.reshape(b, np, sq, hnv).permute(2, 0, 1, 3).contiguous()
    # Flatten: [sq, b, np, hnv] -> [sq, b, np * hnv]
    output = output.reshape(sq, b, np * hnv)
    return output


def _apply_dsa_score_mask(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    q_start: int,
    q_end: int,
    sk: int,
    is_causal: bool,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if is_causal:
        if query_positions is not None or key_positions is not None:
            if query_positions is None or key_positions is None:
                raise ValueError("query_positions and key_positions must be provided together")
            q_pos = query_positions.to(device=scores.device)
            k_pos = key_positions.to(device=scores.device)
            if q_pos.numel() != q_end - q_start:
                raise ValueError(
                    f"query_positions length {q_pos.numel()} does not match chunk length "
                    f"{q_end - q_start}"
                )
            if k_pos.numel() != sk:
                raise ValueError(f"key_positions length {k_pos.numel()} does not match sk {sk}")
        else:
            q_pos = torch.arange(q_start, q_end, device=scores.device)
            k_pos = torch.arange(sk, device=scores.device)
        if scores.dim() == 4:
            q_pos = q_pos.view(1, 1, -1, 1)
            k_pos = k_pos.view(1, 1, 1, -1)
        else:
            q_pos = q_pos.view(1, -1, 1)
            k_pos = k_pos.view(1, 1, -1)
        return scores.masked_fill(k_pos > q_pos, float("-inf"))
    if mask is None:
        return scores
    if mask.dim() == 2:
        return scores + mask[q_start:q_end, :].unsqueeze(0)
    if mask.dim() == 3:
        mask_slice = mask[:, q_start:q_end, :]
        if scores.dim() == 4:
            mask_slice = mask_slice.unsqueeze(1)
        return scores + mask_slice
    raise ValueError(f"DSA mask must be 2D or 3D, got shape {tuple(mask.shape)}")


def _streaming_qk_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk: int,
    mask: Optional[torch.Tensor],
    q_start: int,
    q_end: int,
    sk: int,
    is_causal: bool,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute exact DSA indexer top-k without materializing all per-head scores."""

    topk_k = min(topk, sk)
    key_block_size = _dsa_indexer_key_block_size(topk_k)
    running_scores = None
    running_indices = None
    candidate_scores_buffer = None
    candidate_indices_buffer = None

    with torch.no_grad():
        # The streaming path is used only when the DSA indexer loss is disabled,
        # so top-k selection should not build or retain an autograd graph.
        q_float = q.float()
        k_float = k if k.dtype == torch.float32 else k.float()
        weights_view = weights.unsqueeze(-1)
        if is_causal:
            if query_positions is not None or key_positions is not None:
                if query_positions is None or key_positions is None:
                    raise ValueError("query_positions and key_positions must be provided together")
                q_pos = query_positions.to(device=q.device).view(1, -1, 1)
            else:
                q_pos = torch.arange(q_start, q_end, device=q.device).view(1, -1, 1)

        for key_start in range(0, sk, key_block_size):
            key_end = min(key_start + key_block_size, sk)
            key_block = k_float[key_start:key_end]

            block_scores = torch.einsum("sbhd,tbd->sbht", q_float, key_block)
            block_scores = torch.relu_(block_scores)
            block_scores.mul_(weights_view)
            block_scores = block_scores.sum(dim=2).transpose(0, 1)

            if is_causal:
                if key_positions is not None:
                    k_pos = key_positions[key_start:key_end].to(device=block_scores.device).view(
                        1, 1, -1
                    )
                else:
                    k_pos = torch.arange(
                        key_start, key_end, device=block_scores.device
                    ).view(1, 1, -1)
                block_scores.masked_fill_(k_pos > q_pos, float("-inf"))
            elif mask is not None:
                if mask.dim() == 2:
                    block_scores.add_(
                        mask[q_start:q_end, key_start:key_end].unsqueeze(0)
                    )
                elif mask.dim() == 3:
                    block_scores.add_(mask[:, q_start:q_end, key_start:key_end])
                else:
                    raise ValueError(f"DSA mask must be 2D or 3D, got shape {tuple(mask.shape)}")

            block_topk = min(topk_k, key_end - key_start)
            block_scores, block_indices = block_scores.topk(
                block_topk, dim=-1, sorted=False
            )
            block_indices.add_(key_start)

            if running_scores is None:
                running_scores = block_scores
                running_indices = block_indices
            else:
                candidate_len = running_scores.size(-1) + block_scores.size(-1)
                if (
                    candidate_scores_buffer is None
                    or candidate_scores_buffer.shape[:-1] != running_scores.shape[:-1]
                    or candidate_scores_buffer.size(-1) < candidate_len
                ):
                    candidate_shape = (
                        *running_scores.shape[:-1],
                        topk_k + min(topk_k, key_block_size),
                    )
                    candidate_scores_buffer = torch.empty(
                        candidate_shape, device=running_scores.device, dtype=running_scores.dtype
                    )
                    candidate_indices_buffer = torch.empty(
                        candidate_shape, device=running_indices.device, dtype=running_indices.dtype
                    )
                candidate_scores = candidate_scores_buffer.narrow(-1, 0, candidate_len)
                candidate_indices = candidate_indices_buffer.narrow(-1, 0, candidate_len)
                torch.cat((running_scores, block_scores), dim=-1, out=candidate_scores)
                torch.cat((running_indices, block_indices), dim=-1, out=candidate_indices)
                keep_k = min(topk_k, candidate_len)
                running_scores, selected = candidate_scores.topk(
                    keep_k, dim=-1, sorted=False
                )
                running_indices = candidate_indices.gather(-1, selected)

    return running_indices


def _sparse_dsa_attention_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    mask: Optional[torch.Tensor],
    q_start: int,
    is_causal: bool,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q_len, bsz, num_heads, head_dim = query.size()
    value_head_dim = value.size(3)
    topk_k = topk_indices.size(-1)
    topk_block_size = 64
    batch_outputs = []

    for batch_idx in range(bsz):
        selected_positions = topk_indices[batch_idx]
        valid_selected = selected_positions >= 0
        gather_positions = selected_positions.clamp_min(0)
        query_batch = query[:, batch_idx]
        key_batch = key[:, batch_idx]
        score_blocks = []
        for topk_start in range(0, topk_k, topk_block_size):
            topk_end = min(topk_start + topk_block_size, topk_k)
            block_positions = gather_positions[:, topk_start:topk_end]
            block_index = block_positions.reshape(-1)
            key_block = key_batch.index_select(0, block_index)
            key_block = key_block.view(q_len, topk_end - topk_start, num_heads, head_dim)
            score_blocks.append(
                torch.einsum("qhd,qkhd->qkh", query_batch.float(), key_block.float())
            )
        attention_scores = torch.cat(score_blocks, dim=1) * softmax_scale

        if is_causal:
            if query_positions is not None or key_positions is not None:
                if query_positions is None or key_positions is None:
                    raise ValueError("query_positions and key_positions must be provided together")
                q_pos = query_positions.to(device=query.device).unsqueeze(1)
                selected_abs_positions = key_positions.to(device=query.device).index_select(
                    0, gather_positions.reshape(-1).long()
                )
                selected_abs_positions = selected_abs_positions.view_as(selected_positions)
                invalid = selected_abs_positions > q_pos
            else:
                q_pos = torch.arange(q_start, q_start + q_len, device=query.device).unsqueeze(1)
                invalid = selected_positions > q_pos
            attention_scores = attention_scores.masked_fill(invalid.unsqueeze(-1), float("-inf"))
        elif mask is not None:
            if mask.dim() == 2:
                selected_mask = mask[q_start : q_start + q_len, :].gather(1, gather_positions)
            elif mask.dim() == 3:
                selected_mask = mask[batch_idx, q_start : q_start + q_len, :].gather(
                    1, gather_positions
                )
            else:
                raise ValueError(f"DSA mask must be 2D or 3D, got shape {tuple(mask.shape)}")
            attention_scores = attention_scores + selected_mask.unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(
            (~valid_selected).unsqueeze(-1), float("-inf")
        )

        attention_probs = torch.softmax(attention_scores, dim=1, dtype=torch.float32).to(
            value.dtype
        )
        del attention_scores

        value_batch = value[:, batch_idx]
        output_batch = None
        for topk_start in range(0, topk_k, topk_block_size):
            topk_end = min(topk_start + topk_block_size, topk_k)
            block_positions = gather_positions[:, topk_start:topk_end]
            block_index = block_positions.reshape(-1)
            selected_value = value_batch.index_select(0, block_index)
            selected_value = selected_value.view(
                q_len, topk_end - topk_start, num_heads, value_head_dim
            )
            block_output = torch.einsum(
                "qkh,qkhd->qhd", attention_probs[:, topk_start:topk_end, :], selected_value
            )
            output_batch = block_output if output_batch is None else output_batch + block_output
        del attention_probs
        batch_outputs.append(output_batch)

    output = torch.stack(batch_outputs, dim=1)
    return output.reshape(q_len, bsz, num_heads * value_head_dim)


def _hisa_config_for_topk(config: IndexCacheHISAConfig, topk: int) -> IndexCacheHISAConfig:
    if config.topk_tokens == topk:
        return config
    return replace(config, topk_tokens=topk)


_HISA_CAUSAL_PREFIX_LENS_CACHE: dict[tuple[str, Optional[int], int, int, int], torch.Tensor] = {}


def _hisa_causal_prefix_lens_from_offsets(
    q_len: int,
    sk: int,
    *,
    q_start: int,
    device: torch.device,
) -> torch.Tensor:
    """Build causal prefix lengths without launching CUDA arange on the hot path."""

    q_len = int(q_len)
    sk = int(sk)
    q_start = int(q_start)
    if q_len <= 0:
        return torch.empty((0,), device=device, dtype=torch.int32)

    if device.type != "cuda":
        return torch.arange(q_start + 1, q_start + q_len + 1, device=device, dtype=torch.int32).clamp_(
            0, sk
        )

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
        device = torch.device("cuda", device_index)

    cache_key = (device.type, device_index, q_start, q_len, sk)
    cached = _HISA_CAUSAL_PREFIX_LENS_CACHE.get(cache_key)
    if cached is not None and cached.device == device:
        return cached

    # Some full-stack B200/packed-DSA runs hit cudaErrorInvalidValue in
    # torch.arange(..., device=cuda:N) on pipeline ranks 12-15. Materializing
    # this tiny metadata vector on CPU and copying it to the target device keeps
    # the exact causal-prefix semantics while avoiding that fragile CUDA arange
    # launch. Cache it because the same chunk offsets repeat every layer/step.
    host_prefix_lens = torch.arange(q_start + 1, q_start + q_len + 1, dtype=torch.int32).clamp_(
        0, sk
    )
    with torch.cuda.device(device):
        prefix_lens = host_prefix_lens.to(device=device, non_blocking=False)
    _HISA_CAUSAL_PREFIX_LENS_CACHE[cache_key] = prefix_lens
    return prefix_lens


def _hisa_prefix_lens_for_chunk(
    q_len: int,
    bsz: int,
    sk: int,
    *,
    q_start: int,
    is_causal: bool,
    mask: Optional[torch.Tensor],
    query_positions: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mask is not None:
        return None

    prefix_lens_is_final = False
    if is_causal:
        if query_positions is not None or key_positions is not None:
            if query_positions is None or key_positions is None:
                return None
            if query_positions.dim() != 1 or key_positions.dim() != 1:
                return None
            if query_positions.numel() != q_len or key_positions.numel() != sk:
                return None
            key_positions = key_positions.to(device=device, dtype=torch.long).contiguous()
            query_positions = query_positions.to(device=device, dtype=torch.long).contiguous()
            if (
                not _env_flag_enabled(_HISA_ASSUME_SORTED_POSITIONS_ENV, "1")
                and key_positions.numel() > 1
                and bool((key_positions[1:] < key_positions[:-1]).any().item())
            ):
                return None
            prefix_lens = torch.searchsorted(key_positions, query_positions, right=True)
        else:
            prefix_lens = _hisa_causal_prefix_lens_from_offsets(
                q_len, sk, q_start=q_start, device=device
            )
            prefix_lens_is_final = True
    else:
        prefix_lens = torch.full((q_len,), sk, device=device, dtype=torch.int32)
        prefix_lens_is_final = True

    if not prefix_lens_is_final:
        prefix_lens = prefix_lens.clamp(0, sk).to(torch.int32)

    if bsz == 1:
        return prefix_lens
    return prefix_lens.repeat(bsz)


def _flatten_hisa_indexer_rows(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_len, bsz, num_index_heads, index_head_dim = q.shape
    sk = k.shape[0]
    q_flat = q.permute(1, 0, 2, 3).reshape(bsz * q_len, num_index_heads, index_head_dim)
    weights_flat = weights.permute(1, 0, 2).reshape(bsz * q_len, num_index_heads)
    k_concat = k.permute(1, 0, 2).reshape(bsz * sk, k.shape[-1])
    k_offsets = torch.arange(
        0,
        (bsz + 1) * sk,
        sk,
        device=k.device,
        dtype=torch.long,
    )
    token_to_batch_idx = torch.arange(bsz, device=q.device, dtype=torch.long).repeat_interleave(
        q_len
    )
    return q_flat, weights_flat, k_concat, k_offsets, token_to_batch_idx


def _hisa_attention_target_probs(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    tp_group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    q_len, bsz, num_heads, head_dim = query.shape
    topk_k = topk_indices.shape[-1]
    if is_hisa_attention_target_probs_triton_supported(query, key, topk_indices):
        attention_probs = hisa_attention_target_probs_triton(
            query, key, topk_indices, float(softmax_scale)
        )
        if _dsa_process_group_size(tp_group) > 1:
            torch.distributed.all_reduce(attention_probs.contiguous(), group=tp_group)
        attention_probs = attention_probs / attention_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-20)
        return attention_probs

    row_chunk = _hisa_target_row_chunk(q_len)
    attention_probs = torch.empty(
        (bsz * q_len, topk_k),
        device=query.device,
        dtype=torch.float32,
    )

    for batch_idx in range(bsz):
        query_b = query[:, batch_idx].detach().float()
        key_b = key[:, batch_idx].detach().float()
        selected = topk_indices[batch_idx].to(torch.int32)
        valid = selected >= 0

        for row_start in range(0, q_len, row_chunk):
            row_end = min(row_start + row_chunk, q_len)
            selected_rows = selected[row_start:row_end]
            valid_rows = valid[row_start:row_end]
            safe_selected = selected_rows.clamp_min(0)
            rows = row_end - row_start
            selected_key = key_b.index_select(0, safe_selected.reshape(-1))
            selected_key = selected_key.view(rows, topk_k, num_heads, head_dim)
            scores = (
                torch.einsum(
                    "qhd,qkhd->qkh",
                    query_b[row_start:row_end],
                    selected_key,
                )
                * softmax_scale
            )
            scores = scores.masked_fill(~valid_rows.unsqueeze(-1), float("-inf"))
            probs = torch.softmax(scores, dim=1, dtype=torch.float32).sum(dim=2)
            out_start = batch_idx * q_len + row_start
            attention_probs[out_start : out_start + rows] = probs

    if _dsa_process_group_size(tp_group) > 1:
        torch.distributed.all_reduce(attention_probs.contiguous(), group=tp_group)
    attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True).clamp_min(
        1e-20
    )
    return attention_probs


class _HISAIndexerLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        prefix_lens: torch.Tensor,
        softmax_scale: float,
        topk: int,
        config: IndexCacheHISAConfig,
        tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len, bsz, _, _ = q.shape
        q_flat, weights_flat, k_concat, k_offsets, token_to_batch_idx = _flatten_hisa_indexer_rows(
            q, weights, k
        )
        config = _hisa_config_for_topk(config, int(topk))
        topk_flat, cache = hisa_selector_forward_and_save(
            q_flat,
            k_concat,
            k_offsets,
            weights_flat,
            prefix_lens,
            token_to_batch_idx,
            config,
        )
        if cache is None:
            raise RuntimeError("HISA indexer loss was called for a dense-fallback chunk.")

        topk_indices = topk_flat.reshape(bsz, q_len, int(topk)).to(torch.int32)
        candidate_scores = (
            torch.relu(cache.candidate_dot) * cache.weights.unsqueeze(1)
        ).sum(dim=-1)
        valid = cache.topk_positions >= 0
        selected_scores = candidate_scores.gather(1, cache.topk_positions.clamp_min(0))
        selected_scores = selected_scores.masked_fill(~valid, float("-inf"))

        attention_probs = _hisa_attention_target_probs(
            query, key, topk_indices, float(softmax_scale), tp_group
        )
        index_probs = torch.softmax(selected_scores, dim=-1, dtype=torch.float32)
        index_probs = torch.where(valid, index_probs, torch.zeros_like(index_probs))
        kl = attention_probs * (
            torch.log(attention_probs + 1e-10) - torch.log(index_probs + 1e-10)
        )
        loss_sum = kl.masked_fill(~valid, 0).sum()

        ctx.cache = cache
        ctx.config = config
        ctx.q_shape = tuple(q.shape)
        ctx.k_shape = tuple(k.shape)
        ctx.q_dtype = q.dtype
        ctx.weights_dtype = weights.dtype
        ctx.k_dtype = k.dtype
        ctx.save_for_backward(index_probs, attention_probs, valid)
        return topk_indices, loss_sum

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss_sum):
        index_probs, attention_probs, valid = ctx.saved_tensors
        cache = ctx.cache
        grad_topk_logit = (index_probs - attention_probs) * grad_loss_sum
        grad_topk_logit = grad_topk_logit.masked_fill(~valid, 0)
        grad_candidate_scores, grad_block_scores = hisa_unselect_grad_from_topk_to_scores(
            grad_topk_logit, cache
        )
        grad_q_flat, grad_k_by_batch, grad_weights_flat = apply_hisa_score_backward(
            grad_candidate_scores,
            grad_block_scores,
            cache,
            config=ctx.config,
        )

        q_len, bsz, num_index_heads, index_head_dim = ctx.q_shape
        sk, _, index_k_dim = ctx.k_shape
        grad_q = (
            grad_q_flat.view(bsz, q_len, num_index_heads, index_head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
            .to(ctx.q_dtype)
        )
        grad_weights = (
            grad_weights_flat.view(bsz, q_len, num_index_heads)
            .permute(1, 0, 2)
            .contiguous()
            .to(ctx.weights_dtype)
        )
        grad_k = torch.stack(grad_k_by_batch, dim=1).reshape(sk, bsz, index_k_dim).to(ctx.k_dtype)

        return (
            grad_q,
            grad_weights,
            grad_k,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _try_load_hisa_cuda_ext():
    try:
        from megatron.core.extensions.hisa_indexer.kernels.build import get_ext

        return get_ext()
    except Exception:
        return None


def _hisa_fused_indexer_loss_enabled() -> bool:
    return _env_flag_enabled(_HISA_FUSED_INDEXER_LOSS_ENV, "1")


def _hisa_selected_score_backward_cuda(
    grad_scores: torch.Tensor,
    q_rows: torch.Tensor,
    weights_rows: torch.Tensor,
    k_rows: torch.Tensor,
    topk_indices_i32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _try_load_hisa_cuda_ext()
    if ext is None or not hasattr(ext, "hisa_selected_score_bwd"):
        raise RuntimeError("HISA selected-score backward CUDA extension is unavailable")

    q_f = q_rows.contiguous().float()
    weights_f = weights_rows.contiguous().float()
    k_f = k_rows.contiguous().float()
    topk_i32 = topk_indices_i32.contiguous().to(torch.int32)
    grad_f = grad_scores.contiguous().float()
    grad_q = torch.zeros_like(q_f)
    grad_k = torch.zeros_like(k_f)
    grad_w = torch.zeros_like(weights_f)
    ext.hisa_selected_score_bwd(
        grad_f,
        q_f,
        k_f,
        weights_f,
        topk_i32,
        grad_q,
        grad_k,
        grad_w,
    )
    return grad_q, grad_w, grad_k


def _hisa_selected_score_backward_cuda_batched(
    grad_scores: torch.Tensor,
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk_indices_i32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run selected-score backward for a full microbatch in one CUDA call.

    The underlying CUDA kernel accepts one contiguous key table. For MBS>1 we
    concatenate the per-batch indexer K rows and add a per-batch offset to
    selected top-k indices. This keeps the math identical to one call per batch
    item while avoiding repeated extension launches in the trainable HISA path.
    """

    q_len, bsz, num_heads, head_dim = q.shape
    sk = k.shape[0]
    topk_k = topk_indices_i32.shape[-1]
    q_flat = q.permute(1, 0, 2, 3).reshape(bsz * q_len, num_heads, head_dim)
    weights_flat = weights.permute(1, 0, 2).reshape(bsz * q_len, num_heads)
    k_concat = k.permute(1, 0, 2).reshape(bsz * sk, k.shape[-1])
    topk_batched = topk_indices_i32.contiguous().to(torch.int32)
    batch_offsets = (
        torch.arange(bsz, device=topk_batched.device, dtype=torch.int32).view(bsz, 1, 1)
        * int(sk)
    )
    topk_global = torch.where(topk_batched >= 0, topk_batched + batch_offsets, topk_batched)
    grad_q_flat, grad_w_flat, grad_k_concat = _hisa_selected_score_backward_cuda(
        grad_scores.reshape(bsz * q_len, topk_k),
        q_flat,
        weights_flat,
        k_concat,
        topk_global.reshape(bsz * q_len, topk_k),
    )
    grad_q = grad_q_flat.reshape(bsz, q_len, num_heads, head_dim).permute(1, 0, 2, 3)
    grad_weights = grad_w_flat.reshape(bsz, q_len, num_heads).permute(1, 0, 2)
    grad_k = grad_k_concat.reshape(bsz, sk, k.shape[-1]).permute(1, 0, 2)
    return grad_q.contiguous(), grad_weights.contiguous(), grad_k.contiguous()


class _HISASelectWithScores(torch.autograd.Function):
    """HISA selector with custom selected-logit gradients.

    The selector/top-k itself is non-differentiable. The selected logits remain
    differentiable via ``_hisa_selected_score_backward_cuda`` so the indexer can
    train without keeping the BMM/gather selector graph alive through StreamBP.
    """

    @staticmethod
    def forward(
        ctx,
        q_rows: torch.Tensor,
        weights_rows: torch.Tensor,
        k_rows: torch.Tensor,
        prefix_lens: torch.Tensor,
        topk: int,
        config: IndexCacheHISAConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = indexcache_hisa_select_with_scores(
            q_rows,
            weights_rows,
            k_rows,
            int(topk),
            config=config,
            prefix_lens=prefix_lens,
        )
        if result is None:
            raise RuntimeError("HISA selected-score autograd path cannot handle this chunk")
        topk_i32, selected_scores = result
        topk_i32 = topk_i32.contiguous().to(torch.int32)
        selected_scores = selected_scores.contiguous().float()
        ctx.mark_non_differentiable(topk_i32)
        ctx.save_for_backward(q_rows, weights_rows, k_rows, topk_i32)
        return topk_i32, selected_scores

    @staticmethod
    def backward(ctx, grad_topk: torch.Tensor | None, grad_selected_scores: torch.Tensor | None):
        q_rows, weights_rows, k_rows, topk_i32 = ctx.saved_tensors
        if grad_selected_scores is None:
            return (
                torch.zeros_like(q_rows),
                torch.zeros_like(weights_rows),
                torch.zeros_like(k_rows),
                None,
                None,
                None,
            )
        grad_q, grad_w, grad_k = _hisa_selected_score_backward_cuda(
            grad_selected_scores,
            q_rows,
            weights_rows,
            k_rows,
            topk_i32,
        )
        return (
            grad_q.to(q_rows.dtype),
            grad_w.to(weights_rows.dtype),
            grad_k.to(k_rows.dtype),
            None,
            None,
            None,
        )


class _HISASelectWithScoresBatched(torch.autograd.Function):
    """Batched HISA selector with one selected-score CUDA backward call."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        prefix_lens: torch.Tensor,
        topk: int,
        config: IndexCacheHISAConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len, bsz, _, _ = q.shape
        topk_batches = []
        selected_score_batches = []
        for batch_idx in range(bsz):
            if prefix_lens.numel() == q_len:
                prefix_b = prefix_lens
            else:
                prefix_b = prefix_lens[batch_idx * q_len : (batch_idx + 1) * q_len]
            result = indexcache_hisa_select_with_scores(
                q[:, batch_idx],
                weights[:, batch_idx],
                k[:, batch_idx],
                int(topk),
                config=config,
                prefix_lens=prefix_b,
            )
            if result is None:
                raise RuntimeError("HISA selected-score autograd path cannot handle this chunk")
            topk_i32, selected_scores = result
            topk_batches.append(topk_i32.contiguous().to(torch.int32))
            selected_score_batches.append(selected_scores.contiguous().float())

        topk_i32 = torch.stack(topk_batches, dim=0).contiguous()
        selected_scores = torch.stack(selected_score_batches, dim=0).reshape(
            bsz * q_len, int(topk)
        )
        ctx.mark_non_differentiable(topk_i32)
        ctx.save_for_backward(q, weights, k, topk_i32)
        return topk_i32, selected_scores

    @staticmethod
    def backward(ctx, grad_topk: torch.Tensor | None, grad_selected_scores: torch.Tensor | None):
        q, weights, k, topk_i32 = ctx.saved_tensors
        if grad_selected_scores is None:
            return (
                torch.zeros_like(q),
                torch.zeros_like(weights),
                torch.zeros_like(k),
                None,
                None,
                None,
            )
        grad_q, grad_w, grad_k = _hisa_selected_score_backward_cuda_batched(
            grad_selected_scores,
            q,
            weights,
            k,
            topk_i32,
        )
        return (
            grad_q.to(q.dtype),
            grad_w.to(weights.dtype),
            grad_k.to(k.dtype),
            None,
            None,
            None,
        )


class _SelectedScoresKLLoss(torch.autograd.Function):
    """KL loss with a compact custom gradient for selected indexer logits."""

    @staticmethod
    def forward(
        ctx,
        selected_scores: torch.Tensor,
        teacher_probs: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        selected_scores_f = selected_scores.float()
        teacher_probs_f = teacher_probs.float()
        valid = valid.to(device=selected_scores.device, dtype=torch.bool)
        masked_scores = selected_scores_f.masked_fill(~valid, float("-inf"))
        teacher_probs_f = teacher_probs_f.masked_fill(~valid, 0.0)
        if is_hisa_kl_grad_triton_supported(masked_scores, teacher_probs_f):
            loss_sum, grad_selected_scores = hisa_kl_loss_and_grad_triton(
                masked_scores.contiguous(), teacher_probs_f.contiguous()
            )
        else:
            index_probs = torch.softmax(masked_scores, dim=-1, dtype=torch.float32)
            index_probs = torch.where(valid, index_probs, torch.zeros_like(index_probs))
            kl = teacher_probs_f * (
                torch.log(teacher_probs_f + 1e-10) - torch.log(index_probs + 1e-10)
            )
            loss_sum = kl.masked_fill(~valid, 0).sum()
            prob_ratio = index_probs / (index_probs + 1e-10)
            teacher_prob_ratio = teacher_probs_f * prob_ratio
            row_scale = teacher_prob_ratio.sum(dim=-1, keepdim=True)
            grad_selected_scores = (
                index_probs * row_scale - teacher_prob_ratio
            ).masked_fill(~valid, 0)
        ctx.save_for_backward(grad_selected_scores.to(selected_scores.dtype))
        return loss_sum

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (grad_selected_scores,) = ctx.saved_tensors
        grad = grad_selected_scores
        if grad_output is not None:
            grad = grad * grad_output.to(dtype=grad.dtype)
        return grad, None, None


class _HISAFusedIndexerLoss(torch.autograd.Function):
    """Fused HISA indexer-loss path.

    Forward:
      - CUDA HISA selector emits top-k indices and selected indexer logits.
      - Triton selected-attention teacher emits compact teacher probabilities.
      - Triton KL kernel emits the scalar loss and compact dL/d(selected_logits).

    Backward:
      - CUDA selected-score backward scatters gradients into indexer q/k/weights.

    The HISA top-k selection remains non-differentiable, matching the existing
    trainable indexer path.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        prefix_lens: torch.Tensor,
        softmax_scale: float,
        topk: int,
        config: IndexCacheHISAConfig,
        tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len, bsz, _, _ = q.shape
        topk_k = min(int(topk), int(k.shape[0]))
        config = _hisa_config_for_topk(config, topk_k)

        topk_batches = []
        selected_score_batches = []
        teacher_prob_batches = []
        for batch_idx in range(bsz):
            if prefix_lens.numel() == q_len:
                prefix_b = prefix_lens
            else:
                prefix_b = prefix_lens[batch_idx * q_len : (batch_idx + 1) * q_len]
            result = None
            query_b = query[:, batch_idx : batch_idx + 1]
            key_b = key[:, batch_idx : batch_idx + 1]
            selector_result = indexcache_hisa_select_with_scores(
                q[:, batch_idx],
                weights[:, batch_idx],
                k[:, batch_idx],
                topk_k,
                config=config,
                prefix_lens=prefix_b,
            )
            if selector_result is not None:
                topk_i32, selected_scores = selector_result
                if is_hisa_attention_target_probs_triton_supported(
                    query_b, key_b, topk_i32.unsqueeze(0)
                ):
                    teacher_probs = hisa_attention_target_probs_triton(
                        query_b,
                        key_b,
                        topk_i32.unsqueeze(0),
                        float(softmax_scale),
                    )
                    result = (topk_i32, selected_scores, teacher_probs)
            if result is None:
                result = indexcache_hisa_cuda_select_scores_teacher(
                    q[:, batch_idx],
                    weights[:, batch_idx],
                    k[:, batch_idx],
                    query[:, batch_idx],
                    key[:, batch_idx],
                    topk_k,
                    config=config,
                    prefix_lens=prefix_b,
                    softmax_scale=float(softmax_scale),
                )
            if result is None:
                raise RuntimeError("fused HISA indexer loss cannot handle dense-fallback chunks")
            topk_i32, selected_scores, teacher_probs = result
            topk_batches.append(topk_i32)
            selected_score_batches.append(selected_scores)
            teacher_prob_batches.append(teacher_probs)

        topk_i32 = torch.stack(topk_batches, dim=0).contiguous()
        selected_scores = torch.stack(selected_score_batches, dim=0).reshape(
            bsz * q_len, topk_k
        )
        teacher_probs = torch.stack(teacher_prob_batches, dim=0).reshape(bsz * q_len, topk_k)
        topk_indices = topk_i32
        ctx.mark_non_differentiable(topk_indices)

        if _dsa_process_group_size(tp_group) > 1:
            torch.distributed.all_reduce(teacher_probs.contiguous(), group=tp_group)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(
            1e-20
        )
        if is_hisa_kl_grad_triton_supported(selected_scores, teacher_probs):
            loss_sum, grad_selected_scores = hisa_kl_loss_and_grad_triton(
                selected_scores, teacher_probs
            )
        else:
            index_probs = torch.softmax(selected_scores, dim=-1, dtype=torch.float32)
            valid = topk_i32.reshape(bsz * q_len, topk_k) >= 0
            index_probs = torch.where(valid, index_probs, torch.zeros_like(index_probs))
            kl = teacher_probs * (
                torch.log(teacher_probs + 1e-10) - torch.log(index_probs + 1e-10)
            )
            loss_sum = kl.masked_fill(~valid, 0).sum()
            prob_ratio = index_probs / (index_probs + 1e-10)
            teacher_prob_ratio = teacher_probs * prob_ratio
            row_scale = teacher_prob_ratio.sum(dim=-1, keepdim=True)
            grad_selected_scores = (
                index_probs * row_scale - teacher_prob_ratio
            ).masked_fill(~valid, 0)

        ctx.save_for_backward(q, weights, k, topk_i32, grad_selected_scores)
        ctx.q_len = q_len
        ctx.bsz = bsz
        return topk_indices, loss_sum

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss_sum):
        q, weights, k, topk_i32, grad_selected_scores = ctx.saved_tensors
        q_len = ctx.q_len
        bsz = ctx.bsz
        if grad_loss_sum is None:
            return (
                torch.zeros_like(q),
                torch.zeros_like(weights),
                torch.zeros_like(k),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        scale = grad_loss_sum.float() if torch.is_tensor(grad_loss_sum) else 1.0
        grad_selected_scores = grad_selected_scores * scale

        grad_q, grad_weights, grad_k = _hisa_selected_score_backward_cuda_batched(
            grad_selected_scores,
            q,
            weights,
            k,
            topk_i32,
        )
        grad_q = grad_q.to(q.dtype)
        grad_weights = grad_weights.to(weights.dtype)
        grad_k = grad_k.to(k.dtype)
        return (
            grad_q,
            grad_weights,
            grad_k,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def chunked_dsa_forward(
    q: Union[torch.Tensor, Callable[[int, int], Tuple[torch.Tensor, torch.Tensor]]],
    k: torch.Tensor,
    weights: Optional[torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    topk: int,
    mask: Optional[torch.Tensor],
    is_causal: bool,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
    chunk_size: int,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    indexcache_hisa_config: Optional[IndexCacheHISAConfig] = None,
    dsa_split_qk: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    sq, bsz, num_heads, head_dim = query.size()
    sk = key.size(0)
    if topk <= 0 or sk <= 0:
        raise ValueError(
            "chunked_dsa_forward received non-positive sparse selection extent: "
            f"topk={topk} sk={sk} query_shape={tuple(query.shape)} key_shape={tuple(key.shape)} "
            f"value_shape={tuple(value.shape)} q_shape={'callable' if callable(q) else tuple(q.shape)} "
            f"k_shape={tuple(k.shape)} weights_shape={None if weights is None else tuple(weights.shape)} "
            f"chunk_size={chunk_size} loss_coeff={loss_coeff} "
            f"query_positions_shape={None if query_positions is None else tuple(query_positions.shape)} "
            f"key_positions_shape={None if key_positions is None else tuple(key_positions.shape)}"
        )
    split_query_pe = None
    split_key_pe = None
    if dsa_split_qk is not None:
        split_query_pe, split_key_pe = dsa_split_qk
        if split_query_pe.size(0) != sq or split_query_pe.shape[:3] != query.shape[:3]:
            raise ValueError(
                "DSA split query positional component must match query prefix shape, got "
                f"{tuple(split_query_pe.shape)} for query {tuple(query.shape)}"
            )
        if split_key_pe.size(0) != sk or split_key_pe.size(1) != bsz:
            raise ValueError(
                "DSA split key positional component must match key sequence/batch, got "
                f"{tuple(split_key_pe.shape)} for key {tuple(key.shape)}"
            )

    def materialize_attention_qk(query_part: torch.Tensor, query_pe_part: torch.Tensor):
        if query_pe_part is None or split_key_pe is None:
            return query_part, key
        key_pe = split_key_pe
        if key_pe.size(2) == 1:
            key_pe = key_pe.expand(-1, -1, key.size(2), -1)
        return torch.cat([query_part, query_pe_part], dim=-1), torch.cat([key, key_pe], dim=-1)

    q_weights_provider = q if callable(q) else None
    work_device = query.device
    outputs = []
    topk_buffer = None
    use_triton_attention = None
    use_streaming_indexer_topk = loss_coeff <= 0 and _env_flag_enabled(
        _DSA_STREAMING_INDEXER_TOPK_ENV, "1"
    )
    use_indexcache_hisa_topk = indexcache_hisa_config is not None
    index_scores_buffer = None
    topk_values_buffer = None
    topk_indices_buffer = None
    loss_sum = None
    loss_count = 0
    attention_teacher_score_chunks = []
    attention_teacher_row_indices = []
    stream_triton_attention_chunks = _env_flag_enabled(_DSA_STREAM_TRITON_ATTENTION_CHUNKS_ENV, "0")

    for q_start in range(0, sq, chunk_size):
        q_end = min(q_start + chunk_size, sq)
        if q_weights_provider is not None:
            q_chunk, weights_chunk = q_weights_provider(q_start, q_end)
        else:
            q_chunk = q[q_start:q_end]
            weights_chunk = weights[q_start:q_end]
        query_chunk = query[q_start:q_end]
        query_pe_chunk = None if split_query_pe is None else split_query_pe[q_start:q_end]
        query_positions_chunk = (
            None if query_positions is None else query_positions[q_start:q_end]
        )

        topk_indices = None
        index_scores = None
        hisa_selected_scores = None
        hisa_loss_already_accumulated = False
        hisa_loss_deferred_to_attention = False
        if use_indexcache_hisa_topk:
            if loss_coeff > 0:
                topk_k = min(topk, sk)
                prefix_lens = None
                if (
                    _hisa_fused_indexer_loss_enabled()
                    and q_chunk.is_cuda
                    and k.is_cuda
                    and query_chunk.is_cuda
                    and key.is_cuda
                    and q_chunk.size(-1) == 128
                    and k.size(-1) == 128
                ):
                    prefix_lens = _hisa_prefix_lens_for_chunk(
                        q_end - q_start,
                        bsz,
                        sk,
                        q_start=q_start,
                        is_causal=is_causal,
                        mask=mask,
                        query_positions=query_positions_chunk,
                        key_positions=key_positions,
                        device=work_device,
                    )
                if (
                    prefix_lens is not None
                    and not (
                        indexcache_hisa_config.fallback_to_dense_if_short
                        and int(prefix_lens.max().item()) <= topk_k
                    )
                ):
                    dummy_topk = torch.empty(
                        (bsz, q_end - q_start, topk_k),
                        device=work_device,
                        dtype=torch.int32,
                    )
                    if (
                        query_pe_chunk is not None
                        and split_key_pe is not None
                        and is_sparse_dsa_split_qk_triton_supported(
                            query_chunk,
                            query_pe_chunk,
                            key,
                            split_key_pe,
                            value,
                            dummy_topk,
                            mask,
                            is_causal,
                            query_positions=query_positions_chunk,
                            key_positions=key_positions,
                        )
                    ) or (
                        query_pe_chunk is None
                        and is_sparse_dsa_teacher_triton_supported(
                            query_chunk,
                            key,
                            value,
                            dummy_topk,
                            mask,
                            is_causal,
                            query_positions=query_positions_chunk,
                            key_positions=key_positions,
                        )
                    ):
                        try:
                            topk_i32, selected_scores = _HISASelectWithScoresBatched.apply(
                                q_chunk,
                                weights_chunk,
                                k,
                                prefix_lens,
                                topk_k,
                                indexcache_hisa_config,
                            )
                        except RuntimeError as exc:
                            if "cannot handle this chunk" not in str(exc):
                                raise
                            topk_i32 = None
                            selected_scores = None
                        if topk_i32 is not None and selected_scores is not None:
                            topk_indices = topk_i32
                            hisa_selected_scores = selected_scores
                            hisa_loss_deferred_to_attention = True
                    if topk_indices is None:
                        loss_query_chunk, loss_key = materialize_attention_qk(
                            query_chunk, query_pe_chunk
                        )
                        topk_indices, chunk_loss_sum = _HISAFusedIndexerLoss.apply(
                            q_chunk,
                            weights_chunk,
                            k,
                            loss_query_chunk,
                            loss_key,
                            prefix_lens,
                            float(softmax_scale),
                            topk_k,
                            indexcache_hisa_config,
                            pg_collection.tp if pg_collection is not None else None,
                        )
                        loss_sum = chunk_loss_sum if loss_sum is None else loss_sum + chunk_loss_sum
                        loss_count += bsz * (q_end - q_start)
                        hisa_loss_already_accumulated = True
                else:
                    hisa_result = indexcache_hisa_topk_with_scores(
                        q_chunk,
                        weights_chunk,
                        k,
                        topk_k,
                        config=indexcache_hisa_config,
                        q_start=q_start,
                        is_causal=is_causal,
                        mask=mask,
                        query_positions=query_positions_chunk,
                        key_positions=key_positions,
                        return_scores=True,
                    )
                    if hisa_result is not None:
                        topk_indices, hisa_selected_scores = hisa_result
            else:
                topk_indices = indexcache_hisa_topk(
                    q_chunk,
                    weights_chunk,
                    k,
                    topk,
                    config=indexcache_hisa_config,
                    q_start=q_start,
                    is_causal=is_causal,
                    mask=mask,
                    query_positions=query_positions_chunk,
                    key_positions=key_positions,
                )

        if topk_indices is not None:
            if (
                loss_coeff > 0
                and hisa_selected_scores is None
                and not hisa_loss_already_accumulated
            ):
                index_scores = _compute_index_scores(q_chunk, weights_chunk, k)
                index_scores = _apply_dsa_score_mask(
                    index_scores,
                    mask,
                    q_start,
                    q_end,
                    sk,
                    is_causal,
                    query_positions=query_positions_chunk,
                    key_positions=key_positions,
                )
            else:
                index_scores = None
        elif use_streaming_indexer_topk:
            if is_dsa_indexer_scores_triton_supported(
                q_chunk,
                weights_chunk,
                k,
                mask,
                is_causal,
                query_positions=query_positions_chunk,
                key_positions=key_positions,
            ):
                score_shape = (bsz, q_end - q_start, sk)
                if index_scores_buffer is None or tuple(index_scores_buffer.shape) != score_shape:
                    index_scores_buffer = torch.empty(
                        score_shape, device=work_device, dtype=torch.float32
                    )
                index_scores = dsa_indexer_scores_triton(
                    q_chunk,
                    weights_chunk,
                    k,
                    q_start,
                    query_positions=query_positions_chunk,
                    key_positions=key_positions,
                    out=index_scores_buffer,
                )
                topk_k = min(topk, sk)
                topk_shape = (*score_shape[:-1], topk_k)
                if topk_values_buffer is None or tuple(topk_values_buffer.shape) != topk_shape:
                    topk_values_buffer = torch.empty(
                        topk_shape, device=index_scores.device, dtype=index_scores.dtype
                    )
                    topk_indices_buffer = torch.empty(
                        topk_shape, device=index_scores.device, dtype=torch.long
                    )
                _, topk_indices = torch.topk(
                    index_scores,
                    topk_k,
                    dim=-1,
                    sorted=False,
                    out=(topk_values_buffer, topk_indices_buffer),
                )
            else:
                index_scores = None
                topk_indices = _streaming_qk_topk(
                    q_chunk,
                    weights_chunk,
                    k,
                    topk,
                    mask,
                    q_start,
                    q_end,
                    sk,
                    is_causal,
                    query_positions=query_positions_chunk,
                    key_positions=key_positions,
                )
        else:
            index_scores = _compute_index_scores(q_chunk, weights_chunk, k)
            index_scores = _apply_dsa_score_mask(
                index_scores,
                mask,
                q_start,
                q_end,
                sk,
                is_causal,
                query_positions=query_positions_chunk,
                key_positions=key_positions,
            )
            topk_k = min(topk, sk)
            topk_indices = index_scores.topk(topk_k, dim=-1, sorted=False)[1]

        if topk_indices is not None:
            topk_indices = _maybe_narrow_dsa_topk_indices(topk_indices, sk)

        if use_triton_attention is None:
            if query_pe_chunk is not None and split_key_pe is not None:
                use_triton_attention = is_sparse_dsa_split_qk_triton_supported(
                    query_chunk,
                    query_pe_chunk,
                    key,
                    split_key_pe,
                    value,
                    topk_indices,
                    mask,
                    is_causal,
                    query_positions=query_positions_chunk,
                    key_positions=key_positions,
                )
            else:
                use_triton_attention = is_sparse_dsa_triton_supported(
                    query_chunk,
                    key,
                    value,
                    topk_indices,
                    mask,
                    is_causal,
                    query_positions=query_positions_chunk,
                    key_positions=key_positions,
                )
        if _env_flag_enabled(_DSA_VALIDATE_TOPK_INDICES_ENV, "0") and bool(
            (topk_indices < -1).any().item()
        ):
            use_triton_attention = False

        if loss_coeff > 0 and hisa_loss_already_accumulated:
            pass
        elif (
            loss_coeff > 0
            and hisa_selected_scores is not None
            and not hisa_loss_deferred_to_attention
        ):
            flat_topk = topk_indices.reshape(bsz * (q_end - q_start), -1)
            valid = flat_topk >= 0
            selected_scores = hisa_selected_scores.masked_fill(~valid, float("-inf"))
            attention_query_chunk, attention_key = materialize_attention_qk(
                query_chunk, query_pe_chunk
            )
            attention_probs = _hisa_attention_target_probs(
                attention_query_chunk,
                attention_key,
                topk_indices,
                float(softmax_scale),
                pg_collection.tp if pg_collection is not None else None,
            )
            index_probs = torch.softmax(selected_scores, dim=-1, dtype=torch.float32)
            index_probs = torch.where(valid, index_probs, torch.zeros_like(index_probs))
            kl = attention_probs * (
                torch.log(attention_probs + 1e-10) - torch.log(index_probs + 1e-10)
            )
            chunk_loss_sum = kl.masked_fill(~valid, 0).sum()
            loss_sum = chunk_loss_sum if loss_sum is None else loss_sum + chunk_loss_sum
            loss_count += bsz * (q_end - q_start)
        elif loss_coeff > 0 and not hisa_loss_deferred_to_attention:
            loss_index_scores = index_scores
            attention_query_chunk, attention_key_full = materialize_attention_qk(
                query_chunk, query_pe_chunk
            )
            attention_query = attention_query_chunk.detach().permute(1, 2, 0, 3).reshape(
                bsz * num_heads, q_end - q_start, attention_query_chunk.size(-1)
            )
            attention_key = attention_key_full.detach().permute(1, 2, 3, 0).reshape(
                bsz * num_heads, attention_key_full.size(-1), sk
            )
            attention_scores = torch.bmm(attention_query.float(), attention_key.float())
            attention_scores = attention_scores.reshape(bsz, num_heads, q_end - q_start, sk)
            attention_scores = attention_scores * softmax_scale
            attention_scores = _apply_dsa_score_mask(
                attention_scores,
                mask,
                q_start,
                q_end,
                sk,
                is_causal,
                query_positions=query_positions_chunk,
                key_positions=key_positions,
            )

            if sparse_loss:
                valid_topk = topk_indices >= 0
                selected_counts = torch.zeros_like(loss_index_scores, dtype=torch.int32)
                selected_counts.scatter_add_(
                    -1,
                    topk_indices.clamp_min(0),
                    valid_topk.to(dtype=selected_counts.dtype),
                )
                index_mask = torch.full_like(loss_index_scores, float("-inf"))
                index_mask = index_mask.masked_fill(selected_counts > 0, 0)
                loss_index_scores = loss_index_scores + index_mask
                attention_scores = attention_scores + index_mask.unsqueeze(1)

            attention_probs = torch.softmax(attention_scores, dim=-1, dtype=torch.float32)
            index_probs = torch.softmax(loss_index_scores, dim=-1, dtype=torch.float32)

            attention_probs = attention_probs.sum(dim=1)
            if pg_collection.tp.size() > 1:
                torch.distributed.all_reduce(attention_probs.contiguous(), group=pg_collection.tp)
            attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)

            kl = attention_probs * (
                torch.log(attention_probs + 1e-10) - torch.log(index_probs + 1e-10)
            )
            chunk_loss_sum = kl.sum(dim=-1).sum()
            loss_sum = chunk_loss_sum if loss_sum is None else loss_sum + chunk_loss_sum
            loss_count += bsz * (q_end - q_start)
        elif index_scores is not None:
            del index_scores

        if use_triton_attention:
            topk_indices, hisa_selected_scores = _maybe_sort_dsa_topk_indices_and_scores(
                topk_indices, hisa_selected_scores
            )
            if stream_triton_attention_chunks:
                if hisa_loss_deferred_to_attention:
                    if query_pe_chunk is not None and split_key_pe is not None:
                        (
                            chunk_output,
                            attention_teacher_probs,
                        ) = sparse_dsa_attention_split_qk_with_teacher_triton(
                            query_chunk,
                            query_pe_chunk,
                            key,
                            split_key_pe,
                            value,
                            topk_indices,
                            softmax_scale,
                            q_start,
                            query_positions=query_positions_chunk,
                            key_positions=key_positions,
                        )
                    else:
                        (
                            chunk_output,
                            attention_teacher_probs,
                        ) = sparse_dsa_attention_with_teacher_triton(
                            query_chunk,
                            key,
                            value,
                            topk_indices,
                            softmax_scale,
                            q_start,
                            query_positions=query_positions_chunk,
                            key_positions=key_positions,
                        )
                    if pg_collection is not None and _dsa_process_group_size(pg_collection.tp) > 1:
                        torch.distributed.all_reduce(
                            attention_teacher_probs.contiguous(), group=pg_collection.tp
                        )
                    attention_teacher_probs = attention_teacher_probs / attention_teacher_probs.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-20)
                    flat_topk = topk_indices.reshape(bsz * (q_end - q_start), -1)
                    valid = flat_topk >= 0
                    chunk_loss_sum = _SelectedScoresKLLoss.apply(
                        hisa_selected_scores,
                        attention_teacher_probs.reshape_as(hisa_selected_scores),
                        valid,
                    )
                    loss_sum = chunk_loss_sum if loss_sum is None else loss_sum + chunk_loss_sum
                    loss_count += hisa_selected_scores.size(0)
                    outputs.append(chunk_output)
                else:
                    if query_pe_chunk is not None and split_key_pe is not None:
                        outputs.append(
                            sparse_dsa_attention_split_qk_triton(
                                query_chunk,
                                query_pe_chunk,
                                key,
                                split_key_pe,
                                value,
                                topk_indices,
                                softmax_scale,
                                q_start,
                                query_positions=query_positions_chunk,
                                key_positions=key_positions,
                            )
                        )
                    else:
                        outputs.append(
                            sparse_dsa_attention_triton(
                                query_chunk,
                                key,
                                value,
                                topk_indices,
                                softmax_scale,
                                q_start,
                                query_positions=query_positions_chunk,
                                key_positions=key_positions,
                            )
                        )
                continue
            if topk_buffer is None:
                topk_buffer = torch.empty(
                    (bsz, sq, topk_indices.size(-1)),
                    device=topk_indices.device,
                    dtype=_dsa_topk_buffer_dtype(sk),
                )
            topk_buffer[:, q_start:q_end, :].copy_(topk_indices)
            if hisa_loss_deferred_to_attention:
                attention_teacher_score_chunks.append(hisa_selected_scores)
                attention_teacher_row_indices.append(
                    torch.cat(
                        [
                            torch.arange(
                                batch_idx * sq + q_start,
                                batch_idx * sq + q_end,
                                device=topk_buffer.device,
                                dtype=torch.int32,
                            )
                            for batch_idx in range(bsz)
                        ]
                    )
                )
        else:
            fallback_query_chunk, fallback_key = materialize_attention_qk(
                query_chunk, query_pe_chunk
            )
            outputs.append(
                _sparse_dsa_attention_chunk(
                    fallback_query_chunk,
                    fallback_key,
                    value,
                    topk_indices,
                    softmax_scale,
                    mask,
                    q_start,
                    is_causal,
                    query_positions=query_positions_chunk,
                    key_positions=key_positions,
                )
            )

    if use_triton_attention and stream_triton_attention_chunks:
        output = torch.cat(outputs, dim=0)
    elif use_triton_attention:
        # Keep top-k generation chunked to bound the indexer score tensor, then run
        # the selected-token attention as one autograd op so K/V gradients are
        # accumulated once per layer instead of once per query chunk.
        if attention_teacher_score_chunks:
            if split_query_pe is not None and split_key_pe is not None:
                output, attention_teacher_probs = sparse_dsa_attention_split_qk_with_teacher_triton(
                    query,
                    split_query_pe,
                    key,
                    split_key_pe,
                    value,
                    topk_buffer,
                    softmax_scale,
                    0,
                    query_positions=query_positions,
                    key_positions=key_positions,
                )
            else:
                output, attention_teacher_probs = sparse_dsa_attention_with_teacher_triton(
                    query,
                    key,
                    value,
                    topk_buffer,
                    softmax_scale,
                    0,
                    query_positions=query_positions,
                    key_positions=key_positions,
                )
            if pg_collection is not None and _dsa_process_group_size(pg_collection.tp) > 1:
                torch.distributed.all_reduce(
                    attention_teacher_probs.contiguous(), group=pg_collection.tp
                )
            attention_teacher_probs = attention_teacher_probs / attention_teacher_probs.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-20)
            selected_scores = torch.cat(attention_teacher_score_chunks, dim=0)
            flat_topk_buffer = topk_buffer.reshape(bsz * sq, topk_buffer.size(-1))
            attention_teacher_probs = torch.cat(
                [
                    attention_teacher_probs.index_select(0, row_indices)
                    for row_indices in attention_teacher_row_indices
                ],
                dim=0,
            )
            flat_topk = torch.cat(
                [
                    flat_topk_buffer.index_select(0, row_indices)
                    for row_indices in attention_teacher_row_indices
                ],
                dim=0,
            )
            valid = flat_topk >= 0
            chunk_loss_sum = _SelectedScoresKLLoss.apply(
                selected_scores,
                attention_teacher_probs.reshape_as(selected_scores),
                valid,
            )
            loss_sum = chunk_loss_sum if loss_sum is None else loss_sum + chunk_loss_sum
            loss_count += selected_scores.size(0)
        else:
            if split_query_pe is not None and split_key_pe is not None:
                output = sparse_dsa_attention_split_qk_triton(
                    query,
                    split_query_pe,
                    key,
                    split_key_pe,
                    value,
                    topk_buffer,
                    softmax_scale,
                    0,
                    query_positions=query_positions,
                    key_positions=key_positions,
                )
            else:
                output = sparse_dsa_attention_triton(
                    query,
                    key,
                    value,
                    topk_buffer,
                    softmax_scale,
                    0,
                    query_positions=query_positions,
                    key_positions=key_positions,
                )
    else:
        output = torch.cat(outputs, dim=0)

    indexer_loss = None
    if loss_sum is not None:
        indexer_loss = loss_sum * (loss_coeff / loss_count)
    return output, indexer_loss


class DSAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an DSA Indexer to compute top-k
    attention indices for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.layer_number = layer_number

        self.indexer = build_module(
            submodules.indexer, config=self.config, pg_collection=pg_collection
        )
        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0) or 0.0
        if indexer_loss_coeff <= 0:
            for param in self.indexer.parameters():
                param.requires_grad_(False)

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        streambp_positions: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dsa_split_qk: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor [sq, b, np, hn].
            key: Key tensor [skv, b, np, hn].
            value: Value tensor [skv, b, np, hnv].
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, hidden_size]
        """
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            cp_group = self.indexer.pg_collection.cp
            cp_size = _dsa_process_group_size(cp_group)
            cu_seqlens_q = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            cu_seqlens_kv = (
                packed_seq_params.cu_seqlens_kv_padded
                if packed_seq_params.cu_seqlens_kv_padded is not None
                else packed_seq_params.cu_seqlens_kv
            )
            if cu_seqlens_q is None or cu_seqlens_kv is None:
                raise ValueError("DSAttention THD path requires query and KV cu_seqlens")

            def normalize_cu(cu_seqlens: torch.Tensor, name: str) -> torch.Tensor:
                if cu_seqlens.dim() == 2:
                    if cu_seqlens.size(0) != 1:
                        raise ValueError(
                            f"DSAttention THD path expects micro-batch-size 1, got "
                            f"{name} shape {tuple(cu_seqlens.shape)}"
                        )
                    cu_seqlens = cu_seqlens[0]
                if cu_seqlens.dim() != 1:
                    raise ValueError(
                        f"{name} must be 1D, got shape {tuple(cu_seqlens.shape)}"
                    )
                return cu_seqlens

            cu_seqlens_q = normalize_cu(cu_seqlens_q, "cu_seqlens_q")
            cu_seqlens_kv = normalize_cu(cu_seqlens_kv, "cu_seqlens_kv")
            if cu_seqlens_q.numel() != cu_seqlens_kv.numel():
                raise ValueError(
                    "DSAttention THD StreamBP path requires query and KV cu_seqlens "
                    f"with the same number of sequences, got {cu_seqlens_q.numel()} and "
                    f"{cu_seqlens_kv.numel()}"
                )

            if query.dim() == 4:
                if query.size(1) != 1 or key.size(1) != 1 or value.size(1) != 1:
                    raise ValueError("DSAttention THD path only supports a dummy batch dimension")
                query = query.squeeze(1)
                key = key.squeeze(1)
                value = value.squeeze(1)
            split_query_pe = split_key_pe = None
            if dsa_split_qk is not None:
                split_query_pe, split_key_pe = dsa_split_qk
                if split_query_pe.dim() == 4:
                    if split_query_pe.size(1) != 1:
                        raise ValueError("DSAttention THD split query PE expects dummy batch")
                    split_query_pe = split_query_pe.squeeze(1)
                if split_key_pe.dim() == 4:
                    if split_key_pe.size(1) != 1:
                        raise ValueError("DSAttention THD split key PE expects dummy batch")
                    split_key_pe = split_key_pe.squeeze(1)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            if qr.dim() == 2:
                qr = qr.unsqueeze(1)

            if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
                raise ValueError(
                    "DSAttention THD path expects query/key/value as [tokens, heads, dim]"
                )
            if dsa_split_qk is not None:
                if (
                    split_query_pe.dim() != 3
                    or split_query_pe.shape[:2] != query.shape[:2]
                    or split_key_pe.dim() != 3
                    or split_key_pe.size(0) != key.size(0)
                ):
                    raise ValueError(
                        "DSAttention THD split-QK tensors must align with query/key, got "
                        f"query_pe={tuple(split_query_pe.shape)} key_pe={tuple(split_key_pe.shape)}"
                    )
            if x.dim() != 3 or qr.dim() != 3:
                raise ValueError("DSAttention THD path expects x/qr as [tokens, batch, dim]")

            streambp_query_positions = None
            streambp_key_positions = None
            if streambp_positions is not None:
                if cp_size > 1:
                    raise ValueError("StreamBP DSA currently requires context_parallel_size == 1")
                streambp_query_positions, streambp_key_positions = streambp_positions
                if streambp_query_positions is None or streambp_key_positions is None:
                    raise ValueError("StreamBP DSA requires both query and key positions")
                streambp_query_positions = streambp_query_positions.to(
                    device=query.device, dtype=torch.long
                )
                streambp_key_positions = streambp_key_positions.to(
                    device=key.device, dtype=torch.long
                )
                if streambp_query_positions.numel() != query.size(0):
                    raise ValueError(
                        f"StreamBP DSA query position length "
                        f"{streambp_query_positions.numel()} does not match query length "
                        f"{query.size(0)}"
                    )
                if streambp_key_positions.numel() != key.size(0):
                    raise ValueError(
                        f"StreamBP DSA key position length {streambp_key_positions.numel()} "
                        f"does not match key length {key.size(0)}"
                    )

            tp_group = self.indexer.pg_collection.tp
            tp_size = _dsa_process_group_size(tp_group)
            if tp_size > 1 and (x.size(0) != key.size(0) or qr.size(0) != key.size(0)):
                # TP sequence-parallel packed replay can carry THD metadata while
                # attention K/V are gathered differently from the indexer x/qr
                # inputs. The generic THD recursion slices x/qr by attention-KV
                # offsets first, which can make the trainable indexer see an
                # empty K. Route the whole local packed tensor through the
                # non-packed path so the SP-aware indexer projection can gather
                # or owner-broadcast compact q/k before top-k.
                return self.forward(
                    query.unsqueeze(1),
                    key.unsqueeze(1),
                    value.unsqueeze(1),
                    attention_mask,
                    x,
                    qr,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=None,
                    streambp_positions=(
                        None
                        if streambp_positions is None
                        else (streambp_query_positions, streambp_key_positions)
                    ),
                    dsa_split_qk=(
                        None
                        if dsa_split_qk is None
                        else (split_query_pe.unsqueeze(1), split_key_pe.unsqueeze(1))
                    ),
                )

            outputs = []
            q_cu_total = int(cu_seqlens_q[-1].item())
            kv_cu_total = int(cu_seqlens_kv[-1].item())
            if (
                tp_size > 1
                and cp_size == 1
                and (q_cu_total > query.size(0) or kv_cu_total > key.size(0))
            ):
                local_q_offsets = _dsa_thd_tp_local_sequence_offsets(
                    cu_seqlens_q, query.size(0), tp_group
                )
                local_kv_offsets = _dsa_thd_tp_local_sequence_offsets(
                    cu_seqlens_kv, key.size(0), tp_group
                )
            elif cp_size > 1:
                local_q_offsets = _dsa_thd_local_sequence_offsets(cu_seqlens_q, cp_group)
                local_kv_offsets = _dsa_thd_local_sequence_offsets(cu_seqlens_kv, cp_group)
            else:
                q_offsets = cu_seqlens_q.detach().cpu().tolist()
                kv_offsets = cu_seqlens_kv.detach().cpu().tolist()
                local_q_offsets = [
                    (int(start), int(end)) for start, end in zip(q_offsets[:-1], q_offsets[1:])
                ]
                local_kv_offsets = [
                    (int(start), int(end)) for start, end in zip(kv_offsets[:-1], kv_offsets[1:])
                ]
            for (q_start, q_end), (kv_start, kv_end) in zip(local_q_offsets, local_kv_offsets):
                if q_end <= q_start:
                    continue
                if kv_end <= kv_start:
                    raise ValueError("DSAttention THD path received query tokens with empty KV span")
                if kv_start >= key.size(0) or kv_end > key.size(0):
                    raise ValueError(
                        "DSAttention THD path sliced outside local KV tensor: "
                        f"q_span=({q_start},{q_end}) kv_span=({kv_start},{kv_end}) "
                        f"query_len={query.size(0)} key_len={key.size(0)} value_len={value.size(0)} "
                        f"x_len={x.size(0)} qr_len={qr.size(0)} "
                        f"q_cu_total={q_cu_total} kv_cu_total={kv_cu_total} "
                        f"tp_rank={_dsa_process_group_rank(tp_group)} tp_size={tp_size} "
                        f"cp_size={cp_size}"
                    )
                sequence_streambp_positions = None
                if streambp_positions is not None:
                    key_origin = streambp_key_positions[kv_start]
                    sequence_streambp_positions = (
                        streambp_query_positions[q_start:q_end] - key_origin,
                        streambp_key_positions[kv_start:kv_end] - key_origin,
                    )
                outputs.append(
                    self.forward(
                        query[q_start:q_end].unsqueeze(1),
                        key[kv_start:kv_end].unsqueeze(1),
                        value[kv_start:kv_end].unsqueeze(1),
                        attention_mask,
                        x[kv_start:kv_end],
                        qr[kv_start:kv_end],
                        attn_mask_type=attn_mask_type,
                        attention_bias=attention_bias,
                        packed_seq_params=None,
                        streambp_positions=sequence_streambp_positions,
                        dsa_split_qk=(
                            None
                            if dsa_split_qk is None
                            else (
                                split_query_pe[q_start:q_end].unsqueeze(1),
                                split_key_pe[kv_start:kv_end].unsqueeze(1),
                            )
                        ),
                    )
                )

            if not outputs:
                return query.new_empty((0, 1, value.size(1) * value.size(2)))
            return torch.cat(outputs, dim=0)

        sq, b, np, hn = query.size()
        skv = key.size(0)
        hnv = value.size(3)

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        x = x.detach()
        qr = qr.detach()

        streambp_query_positions = None
        streambp_key_positions = None
        streambp_query_indices = None
        if streambp_positions is not None:
            streambp_query_positions, streambp_key_positions = streambp_positions
            if streambp_query_positions is None or streambp_key_positions is None:
                raise ValueError("StreamBP DSA requires both query and key positions")
            streambp_query_indices = streambp_query_positions.to(device=x.device, dtype=torch.long)
            if streambp_query_indices.numel() != sq:
                raise ValueError(
                    f"StreamBP DSA query position length {streambp_query_indices.numel()} "
                    f"does not match query length {sq}"
                )
            if streambp_key_positions.numel() != skv:
                raise ValueError(
                    f"StreamBP DSA key position length {streambp_key_positions.numel()} "
                    f"does not match key length {skv}"
                )

        is_causal = False
        # Get a FP32 mask with -inf for masked positions.
        if attn_mask_type is not None:
            assert attn_mask_type == AttnMaskType.causal, 'Only causal mask is supported for now'
            is_causal = True
            float_mask = None
        else:
            assert attention_mask.shape == (b, 1, sq, skv), 'attention_mask shape mismatch'
            # [b, 1, sq, skv] -> [b, sq, skv]
            mask = attention_mask.squeeze()
            # float_mask [b, sq, skv]
            float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                mask, float('-inf')
            )

        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)
        chunk_size = int(self.config.dsa_chunk_size)
        chunk_indexer_q = (
            streambp_positions is None
            and packed_seq_params is None
            and chunk_size > 0
            and sq > chunk_size
            and _env_flag_enabled(_DSA_CHUNK_INDEXER_PROJ_ENV, "1")
        )
        if chunk_indexer_q:
            sp_project_before_gather = (
                _env_flag_enabled(_DSA_SP_PROJECT_BEFORE_GATHER_ENV, "1")
                and self.config.sequence_parallel
                and self.indexer.pg_collection.tp.size() > 1
                and x.size(0) * self.indexer.pg_collection.tp.size() == sq
                and chunk_size > 0
                and x.size(0) % chunk_size == 0
            )
            if sp_project_before_gather:
                (
                    indexer_x,
                    indexer_qr,
                    indexer_rotary_pos_emb,
                    indexer_mscale,
                    indexer_orig_seqlen,
                    indexer_pad_len,
                ) = self.indexer._prepare_inputs_before_topk(
                    x, qr, packed_seq_params, gather_sequence_parallel=False
                )
                k = self.indexer._project_local_key_then_gather_before_topk(
                    indexer_x,
                    indexer_rotary_pos_emb,
                    indexer_mscale,
                    indexer_orig_seqlen,
                    indexer_pad_len,
                )

                def q_weights_provider(q_start: int, q_end: int):
                    return self.indexer._project_query_chunk_sp_owner_broadcast_before_topk(
                        indexer_x,
                        indexer_qr,
                        indexer_rotary_pos_emb,
                        indexer_mscale,
                        q_start,
                        q_end,
                    )

            else:
                (
                    indexer_x,
                    indexer_qr,
                    indexer_rotary_pos_emb,
                    indexer_mscale,
                    indexer_orig_seqlen,
                    indexer_pad_len,
                ) = self.indexer._prepare_inputs_before_topk(x, qr, packed_seq_params)
                k = self.indexer._project_key_before_topk(
                    indexer_x,
                    indexer_rotary_pos_emb,
                    indexer_mscale,
                    indexer_orig_seqlen,
                    indexer_pad_len,
                )

                def q_weights_provider(q_start: int, q_end: int):
                    return self.indexer._project_query_chunk_before_topk(
                        indexer_x,
                        indexer_qr,
                        indexer_rotary_pos_emb,
                        indexer_mscale,
                        q_start,
                        q_end,
                    )

            q = q_weights_provider
            weights = None
        else:
            q, k, weights = self.indexer.forward_before_topk(
                x, qr, packed_seq_params, query_indices=streambp_query_indices
            )
        numeric_debug = None
        log_dsa_debug = False
        force_debug = False
        if os.getenv("MEGATRON_NUMERIC_DEBUG_DSA", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            from megatron.core import numeric_debug as _numeric_debug

            numeric_debug = _numeric_debug
            log_dsa_debug = numeric_debug.rank_allowed_for("DSA") and numeric_debug.event_allowed(
                f"dsa.layer{self.layer_number}",
                limit=int(os.getenv("MEGATRON_NUMERIC_DEBUG_DSA_LIMIT", "64")),
            )
            force_debug = os.getenv("MEGATRON_NUMERIC_DEBUG_DSA_FORCE", "").lower() in (
                "1",
                "true",
                "yes",
            )
            if log_dsa_debug:
                numeric_debug.log_tensor(
                    f"dsa.layer{self.layer_number}.query", query, force=force_debug
                )
                numeric_debug.log_tensor(f"dsa.layer{self.layer_number}.key", key, force=force_debug)
                numeric_debug.log_tensor(
                    f"dsa.layer{self.layer_number}.value", value, force=force_debug
                )
                if not callable(q):
                    numeric_debug.log_tensor(
                        f"dsa.layer{self.layer_number}.q_index", q, force=force_debug
                    )
                numeric_debug.log_tensor(f"dsa.layer{self.layer_number}.k_index", k, force=force_debug)
                if weights is not None:
                    numeric_debug.log_tensor(
                        f"dsa.layer{self.layer_number}.weights", weights, force=force_debug
                    )
        query_positions = streambp_query_positions
        key_positions = streambp_key_positions
        cp_group = self.indexer.pg_collection.cp
        cp_size = _dsa_process_group_size(cp_group)
        if cp_size > 1:
            if streambp_positions is not None:
                raise ValueError("StreamBP DSA currently requires context_parallel_size == 1")
            if not is_causal:
                raise NotImplementedError("DSAttention CP path currently supports causal masks only")
            local_skv = key.size(0)
            query_positions = _dsa_cp_local_position_ids(sq, cp_group, query.device)
            key_positions = _dsa_cp_gathered_position_ids(local_skv, cp_group, key.device)
            k = gather_from_sequence_parallel_region(k, group=cp_group)
            key = gather_from_sequence_parallel_region(key, group=cp_group)
            value = gather_from_sequence_parallel_region(value, group=cp_group)
        output, indexer_loss = chunked_dsa_forward(
            q,
            k,
            weights,
            query,
            key,
            value,
            self.softmax_scale,
            self.indexer.index_topk,
            float_mask,
            is_causal,
            indexer_loss_coeff if self.training and torch.is_grad_enabled() else 0.0,
            getattr(self.config, "dsa_indexer_use_sparse_loss", False),
            self.indexer.pg_collection,
            chunk_size,
            query_positions=query_positions,
            key_positions=key_positions,
            indexcache_hisa_config=self.indexer.indexcache_hisa_config,
            dsa_split_qk=dsa_split_qk,
        )
        if indexer_loss is not None:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss, layer_number=self.layer_number, num_layers=self.config.num_layers
            )
            DSAIndexerAuxLossState.add(indexer_loss)
        if numeric_debug is not None and log_dsa_debug:
            numeric_debug.log_tensor(
                f"dsa.layer{self.layer_number}.output", output, force=force_debug
            )
            if indexer_loss is not None:
                numeric_debug.log_tensor(
                    f"dsa.layer{self.layer_number}.indexer_loss",
                    indexer_loss,
                    force=force_debug,
                )

        return output
