# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT GPT."""

# Capture the true program start time BEFORE any heavy imports.
import time
_PROGRAM_START_TIME = time.time()

import json
import math

# Suppress warnings on all ranks but rank 0.
import os
import warnings


def _apply_local_rank_nccl_hca() -> None:
    """Optionally bind each local rank to one RDMA HCA before torch imports."""
    enabled = os.getenv("MEGATRON_LOCAL_RANK_NCCL_HCA", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not enabled:
        return

    local_rank = int(os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "0")))
    hcas = [
        item.strip()
        for item in os.getenv(
            "MEGATRON_NCCL_LOCAL_RANK_HCAS",
            "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
        ).split(",")
        if item.strip()
    ]
    if not hcas:
        return

    hca = hcas[local_rank % len(hcas)]
    if not hca.startswith(("=", "^")):
        hca = f"={hca}"
    os.environ["NCCL_IB_HCA"] = hca
    if os.getenv("MEGATRON_LOCAL_RANK_NCCL_HCA_VERBOSE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        print(
            f"LOCAL_RANK={local_rank} NCCL_IB_HCA={os.environ['NCCL_IB_HCA']}",
            flush=True,
        )


_apply_local_rank_nccl_hca()

rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from fnmatch import fnmatch
from functools import partial
from typing import List, Optional, Tuple

import torch

from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.data_schedule import get_batch_on_this_rank_for_sequence_packing
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import get_attr_wrapped_model, get_thd_batch_on_this_cp_rank, get_batch_on_this_hybrid_cp_rank, StragglerDetector
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank, get_mtp_ranks
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerAuxLossState
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from megatron.training.datasets.sft_dataset import SFTDataset, MockSFTDataset
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _rank_selected(spec: str, rank: int) -> bool:
    spec = (spec or "0").strip()
    if spec.lower() in {"all", "*"}:
        return True
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = item.split("-", 1)
            if lo.strip().isdigit() and hi.strip().isdigit() and int(lo) <= rank <= int(hi):
                return True
        elif item.isdigit() and int(item) == rank:
            return True
        elif fnmatch(str(rank), item):
            return True
    return False


def _dist_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.getenv("RANK", "0"))


def _parallel_diag_payload() -> dict:
    payload = {}
    getters = {
        "tp_rank": parallel_state.get_tensor_model_parallel_rank,
        "pp_rank": parallel_state.get_pipeline_model_parallel_rank,
        "cp_rank": parallel_state.get_context_parallel_rank,
        "dp_rank": parallel_state.get_data_parallel_rank,
        "ep_rank": parallel_state.get_expert_model_parallel_rank,
    }
    for name, getter in getters.items():
        try:
            payload[name] = int(getter())
        except Exception:
            payload[name] = None
    return payload


def _float_item(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().cpu().item())


def _int_item(tensor: torch.Tensor) -> int:
    return int(tensor.detach().cpu().item())


def _parse_float_list(name: str, default: str) -> list[float]:
    values = []
    for item in os.getenv(name, default).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            continue
    return values


_LOSS_DIAG_CALL_COUNT = 0
_BATCH_DIAG_CALL_COUNT = 0
_LOSS_DIAG_TOKENIZER = None


def _loss_diagnostics_enabled() -> bool:
    if not _env_flag("MEGATRON_LOSS_DIAGNOSTICS"):
        return False
    if not _rank_selected(os.getenv("MEGATRON_LOSS_DIAGNOSTIC_RANKS", "0"), _dist_rank()):
        return False
    parallel_filters = {
        "MEGATRON_LOSS_DIAGNOSTIC_TP_RANKS": "tp_rank",
        "MEGATRON_LOSS_DIAGNOSTIC_PP_RANKS": "pp_rank",
        "MEGATRON_LOSS_DIAGNOSTIC_CP_RANKS": "cp_rank",
        "MEGATRON_LOSS_DIAGNOSTIC_DP_RANKS": "dp_rank",
        "MEGATRON_LOSS_DIAGNOSTIC_EP_RANKS": "ep_rank",
    }
    if any(os.getenv(name) not in (None, "") for name in parallel_filters):
        diag = _parallel_diag_payload()
        for env_name, key in parallel_filters.items():
            spec = os.getenv(env_name)
            if spec in (None, ""):
                continue
            value = diag.get(key)
            if value is None or not _rank_selected(spec, int(value)):
                return False
    return True


def _loss_diag_tokenizer():
    global _LOSS_DIAG_TOKENIZER
    if _LOSS_DIAG_TOKENIZER is None:
        _LOSS_DIAG_TOKENIZER = build_tokenizer(get_args())
    return _LOSS_DIAG_TOKENIZER


def _loss_diag_decode(ids: list[int]) -> str:
    if not _env_flag("MEGATRON_LOSS_DIAGNOSTIC_DECODE"):
        return ""
    args = get_args()
    vocab_size = int(getattr(args, "padded_vocab_size", 0) or 0)
    clean = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id < 0:
            continue
        if vocab_size and token_id >= vocab_size:
            continue
        clean.append(token_id)
    if not clean:
        return ""
    try:
        return _loss_diag_tokenizer().detokenize(clean)
    except Exception as exc:
        return f"<decode-error:{type(exc).__name__}:{exc}>"


def _maybe_add_loss_diag_report(
    report: dict,
    losses: torch.Tensor,
    loss_mask: torch.Tensor,
    num_tokens: torch.Tensor,
    loss_dtype: torch.dtype,
) -> None:
    if not _env_flag("MEGATRON_LOSS_DIAGNOSTICS"):
        return
    with torch.no_grad():
        mask = loss_mask > 0
        finite = torch.isfinite(losses)
        active = mask & finite
        active_nonfinite = mask & ~finite
        inactive_nonfinite = (~mask) & ~finite
        denom = num_tokens.detach().to(dtype=loss_dtype).view(1).clamp_min(1)
        report["lm loss diag/sq mean"] = torch.cat(
            [torch.sum((losses * losses).masked_fill(~active, 0.0)).detach().to(loss_dtype).view(1), denom]
        )
        for threshold in (10.0, 20.0, 30.0, 50.0):
            count = torch.sum(active & (losses > threshold)).detach().to(loss_dtype)
            report[f"lm loss diag/gt{int(threshold)} frac"] = torch.cat([count.view(1), denom])
        total = torch.tensor([max(losses.numel(), 1)], device=losses.device, dtype=loss_dtype)
        nonfinite = torch.sum(~finite).detach().to(loss_dtype)
        report["lm loss diag/nonfinite frac"] = torch.cat([nonfinite.view(1), total])
        active_nonfinite_count = torch.sum(active_nonfinite).detach().to(loss_dtype)
        report["lm loss diag/active nonfinite frac"] = torch.cat(
            [active_nonfinite_count.view(1), denom]
        )
        inactive_total = torch.sum(~mask).detach().to(loss_dtype).view(1).clamp_min(1)
        inactive_nonfinite_count = torch.sum(inactive_nonfinite).detach().to(loss_dtype)
        report["lm loss diag/inactive nonfinite frac"] = torch.cat(
            [inactive_nonfinite_count.view(1), inactive_total]
        )
        mask_total = torch.tensor([max(loss_mask.numel(), 1)], device=loss_mask.device, dtype=loss_dtype)
        mask_sum = torch.sum(mask).detach().to(loss_dtype)
        report["lm loss diag/mask frac"] = torch.cat([mask_sum.view(1), mask_total])


def _maybe_print_loss_diagnostics(
    losses: torch.Tensor,
    loss_mask: torch.Tensor,
    loss: torch.Tensor,
    num_tokens: torch.Tensor,
    output_shape: tuple[int, ...],
    dsa_indexer_loss: Optional[torch.Tensor],
    tokens: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> None:
    global _LOSS_DIAG_CALL_COUNT
    _LOSS_DIAG_CALL_COUNT += 1
    if not _loss_diagnostics_enabled():
        return
    every = max(_env_int("MEGATRON_LOSS_DIAGNOSTIC_EVERY", 1), 1)
    if _LOSS_DIAG_CALL_COUNT % every != 0:
        return

    with torch.no_grad():
        rank = _dist_rank()
        mask = loss_mask > 0
        finite = torch.isfinite(losses)
        active = mask & finite
        active_nonfinite = mask & ~finite
        inactive_nonfinite = (~mask) & ~finite
        masked = losses[active].detach().float()
        payload = {
            "call": _LOSS_DIAG_CALL_COUNT,
            "rank": rank,
            "local_rank": _env_int("LOCAL_RANK", -1),
            "output_shape": list(output_shape),
            "loss_numel": int(losses.numel()),
            "mask_numel": int(loss_mask.numel()),
            "mask_sum": _float_item(loss_mask.sum()),
            "num_tokens": _int_item(num_tokens),
            "loss_sum": _float_item(loss),
            "finite_count": _int_item(finite.sum()),
            "nonfinite_count": _int_item((~finite).sum()),
            "active_nonfinite_count": _int_item(active_nonfinite.sum()),
            "inactive_nonfinite_count": _int_item(inactive_nonfinite.sum()),
            "masked_finite_count": int(masked.numel()),
            **_parallel_diag_payload(),
        }
        if dsa_indexer_loss is not None:
            payload["dsa_indexer_loss"] = _float_item(dsa_indexer_loss)
        if masked.numel() > 0:
            quantiles = _parse_float_list(
                "MEGATRON_LOSS_DIAGNOSTIC_QUANTILES", "0.5,0.9,0.95,0.99,0.999"
            )
            payload.update(
                {
                    "masked_mean": _float_item(masked.mean()),
                    "masked_std": _float_item(masked.std(unbiased=False)),
                    "masked_min": _float_item(masked.min()),
                    "masked_max": _float_item(masked.max()),
                    "negative_loss_count": _int_item((masked < 0).sum()),
                }
            )
            if quantiles:
                q = torch.tensor(quantiles, device=masked.device, dtype=torch.float32)
                values = torch.quantile(masked, q)
                payload["quantiles"] = {
                    f"{quantile:g}": float(value)
                    for quantile, value in zip(quantiles, values.detach().cpu().tolist())
                }
            threshold_payload = {}
            for threshold in _parse_float_list(
                "MEGATRON_LOSS_DIAGNOSTIC_THRESHOLDS", "5,10,15,20,30,50"
            ):
                count = _int_item((masked > threshold).sum())
                threshold_payload[f">{threshold:g}"] = {
                    "count": count,
                    "frac": count / max(masked.numel(), 1),
                }
            payload["thresholds"] = threshold_payload
            topn = max(_env_int("MEGATRON_LOSS_DIAGNOSTIC_TOPK", 8), 0)
            if topn > 0:
                masked_losses = losses.detach().float().masked_fill(~active, float("-inf"))
                values, flat_positions = torch.topk(
                    masked_losses, k=min(topn, int(masked.numel())), sorted=True
                )
                labels_flat = labels.detach().reshape(-1) if labels is not None else None
                tokens_flat = tokens.detach().reshape(-1) if tokens is not None else None
                pos_flat = (
                    position_ids.detach().reshape(-1) if position_ids is not None else None
                )
                context = max(_env_int("MEGATRON_LOSS_DIAGNOSTIC_CONTEXT", 3), 0)
                examples = []
                shape = tuple(labels.shape) if labels is not None else output_shape
                for value, flat_pos in zip(
                    values.detach().cpu().tolist(), flat_positions.detach().cpu().tolist()
                ):
                    if not math.isfinite(float(value)):
                        continue
                    item = {
                        "flat": int(flat_pos),
                        "loss": float(value),
                    }
                    if len(shape) >= 2:
                        batch = int(flat_pos // shape[1])
                        seq = int(flat_pos % shape[1])
                        item["seq"] = seq
                        item["batch"] = batch
                    if labels_flat is not None and flat_pos < labels_flat.numel():
                        item["label"] = int(labels_flat[flat_pos].detach().cpu().item())
                        decoded = _loss_diag_decode([item["label"]])
                        if decoded:
                            item["label_text"] = decoded
                    if tokens_flat is not None and flat_pos < tokens_flat.numel():
                        item["token"] = int(tokens_flat[flat_pos].detach().cpu().item())
                        decoded = _loss_diag_decode([item["token"]])
                        if decoded:
                            item["token_text"] = decoded
                    if pos_flat is not None and flat_pos < pos_flat.numel():
                        item["position_id"] = int(pos_flat[flat_pos].detach().cpu().item())
                    if context and labels_flat is not None:
                        left = max(0, int(flat_pos) - context)
                        right = min(int(labels_flat.numel()), int(flat_pos) + context + 1)
                        item["label_window"] = [
                            int(x) for x in labels_flat[left:right].detach().cpu().tolist()
                        ]
                        decoded = _loss_diag_decode(item["label_window"])
                        if decoded:
                            item["label_window_text"] = decoded
                        if tokens_flat is not None:
                            item["token_window"] = [
                                int(x) for x in tokens_flat[left:right].detach().cpu().tolist()
                            ]
                            decoded = _loss_diag_decode(item["token_window"])
                            if decoded:
                                item["token_window_text"] = decoded
                    examples.append(item)
                payload["top_losses"] = examples
        if active_nonfinite.any():
            topn_nonfinite = max(_env_int("MEGATRON_LOSS_DIAGNOSTIC_NONFINITE_TOPK", 16), 0)
            if topn_nonfinite > 0:
                positions = torch.nonzero(active_nonfinite, as_tuple=False).view(-1)
                labels_flat = labels.detach().reshape(-1) if labels is not None else None
                tokens_flat = tokens.detach().reshape(-1) if tokens is not None else None
                pos_flat = (
                    position_ids.detach().reshape(-1) if position_ids is not None else None
                )
                context = max(_env_int("MEGATRON_LOSS_DIAGNOSTIC_CONTEXT", 3), 0)
                shape = tuple(labels.shape) if labels is not None else output_shape
                examples = []
                for flat_pos in positions[:topn_nonfinite].detach().cpu().tolist():
                    item = {"flat": int(flat_pos)}
                    if len(shape) >= 2:
                        batch = int(flat_pos // shape[1])
                        seq = int(flat_pos % shape[1])
                        item["seq"] = seq
                        item["batch"] = batch
                    if labels_flat is not None and flat_pos < labels_flat.numel():
                        item["label"] = int(labels_flat[flat_pos].detach().cpu().item())
                        decoded = _loss_diag_decode([item["label"]])
                        if decoded:
                            item["label_text"] = decoded
                    if tokens_flat is not None and flat_pos < tokens_flat.numel():
                        item["token"] = int(tokens_flat[flat_pos].detach().cpu().item())
                        decoded = _loss_diag_decode([item["token"]])
                        if decoded:
                            item["token_text"] = decoded
                    if pos_flat is not None and flat_pos < pos_flat.numel():
                        item["position_id"] = int(pos_flat[flat_pos].detach().cpu().item())
                    if context and labels_flat is not None:
                        left = max(0, int(flat_pos) - context)
                        right = min(int(labels_flat.numel()), int(flat_pos) + context + 1)
                        item["label_window"] = [
                            int(x) for x in labels_flat[left:right].detach().cpu().tolist()
                        ]
                        decoded = _loss_diag_decode(item["label_window"])
                        if decoded:
                            item["label_window_text"] = decoded
                        if tokens_flat is not None:
                            item["token_window"] = [
                                int(x) for x in tokens_flat[left:right].detach().cpu().tolist()
                            ]
                            decoded = _loss_diag_decode(item["token_window"])
                            if decoded:
                                item["token_window_text"] = decoded
                    examples.append(item)
                payload["active_nonfinite_examples"] = examples
        print("[loss_diag] " + json.dumps(payload, sort_keys=True), flush=True)


def _batch_diagnostics_enabled() -> bool:
    if not _env_flag("MEGATRON_BATCH_DIAGNOSTICS"):
        return False
    return _rank_selected(os.getenv("MEGATRON_BATCH_DIAGNOSTIC_RANKS", "0"), _dist_rank())


def _tensor_min_max_payload(prefix: str, tensor: Optional[torch.Tensor]) -> dict:
    if tensor is None:
        return {f"{prefix}_shape": None}
    detached = tensor.detach()
    payload = {f"{prefix}_shape": list(detached.shape), f"{prefix}_numel": int(detached.numel())}
    if detached.numel() > 0:
        payload[f"{prefix}_min"] = _float_item(detached.float().min())
        payload[f"{prefix}_max"] = _float_item(detached.float().max())
    return payload


def _maybe_print_batch_diagnostics(
    tokens: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    loss_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    padding_mask: Optional[torch.Tensor],
) -> None:
    global _BATCH_DIAG_CALL_COUNT
    _BATCH_DIAG_CALL_COUNT += 1
    if not _batch_diagnostics_enabled():
        return
    every = max(_env_int("MEGATRON_BATCH_DIAGNOSTIC_EVERY", 1), 1)
    if _BATCH_DIAG_CALL_COUNT % every != 0:
        return

    with torch.no_grad():
        args = get_args()
        payload = {
            "call": _BATCH_DIAG_CALL_COUNT,
            "rank": _dist_rank(),
            "local_rank": _env_int("LOCAL_RANK", -1),
            **_parallel_diag_payload(),
        }
        payload.update(_tensor_min_max_payload("tokens", tokens))
        payload.update(_tensor_min_max_payload("labels", labels))
        payload.update(_tensor_min_max_payload("loss_mask", loss_mask))
        payload.update(_tensor_min_max_payload("position_ids", position_ids))
        payload.update(_tensor_min_max_payload("padding_mask", padding_mask))
        if loss_mask is not None:
            mask_f = loss_mask.detach().float()
            payload["loss_mask_sum"] = _float_item(mask_f.sum())
            payload["loss_mask_mean"] = _float_item(mask_f.mean())
            payload["loss_mask_nonzero"] = _int_item((mask_f > 0).sum())
            payload["loss_mask_zero"] = _int_item((mask_f <= 0).sum())
            if mask_f.dim() >= 2:
                mask_rows = mask_f.reshape(mask_f.shape[0], -1)
            else:
                mask_rows = mask_f.reshape(1, -1)
            row_counts = mask_rows.sum(dim=1)
            payload["loss_mask_row_counts"] = [
                float(x) for x in row_counts.detach().cpu().tolist()[:16]
            ]
            payload["loss_mask_zero_rows"] = int((row_counts == 0).sum().item())
            payload["loss_mask_full_rows"] = int(
                (row_counts == mask_rows.shape[1]).sum().item()
            )
            quarters = torch.chunk(mask_rows, chunks=min(4, mask_rows.shape[1]), dim=1)
            payload["loss_mask_quarter_counts"] = [
                [float(x) for x in quarter.sum(dim=1).detach().cpu().tolist()[:16]]
                for quarter in quarters
            ]
            spans = []
            for row_idx, row in enumerate(mask_rows[:16]):
                active = torch.nonzero(row > 0, as_tuple=False).view(-1)
                if active.numel() == 0:
                    spans.append({"row": int(row_idx), "active": 0})
                    continue
                spans.append(
                    {
                        "row": int(row_idx),
                        "active": int(active.numel()),
                        "first": int(active[0].item()),
                        "last": int(active[-1].item()),
                    }
                )
            payload["loss_mask_spans"] = spans
        if labels is not None:
            labels_i = labels.detach()
            payload["label_negative_count"] = _int_item((labels_i < 0).sum())
            vocab_size = getattr(args, "padded_vocab_size", None)
            if vocab_size is not None:
                payload["label_ge_vocab_count"] = _int_item((labels_i >= int(vocab_size)).sum())
                payload["padded_vocab_size"] = int(vocab_size)
        if labels is not None and tokens is not None and loss_mask is not None:
            labels_i = labels.detach().reshape(-1)
            tokens_i = tokens.detach().reshape(-1)
            mask_i = loss_mask.detach().reshape(-1) > 0
            usable = min(labels_i.numel(), tokens_i.numel(), mask_i.numel())
            labels_i = labels_i[:usable]
            tokens_i = tokens_i[:usable]
            mask_i = mask_i[:usable]
            active_labels = labels_i[mask_i]
            active_tokens = tokens_i[mask_i]
            if active_labels.numel() > 0:
                payload["active_label_min"] = int(active_labels.min().detach().cpu().item())
                payload["active_label_max"] = int(active_labels.max().detach().cpu().item())
                payload["active_label_equals_token_frac"] = float(
                    (active_labels == active_tokens).float().mean().detach().cpu().item()
                )
                unique_vals, unique_counts = torch.unique(
                    active_labels, sorted=False, return_counts=True
                )
                topk = min(16, unique_counts.numel())
                if topk > 0:
                    counts, idxs = torch.topk(unique_counts, k=topk)
                    vals = unique_vals[idxs]
                    payload["active_label_top_ids"] = [
                        {"id": int(v.detach().cpu().item()), "count": int(c.detach().cpu().item())}
                        for v, c in zip(vals, counts)
                    ]
                if labels.dim() >= 2 and tokens.dim() >= 2 and loss_mask.dim() >= 2:
                    label_2d = labels.detach().reshape(labels.shape[0], -1)
                    token_2d = tokens.detach().reshape(tokens.shape[0], -1)
                    mask_2d = loss_mask.detach().reshape(loss_mask.shape[0], -1) > 0
                    cols = min(label_2d.shape[1], token_2d.shape[1], mask_2d.shape[1])
                    if cols > 1:
                        same_mask = mask_2d[:, : cols - 1]
                        next_token_match = label_2d[:, : cols - 1] == token_2d[:, 1:cols]
                        prev_token_match = label_2d[:, 1:cols] == token_2d[:, : cols - 1]
                        denom_next = max(int(same_mask.sum().detach().cpu().item()), 1)
                        denom_prev = max(int(mask_2d[:, 1:cols].sum().detach().cpu().item()), 1)
                        payload["active_label_equals_next_token_frac"] = float(
                            (next_token_match & same_mask).sum().detach().cpu().item()
                        ) / denom_next
                        payload["active_label_equals_prev_token_frac"] = float(
                            (prev_token_match & mask_2d[:, 1:cols]).sum().detach().cpu().item()
                        ) / denom_prev
                context = max(_env_int("MEGATRON_BATCH_DIAGNOSTIC_CONTEXT", 8), 0)
                if context > 0:
                    active_positions = torch.nonzero(mask_i, as_tuple=False).view(-1)
                    windows = []
                    for flat_pos in active_positions[: min(8, active_positions.numel())].detach().cpu().tolist():
                        left = max(0, int(flat_pos) - context)
                        right = min(usable, int(flat_pos) + context + 1)
                        windows.append(
                            {
                                "flat": int(flat_pos),
                                "token_window": [
                                    int(x) for x in tokens_i[left:right].detach().cpu().tolist()
                                ],
                                "label_window": [
                                    int(x) for x in labels_i[left:right].detach().cpu().tolist()
                                ],
                                "mask_window": [
                                    float(x)
                                    for x in mask_i[left:right].float().detach().cpu().tolist()
                                ],
                            }
                        )
                    payload["first_active_windows"] = windows
        print("[batch_diag] " + json.dumps(payload, sort_keys=True), flush=True)


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch."""
    args = get_args()
    config = core_transformer_config_from_args(args)

    if args.sequence_packing_scheduler is not None:
        return get_batch_on_this_rank_for_sequence_packing(
            data_iterator,
            vpp_size=config.virtual_pipeline_model_parallel_size,
            mtp_on_this_rank=mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage),
            vp_stage=vp_stage,
        )

    # TODO: this is pretty hacky, find a better way
    if not args.sft and not is_first_or_last_pipeline_stage(vp_stage) and (
    (not mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage))):
        return None, None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(
        data_iterator,
        mtp_on_this_rank=mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage),
        vp_stage=vp_stage,
    )

    cu_seqlens = batch.pop('cu_seqlens', None)
    cu_seqlens_padded = batch.pop('cu_seqlens_padded', None)
    max_seqlen = batch.pop('max_seqlen', None)
    local_cp_size = batch.pop('local_cp_size', None)
    if local_cp_size is not None:
        local_cp_size = int(local_cp_size.item())

    if cu_seqlens is None and local_cp_size is None:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)  # The implementation of this function is in MCore
        packed_seq_params = None
    elif local_cp_size is None:  # Packed THD format
        assert max_seqlen.dim() == 1
        batch, packed_seq_params = get_thd_batch_on_this_cp_rank(batch, cu_seqlens, cu_seqlens_padded, max_seqlen)
    else: # Hybrid CP format
        batch, packed_seq_params = get_batch_on_this_hybrid_cp_rank(batch, local_cp_size)
    
    return (*batch.values(), packed_seq_params)


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    model: Optional[GPTModel] = None,
    tokens: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    numeric_debug = None
    if _env_flag("MEGATRON_NUMERIC_DEBUG_LOSS"):
        from megatron.core import numeric_debug as _numeric_debug

        numeric_debug = _numeric_debug

    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        if numeric_debug is not None:
            numeric_debug.log_tensor("loss.output_tensor.raw", output_tensor, event="loss.output")
            numeric_debug.log_tensor("loss.loss_mask.raw", loss_mask, event="loss.mask")
        output_shape = tuple(output_tensor.shape)
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        # A masked-out token must not poison the local loss. In PyTorch,
        # NaN * 0 is still NaN, so zero inactive positions before applying the
        # mask while preserving loud failure for nonfinite active-token losses.
        active_loss_mask = loss_mask > 0
        masked_losses_for_sum = torch.where(active_loss_mask, losses, torch.zeros_like(losses))
        loss = torch.sum(masked_losses_for_sum * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

        dsa_indexer_loss = DSAIndexerAuxLossState.total()
        DSAIndexerAuxLossState.clear()
        _maybe_add_loss_diag_report(report, losses, loss_mask, num_tokens, loss.dtype)
        if dsa_indexer_loss is not None:
            loss = loss + dsa_indexer_loss
            report["dsa indexer loss"] = torch.cat(
                [dsa_indexer_loss.clone().detach().view(1), num_tokens.view(1)]
            )
            report["total loss"] = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])
            if numeric_debug is not None:
                numeric_debug.log_tensor(
                    "loss.dsa_indexer_loss", dsa_indexer_loss, event="loss.dsa"
                )
        _maybe_print_loss_diagnostics(
            losses,
            loss_mask,
            loss,
            num_tokens,
            output_shape,
            dsa_indexer_loss,
            tokens=tokens,
            labels=labels,
            position_ids=position_ids,
        )

    if numeric_debug is not None:
        numeric_debug.log_tensor("loss.scalar", loss, force=not torch.isfinite(loss).item())
        numeric_debug.log_tensor("loss.num_tokens", num_tokens, event="loss.tokens")

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()
    numeric_debug = None
    if _env_flag("MEGATRON_NUMERIC_DEBUG_BATCH") or _env_flag("MEGATRON_NUMERIC_DEBUG"):
        from megatron.core import numeric_debug as _numeric_debug

        numeric_debug = _numeric_debug

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            padding_mask,
            packed_seq_params,
        ) = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    if numeric_debug is not None and _env_flag("MEGATRON_NUMERIC_DEBUG_BATCH"):
        numeric_debug.set_context(phase="forward_step.batch")
        numeric_debug.log_tensor("batch.tokens", tokens, event="batch.tokens")
        numeric_debug.log_tensor("batch.labels", labels, event="batch.labels")
        numeric_debug.log_tensor("batch.loss_mask", loss_mask, event="batch.loss_mask")
        numeric_debug.log_tensor("batch.position_ids", position_ids, event="batch.position_ids")
        numeric_debug.log_tensor("batch.padding_mask", padding_mask, event="batch.padding_mask")
    _maybe_print_batch_diagnostics(tokens, labels, loss_mask, position_ids, padding_mask)

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=labels,
                    loss_mask=loss_mask,
                    padding_mask=padding_mask,
                )
                return schedule_plan, partial(
                    loss_func,
                    loss_mask,
                    model=model,
                    tokens=tokens,
                    labels=labels,
                    position_ids=position_ids,
                )
            else:
                output_tensor = model(
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=labels,
                    loss_mask=loss_mask,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(
        loss_func,
        loss_mask,
        model=model,
        tokens=tokens,
        labels=labels,
        position_ids=position_ids,
    )


def is_dataset_built_on_rank(vp_stage=None):
    args = get_args()
    config = core_transformer_config_from_args(args)
    return (
        args.sft
        or is_first_or_last_pipeline_stage(vp_stage)
        or mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    data_args = {
        "random_seed": args.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
        "fast_cache_load": args.dataloader_fast_cache_load,
        "sequences_per_dataset": sequences_per_dataset,
        "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
        "context_parallel_size": args.context_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "sequence_parallel_size": args.tensor_model_parallel_size*args.sequence_parallel,
        "hybrid_context_parallel": args.hybrid_context_parallel,
        "sft_mock_dataset_config_json":args.sft_mock_dataset_config_json,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        if args.mock_data:
            dataset_type = MockSFTDataset
        else:
            dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    is_dataset_built = partial(is_dataset_built_on_rank, vp_stage=vp_stage)
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
