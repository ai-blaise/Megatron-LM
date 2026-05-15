# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Gradient clipping."""

import os
import re
from typing import List, Optional, Union

import torch
from torch import inf

try:
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
        multi_tensor_scale,
    )

    l2_norm_impl = multi_tensor_l2norm
    multi_tensor_scale_impl = multi_tensor_scale
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier

        l2_norm_impl = amp_C.multi_tensor_l2norm
        multi_tensor_scale_impl = amp_C.multi_tensor_scale
    except ImportError:
        import warnings

        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of multi_tensor_applier, '
            'multi_tensor_l2norm, and multi_tensor_scale'
        )

        from megatron.core.utils import (
            local_multi_tensor_applier,
            local_multi_tensor_l2_norm,
            local_multi_tensor_scale,
        )

        multi_tensor_applier = local_multi_tensor_applier
        l2_norm_impl = local_multi_tensor_l2_norm
        multi_tensor_scale_impl = local_multi_tensor_scale


from ..tensor_parallel import param_is_not_tensor_parallel_duplicate
from ..transformer.module import param_is_not_shared
from ..utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor


_GRAD_OWNER_LABELS = (
    "embedding",
    "output",
    "attention_q",
    "attention_kv_down",
    "attention_kv_up",
    "attention_kv_norm",
    "attention_kv_other",
    "attention_wo",
    "attention_gate",
    "attention_other",
    "dsa_indexer",
    "moe_router",
    "moe_shared",
    "moe_expert_fc1",
    "moe_expert_fc2",
    "dense_mlp",
    "gated_norm",
    "norm",
    "other",
)
_GRAD_OWNER_TO_INDEX = {label: idx for idx, label in enumerate(_GRAD_OWNER_LABELS)}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _parse_rank_filter(value: str) -> set[int] | None:
    value = value.strip().lower()
    if not value or value in ("all", "*"):
        return None
    ranks: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ranks.update(range(int(start), int(end) + 1))
        else:
            ranks.add(int(part))
    return ranks


def _grad_ownership_rank_allowed() -> bool:
    rank = int(os.getenv("RANK", "0"))
    ranks = _parse_rank_filter(os.getenv("MEGATRON_GRAD_OWNERSHIP_RANKS", "all"))
    return ranks is None or rank in ranks


def _grad_ownership_iteration() -> int:
    try:
        from megatron.core import numeric_debug

        return numeric_debug.context_iteration(0)
    except Exception:
        return 0


def _grad_ownership_should_log() -> bool:
    if not _env_flag("MEGATRON_GRAD_OWNERSHIP"):
        return False
    if not _grad_ownership_rank_allowed():
        return False
    iteration = _grad_ownership_iteration()
    start = int(os.getenv("MEGATRON_GRAD_OWNERSHIP_START_ITER", "1"))
    if iteration and iteration < start:
        return False
    first_n = int(os.getenv("MEGATRON_GRAD_OWNERSHIP_FIRST_N", "8"))
    interval = int(os.getenv("MEGATRON_GRAD_OWNERSHIP_INTERVAL", "1"))
    if iteration == 0:
        return True
    return iteration <= first_n or (interval > 0 and iteration % interval == 0)


def _grad_owner_for_name(name: str) -> str:
    name = re.sub(r"^chunk\d+\.", "", name)
    if "word_embeddings" in name or ".embedding" in name or "embedding" in name:
        return "embedding"
    if "output_layer" in name or "final_linear" in name:
        return "output"
    if "dsa_indexer" in name or ".indexer" in name or "indexcache" in name:
        return "dsa_indexer"
    if ".router" in name or "router." in name:
        return "moe_router"
    if ".shared_experts." in name:
        return "moe_shared"
    if ".experts.linear_fc1." in name:
        return "moe_expert_fc1"
    if ".experts.linear_fc2." in name:
        return "moe_expert_fc2"
    if ".experts." in name:
        return "moe_shared"
    if "linear_gate_proj" in name or "attention_output_gate" in name or "g1" in name or ".gate" in name:
        return "attention_gate"
    if "linear_q" in name or ".q_" in name or "q_layernorm" in name:
        return "attention_q"
    if "linear_kv_down_proj" in name:
        return "attention_kv_down"
    if "linear_kv_up_proj" in name:
        return "attention_kv_up"
    if "kv_layernorm" in name:
        return "attention_kv_norm"
    if "linear_kv" in name or ".kv_" in name:
        return "attention_kv_other"
    if "self_attention.linear_proj" in name or "self_attention.proj" in name:
        return "attention_wo"
    if "self_attention" in name or ".attention." in name:
        return "attention_other"
    if "linear_fc" in name or ".mlp." in name:
        return "dense_mlp"
    if "gated_norm" in name or "gatednorm" in name:
        return "gated_norm"
    if "norm" in name or "layernorm" in name:
        return "norm"
    return "other"


def _grad_for_param(param: torch.Tensor) -> Optional[torch.Tensor]:
    grad = None
    if hasattr(param, "decoupled_grad") and param.decoupled_grad is not None:
        grad = param.decoupled_grad
    elif param.grad is not None:
        grad = param.grad
    if grad is None:
        return None
    return to_local_if_dtensor(grad).detach()


def _grad_l2_norm(grads: list[torch.Tensor]) -> torch.Tensor:
    if not grads:
        return torch.zeros((), dtype=torch.float, device='cuda')
    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device='cuda')
    total_sq = torch.zeros((), dtype=torch.float, device='cuda')
    grads_by_dtype = {}
    for grad in grads:
        grads_by_dtype.setdefault(grad.dtype, []).append(grad)
    for dtype_grads in grads_by_dtype.values():
        norm, _ = multi_tensor_applier(
            l2_norm_impl,
            dummy_overflow_buf,
            [dtype_grads],
            False,
        )
        norm = norm.float().reshape(())
        total_sq += norm.float() * norm.float()
    return torch.sqrt(total_sq)


def _log_grad_ownership(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    *,
    stage: str,
    max_norm: Union[int, float],
    total_norm: float,
    clip_coeff: float,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    if not _env_flag("MEGATRON_GRAD_OWNERSHIP"):
        return
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    rank = int(os.getenv("RANK", "0"))
    iteration = _grad_ownership_iteration()
    should_print = _grad_ownership_should_log()
    owner_grads: dict[str, list[torch.Tensor]] = {label: [] for label in _GRAD_OWNER_LABELS}
    owner_tensors = torch.zeros(len(_GRAD_OWNER_LABELS), dtype=torch.float, device='cuda')
    owner_elems = torch.zeros(len(_GRAD_OWNER_LABELS), dtype=torch.float, device='cuda')
    data_parallel_group = None
    top_candidates: list[tuple[float, str, str, int]] = []
    top_param_limit = (
        int(os.getenv("MEGATRON_GRAD_OWNERSHIP_TOP_PARAMS", "0")) if should_print else 0
    )

    for param in parameters:
        if not param_is_not_shared(param) or not param_is_not_tensor_parallel_duplicate(param):
            continue
        grad = _grad_for_param(param)
        if grad is None:
            continue
        data_parallel_group = get_data_parallel_group_if_dtensor(grad, data_parallel_group)
        name = getattr(param, "_numeric_debug_name", "<unnamed>")
        owner = _grad_owner_for_name(name)
        owner_idx = _GRAD_OWNER_TO_INDEX[owner]
        owner_grads[owner].append(grad)
        owner_tensors[owner_idx] += 1.0
        owner_elems[owner_idx] += float(grad.numel())
        if top_param_limit > 0 and stage == "preclip":
            try:
                norm = float(torch.norm(grad, 2).item())
                top_candidates.append((norm, owner, name, int(grad.numel())))
            except Exception:
                pass

    owner_sq = torch.zeros(len(_GRAD_OWNER_LABELS), dtype=torch.float, device='cuda')
    for owner, grads in owner_grads.items():
        if not grads:
            continue
        norm = _grad_l2_norm(grads)
        owner_sq[_GRAD_OWNER_TO_INDEX[owner]] = norm * norm

    if data_parallel_group:
        torch.distributed.all_reduce(
            owner_sq, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )
        torch.distributed.all_reduce(
            owner_tensors, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )
        torch.distributed.all_reduce(
            owner_elems, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )
    if grad_stats_parallel_group:
        torch.distributed.all_reduce(
            owner_sq, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
        )
        torch.distributed.all_reduce(
            owner_tensors, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
        )
        torch.distributed.all_reduce(
            owner_elems, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
        )

    owner_norms = torch.sqrt(owner_sq).detach().cpu()
    owner_sq_cpu = owner_sq.detach().cpu()
    owner_tensors_cpu = owner_tensors.detach().cpu()
    owner_elems_cpu = owner_elems.detach().cpu()
    if not should_print:
        return

    local_sq_sum = float(owner_sq_cpu.sum().item())
    pct_sq = max(local_sq_sum, 1.0e-30)
    print(
        "[grad_ownership] "
        f"rank={rank} iter={iteration} stage={stage} "
        f"total_norm={float(total_norm):.6e} owner_norm={local_sq_sum ** 0.5:.6e} "
        f"max_norm={float(max_norm):.6e} clip_coeff={float(clip_coeff):.6e}",
        flush=True,
    )

    rows = []
    for idx, owner in enumerate(_GRAD_OWNER_LABELS):
        if owner_tensors_cpu[idx].item() == 0:
            continue
        pct = 100.0 * float(owner_sq_cpu[idx].item()) / pct_sq
        rows.append((pct, owner, float(owner_norms[idx].item()), int(owner_tensors_cpu[idx].item()), int(owner_elems_cpu[idx].item())))
    top_owner_limit = int(os.getenv("MEGATRON_GRAD_OWNERSHIP_TOP_OWNERS", "32"))
    for pct, owner, norm, tensor_count, elem_count in sorted(rows, reverse=True)[:top_owner_limit]:
        print(
            "[grad_ownership.owner] "
            f"rank={rank} iter={iteration} stage={stage} owner={owner} "
            f"norm={norm:.6e} pct_owner_sq={pct:.3f} tensors={tensor_count} elems={elem_count}",
            flush=True,
        )

    if top_param_limit > 0 and stage == "preclip":
        for norm, owner, name, elem_count in sorted(
            top_candidates, key=lambda item: item[0], reverse=True
        )[:top_param_limit]:
            print(
                "[grad_ownership.param] "
                f"rank={rank} iter={iteration} stage={stage} owner={owner} "
                f"norm={norm:.6e} elems={elem_count} name={name}",
                flush=True,
            )


def get_grad_norm_fp32(
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    norm_type: Union[int, float] = 2,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Calculate the norm of gradients in fp32.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters.

    Arguments:
        grads_for_norm (Iterable[Tensor] or Tensor): an iterable of Tensors or a single
            Tensor that will be used for calculating the grad norm.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        grad_stats_parallel_group (group): Process group for reducing the grad norms. This is
            generally the model-parallel group for non-distributed optimizers, and the entire
            world for the distributed optimizer.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    data_parallel_group = None
    for grad in grads_for_norm:
        data_parallel_group = get_data_parallel_group_if_dtensor(grad, data_parallel_group)

    grads_for_norm = [to_local_if_dtensor(grad) for grad in grads_for_norm]

    # Norm parameters.
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
            )
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=grad_stats_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device='cuda')
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:
                total_norm = torch.zeros(1, dtype=torch.float, device='cuda')
                grads_by_dtype = {}
                for grad in grads_for_norm:
                    grads_by_dtype.setdefault(grad.dtype, []).append(grad)
                for dtype_grads_for_norm in grads_by_dtype.values():
                    grad_norm, _ = multi_tensor_applier(
                        l2_norm_impl,
                        dummy_overflow_buf,
                        [dtype_grads_for_norm],
                        False,  # no per-parameter norm
                    )
                    total_norm += grad_norm.float() ** norm_type
            else:
                total_norm = torch.zeros(1, dtype=torch.float, device='cuda')

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm**norm_type

        # Sum across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
            )
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)

    return total_norm


def clip_grad_by_total_norm_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    max_norm: Union[int, float],
    total_norm: float,
    use_decoupled_grad: bool = False,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """Clips gradient of an iterable of parameters in fp32 by total norm.

    Note that the gradients are modified in place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized.
        max_norm (float or int): max norm of the gradients.
        total_norm (float): total norm of the gradients.
        use_decoupled_grad (bool, optional): whether to read grad from ".grad" or ".decoupled_grad",
            default value is False.
    """
    # Grads — check decoupled_grad first (used by precision-aware and
    # FlashAdamW with non-FP32 params), then fall back to .grad.
    grads_by_dtype = {}
    for param in parameters:
        grad = None
        if hasattr(param, "decoupled_grad") and param.decoupled_grad is not None:
            assert param.decoupled_grad.dtype in [torch.float32, torch.bfloat16]
            grad = param.decoupled_grad
        elif param.grad is not None:
            assert param.grad.type() == 'torch.cuda.FloatTensor'
            grad = param.grad
        if grad is not None:
            local_grad = to_local_if_dtensor(grad).detach()
            grads_by_dtype.setdefault(local_grad.dtype, []).append(local_grad)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    _log_grad_ownership(
        parameters,
        stage="preclip",
        max_norm=max_norm,
        total_norm=total_norm,
        clip_coeff=clip_coeff,
        grad_stats_parallel_group=grad_stats_parallel_group,
    )
    if clip_coeff < 1.0:
        fp32_grads = grads_by_dtype.pop(torch.float32, [])
        if fp32_grads:
            dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device='cuda')
            multi_tensor_applier(
                multi_tensor_scale_impl, dummy_overflow_buf, [fp32_grads, fp32_grads], clip_coeff
            )

        # TE/Apex multi_tensor_scale can corrupt BF16 decoupled grad views in the
        # FlashAdamW/NVFP4 path. Use torch's native foreach scaling for non-FP32
        # grads so clipping is still in-place without passing these views through
        # the fused scale kernel.
        for dtype_grads in grads_by_dtype.values():
            try:
                torch._foreach_mul_(dtype_grads, clip_coeff)
            except RuntimeError:
                for grad in dtype_grads:
                    grad.mul_(clip_coeff)
    _log_grad_ownership(
        parameters,
        stage="postclip",
        max_norm=max_norm,
        total_norm=total_norm,
        clip_coeff=min(float(clip_coeff), 1.0),
        grad_stats_parallel_group=grad_stats_parallel_group,
    )


def count_zeros_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    grad_stats_parallel_group: torch.distributed.ProcessGroup,
    use_decoupled_grad: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Counts the number of zeros in gradients associated with the passed-in list of
    parameters.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have the number of zeros in its corresponding
            gradient counted.
        grad_stats_parallel_group (group): Process group for reducing the num_zeros count. This is
            generally the model-parallel group for non-distributed optimizers, and the entire
            world for the distributed optimizer.
        use_decoupled_grad (bool, optional) whether to read grad from ".grad" or ".decoupled_grad",
            default value is False.
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    total_num_zeros = torch.zeros(1, dtype=torch.float, device='cuda')
    data_parallel_group = None
    use_megatron_fsdp = False
    for param in parameters:
        if getattr(param, "__fsdp_param__", False) and param.grad is not None:
            # If the parameter is managed by Megatron FSDP, we need to handle it differently.
            use_megatron_fsdp = True
            grad = param.grad._local_tensor
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros += num_zeros
            continue

        grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
        grad_not_none = hasattr(param, grad_attr) and getattr(param, grad_attr) is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param, tp_group=tp_group)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grad_obj = getattr(param, grad_attr)
            data_parallel_group = get_data_parallel_group_if_dtensor(grad_obj, data_parallel_group)
            grad = to_local_if_dtensor(grad_obj).detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros

    if use_megatron_fsdp and data_parallel_group is not None:
        raise ValueError(
            "Unexpected use of Megatron FSDP with data parallel group. "
            "Please ensure that the parameters are properly managed by Megatron FSDP."
        )

    # Sum across all data-parallel GPUs if using FSDP.
    if data_parallel_group:
        torch.distributed.all_reduce(
            total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(
        total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
    )

    total_num_zeros = total_num_zeros.item()

    return total_num_zeros
