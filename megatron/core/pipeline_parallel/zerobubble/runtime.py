# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Executable ZeroBubble pipeline runtime adapters.

The implementation ports the core reference idea used by ZeroBubble: split the
activation-gradient backward (B) from weight-gradient GEMMs (W), then let the
runtime explicitly drain W work from ``WeightGradStore``.  Unsupported reference
features such as sequence splitting, CPU activation offload, post-validation,
and overlapped P2P are guarded here instead of silently falling back.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterator, List, Optional, Union

import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    _as_tensor_list,
    _detach_pipeline_tensors,
    backward_step,
    check_first_val_step,
    clear_embedding_activation_buffer,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    forward_step,
    get_tensor_shapes,
)
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.zbpp_utils import WeightGradStore


def _coerce_single_model_and_iterator(model, data_iterator, schedule_name):
    if isinstance(model, list):
        if len(model) != 1:
            raise ValueError(f"{schedule_name} supports one local model chunk; use zero_bubble_v for V")
        model = model[0]
    if isinstance(data_iterator, list):
        if len(data_iterator) != 1:
            raise ValueError(
                f"{schedule_name} supports one local data iterator; use zero_bubble_v for V"
            )
        data_iterator = data_iterator[0]
    return model, data_iterator


def _build_default_communicator_and_groups(config):
    p2p_communicator = P2PCommunicator(
        pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
    )
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = parallel_state.get_tensor_model_parallel_group()
    pg_collection.cp = parallel_state.get_context_parallel_group()
    pg_collection.embd = parallel_state.get_embedding_group(check_initialized=False)
    pg_collection.pos_embd = parallel_state.get_position_embedding_group(check_initialized=False)
    pg_collection.pp = parallel_state.get_pipeline_model_parallel_group()
    pg_collection.dp_cp = parallel_state.get_data_parallel_group(
        with_context_parallel=True, partial_data_parallel=False
    )
    return p2p_communicator, pg_collection


def _validate_groups(p2p_communicator, pg_collection):
    assert hasattr(p2p_communicator, "config"), "p2p_communicator must have a config"
    for name in ("tp", "cp", "embd", "pos_embd", "pp", "dp_cp"):
        assert hasattr(pg_collection, name), f"pg_collection must have a {name} group"


def _get_communicator_and_groups(config, p2p_communicator, pg_collection):
    if p2p_communicator is None and pg_collection is None:
        return _build_default_communicator_and_groups(config)
    if p2p_communicator is not None and pg_collection is not None:
        _validate_groups(p2p_communicator, pg_collection)
        return p2p_communicator, pg_collection
    raise ValueError(
        "Invalid combination of p2p_communicator and pg_collection: provide neither or both"
    )


def _validate_common_config(config, schedule_name, *, allow_vp, forward_only):
    if config.overlap_p2p_comm:
        raise ValueError(f"{schedule_name} does not support overlap_p2p_comm yet")
    if getattr(config, "use_ring_exchange_p2p", False):
        raise ValueError(f"{schedule_name} does not support ring-exchange P2P")
    if getattr(config, "overlap_moe_expert_parallel_comm", False):
        raise ValueError(f"{schedule_name} does not support overlap_moe_expert_parallel_comm")
    if getattr(config, "fine_grained_activation_offloading", False):
        raise ValueError(f"{schedule_name} does not support fine-grained activation offload")
    if getattr(config, "cpu_offloading", False) or getattr(config, "cpu_offload", False):
        raise ValueError(f"{schedule_name} does not support CPU activation offload")
    if getattr(config, "variable_seq_lengths", False):
        raise ValueError(f"{schedule_name} does not support variable sequence lengths")
    if getattr(config, "mtp_standalone", False):
        raise ValueError(f"{schedule_name} does not support standalone MTP pipeline exchange")
    if getattr(config, "num_microbatches_with_partial_activation_checkpoints", None) is not None:
        raise ValueError(f"{schedule_name} does not support partial activation checkpoint windows")
    if config.param_sync_func is not None:
        raise ValueError(f"{schedule_name} does not support asynchronous parameter synchronization")
    if not allow_vp and config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError(f"{schedule_name} does not support virtual pipeline stages")
    if config.pipeline_dtype is None:
        raise RuntimeError(f"pipeline_dtype must be provided for {schedule_name}")
    if not forward_only and getattr(config, "overlap_grad_reduce", False):
        raise ValueError(f"{schedule_name} does not support overlapped DDP gradient reduction")
    if not forward_only and getattr(config, "delay_wgrad_compute", False):
        raise ValueError(f"{schedule_name} does not support Transformer Engine delayed WGRAD")
    if not forward_only and getattr(config, "transformer_impl", "local") == "transformer_engine":
        raise ValueError(f"{schedule_name} does not support transformer_engine implementation yet")


def _no_sync_context(config, model):
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        return multi_no_sync
    if no_sync_func is None:
        return contextlib.nullcontext
    return no_sync_func


def _drain_one_wgrad(chunk=0):
    if WeightGradStore.queue_size(chunk=chunk) > 0:
        WeightGradStore.pop(chunk=chunk)


def _drain_all_wgrads(num_chunks):
    for chunk in range(num_chunks):
        WeightGradStore.clear(chunk=chunk)


def _run_backward_split(input_tensor, output_tensor, output_tensor_grad, model_type, config, chunk=0):
    with WeightGradStore.set_split_bw(True):
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )
        WeightGradStore.flush(chunk=chunk)
    return input_tensor_grad


def _validate_single_p2p_tensor_shapes(recv_tensor_shapes, send_tensor_shapes, schedule_name):
    if len(recv_tensor_shapes) != 1 or len(send_tensor_shapes) != 1:
        raise ValueError(f"{schedule_name} currently supports exactly one pipeline tensor")
    if recv_tensor_shapes[0] != send_tensor_shapes[0]:
        raise ValueError(f"{schedule_name} requires matching send and receive tensor shapes")
    return recv_tensor_shapes[0]


def _validate_p2p_send_tensor_shape(tensor, tensor_shape, schedule_name):
    if tuple(tensor.size()) != tuple(tensor_shape):
        raise ValueError(
            f"{schedule_name} expected pipeline tensor shape {tensor_shape}, "
            f"got {tuple(tensor.size())}"
        )


def forward_backward_pipelining_with_zero_bubble(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run a non-interleaved ZeroBubble runtime with explicit B/W split."""

    model, data_iterator = _coerce_single_model_and_iterator(model, data_iterator, "zero_bubble")
    config = get_model_config(model)
    _validate_common_config(config, "zero_bubble", allow_vp=False, forward_only=forward_only)
    p2p_communicator, pg_collection = _get_communicator_and_groups(
        config, p2p_communicator, pg_collection
    )
    pp_group = p2p_communicator.pp_group
    if pp_group.size() <= 1:
        raise ValueError("zero_bubble requires pipeline_model_parallel_size > 1")

    WeightGradStore.reset(num_chunks=1)

    embedding_module = None
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model, is_pp_last_stage(pp_group))

    if config.timers is not None:
        config.timers("forward-backward", log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync = _no_sync_context(config, model)
    no_sync_context = no_sync()
    no_sync_context.__enter__()
    grad_sync_enabled = False

    def enable_grad_sync():
        nonlocal grad_sync_enabled
        if not grad_sync_enabled:
            no_sync_context.__exit__(None, None, None)
            grad_sync_enabled = True

    model_type = get_model_type(model)
    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=pg_collection.tp,
        cp_group=pg_collection.cp,
        pp_group=pp_group,
        is_recv=True,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=pg_collection.tp,
        cp_group=pg_collection.cp,
        pp_group=pp_group,
        is_recv=False,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    num_warmup_microbatches = min(pp_group.size() - pp_group.rank() - 1, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches
    forward_data_store = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
    input_tensors = []
    output_tensors = []

    try:
        for i in range(num_warmup_microbatches):
            input_tensor = p2p_communicator.recv_forward(
                recv_tensor_shapes, is_pp_first_stage(pp_group)
            )
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                cp_group_size=pg_collection.cp.size(),
                collect_non_loss_data=collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                is_last_stage=is_pp_last_stage(pp_group),
            )
            total_num_tokens += num_tokens
            p2p_communicator.send_forward(output_tensor, is_pp_last_stage(pp_group))
            if not forward_only:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        input_tensor = None
        if num_microbatches_remaining > 0:
            input_tensor = p2p_communicator.recv_forward(
                recv_tensor_shapes, is_pp_first_stage(pp_group)
            )

        for i in range(num_microbatches_remaining):
            microbatch_id = i + num_warmup_microbatches
            last_iteration = i == num_microbatches_remaining - 1
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                cp_group_size=pg_collection.cp.size(),
                collect_non_loss_data=collect_non_loss_data,
                is_first_microbatch=check_first_val_step(
                    first_val_step, forward_only, i == 0 and num_warmup_microbatches == 0
                ),
                current_microbatch=microbatch_id,
                is_last_stage=is_pp_last_stage(pp_group),
            )
            total_num_tokens += num_tokens

            if forward_only:
                p2p_communicator.send_forward(output_tensor, is_pp_last_stage(pp_group))
                if not last_iteration:
                    input_tensor = p2p_communicator.recv_forward(
                        recv_tensor_shapes, is_pp_first_stage(pp_group)
                    )
                continue

            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, is_pp_last_stage(pp_group)
            )
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            if num_warmup_microbatches == 0 and last_iteration:
                enable_grad_sync()
            input_tensor_grad = _run_backward_split(
                input_tensor, output_tensor, output_tensor_grad, model_type, config, chunk=0
            )

            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(input_tensor_grad, is_pp_first_stage(pp_group))
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, is_pp_first_stage(pp_group)
                )
            _drain_one_wgrad(chunk=0)

        if not forward_only:
            for i in range(num_warmup_microbatches):
                if i == num_warmup_microbatches - 1:
                    enable_grad_sync()
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)
                output_tensor_grad = p2p_communicator.recv_backward(
                    send_tensor_shapes, is_pp_last_stage(pp_group)
                )
                input_tensor_grad = _run_backward_split(
                    input_tensor, output_tensor, output_tensor_grad, model_type, config, chunk=0
                )
                p2p_communicator.send_backward(input_tensor_grad, is_pp_first_stage(pp_group))
                _drain_one_wgrad(chunk=0)

            if not grad_sync_enabled:
                enable_grad_sync()
            _drain_all_wgrads(num_chunks=1)
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())
            if config.finalize_model_grads_func is not None:
                finish_embedding_wgrad_compute(
                    config, embedding_module, is_pp_last_stage(pp_group), pg_collection.tp
                )
                config.finalize_model_grads_func(
                    [model],
                    total_num_tokens if config.calculate_per_token_loss else None,
                    pg_collection=pg_collection,
                    force_all_reduce=force_all_reduce,
                )
        elif not grad_sync_enabled:
            enable_grad_sync()
    finally:
        if not grad_sync_enabled:
            no_sync_context.__exit__(None, None, None)
        WeightGradStore.assert_empty()
        if config.timers is not None:
            config.timers("forward-backward").stop()

    if (
        hasattr(config, "cuda_graph_impl")
        and config.cuda_graph_impl == "local"
        and CudaGraphScope.full_iteration not in config.cuda_graph_scope
    ):
        create_cudagraphs()
    return forward_data_store


def forward_backward_pipelining_with_zero_bubble_v(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run a conservative V-shaped ZeroBubble runtime with explicit B/W split."""

    if not isinstance(model, list) or len(model) != 2:
        raise ValueError("zero_bubble_v requires exactly two local model chunks")
    if not isinstance(data_iterator, list) or len(data_iterator) != 2:
        raise ValueError("zero_bubble_v requires one data iterator per model chunk")
    if adjust_tensor_shapes_fn is not None:
        raise ValueError("zero_bubble_v does not support adjust_tensor_shapes_fn")

    config = get_model_config(model[0])
    _validate_common_config(config, "zero_bubble_v", allow_vp=True, forward_only=forward_only)
    if config.virtual_pipeline_model_parallel_size != 2:
        raise ValueError("zero_bubble_v requires virtual_pipeline_model_parallel_size == 2")
    p2p_communicator, pg_collection = _get_communicator_and_groups(
        config, p2p_communicator, pg_collection
    )
    pp_group = p2p_communicator.pp_group
    if pp_group.size() <= 1:
        raise ValueError("zero_bubble_v requires pipeline_model_parallel_size > 1")

    WeightGradStore.reset(num_chunks=2)

    if config.timers is not None:
        config.timers("forward-backward", log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync = _no_sync_context(config, model)
    no_sync_context = no_sync()
    no_sync_context.__enter__()
    grad_sync_enabled = False

    def enable_grad_sync():
        nonlocal grad_sync_enabled
        if not grad_sync_enabled:
            no_sync_context.__exit__(None, None, None)
            grad_sync_enabled = True

    embedding_module = None
    is_zbv_last_stage = pp_group.rank() == 0
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model, is_zbv_last_stage)

    model_type = get_model_type(model[0])
    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=pg_collection.tp,
        cp_group=pg_collection.cp,
        pp_group=pp_group,
        is_recv=True,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=pg_collection.tp,
        cp_group=pg_collection.cp,
        pp_group=pp_group,
        is_recv=False,
    )
    tensor_shape = _validate_single_p2p_tensor_shapes(
        recv_tensor_shapes, send_tensor_shapes, "zero_bubble_v"
    )

    forward_data_store = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
    input_tensors = [[], []]
    output_tensors = [[], []]
    input_tensor_grads = [[], []]
    output_tensor_grads = [[], []]
    current_f_chunk_id = [0, 0]
    current_b_chunk_id = [0, 0]
    comm_ops = []
    to_deallocate = []
    rank = pp_group.rank()
    pp_size = pp_group.size()

    def commit_and_wait_comm():
        nonlocal comm_ops, to_deallocate
        if not comm_ops:
            return
        reqs = torch.distributed.batch_isend_irecv(comm_ops)
        for req in reqs:
            req.wait()
        if config.batch_p2p_sync:
            torch.cuda.synchronize()
        comm_ops = []
        for tensor in to_deallocate:
            deallocate_output_tensor(tensor, config.deallocate_pipeline_outputs)
        to_deallocate = []

    def queue_recv(peer_rank):
        tensor = torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )
        comm_ops.append(torch.distributed.P2POp(torch.distributed.irecv, tensor, peer_rank, pp_group))
        return tensor

    def queue_send(tensors, peer_rank, deallocate=False):
        tensor_list = _as_tensor_list(tensors)
        if len(tensor_list) != 1:
            raise ValueError("zero_bubble_v currently supports exactly one pipeline tensor")
        for tensor in tensor_list:
            _validate_p2p_send_tensor_shape(tensor, tensor_shape, "zero_bubble_v")
            comm_ops.append(
                torch.distributed.P2POp(torch.distributed.isend, tensor, peer_rank, pp_group)
            )
            if deallocate:
                to_deallocate.append(tensor)

    def forward_compute_phase(phase):
        nonlocal total_num_tokens
        chunk_id = current_f_chunk_id[phase]
        current_f_chunk_id[phase] += 1
        if phase == 0 and rank == 0 and len(input_tensors[phase]) <= chunk_id:
            input_tensors[phase].append(None)
        input_tensor = input_tensors[phase][chunk_id]
        is_last_stage = phase == 1 and rank == 0
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[phase],
            model[phase],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, chunk_id == 0),
            current_microbatch=chunk_id,
            vp_stage=phase,
            is_last_stage=is_last_stage,
        )
        total_num_tokens += num_tokens
        output_tensors[phase].append(output_tensor)
        if phase == 0 and rank == pp_size - 1:
            input_tensors[1].append(_detach_pipeline_tensors(output_tensor))
            if not forward_only:
                for tensor in _as_tensor_list(output_tensor):
                    deallocate_output_tensor(tensor, config.deallocate_pipeline_outputs)

    def backward_compute_phase(phase):
        if forward_only:
            return
        chunk_id = current_b_chunk_id[phase]
        current_b_chunk_id[phase] += 1
        if phase == 1 and rank == 0 and len(output_tensor_grads[phase]) <= chunk_id:
            output_tensor_grads[phase].append(None)
        input_tensor = input_tensors[phase][chunk_id]
        output_tensor = output_tensors[phase][chunk_id]
        output_tensor_grad = output_tensor_grads[phase][chunk_id]
        input_tensor_grad = _run_backward_split(
            input_tensor, output_tensor, output_tensor_grad, model_type, config, chunk=phase
        )
        if phase == 1 and rank == pp_size - 1:
            output_tensor_grads[0].append(input_tensor_grad)
        else:
            input_tensor_grads[phase].append(input_tensor_grad)

    def forward_microbatch():
        if rank == 0:
            forward_compute_phase(0)
        for edge_rank in range(pp_size - 1):
            if rank == edge_rank:
                queue_send(output_tensors[0][-1], p2p_communicator.next_rank, deallocate=not forward_only)
            elif rank == edge_rank + 1:
                input_tensors[0].append(queue_recv(p2p_communicator.prev_rank))
            commit_and_wait_comm()
            if rank == edge_rank + 1:
                forward_compute_phase(0)

        if rank == pp_size - 1:
            forward_compute_phase(1)
        for edge_rank in range(pp_size - 1, 0, -1):
            if rank == edge_rank:
                queue_send(output_tensors[1][-1], p2p_communicator.prev_rank, deallocate=not forward_only)
            elif rank == edge_rank - 1:
                input_tensors[1].append(queue_recv(p2p_communicator.next_rank))
            commit_and_wait_comm()
            if rank == edge_rank - 1:
                forward_compute_phase(1)

    def backward_microbatch():
        if rank == 0:
            backward_compute_phase(1)
        for edge_rank in range(pp_size - 1):
            if rank == edge_rank:
                queue_send(input_tensor_grads[1][-1], p2p_communicator.next_rank)
            elif rank == edge_rank + 1:
                output_tensor_grads[1].append(queue_recv(p2p_communicator.prev_rank))
            commit_and_wait_comm()
            if rank == edge_rank:
                _drain_one_wgrad(chunk=1)
            if rank == edge_rank + 1:
                backward_compute_phase(1)
        if rank == pp_size - 1:
            _drain_one_wgrad(chunk=1)

        if rank == pp_size - 1:
            backward_compute_phase(0)
        for edge_rank in range(pp_size - 1, 0, -1):
            if rank == edge_rank:
                queue_send(input_tensor_grads[0][-1], p2p_communicator.prev_rank)
            elif rank == edge_rank - 1:
                output_tensor_grads[0].append(queue_recv(p2p_communicator.next_rank))
            commit_and_wait_comm()
            if rank == edge_rank:
                _drain_one_wgrad(chunk=0)
            if rank == edge_rank - 1:
                backward_compute_phase(0)
        if rank == 0:
            _drain_one_wgrad(chunk=0)

    try:
        for _ in range(num_microbatches):
            forward_microbatch()
            if not forward_only:
                backward_microbatch()

        if not forward_only:
            enable_grad_sync()
            _drain_all_wgrads(num_chunks=2)
            grad_sync_func = config.grad_sync_func
            if grad_sync_func is not None:
                if not isinstance(grad_sync_func, list):
                    grad_sync_func = [grad_sync_func for _ in model]
                for model_chunk_id in range(2):
                    grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
            if config.finalize_model_grads_func is not None:
                finish_embedding_wgrad_compute(config, embedding_module, is_zbv_last_stage, pg_collection.tp)
                config.finalize_model_grads_func(
                    model,
                    total_num_tokens if config.calculate_per_token_loss else None,
                    pg_collection=pg_collection,
                    force_all_reduce=force_all_reduce,
                )
        else:
            enable_grad_sync()
    finally:
        if not grad_sync_enabled:
            no_sync_context.__exit__(None, None, None)
        WeightGradStore.assert_empty()
        if config.timers is not None:
            config.timers("forward-backward").stop()

    if (
        hasattr(config, "cuda_graph_impl")
        and config.cuda_graph_impl == "local"
        and CudaGraphScope.full_iteration not in config.cuda_graph_scope
    ):
        create_cudagraphs()
    return forward_data_store
