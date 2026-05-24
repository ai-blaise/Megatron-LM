# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import os
from typing import Optional

from megatron.core.utils import internal_api

try:
    from deep_ep import Buffer
    from deep_ep import Config as DeepEPConfig
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False
    DeepEPConfig = None

import torch

_buffer = None
_combine_config_cache = {}


def _parse_int_env(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _combine_config_group_enabled(group_size: int) -> bool:
    raw = os.getenv("MEGATRON_DEEPEP_COMBINE_CONFIG_GROUP_SIZE", "").strip()
    if not raw:
        return False
    groups = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed <= 0:
            raise ValueError(
                "MEGATRON_DEEPEP_COMBINE_CONFIG_GROUP_SIZE entries must be positive, "
                f"got {parsed}"
            )
        groups.add(parsed)
    return group_size in groups


def _get_combine_config_override(group_size: int):
    """Return an explicit DeepEP combine config override, or None.

    DeepEP's package defaults are tuned per rank count but are opaque at the
    Megatron callsite. This override is deliberately env-gated and group-size
    gated so a config measured for one topology cannot silently apply to another.
    """

    if not HAVE_DEEP_EP or DeepEPConfig is None:
        return None
    sms = _parse_int_env("MEGATRON_DEEPEP_COMBINE_NUM_SMS")
    nvl_send = _parse_int_env("MEGATRON_DEEPEP_COMBINE_NVL_SEND_TOKENS")
    nvl_recv = _parse_int_env("MEGATRON_DEEPEP_COMBINE_NVL_RECV_TOKENS")
    rdma_send = _parse_int_env("MEGATRON_DEEPEP_COMBINE_RDMA_SEND_TOKENS")
    rdma_recv = _parse_int_env("MEGATRON_DEEPEP_COMBINE_RDMA_RECV_TOKENS")

    provided = [sms is not None, nvl_send is not None, nvl_recv is not None]
    if not any(provided):
        return None
    if not all(provided):
        raise RuntimeError(
            "DeepEP combine override requires all of "
            "MEGATRON_DEEPEP_COMBINE_NUM_SMS, "
            "MEGATRON_DEEPEP_COMBINE_NVL_SEND_TOKENS, and "
            "MEGATRON_DEEPEP_COMBINE_NVL_RECV_TOKENS"
        )
    if not os.getenv("MEGATRON_DEEPEP_COMBINE_CONFIG_GROUP_SIZE", "").strip():
        raise RuntimeError(
            "DeepEP combine override requires MEGATRON_DEEPEP_COMBINE_CONFIG_GROUP_SIZE "
            "so a topology-tuned config cannot apply to the wrong process-group size"
        )
    if not _combine_config_group_enabled(group_size):
        return None
    if getattr(Buffer, "num_sms", None) is not None and sms != Buffer.num_sms:
        raise RuntimeError(
            "DeepEP combine override num_sms must match Buffer.num_sms because combine "
            "reuses the dispatch handle channel layout. "
            f"Got override={sms}, Buffer.num_sms={Buffer.num_sms}"
        )

    key = (group_size, sms, nvl_send, nvl_recv, rdma_send, rdma_recv)
    config = _combine_config_cache.get(key)
    if config is None:
        if rdma_send is None and rdma_recv is None:
            config = DeepEPConfig(sms, nvl_send, nvl_recv)
        elif rdma_send is not None and rdma_recv is not None:
            config = DeepEPConfig(sms, nvl_send, nvl_recv, rdma_send, rdma_recv)
        else:
            raise RuntimeError(
                "DeepEP combine RDMA override requires both "
                "MEGATRON_DEEPEP_COMBINE_RDMA_SEND_TOKENS and "
                "MEGATRON_DEEPEP_COMBINE_RDMA_RECV_TOKENS"
            )
        _combine_config_cache[key] = config
    return config


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    configs = [
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ]
    combine_override = _get_combine_config_override(group.size())
    if combine_override is not None:
        configs.append(combine_override)

    for config in configs:
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            config=_get_combine_config_override(ctx.group.size()),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class FusedDispatchExpertMajor(torch.autograd.Function):
    """Strict DeepEP dispatch that writes directly into local expert-major order."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        num_local_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        if not hasattr(buffer, "dispatch_expert_major"):
            raise RuntimeError(
                "Requested strict expert-major DeepEP dispatch, but installed DeepEP "
                "does not expose Buffer.dispatch_expert_major"
            )
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if num_tokens_per_rdma_rank is not None:
            raise RuntimeError(
                "Strict expert-major DeepEP dispatch only supports intranode DeepEP; "
                "refusing to fall back"
            )
        (
            expert_x,
            recv_token_indices,
            permuted_token_probs,
            row_map,
            edge_map,
            edge_to_row,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch_expert_major(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event_overlap.current_stream_wait()

        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.recv_shape = (recv_token_indices.shape[0], x.shape[1])
        ctx.recv_probs_shape = recv_token_indices.shape
        ctx.save_for_backward(row_map, edge_map)
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)
        return (
            expert_x,
            recv_token_indices,
            permuted_token_probs,
            row_map,
            edge_to_row,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(
        ctx,
        grad_expert_x,
        grad_recv_token_indices,
        grad_permuted_token_probs,
        grad_row_map,
        grad_edge_to_row,
        grad_tokens_per_expert,
        grad_handle,
    ):
        row_map, edge_map = ctx.saved_tensors
        from megatron.core.extensions.hisa_indexer.kernels.build import get_ext

        grad_recv_x = torch.zeros(
            ctx.recv_shape,
            device=grad_expert_x.device,
            dtype=grad_expert_x.dtype,
        )
        get_ext().moe_deepep_compact_scatter_add(grad_expert_x.contiguous(), row_map, grad_recv_x)

        grad_recv_probs = None
        if grad_permuted_token_probs is not None:
            grad_recv_probs = torch.zeros(
                ctx.recv_probs_shape,
                device=grad_permuted_token_probs.device,
                dtype=grad_permuted_token_probs.dtype,
            )
            get_ext().moe_deepep_compact_scatter_probs(
                grad_permuted_token_probs.contiguous(), edge_map, grad_recv_probs
            )

        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_recv_x))
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_recv_x.contiguous(),
            ctx.handle,
            topk_weights=None if grad_recv_probs is None else grad_recv_probs.float(),
            config=_get_combine_config_override(ctx.group.size()),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            config=_get_combine_config_override(group.size()),
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


class FusedExpertMajorCombine(torch.autograd.Function):
    """Strict expert-major DeepEP combine.

    The forward consumes expert-major GroupedMLP output directly. DeepEP gathers
    and sums local expert rows in the sender phase, so Megatron does not
    materialize the DeepEP-order unpermuted tensor before combine.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        handle,
        expert_indices,
        edge_to_row,
        row_map,
        num_local_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        if not hasattr(buffer, "combine_expert_major"):
            raise RuntimeError(
                "Requested strict expert-major DeepEP combine, but installed DeepEP "
                "does not expose Buffer.combine_expert_major"
            )
        combined_x, _, after_event = buffer.combine_expert_major(
            x.contiguous(),
            handle=handle,
            expert_indices=expert_indices.contiguous(),
            edge_to_row=edge_to_row.contiguous(),
            num_local_experts=int(num_local_experts),
            config=_get_combine_config_override(group.size()),
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.save_for_backward(row_map)
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_recv_order, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()

        (row_map,) = ctx.saved_tensors
        grad_x = torch.empty(
            (row_map.numel(), grad_recv_order.shape[1]),
            device=grad_recv_order.device,
            dtype=grad_recv_order.dtype,
        )
        from megatron.core.extensions.hisa_indexer.kernels.build import get_ext

        get_ext().moe_deepep_compact_gather(grad_recv_order.contiguous(), row_map, grad_x)
        return grad_x, None, None, None, None, None, None, None, None


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def fused_dispatch_expert_major(
        x,
        token_indices,
        token_probs,
        num_experts,
        num_local_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform strict fused dispatch directly into local expert-major order."""
        return FusedDispatchExpertMajor.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            num_local_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream)

    def fused_combine_expert_major(
        x,
        group,
        handle,
        expert_indices,
        edge_to_row,
        row_map,
        num_local_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform strict expert-major DeepEP combine."""
        return FusedExpertMajorCombine.apply(
            x,
            group,
            handle,
            expert_indices,
            edge_to_row,
            row_map,
            num_local_experts,
            async_finish,
            allocate_on_comm_stream,
        )

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP"""
        Buffer.set_num_sms(num_sms)

else:
    fused_dispatch = None
    fused_dispatch_expert_major = None
    fused_combine = None
    fused_combine_expert_major = None
    set_deepep_num_sms = None


try:
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

_hybrid_ep_buffer = None


def init_hybrid_ep_buffer(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    seq_len: int,
    num_local_experts: int,
    num_sms_dispatch_api: int,
    num_sms_combine_api: int,
    fp8_dispatch: bool,
    num_sms_preprocessing_api: Optional[int] = None,
) -> None:
    '''
    Initialize the HybridEP buffer, including buffer allocation and metadata
    initialization.

    If a runtime dispatch/combine requires a larger buffer than the one
    initialized, the buffer will be reallocated at runtime,
    incuring extra run-time overhead.

    Args:
        group (torch.distributed.ProcessGroup):
            Process group for HybridEP all-to-all communication.
        hidden_dim (int):
            Hidden dimension of the input tensor.
        seq_len (int):
            Maximum sequence length of the input tensor.
        num_local_experts (int):
            Number of local experts.
        num_sms_dispatch_api (int):
            Number of SMs used by the dispatch API.
        num_sms_combine_api (int):
            Number of SMs used by the combine API.
        fp8_dispatch (bool):
            Whether to use FP8 communication during the dispatch phase.
        num_sms_preprocessing_api (Optional[int]):
            Number of SMs used by the preprocessing metadata scan kernel.
    '''
    assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch now"
    kwargs = {}
    if num_sms_preprocessing_api is not None:
        kwargs["num_sms_preprocessing_api"] = num_sms_preprocessing_api
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=seq_len,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        num_sms_dispatch_api=num_sms_dispatch_api,
        num_sms_combine_api=num_sms_combine_api,
        **kwargs,
    )


def reset_hybrid_ep_buffer():
    '''
    Reset the HybridEP buffer
    '''
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = None


class HybridEPDispatch(torch.autograd.Function):
    '''
    Fused dispatch operation for permute + dispatch a2a + permute using the HybridEP backend
    '''

    @staticmethod
    def forward(
        ctx,
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
        num_sms_preprocessing_api=108,
    ):
        '''
        Forward pass of fused dispatch of the HybridEP backend
        '''
        if _hybrid_ep_buffer is None:
            seq_len, hidden_dim = x.shape[-2:]
            fp8_dispatch = False  # Currently, we do not support fp8 dispatch
            init_hybrid_ep_buffer(
                group,
                hidden_dim,
                seq_len,
                num_local_experts,
                num_sms_dispatch_api,
                num_sms_combine_api,
                fp8_dispatch,
                num_sms_preprocessing_api,
            )
        # If we provide the num_permuted_tokens, we do not need to use sync to
        # wait for the data in pinned memory ready
        non_blocking = num_permuted_tokens is not None
        # Process the dispatch
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        ) = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=x,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
        )

        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle):
        '''
        Backward pass of fused dispatch of the HybridEP backend
        '''
        handle = ctx.handle
        combined_hidden, combined_probs = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=grad_x, probs=grad_probs, handle=handle, pad_multiple=ctx.pad_multiple
        )
        return combined_hidden, None, combined_probs, None, None, None, None, None, None, None, None


@internal_api
class HybridEPCombine(torch.autograd.Function):
    '''
    Fused combine operation for permute + combine a2a + permute using the HybridEP backend
    '''

    @staticmethod
    def forward(ctx, x, handle, num_permuted_tokens=None, pad_multiple=None):
        '''
        Forward pass of fused combine of the HybridEP backend
        '''
        combined_hidden, _ = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=x, handle=handle, pad_multiple=pad_multiple
        )
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        return combined_hidden

    @staticmethod
    def backward(ctx, grad_x):
        '''
        Backward pass of fused combine of the HybridEP backend
        '''
        handle = ctx.handle
        dispatched_hidden, _, _, _, _ = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=grad_x,
            scaling_factor=None,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            num_permuted_tokens=ctx.num_permuted_tokens,
        )
        return dispatched_hidden, None, None, None, None


if HAVE_HYBRIDEP:

    @internal_api
    def hybrid_ep_dispatch(
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
        num_sms_preprocessing_api=108,
    ):
        '''
        Perform fused dispatch for "permute + dispatch a2a + permute" using the
        HybridEP backend.

        Args:
            x (torch.Tensor):
                Input hidden states to dispatch.
            routing_map (torch.Tensor):
                Map indicating which expert each token is routed to.
            probs (torch.Tensor):
                Routing probabilities for each token-expert pair.
            group (torch.distributed.ProcessGroup):
                Process group used for communication.
            num_local_experts (int):
                Number of local experts.
            num_sms_dispatch_api (int):
                Number of SMs used by the dispatch API.
            num_sms_combine_api (int):
                Number of SMs used by the combine API.
            num_permuted_tokens (int):
                Number of tokens after permute. HybridEP uses this to allocate buffers.
                If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                Alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
            num_sms_preprocessing_api (int):
                Number of SMs used by the preprocessing metadata scan kernel.
        '''
        return HybridEPDispatch.apply(
            x,
            routing_map,
            probs,
            group,
            num_local_experts,
            num_sms_dispatch_api,
            num_sms_combine_api,
            num_permuted_tokens,
            pad_multiple,
            num_sms_preprocessing_api,
        )

    @internal_api
    def hybrid_ep_combine(x, handle, num_permuted_tokens, pad_multiple):
        '''
        Perform fused combine operation for unpermute + combine a2a + unpermute
        using the HybridEP backend

        args:
            x (torch.Tensor):
                Input hidden states to combine
            handle (EventHandle):
                Communication handle from dispatch operation
            num_permuted_tokens (int): The number of tokens before unpermute. HybridEP uses this
                to allocate buffers. If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                The alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
        '''
        return HybridEPCombine.apply(x, handle, num_permuted_tokens, pad_multiple)

else:
    hybrid_ep_dispatch = None
    hybrid_ep_combine = None
