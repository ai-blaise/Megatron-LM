# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native Megatron-Core helpers for StreamBP sequence chunking.

StreamBP keeps the public transformer forward contract unchanged while the
backward pass recomputes each transformer layer over query chunks. For a causal
decoder layer, chunk ``[i:j]`` attends over the key/value prefix ``[:j]`` and
then runs the MLP only for ``[i:j]``. Gradients from all chunks are accumulated
before the data-parallel bucket is marked ready.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import torch
from torch import Tensor

from megatron.core.tensor_audit import tensor_audit
from megatron.core.packed_seq_params import PackedSeqParams

STREAMBP_PENDING_CHUNKS_ATTR = "_streambp_pending_chunks"
_STREAMBP_FUSED_LCE_LOGGED = False
_STREAMBP_FUSED_LCE_REJECTION_LOGGED = False

ChunkRange = tuple[int, int]
ContextFactory = Optional[Callable[[], Any]]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class _StreamBPHostSavedTensor:
    """Host-resident tensor saved by StreamBP for backward replay."""

    device: torch.device
    cpu_tensor: Tensor
    copy_event: Optional[torch.cuda.Event]

    @staticmethod
    def from_tensor(tensor: Tensor) -> "_StreamBPHostSavedTensor":
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            fine_grained_offloading_copy_to_cpu,
        )

        if not tensor.is_cuda:
            return _StreamBPHostSavedTensor(tensor.device, tensor.detach().clone(), None)

        pin_memory = _streambp_host_save_pin_memory(tensor)
        cpu_tensor = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        fine_grained_offloading_copy_to_cpu(
            cpu_tensor,
            tensor.detach(),
            non_blocking=pin_memory,
        )
        event = torch.cuda.Event() if pin_memory else None
        if event is not None:
            torch.cuda.current_stream(tensor.device).record_event(event)
        return _StreamBPHostSavedTensor(tensor.device, cpu_tensor, event)

    def load(self) -> Tensor:
        if self.copy_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self.copy_event)
        if self.device.type != "cuda":
            return self.cpu_tensor
        tensor = torch.empty(
            self.cpu_tensor.shape,
            dtype=self.cpu_tensor.dtype,
            device=self.device,
        )
        tensor.copy_(self.cpu_tensor, non_blocking=self.cpu_tensor.is_pinned())
        return tensor


def _streambp_host_save_hidden_enabled() -> bool:
    return _env_flag("MEGATRON_STREAMBP_HOST_SAVE_HIDDEN", default=True)


def _streambp_host_save_pin_memory(tensor: Tensor) -> bool:
    mode = os.getenv("MEGATRON_STREAMBP_HOST_SAVE_PIN_MEMORY", "auto").lower()
    if mode in ("0", "false", "off", "no", "never"):
        return False
    if mode in ("1", "true", "on", "yes", "always"):
        return True
    if mode != "auto":
        raise ValueError(
            "MEGATRON_STREAMBP_HOST_SAVE_PIN_MEMORY must be one of auto/always/never"
        )
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        fine_grained_offloading_use_pinned_cpu_backup,
    )

    return fine_grained_offloading_use_pinned_cpu_backup(tensor.shape, tensor.dtype)


def _streambp_save_hidden_for_backward(ctx, hidden_states: Tensor) -> None:
    if _streambp_host_save_hidden_enabled():
        ctx._streambp_host_saved_hidden = _StreamBPHostSavedTensor.from_tensor(hidden_states)
        ctx._streambp_saved_hidden_shape0 = int(hidden_states.size(0))
        ctx._streambp_used_host_save_hidden = True
        return

    ctx._streambp_host_saved_hidden = None
    ctx._streambp_used_host_save_hidden = False
    ctx.save_for_backward(hidden_states)


def _streambp_load_hidden_for_backward(ctx) -> Tensor:
    host_saved = getattr(ctx, "_streambp_host_saved_hidden", None)
    if host_saved is not None:
        hidden_states = host_saved.load()
        ctx._streambp_host_saved_hidden = None
        return hidden_states

    (hidden_states,) = ctx.saved_tensors
    return hidden_states


def _maybe_trim_cuda_cache_before_moe_replay(
    *, force: bool = False, synchronize: bool = False
) -> None:
    """Release cached allocator blocks when StreamBP MoE replay is near OOM."""
    if not torch.cuda.is_available():
        return
    if not _env_flag("MEGATRON_STREAMBP_MOE_REPLAY_TRIM_CACHE", default=True):
        return

    free_bytes, _ = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    cached = max(0, reserved - allocated)

    mib = 1024 * 1024
    free_threshold_mb = int(os.getenv("MEGATRON_STREAMBP_MOE_REPLAY_TRIM_FREE_MB", "2048"))
    cached_threshold_mb = int(os.getenv("MEGATRON_STREAMBP_MOE_REPLAY_TRIM_CACHED_MB", "512"))
    if force or (free_bytes < free_threshold_mb * mib and cached > cached_threshold_mb * mib):
        if synchronize and _env_flag("MEGATRON_STREAMBP_MOE_REPLAY_TRIM_SYNC", default=True):
            torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _streambp_autograd_backward(outputs, grad_outputs) -> None:
    """Run replay backward with StreamBP-owned TE reentrant retention."""

    from megatron.core.tensor_parallel.random import te_reentrant_checkpoint_retention_scope

    with te_reentrant_checkpoint_retention_scope():
        torch.autograd.backward(outputs, grad_outputs)


def _moe_attention_backward_chunks(
    seq_len: int, chunks: Sequence[ChunkRange]
) -> Sequence[ChunkRange]:
    """Return backward-only attention replay chunks for hybrid MoE StreamBP."""

    raw = os.getenv("MEGATRON_STREAMBP_MOE_ATTENTION_BACKWARD_CHUNK_SIZE")
    if raw is None or raw == "":
        return chunks
    try:
        chunk_size = int(raw)
    except ValueError:
        chunk_size = 0
    if chunk_size <= 0:
        return chunks
    return iter_streambp_chunks(seq_len, chunk_size)


def _streambp_autograd_backward_attention_outputs(
    attention_outputs: list[tuple[int, int, Tensor]],
    grad_start: int,
    grad_output: Tensor,
) -> None:
    """Backprop through original attention replay outputs one graph at a time.

    Hybrid MoE replay detaches the MLP input from the attention graph. Once the
    MLP backward has produced d(post_attention), consuming the copied
    post-attention buffer would keep every attention chunk's DSA/TE reentrant
    state under one autograd engine invocation. Backward through the original
    attention outputs preserves the same chain rule while releasing each chunk's
    retained TE checkpoint state before the next chunk starts.
    """

    for output_index, (attention_start, attention_end, attention_output) in enumerate(
        attention_outputs
    ):
        rel_start = attention_start - grad_start
        rel_end = attention_end - grad_start
        if rel_start < 0 or rel_end > grad_output.size(0):
            raise RuntimeError(
                "StreamBP attention gradient range mismatch: "
                f"attention=[{attention_start}, {attention_end}), "
                f"grad_start={grad_start}, grad_len={grad_output.size(0)}"
            )
        _maybe_trim_cuda_cache_before_moe_replay(force=True, synchronize=True)
        _streambp_autograd_backward(attention_output, grad_output[rel_start:rel_end])
        attention_outputs[output_index] = (attention_start, attention_end, None)  # type: ignore[list-item]
        del attention_output
    attention_outputs.clear()


def _split_moe_mlp_attention_backward_enabled() -> bool:
    """Run MoE replay as MLP backward followed by attention backward.

    The default one-shot autograd path keeps the replayed attention graph and
    replayed MoE MLP graph live through the same engine invocation. Splitting
    the chain-rule application preserves gradients while allowing the MLP
    internals to be released before the attention graph is consumed.
    """

    return _env_flag("MEGATRON_STREAMBP_SPLIT_MOE_MLP_ATTENTION_BACKWARD", default=True)


def _suppress_fine_grained_offload_forced_release_context():
    """Keep replay tensors' storage alive while preserving CPU offload hooks."""
    if _env_flag("MEGATRON_FINE_OFFLOAD_FORCE_RELEASE_IN_STREAMBP_REPLAY"):
        return nullcontext()
    names_spec = os.getenv("MEGATRON_FINE_OFFLOAD_STREAMBP_REPLAY_KEEP_STORAGES", "core_attn")
    names = [name for name in names_spec.replace(",", " ").split() if name]
    if not names or names == ["none"]:
        return nullcontext()
    try:
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            fine_grained_offloading_suppress_forced_release,
        )
    except Exception:
        return nullcontext()
    return fine_grained_offloading_suppress_forced_release(names=names)


@contextmanager
def _suppress_fine_grained_offload_replay_context():
    """Bypass fine-grained offload markers inside StreamBP replay graphs."""
    try:
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            fine_grained_offloading_suppress_offload,
        )
    except Exception:
        with _suppress_fine_grained_offload_forced_release_context():
            yield
        return
    with (
        fine_grained_offloading_suppress_offload(),
        _suppress_fine_grained_offload_forced_release_context(),
    ):
        yield


@dataclass
class _ReplayOffloadedTensor:
    device: torch.device
    cpu_tensor: Tensor
    event: Optional[torch.cuda.Event]
    non_blocking: bool
    stride: tuple[int, ...]


@contextmanager
def _streambp_replay_saved_tensor_offload_context(label: str = "replay"):
    """Offload tensors saved by a single StreamBP replay graph.

    The regular fine-grained pipeline offload manager is intentionally
    suppressed inside StreamBP replay because the inner autograd graph does not
    follow the outer pipeline queue ordering. This context provides a local
    saved-tensor hook for one replay chunk: tensors saved during replay forward
    are copied to CPU and reloaded when that same replay graph runs backward.
    """

    if not _env_flag("MEGATRON_STREAMBP_REPLAY_SAVED_TENSOR_OFFLOAD", default=False):
        yield
        return
    if not torch.cuda.is_available():
        yield
        return

    min_mb = _env_int("MEGATRON_STREAMBP_REPLAY_OFFLOAD_MIN_MB", 16)
    free_threshold_mb = _env_int("MEGATRON_STREAMBP_REPLAY_OFFLOAD_FREE_MB", 0)
    max_tensors = _env_int("MEGATRON_STREAMBP_REPLAY_OFFLOAD_MAX_TENSORS", 0)
    min_bytes = max(0, min_mb) * 1024 * 1024
    free_threshold_bytes = max(0, free_threshold_mb) * 1024 * 1024
    packed_count = 0

    try:
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            fine_grained_offloading_copy_to_cpu,
            fine_grained_offloading_use_pinned_cpu_backup,
        )
    except Exception:
        yield
        return

    def pack(tensor: Tensor):
        nonlocal packed_count
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            return tensor
        # TE/NVFP4/FP8 tensor subclasses carry scaling metadata that is not
        # preserved by a plain CPU tensor round-trip. Only offload ordinary
        # CUDA tensors here; replay still keeps quantized wrappers intact.
        if type(tensor) is not torch.Tensor:
            return tensor
        if tensor.numel() == 0 or tensor.numel() * tensor.element_size() < min_bytes:
            return tensor
        if max_tensors > 0 and packed_count >= max_tensors:
            return tensor
        if free_threshold_bytes > 0:
            try:
                free_bytes, _ = torch.cuda.mem_get_info(tensor.device)
            except RuntimeError:
                free_bytes = 0
            if free_bytes > free_threshold_bytes:
                return tensor

        pin_memory = fine_grained_offloading_use_pinned_cpu_backup(tensor.shape, tensor.dtype)
        cpu_tensor = torch.empty_strided(
            tuple(tensor.shape),
            tuple(tensor.stride()),
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        non_blocking = bool(pin_memory)
        fine_grained_offloading_copy_to_cpu(cpu_tensor, tensor, non_blocking=non_blocking)
        event = None
        if non_blocking:
            event = torch.cuda.Event()
            torch.cuda.current_stream(tensor.device).record_event(event)
        packed_count += 1
        return _ReplayOffloadedTensor(
            device=tensor.device,
            cpu_tensor=cpu_tensor,
            event=event,
            non_blocking=non_blocking,
            stride=tuple(tensor.stride()),
        )

    def unpack(state):
        if not isinstance(state, _ReplayOffloadedTensor):
            return state
        if state.event is not None:
            state.event.synchronize()
        tensor = torch.empty_strided(
            tuple(state.cpu_tensor.shape),
            state.stride,
            dtype=state.cpu_tensor.dtype,
            device=state.device,
        )
        tensor.copy_(state.cpu_tensor, non_blocking=state.non_blocking)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


@contextmanager
def _streambp_replay_save_quantized_te_inputs(layer: torch.nn.Module):
    """Prefer TE's saved quantized input path inside StreamBP replay.

    Some TE modules are configured to save the original BF16 input when
    fine-grained offload is enabled. StreamBP suppresses that global offload
    queue during replay, so saving the original input only pushes the TE
    split-quantize allocation into backward where replay memory is tight.
    During replay we can use TE's default saved quantized columnwise input
    path instead and restore the module attributes afterwards.
    """

    if not _env_flag("MEGATRON_STREAMBP_REPLAY_SAVE_QUANTIZED_TE_INPUTS", default=True):
        yield
        return

    saved = []
    for module in layer.modules():
        if getattr(module, "save_original_input", False):
            saved.append((module, True))
            module.save_original_input = False
    try:
        yield
    finally:
        for module, value in saved:
            module.save_original_input = value


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.getenv("RANK", "0"))


def _rank_selected(spec: str, rank: int) -> bool:
    spec = spec.strip()
    if not spec or spec == "all":
        return True
    return rank in {int(item) for item in spec.split(",") if item.strip()}


def _debug_sync(label: str) -> None:
    """Opt-in CUDA sync fence for locating asynchronous StreamBP failures."""
    if os.getenv("MEGATRON_STREAMBP_DEBUG_SYNC", "0").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    if not torch.cuda.is_available():
        return
    rank = _distributed_rank()
    if not _rank_selected(os.getenv("MEGATRON_STREAMBP_DEBUG_RANKS", "all"), rank):
        return
    if os.getenv("MEGATRON_STREAMBP_DEBUG_VERBOSE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        print(f"[rank{rank}] streambp sync: {label}", flush=True)
    torch.cuda.synchronize()


class _StreamBPChunkProfiler:
    """Tiny on-demand profiler for StreamBP chunks.

    Megatron's built-in profiler exports at iteration boundaries. StreamBP
    regressions can make the first iteration too slow to finish, so this helper
    can export a few per-chunk traces from inside the forward/backward replay.
    """

    def __init__(self) -> None:
        self.enabled = _env_flag("MEGATRON_STREAMBP_PROFILE")
        self.rank = int(os.getenv("MEGATRON_STREAMBP_PROFILE_RANK", "0"))
        self.limit = int(os.getenv("MEGATRON_STREAMBP_PROFILE_LIMIT", "4"))
        self.record_shapes = _env_flag("MEGATRON_STREAMBP_PROFILE_RECORD_SHAPES")
        self.with_stack = _env_flag("MEGATRON_STREAMBP_PROFILE_WITH_STACK")
        self.filter = os.getenv("MEGATRON_STREAMBP_PROFILE_FILTER")
        self.capture_count = 0
        self.output_dir = Path(
            os.getenv("MEGATRON_STREAMBP_PROFILE_DIR", "/tmp/streambp_profile")
        ).expanduser()

    def configure(
        self,
        *,
        enabled: bool,
        output_dir: Optional[str],
        rank: int,
        limit: int,
        record_shapes: bool,
        with_stack: bool,
        name_filter: Optional[str],
    ) -> None:
        self.enabled = enabled
        if output_dir:
            self.output_dir = Path(output_dir).expanduser()
        self.rank = rank
        self.limit = limit
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.filter = name_filter

    def should_capture(self, name: str) -> bool:
        if self.filter and self.filter not in name:
            return False
        return (
            self.enabled
            and self.capture_count < self.limit
            and _distributed_rank() == self.rank
        )

    @contextmanager
    def capture(self, name: str) -> Iterator[None]:
        if not self.should_capture(name):
            with torch.profiler.record_function(name):
                yield
            return

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            torch.cuda.synchronize()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        capture_index = self.capture_count
        self.capture_count += 1
        trace_name = name.replace("/", "_").replace(" ", "_")
        trace_path = self.output_dir / (
            f"{trace_name}.rank{_distributed_rank()}.capture{capture_index}.json.gz"
        )

        with torch.profiler.profile(
            activities=activities,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
            with_flops=True,
        ) as prof:
            with torch.profiler.record_function(name):
                yield

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace_path))


_STREAMBP_CHUNK_PROFILER = _StreamBPChunkProfiler()


def configure_streambp_profiler(
    *,
    enabled: bool,
    output_dir: Optional[str],
    rank: int,
    limit: int,
    record_shapes: bool,
    with_stack: bool,
    name_filter: Optional[str] = None,
) -> None:
    _STREAMBP_CHUNK_PROFILER.configure(
        enabled=enabled,
        output_dir=output_dir,
        rank=rank,
        limit=limit,
        record_shapes=record_shapes,
        with_stack=with_stack,
        name_filter=name_filter,
    )


@contextmanager
def _profile_streambp_chunk(name: str) -> Iterator[None]:
    with _STREAMBP_CHUNK_PROFILER.capture(name):
        yield


@dataclass
class StreamBPMoeAuxStats:
    tokens_per_expert: Tensor
    local_num_tokens: Any
    total_num_tokens: Any
    seq_length: int
    bsz: int
    with_padding_mask: bool


class StreamBPMoeAuxState:
    """Full-sequence MoE aux statistics captured for chunked replay."""

    def __init__(self) -> None:
        self._stats: dict[int, dict[str, StreamBPMoeAuxStats]] = {}
        self._chunk_stats: dict[int, dict[str, list[StreamBPMoeAuxStats]]] = {}
        self._quantile_bias: dict[int, list[Tensor]] = {}
        self._replay_chunk_index: Optional[int] = None

    def record(self, router: Any, aux_loss_type: str, stats: StreamBPMoeAuxStats) -> None:
        chunk_router_stats = self._chunk_stats.setdefault(id(router), {})
        chunk_router_stats.setdefault(aux_loss_type, []).append(stats)

        router_stats = self._stats.setdefault(id(router), {})
        existing = router_stats.get(aux_loss_type)
        if existing is None:
            router_stats[aux_loss_type] = stats
            return
        if existing.bsz != stats.bsz or existing.with_padding_mask != stats.with_padding_mask:
            raise ValueError("Inconsistent StreamBP MoE aux stats across forward chunks")
        router_stats[aux_loss_type] = StreamBPMoeAuxStats(
            tokens_per_expert=existing.tokens_per_expert + stats.tokens_per_expert,
            local_num_tokens=existing.local_num_tokens + stats.local_num_tokens,
            total_num_tokens=existing.total_num_tokens + stats.total_num_tokens,
            seq_length=existing.seq_length + stats.seq_length,
            bsz=existing.bsz,
            with_padding_mask=existing.with_padding_mask,
        )

    def get(self, router: Any, aux_loss_type: str) -> Optional[StreamBPMoeAuxStats]:
        router_id = id(router)
        full_stats = self._stats.get(router_id, {}).get(aux_loss_type)
        if full_stats is None:
            return None
        if self._replay_chunk_index is None:
            return full_stats

        chunk_stats = self._chunk_stats.get(router_id, {}).get(aux_loss_type)
        if chunk_stats is None:
            return None
        if self._replay_chunk_index >= len(chunk_stats):
            raise ValueError(
                f"Missing StreamBP MoE aux stats chunk {self._replay_chunk_index} "
                f"for {aux_loss_type}; captured {len(chunk_stats)} chunks"
            )
        chunk = chunk_stats[self._replay_chunk_index]
        return StreamBPMoeAuxStats(
            tokens_per_expert=full_stats.tokens_per_expert,
            local_num_tokens=chunk.local_num_tokens,
            total_num_tokens=full_stats.total_num_tokens,
            seq_length=chunk.seq_length,
            bsz=full_stats.bsz,
            with_padding_mask=full_stats.with_padding_mask,
        )

    def record_quantile_bias(self, router: Any, bias: Tensor) -> None:
        """Record the routing bias used by a StreamBP MoE no-grad forward chunk."""
        self._quantile_bias.setdefault(id(router), []).append(bias.detach())

    def get_quantile_bias(self, router: Any) -> Optional[Tensor]:
        """Return the recorded routing bias for the current replay chunk."""
        biases = self._quantile_bias.get(id(router))
        if not biases:
            return None
        if self._replay_chunk_index is None:
            if len(biases) != 1:
                raise ValueError(
                    "StreamBP MoE quantile replay requested full bias but captured "
                    f"{len(biases)} chunk biases"
                )
            return biases[0]
        if self._replay_chunk_index >= len(biases):
            raise ValueError(
                f"Missing StreamBP MoE quantile bias chunk {self._replay_chunk_index}; "
                f"captured {len(biases)} chunks"
            )
        return biases[self._replay_chunk_index]

    def has_router(self, router: Any) -> bool:
        router_id = id(router)
        return (
            router_id in self._stats
            or router_id in self._chunk_stats
            or router_id in self._quantile_bias
        )


_MOE_AUX_CAPTURE_STACK: list[StreamBPMoeAuxState] = []
_MOE_AUX_REPLAY_STACK: list[StreamBPMoeAuxState] = []


@contextmanager
def capture_streambp_moe_aux_stats() -> Iterator[StreamBPMoeAuxState]:
    state = StreamBPMoeAuxState()
    _MOE_AUX_CAPTURE_STACK.append(state)
    try:
        yield state
    finally:
        popped = _MOE_AUX_CAPTURE_STACK.pop()
        assert popped is state


@contextmanager
def replay_streambp_moe_aux_stats(
    state: Optional[StreamBPMoeAuxState],
    *,
    chunk_index: Optional[int] = None,
) -> Iterator[None]:
    if state is None:
        with nullcontext():
            yield
        return
    previous_chunk_index = state._replay_chunk_index
    state._replay_chunk_index = chunk_index
    _MOE_AUX_REPLAY_STACK.append(state)
    try:
        yield
    finally:
        popped = _MOE_AUX_REPLAY_STACK.pop()
        assert popped is state
        state._replay_chunk_index = previous_chunk_index


def current_streambp_moe_aux_capture() -> Optional[StreamBPMoeAuxState]:
    return _MOE_AUX_CAPTURE_STACK[-1] if _MOE_AUX_CAPTURE_STACK else None


def current_streambp_moe_aux_replay() -> Optional[StreamBPMoeAuxState]:
    return _MOE_AUX_REPLAY_STACK[-1] if _MOE_AUX_REPLAY_STACK else None


def resolve_streambp_chunk_size(seq_len: int, chunk_size: Optional[int]) -> int:
    """Resolve user chunk size or the conservative default heuristic."""
    if seq_len <= 0:
        raise ValueError(f"StreamBP requires positive sequence length, got {seq_len}")
    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError(f"StreamBP chunk size must be positive, got {chunk_size}")
        return min(seq_len, chunk_size)
    return min(seq_len, min(8192, max(2048, seq_len // 4)))


def iter_streambp_chunks(seq_len: int, chunk_size: Optional[int]) -> list[ChunkRange]:
    """Return sequence-major chunk ranges covering ``[0, seq_len)``."""
    size = resolve_streambp_chunk_size(seq_len, chunk_size)
    return [(start, min(start + size, seq_len)) for start in range(0, seq_len, size)]


def iter_streambp_num_chunks(seq_len: int, num_chunks: int) -> list[ChunkRange]:
    """Return ``num_chunks`` large contiguous sequence ranges."""
    if seq_len <= 0:
        raise ValueError(f"StreamBP requires positive sequence length, got {seq_len}")
    if num_chunks <= 0:
        raise ValueError(f"StreamBP chunk count must be positive, got {num_chunks}")
    num_chunks = min(num_chunks, seq_len)
    chunk_size = (seq_len + num_chunks - 1) // num_chunks
    return [(start, min(start + chunk_size, seq_len)) for start in range(0, seq_len, chunk_size)]


def _slice_padding_mask_for_sequence_chunk(
    padding_mask: Optional[Tensor], start: int, end: int, seq_len: int
) -> Optional[Tensor]:
    if padding_mask is None:
        return None
    if padding_mask.dim() >= 2 and padding_mask.size(1) == seq_len:
        return padding_mask[:, start:end]
    return padding_mask


def _intersect_streambp_chunk_ranges(
    chunks: Sequence[ChunkRange], start: int, end: int
) -> list[ChunkRange]:
    """Return chunk intersections covering ``[start, end)`` without gaps."""
    if not (0 <= start < end):
        raise ValueError(f"Invalid StreamBP chunk intersection range [{start}, {end})")

    intersections: list[ChunkRange] = []
    for chunk_start, chunk_end in chunks:
        sub_start = max(start, chunk_start)
        sub_end = min(end, chunk_end)
        if sub_start < sub_end:
            intersections.append((sub_start, sub_end))

    cursor = start
    for sub_start, sub_end in intersections:
        if sub_start != cursor:
            raise ValueError(
                f"StreamBP chunks do not cover range [{start}, {end}); "
                f"missing [{cursor}, {sub_start})"
            )
        cursor = sub_end
    if cursor != end:
        raise ValueError(
            f"StreamBP chunks do not cover range [{start}, {end}); "
            f"missing [{cursor}, {end})"
        )
    return intersections


def _concat_streambp_chunk_outputs(outputs: list[Any]) -> Any:
    first = outputs[0]
    if first is None:
        return None
    if torch.is_tensor(first):
        result = torch.cat(outputs, dim=0)
        outputs.clear()
        return result
    if isinstance(first, tuple):
        return tuple(
            _concat_streambp_chunk_outputs([output[i] for output in outputs])
            for i in range(len(first))
        )
    if isinstance(first, list):
        return [
            _concat_streambp_chunk_outputs([output[i] for output in outputs])
            for i in range(len(first))
        ]
    raise TypeError(f"Unsupported StreamBP chunk output type: {type(first)}")


def _copy_streambp_tensor_chunk(
    destination: Optional[Tensor],
    chunk: Tensor,
    start: int,
    end: int,
    seq_len: int,
) -> Tensor:
    """Copy a sequence chunk into a full-sequence output buffer."""
    if chunk.size(0) != end - start:
        raise RuntimeError(
            "StreamBP chunk output length mismatch: "
            f"got {chunk.size(0)} for range [{start}, {end})"
        )
    if destination is None:
        shape = list(chunk.shape)
        shape[0] = seq_len
        destination = torch.empty(
            tuple(shape),
            dtype=chunk.dtype,
            device=chunk.device,
            layout=chunk.layout,
        )
    destination.narrow(0, start, end - start).copy_(chunk)
    return destination


def _copy_sequence_tensor_ranges(
    chunks: list[tuple[int, int, Tensor]],
    start: int,
    end: int,
    *,
    clear_chunks: bool = True,
) -> Tensor:
    """Concatenate sequence chunks without keeping every chunk plus a cat output live."""
    if not chunks:
        raise RuntimeError("StreamBP expected at least one sequence tensor chunk")
    if len(chunks) == 1:
        chunk_start, chunk_end, tensor = chunks[0]
        if chunk_start != start or chunk_end != end:
            raise RuntimeError(
                "StreamBP single tensor chunk does not cover requested range "
                f"[{start}, {end}); got [{chunk_start}, {chunk_end})"
            )
        if clear_chunks:
            chunks.clear()
        return tensor

    first_start, first_end, first = chunks[0]
    if first_start != start:
        raise RuntimeError(
            f"StreamBP tensor chunks start at {first_start}, expected {start}"
        )
    shape = list(first.shape)
    shape[0] = end - start
    result = torch.empty(tuple(shape), dtype=first.dtype, device=first.device, layout=first.layout)
    cursor = start
    for chunk_start, chunk_end, chunk in chunks:
        if chunk_start != cursor:
            raise RuntimeError(
                "StreamBP tensor chunks are not contiguous: "
                f"expected {cursor}, got {chunk_start}"
            )
        if chunk.size(0) != chunk_end - chunk_start:
            raise RuntimeError(
                "StreamBP tensor chunk length mismatch: "
                f"got {chunk.size(0)} for range [{chunk_start}, {chunk_end})"
            )
        result.narrow(0, chunk_start - start, chunk_end - chunk_start).copy_(chunk)
        cursor = chunk_end
    if cursor != end:
        raise RuntimeError(f"StreamBP tensor chunks ended at {cursor}, expected {end}")
    if clear_chunks:
        chunks.clear()
    return result


def validate_chunk_range(chunk_range: ChunkRange, seq_len: int) -> ChunkRange:
    """Validate and normalize a StreamBP chunk range."""
    start, end = chunk_range
    if not (0 <= start < end <= seq_len):
        raise ValueError(
            f"Invalid StreamBP chunk range {chunk_range} for sequence length {seq_len}"
        )
    return start, end


def _select_cu_seqlens(packed_seq_params: PackedSeqParams, *, padded: bool) -> Tensor:
    if padded:
        cu_seqlens = packed_seq_params.cu_seqlens_q_padded
        if cu_seqlens is None:
            cu_seqlens = packed_seq_params.cu_seqlens_q
    else:
        cu_seqlens = packed_seq_params.cu_seqlens_q
        if cu_seqlens is None:
            cu_seqlens = packed_seq_params.cu_seqlens_q_padded
    if cu_seqlens is None:
        raise ValueError("StreamBP packed sequence chunking requires cu_seqlens")
    if cu_seqlens.dim() == 2:
        if cu_seqlens.size(0) != 1:
            raise ValueError(
                f"StreamBP packed sequence chunking expects micro-batch-size 1, got "
                f"cu_seqlens shape {tuple(cu_seqlens.shape)}"
            )
        cu_seqlens = cu_seqlens[0]
    if cu_seqlens.dim() != 1:
        raise ValueError(f"cu_seqlens must be 1D, got shape {tuple(cu_seqlens.shape)}")
    return cu_seqlens


def _build_cu_from_lengths(lengths: Tensor) -> Tensor:
    cu = torch.empty(lengths.numel() + 1, dtype=torch.int32, device=lengths.device)
    cu[0] = 0
    if lengths.numel() > 0:
        cu[1:] = torch.cumsum(lengths.to(torch.int32), dim=0)
    return cu


def _build_cu_from_contiguous_segments(starts: Tensor, ends: Tensor) -> Tensor:
    if starts.numel() != ends.numel():
        raise ValueError("Packed segment starts and ends must have the same length")
    cu = torch.empty(starts.numel() + 1, dtype=torch.int32, device=starts.device)
    if starts.numel() == 0:
        cu[0] = 0
        return cu
    if starts.numel() > 1 and not bool(torch.equal(starts[1:], ends[:-1])):
        raise ValueError("StreamBP sequence-parallel packed segments must be contiguous")
    cu[0] = starts[0].to(torch.int32)
    cu[1:] = ends.to(torch.int32)
    return cu


def _make_packed_params_from_cu(
    q_cu: Tensor,
    kv_cu: Tensor,
    source: PackedSeqParams,
    *,
    max_seqlen_q: int,
    max_seqlen_kv: int,
) -> PackedSeqParams:
    return PackedSeqParams(
        qkv_format=source.qkv_format,
        cu_seqlens_q=q_cu,
        cu_seqlens_kv=kv_cu,
        cu_seqlens_q_padded=q_cu,
        cu_seqlens_kv_padded=kv_cu,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        local_cp_size=source.local_cp_size,
        cp_group=source.cp_group,
    )


def _make_packed_params_from_lengths(
    q_lengths: Tensor,
    kv_lengths: Tensor,
    source: PackedSeqParams,
) -> PackedSeqParams:
    q_cu = _build_cu_from_lengths(q_lengths)
    kv_cu = _build_cu_from_lengths(kv_lengths)
    max_q = int(q_lengths.max().item()) if q_lengths.numel() > 0 else 0
    max_kv = int(kv_lengths.max().item()) if kv_lengths.numel() > 0 else 0
    return _make_packed_params_from_cu(
        q_cu,
        kv_cu,
        source,
        max_seqlen_q=max_q,
        max_seqlen_kv=max_kv,
    )


def make_streambp_single_sequence_packed_seq_params(
    packed_seq_params: PackedSeqParams,
    query_length: int,
    kv_length: int,
) -> PackedSeqParams:
    """Build local single-sequence THD metadata for sequence-parallel StreamBP.

    Sequence-parallel QKV projections can operate on a TP-local working tensor
    whose packed boundaries no longer match the original global sample
    ``cu_seqlens``. For that path we preserve DSA causality through explicit
    StreamBP positions and use local single-sequence metadata only to describe
    the actual query/KV tensor lengths.
    """
    if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
        raise ValueError("StreamBP local packed metadata currently supports THD only")
    device = _select_cu_seqlens(packed_seq_params, padded=True).device
    q_lengths = torch.tensor([query_length], dtype=torch.int32, device=device)
    kv_lengths = torch.tensor([kv_length], dtype=torch.int32, device=device)
    return _make_packed_params_from_lengths(q_lengths, kv_lengths, packed_seq_params)


def make_streambp_packed_seq_params(
    packed_seq_params: PackedSeqParams,
    start: int,
    end: int,
    *,
    kv_end: Optional[int] = None,
    sequence_offset: int = 0,
) -> tuple[PackedSeqParams, PackedSeqParams]:
    """Build packed THD metadata for a StreamBP query chunk.

    The prefix metadata is used while recomputing Q/K/V and RoPE over
    ``[:end]``. The core-attention metadata uses query lengths for
    ``[start:end]`` and KV lengths for each sequence's prefix through ``kv_end``
    (or ``end`` by default). Keeping those separate is what prevents packed
    samples from attending into each other during DSA chunk replay.

    ``sequence_offset`` is the flattened-token offset for this tensor-parallel
    rank when sequence parallelism has already sharded the THD tokens. Prefix
    metadata keeps that global offset for RoPE, while core-attention metadata
    remains zero-based for the local query/key tensors.
    """
    if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
        raise ValueError("StreamBP packed sequence chunking currently supports THD only")

    cu_seqlens = _select_cu_seqlens(packed_seq_params, padded=True)
    seq_starts = cu_seqlens[:-1]
    seq_ends = cu_seqlens[1:]
    sequence_offset = int(sequence_offset)
    local_start = torch.tensor(
        sequence_offset, dtype=seq_starts.dtype, device=seq_starts.device
    )
    start_tensor = torch.tensor(
        sequence_offset + start, dtype=seq_starts.dtype, device=seq_starts.device
    )
    kv_end = end if kv_end is None else kv_end
    end_tensor = torch.tensor(
        sequence_offset + end, dtype=seq_starts.dtype, device=seq_starts.device
    )
    kv_end_tensor = torch.tensor(
        sequence_offset + kv_end, dtype=seq_starts.dtype, device=seq_starts.device
    )

    kv_starts = torch.maximum(seq_starts, local_start)
    kv_ends = torch.minimum(seq_ends, kv_end_tensor)
    kv_lengths = torch.clamp(kv_ends - kv_starts, min=0)
    query_lengths = torch.clamp(
        torch.minimum(seq_ends, end_tensor) - torch.maximum(seq_starts, start_tensor),
        min=0,
    )
    active = kv_lengths > 0
    if not bool(active.any().item()):
        raise ValueError(
            f"StreamBP packed chunk [{start}, {end}) has no active packed sequence"
        )
    kv_lengths = kv_lengths[active]
    query_lengths = query_lengths[active]

    if int(query_lengths.sum().item()) != end - start:
        raise ValueError(
            f"StreamBP packed query lengths sum to {int(query_lengths.sum().item())}, "
            f"expected {end - start} for chunk [{start}, {end})"
        )

    if sequence_offset:
        kv_starts = kv_starts[active]
        kv_ends = kv_ends[active]
        prefix_cu = _build_cu_from_contiguous_segments(kv_starts, kv_ends)
        prefix_params = _make_packed_params_from_cu(
            prefix_cu,
            prefix_cu,
            packed_seq_params,
            max_seqlen_q=int(prefix_cu[-1].item()),
            max_seqlen_kv=int(prefix_cu[-1].item()),
        )
    else:
        prefix_params = _make_packed_params_from_lengths(
            kv_lengths, kv_lengths, packed_seq_params
        )
    core_params = _make_packed_params_from_lengths(query_lengths, kv_lengths, packed_seq_params)
    return prefix_params, core_params


def slice_streambp_attention_mask(
    attention_mask: Optional[Tensor],
    start: int,
    end: int,
    prefix_end: int,
    *,
    device: torch.device,
    causal: bool,
) -> Optional[Tensor]:
    """Slice an existing mask or synthesize a causal chunk mask.

    Megatron masks use ``True`` for masked positions. The synthesized mask has
    shape ``[1, 1, chunk, prefix]`` and masks keys strictly after each query's
    original sequence position.
    """
    if attention_mask is not None:
        if attention_mask.dim() >= 4:
            return attention_mask[..., start:end, :prefix_end]
        if attention_mask.dim() >= 2:
            return attention_mask[..., start:end, :prefix_end]
        return attention_mask
    if not causal:
        return None
    q_pos = torch.arange(start, end, device=device).view(1, 1, end - start, 1)
    k_pos = torch.arange(prefix_end, device=device).view(1, 1, 1, prefix_end)
    return k_pos > q_pos


def slice_streambp_bias(
    attention_bias: Optional[Tensor], start: int, end: int, prefix_end: int
) -> Optional[Tensor]:
    """Slice an attention bias on query/key dimensions when present."""
    if attention_bias is None or attention_bias.dim() < 2:
        return attention_bias
    return attention_bias[..., start:end, :prefix_end]


def slice_streambp_padding_mask(
    padding_mask: Optional[Tensor], start: int, end: int
) -> Optional[Tensor]:
    """Slice a sequence padding mask for the current query chunk.

    MCore MoE receives padding masks in ``[batch, sequence]`` form from
    TransformerLayer, but a few internal callers/tests use sequence-major masks.
    Preserve either convention.
    """
    if padding_mask is None or padding_mask.dim() < 2:
        return padding_mask
    if padding_mask.size(1) >= end:
        return padding_mask[:, start:end, ...]
    if padding_mask.size(0) >= end:
        return padding_mask[start:end, ...]
    return padding_mask


def slice_streambp_rotary_pos_emb(
    rotary_pos_emb: Optional[tuple[Optional[Tensor], Optional[Tensor]]],
    start: int,
    end: int,
    prefix_end: int,
) -> Optional[tuple[Optional[Tensor], Optional[Tensor]]]:
    """Slice RoPE tensors for chunked query and prefix key/value lengths."""
    if rotary_pos_emb is None:
        return None
    q_pos_emb, k_pos_emb = rotary_pos_emb
    q_pos_emb = None if q_pos_emb is None else q_pos_emb[start:end]
    k_pos_emb = None if k_pos_emb is None else k_pos_emb[:prefix_end]
    return q_pos_emb, k_pos_emb


@contextmanager
def disable_streambp_causal_softmax_fusion(core_attention: Any) -> Iterator[None]:
    """Temporarily disable fused causal softmax for rectangular chunk attention."""
    scale_mask_softmax = getattr(core_attention, "scale_mask_softmax", None)
    if scale_mask_softmax is None or not hasattr(
        scale_mask_softmax, "scaled_masked_softmax_fusion"
    ):
        yield
        return

    old_value = scale_mask_softmax.scaled_masked_softmax_fusion
    scale_mask_softmax.scaled_masked_softmax_fusion = False
    try:
        yield
    finally:
        scale_mask_softmax.scaled_masked_softmax_fusion = old_value


def should_streambp_register_grad_ready(param: torch.nn.Parameter) -> bool:
    """Return True only on the final StreamBP chunk for this parameter."""
    pending = getattr(param, STREAMBP_PENDING_CHUNKS_ATTR, None)
    if pending is None:
        return True
    pending = int(pending) - 1
    if pending <= 0:
        delattr(param, STREAMBP_PENDING_CHUNKS_ATTR)
        return True
    setattr(param, STREAMBP_PENDING_CHUNKS_ATTR, pending)
    return False


def mark_streambp_pending_chunks(
    parameters: Iterable[torch.nn.Parameter], num_chunks: int
) -> list[torch.nn.Parameter]:
    """Mark parameters whose DDP readiness must wait for all chunks."""
    marked: list[torch.nn.Parameter] = []
    if num_chunks <= 1:
        return marked
    for param in parameters:
        if not isinstance(param, torch.nn.Parameter) or not param.requires_grad:
            continue
        existing = getattr(param, STREAMBP_PENDING_CHUNKS_ATTR, None)
        if existing is not None:
            raise RuntimeError(
                "Nested or overlapping StreamBP chunk counters are not supported for one parameter"
            )
        setattr(param, STREAMBP_PENDING_CHUNKS_ATTR, num_chunks)
        marked.append(param)
    return marked


def clear_streambp_pending_chunks(parameters: Iterable[torch.nn.Parameter]) -> None:
    """Clear any leftover StreamBP counters after a recompute block exits."""
    for param in parameters:
        if hasattr(param, STREAMBP_PENDING_CHUNKS_ATTR):
            delattr(param, STREAMBP_PENDING_CHUNKS_ATTR)


def _unique_trainable_parameters(
    module: torch.nn.Module, extra_tensors: Sequence[Optional[Tensor]] = ()
) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    params: list[torch.nn.Parameter] = []

    for param in module.parameters(recurse=True):
        if param.requires_grad and id(param) not in seen:
            seen.add(id(param))
            params.append(param)

    for tensor in extra_tensors:
        if (
            isinstance(tensor, torch.nn.Parameter)
            and tensor.requires_grad
            and id(tensor) not in seen
        ):
            seen.add(id(tensor))
            params.append(tensor)

    return params


def _unique_trainable_parameters_from_modules(
    modules: Iterable[Optional[torch.nn.Module]],
) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    params: list[torch.nn.Parameter] = []
    for module in modules:
        if module is None or not isinstance(module, torch.nn.Module):
            continue
        for param in module.parameters(recurse=True):
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                params.append(param)
    return params


def _streambp_moe_hybrid_pending_marks(
    layer: torch.nn.Module,
    *,
    seq_len: int,
    chunks: Sequence[ChunkRange],
    moe_mlp_chunks: int,
    moe_mlp_backward_chunks: Optional[int],
) -> list[torch.nn.Parameter]:
    """Mark DDP readiness for split MoE replay without over-delaying params.

    Regular StreamBP replays the whole layer once per sequence chunk, so every
    parameter sees the same number of backward hooks.  The DeepSeek MoE hybrid
    path is different: attention parameters are replayed per attention chunk,
    while MoE/MLP parameters are replayed per MLP chunk.  Marking all layer
    parameters with one count either starts communication too early for one
    side or waits for hooks that will never arrive.  Keep the counts matched to
    the replay subgraph that owns each parameter.
    """

    if not _split_moe_mlp_attention_backward_enabled():
        mlp_backward_chunks = moe_mlp_backward_chunks or moe_mlp_chunks
        return mark_streambp_pending_chunks(
            _unique_trainable_parameters(layer),
            len(iter_streambp_num_chunks(seq_len, mlp_backward_chunks)),
        )

    attention_params = _unique_trainable_parameters_from_modules(
        (
            getattr(layer, "input_layernorm", None),
            getattr(layer, "input_gated_norm_down", None),
            getattr(layer, "input_gated_norm_up", None),
            getattr(layer, "self_attention", None),
            getattr(layer, "pre_cross_attn_layernorm", None),
            getattr(layer, "cross_attention", None),
        )
    )
    mlp_params = _unique_trainable_parameters_from_modules(
        (
            getattr(layer, "pre_mlp_layernorm", None),
            getattr(layer, "pre_mlp_gated_norm_down", None),
            getattr(layer, "pre_mlp_gated_norm_up", None),
            getattr(layer, "mlp", None),
        )
    )

    attention_count = len(_moe_attention_backward_chunks(seq_len, chunks))
    mlp_backward_chunks = moe_mlp_backward_chunks or moe_mlp_chunks
    mlp_count = len(iter_streambp_num_chunks(seq_len, mlp_backward_chunks))

    marked: list[torch.nn.Parameter] = []
    marked.extend(mark_streambp_pending_chunks(attention_params, attention_count))
    marked_ids = {id(param) for param in marked}
    marked.extend(
        mark_streambp_pending_chunks(
            (param for param in mlp_params if id(param) not in marked_ids),
            mlp_count,
        )
    )
    return marked


def moe_streambp_requires_full_replay(layer: torch.nn.Module) -> bool:
    """Return True for MoE modes that are not safe for chunked replay yet."""
    if not bool(getattr(layer, "is_moe_layer", False)):
        return False
    config = getattr(layer, "config", None)
    if config is None:
        return True

    routing_type = getattr(config, "moe_router_load_balancing_type", "aux_loss")
    routing_types = routing_type if isinstance(routing_type, list) else [routing_type]
    if "sinkhorn" in routing_types or "global_aux_loss" in routing_types:
        return True
    if getattr(config, "moe_expert_capacity_factor", None) is not None:
        return True
    if getattr(config, "moe_z_loss_coeff", None):
        return True
    return False


def supports_streambp_moe_hybrid_replay(layer: torch.nn.Module) -> bool:
    """Return True when StreamBP can chunk attention but replay MoE full-sequence."""
    return (
        bool(getattr(layer, "is_moe_layer", False))
        and hasattr(layer, "_forward_attention")
        and hasattr(layer, "_forward_mlp")
    )


@contextmanager
def _maybe_context(context_factory: ContextFactory) -> Iterator[None]:
    if context_factory is None:
        with nullcontext():
            yield
        return
    context = context_factory()
    with context:
        yield


@contextmanager
def _te_activation_recompute_context(*, recompute_phase: bool) -> Iterator[None]:
    """Match TransformerEngine checkpoint contexts for custom StreamBP recompute."""
    try:
        from transformer_engine.pytorch.distributed import activation_recompute_forward
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    except (ImportError, ModuleNotFoundError):
        with nullcontext():
            yield
        return

    if not FP8GlobalStateManager.is_fp8_enabled():
        with nullcontext():
            yield
        return

    with activation_recompute_forward(
        activation_recompute=True,
        recompute_phase=recompute_phase,
    ):
        yield


def _call_layer(
    layer: torch.nn.Module,
    hidden_states: Tensor,
    kwargs: dict[str, Any],
    *,
    chunk_range: Optional[ChunkRange] = None,
    context_factory: ContextFactory = None,
    activation_recompute_phase: Optional[bool] = None,
) -> Tensor:
    call_kwargs = dict(kwargs)
    if chunk_range is not None:
        call_kwargs["chunk_range"] = chunk_range
    with _maybe_context(context_factory):
        if activation_recompute_phase is None:
            output, context = layer(hidden_states=hidden_states, **call_kwargs)
        else:
            with _te_activation_recompute_context(
                recompute_phase=activation_recompute_phase
            ):
                output, context = layer(hidden_states=hidden_states, **call_kwargs)
    if context is not None:
        raise ValueError("StreamBP currently supports decoder-only layers with context=None")
    return output


def _make_grad_anchor(hidden_states: Tensor) -> Tensor:
    """Create a scalar grad anchor so checkpointed layers run even for gradless inputs."""
    return torch.empty((), dtype=torch.float32, device=hidden_states.device, requires_grad=True)


def _chunked_no_grad_forward(
    layer: torch.nn.Module,
    hidden_states: Tensor,
    chunks: Sequence[ChunkRange],
    kwargs: dict[str, Any],
    *,
    context_factory: ContextFactory,
) -> Tensor:
    """Run the no-grad StreamBP forward one query chunk at a time."""
    seq_len = hidden_states.size(0)
    output_buffer: Optional[Tensor] = None
    fallback_outputs = []
    for chunk_index, chunk_range in enumerate(chunks):
        start, end = chunk_range
        with _profile_streambp_chunk(f"streambp/no_grad_forward_chunk/{chunk_index}"):
            output = _call_layer(
                layer,
                hidden_states,
                kwargs,
                chunk_range=chunk_range,
                context_factory=context_factory,
                activation_recompute_phase=False,
            )
            if torch.is_tensor(output):
                output_buffer = _copy_streambp_tensor_chunk(
                    output_buffer, output, start, end, seq_len
                )
                del output
            else:
                fallback_outputs.append(output)
    if output_buffer is not None:
        if fallback_outputs:
            raise TypeError("Cannot mix tensor and non-tensor StreamBP chunk outputs")
        return output_buffer
    return _concat_streambp_chunk_outputs(fallback_outputs)


def _reference_no_grad_forward(
    layer: torch.nn.Module,
    hidden_states: Tensor,
    kwargs: dict[str, Any],
    *,
    context_factory: ContextFactory,
) -> Tensor:
    """Run the no-grad StreamBP forward without query chunking.

    This matches the public StreamBP reference implementation: the activation
    checkpoint forward is full-sequence, while backward replay is chunked.
    """
    with _profile_streambp_chunk("streambp/no_grad_forward_full"):
        return _call_layer(
            layer,
            hidden_states,
            kwargs,
            context_factory=context_factory,
            activation_recompute_phase=False,
        )


def _moe_chunk_attention_full_mlp_no_grad_forward(
    layer: torch.nn.Module,
    hidden_states: Tensor,
    chunks: Sequence[ChunkRange],
    kwargs: dict[str, Any],
    *,
    context_factory: ContextFactory,
    moe_mlp_chunks: int,
) -> Tensor:
    """Run chunked attention followed by one full-sequence MoE MLP no-grad forward.

    DeepSeek-style layers can be both MoE layers and DSA-attention layers. A full
    no-grad layer forward would also run DSA attention full-sequence, which defeats
    the packed/chunked DSA path and can hit unsupported kernel shapes. This hybrid
    path keeps attention chunked, then runs the MoE MLP once on the concatenated
    post-attention states to avoid multiplying MoE dispatcher collectives in the
    no-grad forward.
    """
    if not hasattr(layer, "_forward_attention") or not hasattr(layer, "_forward_mlp"):
        return _reference_no_grad_forward(
            layer,
            hidden_states,
            kwargs,
            context_factory=context_factory,
        )

    call_kwargs = dict(kwargs)
    # These are whole-layer wrapper hints. The regular TransformerLayer.forward
    # strips them before entering the attention/MLP internals.
    call_kwargs.pop("dynamic_inference_decode_only", None)
    call_kwargs.pop("mhc_recompute_manager", None)
    with _maybe_context(context_factory):
        if moe_mlp_chunks == 1:
            seq_len = hidden_states.size(0)
            attention_outputs: list[tuple[int, int, Tensor]] = []
            for chunk_index, chunk_range in enumerate(chunks):
                chunk_start, chunk_end = chunk_range
                with _profile_streambp_chunk(
                    f"streambp/no_grad_forward_moe_attention_chunk/{chunk_index}"
                ):
                    _maybe_trim_cuda_cache_before_moe_replay()
                    # Backward replay still recomputes this MoE layer once per chunk.
                    # TE's FP8 activation-recompute context keeps one bookkeeping entry
                    # per replayed forward, so seed that stack even though this hybrid
                    # no-grad path only chunks attention and runs the MoE MLP once.
                    with _te_activation_recompute_context(recompute_phase=False):
                        attention_output, context = layer._forward_attention(
                            hidden_states=hidden_states,
                            chunk_range=chunk_range,
                            **call_kwargs,
                        )
                    if context is not None:
                        raise ValueError(
                            "StreamBP currently supports decoder-only MoE layers with context=None"
                        )
                    attention_outputs.append((chunk_start, chunk_end, attention_output))

            post_attention = _copy_sequence_tensor_ranges(attention_outputs, 0, seq_len)
            del attention_outputs
            tensor_audit(
                "streambp/moe_no_grad/post_attention_full",
                post_attention=post_attention,
                layer=getattr(layer, "layer_number", "?"),
            )
            with _profile_streambp_chunk("streambp/no_grad_forward_moe_mlp_full"):
                _maybe_trim_cuda_cache_before_moe_replay()
                with _te_activation_recompute_context(recompute_phase=False):
                    output = layer._forward_mlp(
                        post_attention,
                        call_kwargs.get("inference_context", None),
                        padding_mask=call_kwargs.get("padding_mask", None),
                    )
            tensor_audit(
                "streambp/moe_no_grad/mlp_output_full",
                output=output,
                layer=getattr(layer, "layer_number", "?"),
            )
            return output

        seq_len = hidden_states.size(0)
        chunk_size = chunks[0][1] - chunks[0][0] if chunks else seq_len
        padding_mask = call_kwargs.get("padding_mask", None)
        mlp_output_buffer: Optional[Tensor] = None
        fallback_mlp_outputs = []
        for mlp_chunk_index, (mlp_start, mlp_end) in enumerate(
            iter_streambp_num_chunks(seq_len, moe_mlp_chunks)
        ):
            attention_outputs: list[tuple[int, int, Tensor]] = []
            for attention_start, attention_end in _intersect_streambp_chunk_ranges(
                chunks, mlp_start, mlp_end
            ):
                attention_chunk_index = attention_start // chunk_size
                with _profile_streambp_chunk(
                    "streambp/no_grad_forward_moe_attention_chunk/"
                    f"{attention_chunk_index}"
                ):
                    _maybe_trim_cuda_cache_before_moe_replay()
                    with _te_activation_recompute_context(recompute_phase=False):
                        attention_output, context = layer._forward_attention(
                            hidden_states=hidden_states,
                            chunk_range=(attention_start, attention_end),
                            **call_kwargs,
                        )
                    if context is not None:
                        raise ValueError(
                            "StreamBP currently supports decoder-only MoE layers with context=None"
                        )
                    attention_outputs.append((attention_start, attention_end, attention_output))
            post_attention = _copy_sequence_tensor_ranges(attention_outputs, mlp_start, mlp_end)
            del attention_outputs
            if post_attention.size(0) != mlp_end - mlp_start:
                raise RuntimeError(
                    "StreamBP MoE no-grad forward produced attention chunk length "
                    f"{post_attention.size(0)} for MLP range [{mlp_start}, {mlp_end})"
                )
            tensor_audit(
                "streambp/moe_no_grad/post_attention_chunk",
                post_attention=post_attention,
                mlp_chunk=mlp_chunk_index,
                layer=getattr(layer, "layer_number", "?"),
            )
            with _profile_streambp_chunk(
                f"streambp/no_grad_forward_moe_mlp_chunk/{mlp_chunk_index}"
            ):
                _maybe_trim_cuda_cache_before_moe_replay()
                with _te_activation_recompute_context(recompute_phase=False):
                    mlp_output = layer._forward_mlp(
                        post_attention,
                        call_kwargs.get("inference_context", None),
                        padding_mask=_slice_padding_mask_for_sequence_chunk(
                            padding_mask, mlp_start, mlp_end, seq_len
                        ),
                    )
                    tensor_audit(
                        "streambp/moe_no_grad/mlp_output_chunk",
                        output=mlp_output,
                        mlp_chunk=mlp_chunk_index,
                        layer=getattr(layer, "layer_number", "?"),
                    )
                    if torch.is_tensor(mlp_output):
                        mlp_output_buffer = _copy_streambp_tensor_chunk(
                            mlp_output_buffer, mlp_output, mlp_start, mlp_end, seq_len
                        )
                        del mlp_output
                    else:
                        fallback_mlp_outputs.append(mlp_output)
            del post_attention
        if mlp_output_buffer is not None:
            if fallback_mlp_outputs:
                raise TypeError("Cannot mix tensor and non-tensor StreamBP MoE chunk outputs")
            return mlp_output_buffer
        return _concat_streambp_chunk_outputs(fallback_mlp_outputs)


def _moe_chunk_attention_full_mlp_backward(
    layer: torch.nn.Module,
    hidden_states: Tensor,
    grad_output: Tensor,
    chunks: Sequence[ChunkRange],
    kwargs: dict[str, Any],
    *,
    context_factory: ContextFactory,
    moe_mlp_chunks: int,
    moe_mlp_backward_chunks: Optional[int],
    moe_aux_stats: Optional[StreamBPMoeAuxState],
) -> None:
    """Replay MoE backward while keeping only one MLP chunk's attention graphs live."""
    call_kwargs = dict(kwargs)
    # These are whole-layer wrapper hints. The regular TransformerLayer.forward
    # strips them before entering the attention/MLP internals.
    call_kwargs.pop("dynamic_inference_decode_only", None)
    call_kwargs.pop("mhc_recompute_manager", None)

    seq_len = hidden_states.size(0)
    chunk_size = chunks[0][1] - chunks[0][0] if chunks else seq_len
    attention_backward_chunks = _moe_attention_backward_chunks(seq_len, chunks)
    layer_number = getattr(layer, "layer_number", "?")
    mlp_backward_chunks = moe_mlp_backward_chunks or moe_mlp_chunks

    def forward_mlp_chunk_index_for_range(start: int) -> int:
        if moe_mlp_chunks <= 1:
            return 0
        return min(moe_mlp_chunks - 1, (start * moe_mlp_chunks) // seq_len)

    with _maybe_context(context_factory):
        if mlp_backward_chunks == 1:
            with _streambp_replay_saved_tensor_offload_context(
                f"moe_full_layer{layer_number}"
            ):
                attention_outputs: list[tuple[int, int, Tensor]] = []
                for chunk_range in chunks:
                    start, end = chunk_range
                    chunk_index = start // chunk_size
                    with _profile_streambp_chunk(
                        f"streambp/backward_replay_moe_attention_chunk/{chunk_index}"
                    ):
                        _maybe_trim_cuda_cache_before_moe_replay()
                        with _te_activation_recompute_context(recompute_phase=True):
                            _debug_sync(
                                f"layer{layer_number}:bwd_moe_attention_chunk{chunk_index}:before"
                            )
                            attention_output, context = layer._forward_attention(
                                hidden_states=hidden_states,
                                chunk_range=chunk_range,
                                **call_kwargs,
                            )
                            _debug_sync(
                                f"layer{layer_number}:bwd_moe_attention_chunk{chunk_index}:after"
                            )
                        if context is not None:
                            raise ValueError(
                                "StreamBP currently supports decoder-only MoE layers with context=None"
                            )
                        attention_outputs.append((start, end, attention_output))
                post_attention = _copy_sequence_tensor_ranges(attention_outputs, 0, seq_len)
                del attention_outputs
                tensor_audit(
                    "streambp/moe_backward/post_attention_full",
                    post_attention=post_attention,
                    layer=layer_number,
                )
                with _profile_streambp_chunk("streambp/backward_replay_moe_mlp_full"):
                    _maybe_trim_cuda_cache_before_moe_replay()
                    with (
                        replay_streambp_moe_aux_stats(moe_aux_stats),
                        _te_activation_recompute_context(recompute_phase=True),
                    ):
                        _debug_sync(f"layer{layer_number}:bwd_moe_mlp_full:before")
                        output = layer._forward_mlp(
                            post_attention,
                            call_kwargs.get("inference_context", None),
                            padding_mask=call_kwargs.get("padding_mask", None),
                        )
                        tensor_audit(
                            "streambp/moe_backward/mlp_output_full",
                            output=output,
                            grad_output=grad_output,
                            layer=layer_number,
                        )
                        _debug_sync(f"layer{layer_number}:bwd_moe_mlp_full:after")
                _debug_sync(f"layer{layer_number}:bwd_moe_full_autograd:before")
                _streambp_autograd_backward(output, grad_output)
                _debug_sync(f"layer{layer_number}:bwd_moe_full_autograd:after")
        else:
            padding_mask = call_kwargs.get("padding_mask", None)
            for mlp_chunk_index, (mlp_start, mlp_end) in enumerate(
                iter_streambp_num_chunks(seq_len, mlp_backward_chunks)
            ):
                with _streambp_replay_saved_tensor_offload_context(
                    f"moe_mlp_chunk{mlp_chunk_index}_layer{layer_number}"
                ):
                    attention_outputs: list[tuple[int, int, Tensor]] = []
                    for attention_start, attention_end in _intersect_streambp_chunk_ranges(
                        attention_backward_chunks, mlp_start, mlp_end
                    ):
                        attention_chunk_index = attention_start // chunk_size
                        with _profile_streambp_chunk(
                            "streambp/backward_replay_moe_attention_chunk/"
                            f"{attention_chunk_index}"
                        ):
                            _maybe_trim_cuda_cache_before_moe_replay()
                            with _te_activation_recompute_context(recompute_phase=True):
                                _debug_sync(
                                    f"layer{layer_number}:bwd_moe_mlp_chunk{mlp_chunk_index}:"
                                    f"attention_chunk{attention_chunk_index}:before"
                                )
                                attention_output, context = layer._forward_attention(
                                    hidden_states=hidden_states,
                                    chunk_range=(attention_start, attention_end),
                                    **call_kwargs,
                                )
                                _debug_sync(
                                    f"layer{layer_number}:bwd_moe_mlp_chunk{mlp_chunk_index}:"
                                    f"attention_chunk{attention_chunk_index}:after"
                                )
                            if context is not None:
                                raise ValueError(
                                    "StreamBP currently supports decoder-only MoE layers "
                                    "with context=None"
                                )
                            attention_outputs.append(
                                (attention_start, attention_end, attention_output)
                            )
                    post_attention = _copy_sequence_tensor_ranges(
                        attention_outputs,
                        mlp_start,
                        mlp_end,
                        clear_chunks=False,
                    )
                    if post_attention.size(0) != mlp_end - mlp_start:
                        raise RuntimeError(
                            "StreamBP MoE replay produced attention chunk length "
                            f"{post_attention.size(0)} for MLP range "
                            f"[{mlp_start}, {mlp_end})"
                        )
                    split_mlp_attention_backward = (
                        _split_moe_mlp_attention_backward_enabled()
                        and post_attention.requires_grad
                    )
                    mlp_input = (
                        post_attention.detach().requires_grad_(True)
                        if split_mlp_attention_backward
                        else post_attention
                    )
                    tensor_audit(
                        "streambp/moe_backward/post_attention_chunk",
                        post_attention=post_attention,
                        grad_chunk=grad_output[mlp_start:mlp_end],
                        mlp_chunk=mlp_chunk_index,
                            layer=layer_number,
                        )
                    if split_mlp_attention_backward:
                        # The detached MLP input shares storage with post_attention.
                        # Release the copied buffer's autograd graph immediately;
                        # the attention graph is consumed below through the original
                        # attention_outputs chunks.
                        del post_attention
                    else:
                        attention_outputs.clear()
                    with _profile_streambp_chunk(
                        f"streambp/backward_replay_moe_mlp_chunk/{mlp_chunk_index}"
                    ):
                        _maybe_trim_cuda_cache_before_moe_replay()
                        with (
                            replay_streambp_moe_aux_stats(
                                moe_aux_stats,
                                chunk_index=forward_mlp_chunk_index_for_range(mlp_start),
                            ),
                            _te_activation_recompute_context(recompute_phase=True),
                            _streambp_replay_save_quantized_te_inputs(layer),
                        ):
                            _debug_sync(
                                f"layer{layer_number}:bwd_moe_mlp_chunk{mlp_chunk_index}:before"
                            )
                            chunk_output = layer._forward_mlp(
                                mlp_input,
                                call_kwargs.get("inference_context", None),
                                padding_mask=_slice_padding_mask_for_sequence_chunk(
                                    padding_mask, mlp_start, mlp_end, seq_len
                                ),
                            )
                            tensor_audit(
                                "streambp/moe_backward/mlp_output_chunk",
                                output=chunk_output,
                                grad_chunk=grad_output[mlp_start:mlp_end],
                                mlp_chunk=mlp_chunk_index,
                                layer=layer_number,
                            )
                            _debug_sync(
                                f"layer{layer_number}:bwd_moe_mlp_chunk{mlp_chunk_index}:after"
                            )
                        if not torch.is_tensor(chunk_output):
                            raise TypeError(
                                "StreamBP split MoE MLP replay expects TransformerLayer._forward_mlp "
                                f"to return a Tensor, got {type(chunk_output)}"
                            )
                        grad_chunk = grad_output[mlp_start:mlp_end]
                        release_mlp_input_before_backward = _env_flag(
                            "MEGATRON_STREAMBP_RELEASE_MLP_INPUT_BEFORE_BACKWARD", default=True
                        )
                        if release_mlp_input_before_backward and not split_mlp_attention_backward:
                            del post_attention
                        _debug_sync(
                            f"layer{layer_number}:bwd_moe_mlp_chunk{mlp_chunk_index}:"
                            "autograd_before"
                        )
                        _maybe_trim_cuda_cache_before_moe_replay(force=True, synchronize=True)
                        _streambp_autograd_backward(chunk_output, grad_chunk)
                        _debug_sync(
                            f"layer{layer_number}:bwd_moe_mlp_chunk{mlp_chunk_index}:"
                            "autograd_after"
                        )
                        del chunk_output, grad_chunk
                        if split_mlp_attention_backward:
                            attention_grad = mlp_input.grad
                            tensor_audit(
                                "streambp/moe_backward/mlp_input_grad",
                                grad=attention_grad,
                                mlp_chunk=mlp_chunk_index,
                                layer=layer_number,
                            )
                            del mlp_input
                            if attention_grad is None:
                                raise RuntimeError(
                                    "StreamBP split MoE MLP replay did not produce an "
                                    "input gradient for the attention replay graph"
                                )
                            _debug_sync(
                                f"layer{layer_number}:bwd_moe_mlp_chunk{mlp_chunk_index}:"
                                "attention_autograd_before"
                            )
                            _maybe_trim_cuda_cache_before_moe_replay(
                                force=True, synchronize=True
                            )
                            _streambp_autograd_backward_attention_outputs(
                                attention_outputs,
                                mlp_start,
                                attention_grad,
                            )
                            _debug_sync(
                                f"layer{layer_number}:bwd_moe_mlp_chunk{mlp_chunk_index}:"
                                "attention_autograd_after"
                            )
                            attention_outputs.clear()
                            del attention_grad
                        elif not release_mlp_input_before_backward:
                            del post_attention


def _full_layer_activation_checkpoint(
    layer: torch.nn.Module,
    hidden_states: Tensor,
    kwargs: dict[str, Any],
    *,
    context_factory: ContextFactory,
) -> Tensor:
    """Replay a whole layer without chunking while preserving StreamBP TE contexts."""
    return _StreamBPFullLayerCheckpoint.apply(
        hidden_states,
        _make_grad_anchor(hidden_states),
        layer,
        context_factory,
        kwargs,
    )


class _StreamBPLayerCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        grad_anchor: Tensor,
        chunk_size: Optional[int],
        chunk_forward: bool,
        moe_mlp_chunks: int,
        moe_mlp_backward_chunks: Optional[int],
        layer: torch.nn.Module,
        context_factory: ContextFactory,
        kwargs: dict[str, Any],
    ) -> Tensor:
        del grad_anchor
        ctx.layer = layer
        ctx.chunk_size = chunk_size
        ctx.chunk_forward = chunk_forward
        ctx.moe_mlp_chunks = moe_mlp_chunks
        ctx.moe_mlp_backward_chunks = moe_mlp_backward_chunks
        ctx.context_factory = context_factory
        ctx.kwargs = kwargs
        chunks = iter_streambp_chunks(hidden_states.size(0), chunk_size)
        with torch.no_grad():
            if getattr(layer, "is_moe_layer", False):
                with capture_streambp_moe_aux_stats() as moe_aux_stats:
                    if chunk_forward:
                        output = _chunked_no_grad_forward(
                            layer,
                            hidden_states,
                            chunks,
                            kwargs,
                            context_factory=context_factory,
                        )
                    else:
                        output = _moe_chunk_attention_full_mlp_no_grad_forward(
                            layer,
                            hidden_states,
                            chunks,
                            kwargs,
                            context_factory=context_factory,
                            moe_mlp_chunks=moe_mlp_chunks,
                        )
                ctx.moe_aux_stats = moe_aux_stats
                _streambp_save_hidden_for_backward(ctx, hidden_states)
                return output
            ctx.moe_aux_stats = None
            if not chunk_forward:
                output = _reference_no_grad_forward(
                    layer,
                    hidden_states,
                    kwargs,
                    context_factory=context_factory,
                )
            else:
                output = _chunked_no_grad_forward(
                    layer,
                    hidden_states,
                    chunks,
                    kwargs,
                    context_factory=context_factory,
                )
            _streambp_save_hidden_for_backward(ctx, hidden_states)
            return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        hidden_states = _streambp_load_hidden_for_backward(ctx)
        chunks = iter_streambp_chunks(hidden_states.size(0), ctx.chunk_size)

        if (
            getattr(ctx.layer, "is_moe_layer", False)
            and not ctx.chunk_forward
            and hasattr(ctx.layer, "_forward_attention")
            and hasattr(ctx.layer, "_forward_mlp")
        ):
            detached_hidden_states = hidden_states.detach().requires_grad_(
                ctx.needs_input_grad[0]
            )
            marked = _streambp_moe_hybrid_pending_marks(
                ctx.layer,
                seq_len=hidden_states.size(0),
                chunks=chunks,
                moe_mlp_chunks=ctx.moe_mlp_chunks,
                moe_mlp_backward_chunks=ctx.moe_mlp_backward_chunks,
            )
            try:
                with _suppress_fine_grained_offload_replay_context(), torch.enable_grad():
                    _moe_chunk_attention_full_mlp_backward(
                        ctx.layer,
                        detached_hidden_states,
                        grad_output,
                        chunks,
                        ctx.kwargs,
                        context_factory=ctx.context_factory,
                        moe_mlp_chunks=ctx.moe_mlp_chunks,
                        moe_mlp_backward_chunks=ctx.moe_mlp_backward_chunks,
                        moe_aux_stats=ctx.moe_aux_stats,
                    )
            finally:
                clear_streambp_pending_chunks(marked)
            hidden_grad = detached_hidden_states.grad if ctx.needs_input_grad[0] else None
            return hidden_grad, None, None, None, None, None, None, None, None

        params = _unique_trainable_parameters(ctx.layer)
        marked = mark_streambp_pending_chunks(params, len(chunks))

        detached_hidden_states = hidden_states.detach().requires_grad_(ctx.needs_input_grad[0])
        try:
            with _suppress_fine_grained_offload_replay_context(), torch.enable_grad():
                for chunk_range in chunks:
                    start, end = chunk_range
                    chunk_index = start // resolve_streambp_chunk_size(
                        hidden_states.size(0), ctx.chunk_size
                    )
                    with _profile_streambp_chunk(
                        f"streambp/backward_replay_chunk/{chunk_index}"
                    ):
                        with replay_streambp_moe_aux_stats(
                            ctx.moe_aux_stats,
                            chunk_index=chunk_index if ctx.moe_aux_stats is not None else None,
                        ):
                            chunk_output = _call_layer(
                                ctx.layer,
                                detached_hidden_states,
                                ctx.kwargs,
                                chunk_range=chunk_range,
                                context_factory=ctx.context_factory,
                                activation_recompute_phase=True,
                            )
                        _streambp_autograd_backward(chunk_output, grad_output[start:end])
            hidden_grad = detached_hidden_states.grad if ctx.needs_input_grad[0] else None
        finally:
            clear_streambp_pending_chunks(marked)

        return hidden_grad, None, None, None, None, None, None, None, None


class _StreamBPFullLayerCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        grad_anchor: Tensor,
        layer: torch.nn.Module,
        context_factory: ContextFactory,
        kwargs: dict[str, Any],
    ) -> Tensor:
        del grad_anchor
        ctx.layer = layer
        ctx.context_factory = context_factory
        ctx.kwargs = kwargs
        ctx.cpu_rng_state = torch.get_rng_state()
        ctx.cuda_device_index = None
        ctx.cuda_rng_state = None
        if hidden_states.is_cuda:
            ctx.cuda_device_index = hidden_states.device.index
            if ctx.cuda_device_index is None:
                ctx.cuda_device_index = torch.cuda.current_device()
            ctx.cuda_rng_state = torch.cuda.get_rng_state(ctx.cuda_device_index)
        with torch.no_grad():
            output = _call_layer(
                layer,
                hidden_states,
                kwargs,
                context_factory=context_factory,
                activation_recompute_phase=False,
            )
        _streambp_save_hidden_for_backward(ctx, hidden_states)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        hidden_states = _streambp_load_hidden_for_backward(ctx)
        detached_hidden_states = hidden_states.detach().requires_grad_(ctx.needs_input_grad[0])
        devices = [ctx.cuda_device_index] if ctx.cuda_device_index is not None else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.set_rng_state(ctx.cpu_rng_state)
            if ctx.cuda_device_index is not None and ctx.cuda_rng_state is not None:
                torch.cuda.set_rng_state(ctx.cuda_rng_state, ctx.cuda_device_index)
            with _suppress_fine_grained_offload_replay_context(), torch.enable_grad():
                output = _call_layer(
                    ctx.layer,
                    detached_hidden_states,
                    ctx.kwargs,
                    context_factory=ctx.context_factory,
                    activation_recompute_phase=True,
                )
                _streambp_autograd_backward(output, grad_output)
        hidden_grad = detached_hidden_states.grad if ctx.needs_input_grad[0] else None
        return hidden_grad, None, None, None, None


def streambp_checkpoint_layer(
    layer: torch.nn.Module,
    hidden_states: Tensor,
    *,
    chunk_size: Optional[int],
    chunk_forward: bool = True,
    moe_mlp_chunks: int = 1,
    moe_mlp_backward_chunks: Optional[int] = None,
    context_factory: ContextFactory = None,
    full_replay: bool = False,
    **kwargs: Any,
) -> tuple[Tensor, None]:
    """Run a transformer layer with StreamBP chunked backward recomputation."""
    if kwargs.get("context") is not None:
        raise ValueError("StreamBP currently supports decoder-only layers with context=None")
    chunks = iter_streambp_chunks(hidden_states.size(0), chunk_size)
    if len(chunks) <= 1 or not torch.is_grad_enabled():
        return _call_layer(layer, hidden_states, kwargs, context_factory=context_factory), None
    requires_full_moe = moe_streambp_requires_full_replay(layer)
    if full_replay or (
        requires_full_moe
        and (chunk_forward or not supports_streambp_moe_hybrid_replay(layer))
    ):
        output = _full_layer_activation_checkpoint(
            layer, hidden_states, kwargs, context_factory=context_factory
        )
        return output, None
    output = _StreamBPLayerCheckpoint.apply(
        hidden_states,
        _make_grad_anchor(hidden_states),
        chunk_size,
        chunk_forward,
        moe_mlp_chunks,
        moe_mlp_backward_chunks,
        layer,
        context_factory,
        kwargs,
    )
    return output, None


def _call_output_layer(
    output_layer: torch.nn.Module, hidden_states: Tensor, kwargs: dict[str, Any]
) -> Tensor:
    call_kwargs = dict(kwargs)
    call_kwargs["input_"] = hidden_states
    logits, _ = output_layer(**call_kwargs)
    return logits


@lru_cache(maxsize=1)
def _load_streambp_fused_lce_entry():
    if not torch.cuda.is_available():
        return None
    try:
        device = torch.cuda.current_device()
        if torch.cuda.get_device_capability(device)[0] != 10:
            return None
        from megatron.core.fusions.linear_cross_entropy.blackwell import entry as blackwell_entry

        if not hasattr(blackwell_entry, "forward") or not hasattr(blackwell_entry, "backward"):
            return None
    except Exception:
        return None
    return blackwell_entry


@lru_cache(maxsize=1)
def _load_streambp_fused_lce() -> Optional[Callable[..., Tensor]]:
    try:
        blackwell_entry = _load_streambp_fused_lce_entry()
        if blackwell_entry is None:
            return None
        from megatron.core.fusions.fused_linear_cross_entropy import linear_cross_entropy
    except Exception:
        return None
    return linear_cross_entropy


def _streambp_lce_sp_tile_size(local_seq_len: int) -> int:
    raw = os.getenv("MEGATRON_STREAMBP_FUSED_LCE_SP_TILE_SIZE", "1024")
    try:
        tile_size = int(raw)
    except ValueError as exc:
        raise ValueError(
            "MEGATRON_STREAMBP_FUSED_LCE_SP_TILE_SIZE must be an integer, "
            f"got {raw!r}"
        ) from exc
    if tile_size <= 0:
        raise ValueError("MEGATRON_STREAMBP_FUSED_LCE_SP_TILE_SIZE must be positive")
    return min(tile_size, local_seq_len)


def _streambp_lce_weight(
    output_layer: torch.nn.Module, kwargs: dict[str, Any]
) -> Optional[Tensor]:
    weight = kwargs.get("weight")
    if weight is None:
        weight = getattr(output_layer, "weight", None)
    return weight if isinstance(weight, Tensor) else None


class _StreamBPFusedSPLinearCrossEntropy(torch.autograd.Function):
    """Sequence-parallel fused LCE that streams TP all-gather tiles.

    The Blackwell LCE entry points are still responsible for the tensor-core
    linear/CE work.  This wrapper changes only the sequence-parallel host
    schedule: it avoids saving one full all-gathered hidden tensor for backward
    and bounds the backward scratch tensors to a local sequence tile.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        weight: Tensor,
        labels: Tensor,
        tp_group: Any,
        tile_size: int,
        ignore_index: int,
    ) -> Tensor:
        if tp_group is None:
            raise RuntimeError("streaming sequence-parallel LCE requires a TP group")
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            raise RuntimeError("streaming sequence-parallel LCE requires distributed init")

        blackwell_entry = _load_streambp_fused_lce_entry()
        if blackwell_entry is None:
            raise RuntimeError("Blackwell fused LCE entry is unavailable")

        tp_rank = torch.distributed.get_rank(tp_group)
        tp_world_size = torch.distributed.get_world_size(tp_group)
        local_seq_len = hidden_states.size(0)
        if labels.size(1) != local_seq_len * tp_world_size:
            raise ValueError(
                "streaming sequence-parallel LCE label length mismatch: "
                f"labels_s={labels.size(1)} local_s={local_seq_len} tp={tp_world_size}"
            )

        loss = torch.empty(labels.shape, device=hidden_states.device, dtype=torch.float32)
        maximum = torch.empty(
            (tp_world_size, local_seq_len, hidden_states.size(1)),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        accumulate = torch.empty_like(maximum)
        num_valid_tokens: list[Tensor] = []
        chunks = iter_streambp_chunks(local_seq_len, tile_size)

        for start, end in chunks:
            local_hidden = hidden_states[start:end].contiguous()
            global_hidden = torch.empty(
                (tp_world_size * (end - start), *hidden_states.shape[1:]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            torch.distributed.all_gather_into_tensor(
                global_hidden, local_hidden, group=tp_group
            )
            labels_chunk = _rank_concatenated_sequence_chunk(
                labels,
                start,
                end,
                local_seq_len=local_seq_len,
                world_size=tp_world_size,
            )
            labels_t = labels_chunk.transpose(0, 1).contiguous()
            (
                logprobs,
                maximum_chunk,
                accumulate_chunk,
                num_valid_chunk,
                _,
                _,
                _,
            ) = blackwell_entry.forward(
                global_hidden,
                weight,
                labels_t,
                tp_group,
                "none",
                ignore_index,
                False,
            )
            _scatter_rank_concatenated_sequence_chunk(
                loss,
                logprobs.view_as(labels_t).transpose(0, 1).contiguous(),
                start,
                end,
                local_seq_len=local_seq_len,
                world_size=tp_world_size,
            )
            maximum[:, start:end, :] = maximum_chunk.view(
                tp_world_size, end - start, hidden_states.size(1)
            )
            accumulate[:, start:end, :] = accumulate_chunk.view(
                tp_world_size, end - start, hidden_states.size(1)
            )
            num_valid_tokens.append(num_valid_chunk)

        ctx.tp_group = tp_group
        ctx.tp_rank = tp_rank
        ctx.tp_world_size = tp_world_size
        ctx.tile_size = tile_size
        ctx.ignore_index = ignore_index
        ctx.save_for_backward(
            hidden_states,
            weight,
            labels,
            maximum,
            accumulate,
            torch.stack(num_valid_tokens),
        )
        tensor_audit(
            "streambp_lce/sp_stream_forward_done",
            hidden=hidden_states,
            weight=weight,
            labels=labels,
            loss=loss,
            tile_size=tile_size,
            num_chunks=len(chunks),
        )
        return loss

    @staticmethod
    def backward(ctx, grad_loss: Tensor):
        hidden_states, weight, labels, maximum, accumulate, num_valid_tokens = ctx.saved_tensors
        blackwell_entry = _load_streambp_fused_lce_entry()
        if blackwell_entry is None:
            raise RuntimeError("Blackwell fused LCE entry is unavailable")

        local_seq_len = hidden_states.size(0)
        chunks = iter_streambp_chunks(local_seq_len, ctx.tile_size)
        hidden_grad = torch.empty_like(hidden_states) if ctx.needs_input_grad[0] else None
        weight_grad = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None

        for chunk_index, (start, end) in enumerate(chunks):
            local_hidden = hidden_states[start:end].contiguous()
            global_hidden = torch.empty(
                (ctx.tp_world_size * (end - start), *hidden_states.shape[1:]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            torch.distributed.all_gather_into_tensor(
                global_hidden, local_hidden, group=ctx.tp_group
            )
            labels_chunk = _rank_concatenated_sequence_chunk(
                labels,
                start,
                end,
                local_seq_len=local_seq_len,
                world_size=ctx.tp_world_size,
            )
            grad_chunk = _rank_concatenated_sequence_chunk(
                grad_loss,
                start,
                end,
                local_seq_len=local_seq_len,
                world_size=ctx.tp_world_size,
            )
            labels_t = labels_chunk.transpose(0, 1).contiguous()
            grad_t = grad_chunk.transpose(0, 1).contiguous()
            maximum_chunk = maximum[:, start:end, :].reshape(-1).contiguous()
            accumulate_chunk = accumulate[:, start:end, :].reshape(-1).contiguous()

            hidden_grad_chunk, weight_grad_chunk = blackwell_entry.backward(
                grad_t.reshape(-1).contiguous(),
                global_hidden,
                weight,
                labels_t,
                maximum_chunk,
                accumulate_chunk,
                num_valid_tokens[chunk_index],
                "none",
                ctx.ignore_index,
                ctx.tp_group,
                ctx.tp_rank,
                ctx.tp_world_size,
                True,
            )
            if hidden_grad is not None:
                hidden_grad[start:end] = hidden_grad_chunk
            if weight_grad is not None:
                weight_grad.add_(weight_grad_chunk)

        tensor_audit(
            "streambp_lce/sp_stream_backward_done",
            hidden_grad=hidden_grad,
            weight_grad=weight_grad,
            tile_size=ctx.tile_size,
            num_chunks=len(chunks),
        )
        return hidden_grad, weight_grad, None, None, None, None


def _streambp_fused_lce_available(
    output_layer: torch.nn.Module,
    hidden_states: Tensor,
    labels: Tensor,
    output_layer_kwargs: dict[str, Any],
    *,
    sequence_parallel_output: bool,
) -> bool:
    if not _env_flag("MEGATRON_STREAMBP_FUSED_LCE", default=True):
        return _log_streambp_fused_lce_rejection("disabled by MEGATRON_STREAMBP_FUSED_LCE=0")
    if _load_streambp_fused_lce() is None:
        return _log_streambp_fused_lce_rejection("Blackwell fused LCE extension is unavailable")
    weight = _streambp_lce_weight(output_layer, output_layer_kwargs)
    if weight is None:
        return _log_streambp_fused_lce_rejection("output weight is unavailable")
    if hidden_states.dim() != 3 or labels.dim() != 2 or weight.dim() != 2:
        return _log_streambp_fused_lce_rejection(
            "unexpected ranks: "
            f"hidden={tuple(hidden_states.shape)} labels={tuple(labels.shape)} "
            f"weight={tuple(weight.shape)}"
        )
    if not (hidden_states.is_cuda and labels.is_cuda and weight.is_cuda):
        return _log_streambp_fused_lce_rejection(
            "inputs are not all CUDA tensors: "
            f"hidden_cuda={hidden_states.is_cuda} labels_cuda={labels.is_cuda} "
            f"weight_cuda={weight.is_cuda}"
        )
    if hidden_states.device != labels.device or hidden_states.device != weight.device:
        return _log_streambp_fused_lce_rejection(
            "device mismatch: "
            f"hidden={hidden_states.device} labels={labels.device} weight={weight.device}"
        )
    if hidden_states.dtype != weight.dtype:
        return _log_streambp_fused_lce_rejection(
            f"dtype mismatch: hidden={hidden_states.dtype} weight={weight.dtype}"
        )
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        return _log_streambp_fused_lce_rejection(f"unsupported dtype: {hidden_states.dtype}")
    if (hidden_states.size(-1) * hidden_states.element_size()) % 128 != 0:
        return _log_streambp_fused_lce_rejection(
            f"hidden row bytes are not 128B aligned: {hidden_states.size(-1) * hidden_states.element_size()}"
        )
    if not weight.is_contiguous():
        return _log_streambp_fused_lce_rejection(
            f"output weight is not contiguous: stride={weight.stride()}"
        )
    if sequence_parallel_output:
        world_size = _tp_world_size(output_layer)
        if labels.size(1) != hidden_states.size(0) * world_size:
            return _log_streambp_fused_lce_rejection(
                "sequence-parallel label length mismatch: "
                f"labels_s={labels.size(1)} hidden_s={hidden_states.size(0)} tp={world_size}"
            )
        return True
    if labels.size(1) != hidden_states.size(0):
        return _log_streambp_fused_lce_rejection(
            f"label length mismatch: labels_s={labels.size(1)} hidden_s={hidden_states.size(0)}"
        )
    return True


def _log_streambp_fused_lce_rejection(reason: str) -> bool:
    global _STREAMBP_FUSED_LCE_REJECTION_LOGGED
    if (
        not _STREAMBP_FUSED_LCE_REJECTION_LOGGED
        and _env_flag("MEGATRON_STREAMBP_FUSED_LCE_LOG_REJECTIONS", default=True)
    ):
        _STREAMBP_FUSED_LCE_REJECTION_LOGGED = True
        print(f"StreamBP fused LCE unavailable: {reason}", flush=True)
    return False


def _streambp_fused_lce_loss(
    output_layer: torch.nn.Module,
    hidden_states: Tensor,
    labels: Tensor,
    output_layer_kwargs: dict[str, Any],
    *,
    sequence_parallel_output: bool,
) -> Tensor:
    linear_cross_entropy = _load_streambp_fused_lce()
    if linear_cross_entropy is None:
        raise RuntimeError("StreamBP fused linear cross entropy is not available")

    weight = _streambp_lce_weight(output_layer, output_layer_kwargs)
    if weight is None:
        raise RuntimeError("StreamBP fused linear cross entropy requires an output weight")
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not labels.is_contiguous():
        labels = labels.contiguous()

    global _STREAMBP_FUSED_LCE_LOGGED
    should_log = not _STREAMBP_FUSED_LCE_LOGGED
    tp_group = getattr(output_layer, "tp_group", None)
    if should_log and torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            if tp_group is not None and torch.distributed.get_rank(tp_group) != 0:
                should_log = False
        except Exception:
            pass
    if should_log:
        _STREAMBP_FUSED_LCE_LOGGED = True
        print(
            "StreamBP fused LCE active: "
            f"hidden={tuple(hidden_states.shape)} "
            f"weight={tuple(weight.shape)} "
            f"labels={tuple(labels.shape)} "
            f"sequence_parallel={sequence_parallel_output} "
            f"fwd_vocab_split={os.getenv('LCE_FWD_VOCAB_SPLIT_SIZE', '3072')} "
            f"bwd_vocab_split={os.getenv('LCE_BWD_VOCAB_SPLIT_SIZE', '3072')}",
            flush=True,
        )

    if sequence_parallel_output and _env_flag(
        "MEGATRON_STREAMBP_FUSED_LCE_SP_STREAMING", default=True
    ):
        tensor_audit(
            "streambp_lce/sp_stream_input",
            hidden=hidden_states,
            weight=weight,
            labels=labels,
            tile_size=_streambp_lce_sp_tile_size(hidden_states.size(0)),
        )
        return _StreamBPFusedSPLinearCrossEntropy.apply(
            hidden_states,
            weight,
            labels,
            getattr(output_layer, "tp_group", None),
            _streambp_lce_sp_tile_size(hidden_states.size(0)),
            -100,
        )

    tensor_audit(
        "streambp_lce/fused_input",
        hidden=hidden_states,
        weight=weight,
        labels=labels,
        sequence_parallel=sequence_parallel_output,
    )
    # The fused LCE kernel consumes sequence-major labels to match [S, B, H]
    # hidden order.  StreamBP keeps public loss tensors in [B, S].
    labels_t = labels.transpose(0, 1).contiguous()
    loss = linear_cross_entropy(
        hidden_states,
        weight,
        labels_t,
        tp_group=getattr(output_layer, "tp_group", None),
        reduction="none",
        ignore_index=-100,
        sequence_parallel=sequence_parallel_output,
    )
    loss_out = loss.view_as(labels_t).transpose(0, 1).contiguous()
    tensor_audit(
        "streambp_lce/fused_output",
        loss=loss_out,
        labels_t=labels_t,
        sequence_parallel=sequence_parallel_output,
    )
    return loss_out


def _tp_world_size(output_layer: torch.nn.Module) -> int:
    tp_group = getattr(output_layer, "tp_group", None)
    if tp_group is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(tp_group)


def _uses_sequence_parallel_output(
    output_layer: torch.nn.Module, hidden_states: Tensor, labels: Tensor
) -> bool:
    world_size = _tp_world_size(output_layer)
    return (
        bool(getattr(output_layer, "sequence_parallel", False))
        and world_size > 1
        and labels.dim() >= 2
        and labels.size(1) == hidden_states.size(0) * world_size
    )


def _rank_concatenated_sequence_chunk(
    tensor: Tensor,
    start: int,
    end: int,
    *,
    local_seq_len: int,
    world_size: int,
) -> Tensor:
    pieces = [
        tensor[:, rank * local_seq_len + start : rank * local_seq_len + end]
        for rank in range(world_size)
    ]
    return torch.cat(pieces, dim=1).contiguous()


def _scatter_rank_concatenated_sequence_chunk(
    destination: Tensor,
    chunk: Tensor,
    start: int,
    end: int,
    *,
    local_seq_len: int,
    world_size: int,
) -> None:
    offset = 0
    chunk_len = end - start
    for rank in range(world_size):
        global_start = rank * local_seq_len + start
        global_end = global_start + chunk_len
        destination[:, global_start:global_end] = chunk[:, offset : offset + chunk_len]
        offset += chunk_len


class _StreamBPLMHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        labels: Tensor,
        chunk_size: Optional[int],
        output_layer: torch.nn.Module,
        loss_func: Callable[[Tensor, Tensor], Tensor],
        output_layer_kwargs: dict[str, Any],
    ) -> Tensor:
        ctx.output_layer = output_layer
        ctx.loss_func = loss_func
        ctx.output_layer_kwargs = output_layer_kwargs
        ctx.chunk_size = chunk_size
        ctx.sequence_parallel_output = _uses_sequence_parallel_output(
            output_layer, hidden_states, labels
        )
        ctx.sequence_parallel_world_size = _tp_world_size(output_layer)
        ctx.use_fused_lce = _streambp_fused_lce_available(
            output_layer,
            hidden_states,
            labels,
            output_layer_kwargs,
            sequence_parallel_output=ctx.sequence_parallel_output,
        )
        ctx.save_for_backward(hidden_states, labels)
        tensor_audit(
            "streambp_lce/forward_start",
            hidden=hidden_states,
            labels=labels,
            chunk_size=chunk_size,
            sequence_parallel=ctx.sequence_parallel_output,
            fused_lce=ctx.use_fused_lce,
        )

        losses: list[Tensor] = []
        full_loss = None
        local_seq_len = hidden_states.size(0)
        with torch.no_grad():
            for start, end in iter_streambp_chunks(hidden_states.size(0), chunk_size):
                if ctx.sequence_parallel_output:
                    labels_chunk = _rank_concatenated_sequence_chunk(
                        labels,
                        start,
                        end,
                        local_seq_len=local_seq_len,
                        world_size=ctx.sequence_parallel_world_size,
                    )
                    if ctx.use_fused_lce:
                        loss_chunk = _streambp_fused_lce_loss(
                            output_layer,
                            hidden_states[start:end],
                            labels_chunk,
                            output_layer_kwargs,
                            sequence_parallel_output=True,
                        )
                    else:
                        logits = _call_output_layer(
                            output_layer, hidden_states[start:end], output_layer_kwargs
                        )
                        loss_chunk = loss_func(labels_chunk, logits)
                    if full_loss is None:
                        full_loss = torch.empty(
                            labels.shape, dtype=loss_chunk.dtype, device=loss_chunk.device
                        )
                        tensor_audit(
                            "streambp_lce/full_loss_alloc",
                            full_loss=full_loss,
                            loss_chunk=loss_chunk,
                            labels_chunk=labels_chunk,
                        )
                    _scatter_rank_concatenated_sequence_chunk(
                        full_loss,
                        loss_chunk,
                        start,
                        end,
                        local_seq_len=local_seq_len,
                        world_size=ctx.sequence_parallel_world_size,
                    )
                else:
                    labels_chunk = labels[:, start:end]
                    if ctx.use_fused_lce:
                        losses.append(
                            _streambp_fused_lce_loss(
                                output_layer,
                                hidden_states[start:end],
                                labels_chunk,
                                output_layer_kwargs,
                                sequence_parallel_output=False,
                            )
                        )
                    else:
                        logits = _call_output_layer(
                            output_layer, hidden_states[start:end], output_layer_kwargs
                        )
                        losses.append(loss_func(labels_chunk, logits))
        if ctx.sequence_parallel_output:
            assert full_loss is not None
            tensor_audit("streambp_lce/forward_done", full_loss=full_loss)
            return full_loss.contiguous()
        result = torch.cat(losses, dim=1).contiguous()
        tensor_audit("streambp_lce/forward_done", full_loss=result)
        return result

    @staticmethod
    def backward(ctx, grad_loss: Tensor):
        hidden_states, labels = ctx.saved_tensors
        chunks = iter_streambp_chunks(hidden_states.size(0), ctx.chunk_size)
        extra_weight = ctx.output_layer_kwargs.get("weight")
        params = _unique_trainable_parameters(ctx.output_layer, extra_tensors=(extra_weight,))
        marked = mark_streambp_pending_chunks(params, len(chunks))

        hidden_grad = torch.empty_like(hidden_states) if ctx.needs_input_grad[0] else None
        tensor_audit(
            "streambp_lce/backward_start",
            hidden=hidden_states,
            labels=labels,
            grad_loss=grad_loss,
            hidden_grad=hidden_grad,
            num_chunks=len(chunks),
            fused_lce=ctx.use_fused_lce,
        )
        local_seq_len = hidden_states.size(0)
        try:
            with torch.enable_grad():
                for start, end in chunks:
                    hidden_chunk = hidden_states[start:end].detach().requires_grad_(
                        ctx.needs_input_grad[0]
                    )
                    if ctx.sequence_parallel_output:
                        labels_chunk = _rank_concatenated_sequence_chunk(
                            labels,
                            start,
                            end,
                            local_seq_len=local_seq_len,
                            world_size=ctx.sequence_parallel_world_size,
                        )
                        grad_loss_chunk = _rank_concatenated_sequence_chunk(
                            grad_loss,
                            start,
                            end,
                            local_seq_len=local_seq_len,
                            world_size=ctx.sequence_parallel_world_size,
                        )
                    else:
                        labels_chunk = labels[:, start:end]
                        grad_loss_chunk = grad_loss[:, start:end]
                    if ctx.use_fused_lce:
                        loss = _streambp_fused_lce_loss(
                            ctx.output_layer,
                            hidden_chunk,
                            labels_chunk,
                            ctx.output_layer_kwargs,
                            sequence_parallel_output=ctx.sequence_parallel_output,
                        )
                    else:
                        logits = _call_output_layer(
                            ctx.output_layer, hidden_chunk, ctx.output_layer_kwargs
                        )
                        loss = ctx.loss_func(labels_chunk, logits)
                    tensor_audit(
                        "streambp_lce/backward_chunk",
                        hidden_chunk=hidden_chunk,
                        labels_chunk=labels_chunk,
                        grad_loss_chunk=grad_loss_chunk,
                        loss=loss,
                        chunk_start=start,
                        chunk_end=end,
                    )
                    _streambp_autograd_backward(loss, grad_loss_chunk)
                    if hidden_grad is not None:
                        hidden_grad[start:end] = hidden_chunk.grad
        finally:
            clear_streambp_pending_chunks(marked)

        return hidden_grad, None, None, None, None, None


def chunked_lm_head_loss(
    output_layer: torch.nn.Module,
    hidden_states: Tensor,
    labels: Tensor,
    *,
    loss_func: Callable[[Tensor, Tensor], Tensor],
    chunk_size: Optional[int],
    **output_layer_kwargs: Any,
) -> Tensor:
    """Compute GPT LM-head loss without materializing full-sequence logits.

    This helper is intentionally usable without layer StreamBP.  The fused
    sequence-parallel LCE path below is a memory-safe LM-head schedule, not a
    replay requirement.
    """
    chunks = iter_streambp_chunks(hidden_states.size(0), chunk_size)
    sequence_parallel_output = _uses_sequence_parallel_output(output_layer, hidden_states, labels)
    fused_lce_available = _streambp_fused_lce_available(
        output_layer,
        hidden_states,
        labels,
        output_layer_kwargs,
        sequence_parallel_output=sequence_parallel_output,
    )
    if fused_lce_available and len(chunks) <= 1:
        return _streambp_fused_lce_loss(
            output_layer,
            hidden_states,
            labels,
            output_layer_kwargs,
            sequence_parallel_output=sequence_parallel_output,
        )
    if len(chunks) <= 1 and torch.is_grad_enabled() and not fused_lce_available:
        fallback_chunk_size = int(
            os.getenv("MEGATRON_STREAMBP_LCE_FALLBACK_CHUNK_SIZE", "8192")
        )
        if 0 < fallback_chunk_size < hidden_states.size(0):
            chunk_size = fallback_chunk_size
            chunks = iter_streambp_chunks(hidden_states.size(0), chunk_size)
    if len(chunks) <= 1 or not torch.is_grad_enabled():
        logits = _call_output_layer(output_layer, hidden_states, output_layer_kwargs)
        return loss_func(labels, logits)
    return _StreamBPLMHeadLoss.apply(
        hidden_states, labels, chunk_size, output_layer, loss_func, output_layer_kwargs
    )


def streambp_lm_head_loss(
    output_layer: torch.nn.Module,
    hidden_states: Tensor,
    labels: Tensor,
    *,
    loss_func: Callable[[Tensor, Tensor], Tensor],
    chunk_size: Optional[int],
    **output_layer_kwargs: Any,
) -> Tensor:
    """Backward-compatible name for StreamBP callers."""
    return chunked_lm_head_loss(
        output_layer,
        hidden_states,
        labels,
        loss_func=loss_func,
        chunk_size=chunk_size,
        **output_layer_kwargs,
    )
