# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime helpers for ZeroBubble pipeline schedules.

This module intentionally stays inside ``megatron.core`` and does not depend on
``megatron.training`` args.  Runtime support is driven by model config and the
current process-group state.
"""

from __future__ import annotations

import queue
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

from megatron.core import parallel_state


WeightGradTask = Tuple[object, Callable[..., tuple], Callable[..., None]]


class WeightGradStore:
    """Queue deferred weight-gradient GEMMs for split B/W execution.

    Tensor-parallel layers enqueue fused weight-gradient work while the pipeline
    runtime executes the activation-gradient part of backward.  Later W runtime
    slots drain these queues without re-entering autograd.
    """

    _split_bw: bool = False
    _cache: List[WeightGradTask] = []
    _queues: Optional[List[List[queue.Queue]]] = None
    _num_chunks: int = 1
    _num_seq_splits: int = 1

    @classmethod
    def lazy_init(cls, num_chunks: Optional[int] = None, num_seq_splits: int = 1) -> None:
        if num_chunks is None:
            num_chunks = parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1
        num_chunks = max(1, int(num_chunks))
        num_seq_splits = max(1, int(num_seq_splits))
        if (
            cls._queues is not None
            and cls._num_chunks == num_chunks
            and cls._num_seq_splits == num_seq_splits
        ):
            return
        cls._num_chunks = num_chunks
        cls._num_seq_splits = num_seq_splits
        cls._queues = [[queue.Queue() for _ in range(num_seq_splits)] for _ in range(num_chunks)]

    @classmethod
    def reset(cls, num_chunks: Optional[int] = None, num_seq_splits: int = 1) -> None:
        cls._split_bw = False
        cls._cache = []
        cls._queues = None
        cls.lazy_init(num_chunks=num_chunks, num_seq_splits=num_seq_splits)

    @classmethod
    def split_bw(cls) -> bool:
        return cls._split_bw

    @classmethod
    def enable_split_bw(cls) -> None:
        cls._split_bw = True

    @classmethod
    def disable_split_bw(cls) -> None:
        cls._split_bw = False

    @classmethod
    @contextmanager
    def set_split_bw(cls, enabled: bool):
        previous = cls._split_bw
        cls._split_bw = enabled
        try:
            yield
        finally:
            cls._split_bw = previous

    @classmethod
    def put(cls, weight, pre_func: Callable[..., tuple], func: Callable[..., None]) -> None:
        if not cls._split_bw:
            raise RuntimeError("WeightGradStore.put() requires split B/W mode")
        cls._cache.append((weight, pre_func, func))

    @classmethod
    def flush(cls, chunk: int = 0, seq_split_idx: int = 0) -> None:
        cls.lazy_init()
        if not cls._split_bw:
            if cls._cache:
                raise RuntimeError("WeightGradStore cache is not empty outside split B/W mode")
            return
        cls._queues[chunk][seq_split_idx].put(cls._cache)
        cls._cache = []

    @classmethod
    def queue_size(cls, chunk: int = 0, seq_split_idx: int = 0) -> int:
        cls.lazy_init()
        return cls._queues[chunk][seq_split_idx].qsize()

    @classmethod
    def pop(cls, chunk: int = 0, seq_split_idx: int = 0) -> None:
        cls.lazy_init()
        if cls._queues[chunk][seq_split_idx].empty():
            rank = parallel_state.get_pipeline_model_parallel_rank()
            raise RuntimeError(f"WeightGradStore queue is empty on pipeline rank {rank}")
        for _, pre_func, func in cls._queues[chunk][seq_split_idx].get():
            func(*pre_func(async_op=False))

    @classmethod
    def clear(cls, chunk: int = 0, seq_split_idx: int = 0) -> None:
        cls.lazy_init()
        while not cls._queues[chunk][seq_split_idx].empty():
            for _, pre_func, func in cls._queues[chunk][seq_split_idx].get():
                func(*pre_func(async_op=False))

    @classmethod
    def assert_empty(cls) -> None:
        rank = parallel_state.get_pipeline_model_parallel_rank()
        if cls._cache:
            raise AssertionError(f"WeightGradStore cache is not empty on pipeline rank {rank}")
        if cls._queues is None:
            return
        for chunk, chunk_queues in enumerate(cls._queues):
            for seq, seq_queue in enumerate(chunk_queues):
                if not seq_queue.empty():
                    raise AssertionError(
                        "WeightGradStore queue is not empty "
                        f"on pipeline rank {rank}, chunk {chunk}, seq {seq}: {seq_queue.qsize()}"
                    )


class RecomputeStore:
    """Placeholder-compatible recompute store for reference runtime imports."""

    @classmethod
    def assert_empty(cls) -> None:
        return
