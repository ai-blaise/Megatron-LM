# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import Any

import torch

from megatron.core import torchcomms_adapter


def _wait(work: Any) -> None:
    if work is None:
        return
    wait_blocking = getattr(work, "wait_blocking", None)
    if wait_blocking is not None:
        wait_blocking()
    else:
        work.wait()


class NCCLXRMAWindow:
    """TorchComms window wrapper for NCCLX/CTran CUDA RMA operations."""

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        group: Any | None = None,
        comm: Any | None = None,
        owning: bool = True,
        collective_barrier: bool = True,
        require_cuda: bool = True,
        keepalive: tuple[Any, ...] = (),
    ) -> None:
        if require_cuda and not tensor.is_cuda:
            raise ValueError("NCCLX RMA windows require a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError("NCCLX RMA windows require a contiguous tensor")
        if comm is None:
            comm = torchcomms_adapter.get_torchcomm(group, tensor=tensor)
        get_backend = getattr(comm, "get_backend", None)
        if get_backend is not None and get_backend() != "ncclx":
            raise RuntimeError(f"NCCLX RMA requires ncclx backend, got {get_backend()}")

        self.comm = comm
        self.tensor = tensor
        self._keepalive = keepalive
        self._closed = False
        if collective_barrier:
            _wait(comm.barrier(False))
        self.window = comm.new_window()
        self.window.tensor_register(tensor, owning)
        if collective_barrier:
            _wait(comm.barrier(False))

    @classmethod
    def allocate(
        cls,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        group: Any | None = None,
        device: torch.device | str | int | None = None,
        fill_value: float | int | None = None,
    ) -> "NCCLXRMAWindow":
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = (
                torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
            )
        comm = torchcomms_adapter.get_torchcomm(group, device_type="cuda")
        allocator = getattr(comm, "mem_allocator", None)
        if allocator is None:
            import torchcomms

            allocator = torchcomms.get_mem_allocator(comm.get_backend())
        pool = torch.cuda.MemPool(allocator)
        with torch.cuda.use_mem_pool(pool):
            tensor = torch.empty(shape, dtype=dtype, device=device)
            if fill_value is not None:
                tensor.fill_(fill_value)
        return cls(tensor, group=group, comm=comm, keepalive=(pool,))

    def put(
        self,
        tensor: torch.Tensor,
        *,
        dst_rank: int,
        target_offset_elems: int = 0,
        async_op: bool = True,
        hints: dict[str, str] | None = None,
        timeout: Any | None = None,
    ) -> Any:
        if not tensor.is_contiguous():
            raise ValueError("NCCLX RMA put source must be contiguous")
        if hints is None and timeout is None:
            return self.window.put(tensor, dst_rank, target_offset_elems, async_op)
        return self.window.put(
            tensor,
            dst_rank,
            target_offset_elems,
            async_op,
            hints,
            timeout,
        )

    def signal(
        self,
        peer_rank: int,
        *,
        async_op: bool = True,
        hints: dict[str, str] | None = None,
        timeout: Any | None = None,
    ) -> Any:
        if hints is None and timeout is None:
            return self.window.signal(peer_rank, async_op)
        return self.window.signal(peer_rank, async_op, hints, timeout)

    def wait_signal(
        self,
        peer_rank: int,
        *,
        async_op: bool = True,
        hints: dict[str, str] | None = None,
        timeout: Any | None = None,
    ) -> Any:
        if hints is None and timeout is None:
            return self.window.wait_signal(peer_rank, async_op)
        return self.window.wait_signal(peer_rank, async_op, hints, timeout)

    def map_rank(self, rank: int) -> torch.Tensor:
        return self.window.map_remote_tensor(rank)

    def get_attr(self, peer_rank: int) -> Any:
        return self.window.get_attr(peer_rank)

    def get_device_window(
        self,
        *,
        signal_count: int = -1,
        counter_count: int = -1,
        barrier_count: int = 1,
    ) -> int:
        return self.window.get_device_window(
            signal_count,
            counter_count,
            barrier_count,
        )

    def register_local_buffer(self, tensor: torch.Tensor) -> Any:
        if not tensor.is_contiguous():
            raise ValueError(
                "NCCLX RMA local buffer registration requires a contiguous tensor"
            )
        return self.window.register_local_buffer(tensor)

    def deregister_local_buffer(self, handle: Any) -> None:
        self.window.deregister_local_buffer(handle)

    @property
    def size_bytes(self) -> int:
        return self.window.get_size()

    @property
    def registered_tensor(self) -> torch.Tensor | None:
        get_tensor = getattr(self.window, "get_tensor", None)
        if get_tensor is None:
            return self.tensor
        return get_tensor()

    def put_then_signal(
        self,
        tensor: torch.Tensor,
        *,
        dst_rank: int,
        target_offset_elems: int = 0,
    ) -> None:
        _wait(
            self.put(
                tensor,
                dst_rank=dst_rank,
                target_offset_elems=target_offset_elems,
                async_op=True,
            )
        )
        _wait(self.signal(dst_rank, async_op=True))

    def wait(self, peer_rank: int) -> None:
        _wait(self.wait_signal(peer_rank, async_op=True))

    def close(self) -> None:
        if self._closed:
            return
        self.window.tensor_deregister()
        self._closed = True

    def __enter__(self) -> "NCCLXRMAWindow":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


def recover_tensor_from_peer(
    *,
    donor_rank: int,
    receiver_rank: int,
    tensor: torch.Tensor,
    group: Any | None = None,
) -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return False
    if not tensor.is_cuda:
        raise ValueError("NCCLX RDMA peer recovery requires a CUDA tensor")

    rank = torch.distributed.get_rank()
    donor_comm_rank = _group_rank(donor_rank, group)
    receiver_comm_rank = _group_rank(receiver_rank, group)
    with NCCLXRMAWindow(tensor, group=group) as window:
        if rank == donor_rank:
            window.put_then_signal(
                tensor,
                dst_rank=receiver_comm_rank,
                target_offset_elems=0,
            )
        elif rank == receiver_rank:
            window.wait(donor_comm_rank)
        _wait(window.comm.barrier(False))
    return rank in (donor_rank, receiver_rank)


def _group_rank(global_rank: int, group: Any | None) -> int:
    if group is None:
        return global_rank
    get_group_rank = getattr(torch.distributed, "get_group_rank", None)
    if get_group_rank is None:
        return global_rank
    return get_group_rank(group, global_rank)
