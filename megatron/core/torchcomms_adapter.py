# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import contextlib
import importlib
import os
from types import ModuleType
from typing import Any

import torch


_ACTIVE = False
_PATCHED = False
_GROUP_CREATE_INDEX = 0
_EXTRA_GROUPS: list[Any] = []
_MEM_POOLS: list[Any] = []
_NATIVE_GROUPS: list[Any] = []
_ORIGINAL_DISTRIBUTED: dict[str, Any] = {}
_ORIGINAL_MEGATRON_COLLECTIVES: dict[str, Any] = {}

_COLLECTIVE_NAMES = (
    "_make_nccl_premul_sum",
    "all_gather",
    "all_gather_into_tensor",
    "all_gather_object",
    "all_reduce",
    "all_to_all",
    "all_to_all_single",
    "barrier",
    "batch_isend_irecv",
    "broadcast",
    "broadcast_object_list",
    "gather",
    "gather_object",
    "irecv",
    "isend",
    "recv",
    "reduce",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "scatter",
    "scatter_object_list",
    "send",
)


def get_distributed_module(args: Any) -> ModuleType:
    if use_torchcomms(args):
        return _import_distwrap()
    return torch.distributed


def use_torchcomms(args: Any | None = None) -> bool:
    if args is not None:
        backend = getattr(args, "distributed_backend", None)
        if backend == "ncclx":
            return True
        if bool(getattr(args, "use_torchcomms", False)):
            return True
    return _env_flag("MEGATRON_USE_TORCHCOMMS")


def is_torchcomms_active() -> bool:
    return _ACTIVE


def ncclx_mem_pool_enabled(args: Any | None = None) -> bool:
    if args is not None and hasattr(args, "ncclx_mem_pool"):
        return bool(getattr(args, "ncclx_mem_pool"))
    return _env_flag("MEGATRON_NCCLX_MEM_POOL")


def ncclx_persist_ag_enabled(args: Any | None = None) -> bool:
    if args is not None and hasattr(args, "ncclx_persist_ag"):
        return bool(getattr(args, "ncclx_persist_ag"))
    return _env_flag("MEGATRON_NCCLX_PERSIST_AG")


def ncclx_ft_enabled(args: Any | None = None) -> bool:
    if args is not None and hasattr(args, "ncclx_ft"):
        return bool(getattr(args, "ncclx_ft"))
    return _env_flag("MEGATRON_NCCLX_FT")


def ncclx_rdma_enabled(args: Any | None = None) -> bool:
    if args is not None and hasattr(args, "ncclx_rdma"):
        return bool(getattr(args, "ncclx_rdma"))
    return _env_flag("MEGATRON_NCCLX_RDMA")


def ncclx_rdma_backends(args: Any | None = None) -> str:
    if args is not None and getattr(args, "ncclx_rdma_backends", None):
        return _normalize_ctran_backends(str(getattr(args, "ncclx_rdma_backends")))
    env_backends = os.getenv("MEGATRON_NCCLX_RDMA_BACKENDS")
    if env_backends:
        return _normalize_ctran_backends(env_backends)
    return "ib,nvl,socket"


def ncclx_rdma_profile(args: Any | None = None) -> str:
    profile = _string_arg_or_env(
        args,
        "ncclx_rdma_profile",
        "MEGATRON_NCCLX_RDMA_PROFILE",
        default="ib",
    )
    return profile.lower()


def configure_ncclx_transport(args: Any | None = None) -> None:
    if not ncclx_rdma_enabled(args):
        return
    os.environ.setdefault("NCCL_CTRAN_ENABLE", "1")
    os.environ.setdefault("NCCL_CTRAN_BACKENDS", ncclx_rdma_backends(args))
    _configure_ncclx_ib_transport(args)


def _configure_ncclx_ib_transport(args: Any | None) -> None:
    backends = {backend.strip().upper() for backend in ncclx_rdma_backends(args).split(",")}
    if "IB" not in backends and ncclx_rdma_profile(args) not in {"roce", "rocev2"}:
        return

    roce_gid_index = _string_arg_or_env(
        args,
        "ncclx_roce_gid_index",
        "MEGATRON_NCCLX_ROCE_GID_INDEX",
        "MEGATRON_NCCLX_RDMA_GID_INDEX",
    )
    if roce_gid_index is not None:
        os.environ.setdefault("NCCL_IB_GID_INDEX", roce_gid_index)

    address_family = _string_arg_or_env(
        args,
        "ncclx_roce_addr_family",
        "MEGATRON_NCCLX_ROCE_ADDR_FAMILY",
        "MEGATRON_NCCLX_RDMA_ADDR_FAMILY",
    )
    if address_family is not None:
        os.environ.setdefault("NCCL_IB_ADDR_FAMILY", _normalize_ib_addr_family(address_family))

    ib_hca = _string_arg_or_env(
        args,
        "ncclx_ib_hca",
        "MEGATRON_NCCLX_IB_HCA",
        "MEGATRON_NCCLX_ROCE_HCA",
    )
    if ib_hca is not None:
        os.environ.setdefault("NCCL_IB_HCA", ib_hca)


def _normalize_ib_addr_family(value: str) -> str:
    normalized = value.strip().upper()
    aliases = {
        "4": "IPV4",
        "6": "IPV6",
        "AF_INET": "IPV4",
        "AF_INET6": "IPV6",
    }
    return aliases.get(normalized, normalized)


def _normalize_ctran_backends(value: str) -> str:
    normalized = ",".join(
        backend.strip().lower() for backend in value.split(",") if backend.strip()
    )
    return normalized or "ib"


def ncclx_backend_string(args: Any | None = None) -> str:
    del args
    return os.getenv("MEGATRON_NCCLX_BACKEND_STRING", "ncclx")


def init_process_group(args: Any, **kwargs: Any) -> None:
    if not use_torchcomms(args) or kwargs.get("backend") == "fake":
        torch.distributed.init_process_group(**kwargs)
        return

    if getattr(args, "distributed_backend", None) != "ncclx":
        raise RuntimeError("TorchComms integration requires --distributed-backend ncclx")

    _raise_nofile_limit()
    configure_ncclx_transport(args)
    _configure_torchcomm_timeout(args, kwargs.get("timeout"))
    _configure_object_collective_serialization()
    distwrap = _import_distwrap()
    kwargs["backend"] = ncclx_backend_string(args)
    kwargs["use_torchcomms"] = True
    distwrap.init_process_group(**kwargs)
    _install_torch_distributed_bridge(distwrap)

    global _ACTIVE
    _ACTIVE = True


def create_group(
    *,
    ranks: list[int] | None,
    timeout: Any = None,
    backend: str | None = None,
    pg_options: Any = None,
    use_local_synchronization: bool = False,
    group_desc: str | None = None,
) -> Any:
    if not is_torchcomms_active():
        return torch.distributed.new_group(
            ranks=ranks,
            timeout=timeout,
            backend=backend,
            pg_options=pg_options,
            use_local_synchronization=use_local_synchronization,
            group_desc=group_desc,
        )

    world_size = torch.distributed.get_world_size()
    world_ranks = list(range(world_size))
    target_ranks = world_ranks if ranks is None else list(ranks)

    if len(target_ranks) <= 1:
        _debug_group("native-singleton", group_desc, target_ranks)
        group = torch.distributed.new_group(
            ranks=target_ranks,
            timeout=timeout,
            backend=backend,
            pg_options=pg_options,
            use_local_synchronization=use_local_synchronization,
            group_desc=group_desc,
        )
        _NATIVE_GROUPS.append(group)
        return group

    _debug_group("ncclx-split-start", group_desc, target_ranks)
    group = _create_registered_torchcomms_group(
        ranks=target_ranks,
        timeout=timeout,
        backend=backend or ncclx_backend_string(),
        pg_options=pg_options,
        use_local_synchronization=use_local_synchronization,
        group_desc=group_desc,
    )
    _debug_group("ncclx-split-done", group_desc, target_ranks)
    return group


def _create_registered_torchcomms_group(
    *,
    ranks: list[int],
    timeout: Any = None,
    backend: str,
    pg_options: Any = None,
    use_local_synchronization: bool = False,
    group_desc: str | None = None,
) -> Any:
    from torchcomms.distwrap.pginfo import (
        pg_info_create,
        pg_info_get_data,
        pg_info_set_data,
    )
    from torchcomms.distwrap.utils import (
        _BACKEND_RENAME_FOR_DIST,
        _pg_options_to_hints,
        format_backend_string,
        parse_backend_string,
    )

    global _GROUP_CREATE_INDEX
    _GROUP_CREATE_INDEX += 1
    split_name = f"{group_desc or 'megatron_group'}:{_GROUP_CREATE_INDEX}"

    device_backends = parse_backend_string(backend)
    dist_device_backends = {
        device: _BACKEND_RENAME_FOR_DIST.get(device_backend, device_backend)
        for device, device_backend in device_backends.items()
    }
    dist_backend = format_backend_string(dist_device_backends)
    group = torch.distributed.new_group(
        ranks=ranks,
        timeout=timeout,
        backend=dist_backend,
        pg_options=pg_options,
        use_local_synchronization=use_local_synchronization,
        group_desc=group_desc,
    )

    world_group = torch.distributed.group.WORLD
    if world_group is None:
        raise AssertionError("World process group is not initialized")
    parent_torchcomms = pg_info_get_data(world_group, "torchcomms")
    my_rank = torch.distributed.get_rank()
    split_ranks = ranks if my_rank in ranks else []
    hints = _pg_options_to_hints(pg_options)

    torchcomms_instances: dict[str, Any] = {}
    for device_type, parent_tc in parent_torchcomms.items():
        if device_type not in device_backends:
            continue
        split_tc = parent_tc.split(split_ranks, name=split_name, hints=hints)
        if split_tc is not None:
            torchcomms_instances[device_type] = split_tc

    if my_rank not in ranks:
        return torch.distributed.GroupMember.NON_GROUP_MEMBER
    if group is torch.distributed.GroupMember.NON_GROUP_MEMBER:
        raise AssertionError(f"Rank {my_rank} expected to be in group {ranks}")

    pg_info_create(group, ranks, group_desc)
    pg_info_set_data(group, "device_backends", device_backends)
    pg_info_set_data(group, "torchcomms", torchcomms_instances)
    return group


def _debug_group(action: str, group_desc: str | None, ranks: list[int]) -> None:
    if not _env_flag("MEGATRON_TORCHCOMMS_GROUP_DEBUG"):
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    if rank == 0 or rank in ranks:
        print(
            f"[torchcomms-group] action={action} rank={rank} "
            f"desc={group_desc or ''} size={len(ranks)} ranks={ranks}",
            flush=True,
        )


def cuda_mem_pool_context(group: Any | None) -> Any:
    if (
        not is_torchcomms_active()
        or not ncclx_mem_pool_enabled()
        or not torch.cuda.is_available()
        or not hasattr(torch.cuda, "MemPool")
    ):
        return contextlib.nullcontext()

    distwrap = _import_distwrap()
    device = torch.device("cuda", torch.cuda.current_device())
    allocator = distwrap.get_mem_allocator(group, device)
    pool = torch.cuda.MemPool(allocator)
    _MEM_POOLS.append(pool)
    return torch.cuda.use_mem_pool(pool)


def get_torchcomm(
    group: Any | None = None,
    *,
    device_type: str | None = None,
    tensor: torch.Tensor | None = None,
) -> Any:
    if not is_torchcomms_active():
        raise RuntimeError("TorchComms is not active")
    from torchcomms.distwrap.utils import get_group, get_torchcomms_instance

    pg = get_group(group)
    return get_torchcomms_instance(pg, device_type=device_type, tensor=tensor)


def _install_torch_distributed_bridge(distwrap: ModuleType) -> None:
    global _PATCHED
    if _PATCHED:
        return
    for name in _COLLECTIVE_NAMES:
        if hasattr(distwrap, name):
            _ORIGINAL_DISTRIBUTED.setdefault(name, getattr(torch.distributed, name, None))
            if name == "all_to_all_single":
                setattr(torch.distributed, name, _all_to_all_single)
            elif name == "batch_isend_irecv":
                setattr(torch.distributed, name, _batch_isend_irecv)
            else:
                setattr(torch.distributed, name, getattr(distwrap, name))
    _ORIGINAL_DISTRIBUTED.setdefault("P2POp", getattr(torch.distributed, "P2POp", None))
    torch.distributed.P2POp = _p2p_op
    _ORIGINAL_DISTRIBUTED.setdefault(
        "destroy_process_group",
        torch.distributed.destroy_process_group,
    )
    torch.distributed.destroy_process_group = _destroy_process_group
    _patch_cached_megatron_collectives(distwrap)
    _PATCHED = True


def _destroy_process_group(group: Any | None = None) -> None:
    distwrap = _import_distwrap()
    original = _ORIGINAL_DISTRIBUTED["destroy_process_group"]
    if group is not None and any(group is native_group for native_group in _NATIVE_GROUPS):
        _NATIVE_GROUPS.remove(group)
        torch.distributed.destroy_process_group = original
        try:
            original(group)
        finally:
            torch.distributed.destroy_process_group = _destroy_process_group
        return

    torch.distributed.destroy_process_group = original
    try:
        distwrap.destroy_process_group(group)
    finally:
        if group is None:
            _restore_torch_distributed_bridge()
        else:
            torch.distributed.destroy_process_group = _destroy_process_group


def _patch_cached_megatron_collectives(distwrap: ModuleType) -> None:
    try:
        param_and_grad_buffer = importlib.import_module(
            "megatron.core.distributed.param_and_grad_buffer"
        )
    except ImportError:
        return
    if hasattr(distwrap, "all_gather_into_tensor"):
        _ORIGINAL_MEGATRON_COLLECTIVES.setdefault(
            "dist_all_gather_func", param_and_grad_buffer.dist_all_gather_func
        )
        param_and_grad_buffer.dist_all_gather_func = distwrap.all_gather_into_tensor
    if hasattr(distwrap, "reduce_scatter_tensor"):
        _ORIGINAL_MEGATRON_COLLECTIVES.setdefault(
            "dist_reduce_scatter_func", param_and_grad_buffer.dist_reduce_scatter_func
        )
        param_and_grad_buffer.dist_reduce_scatter_func = distwrap.reduce_scatter_tensor


def _restore_torch_distributed_bridge() -> None:
    global _ACTIVE, _PATCHED, _GROUP_CREATE_INDEX
    for name, original in _ORIGINAL_DISTRIBUTED.items():
        if original is None:
            if hasattr(torch.distributed, name):
                delattr(torch.distributed, name)
        else:
            setattr(torch.distributed, name, original)
    _ORIGINAL_DISTRIBUTED.clear()
    _restore_cached_megatron_collectives()
    _EXTRA_GROUPS.clear()
    _MEM_POOLS.clear()
    _NATIVE_GROUPS.clear()
    _ACTIVE = False
    _PATCHED = False
    _GROUP_CREATE_INDEX = 0


def _restore_cached_megatron_collectives() -> None:
    if not _ORIGINAL_MEGATRON_COLLECTIVES:
        return
    try:
        param_and_grad_buffer = importlib.import_module(
            "megatron.core.distributed.param_and_grad_buffer"
        )
    except ImportError:
        _ORIGINAL_MEGATRON_COLLECTIVES.clear()
        return
    for name, original in _ORIGINAL_MEGATRON_COLLECTIVES.items():
        setattr(param_and_grad_buffer, name, original)
    _ORIGINAL_MEGATRON_COLLECTIVES.clear()


def _all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: Any = None,
    input_split_sizes: Any = None,
    group: Any | None = None,
    async_op: bool = False,
) -> Any:
    distwrap = _import_distwrap()
    return distwrap.all_to_all_single(
        output,
        input,
        output_split_sizes=_normalize_split_sizes(output_split_sizes),
        input_split_sizes=_normalize_split_sizes(input_split_sizes),
        group=group,
        async_op=async_op,
    )


def _normalize_split_sizes(split_sizes: Any) -> Any:
    if split_sizes is None or isinstance(split_sizes, list):
        return split_sizes
    if isinstance(split_sizes, tuple):
        return list(split_sizes)
    tolist = getattr(split_sizes, "tolist", None)
    if tolist is not None:
        values = tolist()
        if isinstance(values, list):
            return values
    return list(split_sizes)


def _p2p_op(
    op: Any,
    tensor: torch.Tensor,
    peer: int | None = None,
    group: Any | None = None,
    tag: int = 0,
    group_peer: int | None = None,
) -> Any:
    original_p2p_op = _ORIGINAL_DISTRIBUTED.get("P2POp")
    if original_p2p_op is None:
        raise RuntimeError("Original torch.distributed.P2POp is not available")

    distwrap = _import_distwrap()
    op_name = getattr(op, "__name__", "")
    if op is getattr(distwrap, "isend", None) or op_name == "isend":
        op = torch.distributed.distributed_c10d.isend
    elif op is getattr(distwrap, "irecv", None) or op_name == "irecv":
        op = torch.distributed.distributed_c10d.irecv

    return original_p2p_op(
        op=op,
        tensor=tensor,
        peer=peer,
        group=group,
        tag=tag,
        group_peer=group_peer,
    )


def _batch_isend_irecv(p2p_op_list: list[Any]) -> list[Any]:
    if not p2p_op_list:
        return []

    if not is_torchcomms_active():
        original = _ORIGINAL_DISTRIBUTED.get("batch_isend_irecv")
        if original is None:
            raise RuntimeError(
                "Original torch.distributed.batch_isend_irecv is not available"
            )
        unwrapped_ops = [getattr(op, "_p2p_op", op) for op in p2p_op_list]
        works = original(unwrapped_ops)
        if works is None:
            raise AssertionError("torch.distributed.batch_isend_irecv returned None")
        return works

    from torchcomms.distwrap.pginfo import pg_info_assert_registered
    from torchcomms.distwrap.utils import (
        get_group,
        get_group_rank,
        get_torchcomms_instance,
    )

    batches: list[dict[str, Any]] = []
    op_batch_indices: list[int] = []

    for p2p_op in p2p_op_list:
        actual_op = getattr(p2p_op, "_p2p_op", p2p_op)
        pg = get_group(actual_op.group)
        pg_info_assert_registered(pg)

        group_peer = actual_op.group_peer
        if group_peer is None:
            if actual_op.peer is None:
                raise ValueError("TorchComms does not support wildcard P2P operations")
            group_peer = get_group_rank(pg, actual_op.peer)

        tc = get_torchcomms_instance(pg, tensor=actual_op.tensor)
        batch_index = _find_torchcomm_batch(batches, tc)
        if batch_index is None:
            batch_index = len(batches)
            batches.append({"torchcomm": tc, "batch_op": tc.batch_op_create()})

        op_name = getattr(actual_op.op, "__name__", "")
        batch_op = batches[batch_index]["batch_op"]
        if "send" in op_name:
            batch_op.send(actual_op.tensor, group_peer)
        elif "recv" in op_name:
            batch_op.recv(actual_op.tensor, group_peer)
        else:
            raise ValueError(f"Unknown P2P operation: {op_name}")
        op_batch_indices.append(batch_index)

    batch_works = [batch["batch_op"].issue(True) for batch in batches]
    return [batch_works[index] for index in op_batch_indices]


def _find_torchcomm_batch(batches: list[dict[str, Any]], torchcomm: Any) -> int | None:
    for index, batch in enumerate(batches):
        if batch["torchcomm"] is torchcomm:
            return index
    return None


def _configure_object_collective_serialization() -> None:
    # Megatron exchanges checkpoint metadata as ShardedTensor/ShardedObject
    # instances. TorchComms defaults object deserialization to torch.load's
    # weights_only path, which rejects those trusted metadata classes.
    os.environ.setdefault("TORCHCOMMS_SERIALIZATION", "pickle")
    try:
        from torchcomms import objcol
    except ImportError:
        return
    cache_clear = getattr(objcol._get_serialization, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def _raise_nofile_limit() -> None:
    try:
        import resource
    except ImportError:
        return

    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return

    if hard_limit == resource.RLIM_INFINITY or soft_limit >= hard_limit:
        return

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))
    except (OSError, ValueError):
        return


def _configure_torchcomm_timeout(args: Any | None, timeout: Any | None) -> None:
    if os.getenv("TORCHCOMM_TIMEOUT_SECONDS"):
        return

    seconds = _timeout_to_seconds(timeout)
    if seconds is None and args is not None:
        minutes = getattr(args, "distributed_timeout_minutes", None)
        if minutes is not None:
            try:
                seconds = max(1, int(float(minutes) * 60))
            except (TypeError, ValueError):
                seconds = None

    if seconds is not None:
        os.environ["TORCHCOMM_TIMEOUT_SECONDS"] = str(seconds)


def _timeout_to_seconds(timeout: Any | None) -> int | None:
    if timeout is None:
        return None

    total_seconds = getattr(timeout, "total_seconds", None)
    if callable(total_seconds):
        return max(1, int(total_seconds() + 0.999))

    try:
        return max(1, int(float(timeout)))
    except (TypeError, ValueError):
        return None


def _import_distwrap() -> ModuleType:
    try:
        from torchcomms import distwrap
    except ImportError as exc:
        raise RuntimeError(
            "NCCLX mode requires torchcomms with the ncclx backend installed. "
            "Install torchcomms in the active environment or use --distributed-backend nccl."
        ) from exc
    return distwrap


def _string_arg_or_env(
    args: Any | None,
    attr: str,
    *env_names: str,
    default: str | None = None,
) -> str | None:
    if args is not None and getattr(args, attr, None):
        return str(getattr(args, attr))
    for name in env_names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
