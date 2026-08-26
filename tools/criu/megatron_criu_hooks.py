# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CRIU snapshot hooks for flashtraining Megatron workers.

Thin glue over the framework-neutral ``criu_snapshot_hooks`` package
(``ai-blaise/criu-snapshots``, hook contract v1). The package owns the
control-directory protocol, the trigger mechanisms, and the
``torch.distributed`` teardown/rebuild; this module owns only the
Megatron residue, which runs inside the ``on_quiesce`` / ``on_resume``
callbacks:

* completing the distributed-optimizer collectives that overlap the step
  (grad reduce / param gather through the DDP grad buffers),
* capturing dataloader progress (``iteration``, consumed-sample
  counters) into the artifact manifest,
* the post-rebuild collective warm-up, and
* the legacy ``MEGATRON_CRIU_STATE_DIR`` acknowledgment files that
  ``megatron-criu-entrypoint.sh`` and pre-contract agents still watch.

Arming is unchanged: ``sitecustomize.py`` calls :func:`install` at
interpreter startup when ``MEGATRON_CRIU_ENABLE=1``. That is before the
model exists, so steady state is declared separately: the training loop
calls :func:`mark_ready` at the first step boundary after the model,
optimizer, and dataloaders are built, which writes the contract's
``ready-for-checkpoint`` file (and registers the objects the quiesce
residue flushes). With the package present, both the signal mechanism
(contract defaults ``SIGRTMIN+5``/``+6``) and the signal-file mechanism
are active, and callbacks run on mechanism threads — never in the signal
frame, which is what made in-handler CUDA calls wedge ``cuda-checkpoint
lock`` on B200.

Without the package, the pre-contract inline path below runs byte-for-
byte as before.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from collections.abc import Sequence
from typing import Any, Final

try:
    import criu_snapshot_hooks
    from criu_snapshot_hooks import torch_dist as _torch_dist
except ImportError:
    criu_snapshot_hooks = None  # type: ignore[assignment]
    _torch_dist = None  # type: ignore[assignment]

NEUTRAL_PACKAGE: Final[bool] = criu_snapshot_hooks is not None

logger = logging.getLogger(__name__)

STATE_DIR = os.environ.get("MEGATRON_CRIU_STATE_DIR", "/var/run/megatron-criu")
READY_FILE = os.path.join(STATE_DIR, "pre_snapshot.ready")
READY_ERR_FILE = READY_FILE + ".err"
RESUME_FILE = os.path.join(STATE_DIR, "post_restore.done")

# Contract defaults are numeric Linux SIGRTMIN+5/+6; macOS has no
# SIGRTMIN, so fall back to the raw contract values there (dev/test only).
_SIGRTMIN: Final[int | None] = getattr(signal, "SIGRTMIN", None)
PRE_SNAPSHOT_SIGNAL: Final[int] = int(_SIGRTMIN) + 5 if _SIGRTMIN is not None else 39
POST_RESTORE_SIGNAL: Final[int] = int(_SIGRTMIN) + 6 if _SIGRTMIN is not None else 40

_installed = False
_snapshot_state: dict[str, Any] = {}

_registered_model_chunks: list[Any] = []
_registered_optimizer: Any | None = None

_hooks: Any | None = None
_signal_mechanism: Any | None = None
_signal_file_mechanism: Any | None = None
_ready_marked = False


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _write_json(path: str, payload: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
    os.replace(tmp, path)


def _write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _clear_state_files() -> None:
    for path in (READY_FILE, READY_ERR_FILE, RESUME_FILE):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _torch_loaded() -> bool:
    return "torch" in sys.modules


def _cuda_synchronize() -> None:
    if not _torch_loaded():
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --- Megatron residue (neutral-package path) ---------------------------------


def _flush_optimizer() -> None:
    """Complete the optimizer collectives that overlap the training step.

    Megatron's distributed optimizer dispatches its grad reduce-scatter /
    param all-gather through the DDP chunks' shared param-and-grad
    buffers, so ``finish_grad_sync()`` on each registered chunk is the
    flush point; there is no separate flush surface on
    ``MegatronOptimizer`` itself. Contract invariants 1–2 require these
    collectives to be finished before teardown.
    """
    for chunk in _registered_model_chunks:
        finish = getattr(chunk, "finish_grad_sync", None)
        if callable(finish):
            finish()


def _capture_dataloader_state() -> dict[str, Any] | None:
    """Return Megatron dataloader progress counters, or None off-training.

    ``consumed_train_samples`` (plus ``iteration``) is the canonical
    dataloader state in Megatron — the samplers are pure functions of it —
    and CRIU preserves the live iterators in process memory, so the
    counters are captured as artifact-manifest metadata, not for rebuild.
    """
    global_vars = sys.modules.get("megatron.training.global_vars")
    if global_vars is None:
        return None
    try:
        args = global_vars.get_args()
    except (AssertionError, AttributeError):
        logger.info("megatron args are not initialized; skipping dataloader capture")
        return None
    state = {
        name: getattr(args, name)
        for name in ("iteration", "consumed_train_samples", "consumed_valid_samples")
        if getattr(args, name, None) is not None
    }
    return state or None


def _capture_distributed_state() -> dict[str, Any] | None:
    """Snapshot rendezvous material before teardown for in-place resume."""
    if not _torch_loaded():
        return None
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return None
    return {
        "backend": str(dist.get_backend()),
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "master_addr": os.environ.get("MASTER_ADDR", ""),
        "master_port": os.environ.get("MASTER_PORT", ""),
    }


def _rendezvous_from_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map an agent resume payload onto ``torch_dist`` rebuild keys.

    The contract nests camelCase rendezvous material under
    ``rendezvous``; the state captured at quiesce is already flat
    snake_case. Agent-provided material wins over the captured fallback,
    because after a gang restore only the agent knows the new placement.
    """
    rendezvous = (payload or {}).get("rendezvous")
    if isinstance(rendezvous, dict):
        mapped = {
            "master_addr": rendezvous.get("masterAddr", rendezvous.get("master_addr")),
            "master_port": rendezvous.get("masterPort", rendezvous.get("master_port")),
            "rank": rendezvous.get("rank"),
            "world_size": rendezvous.get("worldSize", rendezvous.get("world_size")),
            "backend": rendezvous.get("backend"),
        }
        return {key: value for key, value in mapped.items() if value is not None}
    return _snapshot_state.pop("distributed", None)


def _warmup_collectives() -> None:
    """Pay the NCCL bootstrap latency here instead of on the first step."""
    import torch
    import torch.distributed as dist

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        dist.all_reduce(torch.zeros(1, device=device))
    dist.barrier()


def _legacy_ack(path: str, distributed: bool) -> None:
    """Keep the pre-contract state-dir acks during the migration window.

    ``megatron-criu-entrypoint.sh`` waits on ``post_restore.done`` and
    pre-contract agents on ``pre_snapshot.ready``; retire together with
    the legacy inline path.
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    _write_json(
        path,
        {
            "rank": os.environ.get("RANK", ""),
            "ts": time.time(),
            "distributed": distributed,
        },
    )


def _on_quiesce() -> None:
    """Drain Megatron to the contract invariants; runs off the signal frame."""
    assert _hooks is not None
    _clear_state_files()
    _flush_optimizer()
    dataloader_state = _capture_dataloader_state()
    if dataloader_state is not None:
        _hooks.control.write_manifest({"megatron": dataloader_state})
    _cuda_synchronize()
    dist_state = _capture_distributed_state()
    if dist_state is not None:
        _snapshot_state["distributed"] = dist_state
    if _torch_loaded():
        _torch_dist.destroy_process_groups()
    _legacy_ack(READY_FILE, distributed=dist_state is not None)
    logger.info("megatron quiesce residue complete")


def _on_resume(payload: dict[str, Any] | None) -> None:
    """Rebuild distributed state from the agent payload; runs off the signal frame."""
    if payload and "epoch" in payload:
        logger.info("resuming at snapshot epoch %s", payload["epoch"])
    rendezvous = _rendezvous_from_payload(payload)
    rebuilt = False
    if rendezvous is not None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            logger.info("process group already initialized; skipping rebuild")
        else:
            _torch_dist.rebuild_process_groups(rendezvous)
            _warmup_collectives()
            rebuilt = True
    _cuda_synchronize()
    _legacy_ack(RESUME_FILE, distributed=rebuilt)
    logger.info("megatron resume residue complete")


def mark_ready(
    model: Any | Sequence[Any] | None = None, optimizer: Any | None = None
) -> None:
    """Declare the snapshottable steady state and register flush targets.

    Call from the training loop at the first step boundary after the
    model, optimizer, and dataloaders are built (arming via
    :func:`install` happens at interpreter startup, before any of them
    exist). Writes the contract's ``ready-for-checkpoint`` file once;
    later per-step calls are cheap no-ops, so callers may invoke it at
    every step boundary. Without the neutral package the legacy protocol
    has no readiness marker and this only registers the objects.

    Args:
        model: DDP model chunk or sequence of chunks whose
            ``finish_grad_sync`` the quiesce residue must call.
        optimizer: The Megatron optimizer, retained for residue that
            needs it; its collectives are flushed through the model
            chunks' shared grad buffers.
    """
    global _ready_marked, _registered_optimizer
    if model is not None:
        chunks = list(model) if isinstance(model, (list, tuple)) else [model]
        _registered_model_chunks[:] = chunks
    if optimizer is not None:
        _registered_optimizer = optimizer
    if _hooks is None:
        logger.debug("neutral package inactive; no readiness marker to write")
        return
    if not _ready_marked:
        _hooks.control.write_ready()
        _ready_marked = True
        logger.info("ready-for-checkpoint written; worker is snapshot-eligible")


def _install_neutral(quiesce_signal: int | None, resume_signal: int | None) -> None:
    global _hooks, _signal_mechanism, _signal_file_mechanism
    _hooks = criu_snapshot_hooks.Hooks(
        on_quiesce=_on_quiesce,
        on_resume=_on_resume,
        control_dir=criu_snapshot_hooks.ControlDir(),
    )
    _hooks.control.ensure_dir()
    _signal_mechanism = criu_snapshot_hooks.SignalMechanism(
        _hooks, quiesce_signal=quiesce_signal, resume_signal=resume_signal
    )
    _signal_mechanism.install()
    _signal_file_mechanism = criu_snapshot_hooks.SignalFileMechanism(_hooks)
    _signal_file_mechanism.start()
    os.makedirs(STATE_DIR, exist_ok=True)
    logger.info("Megatron CRIU hooks installed (criu_snapshot_hooks contract v1)")


def _uninstall_neutral() -> None:
    global _hooks, _signal_mechanism, _signal_file_mechanism, _ready_marked
    if _signal_mechanism is not None:
        _signal_mechanism.uninstall()
    if _signal_file_mechanism is not None:
        _signal_file_mechanism.stop()
    _signal_mechanism = None
    _signal_file_mechanism = None
    _hooks = None
    _ready_marked = False


# --- Legacy inline path ------------------------------------------------------
# Predates the criu_snapshot_hooks package; kept byte-for-byte so images
# built without the package keep today's exact protocol (state-dir files,
# in-signal-frame flows, MEGATRON_CRIU_*_IN_SIGNAL gates). Retire this
# section once every flashtraining image ships criu_snapshot_hooks; until
# then, behavior changes land only in the neutral-package path above.

_legacy_previous_handlers: dict[int, Any] = {}


def _torch_cuda_synchronize() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _distributed_state() -> dict[str, Any] | None:
    try:
        import torch.distributed as dist
    except ImportError:
        return None
    if not dist.is_available() or not dist.is_initialized():
        return None
    state = {
        "backend": str(dist.get_backend()),
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "master_addr": os.environ.get("MASTER_ADDR", ""),
        "master_port": os.environ.get("MASTER_PORT", ""),
    }
    dist.destroy_process_group()
    return state


def _restore_distributed(state: dict[str, Any]) -> None:
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return
    if dist.is_available() and dist.is_initialized():
        return
    if state.get("master_addr"):
        os.environ["MASTER_ADDR"] = state["master_addr"]
    if state.get("master_port"):
        os.environ["MASTER_PORT"] = state["master_port"]
    dist.init_process_group(
        backend=state["backend"],
        rank=int(state["rank"]),
        world_size=int(state["world_size"]),
    )
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        tensor = torch.zeros(1, device=device)
        dist.all_reduce(tensor)
    dist.barrier()


def _pre_snapshot_handler(signum: int, frame: Any) -> None:
    del signum, frame
    try:
        _clear_state_files()
        if _env_enabled("MEGATRON_CRIU_SYNC_IN_SIGNAL"):
            _torch_cuda_synchronize()
        dist_state = None
        if _env_enabled("MEGATRON_CRIU_DISTRIBUTED_IN_SIGNAL"):
            dist_state = _distributed_state()
        if dist_state is not None:
            _snapshot_state["distributed"] = dist_state
        if _env_enabled("MEGATRON_CRIU_SYNC_IN_SIGNAL"):
            _torch_cuda_synchronize()
        _write_json(
            READY_FILE,
            {
                "rank": os.environ.get("RANK", ""),
                "ts": time.time(),
                "distributed": dist_state is not None,
            },
        )
        logger.info("Megatron CRIU pre_snapshot complete")
    except Exception as exc:
        logger.exception("Megatron CRIU pre_snapshot failed")
        _write_text(READY_ERR_FILE, repr(exc))


def _post_restore_handler(signum: int, frame: Any) -> None:
    del signum, frame
    try:
        dist_state = _snapshot_state.pop("distributed", None)
        if dist_state is not None:
            _restore_distributed(dist_state)
        if _env_enabled("MEGATRON_CRIU_SYNC_IN_SIGNAL"):
            _torch_cuda_synchronize()
        _write_json(
            RESUME_FILE,
            {
                "rank": os.environ.get("RANK", ""),
                "ts": time.time(),
                "distributed": dist_state is not None,
            },
        )
        logger.info("Megatron CRIU post_restore complete")
    except Exception:
        logger.exception("Megatron CRIU post_restore failed")
        os._exit(1)


def _install_legacy(quiesce_signal: int | None, resume_signal: int | None) -> None:
    quiesce_signal = PRE_SNAPSHOT_SIGNAL if quiesce_signal is None else quiesce_signal
    resume_signal = POST_RESTORE_SIGNAL if resume_signal is None else resume_signal
    os.makedirs(STATE_DIR, exist_ok=True)
    _legacy_previous_handlers[quiesce_signal] = signal.getsignal(quiesce_signal)
    _legacy_previous_handlers[resume_signal] = signal.getsignal(resume_signal)
    signal.signal(quiesce_signal, _pre_snapshot_handler)
    signal.signal(resume_signal, _post_restore_handler)
    logger.info("Megatron CRIU hooks installed (legacy inline path)")


def _uninstall_legacy() -> None:
    for signum, previous in _legacy_previous_handlers.items():
        signal.signal(signum, previous if previous is not None else signal.SIG_DFL)
    _legacy_previous_handlers.clear()


# --- Entry points ------------------------------------------------------------


def install(
    quiesce_signal: int | None = None, resume_signal: int | None = None
) -> None:
    """Arm the snapshot triggers; called at startup via sitecustomize.

    Args:
        quiesce_signal: Override for the quiesce trigger signal; contract
            default (Linux ``SIGRTMIN+5`` = 39) when None. Tests on
            platforms without RT signals substitute ``SIGUSR1``.
        resume_signal: Override for the resume trigger signal; contract
            default (Linux ``SIGRTMIN+6`` = 40) when None.
    """
    global _installed
    if _installed:
        return
    if NEUTRAL_PACKAGE:
        _install_neutral(quiesce_signal, resume_signal)
    else:
        _install_legacy(quiesce_signal, resume_signal)
    _installed = True


def uninstall() -> None:
    """Disarm the triggers and drop registered state. Idempotent."""
    global _installed, _registered_optimizer
    if not _installed:
        return
    if NEUTRAL_PACKAGE:
        _uninstall_neutral()
    else:
        _uninstall_legacy()
    _registered_model_chunks.clear()
    _registered_optimizer = None
    _snapshot_state.clear()
    _installed = False


if os.environ.get("MEGATRON_CRIU_ENABLE") == "1":
    install()
