# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import-light tests for the Megatron CRIU hook glue.

Covers both branches of ``tools/criu/megatron_criu_hooks.py``: the
neutral-package path (``criu_snapshot_hooks`` installed) and the legacy
inline path (import blocked). torch and megatron are stubbed through
``sys.modules`` — nothing heavy is imported — and every signal test
substitutes SIGUSR1/SIGUSR2 for the contract's Linux RT signals, which
do not exist on macOS and must never be raised from tests.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterator

import pytest

HOOKS_PATH = Path(__file__).resolve().parent / "megatron_criu_hooks.py"

_HAVE_NEUTRAL_PACKAGE = importlib.util.find_spec("criu_snapshot_hooks") is not None

requires_neutral_package = pytest.mark.skipif(
    not _HAVE_NEUTRAL_PACKAGE, reason="criu_snapshot_hooks is not installed"
)


class _FakeDistributed:
    """Recording stand-in for torch.distributed."""

    def __init__(self) -> None:
        self.initialized = False
        self.destroy_calls = 0
        self.init_calls: list[dict[str, Any]] = []
        self.barrier_calls = 0

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self.initialized

    def get_backend(self) -> str:
        return "gloo"

    def get_rank(self) -> int:
        return 3

    def get_world_size(self) -> int:
        return 8

    def destroy_process_group(self) -> None:
        self.destroy_calls += 1
        self.initialized = False

    def init_process_group(self, backend: str, rank: int, world_size: int) -> None:
        self.init_calls.append(
            {"backend": backend, "rank": rank, "world_size": world_size}
        )
        self.initialized = True

    def barrier(self) -> None:
        self.barrier_calls += 1


def _stub_torch(monkeypatch: pytest.MonkeyPatch, initialized: bool) -> _FakeDistributed:
    fake_dist = _FakeDistributed()
    fake_dist.initialized = initialized

    dist_module = types.ModuleType("torch.distributed")
    for name in (
        "is_available",
        "is_initialized",
        "get_backend",
        "get_rank",
        "get_world_size",
        "destroy_process_group",
        "init_process_group",
        "barrier",
    ):
        setattr(dist_module, name, getattr(fake_dist, name))

    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_module.distributed = dist_module

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_module)
    return fake_dist


def _stub_megatron_args(monkeypatch: pytest.MonkeyPatch, **fields: Any) -> None:
    args = types.SimpleNamespace(**fields)
    global_vars = types.ModuleType("megatron.training.global_vars")
    global_vars.get_args = lambda: args
    monkeypatch.setitem(sys.modules, "megatron.training.global_vars", global_vars)


def _load_hooks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    neutral: bool,
) -> Any:
    """Load a fresh copy of the hooks module for one branch.

    A unique module name isolates per-test module state; the legacy
    branch is forced by poisoning ``sys.modules`` so the guarded package
    import raises ImportError.
    """
    state_dir = tmp_path / "state"
    control_dir = tmp_path / "control"
    monkeypatch.setenv("MEGATRON_CRIU_STATE_DIR", str(state_dir))
    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", str(control_dir))
    monkeypatch.delenv("MEGATRON_CRIU_ENABLE", raising=False)
    if not neutral:
        for name in [
            loaded
            for loaded in sys.modules
            if loaded == "criu_snapshot_hooks" or loaded.startswith("criu_snapshot_hooks.")
        ]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setitem(sys.modules, "criu_snapshot_hooks", None)

    module_name = f"megatron_criu_hooks_{'neutral' if neutral else 'legacy'}_{time.monotonic_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, HOOKS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def installed_hooks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[Any]:
    """A fresh hooks module for the requested branch, disarmed on teardown."""
    module = _load_hooks(monkeypatch, tmp_path, neutral=request.param)
    yield module
    module.uninstall()


def _wait_for(path: Path, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"{path} did not appear within {timeout_s}s")


def _control(module: Any, name: str) -> Path:
    return Path(os.environ["DYN_SNAPSHOT_CONTROL_DIR"]) / name


# --- Neutral-package branch --------------------------------------------------


@requires_neutral_package
@pytest.mark.parametrize("installed_hooks", [True], indirect=True)
class TestNeutralBranch:
    def test_flag_and_contract_signals(self, installed_hooks: Any) -> None:
        assert installed_hooks.NEUTRAL_PACKAGE is True
        assert installed_hooks.PRE_SNAPSHOT_SIGNAL == 39
        assert installed_hooks.POST_RESTORE_SIGNAL == 40

    def test_mark_ready_writes_contract_ready_file(self, installed_hooks: Any) -> None:
        installed_hooks.install(
            quiesce_signal=signal.SIGUSR1, resume_signal=signal.SIGUSR2
        )
        installed_hooks.mark_ready()
        ready = _control(installed_hooks, "ready-for-checkpoint")
        assert ready.exists()
        ack = json.loads(ready.read_text(encoding="utf-8"))
        assert ack["pid"] == os.getpid()
        assert ack["contract"] == "v1"
        installed_hooks.mark_ready()

    def test_signal_drain_roundtrip(
        self, installed_hooks: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_dist = _stub_torch(monkeypatch, initialized=True)
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.9")
        monkeypatch.setenv("MASTER_PORT", "29500")
        installed_hooks.install(
            quiesce_signal=signal.SIGUSR1, resume_signal=signal.SIGUSR2
        )

        os.kill(os.getpid(), signal.SIGUSR1)
        _wait_for(_control(installed_hooks, "quiesced"))
        assert fake_dist.destroy_calls == 1
        assert Path(installed_hooks.READY_FILE).exists()
        assert not _control(installed_hooks, "error").exists()

        os.kill(os.getpid(), signal.SIGUSR2)
        _wait_for(_control(installed_hooks, "resumed"))
        assert fake_dist.init_calls == [
            {"backend": "gloo", "rank": 3, "world_size": 8}
        ]
        assert fake_dist.barrier_calls == 1
        legacy_resume = json.loads(
            Path(installed_hooks.RESUME_FILE).read_text(encoding="utf-8")
        )
        assert legacy_resume["distributed"] is True

    def test_signal_file_mechanism_consumes_request(
        self, installed_hooks: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_torch(monkeypatch, initialized=False)
        installed_hooks.install(
            quiesce_signal=signal.SIGUSR1, resume_signal=signal.SIGUSR2
        )
        request_file = _control(installed_hooks, "quiesce-requested")
        request_file.touch()
        _wait_for(_control(installed_hooks, "quiesced"))
        assert not request_file.exists()

    def test_quiesce_residue_flushes_and_captures(
        self, installed_hooks: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_torch(monkeypatch, initialized=False)
        _stub_megatron_args(
            monkeypatch,
            iteration=1200,
            consumed_train_samples=614400,
            consumed_valid_samples=1024,
        )
        flushed: list[str] = []
        chunk = types.SimpleNamespace(finish_grad_sync=lambda: flushed.append("chunk"))
        installed_hooks.install(
            quiesce_signal=signal.SIGUSR1, resume_signal=signal.SIGUSR2
        )
        installed_hooks.mark_ready(model=[chunk], optimizer=object())

        installed_hooks._hooks.quiesce()

        assert flushed == ["chunk"]
        manifest = json.loads(
            _control(installed_hooks, "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["megatron"] == {
            "iteration": 1200,
            "consumed_train_samples": 614400,
            "consumed_valid_samples": 1024,
        }

    def test_resume_uses_agent_rendezvous_payload(
        self, installed_hooks: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_dist = _stub_torch(monkeypatch, initialized=False)
        monkeypatch.setenv("MASTER_ADDR", "stale")
        monkeypatch.setenv("MASTER_PORT", "0")
        installed_hooks.install(
            quiesce_signal=signal.SIGUSR1, resume_signal=signal.SIGUSR2
        )

        installed_hooks._hooks.resume(
            {
                "epoch": 2,
                "rendezvous": {
                    "masterAddr": "10.42.0.17",
                    "masterPort": 29501,
                    "rank": 1,
                    "worldSize": 4,
                    "backend": "gloo",
                },
            }
        )

        assert fake_dist.init_calls == [
            {"backend": "gloo", "rank": 1, "world_size": 4}
        ]
        assert os.environ["MASTER_ADDR"] == "10.42.0.17"
        assert os.environ["MASTER_PORT"] == "29501"
        assert _control(installed_hooks, "resumed").exists()

    def test_failed_quiesce_writes_error_nack(
        self, installed_hooks: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_torch(monkeypatch, initialized=False)

        def boom() -> None:
            raise RuntimeError("grad buffer flush failed")

        chunk = types.SimpleNamespace(finish_grad_sync=boom)
        installed_hooks.install(
            quiesce_signal=signal.SIGUSR1, resume_signal=signal.SIGUSR2
        )
        installed_hooks.mark_ready(model=chunk)

        with pytest.raises(RuntimeError, match="grad buffer flush failed"):
            installed_hooks._hooks.quiesce()

        error = _control(installed_hooks, "error").read_text(encoding="utf-8")
        assert "grad buffer flush failed" in error
        assert not _control(installed_hooks, "quiesced").exists()


# --- Legacy inline branch ----------------------------------------------------


@pytest.mark.parametrize("installed_hooks", [False], indirect=True)
class TestLegacyBranch:
    def test_flag(self, installed_hooks: Any) -> None:
        assert installed_hooks.NEUTRAL_PACKAGE is False

    def test_signal_roundtrip_preserves_state_dir_protocol(
        self, installed_hooks: Any
    ) -> None:
        installed_hooks.install(
            quiesce_signal=signal.SIGUSR1, resume_signal=signal.SIGUSR2
        )

        os.kill(os.getpid(), signal.SIGUSR1)
        ready = json.loads(
            Path(installed_hooks.READY_FILE).read_text(encoding="utf-8")
        )
        assert ready["distributed"] is False
        assert not Path(installed_hooks.READY_ERR_FILE).exists()

        os.kill(os.getpid(), signal.SIGUSR2)
        resume = json.loads(
            Path(installed_hooks.RESUME_FILE).read_text(encoding="utf-8")
        )
        assert resume["distributed"] is False

    def test_no_contract_files_written(self, installed_hooks: Any) -> None:
        installed_hooks.install(
            quiesce_signal=signal.SIGUSR1, resume_signal=signal.SIGUSR2
        )
        os.kill(os.getpid(), signal.SIGUSR1)
        assert Path(installed_hooks.READY_FILE).exists()
        control_dir = Path(os.environ["DYN_SNAPSHOT_CONTROL_DIR"])
        assert not control_dir.exists() or not any(control_dir.iterdir())

    def test_mark_ready_is_a_registration_only_noop(
        self, installed_hooks: Any
    ) -> None:
        chunk = types.SimpleNamespace(finish_grad_sync=lambda: None)
        installed_hooks.mark_ready(model=chunk, optimizer=object())
        assert installed_hooks._registered_model_chunks == [chunk]
        control_dir = Path(os.environ["DYN_SNAPSHOT_CONTROL_DIR"])
        assert not control_dir.exists() or not any(control_dir.iterdir())

    def test_uninstall_restores_previous_handlers(self, installed_hooks: Any) -> None:
        previous = signal.getsignal(signal.SIGUSR1)
        installed_hooks.install(
            quiesce_signal=signal.SIGUSR1, resume_signal=signal.SIGUSR2
        )
        assert signal.getsignal(signal.SIGUSR1) is not previous
        installed_hooks.uninstall()
        assert signal.getsignal(signal.SIGUSR1) == previous
