"""Local compatibility shim for Megatron's legacy ``one_logger`` import.

This branch of Megatron expects NVIDIA's older internal package:

    from one_logger import OneLogger

The public ``nv-one-logger`` wheels expose the newer ``nv_one_logger`` namespace
instead, so the import fails before any E2E metrics can be recorded.  This shim
implements the small API surface used by ``megatron.training.one_logger_utils``
and writes JSONL telemetry locally.  It is intentionally conservative: no
background thread, no network export, and no dependency on the newer package.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class OneLogger:
    """Minimal file-backed OneLogger compatible with Megatron's legacy hooks."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self._store: dict[str, Any] = {}
        self._lock = threading.RLock()

        rank = os.getenv("RANK", "0")
        project = str(self.config.get("project") or os.getenv("ONE_LOGGER_PROJECT") or "megatron")
        run_name = (
            self.config.get("name")
            or os.getenv("ONE_LOGGER_RUN_NAME")
            or os.getenv("WANDB_EXP_NAME")
            or f"run-{int(time.time())}"
        )
        safe_project = _safe_name(project)
        safe_run = _safe_name(str(run_name))
        log_dir = Path(os.getenv("ONE_LOGGER_DIR", str(Path.home() / "logs" / "one_logger")))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{safe_project}-{safe_run}-rank{rank}-pid{os.getpid()}.jsonl"
        self._write("config", {"config": self.config, "path": str(self.path)})

    def get_context_manager(self):
        return self._lock

    def store_set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def store_get(self, key: str) -> Any:
        return self._store[key]

    def store_has_key(self, key: str) -> bool:
        return key in self._store

    def store_pop(self, key: str) -> Any:
        return self._store.pop(key)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self._write("metrics", {"metrics": metrics})

    def log_app_tag(self, app_tag: str) -> None:
        self._write("app_tag", {"app_tag": app_tag})

    def finish(self) -> None:
        self._write("finish", {})

    def _write(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "time": time.time(),
            "time_ms": round(time.time() * 1000.0),
            "event": event,
            "rank": _maybe_int(os.getenv("RANK")),
            "local_rank": _maybe_int(os.getenv("LOCAL_RANK")),
            "world_size": _maybe_int(os.getenv("WORLD_SIZE")),
            **payload,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:160]


def _maybe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "detach"):
        try:
            detached = value.detach()
            if getattr(detached, "numel", lambda: 2)() == 1:
                return detached.item()
            return {
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "device": str(detached.device),
            }
        except Exception:
            pass
    if callable(value):
        return f"<callable {getattr(value, '__name__', type(value).__name__)}>"
    return repr(value)
