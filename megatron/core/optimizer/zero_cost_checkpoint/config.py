# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def parse_tensor_attr_list(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = raw.split(",")
    else:
        items = raw
    return tuple(
        dict.fromkeys(item.strip() for item in items if item and item.strip())
    )


@dataclass(frozen=True)
class ZeroCostCheckpointConfig:
    enabled: bool = False
    workers_num: int = 1
    flash_device: str = "/dev/shm/megatron_zcc"
    flash_stripe: tuple[str, ...] = ()
    durable_dir: str | None = None
    durable_interval: int = 10
    compress: str = "zstd:1"
    include_rng: bool = True
    include_dither: bool = True
    bucket_hook: bool = True
    numa_pin: bool = True
    use_gds: bool = False
    fault_inject: bool = False
    recovery_mode: str = "auto"
    fuse_state_buffer: bool = True
    extra_tensor_attrs: tuple[str, ...] = ()
    retain_latest: int = 1

    @classmethod
    def from_optimizer_config(cls, config: Any) -> "ZeroCostCheckpointConfig":
        enabled = _env_bool(
            "MEGATRON_ENABLE_ZCC",
            bool(getattr(config, "enable_zero_cost_checkpoint", False)),
        )
        flash_stripe = os.getenv(
            "MEGATRON_ZCC_FLASH_STRIPE",
            getattr(config, "zcc_flash_stripe", "") or "",
        )
        durable_dir = os.getenv(
            "MEGATRON_ZCC_DURABLE_DIR",
            getattr(config, "zcc_durable_dir", None) or "",
        )
        extra_tensor_attrs = os.getenv(
            "MEGATRON_ZCC_EXTRA_TENSOR_ATTRS",
            getattr(config, "zcc_extra_tensor_attrs", "") or "",
        )
        return cls(
            enabled=enabled,
            workers_num=_env_int(
                "MEGATRON_ZCC_WORKERS_NUM",
                int(getattr(config, "zcc_workers_num", 1)),
            ),
            flash_device=os.getenv(
                "MEGATRON_ZCC_FLASH_DEV",
                getattr(config, "zcc_flash_device", "/dev/shm/megatron_zcc"),
            ),
            flash_stripe=tuple(p for p in flash_stripe.split(",") if p),
            durable_dir=durable_dir or None,
            durable_interval=_env_int(
                "MEGATRON_ZCC_DURABLE_INTERVAL",
                int(getattr(config, "zcc_durable_interval", 10)),
            ),
            compress=os.getenv(
                "MEGATRON_ZCC_COMPRESS",
                getattr(config, "zcc_compress", "zstd:1"),
            ),
            include_rng=_env_bool(
                "MEGATRON_ZCC_INCLUDE_RNG",
                bool(getattr(config, "zcc_include_rng", True)),
            ),
            include_dither=_env_bool(
                "MEGATRON_ZCC_INCLUDE_DITHER",
                bool(getattr(config, "zcc_include_dither", True)),
            ),
            bucket_hook=_env_bool(
                "MEGATRON_ZCC_BUCKET_HOOK",
                bool(getattr(config, "zcc_bucket_hook", True)),
            ),
            numa_pin=_env_bool(
                "MEGATRON_ZCC_NUMA_PIN",
                bool(getattr(config, "zcc_numa_pin", True)),
            ),
            use_gds=_env_bool(
                "MEGATRON_ZCC_USE_GDS",
                bool(getattr(config, "zcc_use_gds", False)),
            ),
            fault_inject=_env_bool(
                "MEGATRON_ZCC_FAULT_INJECT",
                bool(getattr(config, "zcc_fault_inject", False)),
            ),
            recovery_mode=os.getenv(
                "MEGATRON_ZCC_RECOVERY_MODE",
                getattr(config, "zcc_recovery_mode", "auto"),
            ),
            fuse_state_buffer=_env_bool(
                "MEGATRON_FLASH_FUSE_STATE_BUFFER",
                enabled,
            ),
            extra_tensor_attrs=parse_tensor_attr_list(extra_tensor_attrs),
            retain_latest=_env_int(
                "MEGATRON_ZCC_RETAIN_LATEST",
                int(getattr(config, "zcc_retain_latest", 1)),
            ),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.workers_num < 1:
            raise ValueError("ZCC requires at least one worker")
        if self.durable_interval < 1:
            raise ValueError("ZCC durable interval must be positive")
        if self.recovery_mode not in ("auto", "flash", "peer", "durable"):
            raise ValueError(
                "ZCC recovery mode must be one of auto, flash, peer, durable"
            )
        if self.compress != "none" and not self.compress.startswith("zstd:"):
            raise ValueError("ZCC compression must be none or zstd:<level>")
        if self.retain_latest < 0:
            raise ValueError("ZCC retain_latest must be non-negative")
