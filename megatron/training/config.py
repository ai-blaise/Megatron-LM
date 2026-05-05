# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility exports for Megatron Bridge training config imports.

This repository stores the training config dataclasses split across
``training_config.py``, ``common_config.py``, and ``resilience_config.py``.
Megatron Bridge expects the newer consolidated ``megatron.training.config``
module, so re-export the local definitions here.
"""

from dataclasses import dataclass

from megatron.training.common_config import DistributedInitConfig, ProfilingConfig, RNGConfig
from megatron.training.resilience_config import RerunStateMachineConfig, StragglerDetectionConfig
from megatron.training.training_config import (
    CheckpointConfig as _CheckpointConfig,
    LoggerConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)


@dataclass(kw_only=True)
class CheckpointConfig(_CheckpointConfig):
    """Checkpoint config with Bridge-compatible attribute aliases."""

    @property
    def fully_parallel_save(self) -> bool:
        return self.ckpt_fully_parallel_save

    @fully_parallel_save.setter
    def fully_parallel_save(self, value: bool) -> None:
        self.ckpt_fully_parallel_save = value

    @property
    def fully_parallel_load(self) -> bool:
        return self.ckpt_fully_parallel_load

    @fully_parallel_load.setter
    def fully_parallel_load(self, value: bool) -> None:
        self.ckpt_fully_parallel_load = value


__all__ = [
    "CheckpointConfig",
    "DistributedInitConfig",
    "LoggerConfig",
    "ProfilingConfig",
    "RNGConfig",
    "RerunStateMachineConfig",
    "SchedulerConfig",
    "StragglerDetectionConfig",
    "TrainingConfig",
    "ValidationConfig",
]
