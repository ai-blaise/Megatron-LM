# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DualPipeV rank mapping and step-order helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


__all__ = [
    "DualPipeVMapping",
    "DualPipeVStage",
    "DualPipeVStep",
    "build_dualpipe_v_mapping",
    "iter_dualpipe_v_steps",
    "validate_dualpipe_v_config",
]


@dataclass(frozen=True)
class DualPipeVStage:
    """One logical DualPipeV phase hosted by a physical pipeline rank."""

    rank: int
    phase: int
    virtual_stage: int
    layer_group: int
    forward_index: int
    forward_prev_rank: int | None
    forward_next_rank: int | None
    emits_bridge_output: bool = False
    receives_bridge_input: bool = False
    computes_loss: bool = False


@dataclass(frozen=True)
class DualPipeVMapping:
    """Complete conservative sequential DualPipeV mapping for one pipeline group."""

    pipeline_model_parallel_size: int
    virtual_pipeline_model_parallel_size: int
    num_microbatches: int
    stages: tuple[DualPipeVStage, ...]
    schedule_kind: str = "dualpipe_v"
    runtime_backed: bool = True

    def phase_stages(self, phase: int) -> tuple[DualPipeVStage, ...]:
        _validate_phase(phase)
        return tuple(
            sorted(
                (stage for stage in self.stages if stage.phase == phase),
                key=lambda stage: stage.forward_index,
            )
        )

    def rank_stages(self, rank: int) -> tuple[DualPipeVStage, ...]:
        _validate_rank(rank, self.pipeline_model_parallel_size)
        return tuple(stage for stage in self.stages if stage.rank == rank)

    def stage(self, rank: int, phase: int) -> DualPipeVStage:
        _validate_rank(rank, self.pipeline_model_parallel_size)
        _validate_phase(phase)
        matches = [stage for stage in self.stages if stage.rank == rank and stage.phase == phase]
        if len(matches) != 1:
            raise ValueError(f"expected one DualPipeV stage for rank={rank}, phase={phase}")
        return matches[0]

    def forward_ranks(self, phase: int) -> tuple[int, ...]:
        return tuple(stage.rank for stage in self.phase_stages(phase))

    def forward_layer_groups(self, phase: int) -> tuple[int, ...]:
        return tuple(stage.layer_group for stage in self.phase_stages(phase))

    @property
    def bridge_rank(self) -> int:
        return self.pipeline_model_parallel_size - 1

    @property
    def loss_rank(self) -> int:
        return 0


@dataclass(frozen=True)
class DualPipeVStep:
    """One operation in the reference DualPipeV eight-step order."""

    step: int
    op: str
    phase: int | None = None
    defer_weight_grad: bool = False
    recv: bool = True
    send: bool = True


def build_dualpipe_v_mapping(
    pipeline_model_parallel_size: int,
    virtual_pipeline_model_parallel_size: int,
    num_microbatches: int,
) -> DualPipeVMapping:
    """Build the logical DualPipeV phase-to-rank mapping."""

    validate_dualpipe_v_config(
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size,
        num_microbatches,
    )
    pp_size = pipeline_model_parallel_size
    stages = []
    for rank in range(pp_size):
        stages.append(
            DualPipeVStage(
                rank=rank,
                phase=0,
                virtual_stage=0,
                layer_group=rank,
                forward_index=rank,
                forward_prev_rank=rank - 1 if rank > 0 else None,
                forward_next_rank=rank + 1 if rank < pp_size - 1 else None,
                emits_bridge_output=rank == pp_size - 1,
            )
        )
        stages.append(
            DualPipeVStage(
                rank=rank,
                phase=1,
                virtual_stage=1,
                layer_group=2 * pp_size - 1 - rank,
                forward_index=pp_size - 1 - rank,
                forward_prev_rank=rank + 1 if rank < pp_size - 1 else None,
                forward_next_rank=rank - 1 if rank > 0 else None,
                receives_bridge_input=rank == pp_size - 1,
                computes_loss=rank == 0,
            )
        )

    return DualPipeVMapping(
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        num_microbatches=num_microbatches,
        stages=tuple(stages),
    )


def iter_dualpipe_v_steps(
    pipeline_model_parallel_size: int,
    pipeline_model_parallel_rank: int,
    num_microbatches: int,
) -> Iterator[DualPipeVStep]:
    """Yield the reference DualPipeV eight-step order for one pipeline rank."""

    _validate_rank(pipeline_model_parallel_rank, pipeline_model_parallel_size)
    validate_dualpipe_v_config(pipeline_model_parallel_size, 2, num_microbatches)
    rank = pipeline_model_parallel_rank
    pp_size = pipeline_model_parallel_size

    for _ in range((pp_size - rank - 1) * 2):
        yield DualPipeVStep(step=1, op="F", phase=0)

    yield DualPipeVStep(step=2, op="recv_F", phase=0)
    for i in range(rank + 1):
        yield DualPipeVStep(step=2, op="F", phase=0, recv=False, send=False)
        yield DualPipeVStep(step=2, op="recv_F", phase=0)
        yield DualPipeVStep(step=2, op="F", phase=1, send=(rank != pp_size - 1) or (i < rank))
        yield DualPipeVStep(step=2, op="send_F", phase=0)

    for _ in range(pp_size - rank - 1):
        yield DualPipeVStep(step=3, op="B", phase=1, defer_weight_grad=True)
        yield DualPipeVStep(step=3, op="recv_F", phase=1)
        yield DualPipeVStep(step=3, op="W")
        yield DualPipeVStep(step=3, op="F", phase=1, recv=False)

    for i in range(num_microbatches - pp_size * 2 + rank + 1):
        if i == 0 and rank == pp_size - 1:
            yield DualPipeVStep(step=4, op="F", phase=0, recv=False, send=False)
            yield DualPipeVStep(step=4, op="send_F", phase=1)
            yield DualPipeVStep(step=4, op="B", phase=1, send=False)
            yield DualPipeVStep(step=4, op="send_F", phase=0)
            yield DualPipeVStep(step=4, op="send_B", phase=1)
        else:
            yield DualPipeVStep(step=4, op="FB", phase=0)
        yield DualPipeVStep(step=4, op="FB", phase=1)

    for _ in range(pp_size - rank - 1):
        yield DualPipeVStep(step=5, op="B", phase=1)
        yield DualPipeVStep(step=5, op="FB", phase=1)

    for _ in range(rank + 1):
        yield DualPipeVStep(step=6, op="B", phase=1)
        yield DualPipeVStep(step=6, op="B", phase=0)

    for _ in range(pp_size - rank - 1):
        yield DualPipeVStep(step=7, op="W")
        yield DualPipeVStep(step=7, op="B", phase=0, defer_weight_grad=True)

    for _ in range(rank + 1):
        yield DualPipeVStep(step=8, op="W")


def validate_dualpipe_v_config(
    pipeline_model_parallel_size: int,
    virtual_pipeline_model_parallel_size: int | None,
    num_microbatches: int | None,
) -> None:
    """Validate the static DualPipeV shape constraints."""

    if pipeline_model_parallel_size is None:
        raise ValueError("DualPipeV requires pipeline_model_parallel_size >= 2")
    if pipeline_model_parallel_size < 2:
        raise ValueError("DualPipeV requires pipeline_model_parallel_size >= 2")
    if virtual_pipeline_model_parallel_size != 2:
        raise ValueError("DualPipeV requires virtual_pipeline_model_parallel_size == 2")
    min_microbatches = 2 * pipeline_model_parallel_size
    if num_microbatches is not None and num_microbatches < min_microbatches:
        raise ValueError(
            "DualPipeV requires num_microbatches >= "
            f"2 * pipeline_model_parallel_size; got {num_microbatches}"
        )


def _validate_phase(phase: int) -> None:
    if phase not in {0, 1}:
        raise ValueError(f"DualPipeV phase must be 0 or 1, got {phase}")


def _validate_rank(rank: int, pipeline_model_parallel_size: int) -> None:
    if rank < 0 or rank >= pipeline_model_parallel_size:
        raise ValueError(
            "rank must be in [0, pipeline_model_parallel_size); "
            f"got rank={rank}, pipeline_model_parallel_size={pipeline_model_parallel_size}"
        )
