# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Default-off RDEP route-row preparation helpers.

RDEP treats one NVLink domain as both the dense data-parallel domain and the
expert-parallel MoE domain. This module captures the public route-row identity,
expert-owner mapping, and pooled-row accounting needed before a runtime transport
hook is added. It intentionally does not modify TransformerBlock, MoE dispatch,
pipeline schedules, or GPU kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RDEPPreparationConfig:
    """Configuration for building RDEP preparation metadata.

    Args:
        enabled: If False, preparation helpers return None and perform no planning.
        world_size: Dense replica count inside the NVLink domain.
        num_experts: Total routed experts pooled across the domain.
        top_k: Number of selected routed experts per token.
        num_moe_layers: Optional MoE layer count for routing-context memory estimates.
        max_route_rows: Optional guardrail for total route rows in one local plan.
    """

    enabled: bool = False
    world_size: int = 1
    num_experts: int = 1
    top_k: int = 1
    num_moe_layers: Optional[int] = None
    max_route_rows: Optional[int] = None

    def validate(self) -> None:
        """Validate RDEP preparation settings."""
        if self.world_size <= 0:
            raise ValueError(f"RDEP world_size must be positive, got {self.world_size}")
        if self.num_experts <= 0:
            raise ValueError(f"RDEP num_experts must be positive, got {self.num_experts}")
        if self.top_k <= 0:
            raise ValueError(f"RDEP top_k must be positive, got {self.top_k}")
        if self.num_moe_layers is not None and self.num_moe_layers <= 0:
            raise ValueError(
                f"RDEP num_moe_layers must be positive when set, got {self.num_moe_layers}"
            )
        if self.max_route_rows is not None and self.max_route_rows <= 0:
            raise ValueError(
                f"RDEP max_route_rows must be positive when set, got {self.max_route_rows}"
            )
        if self.num_experts % self.world_size != 0:
            raise ValueError(
                "RDEP preparation currently requires balanced expert ownership: "
                f"num_experts={self.num_experts}, world_size={self.world_size}"
            )


@dataclass(frozen=True)
class RDEPRouteRowIdentity:
    """Decoded RDEP route-row identity."""

    rank: int
    token: int
    slot: int


@dataclass(frozen=True)
class RDEPExpertOwner:
    """Contiguous expert ownership for one routed expert."""

    owner_rank: int
    local_expert_id: int


@dataclass(frozen=True)
class RDEPPreparationPlan:
    """Prepared RDEP metadata for a local token shape."""

    tokens_per_rank: int
    world_size: int
    num_experts: int
    top_k: int
    num_local_experts: int
    route_rows_per_rank: int
    pooled_route_rows: int
    expected_rows_per_expert: float
    routing_context_bytes: Optional[int]

    def __post_init__(self) -> None:
        """Validate plan invariants."""
        if self.tokens_per_rank <= 0:
            raise ValueError(f"RDEP tokens_per_rank must be positive, got {self.tokens_per_rank}")
        if self.world_size <= 0:
            raise ValueError(f"RDEP world_size must be positive, got {self.world_size}")
        if self.num_experts <= 0:
            raise ValueError(f"RDEP num_experts must be positive, got {self.num_experts}")
        if self.top_k <= 0:
            raise ValueError(f"RDEP top_k must be positive, got {self.top_k}")
        if self.num_local_experts * self.world_size != self.num_experts:
            raise ValueError(
                "RDEP plan requires balanced contiguous expert ownership: "
                f"num_local_experts={self.num_local_experts}, "
                f"world_size={self.world_size}, num_experts={self.num_experts}"
            )
        expected_route_rows = self.tokens_per_rank * self.top_k
        if self.route_rows_per_rank != expected_route_rows:
            raise ValueError(
                f"RDEP route_rows_per_rank mismatch: expected {expected_route_rows}, "
                f"got {self.route_rows_per_rank}"
            )
        expected_pooled_rows = self.world_size * self.route_rows_per_rank
        if self.pooled_route_rows != expected_pooled_rows:
            raise ValueError(
                f"RDEP pooled_route_rows mismatch: expected {expected_pooled_rows}, "
                f"got {self.pooled_route_rows}"
            )


def encode_route_row_id(rank: int, token: int, slot: int, tokens_per_rank: int, top_k: int) -> int:
    """Encode the RDEP route-row identity ``((rank * T) + token) * K + slot``."""
    _validate_identity_bounds(rank, token, slot, tokens_per_rank, top_k)
    return ((rank * tokens_per_rank) + token) * top_k + slot


def decode_route_row_id(row_id: int, tokens_per_rank: int, top_k: int) -> RDEPRouteRowIdentity:
    """Decode an RDEP route-row identity."""
    if row_id < 0:
        raise ValueError(f"RDEP row_id must be non-negative, got {row_id}")
    if tokens_per_rank <= 0:
        raise ValueError(f"RDEP tokens_per_rank must be positive, got {tokens_per_rank}")
    if top_k <= 0:
        raise ValueError(f"RDEP top_k must be positive, got {top_k}")

    slot = row_id % top_k
    token_and_rank = row_id // top_k
    token = token_and_rank % tokens_per_rank
    rank = token_and_rank // tokens_per_rank
    return RDEPRouteRowIdentity(rank=rank, token=token, slot=slot)


def contiguous_expert_owner(expert_id: int, num_local_experts: int) -> RDEPExpertOwner:
    """Map a global expert id to ``(owner_rank, local_expert_id)``."""
    if expert_id < 0:
        raise ValueError(f"RDEP expert_id must be non-negative, got {expert_id}")
    if num_local_experts <= 0:
        raise ValueError(f"RDEP num_local_experts must be positive, got {num_local_experts}")
    return RDEPExpertOwner(
        owner_rank=expert_id // num_local_experts, local_expert_id=expert_id % num_local_experts
    )


def expected_rows_per_expert(
    world_size: int, tokens_per_rank: int, top_k: int, num_experts: int
) -> float:
    """Return the RDEP pooled-row mean ``world_size * tokens_per_rank * top_k / E``."""
    _validate_positive("world_size", world_size)
    _validate_positive("tokens_per_rank", tokens_per_rank)
    _validate_positive("top_k", top_k)
    _validate_positive("num_experts", num_experts)
    return (world_size * tokens_per_rank * top_k) / num_experts


def estimate_routing_context_bytes(
    num_moe_layers: int,
    world_size: int,
    tokens_per_rank: int,
    top_k: int,
    bytes_per_route_row: int = 8,
) -> int:
    """Estimate compact routing-context bytes for accepted route rows."""
    _validate_positive("num_moe_layers", num_moe_layers)
    _validate_positive("world_size", world_size)
    _validate_positive("tokens_per_rank", tokens_per_rank)
    _validate_positive("top_k", top_k)
    _validate_positive("bytes_per_route_row", bytes_per_route_row)
    return bytes_per_route_row * num_moe_layers * world_size * tokens_per_rank * top_k


def build_rdep_preparation_plan(
    tokens_per_rank: int, config: Optional[RDEPPreparationConfig] = None
) -> Optional[RDEPPreparationPlan]:
    """Build default-off RDEP route-row preparation metadata.

    Args:
        tokens_per_rank: Local token count per dense replica.
        config: RDEP preparation config. Omitted config keeps preparation disabled.

    Returns:
        A preparation plan when enabled, otherwise None.

    Raises:
        ValueError: If the config is invalid or the enabled plan exceeds max_route_rows.
    """
    config = config or RDEPPreparationConfig()
    config.validate()
    if not config.enabled:
        return None
    if tokens_per_rank <= 0:
        raise ValueError(f"RDEP tokens_per_rank must be positive, got {tokens_per_rank}")

    route_rows_per_rank = tokens_per_rank * config.top_k
    pooled_route_rows = config.world_size * route_rows_per_rank
    if config.max_route_rows is not None and pooled_route_rows > config.max_route_rows:
        raise ValueError(
            "RDEP preparation plan exceeds max_route_rows: "
            f"{pooled_route_rows} route rows > {config.max_route_rows}"
        )

    routing_context_bytes = None
    if config.num_moe_layers is not None:
        routing_context_bytes = estimate_routing_context_bytes(
            config.num_moe_layers, config.world_size, tokens_per_rank, config.top_k
        )

    return RDEPPreparationPlan(
        tokens_per_rank=tokens_per_rank,
        world_size=config.world_size,
        num_experts=config.num_experts,
        top_k=config.top_k,
        num_local_experts=config.num_experts // config.world_size,
        route_rows_per_rank=route_rows_per_rank,
        pooled_route_rows=pooled_route_rows,
        expected_rows_per_expert=expected_rows_per_expert(
            config.world_size, tokens_per_rank, config.top_k, config.num_experts
        ),
        routing_context_bytes=routing_context_bytes,
    )


def _validate_identity_bounds(
    rank: int, token: int, slot: int, tokens_per_rank: int, top_k: int
) -> None:
    _validate_positive("tokens_per_rank", tokens_per_rank)
    _validate_positive("top_k", top_k)
    if rank < 0:
        raise ValueError(f"RDEP rank must be non-negative, got {rank}")
    if not 0 <= token < tokens_per_rank:
        raise ValueError(f"RDEP token must be in [0, {tokens_per_rank}), got {token}")
    if not 0 <= slot < top_k:
        raise ValueError(f"RDEP slot must be in [0, {top_k}), got {slot}")


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"RDEP {name} must be positive, got {value}")
