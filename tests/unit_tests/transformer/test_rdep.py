# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.transformer.rdep import (
    RDEPPreparationConfig,
    RDEPPreparationPlan,
    RDEPRouteRowIdentity,
    build_rdep_preparation_plan,
    contiguous_expert_owner,
    decode_route_row_id,
    encode_route_row_id,
    estimate_routing_context_bytes,
    expected_rows_per_expert,
)


def test_rdep_preparation_is_default_off():
    assert build_rdep_preparation_plan(tokens_per_rank=16) is None


def test_route_row_identity_matches_rdep_formula():
    row_id = encode_route_row_id(rank=2, token=17, slot=5, tokens_per_rank=4096, top_k=6)

    assert row_id == ((2 * 4096) + 17) * 6 + 5
    assert decode_route_row_id(row_id, tokens_per_rank=4096, top_k=6) == RDEPRouteRowIdentity(
        rank=2, token=17, slot=5
    )


def test_contiguous_expert_owner_matches_nmoe_layout():
    assert contiguous_expert_owner(expert_id=0, num_local_experts=8).owner_rank == 0
    assert contiguous_expert_owner(expert_id=0, num_local_experts=8).local_expert_id == 0
    assert contiguous_expert_owner(expert_id=63, num_local_experts=8).owner_rank == 7
    assert contiguous_expert_owner(expert_id=63, num_local_experts=8).local_expert_id == 7


def test_pooled_row_accounting_matches_rdep_b200_shape():
    assert (
        expected_rows_per_expert(world_size=8, tokens_per_rank=4096, top_k=6, num_experts=64)
        == 3072.0
    )
    assert (
        estimate_routing_context_bytes(
            num_moe_layers=27, world_size=8, tokens_per_rank=4096, top_k=6
        )
        == 42467328
    )


def test_rdep_preparation_plan_tracks_route_rows_and_memory_budget():
    plan = build_rdep_preparation_plan(
        tokens_per_rank=4096,
        config=RDEPPreparationConfig(
            enabled=True, world_size=8, num_experts=64, top_k=6, num_moe_layers=27
        ),
    )

    assert plan == RDEPPreparationPlan(
        tokens_per_rank=4096,
        world_size=8,
        num_experts=64,
        top_k=6,
        num_local_experts=8,
        route_rows_per_rank=24576,
        pooled_route_rows=196608,
        expected_rows_per_expert=3072.0,
        routing_context_bytes=42467328,
    )


def test_rdep_preparation_rejects_over_budget_plan():
    with pytest.raises(ValueError, match="exceeds max_route_rows"):
        build_rdep_preparation_plan(
            tokens_per_rank=4096,
            config=RDEPPreparationConfig(
                enabled=True, world_size=8, num_experts=64, top_k=6, max_route_rows=196607
            ),
        )


def test_rdep_preparation_rejects_unbalanced_expert_ownership():
    with pytest.raises(ValueError, match="balanced expert ownership"):
        build_rdep_preparation_plan(
            tokens_per_rank=4096,
            config=RDEPPreparationConfig(enabled=True, world_size=8, num_experts=66, top_k=6),
        )


def test_route_row_identity_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="token must be in"):
        encode_route_row_id(rank=0, token=4096, slot=0, tokens_per_rank=4096, top_k=6)
    with pytest.raises(ValueError, match="slot must be in"):
        encode_route_row_id(rank=0, token=0, slot=6, tokens_per_rank=4096, top_k=6)


def test_rdep_plan_validates_derived_counts():
    with pytest.raises(ValueError, match="route_rows_per_rank mismatch"):
        RDEPPreparationPlan(
            tokens_per_rank=4096,
            world_size=8,
            num_experts=64,
            top_k=6,
            num_local_experts=8,
            route_rows_per_rank=1,
            pooled_route_rows=196608,
            expected_rows_per_expert=3072.0,
            routing_context_bytes=None,
        )
