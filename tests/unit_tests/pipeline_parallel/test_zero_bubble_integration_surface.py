# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
from functools import partial
from types import SimpleNamespace

import pytest
import torch

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.model_parallel_config import (
    normalize_pipeline_parallel_schedule,
    validate_pipeline_parallel_schedule,
)
from megatron.training.arguments import (
    add_megatron_arguments,
    _derive_zero_bubble_v_virtual_pipeline_args,
    _validate_zero_bubble_schedule_args,
)
from megatron.training.yaml_arguments import (
    _derive_zero_bubble_v_virtual_pipeline_args as _derive_yaml_zbv_args,
)


@pytest.mark.parametrize(
    "selector,expected",
    [
        ("zero-bubble", "zero_bubble"),
        ("zerobubble", "zero_bubble"),
        ("zb", "zero_bubble"),
        ("zero-bubble-v", "zero_bubble_v"),
        ("zerobubble-v", "zero_bubble_v"),
        ("zbv", "zero_bubble_v"),
    ],
)
def test_zero_bubble_schedule_aliases_normalize(selector, expected):
    assert normalize_pipeline_parallel_schedule(selector) == expected


def test_megatron_argument_parser_registers_pipeline_schedule_once():
    parser = argparse.ArgumentParser()
    parser = add_megatron_arguments(parser)

    args, _ = parser.parse_known_args(["--pipeline-parallel-schedule", "zb"])

    assert args.pipeline_parallel_schedule == "zero_bubble"


@pytest.mark.parametrize(
    "selector,pp_size,vp_size",
    [
        ("zero_bubble", 2, None),
        ("zero_bubble_v", 2, 2),
    ],
)
def test_zero_bubble_schedule_config_validation_accepts_valid_shapes(
    selector, pp_size, vp_size
):
    assert validate_pipeline_parallel_schedule(selector, pp_size, vp_size) == selector


@pytest.mark.parametrize(
    "selector,pp_size,vp_size,match",
    [
        ("zero_bubble", 1, None, "requires pipeline_model_parallel_size > 1"),
        (
            "zero_bubble",
            2,
            2,
            "requires virtual_pipeline_model_parallel_size to be None",
        ),
        ("zero_bubble_v", 1, 2, "requires pipeline_model_parallel_size > 1"),
        ("zero_bubble_v", 2, None, "requires virtual_pipeline_model_parallel_size"),
        ("zero_bubble_v", 2, 4, "virtual_pipeline_model_parallel_size == 2"),
    ],
)
def test_zero_bubble_schedule_config_validation_rejects_invalid_shapes(
    selector, pp_size, vp_size, match
):
    with pytest.raises(ValueError, match=match):
        validate_pipeline_parallel_schedule(selector, pp_size, vp_size)


def test_model_parallel_config_accepts_zero_bubble_selectors():
    zb = ModelParallelConfig(
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float,
        pipeline_parallel_schedule="zero-bubble",
    )
    zbv = ModelParallelConfig(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float,
        pipeline_parallel_schedule="zbv",
    )

    assert zb.pipeline_parallel_schedule == "zero_bubble"
    assert zbv.pipeline_parallel_schedule == "zero_bubble_v"


def _zbv_args(**overrides):
    args = SimpleNamespace(
        pipeline_parallel_schedule="zero_bubble_v",
        pipeline_model_parallel_layout=None,
        num_layers_per_virtual_pipeline_stage=None,
        num_virtual_stages_per_pipeline_rank=None,
        pipeline_model_parallel_size=4,
        transformer_pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=None,
        num_layers=16,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
        untie_embeddings_and_output_weights=True,
        gradient_accumulation_fusion=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_zero_bubble_v_cli_validation_derives_two_virtual_stages():
    args = _zbv_args()

    _derive_zero_bubble_v_virtual_pipeline_args(args)
    _validate_zero_bubble_schedule_args(args)

    assert args.virtual_pipeline_model_parallel_size == 2
    assert args.num_layers_per_virtual_pipeline_stage == 2


def test_zero_bubble_v_cli_validation_rejects_explicit_virtual_shape():
    args = _zbv_args(num_virtual_stages_per_pipeline_rank=2)

    with pytest.raises(AssertionError, match="derives"):
        _derive_zero_bubble_v_virtual_pipeline_args(args)


@pytest.mark.parametrize("flag", ["overlap_grad_reduce", "overlap_param_gather"])
def test_zero_bubble_validation_rejects_unsupported_overlap_combinations(flag):
    args = _zbv_args(pipeline_parallel_schedule="zero_bubble")
    setattr(args, flag, True)

    with pytest.raises(AssertionError, match=flag.replace("_", "-")):
        _validate_zero_bubble_schedule_args(args)


def test_zero_bubble_validation_requires_untied_embeddings():
    with pytest.raises(AssertionError, match="untie"):
        _validate_zero_bubble_schedule_args(
            _zbv_args(
                pipeline_parallel_schedule="zero_bubble",
                untie_embeddings_and_output_weights=False,
            )
        )


def test_zero_bubble_validation_allows_non_fused_weight_grad_path():
    _validate_zero_bubble_schedule_args(
        _zbv_args(
            pipeline_parallel_schedule="zero_bubble",
            gradient_accumulation_fusion=False,
        )
    )


def test_rank_zero_v_training_log_prints_from_rank_zero(monkeypatch):
    from megatron.training import training

    printed = []
    monkeypatch.setattr(
        training, "print_rank_0", lambda message: printed.append(("rank0", message))
    )
    monkeypatch.setattr(
        training, "print_rank_last", lambda message: printed.append(("last", message))
    )

    training._print_pipeline_schedule_training_log(
        SimpleNamespace(pipeline_parallel_schedule="zero_bubble_v"), "loss"
    )
    training._print_pipeline_schedule_training_log(
        SimpleNamespace(pipeline_parallel_schedule="1f1b"), "legacy"
    )

    assert printed == [("rank0", "loss"), ("last", "legacy")]


def test_zero_bubble_v_yaml_validation_derives_two_virtual_stages():
    args = SimpleNamespace(
        model_parallel=SimpleNamespace(
            pipeline_parallel_schedule="zero_bubble_v",
            pipeline_model_parallel_size=4,
            transformer_pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=None,
        ),
        language_model=SimpleNamespace(num_layers=16),
        num_layers_per_virtual_pipeline_stage=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
    )

    _derive_yaml_zbv_args(args)

    assert args.model_parallel.virtual_pipeline_model_parallel_size == 2
    assert args.num_layers_per_virtual_pipeline_stage == 2


@pytest.mark.parametrize(
    "selector,pp_size,vp_size",
    [
        ("zero_bubble", 2, None),
        ("zero_bubble_v", 2, 2),
    ],
)
def test_zero_bubble_schedule_selection_returns_runtime(
    selector, pp_size, vp_size
):
    selected = schedule.get_forward_backward_func(
        pp_size=pp_size,
        vp_size=vp_size,
        pipeline_parallel_schedule=selector,
    )

    assert selected.__name__ == f"forward_backward_pipelining_with_{selector}"


def test_weight_grad_store_drains_deferred_tasks_in_order():
    from megatron.core.zbpp_utils import WeightGradStore

    events = []

    def pre_process(name, async_op=True):
        events.append(("pre", name, async_op))
        return (name,)

    def process(name):
        events.append(("process", name))

    WeightGradStore.reset(num_chunks=1)
    with WeightGradStore.set_split_bw(True):
        WeightGradStore.put(None, partial(pre_process, "a"), process)
        WeightGradStore.put(None, partial(pre_process, "b"), process)
        WeightGradStore.flush()

    assert WeightGradStore.queue_size() == 1
    WeightGradStore.pop()

    assert events == [
        ("pre", "a", False),
        ("process", "a"),
        ("pre", "b", False),
        ("process", "b"),
    ]


def _runtime_config(**overrides):
    config = SimpleNamespace(
        overlap_p2p_comm=False,
        use_ring_exchange_p2p=False,
        overlap_moe_expert_parallel_comm=False,
        fine_grained_activation_offloading=False,
        cpu_offloading=False,
        cpu_offload=False,
        variable_seq_lengths=False,
        mtp_standalone=False,
        num_microbatches_with_partial_activation_checkpoints=None,
        param_sync_func=None,
        virtual_pipeline_model_parallel_size=None,
        pipeline_dtype=torch.float,
        overlap_grad_reduce=False,
        delay_wgrad_compute=False,
        transformer_impl="local",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("overlap_grad_reduce", True, "overlapped DDP gradient reduction"),
        ("delay_wgrad_compute", True, "delayed WGRAD"),
        ("transformer_impl", "transformer_engine", "transformer_engine"),
        ("use_ring_exchange_p2p", True, "ring-exchange P2P"),
    ],
)
def test_zero_bubble_runtime_rejects_unwired_paths(field, value, match):
    from megatron.core.pipeline_parallel.zerobubble.runtime import _validate_common_config

    with pytest.raises(ValueError, match=match):
        _validate_common_config(
            _runtime_config(**{field: value}),
            "zero_bubble",
            allow_vp=False,
            forward_only=False,
        )


def test_zero_bubble_v_runtime_requires_exactly_one_matching_p2p_tensor_shape():
    from megatron.core.pipeline_parallel.zerobubble.runtime import (
        _validate_single_p2p_tensor_shapes,
    )

    assert _validate_single_p2p_tensor_shapes(
        [(4, 2, 8)], [(4, 2, 8)], "zero_bubble_v"
    ) == (4, 2, 8)
    with pytest.raises(ValueError, match="exactly one pipeline tensor"):
        _validate_single_p2p_tensor_shapes(
            [(4, 2, 8), (4, 2, 8)], [(4, 2, 8)], "zero_bubble_v"
        )
    with pytest.raises(ValueError, match="matching send and receive"):
        _validate_single_p2p_tensor_shapes([(4, 2, 8)], [(4, 2, 16)], "zero_bubble_v")


def test_zero_bubble_v_runtime_rejects_mismatched_send_tensor_shape():
    from megatron.core.pipeline_parallel.zerobubble.runtime import _validate_p2p_send_tensor_shape

    _validate_p2p_send_tensor_shape(torch.empty(4, 2, 8), (4, 2, 8), "zero_bubble_v")
    with pytest.raises(ValueError, match=r"expected pipeline tensor shape .* got"):
        _validate_p2p_send_tensor_shape(torch.empty(4, 2, 16), (4, 2, 8), "zero_bubble_v")
