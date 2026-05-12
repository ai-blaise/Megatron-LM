# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_dualpipe_v_schedule_module():
    repo_root = Path(__file__).parents[3]
    module_path = repo_root / "megatron/core/pipeline_parallel/dualpipe_v_schedule.py"
    spec = importlib.util.spec_from_file_location("dualpipe_v_schedule_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dualpipe_v_schedule = _load_dualpipe_v_schedule_module()


def _count_step_ops(steps):
    counts = {
        "F0": 0,
        "F1": 0,
        "B0": 0,
        "B1": 0,
    }
    for step in steps:
        if step.op == "F":
            counts[f"F{step.phase}"] += 1
        elif step.op == "B":
            counts[f"B{step.phase}"] += 1
        elif step.op == "FB":
            counts[f"F{step.phase}"] += 1
            counts[f"B{1 - step.phase}"] += 1
    return counts


def test_dualpipe_v_reference_step_order_counts_all_work():
    seen_steps = set()
    for rank in range(4):
        steps = list(
            dualpipe_v_schedule.iter_dualpipe_v_steps(
                pipeline_model_parallel_size=4,
                pipeline_model_parallel_rank=rank,
                num_microbatches=8,
            )
        )

        assert _count_step_ops(steps) == {"F0": 8, "F1": 8, "B0": 8, "B1": 8}
        seen_steps.update(step.step for step in steps)
    assert seen_steps == set(range(1, 9))


def test_dualpipe_v_mapping_marks_reverse_second_phase():
    mapping = dualpipe_v_schedule.build_dualpipe_v_mapping(4, 2, 8)

    assert mapping.runtime_backed
    assert mapping.forward_ranks(0) == (0, 1, 2, 3)
    assert mapping.forward_ranks(1) == (3, 2, 1, 0)
    assert mapping.stage(rank=3, phase=0).emits_bridge_output
    assert mapping.stage(rank=3, phase=1).receives_bridge_input
    assert mapping.stage(rank=0, phase=1).computes_loss


class _FakeTensor:
    def __init__(self, name="tensor"):
        self.name = name
        self.grad = None
        self.requires_grad = True
        self._base = None

    def detach(self):
        return self

    def requires_grad_(self):
        return self

    def __iadd__(self, other):
        return self


class _FakeGroup:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _FakeReq:
    def wait(self):
        return None


def _runtime_config():
    return SimpleNamespace(
        overlap_p2p_comm=False,
        overlap_moe_expert_parallel_comm=False,
        variable_seq_lengths=False,
        mtp_standalone=False,
        use_ring_exchange_p2p=False,
        pipeline_dtype=object(),
        virtual_pipeline_model_parallel_size=2,
        timers=None,
        finalize_model_grads_func=None,
        no_sync_func=contextlib.nullcontext,
        grad_sync_func=None,
        param_sync_func=None,
        hidden_size=4,
        sequence_parallel=False,
        enable_hyper_connections=False,
        deallocate_pipeline_outputs=False,
        calculate_per_token_loss=False,
        fine_grained_activation_offloading=False,
    )


def _runtime_groups(rank):
    pp_group = _FakeGroup(size=2, rank=rank)
    p2p_communicator = SimpleNamespace(
        pp_group=pp_group,
        prev_rank=rank - 1,
        next_rank=rank + 1,
        config=_runtime_config(),
    )
    size_one_group = _FakeGroup(size=1, rank=0)
    pg_collection = SimpleNamespace(
        tp=size_one_group,
        cp=size_one_group,
        embd=None,
        pos_embd=None,
        pp=pp_group,
        dp_cp=size_one_group,
    )
    return p2p_communicator, pg_collection


def test_dualpipe_v_selector_is_explicit():
    pytest.importorskip("torch")
    import megatron.core.pipeline_parallel.schedules as schedule

    assert (
        schedule.get_forward_backward_func(
            pp_size=2,
            vp_size=2,
            pipeline_parallel_schedule="dualpipe_v",
        )
        == schedule.forward_backward_pipelining_with_dualpipe_v
    )
    with pytest.raises(ValueError, match="virtual_pipeline_model_parallel_size == 2"):
        schedule.get_forward_backward_func(
            pp_size=2,
            vp_size=None,
            pipeline_parallel_schedule="dualpipe_v",
        )


def test_dualpipe_v_runtime_mock_executes_reference_work_counts(monkeypatch):
    pytest.importorskip("torch")
    import megatron.core.pipeline_parallel.schedules as schedule

    for rank in (0, 1):
        events = []
        sends = []
        deallocations = []
        p2p_communicator, pg_collection = _runtime_groups(rank)
        config = p2p_communicator.config

        def fake_empty(*args, **kwargs):
            return _FakeTensor("recv")

        def fake_zeros(*args, **kwargs):
            return _FakeTensor("tokens")

        def fake_forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size,
            collect_non_loss_data=False,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=False,
            current_microbatch=None,
            vp_stage=None,
            is_last_stage=True,
        ):
            del (
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                cp_group_size,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                is_first_microbatch,
                is_last_stage,
            )
            events.append(("F", vp_stage, current_microbatch))
            return _FakeTensor(f"out-{vp_stage}-{current_microbatch}"), _FakeTensor("tok")

        def fake_backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
            del input_tensor, output_tensor_grad, model_type, config
            _, phase, microbatch = output_tensor.name.split("-")
            events.append(("B", int(phase), int(microbatch)))
            return _FakeTensor(f"grad-{phase}-{microbatch}")

        def fake_p2p_op(op, tensor, peer, group):
            sends.append((op.__name__, getattr(tensor, "name", None), peer, group.rank()))
            return (op, tensor, peer, group)

        monkeypatch.setattr(schedule, "get_model_config", lambda model: config)
        monkeypatch.setattr(schedule, "get_model_type", lambda model: "unit-test")
        monkeypatch.setattr(schedule, "forward_step", fake_forward_step)
        monkeypatch.setattr(schedule, "backward_step", fake_backward_step)
        monkeypatch.setattr(
            schedule,
            "deallocate_output_tensor",
            lambda tensor, deallocate_pipeline_outputs=False: deallocations.append(tensor.name),
        )
        monkeypatch.setattr(schedule.torch, "empty", fake_empty)
        monkeypatch.setattr(schedule.torch, "zeros", fake_zeros)
        monkeypatch.setattr(schedule.torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(schedule.torch.distributed, "P2POp", fake_p2p_op)
        monkeypatch.setattr(
            schedule.torch.distributed,
            "batch_isend_irecv",
            lambda ops: [_FakeReq() for _ in ops],
        )

        schedule.forward_backward_pipelining_with_dualpipe_v(
            forward_step_func=lambda data, model: None,
            data_iterator=[iter(()), iter(())],
            model=[object(), object()],
            num_microbatches=4,
            seq_length=2,
            micro_batch_size=1,
            forward_only=False,
            p2p_communicator=p2p_communicator,
            pg_collection=pg_collection,
        )

        assert _count_step_ops(
            SimpleNamespace(op=event[0], phase=event[1]) for event in events
        ) == {"F0": 4, "F1": 4, "B0": 4, "B1": 4}
        assert sends
        if rank == 1:
            assert any(name.startswith("out-0-") for name in deallocations)


def test_dualpipe_v_forward_only_mock_runs_v_path(monkeypatch):
    pytest.importorskip("torch")
    import megatron.core.pipeline_parallel.schedules as schedule

    for rank in (0, 1):
        events = []
        sends = []
        p2p_communicator, pg_collection = _runtime_groups(rank)
        config = p2p_communicator.config

        def fake_empty(*args, **kwargs):
            return _FakeTensor("recv")

        def fake_zeros(*args, **kwargs):
            return _FakeTensor("tokens")

        def fake_forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size,
            collect_non_loss_data=False,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=False,
            current_microbatch=None,
            vp_stage=None,
            is_last_stage=True,
        ):
            del (
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                cp_group_size,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                is_first_microbatch,
                is_last_stage,
            )
            events.append(("F", vp_stage, current_microbatch))
            return _FakeTensor(f"out-{vp_stage}-{current_microbatch}"), _FakeTensor("tok")

        def fake_p2p_op(op, tensor, peer, group):
            sends.append((op.__name__, getattr(tensor, "name", None), peer, group.rank()))
            return (op, tensor, peer, group)

        monkeypatch.setattr(schedule, "get_model_config", lambda model: config)
        monkeypatch.setattr(schedule, "get_model_type", lambda model: "unit-test")
        monkeypatch.setattr(schedule, "forward_step", fake_forward_step)
        monkeypatch.setattr(
            schedule,
            "backward_step",
            lambda *args, **kwargs: pytest.fail("forward_only called backward_step"),
        )
        monkeypatch.setattr(schedule.torch, "empty", fake_empty)
        monkeypatch.setattr(schedule.torch, "zeros", fake_zeros)
        monkeypatch.setattr(schedule.torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(schedule.torch.distributed, "P2POp", fake_p2p_op)
        monkeypatch.setattr(
            schedule.torch.distributed,
            "batch_isend_irecv",
            lambda ops: [_FakeReq() for _ in ops],
        )

        schedule.forward_backward_pipelining_with_dualpipe_v(
            forward_step_func=lambda data, model: None,
            data_iterator=[iter(()), iter(())],
            model=[object(), object()],
            num_microbatches=4,
            seq_length=2,
            micro_batch_size=1,
            forward_only=True,
            p2p_communicator=p2p_communicator,
            pg_collection=pg_collection,
        )

        assert events == [
            ("F", 0, 0),
            ("F", 1, 0),
            ("F", 0, 1),
            ("F", 1, 1),
            ("F", 0, 2),
            ("F", 1, 2),
            ("F", 0, 3),
            ("F", 1, 3),
        ]
        sent_names = [name for op_name, name, _, _ in sends if op_name == "isend"]
        if rank == 0:
            assert all(name.startswith("out-0-") for name in sent_names)
        else:
            assert all(name.startswith("out-1-") for name in sent_names)
