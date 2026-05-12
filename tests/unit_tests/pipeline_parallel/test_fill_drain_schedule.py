# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
from types import SimpleNamespace

import pytest
import torch

from megatron.core.pipeline_parallel import schedules


class _FakePipelineGroup:
    def __init__(self, rank=1, size=3):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _FakeP2PCommunicator:
    def __init__(self):
        self.config = None
        self.pp_group = _FakePipelineGroup()
        self.sent_forward = []
        self.sent_backward = []

    def recv_forward(self, tensor_shapes, is_first_stage):
        return [torch.tensor(float(len(self.sent_forward)), requires_grad=True)]

    def send_forward(self, output_tensor, is_last_stage):
        self.sent_forward.append((output_tensor, is_last_stage))

    def recv_backward(self, tensor_shapes, is_last_stage):
        return [torch.tensor(1.0)]

    def send_backward(self, input_tensor_grad, is_first_stage):
        self.sent_backward.append((input_tensor_grad, is_first_stage))


def test_gpipe_fill_drain_runs_all_forwards_before_reverse_backwards(monkeypatch):
    events = []
    communicator = _FakeP2PCommunicator()

    config = SimpleNamespace(
        overlap_p2p_comm=False,
        finalize_model_grads_func=None,
        timers=None,
        no_sync_func=lambda: _recorded_no_sync(events),
        grad_sync_func=None,
        num_microbatches_with_partial_activation_checkpoints=None,
        deallocate_pipeline_outputs=False,
        calculate_per_token_loss=False,
        fine_grained_activation_offloading=False,
    )
    pg_collection = SimpleNamespace(
        tp=None,
        cp=SimpleNamespace(size=lambda: 1),
        embd=None,
        pos_embd=None,
        pp=None,
        dp_cp=None,
    )
    model = torch.nn.Linear(1, 1)

    real_zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        kwargs.pop("device", None)
        return real_zeros(*args, **kwargs)

    def fake_forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        *,
        current_microbatch,
        **kwargs,
    ):
        events.append(("forward", current_microbatch))
        output = torch.tensor(float(current_microbatch), requires_grad=True)
        return [output], torch.tensor(1)

    def fake_backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
        events.append(("backward", int(output_tensor[0].item())))
        return [torch.tensor(1.0)]

    monkeypatch.setattr(schedules.torch, "zeros", cpu_zeros)
    monkeypatch.setattr(schedules, "get_model_config", lambda model: config)
    monkeypatch.setattr(schedules, "get_model_type", lambda model: "encoder_or_decoder")
    monkeypatch.setattr(schedules, "get_tensor_shapes", lambda **kwargs: [(1, 1, 1)])
    monkeypatch.setattr(schedules, "is_pp_first_stage", lambda pp_group: pp_group.rank() == 0)
    monkeypatch.setattr(
        schedules, "is_pp_last_stage", lambda pp_group: pp_group.rank() == pp_group.size() - 1
    )
    monkeypatch.setattr(schedules, "forward_step", fake_forward_step)
    monkeypatch.setattr(schedules, "backward_step", fake_backward_step)
    monkeypatch.setattr(schedules, "deallocate_output_tensor", lambda *args, **kwargs: None)

    result = schedules.forward_backward_pipelining_with_fill_drain(
        forward_step_func=lambda data_iterator, model: None,
        data_iterator=iter(()),
        model=model,
        num_microbatches=3,
        seq_length=1,
        micro_batch_size=1,
        p2p_communicator=communicator,
        pg_collection=pg_collection,
    )

    assert result == []
    assert events == [
        "disable_grad_sync",
        ("forward", 0),
        ("forward", 1),
        ("forward", 2),
        ("backward", 2),
        ("backward", 1),
        "enable_grad_sync",
        ("backward", 0),
    ]
    assert len(communicator.sent_forward) == 3
    assert len(communicator.sent_backward) == 3


def test_gpipe_fill_drain_rejects_overlap_p2p(monkeypatch):
    config = SimpleNamespace(overlap_p2p_comm=True)
    monkeypatch.setattr(schedules, "get_model_config", lambda model: config)

    with pytest.raises(ValueError, match="does not support overlapping p2p communication"):
        schedules.forward_backward_pipelining_with_fill_drain(
            forward_step_func=lambda data_iterator, model: None,
            data_iterator=iter(()),
            model=torch.nn.Linear(1, 1),
            num_microbatches=1,
            seq_length=1,
            micro_batch_size=1,
            p2p_communicator=_FakeP2PCommunicator(),
            pg_collection=SimpleNamespace(),
        )


@contextlib.contextmanager
def _recorded_no_sync(events):
    events.append("disable_grad_sync")
    try:
        yield
    finally:
        events.append("enable_grad_sync")
