import torch

from megatron.training.training import _adjust_tensor_shapes_for_sft_packed
from megatron.training.utils import _prepare_thd_packed_batch_for_tp_broadcast


def test_prepare_thd_packed_batch_combines_microbatches():
    batch = {
        "tokens": torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=torch.int64),
        "labels": torch.tensor([[1, 2, 3, 4], [11, 12, 13, 14]], dtype=torch.int64),
        "loss_mask": torch.ones((2, 4), dtype=torch.float32),
        "position_ids": torch.tensor([[0, 1, 0, 1], [0, 0, 1, 2]], dtype=torch.int64),
        "padding_mask": torch.tensor([[False, False, True, True], [False, True, True, True]]),
        "cu_seqlens": torch.tensor([[0, 2, 4], [0, 1, 4]], dtype=torch.int32),
        "max_seqlen": torch.tensor([2, 3], dtype=torch.int32),
    }

    out = _prepare_thd_packed_batch_for_tp_broadcast(batch)

    assert out["tokens"].shape == (1, 8)
    assert out["tokens"].tolist() == [[0, 1, 2, 3, 10, 11, 12, 13]]
    assert out["labels"].tolist() == [[1, 2, 3, 4, 11, 12, 13, 14]]
    assert out["position_ids"].tolist() == [[0, 1, 0, 1, 0, 0, 1, 2]]
    assert out["padding_mask"].tolist() == [[False, False, True, True, False, True, True, True]]
    assert out["cu_seqlens"].tolist() == [0, 2, 4, 5, 8]
    assert out["max_seqlen"].shape == (1,)
    assert out["max_seqlen"].item() == 3


def test_adjust_tensor_shapes_for_sft_packed_flattens_pipeline_microbatch_dim():
    recv, send = _adjust_tensor_shapes_for_sft_packed(
        [(8192, 2, 7168)],
        [(8192, 2, 7168)],
    )

    assert recv == [(16384, 1, 7168)]
    assert send == [(16384, 1, 7168)]
