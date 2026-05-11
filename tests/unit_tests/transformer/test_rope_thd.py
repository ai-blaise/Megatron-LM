import torch

from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_thd


class _SingleRankGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


def test_thd_rope_accepts_singleton_batch_dimension():
    t = torch.randn(4, 1, 2, 4)
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    freqs = torch.randn(4, 1, 1, 4)

    out = _apply_rotary_pos_emb_thd(t, cu_seqlens, freqs, cp_group=_SingleRankGroup())
    expected = _apply_rotary_pos_emb_thd(
        t.squeeze(1), cu_seqlens, freqs, cp_group=_SingleRankGroup()
    ).unsqueeze(1)

    assert out.shape == t.shape
    torch.testing.assert_close(out, expected)
