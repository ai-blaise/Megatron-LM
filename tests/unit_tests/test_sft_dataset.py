# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test SFT Dataset."""

import pytest
import os
import sys
from types import SimpleNamespace

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from megatron.training.datasets.sft_dataset import IGNORE_INDEX, SFTDataset, SFTLowLevelDataset


class TestSFTLowLevelDataset:
    """Test class for SFTLowLevelDataset."""

    def setup_method(self, method):
        """Set up test fixtures."""
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        self.parquet_path = os.path.join(
            self.test_data_dir, "sft_test_conversations.parquet"
        )

    def test_sft_parquet_with_conversations(self):
        """Test SFTLowLevelDataset loads parquet with conversations column."""
        dataset = SFTLowLevelDataset(self.parquet_path)
        assert len(dataset) == 1, f"Expected 1 sample, got {len(dataset)}"

        sample = dataset[0]
        assert isinstance(sample, list), f"Expected list, got {type(sample)}"
        assert len(sample) == 3, f"Expected 3 messages, got {len(sample)}"

        # Verify conversation structure
        assert sample[0]["role"] == "system", (
            f"Expected system role, got {sample[0]['role']}"
        )
        assert sample[1]["role"] == "user", (
            f"Expected user role, got {sample[1]['role']}"
        )
        assert sample[2]["role"] == "assistant", (
            f"Expected assistant role, got {sample[2]['role']}"
        )

        # Verify content is present
        assert "content" in sample[0], "Missing content in first message"
        assert "content" in sample[1], "Missing content in second message"
        assert "content" in sample[2], "Missing content in third message"


class _FakeTokenizer:
    eod = 2
    pad = 0

    def tokenize_conversation(self, conversation, return_target, add_generation_prompt):
        del conversation, return_target, add_generation_prompt
        return (
            np.asarray([11, 0, 12, 13], dtype=np.int64),
            np.asarray([IGNORE_INDEX, IGNORE_INDEX, 12, 13], dtype=np.int64),
        )


def test_sft_dataset_returns_true_padding_mask_not_loss_mask_or_pad_id():
    dataset = SFTDataset.__new__(SFTDataset)
    dataset.dataset = [[
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]]
    dataset.indices = np.asarray([0], dtype=np.int64)
    dataset.config = SimpleNamespace(
        tokenizer=_FakeTokenizer(),
        sequence_length=8,
        hybrid_context_parallel=False,
        data_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel_size=1,
        reset_position_ids=False,
        create_attention_mask=False,
        reset_attention_mask=False,
    )

    sample = dataset[0]

    assert sample["tokens"].tolist() == [11, 0, 12, 13, 0, 0, 0, 0]
    assert sample["padding_mask"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert sample["padding_mask"][1].item() is False
    assert sample["loss_mask"].tolist()[:2] == [0.0, 1.0]
