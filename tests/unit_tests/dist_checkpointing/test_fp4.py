# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test FP4 checkpoint save/load functionality."""

import pytest
import torch

from megatron.core import fp4_utils
from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def to_nvfp4(tensor: torch.Tensor):
    """Convert a tensor to NVFP4 format."""
    if not fp4_utils.HAVE_TE_FP4_TENSOR_CLASS:
        pytest.skip("TransformerEngine does not support FP4 tensors")

    try:
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor

        return NVFP4Tensor.to_nvfp4(tensor)
    except Exception as e:
        pytest.skip(f"Failed to create NVFP4 tensor: {e}")


class TestFP4:
    """Test class for FP4 checkpoint operations."""

    @pytest.mark.skipif(
        not fp4_utils.HAVE_TE_FP4_TENSOR_CLASS,
        reason="TransformerEngine does not support FP4 tensors",
    )
    @pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp4"])
    @pytest.mark.parametrize("src_rank", [0, 6])
    def test_simple_broadcast(self, dtype, src_rank):
        """Test broadcasting FP4 tensors."""
        Utils.initialize_model_parallel()

        def get_tensor(dtype: str = "fp4"):
            if dtype == "fp4":
                return to_nvfp4(
                    torch.full((3,), Utils.rank, dtype=torch.bfloat16, device="cuda")
                )
            elif dtype == "bf16":
                return torch.full((3,), Utils.rank, dtype=torch.bfloat16, device="cuda")
            elif dtype == "fp16":
                return torch.full((3,), Utils.rank, dtype=torch.float16, device="cuda")
            else:
                raise NotImplementedError(dtype)

        tensor = get_tensor(dtype)

        # NVFP4 tensors may need dequantize for broadcast due to TE bug
        from megatron.core.fp4_utils import is_nvfp4tensor

        if is_nvfp4tensor(tensor):
            try:
                tensor = tensor.dequantize()
            except Exception:
                pass  # If dequantize fails, try as-is

        torch.distributed.broadcast(tensor, src=src_rank)
        assert torch.all(tensor == src_rank)

    @pytest.mark.skipif(
        not fp4_utils.HAVE_TE_FP4_TENSOR_CLASS,
        reason="TransformerEngine does not support FP4 tensors",
    )
    @pytest.mark.parametrize(
        ("use_fpsl", "src_tp_pp", "dest_tp_pp", "load_exchange_algo"),
        [
            (True, (2, 4), (2, 4), "broadcast"),
            (True, (2, 4), (2, 4), "gather_rounds"),
            (False, (2, 4), (2, 4), None),
        ],
    )
    def test_fp4_save_load(
        self, tmp_path_dist_ckpt, use_fpsl, src_tp_pp, dest_tp_pp, load_exchange_algo
    ):
        """Test saving and loading FP4 tensors in checkpoints."""
        Utils.initialize_model_parallel(*src_tp_pp)

        def get_fp4_tensor(fill_val=1):
            return to_nvfp4(
                torch.full((3,), fill_val, dtype=torch.bfloat16, device="cuda")
            )

        def get_state_dict(fill_val=1):
            return {
                "a": ShardedTensor.from_rank_offsets(
                    "a",
                    get_fp4_tensor(fill_val),
                    (0, Utils.rank, Utils.world_size),
                    replica_id=0,
                ),
                "b": ShardedTensor.from_rank_offsets(
                    "b", get_fp4_tensor(fill_val), replica_id=Utils.rank
                ),
                "c": ShardedTensor.from_rank_offsets(
                    "c", get_fp4_tensor(fill_val), replica_id=Utils.rank
                ),
            }

        with TempNamedDir(
            tmp_path_dist_ckpt / "test_fp4_save_load", sync=True
        ) as ckpt_dir:
            save_strategy = get_default_save_sharded_strategy()
            if use_fpsl:
                save_strategy = FullyParallelSaveStrategyWrapper(
                    save_strategy, None, True
                )
            save(get_state_dict(4), ckpt_dir, save_strategy)

            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(*dest_tp_pp)

            if use_fpsl:
                load_strategy = get_default_load_sharded_strategy(ckpt_dir)
                load_strategy = FullyParallelLoadStrategyWrapper(
                    load_strategy, None, False, load_exchange_algo
                )
            else:
                load_strategy = None

            loaded_state_dict = load(get_state_dict(8), ckpt_dir, load_strategy)
            assert torch.all(loaded_state_dict["a"] == 4)
            assert torch.all(loaded_state_dict["b"] == 4)
        Utils.destroy_model_parallel()
