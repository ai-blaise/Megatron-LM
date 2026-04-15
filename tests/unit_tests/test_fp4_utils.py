# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test FP4 utilities."""

import pytest
import torch

from megatron.core import fp4_utils
from megatron.core.fp4_utils import is_nvfp4tensor
from tests.unit_tests.test_utilities import Utils


class TestFP4Padding:
    """Test class for FP4 padding and utility functions."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_is_nvfp4tensor_detection(self):
        """Test is_nvfp4tensor() correctly detects NVFP4 tensors."""
        # Test with regular tensor - should return False
        regular_tensor = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        assert not is_nvfp4tensor(regular_tensor), (
            "Regular tensor should not be detected as NVFP4"
        )

        # Test with half tensor - should return False
        half_tensor = torch.randn(64, 128, dtype=torch.float16, device="cuda")
        assert not is_nvfp4tensor(half_tensor), (
            "Half tensor should not be detected as NVFP4"
        )

        # Test with float32 tensor - should return False
        float_tensor = torch.randn(32, 64, dtype=torch.float32, device="cuda")
        assert not is_nvfp4tensor(float_tensor), (
            "Float tensor should not be detected as NVFP4"
        )

    def test_get_fp4_align_size(self):
        """Test FP4 alignment size calculation."""
        # NVFP4 requires 128 alignment
        align_size = fp4_utils.get_fp4_align_size(fp4_utils.Fp4Recipe.NVFP4)
        assert align_size == 128, f"Expected alignment size 128, got {align_size}"

    def test_fp4_tensor_class_available(self):
        """Test that FP4 tensor class availability is correctly reported."""
        if fp4_utils.HAVE_TE_FP4_TENSOR_CLASS:
            assert fp4_utils.FP4_TENSOR_CLASS is not None
        else:
            # If TE doesn't have FP4 support, skip detailed tests
            pytest.skip("TransformerEngine does not support FP4 tensors")


class TestFP4UtilFunctions:
    """Additional FP4 utility function tests."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_fp4_recipe_enum(self):
        """Test FP4 recipe enum values."""
        assert hasattr(fp4_utils.Fp4Recipe, "NVFP4")

    def test_have_te_fp4_tensor_class(self):
        """Test that HAVE_TE_FP4_TENSOR_CLASS flag works correctly."""
        # This should be a boolean
        assert isinstance(fp4_utils.HAVE_TE_FP4_TENSOR_CLASS, bool)

    @pytest.mark.skipif(
        not fp4_utils.HAVE_TE_FP4_TENSOR_CLASS,
        reason="FP4 tensor class not available in TransformerEngine",
    )
    def test_nvfp4tensor_creation_and_detection(self):
        """Test NVFP4Tensor creation and detection (when available)."""
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor

        # Create an NVFP4 tensor
        original = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        nvfp4_tensor = NVFP4Tensor.to_nvfp4(original)

        # Verify it is detected as NVFP4
        assert is_nvfp4tensor(nvfp4_tensor), "NVFP4Tensor should be detected as NVFP4"

        # Verify regular tensors still return False
        assert not is_nvfp4tensor(original), "Original bf16 tensor should not be NVFP4"
