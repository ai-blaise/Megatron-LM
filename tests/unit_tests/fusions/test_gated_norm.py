# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

import megatron.core.fusions.gated_norm as gated_norm
from megatron.core.fusions.gated_norm import apply_gated_norm


class TestGatedNorm:
    def test_apply_gated_norm_uses_direct_triton_kernel_structure(self):
        kernel_names = {
            name
            for name in gated_norm.__dict__
            if name.startswith("_gated_norm_") and name.endswith("_kernel")
        }

        assert kernel_names == {
            "_gated_norm_forward_kernel",
            "_gated_norm_backward_kernel",
        }

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for gated_norm")
    def test_apply_gated_norm_forward_backward_matches_reference(self, dtype):
        torch.manual_seed(1234)

        normed = torch.randn(6, 8, device="cuda", dtype=dtype, requires_grad=True)
        w_down = torch.randn(4, 8, device="cuda", dtype=dtype, requires_grad=True)
        w_up = torch.randn(8, 4, device="cuda", dtype=dtype, requires_grad=True)
        grad_output = torch.randn_like(normed)

        out = apply_gated_norm(normed, w_down, w_up)
        ref_normed = normed.detach().clone().requires_grad_(True)
        ref_w_down = w_down.detach().clone().requires_grad_(True)
        ref_w_up = w_up.detach().clone().requires_grad_(True)
        ref_out = (ref_normed.reshape(-1, 8) * torch.sigmoid(
            F.silu(ref_normed.reshape(-1, 8) @ ref_w_down.t()) @ ref_w_up.t()
        )).reshape_as(normed)

        grads = torch.autograd.grad(
            outputs=out,
            inputs=(normed, w_down, w_up),
            grad_outputs=grad_output,
        )
        ref_grads = torch.autograd.grad(
            outputs=ref_out,
            inputs=(ref_normed, ref_w_down, ref_w_up),
            grad_outputs=grad_output,
        )

        assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)
        for grad, ref_grad in zip(grads, ref_grads, strict=True):
            assert torch.allclose(grad, ref_grad, atol=1e-2, rtol=1e-2)

    def test_apply_gated_norm_requires_cuda(self):
        with pytest.raises(RuntimeError, match="requires CUDA tensors"):
            apply_gated_norm(
                torch.randn(2, 4),
                torch.randn(2, 4),
                torch.randn(4, 2),
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for gated_norm")
    def test_apply_gated_norm_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="w_down must have shape"):
            apply_gated_norm(
                torch.randn(2, 4, device="cuda"),
                torch.randn(2, 5, device="cuda"),
                torch.randn(4, 2, device="cuda"),
            )
