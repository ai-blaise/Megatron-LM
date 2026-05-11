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

    def test_torch_mm_thresholds_are_rank_aware(self, monkeypatch):
        monkeypatch.delenv("MEGATRON_GATED_NORM_TORCH_MM_MIN_TOKENS", raising=False)
        for suffix in ("R1", "R8", "R32", "R64"):
            monkeypatch.delenv(f"MEGATRON_GATED_NORM_TORCH_MM_{suffix}_MIN_TOKENS", raising=False)

        assert gated_norm._torch_mm_min_tokens(1) == 4096
        assert gated_norm._torch_mm_min_tokens(8) == 2048
        assert gated_norm._torch_mm_min_tokens(32) == 512
        assert gated_norm._torch_mm_min_tokens(64) == 256
        assert gated_norm._should_use_torch_mm(4096, 1, torch.bfloat16)
        assert not gated_norm._should_use_torch_mm(4096, 1, torch.float16)

    def test_cute_cuda_alloc_failure_falls_back_to_torch_mm(self, monkeypatch):
        class _FakeExt:
            def gated_norm_cute_fwd(self, *args, **kwargs):
                return gated_norm._CUDA_ERROR_MEMORY_ALLOCATION

        monkeypatch.setattr(gated_norm, "_load_cuda_kernel", lambda: _FakeExt())

        normed = torch.randn(2, 8, dtype=torch.bfloat16)
        w_down = torch.randn(4, 8, dtype=torch.bfloat16)
        w_up = torch.randn(8, 4, dtype=torch.bfloat16)
        output = torch.empty_like(normed)

        assert not gated_norm._gated_norm_cute_forward(
            normed,
            w_down,
            w_up,
            output,
            hidden_size=8,
            rank=4,
        )

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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for gated_norm")
    def test_apply_gated_norm_forced_torch_mm_bf16_matches_reference(self, monkeypatch):
        monkeypatch.setenv("MEGATRON_GATED_NORM_TORCH_MM_MIN_TOKENS", "0")
        torch.manual_seed(1234)

        normed = torch.randn(6, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w_down = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w_up = torch.randn(8, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        grad_output = torch.randn_like(normed)

        assert gated_norm._should_use_torch_mm(normed.numel() // normed.shape[-1], 4, normed.dtype)

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
