# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


def _ensure_cuda_home(monkeypatch):
    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home is None:
        candidate = (
            Path(__file__).resolve().parents[3]
            / ".venv"
            / "lib"
            / "python3.12"
            / "site-packages"
            / "nvidia"
            / "cu13"
        )
        if candidate.exists():
            cuda_home = str(candidate)
    if cuda_home is None or not (Path(cuda_home) / "bin" / "nvcc").exists():
        pytest.skip("G1 CUDA extension test requires CUDA_HOME with nvcc")

    monkeypatch.setenv("CUDA_HOME", cuda_home)
    monkeypatch.setenv("CUDA_PATH", cuda_home)
    monkeypatch.setenv("PATH", f"{cuda_home}/bin:{os.getenv('PATH', '')}")

    import torch.utils.cpp_extension as cpp_extension

    cpp_extension.CUDA_HOME = cuda_home


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for fused G1")
def test_fused_g1_gate_forward_backward_matches_bf16_reference(monkeypatch):
    _ensure_cuda_home(monkeypatch)

    from megatron.core.fusions.fused_g1_gate import g1_gate_impl

    torch.manual_seed(1234)
    linear = torch.randn(17, 19, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    attn = torch.randn_like(linear, requires_grad=True)
    grad_output = torch.randn_like(linear)

    output = g1_gate_impl(linear, attn)
    gate = torch.sigmoid(linear.float()).to(torch.bfloat16)
    expected_output = (attn.float() * gate.float()).to(torch.bfloat16)

    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)

    output.backward(grad_output)
    expected_d_attn = (grad_output.float() * gate.float()).to(torch.bfloat16)
    expected_d_linear = (
        grad_output.float() * attn.float() * gate.float() * (1.0 - gate.float())
    ).to(torch.bfloat16)

    torch.testing.assert_close(attn.grad, expected_d_attn, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(linear.grad, expected_d_linear, rtol=2e-2, atol=2e-2)
