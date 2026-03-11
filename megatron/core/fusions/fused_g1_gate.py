# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import torch

_g1_gate_cuda = None


def _load_kernel():
    """JIT-compile the G1 gate CUDA kernel on first use."""
    global _g1_gate_cuda
    if _g1_gate_cuda is not None:
        return _g1_gate_cuda

    from torch.utils.cpp_extension import load

    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    _g1_gate_cuda = load(
        name="g1_gate_cuda",
        sources=[
            os.path.join(kernel_dir, "fused_g1_gate_wrapper.cpp"),
            os.path.join(kernel_dir, "fused_g1_gate.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _g1_gate_cuda


def _g1_gate_fwd(linear_out: torch.Tensor, attn_out: torch.Tensor):
    """Launch the fused G1 gate forward kernel.

    Computes: output = attn_out * sigmoid(linear_out)
              gate   = sigmoid(linear_out)

    Args:
        linear_out: BF16 tensor (flattened or contiguous).
        attn_out:   BF16 tensor, same shape as linear_out.

    Returns:
        (output, gate) — both BF16 tensors with the same shape as the inputs.
    """
    assert linear_out.is_cuda and attn_out.is_cuda, "Inputs must be CUDA tensors"
    assert linear_out.dtype == torch.bfloat16, f"Expected bf16, got {linear_out.dtype}"
    assert attn_out.dtype == torch.bfloat16, f"Expected bf16, got {attn_out.dtype}"
    assert linear_out.shape == attn_out.shape, "Shape mismatch"

    linear_out = linear_out.contiguous()
    attn_out = attn_out.contiguous()

    output = torch.empty_like(attn_out)
    gate = torch.empty_like(linear_out)

    ext = _load_kernel()
    ext.g1_gate_fwd(linear_out, attn_out, output, gate, linear_out.numel())
    return output, gate


def _g1_gate_bwd(
    grad_output: torch.Tensor, attn_out: torch.Tensor, gate: torch.Tensor
):
    """Launch the fused G1 gate backward kernel.

    Args:
        grad_output: Gradient w.r.t. output (BF16).
        attn_out:    The ungated values saved from forward (BF16).
        gate:        sigmoid(linear_out) saved from forward (BF16).

    Returns:
        (d_attn_out, d_linear_out) — both BF16 tensors.
    """
    grad_output = grad_output.contiguous()

    d_attn_out = torch.empty_like(attn_out)
    d_linear_out = torch.empty_like(gate)

    ext = _load_kernel()
    ext.g1_gate_bwd(grad_output, attn_out, gate, d_attn_out, d_linear_out, gate.numel())
    return d_attn_out, d_linear_out


class G1GateFunction(torch.autograd.Function):
    """Fused sigmoid gating: output = attn_out * sigmoid(linear_out).

    Both forward and backward use custom CUDA kernels with vectorised BF16 loads.
    """

    @staticmethod
    def forward(ctx, linear_out, attn_out):
        output, gate = _g1_gate_fwd(linear_out, attn_out)
        ctx.save_for_backward(attn_out, gate)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        attn_out, gate = ctx.saved_tensors
        d_attn_out, d_linear_out = _g1_gate_bwd(grad_output, attn_out, gate)
        return d_linear_out, d_attn_out


def g1_gate_impl(linear_out: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
    """Fused G1 sigmoid gating with autograd support.

    Computes ``attn_out * sigmoid(linear_out)`` using a custom BF16 CUDA kernel
    for the forward pass.  The backward pass is computed analytically.

    Args:
        linear_out: Gating input tensor (BF16, CUDA).
        attn_out:   Value tensor to be gated (BF16, CUDA, same shape).

    Returns:
        Gated output tensor (BF16, same shape as inputs).
    """
    return G1GateFunction.apply(linear_out, attn_out)
