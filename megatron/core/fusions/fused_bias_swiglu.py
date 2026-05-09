# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


# pylint: disable=missing-function-docstring, missing-class-docstring

import os

import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.utils import nvtx_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
    triton = None
    tl = None

###### BIAS SWIGLU FUSION/ NO AUTOGRAD ################


@jit_fuser
def swiglu(y):
    """Performs SwiGLU (Swish-Gated Linear Unit) activation function.

    Args:
        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.

    Returns:
        torch.Tensor: Result of SwiGLU activation: SiLU(y1) * y2, where y1, y2 are the split halves.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


@jit_fuser
def bias_swiglu(y, bias):
    """Performs SwiGLU activation with bias addition.

    Args:
        y (torch.Tensor): Input tensor.
        bias (torch.Tensor): Bias tensor to be added to input.

    Returns:
        torch.Tensor: Result of bias addition followed by SwiGLU activation.
    """
    y = y + bias
    return swiglu(y)


def _weighted_swiglu_fuser(func):
    mode = os.getenv("MEGATRON_WEIGHTED_SWIGLU_FUSER", "eager").lower()
    if mode in ("0", "off", "false", "no", "none", "eager", "triton", "triton_kernel", "kernel"):
        return func
    if mode in ("dynamic", "dynamic_compile"):
        try:
            return torch.compile(dynamic=True)(func)
        except TypeError:
            return torch.compile(func, dynamic=True)
    return jit_fuser(func)


def _env_enabled(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in ("0", "false", "off", "no")


def _weighted_swiglu_triton_enabled() -> bool:
    mode = os.getenv("MEGATRON_WEIGHTED_SWIGLU_FUSER", "eager").lower()
    return mode in ("triton", "triton_kernel", "kernel") or _env_enabled(
        "MEGATRON_WEIGHTED_SWIGLU_TRITON", "0"
    )


def _weighted_swiglu_triton_block_h(hidden_size: int) -> int:
    raw = os.getenv("MEGATRON_WEIGHTED_SWIGLU_TRITON_BLOCK_H")
    value = int(raw) if raw else 1024
    if value <= 0:
        raise ValueError(f"MEGATRON_WEIGHTED_SWIGLU_TRITON_BLOCK_H must be positive, got {value}")
    return min(triton.next_power_of_2(value), triton.next_power_of_2(hidden_size), 2048)


def _can_use_weighted_swiglu_triton(input: torch.Tensor, weights: torch.Tensor) -> bool:
    if not (HAVE_TRITON and _weighted_swiglu_triton_enabled()):
        return False
    if not (input.is_cuda and weights.is_cuda):
        return False
    if input.dim() != 2 or weights.dim() not in (1, 2):
        return False
    if input.size(-1) % 2 != 0 or input.size(0) <= 0:
        return False
    if weights.numel() != input.size(0):
        return False
    if input.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if weights.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    return input.is_contiguous() and weights.is_contiguous()


@_weighted_swiglu_fuser
def weighted_swiglu(y, weights):
    dtype = y.dtype
    y_1, y_2 = torch.chunk(y, 2, -1)
    res = F.silu(y_1) * y_2 * weights
    return res.to(dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@jit_fuser
def swiglu_back(g, y):
    """Computes the gradient for the SwiGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed using the
            chain rule and the derivative of the SiLU activation function.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1
    )


@jit_fuser
def bias_swiglu_back(g, y, bias):
    """Computes the gradient for the biased SwiGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.
        bias (torch.Tensor): Bias tensor that was added in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed after
            applying the bias addition.
    """
    y = y + bias
    return swiglu_back(g, y)


@_weighted_swiglu_fuser
def weighted_swiglu_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    y_1, y_2 = torch.chunk(y, 2, -1)
    weighted_grad = g * weights
    input_grad = torch.cat(
        (
            weighted_grad
            * torch.sigmoid(y_1)
            * (1 + y_1 * (1 - torch.sigmoid(y_1)))
            * y_2,
            weighted_grad * F.silu(y_1),
        ),
        -1,
    )
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = F.silu(y_1) * y_2 * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


if HAVE_TRITON:

    @triton.jit
    def _weighted_swiglu_forward_kernel(
        input_ptr,
        weights_ptr,
        output_ptr,
        rows: tl.constexpr,
        hidden_size: tl.constexpr,
        input_stride_row: tl.constexpr,
        weights_stride_row: tl.constexpr,
        output_stride_row: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offsets < hidden_size

        x1 = tl.load(input_ptr + row * input_stride_row + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        x2 = tl.load(
            input_ptr + row * input_stride_row + hidden_size + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        weights = tl.load(weights_ptr + row * weights_stride_row).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-x1))
        output = x1 * sigmoid * x2 * weights
        tl.store(output_ptr + row * output_stride_row + offsets, output, mask=mask)


    @triton.jit
    def _weighted_swiglu_backward_kernel(
        grad_output_ptr,
        input_ptr,
        weights_ptr,
        grad_input_ptr,
        grad_weights_ptr,
        rows: tl.constexpr,
        hidden_size: tl.constexpr,
        grad_output_stride_row: tl.constexpr,
        input_stride_row: tl.constexpr,
        weights_stride_row: tl.constexpr,
        grad_input_stride_row: tl.constexpr,
        grad_weights_stride_row: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offsets < hidden_size

        grad_output = tl.load(
            grad_output_ptr + row * grad_output_stride_row + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        x1 = tl.load(input_ptr + row * input_stride_row + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        x2 = tl.load(
            input_ptr + row * input_stride_row + hidden_size + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        weights = tl.load(weights_ptr + row * weights_stride_row).to(tl.float32)

        sigmoid = 1.0 / (1.0 + tl.exp(-x1))
        silu = x1 * sigmoid
        weighted_grad = grad_output * weights
        grad_x1 = weighted_grad * sigmoid * (1.0 + x1 * (1.0 - sigmoid)) * x2
        grad_x2 = weighted_grad * silu
        grad_weights = tl.sum(silu * x2 * grad_output, axis=0)

        tl.store(grad_input_ptr + row * grad_input_stride_row + offsets, grad_x1, mask=mask)
        tl.store(
            grad_input_ptr + row * grad_input_stride_row + hidden_size + offsets,
            grad_x2,
            mask=mask,
        )
        tl.atomic_add(
            grad_weights_ptr + row * grad_weights_stride_row,
            grad_weights,
            sem="relaxed",
        )


class WeightedSwiGLUTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, fp8_input_store):
        rows, doubled_hidden = input.shape
        hidden_size = doubled_hidden // 2
        output = torch.empty((rows, hidden_size), device=input.device, dtype=input.dtype)
        block_h = _weighted_swiglu_triton_block_h(hidden_size)
        grid = (rows, triton.cdiv(hidden_size, block_h))

        _weighted_swiglu_forward_kernel[grid](
            input,
            weights,
            output,
            rows,
            hidden_size,
            input.stride(0),
            weights.stride(0) if weights.dim() > 1 else 1,
            output.stride(0),
            BLOCK_H=block_h,
            num_warps=4,
        )

        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.block_h = block_h
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        grad_output = grad_output.contiguous()
        rows, doubled_hidden = input.shape
        hidden_size = doubled_hidden // 2
        grad_input = torch.empty_like(input)
        grad_weights = torch.zeros_like(weights)
        grid = (rows, triton.cdiv(hidden_size, ctx.block_h))

        _weighted_swiglu_backward_kernel[grid](
            grad_output,
            input,
            weights,
            grad_input,
            grad_weights,
            rows,
            hidden_size,
            grad_output.stride(0),
            input.stride(0),
            weights.stride(0) if weights.dim() > 1 else 1,
            grad_input.stride(0),
            grad_weights.stride(0) if grad_weights.dim() > 1 else 1,
            BLOCK_H=ctx.block_h,
            num_warps=4,
        )
        return grad_input, grad_weights, None


class BiasSwiGLUFunction(torch.autograd.Function):
    """Custom autograd function for SwiGLU activation with bias support."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, bias, fp8_input_store, cpu_offload_input):
        """Forward pass of biased SwiGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor to apply SwiGLU to.
            bias (torch.Tensor): Bias tensor to be added to input before SwiGLU.
            fp8_input_store (bool): If True, stores intermediate values in FP8 format.

        Returns:
            torch.Tensor: Result of applying bias addition followed by SwiGLU activation.
        """
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
            bias.activation_offloading = True
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return bias_swiglu(input, bias)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of biased SwiGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Tuple containing:
                - Gradient with respect to the input tensor
                - Gradient with respect to the bias tensor
                - None for fp8_input_store parameter
        """
        input, bias = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = bias_swiglu_back(grad_output, input, bias)
        return tmp, tmp, None, None


class SwiGLUFunction(torch.autograd.Function):
    """Custom autograd function for SwiGLU activation without bias."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, fp8_input_store, cpu_offload_input):
        """Forward pass of SwiGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor to apply SwiGLU to.
            fp8_input_store (bool): If True, stores intermediate values in FP8 format.

        Returns:
            torch.Tensor: Result of applying SwiGLU activation.
        """
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return swiglu(input)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of SwiGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Tuple containing:
                - Gradient with respect to the input tensor
                - None for fp8_input_store parameter
        """
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = swiglu_back(grad_output, input)
        return tmp, None, None


class WeightedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_swiglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = weighted_swiglu_back(grad_output, input, weights)
        return tmp, wgrad, None


def bias_swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
    """Implementation of biased SwiGLU that handles different input shapes.

    This function reshapes the input if necessary, applies the SwiGLU activation
    (with or without bias), and restores the original shape.

    Args:
        input (torch.Tensor): Input tensor to apply SwiGLU activation.
        bias (torch.Tensor, optional): Bias tensor to be added to input. If None,
            uses the bias-free SwiGLU variant.
        fp8_input_store (bool, optional): Whether to store intermediate values in FP8 format.
            Defaults to False.

    Returns:
        torch.Tensor: Result of biased SwiGLU activation.

    Raises:
        AssertionError: If input tensor does not have 2 or 3 dimensions.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasSwiGLUFunction.apply(input, bias, fp8_input_store, cpu_offload_input)
    else:
        output = SwiGLUFunction.apply(input, fp8_input_store, cpu_offload_input)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False):
    """
    Token-wise-weighted bias swiglu fusion.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")
    else:
        if _can_use_weighted_swiglu_triton(input, weights):
            output = WeightedSwiGLUTritonFunction.apply(input, weights, fp8_input_store)
        else:
            output = WeightedSwiGLUFunction.apply(input, weights, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


# bias_swiglu_impl = BiasSwiGLUFunction.apply
# swiglu_impl = SwiGLUFunction.apply
