// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Shared device utilities for the TurboQuant fused fake-quant kernels.
// The math here is a direct port of
// optimization-playground/python/sglang/jit_kernel/csrc/quantization/turboquant_dense_kv.cuh
// adapted from a TVM-FFI store kernel into a Megatron-style fp-quant-dq op
// usable as the forward leg of a torch.autograd.Function.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace megatron {
namespace turboquant {

constexpr int kLatentDim = 512;
constexpr int kHighChannels = 32;
constexpr int kGroupSize = 128;
constexpr int kHighLevels = 8;   // 3-bit codebook
constexpr int kLowLevels = 4;    // 2-bit codebook
constexpr float kInvSqrtLatentDim = 0.044194173824159216f;  // 1 / sqrt(512)

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_reduce_sum_512(
    float value, float* __restrict__ scratch) {
  const int tid = threadIdx.x;
  value = warp_reduce_sum(value);
  if ((tid & 31) == 0) {
    scratch[tid >> 5] = value;
  }
  __syncthreads();

  float total = 0.0f;
  if (tid < 32) {
    total = tid < 16 ? scratch[tid] : 0.0f;
    total = warp_reduce_sum(total);
  }
  __syncthreads();
  if (tid == 0) {
    scratch[0] = total;
  }
  __syncthreads();
  total = scratch[0];
  __syncthreads();
  return total;
}

__device__ __forceinline__ float fwht_512(
    float value, float* __restrict__ scratch) {
  const int tid = threadIdx.x;

  // First 5 stages live entirely inside a warp via shuffle.
#pragma unroll
  for (int len = 1; len < 32; len <<= 1) {
    const float other = __shfl_xor_sync(0xffffffff, value, len);
    value = (tid & len) ? other - value : value + other;
  }

  scratch[tid] = value;
  __syncthreads();

#pragma unroll
  for (int len = 32; len < kLatentDim; len <<= 1) {
    const int wht_group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = wht_group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = scratch[a];
    const float y = scratch[b];
    __syncthreads();
    if (pos < len) {
      scratch[a] = x + y;
      scratch[b] = x - y;
    }
    __syncthreads();
  }

  const float result = scratch[tid];
  __syncthreads();
  return result;
}

template <int N>
__device__ __forceinline__ uint8_t quantize_with_boundaries(
    const float* __restrict__ boundaries, float value) {
  uint8_t index = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    index += value > boundaries[i];
  }
  return index;
}

template <int N>
__device__ __forceinline__ bool inside_codebook(
    const float* __restrict__ boundaries, float value) {
  return value >= boundaries[0] && value <= boundaries[N - 1];
}

}  // namespace turboquant
}  // namespace megatron
