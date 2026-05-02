// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Shared device utilities for the IndexCache fp8 e4m3 fake-quant kernels.
// Algorithm + constants are a direct port of SGLang's
//   optimization-playground/python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh
// adapted from a paged-storage forward kernel into a Megatron-style
// fake-quant op (output is the dequantized tensor + the saved-for-backward
// state instead of packed bytes).

#pragma once

#include <climits>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace megatron {
namespace indexcache {

constexpr int kHeadDim = 128;            // indexer head dim (DeepSeek-V3.2)
constexpr float kFp8Max = 448.0f;        // fp8 e4m3 max representable
constexpr float kFp8MaxInv = 1.0f / kFp8Max;
constexpr float kEpsDefault = 1.0e-4f;   // matches SGLang fused_store_indexer_cache

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

// Block reduction over kHeadDim (128) threads, broadcasting the result to
// every thread. Mirrors the broadcast-via-fresh-slot pattern we adopted
// for TurboQuant (avoids the alias-with-warp-partial trap).
__device__ __forceinline__ float block_reduce_max_128(
    float v, float* __restrict__ scratch) {
  const int tid = threadIdx.x;
  v = warp_reduce_max(v);
  if ((tid & 31) == 0) {
    scratch[tid >> 5] = v;
  }
  __syncthreads();
  if (tid < 32) {
    float m = (tid < (kHeadDim / 32)) ? scratch[tid] : -INFINITY;
    m = warp_reduce_max(m);
    if (tid == 0) {
      scratch[8] = m;  // fresh slot, distinct from scratch[0..3]
    }
  }
  __syncthreads();
  const float total = scratch[8];
  __syncthreads();
  return total;
}

}  // namespace indexcache
}  // namespace megatron
