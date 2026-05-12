// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Shared device utilities for the 2-bit HIGGS dense MLA-KV fused fake-quant
// kernels. The math is a direct port of
//   optimization-playground/python/sglang/jit_kernel/csrc/quantization/
//   higgs_dense_2bit_kv.cuh  (OP main HEAD 2e2f51717)
// adapted from the SGLang TVM-FFI store kernel into a Megatron-style
// fp-quant-dequant op usable as the forward leg of a torch.autograd.Function.
//
// Algorithmic constants:
//   * kLatentDim     = 512  (kv_lora_rank, the MLA latent width)
//   * kPairDim       = 2    (EDEN2-16 codeword dim)
//   * kCodebookSize  = 16   (4 bits per pair => 2 bits per scalar)
//   * kNumPairs      = 256
//
// The kernel does NOT touch the rope passthrough (qk_pos_emb_head_dim=64)
// during fake-quant; rope flows through unchanged at the MLA layer. The
// SGLang store kernel that the OP kernel is derived from copies rope into
// the slot byte buffer; Megatron's fake-quant path doesn't need that and so
// we mirror only the latent-quant portion.

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace megatron {
namespace higgs {

constexpr int kLatentDim = 512;
constexpr int kPairDim = 2;
constexpr int kCodebookSize = 16;
constexpr int kNumPairs = kLatentDim / kPairDim;  // 256
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

  // Lane 0 of warp 0 reduces the 16 warp partials in scratch[0..15] and
  // broadcasts via a fresh slot (scratch[16]) so the read does not alias an
  // earlier write that the compiler could fold across the in-place reuse.
  if (tid < 32) {
    float v = tid < 16 ? scratch[tid] : 0.0f;
    v = warp_reduce_sum(v);
    if (tid == 0) {
      scratch[16] = v;
    }
  }
  __syncthreads();
  const float total = scratch[16];
  __syncthreads();
  return total;
}

// In-place FWHT_512 across 512 threads, one float per thread. Levels 0..4
// live entirely inside a warp via ``__shfl_xor_sync``; levels 5..8 cross
// warps via SMEM. The math is identical to the TurboQuant 2.5-bit kernel
// (turboquant_kv.cuh::fwht_512) so a single ``buf[kLatentDim]`` of fp32 is
// reused as the scratch area.
__device__ __forceinline__ float fwht_512(
    float value, float* __restrict__ scratch) {
  const int tid = threadIdx.x;

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

// Nearest-neighbour lookup in the EDEN2-16 codebook for a (x0, x1) pair.
// Implements argmax_i (2 (x0*G[i,0] + x1*G[i,1]) - ||G[i]||^2), the
// equivalent of nearest-Euclidean for unit-variance pairs. The codebook is
// passed via __ldg so it lands in the read-only cache; both arrays together
// are 192 bytes which fits there comfortably.
__device__ __forceinline__ uint32_t nearest_codebook_index(
    const float* __restrict__ codebook,
    const float* __restrict__ codebook_norm_sq,
    float x0, float x1) {
  float best = -3.4e38f;
  uint32_t best_idx = 0;
#pragma unroll
  for (int i = 0; i < kCodebookSize; ++i) {
    const float g0 = __ldg(&codebook[i * kPairDim + 0]);
    const float g1 = __ldg(&codebook[i * kPairDim + 1]);
    const float gn = __ldg(&codebook_norm_sq[i]);
    const float score = 2.0f * (x0 * g0 + x1 * g1) - gn;
    if (score > best) {
      best = score;
      best_idx = static_cast<uint32_t>(i);
    }
  }
  return best_idx;
}

}  // namespace higgs
}  // namespace megatron
