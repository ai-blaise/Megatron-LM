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

static __device__ __constant__ float kEden2_16Codebook[kCodebookSize * kPairDim] = {
    -0.8996632695198059f, -1.6360418796539307f,
    -0.9611834883689880f,  1.5999565124511719f,
    -1.8820261955261230f,  0.6787783503532410f,
     0.3630079329013824f, -1.9667866230010986f,
    -0.6814072728157043f, -0.5768185853958130f,
     0.7270012497901917f,  0.6186859607696533f,
     0.3359416127204895f,  1.8371193408966064f,
     1.8599303960800171f,  0.0366685986518860f,
     0.1720824837684631f, -0.9401724338537598f,
    -1.7599700689315796f, -0.6244229674339294f,
    -0.8993809223175049f,  0.3226782381534576f,
     0.8394886851310730f, -0.3017036020755768f,
     1.5314953327178955f,  1.2942044734954834f,
    -0.0011779458727688f,  0.0002206907083746f,
     1.4274526834487915f, -1.2078891992568970f,
    -0.1612390577793121f,  0.8787511587142944f,
};

static __device__ __constant__ float kEden2_16CodebookNormSq[kCodebookSize] = {
    3.4860272407531738f,
    3.4837346076965332f,
    4.0027627944946289f,
    4.0000243186950684f,
    0.7970355749130249f,
    0.9113031625747681f,
    3.4878642559051514f,
    3.4606857299804688f,
    0.9135365486145020f,
    3.4873986244201660f,
    0.9130073189735413f,
    0.7957662940025330f,
    4.0204434394836426f,
    0.0000014362608454f,
    3.4966175556182861f,
    0.7982016801834106f,
};

__device__ __forceinline__ float eden2_16_codebook_value(
    uint32_t cb_idx, int coord) {
  return kEden2_16Codebook[cb_idx * kPairDim + coord];
}

__device__ __forceinline__ float eden2_16_codebook_norm_sq(uint32_t cb_idx) {
  return kEden2_16CodebookNormSq[cb_idx];
}

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

__device__ __forceinline__ int smem_swizzle_idx(int logical_idx) {
  const int col = logical_idx & 31;
  const int row = (logical_idx >> 5) & 15;
  return (logical_idx & ~31) | (col ^ (row & 7));
}

// In-place FWHT_512 across 512 threads, one float per thread. Levels 0..4
// live entirely inside a warp via ``__shfl_xor_sync``; levels 5..8 cross
// warps via swizzled SMEM. The swizzle mirrors optimization-playground's
// accepted B200 HIGGS store path and breaks the power-of-two bank stride for
// the cross-warp butterfly levels.
__device__ __forceinline__ float fwht_512(
    float value, float* __restrict__ scratch) {
  const int tid = threadIdx.x;

#pragma unroll
  for (int len = 1; len < 32; len <<= 1) {
    const float other = __shfl_xor_sync(0xffffffff, value, len);
    value = (tid & len) ? other - value : value + other;
  }

  scratch[smem_swizzle_idx(tid)] = value;
  __syncthreads();

#pragma unroll
  for (int len = 32; len < kLatentDim; len <<= 1) {
    const int wht_group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = wht_group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = scratch[smem_swizzle_idx(a)];
    const float y = scratch[smem_swizzle_idx(b)];
    __syncthreads();
    if (pos < len) {
      scratch[smem_swizzle_idx(a)] = x + y;
      scratch[smem_swizzle_idx(b)] = x - y;
    }
    __syncthreads();
  }

  const float result = scratch[smem_swizzle_idx(tid)];
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

__device__ __forceinline__ uint32_t nearest_codebook_index_const(float x0, float x1) {
  float best = -3.4e38f;
  uint32_t best_idx = 0;
#pragma unroll
  for (int i = 0; i < kCodebookSize; ++i) {
    const float g0 = eden2_16_codebook_value(i, 0);
    const float g1 = eden2_16_codebook_value(i, 1);
    const float gn = eden2_16_codebook_norm_sq(i);
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
