// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Fused backward for selected HISA indexer logits:
//   s[row, slot] = sum_h w[row, h] * ReLU(q[row, h] . k[topk[row, slot]])
//
// The selector/top-k is treated as non-differentiable, matching the existing
// HISA training path. This kernel removes the PyTorch row-chunk selected-score
// autograd graph from StreamBP replay while preserving gradients through the
// selected logits.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// torch.utils.cpp_extension defines these guards globally. MathDx/CuBLASDx
// instantiates common half helpers even for float GEMMs, so make those
// operators visible in this translation unit.
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#include <cublasdx.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstdint>

namespace megatron {
namespace hisa_indexer {

namespace {

constexpr int kHeadDim = 128;
constexpr int kMaxIndexerHeads = 64;
constexpr int kMaxHeadGroup = 8;
constexpr int TILE_N32 = 32;
constexpr int TILE_N64 = 64;
constexpr int TILE_N128 = 128;
constexpr int kDTypeFloat32 = 0;
constexpr int kDTypeBFloat16 = 1;
constexpr int kDTypeFloat16 = 2;
constexpr float kFiniteF32Limit = 3.4028234663852886e38F;

using HisaSelectedScoreBwdGemm64x32 = decltype(
    cublasdx::Size<kMaxIndexerHeads, TILE_N32, kHeadDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using HisaSelectedScoreBwdGemm64x64 = decltype(
    cublasdx::Size<kMaxIndexerHeads, TILE_N64, kHeadDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using HisaSelectedScoreBwdGemm64x128 = decltype(
    cublasdx::Size<kMaxIndexerHeads, TILE_N128, kHeadDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using HisaSelectedScoreBwdGemmBF16_64x32 = decltype(
    cublasdx::Size<kMaxIndexerHeads, TILE_N32, kHeadDim>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using HisaSelectedScoreBwdGemmBF16_64x64 = decltype(
    cublasdx::Size<kMaxIndexerHeads, TILE_N64, kHeadDim>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using HisaSelectedScoreBwdGemmBF16_64x128 = decltype(
    cublasdx::Size<kMaxIndexerHeads, TILE_N128, kHeadDim>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using HisaSelectedScoreBwdGemmF16_64x32 = decltype(
    cublasdx::Size<kMaxIndexerHeads, TILE_N32, kHeadDim>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using HisaSelectedScoreBwdGemmF16_64x64 = decltype(
    cublasdx::Size<kMaxIndexerHeads, TILE_N64, kHeadDim>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using HisaSelectedScoreBwdGemmF16_64x128 = decltype(
    cublasdx::Size<kMaxIndexerHeads, TILE_N128, kHeadDim>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
  return static_cast<T>(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half(v);
}

__device__ __forceinline__ bool finite_f32(float v) {
  return v == v && v > -kFiniteF32Limit && v < kFiniteF32Limit;
}

__device__ __forceinline__ float finite_or_zero(float v) {
  return finite_f32(v) ? v : 0.0f;
}

__device__ __forceinline__ float finite_mul_or_zero(float a, float b) {
  const float out = a * b;
  return finite_f32(out) ? out : 0.0f;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_sum(float v, float* scratch) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  v = warp_sum(v);
  if (lane == 0) {
    scratch[warp] = v;
  }
  __syncthreads();
  const int warp_count = blockDim.x >> 5;
  float out = 0.0f;
  if (warp == 0) {
    out = lane < warp_count ? scratch[lane] : 0.0f;
    out = warp_sum(out);
    if (lane == 0) {
      scratch[warp_count] = out;
    }
  }
  __syncthreads();
  return scratch[warp_count];
}

__global__ void hisa_selected_score_bwd_kernel(
    const float* __restrict__ grad_selected_scores, // [Q, K]
    const float* __restrict__ q,                    // [Q, H, D]
    const float* __restrict__ k,                    // [L, D]
    const float* __restrict__ weights,              // [Q, H]
    const int32_t* __restrict__ topk_indices,       // [Q, K]
    float* __restrict__ grad_q,                     // [Q, H, D]
    float* __restrict__ grad_k,                     // [L, D]
    float* __restrict__ grad_w,                     // [Q, H]
    int Q,
    int H,
    int D,
    int L,
    int K) {
  const int head = blockIdx.x;
  const int row = blockIdx.y;
  if (row >= Q || head >= H) {
    return;
  }

  __shared__ float reduce_scratch[9];
  const int tid = threadIdx.x;
  const float* q_head = q + (static_cast<int64_t>(row) * H + head) * D;
  const float weight = weights[static_cast<int64_t>(row) * H + head];
  const bool weight_ok = finite_f32(weight);
  float grad_q_acc = 0.0f;
  float grad_w_acc = 0.0f;
  const float q_d_cached = tid < D ? finite_or_zero(q_head[tid]) : 0.0f;

  for (int slot = 0; slot < K; ++slot) {
    const int token = topk_indices[static_cast<int64_t>(row) * K + slot];
    const float g_score = grad_selected_scores[static_cast<int64_t>(row) * K + slot];
    if (token < 0 || token >= L || g_score == 0.0f || !finite_f32(g_score) || !weight_ok) {
      continue;
    }

    const float* k_row = k + static_cast<int64_t>(token) * D;
    float partial = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
      const float q_v = q_head[d];
      const float k_v = k_row[d];
      partial += finite_f32(q_v) && finite_f32(k_v) ? q_v * k_v : 0.0f;
    }
    const float dot = block_sum(partial, reduce_scratch);
    if (dot <= 0.0f || !finite_f32(dot)) {
      continue;
    }

    const float scale = g_score * weight;
    if (!finite_f32(scale)) {
      continue;
    }
    float* grad_k_row = grad_k + static_cast<int64_t>(token) * D;
    if (tid < D) {
      const float k_d = finite_or_zero(k_row[tid]);
      grad_q_acc += scale * k_d;
      const float grad_k_value = scale * q_d_cached;
      if (finite_f32(grad_k_value) && grad_k_value != 0.0f) {
        atomicAdd(grad_k_row + tid, grad_k_value);
      }
    }
    if (tid == 0) {
      grad_w_acc += finite_mul_or_zero(g_score, dot);
    }
  }

  if (tid < D) {
    grad_q_acc = finite_or_zero(grad_q_acc);
    grad_q[(static_cast<int64_t>(row) * H + head) * D + tid] = grad_q_acc;
  }
  if (tid == 0) {
    grad_w_acc = finite_or_zero(grad_w_acc);
    grad_w[static_cast<int64_t>(row) * H + head] = grad_w_acc;
  }
}

__global__ void hisa_selected_score_bwd_row_grouped_kernel(
    const float* __restrict__ grad_selected_scores, // [Q, K]
    const float* __restrict__ q,                    // [Q, H, D]
    const float* __restrict__ k,                    // [L, D]
    const float* __restrict__ weights,              // [Q, H]
    const int32_t* __restrict__ topk_indices,       // [Q, K]
    float* __restrict__ grad_q,                     // [Q, H, D]
    float* __restrict__ grad_k,                     // [L, D]
    float* __restrict__ grad_w,                     // [Q, H]
    int Q,
    int H,
    int D,
    int L,
    int K) {
  const int row = blockIdx.x;
  if (row >= Q || H > kMaxIndexerHeads) {
    return;
  }

  __shared__ float reduce_scratch[9];
  const int tid = threadIdx.x;
  const float* q_row = q + static_cast<int64_t>(row) * H * D;
  const float* w_row = weights + static_cast<int64_t>(row) * H;
  float grad_q_acc[kMaxIndexerHeads];
  float grad_w_acc[kMaxIndexerHeads];

#pragma unroll
  for (int h = 0; h < kMaxIndexerHeads; ++h) {
    grad_q_acc[h] = 0.0f;
    grad_w_acc[h] = 0.0f;
  }

  for (int slot = 0; slot < K; ++slot) {
    const int token = topk_indices[static_cast<int64_t>(row) * K + slot];
    const float g_score = grad_selected_scores[static_cast<int64_t>(row) * K + slot];
    if (token < 0 || token >= L || g_score == 0.0f || !finite_f32(g_score)) {
      continue;
    }

    const float* k_row = k + static_cast<int64_t>(token) * D;
    const float k_d_cached = tid < D ? finite_or_zero(k_row[tid]) : 0.0f;
    float grad_k_acc = 0.0f;

    for (int h = 0; h < H; ++h) {
      const float* q_head = q_row + static_cast<int64_t>(h) * D;
      const float weight = w_row[h];
      if (!finite_f32(weight)) {
        continue;
      }
      float partial = 0.0f;
      for (int d = tid; d < D; d += blockDim.x) {
        const float q_v = q_head[d];
        const float k_v = k_row[d];
        partial += finite_f32(q_v) && finite_f32(k_v) ? q_v * k_v : 0.0f;
      }
      const float dot = block_sum(partial, reduce_scratch);
      if (dot <= 0.0f || !finite_f32(dot)) {
        continue;
      }

      const float scale = g_score * weight;
      if (!finite_f32(scale)) {
        continue;
      }
      if (tid < D) {
        const float q_d = finite_or_zero(q_head[tid]);
        grad_q_acc[h] += scale * k_d_cached;
        grad_k_acc += scale * q_d;
      }
      if (tid == 0) {
        grad_w_acc[h] += finite_mul_or_zero(g_score, dot);
      }
    }

    if (tid < D && grad_k_acc != 0.0f && finite_f32(grad_k_acc)) {
      atomicAdd(grad_k + static_cast<int64_t>(token) * D + tid, grad_k_acc);
    }
  }

  if (tid < D) {
    for (int h = 0; h < H; ++h) {
      grad_q_acc[h] = finite_or_zero(grad_q_acc[h]);
      grad_q[(static_cast<int64_t>(row) * H + h) * D + tid] = grad_q_acc[h];
    }
  }
  if (tid == 0) {
    for (int h = 0; h < H; ++h) {
      grad_w_acc[h] = finite_or_zero(grad_w_acc[h]);
      grad_w[static_cast<int64_t>(row) * H + h] = grad_w_acc[h];
    }
  }
}

__global__ void hisa_selected_score_bwd_head_grouped_kernel(
    const float* __restrict__ grad_selected_scores, // [Q, K]
    const float* __restrict__ q,                    // [Q, H, D]
    const float* __restrict__ k,                    // [L, D]
    const float* __restrict__ weights,              // [Q, H]
    const int32_t* __restrict__ topk_indices,       // [Q, K]
    float* __restrict__ grad_q,                     // [Q, H, D]
    float* __restrict__ grad_k,                     // [L, D]
    float* __restrict__ grad_w,                     // [Q, H]
    int Q,
    int H,
    int D,
    int L,
    int K,
    int head_group_size) {
  const int group = blockIdx.x;
  const int row = blockIdx.y;
  if (row >= Q || group * head_group_size >= H) {
    return;
  }

  __shared__ float reduce_scratch[9];
  const int tid = threadIdx.x;
  const int head_start = group * head_group_size;
  const int active_heads = min(head_group_size, H - head_start);
  const float* q_row = q + static_cast<int64_t>(row) * H * D;
  const float* w_row = weights + static_cast<int64_t>(row) * H;

  float grad_q_acc[kMaxHeadGroup];
  float grad_w_acc[kMaxHeadGroup];

#pragma unroll
  for (int local_h = 0; local_h < kMaxHeadGroup; ++local_h) {
    grad_q_acc[local_h] = 0.0f;
    grad_w_acc[local_h] = 0.0f;
  }

  for (int slot = 0; slot < K; ++slot) {
    const int token = topk_indices[static_cast<int64_t>(row) * K + slot];
    const float g_score = grad_selected_scores[static_cast<int64_t>(row) * K + slot];
    if (token < 0 || token >= L || g_score == 0.0f || !finite_f32(g_score)) {
      continue;
    }

    const float* k_row = k + static_cast<int64_t>(token) * D;
    const float k_d_cached = tid < D ? finite_or_zero(k_row[tid]) : 0.0f;
    float grad_k_acc = 0.0f;

    for (int local_h = 0; local_h < active_heads; ++local_h) {
      const int head = head_start + local_h;
      const float* q_head = q_row + static_cast<int64_t>(head) * D;
      const float weight = w_row[head];
      if (!finite_f32(weight)) {
        continue;
      }
      float partial = 0.0f;
      for (int d = tid; d < D; d += blockDim.x) {
        const float q_v = q_head[d];
        const float k_v = k_row[d];
        partial += finite_f32(q_v) && finite_f32(k_v) ? q_v * k_v : 0.0f;
      }
      const float dot = block_sum(partial, reduce_scratch);
      if (dot <= 0.0f || !finite_f32(dot)) {
        continue;
      }

      const float scale = g_score * weight;
      if (!finite_f32(scale)) {
        continue;
      }
      if (tid < D) {
        const float q_d = finite_or_zero(q_head[tid]);
        grad_q_acc[local_h] += scale * k_d_cached;
        grad_k_acc += scale * q_d;
      }
      if (tid == 0) {
        grad_w_acc[local_h] += finite_mul_or_zero(g_score, dot);
      }
    }

    if (tid < D && grad_k_acc != 0.0f && finite_f32(grad_k_acc)) {
      atomicAdd(grad_k + static_cast<int64_t>(token) * D + tid, grad_k_acc);
    }
  }

  if (tid < D) {
    for (int local_h = 0; local_h < active_heads; ++local_h) {
      const int head = head_start + local_h;
      grad_q_acc[local_h] = finite_or_zero(grad_q_acc[local_h]);
      grad_q[(static_cast<int64_t>(row) * H + head) * D + tid] = grad_q_acc[local_h];
    }
  }
  if (tid == 0) {
    for (int local_h = 0; local_h < active_heads; ++local_h) {
      const int head = head_start + local_h;
      grad_w_acc[local_h] = finite_or_zero(grad_w_acc[local_h]);
      grad_w[static_cast<int64_t>(row) * H + head] = grad_w_acc[local_h];
    }
  }
}

__global__ void hisa_selected_score_bwd_warp_grouped_kernel(
    const float* __restrict__ grad_selected_scores, // [Q, K]
    const float* __restrict__ q,                    // [Q, H, D]
    const float* __restrict__ k,                    // [L, D]
    const float* __restrict__ weights,              // [Q, H]
    const int32_t* __restrict__ topk_indices,       // [Q, K]
    float* __restrict__ grad_q,                     // [Q, H, D]
    float* __restrict__ grad_k,                     // [L, D]
    float* __restrict__ grad_w,                     // [Q, H]
    int Q,
    int H,
    int D,
    int L,
    int K,
    int head_group_size) {
  const int group = blockIdx.x;
  const int row = blockIdx.y;
  if (row >= Q || group * head_group_size >= H || D != kHeadDim) {
    return;
  }

  __shared__ float grad_k_scratch[kMaxHeadGroup][kHeadDim];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int head_start = group * head_group_size;
  const int active_heads = min(head_group_size, H - head_start);
  const bool active_warp = warp < active_heads;
  const int head = head_start + warp;

  const float* q_row = q + static_cast<int64_t>(row) * H * D;
  const float* w_row = weights + static_cast<int64_t>(row) * H;
  float grad_q_acc0 = 0.0f;
  float grad_q_acc1 = 0.0f;
  float grad_q_acc2 = 0.0f;
  float grad_q_acc3 = 0.0f;
  float grad_w_acc = 0.0f;

  for (int slot = 0; slot < K; ++slot) {
    const int token = topk_indices[static_cast<int64_t>(row) * K + slot];
    const float g_score = grad_selected_scores[static_cast<int64_t>(row) * K + slot];
    const bool token_ok = token >= 0 && token < L && g_score != 0.0f && finite_f32(g_score);
    const int safe_token = token > 0 ? token : 0;
    const float* k_row = k + static_cast<int64_t>(safe_token) * D;

    float partial = 0.0f;
    if (active_warp && token_ok) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        const float q_v = q_row[static_cast<int64_t>(head) * D + d];
        const float k_v = k_row[d];
        partial += finite_f32(q_v) && finite_f32(k_v) ? q_v * k_v : 0.0f;
      }
    }
    float dot = warp_sum(partial);
    dot = __shfl_sync(0xffffffff, dot, 0);
    const float weight = active_warp ? w_row[head] : 0.0f;
    const bool edge_active =
        active_warp && token_ok && dot > 0.0f && finite_f32(dot) && finite_f32(weight);
    const float scale = edge_active ? finite_mul_or_zero(g_score, weight) : 0.0f;

    if (active_warp) {
      const int d0 = lane;
      const int d1 = lane + 32;
      const int d2 = lane + 64;
      const int d3 = lane + 96;
      const float q_base0 = finite_or_zero(q_row[static_cast<int64_t>(head) * D + d0]);
      const float q_base1 = finite_or_zero(q_row[static_cast<int64_t>(head) * D + d1]);
      const float q_base2 = finite_or_zero(q_row[static_cast<int64_t>(head) * D + d2]);
      const float q_base3 = finite_or_zero(q_row[static_cast<int64_t>(head) * D + d3]);
      if (edge_active && scale != 0.0f) {
        grad_q_acc0 += finite_mul_or_zero(scale, finite_or_zero(k_row[d0]));
        grad_q_acc1 += finite_mul_or_zero(scale, finite_or_zero(k_row[d1]));
        grad_q_acc2 += finite_mul_or_zero(scale, finite_or_zero(k_row[d2]));
        grad_q_acc3 += finite_mul_or_zero(scale, finite_or_zero(k_row[d3]));
      }
      grad_k_scratch[warp][d0] = finite_mul_or_zero(scale, q_base0);
      grad_k_scratch[warp][d1] = finite_mul_or_zero(scale, q_base1);
      grad_k_scratch[warp][d2] = finite_mul_or_zero(scale, q_base2);
      grad_k_scratch[warp][d3] = finite_mul_or_zero(scale, q_base3);
      if (lane == 0 && edge_active) {
        grad_w_acc += finite_mul_or_zero(g_score, dot);
      }
    }
    __syncthreads();

    if (token_ok && tid < D) {
      float grad_k_acc = 0.0f;
      for (int local_h = 0; local_h < active_heads; ++local_h) {
        grad_k_acc += grad_k_scratch[local_h][tid];
      }
      if (grad_k_acc != 0.0f && finite_f32(grad_k_acc)) {
        atomicAdd(grad_k + static_cast<int64_t>(token) * D + tid, grad_k_acc);
      }
    }
    __syncthreads();
  }

  if (active_warp) {
    const int d0 = lane;
    const int d1 = lane + 32;
    const int d2 = lane + 64;
    const int d3 = lane + 96;
    float* grad_q_head = grad_q + (static_cast<int64_t>(row) * H + head) * D;
    grad_q_head[d0] = finite_or_zero(grad_q_acc0);
    grad_q_head[d1] = finite_or_zero(grad_q_acc1);
    grad_q_head[d2] = finite_or_zero(grad_q_acc2);
    grad_q_head[d3] = finite_or_zero(grad_q_acc3);
    if (lane == 0) {
      grad_w[static_cast<int64_t>(row) * H + head] = finite_or_zero(grad_w_acc);
    }
  }
}

template <class GEMM, int TILE_N>
__global__ void hisa_selected_score_bwd_cublasdx_kernel(
    const float* __restrict__ grad_selected_scores, // [Q, K]
    const float* __restrict__ q,                    // [Q, 64, 128]
    const float* __restrict__ k,                    // [L, 128]
    const float* __restrict__ weights,              // [Q, 64]
    const int32_t* __restrict__ topk_indices,       // [Q, K]
    float* __restrict__ grad_q,                     // [Q, 64, 128]
    float* __restrict__ grad_k,                     // [L, 128]
    float* __restrict__ grad_w,                     // [Q, 64]
    int Q,
    int L,
    int K) {
  const int row = blockIdx.x;
  const int tile_start = blockIdx.y * TILE_N;
  const int tid = threadIdx.x;
  if (row >= Q || tile_start >= K) {
    return;
  }

  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto gemm_smem = reinterpret_cast<void*>(smem_raw);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  const float* q_row = q + static_cast<int64_t>(row) * kMaxIndexerHeads * kHeadDim;
  const float* w_row = weights + static_cast<int64_t>(row) * kMaxIndexerHeads;
  const int32_t* topk_row = topk_indices + static_cast<int64_t>(row) * K;
  const float* grad_row = grad_selected_scores + static_cast<int64_t>(row) * K;

  for (int idx = tid; idx < kMaxIndexerHeads * kHeadDim; idx += blockDim.x) {
    const int h = idx / kHeadDim;
    const int d = idx - h * kHeadDim;
    a_shared(h, d) = finite_or_zero(q_row[static_cast<int64_t>(h) * kHeadDim + d]);
  }
  for (int idx = tid; idx < kHeadDim * TILE_N; idx += blockDim.x) {
    const int d = idx / TILE_N;
    const int n = idx - d * TILE_N;
    const int slot = tile_start + n;
    float value = 0.0f;
    if (slot < K) {
      const int token = topk_row[slot];
      if (token >= 0 && token < L) {
        value = finite_or_zero(k[static_cast<int64_t>(token) * kHeadDim + d]);
      }
    }
    b_shared(d, n) = value;
  }
  for (int idx = tid; idx < kMaxIndexerHeads * TILE_N; idx += blockDim.x) {
    const int h = idx / TILE_N;
    const int n = idx - h * TILE_N;
    c_shared(h, n) = 0.0f;
  }
  __syncthreads();

  GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
  __syncthreads();

  // dL/dw[row,h] = sum_slot g_slot * relu(q_h dot k_slot).
  for (int h = tid; h < kMaxIndexerHeads; h += blockDim.x) {
    float grad_w_acc = 0.0f;
    for (int n = 0; n < TILE_N; ++n) {
      const int slot = tile_start + n;
      if (slot >= K) {
        continue;
      }
      const int token = topk_row[slot];
      const float g_score = grad_row[slot];
      const float dot = c_shared(h, n);
      if (
          token >= 0 && token < L && g_score != 0.0f && finite_f32(g_score) &&
          dot > 0.0f && finite_f32(dot)) {
        grad_w_acc += finite_mul_or_zero(g_score, dot);
      }
    }
    if (grad_w_acc != 0.0f && finite_f32(grad_w_acc)) {
      atomicAdd(grad_w + static_cast<int64_t>(row) * kMaxIndexerHeads + h, grad_w_acc);
    }
  }
  __syncthreads();

  for (int idx = tid; idx < kMaxIndexerHeads * TILE_N; idx += blockDim.x) {
    const int h = idx / TILE_N;
    const int n = idx - h * TILE_N;
    const int slot = tile_start + n;
    float scale = 0.0f;
    if (slot < K) {
      const int token = topk_row[slot];
      const float g_score = grad_row[slot];
      const float dot = c_shared(h, n);
      const float weight = w_row[h];
      if (
          token >= 0 && token < L && g_score != 0.0f && finite_f32(g_score) &&
          dot > 0.0f && finite_f32(dot) && finite_f32(weight)) {
        scale = finite_mul_or_zero(g_score, weight);
      }
    }
    c_shared(h, n) = scale;
  }
  __syncthreads();

  // dL/dq[row,h,d] accumulates over this tile's selected tokens. There are
  // multiple top-k tiles per row, so grad_q is an atomic accumulation here.
  for (int idx = tid; idx < kMaxIndexerHeads * kHeadDim; idx += blockDim.x) {
    const int h = idx / kHeadDim;
    const int d = idx - h * kHeadDim;
    float grad_q_acc = 0.0f;
    for (int n = 0; n < TILE_N; ++n) {
      const int slot = tile_start + n;
      if (slot >= K) {
        continue;
      }
      const int token = topk_row[slot];
      const float scale = c_shared(h, n);
      if (token >= 0 && token < L && scale != 0.0f) {
        grad_q_acc +=
            finite_mul_or_zero(scale, finite_or_zero(k[static_cast<int64_t>(token) * kHeadDim + d]));
      }
    }
    if (grad_q_acc != 0.0f && finite_f32(grad_q_acc)) {
      atomicAdd(
          grad_q + (static_cast<int64_t>(row) * kMaxIndexerHeads + h) * kHeadDim + d,
          grad_q_acc);
    }
  }

  // dL/dk[token,d] is summed across all 64 indexer heads before one global
  // atomic per selected token/dim. The scalar head-grouped path emits one
  // atomic per head group, so this is the main global-atomic reduction.
  for (int idx = tid; idx < TILE_N * kHeadDim; idx += blockDim.x) {
    const int n = idx / kHeadDim;
    const int d = idx - n * kHeadDim;
    const int slot = tile_start + n;
    if (slot >= K) {
      continue;
    }
    const int token = topk_row[slot];
    if (token < 0 || token >= L) {
      continue;
    }

    float grad_k_acc = 0.0f;
    for (int h = 0; h < kMaxIndexerHeads; ++h) {
      const float scale = c_shared(h, n);
      if (scale != 0.0f) {
        grad_k_acc +=
            finite_mul_or_zero(scale, finite_or_zero(q_row[static_cast<int64_t>(h) * kHeadDim + d]));
      }
    }
    if (grad_k_acc != 0.0f && finite_f32(grad_k_acc)) {
      atomicAdd(grad_k + static_cast<int64_t>(token) * kHeadDim + d, grad_k_acc);
    }
  }
}

template <class GEMM, typename scalar_t, typename weight_t, typename index_t, int TILE_N>
__global__ void hisa_selected_score_bwd_batched_cublasdx_kernel(
    const float* __restrict__ grad_selected_scores, // [B * Q, K]
    const scalar_t* __restrict__ q,                 // [Q, B, 64, 128]
    const scalar_t* __restrict__ k,                 // [L, B, 128]
    const weight_t* __restrict__ weights,           // [Q, B, 64]
    const index_t* __restrict__ topk_indices,       // [B, Q, K]
    float* __restrict__ grad_q,                     // [Q, B, 64, 128]
    float* __restrict__ grad_k,                     // [L, B, 128]
    float* __restrict__ grad_w,                     // [Q, B, 64]
    int Q,
    int B,
    int L,
    int K) {
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  const int tile_start = blockIdx.z * TILE_N;
  const int tid = threadIdx.x;
  if (row >= Q || batch >= B || tile_start >= K) {
    return;
  }

  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto gemm_smem = reinterpret_cast<void*>(smem_raw);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  const int64_t row_batch = static_cast<int64_t>(row) * B + batch;
  const scalar_t* q_row = q + row_batch * kMaxIndexerHeads * kHeadDim;
  const weight_t* w_row = weights + row_batch * kMaxIndexerHeads;
  const index_t* topk_row = topk_indices + (static_cast<int64_t>(batch) * Q + row) * K;
  const float* grad_row = grad_selected_scores + (static_cast<int64_t>(batch) * Q + row) * K;

  for (int idx = tid; idx < kMaxIndexerHeads * kHeadDim; idx += blockDim.x) {
    const int h = idx / kHeadDim;
    const int d = idx - h * kHeadDim;
    a_shared(h, d) =
        from_float<scalar_t>(finite_or_zero(to_float(q_row[static_cast<int64_t>(h) * kHeadDim + d])));
  }
  for (int idx = tid; idx < kHeadDim * TILE_N; idx += blockDim.x) {
    const int d = idx / TILE_N;
    const int n = idx - d * TILE_N;
    const int slot = tile_start + n;
    scalar_t value = from_float<scalar_t>(0.0f);
    if (slot < K) {
      const int token = topk_row[slot];
      if (token >= 0 && token < L) {
        value = from_float<scalar_t>(finite_or_zero(
            to_float(k[(static_cast<int64_t>(token) * B + batch) * kHeadDim + d])));
      }
    }
    b_shared(d, n) = value;
  }
  for (int idx = tid; idx < kMaxIndexerHeads * TILE_N; idx += blockDim.x) {
    const int h = idx / TILE_N;
    const int n = idx - h * TILE_N;
    c_shared(h, n) = 0.0f;
  }
  __syncthreads();

  GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
  __syncthreads();

  for (int h = tid; h < kMaxIndexerHeads; h += blockDim.x) {
    float grad_w_acc = 0.0f;
    for (int n = 0; n < TILE_N; ++n) {
      const int slot = tile_start + n;
      if (slot >= K) {
        continue;
      }
      const int token = topk_row[slot];
      const float g_score = grad_row[slot];
      const float dot = c_shared(h, n);
      if (
          token >= 0 && token < L && g_score != 0.0f && finite_f32(g_score) &&
          dot > 0.0f && finite_f32(dot)) {
        grad_w_acc += finite_mul_or_zero(g_score, dot);
      }
    }
    if (grad_w_acc != 0.0f && finite_f32(grad_w_acc)) {
      atomicAdd(grad_w + row_batch * kMaxIndexerHeads + h, grad_w_acc);
    }
  }
  __syncthreads();

  for (int idx = tid; idx < kMaxIndexerHeads * TILE_N; idx += blockDim.x) {
    const int h = idx / TILE_N;
    const int n = idx - h * TILE_N;
    const int slot = tile_start + n;
    float scale = 0.0f;
    if (slot < K) {
      const int token = topk_row[slot];
      const float g_score = grad_row[slot];
      const float dot = c_shared(h, n);
      const float weight = to_float(w_row[h]);
      if (
          token >= 0 && token < L && g_score != 0.0f && finite_f32(g_score) &&
          dot > 0.0f && finite_f32(dot) && finite_f32(weight)) {
        scale = finite_mul_or_zero(g_score, weight);
      }
    }
    c_shared(h, n) = scale;
  }
  __syncthreads();

  for (int idx = tid; idx < kMaxIndexerHeads * kHeadDim; idx += blockDim.x) {
    const int h = idx / kHeadDim;
    const int d = idx - h * kHeadDim;
    float grad_q_acc = 0.0f;
    for (int n = 0; n < TILE_N; ++n) {
      const int slot = tile_start + n;
      if (slot >= K) {
        continue;
      }
      const int token = topk_row[slot];
      const float scale = c_shared(h, n);
      if (token >= 0 && token < L && scale != 0.0f) {
        grad_q_acc +=
            finite_mul_or_zero(
                scale,
                finite_or_zero(to_float(k[(static_cast<int64_t>(token) * B + batch) * kHeadDim + d])));
      }
    }
    if (grad_q_acc != 0.0f && finite_f32(grad_q_acc)) {
      atomicAdd(
          grad_q + (row_batch * kMaxIndexerHeads + h) * kHeadDim + d,
          grad_q_acc);
    }
  }

  for (int idx = tid; idx < TILE_N * kHeadDim; idx += blockDim.x) {
    const int n = idx / kHeadDim;
    const int d = idx - n * kHeadDim;
    const int slot = tile_start + n;
    if (slot >= K) {
      continue;
    }
    const int token = topk_row[slot];
    if (token < 0 || token >= L) {
      continue;
    }

    float grad_k_acc = 0.0f;
    for (int h = 0; h < kMaxIndexerHeads; ++h) {
      const float scale = c_shared(h, n);
      if (scale != 0.0f) {
        grad_k_acc +=
            finite_mul_or_zero(
                scale,
                finite_or_zero(to_float(q_row[static_cast<int64_t>(h) * kHeadDim + d])));
      }
    }
    if (grad_k_acc != 0.0f && finite_f32(grad_k_acc)) {
      atomicAdd(grad_k + (static_cast<int64_t>(token) * B + batch) * kHeadDim + d, grad_k_acc);
    }
  }
}

}  // namespace

template <class GEMM, typename scalar_t, typename weight_t, int TILE_N>
void launch_hisa_selected_score_bwd_batched_cublasdx_typed(
    const float* grad_selected_scores, const void* q, const void* k,
    const void* weights, const void* topk_indices, float* grad_q,
    float* grad_k, float* grad_w, int Q, int B, int L, int K,
    int topk_index_dtype, cudaStream_t stream) {
  dim3 block(kHeadDim);
  dim3 grid(Q, B, (K + TILE_N - 1) / TILE_N);
  const size_t smem_bytes = cublasdx::get_shared_storage_size<GEMM>();
  if (topk_index_dtype == 0) {
    auto kernel =
        hisa_selected_score_bwd_batched_cublasdx_kernel<
            GEMM, scalar_t, weight_t, int16_t, TILE_N>;
    if (smem_bytes > 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
    kernel<<<grid, block, smem_bytes, stream>>>(
        grad_selected_scores,
        static_cast<const scalar_t*>(q),
        static_cast<const scalar_t*>(k),
        static_cast<const weight_t*>(weights),
        static_cast<const int16_t*>(topk_indices),
        grad_q,
        grad_k,
        grad_w,
        Q,
        B,
        L,
        K);
  } else if (topk_index_dtype == 1) {
    auto kernel =
        hisa_selected_score_bwd_batched_cublasdx_kernel<
            GEMM, scalar_t, weight_t, int32_t, TILE_N>;
    if (smem_bytes > 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
    kernel<<<grid, block, smem_bytes, stream>>>(
        grad_selected_scores,
        static_cast<const scalar_t*>(q),
        static_cast<const scalar_t*>(k),
        static_cast<const weight_t*>(weights),
        static_cast<const int32_t*>(topk_indices),
        grad_q,
        grad_k,
        grad_w,
        Q,
        B,
        L,
        K);
  } else {
    C10_THROW_ERROR(ValueError, "batched HISA selected-score backward supports int16/int32 topk only");
  }
}

template <class GEMM, typename scalar_t, int TILE_N>
void dispatch_hisa_selected_score_bwd_batched_weight(
    const float* grad_selected_scores, const void* q, const void* k,
    const void* weights, const void* topk_indices, float* grad_q,
    float* grad_k, float* grad_w, int Q, int B, int L, int K,
    int weight_dtype, int topk_index_dtype, cudaStream_t stream) {
  if (weight_dtype == kDTypeFloat32) {
    launch_hisa_selected_score_bwd_batched_cublasdx_typed<GEMM, scalar_t, float, TILE_N>(
        grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
        grad_w, Q, B, L, K, topk_index_dtype, stream);
  } else if (weight_dtype == kDTypeBFloat16) {
    launch_hisa_selected_score_bwd_batched_cublasdx_typed<
        GEMM, scalar_t, __nv_bfloat16, TILE_N>(
        grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
        grad_w, Q, B, L, K, topk_index_dtype, stream);
  } else if (weight_dtype == kDTypeFloat16) {
    launch_hisa_selected_score_bwd_batched_cublasdx_typed<GEMM, scalar_t, __half, TILE_N>(
        grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
        grad_w, Q, B, L, K, topk_index_dtype, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported batched HISA selected-score weight dtype");
  }
}

void launch_hisa_selected_score_bwd(
    const float* grad_selected_scores, const float* q, const float* k,
    const float* weights, const int32_t* topk_indices, float* grad_q,
    float* grad_k, float* grad_w, int Q, int H, int D, int L, int K,
    cudaStream_t stream) {
  dim3 block(kHeadDim);
  const char* cublasdx_env = std::getenv("MEGATRON_HISA_SELECTED_SCORE_BWD_CUBLASDX");
  const bool use_cublasdx =
      cublasdx_env != nullptr && cublasdx_env[0] != '0' &&
      H == kMaxIndexerHeads && D == kHeadDim;
  if (use_cublasdx) {
    const char* tile_env = std::getenv("MEGATRON_HISA_SELECTED_SCORE_BWD_CUBLASDX_TILE_N");
    const int tile_n = tile_env == nullptr ? TILE_N32 : std::atoi(tile_env);
    if (tile_n == TILE_N32) {
      using GEMM = HisaSelectedScoreBwdGemm64x32;
      dim3 grid(Q, (K + TILE_N32 - 1) / TILE_N32);
      const size_t smem_bytes = cublasdx::get_shared_storage_size<GEMM>();
      if (smem_bytes > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            hisa_selected_score_bwd_cublasdx_kernel<GEMM, TILE_N32>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
      }
      hisa_selected_score_bwd_cublasdx_kernel<GEMM, TILE_N32>
          <<<grid, block, smem_bytes, stream>>>(
              grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
              grad_w, Q, L, K);
    } else if (tile_n == TILE_N128) {
      using GEMM = HisaSelectedScoreBwdGemm64x128;
      dim3 grid(Q, (K + TILE_N128 - 1) / TILE_N128);
      const size_t smem_bytes = cublasdx::get_shared_storage_size<GEMM>();
      if (smem_bytes > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            hisa_selected_score_bwd_cublasdx_kernel<GEMM, TILE_N128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
      }
      hisa_selected_score_bwd_cublasdx_kernel<GEMM, TILE_N128>
          <<<grid, block, smem_bytes, stream>>>(
              grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
              grad_w, Q, L, K);
    } else {
      using GEMM = HisaSelectedScoreBwdGemm64x64;
      dim3 grid(Q, (K + TILE_N64 - 1) / TILE_N64);
      const size_t smem_bytes = cublasdx::get_shared_storage_size<GEMM>();
      if (smem_bytes > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            hisa_selected_score_bwd_cublasdx_kernel<GEMM, TILE_N64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
      }
      hisa_selected_score_bwd_cublasdx_kernel<GEMM, TILE_N64>
          <<<grid, block, smem_bytes, stream>>>(
              grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
              grad_w, Q, L, K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const char* row_grouped_env = std::getenv("MEGATRON_HISA_SELECTED_SCORE_BWD_ROW_GROUPED");
  const bool use_row_grouped =
      row_grouped_env != nullptr && row_grouped_env[0] != '0';
  const char* head_group_env = std::getenv("MEGATRON_HISA_SELECTED_SCORE_BWD_HEAD_GROUP");
  const int requested_head_group =
      head_group_env == nullptr
          ? 1
          : std::max(1, std::min(std::atoi(head_group_env), kMaxHeadGroup));
  const char* warp_grouped_env = std::getenv("MEGATRON_HISA_SELECTED_SCORE_BWD_WARP_GROUPED");
  const bool use_warp_grouped =
      warp_grouped_env != nullptr && warp_grouped_env[0] != '0';
  if (use_warp_grouped && requested_head_group > 1 && H <= kMaxIndexerHeads && D == kHeadDim) {
    dim3 grid((H + requested_head_group - 1) / requested_head_group, Q);
    dim3 warp_block(32 * kMaxHeadGroup);
    hisa_selected_score_bwd_warp_grouped_kernel<<<grid, warp_block, 0, stream>>>(
        grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
        grad_w, Q, H, D, L, K, requested_head_group);
  } else if (requested_head_group > 1 && H <= kMaxIndexerHeads) {
    dim3 grid((H + requested_head_group - 1) / requested_head_group, Q);
    hisa_selected_score_bwd_head_grouped_kernel<<<grid, block, 0, stream>>>(
        grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
        grad_w, Q, H, D, L, K, requested_head_group);
  } else if (use_row_grouped && H <= kMaxIndexerHeads) {
    dim3 grid(Q);
    hisa_selected_score_bwd_row_grouped_kernel<<<grid, block, 0, stream>>>(
        grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
        grad_w, Q, H, D, L, K);
  } else {
    dim3 grid(H, Q);
    hisa_selected_score_bwd_kernel<<<grid, block, 0, stream>>>(
        grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
        grad_w, Q, H, D, L, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_hisa_selected_score_bwd_batched(
    const float* grad_selected_scores, const void* q, const void* k,
    const void* weights, const void* topk_indices, float* grad_q,
    float* grad_k, float* grad_w, int Q, int B, int H, int D, int L, int K,
    int scalar_dtype, int weight_dtype, int topk_index_dtype, cudaStream_t stream) {
  if (H != kMaxIndexerHeads || D != kHeadDim) {
    C10_THROW_ERROR(ValueError, "batched HISA selected-score backward requires H=64 and D=128");
  }
  const char* tile_env = std::getenv("MEGATRON_HISA_SELECTED_SCORE_BWD_CUBLASDX_TILE_N");
  const int tile_n = tile_env == nullptr ? TILE_N32 : std::atoi(tile_env);
  if (scalar_dtype == kDTypeFloat32) {
    if (tile_n == TILE_N32) {
      using GEMM = HisaSelectedScoreBwdGemm64x32;
      dispatch_hisa_selected_score_bwd_batched_weight<GEMM, float, TILE_N32>(
          grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
          grad_w, Q, B, L, K, weight_dtype, topk_index_dtype, stream);
    } else if (tile_n == TILE_N128) {
      using GEMM = HisaSelectedScoreBwdGemm64x128;
      dispatch_hisa_selected_score_bwd_batched_weight<GEMM, float, TILE_N128>(
          grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
          grad_w, Q, B, L, K, weight_dtype, topk_index_dtype, stream);
    } else {
      using GEMM = HisaSelectedScoreBwdGemm64x64;
      dispatch_hisa_selected_score_bwd_batched_weight<GEMM, float, TILE_N64>(
          grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
          grad_w, Q, B, L, K, weight_dtype, topk_index_dtype, stream);
    }
  } else if (scalar_dtype == kDTypeBFloat16) {
    if (tile_n == TILE_N32) {
      using GEMM = HisaSelectedScoreBwdGemmBF16_64x32;
      dispatch_hisa_selected_score_bwd_batched_weight<GEMM, __nv_bfloat16, TILE_N32>(
          grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
          grad_w, Q, B, L, K, weight_dtype, topk_index_dtype, stream);
    } else if (tile_n == TILE_N128) {
      using GEMM = HisaSelectedScoreBwdGemmBF16_64x128;
      dispatch_hisa_selected_score_bwd_batched_weight<GEMM, __nv_bfloat16, TILE_N128>(
          grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
          grad_w, Q, B, L, K, weight_dtype, topk_index_dtype, stream);
    } else {
      using GEMM = HisaSelectedScoreBwdGemmBF16_64x64;
      dispatch_hisa_selected_score_bwd_batched_weight<GEMM, __nv_bfloat16, TILE_N64>(
          grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
          grad_w, Q, B, L, K, weight_dtype, topk_index_dtype, stream);
    }
  } else if (scalar_dtype == kDTypeFloat16) {
    if (tile_n == TILE_N32) {
      using GEMM = HisaSelectedScoreBwdGemmF16_64x32;
      dispatch_hisa_selected_score_bwd_batched_weight<GEMM, __half, TILE_N32>(
          grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
          grad_w, Q, B, L, K, weight_dtype, topk_index_dtype, stream);
    } else if (tile_n == TILE_N128) {
      using GEMM = HisaSelectedScoreBwdGemmF16_64x128;
      dispatch_hisa_selected_score_bwd_batched_weight<GEMM, __half, TILE_N128>(
          grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
          grad_w, Q, B, L, K, weight_dtype, topk_index_dtype, stream);
    } else {
      using GEMM = HisaSelectedScoreBwdGemmF16_64x64;
      dispatch_hisa_selected_score_bwd_batched_weight<GEMM, __half, TILE_N64>(
          grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
          grad_w, Q, B, L, K, weight_dtype, topk_index_dtype, stream);
    }
  } else {
    C10_THROW_ERROR(ValueError, "unsupported batched HISA selected-score q/k dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace hisa_indexer
}  // namespace megatron
