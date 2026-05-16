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
#include <cuda_runtime.h>

#include <cstdint>

namespace megatron {
namespace hisa_indexer {

namespace {

constexpr int kHeadDim = 128;

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
  const int slot = blockIdx.x;
  const int row = blockIdx.y;
  if (row >= Q || slot >= K) {
    return;
  }

  const int token = topk_indices[static_cast<int64_t>(row) * K + slot];
  if (token < 0 || token >= L) {
    return;
  }
  const float g_score = grad_selected_scores[static_cast<int64_t>(row) * K + slot];
  if (g_score == 0.0f) {
    return;
  }

  __shared__ float reduce_scratch[9];
  const int tid = threadIdx.x;
  const float* q_row = q + static_cast<int64_t>(row) * H * D;
  const float* k_row = k + static_cast<int64_t>(token) * D;
  const float* w_row = weights + static_cast<int64_t>(row) * H;

  for (int h = 0; h < H; ++h) {
    float partial = 0.0f;
    const float* q_head = q_row + static_cast<int64_t>(h) * D;
    for (int d = tid; d < D; d += blockDim.x) {
      partial += q_head[d] * k_row[d];
    }
    const float dot = block_sum(partial, reduce_scratch);
    if (dot <= 0.0f) {
      continue;
    }

    const float scale = g_score * w_row[h];
    float* grad_q_head = grad_q + (static_cast<int64_t>(row) * H + h) * D;
    float* grad_k_row = grad_k + static_cast<int64_t>(token) * D;
    for (int d = tid; d < D; d += blockDim.x) {
      const float q_d = q_head[d];
      const float k_d = k_row[d];
      atomicAdd(grad_q_head + d, scale * k_d);
      atomicAdd(grad_k_row + d, scale * q_d);
    }
    if (tid == 0) {
      atomicAdd(grad_w + static_cast<int64_t>(row) * H + h, g_score * dot);
    }
    __syncthreads();
  }
}

}  // namespace

void launch_hisa_selected_score_bwd(
    const float* grad_selected_scores, const float* q, const float* k,
    const float* weights, const int32_t* topk_indices, float* grad_q,
    float* grad_k, float* grad_w, int Q, int H, int D, int L, int K,
    cudaStream_t stream) {
  dim3 grid(K, Q);
  dim3 block(kHeadDim);
  hisa_selected_score_bwd_kernel<<<grid, block, 0, stream>>>(
      grad_selected_scores, q, k, weights, topk_indices, grad_q, grad_k,
      grad_w, Q, H, D, L, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace hisa_indexer
}  // namespace megatron
