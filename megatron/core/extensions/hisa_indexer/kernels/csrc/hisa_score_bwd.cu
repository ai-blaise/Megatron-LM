// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// HISA 4:1 indexer backward CUDA kernel.
//
// Implements the analytic backward through the two weighted-ReLU DSA score
// formulas + the mean-pool stage. The forward kernel that produces the
// intermediates lives at
//   optimization-playground/python/sglang/jit_kernel/csrc/nsa/nvfp4_indexer_quant.cuh
// (commit 07ead85c5). The math derivation is documented in
//   megatron/core/extensions/hisa_indexer/__init__.py and
//   megatron/core/extensions/hisa_indexer/reference.py.
//
// References cited (project rule 6):
//   * CUTLASS examples/65_blackwell_pingpong_grouped_gemm — Blackwell CTA-pair
//     workflow we adopt in the perf-optimization iterations.
//   * CuTe DSL / Colfax blockscaled MMA tutorial — NVF4 dequant ladder for
//     recomputing k_s on the bwd path.
//   * vLLM csrc/moe/topk_softmax_kernels.cu::topkGating — XOR-butterfly warp
//     reduction.
//   * HIGGS Megatron backward
//     (megatron/core/quantization/higgs/kernels/csrc/higgs_kv_bwd.cu) — STE
//     detach pattern.
//   * fast-hadamard-transform (Dao-AILab) csrc/*.cu — warp shuffle reductions.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace megatron {
namespace hisa_indexer {

namespace {

// Warp-level shuffle reduction; mirrors vLLM and fast-hadamard-transform.
__device__ inline float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, offset, 32);
  }
  return v;
}

constexpr int kHeadDim = 128;

}  // namespace

// ============================================================================
// candidate_score backward
//   I_{t,s} = sum_h w_{t,h} * ReLU(q_{t,h} . k_s)
//   d I / d q_{t,h} = w_{t,h} * H(dot_{t,s,h}) * k_s
//   d I / d k_s     = sum_h w_{t,h} * H(dot_{t,s,h}) * q_{t,h}
//   d I / d w_{t,h} = ReLU(dot_{t,s,h})
// One CTA per (row, candidate-slot). Each warp owns one head; reduces along D
// for k_s, then atomically scatters into grad_q[row,head,:] and
// grad_k[token,:]. grad_w is summed per row across candidates with one CTA
// per row.
// ============================================================================

__global__ void hisa_candidate_score_bwd_kernel(
    const float* __restrict__ grad_cand_score,    // [Q, CL]
    const float* __restrict__ q,                  // [Q, H, D]
    const float* __restrict__ k_concat,           // [N_K, D]
    const int32_t* __restrict__ k_offsets,        // [B+1]
    const float* __restrict__ weights,            // [Q, H]
    const int32_t* __restrict__ token_to_batch,   // [Q]
    const int32_t* __restrict__ candidate_indices,// [Q, CL] in concat space (per-batch row + offset)
    const float* __restrict__ candidate_dot,      // [Q, CL, H]
    const uint8_t* __restrict__ selected_cand_mask, // [Q, CL]
    float* __restrict__ grad_q,                   // [Q, H, D]
    float* __restrict__ grad_k_concat,            // [N_K, D]
    float* __restrict__ grad_w,                   // [Q, H]
    int Q, int H, int CL, int D) {
  const int row = blockIdx.y;
  const int slot = blockIdx.x;
  if (row >= Q || slot >= CL) return;

  const int idx = row * CL + slot;
  if (selected_cand_mask[idx] == 0) return;
  const float g_score = grad_cand_score[idx];
  if (g_score == 0.f) return;
  const int32_t token_in_batch = candidate_indices[idx];
  if (token_in_batch < 0) return;
  const int batch = token_to_batch[row];
  const int token_global = k_offsets[batch] + token_in_batch;

  const float* q_row = q + (static_cast<int64_t>(row) * H) * D;
  const float* w_row = weights + static_cast<int64_t>(row) * H;
  const float* dot_row = candidate_dot + (static_cast<int64_t>(row) * CL + slot) * H;

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  // One warp per head, but if H > 8 we tile.
  for (int h = warp_id; h < H; h += blockDim.x / 32) {
    const float dot_th = dot_row[h];
    if (dot_th <= 0.f) continue;  // ReLU gate
    const float w_th = w_row[h];
    const float scale_qk = g_score * w_th;
    const float* k_ptr = k_concat + static_cast<int64_t>(token_global) * D;
    const float* q_ptr = q_row + static_cast<int64_t>(h) * D;
    float* gq_ptr = grad_q + (static_cast<int64_t>(row) * H + h) * D;
    float* gk_ptr = grad_k_concat + static_cast<int64_t>(token_global) * D;
    // d q[h,d] += scale_qk * k[d]
    // d k[d]   += scale_qk * q[h,d]
    for (int d = lane; d < D; d += 32) {
      const float k_d = k_ptr[d];
      const float q_d = q_ptr[d];
      atomicAdd(gq_ptr + d, scale_qk * k_d);
      atomicAdd(gk_ptr + d, scale_qk * q_d);
    }
    if (lane == 0) {
      // d w[h] += g_score * ReLU(dot[h]) (ReLU == dot since dot > 0).
      atomicAdd(grad_w + static_cast<int64_t>(row) * H + h, g_score * dot_th);
    }
  }
}

// ============================================================================
// block_score backward — same structure but the K row is the mean of a
// block. mean_pool jacobian distributes 1/N_b to each token in the block.
// ============================================================================

__global__ void hisa_block_score_bwd_kernel(
    const float* __restrict__ grad_block_score,    // [Q, MB]
    const float* __restrict__ q,                   // [Q, H, D]
    const float* __restrict__ k_concat,            // [N_K, D]
    const int32_t* __restrict__ k_offsets,         // [B+1]
    const float* __restrict__ weights,             // [Q, H]
    const int32_t* __restrict__ token_to_batch,    // [Q]
    const int32_t* __restrict__ prefix_lens,       // [Q]
    const int32_t* __restrict__ top_blocks,        // [Q, TB]
    const float* __restrict__ block_dot,           // [Q, MB, H]
    const uint8_t* __restrict__ selected_block_mask, // [Q, MB]
    float* __restrict__ grad_q,                    // [Q, H, D]
    float* __restrict__ grad_k_concat,             // [N_K, D]
    float* __restrict__ grad_w,                    // [Q, H]
    int Q, int H, int MB, int D, int block_size, int TB) {
  const int row = blockIdx.y;
  const int slot = blockIdx.x;
  if (row >= Q || slot >= TB) return;
  const int b = top_blocks[row * TB + slot];
  if (b < 0 || b >= MB) return;
  const int idx = row * MB + b;
  if (selected_block_mask[idx] == 0) return;
  const int batch = token_to_batch[row];
  const int L = prefix_lens[row];
  const int token_start_in_batch = b * block_size;
  if (token_start_in_batch >= L) return;
  const int token_count = min(block_size, L - token_start_in_batch);
  if (token_count <= 0) return;
  const float inv_n = 1.f / static_cast<float>(token_count);
  const float g_score = grad_block_score[idx];
  if (g_score == 0.f) return;

  const float* q_row = q + (static_cast<int64_t>(row) * H) * D;
  const float* w_row = weights + static_cast<int64_t>(row) * H;
  const float* dot_row = block_dot + (static_cast<int64_t>(row) * MB + b) * H;

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  // Compute mean k_block locally (reusing k_concat). We need k_block to
  // accumulate grad_q.
  __shared__ float k_block_smem[kHeadDim];
  for (int d = threadIdx.x; d < D; d += blockDim.x) k_block_smem[d] = 0.f;
  __syncthreads();
  const int token_start_global = k_offsets[batch] + token_start_in_batch;
  for (int s = 0; s < token_count; ++s) {
    const float* k_ptr = k_concat + static_cast<int64_t>(token_start_global + s) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      k_block_smem[d] += k_ptr[d] * inv_n;
    }
  }
  __syncthreads();

  for (int h = warp_id; h < H; h += blockDim.x / 32) {
    const float dot_bh = dot_row[h];
    if (dot_bh <= 0.f) continue;  // ReLU gate
    const float w_h = w_row[h];
    const float scale = g_score * w_h;
    const float* q_ptr = q_row + static_cast<int64_t>(h) * D;
    float* gq_ptr = grad_q + (static_cast<int64_t>(row) * H + h) * D;
    for (int d = lane; d < D; d += 32) {
      // d q[h,d] += scale * k_block[d]
      atomicAdd(gq_ptr + d, scale * k_block_smem[d]);
      // d k_block[d] += scale * q[h,d]; then mean_pool jacobian gives
      // d k_s[d] += scale * q[h,d] / N_b for s in block.
      const float dk = scale * q_ptr[d] * inv_n;
      for (int s = 0; s < token_count; ++s) {
        float* gk_ptr = grad_k_concat + static_cast<int64_t>(token_start_global + s) * D;
        atomicAdd(gk_ptr + d, dk);
      }
    }
    if (lane == 0) {
      // d w[h] += g_score * ReLU(block_dot[h])
      atomicAdd(grad_w + static_cast<int64_t>(row) * H + h, g_score * dot_bh);
    }
  }
}

// ============================================================================
// Launcher (called from pybind shim).
// ============================================================================

void launch_hisa_score_bwd(
    const float* grad_cand_score, const float* grad_block_score,
    const float* q, const float* k_concat, const int32_t* k_offsets,
    const float* weights, const int32_t* token_to_batch,
    const int32_t* prefix_lens, const int32_t* top_blocks,
    const int32_t* candidate_indices, const float* candidate_dot,
    const float* block_dot, const uint8_t* selected_block_mask,
    const uint8_t* selected_cand_mask, float* grad_q, float* grad_k_concat,
    float* grad_w, int Q, int H, int D, int CL, int MB, int block_size,
    int TB, cudaStream_t stream) {
  // Candidate-score backward.
  {
    dim3 grid(CL, Q);
    dim3 block(256);
    hisa_candidate_score_bwd_kernel<<<grid, block, 0, stream>>>(
        grad_cand_score, q, k_concat, k_offsets, weights, token_to_batch,
        candidate_indices, candidate_dot, selected_cand_mask, grad_q,
        grad_k_concat, grad_w, Q, H, CL, D);
  }
  // Block-score backward.
  {
    dim3 grid(TB, Q);
    dim3 block(256);
    hisa_block_score_bwd_kernel<<<grid, block, 0, stream>>>(
        grad_block_score, q, k_concat, k_offsets, weights, token_to_batch,
        prefix_lens, top_blocks, block_dot, selected_block_mask, grad_q,
        grad_k_concat, grad_w, Q, H, MB, D, block_size, TB);
  }
}

}  // namespace hisa_indexer
}  // namespace megatron
