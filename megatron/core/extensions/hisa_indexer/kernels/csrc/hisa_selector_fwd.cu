// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// HISA selector forward CUDA kernel.
//
// This kernel removes the reference path's Python/CPU top-k selection from the
// training hot path. It intentionally emits only selector outputs:
//   * top-k token indices per query row
//   * selected indexer scores for validation / inference consumers
//
// Training-time gradients are produced by recomputing the final selected
// scores in PyTorch from the emitted indices. That keeps the memory footprint
// bounded and avoids materializing the full HISA backward cache
// [Q, candidate_len, H] in the normal SFT path.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cfloat>
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

__device__ __forceinline__ float block_sum_128(float v, float* scratch) {
  const int tid = threadIdx.x;
  v = warp_sum(v);
  if ((tid & 31) == 0) {
    scratch[tid >> 5] = v;
  }
  __syncthreads();
  if (tid < 32) {
    float s = tid < 4 ? scratch[tid] : 0.0f;
    s = warp_sum(s);
    if (tid == 0) {
      scratch[8] = s;
    }
  }
  __syncthreads();
  return scratch[8];
}

__device__ __forceinline__ int ceil_div_int(int x, int y) {
  return (x + y - 1) / y;
}

__device__ __forceinline__ bool is_selected_block(
    const int* selected_blocks, int count, int block_id) {
  for (int i = 0; i < count; ++i) {
    if (selected_blocks[i] == block_id) {
      return true;
    }
  }
  return false;
}

__device__ __forceinline__ void insert_topk(
    float score,
    int token_idx,
    float* top_scores,
    int32_t* top_indices,
    int topk,
    int* count) {
  if (token_idx < 0) {
    return;
  }
  if (*count < topk) {
    const int pos = *count;
    top_scores[pos] = score;
    top_indices[pos] = token_idx;
    *count += 1;
    return;
  }

  int min_pos = 0;
  float min_score = top_scores[0];
  for (int i = 1; i < topk; ++i) {
    const float s = top_scores[i];
    if (s < min_score) {
      min_score = s;
      min_pos = i;
    }
  }
  if (score > min_score) {
    top_scores[min_pos] = score;
    top_indices[min_pos] = token_idx;
  }
}

}  // namespace

__global__ void hisa_selector_fwd_kernel(
    const float* __restrict__ q,                  // [Q, H, D]
    const float* __restrict__ k,                  // [L, D]
    const float* __restrict__ block_reps,         // [MB, D]
    const float* __restrict__ weights,            // [Q, H]
    const int32_t* __restrict__ prefix_lens,      // [Q]
    const int32_t* __restrict__ block_topk_counts,// [Q]
    int32_t* __restrict__ topk_indices,           // [Q, K]
    float* __restrict__ selected_scores,          // [Q, K]
    int Q, int H, int D, int L, int MB, int block_size,
    int effective_block_topk, int topk_tokens,
    int force_first, int force_last, int force_last_minus_one) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= Q) {
    return;
  }

  extern __shared__ unsigned char smem_raw[];
  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* block_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * MB;
  int* selected_blocks = reinterpret_cast<int*>(cursor);
  cursor += sizeof(int) * effective_block_topk;
  float* top_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * topk_tokens;
  int32_t* top_indices = reinterpret_cast<int32_t*>(cursor);

  __shared__ float reduce_scratch[9];
  __shared__ float score_accum;
  __shared__ int candidate_count;

  for (int i = tid; i < MB; i += blockDim.x) {
    block_scores[i] = -INFINITY;
  }
  for (int i = tid; i < effective_block_topk; i += blockDim.x) {
    selected_blocks[i] = -1;
  }
  for (int i = tid; i < topk_tokens; i += blockDim.x) {
    top_scores[i] = -INFINITY;
    top_indices[i] = -1;
  }
  if (tid == 0) {
    candidate_count = 0;
  }
  __syncthreads();

  const int prefix_len = max(0, min(static_cast<int>(prefix_lens[row]), L));
  const int row_blocks = min(MB, ceil_div_int(prefix_len, block_size));
  if (prefix_len <= 0 || row_blocks <= 0) {
    for (int i = tid; i < topk_tokens; i += blockDim.x) {
      topk_indices[row * topk_tokens + i] = -1;
      selected_scores[row * topk_tokens + i] = -INFINITY;
    }
    return;
  }

  const float* q_row = q + static_cast<int64_t>(row) * H * D;
  const float* w_row = weights + static_cast<int64_t>(row) * H;

  // Stage 1: block scores over mean-pooled block representatives.
  for (int block_id = 0; block_id < row_blocks; ++block_id) {
    if (tid == 0) {
      score_accum = 0.0f;
    }
    __syncthreads();

    const float* rep = block_reps + static_cast<int64_t>(block_id) * D;
    const int block_start = block_id * block_size;
    const int block_end = min(block_start + block_size, prefix_len);
    const int block_token_count = max(1, block_end - block_start);
    const bool partial_final_block =
        block_end == prefix_len && block_token_count < block_size;
    for (int h = 0; h < H; ++h) {
      float partial = 0.0f;
      for (int d = tid; d < D; d += blockDim.x) {
        float rep_d = rep[d];
        if (partial_final_block) {
          float sum = 0.0f;
          for (int tok = block_start; tok < block_end; ++tok) {
            sum += k[static_cast<int64_t>(tok) * D + d];
          }
          rep_d = sum / static_cast<float>(block_token_count);
        }
        partial += q_row[static_cast<int64_t>(h) * D + d] * rep_d;
      }
      const float dot = block_sum_128(partial, reduce_scratch);
      if (tid == 0 && dot > 0.0f) {
        score_accum += dot * w_row[h];
      }
      __syncthreads();
    }
    if (tid == 0) {
      block_scores[block_id] = score_accum;
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (force_first) {
      block_scores[0] = INFINITY;
    }
    if (force_last) {
      block_scores[row_blocks - 1] = INFINITY;
    }
    if (force_last_minus_one && row_blocks >= 2) {
      block_scores[row_blocks - 2] = INFINITY;
    }

    int keep = block_topk_counts[row];
    keep = max(0, min(keep, effective_block_topk));
    keep = min(keep, row_blocks);

    for (int slot = 0; slot < keep; ++slot) {
      int best = -1;
      float best_score = -INFINITY;
      for (int block_id = 0; block_id < row_blocks; ++block_id) {
        if (is_selected_block(selected_blocks, slot, block_id)) {
          continue;
        }
        const float s = block_scores[block_id];
        if (s > best_score || (s == best_score && block_id < best)) {
          best_score = s;
          best = block_id;
        }
      }
      selected_blocks[slot] = best;
    }
  }
  __syncthreads();

  // Stage 2: candidate scores inside selected blocks + row-local top-k.
  const int keep = min(
      min(static_cast<int>(block_topk_counts[row]), effective_block_topk),
      row_blocks);
  for (int slot = 0; slot < keep; ++slot) {
    const int block_id = selected_blocks[slot];
    if (block_id < 0) {
      continue;
    }
    const int start = block_id * block_size;
    const int end = min(start + block_size, prefix_len);
    for (int tok = start; tok < end; ++tok) {
      if (tid == 0) {
        score_accum = 0.0f;
      }
      __syncthreads();

      const float* k_row = k + static_cast<int64_t>(tok) * D;
      for (int h = 0; h < H; ++h) {
        float partial = 0.0f;
        for (int d = tid; d < D; d += blockDim.x) {
          partial += q_row[static_cast<int64_t>(h) * D + d] * k_row[d];
        }
        const float dot = block_sum_128(partial, reduce_scratch);
        if (tid == 0 && dot > 0.0f) {
          score_accum += dot * w_row[h];
        }
        __syncthreads();
      }
      if (tid == 0) {
        insert_topk(score_accum, tok, top_scores, top_indices, topk_tokens, &candidate_count);
      }
      __syncthreads();
    }
  }

  for (int i = tid; i < topk_tokens; i += blockDim.x) {
    topk_indices[static_cast<int64_t>(row) * topk_tokens + i] = top_indices[i];
    selected_scores[static_cast<int64_t>(row) * topk_tokens + i] = top_scores[i];
  }
}

void launch_hisa_selector_fwd(
    const float* q, const float* k, const float* block_reps,
    const float* weights, const int32_t* prefix_lens,
    const int32_t* block_topk_counts, int32_t* topk_indices,
    float* selected_scores, int Q, int H, int D, int L, int MB,
    int block_size, int effective_block_topk, int topk_tokens,
    int force_first, int force_last, int force_last_minus_one,
    cudaStream_t stream) {
  const int threads = kHeadDim;
  const size_t smem_bytes =
      sizeof(float) * static_cast<size_t>(MB)
      + sizeof(int) * static_cast<size_t>(effective_block_topk)
      + sizeof(float) * static_cast<size_t>(topk_tokens)
      + sizeof(int32_t) * static_cast<size_t>(topk_tokens);
  hisa_selector_fwd_kernel<<<Q, threads, smem_bytes, stream>>>(
      q, k, block_reps, weights, prefix_lens, block_topk_counts,
      topk_indices, selected_scores, Q, H, D, L, MB, block_size,
      effective_block_topk, topk_tokens, force_first, force_last,
      force_last_minus_one);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void hisa_selector_teacher_fwd_kernel(
    const float* __restrict__ q,                  // [Q, H, D]
    const float* __restrict__ k,                  // [L, D]
    const float* __restrict__ block_reps,         // [MB, D]
    const float* __restrict__ weights,            // [Q, H]
    const float* __restrict__ attn_query,         // [Q, AH, AD]
    const float* __restrict__ attn_key,           // [L, AH, AD]
    const int32_t* __restrict__ prefix_lens,      // [Q]
    const int32_t* __restrict__ block_topk_counts,// [Q]
    int32_t* __restrict__ topk_indices,           // [Q, K]
    float* __restrict__ selected_scores,          // [Q, K]
    float* __restrict__ teacher_probs,            // [Q, K]
    int Q, int H, int D, int L, int MB, int AH, int AD, int block_size,
    int effective_block_topk, int topk_tokens, float softmax_scale,
    int force_first, int force_last, int force_last_minus_one) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= Q) {
    return;
  }

  extern __shared__ unsigned char smem_raw[];
  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* block_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * MB;
  int* selected_blocks = reinterpret_cast<int*>(cursor);
  cursor += sizeof(int) * effective_block_topk;
  float* top_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * topk_tokens;
  int32_t* top_indices = reinterpret_cast<int32_t*>(cursor);
  cursor += sizeof(int32_t) * topk_tokens;
  float* teacher_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * topk_tokens;
  float* teacher_mass = reinterpret_cast<float*>(cursor);

  __shared__ float reduce_scratch[9];
  __shared__ float score_accum;
  __shared__ int candidate_count;

  for (int i = tid; i < MB; i += blockDim.x) {
    block_scores[i] = -INFINITY;
  }
  for (int i = tid; i < effective_block_topk; i += blockDim.x) {
    selected_blocks[i] = -1;
  }
  for (int i = tid; i < topk_tokens; i += blockDim.x) {
    top_scores[i] = -INFINITY;
    top_indices[i] = -1;
    teacher_scores[i] = -INFINITY;
    teacher_mass[i] = 0.0f;
  }
  if (tid == 0) {
    candidate_count = 0;
  }
  __syncthreads();

  const int prefix_len = max(0, min(static_cast<int>(prefix_lens[row]), L));
  const int row_blocks = min(MB, ceil_div_int(prefix_len, block_size));
  if (prefix_len <= 0 || row_blocks <= 0) {
    for (int i = tid; i < topk_tokens; i += blockDim.x) {
      topk_indices[row * topk_tokens + i] = -1;
      selected_scores[row * topk_tokens + i] = -INFINITY;
      teacher_probs[row * topk_tokens + i] = 0.0f;
    }
    return;
  }

  const float* q_row = q + static_cast<int64_t>(row) * H * D;
  const float* w_row = weights + static_cast<int64_t>(row) * H;

  // Stage 1: block scores over mean-pooled block representatives.
  for (int block_id = 0; block_id < row_blocks; ++block_id) {
    if (tid == 0) {
      score_accum = 0.0f;
    }
    __syncthreads();

    const float* rep = block_reps + static_cast<int64_t>(block_id) * D;
    const int block_start = block_id * block_size;
    const int block_end = min(block_start + block_size, prefix_len);
    const int block_token_count = max(1, block_end - block_start);
    const bool partial_final_block =
        block_end == prefix_len && block_token_count < block_size;
    for (int h = 0; h < H; ++h) {
      float partial = 0.0f;
      for (int d = tid; d < D; d += blockDim.x) {
        float rep_d = rep[d];
        if (partial_final_block) {
          float sum = 0.0f;
          for (int tok = block_start; tok < block_end; ++tok) {
            sum += k[static_cast<int64_t>(tok) * D + d];
          }
          rep_d = sum / static_cast<float>(block_token_count);
        }
        partial += q_row[static_cast<int64_t>(h) * D + d] * rep_d;
      }
      const float dot = block_sum_128(partial, reduce_scratch);
      if (tid == 0 && dot > 0.0f) {
        score_accum += dot * w_row[h];
      }
      __syncthreads();
    }
    if (tid == 0) {
      block_scores[block_id] = score_accum;
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (force_first) {
      block_scores[0] = INFINITY;
    }
    if (force_last) {
      block_scores[row_blocks - 1] = INFINITY;
    }
    if (force_last_minus_one && row_blocks >= 2) {
      block_scores[row_blocks - 2] = INFINITY;
    }

    int keep = block_topk_counts[row];
    keep = max(0, min(keep, effective_block_topk));
    keep = min(keep, row_blocks);

    for (int slot = 0; slot < keep; ++slot) {
      int best = -1;
      float best_score = -INFINITY;
      for (int block_id = 0; block_id < row_blocks; ++block_id) {
        if (is_selected_block(selected_blocks, slot, block_id)) {
          continue;
        }
        const float s = block_scores[block_id];
        if (s > best_score || (s == best_score && block_id < best)) {
          best_score = s;
          best = block_id;
        }
      }
      selected_blocks[slot] = best;
    }
  }
  __syncthreads();

  // Stage 2: candidate scores inside selected blocks + row-local top-k.
  const int keep = min(
      min(static_cast<int>(block_topk_counts[row]), effective_block_topk),
      row_blocks);
  for (int slot = 0; slot < keep; ++slot) {
    const int block_id = selected_blocks[slot];
    if (block_id < 0) {
      continue;
    }
    const int start = block_id * block_size;
    const int end = min(start + block_size, prefix_len);
    for (int tok = start; tok < end; ++tok) {
      if (tid == 0) {
        score_accum = 0.0f;
      }
      __syncthreads();

      const float* k_row = k + static_cast<int64_t>(tok) * D;
      for (int h = 0; h < H; ++h) {
        float partial = 0.0f;
        for (int d = tid; d < D; d += blockDim.x) {
          partial += q_row[static_cast<int64_t>(h) * D + d] * k_row[d];
        }
        const float dot = block_sum_128(partial, reduce_scratch);
        if (tid == 0 && dot > 0.0f) {
          score_accum += dot * w_row[h];
        }
        __syncthreads();
      }
      if (tid == 0) {
        insert_topk(score_accum, tok, top_scores, top_indices, topk_tokens, &candidate_count);
      }
      __syncthreads();
    }
  }

  // Stage 3: local attention-teacher distribution over the selected top-k.
  for (int ah = 0; ah < AH; ++ah) {
    for (int i = tid; i < topk_tokens; i += blockDim.x) {
      const int tok = top_indices[i];
      float score = -INFINITY;
      if (tok >= 0 && tok < prefix_len) {
        float partial = 0.0f;
        const float* query_head =
            attn_query + (static_cast<int64_t>(row) * AH + ah) * AD;
        const float* key_head =
            attn_key + (static_cast<int64_t>(tok) * AH + ah) * AD;
        for (int d = 0; d < AD; ++d) {
          partial += query_head[d] * key_head[d];
        }
        score = partial * softmax_scale;
      }
      teacher_scores[i] = score;
    }
    __syncthreads();

    if (tid == 0) {
      float max_score = -INFINITY;
      for (int i = 0; i < topk_tokens; ++i) {
        max_score = max(max_score, teacher_scores[i]);
      }
      float denom = 0.0f;
      for (int i = 0; i < topk_tokens; ++i) {
        const float s = teacher_scores[i];
        if (s > -INFINITY) {
          denom += expf(s - max_score);
        }
      }
      const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
      for (int i = 0; i < topk_tokens; ++i) {
        const float s = teacher_scores[i];
        if (s > -INFINITY) {
          teacher_mass[i] += expf(s - max_score) * inv_denom;
        }
      }
    }
    __syncthreads();
  }

  for (int i = tid; i < topk_tokens; i += blockDim.x) {
    topk_indices[static_cast<int64_t>(row) * topk_tokens + i] = top_indices[i];
    selected_scores[static_cast<int64_t>(row) * topk_tokens + i] = top_scores[i];
    teacher_probs[static_cast<int64_t>(row) * topk_tokens + i] = teacher_mass[i];
  }
}

void launch_hisa_selector_teacher_fwd(
    const float* q, const float* k, const float* block_reps,
    const float* weights, const float* attn_query, const float* attn_key,
    const int32_t* prefix_lens, const int32_t* block_topk_counts,
    int32_t* topk_indices, float* selected_scores, float* teacher_probs,
    int Q, int H, int D, int L, int MB, int AH, int AD, int block_size,
    int effective_block_topk, int topk_tokens, float softmax_scale,
    int force_first, int force_last, int force_last_minus_one,
    cudaStream_t stream) {
  const int threads = kHeadDim;
  const size_t smem_bytes =
      sizeof(float) * static_cast<size_t>(MB)
      + sizeof(int) * static_cast<size_t>(effective_block_topk)
      + sizeof(float) * static_cast<size_t>(topk_tokens)
      + sizeof(int32_t) * static_cast<size_t>(topk_tokens)
      + sizeof(float) * static_cast<size_t>(topk_tokens)
      + sizeof(float) * static_cast<size_t>(topk_tokens);
  hisa_selector_teacher_fwd_kernel<<<Q, threads, smem_bytes, stream>>>(
      q, k, block_reps, weights, attn_query, attn_key, prefix_lens,
      block_topk_counts, topk_indices, selected_scores, teacher_probs,
      Q, H, D, L, MB, AH, AD, block_size, effective_block_topk, topk_tokens,
      softmax_scale, force_first, force_last, force_last_minus_one);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace hisa_indexer
}  // namespace megatron
