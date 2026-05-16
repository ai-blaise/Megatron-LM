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
#include <c10/util/Exception.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// torch.utils.cpp_extension injects these guards globally. MathDx/CuBLASDx
// instantiates common half/half2 helpers even for float GEMMs, so the operators
// must be available in this translation unit.
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#include <cublasdx.hpp>
#include <cub/block/block_radix_sort.cuh>

#include <cfloat>
#include <climits>
#include <cstdint>
#include <type_traits>

namespace megatron {
namespace hisa_indexer {

namespace {

constexpr int kHeadDim = 128;
constexpr int kCublasDxIndexerHeads = 64;
constexpr int kCublasDxTileN = 64;

using HisaSelectorGemm64x16 = decltype(
    cublasdx::Size<kCublasDxIndexerHeads, kCublasDxTileN, kHeadDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using HisaSelectorGemm64x16Fp8 = decltype(
    cublasdx::Size<kCublasDxIndexerHeads, kCublasDxTileN, kHeadDim>()
    + cublasdx::Precision<__nv_fp8_e4m3, __nv_fp8_e4m3, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
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

__device__ __forceinline__ float block_max_128(float v, float* scratch) {
  const int tid = threadIdx.x;
  v = warp_max(v);
  if ((tid & 31) == 0) {
    scratch[tid >> 5] = v;
  }
  __syncthreads();
  if (tid < 32) {
    float s = tid < 4 ? scratch[tid] : -INFINITY;
    s = warp_max(s);
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

__device__ __forceinline__ bool candidate_better(
    float score_a,
    int ordinal_a,
    float score_b,
    int ordinal_b) {
  if (score_a > score_b) {
    return true;
  }
  if (score_a < score_b) {
    return false;
  }
  return ordinal_a < ordinal_b;
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

inline int next_power_of_two_int(int x) {
  if (x <= 1) {
    return 1;
  }
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

__device__ __forceinline__ float e2m1_code_to_float(uint8_t code) {
  constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = lut[code & 0x7];
  return (code & 0x8) ? -magnitude : magnitude;
}

__device__ __forceinline__ float nvfp4_scale_from_word(int32_t packed_scale, int group) {
  const uint32_t word = static_cast<uint32_t>(packed_scale);
  const uint32_t exp = (word >> (group * 8)) & 0xffu;
  return __uint_as_float(exp << 23);
}

__device__ __forceinline__ float load_nvfp4_packed_row_dim(
    const uint8_t* __restrict__ packed_values,
    const int32_t* __restrict__ packed_scales,
    int packed_row,
    int dim) {
  const uint8_t byte = packed_values[static_cast<int64_t>(packed_row) * 64 + (dim >> 1)];
  const uint8_t code = ((dim & 1) == 0) ? (byte & 0x0f) : ((byte >> 4) & 0x0f);
  const int group = dim >> 5;
  return e2m1_code_to_float(code) * nvfp4_scale_from_word(packed_scales[packed_row], group);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t make_mma_value(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ __forceinline__ __nv_fp8_e4m3 make_mma_value<__nv_fp8_e4m3>(float value) {
  return __nv_fp8_e4m3(value);
}

__device__ __forceinline__ char* align_dynamic_smem(char* ptr, uintptr_t alignment) {
  const uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
  return reinterpret_cast<char*>((raw + alignment - 1) & ~(alignment - 1));
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

__global__ void hisa_selector_nvfp4_fwd_kernel(
    const float* __restrict__ q,                  // [Q, H, D]
    const uint8_t* __restrict__ packed_values,    // [N, 64]
    const int32_t* __restrict__ packed_scales,    // [N]
    const float* __restrict__ block_reps,         // [MB, D]
    const float* __restrict__ weights,            // [Q, H]
    const int32_t* __restrict__ prefix_lens,      // [Q]
    const int32_t* __restrict__ block_topk_counts,// [Q]
    int32_t* __restrict__ topk_indices,           // [Q, K]
    float* __restrict__ selected_scores,          // [Q, K]
    int Q, int H, int D, int L, int packed_row_offset,
    int packed_row_stride, int MB, int block_size,
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

  // Stage 1: block scores over dense mean-pooled reps. The partial final
  // block is recomputed from packed NVFP4 rows so causal rows never see future
  // tokens inside the current block.
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
            const int packed_row = packed_row_offset + tok * packed_row_stride;
            sum += load_nvfp4_packed_row_dim(
                packed_values, packed_scales, packed_row, d);
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

  // Stage 2: candidate scores inside selected blocks + row-local top-k, with
  // K loaded directly from the packed NVFP4 IndexCache sidecar.
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

      const int packed_row = packed_row_offset + tok * packed_row_stride;
      for (int h = 0; h < H; ++h) {
        float partial = 0.0f;
        for (int d = tid; d < D; d += blockDim.x) {
          const float k_d = load_nvfp4_packed_row_dim(
              packed_values, packed_scales, packed_row, d);
          partial += q_row[static_cast<int64_t>(h) * D + d] * k_d;
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

void launch_hisa_selector_nvfp4_fwd(
    const float* q, const uint8_t* packed_values,
    const int32_t* packed_scales, const float* block_reps,
    const float* weights, const int32_t* prefix_lens,
    const int32_t* block_topk_counts, int32_t* topk_indices,
    float* selected_scores, int Q, int H, int D, int L, int packed_row_offset,
    int packed_row_stride, int MB, int block_size, int effective_block_topk,
    int topk_tokens, int force_first, int force_last,
    int force_last_minus_one, cudaStream_t stream) {
  const int threads = kHeadDim;
  const size_t smem_bytes =
      sizeof(float) * static_cast<size_t>(MB)
      + sizeof(int) * static_cast<size_t>(effective_block_topk)
      + sizeof(float) * static_cast<size_t>(topk_tokens)
      + sizeof(int32_t) * static_cast<size_t>(topk_tokens);
  hisa_selector_nvfp4_fwd_kernel<<<Q, threads, smem_bytes, stream>>>(
      q, packed_values, packed_scales, block_reps, weights, prefix_lens,
      block_topk_counts, topk_indices, selected_scores, Q, H, D, L,
      packed_row_offset, packed_row_stride, MB, block_size,
      effective_block_topk, topk_tokens, force_first, force_last,
      force_last_minus_one);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class GEMM, class MmaScalar>
__global__ void hisa_selector_nvfp4_cublasdx_fwd_kernel(
    const float* __restrict__ q,                  // [Q, 64, 128]
    const uint8_t* __restrict__ packed_values,    // [N, 64]
    const int32_t* __restrict__ packed_scales,    // [N]
    const float* __restrict__ block_reps,         // [MB, 128]
    const float* __restrict__ weights,            // [Q, 64]
    const int32_t* __restrict__ prefix_lens,      // [Q]
    const int32_t* __restrict__ block_topk_counts,// [Q]
    int32_t* __restrict__ topk_indices,           // [Q, K]
    float* __restrict__ selected_scores,          // [Q, K]
    int Q, int L, int packed_row_offset, int packed_row_stride, int MB,
    int block_size, int effective_block_topk, int topk_tokens,
    int candidate_capacity,
    int force_first, int force_last, int force_last_minus_one) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= Q) {
    return;
  }

  extern __shared__ __align__(16) unsigned char smem_raw[];
  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* block_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * MB;
  int* selected_blocks = reinterpret_cast<int*>(cursor);
  cursor += sizeof(int) * effective_block_topk;
  float* candidate_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * candidate_capacity;
  int32_t* candidate_indices = reinterpret_cast<int32_t*>(cursor);
  cursor += sizeof(int32_t) * candidate_capacity;
  int32_t* candidate_ordinals = reinterpret_cast<int32_t*>(cursor);
  cursor += sizeof(int32_t) * candidate_capacity;
  cursor = align_dynamic_smem(cursor, 16);
  auto gemm_smem = reinterpret_cast<void*>(cursor);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  __shared__ float reduce_scratch[9];
  __shared__ float score_accum;

  for (int i = tid; i < MB; i += blockDim.x) {
    block_scores[i] = -INFINITY;
  }
  for (int i = tid; i < effective_block_topk; i += blockDim.x) {
    selected_blocks[i] = -1;
  }
  for (int i = tid; i < candidate_capacity; i += blockDim.x) {
    candidate_scores[i] = -INFINITY;
    candidate_indices[i] = -1;
    candidate_ordinals[i] = INT_MAX;
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

  const float* q_row = q + static_cast<int64_t>(row) * kCublasDxIndexerHeads * kHeadDim;
  const float* w_row = weights + static_cast<int64_t>(row) * kCublasDxIndexerHeads;

  // A is reused for block-rep scoring and candidate-token scoring in this row.
  for (int idx = tid; idx < kCublasDxIndexerHeads * kHeadDim; idx += blockDim.x) {
    const int h = idx / kHeadDim;
    const int d = idx - h * kHeadDim;
    a_shared(h, d) = make_mma_value<MmaScalar>(
        q_row[static_cast<int64_t>(h) * kHeadDim + d]);
  }
  __syncthreads();

  if constexpr (std::is_same_v<MmaScalar, float>) {
    // Stage 1: score dense block representatives with the same cuBLASDx tile
    // used for token refinement. This keeps the HISA selector's first phase on
    // tensor cores for the exact-fp32 backend. The only scalar correction is
    // the causal partial final block, whose representative depends on this
    // query row's prefix length.
    for (int tile_start = 0; tile_start < row_blocks; tile_start += kCublasDxTileN) {
      const int tile_count = min(kCublasDxTileN, row_blocks - tile_start);

      for (int idx = tid; idx < kHeadDim * kCublasDxTileN; idx += blockDim.x) {
        const int d = idx / kCublasDxTileN;
        const int n = idx - d * kCublasDxTileN;
        float value = 0.0f;
        if (n < tile_count) {
          value = block_reps[static_cast<int64_t>(tile_start + n) * kHeadDim + d];
        }
        b_shared(d, n) = make_mma_value<MmaScalar>(value);
      }
      for (int idx = tid; idx < kCublasDxIndexerHeads * kCublasDxTileN; idx += blockDim.x) {
        const int h = idx / kCublasDxTileN;
        const int n = idx - h * kCublasDxTileN;
        c_shared(h, n) = 0.0f;
      }
      __syncthreads();

      GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
      __syncthreads();

      for (int n = tid; n < kCublasDxTileN; n += blockDim.x) {
        if (n < tile_count) {
          float score = 0.0f;
          for (int h = 0; h < kCublasDxIndexerHeads; ++h) {
            const float dot = c_shared(h, n);
            if (dot > 0.0f) {
              score += dot * w_row[h];
            }
          }
          block_scores[tile_start + n] = score;
        }
      }
      __syncthreads();
    }

    const int final_block = row_blocks - 1;
    const int block_start = final_block * block_size;
    const int block_end = min(block_start + block_size, prefix_len);
    const int block_token_count = max(1, block_end - block_start);
    if (block_token_count < block_size) {
      if (tid == 0) {
        score_accum = 0.0f;
      }
      __syncthreads();
      for (int h = 0; h < kCublasDxIndexerHeads; ++h) {
        float partial = 0.0f;
        for (int d = tid; d < kHeadDim; d += blockDim.x) {
          float sum = 0.0f;
          for (int tok = block_start; tok < block_end; ++tok) {
            const int packed_row = packed_row_offset + tok * packed_row_stride;
            sum += load_nvfp4_packed_row_dim(
                packed_values, packed_scales, packed_row, d);
          }
          partial += q_row[static_cast<int64_t>(h) * kHeadDim + d]
              * (sum / static_cast<float>(block_token_count));
        }
        const float dot = block_sum_128(partial, reduce_scratch);
        if (tid == 0 && dot > 0.0f) {
          score_accum += dot * w_row[h];
        }
        __syncthreads();
      }
      if (tid == 0) {
        block_scores[final_block] = score_accum;
      }
      __syncthreads();
    }
  } else {
    // Stage 1 for the FP8 MMA experiment intentionally stays scalar-fp32.
    // The FP8 backend validates the candidate-token payload against an FP8
    // oracle; changing block selection precision here would mix two separate
    // experiments and make top-k changes harder to reason about.
    for (int block_id = 0; block_id < row_blocks; ++block_id) {
      if (tid == 0) {
        score_accum = 0.0f;
      }
      __syncthreads();

      const float* rep = block_reps + static_cast<int64_t>(block_id) * kHeadDim;
      const int block_start = block_id * block_size;
      const int block_end = min(block_start + block_size, prefix_len);
      const int block_token_count = max(1, block_end - block_start);
      const bool partial_final_block =
          block_end == prefix_len && block_token_count < block_size;
      for (int h = 0; h < kCublasDxIndexerHeads; ++h) {
        float partial = 0.0f;
        for (int d = tid; d < kHeadDim; d += blockDim.x) {
          float rep_d = rep[d];
          if (partial_final_block) {
            float sum = 0.0f;
            for (int tok = block_start; tok < block_end; ++tok) {
              const int packed_row = packed_row_offset + tok * packed_row_stride;
              sum += load_nvfp4_packed_row_dim(
                  packed_values, packed_scales, packed_row, d);
            }
            rep_d = sum / static_cast<float>(block_token_count);
          }
          partial += q_row[static_cast<int64_t>(h) * kHeadDim + d] * rep_d;
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
    for (int tile_start = start; tile_start < end; tile_start += kCublasDxTileN) {
      const int tile_count = min(kCublasDxTileN, end - tile_start);

      for (int idx = tid; idx < kHeadDim * kCublasDxTileN; idx += blockDim.x) {
        const int d = idx / kCublasDxTileN;
        const int n = idx - d * kCublasDxTileN;
        float value = 0.0f;
        if (n < tile_count) {
          const int tok = tile_start + n;
          const int packed_row = packed_row_offset + tok * packed_row_stride;
          value = load_nvfp4_packed_row_dim(
              packed_values, packed_scales, packed_row, d);
        }
        b_shared(d, n) = make_mma_value<MmaScalar>(value);
      }
      for (int idx = tid; idx < kCublasDxIndexerHeads * kCublasDxTileN; idx += blockDim.x) {
        const int h = idx / kCublasDxTileN;
        const int n = idx - h * kCublasDxTileN;
        c_shared(h, n) = 0.0f;
      }
      __syncthreads();

      GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
      __syncthreads();

      for (int n = tid; n < kCublasDxTileN; n += blockDim.x) {
        float score = -INFINITY;
        int32_t token_idx = -1;
        if (n < tile_count) {
          score = 0.0f;
          for (int h = 0; h < kCublasDxIndexerHeads; ++h) {
            const float dot = c_shared(h, n);
            if (dot > 0.0f) {
              score += dot * w_row[h];
            }
          }
          token_idx = tile_start + n;
        }
        const int candidate_pos = slot * block_size + (tile_start - start) + n;
        if (candidate_pos < candidate_capacity) {
          candidate_scores[candidate_pos] = score;
          candidate_indices[candidate_pos] = token_idx;
          candidate_ordinals[candidate_pos] = n < tile_count ? candidate_pos : INT_MAX;
        }
      }
      __syncthreads();
    }
  }

  // Deterministic row-local top-k over the scored candidate pool. This replaces
  // the old thread-0 insert path, whose runtime scaled badly at topk=1024
  // because every replacement scanned the full top-k buffer. The comparator
  // preserves the selected set's tie policy: higher score first, then earlier
  // candidate ordinal for exact ties.
  for (int k = 2; k <= candidate_capacity; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < candidate_capacity; i += blockDim.x) {
        const int other = i ^ j;
        if (other > i && other < candidate_capacity) {
          const float score_i = candidate_scores[i];
          const float score_o = candidate_scores[other];
          const int ord_i = candidate_ordinals[i];
          const int ord_o = candidate_ordinals[other];
          const bool left_should_be_better = (i & k) == 0;
          const bool i_better = candidate_better(score_i, ord_i, score_o, ord_o);
          const bool should_swap =
              (left_should_be_better && !i_better)
              || (!left_should_be_better && i_better);
          if (should_swap) {
            candidate_scores[i] = score_o;
            candidate_scores[other] = score_i;
            const int32_t idx_i = candidate_indices[i];
            candidate_indices[i] = candidate_indices[other];
            candidate_indices[other] = idx_i;
            candidate_ordinals[i] = ord_o;
            candidate_ordinals[other] = ord_i;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int i = tid; i < topk_tokens; i += blockDim.x) {
    topk_indices[static_cast<int64_t>(row) * topk_tokens + i] = candidate_indices[i];
    selected_scores[static_cast<int64_t>(row) * topk_tokens + i] = candidate_scores[i];
  }
}

void launch_hisa_selector_nvfp4_cublasdx_fwd(
    const float* q, const uint8_t* packed_values,
    const int32_t* packed_scales, const float* block_reps,
    const float* weights, const int32_t* prefix_lens,
    const int32_t* block_topk_counts, int32_t* topk_indices,
    float* selected_scores, int Q, int H, int D, int L, int packed_row_offset,
    int packed_row_stride, int MB, int block_size, int effective_block_topk,
    int topk_tokens, int force_first, int force_last,
    int force_last_minus_one, cudaStream_t stream) {
  if (H != kCublasDxIndexerHeads || D != kHeadDim) {
    C10_THROW_ERROR(ValueError, "cuBLASDx HISA selector requires H=64 and D=128");
  }
  using GEMM = HisaSelectorGemm64x16;
  const dim3 block = GEMM::block_dim;
  const int candidate_capacity =
      next_power_of_two_int(max(topk_tokens, max(1, effective_block_topk * block_size)));
  size_t smem_bytes =
      sizeof(float) * static_cast<size_t>(MB)
      + sizeof(int) * static_cast<size_t>(effective_block_topk)
      + sizeof(float) * static_cast<size_t>(candidate_capacity)
      + sizeof(int32_t) * static_cast<size_t>(candidate_capacity)
      + sizeof(int32_t) * static_cast<size_t>(candidate_capacity)
      + 16
      + cublasdx::get_shared_storage_size<GEMM>();
  if (smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hisa_selector_nvfp4_cublasdx_fwd_kernel<GEMM, float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }
  hisa_selector_nvfp4_cublasdx_fwd_kernel<GEMM, float><<<Q, block, smem_bytes, stream>>>(
      q, packed_values, packed_scales, block_reps, weights, prefix_lens,
      block_topk_counts, topk_indices, selected_scores, Q, L,
      packed_row_offset, packed_row_stride, MB, block_size,
      effective_block_topk, topk_tokens, candidate_capacity, force_first, force_last,
      force_last_minus_one);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class GEMM>
__global__ void hisa_selector_nvfp4_cublasdx_stage1_kernel(
    const float* __restrict__ q,                  // [Q, 64, 128]
    const uint8_t* __restrict__ packed_values,    // [N, 64]
    const int32_t* __restrict__ packed_scales,    // [N]
    const float* __restrict__ block_reps,         // [MB, 128]
    const float* __restrict__ weights,            // [Q, 64]
    const int32_t* __restrict__ prefix_lens,      // [Q]
    const int32_t* __restrict__ block_topk_counts,// [Q]
    int32_t* __restrict__ selected_blocks_out,    // [Q, effective_block_topk]
    int Q, int L, int packed_row_offset, int packed_row_stride, int MB,
    int block_size, int effective_block_topk,
    int force_first, int force_last, int force_last_minus_one) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= Q) {
    return;
  }

  extern __shared__ __align__(16) unsigned char smem_raw[];
  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* block_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * MB;
  int* selected_blocks = reinterpret_cast<int*>(cursor);
  cursor += sizeof(int) * effective_block_topk;
  cursor = align_dynamic_smem(cursor, 16);
  auto gemm_smem = reinterpret_cast<void*>(cursor);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  __shared__ float reduce_scratch[9];
  __shared__ float score_accum;

  for (int i = tid; i < MB; i += blockDim.x) {
    block_scores[i] = -INFINITY;
  }
  for (int i = tid; i < effective_block_topk; i += blockDim.x) {
    selected_blocks[i] = -1;
  }
  __syncthreads();

  const int prefix_len = max(0, min(static_cast<int>(prefix_lens[row]), L));
  const int row_blocks = min(MB, ceil_div_int(prefix_len, block_size));
  if (prefix_len <= 0 || row_blocks <= 0) {
    for (int i = tid; i < effective_block_topk; i += blockDim.x) {
      selected_blocks_out[static_cast<int64_t>(row) * effective_block_topk + i] = -1;
    }
    return;
  }

  const float* q_row = q + static_cast<int64_t>(row) * kCublasDxIndexerHeads * kHeadDim;
  const float* w_row = weights + static_cast<int64_t>(row) * kCublasDxIndexerHeads;

  for (int idx = tid; idx < kCublasDxIndexerHeads * kHeadDim; idx += blockDim.x) {
    const int h = idx / kHeadDim;
    const int d = idx - h * kHeadDim;
    a_shared(h, d) = q_row[static_cast<int64_t>(h) * kHeadDim + d];
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < row_blocks; tile_start += kCublasDxTileN) {
    const int tile_count = min(kCublasDxTileN, row_blocks - tile_start);

    for (int idx = tid; idx < kHeadDim * kCublasDxTileN; idx += blockDim.x) {
      const int d = idx / kCublasDxTileN;
      const int n = idx - d * kCublasDxTileN;
      float value = 0.0f;
      if (n < tile_count) {
        value = block_reps[static_cast<int64_t>(tile_start + n) * kHeadDim + d];
      }
      b_shared(d, n) = value;
    }
    for (int idx = tid; idx < kCublasDxIndexerHeads * kCublasDxTileN; idx += blockDim.x) {
      const int h = idx / kCublasDxTileN;
      const int n = idx - h * kCublasDxTileN;
      c_shared(h, n) = 0.0f;
    }
    __syncthreads();

    GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();

    for (int n = tid; n < kCublasDxTileN; n += blockDim.x) {
      if (n < tile_count) {
        float score = 0.0f;
        for (int h = 0; h < kCublasDxIndexerHeads; ++h) {
          const float dot = c_shared(h, n);
          if (dot > 0.0f) {
            score += dot * w_row[h];
          }
        }
        block_scores[tile_start + n] = score;
      }
    }
    __syncthreads();
  }

  const int final_block = row_blocks - 1;
  const int block_start = final_block * block_size;
  const int block_end = min(block_start + block_size, prefix_len);
  const int block_token_count = max(1, block_end - block_start);
  if (block_token_count < block_size) {
    if (tid == 0) {
      score_accum = 0.0f;
    }
    __syncthreads();
    for (int h = 0; h < kCublasDxIndexerHeads; ++h) {
      float partial = 0.0f;
      for (int d = tid; d < kHeadDim; d += blockDim.x) {
        float sum = 0.0f;
        for (int tok = block_start; tok < block_end; ++tok) {
          const int packed_row = packed_row_offset + tok * packed_row_stride;
          sum += load_nvfp4_packed_row_dim(packed_values, packed_scales, packed_row, d);
        }
        partial += q_row[static_cast<int64_t>(h) * kHeadDim + d]
            * (sum / static_cast<float>(block_token_count));
      }
      const float dot = block_sum_128(partial, reduce_scratch);
      if (tid == 0 && dot > 0.0f) {
        score_accum += dot * w_row[h];
      }
      __syncthreads();
    }
    if (tid == 0) {
      block_scores[final_block] = score_accum;
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

  for (int i = tid; i < effective_block_topk; i += blockDim.x) {
    selected_blocks_out[static_cast<int64_t>(row) * effective_block_topk + i] =
        selected_blocks[i];
  }
}

template <class GEMM>
__global__ void hisa_selector_nvfp4_cublasdx_score_tiles_kernel(
    const float* __restrict__ q,                  // [Q, 64, 128]
    const uint8_t* __restrict__ packed_values,    // [N, 64]
    const int32_t* __restrict__ packed_scales,    // [N]
    const float* __restrict__ weights,            // [Q, 64]
    const int32_t* __restrict__ prefix_lens,      // [Q]
    const int32_t* __restrict__ selected_blocks,  // [Q, effective_block_topk]
    float* __restrict__ candidate_scores,         // [Q, candidate_capacity]
    int32_t* __restrict__ candidate_indices,      // [Q, candidate_capacity]
    int Q, int L, int packed_row_offset, int packed_row_stride,
    int block_size, int effective_block_topk, int candidate_capacity) {
  const int row = blockIdx.x;
  const int tile_id = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= Q) {
    return;
  }

  const int tile_start = tile_id * kCublasDxTileN;
  if (tile_start >= candidate_capacity) {
    return;
  }

  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto gemm_smem = reinterpret_cast<void*>(smem_raw);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  const int prefix_len = max(0, min(static_cast<int>(prefix_lens[row]), L));
  const float* q_row = q + static_cast<int64_t>(row) * kCublasDxIndexerHeads * kHeadDim;
  const float* w_row = weights + static_cast<int64_t>(row) * kCublasDxIndexerHeads;

  for (int idx = tid; idx < kCublasDxIndexerHeads * kHeadDim; idx += blockDim.x) {
    const int h = idx / kHeadDim;
    const int d = idx - h * kHeadDim;
    a_shared(h, d) = q_row[static_cast<int64_t>(h) * kHeadDim + d];
  }

  for (int idx = tid; idx < kHeadDim * kCublasDxTileN; idx += blockDim.x) {
    const int d = idx / kCublasDxTileN;
    const int n = idx - d * kCublasDxTileN;
    const int candidate_pos = tile_start + n;
    float value = 0.0f;
    if (candidate_pos < candidate_capacity) {
      const int slot = candidate_pos / block_size;
      const int offset = candidate_pos - slot * block_size;
      int token = -1;
      if (slot < effective_block_topk) {
        const int block_id =
            selected_blocks[static_cast<int64_t>(row) * effective_block_topk + slot];
        token = block_id >= 0 ? block_id * block_size + offset : -1;
      }
      if (token >= 0 && token < prefix_len && token < L) {
        const int packed_row = packed_row_offset + token * packed_row_stride;
        value = load_nvfp4_packed_row_dim(packed_values, packed_scales, packed_row, d);
      }
    }
    b_shared(d, n) = value;
  }
  for (int idx = tid; idx < kCublasDxIndexerHeads * kCublasDxTileN; idx += blockDim.x) {
    const int h = idx / kCublasDxTileN;
    const int n = idx - h * kCublasDxTileN;
    c_shared(h, n) = 0.0f;
  }
  __syncthreads();

  GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
  __syncthreads();

  for (int n = tid; n < kCublasDxTileN; n += blockDim.x) {
    const int candidate_pos = tile_start + n;
    if (candidate_pos >= candidate_capacity) {
      continue;
    }
    const int slot = candidate_pos / block_size;
    const int offset = candidate_pos - slot * block_size;
    int32_t token_idx = -1;
    float score = -INFINITY;
    if (slot < effective_block_topk) {
      const int block_id =
          selected_blocks[static_cast<int64_t>(row) * effective_block_topk + slot];
      const int token = block_id >= 0 ? block_id * block_size + offset : -1;
      if (token >= 0 && token < prefix_len && token < L) {
        float accum = 0.0f;
        for (int h = 0; h < kCublasDxIndexerHeads; ++h) {
          const float dot = c_shared(h, n);
          if (dot > 0.0f) {
            accum += dot * w_row[h];
          }
        }
        score = accum;
        token_idx = token;
      }
    }
    candidate_scores[static_cast<int64_t>(row) * candidate_capacity + candidate_pos] = score;
    candidate_indices[static_cast<int64_t>(row) * candidate_capacity + candidate_pos] =
        token_idx;
  }
}

__global__ void hisa_selector_tiled_topk_kernel(
    float* __restrict__ candidate_scores,          // [Q, candidate_capacity]
    int32_t* __restrict__ candidate_indices,       // [Q, candidate_capacity]
    int32_t* __restrict__ topk_indices,            // [Q, K]
    float* __restrict__ selected_scores,           // [Q, K]
    int Q, int candidate_capacity, int topk_tokens) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= Q) {
    return;
  }

  float* scores = candidate_scores + static_cast<int64_t>(row) * candidate_capacity;
  int32_t* indices = candidate_indices + static_cast<int64_t>(row) * candidate_capacity;

  for (int k = 2; k <= candidate_capacity; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < candidate_capacity; i += blockDim.x) {
        const int other = i ^ j;
        if (other > i && other < candidate_capacity) {
          const float score_i = scores[i];
          const float score_o = scores[other];
          const bool left_should_be_better = (i & k) == 0;
          const bool i_better = candidate_better(score_i, i, score_o, other);
          const bool should_swap =
              (left_should_be_better && !i_better)
              || (!left_should_be_better && i_better);
          if (should_swap) {
            scores[i] = score_o;
            scores[other] = score_i;
            const int32_t idx_i = indices[i];
            indices[i] = indices[other];
            indices[other] = idx_i;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int i = tid; i < topk_tokens; i += blockDim.x) {
    topk_indices[static_cast<int64_t>(row) * topk_tokens + i] = indices[i];
    selected_scores[static_cast<int64_t>(row) * topk_tokens + i] = scores[i];
  }
}

__global__ void hisa_selector_tiled_topk_cub4096_kernel(
    float* __restrict__ candidate_scores,          // [Q, candidate_capacity]
    int32_t* __restrict__ candidate_indices,       // [Q, candidate_capacity]
    int32_t* __restrict__ topk_indices,            // [Q, K]
    float* __restrict__ selected_scores,           // [Q, K]
    int Q, int candidate_capacity, int topk_tokens) {
  constexpr int kThreads = 128;
  constexpr int kItemsPerThread = 32;
  using Sort = cub::BlockRadixSort<uint64_t, kThreads, kItemsPerThread, int32_t>;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= Q) {
    return;
  }

  __shared__ typename Sort::TempStorage sort_storage;
  uint64_t keys[kItemsPerThread];
  int32_t values[kItemsPerThread];

  float* scores = candidate_scores + static_cast<int64_t>(row) * candidate_capacity;
  int32_t* indices = candidate_indices + static_cast<int64_t>(row) * candidate_capacity;

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int candidate_pos = tid * kItemsPerThread + item;
    float score = -INFINITY;
    int32_t token = -1;
    if (candidate_pos < candidate_capacity) {
      score = scores[candidate_pos];
      token = indices[candidate_pos];
    }
    const bool valid = token >= 0 && score > -3.0e38f;
    const uint32_t score_key = valid ? __float_as_uint(score) : 0u;
    const uint32_t ordinal_key = 0xffffffffu - static_cast<uint32_t>(candidate_pos);
    keys[item] = (static_cast<uint64_t>(score_key) << 32) | ordinal_key;
    values[item] = valid ? token : -1;
  }

  Sort(sort_storage).SortDescending(keys, values);
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int rank = tid * kItemsPerThread + item;
    if (rank < topk_tokens) {
      const uint32_t score_key = static_cast<uint32_t>(keys[item] >> 32);
      topk_indices[static_cast<int64_t>(row) * topk_tokens + rank] = values[item];
      selected_scores[static_cast<int64_t>(row) * topk_tokens + rank] =
          values[item] >= 0 ? __uint_as_float(score_key) : -INFINITY;
    }
  }
}

void launch_hisa_selector_nvfp4_cublasdx_tiled_fwd(
    const float* q, const uint8_t* packed_values,
    const int32_t* packed_scales, const float* block_reps,
    const float* weights, const int32_t* prefix_lens,
    const int32_t* block_topk_counts, int32_t* selected_blocks,
    float* candidate_scores, int32_t* candidate_indices,
    int32_t* topk_indices, float* selected_scores,
    int Q, int H, int D, int L, int packed_row_offset,
    int packed_row_stride, int MB, int block_size, int effective_block_topk,
    int topk_tokens, int candidate_capacity, int force_first, int force_last,
    int force_last_minus_one, cudaStream_t stream) {
  if (H != kCublasDxIndexerHeads || D != kHeadDim) {
    C10_THROW_ERROR(ValueError, "tiled cuBLASDx HISA selector requires H=64 and D=128");
  }
  using GEMM = HisaSelectorGemm64x16;
  const dim3 block = GEMM::block_dim;
  const size_t gemm_smem = cublasdx::get_shared_storage_size<GEMM>();
  size_t stage1_smem =
      sizeof(float) * static_cast<size_t>(MB)
      + sizeof(int) * static_cast<size_t>(effective_block_topk)
      + 16
      + gemm_smem;
  size_t score_smem = gemm_smem;
  if (stage1_smem > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hisa_selector_nvfp4_cublasdx_stage1_kernel<GEMM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(stage1_smem)));
  }
  if (score_smem > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hisa_selector_nvfp4_cublasdx_score_tiles_kernel<GEMM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(score_smem)));
  }

  hisa_selector_nvfp4_cublasdx_stage1_kernel<GEMM><<<Q, block, stage1_smem, stream>>>(
      q, packed_values, packed_scales, block_reps, weights, prefix_lens,
      block_topk_counts, selected_blocks, Q, L, packed_row_offset,
      packed_row_stride, MB, block_size, effective_block_topk, force_first,
      force_last, force_last_minus_one);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int tile_count = (candidate_capacity + kCublasDxTileN - 1) / kCublasDxTileN;
  dim3 score_grid(Q, tile_count);
  hisa_selector_nvfp4_cublasdx_score_tiles_kernel<GEMM>
      <<<score_grid, block, score_smem, stream>>>(
          q, packed_values, packed_scales, weights, prefix_lens, selected_blocks,
          candidate_scores, candidate_indices, Q, L, packed_row_offset,
          packed_row_stride, block_size, effective_block_topk, candidate_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (candidate_capacity <= 4096) {
    hisa_selector_tiled_topk_cub4096_kernel<<<Q, 128, 0, stream>>>(
        candidate_scores, candidate_indices, topk_indices, selected_scores, Q,
        candidate_capacity, topk_tokens);
  } else {
    hisa_selector_tiled_topk_kernel<<<Q, 128, 0, stream>>>(
        candidate_scores, candidate_indices, topk_indices, selected_scores, Q,
        candidate_capacity, topk_tokens);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_hisa_selector_nvfp4_cublasdx_fp8_fwd(
    const float* q, const uint8_t* packed_values,
    const int32_t* packed_scales, const float* block_reps,
    const float* weights, const int32_t* prefix_lens,
    const int32_t* block_topk_counts, int32_t* topk_indices,
    float* selected_scores, int Q, int H, int D, int L, int packed_row_offset,
    int packed_row_stride, int MB, int block_size, int effective_block_topk,
    int topk_tokens, int force_first, int force_last,
    int force_last_minus_one, cudaStream_t stream) {
  if (H != kCublasDxIndexerHeads || D != kHeadDim) {
    C10_THROW_ERROR(ValueError, "FP8 cuBLASDx HISA selector requires H=64 and D=128");
  }
  using GEMM = HisaSelectorGemm64x16Fp8;
  const dim3 block = GEMM::block_dim;
  const int candidate_capacity =
      next_power_of_two_int(max(topk_tokens, max(1, effective_block_topk * block_size)));
  size_t smem_bytes =
      sizeof(float) * static_cast<size_t>(MB)
      + sizeof(int) * static_cast<size_t>(effective_block_topk)
      + sizeof(float) * static_cast<size_t>(candidate_capacity)
      + sizeof(int32_t) * static_cast<size_t>(candidate_capacity)
      + sizeof(int32_t) * static_cast<size_t>(candidate_capacity)
      + 16
      + cublasdx::get_shared_storage_size<GEMM>();
  if (smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hisa_selector_nvfp4_cublasdx_fwd_kernel<GEMM, __nv_fp8_e4m3>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }
  hisa_selector_nvfp4_cublasdx_fwd_kernel<GEMM, __nv_fp8_e4m3>
      <<<Q, block, smem_bytes, stream>>>(
          q, packed_values, packed_scales, block_reps, weights, prefix_lens,
          block_topk_counts, topk_indices, selected_scores, Q, L,
          packed_row_offset, packed_row_stride, MB, block_size,
          effective_block_topk, topk_tokens, candidate_capacity, force_first, force_last,
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
    float local_max = -INFINITY;
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
      local_max = fmaxf(local_max, score);
    }
    __syncthreads();

    const float max_score = block_max_128(local_max, reduce_scratch);
    float local_denom = 0.0f;
    for (int i = tid; i < topk_tokens; i += blockDim.x) {
      const float s = teacher_scores[i];
      if (s > -INFINITY) {
        local_denom += expf(s - max_score);
      }
    }
    const float denom = block_sum_128(local_denom, reduce_scratch);
    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
    for (int i = tid; i < topk_tokens; i += blockDim.x) {
      const float s = teacher_scores[i];
      if (s > -INFINITY) {
        teacher_mass[i] += expf(s - max_score) * inv_denom;
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
  if (smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hisa_selector_teacher_fwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }
  hisa_selector_teacher_fwd_kernel<<<Q, threads, smem_bytes, stream>>>(
      q, k, block_reps, weights, attn_query, attn_key, prefix_lens,
      block_topk_counts, topk_indices, selected_scores, teacher_probs,
      Q, H, D, L, MB, AH, AD, block_size, effective_block_topk, topk_tokens,
      softmax_scale, force_first, force_last, force_last_minus_one);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace hisa_indexer
}  // namespace megatron
