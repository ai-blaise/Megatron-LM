// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Batched HISA selector megakernel.
//
// This translation unit is intentionally separate from hisa_selector_fwd.cu.
// It provides a new batched path for the training selector contract:
//   q             [Q, B, 64, 128]
//   k             [L, B, 128]
//   block_reps    [B, MB, 128]
//   weights       [Q, B, 64]
//   prefix_lens   [B, Q]
//
// One CTA owns one (batch, query-row), scores HISA blocks, refines candidate
// tokens, and emits top-k token ids plus selected scores. The GEMM work uses
// cuBLASDx block-level MMA so this path removes the Python/ATen BMM/topk/gather
// orchestration from the selected-score forward.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

constexpr int kIndexerHeads = 64;
constexpr int kHeadDim = 128;
constexpr int kTileN = 128;
constexpr int kDTypeFloat32 = 0;
constexpr int kDTypeBFloat16 = 1;
constexpr int kDTypeFloat16 = 2;

using HisaMegaGemmF32 = decltype(
    cublasdx::Size<kIndexerHeads, kTileN, kHeadDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using HisaMegaGemmBF16 = decltype(
    cublasdx::Size<kIndexerHeads, kTileN, kHeadDim>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using HisaMegaGemmF16 = decltype(
    cublasdx::Size<kIndexerHeads, kTileN, kHeadDim>()
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

__device__ __forceinline__ int ceil_div_int(int x, int y) {
  return (x + y - 1) / y;
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

__device__ __forceinline__ char* align_dynamic_smem(char* ptr, uintptr_t alignment) {
  const uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
  return reinterpret_cast<char*>((raw + alignment - 1) & ~(alignment - 1));
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

__device__ __forceinline__ bool is_power_of_two_int_device(int x) {
  return x > 0 && (x & (x - 1)) == 0;
}

__device__ __forceinline__ uint64_t candidate_key_from_score_ordinal(
    float score,
    int32_t ordinal) {
  if (ordinal == INT_MAX || !(score > -3.0e38f)) {
    return 0ull;
  }
  const uint32_t score_key = __float_as_uint(score);
  const uint32_t ordinal_key = 0xffffffffu - static_cast<uint32_t>(ordinal);
  return (static_cast<uint64_t>(score_key) << 32) | ordinal_key;
}

__device__ __forceinline__ float candidate_key_score(uint64_t key) {
  return key == 0ull ? -INFINITY : __uint_as_float(static_cast<uint32_t>(key >> 32));
}

__device__ __forceinline__ int32_t candidate_key_ordinal(uint64_t key) {
  if (key == 0ull) {
    return INT_MAX;
  }
  const uint32_t ordinal_key = static_cast<uint32_t>(key);
  return static_cast<int32_t>(0xffffffffu - ordinal_key);
}

__device__ __forceinline__ int32_t candidate_token_from_ordinal(
    int32_t ordinal,
    const int32_t* __restrict__ selected_blocks,
    int row_linear,
    int effective_block_topk,
    int block_size) {
  if (ordinal == INT_MAX || ordinal < 0) {
    return -1;
  }
  const int slot = ordinal / block_size;
  const int offset = ordinal - slot * block_size;
  if (slot < 0 || slot >= effective_block_topk) {
    return -1;
  }
  const int block_id = selected_blocks[row_linear * effective_block_topk + slot];
  return block_id >= 0 ? block_id * block_size + offset : -1;
}

__device__ __forceinline__ void store_candidate_token(
    void* candidate_tokens,
    int pos,
    int32_t token,
    int use_u16_tokens) {
  if (use_u16_tokens) {
    reinterpret_cast<uint16_t*>(candidate_tokens)[pos] =
        token >= 0 ? static_cast<uint16_t>(token) : UINT16_MAX;
  } else {
    reinterpret_cast<int32_t*>(candidate_tokens)[pos] = token;
  }
}

__device__ __forceinline__ int32_t load_candidate_token(
    const void* candidate_tokens,
    int pos,
    int use_u16_tokens) {
  if (use_u16_tokens) {
    const uint16_t token = reinterpret_cast<const uint16_t*>(candidate_tokens)[pos];
    return token == UINT16_MAX ? -1 : static_cast<int32_t>(token);
  }
  return reinterpret_cast<const int32_t*>(candidate_tokens)[pos];
}

__device__ __forceinline__ int hisa_forced_block_budget(
    int row_blocks, int force_first, int force_last, int force_last_minus_one) {
  int forced = 0;
  if (force_first && row_blocks >= 1) {
    forced += 1;
  }
  if (force_last && row_blocks >= 1) {
    forced += 1;
    if (force_first && row_blocks == 1) {
      forced -= 1;
    }
  }
  if (force_last_minus_one && row_blocks >= 2) {
    forced += 1;
    if (force_first && row_blocks == 2) {
      forced -= 1;
    }
  }
  return max(0, forced);
}

__device__ __forceinline__ int hisa_row_block_keep(
    int row_blocks,
    int block_topk,
    float compression_ratio,
    int effective_block_topk,
    int force_first,
    int force_last,
    int force_last_minus_one) {
  int keep = 0;
  if (compression_ratio > 0.0f) {
    keep = static_cast<int>(ceilf(static_cast<float>(row_blocks) / compression_ratio));
  } else {
    keep = block_topk;
  }
  keep = max(keep, hisa_forced_block_budget(
      row_blocks, force_first, force_last, force_last_minus_one));
  keep = max(0, min(keep, effective_block_topk));
  return min(keep, row_blocks);
}

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

template <typename scalar_t>
__global__ void hisa_block_reps_batched_kernel(
    const scalar_t* __restrict__ k,       // [L, B, D]
    scalar_t* __restrict__ block_reps,    // [B, MB, D]
    int L, int B, int D, int block_size, int MB) {
  const int block_id = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  if (block_id >= MB || batch >= B) {
    return;
  }
  const int start = block_id * block_size;
  const int end = min(start + block_size, L);
  const int count = max(1, end - start);
  for (int d = tid; d < D; d += blockDim.x) {
    float sum = 0.0f;
    for (int tok = start; tok < end; ++tok) {
      sum += to_float(k[(static_cast<int64_t>(tok) * B + batch) * D + d]);
    }
    block_reps[(static_cast<int64_t>(batch) * MB + block_id) * D + d] =
        from_float<scalar_t>(sum / static_cast<float>(count));
  }
}

template <class GEMM, typename scalar_t, typename weight_t, typename prefix_t>
__global__ void hisa_selector_megakernel_batched_kernel(
    const scalar_t* __restrict__ q,             // [Q, B, 64, 128]
    const scalar_t* __restrict__ k,             // [L, B, 128]
    const scalar_t* __restrict__ block_reps,    // [B, MB, 128]
    const weight_t* __restrict__ weights,       // [Q, B, 64]
    const prefix_t* __restrict__ prefix_lens,   // [Q] or [B, Q]
    int32_t* __restrict__ topk_indices,         // [B, Q, K]
    float* __restrict__ selected_scores,        // [B, Q, K]
    int Q, int B, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int block_score_capacity,
    int candidate_capacity,
    int use_u16_tokens,
    int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one) {
  const int row_linear = blockIdx.x;
  const int tid = threadIdx.x;
  if (row_linear >= Q * B) {
    return;
  }
  const int batch = row_linear / Q;
  const int row = row_linear - batch * Q;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* block_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * block_score_capacity;
  int* block_indices = reinterpret_cast<int*>(cursor);
  cursor += sizeof(int) * block_score_capacity;
  int* selected_blocks = reinterpret_cast<int*>(cursor);
  cursor += sizeof(int) * effective_block_topk;
  float* candidate_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * candidate_capacity;
  void* candidate_tokens = reinterpret_cast<void*>(cursor);
  cursor += (use_u16_tokens ? sizeof(uint16_t) : sizeof(int32_t)) * candidate_capacity;
  uint16_t* candidate_ordinals = reinterpret_cast<uint16_t*>(cursor);
  cursor += sizeof(uint16_t) * candidate_capacity;
  cursor = align_dynamic_smem(cursor, 16);
  auto gemm_smem = reinterpret_cast<void*>(cursor);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  __shared__ float reduce_scratch[9];
  __shared__ float score_accum;

  for (int i = tid; i < block_score_capacity; i += blockDim.x) {
    block_scores[i] = -INFINITY;
    block_indices[i] = INT_MAX;
  }
  for (int i = tid; i < effective_block_topk; i += blockDim.x) {
    selected_blocks[i] = -1;
  }
  for (int i = tid; i < candidate_capacity; i += blockDim.x) {
    candidate_scores[i] = -INFINITY;
    store_candidate_token(candidate_tokens, i, -1, use_u16_tokens);
    candidate_ordinals[i] = UINT16_MAX;
  }
  __syncthreads();

  const int prefix_index = prefix_lens_shared ? row : (batch * Q + row);
  const int prefix_len = max(0, min(static_cast<int>(prefix_lens[prefix_index]), L));
  const int row_blocks = min(MB, ceil_div_int(prefix_len, block_size));
  if (prefix_len <= 0 || row_blocks <= 0) {
    for (int i = tid; i < topk_tokens; i += blockDim.x) {
      topk_indices[(static_cast<int64_t>(batch) * Q + row) * topk_tokens + i] = -1;
      selected_scores[(static_cast<int64_t>(batch) * Q + row) * topk_tokens + i] =
          -INFINITY;
    }
    return;
  }

  const scalar_t* q_row =
      q + (static_cast<int64_t>(row) * B + batch) * kIndexerHeads * kHeadDim;
  const weight_t* w_row =
      weights + (static_cast<int64_t>(row) * B + batch) * kIndexerHeads;
  const scalar_t* block_rep_batch =
      block_reps + static_cast<int64_t>(batch) * MB * kHeadDim;

  for (int idx = tid; idx < kIndexerHeads * kHeadDim; idx += blockDim.x) {
    const int h = idx / kHeadDim;
    const int d = idx - h * kHeadDim;
    a_shared(h, d) = q_row[static_cast<int64_t>(h) * kHeadDim + d];
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < row_blocks; tile_start += kTileN) {
    const int tile_count = min(kTileN, row_blocks - tile_start);
    for (int idx = tid; idx < kHeadDim * kTileN; idx += blockDim.x) {
      const int d = idx / kTileN;
      const int n = idx - d * kTileN;
      scalar_t value = from_float<scalar_t>(0.0f);
      if (n < tile_count) {
        value = block_rep_batch[static_cast<int64_t>(tile_start + n) * kHeadDim + d];
      }
      b_shared(d, n) = value;
    }
    for (int idx = tid; idx < kIndexerHeads * kTileN; idx += blockDim.x) {
      const int h = idx / kTileN;
      const int n = idx - h * kTileN;
      c_shared(h, n) = 0.0f;
    }
    __syncthreads();

    GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();

    for (int n = tid; n < kTileN; n += blockDim.x) {
      if (n < tile_count) {
        float score = 0.0f;
        for (int h = 0; h < kIndexerHeads; ++h) {
          const float dot = c_shared(h, n);
          if (dot > 0.0f) {
            score += dot * to_float(w_row[h]);
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
    for (int h = 0; h < kIndexerHeads; ++h) {
      float partial = 0.0f;
      for (int d = tid; d < kHeadDim; d += blockDim.x) {
        float sum = 0.0f;
        for (int tok = block_start; tok < block_end; ++tok) {
          sum += to_float(k[(static_cast<int64_t>(tok) * B + batch) * kHeadDim + d]);
        }
        partial +=
            to_float(q_row[static_cast<int64_t>(h) * kHeadDim + d])
            * (sum / static_cast<float>(block_token_count));
      }
      const float dot = block_sum_128(partial, reduce_scratch);
      if (tid == 0 && dot > 0.0f) {
        score_accum += dot * to_float(w_row[h]);
      }
      __syncthreads();
    }
    if (tid == 0) {
      block_scores[final_block] = score_accum;
    }
    __syncthreads();
  }

  for (int i = tid; i < block_score_capacity; i += blockDim.x) {
    block_indices[i] = i < row_blocks ? i : INT_MAX;
    if (i >= row_blocks) {
      block_scores[i] = -INFINITY;
    }
  }
  __syncthreads();

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
  }
  __syncthreads();

  for (int k_size = 2; k_size <= block_score_capacity; k_size <<= 1) {
    for (int j = k_size >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < block_score_capacity; i += blockDim.x) {
        const int other = i ^ j;
        if (other > i && other < block_score_capacity) {
          const float score_i = block_scores[i];
          const float score_o = block_scores[other];
          const int block_i = block_indices[i];
          const int block_o = block_indices[other];
          const bool left_should_be_better = (i & k_size) == 0;
          const bool i_better = candidate_better(score_i, block_i, score_o, block_o);
          const bool should_swap =
              (left_should_be_better && !i_better)
              || (!left_should_be_better && i_better);
          if (should_swap) {
            block_scores[i] = score_o;
            block_scores[other] = score_i;
            block_indices[i] = block_o;
            block_indices[other] = block_i;
          }
        }
      }
      __syncthreads();
    }
  }

  const int keep = hisa_row_block_keep(
      row_blocks,
      block_topk,
      compression_ratio,
      effective_block_topk,
      force_first,
      force_last,
      force_last_minus_one);
  for (int slot = tid; slot < effective_block_topk; slot += blockDim.x) {
    selected_blocks[slot] = slot < keep ? block_indices[slot] : -1;
  }
  __syncthreads();

  for (int slot = 0; slot < keep; ++slot) {
    const int block_id = selected_blocks[slot];
    if (block_id < 0) {
      continue;
    }
    const int start = block_id * block_size;
    const int end = min(start + block_size, prefix_len);
    for (int tile_start = start; tile_start < end; tile_start += kTileN) {
      const int tile_count = min(kTileN, end - tile_start);
      for (int idx = tid; idx < kHeadDim * kTileN; idx += blockDim.x) {
        const int d = idx / kTileN;
        const int n = idx - d * kTileN;
        scalar_t value = from_float<scalar_t>(0.0f);
        if (n < tile_count) {
          const int tok = tile_start + n;
          value = k[(static_cast<int64_t>(tok) * B + batch) * kHeadDim + d];
        }
        b_shared(d, n) = value;
      }
      for (int idx = tid; idx < kIndexerHeads * kTileN; idx += blockDim.x) {
        const int h = idx / kTileN;
        const int n = idx - h * kTileN;
        c_shared(h, n) = 0.0f;
      }
      __syncthreads();

      GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
      __syncthreads();

      for (int n = tid; n < kTileN; n += blockDim.x) {
        float score = -INFINITY;
        int32_t token_idx = -1;
        if (n < tile_count) {
          score = 0.0f;
          for (int h = 0; h < kIndexerHeads; ++h) {
            const float dot = c_shared(h, n);
            if (dot > 0.0f) {
              score += dot * to_float(w_row[h]);
            }
          }
          token_idx = tile_start + n;
        }
        const int candidate_pos = slot * block_size + (tile_start - start) + n;
        if (candidate_pos < candidate_capacity) {
          candidate_scores[candidate_pos] = score;
          store_candidate_token(candidate_tokens, candidate_pos, token_idx, use_u16_tokens);
          candidate_ordinals[candidate_pos] =
              n < tile_count ? static_cast<uint16_t>(candidate_pos) : UINT16_MAX;
        }
      }
      __syncthreads();
    }
  }

  const bool exact_prefix_select =
      is_power_of_two_int_device(topk_tokens) && topk_tokens <= (candidate_capacity >> 1);
  const int final_stop_stride = exact_prefix_select ? topk_tokens : 1;
  for (int k_size = 2; k_size <= candidate_capacity; k_size <<= 1) {
    const int stop_stride = (k_size == candidate_capacity) ? final_stop_stride : 1;
    for (int j = k_size >> 1; j >= stop_stride && j > 0; j >>= 1) {
      for (int i = tid; i < candidate_capacity; i += blockDim.x) {
        const int other = i ^ j;
        if (other > i && other < candidate_capacity) {
          const float score_i = candidate_scores[i];
          const float score_o = candidate_scores[other];
          const int ord_i = static_cast<int>(candidate_ordinals[i]);
          const int ord_o = static_cast<int>(candidate_ordinals[other]);
          const bool left_should_be_better = (i & k_size) == 0;
          const bool i_better = candidate_better(score_i, ord_i, score_o, ord_o);
          const bool should_swap =
              (left_should_be_better && !i_better)
              || (!left_should_be_better && i_better);
          if (should_swap) {
            candidate_scores[i] = score_o;
            candidate_scores[other] = score_i;
            const int32_t idx_i =
                load_candidate_token(candidate_tokens, i, use_u16_tokens);
            store_candidate_token(
                candidate_tokens,
                i,
                load_candidate_token(candidate_tokens, other, use_u16_tokens),
                use_u16_tokens);
            store_candidate_token(candidate_tokens, other, idx_i, use_u16_tokens);
            const uint16_t ord_swap = candidate_ordinals[i];
            candidate_ordinals[i] = candidate_ordinals[other];
            candidate_ordinals[other] = ord_swap;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int i = tid; i < topk_tokens; i += blockDim.x) {
    const int64_t out = (static_cast<int64_t>(batch) * Q + row) * topk_tokens + i;
    topk_indices[out] = load_candidate_token(candidate_tokens, i, use_u16_tokens);
    selected_scores[out] = candidate_scores[i];
  }
}

template <class GEMM, typename scalar_t, typename weight_t, typename prefix_t>
__global__ void hisa_selector_megakernel_select_blocks_batched_kernel(
    const scalar_t* __restrict__ q,             // [Q, B, 64, 128]
    const scalar_t* __restrict__ k,             // [L, B, 128]
    const scalar_t* __restrict__ block_reps,    // [B, MB, 128]
    const weight_t* __restrict__ weights,       // [Q, B, 64]
    const prefix_t* __restrict__ prefix_lens,   // [Q] or [B, Q]
    int32_t* __restrict__ selected_blocks_out,  // [B, Q, effective_block_topk]
    int Q, int B, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int block_score_capacity,
    int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one) {
  const int row_linear = blockIdx.x;
  const int tid = threadIdx.x;
  if (row_linear >= Q * B) {
    return;
  }
  const int batch = row_linear / Q;
  const int row = row_linear - batch * Q;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* block_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * block_score_capacity;
  int* block_indices = reinterpret_cast<int*>(cursor);
  cursor += sizeof(int) * block_score_capacity;
  cursor = align_dynamic_smem(cursor, 16);
  auto gemm_smem = reinterpret_cast<void*>(cursor);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  __shared__ float reduce_scratch[9];
  __shared__ float score_accum;

  for (int i = tid; i < block_score_capacity; i += blockDim.x) {
    block_scores[i] = -INFINITY;
    block_indices[i] = INT_MAX;
  }
  for (int i = tid; i < effective_block_topk; i += blockDim.x) {
    selected_blocks_out[
        (static_cast<int64_t>(batch) * Q + row) * effective_block_topk + i] = -1;
  }
  __syncthreads();

  const int prefix_index = prefix_lens_shared ? row : (batch * Q + row);
  const int prefix_len = max(0, min(static_cast<int>(prefix_lens[prefix_index]), L));
  const int row_blocks = min(MB, ceil_div_int(prefix_len, block_size));
  if (prefix_len <= 0 || row_blocks <= 0) {
    return;
  }

  const scalar_t* q_row =
      q + (static_cast<int64_t>(row) * B + batch) * kIndexerHeads * kHeadDim;
  const weight_t* w_row =
      weights + (static_cast<int64_t>(row) * B + batch) * kIndexerHeads;
  const scalar_t* block_rep_batch =
      block_reps + static_cast<int64_t>(batch) * MB * kHeadDim;

  for (int idx = tid; idx < kIndexerHeads * kHeadDim; idx += blockDim.x) {
    const int h = idx / kHeadDim;
    const int d = idx - h * kHeadDim;
    a_shared(h, d) = q_row[static_cast<int64_t>(h) * kHeadDim + d];
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < row_blocks; tile_start += kTileN) {
    const int tile_count = min(kTileN, row_blocks - tile_start);
    for (int idx = tid; idx < kHeadDim * kTileN; idx += blockDim.x) {
      const int d = idx / kTileN;
      const int n = idx - d * kTileN;
      scalar_t value = from_float<scalar_t>(0.0f);
      if (n < tile_count) {
        value = block_rep_batch[static_cast<int64_t>(tile_start + n) * kHeadDim + d];
      }
      b_shared(d, n) = value;
    }
    for (int idx = tid; idx < kIndexerHeads * kTileN; idx += blockDim.x) {
      const int h = idx / kTileN;
      const int n = idx - h * kTileN;
      c_shared(h, n) = 0.0f;
    }
    __syncthreads();

    GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();

    for (int n = tid; n < kTileN; n += blockDim.x) {
      if (n < tile_count) {
        float score = 0.0f;
        for (int h = 0; h < kIndexerHeads; ++h) {
          const float dot = c_shared(h, n);
          if (dot > 0.0f) {
            score += dot * to_float(w_row[h]);
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
    for (int h = 0; h < kIndexerHeads; ++h) {
      float partial = 0.0f;
      for (int d = tid; d < kHeadDim; d += blockDim.x) {
        float sum = 0.0f;
        for (int tok = block_start; tok < block_end; ++tok) {
          sum += to_float(k[(static_cast<int64_t>(tok) * B + batch) * kHeadDim + d]);
        }
        partial +=
            to_float(q_row[static_cast<int64_t>(h) * kHeadDim + d])
            * (sum / static_cast<float>(block_token_count));
      }
      const float dot = block_sum_128(partial, reduce_scratch);
      if (tid == 0 && dot > 0.0f) {
        score_accum += dot * to_float(w_row[h]);
      }
      __syncthreads();
    }
    if (tid == 0) {
      block_scores[final_block] = score_accum;
    }
    __syncthreads();
  }

  for (int i = tid; i < block_score_capacity; i += blockDim.x) {
    block_indices[i] = i < row_blocks ? i : INT_MAX;
    if (i >= row_blocks) {
      block_scores[i] = -INFINITY;
    }
  }
  __syncthreads();

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
  }
  __syncthreads();

  for (int k_size = 2; k_size <= block_score_capacity; k_size <<= 1) {
    for (int j = k_size >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < block_score_capacity; i += blockDim.x) {
        const int other = i ^ j;
        if (other > i && other < block_score_capacity) {
          const float score_i = block_scores[i];
          const float score_o = block_scores[other];
          const int block_i = block_indices[i];
          const int block_o = block_indices[other];
          const bool left_should_be_better = (i & k_size) == 0;
          const bool i_better = candidate_better(score_i, block_i, score_o, block_o);
          const bool should_swap =
              (left_should_be_better && !i_better)
              || (!left_should_be_better && i_better);
          if (should_swap) {
            block_scores[i] = score_o;
            block_scores[other] = score_i;
            block_indices[i] = block_o;
            block_indices[other] = block_i;
          }
        }
      }
      __syncthreads();
    }
  }

  const int keep = hisa_row_block_keep(
      row_blocks,
      block_topk,
      compression_ratio,
      effective_block_topk,
      force_first,
      force_last,
      force_last_minus_one);
  for (int slot = tid; slot < effective_block_topk; slot += blockDim.x) {
    selected_blocks_out[
        (static_cast<int64_t>(batch) * Q + row) * effective_block_topk + slot] =
            slot < keep ? block_indices[slot] : -1;
  }
}

template <class GEMM, typename scalar_t, typename weight_t, typename prefix_t>
__global__ void hisa_selector_megakernel_score_candidates_batched_kernel(
    const scalar_t* __restrict__ q,              // [Q, B, 64, 128]
    const scalar_t* __restrict__ k,              // [L, B, 128]
    const weight_t* __restrict__ weights,        // [Q, B, 64]
    const prefix_t* __restrict__ prefix_lens,    // [Q] or [B, Q]
    const int32_t* __restrict__ selected_blocks, // [B, Q, effective_block_topk]
    uint64_t* __restrict__ candidate_keys,       // [B, Q, candidate_capacity]
    int Q, int B, int L, int block_size,
    int effective_block_topk, int candidate_capacity,
    int prefix_lens_shared, int candidate_global_offset) {
  const int row_linear = blockIdx.x;
  const int tile_id = blockIdx.y;
  const int tid = threadIdx.x;
  if (row_linear >= Q * B) {
    return;
  }
  const int batch = row_linear / Q;
  const int row = row_linear - batch * Q;
  const int tile_start = tile_id * kTileN;
  if (tile_start >= candidate_capacity) {
    return;
  }

  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto gemm_smem = reinterpret_cast<void*>(smem_raw);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  const int prefix_index = prefix_lens_shared ? row : (batch * Q + row);
  const int prefix_len = max(0, min(static_cast<int>(prefix_lens[prefix_index]), L));
  const scalar_t* q_row =
      q + (static_cast<int64_t>(row) * B + batch) * kIndexerHeads * kHeadDim;
  const weight_t* w_row =
      weights + (static_cast<int64_t>(row) * B + batch) * kIndexerHeads;

  for (int idx = tid; idx < kIndexerHeads * kHeadDim; idx += blockDim.x) {
    const int h = idx / kHeadDim;
    const int d = idx - h * kHeadDim;
    a_shared(h, d) = q_row[static_cast<int64_t>(h) * kHeadDim + d];
  }
  for (int idx = tid; idx < kHeadDim * kTileN; idx += blockDim.x) {
    const int d = idx / kTileN;
    const int n = idx - d * kTileN;
    const int candidate_pos = tile_start + n;
    scalar_t value = from_float<scalar_t>(0.0f);
    if (candidate_pos < candidate_capacity) {
      const int global_candidate_pos = candidate_global_offset + candidate_pos;
      const int slot = global_candidate_pos / block_size;
      const int offset = global_candidate_pos - slot * block_size;
      int token = -1;
      if (slot < effective_block_topk) {
        const int block_id =
            selected_blocks[row_linear * effective_block_topk + slot];
        token = block_id >= 0 ? block_id * block_size + offset : -1;
      }
      if (token >= 0 && token < prefix_len && token < L) {
        value = k[(static_cast<int64_t>(token) * B + batch) * kHeadDim + d];
      }
    }
    b_shared(d, n) = value;
  }
  for (int idx = tid; idx < kIndexerHeads * kTileN; idx += blockDim.x) {
    const int h = idx / kTileN;
    const int n = idx - h * kTileN;
    c_shared(h, n) = 0.0f;
  }
  __syncthreads();

  GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
  __syncthreads();

  for (int n = tid; n < kTileN; n += blockDim.x) {
    const int candidate_pos = tile_start + n;
    if (candidate_pos >= candidate_capacity) {
      continue;
    }
    const int global_candidate_pos = candidate_global_offset + candidate_pos;
    const int slot = global_candidate_pos / block_size;
    const int offset = global_candidate_pos - slot * block_size;
    float score = -INFINITY;
    int32_t ordinal = INT_MAX;
    if (slot < effective_block_topk) {
      const int block_id = selected_blocks[row_linear * effective_block_topk + slot];
      const int token = block_id >= 0 ? block_id * block_size + offset : -1;
      if (token >= 0 && token < prefix_len && token < L) {
        float accum = 0.0f;
        for (int h = 0; h < kIndexerHeads; ++h) {
          const float dot = c_shared(h, n);
          if (dot > 0.0f) {
            accum += dot * to_float(w_row[h]);
          }
        }
        score = accum;
        ordinal = global_candidate_pos;
      }
    }
    const int64_t out = static_cast<int64_t>(row_linear) * candidate_capacity + candidate_pos;
    candidate_keys[out] = candidate_key_from_score_ordinal(score, ordinal);
  }
}

__global__ void hisa_selector_megakernel_candidate_topk_batched_kernel(
    uint64_t* __restrict__ candidate_keys,      // [B, Q, candidate_capacity]
    const int32_t* __restrict__ selected_blocks,
    int32_t* __restrict__ topk_indices,         // [B, Q, K]
    float* __restrict__ selected_scores,        // [B, Q, K]
    int row_count, int candidate_capacity, int topk_tokens,
    int effective_block_topk, int block_size) {
  const int row_linear = blockIdx.x;
  const int tid = threadIdx.x;
  if (row_linear >= row_count) {
    return;
  }

  uint64_t* keys = candidate_keys + static_cast<int64_t>(row_linear) * candidate_capacity;

  const bool exact_prefix_select =
      is_power_of_two_int_device(topk_tokens) && topk_tokens <= (candidate_capacity >> 1);
  const int final_stop_stride = exact_prefix_select ? topk_tokens : 1;
  for (int k_size = 2; k_size <= candidate_capacity; k_size <<= 1) {
    const int stop_stride = (k_size == candidate_capacity) ? final_stop_stride : 1;
    for (int j = k_size >> 1; j >= stop_stride && j > 0; j >>= 1) {
      for (int i = tid; i < candidate_capacity; i += blockDim.x) {
        const int other = i ^ j;
        if (other > i && other < candidate_capacity) {
          const uint64_t key_i = keys[i];
          const uint64_t key_o = keys[other];
          const bool left_should_be_better = (i & k_size) == 0;
          const bool i_better = key_i >= key_o;
          const bool should_swap =
              (left_should_be_better && !i_better)
              || (!left_should_be_better && i_better);
          if (should_swap) {
            keys[i] = key_o;
            keys[other] = key_i;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int i = tid; i < topk_tokens; i += blockDim.x) {
    const int64_t out = static_cast<int64_t>(row_linear) * topk_tokens + i;
    const uint64_t key = keys[i];
    const int32_t ordinal = candidate_key_ordinal(key);
    topk_indices[out] = candidate_token_from_ordinal(
        ordinal, selected_blocks, row_linear, effective_block_topk, block_size);
    selected_scores[out] = candidate_key_score(key);
  }
}

__global__ void hisa_selector_megakernel_candidate_sort4096_batched_kernel(
    uint64_t* __restrict__ candidate_keys,          // [B, Q, candidate_capacity]
    int row_count, int candidate_capacity) {
  constexpr int kThreads = 128;
  constexpr int kItemsPerThread = 32;
  using Sort = cub::BlockRadixSort<uint64_t, kThreads, kItemsPerThread>;
  constexpr int kSortWidth = kThreads * kItemsPerThread;

  const int row_linear = blockIdx.x;
  const int half = blockIdx.y;
  const int tid = threadIdx.x;
  if (row_linear >= row_count) {
    return;
  }

  __shared__ typename Sort::TempStorage sort_storage;
  uint64_t keys[kItemsPerThread];

  const int half_offset = half * kSortWidth;
  uint64_t* row_keys =
      candidate_keys + static_cast<int64_t>(row_linear) * candidate_capacity;

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int local_pos = tid * kItemsPerThread + item;
    const int candidate_pos = half_offset + local_pos;
    keys[item] = candidate_pos < candidate_capacity ? row_keys[candidate_pos] : 0ull;
  }

  Sort(sort_storage).SortDescending(keys);
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int local_rank = tid * kItemsPerThread + item;
    const int candidate_pos = half_offset + local_rank;
    if (candidate_pos < candidate_capacity) {
      row_keys[candidate_pos] = keys[item];
    }
  }
}

__global__ void hisa_selector_megakernel_candidate_merge_topk_batched_kernel(
    const uint64_t* __restrict__ candidate_keys,    // sorted 4096 chunks
    const int32_t* __restrict__ selected_blocks,
    int32_t* __restrict__ topk_indices,
    float* __restrict__ selected_scores,
    int32_t* __restrict__ topk_ordinals,
    int row_count, int candidate_capacity, int topk_tokens,
    int effective_block_topk, int block_size) {
  const int row_linear = blockIdx.x;
  const int tid = threadIdx.x;
  if (row_linear >= row_count) {
    return;
  }
  const int half_count = (candidate_capacity + 4095) / 4096;
  const int merge_capacity = topk_tokens * half_count;

  extern __shared__ unsigned char smem_raw[];
  uint64_t* keys = reinterpret_cast<uint64_t*>(smem_raw);

  const uint64_t* row_keys =
      candidate_keys + static_cast<int64_t>(row_linear) * candidate_capacity;

  for (int i = tid; i < merge_capacity; i += blockDim.x) {
    const int half = i / topk_tokens;
    const int rank = i - half * topk_tokens;
    const int candidate_pos = half * 4096 + rank;
    if (half < half_count && candidate_pos < candidate_capacity) {
      keys[i] = row_keys[candidate_pos];
    } else {
      keys[i] = 0ull;
    }
  }
  __syncthreads();

  for (int k_size = 2; k_size <= merge_capacity; k_size <<= 1) {
    const int stop_stride = (k_size == merge_capacity) ? topk_tokens : 1;
    for (int j = k_size >> 1; j >= stop_stride && j > 0; j >>= 1) {
      for (int i = tid; i < merge_capacity; i += blockDim.x) {
        const int other = i ^ j;
        if (other > i && other < merge_capacity) {
          const uint64_t key_i = keys[i];
          const uint64_t key_o = keys[other];
          const bool left_should_be_better = (i & k_size) == 0;
          const bool i_better = key_i >= key_o;
          const bool should_swap =
              (left_should_be_better && !i_better)
              || (!left_should_be_better && i_better);
          if (should_swap) {
            keys[i] = key_o;
            keys[other] = key_i;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int i = tid; i < topk_tokens; i += blockDim.x) {
    const int64_t out = static_cast<int64_t>(row_linear) * topk_tokens + i;
    const uint64_t key = keys[i];
    const int32_t ordinal = candidate_key_ordinal(key);
    topk_indices[out] = candidate_token_from_ordinal(
        ordinal, selected_blocks, row_linear, effective_block_topk, block_size);
    selected_scores[out] = candidate_key_score(key);
    if (topk_ordinals != nullptr) {
      topk_ordinals[out] = ordinal;
    }
  }
}

__global__ void hisa_selector_megakernel_merge_running_topk_batched_kernel(
    const uint64_t* __restrict__ candidate_keys,    // sorted 4096 chunks
    const int32_t* __restrict__ selected_blocks,
    int32_t* __restrict__ topk_indices,
    float* __restrict__ selected_scores,
    int32_t* __restrict__ topk_ordinals,
    int row_count, int candidate_capacity, int topk_tokens, int merge_capacity,
    int effective_block_topk, int block_size) {
  const int row_linear = blockIdx.x;
  const int tid = threadIdx.x;
  if (row_linear >= row_count) {
    return;
  }
  const int candidate_chunk_count = (candidate_capacity + 4095) / 4096;
  const int merge_items = topk_tokens * (candidate_chunk_count + 1);

  extern __shared__ unsigned char smem_raw[];
  uint64_t* keys = reinterpret_cast<uint64_t*>(smem_raw);

  const uint64_t* row_keys =
      candidate_keys + static_cast<int64_t>(row_linear) * candidate_capacity;
  int32_t* row_topk =
      topk_indices + static_cast<int64_t>(row_linear) * topk_tokens;
  float* row_selected =
      selected_scores + static_cast<int64_t>(row_linear) * topk_tokens;
  int32_t* row_topk_ord =
      topk_ordinals + static_cast<int64_t>(row_linear) * topk_tokens;

  for (int i = tid; i < merge_capacity; i += blockDim.x) {
    uint64_t key = 0ull;
    if (i < topk_tokens) {
      key = candidate_key_from_score_ordinal(row_selected[i], row_topk_ord[i]);
    } else if (i < merge_items) {
      const int rel = i - topk_tokens;
      const int chunk = rel / topk_tokens;
      const int rank = rel - chunk * topk_tokens;
      const int candidate_pos = chunk * 4096 + rank;
      if (chunk < candidate_chunk_count && candidate_pos < candidate_capacity) {
        key = row_keys[candidate_pos];
      }
    }
    keys[i] = key;
  }
  __syncthreads();

  for (int k_size = 2; k_size <= merge_capacity; k_size <<= 1) {
    const int stop_stride = (k_size == merge_capacity) ? topk_tokens : 1;
    for (int j = k_size >> 1; j >= stop_stride && j > 0; j >>= 1) {
      for (int i = tid; i < merge_capacity; i += blockDim.x) {
        const int other = i ^ j;
        if (other > i && other < merge_capacity) {
          const uint64_t key_i = keys[i];
          const uint64_t key_o = keys[other];
          const bool left_should_be_better = (i & k_size) == 0;
          const bool i_better = key_i >= key_o;
          const bool should_swap =
              (left_should_be_better && !i_better)
              || (!left_should_be_better && i_better);
          if (should_swap) {
            keys[i] = key_o;
            keys[other] = key_i;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int i = tid; i < topk_tokens; i += blockDim.x) {
    const uint64_t key = keys[i];
    const int32_t ordinal = candidate_key_ordinal(key);
    row_topk[i] = candidate_token_from_ordinal(
        ordinal, selected_blocks, row_linear, effective_block_topk, block_size);
    row_selected[i] = candidate_key_score(key);
    row_topk_ord[i] = ordinal;
  }
}

template <typename scalar_t>
void launch_hisa_block_reps_batched_typed(
    const void* k, void* block_reps,
    int L, int B, int D, int block_size, int MB, cudaStream_t stream) {
  const dim3 grid(MB, B);
  const dim3 block(128);
  hisa_block_reps_batched_kernel<scalar_t><<<grid, block, 0, stream>>>(
      static_cast<const scalar_t*>(k),
      static_cast<scalar_t*>(block_reps),
      L, B, D, block_size, MB);
}

template <class GEMM, typename scalar_t, typename weight_t, typename prefix_t>
void launch_hisa_selector_megakernel_batched_typed(
    const void* q, const void* k, const void* block_reps, const void* weights,
    const void* prefix_lens,
    int32_t* topk_indices, float* selected_scores,
    int Q, int B, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one,
    cudaStream_t stream) {
  const dim3 block = GEMM::block_dim;
  const int block_score_capacity = next_power_of_two_int(max(1, MB));
  const int candidate_capacity =
      next_power_of_two_int(max(topk_tokens, max(1, effective_block_topk * block_size)));
  const int use_u16_tokens = L <= static_cast<int>(UINT16_MAX) ? 1 : 0;
  const size_t candidate_token_bytes =
      (use_u16_tokens ? sizeof(uint16_t) : sizeof(int32_t))
      * static_cast<size_t>(candidate_capacity);
  size_t smem_bytes =
      sizeof(float) * static_cast<size_t>(block_score_capacity)
      + sizeof(int) * static_cast<size_t>(block_score_capacity)
      + sizeof(int) * static_cast<size_t>(effective_block_topk)
      + sizeof(float) * static_cast<size_t>(candidate_capacity)
      + sizeof(uint16_t) * static_cast<size_t>(candidate_capacity)
      + candidate_token_bytes
      + 16
      + cublasdx::get_shared_storage_size<GEMM>();
  auto kernel = hisa_selector_megakernel_batched_kernel<GEMM, scalar_t, weight_t, prefix_t>;
  if (smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }
  kernel<<<Q * B, block, smem_bytes, stream>>>(
      static_cast<const scalar_t*>(q),
      static_cast<const scalar_t*>(k),
      static_cast<const scalar_t*>(block_reps),
      static_cast<const weight_t*>(weights),
      static_cast<const prefix_t*>(prefix_lens),
      topk_indices,
      selected_scores,
      Q,
      B,
      L,
      MB,
      block_size,
      block_topk,
      compression_ratio,
      effective_block_topk,
      topk_tokens,
      block_score_capacity,
      candidate_capacity,
      use_u16_tokens,
      prefix_lens_shared,
      force_first,
      force_last,
      force_last_minus_one);
}

template <class GEMM, typename scalar_t, typename weight_t, typename prefix_t>
void launch_hisa_selector_megakernel_parallel_batched_typed(
    const void* q, const void* k, const void* block_reps, const void* weights,
    const void* prefix_lens,
    int32_t* selected_blocks,
    uint64_t* candidate_keys,
    int32_t* topk_indices, float* selected_scores,
    int Q, int B, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int candidate_capacity,
    int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one,
    cudaStream_t stream) {
  const dim3 block = GEMM::block_dim;
  const int block_score_capacity = next_power_of_two_int(max(1, MB));

  size_t select_smem_bytes =
      sizeof(float) * static_cast<size_t>(block_score_capacity)
      + sizeof(int) * static_cast<size_t>(block_score_capacity)
      + 16
      + cublasdx::get_shared_storage_size<GEMM>();
  auto select_kernel =
      hisa_selector_megakernel_select_blocks_batched_kernel<
          GEMM, scalar_t, weight_t, prefix_t>;
  if (select_smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        select_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(select_smem_bytes)));
  }
  select_kernel<<<Q * B, block, select_smem_bytes, stream>>>(
      static_cast<const scalar_t*>(q),
      static_cast<const scalar_t*>(k),
      static_cast<const scalar_t*>(block_reps),
      static_cast<const weight_t*>(weights),
      static_cast<const prefix_t*>(prefix_lens),
      selected_blocks,
      Q,
      B,
      L,
      MB,
      block_size,
      block_topk,
      compression_ratio,
      effective_block_topk,
      block_score_capacity,
      prefix_lens_shared,
      force_first,
      force_last,
      force_last_minus_one);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const size_t score_smem_bytes = cublasdx::get_shared_storage_size<GEMM>();
  auto score_kernel =
      hisa_selector_megakernel_score_candidates_batched_kernel<
          GEMM, scalar_t, weight_t, prefix_t>;
  if (score_smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        score_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(score_smem_bytes)));
  }
  const int candidate_tile_count = (candidate_capacity + kTileN - 1) / kTileN;
  const dim3 score_grid(Q * B, candidate_tile_count);
  score_kernel<<<score_grid, block, score_smem_bytes, stream>>>(
      static_cast<const scalar_t*>(q),
      static_cast<const scalar_t*>(k),
      static_cast<const weight_t*>(weights),
      static_cast<const prefix_t*>(prefix_lens),
      selected_blocks,
      candidate_keys,
      Q,
      B,
      L,
      block_size,
	      effective_block_topk,
	      candidate_capacity,
	      prefix_lens_shared,
	      0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (candidate_capacity >= 8192 && candidate_capacity <= 16384 && topk_tokens <= 2048) {
    const int sort_chunk_count = (candidate_capacity + 4095) / 4096;
    hisa_selector_megakernel_candidate_sort4096_batched_kernel
        <<<dim3(Q * B, sort_chunk_count), 128, 0, stream>>>(
            candidate_keys,
            Q * B,
            candidate_capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int merge_capacity = topk_tokens * sort_chunk_count;
    const size_t merge_smem_bytes =
        sizeof(uint64_t) * static_cast<size_t>(merge_capacity);
    auto merge_kernel = hisa_selector_megakernel_candidate_merge_topk_batched_kernel;
    if (merge_smem_bytes > 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          merge_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(merge_smem_bytes)));
    }
    merge_kernel<<<Q * B, 128, merge_smem_bytes, stream>>>(
        candidate_keys,
        selected_blocks,
        topk_indices,
        selected_scores,
        nullptr,
        Q * B,
        candidate_capacity,
        topk_tokens,
        effective_block_topk,
        block_size);
  } else {
    hisa_selector_megakernel_candidate_topk_batched_kernel<<<Q * B, 128, 0, stream>>>(
        candidate_keys,
        selected_blocks,
        topk_indices,
        selected_scores,
        Q * B,
        candidate_capacity,
        topk_tokens,
        effective_block_topk,
        block_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class GEMM, typename scalar_t, typename weight_t, typename prefix_t>
void launch_hisa_selector_megakernel_parallel_streaming_batched_typed(
    const void* q, const void* k, const void* block_reps, const void* weights,
    const void* prefix_lens,
    int32_t* selected_blocks,
    uint64_t* candidate_keys,
    int32_t* topk_indices, float* selected_scores, int32_t* topk_ordinals,
    int Q, int B, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int candidate_scratch_capacity,
    int total_candidate_capacity, int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one,
    cudaStream_t stream) {
  const dim3 block = GEMM::block_dim;
  const int block_score_capacity = next_power_of_two_int(max(1, MB));

  size_t select_smem_bytes =
      sizeof(float) * static_cast<size_t>(block_score_capacity)
      + sizeof(int) * static_cast<size_t>(block_score_capacity)
      + 16
      + cublasdx::get_shared_storage_size<GEMM>();
  auto select_kernel =
      hisa_selector_megakernel_select_blocks_batched_kernel<
          GEMM, scalar_t, weight_t, prefix_t>;
  if (select_smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        select_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(select_smem_bytes)));
  }
  select_kernel<<<Q * B, block, select_smem_bytes, stream>>>(
      static_cast<const scalar_t*>(q),
      static_cast<const scalar_t*>(k),
      static_cast<const scalar_t*>(block_reps),
      static_cast<const weight_t*>(weights),
      static_cast<const prefix_t*>(prefix_lens),
      selected_blocks,
      Q,
      B,
      L,
      MB,
      block_size,
      block_topk,
      compression_ratio,
      effective_block_topk,
      block_score_capacity,
      prefix_lens_shared,
      force_first,
      force_last,
      force_last_minus_one);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const size_t score_smem_bytes = cublasdx::get_shared_storage_size<GEMM>();
  auto score_kernel =
      hisa_selector_megakernel_score_candidates_batched_kernel<
          GEMM, scalar_t, weight_t, prefix_t>;
  if (score_smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        score_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(score_smem_bytes)));
  }

  const int candidate_tile_count =
      (candidate_scratch_capacity + kTileN - 1) / kTileN;
  const int sort_chunk_count = (candidate_scratch_capacity + 4095) / 4096;
  const int merge_capacity = topk_tokens * sort_chunk_count;
  const size_t merge_smem_bytes =
      sizeof(uint64_t) * static_cast<size_t>(merge_capacity);
  auto merge_kernel = hisa_selector_megakernel_candidate_merge_topk_batched_kernel;
  if (merge_smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        merge_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(merge_smem_bytes)));
  }

  const int running_merge_capacity =
      next_power_of_two_int(topk_tokens * (sort_chunk_count + 1));
  const size_t running_merge_smem_bytes =
      sizeof(uint64_t) * static_cast<size_t>(running_merge_capacity);
  auto running_merge_kernel =
      hisa_selector_megakernel_merge_running_topk_batched_kernel;
  if (running_merge_smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        running_merge_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(running_merge_smem_bytes)));
  }

  bool have_running_topk = false;
  for (int candidate_offset = 0; candidate_offset < total_candidate_capacity;
       candidate_offset += candidate_scratch_capacity) {
    const dim3 score_grid(Q * B, candidate_tile_count);
    score_kernel<<<score_grid, block, score_smem_bytes, stream>>>(
        static_cast<const scalar_t*>(q),
        static_cast<const scalar_t*>(k),
        static_cast<const weight_t*>(weights),
        static_cast<const prefix_t*>(prefix_lens),
        selected_blocks,
        candidate_keys,
        Q,
        B,
        L,
        block_size,
        effective_block_topk,
        candidate_scratch_capacity,
        prefix_lens_shared,
        candidate_offset);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    hisa_selector_megakernel_candidate_sort4096_batched_kernel
        <<<dim3(Q * B, sort_chunk_count), 128, 0, stream>>>(
            candidate_keys,
            Q * B,
            candidate_scratch_capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (!have_running_topk) {
      merge_kernel<<<Q * B, 128, merge_smem_bytes, stream>>>(
          candidate_keys,
          selected_blocks,
          topk_indices,
          selected_scores,
          topk_ordinals,
          Q * B,
          candidate_scratch_capacity,
          topk_tokens,
          effective_block_topk,
          block_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      have_running_topk = true;
    } else {
      running_merge_kernel<<<Q * B, 128, running_merge_smem_bytes, stream>>>(
          candidate_keys,
          selected_blocks,
          topk_indices,
          selected_scores,
          topk_ordinals,
          Q * B,
          candidate_scratch_capacity,
          topk_tokens,
          running_merge_capacity,
          effective_block_topk,
          block_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
}

template <class GEMM, typename scalar_t, typename prefix_t>
void dispatch_hisa_selector_megakernel_weight_prefix(
    const void* q, const void* k, const void* block_reps, const void* weights,
    const void* prefix_lens,
    int32_t* topk_indices, float* selected_scores,
    int Q, int B, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one,
    int weight_dtype, cudaStream_t stream) {
  if (weight_dtype == kDTypeFloat32) {
    launch_hisa_selector_megakernel_batched_typed<GEMM, scalar_t, float, prefix_t>(
        q, k, block_reps, weights, prefix_lens,
        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        prefix_lens_shared, force_first, force_last,
        force_last_minus_one, stream);
  } else if (weight_dtype == kDTypeBFloat16) {
    launch_hisa_selector_megakernel_batched_typed<GEMM, scalar_t, __nv_bfloat16, prefix_t>(
        q, k, block_reps, weights, prefix_lens,
        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        prefix_lens_shared, force_first, force_last,
        force_last_minus_one, stream);
  } else if (weight_dtype == kDTypeFloat16) {
    launch_hisa_selector_megakernel_batched_typed<GEMM, scalar_t, __half, prefix_t>(
        q, k, block_reps, weights, prefix_lens,
        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        prefix_lens_shared, force_first, force_last,
        force_last_minus_one, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported HISA megakernel weight dtype");
  }
}

template <class GEMM, typename scalar_t>
void dispatch_hisa_selector_megakernel_weight(
    const void* q, const void* k, const void* block_reps, const void* weights,
    const void* prefix_lens,
    int32_t* topk_indices, float* selected_scores,
    int Q, int B, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one,
    int weight_dtype, int prefix_dtype, cudaStream_t stream) {
  if (prefix_dtype == 0) {
    dispatch_hisa_selector_megakernel_weight_prefix<GEMM, scalar_t, int32_t>(
        q, k, block_reps, weights, prefix_lens, topk_indices, selected_scores,
        Q, B, L, MB, block_size, block_topk, compression_ratio,
        effective_block_topk, topk_tokens, prefix_lens_shared, force_first,
        force_last, force_last_minus_one, weight_dtype, stream);
  } else if (prefix_dtype == 1) {
    dispatch_hisa_selector_megakernel_weight_prefix<GEMM, scalar_t, int64_t>(
        q, k, block_reps, weights, prefix_lens, topk_indices, selected_scores,
        Q, B, L, MB, block_size, block_topk, compression_ratio,
        effective_block_topk, topk_tokens, prefix_lens_shared, force_first,
        force_last, force_last_minus_one, weight_dtype, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported HISA megakernel prefix dtype");
  }
}

template <class GEMM, typename scalar_t, typename prefix_t>
void dispatch_hisa_selector_megakernel_parallel_weight_prefix(
    const void* q, const void* k, const void* block_reps, const void* weights,
    const void* prefix_lens,
    int32_t* selected_blocks,
    uint64_t* candidate_keys,
    int32_t* topk_indices, float* selected_scores,
    int Q, int B, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int candidate_capacity,
    int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one,
    int weight_dtype, cudaStream_t stream) {
  if (weight_dtype == kDTypeFloat32) {
	    launch_hisa_selector_megakernel_parallel_batched_typed<
	        GEMM, scalar_t, float, prefix_t>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_capacity, prefix_lens_shared, force_first, force_last,
        force_last_minus_one, stream);
  } else if (weight_dtype == kDTypeBFloat16) {
	    launch_hisa_selector_megakernel_parallel_batched_typed<
	        GEMM, scalar_t, __nv_bfloat16, prefix_t>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_capacity, prefix_lens_shared, force_first, force_last,
        force_last_minus_one, stream);
  } else if (weight_dtype == kDTypeFloat16) {
	    launch_hisa_selector_megakernel_parallel_batched_typed<
	        GEMM, scalar_t, __half, prefix_t>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_capacity, prefix_lens_shared, force_first, force_last,
        force_last_minus_one, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported HISA parallel megakernel weight dtype");
  }
}

template <class GEMM, typename scalar_t>
void dispatch_hisa_selector_megakernel_parallel_weight(
    const void* q, const void* k, const void* block_reps, const void* weights,
    const void* prefix_lens,
    int32_t* selected_blocks,
    uint64_t* candidate_keys,
    int32_t* topk_indices, float* selected_scores,
    int Q, int B, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int candidate_capacity,
    int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one,
    int weight_dtype, int prefix_dtype, cudaStream_t stream) {
  if (prefix_dtype == 0) {
	    dispatch_hisa_selector_megakernel_parallel_weight_prefix<GEMM, scalar_t, int32_t>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_capacity, prefix_lens_shared, force_first, force_last,
        force_last_minus_one, weight_dtype, stream);
  } else if (prefix_dtype == 1) {
	    dispatch_hisa_selector_megakernel_parallel_weight_prefix<GEMM, scalar_t, int64_t>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_capacity, prefix_lens_shared, force_first, force_last,
        force_last_minus_one, weight_dtype, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported HISA parallel megakernel prefix dtype");
  }
}

template <class GEMM, typename scalar_t, typename prefix_t>
void dispatch_hisa_selector_megakernel_parallel_streaming_weight_prefix(
    const void* q, const void* k, const void* block_reps, const void* weights,
    const void* prefix_lens,
    int32_t* selected_blocks,
    uint64_t* candidate_keys,
    int32_t* topk_indices, float* selected_scores, int32_t* topk_ordinals,
    int Q, int B, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int candidate_scratch_capacity,
    int total_candidate_capacity, int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one,
    int weight_dtype, cudaStream_t stream) {
  if (weight_dtype == kDTypeFloat32) {
	    launch_hisa_selector_megakernel_parallel_streaming_batched_typed<
	        GEMM, scalar_t, float, prefix_t>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, topk_ordinals, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_scratch_capacity, total_candidate_capacity, prefix_lens_shared,
        force_first, force_last, force_last_minus_one, stream);
  } else if (weight_dtype == kDTypeBFloat16) {
	    launch_hisa_selector_megakernel_parallel_streaming_batched_typed<
	        GEMM, scalar_t, __nv_bfloat16, prefix_t>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, topk_ordinals, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_scratch_capacity, total_candidate_capacity, prefix_lens_shared,
        force_first, force_last, force_last_minus_one, stream);
  } else if (weight_dtype == kDTypeFloat16) {
	    launch_hisa_selector_megakernel_parallel_streaming_batched_typed<
	        GEMM, scalar_t, __half, prefix_t>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, topk_ordinals, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_scratch_capacity, total_candidate_capacity, prefix_lens_shared,
        force_first, force_last, force_last_minus_one, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported HISA streaming megakernel weight dtype");
  }
}

template <class GEMM, typename scalar_t>
void dispatch_hisa_selector_megakernel_parallel_streaming_weight(
    const void* q, const void* k, const void* block_reps, const void* weights,
    const void* prefix_lens,
    int32_t* selected_blocks,
    uint64_t* candidate_keys,
    int32_t* topk_indices, float* selected_scores, int32_t* topk_ordinals,
    int Q, int B, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int candidate_scratch_capacity,
    int total_candidate_capacity, int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one,
    int weight_dtype, int prefix_dtype, cudaStream_t stream) {
  if (prefix_dtype == 0) {
	    dispatch_hisa_selector_megakernel_parallel_streaming_weight_prefix<
	        GEMM, scalar_t, int32_t>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, topk_ordinals, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_scratch_capacity, total_candidate_capacity, prefix_lens_shared,
        force_first, force_last, force_last_minus_one, weight_dtype, stream);
  } else if (prefix_dtype == 1) {
	    dispatch_hisa_selector_megakernel_parallel_streaming_weight_prefix<
	        GEMM, scalar_t, int64_t>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, topk_ordinals, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_scratch_capacity, total_candidate_capacity, prefix_lens_shared,
        force_first, force_last, force_last_minus_one, weight_dtype, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported HISA streaming megakernel prefix dtype");
  }
}

}  // namespace

void launch_hisa_block_reps_batched_fwd(
    const void* k, void* block_reps,
    int L, int B, int D, int block_size, int MB, int scalar_dtype,
    cudaStream_t stream) {
  if (scalar_dtype == kDTypeFloat32) {
    launch_hisa_block_reps_batched_typed<float>(
        k, block_reps, L, B, D, block_size, MB, stream);
  } else if (scalar_dtype == kDTypeBFloat16) {
    launch_hisa_block_reps_batched_typed<__nv_bfloat16>(
        k, block_reps, L, B, D, block_size, MB, stream);
  } else if (scalar_dtype == kDTypeFloat16) {
    launch_hisa_block_reps_batched_typed<__half>(
        k, block_reps, L, B, D, block_size, MB, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported HISA block-rep dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_hisa_selector_megakernel_batched_fwd(
    const void* q, const void* k, const void* block_reps, const void* weights,
    const void* prefix_lens,
    int32_t* topk_indices, float* selected_scores,
    int Q, int B, int H, int D, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one,
    int scalar_dtype, int weight_dtype, int prefix_dtype, cudaStream_t stream) {
  if (H != kIndexerHeads || D != kHeadDim) {
    C10_THROW_ERROR(ValueError, "HISA megakernel requires H=64 and D=128");
  }
  if (effective_block_topk <= 0 || topk_tokens <= 0 || block_size <= 0) {
    C10_THROW_ERROR(ValueError, "HISA megakernel got non-positive selector dimensions");
  }
  const int candidate_capacity =
      next_power_of_two_int(max(topk_tokens, max(1, effective_block_topk * block_size)));
  if (candidate_capacity > 8192) {
    C10_THROW_ERROR(
        ValueError,
        "HISA megakernel candidate_capacity exceeds 8192; reduce local context or "
        "compression before using this backend");
  }

  if (scalar_dtype == kDTypeFloat32) {
    dispatch_hisa_selector_megakernel_weight<HisaMegaGemmF32, float>(
        q, k, block_reps, weights, prefix_lens,
        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        prefix_lens_shared, force_first, force_last, force_last_minus_one,
        weight_dtype, prefix_dtype, stream);
  } else if (scalar_dtype == kDTypeBFloat16) {
    dispatch_hisa_selector_megakernel_weight<HisaMegaGemmBF16, __nv_bfloat16>(
        q, k, block_reps, weights, prefix_lens,
        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        prefix_lens_shared, force_first, force_last, force_last_minus_one,
        weight_dtype, prefix_dtype, stream);
  } else if (scalar_dtype == kDTypeFloat16) {
    dispatch_hisa_selector_megakernel_weight<HisaMegaGemmF16, __half>(
        q, k, block_reps, weights, prefix_lens,
        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        prefix_lens_shared, force_first, force_last, force_last_minus_one,
        weight_dtype, prefix_dtype, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported HISA megakernel q/k dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_hisa_selector_megakernel_parallel_batched_fwd(
    const void* q, const void* k, const void* block_reps, const void* weights,
    const void* prefix_lens,
    int32_t* selected_blocks,
    uint64_t* candidate_keys,
    int32_t* topk_indices, float* selected_scores,
    int Q, int B, int H, int D, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int candidate_capacity,
    int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one,
    int scalar_dtype, int weight_dtype, int prefix_dtype, cudaStream_t stream) {
  if (H != kIndexerHeads || D != kHeadDim) {
    C10_THROW_ERROR(ValueError, "HISA parallel megakernel requires H=64 and D=128");
  }
  if (effective_block_topk <= 0 || topk_tokens <= 0 || block_size <= 0) {
    C10_THROW_ERROR(ValueError, "HISA parallel megakernel got non-positive selector dimensions");
  }
  const int expected_candidate_capacity =
      next_power_of_two_int(max(topk_tokens, max(1, effective_block_topk * block_size)));
  if (candidate_capacity != expected_candidate_capacity) {
    C10_THROW_ERROR(ValueError, "HISA parallel megakernel got mismatched candidate capacity");
  }
  if (candidate_capacity > 8192) {
    C10_THROW_ERROR(
        ValueError,
        "HISA parallel megakernel candidate_capacity exceeds 8192; this production "
        "backend is hard-capped to the validated 8k path");
  }

	  if (scalar_dtype == kDTypeFloat32) {
	    dispatch_hisa_selector_megakernel_parallel_weight<HisaMegaGemmF32, float>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_capacity, prefix_lens_shared, force_first, force_last,
        force_last_minus_one, weight_dtype, prefix_dtype, stream);
	  } else if (scalar_dtype == kDTypeBFloat16) {
	    dispatch_hisa_selector_megakernel_parallel_weight<HisaMegaGemmBF16, __nv_bfloat16>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_capacity, prefix_lens_shared, force_first, force_last,
        force_last_minus_one, weight_dtype, prefix_dtype, stream);
	  } else if (scalar_dtype == kDTypeFloat16) {
	    dispatch_hisa_selector_megakernel_parallel_weight<HisaMegaGemmF16, __half>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_capacity, prefix_lens_shared, force_first, force_last,
        force_last_minus_one, weight_dtype, prefix_dtype, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported HISA parallel megakernel q/k dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_hisa_selector_megakernel_parallel_streaming_batched_fwd(
    const void* q, const void* k, const void* block_reps, const void* weights,
    const void* prefix_lens,
    int32_t* selected_blocks,
    uint64_t* candidate_keys,
    int32_t* topk_indices, float* selected_scores, int32_t* topk_ordinals,
    int Q, int B, int H, int D, int L, int MB, int block_size,
    int block_topk, float compression_ratio,
    int effective_block_topk, int topk_tokens, int candidate_scratch_capacity,
    int total_candidate_capacity, int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one,
    int scalar_dtype, int weight_dtype, int prefix_dtype, cudaStream_t stream) {
  if (H != kIndexerHeads || D != kHeadDim) {
    C10_THROW_ERROR(ValueError, "HISA streaming megakernel requires H=64 and D=128");
  }
  if (effective_block_topk <= 0 || topk_tokens <= 0 || block_size <= 0) {
    C10_THROW_ERROR(ValueError, "HISA streaming megakernel got non-positive selector dimensions");
  }
  const int expected_candidate_capacity =
      next_power_of_two_int(max(topk_tokens, max(1, effective_block_topk * block_size)));
  if (total_candidate_capacity != expected_candidate_capacity) {
    C10_THROW_ERROR(ValueError, "HISA streaming megakernel got mismatched total candidate capacity");
  }
  if (total_candidate_capacity > 8192) {
    C10_THROW_ERROR(
        ValueError,
        "HISA streaming megakernel total candidate_capacity exceeds 8192; this "
        "experimental path is not the production 8k selector");
  }
  if (candidate_scratch_capacity <= 0 || candidate_scratch_capacity > 8192) {
    C10_THROW_ERROR(ValueError, "HISA streaming megakernel scratch capacity must be <= 8192");
  }
  if ((candidate_scratch_capacity & (candidate_scratch_capacity - 1)) != 0) {
    C10_THROW_ERROR(ValueError, "HISA streaming megakernel scratch capacity must be power of two");
  }
  if (candidate_scratch_capacity < topk_tokens) {
    C10_THROW_ERROR(ValueError, "HISA streaming megakernel scratch must fit topk");
  }
  if (topk_tokens > 1024) {
    C10_THROW_ERROR(ValueError, "HISA streaming megakernel currently requires topk <= 1024");
  }

	  if (scalar_dtype == kDTypeFloat32) {
	    dispatch_hisa_selector_megakernel_parallel_streaming_weight<HisaMegaGemmF32, float>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, topk_ordinals, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_scratch_capacity, total_candidate_capacity, prefix_lens_shared,
        force_first, force_last, force_last_minus_one, weight_dtype, prefix_dtype, stream);
  } else if (scalar_dtype == kDTypeBFloat16) {
	    dispatch_hisa_selector_megakernel_parallel_streaming_weight<
	        HisaMegaGemmBF16, __nv_bfloat16>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, topk_ordinals, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_scratch_capacity, total_candidate_capacity, prefix_lens_shared,
        force_first, force_last, force_last_minus_one, weight_dtype, prefix_dtype, stream);
	  } else if (scalar_dtype == kDTypeFloat16) {
	    dispatch_hisa_selector_megakernel_parallel_streaming_weight<HisaMegaGemmF16, __half>(
	        q, k, block_reps, weights, prefix_lens, selected_blocks,
	        candidate_keys,
	        topk_indices, selected_scores, topk_ordinals, Q, B, L, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, topk_tokens,
        candidate_scratch_capacity, total_candidate_capacity, prefix_lens_shared,
        force_first, force_last, force_last_minus_one, weight_dtype, prefix_dtype, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported HISA streaming megakernel q/k dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace hisa_indexer
}  // namespace megatron
