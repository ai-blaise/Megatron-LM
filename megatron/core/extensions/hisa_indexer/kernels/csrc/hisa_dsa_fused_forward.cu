// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Fused HISA selector + split-QK selected DSA attention forward.
//
// This path intentionally keeps the distributed teacher all-reduce outside the
// kernel: NCCL is the semantic boundary between local attention-head teacher
// mass and globally normalized teacher probabilities. Everything local to the
// selected-attention/indexer forward is fused into one CUDA launch.

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

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <type_traits>

namespace megatron {
namespace hisa_indexer {

namespace {

constexpr int kIndexerHeads = 64;
constexpr int kIndexerDim = 128;
constexpr int kTileN = 128;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;

constexpr int kDTypeF32 = 0;
constexpr int kDTypeBF16 = 1;
constexpr int kDTypeF16 = 2;

constexpr int kPrefixI32 = 0;
constexpr int kPrefixI64 = 1;

constexpr int kTopkI16 = 0;
constexpr int kTopkI32 = 1;
constexpr int kTopkI64 = 2;

using HisaDsaGemmF32 = decltype(
    cublasdx::Size<kIndexerHeads, kTileN, kIndexerDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using HisaDsaGemmBF16 = decltype(
    cublasdx::Size<kIndexerHeads, kTileN, kIndexerDim>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using HisaDsaGemmF16 = decltype(
    cublasdx::Size<kIndexerHeads, kTileN, kIndexerDim>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ bool finite_float(float v) {
  return isfinite(v);
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

__device__ __forceinline__ float load_typed(const void* ptr, int64_t idx, int dtype) {
  if (dtype == kDTypeF32) {
    return reinterpret_cast<const float*>(ptr)[idx];
  }
  if (dtype == kDTypeBF16) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
  }
  return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

__device__ __forceinline__ void store_typed(void* ptr, int64_t idx, float v, int dtype) {
  if (dtype == kDTypeF32) {
    reinterpret_cast<float*>(ptr)[idx] = v;
  } else if (dtype == kDTypeBF16) {
    reinterpret_cast<__nv_bfloat16*>(ptr)[idx] = __float2bfloat16(v);
  } else {
    reinterpret_cast<__half*>(ptr)[idx] = __float2half(v);
  }
}

__device__ __forceinline__ int load_prefix_len(
    const void* prefix_lens,
    int64_t idx,
    int prefix_dtype) {
  if (prefix_dtype == kPrefixI32) {
    return reinterpret_cast<const int32_t*>(prefix_lens)[idx];
  }
  return static_cast<int>(reinterpret_cast<const int64_t*>(prefix_lens)[idx]);
}

__device__ __forceinline__ void store_topk(void* ptr, int64_t idx, int32_t value, int dtype) {
  if (dtype == kTopkI16) {
    reinterpret_cast<int16_t*>(ptr)[idx] = static_cast<int16_t>(value);
  } else if (dtype == kTopkI32) {
    reinterpret_cast<int32_t*>(ptr)[idx] = value;
  } else {
    reinterpret_cast<int64_t*>(ptr)[idx] = static_cast<int64_t>(value);
  }
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
    float s = tid < kWarps ? scratch[tid] : 0.0f;
    s = warp_sum(s);
    if (tid == 0) {
      scratch[8] = s;
    }
  }
  __syncthreads();
  return scratch[8];
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

__device__ __forceinline__ int hisa_forced_block_budget(
    int row_blocks,
    int force_first,
    int force_last,
    int force_last_minus_one) {
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

template <class GEMM, typename scalar_t, typename weight_t>
__global__ void hisa_dsa_split_qk_fused_fwd_kernel(
    const scalar_t* __restrict__ q_indexer,       // [Q, B, 64, 128]
    const weight_t* __restrict__ weights,         // [Q, B, 64]
    const scalar_t* __restrict__ k_indexer,       // [S, B, 128]
    const scalar_t* __restrict__ block_reps,      // [B, MB, 128]
    const void* __restrict__ prefix_lens,         // [Q] or [B, Q]
    const void* __restrict__ query_nope,          // [Q, B, H, D]
    const void* __restrict__ query_pe,            // [Q, B, H, P]
    const void* __restrict__ key_nope,            // [S, B, H, D]
    const void* __restrict__ key_pe,              // [S, B, KPH, P]
    const void* __restrict__ value,               // [S, B, H, V]
    const int64_t* __restrict__ query_pos,
    const int64_t* __restrict__ key_pos,
    void* __restrict__ topk_out,                  // [B, Q, K]
    float* __restrict__ selected_scores_out,      // [B, Q, K]
    void* __restrict__ output,                    // [Q, B, H, V]
    float* __restrict__ lse,                      // [B * Q, H]
    float* __restrict__ teacher_probs,            // [B * Q, K]
    int Q,
    int B,
    int S,
    int H,
    int D,
    int P,
    int KPH,
    int V,
    int MB,
    int block_size,
    int block_topk,
    float compression_ratio,
    int effective_block_topk,
    int K,
    int q_start,
    int block_score_capacity,
    int candidate_capacity,
    int64_t query_nope_stride_s,
    int64_t query_nope_stride_b,
    int64_t query_nope_stride_h,
    int64_t query_nope_stride_d,
    int64_t key_nope_stride_s,
    int64_t key_nope_stride_b,
    int64_t key_nope_stride_h,
    int64_t key_nope_stride_d,
    int64_t value_stride_s,
    int64_t value_stride_b,
    int64_t value_stride_h,
    int64_t value_stride_v,
    float softmax_scale,
    int scalar_dtype,
    int prefix_dtype,
    int topk_dtype,
    int has_positions,
    int prefix_lens_shared,
    int force_first,
    int force_last,
    int force_last_minus_one) {
  const int row_linear = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (row_linear >= Q * B) {
    return;
  }
  const int batch = row_linear / Q;
  const int row = row_linear - batch * Q;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* block_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * static_cast<size_t>(block_score_capacity);
  int* block_indices = reinterpret_cast<int*>(cursor);
  cursor += sizeof(int) * static_cast<size_t>(block_score_capacity);
  int* selected_blocks = reinterpret_cast<int*>(cursor);
  cursor += sizeof(int) * static_cast<size_t>(effective_block_topk);
  float* candidate_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * static_cast<size_t>(candidate_capacity);
  int32_t* candidate_tokens = reinterpret_cast<int32_t*>(cursor);
  cursor += sizeof(int32_t) * static_cast<size_t>(candidate_capacity);
  uint16_t* candidate_ordinals = reinterpret_cast<uint16_t*>(cursor);
  cursor += sizeof(uint16_t) * static_cast<size_t>(candidate_capacity);
  cursor = align_dynamic_smem(cursor, 16);
  auto gemm_smem = reinterpret_cast<void*>(cursor);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());
  cursor += cublasdx::get_shared_storage_size<GEMM>();
  cursor = align_dynamic_smem(cursor, 16);
  float* q_nope_s = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * static_cast<size_t>(H) * static_cast<size_t>(D);
  float* q_pe_s = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * static_cast<size_t>(H) * static_cast<size_t>(P);
  float* teacher_s = reinterpret_cast<float*>(cursor);

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
    candidate_tokens[i] = -1;
    candidate_ordinals[i] = UINT16_MAX;
  }
  __syncthreads();

  const int prefix_index = prefix_lens_shared ? row : (batch * Q + row);
  const int prefix_len = max(0, min(load_prefix_len(prefix_lens, prefix_index, prefix_dtype), S));
  const int row_blocks = min(MB, ceil_div_int(prefix_len, block_size));
  const int64_t row_batch = static_cast<int64_t>(batch) * Q + row;
  if (prefix_len <= 0 || row_blocks <= 0) {
    for (int i = tid; i < K; i += blockDim.x) {
      store_topk(topk_out, row_batch * K + i, -1, topk_dtype);
      selected_scores_out[row_batch * K + i] = -INFINITY;
      teacher_probs[row_batch * K + i] = 0.0f;
    }
    for (int idx = tid; idx < H * V; idx += blockDim.x) {
      store_typed(
          output,
          ((static_cast<int64_t>(row) * B + batch) * H * V) + idx,
          0.0f,
          scalar_dtype);
    }
    for (int h = tid; h < H; h += blockDim.x) {
      lse[row_batch * H + h] = -FLT_MAX;
    }
    return;
  }

  const scalar_t* q_row =
      q_indexer + (static_cast<int64_t>(row) * B + batch) * kIndexerHeads * kIndexerDim;
  const weight_t* w_row =
      weights + (static_cast<int64_t>(row) * B + batch) * kIndexerHeads;
  const scalar_t* block_rep_batch =
      block_reps + static_cast<int64_t>(batch) * MB * kIndexerDim;

  for (int idx = tid; idx < kIndexerHeads * kIndexerDim; idx += blockDim.x) {
    const int h = idx / kIndexerDim;
    const int d = idx - h * kIndexerDim;
    a_shared(h, d) = q_row[static_cast<int64_t>(h) * kIndexerDim + d];
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < row_blocks; tile_start += kTileN) {
    const int tile_count = min(kTileN, row_blocks - tile_start);
    for (int idx = tid; idx < kIndexerDim * kTileN; idx += blockDim.x) {
      const int d = idx / kTileN;
      const int n = idx - d * kTileN;
      scalar_t v = from_float<scalar_t>(0.0f);
      if (n < tile_count) {
        v = block_rep_batch[static_cast<int64_t>(tile_start + n) * kIndexerDim + d];
      }
      b_shared(d, n) = v;
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
  const int final_start = final_block * block_size;
  const int final_end = min(final_start + block_size, prefix_len);
  const int final_count = max(1, final_end - final_start);
  if (final_count < block_size) {
    if (tid == 0) {
      score_accum = 0.0f;
    }
    __syncthreads();
    for (int h = 0; h < kIndexerHeads; ++h) {
      float partial = 0.0f;
      for (int d = tid; d < kIndexerDim; d += blockDim.x) {
        float sum = 0.0f;
        for (int tok = final_start; tok < final_end; ++tok) {
          sum += to_float(k_indexer[(static_cast<int64_t>(tok) * B + batch) * kIndexerDim + d]);
        }
        partial +=
            to_float(q_row[static_cast<int64_t>(h) * kIndexerDim + d])
            * (sum / static_cast<float>(final_count));
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
      for (int idx = tid; idx < kIndexerDim * kTileN; idx += blockDim.x) {
        const int d = idx / kTileN;
        const int n = idx - d * kTileN;
        scalar_t v = from_float<scalar_t>(0.0f);
        if (n < tile_count) {
          const int tok = tile_start + n;
          v = k_indexer[(static_cast<int64_t>(tok) * B + batch) * kIndexerDim + d];
        }
        b_shared(d, n) = v;
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
          candidate_tokens[candidate_pos] = token_idx;
          candidate_ordinals[candidate_pos] =
              n < tile_count ? static_cast<uint16_t>(candidate_pos) : UINT16_MAX;
        }
      }
      __syncthreads();
    }
  }

  const bool exact_prefix_select =
      is_power_of_two_int_device(K) && K <= (candidate_capacity >> 1);
  const int final_stop_stride = exact_prefix_select ? K : 1;
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
            const int32_t token_i = candidate_tokens[i];
            candidate_tokens[i] = candidate_tokens[other];
            candidate_tokens[other] = token_i;
            const uint16_t ord_i_swap = candidate_ordinals[i];
            candidate_ordinals[i] = candidate_ordinals[other];
            candidate_ordinals[other] = ord_i_swap;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int i = tid; i < K; i += blockDim.x) {
    const int64_t out = row_batch * K + i;
    store_topk(topk_out, out, candidate_tokens[i], topk_dtype);
    selected_scores_out[out] = candidate_scores[i];
    teacher_s[i] = 0.0f;
  }
  for (int idx = tid; idx < H * D; idx += blockDim.x) {
    const int head = idx / D;
    const int d = idx - head * D;
    const int64_t q_base =
        static_cast<int64_t>(row) * query_nope_stride_s +
        static_cast<int64_t>(batch) * query_nope_stride_b +
        static_cast<int64_t>(head) * query_nope_stride_h;
    q_nope_s[idx] = load_typed(query_nope, q_base + d * query_nope_stride_d, scalar_dtype);
  }
  for (int idx = tid; idx < H * P; idx += blockDim.x) {
    const int head = idx / P;
    const int d = idx - head * P;
    const int64_t q_base = ((static_cast<int64_t>(row) * B + batch) * H + head) * P;
    q_pe_s[idx] = load_typed(query_pe, q_base + d, scalar_dtype);
  }
  __syncthreads();

  const int64_t q_abs = has_positions ? query_pos[row] : static_cast<int64_t>(q_start + row);

  for (int head = warp; head < H; head += kWarps) {
    const int pe_head = min(head, KPH - 1);
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    for (int slot = 0; slot < K; ++slot) {
      const int32_t selected = candidate_tokens[slot];
      const int safe_selected = selected > 0 ? selected : 0;
      bool valid = selected >= 0 && selected < S;
      if (valid) {
        const int64_t selected_abs = has_positions ? key_pos[safe_selected] : selected;
        valid = selected_abs <= q_abs;
      }

      const int64_t k_nope_base =
          static_cast<int64_t>(safe_selected) * key_nope_stride_s +
          static_cast<int64_t>(batch) * key_nope_stride_b +
          static_cast<int64_t>(head) * key_nope_stride_h;
      const int64_t k_pe_base =
          ((static_cast<int64_t>(safe_selected) * B + batch) * KPH + pe_head) * P;

      float score_nope = 0.0f;
      float score_pe = 0.0f;
      if (valid) {
        for (int d = lane; d < D; d += 32) {
          score_nope += q_nope_s[head * D + d] *
                        load_typed(key_nope, k_nope_base + d * key_nope_stride_d, scalar_dtype);
        }
        for (int d = lane; d < P; d += 32) {
          score_pe += q_pe_s[head * P + d] * load_typed(key_pe, k_pe_base + d, scalar_dtype);
        }
      }
      score_nope = warp_sum(score_nope);
      score_pe = warp_sum(score_pe);
      float score = (score_nope + score_pe) * softmax_scale;
      if (lane == 0) {
        score = valid && finite_float(score) ? score : -FLT_MAX;
      }
      score = __shfl_sync(0xffffffff, score, 0);
      valid = valid && finite_float(score);
      if (valid) {
        const float m_new = fmaxf(m_i, score);
        l_i = l_i * expf(m_i - m_new) + expf(score - m_new);
        m_i = m_new;
      }
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int slot = 0; slot < K; ++slot) {
      const int32_t selected = candidate_tokens[slot];
      const int safe_selected = selected > 0 ? selected : 0;
      bool valid = selected >= 0 && selected < S;
      if (valid) {
        const int64_t selected_abs = has_positions ? key_pos[safe_selected] : selected;
        valid = selected_abs <= q_abs;
      }

      const int64_t k_nope_base =
          static_cast<int64_t>(safe_selected) * key_nope_stride_s +
          static_cast<int64_t>(batch) * key_nope_stride_b +
          static_cast<int64_t>(head) * key_nope_stride_h;
      const int64_t k_pe_base =
          ((static_cast<int64_t>(safe_selected) * B + batch) * KPH + pe_head) * P;

      float score_nope = 0.0f;
      float score_pe = 0.0f;
      if (valid) {
        for (int d = lane; d < D; d += 32) {
          score_nope += q_nope_s[head * D + d] *
                        load_typed(key_nope, k_nope_base + d * key_nope_stride_d, scalar_dtype);
        }
        for (int d = lane; d < P; d += 32) {
          score_pe += q_pe_s[head * P + d] * load_typed(key_pe, k_pe_base + d, scalar_dtype);
        }
      }
      score_nope = warp_sum(score_nope);
      score_pe = warp_sum(score_pe);
      float score = (score_nope + score_pe) * softmax_scale;
      if (lane == 0) {
        score = valid && finite_float(score) ? score : -FLT_MAX;
      }
      score = __shfl_sync(0xffffffff, score, 0);

      valid = valid && finite_float(score) && finite_float(m_i) && finite_float(l_i);
      float prob = (valid && l_i > 0.0f) ? expf(score - m_i) / l_i : 0.0f;
      if (!finite_float(prob)) {
        prob = 0.0f;
        valid = false;
      }
      if (lane == 0 && prob != 0.0f) {
        atomicAdd(teacher_s + slot, prob);
      }
      if (valid && prob != 0.0f) {
        const int64_t v_base =
            static_cast<int64_t>(safe_selected) * value_stride_s +
            static_cast<int64_t>(batch) * value_stride_b +
            static_cast<int64_t>(head) * value_stride_h;
        const int d0 = lane;
        const int d1 = lane + 32;
        const int d2 = lane + 64;
        const int d3 = lane + 96;
        if (d0 < V) {
          acc0 += prob * load_typed(value, v_base + d0 * value_stride_v, scalar_dtype);
        }
        if (d1 < V) {
          acc1 += prob * load_typed(value, v_base + d1 * value_stride_v, scalar_dtype);
        }
        if (d2 < V) {
          acc2 += prob * load_typed(value, v_base + d2 * value_stride_v, scalar_dtype);
        }
        if (d3 < V) {
          acc3 += prob * load_typed(value, v_base + d3 * value_stride_v, scalar_dtype);
        }
      }
    }

    const int64_t out_base = ((static_cast<int64_t>(row) * B + batch) * H + head) * V;
    const int d0 = lane;
    const int d1 = lane + 32;
    const int d2 = lane + 64;
    const int d3 = lane + 96;
    if (d0 < V) {
      store_typed(output, out_base + d0, acc0, scalar_dtype);
    }
    if (d1 < V) {
      store_typed(output, out_base + d1, acc1, scalar_dtype);
    }
    if (d2 < V) {
      store_typed(output, out_base + d2, acc2, scalar_dtype);
    }
    if (d3 < V) {
      store_typed(output, out_base + d3, acc3, scalar_dtype);
    }
    if (lane == 0) {
      lse[row_batch * H + head] = l_i > 0.0f ? m_i + logf(l_i) : -FLT_MAX;
    }
  }

  __syncthreads();
  for (int slot = tid; slot < K; slot += blockDim.x) {
    teacher_probs[row_batch * K + slot] = teacher_s[slot];
  }
}

template <class GEMM, typename scalar_t, typename weight_t>
void launch_hisa_dsa_split_qk_fused_fwd_typed(
    const void* q_indexer, const void* weights, const void* k_indexer,
    const void* block_reps, const void* prefix_lens,
    const void* query_nope, const void* query_pe, const void* key_nope,
    const void* key_pe, const void* value,
    const int64_t* query_positions, const int64_t* key_positions,
    void* topk_indices, float* selected_scores, void* output, float* lse,
    float* teacher_probs, int Q, int B, int S, int H, int D, int P,
    int KPH, int V, int MB, int block_size, int block_topk,
    float compression_ratio, int effective_block_topk, int K, int q_start,
    int64_t query_nope_stride_s, int64_t query_nope_stride_b,
    int64_t query_nope_stride_h, int64_t query_nope_stride_d,
    int64_t key_nope_stride_s, int64_t key_nope_stride_b,
    int64_t key_nope_stride_h, int64_t key_nope_stride_d,
    int64_t value_stride_s, int64_t value_stride_b, int64_t value_stride_h,
    int64_t value_stride_v, float softmax_scale, int scalar_dtype,
    int prefix_dtype, int topk_dtype, int has_positions, int prefix_lens_shared,
    int force_first, int force_last, int force_last_minus_one, cudaStream_t stream) {
  const int block_score_capacity = next_power_of_two_int(std::max(1, MB));
  const int candidate_capacity =
      next_power_of_two_int(std::max(K, std::max(1, effective_block_topk * block_size)));
  const size_t smem_bytes =
      sizeof(float) * static_cast<size_t>(block_score_capacity)
      + sizeof(int) * static_cast<size_t>(block_score_capacity)
      + sizeof(int) * static_cast<size_t>(effective_block_topk)
      + sizeof(float) * static_cast<size_t>(candidate_capacity)
      + sizeof(int32_t) * static_cast<size_t>(candidate_capacity)
      + sizeof(uint16_t) * static_cast<size_t>(candidate_capacity)
      + 16
      + cublasdx::get_shared_storage_size<GEMM>()
      + 16
      + sizeof(float) * static_cast<size_t>(H) * static_cast<size_t>(D)
      + sizeof(float) * static_cast<size_t>(H) * static_cast<size_t>(P)
      + sizeof(float) * static_cast<size_t>(K);
  auto kernel = hisa_dsa_split_qk_fused_fwd_kernel<GEMM, scalar_t, weight_t>;
  if (smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }
  kernel<<<Q * B, GEMM::block_dim, smem_bytes, stream>>>(
      static_cast<const scalar_t*>(q_indexer),
      static_cast<const weight_t*>(weights),
      static_cast<const scalar_t*>(k_indexer),
      static_cast<const scalar_t*>(block_reps),
      prefix_lens,
      query_nope,
      query_pe,
      key_nope,
      key_pe,
      value,
      query_positions,
      key_positions,
      topk_indices,
      selected_scores,
      output,
      lse,
      teacher_probs,
      Q,
      B,
      S,
      H,
      D,
      P,
      KPH,
      V,
      MB,
      block_size,
      block_topk,
      compression_ratio,
      effective_block_topk,
      K,
      q_start,
      block_score_capacity,
      candidate_capacity,
      query_nope_stride_s,
      query_nope_stride_b,
      query_nope_stride_h,
      query_nope_stride_d,
      key_nope_stride_s,
      key_nope_stride_b,
      key_nope_stride_h,
      key_nope_stride_d,
      value_stride_s,
      value_stride_b,
      value_stride_h,
      value_stride_v,
      softmax_scale,
      scalar_dtype,
      prefix_dtype,
      topk_dtype,
      has_positions,
      prefix_lens_shared,
      force_first,
      force_last,
      force_last_minus_one);
}

template <class GEMM, typename scalar_t>
void dispatch_hisa_dsa_weight(
    const void* q_indexer, const void* weights, const void* k_indexer,
    const void* block_reps, const void* prefix_lens,
    const void* query_nope, const void* query_pe, const void* key_nope,
    const void* key_pe, const void* value,
    const int64_t* query_positions, const int64_t* key_positions,
    void* topk_indices, float* selected_scores, void* output, float* lse,
    float* teacher_probs, int Q, int B, int S, int H, int D, int P,
    int KPH, int V, int MB, int block_size, int block_topk,
    float compression_ratio, int effective_block_topk, int K, int q_start,
    int64_t query_nope_stride_s, int64_t query_nope_stride_b,
    int64_t query_nope_stride_h, int64_t query_nope_stride_d,
    int64_t key_nope_stride_s, int64_t key_nope_stride_b,
    int64_t key_nope_stride_h, int64_t key_nope_stride_d,
    int64_t value_stride_s, int64_t value_stride_b, int64_t value_stride_h,
    int64_t value_stride_v, float softmax_scale, int scalar_dtype,
    int weight_dtype, int prefix_dtype, int topk_dtype, int has_positions,
    int prefix_lens_shared, int force_first, int force_last,
    int force_last_minus_one, cudaStream_t stream) {
  if (weight_dtype == kDTypeF32) {
    launch_hisa_dsa_split_qk_fused_fwd_typed<GEMM, scalar_t, float>(
        q_indexer, weights, k_indexer, block_reps, prefix_lens,
        query_nope, query_pe, key_nope, key_pe, value,
        query_positions, key_positions, topk_indices, selected_scores, output,
        lse, teacher_probs, Q, B, S, H, D, P, KPH, V, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, K, q_start,
        query_nope_stride_s, query_nope_stride_b, query_nope_stride_h,
        query_nope_stride_d, key_nope_stride_s, key_nope_stride_b,
        key_nope_stride_h, key_nope_stride_d, value_stride_s, value_stride_b,
        value_stride_h, value_stride_v, softmax_scale, scalar_dtype,
        prefix_dtype, topk_dtype, has_positions, prefix_lens_shared,
        force_first, force_last, force_last_minus_one, stream);
  } else if (weight_dtype == kDTypeBF16) {
    launch_hisa_dsa_split_qk_fused_fwd_typed<GEMM, scalar_t, __nv_bfloat16>(
        q_indexer, weights, k_indexer, block_reps, prefix_lens,
        query_nope, query_pe, key_nope, key_pe, value,
        query_positions, key_positions, topk_indices, selected_scores, output,
        lse, teacher_probs, Q, B, S, H, D, P, KPH, V, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, K, q_start,
        query_nope_stride_s, query_nope_stride_b, query_nope_stride_h,
        query_nope_stride_d, key_nope_stride_s, key_nope_stride_b,
        key_nope_stride_h, key_nope_stride_d, value_stride_s, value_stride_b,
        value_stride_h, value_stride_v, softmax_scale, scalar_dtype,
        prefix_dtype, topk_dtype, has_positions, prefix_lens_shared,
        force_first, force_last, force_last_minus_one, stream);
  } else {
    launch_hisa_dsa_split_qk_fused_fwd_typed<GEMM, scalar_t, __half>(
        q_indexer, weights, k_indexer, block_reps, prefix_lens,
        query_nope, query_pe, key_nope, key_pe, value,
        query_positions, key_positions, topk_indices, selected_scores, output,
        lse, teacher_probs, Q, B, S, H, D, P, KPH, V, MB, block_size,
        block_topk, compression_ratio, effective_block_topk, K, q_start,
        query_nope_stride_s, query_nope_stride_b, query_nope_stride_h,
        query_nope_stride_d, key_nope_stride_s, key_nope_stride_b,
        key_nope_stride_h, key_nope_stride_d, value_stride_s, value_stride_b,
        value_stride_h, value_stride_v, softmax_scale, scalar_dtype,
        prefix_dtype, topk_dtype, has_positions, prefix_lens_shared,
        force_first, force_last, force_last_minus_one, stream);
  }
}

}  // namespace

void launch_hisa_dsa_split_qk_fused_fwd(
    const void* q_indexer, const void* weights, const void* k_indexer,
    const void* block_reps, const void* prefix_lens,
    const void* query_nope, const void* query_pe, const void* key_nope,
    const void* key_pe, const void* value,
    const int64_t* query_positions, const int64_t* key_positions,
    void* topk_indices, float* selected_scores, void* output, float* lse,
    float* teacher_probs, int q_len, int bsz, int sk, int indexer_heads,
    int indexer_dim, int num_heads, int head_dim, int pos_dim,
    int key_pe_heads, int value_dim, int mb, int block_size,
    int block_topk, float compression_ratio, int effective_block_topk,
    int topk_count, int q_start,
    int64_t query_nope_stride_s, int64_t query_nope_stride_b,
    int64_t query_nope_stride_h, int64_t query_nope_stride_d,
    int64_t key_nope_stride_s, int64_t key_nope_stride_b,
    int64_t key_nope_stride_h, int64_t key_nope_stride_d,
    int64_t value_stride_s, int64_t value_stride_b, int64_t value_stride_h,
    int64_t value_stride_v, float softmax_scale, int scalar_dtype,
    int weight_dtype, int prefix_dtype, int topk_dtype, int has_positions,
    int prefix_lens_shared, int force_first, int force_last,
    int force_last_minus_one, cudaStream_t stream) {
  if (indexer_heads != kIndexerHeads || indexer_dim != kIndexerDim) {
    C10_THROW_ERROR(
        ValueError,
        "fused HISA/DSA forward requires indexer shape [Q,B,64,128]");
  }
  if (topk_count <= 0 || topk_count > 2048) {
    C10_THROW_ERROR(ValueError, "fused HISA/DSA forward supports topk in (0, 2048]");
  }
  if (num_heads <= 0 || num_heads > 64 || head_dim <= 0 || head_dim > 256 ||
      pos_dim <= 0 || pos_dim > 256 || value_dim <= 0 || value_dim > 128) {
    C10_THROW_ERROR(ValueError, "fused HISA/DSA forward received unsupported attention shape");
  }
  if (block_size <= 0 || effective_block_topk <= 0 || mb <= 0) {
    C10_THROW_ERROR(ValueError, "fused HISA/DSA forward received invalid HISA block shape");
  }
  const int candidate_capacity = next_power_of_two_int(
      std::max(topk_count, std::max(1, effective_block_topk * block_size)));
  if (candidate_capacity > 8192) {
    C10_THROW_ERROR(
        ValueError,
        "fused HISA/DSA forward currently requires candidate capacity <= 8192");
  }
  if (topk_dtype == kTopkI16 && sk > 32768) {
    C10_THROW_ERROR(ValueError, "int16 topk output is only valid for sk <= 32768");
  }

  if (scalar_dtype == kDTypeF32) {
    dispatch_hisa_dsa_weight<HisaDsaGemmF32, float>(
        q_indexer, weights, k_indexer, block_reps, prefix_lens,
        query_nope, query_pe, key_nope, key_pe, value,
        query_positions, key_positions, topk_indices, selected_scores, output,
        lse, teacher_probs, q_len, bsz, sk, num_heads, head_dim, pos_dim,
        key_pe_heads, value_dim, mb, block_size, block_topk, compression_ratio,
        effective_block_topk, topk_count, q_start, query_nope_stride_s,
        query_nope_stride_b, query_nope_stride_h, query_nope_stride_d,
        key_nope_stride_s, key_nope_stride_b, key_nope_stride_h,
        key_nope_stride_d, value_stride_s, value_stride_b, value_stride_h,
        value_stride_v, softmax_scale, scalar_dtype, weight_dtype, prefix_dtype,
        topk_dtype, has_positions, prefix_lens_shared, force_first, force_last,
        force_last_minus_one, stream);
  } else if (scalar_dtype == kDTypeBF16) {
    dispatch_hisa_dsa_weight<HisaDsaGemmBF16, __nv_bfloat16>(
        q_indexer, weights, k_indexer, block_reps, prefix_lens,
        query_nope, query_pe, key_nope, key_pe, value,
        query_positions, key_positions, topk_indices, selected_scores, output,
        lse, teacher_probs, q_len, bsz, sk, num_heads, head_dim, pos_dim,
        key_pe_heads, value_dim, mb, block_size, block_topk, compression_ratio,
        effective_block_topk, topk_count, q_start, query_nope_stride_s,
        query_nope_stride_b, query_nope_stride_h, query_nope_stride_d,
        key_nope_stride_s, key_nope_stride_b, key_nope_stride_h,
        key_nope_stride_d, value_stride_s, value_stride_b, value_stride_h,
        value_stride_v, softmax_scale, scalar_dtype, weight_dtype, prefix_dtype,
        topk_dtype, has_positions, prefix_lens_shared, force_first, force_last,
        force_last_minus_one, stream);
  } else if (scalar_dtype == kDTypeF16) {
    dispatch_hisa_dsa_weight<HisaDsaGemmF16, __half>(
        q_indexer, weights, k_indexer, block_reps, prefix_lens,
        query_nope, query_pe, key_nope, key_pe, value,
        query_positions, key_positions, topk_indices, selected_scores, output,
        lse, teacher_probs, q_len, bsz, sk, num_heads, head_dim, pos_dim,
        key_pe_heads, value_dim, mb, block_size, block_topk, compression_ratio,
        effective_block_topk, topk_count, q_start, query_nope_stride_s,
        query_nope_stride_b, query_nope_stride_h, query_nope_stride_d,
        key_nope_stride_s, key_nope_stride_b, key_nope_stride_h,
        key_nope_stride_d, value_stride_s, value_stride_b, value_stride_h,
        value_stride_v, softmax_scale, scalar_dtype, weight_dtype, prefix_dtype,
        topk_dtype, has_positions, prefix_lens_shared, force_first, force_last,
        force_last_minus_one, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported fused HISA/DSA scalar dtype");
  }
}

}  // namespace hisa_indexer
}  // namespace megatron
