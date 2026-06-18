// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Persistent cooperative HISA selector + split-QK selected DSA forward.
//
// This is the throughput-oriented replacement for the rejected fused v0 row
// kernel. The important property is the work decomposition: one CUDA launch,
// but many CTAs cooperate across phases so candidate refinement remains
// row x candidate-tile parallel and selected attention remains row x head-group
// parallel.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#include <cublasdx.hpp>
#include <cub/block/block_radix_sort.cuh>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <type_traits>

namespace megatron {
namespace hisa_indexer {

namespace {

namespace cg = cooperative_groups;

constexpr int kIndexerHeads = 64;
constexpr int kIndexerDim = 128;
constexpr int kTileN = 128;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kSortThreads = 128;
constexpr int kSortItemsPerThread = 32;
constexpr int kSortWidth = kSortThreads * kSortItemsPerThread;
constexpr int kNopeDim = 128;
constexpr int kPeDim = 64;
constexpr int kValueDim = 128;
constexpr int kAttentionTileK = 64;
constexpr int kAttentionHeadsPerCTA = 4;
constexpr int kAttentionNopeGroupK = kAttentionHeadsPerCTA * kNopeDim;
constexpr int kAttentionPeGroupK = kAttentionHeadsPerCTA * kPeDim;
constexpr int kAttentionValueGroupK = kAttentionHeadsPerCTA * kAttentionTileK;

constexpr int kDTypeF32 = 0;
constexpr int kDTypeBF16 = 1;
constexpr int kDTypeF16 = 2;
constexpr int kPrefixI32 = 0;
constexpr int kPrefixI64 = 1;
constexpr int kTopkI16 = 0;
constexpr int kTopkI32 = 1;
constexpr int kTopkI64 = 2;

using HisaDsaPersistentGemmF32 = decltype(
    cublasdx::Size<kIndexerHeads, kTileN, kIndexerDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using HisaDsaPersistentGemmBF16 = decltype(
    cublasdx::Size<kIndexerHeads, kTileN, kIndexerDim>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using HisaDsaPersistentGemmF16 = decltype(
    cublasdx::Size<kIndexerHeads, kTileN, kIndexerDim>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaScoreNopePersistentGemmF32 = decltype(
    cublasdx::Size<1, kAttentionTileK, kNopeDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaScorePePersistentGemmF32 = decltype(
    cublasdx::Size<1, kAttentionTileK, kPeDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaValuePersistentGemmF32 = decltype(
    cublasdx::Size<1, kValueDim, kAttentionTileK>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaScoreNopePersistentGemmBF16 = decltype(
    cublasdx::Size<1, kAttentionTileK, kNopeDim>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaScorePePersistentGemmBF16 = decltype(
    cublasdx::Size<1, kAttentionTileK, kPeDim>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaValuePersistentGemmBF16 = decltype(
    cublasdx::Size<1, kValueDim, kAttentionTileK>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaScoreNopePersistentGemmF16 = decltype(
    cublasdx::Size<1, kAttentionTileK, kNopeDim>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaScorePePersistentGemmF16 = decltype(
    cublasdx::Size<1, kAttentionTileK, kPeDim>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaValuePersistentGemmF16 = decltype(
    cublasdx::Size<1, kValueDim, kAttentionTileK>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaScoreNopeGroupPersistentGemmF32 = decltype(
    cublasdx::Size<kAttentionHeadsPerCTA, kAttentionTileK, kAttentionNopeGroupK>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaScorePeGroupPersistentGemmF32 = decltype(
    cublasdx::Size<kAttentionHeadsPerCTA, kAttentionTileK, kAttentionPeGroupK>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaValueGroupPersistentGemmF32 = decltype(
    cublasdx::Size<kAttentionHeadsPerCTA, kValueDim, kAttentionValueGroupK>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaScoreNopeGroupPersistentGemmBF16 = decltype(
    cublasdx::Size<kAttentionHeadsPerCTA, kAttentionTileK, kAttentionNopeGroupK>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaScorePeGroupPersistentGemmBF16 = decltype(
    cublasdx::Size<kAttentionHeadsPerCTA, kAttentionTileK, kAttentionPeGroupK>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaValueGroupPersistentGemmBF16 = decltype(
    cublasdx::Size<kAttentionHeadsPerCTA, kValueDim, kAttentionValueGroupK>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaScoreNopeGroupPersistentGemmF16 = decltype(
    cublasdx::Size<kAttentionHeadsPerCTA, kAttentionTileK, kAttentionNopeGroupK>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaScorePeGroupPersistentGemmF16 = decltype(
    cublasdx::Size<kAttentionHeadsPerCTA, kAttentionTileK, kAttentionPeGroupK>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

using DsaValueGroupPersistentGemmF16 = decltype(
    cublasdx::Size<kAttentionHeadsPerCTA, kValueDim, kAttentionValueGroupK>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Block());

struct HisaDsaPersistentParams {
  const void* q_indexer;
  const void* weights;
  const void* k_indexer;
  const void* block_reps;
  const void* prefix_lens;
  const void* query_nope;
  const void* query_pe;
  const void* key_nope;
  const void* key_pe;
  const void* value;
  const int64_t* query_pos;
  const int64_t* key_pos;
  void* topk_out;
  float* selected_scores_out;
  void* output;
  float* lse;
  float* teacher_probs;
  int32_t* selected_blocks;
  uint64_t* candidate_keys;
  float* attention_scores;
  float* output_accum;
  int Q;
  int B;
  int S;
  int H;
  int D;
  int P;
  int KPH;
  int V;
  int MB;
  int block_size;
  int block_topk;
  float compression_ratio;
  int effective_block_topk;
  int K;
  int q_start;
  int block_score_capacity;
  int candidate_capacity;
  int candidate_tile_count;
  int sort_chunk_count;
  int attention_tile_count;
  int attention_head_block_count;
  int64_t query_nope_stride_s;
  int64_t query_nope_stride_b;
  int64_t query_nope_stride_h;
  int64_t query_nope_stride_d;
  int64_t key_nope_stride_s;
  int64_t key_nope_stride_b;
  int64_t key_nope_stride_h;
  int64_t key_nope_stride_d;
  int64_t value_stride_s;
  int64_t value_stride_b;
  int64_t value_stride_h;
  int64_t value_stride_v;
  float softmax_scale;
  int scalar_dtype;
  int prefix_dtype;
  int topk_dtype;
  int has_positions;
  int prefix_lens_shared;
  int force_first;
  int force_last;
  int force_last_minus_one;
};

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

__device__ __forceinline__ bool finite_float(float v) {
  return isfinite(v);
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

__device__ __forceinline__ int32_t load_topk(const void* ptr, int64_t idx, int dtype) {
  if (dtype == kTopkI16) {
    return static_cast<int32_t>(reinterpret_cast<const int16_t*>(ptr)[idx]);
  }
  if (dtype == kTopkI32) {
    return reinterpret_cast<const int32_t*>(ptr)[idx];
  }
  return static_cast<int32_t>(reinterpret_cast<const int64_t*>(ptr)[idx]);
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

__device__ __forceinline__ float block_reduce_max_128(float v, float* scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = v;
  __syncthreads();
#pragma unroll
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ __forceinline__ float block_reduce_sum_128(float v, float* scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = v;
  __syncthreads();
#pragma unroll
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  return scratch[0];
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
__device__ void select_blocks_task(
    const HisaDsaPersistentParams& p,
    int row_linear,
    unsigned char* smem_raw) {
  const int tid = threadIdx.x;
  const int batch = row_linear / p.Q;
  const int row = row_linear - batch * p.Q;

  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* block_scores = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * static_cast<size_t>(p.block_score_capacity);
  int* block_indices = reinterpret_cast<int*>(cursor);
  cursor += sizeof(int) * static_cast<size_t>(p.block_score_capacity);
  cursor = align_dynamic_smem(cursor, 16);
  auto gemm_smem = reinterpret_cast<void*>(cursor);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  __shared__ float reduce_scratch[9];
  __shared__ float score_accum;

  for (int i = tid; i < p.block_score_capacity; i += blockDim.x) {
    block_scores[i] = -INFINITY;
    block_indices[i] = INT_MAX;
  }
  for (int i = tid; i < p.effective_block_topk; i += blockDim.x) {
    p.selected_blocks[row_linear * p.effective_block_topk + i] = -1;
  }
  __syncthreads();

  const int prefix_index = p.prefix_lens_shared ? row : (batch * p.Q + row);
  const int prefix_len =
      max(0, min(load_prefix_len(p.prefix_lens, prefix_index, p.prefix_dtype), p.S));
  const int row_blocks = min(p.MB, ceil_div_int(prefix_len, p.block_size));
  if (prefix_len <= 0 || row_blocks <= 0) {
    return;
  }

  const scalar_t* q = reinterpret_cast<const scalar_t*>(p.q_indexer);
  const scalar_t* k = reinterpret_cast<const scalar_t*>(p.k_indexer);
  const scalar_t* block_reps = reinterpret_cast<const scalar_t*>(p.block_reps);
  const weight_t* weights = reinterpret_cast<const weight_t*>(p.weights);
  const scalar_t* q_row =
      q + (static_cast<int64_t>(row) * p.B + batch) * kIndexerHeads * kIndexerDim;
  const weight_t* w_row =
      weights + (static_cast<int64_t>(row) * p.B + batch) * kIndexerHeads;
  const scalar_t* block_rep_batch =
      block_reps + static_cast<int64_t>(batch) * p.MB * kIndexerDim;

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
      scalar_t value = from_float<scalar_t>(0.0f);
      if (n < tile_count) {
        value = block_rep_batch[static_cast<int64_t>(tile_start + n) * kIndexerDim + d];
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
  const int block_start = final_block * p.block_size;
  const int block_end = min(block_start + p.block_size, prefix_len);
  const int block_token_count = max(1, block_end - block_start);
  if (block_token_count < p.block_size) {
    if (tid == 0) {
      score_accum = 0.0f;
    }
    __syncthreads();
    for (int h = 0; h < kIndexerHeads; ++h) {
      float partial = 0.0f;
      for (int d = tid; d < kIndexerDim; d += blockDim.x) {
        float sum = 0.0f;
        for (int tok = block_start; tok < block_end; ++tok) {
          sum += to_float(k[(static_cast<int64_t>(tok) * p.B + batch) * kIndexerDim + d]);
        }
        partial +=
            to_float(q_row[static_cast<int64_t>(h) * kIndexerDim + d])
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

  for (int i = tid; i < p.block_score_capacity; i += blockDim.x) {
    block_indices[i] = i < row_blocks ? i : INT_MAX;
    if (i >= row_blocks) {
      block_scores[i] = -INFINITY;
    }
  }
  __syncthreads();

  if (tid == 0) {
    if (p.force_first) {
      block_scores[0] = INFINITY;
    }
    if (p.force_last) {
      block_scores[row_blocks - 1] = INFINITY;
    }
    if (p.force_last_minus_one && row_blocks >= 2) {
      block_scores[row_blocks - 2] = INFINITY;
    }
  }
  __syncthreads();

  for (int k_size = 2; k_size <= p.block_score_capacity; k_size <<= 1) {
    for (int j = k_size >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < p.block_score_capacity; i += blockDim.x) {
        const int other = i ^ j;
        if (other > i && other < p.block_score_capacity) {
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
      p.block_topk,
      p.compression_ratio,
      p.effective_block_topk,
      p.force_first,
      p.force_last,
      p.force_last_minus_one);
  for (int slot = tid; slot < p.effective_block_topk; slot += blockDim.x) {
    p.selected_blocks[row_linear * p.effective_block_topk + slot] =
        slot < keep ? block_indices[slot] : -1;
  }
}

template <class GEMM, typename scalar_t, typename weight_t>
__device__ void score_candidates_task(
    const HisaDsaPersistentParams& p,
    int row_linear,
    int tile_id,
    unsigned char* smem_raw) {
  const int tid = threadIdx.x;
  const int batch = row_linear / p.Q;
  const int row = row_linear - batch * p.Q;
  const int tile_start = tile_id * kTileN;
  if (tile_start >= p.candidate_capacity) {
    return;
  }

  auto gemm_smem = reinterpret_cast<void*>(smem_raw);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(gemm_smem);
  auto a_shared = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

  const int prefix_index = p.prefix_lens_shared ? row : (batch * p.Q + row);
  const int prefix_len =
      max(0, min(load_prefix_len(p.prefix_lens, prefix_index, p.prefix_dtype), p.S));
  const scalar_t* q = reinterpret_cast<const scalar_t*>(p.q_indexer);
  const scalar_t* k = reinterpret_cast<const scalar_t*>(p.k_indexer);
  const weight_t* weights = reinterpret_cast<const weight_t*>(p.weights);
  const scalar_t* q_row =
      q + (static_cast<int64_t>(row) * p.B + batch) * kIndexerHeads * kIndexerDim;
  const weight_t* w_row =
      weights + (static_cast<int64_t>(row) * p.B + batch) * kIndexerHeads;

  for (int idx = tid; idx < kIndexerHeads * kIndexerDim; idx += blockDim.x) {
    const int h = idx / kIndexerDim;
    const int d = idx - h * kIndexerDim;
    a_shared(h, d) = q_row[static_cast<int64_t>(h) * kIndexerDim + d];
  }
  for (int idx = tid; idx < kIndexerDim * kTileN; idx += blockDim.x) {
    const int d = idx / kTileN;
    const int n = idx - d * kTileN;
    const int candidate_pos = tile_start + n;
    scalar_t value = from_float<scalar_t>(0.0f);
    if (candidate_pos < p.candidate_capacity) {
      const int slot = candidate_pos / p.block_size;
      const int offset = candidate_pos - slot * p.block_size;
      int token = -1;
      if (slot < p.effective_block_topk) {
        const int block_id = p.selected_blocks[row_linear * p.effective_block_topk + slot];
        token = block_id >= 0 ? block_id * p.block_size + offset : -1;
      }
      if (token >= 0 && token < prefix_len && token < p.S) {
        value = k[(static_cast<int64_t>(token) * p.B + batch) * kIndexerDim + d];
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
    if (candidate_pos >= p.candidate_capacity) {
      continue;
    }
    const int slot = candidate_pos / p.block_size;
    const int offset = candidate_pos - slot * p.block_size;
    float score = -INFINITY;
    int32_t ordinal = INT_MAX;
    if (slot < p.effective_block_topk) {
      const int block_id = p.selected_blocks[row_linear * p.effective_block_topk + slot];
      const int token = block_id >= 0 ? block_id * p.block_size + offset : -1;
      if (token >= 0 && token < prefix_len && token < p.S) {
        float accum = 0.0f;
        for (int h = 0; h < kIndexerHeads; ++h) {
          const float dot = c_shared(h, n);
          if (dot > 0.0f) {
            accum += dot * to_float(w_row[h]);
          }
        }
        score = accum;
        ordinal = candidate_pos;
      }
    }
    p.candidate_keys[static_cast<int64_t>(row_linear) * p.candidate_capacity + candidate_pos] =
        candidate_key_from_score_ordinal(score, ordinal);
  }
}

__device__ void sort_candidates_task(
    const HisaDsaPersistentParams& p,
    int row_linear,
    int sort_chunk,
    unsigned char* smem_raw) {
  using Sort = cub::BlockRadixSort<uint64_t, kSortThreads, kSortItemsPerThread>;
  const int tid = threadIdx.x;
  auto* storage = reinterpret_cast<typename Sort::TempStorage*>(smem_raw);
  uint64_t keys[kSortItemsPerThread];
  const int chunk_offset = sort_chunk * kSortWidth;
  uint64_t* row_keys = p.candidate_keys + static_cast<int64_t>(row_linear) * p.candidate_capacity;

#pragma unroll
  for (int item = 0; item < kSortItemsPerThread; ++item) {
    const int local_pos = tid * kSortItemsPerThread + item;
    const int candidate_pos = chunk_offset + local_pos;
    keys[item] = candidate_pos < p.candidate_capacity ? row_keys[candidate_pos] : 0ull;
  }
  Sort(*storage).SortDescending(keys);
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kSortItemsPerThread; ++item) {
    const int local_rank = tid * kSortItemsPerThread + item;
    const int candidate_pos = chunk_offset + local_rank;
    if (candidate_pos < p.candidate_capacity) {
      row_keys[candidate_pos] = keys[item];
    }
  }
}

__device__ void merge_topk_task(
    const HisaDsaPersistentParams& p,
    int row_linear,
    unsigned char* smem_raw) {
  const int tid = threadIdx.x;
  const int merge_capacity = p.K * p.sort_chunk_count;
  uint64_t* keys = reinterpret_cast<uint64_t*>(smem_raw);
  const uint64_t* row_keys =
      p.candidate_keys + static_cast<int64_t>(row_linear) * p.candidate_capacity;

  for (int i = tid; i < merge_capacity; i += blockDim.x) {
    const int chunk = i / p.K;
    const int rank = i - chunk * p.K;
    const int candidate_pos = chunk * kSortWidth + rank;
    keys[i] = (chunk < p.sort_chunk_count && candidate_pos < p.candidate_capacity)
        ? row_keys[candidate_pos]
        : 0ull;
  }
  __syncthreads();

  for (int k_size = 2; k_size <= merge_capacity; k_size <<= 1) {
    const int stop_stride = (k_size == merge_capacity) ? p.K : 1;
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

  for (int i = tid; i < p.K; i += blockDim.x) {
    const int64_t out = static_cast<int64_t>(row_linear) * p.K + i;
    const uint64_t key = keys[i];
    const int32_t ordinal = candidate_key_ordinal(key);
    const int32_t token = candidate_token_from_ordinal(
        ordinal, p.selected_blocks, row_linear, p.effective_block_topk, p.block_size);
    store_topk(p.topk_out, out, token, p.topk_dtype);
    p.selected_scores_out[out] = candidate_key_score(key);
    p.teacher_probs[out] = 0.0f;
  }
}

__device__ void attention_head_group_task(
    const HisaDsaPersistentParams& p,
    int row_linear,
    int head_group,
    unsigned char* smem_raw) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int batch = row_linear / p.Q;
  const int row = row_linear - batch * p.Q;
  const int head_start = head_group * kAttentionHeadsPerCTA;

  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* q_nope_s = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * static_cast<size_t>(kAttentionHeadsPerCTA) * p.D;
  cursor = align_dynamic_smem(cursor, 16);
  float* q_pe_s = reinterpret_cast<float*>(cursor);

  for (int idx = tid; idx < kAttentionHeadsPerCTA * p.D; idx += blockDim.x) {
    const int local_head = idx / p.D;
    const int d = idx - local_head * p.D;
    const int head = head_start + local_head;
    float v = 0.0f;
    if (head < p.H) {
      const int64_t q_base =
          static_cast<int64_t>(row) * p.query_nope_stride_s
          + static_cast<int64_t>(batch) * p.query_nope_stride_b
          + static_cast<int64_t>(head) * p.query_nope_stride_h;
      v = load_typed(p.query_nope, q_base + d * p.query_nope_stride_d, p.scalar_dtype);
    }
    q_nope_s[idx] = v;
  }
  for (int idx = tid; idx < kAttentionHeadsPerCTA * p.P; idx += blockDim.x) {
    const int local_head = idx / p.P;
    const int d = idx - local_head * p.P;
    const int head = head_start + local_head;
    float v = 0.0f;
    if (head < p.H) {
      const int64_t q_base =
          ((static_cast<int64_t>(row) * p.B + batch) * p.H + head) * p.P;
      v = load_typed(p.query_pe, q_base + d, p.scalar_dtype);
    }
    q_pe_s[idx] = v;
  }
  __syncthreads();

  const int64_t q_abs =
      p.has_positions ? p.query_pos[row] : static_cast<int64_t>(p.q_start + row);

  if (warp >= kAttentionHeadsPerCTA) {
    return;
  }
  const int head = head_start + warp;
  if (head >= p.H) {
    return;
  }
  const int pe_head = min(head, p.KPH - 1);

  float m_i = -FLT_MAX;
  float l_i = 0.0f;
  for (int slot = 0; slot < p.K; ++slot) {
    const int64_t topk_off = static_cast<int64_t>(row_linear) * p.K + slot;
    const int32_t selected = load_topk(p.topk_out, topk_off, p.topk_dtype);
    const int safe_selected = selected > 0 ? selected : 0;
    bool valid = selected >= 0 && selected < p.S;
    if (valid) {
      const int64_t selected_abs = p.has_positions ? p.key_pos[safe_selected] : selected;
      valid = selected_abs <= q_abs;
    }

    const int64_t k_nope_base =
        static_cast<int64_t>(safe_selected) * p.key_nope_stride_s
        + static_cast<int64_t>(batch) * p.key_nope_stride_b
        + static_cast<int64_t>(head) * p.key_nope_stride_h;
    const int64_t k_pe_base =
        ((static_cast<int64_t>(safe_selected) * p.B + batch) * p.KPH + pe_head) * p.P;

    float score_nope = 0.0f;
    float score_pe = 0.0f;
    if (valid) {
      for (int d = lane; d < p.D; d += 32) {
        score_nope += q_nope_s[warp * p.D + d] *
                      load_typed(p.key_nope, k_nope_base + d * p.key_nope_stride_d, p.scalar_dtype);
      }
      for (int d = lane; d < p.P; d += 32) {
        score_pe += q_pe_s[warp * p.P + d] * load_typed(p.key_pe, k_pe_base + d, p.scalar_dtype);
      }
    }
    score_nope = warp_sum(score_nope);
    score_pe = warp_sum(score_pe);
    float score = (score_nope + score_pe) * p.softmax_scale;
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

  for (int slot = 0; slot < p.K; ++slot) {
    const int64_t topk_off = static_cast<int64_t>(row_linear) * p.K + slot;
    const int32_t selected = load_topk(p.topk_out, topk_off, p.topk_dtype);
    const int safe_selected = selected > 0 ? selected : 0;
    bool valid = selected >= 0 && selected < p.S;
    if (valid) {
      const int64_t selected_abs = p.has_positions ? p.key_pos[safe_selected] : selected;
      valid = selected_abs <= q_abs;
    }

    const int64_t k_nope_base =
        static_cast<int64_t>(safe_selected) * p.key_nope_stride_s
        + static_cast<int64_t>(batch) * p.key_nope_stride_b
        + static_cast<int64_t>(head) * p.key_nope_stride_h;
    const int64_t k_pe_base =
        ((static_cast<int64_t>(safe_selected) * p.B + batch) * p.KPH + pe_head) * p.P;

    float score_nope = 0.0f;
    float score_pe = 0.0f;
    if (valid) {
      for (int d = lane; d < p.D; d += 32) {
        score_nope += q_nope_s[warp * p.D + d] *
                      load_typed(p.key_nope, k_nope_base + d * p.key_nope_stride_d, p.scalar_dtype);
      }
      for (int d = lane; d < p.P; d += 32) {
        score_pe += q_pe_s[warp * p.P + d] * load_typed(p.key_pe, k_pe_base + d, p.scalar_dtype);
      }
    }
    score_nope = warp_sum(score_nope);
    score_pe = warp_sum(score_pe);
    float score = (score_nope + score_pe) * p.softmax_scale;
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
      atomicAdd(p.teacher_probs + topk_off, prob);
    }
    if (valid && prob != 0.0f) {
      const int64_t v_base =
          static_cast<int64_t>(safe_selected) * p.value_stride_s
          + static_cast<int64_t>(batch) * p.value_stride_b
          + static_cast<int64_t>(head) * p.value_stride_h;
      const int d0 = lane;
      const int d1 = lane + 32;
      const int d2 = lane + 64;
      const int d3 = lane + 96;
      if (d0 < p.V) {
        acc0 += prob * load_typed(p.value, v_base + d0 * p.value_stride_v, p.scalar_dtype);
      }
      if (d1 < p.V) {
        acc1 += prob * load_typed(p.value, v_base + d1 * p.value_stride_v, p.scalar_dtype);
      }
      if (d2 < p.V) {
        acc2 += prob * load_typed(p.value, v_base + d2 * p.value_stride_v, p.scalar_dtype);
      }
      if (d3 < p.V) {
        acc3 += prob * load_typed(p.value, v_base + d3 * p.value_stride_v, p.scalar_dtype);
      }
    }
  }

  const int64_t out_base =
      ((static_cast<int64_t>(row) * p.B + batch) * p.H + head) * p.V;
  const int d0 = lane;
  const int d1 = lane + 32;
  const int d2 = lane + 64;
  const int d3 = lane + 96;
  if (d0 < p.V) {
    store_typed(p.output, out_base + d0, acc0, p.scalar_dtype);
  }
  if (d1 < p.V) {
    store_typed(p.output, out_base + d1, acc1, p.scalar_dtype);
  }
  if (d2 < p.V) {
    store_typed(p.output, out_base + d2, acc2, p.scalar_dtype);
  }
  if (d3 < p.V) {
    store_typed(p.output, out_base + d3, acc3, p.scalar_dtype);
  }
  if (lane == 0) {
    p.lse[static_cast<int64_t>(row_linear) * p.H + head] =
        l_i > 0.0f ? m_i + logf(l_i) : -FLT_MAX;
  }
}

__device__ void attention_init_task(const HisaDsaPersistentParams& p, int row_head) {
  const int tid = threadIdx.x;
  const int row_linear = row_head / p.H;
  const int head = row_head - row_linear * p.H;
  for (int v = tid; v < kValueDim; v += blockDim.x) {
    p.output_accum[(static_cast<int64_t>(row_linear) * p.H + head) * kValueDim + v] = 0.0f;
  }
  if (tid == 0) {
    p.lse[static_cast<int64_t>(row_linear) * p.H + head] = -FLT_MAX;
  }
}

template <class SCORE_NOPE, class SCORE_PE, typename scalar_t>
__device__ void attention_score_tile_task(
    const HisaDsaPersistentParams& p,
    int row_linear,
    int head,
    int tile_id,
    unsigned char* smem_raw) {
  const int tid = threadIdx.x;
  const int batch = row_linear / p.Q;
  const int row = row_linear - batch * p.Q;
  const int tile_start = tile_id * kAttentionTileK;
  const int tile_count = min(kAttentionTileK, p.K - tile_start);
  if (tile_count <= 0 || head >= p.H) {
    return;
  }

  const scalar_t* query_nope = reinterpret_cast<const scalar_t*>(p.query_nope);
  const scalar_t* query_pe = reinterpret_cast<const scalar_t*>(p.query_pe);
  const scalar_t* key_nope = reinterpret_cast<const scalar_t*>(p.key_nope);
  const scalar_t* key_pe = reinterpret_cast<const scalar_t*>(p.key_pe);

  const int pe_head = min(head, p.KPH - 1);
  const int64_t q_abs =
      p.has_positions ? p.query_pos[row] : static_cast<int64_t>(p.q_start + row);
  const int64_t q_nope_base =
      static_cast<int64_t>(row) * p.query_nope_stride_s
      + static_cast<int64_t>(batch) * p.query_nope_stride_b
      + static_cast<int64_t>(head) * p.query_nope_stride_h;
  const int64_t q_pe_base =
      ((static_cast<int64_t>(row) * p.B + batch) * p.H + head) * kPeDim;
  const int64_t row_batch = static_cast<int64_t>(row_linear);

  {
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<SCORE_NOPE>(smem_raw);
    auto a_shared = cublasdx::make_tensor(smem_a, SCORE_NOPE::get_layout_smem_a());
    auto b_shared = cublasdx::make_tensor(smem_b, SCORE_NOPE::get_layout_smem_b());
    auto c_shared = cublasdx::make_tensor(smem_c, SCORE_NOPE::get_layout_smem_c());
    for (int d = tid; d < kNopeDim; d += blockDim.x) {
      a_shared(0, d) = query_nope[q_nope_base + static_cast<int64_t>(d) * p.query_nope_stride_d];
    }
    for (int idx = tid; idx < kNopeDim * kAttentionTileK; idx += blockDim.x) {
      const int d = idx / kAttentionTileK;
      const int n = idx - d * kAttentionTileK;
      scalar_t kv = from_float<scalar_t>(0.0f);
      if (n < tile_count) {
        const int slot = tile_start + n;
        const int32_t selected = load_topk(p.topk_out, row_batch * p.K + slot, p.topk_dtype);
        const int safe_selected = selected > 0 ? selected : 0;
        bool valid = selected >= 0 && selected < p.S;
        if (valid) {
          const int64_t selected_abs = p.has_positions ? p.key_pos[safe_selected] : selected;
          valid = selected_abs <= q_abs;
        }
        if (valid) {
          const int64_t k_base =
              static_cast<int64_t>(safe_selected) * p.key_nope_stride_s
              + static_cast<int64_t>(batch) * p.key_nope_stride_b
              + static_cast<int64_t>(head) * p.key_nope_stride_h;
          kv = key_nope[k_base + static_cast<int64_t>(d) * p.key_nope_stride_d];
        }
      }
      b_shared(d, n) = kv;
    }
    for (int n = tid; n < kAttentionTileK; n += blockDim.x) {
      c_shared(0, n) = 0.0f;
    }
    __syncthreads();
    SCORE_NOPE().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();
    for (int n = tid; n < tile_count; n += blockDim.x) {
      p.attention_scores[(row_batch * p.H + head) * p.K + tile_start + n] = c_shared(0, n);
    }
    __syncthreads();
  }

  {
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<SCORE_PE>(smem_raw);
    auto a_shared = cublasdx::make_tensor(smem_a, SCORE_PE::get_layout_smem_a());
    auto b_shared = cublasdx::make_tensor(smem_b, SCORE_PE::get_layout_smem_b());
    auto c_shared = cublasdx::make_tensor(smem_c, SCORE_PE::get_layout_smem_c());
    for (int d = tid; d < kPeDim; d += blockDim.x) {
      a_shared(0, d) = query_pe[q_pe_base + d];
    }
    for (int idx = tid; idx < kPeDim * kAttentionTileK; idx += blockDim.x) {
      const int d = idx / kAttentionTileK;
      const int n = idx - d * kAttentionTileK;
      scalar_t kv = from_float<scalar_t>(0.0f);
      if (n < tile_count) {
        const int slot = tile_start + n;
        const int32_t selected = load_topk(p.topk_out, row_batch * p.K + slot, p.topk_dtype);
        const int safe_selected = selected > 0 ? selected : 0;
        bool valid = selected >= 0 && selected < p.S;
        if (valid) {
          const int64_t selected_abs = p.has_positions ? p.key_pos[safe_selected] : selected;
          valid = selected_abs <= q_abs;
        }
        if (valid) {
          const int64_t k_base =
              ((static_cast<int64_t>(safe_selected) * p.B + batch) * p.KPH + pe_head) * kPeDim;
          kv = key_pe[k_base + d];
        }
      }
      b_shared(d, n) = kv;
    }
    for (int n = tid; n < kAttentionTileK; n += blockDim.x) {
      c_shared(0, n) = 0.0f;
    }
    __syncthreads();
    SCORE_PE().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();
    for (int n = tid; n < tile_count; n += blockDim.x) {
      const int slot = tile_start + n;
      const int32_t selected = load_topk(p.topk_out, row_batch * p.K + slot, p.topk_dtype);
      const int safe_selected = selected > 0 ? selected : 0;
      bool valid = selected >= 0 && selected < p.S;
      if (valid) {
        const int64_t selected_abs = p.has_positions ? p.key_pos[safe_selected] : selected;
        valid = selected_abs <= q_abs;
      }
      const float no_pe = p.attention_scores[(row_batch * p.H + head) * p.K + slot];
      const float score = valid ? (no_pe + c_shared(0, n)) * p.softmax_scale : -FLT_MAX;
      p.attention_scores[(row_batch * p.H + head) * p.K + slot] =
          finite_float(score) ? score : -FLT_MAX;
    }
  }
}

__device__ void attention_lse_task(const HisaDsaPersistentParams& p, int row_head, unsigned char* smem_raw) {
  const int tid = threadIdx.x;
  float* scratch = reinterpret_cast<float*>(smem_raw);
  float local_m = -FLT_MAX;
  const float* scores = p.attention_scores + static_cast<int64_t>(row_head) * p.K;
  for (int slot = tid; slot < p.K; slot += blockDim.x) {
    local_m = fmaxf(local_m, scores[slot]);
  }
  const float m_i = block_reduce_max_128(local_m, scratch);
  float local_l = 0.0f;
  if (m_i > -3.0e38f) {
    for (int slot = tid; slot < p.K; slot += blockDim.x) {
      const float score = scores[slot];
      if (score > -3.0e38f && finite_float(score)) {
        local_l += expf(score - m_i);
      }
    }
  }
  const float l_i = block_reduce_sum_128(local_l, scratch);
  if (tid == 0) {
    p.lse[row_head] = l_i > 0.0f ? m_i + logf(l_i) : -FLT_MAX;
  }
}

template <class VALUE_GEMM, typename scalar_t>
__device__ void attention_value_tile_task(
    const HisaDsaPersistentParams& p,
    int row_linear,
    int head,
    int tile_id,
    unsigned char* smem_raw) {
  const int tid = threadIdx.x;
  const int batch = row_linear / p.Q;
  const int tile_start = tile_id * kAttentionTileK;
  const int tile_count = min(kAttentionTileK, p.K - tile_start);
  if (tile_count <= 0 || head >= p.H) {
    return;
  }

  const scalar_t* value = reinterpret_cast<const scalar_t*>(p.value);
  const int64_t row_head = static_cast<int64_t>(row_linear) * p.H + head;
  const float lse_val = p.lse[row_head];
  const bool have_lse = lse_val > -3.0e38f && finite_float(lse_val);

  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<VALUE_GEMM>(smem_raw);
  auto a_shared = cublasdx::make_tensor(smem_a, VALUE_GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, VALUE_GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, VALUE_GEMM::get_layout_smem_c());

  for (int n = tid; n < kAttentionTileK; n += blockDim.x) {
    float prob = 0.0f;
    if (n < tile_count && have_lse) {
      const int slot = tile_start + n;
      const float score = p.attention_scores[row_head * p.K + slot];
      if (score > -3.0e38f && finite_float(score)) {
        prob = expf(score - lse_val);
        if (!finite_float(prob)) {
          prob = 0.0f;
        }
      }
      if (prob != 0.0f) {
        atomicAdd(p.teacher_probs + static_cast<int64_t>(row_linear) * p.K + slot, prob);
      }
    }
    a_shared(0, n) = from_float<scalar_t>(prob);
  }
  for (int idx = tid; idx < kAttentionTileK * kValueDim; idx += blockDim.x) {
    const int n = idx / kValueDim;
    const int v = idx - n * kValueDim;
    scalar_t vv = from_float<scalar_t>(0.0f);
    if (n < tile_count && have_lse) {
      const int slot = tile_start + n;
      const float score = p.attention_scores[row_head * p.K + slot];
      const int32_t selected = load_topk(
          p.topk_out, static_cast<int64_t>(row_linear) * p.K + slot, p.topk_dtype);
      if (score > -3.0e38f && finite_float(score) && selected >= 0 && selected < p.S) {
        const int64_t v_base =
            static_cast<int64_t>(selected) * p.value_stride_s
            + static_cast<int64_t>(batch) * p.value_stride_b
            + static_cast<int64_t>(head) * p.value_stride_h;
        vv = value[v_base + static_cast<int64_t>(v) * p.value_stride_v];
      }
    }
    b_shared(n, v) = vv;
  }
  for (int v = tid; v < kValueDim; v += blockDim.x) {
    c_shared(0, v) = 0.0f;
  }
  __syncthreads();
  VALUE_GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
  __syncthreads();
  for (int v = tid; v < kValueDim; v += blockDim.x) {
    atomicAdd(p.output_accum + row_head * kValueDim + v, c_shared(0, v));
  }
}

template <class SCORE_NOPE, class SCORE_PE, typename scalar_t>
__device__ void attention_score_headblock_tile_task(
    const HisaDsaPersistentParams& p,
    int row_linear,
    int head_block,
    int tile_id,
    unsigned char* smem_raw) {
  const int tid = threadIdx.x;
  const int batch = row_linear / p.Q;
  const int row = row_linear - batch * p.Q;
  const int head_start = head_block * kAttentionHeadsPerCTA;
  const int tile_start = tile_id * kAttentionTileK;
  const int tile_count = min(kAttentionTileK, p.K - tile_start);
  if (tile_count <= 0) {
    return;
  }

  const scalar_t* query_nope = reinterpret_cast<const scalar_t*>(p.query_nope);
  const scalar_t* query_pe = reinterpret_cast<const scalar_t*>(p.query_pe);
  const scalar_t* key_nope = reinterpret_cast<const scalar_t*>(p.key_nope);
  const scalar_t* key_pe = reinterpret_cast<const scalar_t*>(p.key_pe);

  const int64_t row_batch = static_cast<int64_t>(row_linear);
  const int64_t q_abs =
      p.has_positions ? p.query_pos[row] : static_cast<int64_t>(p.q_start + row);

  {
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<SCORE_NOPE>(smem_raw);
    auto a_shared = cublasdx::make_tensor(smem_a, SCORE_NOPE::get_layout_smem_a());
    auto b_shared = cublasdx::make_tensor(smem_b, SCORE_NOPE::get_layout_smem_b());
    auto c_shared = cublasdx::make_tensor(smem_c, SCORE_NOPE::get_layout_smem_c());
    for (int idx = tid; idx < kAttentionHeadsPerCTA * kAttentionNopeGroupK; idx += blockDim.x) {
      const int local_head = idx / kAttentionNopeGroupK;
      const int kk = idx - local_head * kAttentionNopeGroupK;
      const int segment = kk / kNopeDim;
      const int d = kk - segment * kNopeDim;
      const int head = head_start + local_head;
      scalar_t qv = from_float<scalar_t>(0.0f);
      if (segment == local_head && head < p.H) {
        const int64_t q_base =
            static_cast<int64_t>(row) * p.query_nope_stride_s
            + static_cast<int64_t>(batch) * p.query_nope_stride_b
            + static_cast<int64_t>(head) * p.query_nope_stride_h;
        qv = query_nope[q_base + static_cast<int64_t>(d) * p.query_nope_stride_d];
      }
      a_shared(local_head, kk) = qv;
    }
    for (int idx = tid; idx < kAttentionNopeGroupK * kAttentionTileK; idx += blockDim.x) {
      const int kk = idx / kAttentionTileK;
      const int n = idx - kk * kAttentionTileK;
      const int segment = kk / kNopeDim;
      const int d = kk - segment * kNopeDim;
      const int head = head_start + segment;
      scalar_t kv = from_float<scalar_t>(0.0f);
      if (head < p.H && n < tile_count) {
        const int slot = tile_start + n;
        const int32_t selected = load_topk(p.topk_out, row_batch * p.K + slot, p.topk_dtype);
        const int safe_selected = selected > 0 ? selected : 0;
        bool valid = selected >= 0 && selected < p.S;
        if (valid) {
          const int64_t selected_abs = p.has_positions ? p.key_pos[safe_selected] : selected;
          valid = selected_abs <= q_abs;
        }
        if (valid) {
          const int64_t k_base =
              static_cast<int64_t>(safe_selected) * p.key_nope_stride_s
              + static_cast<int64_t>(batch) * p.key_nope_stride_b
              + static_cast<int64_t>(head) * p.key_nope_stride_h;
          kv = key_nope[k_base + static_cast<int64_t>(d) * p.key_nope_stride_d];
        }
      }
      b_shared(kk, n) = kv;
    }
    for (int idx = tid; idx < kAttentionHeadsPerCTA * kAttentionTileK; idx += blockDim.x) {
      const int local_head = idx / kAttentionTileK;
      const int n = idx - local_head * kAttentionTileK;
      c_shared(local_head, n) = 0.0f;
    }
    __syncthreads();
    SCORE_NOPE().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();
    for (int idx = tid; idx < kAttentionHeadsPerCTA * tile_count; idx += blockDim.x) {
      const int local_head = idx / tile_count;
      const int n = idx - local_head * tile_count;
      const int head = head_start + local_head;
      if (head < p.H) {
        p.attention_scores[(row_batch * p.H + head) * p.K + tile_start + n] =
            c_shared(local_head, n);
      }
    }
    __syncthreads();
  }

  {
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<SCORE_PE>(smem_raw);
    auto a_shared = cublasdx::make_tensor(smem_a, SCORE_PE::get_layout_smem_a());
    auto b_shared = cublasdx::make_tensor(smem_b, SCORE_PE::get_layout_smem_b());
    auto c_shared = cublasdx::make_tensor(smem_c, SCORE_PE::get_layout_smem_c());
    for (int idx = tid; idx < kAttentionHeadsPerCTA * kAttentionPeGroupK; idx += blockDim.x) {
      const int local_head = idx / kAttentionPeGroupK;
      const int kk = idx - local_head * kAttentionPeGroupK;
      const int segment = kk / kPeDim;
      const int d = kk - segment * kPeDim;
      const int head = head_start + local_head;
      scalar_t qv = from_float<scalar_t>(0.0f);
      if (segment == local_head && head < p.H) {
        const int64_t q_base =
            ((static_cast<int64_t>(row) * p.B + batch) * p.H + head) * kPeDim;
        qv = query_pe[q_base + d];
      }
      a_shared(local_head, kk) = qv;
    }
    for (int idx = tid; idx < kAttentionPeGroupK * kAttentionTileK; idx += blockDim.x) {
      const int kk = idx / kAttentionTileK;
      const int n = idx - kk * kAttentionTileK;
      const int segment = kk / kPeDim;
      const int d = kk - segment * kPeDim;
      const int head = head_start + segment;
      scalar_t kv = from_float<scalar_t>(0.0f);
      if (head < p.H && n < tile_count) {
        const int slot = tile_start + n;
        const int32_t selected = load_topk(p.topk_out, row_batch * p.K + slot, p.topk_dtype);
        const int safe_selected = selected > 0 ? selected : 0;
        bool valid = selected >= 0 && selected < p.S;
        if (valid) {
          const int64_t selected_abs = p.has_positions ? p.key_pos[safe_selected] : selected;
          valid = selected_abs <= q_abs;
        }
        if (valid) {
          const int pe_head = min(head, p.KPH - 1);
          const int64_t k_base =
              ((static_cast<int64_t>(safe_selected) * p.B + batch) * p.KPH + pe_head) * kPeDim;
          kv = key_pe[k_base + d];
        }
      }
      b_shared(kk, n) = kv;
    }
    for (int idx = tid; idx < kAttentionHeadsPerCTA * kAttentionTileK; idx += blockDim.x) {
      const int local_head = idx / kAttentionTileK;
      const int n = idx - local_head * kAttentionTileK;
      c_shared(local_head, n) = 0.0f;
    }
    __syncthreads();
    SCORE_PE().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();
    for (int idx = tid; idx < kAttentionHeadsPerCTA * tile_count; idx += blockDim.x) {
      const int local_head = idx / tile_count;
      const int n = idx - local_head * tile_count;
      const int head = head_start + local_head;
      if (head < p.H) {
        const int slot = tile_start + n;
        const int32_t selected = load_topk(p.topk_out, row_batch * p.K + slot, p.topk_dtype);
        const int safe_selected = selected > 0 ? selected : 0;
        bool valid = selected >= 0 && selected < p.S;
        if (valid) {
          const int64_t selected_abs = p.has_positions ? p.key_pos[safe_selected] : selected;
          valid = selected_abs <= q_abs;
        }
        const int64_t off = (row_batch * p.H + head) * p.K + slot;
        const float score =
            valid ? (p.attention_scores[off] + c_shared(local_head, n)) * p.softmax_scale
                  : -FLT_MAX;
        p.attention_scores[off] = finite_float(score) ? score : -FLT_MAX;
      }
    }
  }
}

template <class VALUE_GEMM, typename scalar_t>
__device__ void attention_value_headblock_tile_task(
    const HisaDsaPersistentParams& p,
    int row_linear,
    int head_block,
    int tile_id,
    unsigned char* smem_raw) {
  const int tid = threadIdx.x;
  const int batch = row_linear / p.Q;
  const int head_start = head_block * kAttentionHeadsPerCTA;
  const int tile_start = tile_id * kAttentionTileK;
  const int tile_count = min(kAttentionTileK, p.K - tile_start);
  if (tile_count <= 0) {
    return;
  }

  const scalar_t* value = reinterpret_cast<const scalar_t*>(p.value);
  auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<VALUE_GEMM>(smem_raw);
  auto a_shared = cublasdx::make_tensor(smem_a, VALUE_GEMM::get_layout_smem_a());
  auto b_shared = cublasdx::make_tensor(smem_b, VALUE_GEMM::get_layout_smem_b());
  auto c_shared = cublasdx::make_tensor(smem_c, VALUE_GEMM::get_layout_smem_c());

  const int64_t row_batch = static_cast<int64_t>(row_linear);
  for (int idx = tid; idx < kAttentionHeadsPerCTA * kAttentionValueGroupK; idx += blockDim.x) {
    const int local_head = idx / kAttentionValueGroupK;
    const int kk = idx - local_head * kAttentionValueGroupK;
    const int segment = kk / kAttentionTileK;
    const int n = kk - segment * kAttentionTileK;
    const int head = head_start + local_head;
    float prob = 0.0f;
    if (segment == local_head && head < p.H && n < tile_count) {
      const int slot = tile_start + n;
      const int64_t row_head = row_batch * p.H + head;
      const float lse_val = p.lse[row_head];
      const float score = p.attention_scores[row_head * p.K + slot];
      if (lse_val > -3.0e38f && score > -3.0e38f && finite_float(lse_val) &&
          finite_float(score)) {
        prob = expf(score - lse_val);
        if (!finite_float(prob)) {
          prob = 0.0f;
        }
      }
      if (prob != 0.0f) {
        atomicAdd(p.teacher_probs + row_batch * p.K + slot, prob);
      }
    }
    a_shared(local_head, kk) = from_float<scalar_t>(prob);
  }
  for (int idx = tid; idx < kAttentionValueGroupK * kValueDim; idx += blockDim.x) {
    const int kk = idx / kValueDim;
    const int v = idx - kk * kValueDim;
    const int segment = kk / kAttentionTileK;
    const int n = kk - segment * kAttentionTileK;
    const int head = head_start + segment;
    scalar_t vv = from_float<scalar_t>(0.0f);
    if (head < p.H && n < tile_count) {
      const int slot = tile_start + n;
      const int64_t row_head = row_batch * p.H + head;
      const float score = p.attention_scores[row_head * p.K + slot];
      const int32_t selected = load_topk(p.topk_out, row_batch * p.K + slot, p.topk_dtype);
      if (score > -3.0e38f && finite_float(score) && selected >= 0 && selected < p.S) {
        const int64_t v_base =
            static_cast<int64_t>(selected) * p.value_stride_s
            + static_cast<int64_t>(batch) * p.value_stride_b
            + static_cast<int64_t>(head) * p.value_stride_h;
        vv = value[v_base + static_cast<int64_t>(v) * p.value_stride_v];
      }
    }
    b_shared(kk, v) = vv;
  }
  for (int idx = tid; idx < kAttentionHeadsPerCTA * kValueDim; idx += blockDim.x) {
    const int local_head = idx / kValueDim;
    const int v = idx - local_head * kValueDim;
    c_shared(local_head, v) = 0.0f;
  }
  __syncthreads();
  VALUE_GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
  __syncthreads();
  for (int idx = tid; idx < kAttentionHeadsPerCTA * kValueDim; idx += blockDim.x) {
    const int local_head = idx / kValueDim;
    const int v = idx - local_head * kValueDim;
    const int head = head_start + local_head;
    if (head < p.H) {
      const int64_t row_head = row_batch * p.H + head;
      atomicAdd(p.output_accum + row_head * kValueDim + v, c_shared(local_head, v));
    }
  }
}

__device__ void attention_finalize_task(const HisaDsaPersistentParams& p, int row_head) {
  const int tid = threadIdx.x;
  const int row_linear = row_head / p.H;
  const int head = row_head - row_linear * p.H;
  const int batch = row_linear / p.Q;
  const int row = row_linear - batch * p.Q;
  const int64_t out_base =
      ((static_cast<int64_t>(row) * p.B + batch) * p.H + head) * kValueDim;
  for (int v = tid; v < kValueDim; v += blockDim.x) {
    store_typed(p.output, out_base + v, p.output_accum[static_cast<int64_t>(row_head) * kValueDim + v], p.scalar_dtype);
  }
}

template <
    class GEMM,
    class SCORE_NOPE,
    class SCORE_PE,
    class VALUE_GEMM,
    class SCORE_NOPE_GROUP,
    class SCORE_PE_GROUP,
    class VALUE_GEMM_GROUP,
    typename scalar_t,
    typename weight_t>
__global__ void hisa_dsa_split_qk_persistent_fwd_kernel(HisaDsaPersistentParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  cg::grid_group grid = cg::this_grid();
  const int row_count = p.Q * p.B;

  for (int row_linear = blockIdx.x; row_linear < row_count; row_linear += gridDim.x) {
    select_blocks_task<GEMM, scalar_t, weight_t>(p, row_linear, smem_raw);
  }
  grid.sync();

  const int candidate_tasks = row_count * p.candidate_tile_count;
  for (int task = blockIdx.x; task < candidate_tasks; task += gridDim.x) {
    const int row_linear = task / p.candidate_tile_count;
    const int tile_id = task - row_linear * p.candidate_tile_count;
    score_candidates_task<GEMM, scalar_t, weight_t>(p, row_linear, tile_id, smem_raw);
  }
  grid.sync();

  const int sort_tasks = row_count * p.sort_chunk_count;
  for (int task = blockIdx.x; task < sort_tasks; task += gridDim.x) {
    const int row_linear = task / p.sort_chunk_count;
    const int sort_chunk = task - row_linear * p.sort_chunk_count;
    sort_candidates_task(p, row_linear, sort_chunk, smem_raw);
  }
  grid.sync();

  for (int row_linear = blockIdx.x; row_linear < row_count; row_linear += gridDim.x) {
    merge_topk_task(p, row_linear, smem_raw);
  }
  grid.sync();

  const int row_head_count = row_count * p.H;
  for (int row_head = blockIdx.x; row_head < row_head_count; row_head += gridDim.x) {
    attention_init_task(p, row_head);
  }
  grid.sync();

  const int attention_score_tasks =
      row_count * p.attention_head_block_count * p.attention_tile_count;
  for (int task = blockIdx.x; task < attention_score_tasks; task += gridDim.x) {
    const int tile_id = task % p.attention_tile_count;
    const int row_head_block = task / p.attention_tile_count;
    const int head_block = row_head_block % p.attention_head_block_count;
    const int row_linear = row_head_block / p.attention_head_block_count;
    attention_score_headblock_tile_task<SCORE_NOPE_GROUP, SCORE_PE_GROUP, scalar_t>(
        p, row_linear, head_block, tile_id, smem_raw);
  }
  grid.sync();

  for (int row_head = blockIdx.x; row_head < row_head_count; row_head += gridDim.x) {
    attention_lse_task(p, row_head, smem_raw);
  }
  grid.sync();

  const int attention_value_tasks =
      row_count * p.attention_head_block_count * p.attention_tile_count;
  for (int task = blockIdx.x; task < attention_value_tasks; task += gridDim.x) {
    const int tile_id = task % p.attention_tile_count;
    const int row_head_block = task / p.attention_tile_count;
    const int head_block = row_head_block % p.attention_head_block_count;
    const int row_linear = row_head_block / p.attention_head_block_count;
    attention_value_headblock_tile_task<VALUE_GEMM_GROUP, scalar_t>(
        p, row_linear, head_block, tile_id, smem_raw);
  }
  grid.sync();

  for (int row_head = blockIdx.x; row_head < row_head_count; row_head += gridDim.x) {
    attention_finalize_task(p, row_head);
  }
}

template <
    class GEMM,
    class SCORE_NOPE,
    class SCORE_PE,
    class VALUE_GEMM,
    class SCORE_NOPE_GROUP,
    class SCORE_PE_GROUP,
    class VALUE_GEMM_GROUP,
    typename scalar_t,
    typename weight_t>
void launch_hisa_dsa_split_qk_persistent_fwd_typed(
    HisaDsaPersistentParams p,
    cudaStream_t stream) {
  using Sort = cub::BlockRadixSort<uint64_t, kSortThreads, kSortItemsPerThread>;
  const size_t gemm_smem = cublasdx::get_shared_storage_size<GEMM>();
  const size_t score_attn_smem = std::max(
      cublasdx::get_shared_storage_size<SCORE_NOPE>(),
      cublasdx::get_shared_storage_size<SCORE_PE>());
  const size_t value_attn_smem = cublasdx::get_shared_storage_size<VALUE_GEMM>();
  const size_t score_group_attn_smem = std::max(
      cublasdx::get_shared_storage_size<SCORE_NOPE_GROUP>(),
      cublasdx::get_shared_storage_size<SCORE_PE_GROUP>());
  const size_t value_group_attn_smem = cublasdx::get_shared_storage_size<VALUE_GEMM_GROUP>();
  const size_t select_smem =
      sizeof(float) * static_cast<size_t>(p.block_score_capacity)
      + sizeof(int) * static_cast<size_t>(p.block_score_capacity)
      + 16
      + gemm_smem;
  const size_t score_smem = gemm_smem;
  const size_t sort_smem = sizeof(typename Sort::TempStorage);
  const size_t merge_smem =
      sizeof(uint64_t) * static_cast<size_t>(p.K) * static_cast<size_t>(p.sort_chunk_count);
  const size_t lse_smem = sizeof(float) * kThreads;
  const size_t smem_bytes =
      std::max(
          std::max(std::max(select_smem, score_smem), std::max(sort_smem, merge_smem)),
          std::max(
              std::max(std::max(score_attn_smem, value_attn_smem), lse_smem),
              std::max(score_group_attn_smem, value_group_attn_smem)));

  auto kernel =
      hisa_dsa_split_qk_persistent_fwd_kernel<
          GEMM,
          SCORE_NOPE,
          SCORE_PE,
          VALUE_GEMM,
          SCORE_NOPE_GROUP,
          SCORE_PE_GROUP,
          VALUE_GEMM_GROUP,
          scalar_t,
          weight_t>;
  if (smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }

  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int cooperative = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(cooperative, "persistent HISA/DSA forward requires cooperative launch support");

  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  int active_blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm,
      kernel,
      kThreads,
      smem_bytes));
  TORCH_CHECK(active_blocks_per_sm > 0, "persistent HISA/DSA forward has zero CTA occupancy");

  const int row_count = p.Q * p.B;
  const int row_head_count = row_count * p.H;
  const int max_task_count = std::max(
      std::max(row_count, row_count * p.candidate_tile_count),
      std::max(
          row_count * p.sort_chunk_count,
          row_count * p.attention_head_block_count * p.attention_tile_count));
  const int max_blocks = sm_count * active_blocks_per_sm;
  const int blocks = std::max(1, std::min(max_blocks, max_task_count));
  void* args[] = {&p};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(kernel),
      dim3(blocks),
      dim3(kThreads),
      args,
      smem_bytes,
      stream));
}

template <
    class GEMM,
    class SCORE_NOPE,
    class SCORE_PE,
    class VALUE_GEMM,
    class SCORE_NOPE_GROUP,
    class SCORE_PE_GROUP,
    class VALUE_GEMM_GROUP,
    typename scalar_t>
void dispatch_persistent_weight(
    HisaDsaPersistentParams p,
    int weight_dtype,
    cudaStream_t stream) {
  if (weight_dtype == kDTypeF32) {
    launch_hisa_dsa_split_qk_persistent_fwd_typed<
        GEMM,
        SCORE_NOPE,
        SCORE_PE,
        VALUE_GEMM,
        SCORE_NOPE_GROUP,
        SCORE_PE_GROUP,
        VALUE_GEMM_GROUP,
        scalar_t,
        float>(p, stream);
  } else if (weight_dtype == kDTypeBF16) {
    launch_hisa_dsa_split_qk_persistent_fwd_typed<
        GEMM,
        SCORE_NOPE,
        SCORE_PE,
        VALUE_GEMM,
        SCORE_NOPE_GROUP,
        SCORE_PE_GROUP,
        VALUE_GEMM_GROUP,
        scalar_t,
        __nv_bfloat16>(p, stream);
  } else if (weight_dtype == kDTypeF16) {
    launch_hisa_dsa_split_qk_persistent_fwd_typed<
        GEMM,
        SCORE_NOPE,
        SCORE_PE,
        VALUE_GEMM,
        SCORE_NOPE_GROUP,
        SCORE_PE_GROUP,
        VALUE_GEMM_GROUP,
        scalar_t,
        __half>(p, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported persistent HISA/DSA weight dtype");
  }
}

}  // namespace

void launch_hisa_dsa_split_qk_persistent_fwd(
    const void* q_indexer,
    const void* weights,
    const void* k_indexer,
    const void* block_reps,
    const void* prefix_lens,
    const void* query_nope,
    const void* query_pe,
    const void* key_nope,
    const void* key_pe,
    const void* value,
    const int64_t* query_pos,
    const int64_t* key_pos,
    void* topk_out,
    float* selected_scores,
    void* output,
    float* lse,
    float* teacher_probs,
    int32_t* selected_blocks,
    int64_t* candidate_keys,
    float* attention_scores,
    float* output_accum,
    int Q,
    int B,
    int S,
    int IH,
    int ID,
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
    int weight_dtype,
    int prefix_dtype,
    int topk_dtype,
    int has_positions,
    int prefix_lens_shared,
    int force_first,
    int force_last,
    int force_last_minus_one,
    cudaStream_t stream) {
  TORCH_CHECK(IH == kIndexerHeads && ID == kIndexerDim,
              "persistent HISA/DSA requires indexer shape [Q, B, 64, 128]");
  TORCH_CHECK(Q > 0 && B > 0 && S > 0, "persistent HISA/DSA got empty sequence/batch");
  TORCH_CHECK(H > 0 && H <= 64, "persistent HISA/DSA supports H in (0, 64]");
  TORCH_CHECK(D == kNopeDim && P == kPeDim && V == kValueDim,
              "persistent HISA/DSA tiled attention requires noPE=128, PE=64, V=128");
  TORCH_CHECK(block_size > 0 && effective_block_topk > 0, "invalid HISA block config");
  TORCH_CHECK(K > 0 && K <= 2048, "persistent HISA/DSA supports topk in (0, 2048]");
  const int candidate_capacity = effective_block_topk * block_size;
  TORCH_CHECK(candidate_capacity >= K, "persistent HISA/DSA candidate capacity must cover topk");
  TORCH_CHECK(candidate_capacity <= 16384,
              "persistent HISA/DSA currently supports candidate capacity <= 16384");
  const int candidate_tile_count = (candidate_capacity + kTileN - 1) / kTileN;
  const int sort_chunk_count = (candidate_capacity + kSortWidth - 1) / kSortWidth;
  const int attention_tile_count = (K + kAttentionTileK - 1) / kAttentionTileK;
  const int attention_head_block_count = (H + kAttentionHeadsPerCTA - 1) / kAttentionHeadsPerCTA;
  const int merge_capacity = K * sort_chunk_count;
  TORCH_CHECK((merge_capacity & (merge_capacity - 1)) == 0,
              "persistent HISA/DSA top-k merge capacity must be power of two");

  HisaDsaPersistentParams p{};
  p.q_indexer = q_indexer;
  p.weights = weights;
  p.k_indexer = k_indexer;
  p.block_reps = block_reps;
  p.prefix_lens = prefix_lens;
  p.query_nope = query_nope;
  p.query_pe = query_pe;
  p.key_nope = key_nope;
  p.key_pe = key_pe;
  p.value = value;
  p.query_pos = query_pos;
  p.key_pos = key_pos;
  p.topk_out = topk_out;
  p.selected_scores_out = selected_scores;
  p.output = output;
  p.lse = lse;
  p.teacher_probs = teacher_probs;
  p.selected_blocks = selected_blocks;
  p.candidate_keys = reinterpret_cast<uint64_t*>(candidate_keys);
  p.attention_scores = attention_scores;
  p.output_accum = output_accum;
  p.Q = Q;
  p.B = B;
  p.S = S;
  p.H = H;
  p.D = D;
  p.P = P;
  p.KPH = KPH;
  p.V = V;
  p.MB = MB;
  p.block_size = block_size;
  p.block_topk = block_topk;
  p.compression_ratio = compression_ratio;
  p.effective_block_topk = effective_block_topk;
  p.K = K;
  p.q_start = q_start;
  p.block_score_capacity = next_power_of_two_int(std::max(1, MB));
  p.candidate_capacity = candidate_capacity;
  p.candidate_tile_count = candidate_tile_count;
  p.sort_chunk_count = sort_chunk_count;
  p.attention_tile_count = attention_tile_count;
  p.attention_head_block_count = attention_head_block_count;
  p.query_nope_stride_s = query_nope_stride_s;
  p.query_nope_stride_b = query_nope_stride_b;
  p.query_nope_stride_h = query_nope_stride_h;
  p.query_nope_stride_d = query_nope_stride_d;
  p.key_nope_stride_s = key_nope_stride_s;
  p.key_nope_stride_b = key_nope_stride_b;
  p.key_nope_stride_h = key_nope_stride_h;
  p.key_nope_stride_d = key_nope_stride_d;
  p.value_stride_s = value_stride_s;
  p.value_stride_b = value_stride_b;
  p.value_stride_h = value_stride_h;
  p.value_stride_v = value_stride_v;
  p.softmax_scale = softmax_scale;
  p.scalar_dtype = scalar_dtype;
  p.prefix_dtype = prefix_dtype;
  p.topk_dtype = topk_dtype;
  p.has_positions = has_positions;
  p.prefix_lens_shared = prefix_lens_shared;
  p.force_first = force_first;
  p.force_last = force_last;
  p.force_last_minus_one = force_last_minus_one;

  if (scalar_dtype == kDTypeF32) {
    dispatch_persistent_weight<
        HisaDsaPersistentGemmF32,
        DsaScoreNopePersistentGemmF32,
        DsaScorePePersistentGemmF32,
        DsaValuePersistentGemmF32,
        DsaScoreNopeGroupPersistentGemmF32,
        DsaScorePeGroupPersistentGemmF32,
        DsaValueGroupPersistentGemmF32,
        float>(p, weight_dtype, stream);
  } else if (scalar_dtype == kDTypeBF16) {
    dispatch_persistent_weight<
        HisaDsaPersistentGemmBF16,
        DsaScoreNopePersistentGemmBF16,
        DsaScorePePersistentGemmBF16,
        DsaValuePersistentGemmBF16,
        DsaScoreNopeGroupPersistentGemmBF16,
        DsaScorePeGroupPersistentGemmBF16,
        DsaValueGroupPersistentGemmBF16,
        __nv_bfloat16>(p, weight_dtype, stream);
  } else if (scalar_dtype == kDTypeF16) {
    dispatch_persistent_weight<
        HisaDsaPersistentGemmF16,
        DsaScoreNopePersistentGemmF16,
        DsaScorePePersistentGemmF16,
        DsaValuePersistentGemmF16,
        DsaScoreNopeGroupPersistentGemmF16,
        DsaScorePeGroupPersistentGemmF16,
        DsaValueGroupPersistentGemmF16,
        __half>(p, weight_dtype, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported persistent HISA/DSA scalar dtype");
  }
}

}  // namespace hisa_indexer
}  // namespace megatron
