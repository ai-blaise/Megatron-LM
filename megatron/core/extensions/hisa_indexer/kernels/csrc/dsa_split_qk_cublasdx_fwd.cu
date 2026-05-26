// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// cuBLASDx selected split-QK DSA forward.
//
// This is the first stage toward the fused HISA/DSA megakernel.  It consumes
// HISA-selected token ids directly and replaces the scalar row-dot forward with
// block-level MMA tiles for the real training shape:
//   local heads <= 64, noPE=128, PE=64, V=128, selected top-k tiled by 64.
//
// The noPE key is head-specific, so the work is intentionally grouped by
// attention head instead of pretending the selected key tile is shared across
// heads.  A later selector+attention fusion can reuse this head-owned score and
// value tile structure while removing the global top-k handoff.

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

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace megatron {
namespace hisa_indexer {

namespace {

constexpr int kNopeDim = 128;
constexpr int kPeDim = 64;
constexpr int kValueDim = 128;
constexpr int kTileK = 64;
constexpr int kDTypeFloat32 = 0;
constexpr int kDTypeBFloat16 = 1;
constexpr int kDTypeFloat16 = 2;
constexpr int kTopkInt16 = 0;
constexpr int kTopkInt32 = 1;

using ScoreNopeGemmF32 = decltype(
    cublasdx::Size<1, kTileK, kNopeDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using ScorePeGemmF32 = decltype(
    cublasdx::Size<1, kTileK, kPeDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using ValueGemmF32 = decltype(
    cublasdx::Size<1, kValueDim, kTileK>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using ScoreNopeGemmBF16 = decltype(
    cublasdx::Size<1, kTileK, kNopeDim>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using ScorePeGemmBF16 = decltype(
    cublasdx::Size<1, kTileK, kPeDim>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using ValueGemmBF16 = decltype(
    cublasdx::Size<1, kValueDim, kTileK>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using ScoreNopeGemmF16 = decltype(
    cublasdx::Size<1, kTileK, kNopeDim>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using ScorePeGemmF16 = decltype(
    cublasdx::Size<1, kTileK, kPeDim>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using ValueGemmF16 = decltype(
    cublasdx::Size<1, kValueDim, kTileK>()
    + cublasdx::Precision<__half, __half, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using SharedPeGemmF32 = decltype(
    cublasdx::Size<64, kTileK, kPeDim>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using SharedPeGemmBF16 = decltype(
    cublasdx::Size<64, kTileK, kPeDim>()
    + cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
    + cublasdx::SM<1000>()
    + cublasdx::BlockDim<128>()
    + cublasdx::Block());

using SharedPeGemmF16 = decltype(
    cublasdx::Size<64, kTileK, kPeDim>()
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

__device__ __forceinline__ int32_t load_topk_index(
    const void* ptr, int64_t idx, int dtype) {
  if (dtype == kTopkInt16) {
    return static_cast<int32_t>(reinterpret_cast<const int16_t*>(ptr)[idx]);
  }
  if (dtype == kTopkInt32) {
    return reinterpret_cast<const int32_t*>(ptr)[idx];
  }
  return static_cast<int32_t>(reinterpret_cast<const int64_t*>(ptr)[idx]);
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

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
  }
  return v;
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

template <class SCORE_NOPE, class SCORE_PE, class VALUE_GEMM, typename scalar_t>
__global__ void dsa_split_qk_fwd_cublasdx_kernel(
    const scalar_t* __restrict__ query_nope,      // [Q, B, H, 128], strided
    const scalar_t* __restrict__ query_pe,        // [Q, B, H, 64], contiguous
    const scalar_t* __restrict__ key_nope,        // [S, B, H, 128], strided
    const scalar_t* __restrict__ key_pe,          // [S, B, KPH, 64], contiguous
    const scalar_t* __restrict__ value,           // [S, B, H, 128], strided
    const void* __restrict__ topk,                // [B, Q, K]
    const int64_t* __restrict__ query_pos,        // [Q], optional
    const int64_t* __restrict__ key_pos,          // [S], optional
    scalar_t* __restrict__ output,                // [Q, B, H, 128]
    float* __restrict__ lse,                      // [B * Q, H]
    float* __restrict__ teacher_probs,            // [B * Q, K], optional
    float* __restrict__ teacher_score_scratch,    // [B * Q, H, K], optional
    int Q,
    int B,
    int S,
    int H,
    int KPH,
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
    int topk_dtype,
    int has_positions,
    int emit_teacher,
    int use_teacher_score_scratch) {
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  const int head = blockIdx.z;
  const int tid = threadIdx.x;
  if (row >= Q || batch >= B || head >= H) {
    return;
  }

  extern __shared__ __align__(16) unsigned char smem_raw[];
  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* score_s = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * K;
  float* out_s = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * kValueDim;
  cursor = align_dynamic_smem(cursor, 16);
  void* gemm_smem = reinterpret_cast<void*>(cursor);

  for (int i = tid; i < K; i += blockDim.x) {
    score_s[i] = -FLT_MAX;
  }
  __syncthreads();

  const int pe_head = min(head, KPH - 1);
  const int64_t row_batch = static_cast<int64_t>(batch) * Q + row;
  const int64_t q_abs = has_positions ? query_pos[row] : static_cast<int64_t>(q_start + row);
  const int64_t q_nope_base =
      static_cast<int64_t>(row) * query_nope_stride_s +
      static_cast<int64_t>(batch) * query_nope_stride_b +
      static_cast<int64_t>(head) * query_nope_stride_h;
  const int64_t q_pe_base =
      ((static_cast<int64_t>(row) * B + batch) * H + head) * kPeDim;

  for (int tile_start = 0; tile_start < K; tile_start += kTileK) {
    const int tile_count = min(kTileK, K - tile_start);

    {
      auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<SCORE_NOPE>(gemm_smem);
      auto a_shared = cublasdx::make_tensor(smem_a, SCORE_NOPE::get_layout_smem_a());
      auto b_shared = cublasdx::make_tensor(smem_b, SCORE_NOPE::get_layout_smem_b());
      auto c_shared = cublasdx::make_tensor(smem_c, SCORE_NOPE::get_layout_smem_c());
      for (int d = tid; d < kNopeDim; d += blockDim.x) {
        a_shared(0, d) = query_nope[q_nope_base + static_cast<int64_t>(d) * query_nope_stride_d];
      }
      for (int idx = tid; idx < kNopeDim * kTileK; idx += blockDim.x) {
        const int d = idx / kTileK;
        const int n = idx - d * kTileK;
        scalar_t kv = from_float<scalar_t>(0.0f);
        if (n < tile_count) {
          const int slot = tile_start + n;
          const int32_t selected = load_topk_index(topk, row_batch * K + slot, topk_dtype);
          const int safe_selected = selected > 0 ? selected : 0;
          bool valid = selected >= 0 && selected < S;
          if (valid) {
            const int64_t selected_abs = has_positions ? key_pos[safe_selected] : selected;
            valid = selected_abs <= q_abs;
          }
          if (valid) {
            const int64_t k_base =
                static_cast<int64_t>(safe_selected) * key_nope_stride_s +
                static_cast<int64_t>(batch) * key_nope_stride_b +
                static_cast<int64_t>(head) * key_nope_stride_h;
            kv = key_nope[k_base + static_cast<int64_t>(d) * key_nope_stride_d];
          }
        }
        b_shared(d, n) = kv;
      }
      for (int n = tid; n < kTileK; n += blockDim.x) {
        c_shared(0, n) = 0.0f;
      }
      __syncthreads();
      SCORE_NOPE().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
      __syncthreads();
      for (int n = tid; n < tile_count; n += blockDim.x) {
        score_s[tile_start + n] = c_shared(0, n);
      }
      __syncthreads();
    }

    {
      auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<SCORE_PE>(gemm_smem);
      auto a_shared = cublasdx::make_tensor(smem_a, SCORE_PE::get_layout_smem_a());
      auto b_shared = cublasdx::make_tensor(smem_b, SCORE_PE::get_layout_smem_b());
      auto c_shared = cublasdx::make_tensor(smem_c, SCORE_PE::get_layout_smem_c());
      for (int d = tid; d < kPeDim; d += blockDim.x) {
        a_shared(0, d) = query_pe[q_pe_base + d];
      }
      for (int idx = tid; idx < kPeDim * kTileK; idx += blockDim.x) {
        const int d = idx / kTileK;
        const int n = idx - d * kTileK;
        scalar_t kv = from_float<scalar_t>(0.0f);
        if (n < tile_count) {
          const int slot = tile_start + n;
          const int32_t selected = load_topk_index(topk, row_batch * K + slot, topk_dtype);
          const int safe_selected = selected > 0 ? selected : 0;
          bool valid = selected >= 0 && selected < S;
          if (valid) {
            const int64_t selected_abs = has_positions ? key_pos[safe_selected] : selected;
            valid = selected_abs <= q_abs;
          }
          if (valid) {
            const int64_t k_base =
                ((static_cast<int64_t>(safe_selected) * B + batch) * KPH + pe_head) * kPeDim;
            kv = key_pe[k_base + d];
          }
        }
        b_shared(d, n) = kv;
      }
      for (int n = tid; n < kTileK; n += blockDim.x) {
        c_shared(0, n) = 0.0f;
      }
      __syncthreads();
      SCORE_PE().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
      __syncthreads();
      for (int n = tid; n < tile_count; n += blockDim.x) {
        const int slot = tile_start + n;
        const int32_t selected = load_topk_index(topk, row_batch * K + slot, topk_dtype);
        const int safe_selected = selected > 0 ? selected : 0;
        bool valid = selected >= 0 && selected < S;
        if (valid) {
          const int64_t selected_abs = has_positions ? key_pos[safe_selected] : selected;
          valid = selected_abs <= q_abs;
        }
        float score = valid ? (score_s[slot] + c_shared(0, n)) * softmax_scale : -FLT_MAX;
        score_s[slot] = score;
        if (use_teacher_score_scratch) {
          teacher_score_scratch[(row_batch * H + head) * K + slot] = score;
        }
      }
      __syncthreads();
    }
  }

  float local_m = -FLT_MAX;
  for (int slot = tid; slot < K; slot += blockDim.x) {
    local_m = fmaxf(local_m, score_s[slot]);
  }
  const float m_i = block_reduce_max_128(local_m, out_s);
  float local_l = 0.0f;
  if (m_i > -3.0e38f) {
    for (int slot = tid; slot < K; slot += blockDim.x) {
      const float score = score_s[slot];
      if (score > -3.0e38f) {
        local_l += expf(score - m_i);
      }
    }
  }
  const float l_i = block_reduce_sum_128(local_l, out_s);
  if (tid == 0) {
    lse[row_batch * H + head] = l_i > 0.0f ? m_i + logf(l_i) : -FLT_MAX;
  }
  __syncthreads();
  for (int v = tid; v < kValueDim; v += blockDim.x) {
    out_s[v] = 0.0f;
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < K; tile_start += kTileK) {
    const int tile_count = min(kTileK, K - tile_start);
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<VALUE_GEMM>(gemm_smem);
    auto a_shared = cublasdx::make_tensor(smem_a, VALUE_GEMM::get_layout_smem_a());
    auto b_shared = cublasdx::make_tensor(smem_b, VALUE_GEMM::get_layout_smem_b());
    auto c_shared = cublasdx::make_tensor(smem_c, VALUE_GEMM::get_layout_smem_c());

    for (int n = tid; n < kTileK; n += blockDim.x) {
      float prob = 0.0f;
      if (n < tile_count && l_i > 0.0f) {
        const int slot = tile_start + n;
        const float score = score_s[slot];
        if (score > -3.0e38f) {
          prob = expf(score - m_i) / l_i;
        }
        if (emit_teacher && prob != 0.0f) {
          atomicAdd(teacher_probs + row_batch * K + slot, prob);
        }
      }
      a_shared(0, n) = from_float<scalar_t>(prob);
    }
    for (int idx = tid; idx < kTileK * kValueDim; idx += blockDim.x) {
      const int n = idx / kValueDim;
      const int v = idx - n * kValueDim;
      scalar_t vv = from_float<scalar_t>(0.0f);
      if (n < tile_count) {
        const int slot = tile_start + n;
        const int32_t selected = load_topk_index(topk, row_batch * K + slot, topk_dtype);
        const int safe_selected = selected > 0 ? selected : 0;
        bool valid = selected >= 0 && selected < S && score_s[slot] > -3.0e38f;
        if (valid) {
          const int64_t v_base =
              static_cast<int64_t>(safe_selected) * value_stride_s +
              static_cast<int64_t>(batch) * value_stride_b +
              static_cast<int64_t>(head) * value_stride_h;
          vv = value[v_base + static_cast<int64_t>(v) * value_stride_v];
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
      out_s[v] += c_shared(0, v);
    }
    __syncthreads();
  }

  const int64_t out_base =
      ((static_cast<int64_t>(row) * B + batch) * H + head) * kValueDim;
  for (int v = tid; v < kValueDim; v += blockDim.x) {
    output[out_base + v] = from_float<scalar_t>(out_s[v]);
  }
}

template <class PE_GEMM, typename scalar_t>
__global__ void dsa_split_qk_fwd_cublasdx_pe_shared_kernel(
    const scalar_t* __restrict__ query_nope,      // [Q, B, H, 128], strided
    const scalar_t* __restrict__ query_pe,        // [Q, B, H, 64], contiguous
    const scalar_t* __restrict__ key_nope,        // [S, B, H, 128], strided
    const scalar_t* __restrict__ key_pe,          // [S, B, 1, 64], contiguous
    const scalar_t* __restrict__ value,           // [S, B, H, 128], strided
    const void* __restrict__ topk,                // [B, Q, K]
    const int64_t* __restrict__ query_pos,        // [Q], optional
    const int64_t* __restrict__ key_pos,          // [S], optional
    scalar_t* __restrict__ output,                // [Q, B, H, 128]
    float* __restrict__ lse,                      // [B * Q, H]
    float* __restrict__ teacher_probs,            // [B * Q, K]
    float* __restrict__ teacher_score_scratch,    // [B * Q, H, K]
    int Q,
    int B,
    int S,
    int H,
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
    int topk_dtype,
    int has_positions) {
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  constexpr int kBlockWarps = 4;
  if (row >= Q || batch >= B) {
    return;
  }

  extern __shared__ __align__(16) unsigned char smem_raw[];
  char* cursor = reinterpret_cast<char*>(smem_raw);
  float* teacher_s = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * K;
  cursor = align_dynamic_smem(cursor, 16);
  float* q_nope_s = reinterpret_cast<float*>(cursor);
  cursor += sizeof(float) * static_cast<size_t>(H) * kNopeDim;
  cursor = align_dynamic_smem(cursor, 16);
  void* gemm_smem = reinterpret_cast<void*>(cursor);

  const int64_t row_batch = static_cast<int64_t>(batch) * Q + row;
  const int64_t q_abs = has_positions ? query_pos[row] : static_cast<int64_t>(q_start + row);

  for (int idx = tid; idx < H * kNopeDim; idx += blockDim.x) {
    const int head = idx / kNopeDim;
    const int d = idx - head * kNopeDim;
    const int64_t q_base =
        static_cast<int64_t>(row) * query_nope_stride_s +
        static_cast<int64_t>(batch) * query_nope_stride_b +
        static_cast<int64_t>(head) * query_nope_stride_h;
    q_nope_s[idx] = to_float(query_nope[q_base + static_cast<int64_t>(d) * query_nope_stride_d]);
  }
  for (int slot = tid; slot < K; slot += blockDim.x) {
    teacher_s[slot] = 0.0f;
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < K; tile_start += kTileK) {
    const int tile_count = min(kTileK, K - tile_start);
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<PE_GEMM>(gemm_smem);
    auto a_shared = cublasdx::make_tensor(smem_a, PE_GEMM::get_layout_smem_a());
    auto b_shared = cublasdx::make_tensor(smem_b, PE_GEMM::get_layout_smem_b());
    auto c_shared = cublasdx::make_tensor(smem_c, PE_GEMM::get_layout_smem_c());

    for (int idx = tid; idx < 64 * kPeDim; idx += blockDim.x) {
      const int head = idx / kPeDim;
      const int d = idx - head * kPeDim;
      scalar_t qv = from_float<scalar_t>(0.0f);
      if (head < H) {
        const int64_t q_base =
            ((static_cast<int64_t>(row) * B + batch) * H + head) * kPeDim;
        qv = query_pe[q_base + d];
      }
      a_shared(head, d) = qv;
    }
    for (int idx = tid; idx < kPeDim * kTileK; idx += blockDim.x) {
      const int d = idx / kTileK;
      const int n = idx - d * kTileK;
      scalar_t kv = from_float<scalar_t>(0.0f);
      if (n < tile_count) {
        const int slot = tile_start + n;
        const int32_t selected = load_topk_index(topk, row_batch * K + slot, topk_dtype);
        const int safe_selected = selected > 0 ? selected : 0;
        bool valid = selected >= 0 && selected < S;
        if (valid) {
          const int64_t selected_abs = has_positions ? key_pos[safe_selected] : selected;
          valid = selected_abs <= q_abs;
        }
        if (valid) {
          const int64_t k_base =
              ((static_cast<int64_t>(safe_selected) * B + batch) * 1) * kPeDim;
          kv = key_pe[k_base + d];
        }
      }
      b_shared(d, n) = kv;
    }
    for (int idx = tid; idx < 64 * kTileK; idx += blockDim.x) {
      const int head = idx / kTileK;
      const int n = idx - head * kTileK;
      c_shared(head, n) = 0.0f;
    }
    __syncthreads();
    PE_GEMM().execute(1.0f, a_shared, b_shared, 0.0f, c_shared);
    __syncthreads();

    for (int head = warp_id; head < H; head += kBlockWarps) {
      for (int n = 0; n < tile_count; ++n) {
        const int slot = tile_start + n;
        const int32_t selected = load_topk_index(topk, row_batch * K + slot, topk_dtype);
        const int safe_selected = selected > 0 ? selected : 0;
        bool valid = selected >= 0 && selected < S;
        if (valid) {
          const int64_t selected_abs = has_positions ? key_pos[safe_selected] : selected;
          valid = selected_abs <= q_abs;
        }
        float partial = 0.0f;
        if (valid) {
          const int64_t k_base =
              static_cast<int64_t>(safe_selected) * key_nope_stride_s +
              static_cast<int64_t>(batch) * key_nope_stride_b +
              static_cast<int64_t>(head) * key_nope_stride_h;
          for (int d = lane; d < kNopeDim; d += 32) {
            partial += q_nope_s[head * kNopeDim + d] *
                       to_float(key_nope[k_base + static_cast<int64_t>(d) * key_nope_stride_d]);
          }
        }
        const float nope_score = warp_sum(partial);
        if (lane == 0) {
          const float score = valid ? (nope_score + c_shared(head, n)) * softmax_scale : -FLT_MAX;
          teacher_score_scratch[(row_batch * H + head) * K + slot] = score;
        }
      }
    }
    __syncthreads();
  }

  __threadfence_block();
  __syncthreads();

  for (int head = warp_id; head < H; head += kBlockWarps) {
    float local_m = -FLT_MAX;
    for (int slot = lane; slot < K; slot += 32) {
      const float score = teacher_score_scratch[(row_batch * H + head) * K + slot];
      local_m = fmaxf(local_m, score);
    }
    const float m_reduce = warp_max(local_m);
    const float m_i = __shfl_sync(0xffffffff, m_reduce, 0);
    float local_l = 0.0f;
    if (m_i > -3.0e38f) {
      for (int slot = lane; slot < K; slot += 32) {
        const float score = teacher_score_scratch[(row_batch * H + head) * K + slot];
        if (score > -3.0e38f) {
          local_l += expf(score - m_i);
        }
      }
    }
    const float l_reduce = warp_sum(local_l);
    const float l_i = __shfl_sync(0xffffffff, l_reduce, 0);
    if (lane == 0) {
      lse[row_batch * H + head] = l_i > 0.0f ? m_i + logf(l_i) : -FLT_MAX;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const int v0 = lane;
    const int v1 = lane + 32;
    const int v2 = lane + 64;
    const int v3 = lane + 96;
    for (int slot = 0; slot < K; ++slot) {
      const float score = teacher_score_scratch[(row_batch * H + head) * K + slot];
      if (score <= -3.0e38f || l_i <= 0.0f) {
        continue;
      }
      const float prob = expf(score - m_i) / l_i;
      if (lane == 0) {
        atomicAdd(teacher_s + slot, prob);
      }
      const int32_t selected = load_topk_index(topk, row_batch * K + slot, topk_dtype);
      const int safe_selected = selected > 0 ? selected : 0;
      const int64_t v_base =
          static_cast<int64_t>(safe_selected) * value_stride_s +
          static_cast<int64_t>(batch) * value_stride_b +
          static_cast<int64_t>(head) * value_stride_h;
      acc0 += prob * to_float(value[v_base + static_cast<int64_t>(v0) * value_stride_v]);
      acc1 += prob * to_float(value[v_base + static_cast<int64_t>(v1) * value_stride_v]);
      acc2 += prob * to_float(value[v_base + static_cast<int64_t>(v2) * value_stride_v]);
      acc3 += prob * to_float(value[v_base + static_cast<int64_t>(v3) * value_stride_v]);
    }
    const int64_t out_base =
        ((static_cast<int64_t>(row) * B + batch) * H + head) * kValueDim;
    output[out_base + v0] = from_float<scalar_t>(acc0);
    output[out_base + v1] = from_float<scalar_t>(acc1);
    output[out_base + v2] = from_float<scalar_t>(acc2);
    output[out_base + v3] = from_float<scalar_t>(acc3);
  }
  __syncthreads();

  for (int slot = tid; slot < K; slot += blockDim.x) {
    teacher_probs[row_batch * K + slot] = teacher_s[slot];
  }
}

template <class SCORE_NOPE, class SCORE_PE, class VALUE_GEMM, typename scalar_t>
void launch_dsa_split_qk_fwd_cublasdx_typed(
    const void* query_nope, const void* query_pe, const void* key_nope,
    const void* key_pe, const void* value, const void* topk_indices,
    const int64_t* query_positions, const int64_t* key_positions,
    void* output, float* lse, float* teacher_probs,
    float* teacher_score_scratch, int q_len, int bsz, int sk,
    int num_heads, int key_pe_heads, int topk_count, int q_start,
    int64_t query_nope_stride_s, int64_t query_nope_stride_b,
    int64_t query_nope_stride_h, int64_t query_nope_stride_d,
    int64_t key_nope_stride_s, int64_t key_nope_stride_b,
    int64_t key_nope_stride_h, int64_t key_nope_stride_d,
    int64_t value_stride_s, int64_t value_stride_b, int64_t value_stride_h,
    int64_t value_stride_v, float softmax_scale, int topk_dtype,
    int has_positions, int emit_teacher, int use_teacher_score_scratch,
    cudaStream_t stream) {
  const dim3 grid(q_len, bsz, num_heads);
  const dim3 block(128);
  const size_t gemm_smem = std::max(
      std::max(
          cublasdx::get_shared_storage_size<SCORE_NOPE>(),
          cublasdx::get_shared_storage_size<SCORE_PE>()),
      cublasdx::get_shared_storage_size<VALUE_GEMM>());
  const size_t smem_bytes =
      sizeof(float) * static_cast<size_t>(topk_count)
      + sizeof(float) * static_cast<size_t>(kValueDim)
      + 16
      + gemm_smem;
  auto kernel =
      dsa_split_qk_fwd_cublasdx_kernel<SCORE_NOPE, SCORE_PE, VALUE_GEMM, scalar_t>;
  if (smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }
  if (emit_teacher) {
    C10_CUDA_CHECK(cudaMemsetAsync(
        teacher_probs,
        0,
        sizeof(float) * static_cast<size_t>(bsz) * q_len * topk_count,
        stream));
  }
  kernel<<<grid, block, smem_bytes, stream>>>(
      static_cast<const scalar_t*>(query_nope),
      static_cast<const scalar_t*>(query_pe),
      static_cast<const scalar_t*>(key_nope),
      static_cast<const scalar_t*>(key_pe),
      static_cast<const scalar_t*>(value),
      topk_indices,
      query_positions,
      key_positions,
      static_cast<scalar_t*>(output),
      lse,
      teacher_probs,
      teacher_score_scratch,
      q_len,
      bsz,
      sk,
      num_heads,
      key_pe_heads,
      topk_count,
      q_start,
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
      topk_dtype,
      has_positions,
      emit_teacher,
      use_teacher_score_scratch);
}

template <class PE_GEMM, typename scalar_t>
void launch_dsa_split_qk_fwd_cublasdx_pe_typed(
    const void* query_nope, const void* query_pe, const void* key_nope,
    const void* key_pe, const void* value, const void* topk_indices,
    const int64_t* query_positions, const int64_t* key_positions,
    void* output, float* lse, float* teacher_probs,
    float* teacher_score_scratch, int q_len, int bsz, int sk,
    int num_heads, int topk_count, int q_start,
    int64_t query_nope_stride_s, int64_t query_nope_stride_b,
    int64_t query_nope_stride_h, int64_t query_nope_stride_d,
    int64_t key_nope_stride_s, int64_t key_nope_stride_b,
    int64_t key_nope_stride_h, int64_t key_nope_stride_d,
    int64_t value_stride_s, int64_t value_stride_b, int64_t value_stride_h,
    int64_t value_stride_v, float softmax_scale, int topk_dtype,
    int has_positions, cudaStream_t stream) {
  const dim3 grid(q_len, bsz);
  const dim3 block(128);
  const size_t smem_bytes =
      sizeof(float) * static_cast<size_t>(topk_count)
      + sizeof(float) * static_cast<size_t>(num_heads) * kNopeDim
      + 32
      + cublasdx::get_shared_storage_size<PE_GEMM>();
  auto kernel = dsa_split_qk_fwd_cublasdx_pe_shared_kernel<PE_GEMM, scalar_t>;
  if (smem_bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }
  kernel<<<grid, block, smem_bytes, stream>>>(
      static_cast<const scalar_t*>(query_nope),
      static_cast<const scalar_t*>(query_pe),
      static_cast<const scalar_t*>(key_nope),
      static_cast<const scalar_t*>(key_pe),
      static_cast<const scalar_t*>(value),
      topk_indices,
      query_positions,
      key_positions,
      static_cast<scalar_t*>(output),
      lse,
      teacher_probs,
      teacher_score_scratch,
      q_len,
      bsz,
      sk,
      num_heads,
      topk_count,
      q_start,
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
      topk_dtype,
      has_positions);
}

}  // namespace

void launch_dsa_split_qk_fwd_cublasdx(
    const void* query_nope, const void* query_pe, const void* key_nope,
    const void* key_pe, const void* value, const void* topk_indices,
    const int64_t* query_positions, const int64_t* key_positions,
    void* output, float* lse, float* teacher_probs,
    float* teacher_score_scratch, int q_len, int bsz, int sk,
    int num_heads, int head_dim, int pos_dim, int key_pe_heads,
    int value_dim, int topk_count, int q_start,
    int64_t query_nope_stride_s, int64_t query_nope_stride_b,
    int64_t query_nope_stride_h, int64_t query_nope_stride_d,
    int64_t key_nope_stride_s, int64_t key_nope_stride_b,
    int64_t key_nope_stride_h, int64_t key_nope_stride_d,
    int64_t value_stride_s, int64_t value_stride_b, int64_t value_stride_h,
    int64_t value_stride_v, float softmax_scale, int scalar_dtype,
    int topk_dtype, int has_positions, int emit_teacher,
    int use_teacher_score_scratch, cudaStream_t stream) {
  if (head_dim != kNopeDim || pos_dim != kPeDim || value_dim != kValueDim) {
    C10_THROW_ERROR(
        ValueError,
        "cuBLASDx split-QK forward currently requires head_dim=128, pos_dim=64, value_dim=128");
  }
  if (num_heads <= 0 || num_heads > 64 || key_pe_heads <= 0 || key_pe_heads > num_heads) {
    C10_THROW_ERROR(ValueError, "cuBLASDx split-QK forward got invalid head counts");
  }
  if (topk_count <= 0 || topk_count > 4096) {
    C10_THROW_ERROR(ValueError, "cuBLASDx split-QK forward supports topk in (0, 4096]");
  }
  if (scalar_dtype == kDTypeFloat32) {
    launch_dsa_split_qk_fwd_cublasdx_typed<
        ScoreNopeGemmF32, ScorePeGemmF32, ValueGemmF32, float>(
        query_nope, query_pe, key_nope, key_pe, value, topk_indices,
        query_positions, key_positions, output, lse, teacher_probs,
        teacher_score_scratch, q_len, bsz, sk, num_heads, key_pe_heads,
        topk_count, q_start, query_nope_stride_s, query_nope_stride_b,
        query_nope_stride_h, query_nope_stride_d, key_nope_stride_s,
        key_nope_stride_b, key_nope_stride_h, key_nope_stride_d,
        value_stride_s, value_stride_b, value_stride_h, value_stride_v,
        softmax_scale, topk_dtype, has_positions, emit_teacher,
        use_teacher_score_scratch, stream);
  } else if (scalar_dtype == kDTypeBFloat16) {
    launch_dsa_split_qk_fwd_cublasdx_typed<
        ScoreNopeGemmBF16, ScorePeGemmBF16, ValueGemmBF16, __nv_bfloat16>(
        query_nope, query_pe, key_nope, key_pe, value, topk_indices,
        query_positions, key_positions, output, lse, teacher_probs,
        teacher_score_scratch, q_len, bsz, sk, num_heads, key_pe_heads,
        topk_count, q_start, query_nope_stride_s, query_nope_stride_b,
        query_nope_stride_h, query_nope_stride_d, key_nope_stride_s,
        key_nope_stride_b, key_nope_stride_h, key_nope_stride_d,
        value_stride_s, value_stride_b, value_stride_h, value_stride_v,
        softmax_scale, topk_dtype, has_positions, emit_teacher,
        use_teacher_score_scratch, stream);
  } else if (scalar_dtype == kDTypeFloat16) {
    launch_dsa_split_qk_fwd_cublasdx_typed<
        ScoreNopeGemmF16, ScorePeGemmF16, ValueGemmF16, __half>(
        query_nope, query_pe, key_nope, key_pe, value, topk_indices,
        query_positions, key_positions, output, lse, teacher_probs,
        teacher_score_scratch, q_len, bsz, sk, num_heads, key_pe_heads,
        topk_count, q_start, query_nope_stride_s, query_nope_stride_b,
        query_nope_stride_h, query_nope_stride_d, key_nope_stride_s,
        key_nope_stride_b, key_nope_stride_h, key_nope_stride_d,
        value_stride_s, value_stride_b, value_stride_h, value_stride_v,
        softmax_scale, topk_dtype, has_positions, emit_teacher,
        use_teacher_score_scratch, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported cuBLASDx split-QK forward dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_dsa_split_qk_fwd_cublasdx_pe(
    const void* query_nope, const void* query_pe, const void* key_nope,
    const void* key_pe, const void* value, const void* topk_indices,
    const int64_t* query_positions, const int64_t* key_positions,
    void* output, float* lse, float* teacher_probs,
    float* teacher_score_scratch, int q_len, int bsz, int sk,
    int num_heads, int head_dim, int pos_dim, int key_pe_heads,
    int value_dim, int topk_count, int q_start,
    int64_t query_nope_stride_s, int64_t query_nope_stride_b,
    int64_t query_nope_stride_h, int64_t query_nope_stride_d,
    int64_t key_nope_stride_s, int64_t key_nope_stride_b,
    int64_t key_nope_stride_h, int64_t key_nope_stride_d,
    int64_t value_stride_s, int64_t value_stride_b, int64_t value_stride_h,
    int64_t value_stride_v, float softmax_scale, int scalar_dtype,
    int topk_dtype, int has_positions, cudaStream_t stream) {
  if (head_dim != kNopeDim || pos_dim != kPeDim || value_dim != kValueDim) {
    C10_THROW_ERROR(
        ValueError,
        "PE-shared cuBLASDx split-QK forward requires head_dim=128, pos_dim=64, value_dim=128");
  }
  if (num_heads <= 0 || num_heads > 64 || key_pe_heads != 1) {
    C10_THROW_ERROR(
        ValueError,
        "PE-shared cuBLASDx split-QK forward requires 1..64 local heads and key_pe_heads=1");
  }
  if (topk_count <= 0 || topk_count > 4096) {
    C10_THROW_ERROR(
        ValueError,
        "PE-shared cuBLASDx split-QK forward supports topk in (0, 4096]");
  }
  if (teacher_probs == nullptr || teacher_score_scratch == nullptr) {
    C10_THROW_ERROR(
        ValueError,
        "PE-shared cuBLASDx split-QK forward requires teacher_probs and teacher_score_scratch");
  }
  if (scalar_dtype == kDTypeFloat32) {
    launch_dsa_split_qk_fwd_cublasdx_pe_typed<SharedPeGemmF32, float>(
        query_nope, query_pe, key_nope, key_pe, value, topk_indices,
        query_positions, key_positions, output, lse, teacher_probs,
        teacher_score_scratch, q_len, bsz, sk, num_heads, topk_count,
        q_start, query_nope_stride_s, query_nope_stride_b,
        query_nope_stride_h, query_nope_stride_d, key_nope_stride_s,
        key_nope_stride_b, key_nope_stride_h, key_nope_stride_d,
        value_stride_s, value_stride_b, value_stride_h, value_stride_v,
        softmax_scale, topk_dtype, has_positions, stream);
  } else if (scalar_dtype == kDTypeBFloat16) {
    launch_dsa_split_qk_fwd_cublasdx_pe_typed<SharedPeGemmBF16, __nv_bfloat16>(
        query_nope, query_pe, key_nope, key_pe, value, topk_indices,
        query_positions, key_positions, output, lse, teacher_probs,
        teacher_score_scratch, q_len, bsz, sk, num_heads, topk_count,
        q_start, query_nope_stride_s, query_nope_stride_b,
        query_nope_stride_h, query_nope_stride_d, key_nope_stride_s,
        key_nope_stride_b, key_nope_stride_h, key_nope_stride_d,
        value_stride_s, value_stride_b, value_stride_h, value_stride_v,
        softmax_scale, topk_dtype, has_positions, stream);
  } else if (scalar_dtype == kDTypeFloat16) {
    launch_dsa_split_qk_fwd_cublasdx_pe_typed<SharedPeGemmF16, __half>(
        query_nope, query_pe, key_nope, key_pe, value, topk_indices,
        query_positions, key_positions, output, lse, teacher_probs,
        teacher_score_scratch, q_len, bsz, sk, num_heads, topk_count,
        q_start, query_nope_stride_s, query_nope_stride_b,
        query_nope_stride_h, query_nope_stride_d, key_nope_stride_s,
        key_nope_stride_b, key_nope_stride_h, key_nope_stride_d,
        value_stride_s, value_stride_b, value_stride_h, value_stride_v,
        softmax_scale, topk_dtype, has_positions, stream);
  } else {
    C10_THROW_ERROR(ValueError, "unsupported PE-shared cuBLASDx split-QK forward dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace hisa_indexer
}  // namespace megatron
