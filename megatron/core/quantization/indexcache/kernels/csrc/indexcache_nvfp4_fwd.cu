// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Blackwell-gated NVFP4 E2M1/UE8M0 forward for DSA IndexCache fake-quant.
// One 128-thread block handles one indexer row. Each warp handles one 32-dim
// group and emits the OP-compatible 64 value bytes + one packed scale word.

#include "indexcache.cuh"

#include <climits>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace megatron {
namespace indexcache {

constexpr float kNvfp4E2M1Max = 6.0f;
constexpr float kNvfp4E2M1MaxInv = 1.0f / kNvfp4E2M1Max;

template <typename scalar_t>
__device__ __forceinline__ float nvfp4_load_as_float(const scalar_t* p);
template <>
__device__ __forceinline__ float nvfp4_load_as_float<float>(const float* p) { return *p; }
template <>
__device__ __forceinline__ float nvfp4_load_as_float<__nv_bfloat16>(
    const __nv_bfloat16* p) {
  return __bfloat162float(*p);
}
template <>
__device__ __forceinline__ float nvfp4_load_as_float<__half>(const __half* p) {
  return __half2float(*p);
}

template <typename scalar_t>
__device__ __forceinline__ void nvfp4_store_from_float(scalar_t* p, float v);
template <>
__device__ __forceinline__ void nvfp4_store_from_float<float>(float* p, float v) {
  *p = v;
}
template <>
__device__ __forceinline__ void nvfp4_store_from_float<__nv_bfloat16>(
    __nv_bfloat16* p, float v) {
  *p = __float2bfloat16(v);
}
template <>
__device__ __forceinline__ void nvfp4_store_from_float<__half>(__half* p, float v) {
  *p = __float2half(v);
}

__device__ __forceinline__ float warp_broadcast_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return __shfl_sync(0xffffffff, v, 0);
}

__device__ __forceinline__ uint8_t ceil_to_ue8m0_exp(float value) {
  uint32_t bits = __float_as_uint(fabsf(value));
  uint32_t exp = (bits >> 23) & 0xffu;
  exp += (bits & 0x7fffffu) != 0;
  exp = max(1u, min(exp, 254u));
  return static_cast<uint8_t>(exp);
}

__device__ __forceinline__ uint8_t quantize_e2m1_code(float x) {
  const float ax = fminf(fabsf(x), kNvfp4E2M1Max);
  uint8_t idx = 0;
  idx = (ax > 0.25f) ? 1 : idx;
  idx = (ax >= 0.75f) ? 2 : idx;
  idx = (ax > 1.25f) ? 3 : idx;
  idx = (ax >= 1.75f) ? 4 : idx;
  idx = (ax > 2.5f) ? 5 : idx;
  idx = (ax >= 3.5f) ? 6 : idx;
  idx = (ax > 5.0f) ? 7 : idx;
  if (x < 0.0f && idx != 0) {
    idx |= 0x8;
  }
  return idx;
}

__device__ __forceinline__ float e2m1_code_to_float(uint8_t code) {
  constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = lut[code & 0x7];
  return (code & 0x8) ? -magnitude : magnitude;
}

template <typename scalar_t>
__global__ void indexcache_nvfp4_fwd_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    float* __restrict__ scale_out,
    float* __restrict__ q_e2m1_out,
    uint8_t* __restrict__ clip_mask_out,
    int32_t* __restrict__ argmax_out,
    uint8_t* __restrict__ eps_active_out,
    uint8_t* __restrict__ packed_values_out,
    int32_t* __restrict__ packed_scales_out,
    float eps,
    int64_t num_rows,
    int64_t row_stride) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int group = tid >> 5;
  const int lane = tid & 31;
  const int64_t row_offset = row * row_stride;

  __shared__ float scale_sh[4];
  __shared__ uint8_t scale_exp_sh[4];
  __shared__ int argmax_sh[4];

  if (tid < 4) {
    argmax_sh[tid] = INT_MAX;
  }
  __syncthreads();

  const float xv = nvfp4_load_as_float(x + row_offset + tid);
  const float abs_xv = fabsf(xv);
  const float group_max = warp_reduce_max(abs_xv);

  if (lane == 0) {
    const uint8_t exp = ceil_to_ue8m0_exp(fmaxf(eps, group_max) * kNvfp4E2M1MaxInv);
    scale_exp_sh[group] = exp;
    scale_sh[group] = __uint_as_float(static_cast<uint32_t>(exp) << 23);
    scale_out[row * 4 + group] = scale_sh[group];
    eps_active_out[row * 4 + group] = static_cast<uint8_t>(group_max >= eps);
  }

  if (abs_xv == group_max) {
    atomicMin(&argmax_sh[group], lane);
  }
  __syncthreads();

  const float scale = scale_sh[group];
  const float pre_clip = xv / scale;
  const bool in_range = (pre_clip >= -kNvfp4E2M1Max) && (pre_clip <= kNvfp4E2M1Max);
  const float clipped = fminf(fmaxf(pre_clip, -kNvfp4E2M1Max), kNvfp4E2M1Max);
  const uint8_t code = quantize_e2m1_code(clipped);
  const float q = e2m1_code_to_float(code);

  q_e2m1_out[row * kHeadDim + tid] = q;
  clip_mask_out[row * kHeadDim + tid] = static_cast<uint8_t>(in_range);
  nvfp4_store_from_float(out + row_offset + tid, q * scale);

  const uint8_t next_code = __shfl_down_sync(0xffffffff, code, 1);
  if ((tid & 1) == 0) {
    packed_values_out[row * (kHeadDim / 2) + (tid >> 1)] =
        static_cast<uint8_t>((code & 0x0f) | ((next_code & 0x0f) << 4));
  }

  if (lane == 0) {
    argmax_out[row * 4 + group] = argmax_sh[group];
  }
  if (tid == 0) {
    const uint32_t scale_word =
        static_cast<uint32_t>(scale_exp_sh[0]) |
        (static_cast<uint32_t>(scale_exp_sh[1]) << 8) |
        (static_cast<uint32_t>(scale_exp_sh[2]) << 16) |
        (static_cast<uint32_t>(scale_exp_sh[3]) << 24);
    packed_scales_out[row] = static_cast<int32_t>(scale_word);
  }
}

template <typename scalar_t>
void launch_indexcache_nvfp4_fwd(
    const scalar_t* x,
    scalar_t* out,
    float* scale,
    float* q_e2m1,
    uint8_t* clip_mask,
    int32_t* argmax,
    uint8_t* eps_active,
    uint8_t* packed_values,
    int32_t* packed_scales,
    float eps,
    int64_t num_rows,
    int64_t row_stride,
    cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(num_rows));
  const dim3 block(kHeadDim);
  indexcache_nvfp4_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
      x, out, scale, q_e2m1, clip_mask, argmax, eps_active,
      packed_values, packed_scales, eps, num_rows, row_stride);
}

template void launch_indexcache_nvfp4_fwd<float>(
    const float*, float*, float*, float*, uint8_t*, int32_t*, uint8_t*,
    uint8_t*, int32_t*, float, int64_t, int64_t, cudaStream_t);
template void launch_indexcache_nvfp4_fwd<__nv_bfloat16>(
    const __nv_bfloat16*, __nv_bfloat16*, float*, float*, uint8_t*, int32_t*,
    uint8_t*, uint8_t*, int32_t*, float, int64_t, int64_t, cudaStream_t);
template void launch_indexcache_nvfp4_fwd<__half>(
    const __half*, __half*, float*, float*, uint8_t*, int32_t*, uint8_t*,
    uint8_t*, int32_t*, float, int64_t, int64_t, cudaStream_t);

}  // namespace indexcache
}  // namespace megatron
