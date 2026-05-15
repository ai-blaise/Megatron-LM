// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Blackwell-gated NVFP4 E2M1/UE8M0 backward for DSA IndexCache fake-quant.
// Each warp owns one 32-dim group and applies the STE direct term plus the
// per-group argmax rank update.

#include "indexcache.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace megatron {
namespace indexcache {

template <typename scalar_t>
__device__ __forceinline__ float nvfp4_bwd_load_as_float(const scalar_t* p);
template <>
__device__ __forceinline__ float nvfp4_bwd_load_as_float<float>(const float* p) {
  return *p;
}
template <>
__device__ __forceinline__ float nvfp4_bwd_load_as_float<__nv_bfloat16>(
    const __nv_bfloat16* p) {
  return __bfloat162float(*p);
}
template <>
__device__ __forceinline__ float nvfp4_bwd_load_as_float<__half>(const __half* p) {
  return __half2float(*p);
}

template <typename scalar_t>
__device__ __forceinline__ void nvfp4_bwd_store_from_float(scalar_t* p, float v);
template <>
__device__ __forceinline__ void nvfp4_bwd_store_from_float<float>(float* p, float v) {
  *p = v;
}
template <>
__device__ __forceinline__ void nvfp4_bwd_store_from_float<__nv_bfloat16>(
    __nv_bfloat16* p, float v) {
  *p = __float2bfloat16(v);
}
template <>
__device__ __forceinline__ void nvfp4_bwd_store_from_float<__half>(__half* p, float v) {
  *p = __float2half(v);
}

__device__ __forceinline__ float warp_reduce_sum_broadcast(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return __shfl_sync(0xffffffff, v, 0);
}

__device__ __forceinline__ float nvfp4_bwd_e2m1_code_to_float(uint8_t code) {
  constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = lut[code & 0x7];
  return (code & 0x8) ? -magnitude : magnitude;
}

template <typename scalar_t>
__global__ void indexcache_nvfp4_bwd_kernel(
    const scalar_t* __restrict__ grad_y,
    const scalar_t* __restrict__ x,
    const float* __restrict__ scale_arr,
    const float* __restrict__ q_e2m1,
    const uint8_t* __restrict__ clip_mask,
    const int32_t* __restrict__ argmax_arr,
    const uint8_t* __restrict__ eps_active_arr,
    scalar_t* __restrict__ grad_x,
    float fp4_max,
    int64_t num_rows,
    int64_t row_stride) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int group = tid >> 5;
  const int lane = tid & 31;
  const int64_t row_offset = row * row_stride;
  const int64_t group_offset = row * 4 + group;

  const float gy = nvfp4_bwd_load_as_float(grad_y + row_offset + tid);
  const float xv = nvfp4_bwd_load_as_float(x + row_offset + tid);
  const float scale = scale_arr[group_offset];
  const float q = q_e2m1[row * kHeadDim + tid];
  const float mask = static_cast<float>(clip_mask[row * kHeadDim + tid]);

  const float grad_direct = gy * mask;
  const float per_coord = gy * (q - mask * xv / scale);
  const float inner = warp_reduce_sum_broadcast(per_coord);

  float gx = grad_direct;
  if (lane == argmax_arr[group_offset]) {
    const float sign_x = (xv > 0.0f) ? 1.0f : ((xv < 0.0f) ? -1.0f : 0.0f);
    const float eps_active = static_cast<float>(eps_active_arr[group_offset]);
    gx += inner * sign_x * eps_active * (1.0f / fp4_max);
  }

  nvfp4_bwd_store_from_float(grad_x + row_offset + tid, gx);
}

template <typename scalar_t>
__global__ void indexcache_nvfp4_bwd_packed_kernel(
    const scalar_t* __restrict__ grad_y,
    const scalar_t* __restrict__ x,
    const float* __restrict__ scale_arr,
    const uint8_t* __restrict__ packed_values,
    const uint8_t* __restrict__ clip_mask,
    const int32_t* __restrict__ argmax_arr,
    const uint8_t* __restrict__ eps_active_arr,
    scalar_t* __restrict__ grad_x,
    float fp4_max,
    int64_t num_rows,
    int64_t row_stride) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int group = tid >> 5;
  const int lane = tid & 31;
  const int64_t row_offset = row * row_stride;
  const int64_t group_offset = row * 4 + group;
  const int64_t head_offset = row * kHeadDim + tid;

  const float gy = nvfp4_bwd_load_as_float(grad_y + row_offset + tid);
  const float xv = nvfp4_bwd_load_as_float(x + row_offset + tid);
  const float scale = scale_arr[group_offset];
  const uint8_t packed = packed_values[row * (kHeadDim / 2) + (tid >> 1)];
  const uint8_t code = (tid & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
  const float q = nvfp4_bwd_e2m1_code_to_float(code);
  const float mask = static_cast<float>(clip_mask[head_offset]);

  const float grad_direct = gy * mask;
  const float per_coord = gy * (q - mask * xv / scale);
  const float inner = warp_reduce_sum_broadcast(per_coord);

  float gx = grad_direct;
  if (lane == argmax_arr[group_offset]) {
    const float sign_x = (xv > 0.0f) ? 1.0f : ((xv < 0.0f) ? -1.0f : 0.0f);
    const float eps_active = static_cast<float>(eps_active_arr[group_offset]);
    gx += inner * sign_x * eps_active * (1.0f / fp4_max);
  }

  nvfp4_bwd_store_from_float(grad_x + row_offset + tid, gx);
}

template <typename scalar_t>
void launch_indexcache_nvfp4_bwd(
    const scalar_t* grad_y,
    const scalar_t* x,
    const float* scale,
    const float* q_e2m1,
    const uint8_t* clip_mask,
    const int32_t* argmax,
    const uint8_t* eps_active,
    scalar_t* grad_x,
    float fp4_max,
    int64_t num_rows,
    int64_t row_stride,
    cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(num_rows));
  const dim3 block(kHeadDim);
  indexcache_nvfp4_bwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
      grad_y, x, scale, q_e2m1, clip_mask, argmax, eps_active,
      grad_x, fp4_max, num_rows, row_stride);
}

template <typename scalar_t>
void launch_indexcache_nvfp4_bwd_packed(
    const scalar_t* grad_y,
    const scalar_t* x,
    const float* scale,
    const uint8_t* packed_values,
    const uint8_t* clip_mask,
    const int32_t* argmax,
    const uint8_t* eps_active,
    scalar_t* grad_x,
    float fp4_max,
    int64_t num_rows,
    int64_t row_stride,
    cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(num_rows));
  const dim3 block(kHeadDim);
  indexcache_nvfp4_bwd_packed_kernel<scalar_t><<<grid, block, 0, stream>>>(
      grad_y, x, scale, packed_values, clip_mask, argmax, eps_active,
      grad_x, fp4_max, num_rows, row_stride);
}

template void launch_indexcache_nvfp4_bwd<float>(
    const float*, const float*, const float*, const float*, const uint8_t*,
    const int32_t*, const uint8_t*, float*, float, int64_t, int64_t, cudaStream_t);
template void launch_indexcache_nvfp4_bwd<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*,
    const uint8_t*, const int32_t*, const uint8_t*, __nv_bfloat16*,
    float, int64_t, int64_t, cudaStream_t);
template void launch_indexcache_nvfp4_bwd<__half>(
    const __half*, const __half*, const float*, const float*, const uint8_t*,
    const int32_t*, const uint8_t*, __half*, float, int64_t, int64_t, cudaStream_t);

template void launch_indexcache_nvfp4_bwd_packed<float>(
    const float*, const float*, const float*, const uint8_t*, const uint8_t*,
    const int32_t*, const uint8_t*, float*, float, int64_t, int64_t, cudaStream_t);
template void launch_indexcache_nvfp4_bwd_packed<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const float*, const uint8_t*,
    const uint8_t*, const int32_t*, const uint8_t*, __nv_bfloat16*,
    float, int64_t, int64_t, cudaStream_t);
template void launch_indexcache_nvfp4_bwd_packed<__half>(
    const __half*, const __half*, const float*, const uint8_t*, const uint8_t*,
    const int32_t*, const uint8_t*, __half*, float, int64_t, int64_t, cudaStream_t);

}  // namespace indexcache
}  // namespace megatron
