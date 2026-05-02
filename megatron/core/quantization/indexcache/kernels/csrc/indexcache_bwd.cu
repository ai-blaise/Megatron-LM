// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Backward CUDA kernel for IndexCache fp8 fake-quant. The forward kernel
// stores per-coord saved state (q_fp8, clip_mask) and per-row scalars
// (scale, argmax, eps_active); this kernel reads them and emits grad_x.
//
// Math (matches reference.py:indexcache_backward exactly):
//   grad_x_j = grad_y_j * clip_mask_j
//            + delta_{j, argmax} * sign(x_argmax) * eps_active * (1/fp8_max)
//              * sum_i [ grad_y_i * (q_fp8_i - clip_mask_i * x_i / scale) ]
//
// One block per token, kHeadDim threads per block.

#include "indexcache.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace megatron {
namespace indexcache {

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* p);
template <>
__device__ __forceinline__ float load_as_float<float>(const float* p) { return *p; }
template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16* p) {
  return __bfloat162float(*p);
}
template <>
__device__ __forceinline__ float load_as_float<__half>(const __half* p) {
  return __half2float(*p);
}

template <typename scalar_t>
__device__ __forceinline__ void store_from_float(scalar_t* p, float v);
template <>
__device__ __forceinline__ void store_from_float<float>(float* p, float v) { *p = v; }
template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* p, float v) {
  *p = __float2bfloat16(v);
}
template <>
__device__ __forceinline__ void store_from_float<__half>(__half* p, float v) {
  *p = __float2half(v);
}

__device__ __forceinline__ float block_reduce_sum_128(
    float v, float* __restrict__ scratch) {
  const int tid = threadIdx.x;
  v = warp_reduce_sum(v);
  if ((tid & 31) == 0) {
    scratch[tid >> 5] = v;
  }
  __syncthreads();
  if (tid < 32) {
    float s = (tid < (kHeadDim / 32)) ? scratch[tid] : 0.0f;
    s = warp_reduce_sum(s);
    if (tid == 0) {
      scratch[8] = s;  // fresh slot per the TurboQuant lesson
    }
  }
  __syncthreads();
  const float total = scratch[8];
  __syncthreads();
  return total;
}

template <typename scalar_t>
__global__ void indexcache_kv_bwd_kernel(
    const scalar_t* __restrict__ grad_y,
    const scalar_t* __restrict__ x,
    const float* __restrict__ scale_arr,
    const __nv_fp8_e4m3* __restrict__ q_fp8,
    const uint8_t* __restrict__ clip_mask,
    const int32_t* __restrict__ argmax_arr,
    const uint8_t* __restrict__ eps_active_arr,
    scalar_t* __restrict__ grad_x,
    float fp8_max,
    int64_t num_rows,
    int64_t row_stride) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  __shared__ float reduce_scratch[kHeadDim];

  const float gy = load_as_float(grad_y + row * row_stride + tid);
  const float xv = load_as_float(x + row * row_stride + tid);
  const float scale = scale_arr[row];
  const float q_fp32 = static_cast<float>(q_fp8[row * kHeadDim + tid]);
  const float mask_v = static_cast<float>(clip_mask[row * kHeadDim + tid]);
  const float eps_active = static_cast<float>(eps_active_arr[row]);
  const int32_t argmax = argmax_arr[row];

  // Direct STE: grad_y * mask  (per coord).
  const float grad_direct = gy * mask_v;

  // Rank-1 scale-path inner product:
  //   inner = sum_i [ gy_i * (q_fp8_i - mask_i * x_i / scale) ]
  const float per_coord = gy * (q_fp32 - mask_v * xv / scale);
  const float inner = block_reduce_sum_128(per_coord, reduce_scratch);

  // Apply rank-1 update only on the row's argmax(|x|) coord.
  float gx = grad_direct;
  if (tid == argmax) {
    const float sign_x = (xv >= 0.0f) ? 1.0f : -1.0f;
    gx += inner * sign_x * eps_active * (1.0f / fp8_max);
  }

  store_from_float(grad_x + row * row_stride + tid, gx);
}

template <typename scalar_t>
void launch_indexcache_kv_bwd(
    const scalar_t* grad_y,
    const scalar_t* x,
    const float* scale,
    const __nv_fp8_e4m3* q_fp8,
    const uint8_t* clip_mask,
    const int32_t* argmax,
    const uint8_t* eps_active,
    scalar_t* grad_x,
    float fp8_max,
    int64_t num_rows,
    int64_t row_stride,
    cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(num_rows));
  const dim3 block(kHeadDim);
  indexcache_kv_bwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
      grad_y, x, scale, q_fp8, clip_mask, argmax, eps_active,
      grad_x, fp8_max, num_rows, row_stride);
}

template void launch_indexcache_kv_bwd<float>(
    const float*, const float*, const float*, const __nv_fp8_e4m3*,
    const uint8_t*, const int32_t*, const uint8_t*, float*,
    float, int64_t, int64_t, cudaStream_t);
template void launch_indexcache_kv_bwd<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const float*, const __nv_fp8_e4m3*,
    const uint8_t*, const int32_t*, const uint8_t*, __nv_bfloat16*,
    float, int64_t, int64_t, cudaStream_t);
template void launch_indexcache_kv_bwd<__half>(
    const __half*, const __half*, const float*, const __nv_fp8_e4m3*,
    const uint8_t*, const int32_t*, const uint8_t*, __half*,
    float, int64_t, int64_t, cudaStream_t);

}  // namespace indexcache
}  // namespace megatron
