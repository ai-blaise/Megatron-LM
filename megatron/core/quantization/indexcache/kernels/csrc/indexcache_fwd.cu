// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Forward CUDA kernel for the IndexCache fp8 e4m3 fake-quant on the DSA
// indexer K tensor. One block per token, kHeadDim (128) threads per block.
// Direct port of the per-token math in SGLang's
//   optimization-playground/python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh
// with two adaptations:
//   1. Output is the dequantized fake-quant tensor (q_fp8 * scale) in the
//      input dtype, instead of packed bytes in a paged cache.
//   2. We additionally save the per-coord fp8 index, the clip mask, the
//      per-row scale, the per-row argmax-of-|x|, and an eps-active flag
//      so the backward kernel can skip recomputing them.

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

template <typename scalar_t>
__global__ void indexcache_kv_fwd_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    float* __restrict__ scale_out,
    __nv_fp8_e4m3* __restrict__ q_fp8_out,
    uint8_t* __restrict__ clip_mask_out,
    int32_t* __restrict__ argmax_out,
    uint8_t* __restrict__ eps_active_out,
    float eps,
    float fp8_max,
    int64_t num_rows,
    int64_t row_stride) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  __shared__ float scratch[kHeadDim];
  __shared__ float scale_sh;
  __shared__ int   argmax_sh;

  if (tid == 0) {
    argmax_sh = INT_MAX;
  }
  __syncthreads();

  const scalar_t* x_row = x + row * row_stride;
  const float xv = load_as_float(x_row + tid);
  const float abs_xv = fabsf(xv);

  // Block-wide max(|x|) — port of warp::reduce_max in SGLang fused_store kernel.
  const float abs_max = block_reduce_max_128(abs_xv, scratch);

  // scale = max(eps, abs_max) / fp8_max  (line 67 of SGLang kernel).
  if (tid == 0) {
    scale_sh = fmaxf(eps, abs_max) * (1.0f / fp8_max);
  }
  __syncthreads();
  const float scale = scale_sh;

  // Quantize: clamp(x / scale, +-fp8_max), then cast to fp8_e4m3.
  const float pre_clip = xv / scale;
  const bool in_range = (pre_clip >= -fp8_max) && (pre_clip <= fp8_max);
  const float clipped = fminf(fmaxf(pre_clip, -fp8_max), fp8_max);
  const __nv_fp8_e4m3 q_fp8 = __nv_fp8_e4m3(clipped);
  const float q_fp32 = static_cast<float>(q_fp8);

  // Stores per coord.
  q_fp8_out[row * kHeadDim + tid] = q_fp8;
  clip_mask_out[row * kHeadDim + tid] = static_cast<uint8_t>(in_range);

  // Find argmax(|x|) for the rank-1 scale-path gradient. We pick the lane
  // whose abs equals the block max; ties broken by lowest tid via atomicMin
  // on argmax_sh which was initialised to INT_MAX above.
  if (abs_xv == abs_max) {
    atomicMin(&argmax_sh, tid);
  }
  __syncthreads();

  // Per-row scalar writes (one thread).
  if (tid == 0) {
    scale_out[row] = scale;
    argmax_out[row] = argmax_sh;
    eps_active_out[row] = static_cast<uint8_t>(abs_max >= eps);
  }

  // Output: y = q_fp8 * scale, in input dtype.
  store_from_float(out + row * row_stride + tid, q_fp32 * scale);
}

template <typename scalar_t>
void launch_indexcache_kv_fwd(
    const scalar_t* x,
    scalar_t* out,
    float* scale,
    __nv_fp8_e4m3* q_fp8,
    uint8_t* clip_mask,
    int32_t* argmax,
    uint8_t* eps_active,
    float eps,
    float fp8_max,
    int64_t num_rows,
    int64_t row_stride,
    cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(num_rows));
  const dim3 block(kHeadDim);
  indexcache_kv_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
      x, out, scale, q_fp8, clip_mask, argmax, eps_active,
      eps, fp8_max, num_rows, row_stride);
}

template void launch_indexcache_kv_fwd<float>(
    const float*, float*, float*, __nv_fp8_e4m3*, uint8_t*, int32_t*, uint8_t*,
    float, float, int64_t, int64_t, cudaStream_t);
template void launch_indexcache_kv_fwd<__nv_bfloat16>(
    const __nv_bfloat16*, __nv_bfloat16*, float*, __nv_fp8_e4m3*, uint8_t*, int32_t*, uint8_t*,
    float, float, int64_t, int64_t, cudaStream_t);
template void launch_indexcache_kv_fwd<__half>(
    const __half*, __half*, float*, __nv_fp8_e4m3*, uint8_t*, int32_t*, uint8_t*,
    float, float, int64_t, int64_t, cudaStream_t);

}  // namespace indexcache
}  // namespace megatron
