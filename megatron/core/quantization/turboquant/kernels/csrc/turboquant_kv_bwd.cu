// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Fused backward kernel for the 2.5-bit TurboQuant fake-quant. Mirrors the
// forward layout: one block per token, 512 threads per block. Inputs are
//   grad_out[row, :]  -- upstream gradient (same dtype as forward output)
//   x[row, :]         -- saved fp32 input
//   indices[row, :]   -- forward-time quantizer indices
//   ste_mask[row, :]  -- 0/1 byte per coord
//   norm[row], inner_norm[row]
// Output: grad_x[row, :] (same dtype as upstream).
//
// We recompute w_hat from indices+codebooks rather than save it in fp32 ---
// 8x memory saving over saving the full 512 fp32 vector per token, and the
// recompute cost is dominated by the FWHT which we'd do regardless.
//
// Math reference: see ../../reference.py turboquant_backward; this kernel is
// a fp32 implementation of the same closed-form.

#include "turboquant_kv.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef IKP_ENABLED
#include "ikp_runtime.cuh"
#define TQ_REGION_BEGIN(name) IKP_REGION_BEGIN(name)
#define TQ_REGION_END(name) IKP_REGION_END(name)
#else
#define TQ_REGION_BEGIN(name)
#define TQ_REGION_END(name)
#endif

namespace megatron {
namespace turboquant {

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

template <typename scalar_t, bool kNormCorrection>
__global__ void turboquant_kv_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ indices,
    const uint8_t* __restrict__ ste_mask,
    const float* __restrict__ norm_arr,
    const float* __restrict__ inner_norm_arr,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    scalar_t* __restrict__ grad_x,
    int64_t num_rows,
    int64_t row_stride) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  __shared__ float buf[kLatentDim];
  __shared__ float scratch[kLatentDim];   // for w_hat recompute
  __shared__ float reduce_scratch[kLatentDim];

  const scalar_t* grad_row = grad_out + row * row_stride;
  const scalar_t* x_row = x + row * row_stride;
  const uint8_t* idx_row = indices + row * kLatentDim;
  const uint8_t* mask_row = ste_mask + row * kLatentDim;

  const float norm = norm_arr[row];
  const float inner_norm = inner_norm_arr[row];

  // ------------------- region: recompute_w_hat -------------------
  TQ_REGION_BEGIN(recompute_w_hat);
  const int channel = tid & (kGroupSize - 1);
  float centroid;
  if (channel < kHighChannels) {
    centroid = centroids_high[idx_row[tid]];
  } else {
    centroid = centroids_low[idx_row[tid]];
  }
  const float r_back = centroid * signs2[tid];
  const float w_hat = fwht_512(r_back, scratch) * kInvSqrtLatentDim;
  TQ_REGION_END(recompute_w_hat);

  // ------------------- region: chain_through_outputs -------------------
  TQ_REGION_BEGIN(chain_outputs);
  const float gv = load_as_float(grad_row + tid);
  const float z_hat = w_hat * signs1[tid];

  float norm_hat;
  if (kNormCorrection) {
    norm_hat = norm / inner_norm;
  } else {
    norm_hat = norm;
  }

  const float grad_z_hat = gv * norm_hat;
  const float grad_norm_hat = block_reduce_sum_512(gv * z_hat, reduce_scratch);

  float grad_w_hat = grad_z_hat * signs1[tid];
  float grad_norm_external;
  if (kNormCorrection) {
    grad_norm_external = grad_norm_hat / inner_norm;
    const float grad_inner_norm = -grad_norm_hat * norm / (inner_norm * inner_norm);
    grad_w_hat += (grad_inner_norm / inner_norm) * w_hat;
  } else {
    grad_norm_external = grad_norm_hat;
  }
  TQ_REGION_END(chain_outputs);

  // ------------------- region: invert_rotation -------------------
  TQ_REGION_BEGIN(invert_rotation);
  const float grad_quantized = fwht_512(grad_w_hat, scratch) * kInvSqrtLatentDim
                               * signs2[tid];
  const float grad_rotated = grad_quantized * static_cast<float>(mask_row[tid]);
  const float grad_unit = fwht_512(grad_rotated * signs2[tid], scratch)
                          * kInvSqrtLatentDim * signs1[tid];
  TQ_REGION_END(invert_rotation);

  // ------------------- region: chain_through_norm -------------------
  TQ_REGION_BEGIN(chain_norm);
  const float xv = load_as_float(x_row + tid);
  const float inv_norm = 1.0f / norm;
  const float proj = block_reduce_sum_512(grad_unit * xv, reduce_scratch);
  const float inv_norm3 = inv_norm * inv_norm * inv_norm;
  float gx = grad_unit * inv_norm - proj * inv_norm3 * xv;
  gx += grad_norm_external * inv_norm * xv;
  TQ_REGION_END(chain_norm);

  store_from_float(grad_x + row * row_stride + tid, gx);
}

template <typename scalar_t>
void launch_turboquant_kv_bwd(
    const scalar_t* grad_out,
    const scalar_t* x,
    const uint8_t* indices,
    const uint8_t* ste_mask,
    const float* norm_arr,
    const float* inner_norm_arr,
    const float* signs1,
    const float* signs2,
    const float* centroids_high,
    const float* centroids_low,
    scalar_t* grad_x,
    int64_t num_rows,
    int64_t row_stride,
    bool norm_correction,
    cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(num_rows));
  const dim3 block(kLatentDim);
  if (norm_correction) {
    turboquant_kv_bwd_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
        grad_out, x, indices, ste_mask, norm_arr, inner_norm_arr,
        signs1, signs2, centroids_high, centroids_low,
        grad_x, num_rows, row_stride);
  } else {
    turboquant_kv_bwd_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
        grad_out, x, indices, ste_mask, norm_arr, inner_norm_arr,
        signs1, signs2, centroids_high, centroids_low,
        grad_x, num_rows, row_stride);
  }
}

template void launch_turboquant_kv_bwd<float>(
    const float*, const float*, const uint8_t*, const uint8_t*,
    const float*, const float*, const float*, const float*,
    const float*, const float*, float*,
    int64_t, int64_t, bool, cudaStream_t);
template void launch_turboquant_kv_bwd<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const uint8_t*, const uint8_t*,
    const float*, const float*, const float*, const float*,
    const float*, const float*, __nv_bfloat16*,
    int64_t, int64_t, bool, cudaStream_t);
template void launch_turboquant_kv_bwd<__half>(
    const __half*, const __half*, const uint8_t*, const uint8_t*,
    const float*, const float*, const float*, const float*,
    const float*, const float*, __half*,
    int64_t, int64_t, bool, cudaStream_t);

}  // namespace turboquant
}  // namespace megatron
