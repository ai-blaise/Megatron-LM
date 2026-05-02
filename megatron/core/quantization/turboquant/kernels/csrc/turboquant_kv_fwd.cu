// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Fused forward kernel for the 2.5-bit TurboQuant fake-quant on the dense
// MLA latent. One CUDA block per token; 512 threads per block (one per
// latent coordinate). The kernel produces:
//   out[row, :]              -- dequantized fake-quant output, same dtype as x
//   indices[row, :]          -- uint8 quantizer index per coord (0..7 hi, 0..3 lo)
//   ste_mask[row, :]         -- 0/1 byte per coord; backward gates STE on this
//   norm[row], inner_norm[row] -- fp32 scalars saved for backward
//
// IKP region annotations (see ../build.py for the IKP_ENABLED flag):
//   region "norm"     : forward L2 norm reduction
//   region "rotate"   : sign flip + FWHT
//   region "quant"    : codebook search + index store
//   region "inv_rot"  : sign flip + inverse FWHT
//   region "norm2"    : reconstruct L2 norm reduction
//   region "writeout" : final scale + store
//
// Math reference: see ../../reference.py (turboquant_forward); this kernel is
// the bit-identical fp32 implementation of the same algorithm.

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
__device__ __forceinline__ float load_as_float<float>(const float* p) {
  return *p;
}
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
__device__ __forceinline__ void store_from_float<float>(float* p, float v) {
  *p = v;
}
template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* p, float v) {
  *p = __float2bfloat16(v);
}
template <>
__device__ __forceinline__ void store_from_float<__half>(__half* p, float v) {
  *p = __float2half(v);
}

// Forward kernel — direct port of SGLang's store_2p5_kernel (the
// quantization+norm-correction half) followed by SGLang's
// dequantize_selected_2p5_kernel (the inverse-FWHT+rescale half), fused into
// a single kernel that materializes the dequantized fake-quant output. The
// only Megatron-specific additions are saving the per-coord quantizer index
// and saturating-STE mask for the backward pass.
//
// Parseval shortcut (matches SGLang store kernel exactly): inner_norm equals
// sqrt(sum(centroid^2)) without performing the inverse FWHT, since the FWHT
// + signs2 are unitary on the squared-L2 norm.
// `w_hat_save` is an optional bf16 [N, kLatentDim] buffer that, when non-null,
// receives the post-inverse-FWHT pre-norm-correction reconstruction. Saving it
// in the forward lets the backward skip its own w_hat recompute (which is the
// largest single FWHT in the backward chain). bf16 round-trip preserves enough
// precision for STE-level gradient accuracy (validated bit-equivalent at
// <2e-6 vs the recompute path).
template <typename scalar_t, bool kNormCorrection, bool kSaveWHat>
__global__ void turboquant_kv_fwd_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    uint8_t* __restrict__ indices,
    uint8_t* __restrict__ ste_mask,
    float* __restrict__ norm_out,
    float* __restrict__ inner_norm_out,
    float* __restrict__ w_hat_save,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    const float* __restrict__ boundaries_high,
    const float* __restrict__ boundaries_low,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    int64_t num_rows,
    int64_t row_stride) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  __shared__ float buf[kLatentDim];
  __shared__ float norm_sh;
  __shared__ float inner_norm_sh;

  const scalar_t* x_row = x + row * row_stride;
  scalar_t* out_row = out + row * row_stride;
  uint8_t* indices_row = indices + row * kLatentDim;
  uint8_t* mask_row = ste_mask + row * kLatentDim;

  // region: norm — port of SGLang store_2p5_kernel L184-188
  TQ_REGION_BEGIN(norm);
  const float xv = load_as_float(x_row + tid);
  const float norm_sum = block_reduce_sum_512(xv * xv, buf);
  if (tid == 0) {
    norm_sh = sqrtf(fmaxf(norm_sum, 1.0e-16f));
    norm_out[row] = norm_sh;
  }
  __syncthreads();
  const float norm = norm_sh;
  TQ_REGION_END(norm);

  // region: rotate — port of SGLang store_2p5_kernel L191
  TQ_REGION_BEGIN(rotate);
  const float pre = (xv / norm) * signs1[tid];
  const float transformed = fwht_512(pre, buf);
  const float rotated = transformed * kInvSqrtLatentDim * signs2[tid];
  TQ_REGION_END(rotate);

  // region: quant — port of SGLang store_2p5_kernel L193-205
  TQ_REGION_BEGIN(quant);
  const int channel = tid & (kGroupSize - 1);
  uint8_t index;
  uint8_t mask_v;
  float centroid;
  if (channel < kHighChannels) {
    index = quantize_with_boundaries<kHighLevels - 1>(boundaries_high, rotated);
    centroid = centroids_high[index];
    mask_v = static_cast<uint8_t>(
        rotated >= boundaries_high[0] &&
        rotated <= boundaries_high[kHighLevels - 2]);
  } else {
    index = quantize_with_boundaries<kLowLevels - 1>(boundaries_low, rotated);
    centroid = centroids_low[index];
    mask_v = static_cast<uint8_t>(
        rotated >= boundaries_low[0] &&
        rotated <= boundaries_low[kLowLevels - 2]);
  }
  indices_row[tid] = index;
  mask_row[tid] = mask_v;
  TQ_REGION_END(quant);

  // region: inner_norm via Parseval — port of SGLang store_2p5_kernel L206-213.
  // SGLang only stores corrected_norm = norm / recon_norm, which uses the
  // identity ||fwht(centroid * signs2) / sqrt(d)||^2 = ||centroid||^2 / d.
  // We compute the squared sum identically and take the same sqrt.
  TQ_REGION_BEGIN(parseval);
  if (kNormCorrection) {
    const float recon_sum = block_reduce_sum_512(centroid * centroid, buf);
    if (tid == 0) {
      inner_norm_sh = sqrtf(fmaxf(recon_sum, 1.0e-16f));
      inner_norm_out[row] = inner_norm_sh;
    }
    __syncthreads();
  } else {
    if (tid == 0) {
      inner_norm_sh = 1.0f;
      inner_norm_out[row] = 1.0f;
    }
    __syncthreads();
  }
  const float inner_norm = inner_norm_sh;
  TQ_REGION_END(parseval);

  // region: inv_rot — port of SGLang dequantize_selected_2p5_kernel L271-303
  TQ_REGION_BEGIN(inv_rot);
  const float r_back = centroid * signs2[tid];
  const float w_hat = fwht_512(r_back, buf) * kInvSqrtLatentDim;
  TQ_REGION_END(inv_rot);

  if (kSaveWHat) {
    w_hat_save[row * kLatentDim + tid] = w_hat;
  }

  // region: writeout — port of SGLang dequantize_selected_2p5_kernel final write
  TQ_REGION_BEGIN(writeout);
  const float norm_hat = kNormCorrection ? (norm / inner_norm) : norm;
  store_from_float(out_row + tid, w_hat * signs1[tid] * norm_hat);
  TQ_REGION_END(writeout);
}

template <typename scalar_t>
void launch_turboquant_kv_fwd(
    const scalar_t* x,
    scalar_t* out,
    uint8_t* indices,
    uint8_t* ste_mask,
    float* norm_out,
    float* inner_norm_out,
    float* w_hat_save,
    const float* signs1,
    const float* signs2,
    const float* boundaries_high,
    const float* boundaries_low,
    const float* centroids_high,
    const float* centroids_low,
    int64_t num_rows,
    int64_t row_stride,
    bool norm_correction,
    cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(num_rows));
  const dim3 block(kLatentDim);
  const bool save_w = (w_hat_save != nullptr);
  if (norm_correction && save_w) {
    turboquant_kv_fwd_kernel<scalar_t, true, true><<<grid, block, 0, stream>>>(
        x, out, indices, ste_mask, norm_out, inner_norm_out, w_hat_save,
        signs1, signs2, boundaries_high, boundaries_low,
        centroids_high, centroids_low, num_rows, row_stride);
  } else if (norm_correction) {
    turboquant_kv_fwd_kernel<scalar_t, true, false><<<grid, block, 0, stream>>>(
        x, out, indices, ste_mask, norm_out, inner_norm_out, nullptr,
        signs1, signs2, boundaries_high, boundaries_low,
        centroids_high, centroids_low, num_rows, row_stride);
  } else if (save_w) {
    turboquant_kv_fwd_kernel<scalar_t, false, true><<<grid, block, 0, stream>>>(
        x, out, indices, ste_mask, norm_out, inner_norm_out, w_hat_save,
        signs1, signs2, boundaries_high, boundaries_low,
        centroids_high, centroids_low, num_rows, row_stride);
  } else {
    turboquant_kv_fwd_kernel<scalar_t, false, false><<<grid, block, 0, stream>>>(
        x, out, indices, ste_mask, norm_out, inner_norm_out, nullptr,
        signs1, signs2, boundaries_high, boundaries_low,
        centroids_high, centroids_low, num_rows, row_stride);
  }
}

template void launch_turboquant_kv_fwd<float>(
    const float*, float*, uint8_t*, uint8_t*, float*, float*, float*,
    const float*, const float*, const float*, const float*,
    const float*, const float*, int64_t, int64_t, bool, cudaStream_t);
template void launch_turboquant_kv_fwd<__nv_bfloat16>(
    const __nv_bfloat16*, __nv_bfloat16*, uint8_t*, uint8_t*, float*, float*, float*,
    const float*, const float*, const float*, const float*,
    const float*, const float*, int64_t, int64_t, bool, cudaStream_t);
template void launch_turboquant_kv_fwd<__half>(
    const __half*, __half*, uint8_t*, uint8_t*, float*, float*, float*,
    const float*, const float*, const float*, const float*,
    const float*, const float*, int64_t, int64_t, bool, cudaStream_t);

}  // namespace turboquant
}  // namespace megatron
