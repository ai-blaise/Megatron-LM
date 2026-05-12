// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Fused backward kernel for the 2-bit HIGGS dense MLA-latent KV fake-quant.
// Mirrors the forward layout: one block per token, 512 threads per block.
//
// Inputs:
//   grad_y[row, :]            -- upstream gradient (same dtype as forward output)
//   x[row, :]                 -- saved input (unused for the closed form here,
//                                kept for parity with TurboQuant's backward
//                                signature and for future variants that need it)
//   indices[row, :]           -- forward-time codebook indices (per pair)
//   ste_mask[row, :]          -- 0/1 byte per coord; STE gate
//   rot_norm[row]             -- saved ||rotated|| in fp32
//   scale[row]                -- saved scale = ||rotated|| / sqrt(N)
//   rotated_save[row, :]      -- saved rotated tensor in bf16 (or null to
//                                recompute via FWHT(x))
//   recon_unit_save[row, :]   -- saved per-thread codebook coord in bf16 (or
//                                null to recompute via codebook[indices])
//   codebook, codebook_norm_sq -- (16, 2) + (16,) frozen lattice
// Output: grad_x[row, :] (same dtype as grad_y).
//
// IKP region annotations (see ../build.py for the IKP_ENABLED flag):
//   region "load_grad"        : upstream + saved scalars load
//   region "invert_outer_fwht": FWHT(grad_y) -> grad_rotated_recon
//   region "chain_scale"      : split grad_rotated_recon = scale * grad_rotated_recon_unit
//                               + reduce grad_rotated_recon . recon_unit -> grad_scale
//   region "chain_normalize"  : grad_rotated_from_normalized + grad_rotated_from_scale
//   region "invert_input_fwht": FWHT(grad_rotated) -> grad_x
//   region "writeout"         : final store
//
// Math reference: see ../../reference.py higgs_backward; this kernel is the
// fp32 implementation of the same closed form.

#include "higgs_kv.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef IKP_ENABLED
#include "ikp_runtime.cuh"
#define HG_REGION_BEGIN(name) IKP_REGION_BEGIN(name)
#define HG_REGION_END(name) IKP_REGION_END(name)
#else
#define HG_REGION_BEGIN(name)
#define HG_REGION_END(name)
#endif

namespace megatron {
namespace higgs {

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* p);
template <>
__device__ __forceinline__ float load_as_float<float>(const float* p) {
  return *p;
}
template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(
    const __nv_bfloat16* p) {
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
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(
    __nv_bfloat16* p, float v) {
  *p = __float2bfloat16(v);
}
template <>
__device__ __forceinline__ void store_from_float<__half>(__half* p, float v) {
  *p = __float2half(v);
}

// ``rotated_save`` and ``recon_unit_save`` are bf16 [N, kLatentDim] tensors
// produced by the forward kernel. Reading them removes the inner-FWHT and
// codebook-lookup recomputations from the backward, the two largest
// hotspots after invert_outer_fwht. When null we fall back to a recompute
// path (FWHT(x) for rotated, codebook[indices] for recon_unit) so external
// callers without a saved forward stay compatible.
template <typename scalar_t, bool kHaveRotated, bool kHaveReconUnit>
__global__ void higgs_kv_bwd_kernel(
    const scalar_t* __restrict__ grad_y,
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ indices,
    const uint8_t* __restrict__ ste_mask,
    const float* __restrict__ rot_norm_arr,
    const float* __restrict__ scale_arr,
    const __nv_bfloat16* __restrict__ rotated_save,
    const __nv_bfloat16* __restrict__ recon_unit_save,
    const float* __restrict__ codebook,
    const float* __restrict__ codebook_norm_sq,  // unused, kept for symmetry
    scalar_t* __restrict__ grad_x,
    int64_t num_rows,
    int64_t row_stride) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  (void)codebook_norm_sq;

  __shared__ float scratch[kLatentDim];          // FWHT working area
  __shared__ float reduce_scratch[64];           // block_reduce_sum partials
  __shared__ float grad_scale_sh;

  const scalar_t* grad_row = grad_y + row * row_stride;
  const scalar_t* x_row = x + row * row_stride;
  const uint8_t* idx_row = indices + row * kNumPairs;
  const uint8_t* mask_row = ste_mask + row * kLatentDim;

  const float rot_norm = rot_norm_arr[row];
  const float scale = scale_arr[row];

  // region: load_grad
  HG_REGION_BEGIN(load_grad);
  const float gy = load_as_float(grad_row + tid);
  HG_REGION_END(load_grad);

  // region: invert_outer_fwht -- FWHT is involutory under ×1/sqrt(N).
  // dL/d rotated_recon = FWHT(dL/d y) * 1/sqrt(N).
  HG_REGION_BEGIN(invert_outer_fwht);
  const float grad_rotated_recon = fwht_512(gy, scratch) * kInvSqrtLatentDim;
  HG_REGION_END(invert_outer_fwht);

  // region: chain_scale
  // Forward did rotated_recon = scale * recon_unit; split the gradient:
  //   dL/d recon_unit = scale * dL/d rotated_recon
  //   dL/d scale      = sum_i (recon_unit_i * dL/d rotated_recon_i)
  HG_REGION_BEGIN(chain_scale);
  float recon_unit;
  if (kHaveReconUnit) {
    recon_unit = __bfloat162float(recon_unit_save[row * kLatentDim + tid]);
  } else {
    const int pair_idx = tid >> 1;
    const int coord = tid & 1;
    const uint32_t cb_idx = static_cast<uint32_t>(idx_row[pair_idx]);
    recon_unit = __ldg(&codebook[cb_idx * kPairDim + coord]);
  }
  const float grad_recon_unit = scale * grad_rotated_recon;
  const float grad_scale_partial = recon_unit * grad_rotated_recon;
  const float grad_scale = block_reduce_sum_512(grad_scale_partial, reduce_scratch);
  if (tid == 0) {
    grad_scale_sh = grad_scale;
  }
  __syncthreads();
  HG_REGION_END(chain_scale);

  // STE: dL/d normalized = dL/d recon_unit * mask (mask=1 for HIGGS today).
  const float mask = static_cast<float>(mask_row[tid]);
  const float grad_normalized = grad_recon_unit * mask;

  // region: chain_normalize -- split rotated -> (normalized, scale)
  //   normalized = rotated / scale ; scale = ||rotated|| / sqrt(N).
  // d normalized_j / d rotated_i = (1/scale) * delta_ij
  //                                - (rotated_j / scale^2) * d scale / d rotated_i
  // d scale / d rotated_i       = rotated_i / (sqrt(N) * rot_norm)
  // Combining the explicit and implicit-through-scale paths:
  //   dL/d rotated_i = (dL/d normalized_i) / scale
  //                  + (rotated_i / (sqrt(N) * rot_norm)) *
  //                      ( dL/d scale - (1/scale^2) *
  //                                     sum_j (rotated_j * dL/d normalized_j) )
  HG_REGION_BEGIN(chain_normalize);
  float rotated_i;
  if (kHaveRotated) {
    rotated_i = __bfloat162float(rotated_save[row * kLatentDim + tid]);
  } else {
    // Recompute rotated_i = FWHT(x) * 1/sqrt(N) on the fly.
    const float xv = load_as_float(x_row + tid);
    rotated_i = fwht_512(xv, scratch) * kInvSqrtLatentDim;
  }
  const float inv_scale = 1.0f / fmaxf(scale, 1.0e-30f);
  const float inv_rot_norm = 1.0f / fmaxf(rot_norm, 1.0e-30f);

  // Compute the projection ``sum_j (rotated_j * grad_normalized_j)`` that
  // appears in the implicit-through-scale correction. We need a block-wide
  // reduction in fp32; reuse the dedicated reduction scratch.
  const float proj = block_reduce_sum_512(
      rotated_i * grad_normalized, reduce_scratch);
  const float correction =
      grad_scale_sh - proj * inv_scale * inv_scale;
  const float grad_rotated =
      grad_normalized * inv_scale
      + correction * (kInvSqrtLatentDim * inv_rot_norm) * rotated_i;
  HG_REGION_END(chain_normalize);

  __syncthreads();  // sync before reusing scratch for the inverse FWHT

  // region: invert_input_fwht -- backprop through the forward FWHT.
  // dL/d x = FWHT(dL/d rotated) * 1/sqrt(N).
  HG_REGION_BEGIN(invert_input_fwht);
  const float gx = fwht_512(grad_rotated, scratch) * kInvSqrtLatentDim;
  HG_REGION_END(invert_input_fwht);

  // region: writeout
  HG_REGION_BEGIN(writeout);
  store_from_float(grad_x + row * row_stride + tid, gx);
  HG_REGION_END(writeout);
}

template <typename scalar_t>
void launch_higgs_kv_bwd(
    const scalar_t* grad_y,
    const scalar_t* x,
    const uint8_t* indices,
    const uint8_t* ste_mask,
    const float* rot_norm_arr,
    const float* scale_arr,
    const __nv_bfloat16* rotated_save,
    const __nv_bfloat16* recon_unit_save,
    const float* codebook,
    const float* codebook_norm_sq,
    scalar_t* grad_x,
    int64_t num_rows,
    int64_t row_stride,
    cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(num_rows));
  const dim3 block(kLatentDim);
  const bool have_rot = (rotated_save != nullptr);
  const bool have_ru = (recon_unit_save != nullptr);
  if (have_rot && have_ru) {
    higgs_kv_bwd_kernel<scalar_t, true, true><<<grid, block, 0, stream>>>(
        grad_y, x, indices, ste_mask, rot_norm_arr, scale_arr,
        rotated_save, recon_unit_save, codebook, codebook_norm_sq,
        grad_x, num_rows, row_stride);
  } else if (have_rot) {
    higgs_kv_bwd_kernel<scalar_t, true, false><<<grid, block, 0, stream>>>(
        grad_y, x, indices, ste_mask, rot_norm_arr, scale_arr,
        rotated_save, nullptr, codebook, codebook_norm_sq,
        grad_x, num_rows, row_stride);
  } else if (have_ru) {
    higgs_kv_bwd_kernel<scalar_t, false, true><<<grid, block, 0, stream>>>(
        grad_y, x, indices, ste_mask, rot_norm_arr, scale_arr,
        nullptr, recon_unit_save, codebook, codebook_norm_sq,
        grad_x, num_rows, row_stride);
  } else {
    higgs_kv_bwd_kernel<scalar_t, false, false><<<grid, block, 0, stream>>>(
        grad_y, x, indices, ste_mask, rot_norm_arr, scale_arr,
        nullptr, nullptr, codebook, codebook_norm_sq,
        grad_x, num_rows, row_stride);
  }
}

template void launch_higgs_kv_bwd<float>(
    const float*, const float*, const uint8_t*, const uint8_t*,
    const float*, const float*, const __nv_bfloat16*, const __nv_bfloat16*,
    const float*, const float*, float*, int64_t, int64_t, cudaStream_t);
template void launch_higgs_kv_bwd<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const uint8_t*, const uint8_t*,
    const float*, const float*, const __nv_bfloat16*, const __nv_bfloat16*,
    const float*, const float*, __nv_bfloat16*, int64_t, int64_t, cudaStream_t);
template void launch_higgs_kv_bwd<__half>(
    const __half*, const __half*, const uint8_t*, const uint8_t*,
    const float*, const float*, const __nv_bfloat16*, const __nv_bfloat16*,
    const float*, const float*, __half*, int64_t, int64_t, cudaStream_t);

}  // namespace higgs
}  // namespace megatron
