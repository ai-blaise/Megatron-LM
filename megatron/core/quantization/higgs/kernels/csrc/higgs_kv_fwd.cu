// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Fused forward kernel for the 2-bit HIGGS dense MLA-latent KV fake-quant.
// One CUDA block per token; 512 threads per block (one per latent
// coordinate). The kernel produces:
//   out[row, :]          -- dequantized fake-quant output, same dtype as x
//   indices[row, :]      -- uint8 codebook index per pair (256 pairs/token)
//   ste_mask[row, :]     -- 0/1 byte per coord; backward gates STE on this
//   rot_norm[row]        -- ||rotated(x)|| in fp32, saved for backward
//   scale[row]           -- ||rotated|| / sqrt(N), the per-token block scale
//   rotated_save[row, :] -- bf16 cached rotated tensor (saves a backward FWHT)
//   recon_unit_save[row, :] -- bf16 cached codebook reconstruction (un-scaled)
//
// IKP region annotations (see ../build.py for the IKP_ENABLED flag):
//   region "rotate"        : forward FWHT
//   region "norm"          : sum-of-squares reduction -> scale
//   region "normalize"     : per-thread scale division + SMEM exchange
//   region "quant"         : pair-wise codebook NN lookup + index store
//   region "recon_unit"    : codebook lookup back to per-thread coord
//   region "inv_rot"       : inverse FWHT (involutory => same kernel as forward)
//   region "writeout"      : final scale + store
//
// Math reference: see ../../reference.py (higgs_forward); this kernel is
// the bit-identical fp32 implementation of the same algorithm.
//
// Algorithmic source: optimization-playground main HEAD ``2e2f51717``,
// python/sglang/jit_kernel/csrc/quantization/higgs_dense_2bit_kv.cuh.

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

template <typename scalar_t>
__global__ void higgs_kv_fwd_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    uint8_t* __restrict__ indices,       // (N, kNumPairs)
    uint8_t* __restrict__ ste_mask,      // (N, kLatentDim)
    float* __restrict__ rot_norm_out,    // (N,)
    float* __restrict__ scale_out,       // (N,)
    __nv_bfloat16* __restrict__ rotated_save,     // (N, kLatentDim) or null
    __nv_bfloat16* __restrict__ recon_unit_save,  // (N, kLatentDim) or null
    const float* __restrict__ codebook,           // (kCodebookSize, kPairDim)
    const float* __restrict__ codebook_norm_sq,   // (kCodebookSize,)
    int64_t num_rows,
    int64_t row_stride) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  __shared__ float buf[kLatentDim];     // FWHT working area
  __shared__ float reduce_scratch[64];  // block_reduce_sum partials
  __shared__ float scale_sh;
  __shared__ float rot_norm_sh;

  const scalar_t* x_row = x + row * row_stride;
  scalar_t* out_row = out + row * row_stride;
  uint8_t* indices_row = indices + row * kNumPairs;
  uint8_t* mask_row = ste_mask + row * kLatentDim;
  (void)codebook;
  (void)codebook_norm_sq;

  // region: rotate -- forward orthonormal FWHT
  HG_REGION_BEGIN(rotate);
  const float xin = load_as_float(x_row + tid);
  const float xrot = fwht_512(xin, buf) * kInvSqrtLatentDim;
  HG_REGION_END(rotate);

  // region: norm -- sum of squares reduction -> per-token scale
  HG_REGION_BEGIN(norm);
  const float sum_sq = block_reduce_sum_512(xrot * xrot, reduce_scratch);
  if (tid == 0) {
    const float rn = sqrtf(fmaxf(sum_sq, 1.0e-32f));
    rot_norm_sh = rn;
    scale_sh = rn * kInvSqrtLatentDim;
    rot_norm_out[row] = rn;
    scale_out[row] = scale_sh;
  }
  __syncthreads();
  const float scale = scale_sh;
  HG_REGION_END(norm);

  // Optionally save rotated[row, tid] as bf16 so the backward can skip its
  // own input-side FWHT recompute.
  if (rotated_save != nullptr) {
    rotated_save[row * kLatentDim + tid] = __float2bfloat16(xrot);
  }

  // region: normalize -- per-coord scale division. Adjacent pair values are
  // exchanged with warp shuffles, matching the accepted OP HIGGS B200 path.
  HG_REGION_BEGIN(normalize);
  const float xnorm = xrot / fmaxf(scale, 1.0e-30f);
  HG_REGION_END(normalize);

  // region: quant -- pair-wise codebook nearest-neighbour lookup
  HG_REGION_BEGIN(quant);
  uint32_t cb_idx = 0;
  const float x1 = __shfl_xor_sync(0xffffffff, xnorm, 1);
  if ((tid & 1) == 0) {
    cb_idx = nearest_codebook_index_const(xnorm, x1);
    indices_row[tid >> 1] = static_cast<uint8_t>(cb_idx);
  }
  const uint32_t peer_idx = __shfl_xor_sync(0xffffffff, cb_idx, 1);
  const uint32_t cb_idx_pair = (tid & 1) ? peer_idx : cb_idx;

  // Saturating-STE mask is 1 everywhere by construction for the EDEN2-16
  // lattice (no bounded support). We still write the mask so the backward
  // kernel reads it uniformly and so future bounded-codebook variants can
  // change saturation behaviour without changing the slot layout.
  mask_row[tid] = static_cast<uint8_t>(1);
  HG_REGION_END(quant);

  // region: recon_unit -- codebook lookup back to per-thread coord
  HG_REGION_BEGIN(recon_unit);
  const int coord = tid & 1;
  const float recon_unit = eden2_16_codebook_value(cb_idx_pair, coord);
  if (recon_unit_save != nullptr) {
    recon_unit_save[row * kLatentDim + tid] = __float2bfloat16(recon_unit);
  }
  const float rot_recon = scale * recon_unit;
  HG_REGION_END(recon_unit);

  __syncthreads();  // sync before reusing buf for the inverse FWHT

  // region: inv_rot -- inverse FWHT (involutory; ×1/sqrt(N) keeps it so)
  HG_REGION_BEGIN(inv_rot);
  const float y = fwht_512(rot_recon, buf) * kInvSqrtLatentDim;
  HG_REGION_END(inv_rot);

  // region: writeout
  HG_REGION_BEGIN(writeout);
  store_from_float(out_row + tid, y);
  HG_REGION_END(writeout);
}

template <typename scalar_t>
void launch_higgs_kv_fwd(
    const scalar_t* x,
    scalar_t* out,
    uint8_t* indices,
    uint8_t* ste_mask,
    float* rot_norm_out,
    float* scale_out,
    __nv_bfloat16* rotated_save,
    __nv_bfloat16* recon_unit_save,
    const float* codebook,
    const float* codebook_norm_sq,
    int64_t num_rows,
    int64_t row_stride,
    cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(num_rows));
  const dim3 block(kLatentDim);
  higgs_kv_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
      x, out, indices, ste_mask, rot_norm_out, scale_out,
      rotated_save, recon_unit_save,
      codebook, codebook_norm_sq, num_rows, row_stride);
}

template void launch_higgs_kv_fwd<float>(
    const float*, float*, uint8_t*, uint8_t*, float*, float*,
    __nv_bfloat16*, __nv_bfloat16*, const float*, const float*,
    int64_t, int64_t, cudaStream_t);
template void launch_higgs_kv_fwd<__nv_bfloat16>(
    const __nv_bfloat16*, __nv_bfloat16*, uint8_t*, uint8_t*, float*, float*,
    __nv_bfloat16*, __nv_bfloat16*, const float*, const float*,
    int64_t, int64_t, cudaStream_t);
template void launch_higgs_kv_fwd<__half>(
    const __half*, __half*, uint8_t*, uint8_t*, float*, float*,
    __nv_bfloat16*, __nv_bfloat16*, const float*, const float*,
    int64_t, int64_t, cudaStream_t);

}  // namespace higgs
}  // namespace megatron
