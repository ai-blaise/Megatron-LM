// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Full-head DSA indexer RoPE writer.
//
// The hot DSA indexer path only applies RoPE to the positional prefix of each
// 128-wide indexer head, then concatenates the untouched non-positional suffix.
// That split/concat materializes another full [S, B, H, D] tensor at exactly
// the point where StreamBP replay is tightest on memory.  This kernel writes
// the final head directly: one warp owns one logical [S, B, H] row, computes the
// RoPE prefix in registers, and copies the suffix.  Eight rows per CTA keeps
// the memory-dominated work coalesced without creating one CTA per row.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace megatron {
namespace hisa_indexer {

namespace {

constexpr int kDTypeF32 = 0;
constexpr int kDTypeBF16 = 1;
constexpr int kDTypeF16 = 2;
constexpr int kRowsPerBlock = 8;
constexpr int kThreadsPerBlock = kRowsPerBlock * 32;

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

__device__ __forceinline__ int rope_partner(int d, int pe_dim, int interleaved) {
  if (interleaved) {
    return d ^ 1;
  }
  const int half = pe_dim >> 1;
  return d < half ? d + half : d - half;
}

__device__ __forceinline__ float rope_sign(int d, int pe_dim, int interleaved) {
  if (interleaved) {
    return (d & 1) == 0 ? -1.0f : 1.0f;
  }
  return d < (pe_dim >> 1) ? -1.0f : 1.0f;
}

__device__ __forceinline__ int64_t freq_offset(
    int s,
    int b,
    int h,
    int d,
    int freq_seqlen,
    int freq_batch,
    int freq_heads,
    int64_t freq_stride_s,
    int64_t freq_stride_b,
    int64_t freq_stride_h,
    int64_t freq_stride_d) {
  const int fs = freq_seqlen == 1 ? 0 : s;
  const int fb = freq_batch == 1 ? 0 : b;
  const int fh = freq_heads == 1 ? 0 : h;
  return static_cast<int64_t>(fs) * freq_stride_s +
         static_cast<int64_t>(fb) * freq_stride_b +
         static_cast<int64_t>(fh) * freq_stride_h +
         static_cast<int64_t>(d) * freq_stride_d;
}

__global__ void dsa_indexer_rope_fwd_kernel(
    const void* __restrict__ x,
    const void* __restrict__ freqs,
    void* __restrict__ out,
    int64_t total_rows,
    int batch,
    int heads,
    int head_dim,
    int pe_dim,
    int freq_seqlen,
    int freq_batch,
    int freq_heads,
    int64_t freq_stride_s,
    int64_t freq_stride_b,
    int64_t freq_stride_h,
    int64_t freq_stride_d,
    int x_dtype,
    int freqs_dtype,
    float mscale,
    int interleaved) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + warp;
  if (row >= total_rows) {
    return;
  }

  const int h = static_cast<int>(row % heads);
  const int64_t row_div_h = row / heads;
  const int b = static_cast<int>(row_div_h % batch);
  const int s = static_cast<int>(row_div_h / batch);
  const int64_t row_base = row * head_dim;

  for (int d = lane; d < head_dim; d += 32) {
    float y;
    if (d < pe_dim) {
      const int partner = rope_partner(d, pe_dim, interleaved);
      float angle = load_typed(
          freqs,
          freq_offset(
              s, b, h, d, freq_seqlen, freq_batch, freq_heads, freq_stride_s,
              freq_stride_b, freq_stride_h, freq_stride_d),
          freqs_dtype);
      float sn;
      float cs;
      sincosf(angle, &sn, &cs);
      const float x_d = load_typed(x, row_base + d, x_dtype);
      const float x_partner = load_typed(x, row_base + partner, x_dtype);
      y = mscale * (x_d * cs + rope_sign(d, pe_dim, interleaved) * x_partner * sn);
    } else {
      y = load_typed(x, row_base + d, x_dtype);
    }
    store_typed(out, row_base + d, y, x_dtype);
  }
}

__global__ void dsa_indexer_rope_fwd_inplace_kernel(
    void* __restrict__ x,
    const void* __restrict__ freqs,
    int64_t total_rows,
    int batch,
    int heads,
    int head_dim,
    int pe_dim,
    int freq_seqlen,
    int freq_batch,
    int freq_heads,
    int64_t freq_stride_s,
    int64_t freq_stride_b,
    int64_t freq_stride_h,
    int64_t freq_stride_d,
    int x_dtype,
    int freqs_dtype,
    float mscale,
    int interleaved) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + warp;
  if (row >= total_rows) {
    return;
  }

  const int h = static_cast<int>(row % heads);
  const int64_t row_div_h = row / heads;
  const int b = static_cast<int>(row_div_h % batch);
  const int s = static_cast<int>(row_div_h / batch);
  const int64_t row_base = row * head_dim;
  const int pair_count = pe_dim >> 1;

  for (int pair = lane; pair < pair_count; pair += 32) {
    const int d0 = interleaved ? (pair << 1) : pair;
    const int d1 = interleaved ? ((pair << 1) + 1) : (pair + pair_count);

    const float angle0 = load_typed(
        freqs,
        freq_offset(
            s, b, h, d0, freq_seqlen, freq_batch, freq_heads, freq_stride_s,
            freq_stride_b, freq_stride_h, freq_stride_d),
        freqs_dtype);
    const float angle1 = load_typed(
        freqs,
        freq_offset(
            s, b, h, d1, freq_seqlen, freq_batch, freq_heads, freq_stride_s,
            freq_stride_b, freq_stride_h, freq_stride_d),
        freqs_dtype);

    float sin0;
    float cos0;
    float sin1;
    float cos1;
    sincosf(angle0, &sin0, &cos0);
    sincosf(angle1, &sin1, &cos1);

    const float x0 = load_typed(x, row_base + d0, x_dtype);
    const float x1 = load_typed(x, row_base + d1, x_dtype);
    const float y0 = mscale * (x0 * cos0 + rope_sign(d0, pe_dim, interleaved) * x1 * sin0);
    const float y1 = mscale * (x1 * cos1 + rope_sign(d1, pe_dim, interleaved) * x0 * sin1);

    store_typed(x, row_base + d0, y0, x_dtype);
    store_typed(x, row_base + d1, y1, x_dtype);
  }
}

__global__ void dsa_indexer_rope_bwd_kernel(
    const void* __restrict__ grad_out,
    const void* __restrict__ freqs,
    void* __restrict__ grad_x,
    int64_t total_rows,
    int batch,
    int heads,
    int head_dim,
    int pe_dim,
    int freq_seqlen,
    int freq_batch,
    int freq_heads,
    int64_t freq_stride_s,
    int64_t freq_stride_b,
    int64_t freq_stride_h,
    int64_t freq_stride_d,
    int grad_dtype,
    int freqs_dtype,
    float mscale,
    int interleaved) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + warp;
  if (row >= total_rows) {
    return;
  }

  const int h = static_cast<int>(row % heads);
  const int64_t row_div_h = row / heads;
  const int b = static_cast<int>(row_div_h % batch);
  const int s = static_cast<int>(row_div_h / batch);
  const int64_t row_base = row * head_dim;

  for (int d = lane; d < head_dim; d += 32) {
    float gx;
    if (d < pe_dim) {
      const int partner = rope_partner(d, pe_dim, interleaved);
      const float angle_d = load_typed(
          freqs,
          freq_offset(
              s, b, h, d, freq_seqlen, freq_batch, freq_heads, freq_stride_s,
              freq_stride_b, freq_stride_h, freq_stride_d),
          freqs_dtype);
      const float angle_partner = load_typed(
          freqs,
          freq_offset(
              s, b, h, partner, freq_seqlen, freq_batch, freq_heads, freq_stride_s,
              freq_stride_b, freq_stride_h, freq_stride_d),
          freqs_dtype);
      float sin_d;
      float cos_d;
      float sin_partner;
      float cos_partner;
      sincosf(angle_d, &sin_d, &cos_d);
      sincosf(angle_partner, &sin_partner, &cos_partner);
      (void)sin_d;
      (void)cos_partner;
      const float gy_d = load_typed(grad_out, row_base + d, grad_dtype);
      const float gy_partner = load_typed(grad_out, row_base + partner, grad_dtype);
      gx = mscale *
           (gy_d * cos_d +
            gy_partner * rope_sign(partner, pe_dim, interleaved) * sin_partner);
    } else {
      gx = load_typed(grad_out, row_base + d, grad_dtype);
    }
    store_typed(grad_x, row_base + d, gx, grad_dtype);
  }
}

}  // namespace

void launch_dsa_indexer_rope_fwd(
    const void* x,
    const void* freqs,
    void* out,
    int64_t total_rows,
    int seqlen,
    int batch,
    int heads,
    int head_dim,
    int pe_dim,
    int freq_seqlen,
    int freq_batch,
    int freq_heads,
    int64_t freq_stride_s,
    int64_t freq_stride_b,
    int64_t freq_stride_h,
    int64_t freq_stride_d,
    int x_dtype,
    int freqs_dtype,
    float mscale,
    int interleaved,
    cudaStream_t stream) {
  (void)seqlen;
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((total_rows + kRowsPerBlock - 1) / kRowsPerBlock));
  dsa_indexer_rope_fwd_kernel<<<grid, block, 0, stream>>>(
      x,
      freqs,
      out,
      total_rows,
      batch,
      heads,
      head_dim,
      pe_dim,
      freq_seqlen,
      freq_batch,
      freq_heads,
      freq_stride_s,
      freq_stride_b,
      freq_stride_h,
      freq_stride_d,
      x_dtype,
      freqs_dtype,
      mscale,
      interleaved);
}

void launch_dsa_indexer_rope_fwd_inplace(
    void* x,
    const void* freqs,
    int64_t total_rows,
    int seqlen,
    int batch,
    int heads,
    int head_dim,
    int pe_dim,
    int freq_seqlen,
    int freq_batch,
    int freq_heads,
    int64_t freq_stride_s,
    int64_t freq_stride_b,
    int64_t freq_stride_h,
    int64_t freq_stride_d,
    int x_dtype,
    int freqs_dtype,
    float mscale,
    int interleaved,
    cudaStream_t stream) {
  (void)seqlen;
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((total_rows + kRowsPerBlock - 1) / kRowsPerBlock));
  dsa_indexer_rope_fwd_inplace_kernel<<<grid, block, 0, stream>>>(
      x,
      freqs,
      total_rows,
      batch,
      heads,
      head_dim,
      pe_dim,
      freq_seqlen,
      freq_batch,
      freq_heads,
      freq_stride_s,
      freq_stride_b,
      freq_stride_h,
      freq_stride_d,
      x_dtype,
      freqs_dtype,
      mscale,
      interleaved);
}

void launch_dsa_indexer_rope_bwd(
    const void* grad_out,
    const void* freqs,
    void* grad_x,
    int64_t total_rows,
    int seqlen,
    int batch,
    int heads,
    int head_dim,
    int pe_dim,
    int freq_seqlen,
    int freq_batch,
    int freq_heads,
    int64_t freq_stride_s,
    int64_t freq_stride_b,
    int64_t freq_stride_h,
    int64_t freq_stride_d,
    int grad_dtype,
    int freqs_dtype,
    float mscale,
    int interleaved,
    cudaStream_t stream) {
  (void)seqlen;
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((total_rows + kRowsPerBlock - 1) / kRowsPerBlock));
  dsa_indexer_rope_bwd_kernel<<<grid, block, 0, stream>>>(
      grad_out,
      freqs,
      grad_x,
      total_rows,
      batch,
      heads,
      head_dim,
      pe_dim,
      freq_seqlen,
      freq_batch,
      freq_heads,
      freq_stride_s,
      freq_stride_b,
      freq_stride_h,
      freq_stride_d,
      grad_dtype,
      freqs_dtype,
      mscale,
      interleaved);
}

}  // namespace hisa_indexer
}  // namespace megatron
