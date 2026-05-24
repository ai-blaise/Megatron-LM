// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// CUDA K/V-gradient reducer for fused sparse DSA attention.
//
// This is intentionally a lower-level path than the Triton row replay.  The
// launch maps one warp to one selected edge inside a small query/top-k tile,
// computes the attention backward scalars, then uses a representative edge per
// selected key inside the CTA to accumulate duplicate-key contributions before
// issuing global atomics.  The selector/top-k ordering remains unchanged.

#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>

namespace megatron {
namespace hisa_indexer {

namespace {

constexpr int kDTypeF32 = 0;
constexpr int kDTypeBF16 = 1;
constexpr int kDTypeF16 = 2;

constexpr int kTopkI16 = 0;
constexpr int kTopkI32 = 1;
constexpr int kTopkI64 = 2;
constexpr uint64_t kInvalidEdgeKey = 0xffffffffffffffffULL;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ float load_typed(const void* ptr, int64_t idx, int dtype) {
  if (dtype == kDTypeF32) {
    return reinterpret_cast<const float*>(ptr)[idx];
  }
  if (dtype == kDTypeBF16) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
  }
  return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

__device__ __forceinline__ bool finite_float(float v) {
  return isfinite(v);
}

__device__ __forceinline__ bool valid_lse(float v) {
  return finite_float(v) && v > -3.0e38f;
}

__device__ __forceinline__ int32_t load_topk(const void* ptr, int64_t idx, int dtype) {
  if (dtype == kTopkI16) {
    return static_cast<int32_t>(reinterpret_cast<const int16_t*>(ptr)[idx]);
  }
  if (dtype == kTopkI32) {
    return reinterpret_cast<const int32_t*>(ptr)[idx];
  }
  return static_cast<int32_t>(reinterpret_cast<const int64_t*>(ptr)[idx]);
}

__device__ __forceinline__ void atomic_add_typed(void* ptr, int64_t idx, float v, int dtype) {
  if (v == 0.0f) {
    return;
  }
  if (dtype == kDTypeF32) {
    atomicAdd(reinterpret_cast<float*>(ptr) + idx, v);
  } else if (dtype == kDTypeBF16) {
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(ptr) + idx, __float2bfloat16(v));
  } else {
    atomicAdd(reinterpret_cast<__half*>(ptr) + idx, __float2half(v));
  }
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

template <int WARPS>
__global__ void dsa_split_qk_fwd_row_kernel(
    const void* __restrict__ query_nope,      // [Q, B, H, D]
    const void* __restrict__ query_pe,        // [Q, B, H, P]
    const void* __restrict__ key_nope,        // [S, B, H, D]
    const void* __restrict__ key_pe,          // [S, B, KPH, P]
    const void* __restrict__ value,           // [S, B, H, V]
    const void* __restrict__ topk,            // [B, Q, K]
    const int64_t* __restrict__ query_pos,    // [Q], optional when has_positions=1
    const int64_t* __restrict__ key_pos,      // [S], optional when has_positions=1
    void* __restrict__ output,                // [Q, B, H, V]
    float* __restrict__ lse,                  // [B * Q, H]
    float* __restrict__ teacher_probs,        // [B * Q, K], optional
    float* __restrict__ teacher_score_scratch,// [B * Q, H, K], optional
    int Q,
    int B,
    int S,
    int H,
    int D,
    int P,
    int KPH,
    int V,
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
    int scalar_dtype,
    int topk_dtype,
    int has_positions,
    int emit_teacher,
    int use_teacher_score_scratch) {
  static_assert(WARPS > 0 && WARPS <= 16, "split-QK row forward supports 1..16 warps");
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (row >= Q || batch >= B || warp >= WARPS) {
    return;
  }

  extern __shared__ float smem[];
  float* q_nope_s = smem;
  float* q_pe_s = q_nope_s + static_cast<int64_t>(H) * D;
  float* teacher_s = q_pe_s + static_cast<int64_t>(H) * P;

  for (int idx = tid; idx < H * D; idx += WARPS * 32) {
    const int head = idx / D;
    const int d = idx - head * D;
    const int64_t q_base =
        static_cast<int64_t>(row) * query_nope_stride_s +
        static_cast<int64_t>(batch) * query_nope_stride_b +
        static_cast<int64_t>(head) * query_nope_stride_h;
    q_nope_s[idx] = load_typed(query_nope, q_base + d * query_nope_stride_d, scalar_dtype);
  }
  for (int idx = tid; idx < H * P; idx += WARPS * 32) {
    const int head = idx / P;
    const int d = idx - head * P;
    const int64_t q_base = ((static_cast<int64_t>(row) * B + batch) * H + head) * P;
    q_pe_s[idx] = load_typed(query_pe, q_base + d, scalar_dtype);
  }
  if (emit_teacher) {
    for (int slot = tid; slot < K; slot += WARPS * 32) {
      teacher_s[slot] = 0.0f;
    }
  }
  __syncthreads();

  const int64_t q_abs = has_positions ? query_pos[row] : static_cast<int64_t>(q_start + row);
  const int64_t row_batch = static_cast<int64_t>(batch) * Q + row;

  for (int head = warp; head < H; head += WARPS) {
    const int pe_head = min(head, KPH - 1);
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    for (int slot = 0; slot < K; ++slot) {
      int32_t selected = -1;
      if (lane == 0) {
        selected = load_topk(topk, row_batch * K + slot, topk_dtype);
      }
      selected = __shfl_sync(0xffffffff, selected, 0);
      const int safe_selected = selected > 0 ? selected : 0;
      bool valid = selected >= 0 && selected < S;
      if (valid) {
        const int64_t selected_abs = has_positions ? key_pos[safe_selected] : selected;
        valid = selected_abs <= q_abs;
      }

      const int64_t k_nope_base =
          static_cast<int64_t>(safe_selected) * key_nope_stride_s +
          static_cast<int64_t>(batch) * key_nope_stride_b +
          static_cast<int64_t>(head) * key_nope_stride_h;
      const int64_t k_pe_base =
          ((static_cast<int64_t>(safe_selected) * B + batch) * KPH + pe_head) * P;

      float score_nope = 0.0f;
      float score_pe = 0.0f;
      if (valid) {
        for (int d = lane; d < D; d += 32) {
          score_nope += q_nope_s[head * D + d] *
                        load_typed(key_nope, k_nope_base + d * key_nope_stride_d, scalar_dtype);
        }
        for (int d = lane; d < P; d += 32) {
          score_pe += q_pe_s[head * P + d] * load_typed(key_pe, k_pe_base + d, scalar_dtype);
        }
      }
      score_nope = warp_sum(score_nope);
      score_pe = warp_sum(score_pe);
      float score = (score_nope + score_pe) * softmax_scale;
      if (lane == 0) {
        score = valid && finite_float(score) ? score : -FLT_MAX;
        if (use_teacher_score_scratch) {
          teacher_score_scratch[(row_batch * H + head) * K + slot] = score;
        }
      }
      score = __shfl_sync(0xffffffff, score, 0);
      valid = valid && finite_float(score);
      if (valid) {
        const float m_new = fmaxf(m_i, score);
        l_i = l_i * expf(m_i - m_new) + expf(score - m_new);
        m_i = m_new;
      }
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int slot = 0; slot < K; ++slot) {
      int32_t selected = -1;
      if (lane == 0) {
        selected = load_topk(topk, row_batch * K + slot, topk_dtype);
      }
      selected = __shfl_sync(0xffffffff, selected, 0);
      const int safe_selected = selected > 0 ? selected : 0;
      bool valid = selected >= 0 && selected < S;
      if (valid) {
        const int64_t selected_abs = has_positions ? key_pos[safe_selected] : selected;
        valid = selected_abs <= q_abs;
      }

      float score = -FLT_MAX;
      if (use_teacher_score_scratch) {
        if (lane == 0) {
          score = teacher_score_scratch[(row_batch * H + head) * K + slot];
        }
        score = __shfl_sync(0xffffffff, score, 0);
        valid = valid && score > -3.0e38f && finite_float(score);
      } else {
        const int pe_head_local = min(head, KPH - 1);
        const int64_t k_nope_base =
            static_cast<int64_t>(safe_selected) * key_nope_stride_s +
            static_cast<int64_t>(batch) * key_nope_stride_b +
            static_cast<int64_t>(head) * key_nope_stride_h;
        const int64_t k_pe_base =
            ((static_cast<int64_t>(safe_selected) * B + batch) * KPH + pe_head_local) * P;
        float score_nope = 0.0f;
        float score_pe = 0.0f;
        if (valid) {
          for (int d = lane; d < D; d += 32) {
            score_nope += q_nope_s[head * D + d] *
                          load_typed(key_nope, k_nope_base + d * key_nope_stride_d, scalar_dtype);
          }
          for (int d = lane; d < P; d += 32) {
            score_pe += q_pe_s[head * P + d] * load_typed(key_pe, k_pe_base + d, scalar_dtype);
          }
        }
        score = (warp_sum(score_nope) + warp_sum(score_pe)) * softmax_scale;
        if (lane == 0) {
          score = valid && finite_float(score) ? score : -FLT_MAX;
        }
        score = __shfl_sync(0xffffffff, score, 0);
      }
      valid = valid && finite_float(score) && finite_float(m_i) && finite_float(l_i);
      float prob = (valid && l_i > 0.0f) ? expf(score - m_i) / l_i : 0.0f;
      if (!finite_float(prob)) {
        prob = 0.0f;
        valid = false;
      }
      if (emit_teacher && lane == 0 && prob != 0.0f) {
        atomicAdd(teacher_s + slot, prob);
      }

      if (valid && prob != 0.0f) {
        const int64_t v_base =
            static_cast<int64_t>(safe_selected) * value_stride_s +
            static_cast<int64_t>(batch) * value_stride_b +
            static_cast<int64_t>(head) * value_stride_h;
        const int d0 = lane;
        const int d1 = lane + 32;
        const int d2 = lane + 64;
        const int d3 = lane + 96;
        if (d0 < V) {
          acc0 += prob * load_typed(value, v_base + d0 * value_stride_v, scalar_dtype);
        }
        if (d1 < V) {
          acc1 += prob * load_typed(value, v_base + d1 * value_stride_v, scalar_dtype);
        }
        if (d2 < V) {
          acc2 += prob * load_typed(value, v_base + d2 * value_stride_v, scalar_dtype);
        }
        if (d3 < V) {
          acc3 += prob * load_typed(value, v_base + d3 * value_stride_v, scalar_dtype);
        }
      }
    }

    const int64_t out_base = ((static_cast<int64_t>(row) * B + batch) * H + head) * V;
    const int d0 = lane;
    const int d1 = lane + 32;
    const int d2 = lane + 64;
    const int d3 = lane + 96;
    if (d0 < V) {
      store_typed(output, out_base + d0, acc0, scalar_dtype);
    }
    if (d1 < V) {
      store_typed(output, out_base + d1, acc1, scalar_dtype);
    }
    if (d2 < V) {
      store_typed(output, out_base + d2, acc2, scalar_dtype);
    }
    if (d3 < V) {
      store_typed(output, out_base + d3, acc3, scalar_dtype);
    }
    if (lane == 0) {
      lse[row_batch * H + head] = l_i > 0.0f ? m_i + logf(l_i) : -FLT_MAX;
    }
  }

  __syncthreads();
  if (emit_teacher) {
    for (int slot = tid; slot < K; slot += WARPS * 32) {
      teacher_probs[row_batch * K + slot] = teacher_s[slot];
    }
  }
}

template <int TILE_Q, int TILE_K>
__global__ void dsa_sparse_kv_bwd_kernel(
    const void* __restrict__ query,       // [Q, B, H, D]
    const void* __restrict__ key,         // [S, B, H, D]
    const void* __restrict__ value,       // [S, B, H, V]
    const void* __restrict__ topk,        // [B, Q, K]
    const void* __restrict__ output,      // [Q, B, H, V]
    const float* __restrict__ lse,        // [B * Q, H]
    const void* __restrict__ grad_output, // [Q, B, H, V]
    void* __restrict__ grad_key,          // [S, B, H, D]
    void* __restrict__ grad_value,        // [S, B, H, V]
    float softmax_scale,
    int Q,
    int B,
    int S,
    int H,
    int D,
    int V,
    int K,
    int q_start,
    int scalar_dtype,
    int topk_dtype,
    int grad_dtype) {
  constexpr int EDGES = TILE_Q * TILE_K;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (warp >= EDGES) {
    return;
  }

  const int q_block = blockIdx.x;
  const int k_block = blockIdx.y;
  const int head_batch = blockIdx.z;
  const int head = head_batch % H;
  const int batch = head_batch / H;

  const int local_q = warp / TILE_K;
  const int local_k = warp - local_q * TILE_K;
  const int q = q_block * TILE_Q + local_q;
  const int slot = k_block * TILE_K + local_k;

  __shared__ int32_t selected_s[EDGES];
  __shared__ int32_t q_s[EDGES];
  __shared__ uint8_t valid_s[EDGES];
  __shared__ float prob_s[EDGES];
  __shared__ float ds_s[EDGES];

  const int edge = warp;
  int32_t selected = -1;
  bool valid = q < Q && slot < K;
  if (valid) {
    selected = load_topk(
        topk,
        (static_cast<int64_t>(batch) * Q + q) * K + slot,
        topk_dtype);
    valid = selected >= 0 && selected < S && selected <= q_start + q;
  }

  float dot_qk = 0.0f;
  float dot_vg = 0.0f;
  float dot_og = 0.0f;
  if (valid) {
    const int64_t q_base = ((static_cast<int64_t>(q) * B + batch) * H + head);
    const int64_t k_base = ((static_cast<int64_t>(selected) * B + batch) * H + head);
    for (int d = lane; d < D; d += 32) {
      dot_qk += load_typed(query, q_base * D + d, scalar_dtype) *
                load_typed(key, k_base * D + d, scalar_dtype);
    }
    for (int d = lane; d < V; d += 32) {
      const float go = load_typed(grad_output, q_base * V + d, scalar_dtype);
      dot_vg += load_typed(value, k_base * V + d, scalar_dtype) * go;
      dot_og += load_typed(output, q_base * V + d, scalar_dtype) * go;
    }
  }
  dot_qk = warp_sum(dot_qk);
  dot_vg = warp_sum(dot_vg);
  dot_og = warp_sum(dot_og);

  float prob = 0.0f;
  float ds = 0.0f;
  if (valid) {
    const float row_lse = lse[(static_cast<int64_t>(batch) * Q + q) * H + head];
    const float score = dot_qk * softmax_scale;
    valid = finite_float(score) && valid_lse(row_lse) && finite_float(dot_vg) &&
            finite_float(dot_og);
    if (valid) {
      prob = expf(score - row_lse);
      valid = finite_float(prob);
      ds = valid ? prob * (dot_vg - dot_og) * softmax_scale : 0.0f;
      valid = valid && finite_float(ds);
    }
  }

  if (lane == 0) {
    selected_s[edge] = selected;
    q_s[edge] = q;
    valid_s[edge] = valid ? 1 : 0;
    prob_s[edge] = prob;
    ds_s[edge] = ds;
  }
  __syncthreads();

  bool representative = valid;
#pragma unroll
  for (int prev = 0; prev < EDGES; ++prev) {
    if (prev < edge && valid_s[prev] && selected_s[prev] == selected) {
      representative = false;
    }
  }
  if (!representative) {
    return;
  }

  const int64_t selected_base = ((static_cast<int64_t>(selected) * B + batch) * H + head);
  for (int d = lane; d < D; d += 32) {
    float acc = 0.0f;
#pragma unroll
    for (int e = 0; e < EDGES; ++e) {
      if (valid_s[e] && selected_s[e] == selected) {
        const int qe = q_s[e];
        const int64_t q_base = ((static_cast<int64_t>(qe) * B + batch) * H + head);
        acc += ds_s[e] * load_typed(query, q_base * D + d, scalar_dtype);
      }
    }
    atomic_add_typed(grad_key, selected_base * D + d, acc, grad_dtype);
  }
  for (int d = lane; d < V; d += 32) {
    float acc = 0.0f;
#pragma unroll
    for (int e = 0; e < EDGES; ++e) {
      if (valid_s[e] && selected_s[e] == selected) {
        const int qe = q_s[e];
        const int64_t q_base = ((static_cast<int64_t>(qe) * B + batch) * H + head);
        acc += prob_s[e] * load_typed(grad_output, q_base * V + d, scalar_dtype);
      }
    }
    atomic_add_typed(grad_value, selected_base * V + d, acc, grad_dtype);
  }
}

template <int TILE_Q, int TILE_K>
__global__ void dsa_sparse_bwd_from_scores_kernel(
    const void* __restrict__ query,          // [Q, B, H, D]
    const void* __restrict__ key,            // [S, B, H, D]
    const void* __restrict__ value,          // [S, B, H, V]
    const void* __restrict__ topk,           // [B, Q, K]
    const float* __restrict__ selected_score, // [B * Q, H, K]
    const void* __restrict__ output,         // [Q, B, H, V]
    const float* __restrict__ lse,           // [B * Q, H]
    const void* __restrict__ grad_output,    // [Q, B, H, V]
    float* __restrict__ grad_query,          // [Q, B, H, D]
    void* __restrict__ grad_key,             // [S, B, H, D]
    void* __restrict__ grad_value,           // [S, B, H, V]
    int Q,
    int B,
    int S,
    int H,
    int D,
    int V,
    int K,
    float softmax_scale,
    int scalar_dtype,
    int topk_dtype,
    int grad_dtype) {
  constexpr int EDGES = TILE_Q * TILE_K;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (warp >= EDGES) {
    return;
  }

  const int q_block = blockIdx.x;
  const int k_block = blockIdx.y;
  const int head_batch = blockIdx.z;
  const int head = head_batch % H;
  const int batch = head_batch / H;

  const int local_q = warp / TILE_K;
  const int local_k = warp - local_q * TILE_K;
  const int q = q_block * TILE_Q + local_q;
  const int slot = k_block * TILE_K + local_k;

  __shared__ int32_t selected_s[EDGES];
  __shared__ int32_t q_s[EDGES];
  __shared__ uint8_t valid_s[EDGES];
  __shared__ float prob_s[EDGES];
  __shared__ float ds_s[EDGES];

  const int edge = warp;
  int32_t selected = -1;
  float score = -INFINITY;
  bool valid = q < Q && slot < K;
  if (valid) {
    selected = load_topk(
        topk,
        (static_cast<int64_t>(batch) * Q + q) * K + slot,
        topk_dtype);
    score = selected_score[(static_cast<int64_t>(batch) * Q + q) * H * K +
                           static_cast<int64_t>(head) * K + slot];
    valid = selected >= 0 && selected < S && score > -3.0e38f && finite_float(score);
  }

  float dot_vg = 0.0f;
  float dot_og = 0.0f;
  if (valid) {
    const int64_t q_base = ((static_cast<int64_t>(q) * B + batch) * H + head);
    const int64_t k_base = ((static_cast<int64_t>(selected) * B + batch) * H + head);
    for (int d = lane; d < V; d += 32) {
      const float go = load_typed(grad_output, q_base * V + d, scalar_dtype);
      dot_vg += load_typed(value, k_base * V + d, scalar_dtype) * go;
      dot_og += load_typed(output, q_base * V + d, scalar_dtype) * go;
    }
  }
  dot_vg = warp_sum(dot_vg);
  dot_og = warp_sum(dot_og);

  float prob = 0.0f;
  float ds = 0.0f;
  if (valid) {
    const float row_lse = lse[(static_cast<int64_t>(batch) * Q + q) * H + head];
    valid = valid_lse(row_lse) && finite_float(dot_vg) && finite_float(dot_og);
    if (valid) {
      prob = expf(score - row_lse);
      valid = finite_float(prob);
      ds = valid ? prob * (dot_vg - dot_og) * softmax_scale : 0.0f;
      valid = valid && finite_float(ds);
    }
  }

  if (lane == 0) {
    selected_s[edge] = selected;
    q_s[edge] = q;
    valid_s[edge] = valid ? 1 : 0;
    prob_s[edge] = prob;
    ds_s[edge] = ds;
  }
  __syncthreads();

  const bool q_representative = valid && local_k == 0;
  if (q_representative) {
    for (int d = lane; d < D; d += 32) {
      float acc = 0.0f;
#pragma unroll
      for (int e = 0; e < EDGES; ++e) {
        if (valid_s[e] && q_s[e] == q) {
          const int64_t selected_base =
              ((static_cast<int64_t>(selected_s[e]) * B + batch) * H + head);
          acc += ds_s[e] * load_typed(key, selected_base * D + d, scalar_dtype);
        }
      }
      atomicAdd(
          grad_query + ((static_cast<int64_t>(q) * B + batch) * H + head) * D + d,
          acc);
    }
  }

  bool key_representative = valid;
#pragma unroll
  for (int prev = 0; prev < EDGES; ++prev) {
    if (prev < edge && valid_s[prev] && selected_s[prev] == selected) {
      key_representative = false;
    }
  }
  if (!key_representative) {
    return;
  }

  const int64_t selected_base = ((static_cast<int64_t>(selected) * B + batch) * H + head);
  for (int d = lane; d < D; d += 32) {
    float acc = 0.0f;
#pragma unroll
    for (int e = 0; e < EDGES; ++e) {
      if (valid_s[e] && selected_s[e] == selected) {
        const int qe = q_s[e];
        const int64_t q_base = ((static_cast<int64_t>(qe) * B + batch) * H + head);
        acc += ds_s[e] * load_typed(query, q_base * D + d, scalar_dtype);
      }
    }
    atomic_add_typed(grad_key, selected_base * D + d, acc, grad_dtype);
  }
  for (int d = lane; d < V; d += 32) {
    float acc = 0.0f;
#pragma unroll
    for (int e = 0; e < EDGES; ++e) {
      if (valid_s[e] && selected_s[e] == selected) {
        const int qe = q_s[e];
        const int64_t q_base = ((static_cast<int64_t>(qe) * B + batch) * H + head);
        acc += prob_s[e] * load_typed(grad_output, q_base * V + d, scalar_dtype);
      }
    }
    atomic_add_typed(grad_value, selected_base * V + d, acc, grad_dtype);
  }
}

template <int WARPS>
__global__ void dsa_sparse_bwd_from_scores_row_kernel(
    const void* __restrict__ query,           // [Q, B, H, D]
    const void* __restrict__ key,             // [S, B, H, D]
    const void* __restrict__ value,           // [S, B, H, V]
    const void* __restrict__ topk,            // [B, Q, K]
    const float* __restrict__ selected_score, // [B * Q, H, K]
    const void* __restrict__ output,          // [Q, B, H, V]
    const float* __restrict__ lse,            // [B * Q, H]
    const void* __restrict__ grad_output,     // [Q, B, H, V]
    float* __restrict__ grad_query,           // [Q, B, H, D]
    void* __restrict__ grad_key,              // [S, B, H, D]
    void* __restrict__ grad_value,            // [S, B, H, V]
    int Q,
    int B,
    int S,
    int H,
    int D,
    int V,
    int K,
    float softmax_scale,
    int scalar_dtype,
    int topk_dtype,
    int grad_dtype) {
  static_assert(WARPS > 0 && WARPS <= 16, "row scheduler supports 1..16 warps");
  constexpr int kMaxDim = 256;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (warp >= WARPS) {
    return;
  }

  const int row = blockIdx.x;
  const int head_batch = blockIdx.y;
  const int head = head_batch % H;
  const int batch = head_batch / H;
  if (row >= Q || D > kMaxDim || V > kMaxDim) {
    return;
  }

  __shared__ float q_s[kMaxDim];
  __shared__ float go_s[kMaxDim];
  __shared__ float grad_q_s[WARPS][kMaxDim];
  __shared__ float delta_s;

  const int64_t q_base = ((static_cast<int64_t>(row) * B + batch) * H + head);
  for (int d = tid; d < D; d += WARPS * 32) {
    q_s[d] = load_typed(query, q_base * D + d, scalar_dtype);
  }
  for (int d = tid; d < V; d += WARPS * 32) {
    go_s[d] = load_typed(grad_output, q_base * V + d, scalar_dtype);
  }
  for (int idx = tid; idx < WARPS * kMaxDim; idx += WARPS * 32) {
    reinterpret_cast<float*>(grad_q_s)[idx] = 0.0f;
  }
  __syncthreads();

  float delta = 0.0f;
  if (warp == 0) {
    for (int d = lane; d < V; d += 32) {
      delta += load_typed(output, q_base * V + d, scalar_dtype) * go_s[d];
    }
    delta = warp_sum(delta);
    if (lane == 0) {
      delta_s = delta;
    }
  }
  __syncthreads();
  delta = delta_s;
  const float row_lse = lse[(static_cast<int64_t>(batch) * Q + row) * H + head];

  for (int slot = warp; slot < K; slot += WARPS) {
    int32_t selected = -1;
    float score = -INFINITY;
    if (lane == 0) {
      selected = load_topk(
          topk,
          (static_cast<int64_t>(batch) * Q + row) * K + slot,
          topk_dtype);
      score = selected_score[(static_cast<int64_t>(batch) * Q + row) * H * K +
                             static_cast<int64_t>(head) * K + slot];
    }
    selected = __shfl_sync(0xffffffff, selected, 0);
    score = __shfl_sync(0xffffffff, score, 0);
    bool valid = selected >= 0 && selected < S && score > -3.0e38f &&
                 finite_float(score) && valid_lse(row_lse) && finite_float(delta);
    const int safe_selected = selected > 0 ? selected : 0;
    const int64_t selected_base =
        ((static_cast<int64_t>(safe_selected) * B + batch) * H + head);

    float dot_vg = 0.0f;
    if (valid) {
      for (int d = lane; d < V; d += 32) {
        dot_vg += load_typed(value, selected_base * V + d, scalar_dtype) * go_s[d];
      }
    }
    dot_vg = warp_sum(dot_vg);
    dot_vg = __shfl_sync(0xffffffff, dot_vg, 0);

    valid = valid && finite_float(dot_vg);
    const float prob = valid ? expf(score - row_lse) : 0.0f;
    valid = valid && finite_float(prob);
    const float ds = valid ? prob * (dot_vg - delta) * softmax_scale : 0.0f;
    valid = valid && finite_float(ds);

    if (valid) {
      for (int d = lane; d < D; d += 32) {
        const float k_val = load_typed(key, selected_base * D + d, scalar_dtype);
        grad_q_s[warp][d] += ds * k_val;
        atomic_add_typed(grad_key, selected_base * D + d, ds * q_s[d], grad_dtype);
      }
      for (int d = lane; d < V; d += 32) {
        atomic_add_typed(grad_value, selected_base * V + d, prob * go_s[d], grad_dtype);
      }
    }
  }
  __syncthreads();

  for (int d = tid; d < D; d += WARPS * 32) {
    float acc = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      acc += grad_q_s[w][d];
    }
    grad_query[q_base * D + d] = acc;
  }
}

template <int WARPS>
__global__ void dsa_split_qk_bwd_row_kernel(
    const void* __restrict__ query_nope,      // [Q, B, H, D]
    const void* __restrict__ query_pe,        // [Q, B, H, P]
    const void* __restrict__ key_nope,        // [S, B, H, D]
    const void* __restrict__ key_pe,          // [S, B, KPH, P]
    const void* __restrict__ value,           // [S, B, H, V]
    const void* __restrict__ topk,            // [B, Q, K]
    const int64_t* __restrict__ query_pos,    // [Q], optional when has_positions=1
    const int64_t* __restrict__ key_pos,      // [S], optional when has_positions=1
    const void* __restrict__ output,          // [Q, B, H, V]
    const float* __restrict__ lse,            // [B * Q, H]
    const void* __restrict__ grad_output,     // [Q, B, H, V]
    void* __restrict__ grad_query_nope,       // [Q, B, H, D]
    void* __restrict__ grad_query_pe,         // [Q, B, H, P]
    void* __restrict__ grad_key_nope,         // [local_S, B, H, D]
    void* __restrict__ grad_key_pe,           // [local_S, B, KPH, P]
    void* __restrict__ grad_value,            // [local_S, B, H, V]
    int Q,
    int B,
    int S,
    int H,
    int D,
    int P,
    int KPH,
    int V,
    int K,
    int q_start,
    int kv_start,
    int kv_end,
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
    int64_t grad_key_nope_stride_s,
    int64_t grad_key_nope_stride_b,
    int64_t grad_key_nope_stride_h,
    int64_t grad_key_nope_stride_d,
    int64_t grad_value_stride_s,
    int64_t grad_value_stride_b,
    int64_t grad_value_stride_h,
    int64_t grad_value_stride_v,
    float softmax_scale,
    int scalar_dtype,
    int topk_dtype,
    int grad_dtype,
    int has_positions,
    int emit_query,
    int emit_key_nope,
    int emit_key_pe,
    int emit_value) {
  static_assert(WARPS > 0 && WARPS <= 16, "split-QK row scheduler supports 1..16 warps");
  constexpr int kMaxDim = 256;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (warp >= WARPS) {
    return;
  }

  const int row = blockIdx.x;
  const int head_batch = blockIdx.y;
  const int head = head_batch % H;
  const int batch = head_batch / H;
  if (row >= Q || D > kMaxDim || P > kMaxDim || V > kMaxDim) {
    return;
  }

  __shared__ float q_nope_s[kMaxDim];
  __shared__ float q_pe_s[kMaxDim];
  __shared__ float go_s[kMaxDim];
  __shared__ float grad_q_nope_s[WARPS][kMaxDim];
  __shared__ float grad_q_pe_s[WARPS][kMaxDim];
  __shared__ float delta_s;
  __shared__ int64_t q_abs_s;

  const int pe_head = min(head, KPH - 1);
  const int64_t q_base = ((static_cast<int64_t>(row) * B + batch) * H + head);
  const int64_t q_nope_base =
      static_cast<int64_t>(row) * query_nope_stride_s +
      static_cast<int64_t>(batch) * query_nope_stride_b +
      static_cast<int64_t>(head) * query_nope_stride_h;
  for (int d = tid; d < D; d += WARPS * 32) {
    q_nope_s[d] =
        load_typed(query_nope, q_nope_base + d * query_nope_stride_d, scalar_dtype);
  }
  for (int d = tid; d < P; d += WARPS * 32) {
    q_pe_s[d] = load_typed(query_pe, q_base * P + d, scalar_dtype);
  }
  for (int d = tid; d < V; d += WARPS * 32) {
    go_s[d] = load_typed(grad_output, q_base * V + d, scalar_dtype);
  }
  for (int idx = tid; idx < WARPS * kMaxDim; idx += WARPS * 32) {
    reinterpret_cast<float*>(grad_q_nope_s)[idx] = 0.0f;
  }
  for (int idx = tid; idx < WARPS * kMaxDim; idx += WARPS * 32) {
    reinterpret_cast<float*>(grad_q_pe_s)[idx] = 0.0f;
  }
  if (tid == 0) {
    q_abs_s = has_positions ? query_pos[row] : static_cast<int64_t>(q_start + row);
  }
  __syncthreads();

  float delta = 0.0f;
  if (warp == 0) {
    for (int d = lane; d < V; d += 32) {
      delta += load_typed(output, q_base * V + d, scalar_dtype) * go_s[d];
    }
    delta = warp_sum(delta);
    if (lane == 0) {
      delta_s = delta;
    }
  }
  __syncthreads();
  delta = delta_s;
  const int64_t q_abs = q_abs_s;
  const float row_lse = lse[(static_cast<int64_t>(batch) * Q + row) * H + head];

  for (int slot = warp; slot < K; slot += WARPS) {
    int32_t selected = -1;
    if (lane == 0) {
      selected = load_topk(
          topk,
          (static_cast<int64_t>(batch) * Q + row) * K + slot,
          topk_dtype);
    }
    selected = __shfl_sync(0xffffffff, selected, 0);
    const int safe_selected = selected > 0 ? selected : 0;
    bool valid = selected >= 0 && selected < S;
    if (valid) {
      const int64_t selected_abs = has_positions ? key_pos[safe_selected] : selected;
      valid = selected_abs <= q_abs;
    }

    const int64_t k_nope_base =
        static_cast<int64_t>(safe_selected) * key_nope_stride_s +
        static_cast<int64_t>(batch) * key_nope_stride_b +
        static_cast<int64_t>(head) * key_nope_stride_h;
    const int64_t k_pe_base =
        ((static_cast<int64_t>(safe_selected) * B + batch) * KPH + pe_head);

    float score_nope = 0.0f;
    float score_pe = 0.0f;
    if (valid) {
      for (int d = lane; d < D; d += 32) {
        score_nope += q_nope_s[d] *
                      load_typed(key_nope, k_nope_base + d * key_nope_stride_d, scalar_dtype);
      }
      for (int d = lane; d < P; d += 32) {
        score_pe += q_pe_s[d] * load_typed(key_pe, k_pe_base * P + d, scalar_dtype);
      }
    }
    score_nope = warp_sum(score_nope);
    score_pe = warp_sum(score_pe);
    float score = (score_nope + score_pe) * softmax_scale;
    score = __shfl_sync(0xffffffff, score, 0);
    valid = valid && finite_float(score) && valid_lse(row_lse) && finite_float(delta);

    float dot_vg = 0.0f;
    if (valid) {
      const int64_t v_base =
          static_cast<int64_t>(safe_selected) * value_stride_s +
          static_cast<int64_t>(batch) * value_stride_b +
          static_cast<int64_t>(head) * value_stride_h;
      for (int d = lane; d < V; d += 32) {
        dot_vg += load_typed(value, v_base + d * value_stride_v, scalar_dtype) * go_s[d];
      }
    }
    dot_vg = warp_sum(dot_vg);
    dot_vg = __shfl_sync(0xffffffff, dot_vg, 0);

    valid = valid && finite_float(dot_vg);
    const float prob = valid ? expf(score - row_lse) : 0.0f;
    valid = valid && finite_float(prob);
    const float ds = valid ? prob * (dot_vg - delta) * softmax_scale : 0.0f;
    valid = valid && finite_float(ds);

    if (valid && emit_query) {
      for (int d = lane; d < D; d += 32) {
        grad_q_nope_s[warp][d] +=
            ds * load_typed(key_nope, k_nope_base + d * key_nope_stride_d, scalar_dtype);
      }
      for (int d = lane; d < P; d += 32) {
        grad_q_pe_s[warp][d] += ds * load_typed(key_pe, k_pe_base * P + d, scalar_dtype);
      }
    }

    const bool in_kv_range = valid && (kv_end <= 0 || (selected >= kv_start && selected < kv_end));
    const int local_selected = selected - kv_start;
    if (in_kv_range && emit_key_nope) {
      const int64_t grad_k_base =
          static_cast<int64_t>(local_selected) * grad_key_nope_stride_s +
          static_cast<int64_t>(batch) * grad_key_nope_stride_b +
          static_cast<int64_t>(head) * grad_key_nope_stride_h;
      for (int d = lane; d < D; d += 32) {
        atomic_add_typed(
            grad_key_nope,
            grad_k_base + static_cast<int64_t>(d) * grad_key_nope_stride_d,
            ds * q_nope_s[d],
            grad_dtype);
      }
    }
    if (in_kv_range && emit_key_pe) {
      const int64_t grad_kp_base =
          ((static_cast<int64_t>(local_selected) * B + batch) * KPH + pe_head);
      for (int d = lane; d < P; d += 32) {
        atomic_add_typed(grad_key_pe, grad_kp_base * P + d, ds * q_pe_s[d], grad_dtype);
      }
    }
    if (in_kv_range && emit_value) {
      const int64_t grad_v_base =
          static_cast<int64_t>(local_selected) * grad_value_stride_s +
          static_cast<int64_t>(batch) * grad_value_stride_b +
          static_cast<int64_t>(head) * grad_value_stride_h;
      for (int d = lane; d < V; d += 32) {
        atomic_add_typed(
            grad_value,
            grad_v_base + static_cast<int64_t>(d) * grad_value_stride_v,
            prob * go_s[d],
            grad_dtype);
      }
    }
  }
  __syncthreads();

  if (emit_query) {
    for (int d = tid; d < D; d += WARPS * 32) {
      float acc = 0.0f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) {
        acc += grad_q_nope_s[w][d];
      }
      store_typed(grad_query_nope, q_base * D + d, acc, grad_dtype);
    }
    for (int d = tid; d < P; d += WARPS * 32) {
      float acc = 0.0f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) {
        acc += grad_q_pe_s[w][d];
      }
      store_typed(grad_query_pe, q_base * P + d, acc, grad_dtype);
    }
  }
}

__global__ void dsa_sorted_delta_kernel(
    const void* __restrict__ output,      // [Q, B, H, V]
    const void* __restrict__ grad_output, // [Q, B, H, V]
    float* __restrict__ delta,            // [B * Q, H]
    int Q,
    int B,
    int H,
    int V,
    int scalar_dtype) {
  const int row = blockIdx.x;
  const int head_batch = blockIdx.y;
  const int head = head_batch % H;
  const int batch = head_batch / H;
  const int tid = threadIdx.x;
  __shared__ float scratch[256];
  const int64_t q_base = ((static_cast<int64_t>(row) * B + batch) * H + head);
  float acc = 0.0f;
  for (int d = tid; d < V; d += blockDim.x) {
    acc += load_typed(output, q_base * V + d, scalar_dtype) *
           load_typed(grad_output, q_base * V + d, scalar_dtype);
  }
  scratch[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    delta[(static_cast<int64_t>(batch) * Q + row) * H + head] = scratch[0];
  }
}

template <int EDGE_WARPS>
__global__ void dsa_sorted_edge_stats_kernel(
    const void* __restrict__ value,           // [S, B, H, V]
    const void* __restrict__ topk,            // [B, Q, K]
    const float* __restrict__ selected_score, // [B * Q, H, K]
    const float* __restrict__ lse,            // [B * Q, H]
    const void* __restrict__ grad_output,     // [Q, B, H, V]
    const float* __restrict__ delta,          // [B * Q, H]
    uint64_t* __restrict__ edge_keys,         // [Q * B * H * K]
    int64_t* __restrict__ edge_ids,           // [Q * B * H * K]
    float* __restrict__ edge_prob,            // [Q * B * H * K]
    float* __restrict__ edge_ds,              // [Q * B * H * K]
    int Q,
    int B,
    int S,
    int H,
    int V,
    int K,
    float softmax_scale,
    int scalar_dtype,
    int topk_dtype,
    int64_t total_edges) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int64_t edge = static_cast<int64_t>(blockIdx.x) * EDGE_WARPS + warp;
  if (warp >= EDGE_WARPS || edge >= total_edges) {
    return;
  }

  const int slot = static_cast<int>(edge % K);
  const int64_t tmp = edge / K;
  const int head_batch = static_cast<int>(tmp % (B * H));
  const int row = static_cast<int>(tmp / (B * H));
  const int head = head_batch % H;
  const int batch = head_batch / H;

  int32_t selected = -1;
  float score = -INFINITY;
  if (lane == 0) {
    selected = load_topk(
        topk,
        (static_cast<int64_t>(batch) * Q + row) * K + slot,
        topk_dtype);
    score = selected_score[(static_cast<int64_t>(batch) * Q + row) * H * K +
                           static_cast<int64_t>(head) * K + slot];
  }
  selected = __shfl_sync(0xffffffff, selected, 0);
  score = __shfl_sync(0xffffffff, score, 0);
  bool valid = selected >= 0 && selected < S && score > -3.0e38f &&
               finite_float(score);
  const int safe_selected = selected > 0 ? selected : 0;
  const int64_t q_base = ((static_cast<int64_t>(row) * B + batch) * H + head);
  const int64_t selected_base =
      ((static_cast<int64_t>(safe_selected) * B + batch) * H + head);

  float dot_vg = 0.0f;
  if (valid) {
    for (int d = lane; d < V; d += 32) {
      dot_vg += load_typed(value, selected_base * V + d, scalar_dtype) *
                load_typed(grad_output, q_base * V + d, scalar_dtype);
    }
  }
  dot_vg = warp_sum(dot_vg);
  dot_vg = __shfl_sync(0xffffffff, dot_vg, 0);

  if (lane == 0) {
    if (valid) {
      const float row_lse = lse[(static_cast<int64_t>(batch) * Q + row) * H + head];
      const float row_delta =
          delta[(static_cast<int64_t>(batch) * Q + row) * H + head];
      valid = valid_lse(row_lse) && finite_float(dot_vg) && finite_float(row_delta);
      const float prob = valid ? expf(score - row_lse) : 0.0f;
      const float ds = valid ? prob * (dot_vg - row_delta) * softmax_scale : 0.0f;
      valid = valid && finite_float(prob) && finite_float(ds);
      if (valid) {
        edge_keys[edge] =
            (static_cast<uint64_t>(static_cast<uint32_t>(head_batch)) << 32) |
            static_cast<uint32_t>(selected);
        edge_prob[edge] = prob;
        edge_ds[edge] = ds;
      }
    }
    if (!valid) {
      edge_keys[edge] = kInvalidEdgeKey;
      edge_prob[edge] = 0.0f;
      edge_ds[edge] = 0.0f;
    }
    edge_ids[edge] = edge;
  }
}

__global__ void dsa_sorted_kv_reduce_kernel(
    const void* __restrict__ query,       // [Q, B, H, D]
    const void* __restrict__ grad_output, // [Q, B, H, V]
    const uint64_t* __restrict__ edge_keys,
    const int64_t* __restrict__ edge_ids,
    const float* __restrict__ edge_prob,
    const float* __restrict__ edge_ds,
    void* __restrict__ grad_key,   // [S, B, H, D]
    void* __restrict__ grad_value, // [S, B, H, V]
    int Q,
    int B,
    int H,
    int D,
    int V,
    int K,
    int scalar_dtype,
    int grad_dtype,
    int64_t total_edges) {
  const int tid = threadIdx.x;
  for (int64_t start = blockIdx.x; start < total_edges; start += gridDim.x) {
    const uint64_t key = edge_keys[start];
    if (key == kInvalidEdgeKey || (start > 0 && edge_keys[start - 1] == key)) {
      continue;
    }
    int64_t end = start + 1;
    while (end < total_edges && edge_keys[end] == key) {
      ++end;
    }

    const int head_batch = static_cast<int>(key >> 32);
    const int selected = static_cast<int>(key & 0xffffffffULL);
    const int head = head_batch % H;
    const int batch = head_batch / H;
    const int64_t selected_base =
        ((static_cast<int64_t>(selected) * B + batch) * H + head);

    for (int d = tid; d < D; d += blockDim.x) {
      float acc = 0.0f;
      for (int64_t pos = start; pos < end; ++pos) {
        const int64_t edge = edge_ids[pos];
        const int64_t tmp = edge / K;
        const int row = static_cast<int>(tmp / (B * H));
        const int64_t q_base = ((static_cast<int64_t>(row) * B + batch) * H + head);
        acc += edge_ds[edge] * load_typed(query, q_base * D + d, scalar_dtype);
      }
      store_typed(grad_key, selected_base * D + d, acc, grad_dtype);
    }
    for (int d = tid; d < V; d += blockDim.x) {
      float acc = 0.0f;
      for (int64_t pos = start; pos < end; ++pos) {
        const int64_t edge = edge_ids[pos];
        const int64_t tmp = edge / K;
        const int row = static_cast<int>(tmp / (B * H));
        const int64_t q_base = ((static_cast<int64_t>(row) * B + batch) * H + head);
        acc += edge_prob[edge] * load_typed(grad_output, q_base * V + d, scalar_dtype);
      }
      store_typed(grad_value, selected_base * V + d, acc, grad_dtype);
    }
  }
}

template <int TILE_Q, int TILE_K>
void launch_typed_tile(
    const void* query, const void* key, const void* value,
    const void* topk_indices, const void* output, const float* lse,
    const void* grad_output, void* grad_key, void* grad_value,
    float softmax_scale, int q_len, int bsz, int sk, int num_heads,
    int head_dim, int value_dim, int topk_count, int q_start,
    int scalar_dtype, int topk_dtype, int grad_dtype, cudaStream_t stream) {
  constexpr int EDGES = TILE_Q * TILE_K;
  static_assert(EDGES <= 8, "DSA sparse K/V CUDA reducer supports <=8 edge-warps");
  const dim3 block(EDGES * 32);
  const dim3 grid(
      (q_len + TILE_Q - 1) / TILE_Q,
      (topk_count + TILE_K - 1) / TILE_K,
      bsz * num_heads);
  dsa_sparse_kv_bwd_kernel<TILE_Q, TILE_K><<<grid, block, 0, stream>>>(
      query,
      key,
      value,
      topk_indices,
      output,
      lse,
      grad_output,
      grad_key,
      grad_value,
      softmax_scale,
      q_len,
      bsz,
      sk,
      num_heads,
      head_dim,
      value_dim,
      topk_count,
      q_start,
      scalar_dtype,
      topk_dtype,
      grad_dtype);
}

template <int TILE_Q, int TILE_K>
void launch_scores_tile(
    const void* query, const void* key, const void* value,
    const void* topk_indices, const float* selected_scores, const void* output,
    const float* lse, const void* grad_output, float* grad_query,
    void* grad_key, void* grad_value, int q_len, int bsz, int sk,
    int num_heads, int head_dim, int value_dim, int topk_count,
    float softmax_scale, int scalar_dtype, int topk_dtype, int grad_dtype,
    cudaStream_t stream) {
  constexpr int EDGES = TILE_Q * TILE_K;
  static_assert(EDGES <= 8, "DSA sparse backward CUDA reducer supports <=8 edge-warps");
  const dim3 block(EDGES * 32);
  const dim3 grid(
      (q_len + TILE_Q - 1) / TILE_Q,
      (topk_count + TILE_K - 1) / TILE_K,
      bsz * num_heads);
  dsa_sparse_bwd_from_scores_kernel<TILE_Q, TILE_K><<<grid, block, 0, stream>>>(
      query,
      key,
      value,
      topk_indices,
      selected_scores,
      output,
      lse,
      grad_output,
      grad_query,
      grad_key,
      grad_value,
      q_len,
      bsz,
      sk,
      num_heads,
      head_dim,
      value_dim,
      topk_count,
      softmax_scale,
      scalar_dtype,
      topk_dtype,
      grad_dtype);
}

template <int WARPS>
void launch_scores_row(
    const void* query, const void* key, const void* value,
    const void* topk_indices, const float* selected_scores, const void* output,
    const float* lse, const void* grad_output, float* grad_query,
    void* grad_key, void* grad_value, int q_len, int bsz, int sk,
    int num_heads, int head_dim, int value_dim, int topk_count,
    float softmax_scale, int scalar_dtype, int topk_dtype, int grad_dtype,
    cudaStream_t stream) {
  const dim3 block(WARPS * 32);
  const dim3 grid(q_len, bsz * num_heads);
  dsa_sparse_bwd_from_scores_row_kernel<WARPS><<<grid, block, 0, stream>>>(
      query,
      key,
      value,
      topk_indices,
      selected_scores,
      output,
      lse,
      grad_output,
      grad_query,
      grad_key,
      grad_value,
      q_len,
      bsz,
      sk,
      num_heads,
      head_dim,
      value_dim,
      topk_count,
      softmax_scale,
      scalar_dtype,
      topk_dtype,
      grad_dtype);
}

}  // namespace

void launch_dsa_sparse_kv_bwd(
    const void* query, const void* key, const void* value,
    const void* topk_indices, const void* output, const float* lse,
    const void* grad_output, void* grad_key, void* grad_value,
    float softmax_scale, int q_len, int bsz, int sk, int num_heads,
    int head_dim, int value_dim, int topk_count, int q_start,
    int scalar_dtype, int topk_dtype, int grad_dtype, int tile_q,
    int tile_k, cudaStream_t stream) {
  if (tile_q == 1 && tile_k == 8) {
    launch_typed_tile<1, 8>(
        query, key, value, topk_indices, output, lse, grad_output, grad_key,
        grad_value, softmax_scale, q_len, bsz, sk, num_heads, head_dim,
        value_dim, topk_count, q_start, scalar_dtype, topk_dtype, grad_dtype,
        stream);
  } else if (tile_q == 2 && tile_k == 4) {
    launch_typed_tile<2, 4>(
        query, key, value, topk_indices, output, lse, grad_output, grad_key,
        grad_value, softmax_scale, q_len, bsz, sk, num_heads, head_dim,
        value_dim, topk_count, q_start, scalar_dtype, topk_dtype, grad_dtype,
        stream);
  } else if (tile_q == 4 && tile_k == 2) {
    launch_typed_tile<4, 2>(
        query, key, value, topk_indices, output, lse, grad_output, grad_key,
        grad_value, softmax_scale, q_len, bsz, sk, num_heads, head_dim,
        value_dim, topk_count, q_start, scalar_dtype, topk_dtype, grad_dtype,
        stream);
  } else if (tile_q == 1 && tile_k == 4) {
    launch_typed_tile<1, 4>(
        query, key, value, topk_indices, output, lse, grad_output, grad_key,
        grad_value, softmax_scale, q_len, bsz, sk, num_heads, head_dim,
        value_dim, topk_count, q_start, scalar_dtype, topk_dtype, grad_dtype,
        stream);
  } else if (tile_q == 2 && tile_k == 2) {
    launch_typed_tile<2, 2>(
        query, key, value, topk_indices, output, lse, grad_output, grad_key,
        grad_value, softmax_scale, q_len, bsz, sk, num_heads, head_dim,
        value_dim, topk_count, q_start, scalar_dtype, topk_dtype, grad_dtype,
        stream);
  } else {
    C10_CUDA_CHECK(cudaErrorInvalidConfiguration);
  }
}

void launch_dsa_sparse_bwd_from_scores(
    const void* query, const void* key, const void* value,
    const void* topk_indices, const float* selected_scores, const void* output,
    const float* lse, const void* grad_output, float* grad_query,
    void* grad_key, void* grad_value, int q_len, int bsz, int sk,
    int num_heads, int head_dim, int value_dim, int topk_count,
    float softmax_scale, int scalar_dtype, int topk_dtype, int grad_dtype,
    int tile_q, int tile_k, cudaStream_t stream) {
  if (tile_q == 1 && tile_k == 8) {
    launch_scores_tile<1, 8>(
        query, key, value, topk_indices, selected_scores, output, lse,
        grad_output, grad_query, grad_key, grad_value, q_len, bsz, sk,
        num_heads, head_dim, value_dim, topk_count, softmax_scale, scalar_dtype,
        topk_dtype, grad_dtype, stream);
  } else if (tile_q == 2 && tile_k == 4) {
    launch_scores_tile<2, 4>(
        query, key, value, topk_indices, selected_scores, output, lse,
        grad_output, grad_query, grad_key, grad_value, q_len, bsz, sk,
        num_heads, head_dim, value_dim, topk_count, softmax_scale, scalar_dtype,
        topk_dtype, grad_dtype, stream);
  } else if (tile_q == 4 && tile_k == 2) {
    launch_scores_tile<4, 2>(
        query, key, value, topk_indices, selected_scores, output, lse,
        grad_output, grad_query, grad_key, grad_value, q_len, bsz, sk,
        num_heads, head_dim, value_dim, topk_count, softmax_scale, scalar_dtype,
        topk_dtype, grad_dtype, stream);
  } else if (tile_q == 1 && tile_k == 4) {
    launch_scores_tile<1, 4>(
        query, key, value, topk_indices, selected_scores, output, lse,
        grad_output, grad_query, grad_key, grad_value, q_len, bsz, sk,
        num_heads, head_dim, value_dim, topk_count, softmax_scale, scalar_dtype,
        topk_dtype, grad_dtype, stream);
  } else if (tile_q == 2 && tile_k == 2) {
    launch_scores_tile<2, 2>(
        query, key, value, topk_indices, selected_scores, output, lse,
        grad_output, grad_query, grad_key, grad_value, q_len, bsz, sk,
        num_heads, head_dim, value_dim, topk_count, softmax_scale, scalar_dtype,
        topk_dtype, grad_dtype, stream);
  } else {
    C10_CUDA_CHECK(cudaErrorInvalidConfiguration);
  }
}

void launch_dsa_sparse_bwd_from_scores_row(
    const void* query, const void* key, const void* value,
    const void* topk_indices, const float* selected_scores, const void* output,
    const float* lse, const void* grad_output, float* grad_query,
    void* grad_key, void* grad_value, int q_len, int bsz, int sk,
    int num_heads, int head_dim, int value_dim, int topk_count,
    float softmax_scale, int scalar_dtype, int topk_dtype, int grad_dtype,
    cudaStream_t stream) {
  const char* warps_env = std::getenv("MEGATRON_DSA_CUDA_ROW_BWD_WARPS");
  const int warps = warps_env == nullptr ? 8 : std::atoi(warps_env);
  if (warps == 16) {
    launch_scores_row<16>(
        query,
        key,
        value,
        topk_indices,
        selected_scores,
        output,
        lse,
        grad_output,
        grad_query,
        grad_key,
        grad_value,
        q_len,
        bsz,
        sk,
        num_heads,
        head_dim,
        value_dim,
        topk_count,
        softmax_scale,
        scalar_dtype,
        topk_dtype,
        grad_dtype,
        stream);
    return;
  }
  if (warps == 4) {
    launch_scores_row<4>(
        query,
        key,
        value,
        topk_indices,
        selected_scores,
        output,
        lse,
        grad_output,
        grad_query,
        grad_key,
        grad_value,
        q_len,
        bsz,
        sk,
        num_heads,
        head_dim,
        value_dim,
        topk_count,
        softmax_scale,
        scalar_dtype,
        topk_dtype,
        grad_dtype,
        stream);
    return;
  }
  launch_scores_row<8>(
      query,
      key,
      value,
      topk_indices,
      selected_scores,
      output,
      lse,
      grad_output,
      grad_query,
      grad_key,
      grad_value,
      q_len,
      bsz,
      sk,
      num_heads,
      head_dim,
      value_dim,
      topk_count,
      softmax_scale,
      scalar_dtype,
      topk_dtype,
      grad_dtype,
      stream);
}

template <int WARPS>
void launch_split_qk_fwd_row(
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
  const size_t smem_bytes =
      static_cast<size_t>(num_heads) * static_cast<size_t>(head_dim + pos_dim) * sizeof(float) +
      (emit_teacher ? static_cast<size_t>(topk_count) * sizeof(float) : 0);
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(
        dsa_split_qk_fwd_row_kernel<WARPS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
  }
  const dim3 block(WARPS * 32);
  const dim3 grid(q_len, bsz);
  dsa_split_qk_fwd_row_kernel<WARPS><<<grid, block, smem_bytes, stream>>>(
      query_nope,
      query_pe,
      key_nope,
      key_pe,
      value,
      topk_indices,
      query_positions,
      key_positions,
      output,
      lse,
      teacher_probs,
      teacher_score_scratch,
      q_len,
      bsz,
      sk,
      num_heads,
      head_dim,
      pos_dim,
      key_pe_heads,
      value_dim,
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
      scalar_dtype,
      topk_dtype,
      has_positions,
      emit_teacher,
      use_teacher_score_scratch);
}

void launch_dsa_split_qk_fwd_row(
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
    int use_teacher_score_scratch, int warps, cudaStream_t stream) {
  if (warps == 16) {
    launch_split_qk_fwd_row<16>(
        query_nope, query_pe, key_nope, key_pe, value, topk_indices,
        query_positions, key_positions, output, lse, teacher_probs,
        teacher_score_scratch, q_len, bsz, sk, num_heads, head_dim, pos_dim,
        key_pe_heads, value_dim, topk_count, q_start, query_nope_stride_s,
        query_nope_stride_b, query_nope_stride_h, query_nope_stride_d,
        key_nope_stride_s, key_nope_stride_b, key_nope_stride_h,
        key_nope_stride_d, value_stride_s, value_stride_b, value_stride_h,
        value_stride_v, softmax_scale, scalar_dtype, topk_dtype, has_positions,
        emit_teacher, use_teacher_score_scratch, stream);
    return;
  }
  if (warps == 4) {
    launch_split_qk_fwd_row<4>(
        query_nope, query_pe, key_nope, key_pe, value, topk_indices,
        query_positions, key_positions, output, lse, teacher_probs,
        teacher_score_scratch, q_len, bsz, sk, num_heads, head_dim, pos_dim,
        key_pe_heads, value_dim, topk_count, q_start, query_nope_stride_s,
        query_nope_stride_b, query_nope_stride_h, query_nope_stride_d,
        key_nope_stride_s, key_nope_stride_b, key_nope_stride_h,
        key_nope_stride_d, value_stride_s, value_stride_b, value_stride_h,
        value_stride_v, softmax_scale, scalar_dtype, topk_dtype, has_positions,
        emit_teacher, use_teacher_score_scratch, stream);
    return;
  }
  launch_split_qk_fwd_row<8>(
      query_nope, query_pe, key_nope, key_pe, value, topk_indices,
      query_positions, key_positions, output, lse, teacher_probs,
      teacher_score_scratch, q_len, bsz, sk, num_heads, head_dim, pos_dim,
      key_pe_heads, value_dim, topk_count, q_start, query_nope_stride_s,
      query_nope_stride_b, query_nope_stride_h, query_nope_stride_d,
      key_nope_stride_s, key_nope_stride_b, key_nope_stride_h,
      key_nope_stride_d, value_stride_s, value_stride_b, value_stride_h,
      value_stride_v, softmax_scale, scalar_dtype, topk_dtype, has_positions,
      emit_teacher, use_teacher_score_scratch, stream);
}

template <int WARPS>
void launch_split_qk_row(
    const void* query_nope, const void* query_pe, const void* key_nope,
    const void* key_pe, const void* value, const void* topk_indices,
    const int64_t* query_positions, const int64_t* key_positions,
    const void* output, const float* lse, const void* grad_output,
    void* grad_query_nope, void* grad_query_pe, void* grad_key_nope,
    void* grad_key_pe, void* grad_value, int q_len, int bsz, int sk,
    int num_heads, int head_dim, int pos_dim, int key_pe_heads,
    int value_dim, int topk_count, int q_start, int kv_start, int kv_end,
    int64_t query_nope_stride_s, int64_t query_nope_stride_b,
    int64_t query_nope_stride_h, int64_t query_nope_stride_d,
    int64_t key_nope_stride_s, int64_t key_nope_stride_b,
    int64_t key_nope_stride_h, int64_t key_nope_stride_d,
    int64_t value_stride_s, int64_t value_stride_b, int64_t value_stride_h,
    int64_t value_stride_v,
    int64_t grad_key_nope_stride_s, int64_t grad_key_nope_stride_b,
    int64_t grad_key_nope_stride_h, int64_t grad_key_nope_stride_d,
    int64_t grad_value_stride_s, int64_t grad_value_stride_b,
    int64_t grad_value_stride_h, int64_t grad_value_stride_v,
    float softmax_scale, int scalar_dtype, int topk_dtype, int grad_dtype,
    int has_positions, int emit_query, int emit_key_nope, int emit_key_pe,
    int emit_value, cudaStream_t stream) {
  const dim3 block(WARPS * 32);
  const dim3 grid(q_len, bsz * num_heads);
  dsa_split_qk_bwd_row_kernel<WARPS><<<grid, block, 0, stream>>>(
      query_nope,
      query_pe,
      key_nope,
      key_pe,
      value,
      topk_indices,
      query_positions,
      key_positions,
      output,
      lse,
      grad_output,
      grad_query_nope,
      grad_query_pe,
      grad_key_nope,
      grad_key_pe,
      grad_value,
      q_len,
      bsz,
      sk,
      num_heads,
      head_dim,
      pos_dim,
      key_pe_heads,
      value_dim,
      topk_count,
      q_start,
      kv_start,
      kv_end,
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
      grad_key_nope_stride_s,
      grad_key_nope_stride_b,
      grad_key_nope_stride_h,
      grad_key_nope_stride_d,
      grad_value_stride_s,
      grad_value_stride_b,
      grad_value_stride_h,
      grad_value_stride_v,
      softmax_scale,
      scalar_dtype,
      topk_dtype,
      grad_dtype,
      has_positions,
      emit_query,
      emit_key_nope,
      emit_key_pe,
      emit_value);
}

void launch_dsa_split_qk_bwd_row(
    const void* query_nope, const void* query_pe, const void* key_nope,
    const void* key_pe, const void* value, const void* topk_indices,
    const int64_t* query_positions, const int64_t* key_positions,
    const void* output, const float* lse, const void* grad_output,
    void* grad_query_nope, void* grad_query_pe, void* grad_key_nope,
    void* grad_key_pe, void* grad_value, int q_len, int bsz, int sk,
    int num_heads, int head_dim, int pos_dim, int key_pe_heads,
    int value_dim, int topk_count, int q_start, int kv_start, int kv_end,
    int64_t query_nope_stride_s, int64_t query_nope_stride_b,
    int64_t query_nope_stride_h, int64_t query_nope_stride_d,
    int64_t key_nope_stride_s, int64_t key_nope_stride_b,
    int64_t key_nope_stride_h, int64_t key_nope_stride_d,
    int64_t value_stride_s, int64_t value_stride_b, int64_t value_stride_h,
    int64_t value_stride_v,
    int64_t grad_key_nope_stride_s, int64_t grad_key_nope_stride_b,
    int64_t grad_key_nope_stride_h, int64_t grad_key_nope_stride_d,
    int64_t grad_value_stride_s, int64_t grad_value_stride_b,
    int64_t grad_value_stride_h, int64_t grad_value_stride_v,
    float softmax_scale, int scalar_dtype, int topk_dtype, int grad_dtype,
    int has_positions, int emit_query, int emit_key_nope, int emit_key_pe,
    int emit_value, int warps, cudaStream_t stream) {
  if (warps == 16) {
    launch_split_qk_row<16>(
        query_nope, query_pe, key_nope, key_pe, value, topk_indices,
        query_positions, key_positions, output, lse, grad_output,
        grad_query_nope, grad_query_pe, grad_key_nope, grad_key_pe,
        grad_value, q_len, bsz, sk, num_heads, head_dim, pos_dim,
        key_pe_heads, value_dim, topk_count, q_start, kv_start, kv_end,
        query_nope_stride_s, query_nope_stride_b,
        query_nope_stride_h, query_nope_stride_d,
        key_nope_stride_s, key_nope_stride_b,
        key_nope_stride_h, key_nope_stride_d,
        value_stride_s, value_stride_b, value_stride_h, value_stride_v,
        grad_key_nope_stride_s, grad_key_nope_stride_b,
        grad_key_nope_stride_h, grad_key_nope_stride_d,
        grad_value_stride_s, grad_value_stride_b,
        grad_value_stride_h, grad_value_stride_v,
        softmax_scale, scalar_dtype, topk_dtype, grad_dtype, has_positions,
        emit_query, emit_key_nope, emit_key_pe, emit_value, stream);
    return;
  }
  if (warps == 4) {
    launch_split_qk_row<4>(
        query_nope, query_pe, key_nope, key_pe, value, topk_indices,
        query_positions, key_positions, output, lse, grad_output,
        grad_query_nope, grad_query_pe, grad_key_nope, grad_key_pe,
        grad_value, q_len, bsz, sk, num_heads, head_dim, pos_dim,
        key_pe_heads, value_dim, topk_count, q_start, kv_start, kv_end,
        query_nope_stride_s, query_nope_stride_b,
        query_nope_stride_h, query_nope_stride_d,
        key_nope_stride_s, key_nope_stride_b,
        key_nope_stride_h, key_nope_stride_d,
        value_stride_s, value_stride_b, value_stride_h, value_stride_v,
        grad_key_nope_stride_s, grad_key_nope_stride_b,
        grad_key_nope_stride_h, grad_key_nope_stride_d,
        grad_value_stride_s, grad_value_stride_b,
        grad_value_stride_h, grad_value_stride_v,
        softmax_scale, scalar_dtype, topk_dtype, grad_dtype, has_positions,
        emit_query, emit_key_nope, emit_key_pe, emit_value, stream);
    return;
  }
  launch_split_qk_row<8>(
      query_nope, query_pe, key_nope, key_pe, value, topk_indices,
      query_positions, key_positions, output, lse, grad_output,
      grad_query_nope, grad_query_pe, grad_key_nope, grad_key_pe,
      grad_value, q_len, bsz, sk, num_heads, head_dim, pos_dim,
      key_pe_heads, value_dim, topk_count, q_start, kv_start, kv_end,
      query_nope_stride_s, query_nope_stride_b,
      query_nope_stride_h, query_nope_stride_d,
      key_nope_stride_s, key_nope_stride_b,
      key_nope_stride_h, key_nope_stride_d,
      value_stride_s, value_stride_b, value_stride_h, value_stride_v,
      grad_key_nope_stride_s, grad_key_nope_stride_b,
      grad_key_nope_stride_h, grad_key_nope_stride_d,
      grad_value_stride_s, grad_value_stride_b,
      grad_value_stride_h, grad_value_stride_v,
      softmax_scale, scalar_dtype, topk_dtype, grad_dtype, has_positions,
      emit_query, emit_key_nope, emit_key_pe, emit_value, stream);
}

void launch_dsa_sparse_kv_bwd_sorted_from_scores(
    const void* query, const void* value, const void* topk_indices,
    const float* selected_scores, const void* output, const float* lse,
    const void* grad_output, uint64_t* edge_keys, int64_t* edge_ids,
    float* edge_prob, float* edge_ds, float* delta, void* grad_key,
    void* grad_value, int q_len, int bsz, int sk, int num_heads,
    int head_dim, int value_dim, int topk_count, float softmax_scale,
    int scalar_dtype, int topk_dtype, int grad_dtype, cudaStream_t stream) {
  const int64_t total_edges =
      static_cast<int64_t>(q_len) * bsz * num_heads * topk_count;
  const dim3 delta_grid(q_len, bsz * num_heads);
  dsa_sorted_delta_kernel<<<delta_grid, 256, 0, stream>>>(
      output, grad_output, delta, q_len, bsz, num_heads, value_dim, scalar_dtype);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int kEdgeWarps = 8;
  const dim3 edge_block(kEdgeWarps * 32);
  const dim3 edge_grid((total_edges + kEdgeWarps - 1) / kEdgeWarps);
  dsa_sorted_edge_stats_kernel<kEdgeWarps><<<edge_grid, edge_block, 0, stream>>>(
      value,
      topk_indices,
      selected_scores,
      lse,
      grad_output,
      delta,
      edge_keys,
      edge_ids,
      edge_prob,
      edge_ds,
      q_len,
      bsz,
      sk,
      num_heads,
      value_dim,
      topk_count,
      softmax_scale,
      scalar_dtype,
      topk_dtype,
      total_edges);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto keys_begin = thrust::device_pointer_cast(edge_keys);
  auto keys_end = keys_begin + total_edges;
  auto values_begin = thrust::device_pointer_cast(edge_ids);
  thrust::sort_by_key(thrust::cuda::par.on(stream), keys_begin, keys_end, values_begin);

  const int reduce_grid = static_cast<int>(std::min<int64_t>(total_edges, 65535));
  dsa_sorted_kv_reduce_kernel<<<reduce_grid, 256, 0, stream>>>(
      query,
      grad_output,
      edge_keys,
      edge_ids,
      edge_prob,
      edge_ds,
      grad_key,
      grad_value,
      q_len,
      bsz,
      num_heads,
      head_dim,
      value_dim,
      topk_count,
      scalar_dtype,
      grad_dtype,
      total_edges);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace hisa_indexer
}  // namespace megatron
