// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// CUDA compact-index local permutation helpers for DeepEP MoE dispatch.
//
// DeepEP returns compact local expert indices after fused dispatch. The older
// path built row/edge maps in one kernel and then launched separate row gather
// or scatter kernels. These kernels use the compact DeepEP representation
// directly: the forward pack assigns each valid edge to its expert slot and
// copies the hidden row in the same CTA.

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

constexpr int kIndexI16 = 0;
constexpr int kIndexI32 = 1;
constexpr int kIndexI64 = 2;

__device__ __forceinline__ float load_typed(const void* ptr, int64_t idx, int dtype) {
  if (dtype == kDTypeF32) {
    return reinterpret_cast<const float*>(ptr)[idx];
  }
  if (dtype == kDTypeBF16) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
  }
  return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

__device__ __forceinline__ void store_typed(void* ptr, int64_t idx, float value, int dtype) {
  if (dtype == kDTypeF32) {
    reinterpret_cast<float*>(ptr)[idx] = value;
  } else if (dtype == kDTypeBF16) {
    reinterpret_cast<__nv_bfloat16*>(ptr)[idx] = __float2bfloat16(value);
  } else {
    reinterpret_cast<__half*>(ptr)[idx] = __float2half(value);
  }
}

__device__ __forceinline__ void atomic_add_typed(void* ptr, int64_t idx, float value, int dtype) {
  if (dtype == kDTypeF32) {
    atomicAdd(reinterpret_cast<float*>(ptr) + idx, value);
  } else if (dtype == kDTypeBF16) {
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(ptr) + idx, __float2bfloat16(value));
  } else {
    atomicAdd(reinterpret_cast<__half*>(ptr) + idx, __float2half(value));
  }
}

__device__ __forceinline__ int dtype_size_bytes(int dtype) {
  return dtype == kDTypeF32 ? 4 : 2;
}

__device__ __forceinline__ int4 make_zero_int4() {
  return make_int4(0, 0, 0, 0);
}

union Int4TypedPack {
  int4 raw;
  float f32[4];
  __nv_bfloat16 bf16[8];
  __half f16[8];
};

__device__ __forceinline__ int4 sum_row_vecs_as_float(
    const int4* __restrict__ src,
    const int32_t* __restrict__ src_rows,
    int valid_count,
    int64_t hidden_vecs,
    int64_t vec,
    int dtype) {
  Int4TypedPack in;
  Int4TypedPack out;

  if (dtype == kDTypeF32) {
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int slot = 0; slot < valid_count; ++slot) {
      in.raw = src[static_cast<int64_t>(src_rows[slot]) * hidden_vecs + vec];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        accum[j] += in.f32[j];
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out.f32[j] = accum[j];
    }
    return out.raw;
  }

  if (dtype == kDTypeBF16) {
    float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int slot = 0; slot < valid_count; ++slot) {
      in.raw = src[static_cast<int64_t>(src_rows[slot]) * hidden_vecs + vec];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        accum[j] += __bfloat162float(in.bf16[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      out.bf16[j] = __float2bfloat16(accum[j]);
    }
    return out.raw;
  }

  float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int slot = 0; slot < valid_count; ++slot) {
    in.raw = src[static_cast<int64_t>(src_rows[slot]) * hidden_vecs + vec];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      accum[j] += __half2float(in.f16[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    out.f16[j] = __float2half(accum[j]);
  }
  return out.raw;
}

__device__ __forceinline__ void copy_row_raw(
    const void* src,
    int64_t src_row,
    void* dst,
    int64_t dst_row,
    int hidden_size,
    int dtype) {
  const int64_t row_bytes = static_cast<int64_t>(hidden_size) * dtype_size_bytes(dtype);
  const char* src_bytes = reinterpret_cast<const char*>(src) + src_row * row_bytes;
  char* dst_bytes = reinterpret_cast<char*>(dst) + dst_row * row_bytes;

  if ((row_bytes & 15) == 0) {
    const int64_t vecs = row_bytes >> 4;
    const uint4* src_vec = reinterpret_cast<const uint4*>(src_bytes);
    uint4* dst_vec = reinterpret_cast<uint4*>(dst_bytes);
    for (int64_t i = threadIdx.x; i < vecs; i += blockDim.x) {
      dst_vec[i] = src_vec[i];
    }
    return;
  }

  for (int64_t i = threadIdx.x; i < row_bytes; i += blockDim.x) {
    dst_bytes[i] = src_bytes[i];
  }
}

__device__ __forceinline__ int64_t load_index(const void* ptr, int64_t idx, int dtype) {
  if (dtype == kIndexI64) {
    return reinterpret_cast<const int64_t*>(ptr)[idx];
  }
  if (dtype == kIndexI32) {
    return static_cast<int64_t>(reinterpret_cast<const int32_t*>(ptr)[idx]);
  }
  return static_cast<int64_t>(reinterpret_cast<const int16_t*>(ptr)[idx]);
}

__global__ void compact_permute_fwd_kernel(
    const void* __restrict__ hidden,
    const void* __restrict__ indices,
    const float* __restrict__ probs,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ counts,
    int32_t* __restrict__ counters,
    void* __restrict__ output,
    float* __restrict__ permuted_probs,
    int64_t* __restrict__ row_map,
    int64_t* __restrict__ edge_map,
    int64_t num_edges,
    int topk,
    int num_experts,
    int hidden_size,
    int hidden_dtype,
    int index_dtype) {
  const int64_t edge = static_cast<int64_t>(blockIdx.x);
  if (edge >= num_edges) {
    return;
  }

  __shared__ int64_t out_row_shared;
  __shared__ int64_t src_row_shared;
  __shared__ int valid_shared;

  if (threadIdx.x == 0) {
    valid_shared = 0;
    const int64_t expert64 = load_index(indices, edge, index_dtype);
    if (expert64 >= 0 && expert64 < num_experts) {
      const int expert = static_cast<int>(expert64);
      const int pos = atomicAdd(counters + expert, 1);
      if (static_cast<int64_t>(pos) < counts[expert]) {
        const int64_t out_row = offsets[expert] + static_cast<int64_t>(pos);
        const int64_t src_row = edge / topk;
        row_map[out_row] = src_row;
        edge_map[out_row] = edge;
        permuted_probs[out_row] = probs[edge];
        out_row_shared = out_row;
        src_row_shared = src_row;
        valid_shared = 1;
      }
    }
  }
  __syncthreads();

  if (!valid_shared) {
    return;
  }

  copy_row_raw(hidden, src_row_shared, output, out_row_shared, hidden_size, hidden_dtype);
}

__global__ void compact_permute_rows_fwd_kernel(
    const void* __restrict__ hidden,
    const void* __restrict__ indices,
    const float* __restrict__ probs,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ counts,
    int32_t* __restrict__ counters,
    void* __restrict__ output,
    float* __restrict__ permuted_probs,
    int64_t* __restrict__ row_map,
    int64_t* __restrict__ edge_map,
    int32_t* __restrict__ edge_to_row,
    int64_t num_rows,
    int topk,
    int num_experts,
    int hidden_size,
    int hidden_dtype,
    int index_dtype) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= num_rows) {
    return;
  }

  __shared__ int valid_count;
  __shared__ int64_t out_rows[64];

  if (threadIdx.x == 0) {
    valid_count = 0;
    const int64_t edge_base = row * static_cast<int64_t>(topk);
    for (int k = 0; k < topk; ++k) {
      const int64_t edge = edge_base + k;
      const int64_t expert64 = load_index(indices, edge, index_dtype);
      if (expert64 < 0 || expert64 >= num_experts) {
        continue;
      }
      const int expert = static_cast<int>(expert64);
      const int pos = atomicAdd(counters + expert, 1);
      if (static_cast<int64_t>(pos) >= counts[expert]) {
        continue;
      }
      const int slot = valid_count++;
      const int64_t out_row = offsets[expert] + static_cast<int64_t>(pos);
      row_map[out_row] = row;
      edge_map[out_row] = edge;
      edge_to_row[edge] = static_cast<int32_t>(out_row);
      permuted_probs[out_row] = probs[edge];
      out_rows[slot] = out_row;
    }
  }
  __syncthreads();

  for (int slot = 0; slot < valid_count; ++slot) {
    copy_row_raw(hidden, row, output, out_rows[slot], hidden_size, hidden_dtype);
  }
}

__global__ void compact_unpermute_rows_kernel(
    const void* __restrict__ permuted_hidden,
    const void* __restrict__ indices,
    const int32_t* __restrict__ edge_to_row,
    void* __restrict__ output,
    int64_t num_rows,
    int topk,
    int num_experts,
    int hidden_size,
    int hidden_dtype,
    int index_dtype) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= num_rows) {
    return;
  }

  __shared__ int valid_count;
  __shared__ int32_t src_rows[64];

  if (threadIdx.x == 0) {
    valid_count = 0;
    const int64_t edge_base = row * static_cast<int64_t>(topk);
    for (int k = 0; k < topk; ++k) {
      const int64_t edge = edge_base + k;
      const int64_t expert64 = load_index(indices, edge, index_dtype);
      if (expert64 < 0 || expert64 >= num_experts) {
        continue;
      }
      const int32_t src_row = edge_to_row[edge];
      if (src_row >= 0) {
        src_rows[valid_count++] = src_row;
      }
    }
  }
  __syncthreads();

  const int64_t row_bytes = static_cast<int64_t>(hidden_size) * dtype_size_bytes(hidden_dtype);
  if ((row_bytes & 15) == 0) {
    const int64_t hidden_vecs = row_bytes >> 4;
    const int4* __restrict__ src_vec = reinterpret_cast<const int4*>(permuted_hidden);
    int4* __restrict__ dst_vec = reinterpret_cast<int4*>(output);
    const int64_t dst_base = row * hidden_vecs;

    if (valid_count == 0) {
      for (int64_t vec = threadIdx.x; vec < hidden_vecs; vec += blockDim.x) {
        dst_vec[dst_base + vec] = make_zero_int4();
      }
      return;
    }

    if (valid_count == 1) {
      const int64_t src_base = static_cast<int64_t>(src_rows[0]) * hidden_vecs;
      for (int64_t vec = threadIdx.x; vec < hidden_vecs; vec += blockDim.x) {
        dst_vec[dst_base + vec] = src_vec[src_base + vec];
      }
      return;
    }

    for (int64_t vec = threadIdx.x; vec < hidden_vecs; vec += blockDim.x) {
      dst_vec[dst_base + vec] = sum_row_vecs_as_float(
          src_vec, src_rows, valid_count, hidden_vecs, vec, hidden_dtype);
    }
    return;
  }

  const int64_t dst_base = row * static_cast<int64_t>(hidden_size);
  for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    float sum = 0.0f;
    for (int slot = 0; slot < valid_count; ++slot) {
      const int64_t src_base = static_cast<int64_t>(src_rows[slot]) * hidden_size;
      sum += load_typed(permuted_hidden, src_base + col, hidden_dtype);
    }
    store_typed(output, dst_base + col, sum, hidden_dtype);
  }
}

__global__ void compact_scatter_add_kernel(
    const void* __restrict__ src,
    const int64_t* __restrict__ row_map,
    void* __restrict__ dst,
    int64_t num_rows,
    int hidden_size,
    int dtype) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int col = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (row >= num_rows) {
    return;
  }
  const int64_t dst_row = row_map[row];
  if (dst_row < 0 || col >= hidden_size) {
    return;
  }
  const int64_t src_base = row * static_cast<int64_t>(hidden_size);
  const int64_t dst_base = dst_row * static_cast<int64_t>(hidden_size);
  const float value = load_typed(src, src_base + col, dtype);
  atomic_add_typed(dst, dst_base + col, value, dtype);
}

__global__ void compact_gather_kernel(
    const void* __restrict__ src,
    const int64_t* __restrict__ row_map,
    void* __restrict__ dst,
    int64_t num_rows,
    int hidden_size,
    int dtype) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= num_rows) {
    return;
  }
  const int64_t src_row = row_map[row];
  if (src_row < 0) {
    return;
  }
  copy_row_raw(src, src_row, dst, row, hidden_size, dtype);
}

__global__ void compact_scatter_probs_kernel(
    const float* __restrict__ grad_permuted_probs,
    const int64_t* __restrict__ edge_map,
    float* __restrict__ grad_probs,
    int64_t num_rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= num_rows) {
    return;
  }
  const int64_t edge = edge_map[row];
  if (edge >= 0) {
    grad_probs[edge] = grad_permuted_probs[row];
  }
}

}  // namespace

void launch_moe_deepep_compact_permute_fwd(
    const void* hidden,
    const void* indices,
    const float* probs,
    const int64_t* offsets,
    const int64_t* counts,
    int32_t* counters,
    void* output,
    float* permuted_probs,
    int64_t* row_map,
    int64_t* edge_map,
    int64_t num_edges,
    int topk,
    int num_experts,
    int hidden_size,
    int hidden_dtype,
    int index_dtype,
    cudaStream_t stream) {
  if (num_edges <= 0 || hidden_size <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  compact_permute_fwd_kernel<<<static_cast<unsigned int>(num_edges), kThreads, 0, stream>>>(
      hidden,
      indices,
      probs,
      offsets,
      counts,
      counters,
      output,
      permuted_probs,
      row_map,
      edge_map,
      num_edges,
      topk,
      num_experts,
      hidden_size,
      hidden_dtype,
      index_dtype);
}

void launch_moe_deepep_compact_permute_rows_fwd(
    const void* hidden,
    const void* indices,
    const float* probs,
    const int64_t* offsets,
    const int64_t* counts,
    int32_t* counters,
    void* output,
    float* permuted_probs,
    int64_t* row_map,
    int64_t* edge_map,
    int32_t* edge_to_row,
    int64_t num_rows,
    int topk,
    int num_experts,
    int hidden_size,
    int hidden_dtype,
    int index_dtype,
    cudaStream_t stream) {
  if (num_rows <= 0 || hidden_size <= 0 || topk <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  compact_permute_rows_fwd_kernel<<<static_cast<unsigned int>(num_rows), kThreads, 0, stream>>>(
      hidden,
      indices,
      probs,
      offsets,
      counts,
      counters,
      output,
      permuted_probs,
      row_map,
      edge_map,
      edge_to_row,
      num_rows,
      topk,
      num_experts,
      hidden_size,
      hidden_dtype,
      index_dtype);
}

void launch_moe_deepep_compact_unpermute_rows(
    const void* permuted_hidden,
    const void* indices,
    const int32_t* edge_to_row,
    void* output,
    int64_t num_rows,
    int topk,
    int num_experts,
    int hidden_size,
    int hidden_dtype,
    int index_dtype,
    cudaStream_t stream) {
  if (num_rows <= 0 || hidden_size <= 0 || topk <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  compact_unpermute_rows_kernel<<<static_cast<unsigned int>(num_rows), kThreads, 0, stream>>>(
      permuted_hidden,
      indices,
      edge_to_row,
      output,
      num_rows,
      topk,
      num_experts,
      hidden_size,
      hidden_dtype,
      index_dtype);
}

void launch_moe_deepep_compact_scatter_add(
    const void* src,
    const int64_t* row_map,
    void* dst,
    int64_t num_rows,
    int hidden_size,
    int dtype,
    cudaStream_t stream) {
  if (num_rows <= 0 || hidden_size <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  const dim3 grid(
      static_cast<unsigned int>(num_rows),
      static_cast<unsigned int>((hidden_size + kThreads - 1) / kThreads));
  compact_scatter_add_kernel<<<grid, kThreads, 0, stream>>>(
      src, row_map, dst, num_rows, hidden_size, dtype);
}

void launch_moe_deepep_compact_gather(
    const void* src,
    const int64_t* row_map,
    void* dst,
    int64_t num_rows,
    int hidden_size,
    int dtype,
    cudaStream_t stream) {
  if (num_rows <= 0 || hidden_size <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  compact_gather_kernel<<<static_cast<unsigned int>(num_rows), kThreads, 0, stream>>>(
      src, row_map, dst, num_rows, hidden_size, dtype);
}

void launch_moe_deepep_compact_scatter_probs(
    const float* grad_permuted_probs,
    const int64_t* edge_map,
    float* grad_probs,
    int64_t num_rows,
    cudaStream_t stream) {
  if (num_rows <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  const int64_t blocks = (num_rows + kThreads - 1) / kThreads;
  compact_scatter_probs_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      grad_permuted_probs, edge_map, grad_probs, num_rows);
}

}  // namespace hisa_indexer
}  // namespace megatron
