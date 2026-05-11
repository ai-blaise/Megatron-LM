// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Adapted from: optimization-playground/sgl-kernel/csrc/elementwise/gated_norm_cute.{cu,cuh}
//   Copyright 2025 SGLang Team, Apache 2.0.
//
// GatedNorm fused CuTe kernel for SM100 (B200).
//
//   output = normed * sigmoid(silu(normed @ w_down^T) @ w_up^T)
//
// Two-pass tensor-core kernel. Pass 1 computes `silu(normed @ w_down^T)`;
// pass 2 computes `sigmoid(... @ w_up^T) * normed`. Both passes use
// `mma.sync.aligned.m16n8k16` with `cp.async` double-buffered SMEM staging
// (prefetch issued AFTER the mma so the released stage is the prefetch
// target). SMEM rows are padded by 8 bf16 to break the power-of-2 stride
// that would otherwise cause bank conflicts on `ldmatrix` loads.
//
// At low-N (BM=16), pass 2 partitions across the output N axis with
// NUM_N_WARPS=4 warps per CTA — each warp owns a disjoint BN_PER_WARP=64
// column tile, sharing the small sA[BM, BK_R] activation tile across warps.
//
// At rank == 64 && num_tokens >= 16 the per-warp work is too large for
// hand-written mma to compete with cuBLAS; the launcher returns
// cudaErrorInvalidValue so callers fall back to torch.mm (matches the
// existing SMEM-overflow contract).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace megatron_gated_norm {

static constexpr int kMaxRank = 64;
static constexpr int kWarpSize = 32;
static constexpr int kMaxSmemBytes = 228 * 1024;
static constexpr int kSmemPad = 8;  // bf16 padding to break power-of-2 stride

__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float fast_silu(float x) {
  return x * fast_sigmoid(x);
}

__device__ __forceinline__ void cp_async_16(uint32_t smem_addr,
                                            const void* gmem_addr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(gmem_addr));
}
__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ uint32_t smem_to_uint(const void* ptr) {
  uint32_t addr;
  asm("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }"
      : "=r"(addr)
      : "l"(ptr));
  return addr;
}

__device__ __forceinline__ void mma_m16n8k16_bf16_f32(float c[4],
                                                      const uint32_t a[4],
                                                      const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t out[4],
                                            uint32_t smem_addr) {
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
      : "r"(smem_addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t out[2],
                                            uint32_t smem_addr) {
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(out[0]), "=r"(out[1])
               : "r"(smem_addr));
}

// ----------------------------------------------------------------------------
// Pass 1: z[N, R] += normed[N, D] @ w_down^T[D, R]
// ----------------------------------------------------------------------------
template <int BM_TOTAL, int BN, int BK_STEPS, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS* kWarpSize)
    gated_norm_pass1_mma(const __nv_bfloat16* __restrict__ normed,
                         const __nv_bfloat16* __restrict__ w_down,
                         float* __restrict__ z_workspace,
                         int num_tokens,
                         int D,
                         int rank,
                         int split_k,
                         int chunk_per_split,
                         int max_rank) {
  static constexpr int BK = 16 * BK_STEPS;
  static constexpr int BK_S = BK + kSmemPad;
  static constexpr int kBNTiles = BN / 8;
  static constexpr int kBMPerWarp = 16;
  static_assert(BM_TOTAL == NUM_WARPS * 16, "BM_TOTAL must equal NUM_WARPS*16");

  const int chunk_id = blockIdx.x;
  const int row_block_id = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;

  const int row_base = row_block_id * BM_TOTAL;
  if (row_base >= num_tokens) return;

  const int h_start_global = chunk_id * chunk_per_split;
  const int h_end_global = min(h_start_global + chunk_per_split, D);

  static constexpr int kStages = 2;
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* sA[kStages];
  __nv_bfloat16* sB[kStages];
  sA[0] = smem;
  sB[0] = sA[0] + BM_TOTAL * BK_S;
  sA[1] = sB[0] + BN * BK_S;
  sB[1] = sA[1] + BM_TOTAL * BK_S;

  const int warp_m_offset = warp_id * kBMPerWarp;

  float acc[kBNTiles][4];
#pragma unroll
  for (int i = 0; i < kBNTiles; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
  }

  constexpr int VEC = 8;
  const int threads_per_block = NUM_WARPS * kWarpSize;
  const int total_vec_a = (BM_TOTAL * BK) / VEC;
  const int total_vec_b = (BN * BK) / VEC;

  const int tile = lane_id / 8;
  const int row_in_tile = lane_id % 8;
  const int a_row_local =
      (tile == 2 || tile == 3) ? (row_in_tile + 8) : row_in_tile;
  const int a_row = warp_m_offset + a_row_local;
  const int a_col_offset = (tile == 1 || tile == 3) ? 8 : 0;
  int b_row_local, b_col_offset;
  if (lane_id < 8) {
    b_row_local = lane_id;
    b_col_offset = 0;
  } else if (lane_id < 16) {
    b_row_local = lane_id - 8;
    b_col_offset = 8;
  } else if (lane_id < 24) {
    b_row_local = lane_id - 16;
    b_col_offset = 0;
  } else {
    b_row_local = lane_id - 24;
    b_col_offset = 8;
  }

  auto issue_loads = [&](int k_off_local, int stage) {
    const int k_left_local = min(BK, h_end_global - k_off_local);
    for (int v = tid; v < total_vec_a; v += threads_per_block) {
      int row_local = v / (BK / VEC);
      int col = (v % (BK / VEC)) * VEC;
      int row_global = row_base + row_local;
      uint32_t s_addr = smem_to_uint(&sA[stage][row_local * BK_S + col]);
      if (row_global < num_tokens && (col + VEC) <= k_left_local) {
        cp_async_16(s_addr, normed + row_global * D + k_off_local + col);
      } else {
        ((float4*)&sA[stage][row_local * BK_S + col])[0] =
            make_float4(0, 0, 0, 0);
        if (row_global < num_tokens) {
          for (int e = 0; e < VEC; e++) {
            if (col + e < k_left_local) {
              sA[stage][row_local * BK_S + col + e] =
                  normed[row_global * D + k_off_local + col + e];
            }
          }
        }
      }
    }
    for (int v = tid; v < total_vec_b; v += threads_per_block) {
      int row_local = v / (BK / VEC);
      int col = (v % (BK / VEC)) * VEC;
      uint32_t s_addr = smem_to_uint(&sB[stage][row_local * BK_S + col]);
      if (row_local < rank && (col + VEC) <= k_left_local) {
        cp_async_16(s_addr, w_down + row_local * D + k_off_local + col);
      } else {
        ((float4*)&sB[stage][row_local * BK_S + col])[0] =
            make_float4(0, 0, 0, 0);
        if (row_local < rank) {
          for (int e = 0; e < VEC; e++) {
            if (col + e < k_left_local) {
              sB[stage][row_local * BK_S + col + e] =
                  w_down[row_local * D + k_off_local + col + e];
            }
          }
        }
      }
    }
    cp_async_commit_group();
  };

  auto do_mma = [&](int stage) {
#pragma unroll
    for (int ks = 0; ks < BK_STEPS; ks++) {
      const int a_col = ks * 16 + a_col_offset;
      uint32_t a_raw[4];
      ldmatrix_x4(a_raw, smem_to_uint(&sA[stage][a_row * BK_S + a_col]));
      uint32_t a_reg[4] = {a_raw[0], a_raw[2], a_raw[1], a_raw[3]};

      const int b_col = ks * 16 + b_col_offset;
#pragma unroll
      for (int nt = 0; nt < kBNTiles; nt++) {
        uint32_t b_reg[2];
        ldmatrix_x2(b_reg, smem_to_uint(
                               &sB[stage][(nt * 8 + b_row_local) * BK_S + b_col]));
        mma_m16n8k16_bf16_f32(acc[nt], a_reg, b_reg);
      }
    }
  };

  if (h_start_global < h_end_global) {
    int n_iters = 0;
    for (int k = h_start_global; k < h_end_global; k += BK) n_iters++;

    issue_loads(h_start_global, 0);

    if (n_iters == 1) {
      cp_async_wait_all();
      __syncthreads();
      do_mma(0);
      __syncthreads();
    } else {
      const int second_k = h_start_global + BK;
      issue_loads(second_k, 1);

      int compute_stage = 0;
      for (int it = 0; it < n_iters - 1; it++) {
        cp_async_wait_group<1>();
        __syncthreads();

        do_mma(compute_stage);
        __syncthreads();

        const int prefetch_k = h_start_global + (it + 2) * BK;
        if (prefetch_k < h_end_global) {
          issue_loads(prefetch_k, compute_stage);
        }

        compute_stage = 1 - compute_stage;
      }
      cp_async_wait_all();
      __syncthreads();
      do_mma(compute_stage);
      __syncthreads();
    }
  }

  {
    const int row0 = lane_id / 4;
    const int col_base_inner = (lane_id % 4) * 2;
#pragma unroll
    for (int nt = 0; nt < kBNTiles; nt++) {
      const int n_base = nt * 8;
      const int g_row0 = row_base + warp_m_offset + row0;
      const int g_row1 = row_base + warp_m_offset + row0 + 8;
      const int g_col0 = n_base + col_base_inner;
      const int g_col1 = n_base + col_base_inner + 1;

#define WRITE_OR_ATOMIC(cond_row, cond_col, idx, val)            \
  if ((cond_row) && (cond_col)) {                                \
    if (split_k == 1)                                            \
      z_workspace[(int64_t)(idx)] = (val);                       \
    else                                                         \
      atomicAdd(&z_workspace[(int64_t)(idx)], (val));            \
  }

      WRITE_OR_ATOMIC(g_row0 < num_tokens, g_col0 < rank,
                      g_row0 * max_rank + g_col0, acc[nt][0]);
      WRITE_OR_ATOMIC(g_row0 < num_tokens, g_col1 < rank,
                      g_row0 * max_rank + g_col1, acc[nt][1]);
      WRITE_OR_ATOMIC(g_row1 < num_tokens, g_col0 < rank,
                      g_row1 * max_rank + g_col0, acc[nt][2]);
      WRITE_OR_ATOMIC(g_row1 < num_tokens, g_col1 < rank,
                      g_row1 * max_rank + g_col1, acc[nt][3]);
#undef WRITE_OR_ATOMIC
    }
  }
}

// ----------------------------------------------------------------------------
// Pass 2: output = sigmoid(silu(z) @ w_up^T) * normed
// ----------------------------------------------------------------------------
template <int BM_TOTAL, int BN_D, int BK_R_STEPS, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS* kWarpSize)
    gated_norm_pass2_mma(const __nv_bfloat16* __restrict__ normed,
                         const __nv_bfloat16* __restrict__ w_up,
                         const float* __restrict__ z_workspace,
                         __nv_bfloat16* __restrict__ output,
                         int num_tokens,
                         int D,
                         int rank,
                         int max_rank) {
  static constexpr int BK_R = 16 * BK_R_STEPS;
  static constexpr int BK_R_S = BK_R + kSmemPad;
  static constexpr int kBNTiles = BN_D / 8;
  static constexpr int kBMPerWarp = 16;
  static_assert(BM_TOTAL == NUM_WARPS * 16, "BM_TOTAL must equal NUM_WARPS*16");

  const int row_block_id = blockIdx.y;
  const int col_block_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;

  const int row_base = row_block_id * BM_TOTAL;
  const int col_base = col_block_id * BN_D;
  if (row_base >= num_tokens || col_base >= D) return;

  static constexpr int BN_D_S = BN_D + kSmemPad;
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* sA = smem;
  __nv_bfloat16* sB = smem + BM_TOTAL * BK_R_S;
  __nv_bfloat16* sN = sB + BN_D * BK_R_S;

  const int threads_per_block = NUM_WARPS * kWarpSize;
  const int warp_m_offset = warp_id * kBMPerWarp;

  const int total_a = BM_TOTAL * BK_R;
  for (int i = tid; i < total_a; i += threads_per_block) {
    int m = i / BK_R;
    int r = i % BK_R;
    int row_global = row_base + m;
    float val = 0.0f;
    if (row_global < num_tokens && r < rank) {
      val = z_workspace[row_global * max_rank + r];
      val = fast_silu(val);
    }
    sA[m * BK_R_S + r] = __float2bfloat16(val);
  }

  constexpr int VEC = 8;

  const int total_vec_b = (BN_D * BK_R) / VEC;
  for (int v = tid; v < total_vec_b; v += threads_per_block) {
    int n = v / (BK_R / VEC);
    int r = (v % (BK_R / VEC)) * VEC;
    int col_global = col_base + n;
    if (col_global < D && (r + VEC) <= rank) {
      cp_async_16(smem_to_uint(&sB[n * BK_R_S + r]),
                  w_up + (int64_t)col_global * rank + r);
    } else {
      ((float4*)&sB[n * BK_R_S + r])[0] = make_float4(0, 0, 0, 0);
      if (col_global < D) {
        for (int e = 0; e < VEC; e++) {
          if (r + e < rank) {
            sB[n * BK_R_S + r + e] = w_up[(int64_t)col_global * rank + r + e];
          }
        }
      }
    }
  }
  cp_async_commit_group();

  const int total_vec_n = (BM_TOTAL * BN_D) / VEC;
  for (int v = tid; v < total_vec_n; v += threads_per_block) {
    int m = v / (BN_D / VEC);
    int n = (v % (BN_D / VEC)) * VEC;
    int row_global = row_base + m;
    int col_global = col_base + n;
    if (row_global < num_tokens && (col_global + VEC) <= D) {
      cp_async_16(smem_to_uint(&sN[m * BN_D_S + n]),
                  normed + (int64_t)row_global * D + col_global);
    } else {
      ((float4*)&sN[m * BN_D_S + n])[0] = make_float4(0, 0, 0, 0);
      if (row_global < num_tokens) {
        for (int e = 0; e < VEC; e++) {
          if (col_global + e < D) {
            sN[m * BN_D_S + n + e] =
                normed[(int64_t)row_global * D + col_global + e];
          }
        }
      }
    }
  }
  cp_async_commit_group();

  cp_async_wait_group<1>();
  __syncthreads();

  float acc[kBNTiles][4];
#pragma unroll
  for (int i = 0; i < kBNTiles; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
  }

  {
    const int tile = lane_id / 8;
    const int row_in_tile = lane_id % 8;
    const int a_row_local =
        (tile == 2 || tile == 3) ? (row_in_tile + 8) : row_in_tile;
    const int a_row = warp_m_offset + a_row_local;

#pragma unroll
    for (int ks = 0; ks < BK_R_STEPS; ks++) {
      const int a_col = ks * 16 + ((tile == 1 || tile == 3) ? 8 : 0);
      uint32_t a_raw[4];
      ldmatrix_x4(a_raw, smem_to_uint(&sA[a_row * BK_R_S + a_col]));
      uint32_t a_reg[4] = {a_raw[0], a_raw[2], a_raw[1], a_raw[3]};

      int b_row_local, b_col;
      if (lane_id < 8) {
        b_row_local = lane_id;
        b_col = ks * 16;
      } else if (lane_id < 16) {
        b_row_local = lane_id - 8;
        b_col = ks * 16 + 8;
      } else if (lane_id < 24) {
        b_row_local = lane_id - 16;
        b_col = ks * 16;
      } else {
        b_row_local = lane_id - 24;
        b_col = ks * 16 + 8;
      }

#pragma unroll
      for (int nt = 0; nt < kBNTiles; nt++) {
        uint32_t b_reg[2];
        ldmatrix_x2(b_reg, smem_to_uint(
                               &sB[(nt * 8 + b_row_local) * BK_R_S + b_col]));
        mma_m16n8k16_bf16_f32(acc[nt], a_reg, b_reg);
      }
    }
  }

  cp_async_wait_all();
  __syncthreads();

  {
    const int row0_local = lane_id / 4;
    const int row1_local = row0_local + 8;
    const int col_local_base = (lane_id % 4) * 2;
    const int m0 = warp_m_offset + row0_local;
    const int m1 = warp_m_offset + row1_local;
    const int g_row0 = row_base + m0;
    const int g_row1 = row_base + m1;

#pragma unroll
    for (int nt = 0; nt < kBNTiles; nt++) {
      const int n_base = nt * 8;
      const int n0_local = n_base + col_local_base;
      const int g_col0 = col_base + n0_local;
      const int g_col1 = g_col0 + 1;

      if (g_row0 < num_tokens && g_col1 < D) {
        __nv_bfloat162 n_pair =
            *reinterpret_cast<__nv_bfloat162*>(&sN[m0 * BN_D_S + n0_local]);
        float2 n_f = __bfloat1622float2(n_pair);
        float gate0 = fast_sigmoid(acc[nt][0]);
        float gate1 = fast_sigmoid(acc[nt][1]);
        __nv_bfloat162 out_pair =
            __float22bfloat162_rn({gate0 * n_f.x, gate1 * n_f.y});
        *reinterpret_cast<__nv_bfloat162*>(&output[(int64_t)g_row0 * D + g_col0]) =
            out_pair;
      } else if (g_row0 < num_tokens && g_col0 < D) {
        float gate = fast_sigmoid(acc[nt][0]);
        float n_val = __bfloat162float(sN[m0 * BN_D_S + n0_local]);
        output[(int64_t)g_row0 * D + g_col0] = __float2bfloat16(gate * n_val);
      }

      if (g_row1 < num_tokens && g_col1 < D) {
        __nv_bfloat162 n_pair =
            *reinterpret_cast<__nv_bfloat162*>(&sN[m1 * BN_D_S + n0_local]);
        float2 n_f = __bfloat1622float2(n_pair);
        float gate0 = fast_sigmoid(acc[nt][2]);
        float gate1 = fast_sigmoid(acc[nt][3]);
        __nv_bfloat162 out_pair =
            __float22bfloat162_rn({gate0 * n_f.x, gate1 * n_f.y});
        *reinterpret_cast<__nv_bfloat162*>(&output[(int64_t)g_row1 * D + g_col0]) =
            out_pair;
      } else if (g_row1 < num_tokens && g_col0 < D) {
        float gate = fast_sigmoid(acc[nt][2]);
        float n_val = __bfloat162float(sN[m1 * BN_D_S + n0_local]);
        output[(int64_t)g_row1 * D + g_col0] = __float2bfloat16(gate * n_val);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Pass 2 (N-warp variant): each warp owns BN_PER_WARP=64 column tile.
// ----------------------------------------------------------------------------
template <int BM_TOTAL, int BN_PER_WARP, int BK_R_STEPS, int NUM_N_WARPS>
__global__ void __launch_bounds__(NUM_N_WARPS* kWarpSize)
    gated_norm_pass2_mma_n_warps(const __nv_bfloat16* __restrict__ normed,
                                 const __nv_bfloat16* __restrict__ w_up,
                                 const float* __restrict__ z_workspace,
                                 __nv_bfloat16* __restrict__ output,
                                 int num_tokens,
                                 int D,
                                 int rank,
                                 int max_rank) {
  static constexpr int BK_R = 16 * BK_R_STEPS;
  static constexpr int BK_R_S = BK_R + kSmemPad;
  static constexpr int kBNTiles = BN_PER_WARP / 8;
  static constexpr int BN_D_TOTAL = BN_PER_WARP * NUM_N_WARPS;
  static constexpr int BN_PER_WARP_S = BN_PER_WARP + kSmemPad;
  static_assert(BM_TOTAL == 16, "BM_TOTAL must be 16 for n-warp variant");

  const int row_block_id = blockIdx.y;
  const int col_block_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;

  const int row_base = row_block_id * BM_TOTAL;
  const int col_base_total = col_block_id * BN_D_TOTAL;
  const int col_base = col_base_total + warp_id * BN_PER_WARP;
  if (row_base >= num_tokens || col_base_total >= D) return;

  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* sA = smem;
  __nv_bfloat16* sB_all = smem + BM_TOTAL * BK_R_S;
  __nv_bfloat16* sN_all = sB_all + NUM_N_WARPS * BN_PER_WARP * BK_R_S;
  __nv_bfloat16* sB = sB_all + warp_id * BN_PER_WARP * BK_R_S;
  __nv_bfloat16* sN = sN_all + warp_id * BM_TOTAL * BN_PER_WARP_S;

  const int threads_per_block = NUM_N_WARPS * kWarpSize;

  const int total_a = BM_TOTAL * BK_R;
  for (int i = tid; i < total_a; i += threads_per_block) {
    int m = i / BK_R;
    int r = i % BK_R;
    int row_global = row_base + m;
    float val = 0.0f;
    if (row_global < num_tokens && r < rank) {
      val = z_workspace[row_global * max_rank + r];
      val = fast_silu(val);
    }
    sA[m * BK_R_S + r] = __float2bfloat16(val);
  }

  constexpr int VEC = 8;

  const int total_vec_b_per_warp = (BN_PER_WARP * BK_R) / VEC;
  for (int v = lane_id; v < total_vec_b_per_warp; v += kWarpSize) {
    int n = v / (BK_R / VEC);
    int r = (v % (BK_R / VEC)) * VEC;
    int col_global = col_base + n;
    if (col_global < D && (r + VEC) <= rank) {
      cp_async_16(smem_to_uint(&sB[n * BK_R_S + r]),
                  w_up + (int64_t)col_global * rank + r);
    } else {
      ((float4*)&sB[n * BK_R_S + r])[0] = make_float4(0, 0, 0, 0);
      if (col_global < D) {
        for (int e = 0; e < VEC; e++) {
          if (r + e < rank) {
            sB[n * BK_R_S + r + e] = w_up[(int64_t)col_global * rank + r + e];
          }
        }
      }
    }
  }
  cp_async_commit_group();

  const int total_vec_n_per_warp = (BM_TOTAL * BN_PER_WARP) / VEC;
  for (int v = lane_id; v < total_vec_n_per_warp; v += kWarpSize) {
    int m = v / (BN_PER_WARP / VEC);
    int n = (v % (BN_PER_WARP / VEC)) * VEC;
    int row_global = row_base + m;
    int col_global = col_base + n;
    if (row_global < num_tokens && (col_global + VEC) <= D) {
      cp_async_16(smem_to_uint(&sN[m * BN_PER_WARP_S + n]),
                  normed + (int64_t)row_global * D + col_global);
    } else {
      ((float4*)&sN[m * BN_PER_WARP_S + n])[0] = make_float4(0, 0, 0, 0);
      if (row_global < num_tokens) {
        for (int e = 0; e < VEC; e++) {
          if (col_global + e < D) {
            sN[m * BN_PER_WARP_S + n + e] =
                normed[(int64_t)row_global * D + col_global + e];
          }
        }
      }
    }
  }
  cp_async_commit_group();

  cp_async_wait_group<1>();
  __syncthreads();

  float acc[kBNTiles][4];
#pragma unroll
  for (int i = 0; i < kBNTiles; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
  }

  {
    const int tile = lane_id / 8;
    const int row_in_tile = lane_id % 8;
    const int a_row_local =
        (tile == 2 || tile == 3) ? (row_in_tile + 8) : row_in_tile;
    const int a_row = a_row_local;

#pragma unroll
    for (int ks = 0; ks < BK_R_STEPS; ks++) {
      const int a_col = ks * 16 + ((tile == 1 || tile == 3) ? 8 : 0);
      uint32_t a_raw[4];
      ldmatrix_x4(a_raw, smem_to_uint(&sA[a_row * BK_R_S + a_col]));
      uint32_t a_reg[4] = {a_raw[0], a_raw[2], a_raw[1], a_raw[3]};

      int b_row_local, b_col;
      if (lane_id < 8) {
        b_row_local = lane_id;
        b_col = ks * 16;
      } else if (lane_id < 16) {
        b_row_local = lane_id - 8;
        b_col = ks * 16 + 8;
      } else if (lane_id < 24) {
        b_row_local = lane_id - 16;
        b_col = ks * 16;
      } else {
        b_row_local = lane_id - 24;
        b_col = ks * 16 + 8;
      }

#pragma unroll
      for (int nt = 0; nt < kBNTiles; nt++) {
        uint32_t b_reg[2];
        ldmatrix_x2(b_reg, smem_to_uint(
                               &sB[(nt * 8 + b_row_local) * BK_R_S + b_col]));
        mma_m16n8k16_bf16_f32(acc[nt], a_reg, b_reg);
      }
    }
  }

  cp_async_wait_all();
  __syncthreads();

  {
    const int row0_local = lane_id / 4;
    const int row1_local = row0_local + 8;
    const int col_local_base = (lane_id % 4) * 2;
    const int g_row0 = row_base + row0_local;
    const int g_row1 = row_base + row1_local;

#pragma unroll
    for (int nt = 0; nt < kBNTiles; nt++) {
      const int n_base = nt * 8;
      const int n0_local = n_base + col_local_base;
      const int g_col0 = col_base + n0_local;
      const int g_col1 = g_col0 + 1;

      if (g_row0 < num_tokens && g_col1 < D) {
        __nv_bfloat162 n_pair = *reinterpret_cast<__nv_bfloat162*>(
            &sN[row0_local * BN_PER_WARP_S + n0_local]);
        float2 n_f = __bfloat1622float2(n_pair);
        float gate0 = fast_sigmoid(acc[nt][0]);
        float gate1 = fast_sigmoid(acc[nt][1]);
        __nv_bfloat162 out_pair =
            __float22bfloat162_rn({gate0 * n_f.x, gate1 * n_f.y});
        *reinterpret_cast<__nv_bfloat162*>(&output[(int64_t)g_row0 * D + g_col0]) =
            out_pair;
      } else if (g_row0 < num_tokens && g_col0 < D) {
        float gate = fast_sigmoid(acc[nt][0]);
        float n_val = __bfloat162float(sN[row0_local * BN_PER_WARP_S + n0_local]);
        output[(int64_t)g_row0 * D + g_col0] = __float2bfloat16(gate * n_val);
      }

      if (g_row1 < num_tokens && g_col1 < D) {
        __nv_bfloat162 n_pair = *reinterpret_cast<__nv_bfloat162*>(
            &sN[row1_local * BN_PER_WARP_S + n0_local]);
        float2 n_f = __bfloat1622float2(n_pair);
        float gate0 = fast_sigmoid(acc[nt][2]);
        float gate1 = fast_sigmoid(acc[nt][3]);
        __nv_bfloat162 out_pair =
            __float22bfloat162_rn({gate0 * n_f.x, gate1 * n_f.y});
        *reinterpret_cast<__nv_bfloat162*>(&output[(int64_t)g_row1 * D + g_col0]) =
            out_pair;
      } else if (g_row1 < num_tokens && g_col0 < D) {
        float gate = fast_sigmoid(acc[nt][2]);
        float n_val = __bfloat162float(sN[row1_local * BN_PER_WARP_S + n0_local]);
        output[(int64_t)g_row1 * D + g_col0] = __float2bfloat16(gate * n_val);
      }
    }
  }
}

__global__ void zero_workspace_kernel(float* z_workspace, int n_elems) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_elems) z_workspace[idx] = 0.0f;
}

// ----------------------------------------------------------------------------
// Host-side launcher.
// ----------------------------------------------------------------------------
inline cudaError_t launch_gated_norm_cute(const __nv_bfloat16* normed,
                                          const __nv_bfloat16* w_down,
                                          const __nv_bfloat16* w_up,
                                          __nv_bfloat16* output,
                                          int num_tokens,
                                          int hidden_size,
                                          int rank,
                                          cudaStream_t stream) {
  if (num_tokens <= 0) return cudaSuccess;
  if (rank > kMaxRank) return cudaErrorInvalidValue;

  // R=64 with N>=16: hand-written mma loses to cuBLAS structurally.
  if (rank > 48 && num_tokens >= 16) return cudaErrorInvalidValue;
  // R=32 at very large N: cuBLAS heavily tuned; gating overhead pushes < 1.0x.
  if (rank > 16 && rank <= 48 && num_tokens >= 4096) return cudaErrorInvalidValue;

  auto next_po2 = [](int v) -> int {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
  };
  const int max_rank = next_po2(rank);
  const int rank_up_8 = ((rank + 7) / 8) * 8;
  const int rank_up_16 = ((rank + 15) / 16) * 16;
  const int sms = 148;

  // ---------- Pass 1 ----------
  int p1_BM;
  if (num_tokens >= 1024)
    p1_BM = 128;
  else if (num_tokens >= 64)
    p1_BM = 64;
  else
    p1_BM = 16;
  const int BK_P1 = 64;
  const int BK_S_P1 = BK_P1 + kSmemPad;
  int p1_smem = 2 * (p1_BM * BK_S_P1 + rank_up_8 * BK_S_P1) *
                (int)sizeof(__nv_bfloat16);
  if (p1_smem > kMaxSmemBytes) return cudaErrorInvalidValue;

  const int row_blocks_p1 = (num_tokens + p1_BM - 1) / p1_BM;
  int split_k = 1;
  if (row_blocks_p1 < sms) {
    int target = (sms + row_blocks_p1 - 1) / row_blocks_p1;
    split_k = max(1, min(target, (hidden_size + BK_P1 - 1) / BK_P1));
  }
  int chunk_per_split = ((hidden_size + split_k - 1) / split_k);
  chunk_per_split = ((chunk_per_split + BK_P1 - 1) / BK_P1) * BK_P1;
  split_k = (hidden_size + chunk_per_split - 1) / chunk_per_split;

  float* z_workspace = nullptr;
  size_t ws_bytes = (size_t)num_tokens * max_rank * sizeof(float);
  cudaError_t err = cudaMallocAsync(&z_workspace, ws_bytes, stream);
  if (err != cudaSuccess) return err;

  if (split_k > 1) {
    int n_elems = num_tokens * max_rank;
    int blocks = (n_elems + 255) / 256;
    zero_workspace_kernel<<<blocks, 256, 0, stream>>>(z_workspace, n_elems);
  }

  dim3 p1_grid(split_k, row_blocks_p1);

#define LAUNCH_P1(BM_TOT, BN_TPL, NW)                                         \
  do {                                                                        \
    auto* fn = gated_norm_pass1_mma<BM_TOT, BN_TPL, 4, NW>;                   \
    if (p1_smem > 48 * 1024) {                                                \
      cudaError_t e = cudaFuncSetAttribute(                                   \
          fn, cudaFuncAttributeMaxDynamicSharedMemorySize, p1_smem);          \
      if (e != cudaSuccess) {                                                 \
        cudaFreeAsync(z_workspace, stream);                                   \
        return e;                                                             \
      }                                                                       \
    }                                                                         \
    fn<<<p1_grid, NW * kWarpSize, p1_smem, stream>>>(                         \
        normed, w_down, z_workspace, num_tokens, hidden_size, rank, split_k,  \
        chunk_per_split, max_rank);                                           \
  } while (0)

  if (p1_BM == 16) {
    switch (rank_up_8) {
      case 8:  LAUNCH_P1(16, 8, 1);  break;
      case 16: LAUNCH_P1(16, 16, 1); break;
      case 24: LAUNCH_P1(16, 24, 1); break;
      case 32: LAUNCH_P1(16, 32, 1); break;
      case 40: LAUNCH_P1(16, 40, 1); break;
      case 48: LAUNCH_P1(16, 48, 1); break;
      case 56: LAUNCH_P1(16, 56, 1); break;
      case 64: LAUNCH_P1(16, 64, 1); break;
      default:
        cudaFreeAsync(z_workspace, stream);
        return cudaErrorInvalidValue;
    }
  } else if (p1_BM == 64) {
    switch (rank_up_8) {
      case 8:  LAUNCH_P1(64, 8, 4);  break;
      case 16: LAUNCH_P1(64, 16, 4); break;
      case 24: LAUNCH_P1(64, 24, 4); break;
      case 32: LAUNCH_P1(64, 32, 4); break;
      case 40: LAUNCH_P1(64, 40, 4); break;
      case 48: LAUNCH_P1(64, 48, 4); break;
      case 56: LAUNCH_P1(64, 56, 4); break;
      case 64: LAUNCH_P1(64, 64, 4); break;
      default:
        cudaFreeAsync(z_workspace, stream);
        return cudaErrorInvalidValue;
    }
  } else {
    switch (rank_up_8) {
      case 8:  LAUNCH_P1(128, 8, 8);  break;
      case 16: LAUNCH_P1(128, 16, 8); break;
      case 24: LAUNCH_P1(128, 24, 8); break;
      case 32: LAUNCH_P1(128, 32, 8); break;
      case 40: LAUNCH_P1(128, 40, 8); break;
      case 48: LAUNCH_P1(128, 48, 8); break;
      case 56: LAUNCH_P1(128, 56, 8); break;
      case 64: LAUNCH_P1(128, 64, 8); break;
      default:
        cudaFreeAsync(z_workspace, stream);
        return cudaErrorInvalidValue;
    }
  }
#undef LAUNCH_P1

  // ---------- Pass 2 ----------
  int p2_BM;
  const bool prefer_n_warp_path = (rank_up_16 == 32);
  if (prefer_n_warp_path)
    p2_BM = 16;
  else if (num_tokens >= 64)
    p2_BM = 64;
  else
    p2_BM = 16;
  static constexpr int BN_P2 = 64;
  int p2_smem = ((p2_BM + BN_P2) * (rank_up_16 + kSmemPad) +
                 p2_BM * (BN_P2 + kSmemPad)) *
                (int)sizeof(__nv_bfloat16);
  if (p2_smem > kMaxSmemBytes) {
    cudaFreeAsync(z_workspace, stream);
    return cudaErrorInvalidValue;
  }

  const int row_blocks_p2 = (num_tokens + p2_BM - 1) / p2_BM;
  const int col_blocks_p2 = (hidden_size + BN_P2 - 1) / BN_P2;
  dim3 p2_grid(col_blocks_p2, row_blocks_p2);

#define LAUNCH_P2(BM_TOT, BK_R_STEPS_TPL, NW)                                 \
  do {                                                                        \
    auto* fn = gated_norm_pass2_mma<BM_TOT, BN_P2, BK_R_STEPS_TPL, NW>;       \
    if (p2_smem > 48 * 1024) {                                                \
      cudaError_t e = cudaFuncSetAttribute(                                   \
          fn, cudaFuncAttributeMaxDynamicSharedMemorySize, p2_smem);          \
      if (e != cudaSuccess) {                                                 \
        cudaFreeAsync(z_workspace, stream);                                   \
        return e;                                                             \
      }                                                                       \
    }                                                                         \
    fn<<<p2_grid, NW * kWarpSize, p2_smem, stream>>>(                         \
        normed, w_up, z_workspace, output, num_tokens, hidden_size, rank,     \
        max_rank);                                                            \
  } while (0)

  static constexpr int kNNW_BN_PER_WARP = 64;
  static constexpr int kNNW = 4;
  static constexpr int kNNW_BN_PER_WARP_S = kNNW_BN_PER_WARP + kSmemPad;
  const int kNNW_BK_R_S = rank_up_16 + kSmemPad;
  const int p2_nnw_smem =
      (16 * kNNW_BK_R_S + kNNW * kNNW_BN_PER_WARP * kNNW_BK_R_S +
       kNNW * 16 * kNNW_BN_PER_WARP_S) *
      (int)sizeof(__nv_bfloat16);
  const int col_blocks_p2_nnw =
      (hidden_size + kNNW * kNNW_BN_PER_WARP - 1) / (kNNW * kNNW_BN_PER_WARP);
  dim3 p2_nnw_grid(col_blocks_p2_nnw, row_blocks_p2);

#define LAUNCH_P2_NNW(BK_R_STEPS_TPL)                                         \
  do {                                                                        \
    auto* fn =                                                                \
        gated_norm_pass2_mma_n_warps<16, kNNW_BN_PER_WARP, BK_R_STEPS_TPL,    \
                                     kNNW>;                                   \
    if (p2_nnw_smem > 48 * 1024) {                                            \
      cudaError_t e = cudaFuncSetAttribute(                                   \
          fn, cudaFuncAttributeMaxDynamicSharedMemorySize, p2_nnw_smem);      \
      if (e != cudaSuccess) {                                                 \
        cudaFreeAsync(z_workspace, stream);                                   \
        return e;                                                             \
      }                                                                       \
    }                                                                         \
    fn<<<p2_nnw_grid, kNNW * kWarpSize, p2_nnw_smem, stream>>>(               \
        normed, w_up, z_workspace, output, num_tokens, hidden_size, rank,     \
        max_rank);                                                            \
  } while (0)

  if (p2_BM == 16) {
    if (p2_nnw_smem <= kMaxSmemBytes &&
        (rank_up_16 == 32 || rank_up_16 == 48 || rank_up_16 == 64)) {
      switch (rank_up_16) {
        case 32: LAUNCH_P2_NNW(2); break;
        case 48: LAUNCH_P2_NNW(3); break;
        case 64: LAUNCH_P2_NNW(4); break;
      }
    } else {
      switch (rank_up_16) {
        case 16: LAUNCH_P2(16, 1, 1); break;
        case 32: LAUNCH_P2(16, 2, 1); break;
        case 48: LAUNCH_P2(16, 3, 1); break;
        case 64: LAUNCH_P2(16, 4, 1); break;
        default:
          cudaFreeAsync(z_workspace, stream);
          return cudaErrorInvalidValue;
      }
    }
  } else if (p2_BM == 64) {
    switch (rank_up_16) {
      case 16: LAUNCH_P2(64, 1, 4); break;
      case 32: LAUNCH_P2(64, 2, 4); break;
      case 48: LAUNCH_P2(64, 3, 4); break;
      case 64: LAUNCH_P2(64, 4, 4); break;
      default:
        cudaFreeAsync(z_workspace, stream);
        return cudaErrorInvalidValue;
    }
  } else {
    switch (rank_up_16) {
      case 16: LAUNCH_P2(128, 1, 8); break;
      case 32: LAUNCH_P2(128, 2, 8); break;
      case 48: LAUNCH_P2(128, 3, 8); break;
      case 64: LAUNCH_P2(128, 4, 8); break;
      default:
        cudaFreeAsync(z_workspace, stream);
        return cudaErrorInvalidValue;
    }
  }
#undef LAUNCH_P2
#undef LAUNCH_P2_NNW

  cudaError_t last = cudaGetLastError();
  cudaFreeAsync(z_workspace, stream);
  return last;
}

}  // namespace megatron_gated_norm

// ----------------------------------------------------------------------------
// Extern "C" entry called from the cpp wrapper.
// ----------------------------------------------------------------------------
extern "C" int megatron_gated_norm_cute_forward(const void* normed,
                                                const void* w_down,
                                                const void* w_up,
                                                void* output,
                                                int num_tokens,
                                                int hidden_size,
                                                int rank,
                                                cudaStream_t stream) {
  cudaError_t err = megatron_gated_norm::launch_gated_norm_cute(
      reinterpret_cast<const __nv_bfloat16*>(normed),
      reinterpret_cast<const __nv_bfloat16*>(w_down),
      reinterpret_cast<const __nv_bfloat16*>(w_up),
      reinterpret_cast<__nv_bfloat16*>(output),
      num_tokens,
      hidden_size,
      rank,
      stream);
  return static_cast<int>(err);
}
