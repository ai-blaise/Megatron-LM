// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace megatron {
namespace hisa_indexer {

void launch_hisa_score_bwd(
    const float* grad_cand_score, const float* grad_block_score,
    const float* q, const float* k_concat, const int32_t* k_offsets,
    const float* weights, const int32_t* token_to_batch,
    const int32_t* prefix_lens, const int32_t* top_blocks,
    const int32_t* candidate_indices, const float* candidate_dot,
    const float* block_dot, const uint8_t* selected_block_mask,
    const uint8_t* selected_cand_mask, float* grad_q, float* grad_k_concat,
    float* grad_w, int Q, int H, int D, int CL, int MB, int block_size,
    int TB, cudaStream_t stream);

void launch_hisa_selector_fwd(
    const float* q, const float* k, const float* block_reps,
    const float* weights, const int32_t* prefix_lens,
    const int32_t* block_topk_counts, int32_t* topk_indices,
    float* selected_scores, int Q, int H, int D, int L, int MB,
    int block_size, int effective_block_topk, int topk_tokens,
    int force_first, int force_last, int force_last_minus_one,
    cudaStream_t stream);

void launch_hisa_selector_nvfp4_fwd(
    const float* q, const uint8_t* packed_values,
    const int32_t* packed_scales, const float* block_reps,
    const float* weights, const int32_t* prefix_lens,
    const int32_t* block_topk_counts, int32_t* topk_indices,
    float* selected_scores, int Q, int H, int D, int L, int packed_row_offset,
    int packed_row_stride, int MB, int block_size, int effective_block_topk,
    int topk_tokens, int force_first, int force_last,
    int force_last_minus_one, cudaStream_t stream);

void launch_hisa_selector_nvfp4_cublasdx_fwd(
    const float* q, const uint8_t* packed_values,
    const int32_t* packed_scales, const float* block_reps,
    const float* weights, const int32_t* prefix_lens,
    const int32_t* block_topk_counts, int32_t* topk_indices,
    float* selected_scores, int Q, int H, int D, int L, int packed_row_offset,
    int packed_row_stride, int MB, int block_size, int effective_block_topk,
    int topk_tokens, int force_first, int force_last,
    int force_last_minus_one, cudaStream_t stream);

void launch_hisa_selector_nvfp4_cublasdx_tiled_fwd(
    const float* q, const uint8_t* packed_values,
    const int32_t* packed_scales, const float* block_reps,
    const float* weights, const int32_t* prefix_lens,
    const int32_t* block_topk_counts, int32_t* selected_blocks,
    float* candidate_scores, int32_t* candidate_indices,
    int32_t* topk_indices, float* selected_scores,
    int Q, int H, int D, int L, int packed_row_offset,
    int packed_row_stride, int MB, int block_size, int effective_block_topk,
    int topk_tokens, int candidate_capacity, int force_first, int force_last,
    int force_last_minus_one, cudaStream_t stream);

void launch_hisa_selector_nvfp4_cublasdx_fp8_fwd(
    const float* q, const uint8_t* packed_values,
    const int32_t* packed_scales, const float* block_reps,
    const float* weights, const int32_t* prefix_lens,
    const int32_t* block_topk_counts, int32_t* topk_indices,
    float* selected_scores, int Q, int H, int D, int L, int packed_row_offset,
    int packed_row_stride, int MB, int block_size, int effective_block_topk,
    int topk_tokens, int force_first, int force_last,
    int force_last_minus_one, cudaStream_t stream);

void launch_hisa_selected_score_bwd(
    const float* grad_selected_scores, const float* q, const float* k,
    const float* weights, const int32_t* topk_indices, float* grad_q,
    float* grad_k, float* grad_w, int Q, int H, int D, int L, int K,
    cudaStream_t stream);

void launch_hisa_selector_teacher_fwd(
    const float* q, const float* k, const float* block_reps,
    const float* weights, const float* attn_query, const float* attn_key,
    const int32_t* prefix_lens, const int32_t* block_topk_counts,
    int32_t* topk_indices, float* selected_scores, float* teacher_probs,
    int Q, int H, int D, int L, int MB, int AH, int AD, int block_size,
    int effective_block_topk, int topk_tokens, float softmax_scale,
    int force_first, int force_last, int force_last_minus_one,
    cudaStream_t stream);

void launch_dsa_sparse_kv_bwd(
    const void* query, const void* key, const void* value,
    const void* topk_indices, const void* output, const float* lse,
    const void* grad_output, void* grad_key, void* grad_value,
    float softmax_scale, int q_len, int bsz, int sk, int num_heads,
    int head_dim, int value_dim, int topk_count, int q_start,
    int scalar_dtype, int topk_dtype, int grad_dtype, int tile_q,
    int tile_k, cudaStream_t stream);

void launch_dsa_sparse_bwd_from_scores(
    const void* query, const void* key, const void* value,
    const void* topk_indices, const float* selected_scores, const void* output,
    const float* lse, const void* grad_output, float* grad_query,
    void* grad_key, void* grad_value, int q_len, int bsz, int sk,
    int num_heads, int head_dim, int value_dim, int topk_count,
    float softmax_scale, int scalar_dtype, int topk_dtype, int grad_dtype,
    int tile_q, int tile_k, cudaStream_t stream);

void launch_dsa_sparse_bwd_from_scores_row(
    const void* query, const void* key, const void* value,
    const void* topk_indices, const float* selected_scores, const void* output,
    const float* lse, const void* grad_output, float* grad_query,
    void* grad_key, void* grad_value, int q_len, int bsz, int sk,
    int num_heads, int head_dim, int value_dim, int topk_count,
    float softmax_scale, int scalar_dtype, int topk_dtype, int grad_dtype,
    cudaStream_t stream);

void launch_dsa_sparse_kv_bwd_sorted_from_scores(
    const void* query, const void* value, const void* topk_indices,
    const float* selected_scores, const void* output, const float* lse,
    const void* grad_output, uint64_t* edge_keys, int64_t* edge_ids,
    float* edge_prob, float* edge_ds, float* delta, void* grad_key,
    void* grad_value, int q_len, int bsz, int sk, int num_heads,
    int head_dim, int value_dim, int topk_count, float softmax_scale,
    int scalar_dtype, int topk_dtype, int grad_dtype, cudaStream_t stream);

void launch_dsa_indexer_rope_fwd(
    const void* x, const void* freqs, void* out, int64_t total_rows,
    int seqlen, int batch, int heads, int head_dim, int pe_dim,
    int freq_seqlen, int freq_batch, int freq_heads, int64_t freq_stride_s,
    int64_t freq_stride_b, int64_t freq_stride_h, int64_t freq_stride_d,
    int x_dtype, int freqs_dtype, float mscale, int interleaved,
    cudaStream_t stream);

void launch_dsa_indexer_rope_fwd_inplace(
    void* x, const void* freqs, int64_t total_rows, int seqlen, int batch,
    int heads, int head_dim, int pe_dim, int freq_seqlen, int freq_batch,
    int freq_heads, int64_t freq_stride_s, int64_t freq_stride_b,
    int64_t freq_stride_h, int64_t freq_stride_d, int x_dtype,
    int freqs_dtype, float mscale, int interleaved, cudaStream_t stream);

void launch_dsa_indexer_rope_bwd(
    const void* grad_out, const void* freqs, void* grad_x,
    int64_t total_rows, int seqlen, int batch, int heads, int head_dim,
    int pe_dim, int freq_seqlen, int freq_batch, int freq_heads,
    int64_t freq_stride_s, int64_t freq_stride_b, int64_t freq_stride_h,
    int64_t freq_stride_d, int grad_dtype, int freqs_dtype, float mscale,
    int interleaved, cudaStream_t stream);

}  // namespace hisa_indexer
}  // namespace megatron

namespace {

#define HISA_CHECK_CUDA(t) TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor")
#define HISA_CHECK_CONTIG(t) TORCH_CHECK((t).is_contiguous(), #t " must be contiguous")
#define HISA_CHECK_DTYPE(t, dt) TORCH_CHECK((t).scalar_type() == (dt), #t " has wrong dtype")

void hisa_score_bwd(
    torch::Tensor grad_cand_score, torch::Tensor grad_block_score,
    torch::Tensor q, torch::Tensor k_concat, torch::Tensor k_offsets,
    torch::Tensor weights, torch::Tensor token_to_batch,
    torch::Tensor prefix_lens, torch::Tensor top_blocks,
    torch::Tensor candidate_indices, torch::Tensor candidate_dot,
    torch::Tensor block_dot, torch::Tensor selected_block_mask,
    torch::Tensor selected_cand_mask, torch::Tensor grad_q,
    torch::Tensor grad_k_concat, torch::Tensor grad_w, int64_t block_size) {
  HISA_CHECK_CUDA(grad_cand_score);
  HISA_CHECK_CUDA(grad_block_score);
  HISA_CHECK_CUDA(q);
  HISA_CHECK_CUDA(k_concat);
  HISA_CHECK_CUDA(k_offsets);
  HISA_CHECK_CUDA(weights);
  HISA_CHECK_CUDA(token_to_batch);
  HISA_CHECK_CUDA(prefix_lens);
  HISA_CHECK_CUDA(top_blocks);
  HISA_CHECK_CUDA(candidate_indices);
  HISA_CHECK_CUDA(candidate_dot);
  HISA_CHECK_CUDA(block_dot);
  HISA_CHECK_CUDA(selected_block_mask);
  HISA_CHECK_CUDA(selected_cand_mask);
  HISA_CHECK_CUDA(grad_q);
  HISA_CHECK_CUDA(grad_k_concat);
  HISA_CHECK_CUDA(grad_w);

  HISA_CHECK_CONTIG(q);
  HISA_CHECK_CONTIG(k_concat);
  HISA_CHECK_CONTIG(grad_q);
  HISA_CHECK_CONTIG(grad_k_concat);
  HISA_CHECK_CONTIG(grad_w);

  HISA_CHECK_DTYPE(grad_cand_score, torch::kFloat32);
  HISA_CHECK_DTYPE(grad_block_score, torch::kFloat32);
  HISA_CHECK_DTYPE(q, torch::kFloat32);
  HISA_CHECK_DTYPE(k_concat, torch::kFloat32);
  HISA_CHECK_DTYPE(k_offsets, torch::kInt32);
  HISA_CHECK_DTYPE(weights, torch::kFloat32);
  HISA_CHECK_DTYPE(token_to_batch, torch::kInt32);
  HISA_CHECK_DTYPE(prefix_lens, torch::kInt32);
  HISA_CHECK_DTYPE(top_blocks, torch::kInt32);
  HISA_CHECK_DTYPE(candidate_indices, torch::kInt32);
  HISA_CHECK_DTYPE(candidate_dot, torch::kFloat32);
  HISA_CHECK_DTYPE(block_dot, torch::kFloat32);
  HISA_CHECK_DTYPE(selected_block_mask, torch::kUInt8);
  HISA_CHECK_DTYPE(selected_cand_mask, torch::kUInt8);
  HISA_CHECK_DTYPE(grad_q, torch::kFloat32);
  HISA_CHECK_DTYPE(grad_k_concat, torch::kFloat32);
  HISA_CHECK_DTYPE(grad_w, torch::kFloat32);

  TORCH_CHECK(q.dim() == 3, "q must be [Q, H, D]");
  const int Q = q.size(0);
  const int H = q.size(1);
  const int D = q.size(2);
  const int CL = candidate_dot.size(1);
  const int MB = block_dot.size(1);
  const int TB = top_blocks.size(1);
  TORCH_CHECK(D == 128, "HISA requires head_dim=128 (got ", D, ")");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_hisa_score_bwd(
      grad_cand_score.data_ptr<float>(), grad_block_score.data_ptr<float>(),
      q.data_ptr<float>(), k_concat.data_ptr<float>(),
      k_offsets.data_ptr<int32_t>(), weights.data_ptr<float>(),
      token_to_batch.data_ptr<int32_t>(), prefix_lens.data_ptr<int32_t>(),
      top_blocks.data_ptr<int32_t>(), candidate_indices.data_ptr<int32_t>(),
      candidate_dot.data_ptr<float>(), block_dot.data_ptr<float>(),
      selected_block_mask.data_ptr<uint8_t>(),
      selected_cand_mask.data_ptr<uint8_t>(),
      grad_q.data_ptr<float>(), grad_k_concat.data_ptr<float>(),
      grad_w.data_ptr<float>(), Q, H, D, CL, MB, static_cast<int>(block_size),
      TB, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hisa_selector_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor block_reps,
    torch::Tensor weights,
    torch::Tensor prefix_lens,
    torch::Tensor block_topk_counts,
    torch::Tensor topk_indices,
    torch::Tensor selected_scores,
    int64_t block_size,
    int64_t effective_block_topk,
    int64_t topk_tokens,
    bool force_first,
    bool force_last,
    bool force_last_minus_one) {
  HISA_CHECK_CUDA(q);
  HISA_CHECK_CUDA(k);
  HISA_CHECK_CUDA(block_reps);
  HISA_CHECK_CUDA(weights);
  HISA_CHECK_CUDA(prefix_lens);
  HISA_CHECK_CUDA(block_topk_counts);
  HISA_CHECK_CUDA(topk_indices);
  HISA_CHECK_CUDA(selected_scores);

  HISA_CHECK_CONTIG(q);
  HISA_CHECK_CONTIG(k);
  HISA_CHECK_CONTIG(block_reps);
  HISA_CHECK_CONTIG(weights);
  HISA_CHECK_CONTIG(prefix_lens);
  HISA_CHECK_CONTIG(block_topk_counts);
  HISA_CHECK_CONTIG(topk_indices);
  HISA_CHECK_CONTIG(selected_scores);

  HISA_CHECK_DTYPE(q, torch::kFloat32);
  HISA_CHECK_DTYPE(k, torch::kFloat32);
  HISA_CHECK_DTYPE(block_reps, torch::kFloat32);
  HISA_CHECK_DTYPE(weights, torch::kFloat32);
  HISA_CHECK_DTYPE(prefix_lens, torch::kInt32);
  HISA_CHECK_DTYPE(block_topk_counts, torch::kInt32);
  HISA_CHECK_DTYPE(topk_indices, torch::kInt32);
  HISA_CHECK_DTYPE(selected_scores, torch::kFloat32);

  TORCH_CHECK(q.dim() == 3, "q must be [Q, H, D]");
  TORCH_CHECK(k.dim() == 2, "k must be [L, D]");
  TORCH_CHECK(block_reps.dim() == 2, "block_reps must be [MB, D]");
  TORCH_CHECK(weights.dim() == 2, "weights must be [Q, H]");
  TORCH_CHECK(prefix_lens.dim() == 1, "prefix_lens must be [Q]");
  TORCH_CHECK(block_topk_counts.dim() == 1, "block_topk_counts must be [Q]");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be [Q, K]");
  TORCH_CHECK(selected_scores.dim() == 2, "selected_scores must be [Q, K]");

  const int Q = q.size(0);
  const int H = q.size(1);
  const int D = q.size(2);
  const int L = k.size(0);
  const int MB = block_reps.size(0);
  const int K = topk_indices.size(1);

  TORCH_CHECK(D == 128, "HISA selector requires head_dim=128 (got ", D, ")");
  TORCH_CHECK(k.size(1) == D, "k dim must match q head dim");
  TORCH_CHECK(block_reps.size(1) == D, "block_reps dim must match q head dim");
  TORCH_CHECK(weights.size(0) == Q && weights.size(1) == H, "weights must be [Q, H]");
  TORCH_CHECK(prefix_lens.size(0) == Q, "prefix_lens must be [Q]");
  TORCH_CHECK(block_topk_counts.size(0) == Q, "block_topk_counts must be [Q]");
  TORCH_CHECK(selected_scores.size(0) == Q && selected_scores.size(1) == K,
              "selected_scores must match topk_indices shape");
  TORCH_CHECK(topk_tokens == K, "topk_tokens must equal topk_indices.size(1)");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(effective_block_topk > 0, "effective_block_topk must be positive");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_hisa_selector_fwd(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      block_reps.data_ptr<float>(),
      weights.data_ptr<float>(),
      prefix_lens.data_ptr<int32_t>(),
      block_topk_counts.data_ptr<int32_t>(),
      topk_indices.data_ptr<int32_t>(),
      selected_scores.data_ptr<float>(),
      Q, H, D, L, MB, static_cast<int>(block_size),
      static_cast<int>(effective_block_topk), static_cast<int>(topk_tokens),
      force_first ? 1 : 0, force_last ? 1 : 0,
      force_last_minus_one ? 1 : 0, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hisa_selector_nvfp4_fwd(
    torch::Tensor q,
    torch::Tensor packed_values,
    torch::Tensor packed_scales,
    torch::Tensor block_reps,
    torch::Tensor weights,
    torch::Tensor prefix_lens,
    torch::Tensor block_topk_counts,
    torch::Tensor topk_indices,
    torch::Tensor selected_scores,
    int64_t key_length,
    int64_t packed_row_offset,
    int64_t packed_row_stride,
    int64_t block_size,
    int64_t effective_block_topk,
    int64_t topk_tokens,
    bool force_first,
    bool force_last,
    bool force_last_minus_one) {
  HISA_CHECK_CUDA(q);
  HISA_CHECK_CUDA(packed_values);
  HISA_CHECK_CUDA(packed_scales);
  HISA_CHECK_CUDA(block_reps);
  HISA_CHECK_CUDA(weights);
  HISA_CHECK_CUDA(prefix_lens);
  HISA_CHECK_CUDA(block_topk_counts);
  HISA_CHECK_CUDA(topk_indices);
  HISA_CHECK_CUDA(selected_scores);

  HISA_CHECK_CONTIG(q);
  HISA_CHECK_CONTIG(packed_values);
  HISA_CHECK_CONTIG(packed_scales);
  HISA_CHECK_CONTIG(block_reps);
  HISA_CHECK_CONTIG(weights);
  HISA_CHECK_CONTIG(prefix_lens);
  HISA_CHECK_CONTIG(block_topk_counts);
  HISA_CHECK_CONTIG(topk_indices);
  HISA_CHECK_CONTIG(selected_scores);

  HISA_CHECK_DTYPE(q, torch::kFloat32);
  HISA_CHECK_DTYPE(packed_values, torch::kUInt8);
  HISA_CHECK_DTYPE(packed_scales, torch::kInt32);
  HISA_CHECK_DTYPE(block_reps, torch::kFloat32);
  HISA_CHECK_DTYPE(weights, torch::kFloat32);
  HISA_CHECK_DTYPE(prefix_lens, torch::kInt32);
  HISA_CHECK_DTYPE(block_topk_counts, torch::kInt32);
  HISA_CHECK_DTYPE(topk_indices, torch::kInt32);
  HISA_CHECK_DTYPE(selected_scores, torch::kFloat32);

  TORCH_CHECK(q.dim() == 3, "q must be [Q, H, D]");
  TORCH_CHECK(packed_values.dim() == 2, "packed_values must be [N, 64]");
  TORCH_CHECK(packed_scales.dim() == 1, "packed_scales must be [N]");
  TORCH_CHECK(block_reps.dim() == 2, "block_reps must be [MB, D]");
  TORCH_CHECK(weights.dim() == 2, "weights must be [Q, H]");
  TORCH_CHECK(prefix_lens.dim() == 1, "prefix_lens must be [Q]");
  TORCH_CHECK(block_topk_counts.dim() == 1, "block_topk_counts must be [Q]");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be [Q, K]");
  TORCH_CHECK(selected_scores.dim() == 2, "selected_scores must be [Q, K]");

  const int Q = q.size(0);
  const int H = q.size(1);
  const int D = q.size(2);
  const int L = static_cast<int>(key_length);
  const int MB = block_reps.size(0);
  const int K = topk_indices.size(1);

  TORCH_CHECK(D == 128, "HISA selector requires head_dim=128 (got ", D, ")");
  TORCH_CHECK(packed_values.size(1) == 64, "NVFP4 packed values must have 64 bytes per row");
  TORCH_CHECK(packed_scales.size(0) == packed_values.size(0),
              "packed scale/value row count mismatch");
  TORCH_CHECK(block_reps.size(1) == D, "block_reps dim must match q head dim");
  TORCH_CHECK(weights.size(0) == Q && weights.size(1) == H, "weights must be [Q, H]");
  TORCH_CHECK(prefix_lens.size(0) == Q, "prefix_lens must be [Q]");
  TORCH_CHECK(block_topk_counts.size(0) == Q, "block_topk_counts must be [Q]");
  TORCH_CHECK(selected_scores.size(0) == Q && selected_scores.size(1) == K,
              "selected_scores must match topk_indices shape");
  TORCH_CHECK(topk_tokens == K, "topk_tokens must equal topk_indices.size(1)");
  TORCH_CHECK(L >= 0, "key_length must be non-negative");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(effective_block_topk > 0, "effective_block_topk must be positive");
  TORCH_CHECK(packed_row_stride > 0, "packed_row_stride must be positive");
  TORCH_CHECK(packed_row_offset >= 0, "packed_row_offset must be non-negative");
  if (L > 0) {
    TORCH_CHECK(
        packed_row_offset + (static_cast<int64_t>(L) - 1) * packed_row_stride <
            packed_values.size(0),
        "packed row mapping exceeds packed_values rows");
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_hisa_selector_nvfp4_fwd(
      q.data_ptr<float>(),
      packed_values.data_ptr<uint8_t>(),
      packed_scales.data_ptr<int32_t>(),
      block_reps.data_ptr<float>(),
      weights.data_ptr<float>(),
      prefix_lens.data_ptr<int32_t>(),
      block_topk_counts.data_ptr<int32_t>(),
      topk_indices.data_ptr<int32_t>(),
      selected_scores.data_ptr<float>(),
      Q, H, D, L, static_cast<int>(packed_row_offset),
      static_cast<int>(packed_row_stride), MB, static_cast<int>(block_size),
      static_cast<int>(effective_block_topk), static_cast<int>(topk_tokens),
      force_first ? 1 : 0, force_last ? 1 : 0,
      force_last_minus_one ? 1 : 0, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hisa_selector_nvfp4_cublasdx_fwd(
    torch::Tensor q,
    torch::Tensor packed_values,
    torch::Tensor packed_scales,
    torch::Tensor block_reps,
    torch::Tensor weights,
    torch::Tensor prefix_lens,
    torch::Tensor block_topk_counts,
    torch::Tensor topk_indices,
    torch::Tensor selected_scores,
    int64_t key_length,
    int64_t packed_row_offset,
    int64_t packed_row_stride,
    int64_t block_size,
    int64_t effective_block_topk,
    int64_t topk_tokens,
    bool force_first,
    bool force_last,
    bool force_last_minus_one) {
  HISA_CHECK_CUDA(q);
  HISA_CHECK_CUDA(packed_values);
  HISA_CHECK_CUDA(packed_scales);
  HISA_CHECK_CUDA(block_reps);
  HISA_CHECK_CUDA(weights);
  HISA_CHECK_CUDA(prefix_lens);
  HISA_CHECK_CUDA(block_topk_counts);
  HISA_CHECK_CUDA(topk_indices);
  HISA_CHECK_CUDA(selected_scores);

  HISA_CHECK_CONTIG(q);
  HISA_CHECK_CONTIG(packed_values);
  HISA_CHECK_CONTIG(packed_scales);
  HISA_CHECK_CONTIG(block_reps);
  HISA_CHECK_CONTIG(weights);
  HISA_CHECK_CONTIG(prefix_lens);
  HISA_CHECK_CONTIG(block_topk_counts);
  HISA_CHECK_CONTIG(topk_indices);
  HISA_CHECK_CONTIG(selected_scores);

  HISA_CHECK_DTYPE(q, torch::kFloat32);
  HISA_CHECK_DTYPE(packed_values, torch::kUInt8);
  HISA_CHECK_DTYPE(packed_scales, torch::kInt32);
  HISA_CHECK_DTYPE(block_reps, torch::kFloat32);
  HISA_CHECK_DTYPE(weights, torch::kFloat32);
  HISA_CHECK_DTYPE(prefix_lens, torch::kInt32);
  HISA_CHECK_DTYPE(block_topk_counts, torch::kInt32);
  HISA_CHECK_DTYPE(topk_indices, torch::kInt32);
  HISA_CHECK_DTYPE(selected_scores, torch::kFloat32);

  TORCH_CHECK(q.dim() == 3, "q must be [Q, H, D]");
  TORCH_CHECK(packed_values.dim() == 2, "packed_values must be [N, 64]");
  TORCH_CHECK(packed_scales.dim() == 1, "packed_scales must be [N]");
  TORCH_CHECK(block_reps.dim() == 2, "block_reps must be [MB, D]");
  TORCH_CHECK(weights.dim() == 2, "weights must be [Q, H]");
  TORCH_CHECK(prefix_lens.dim() == 1, "prefix_lens must be [Q]");
  TORCH_CHECK(block_topk_counts.dim() == 1, "block_topk_counts must be [Q]");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be [Q, K]");
  TORCH_CHECK(selected_scores.dim() == 2, "selected_scores must be [Q, K]");

  const int Q = q.size(0);
  const int H = q.size(1);
  const int D = q.size(2);
  const int L = static_cast<int>(key_length);
  const int MB = block_reps.size(0);
  const int K = topk_indices.size(1);

  TORCH_CHECK(H == 64, "cuBLASDx NVFP4 HISA selector requires 64 indexer heads");
  TORCH_CHECK(D == 128, "cuBLASDx NVFP4 HISA selector requires head_dim=128");
  TORCH_CHECK(packed_values.size(1) == 64, "NVFP4 packed values must have 64 bytes per row");
  TORCH_CHECK(packed_scales.size(0) == packed_values.size(0),
              "packed scale/value row count mismatch");
  TORCH_CHECK(block_reps.size(1) == D, "block_reps dim must match q head dim");
  TORCH_CHECK(weights.size(0) == Q && weights.size(1) == H, "weights must be [Q, H]");
  TORCH_CHECK(prefix_lens.size(0) == Q, "prefix_lens must be [Q]");
  TORCH_CHECK(block_topk_counts.size(0) == Q, "block_topk_counts must be [Q]");
  TORCH_CHECK(selected_scores.size(0) == Q && selected_scores.size(1) == K,
              "selected_scores must match topk_indices shape");
  TORCH_CHECK(topk_tokens == K, "topk_tokens must equal topk_indices.size(1)");
  TORCH_CHECK(L >= 0, "key_length must be non-negative");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(effective_block_topk > 0, "effective_block_topk must be positive");
  TORCH_CHECK(packed_row_stride > 0, "packed_row_stride must be positive");
  TORCH_CHECK(packed_row_offset >= 0, "packed_row_offset must be non-negative");
  if (L > 0) {
    TORCH_CHECK(
        packed_row_offset + (static_cast<int64_t>(L) - 1) * packed_row_stride <
            packed_values.size(0),
        "packed row mapping exceeds packed_values rows");
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_hisa_selector_nvfp4_cublasdx_fwd(
      q.data_ptr<float>(),
      packed_values.data_ptr<uint8_t>(),
      packed_scales.data_ptr<int32_t>(),
      block_reps.data_ptr<float>(),
      weights.data_ptr<float>(),
      prefix_lens.data_ptr<int32_t>(),
      block_topk_counts.data_ptr<int32_t>(),
      topk_indices.data_ptr<int32_t>(),
      selected_scores.data_ptr<float>(),
      Q, H, D, L, static_cast<int>(packed_row_offset),
      static_cast<int>(packed_row_stride), MB, static_cast<int>(block_size),
      static_cast<int>(effective_block_topk), static_cast<int>(topk_tokens),
      force_first ? 1 : 0, force_last ? 1 : 0,
      force_last_minus_one ? 1 : 0, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hisa_selector_nvfp4_cublasdx_tiled_fwd(
    torch::Tensor q,
    torch::Tensor packed_values,
    torch::Tensor packed_scales,
    torch::Tensor block_reps,
    torch::Tensor weights,
    torch::Tensor prefix_lens,
    torch::Tensor block_topk_counts,
    torch::Tensor selected_blocks,
    torch::Tensor candidate_scores,
    torch::Tensor candidate_indices,
    torch::Tensor topk_indices,
    torch::Tensor selected_scores,
    int64_t key_length,
    int64_t packed_row_offset,
    int64_t packed_row_stride,
    int64_t block_size,
    int64_t effective_block_topk,
    int64_t topk_tokens,
    bool force_first,
    bool force_last,
    bool force_last_minus_one) {
  HISA_CHECK_CUDA(q);
  HISA_CHECK_CUDA(packed_values);
  HISA_CHECK_CUDA(packed_scales);
  HISA_CHECK_CUDA(block_reps);
  HISA_CHECK_CUDA(weights);
  HISA_CHECK_CUDA(prefix_lens);
  HISA_CHECK_CUDA(block_topk_counts);
  HISA_CHECK_CUDA(selected_blocks);
  HISA_CHECK_CUDA(candidate_scores);
  HISA_CHECK_CUDA(candidate_indices);
  HISA_CHECK_CUDA(topk_indices);
  HISA_CHECK_CUDA(selected_scores);

  HISA_CHECK_CONTIG(q);
  HISA_CHECK_CONTIG(packed_values);
  HISA_CHECK_CONTIG(packed_scales);
  HISA_CHECK_CONTIG(block_reps);
  HISA_CHECK_CONTIG(weights);
  HISA_CHECK_CONTIG(prefix_lens);
  HISA_CHECK_CONTIG(block_topk_counts);
  HISA_CHECK_CONTIG(selected_blocks);
  HISA_CHECK_CONTIG(candidate_scores);
  HISA_CHECK_CONTIG(candidate_indices);
  HISA_CHECK_CONTIG(topk_indices);
  HISA_CHECK_CONTIG(selected_scores);

  HISA_CHECK_DTYPE(q, torch::kFloat32);
  HISA_CHECK_DTYPE(packed_values, torch::kUInt8);
  HISA_CHECK_DTYPE(packed_scales, torch::kInt32);
  HISA_CHECK_DTYPE(block_reps, torch::kFloat32);
  HISA_CHECK_DTYPE(weights, torch::kFloat32);
  HISA_CHECK_DTYPE(prefix_lens, torch::kInt32);
  HISA_CHECK_DTYPE(block_topk_counts, torch::kInt32);
  HISA_CHECK_DTYPE(selected_blocks, torch::kInt32);
  HISA_CHECK_DTYPE(candidate_scores, torch::kFloat32);
  HISA_CHECK_DTYPE(candidate_indices, torch::kInt32);
  HISA_CHECK_DTYPE(topk_indices, torch::kInt32);
  HISA_CHECK_DTYPE(selected_scores, torch::kFloat32);

  TORCH_CHECK(q.dim() == 3, "q must be [Q, H, D]");
  TORCH_CHECK(packed_values.dim() == 2, "packed_values must be [N, 64]");
  TORCH_CHECK(packed_scales.dim() == 1, "packed_scales must be [N]");
  TORCH_CHECK(block_reps.dim() == 2, "block_reps must be [MB, D]");
  TORCH_CHECK(weights.dim() == 2, "weights must be [Q, H]");
  TORCH_CHECK(prefix_lens.dim() == 1, "prefix_lens must be [Q]");
  TORCH_CHECK(block_topk_counts.dim() == 1, "block_topk_counts must be [Q]");
  TORCH_CHECK(selected_blocks.dim() == 2, "selected_blocks must be [Q, TB]");
  TORCH_CHECK(candidate_scores.dim() == 2, "candidate_scores must be [Q, C]");
  TORCH_CHECK(candidate_indices.dim() == 2, "candidate_indices must be [Q, C]");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be [Q, K]");
  TORCH_CHECK(selected_scores.dim() == 2, "selected_scores must be [Q, K]");

  const int Q = q.size(0);
  const int H = q.size(1);
  const int D = q.size(2);
  const int L = static_cast<int>(key_length);
  const int MB = block_reps.size(0);
  const int K = topk_indices.size(1);
  const int candidate_capacity = candidate_scores.size(1);

  TORCH_CHECK(H == 64, "tiled cuBLASDx NVFP4 HISA selector requires 64 indexer heads");
  TORCH_CHECK(D == 128, "tiled cuBLASDx NVFP4 HISA selector requires head_dim=128");
  TORCH_CHECK(packed_values.size(1) == 64, "NVFP4 packed values must have 64 bytes per row");
  TORCH_CHECK(packed_scales.size(0) == packed_values.size(0),
              "packed scale/value row count mismatch");
  TORCH_CHECK(block_reps.size(1) == D, "block_reps dim must match q head dim");
  TORCH_CHECK(weights.size(0) == Q && weights.size(1) == H, "weights must be [Q, H]");
  TORCH_CHECK(prefix_lens.size(0) == Q, "prefix_lens must be [Q]");
  TORCH_CHECK(block_topk_counts.size(0) == Q, "block_topk_counts must be [Q]");
  TORCH_CHECK(selected_blocks.size(0) == Q, "selected_blocks row count mismatch");
  TORCH_CHECK(selected_blocks.size(1) == effective_block_topk,
              "selected_blocks must be [Q, effective_block_topk]");
  TORCH_CHECK(candidate_indices.size(0) == Q && candidate_indices.size(1) == candidate_capacity,
              "candidate_indices must match candidate_scores shape");
  TORCH_CHECK(selected_scores.size(0) == Q && selected_scores.size(1) == K,
              "selected_scores must match topk_indices shape");
  TORCH_CHECK(topk_tokens == K, "topk_tokens must equal topk_indices.size(1)");
  TORCH_CHECK(candidate_capacity >= K, "candidate scratch must be at least topk_tokens");
  TORCH_CHECK((candidate_capacity & (candidate_capacity - 1)) == 0,
              "candidate scratch width must be a power of two");
  TORCH_CHECK(L >= 0, "key_length must be non-negative");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(effective_block_topk > 0, "effective_block_topk must be positive");
  TORCH_CHECK(packed_row_stride > 0, "packed_row_stride must be positive");
  TORCH_CHECK(packed_row_offset >= 0, "packed_row_offset must be non-negative");
  if (L > 0) {
    TORCH_CHECK(
        packed_row_offset + (static_cast<int64_t>(L) - 1) * packed_row_stride <
            packed_values.size(0),
        "packed row mapping exceeds packed_values rows");
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_hisa_selector_nvfp4_cublasdx_tiled_fwd(
      q.data_ptr<float>(),
      packed_values.data_ptr<uint8_t>(),
      packed_scales.data_ptr<int32_t>(),
      block_reps.data_ptr<float>(),
      weights.data_ptr<float>(),
      prefix_lens.data_ptr<int32_t>(),
      block_topk_counts.data_ptr<int32_t>(),
      selected_blocks.data_ptr<int32_t>(),
      candidate_scores.data_ptr<float>(),
      candidate_indices.data_ptr<int32_t>(),
      topk_indices.data_ptr<int32_t>(),
      selected_scores.data_ptr<float>(),
      Q, H, D, L, static_cast<int>(packed_row_offset),
      static_cast<int>(packed_row_stride), MB, static_cast<int>(block_size),
      static_cast<int>(effective_block_topk), static_cast<int>(topk_tokens),
      candidate_capacity, force_first ? 1 : 0, force_last ? 1 : 0,
      force_last_minus_one ? 1 : 0, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hisa_selector_nvfp4_cublasdx_fp8_fwd(
    torch::Tensor q,
    torch::Tensor packed_values,
    torch::Tensor packed_scales,
    torch::Tensor block_reps,
    torch::Tensor weights,
    torch::Tensor prefix_lens,
    torch::Tensor block_topk_counts,
    torch::Tensor topk_indices,
    torch::Tensor selected_scores,
    int64_t key_length,
    int64_t packed_row_offset,
    int64_t packed_row_stride,
    int64_t block_size,
    int64_t effective_block_topk,
    int64_t topk_tokens,
    bool force_first,
    bool force_last,
    bool force_last_minus_one) {
  HISA_CHECK_CUDA(q);
  HISA_CHECK_CUDA(packed_values);
  HISA_CHECK_CUDA(packed_scales);
  HISA_CHECK_CUDA(block_reps);
  HISA_CHECK_CUDA(weights);
  HISA_CHECK_CUDA(prefix_lens);
  HISA_CHECK_CUDA(block_topk_counts);
  HISA_CHECK_CUDA(topk_indices);
  HISA_CHECK_CUDA(selected_scores);

  HISA_CHECK_CONTIG(q);
  HISA_CHECK_CONTIG(packed_values);
  HISA_CHECK_CONTIG(packed_scales);
  HISA_CHECK_CONTIG(block_reps);
  HISA_CHECK_CONTIG(weights);
  HISA_CHECK_CONTIG(prefix_lens);
  HISA_CHECK_CONTIG(block_topk_counts);
  HISA_CHECK_CONTIG(topk_indices);
  HISA_CHECK_CONTIG(selected_scores);

  HISA_CHECK_DTYPE(q, torch::kFloat32);
  HISA_CHECK_DTYPE(packed_values, torch::kUInt8);
  HISA_CHECK_DTYPE(packed_scales, torch::kInt32);
  HISA_CHECK_DTYPE(block_reps, torch::kFloat32);
  HISA_CHECK_DTYPE(weights, torch::kFloat32);
  HISA_CHECK_DTYPE(prefix_lens, torch::kInt32);
  HISA_CHECK_DTYPE(block_topk_counts, torch::kInt32);
  HISA_CHECK_DTYPE(topk_indices, torch::kInt32);
  HISA_CHECK_DTYPE(selected_scores, torch::kFloat32);

  TORCH_CHECK(q.dim() == 3, "q must be [Q, H, D]");
  TORCH_CHECK(packed_values.dim() == 2, "packed_values must be [N, 64]");
  TORCH_CHECK(packed_scales.dim() == 1, "packed_scales must be [N]");
  TORCH_CHECK(block_reps.dim() == 2, "block_reps must be [MB, D]");
  TORCH_CHECK(weights.dim() == 2, "weights must be [Q, H]");
  TORCH_CHECK(prefix_lens.dim() == 1, "prefix_lens must be [Q]");
  TORCH_CHECK(block_topk_counts.dim() == 1, "block_topk_counts must be [Q]");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be [Q, K]");
  TORCH_CHECK(selected_scores.dim() == 2, "selected_scores must be [Q, K]");

  const int Q = q.size(0);
  const int H = q.size(1);
  const int D = q.size(2);
  const int L = static_cast<int>(key_length);
  const int MB = block_reps.size(0);
  const int K = topk_indices.size(1);

  TORCH_CHECK(H == 64, "FP8 cuBLASDx NVFP4 HISA selector requires 64 indexer heads");
  TORCH_CHECK(D == 128, "FP8 cuBLASDx NVFP4 HISA selector requires head_dim=128");
  TORCH_CHECK(packed_values.size(1) == 64, "NVFP4 packed values must have 64 bytes per row");
  TORCH_CHECK(packed_scales.size(0) == packed_values.size(0),
              "packed scale/value row count mismatch");
  TORCH_CHECK(block_reps.size(1) == D, "block_reps dim must match q head dim");
  TORCH_CHECK(weights.size(0) == Q && weights.size(1) == H, "weights must be [Q, H]");
  TORCH_CHECK(prefix_lens.size(0) == Q, "prefix_lens must be [Q]");
  TORCH_CHECK(block_topk_counts.size(0) == Q, "block_topk_counts must be [Q]");
  TORCH_CHECK(selected_scores.size(0) == Q && selected_scores.size(1) == K,
              "selected_scores must match topk_indices shape");
  TORCH_CHECK(topk_tokens == K, "topk_tokens must equal topk_indices.size(1)");
  TORCH_CHECK(L >= 0, "key_length must be non-negative");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(effective_block_topk > 0, "effective_block_topk must be positive");
  TORCH_CHECK(packed_row_stride > 0, "packed_row_stride must be positive");
  TORCH_CHECK(packed_row_offset >= 0, "packed_row_offset must be non-negative");
  if (L > 0) {
    TORCH_CHECK(
        packed_row_offset + (static_cast<int64_t>(L) - 1) * packed_row_stride <
            packed_values.size(0),
        "packed row mapping exceeds packed_values rows");
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_hisa_selector_nvfp4_cublasdx_fp8_fwd(
      q.data_ptr<float>(),
      packed_values.data_ptr<uint8_t>(),
      packed_scales.data_ptr<int32_t>(),
      block_reps.data_ptr<float>(),
      weights.data_ptr<float>(),
      prefix_lens.data_ptr<int32_t>(),
      block_topk_counts.data_ptr<int32_t>(),
      topk_indices.data_ptr<int32_t>(),
      selected_scores.data_ptr<float>(),
      Q, H, D, L, static_cast<int>(packed_row_offset),
      static_cast<int>(packed_row_stride), MB, static_cast<int>(block_size),
      static_cast<int>(effective_block_topk), static_cast<int>(topk_tokens),
      force_first ? 1 : 0, force_last ? 1 : 0,
      force_last_minus_one ? 1 : 0, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hisa_selected_score_bwd(
    torch::Tensor grad_selected_scores,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor weights,
    torch::Tensor topk_indices,
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor grad_w) {
  HISA_CHECK_CUDA(grad_selected_scores);
  HISA_CHECK_CUDA(q);
  HISA_CHECK_CUDA(k);
  HISA_CHECK_CUDA(weights);
  HISA_CHECK_CUDA(topk_indices);
  HISA_CHECK_CUDA(grad_q);
  HISA_CHECK_CUDA(grad_k);
  HISA_CHECK_CUDA(grad_w);

  HISA_CHECK_CONTIG(grad_selected_scores);
  HISA_CHECK_CONTIG(q);
  HISA_CHECK_CONTIG(k);
  HISA_CHECK_CONTIG(weights);
  HISA_CHECK_CONTIG(topk_indices);
  HISA_CHECK_CONTIG(grad_q);
  HISA_CHECK_CONTIG(grad_k);
  HISA_CHECK_CONTIG(grad_w);

  HISA_CHECK_DTYPE(grad_selected_scores, torch::kFloat32);
  HISA_CHECK_DTYPE(q, torch::kFloat32);
  HISA_CHECK_DTYPE(k, torch::kFloat32);
  HISA_CHECK_DTYPE(weights, torch::kFloat32);
  HISA_CHECK_DTYPE(topk_indices, torch::kInt32);
  HISA_CHECK_DTYPE(grad_q, torch::kFloat32);
  HISA_CHECK_DTYPE(grad_k, torch::kFloat32);
  HISA_CHECK_DTYPE(grad_w, torch::kFloat32);

  TORCH_CHECK(q.dim() == 3, "q must be [Q, H, D]");
  TORCH_CHECK(k.dim() == 2, "k must be [L, D]");
  TORCH_CHECK(weights.dim() == 2, "weights must be [Q, H]");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be [Q, K]");
  TORCH_CHECK(grad_selected_scores.dim() == 2, "grad_selected_scores must be [Q, K]");

  const int Q = q.size(0);
  const int H = q.size(1);
  const int D = q.size(2);
  const int L = k.size(0);
  const int K = topk_indices.size(1);

  TORCH_CHECK(D == 128, "HISA selected-score backward requires head_dim=128 (got ", D, ")");
  TORCH_CHECK(k.size(1) == D, "k dim must match q head dim");
  TORCH_CHECK(weights.size(0) == Q && weights.size(1) == H, "weights must be [Q, H]");
  TORCH_CHECK(grad_selected_scores.size(0) == Q && grad_selected_scores.size(1) == K,
              "grad_selected_scores must match topk_indices shape");
  TORCH_CHECK(grad_q.sizes() == q.sizes(), "grad_q shape must match q");
  TORCH_CHECK(grad_k.sizes() == k.sizes(), "grad_k shape must match k");
  TORCH_CHECK(grad_w.sizes() == weights.sizes(), "grad_w shape must match weights");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_hisa_selected_score_bwd(
      grad_selected_scores.data_ptr<float>(),
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      weights.data_ptr<float>(),
      topk_indices.data_ptr<int32_t>(),
      grad_q.data_ptr<float>(),
      grad_k.data_ptr<float>(),
      grad_w.data_ptr<float>(),
      Q, H, D, L, K, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hisa_selector_teacher_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor block_reps,
    torch::Tensor weights,
    torch::Tensor attn_query,
    torch::Tensor attn_key,
    torch::Tensor prefix_lens,
    torch::Tensor block_topk_counts,
    torch::Tensor topk_indices,
    torch::Tensor selected_scores,
    torch::Tensor teacher_probs,
    int64_t block_size,
    int64_t effective_block_topk,
    int64_t topk_tokens,
    double softmax_scale,
    bool force_first,
    bool force_last,
    bool force_last_minus_one) {
  HISA_CHECK_CUDA(q);
  HISA_CHECK_CUDA(k);
  HISA_CHECK_CUDA(block_reps);
  HISA_CHECK_CUDA(weights);
  HISA_CHECK_CUDA(attn_query);
  HISA_CHECK_CUDA(attn_key);
  HISA_CHECK_CUDA(prefix_lens);
  HISA_CHECK_CUDA(block_topk_counts);
  HISA_CHECK_CUDA(topk_indices);
  HISA_CHECK_CUDA(selected_scores);
  HISA_CHECK_CUDA(teacher_probs);

  HISA_CHECK_CONTIG(q);
  HISA_CHECK_CONTIG(k);
  HISA_CHECK_CONTIG(block_reps);
  HISA_CHECK_CONTIG(weights);
  HISA_CHECK_CONTIG(attn_query);
  HISA_CHECK_CONTIG(attn_key);
  HISA_CHECK_CONTIG(prefix_lens);
  HISA_CHECK_CONTIG(block_topk_counts);
  HISA_CHECK_CONTIG(topk_indices);
  HISA_CHECK_CONTIG(selected_scores);
  HISA_CHECK_CONTIG(teacher_probs);

  HISA_CHECK_DTYPE(q, torch::kFloat32);
  HISA_CHECK_DTYPE(k, torch::kFloat32);
  HISA_CHECK_DTYPE(block_reps, torch::kFloat32);
  HISA_CHECK_DTYPE(weights, torch::kFloat32);
  HISA_CHECK_DTYPE(attn_query, torch::kFloat32);
  HISA_CHECK_DTYPE(attn_key, torch::kFloat32);
  HISA_CHECK_DTYPE(prefix_lens, torch::kInt32);
  HISA_CHECK_DTYPE(block_topk_counts, torch::kInt32);
  HISA_CHECK_DTYPE(topk_indices, torch::kInt32);
  HISA_CHECK_DTYPE(selected_scores, torch::kFloat32);
  HISA_CHECK_DTYPE(teacher_probs, torch::kFloat32);

  TORCH_CHECK(q.dim() == 3, "q must be [Q, H, D]");
  TORCH_CHECK(k.dim() == 2, "k must be [L, D]");
  TORCH_CHECK(block_reps.dim() == 2, "block_reps must be [MB, D]");
  TORCH_CHECK(weights.dim() == 2, "weights must be [Q, H]");
  TORCH_CHECK(attn_query.dim() == 3, "attn_query must be [Q, AH, AD]");
  TORCH_CHECK(attn_key.dim() == 3, "attn_key must be [L, AH, AD]");
  TORCH_CHECK(prefix_lens.dim() == 1, "prefix_lens must be [Q]");
  TORCH_CHECK(block_topk_counts.dim() == 1, "block_topk_counts must be [Q]");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be [Q, K]");
  TORCH_CHECK(selected_scores.dim() == 2, "selected_scores must be [Q, K]");
  TORCH_CHECK(teacher_probs.dim() == 2, "teacher_probs must be [Q, K]");

  const int Q = q.size(0);
  const int H = q.size(1);
  const int D = q.size(2);
  const int L = k.size(0);
  const int MB = block_reps.size(0);
  const int AH = attn_query.size(1);
  const int AD = attn_query.size(2);
  const int K = topk_indices.size(1);

  TORCH_CHECK(D == 128, "HISA selector requires head_dim=128 (got ", D, ")");
  TORCH_CHECK(k.size(1) == D, "k dim must match q head dim");
  TORCH_CHECK(block_reps.size(1) == D, "block_reps dim must match q head dim");
  TORCH_CHECK(attn_key.size(0) == L, "attn_key length must match k length");
  TORCH_CHECK(attn_key.size(1) == AH && attn_key.size(2) == AD,
              "attn_key must match attn_query head shape");
  TORCH_CHECK(weights.size(0) == Q && weights.size(1) == H, "weights must be [Q, H]");
  TORCH_CHECK(prefix_lens.size(0) == Q, "prefix_lens must be [Q]");
  TORCH_CHECK(block_topk_counts.size(0) == Q, "block_topk_counts must be [Q]");
  TORCH_CHECK(selected_scores.size(0) == Q && selected_scores.size(1) == K,
              "selected_scores must match topk_indices shape");
  TORCH_CHECK(teacher_probs.size(0) == Q && teacher_probs.size(1) == K,
              "teacher_probs must match topk_indices shape");
  TORCH_CHECK(topk_tokens == K, "topk_tokens must equal topk_indices.size(1)");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(effective_block_topk > 0, "effective_block_topk must be positive");
  TORCH_CHECK(AD > 0 && AD <= 256, "attention head_dim must be in (0, 256]");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_hisa_selector_teacher_fwd(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      block_reps.data_ptr<float>(),
      weights.data_ptr<float>(),
      attn_query.data_ptr<float>(),
      attn_key.data_ptr<float>(),
      prefix_lens.data_ptr<int32_t>(),
      block_topk_counts.data_ptr<int32_t>(),
      topk_indices.data_ptr<int32_t>(),
      selected_scores.data_ptr<float>(),
      teacher_probs.data_ptr<float>(),
      Q, H, D, L, MB, AH, AD, static_cast<int>(block_size),
      static_cast<int>(effective_block_topk), static_cast<int>(topk_tokens),
      static_cast<float>(softmax_scale), force_first ? 1 : 0,
      force_last ? 1 : 0, force_last_minus_one ? 1 : 0, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int dtype_code(torch::ScalarType dtype) {
  if (dtype == torch::kFloat32) {
    return 0;
  }
  if (dtype == torch::kBFloat16) {
    return 1;
  }
  if (dtype == torch::kFloat16) {
    return 2;
  }
  TORCH_CHECK(false, "unsupported dtype for DSA sparse K/V backward: ", dtype);
}

void check_dsa_indexer_rope_common(
    torch::Tensor x_or_grad,
    torch::Tensor freqs,
    torch::Tensor out,
    int64_t pe_dim) {
  HISA_CHECK_CUDA(x_or_grad);
  HISA_CHECK_CUDA(freqs);
  HISA_CHECK_CUDA(out);

  HISA_CHECK_CONTIG(x_or_grad);
  HISA_CHECK_CONTIG(out);

  TORCH_CHECK(x_or_grad.dim() == 4, "DSA indexer RoPE input must be [S, B, H, D]");
  TORCH_CHECK(freqs.dim() == 4, "DSA indexer RoPE freqs must be [S, B|1, H|1, pe_dim]");
  TORCH_CHECK(out.sizes() == x_or_grad.sizes(), "DSA indexer RoPE output shape mismatch");
  TORCH_CHECK(out.scalar_type() == x_or_grad.scalar_type(),
              "DSA indexer RoPE output dtype must match input dtype");
  TORCH_CHECK(
      x_or_grad.scalar_type() == torch::kFloat32 ||
          x_or_grad.scalar_type() == torch::kBFloat16 ||
          x_or_grad.scalar_type() == torch::kFloat16,
      "DSA indexer RoPE supports fp32/bf16/fp16 inputs");
  TORCH_CHECK(
      freqs.scalar_type() == torch::kFloat32 ||
          freqs.scalar_type() == torch::kBFloat16 ||
          freqs.scalar_type() == torch::kFloat16,
      "DSA indexer RoPE supports fp32/bf16/fp16 freqs");

  const int64_t S = x_or_grad.size(0);
  const int64_t B = x_or_grad.size(1);
  const int64_t H = x_or_grad.size(2);
  const int64_t D = x_or_grad.size(3);
  TORCH_CHECK(D > 0 && D <= 256, "DSA indexer RoPE head_dim must be in (0, 256]");
  TORCH_CHECK(pe_dim > 0 && pe_dim <= D, "DSA indexer RoPE pe_dim must be in (0, head_dim]");
  TORCH_CHECK((pe_dim % 2) == 0, "DSA indexer RoPE pe_dim must be even");
  TORCH_CHECK(freqs.size(3) == pe_dim, "DSA indexer RoPE freqs last dim must equal pe_dim");
  TORCH_CHECK(freqs.size(0) == 1 || freqs.size(0) == S,
              "DSA indexer RoPE freqs sequence dim must be 1 or match input");
  TORCH_CHECK(freqs.size(1) == 1 || freqs.size(1) == B,
              "DSA indexer RoPE freqs batch dim must be 1 or match input");
  TORCH_CHECK(freqs.size(2) == 1 || freqs.size(2) == H,
              "DSA indexer RoPE freqs head dim must be 1 or match input");
}

void dsa_indexer_rope_fwd(
    torch::Tensor x,
    torch::Tensor freqs,
    torch::Tensor out,
    int64_t pe_dim,
    double mscale,
    bool interleaved) {
  check_dsa_indexer_rope_common(x, freqs, out, pe_dim);

  const int S = static_cast<int>(x.size(0));
  const int B = static_cast<int>(x.size(1));
  const int H = static_cast<int>(x.size(2));
  const int D = static_cast<int>(x.size(3));
  const int64_t total_rows = static_cast<int64_t>(S) * B * H;
  if (total_rows == 0) {
    return;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_dsa_indexer_rope_fwd(
      x.data_ptr(),
      freqs.data_ptr(),
      out.data_ptr(),
      total_rows,
      S,
      B,
      H,
      D,
      static_cast<int>(pe_dim),
      static_cast<int>(freqs.size(0)),
      static_cast<int>(freqs.size(1)),
      static_cast<int>(freqs.size(2)),
      freqs.stride(0),
      freqs.stride(1),
      freqs.stride(2),
      freqs.stride(3),
      dtype_code(x.scalar_type()),
      dtype_code(freqs.scalar_type()),
      static_cast<float>(mscale),
      interleaved ? 1 : 0,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsa_indexer_rope_fwd_inplace(
    torch::Tensor x,
    torch::Tensor freqs,
    int64_t pe_dim,
    double mscale,
    bool interleaved) {
  check_dsa_indexer_rope_common(x, freqs, x, pe_dim);

  const int S = static_cast<int>(x.size(0));
  const int B = static_cast<int>(x.size(1));
  const int H = static_cast<int>(x.size(2));
  const int D = static_cast<int>(x.size(3));
  const int64_t total_rows = static_cast<int64_t>(S) * B * H;
  if (total_rows == 0) {
    return;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_dsa_indexer_rope_fwd_inplace(
      x.data_ptr(),
      freqs.data_ptr(),
      total_rows,
      S,
      B,
      H,
      D,
      static_cast<int>(pe_dim),
      static_cast<int>(freqs.size(0)),
      static_cast<int>(freqs.size(1)),
      static_cast<int>(freqs.size(2)),
      freqs.stride(0),
      freqs.stride(1),
      freqs.stride(2),
      freqs.stride(3),
      dtype_code(x.scalar_type()),
      dtype_code(freqs.scalar_type()),
      static_cast<float>(mscale),
      interleaved ? 1 : 0,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_dsa_indexer_rope_flat_common(
    torch::Tensor x_or_grad,
    torch::Tensor freqs,
    torch::Tensor out,
    int64_t heads,
    int64_t head_dim,
    int64_t pe_dim) {
  HISA_CHECK_CUDA(x_or_grad);
  HISA_CHECK_CUDA(freqs);
  HISA_CHECK_CUDA(out);

  HISA_CHECK_CONTIG(x_or_grad);
  HISA_CHECK_CONTIG(out);

  TORCH_CHECK(x_or_grad.dim() == 3, "flat DSA indexer RoPE input must be [S, B, H*D]");
  TORCH_CHECK(freqs.dim() == 4, "flat DSA indexer RoPE freqs must be [S, B|1, H|1, pe_dim]");
  TORCH_CHECK(out.sizes() == x_or_grad.sizes(), "flat DSA indexer RoPE output shape mismatch");
  TORCH_CHECK(out.scalar_type() == x_or_grad.scalar_type(),
              "flat DSA indexer RoPE output dtype must match input dtype");
  TORCH_CHECK(
      x_or_grad.scalar_type() == torch::kFloat32 ||
          x_or_grad.scalar_type() == torch::kBFloat16 ||
          x_or_grad.scalar_type() == torch::kFloat16,
      "flat DSA indexer RoPE supports fp32/bf16/fp16 inputs");
  TORCH_CHECK(
      freqs.scalar_type() == torch::kFloat32 ||
          freqs.scalar_type() == torch::kBFloat16 ||
          freqs.scalar_type() == torch::kFloat16,
      "flat DSA indexer RoPE supports fp32/bf16/fp16 freqs");

  const int64_t S = x_or_grad.size(0);
  const int64_t B = x_or_grad.size(1);
  TORCH_CHECK(heads > 0, "flat DSA indexer RoPE heads must be positive");
  TORCH_CHECK(head_dim > 0 && head_dim <= 256,
              "flat DSA indexer RoPE head_dim must be in (0, 256]");
  TORCH_CHECK(x_or_grad.size(2) == heads * head_dim,
              "flat DSA indexer RoPE last dim must equal heads * head_dim");
  TORCH_CHECK(pe_dim > 0 && pe_dim <= head_dim,
              "flat DSA indexer RoPE pe_dim must be in (0, head_dim]");
  TORCH_CHECK((pe_dim % 2) == 0, "flat DSA indexer RoPE pe_dim must be even");
  TORCH_CHECK(freqs.size(3) == pe_dim, "flat DSA indexer RoPE freqs last dim must equal pe_dim");
  TORCH_CHECK(freqs.size(0) == 1 || freqs.size(0) == S,
              "flat DSA indexer RoPE freqs sequence dim must be 1 or match input");
  TORCH_CHECK(freqs.size(1) == 1 || freqs.size(1) == B,
              "flat DSA indexer RoPE freqs batch dim must be 1 or match input");
  TORCH_CHECK(freqs.size(2) == 1 || freqs.size(2) == heads,
              "flat DSA indexer RoPE freqs head dim must be 1 or match heads");
}

void dsa_indexer_rope_fwd_inplace_flat(
    torch::Tensor x,
    torch::Tensor freqs,
    int64_t heads,
    int64_t head_dim,
    int64_t pe_dim,
    double mscale,
    bool interleaved) {
  check_dsa_indexer_rope_flat_common(x, freqs, x, heads, head_dim, pe_dim);

  const int S = static_cast<int>(x.size(0));
  const int B = static_cast<int>(x.size(1));
  const int64_t total_rows = static_cast<int64_t>(S) * B * heads;
  if (total_rows == 0) {
    return;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_dsa_indexer_rope_fwd_inplace(
      x.data_ptr(),
      freqs.data_ptr(),
      total_rows,
      S,
      B,
      static_cast<int>(heads),
      static_cast<int>(head_dim),
      static_cast<int>(pe_dim),
      static_cast<int>(freqs.size(0)),
      static_cast<int>(freqs.size(1)),
      static_cast<int>(freqs.size(2)),
      freqs.stride(0),
      freqs.stride(1),
      freqs.stride(2),
      freqs.stride(3),
      dtype_code(x.scalar_type()),
      dtype_code(freqs.scalar_type()),
      static_cast<float>(mscale),
      interleaved ? 1 : 0,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsa_indexer_rope_bwd_flat(
    torch::Tensor grad_out,
    torch::Tensor freqs,
    torch::Tensor grad_x,
    int64_t heads,
    int64_t head_dim,
    int64_t pe_dim,
    double mscale,
    bool interleaved) {
  check_dsa_indexer_rope_flat_common(grad_out, freqs, grad_x, heads, head_dim, pe_dim);

  const int S = static_cast<int>(grad_out.size(0));
  const int B = static_cast<int>(grad_out.size(1));
  const int64_t total_rows = static_cast<int64_t>(S) * B * heads;
  if (total_rows == 0) {
    return;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_dsa_indexer_rope_bwd(
      grad_out.data_ptr(),
      freqs.data_ptr(),
      grad_x.data_ptr(),
      total_rows,
      S,
      B,
      static_cast<int>(heads),
      static_cast<int>(head_dim),
      static_cast<int>(pe_dim),
      static_cast<int>(freqs.size(0)),
      static_cast<int>(freqs.size(1)),
      static_cast<int>(freqs.size(2)),
      freqs.stride(0),
      freqs.stride(1),
      freqs.stride(2),
      freqs.stride(3),
      dtype_code(grad_out.scalar_type()),
      dtype_code(freqs.scalar_type()),
      static_cast<float>(mscale),
      interleaved ? 1 : 0,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsa_indexer_rope_bwd(
    torch::Tensor grad_out,
    torch::Tensor freqs,
    torch::Tensor grad_x,
    int64_t pe_dim,
    double mscale,
    bool interleaved) {
  check_dsa_indexer_rope_common(grad_out, freqs, grad_x, pe_dim);

  const int S = static_cast<int>(grad_out.size(0));
  const int B = static_cast<int>(grad_out.size(1));
  const int H = static_cast<int>(grad_out.size(2));
  const int D = static_cast<int>(grad_out.size(3));
  const int64_t total_rows = static_cast<int64_t>(S) * B * H;
  if (total_rows == 0) {
    return;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_dsa_indexer_rope_bwd(
      grad_out.data_ptr(),
      freqs.data_ptr(),
      grad_x.data_ptr(),
      total_rows,
      S,
      B,
      H,
      D,
      static_cast<int>(pe_dim),
      static_cast<int>(freqs.size(0)),
      static_cast<int>(freqs.size(1)),
      static_cast<int>(freqs.size(2)),
      freqs.stride(0),
      freqs.stride(1),
      freqs.stride(2),
      freqs.stride(3),
      dtype_code(grad_out.scalar_type()),
      dtype_code(freqs.scalar_type()),
      static_cast<float>(mscale),
      interleaved ? 1 : 0,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int topk_dtype_code(torch::ScalarType dtype) {
  if (dtype == torch::kInt16) {
    return 0;
  }
  if (dtype == torch::kInt32) {
    return 1;
  }
  if (dtype == torch::kInt64) {
    return 2;
  }
  TORCH_CHECK(false, "unsupported topk dtype for DSA sparse K/V backward: ", dtype);
}

void dsa_sparse_kv_bwd(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor topk_indices,
    torch::Tensor output,
    torch::Tensor lse,
    torch::Tensor grad_output,
    torch::Tensor grad_key,
    torch::Tensor grad_value,
    double softmax_scale,
    int64_t q_start,
    int64_t tile_q,
    int64_t tile_k) {
  HISA_CHECK_CUDA(query);
  HISA_CHECK_CUDA(key);
  HISA_CHECK_CUDA(value);
  HISA_CHECK_CUDA(topk_indices);
  HISA_CHECK_CUDA(output);
  HISA_CHECK_CUDA(lse);
  HISA_CHECK_CUDA(grad_output);
  HISA_CHECK_CUDA(grad_key);
  HISA_CHECK_CUDA(grad_value);

  HISA_CHECK_CONTIG(query);
  HISA_CHECK_CONTIG(key);
  HISA_CHECK_CONTIG(value);
  HISA_CHECK_CONTIG(topk_indices);
  HISA_CHECK_CONTIG(output);
  HISA_CHECK_CONTIG(lse);
  HISA_CHECK_CONTIG(grad_output);
  HISA_CHECK_CONTIG(grad_key);
  HISA_CHECK_CONTIG(grad_value);

  TORCH_CHECK(query.dim() == 4, "query must be [Q, B, H, D]");
  TORCH_CHECK(key.dim() == 4, "key must be [S, B, H, D]");
  TORCH_CHECK(value.dim() == 4, "value must be [S, B, H, V]");
  TORCH_CHECK(topk_indices.dim() == 3, "topk_indices must be [B, Q, K]");
  TORCH_CHECK(output.dim() == 4, "output must be [Q, B, H, V]");
  TORCH_CHECK(grad_output.dim() == 4, "grad_output must be [Q, B, H, V]");
  TORCH_CHECK(lse.dim() == 2, "lse must be [B * Q, H]");

  const int Q = query.size(0);
  const int B = query.size(1);
  const int H = query.size(2);
  const int D = query.size(3);
  const int S = key.size(0);
  const int V = value.size(3);
  const int K = topk_indices.size(2);

  TORCH_CHECK(key.size(1) == B && key.size(2) == H && key.size(3) == D,
              "key shape must match query batch/head/head_dim");
  TORCH_CHECK(value.size(0) == S && value.size(1) == B && value.size(2) == H,
              "value shape must match key sequence/batch/head");
  TORCH_CHECK(topk_indices.size(0) == B && topk_indices.size(1) == Q,
              "topk_indices must be [B, Q, K]");
  TORCH_CHECK(output.size(0) == Q && output.size(1) == B && output.size(2) == H &&
                  output.size(3) == V,
              "output shape must be [Q, B, H, V]");
  TORCH_CHECK(grad_output.size(0) == Q && grad_output.size(1) == B &&
                  grad_output.size(2) == H && grad_output.size(3) == V,
              "grad_output shape must match output");
  TORCH_CHECK(lse.size(0) == B * Q && lse.size(1) == H,
              "lse must be [B * Q, H]");
  TORCH_CHECK(grad_key.sizes() == key.sizes(), "grad_key shape must match key");
  TORCH_CHECK(grad_value.sizes() == value.sizes(), "grad_value shape must match value");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type() &&
                  query.scalar_type() == output.scalar_type() &&
                  query.scalar_type() == grad_output.scalar_type(),
              "query/key/value/output/grad_output must share dtype");
  TORCH_CHECK(grad_key.scalar_type() == grad_value.scalar_type(),
              "grad_key and grad_value must share dtype");
  HISA_CHECK_DTYPE(lse, torch::kFloat32);
  TORCH_CHECK(D > 0 && D <= 256, "DSA sparse K/V backward supports head_dim in (0, 256]");
  TORCH_CHECK(V > 0 && V <= 256, "DSA sparse K/V backward supports value_dim in (0, 256]");
  TORCH_CHECK(K > 0, "topk count must be positive");
  TORCH_CHECK(tile_q > 0 && tile_k > 0, "tile_q and tile_k must be positive");
  TORCH_CHECK(tile_q * tile_k <= 8,
              "DSA sparse K/V backward currently supports at most 8 edge-warps per CTA");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_dsa_sparse_kv_bwd(
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      topk_indices.data_ptr(),
      output.data_ptr(),
      lse.data_ptr<float>(),
      grad_output.data_ptr(),
      grad_key.data_ptr(),
      grad_value.data_ptr(),
      static_cast<float>(softmax_scale),
      Q,
      B,
      S,
      H,
      D,
      V,
      K,
      static_cast<int>(q_start),
      dtype_code(query.scalar_type()),
      topk_dtype_code(topk_indices.scalar_type()),
      dtype_code(grad_key.scalar_type()),
      static_cast<int>(tile_q),
      static_cast<int>(tile_k),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsa_sparse_bwd_from_scores(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor topk_indices,
    torch::Tensor selected_scores,
    torch::Tensor output,
    torch::Tensor lse,
    torch::Tensor grad_output,
    torch::Tensor grad_query,
    torch::Tensor grad_key,
    torch::Tensor grad_value,
    double softmax_scale,
    int64_t tile_q,
    int64_t tile_k) {
  HISA_CHECK_CUDA(query);
  HISA_CHECK_CUDA(key);
  HISA_CHECK_CUDA(value);
  HISA_CHECK_CUDA(topk_indices);
  HISA_CHECK_CUDA(selected_scores);
  HISA_CHECK_CUDA(output);
  HISA_CHECK_CUDA(lse);
  HISA_CHECK_CUDA(grad_output);
  HISA_CHECK_CUDA(grad_query);
  HISA_CHECK_CUDA(grad_key);
  HISA_CHECK_CUDA(grad_value);

  HISA_CHECK_CONTIG(query);
  HISA_CHECK_CONTIG(key);
  HISA_CHECK_CONTIG(value);
  HISA_CHECK_CONTIG(topk_indices);
  HISA_CHECK_CONTIG(selected_scores);
  HISA_CHECK_CONTIG(output);
  HISA_CHECK_CONTIG(lse);
  HISA_CHECK_CONTIG(grad_output);
  HISA_CHECK_CONTIG(grad_query);
  HISA_CHECK_CONTIG(grad_key);
  HISA_CHECK_CONTIG(grad_value);

  TORCH_CHECK(query.dim() == 4, "query must be [Q, B, H, D]");
  TORCH_CHECK(key.dim() == 4, "key must be [S, B, H, D]");
  TORCH_CHECK(value.dim() == 4, "value must be [S, B, H, V]");
  TORCH_CHECK(topk_indices.dim() == 3, "topk_indices must be [B, Q, K]");
  TORCH_CHECK(selected_scores.dim() == 3, "selected_scores must be [B * Q, H, K]");
  TORCH_CHECK(output.dim() == 4, "output must be [Q, B, H, V]");
  TORCH_CHECK(grad_output.dim() == 4, "grad_output must be [Q, B, H, V]");
  TORCH_CHECK(grad_query.dim() == 4, "grad_query must be [Q, B, H, D]");
  TORCH_CHECK(lse.dim() == 2, "lse must be [B * Q, H]");

  const int Q = query.size(0);
  const int B = query.size(1);
  const int H = query.size(2);
  const int D = query.size(3);
  const int S = key.size(0);
  const int V = value.size(3);
  const int K = topk_indices.size(2);

  TORCH_CHECK(key.size(1) == B && key.size(2) == H && key.size(3) == D,
              "key shape must match query batch/head/head_dim");
  TORCH_CHECK(value.size(0) == S && value.size(1) == B && value.size(2) == H,
              "value shape must match key sequence/batch/head");
  TORCH_CHECK(topk_indices.size(0) == B && topk_indices.size(1) == Q,
              "topk_indices must be [B, Q, K]");
  TORCH_CHECK(selected_scores.size(0) == B * Q && selected_scores.size(1) == H &&
                  selected_scores.size(2) == K,
              "selected_scores must be [B * Q, H, K]");
  TORCH_CHECK(output.size(0) == Q && output.size(1) == B && output.size(2) == H &&
                  output.size(3) == V,
              "output shape must be [Q, B, H, V]");
  TORCH_CHECK(grad_output.size(0) == Q && grad_output.size(1) == B &&
                  grad_output.size(2) == H && grad_output.size(3) == V,
              "grad_output shape must match output");
  TORCH_CHECK(grad_query.sizes() == query.sizes(), "grad_query shape must match query");
  TORCH_CHECK(lse.size(0) == B * Q && lse.size(1) == H,
              "lse must be [B * Q, H]");
  TORCH_CHECK(grad_key.sizes() == key.sizes(), "grad_key shape must match key");
  TORCH_CHECK(grad_value.sizes() == value.sizes(), "grad_value shape must match value");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type() &&
                  query.scalar_type() == output.scalar_type() &&
                  query.scalar_type() == grad_output.scalar_type(),
              "query/key/value/output/grad_output must share dtype");
  HISA_CHECK_DTYPE(selected_scores, torch::kFloat32);
  HISA_CHECK_DTYPE(lse, torch::kFloat32);
  HISA_CHECK_DTYPE(grad_query, torch::kFloat32);
  TORCH_CHECK(grad_key.scalar_type() == grad_value.scalar_type(),
              "grad_key and grad_value must share dtype");
  TORCH_CHECK(D > 0 && D <= 256, "DSA sparse backward supports head_dim in (0, 256]");
  TORCH_CHECK(V > 0 && V <= 256, "DSA sparse backward supports value_dim in (0, 256]");
  TORCH_CHECK(K > 0, "topk count must be positive");
  TORCH_CHECK(tile_q > 0 && tile_k > 0, "tile_q and tile_k must be positive");
  TORCH_CHECK(tile_q * tile_k <= 8,
              "DSA sparse backward currently supports at most 8 edge-warps per CTA");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_dsa_sparse_bwd_from_scores(
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      topk_indices.data_ptr(),
      selected_scores.data_ptr<float>(),
      output.data_ptr(),
      lse.data_ptr<float>(),
      grad_output.data_ptr(),
      grad_query.data_ptr<float>(),
      grad_key.data_ptr(),
      grad_value.data_ptr(),
      Q,
      B,
      S,
      H,
      D,
      V,
      K,
      static_cast<float>(softmax_scale),
      dtype_code(query.scalar_type()),
      topk_dtype_code(topk_indices.scalar_type()),
      dtype_code(grad_key.scalar_type()),
      static_cast<int>(tile_q),
      static_cast<int>(tile_k),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsa_sparse_bwd_from_scores_row(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor topk_indices,
    torch::Tensor selected_scores,
    torch::Tensor output,
    torch::Tensor lse,
    torch::Tensor grad_output,
    torch::Tensor grad_query,
    torch::Tensor grad_key,
    torch::Tensor grad_value,
    double softmax_scale) {
  HISA_CHECK_CUDA(query);
  HISA_CHECK_CUDA(key);
  HISA_CHECK_CUDA(value);
  HISA_CHECK_CUDA(topk_indices);
  HISA_CHECK_CUDA(selected_scores);
  HISA_CHECK_CUDA(output);
  HISA_CHECK_CUDA(lse);
  HISA_CHECK_CUDA(grad_output);
  HISA_CHECK_CUDA(grad_query);
  HISA_CHECK_CUDA(grad_key);
  HISA_CHECK_CUDA(grad_value);

  HISA_CHECK_CONTIG(query);
  HISA_CHECK_CONTIG(key);
  HISA_CHECK_CONTIG(value);
  HISA_CHECK_CONTIG(topk_indices);
  HISA_CHECK_CONTIG(selected_scores);
  HISA_CHECK_CONTIG(output);
  HISA_CHECK_CONTIG(lse);
  HISA_CHECK_CONTIG(grad_output);
  HISA_CHECK_CONTIG(grad_query);
  HISA_CHECK_CONTIG(grad_key);
  HISA_CHECK_CONTIG(grad_value);

  TORCH_CHECK(query.dim() == 4, "query must be [Q, B, H, D]");
  TORCH_CHECK(key.dim() == 4, "key must be [S, B, H, D]");
  TORCH_CHECK(value.dim() == 4, "value must be [S, B, H, V]");
  TORCH_CHECK(topk_indices.dim() == 3, "topk_indices must be [B, Q, K]");
  TORCH_CHECK(selected_scores.dim() == 3, "selected_scores must be [B * Q, H, K]");
  TORCH_CHECK(output.dim() == 4, "output must be [Q, B, H, V]");
  TORCH_CHECK(grad_output.dim() == 4, "grad_output must be [Q, B, H, V]");
  TORCH_CHECK(grad_query.dim() == 4, "grad_query must be [Q, B, H, D]");
  TORCH_CHECK(lse.dim() == 2, "lse must be [B * Q, H]");

  const int Q = query.size(0);
  const int B = query.size(1);
  const int H = query.size(2);
  const int D = query.size(3);
  const int S = key.size(0);
  const int V = value.size(3);
  const int K = topk_indices.size(2);

  TORCH_CHECK(key.size(1) == B && key.size(2) == H && key.size(3) == D,
              "key shape must match query batch/head/head_dim");
  TORCH_CHECK(value.size(0) == S && value.size(1) == B && value.size(2) == H,
              "value shape must match key sequence/batch/head");
  TORCH_CHECK(topk_indices.size(0) == B && topk_indices.size(1) == Q,
              "topk_indices must be [B, Q, K]");
  TORCH_CHECK(selected_scores.size(0) == B * Q && selected_scores.size(1) == H &&
                  selected_scores.size(2) == K,
              "selected_scores must be [B * Q, H, K]");
  TORCH_CHECK(output.size(0) == Q && output.size(1) == B && output.size(2) == H &&
                  output.size(3) == V,
              "output shape must be [Q, B, H, V]");
  TORCH_CHECK(grad_output.size(0) == Q && grad_output.size(1) == B &&
                  grad_output.size(2) == H && grad_output.size(3) == V,
              "grad_output shape must match output");
  TORCH_CHECK(grad_query.sizes() == query.sizes(), "grad_query shape must match query");
  TORCH_CHECK(lse.size(0) == B * Q && lse.size(1) == H,
              "lse must be [B * Q, H]");
  TORCH_CHECK(grad_key.sizes() == key.sizes(), "grad_key shape must match key");
  TORCH_CHECK(grad_value.sizes() == value.sizes(), "grad_value shape must match value");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type() &&
                  query.scalar_type() == output.scalar_type() &&
                  query.scalar_type() == grad_output.scalar_type(),
              "query/key/value/output/grad_output must share dtype");
  HISA_CHECK_DTYPE(selected_scores, torch::kFloat32);
  HISA_CHECK_DTYPE(lse, torch::kFloat32);
  HISA_CHECK_DTYPE(grad_query, torch::kFloat32);
  TORCH_CHECK(grad_key.scalar_type() == grad_value.scalar_type(),
              "grad_key and grad_value must share dtype");
  TORCH_CHECK(D > 0 && D <= 256, "DSA row backward supports head_dim in (0, 256]");
  TORCH_CHECK(V > 0 && V <= 256, "DSA row backward supports value_dim in (0, 256]");
  TORCH_CHECK(K > 0, "topk count must be positive");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_dsa_sparse_bwd_from_scores_row(
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      topk_indices.data_ptr(),
      selected_scores.data_ptr<float>(),
      output.data_ptr(),
      lse.data_ptr<float>(),
      grad_output.data_ptr(),
      grad_query.data_ptr<float>(),
      grad_key.data_ptr(),
      grad_value.data_ptr(),
      Q,
      B,
      S,
      H,
      D,
      V,
      K,
      static_cast<float>(softmax_scale),
      dtype_code(query.scalar_type()),
      topk_dtype_code(topk_indices.scalar_type()),
      dtype_code(grad_key.scalar_type()),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dsa_sparse_kv_bwd_sorted_from_scores(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor topk_indices,
    torch::Tensor selected_scores,
    torch::Tensor output,
    torch::Tensor lse,
    torch::Tensor grad_output,
    torch::Tensor grad_key,
    torch::Tensor grad_value,
    double softmax_scale) {
  HISA_CHECK_CUDA(query);
  HISA_CHECK_CUDA(key);
  HISA_CHECK_CUDA(value);
  HISA_CHECK_CUDA(topk_indices);
  HISA_CHECK_CUDA(selected_scores);
  HISA_CHECK_CUDA(output);
  HISA_CHECK_CUDA(lse);
  HISA_CHECK_CUDA(grad_output);
  HISA_CHECK_CUDA(grad_key);
  HISA_CHECK_CUDA(grad_value);

  HISA_CHECK_CONTIG(query);
  HISA_CHECK_CONTIG(key);
  HISA_CHECK_CONTIG(value);
  HISA_CHECK_CONTIG(topk_indices);
  HISA_CHECK_CONTIG(selected_scores);
  HISA_CHECK_CONTIG(output);
  HISA_CHECK_CONTIG(lse);
  HISA_CHECK_CONTIG(grad_output);
  HISA_CHECK_CONTIG(grad_key);
  HISA_CHECK_CONTIG(grad_value);

  TORCH_CHECK(query.dim() == 4, "query must be [Q, B, H, D]");
  TORCH_CHECK(key.dim() == 4, "key must be [S, B, H, D]");
  TORCH_CHECK(value.dim() == 4, "value must be [S, B, H, V]");
  TORCH_CHECK(topk_indices.dim() == 3, "topk_indices must be [B, Q, K]");
  TORCH_CHECK(selected_scores.dim() == 3, "selected_scores must be [B * Q, H, K]");
  TORCH_CHECK(output.dim() == 4, "output must be [Q, B, H, V]");
  TORCH_CHECK(grad_output.dim() == 4, "grad_output must be [Q, B, H, V]");
  TORCH_CHECK(lse.dim() == 2, "lse must be [B * Q, H]");

  const int Q = query.size(0);
  const int B = query.size(1);
  const int H = query.size(2);
  const int D = query.size(3);
  const int S = key.size(0);
  const int V = value.size(3);
  const int K = topk_indices.size(2);

  TORCH_CHECK(key.size(1) == B && key.size(2) == H && key.size(3) == D,
              "key shape must match query batch/head/head_dim");
  TORCH_CHECK(value.size(0) == S && value.size(1) == B && value.size(2) == H,
              "value shape must match key sequence/batch/head");
  TORCH_CHECK(topk_indices.size(0) == B && topk_indices.size(1) == Q,
              "topk_indices must be [B, Q, K]");
  TORCH_CHECK(selected_scores.size(0) == B * Q && selected_scores.size(1) == H &&
                  selected_scores.size(2) == K,
              "selected_scores must be [B * Q, H, K]");
  TORCH_CHECK(output.size(0) == Q && output.size(1) == B && output.size(2) == H &&
                  output.size(3) == V,
              "output shape must be [Q, B, H, V]");
  TORCH_CHECK(grad_output.size(0) == Q && grad_output.size(1) == B &&
                  grad_output.size(2) == H && grad_output.size(3) == V,
              "grad_output shape must match output");
  TORCH_CHECK(lse.size(0) == B * Q && lse.size(1) == H,
              "lse must be [B * Q, H]");
  TORCH_CHECK(grad_key.sizes() == key.sizes(), "grad_key shape must match key");
  TORCH_CHECK(grad_value.sizes() == value.sizes(), "grad_value shape must match value");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type() &&
                  query.scalar_type() == output.scalar_type() &&
                  query.scalar_type() == grad_output.scalar_type(),
              "query/key/value/output/grad_output must share dtype");
  HISA_CHECK_DTYPE(selected_scores, torch::kFloat32);
  HISA_CHECK_DTYPE(lse, torch::kFloat32);
  TORCH_CHECK(grad_key.scalar_type() == grad_value.scalar_type(),
              "grad_key and grad_value must share dtype");
  TORCH_CHECK(D > 0 && D <= 256, "DSA sorted K/V backward supports head_dim in (0, 256]");
  TORCH_CHECK(V > 0 && V <= 256, "DSA sorted K/V backward supports value_dim in (0, 256]");
  TORCH_CHECK(K > 0, "topk count must be positive");

  const int64_t total_edges = static_cast<int64_t>(Q) * B * H * K;
  auto edge_keys = torch::empty({total_edges}, topk_indices.options().dtype(torch::kInt64));
  auto edge_ids = torch::empty({total_edges}, topk_indices.options().dtype(torch::kInt64));
  auto edge_prob = torch::empty({total_edges}, selected_scores.options().dtype(torch::kFloat32));
  auto edge_ds = torch::empty({total_edges}, selected_scores.options().dtype(torch::kFloat32));
  auto delta = torch::empty({B * Q, H}, selected_scores.options().dtype(torch::kFloat32));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  megatron::hisa_indexer::launch_dsa_sparse_kv_bwd_sorted_from_scores(
      query.data_ptr(),
      value.data_ptr(),
      topk_indices.data_ptr(),
      selected_scores.data_ptr<float>(),
      output.data_ptr(),
      lse.data_ptr<float>(),
      grad_output.data_ptr(),
      reinterpret_cast<uint64_t*>(edge_keys.data_ptr<int64_t>()),
      edge_ids.data_ptr<int64_t>(),
      edge_prob.data_ptr<float>(),
      edge_ds.data_ptr<float>(),
      delta.data_ptr<float>(),
      grad_key.data_ptr(),
      grad_value.data_ptr(),
      Q,
      B,
      S,
      H,
      D,
      V,
      K,
      static_cast<float>(softmax_scale),
      dtype_code(query.scalar_type()),
      topk_dtype_code(topk_indices.scalar_type()),
      dtype_code(grad_key.scalar_type()),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hisa_score_bwd", &hisa_score_bwd, "HISA score backward (CUDA)");
  m.def("hisa_selector_fwd", &hisa_selector_fwd, "HISA selector forward (CUDA)");
  m.def(
      "hisa_selector_nvfp4_fwd",
      &hisa_selector_nvfp4_fwd,
      "HISA selector forward from packed NVFP4 IndexCache rows (CUDA)");
  m.def(
      "hisa_selector_nvfp4_cublasdx_fwd",
      &hisa_selector_nvfp4_cublasdx_fwd,
      "HISA selector forward from packed NVFP4 IndexCache rows (cuBLASDx/CuTe)");
  m.def(
      "hisa_selector_nvfp4_cublasdx_tiled_fwd",
      &hisa_selector_nvfp4_cublasdx_tiled_fwd,
      "Tiled HISA selector forward from packed NVFP4 IndexCache rows (cuBLASDx/CuTe)");
  m.def(
      "hisa_selector_nvfp4_cublasdx_fp8_fwd",
      &hisa_selector_nvfp4_cublasdx_fp8_fwd,
      "HISA selector forward from packed NVFP4 rows with FP8 cuBLASDx MMA (CUDA)");
  m.def(
      "hisa_selected_score_bwd",
      &hisa_selected_score_bwd,
      "HISA selected-score backward (CUDA)");
  m.def(
      "hisa_selector_teacher_fwd",
      &hisa_selector_teacher_fwd,
      "HISA selector + local teacher forward (CUDA)");
  m.def(
      "dsa_sparse_kv_bwd",
      &dsa_sparse_kv_bwd,
      "Sparse DSA selected K/V backward edge-tile reducer (CUDA)");
  m.def(
      "dsa_sparse_bwd_from_scores",
      &dsa_sparse_bwd_from_scores,
      "Sparse DSA selected backward from forward-saved scores (CUDA)");
  m.def(
      "dsa_sparse_bwd_from_scores_row",
      &dsa_sparse_bwd_from_scores_row,
      "Sparse DSA row-owned backward from forward-saved scores (CUDA)");
  m.def(
      "dsa_sparse_kv_bwd_sorted_from_scores",
      &dsa_sparse_kv_bwd_sorted_from_scores,
      "Sparse DSA sorted-segment K/V backward from forward-saved scores (CUDA)");
  m.def(
      "dsa_indexer_rope_fwd",
      &dsa_indexer_rope_fwd,
      "DSA indexer RoPE full-head writer forward (CUDA)");
  m.def(
      "dsa_indexer_rope_fwd_inplace",
      &dsa_indexer_rope_fwd_inplace,
      "DSA indexer RoPE full-head writer in-place forward (CUDA)");
  m.def(
      "dsa_indexer_rope_fwd_inplace_flat",
      &dsa_indexer_rope_fwd_inplace_flat,
      "DSA indexer RoPE flat projected in-place forward (CUDA)");
  m.def(
      "dsa_indexer_rope_bwd_flat",
      &dsa_indexer_rope_bwd_flat,
      "DSA indexer RoPE flat projected backward (CUDA)");
  m.def(
      "dsa_indexer_rope_bwd",
      &dsa_indexer_rope_bwd,
      "DSA indexer RoPE full-head writer backward (CUDA)");
}
