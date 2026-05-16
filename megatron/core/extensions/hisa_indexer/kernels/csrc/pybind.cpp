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

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hisa_score_bwd", &hisa_score_bwd, "HISA score backward (CUDA)");
  m.def("hisa_selector_fwd", &hisa_selector_fwd, "HISA selector forward (CUDA)");
  m.def(
      "hisa_selected_score_bwd",
      &hisa_selected_score_bwd,
      "HISA selected-score backward (CUDA)");
  m.def(
      "hisa_selector_teacher_fwd",
      &hisa_selector_teacher_fwd,
      "HISA selector + local teacher forward (CUDA)");
}
