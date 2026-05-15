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

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hisa_score_bwd", &hisa_score_bwd, "HISA score backward (CUDA)");
}
