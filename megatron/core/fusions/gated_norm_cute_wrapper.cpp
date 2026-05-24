// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

extern "C" int megatron_gated_norm_cute_forward(const void* normed,
                                                const void* w_down,
                                                const void* w_up,
                                                void* z_workspace,
                                                void* output,
                                                int num_tokens,
                                                int hidden_size,
                                                int rank,
                                                cudaStream_t stream);

// Returns the cudaError_t encoding (0 = cudaSuccess; cudaErrorInvalidValue is
// the documented "fall back to torch.mm" signal — it's NOT a Python exception).
int64_t gated_norm_cute_fwd_torch(torch::Tensor normed,
                                  torch::Tensor w_down,
                                  torch::Tensor w_up,
                                  torch::Tensor z_workspace,
                                  torch::Tensor output,
                                  int64_t num_tokens,
                                  int64_t hidden_size,
                                  int64_t rank) {
  TORCH_CHECK(normed.is_cuda() && w_down.is_cuda() && w_up.is_cuda() &&
                  z_workspace.is_cuda() && output.is_cuda(),
              "All tensors must be CUDA");
  TORCH_CHECK(normed.dtype() == at::kBFloat16, "normed must be bf16");
  TORCH_CHECK(w_down.dtype() == at::kBFloat16, "w_down must be bf16");
  TORCH_CHECK(w_up.dtype() == at::kBFloat16, "w_up must be bf16");
  TORCH_CHECK(z_workspace.dtype() == at::kFloat, "z_workspace must be fp32");
  TORCH_CHECK(output.dtype() == at::kBFloat16, "output must be bf16");
  TORCH_CHECK(normed.is_contiguous(), "normed must be contiguous");
  TORCH_CHECK(w_down.is_contiguous(), "w_down must be contiguous");
  TORCH_CHECK(w_up.is_contiguous(), "w_up must be contiguous");
  TORCH_CHECK(z_workspace.is_contiguous(), "z_workspace must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(z_workspace.numel() >= num_tokens * rank,
              "z_workspace must have at least num_tokens * rank elements");

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  int err = megatron_gated_norm_cute_forward(normed.data_ptr(),
                                             w_down.data_ptr(),
                                             w_up.data_ptr(),
                                             z_workspace.data_ptr(),
                                             output.data_ptr(),
                                             static_cast<int>(num_tokens),
                                             static_cast<int>(hidden_size),
                                             static_cast<int>(rank),
                                             stream);
  return static_cast<int64_t>(err);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gated_norm_cute_fwd",
        &gated_norm_cute_fwd_torch,
        "Fused GatedNorm forward (CuTe SM100 BF16) writing both output and "
        "the fp32 z workspace needed by backward. Returns cudaError_t as int64. "
        "cudaErrorInvalidValue (1) signals SMEM-overflow / unsupported (R,N) "
        "→ caller must fall back to torch.mm.");
}
