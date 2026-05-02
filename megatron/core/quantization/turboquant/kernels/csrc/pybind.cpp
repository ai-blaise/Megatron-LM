// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace megatron {
namespace turboquant {

template <typename scalar_t>
void launch_turboquant_kv_fwd(
    const scalar_t* x, scalar_t* out, uint8_t* indices, uint8_t* ste_mask,
    float* norm_out, float* inner_norm_out, __nv_bfloat16* w_hat_save,
    const float* signs1, const float* signs2,
    const float* boundaries_high, const float* boundaries_low,
    const float* centroids_high, const float* centroids_low,
    int64_t num_rows, int64_t row_stride, bool norm_correction,
    cudaStream_t stream);

template <typename scalar_t>
void launch_turboquant_kv_bwd(
    const scalar_t* grad_out, const scalar_t* x,
    const uint8_t* indices, const uint8_t* ste_mask,
    const float* norm_arr, const float* inner_norm_arr,
    const __nv_bfloat16* w_hat_saved,
    const float* signs1, const float* signs2,
    const float* centroids_high, const float* centroids_low,
    scalar_t* grad_x, int64_t num_rows, int64_t row_stride,
    bool norm_correction, cudaStream_t stream);

}  // namespace turboquant
}  // namespace megatron

namespace {

#define TQ_CHECK_CUDA(t) TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor")
#define TQ_CHECK_CONTIG(t) TORCH_CHECK((t).is_contiguous(), #t " must be contiguous")
#define TQ_CHECK_DTYPE(t, dt) TORCH_CHECK((t).scalar_type() == (dt), #t " has wrong dtype")

__nv_bfloat16* maybe_w_hat_ptr(c10::optional<torch::Tensor> w_hat) {
  if (!w_hat.has_value() || !w_hat->defined()) return nullptr;
  TQ_CHECK_CUDA((*w_hat));
  TQ_CHECK_DTYPE((*w_hat), torch::kBFloat16);
  return reinterpret_cast<__nv_bfloat16*>(w_hat->data_ptr());
}

const __nv_bfloat16* maybe_w_hat_const(c10::optional<torch::Tensor> w_hat) {
  if (!w_hat.has_value() || !w_hat->defined()) return nullptr;
  TQ_CHECK_CUDA((*w_hat));
  TQ_CHECK_DTYPE((*w_hat), torch::kBFloat16);
  return reinterpret_cast<const __nv_bfloat16*>(w_hat->data_ptr());
}

void turboquant_kv_fwd(
    torch::Tensor x, torch::Tensor out,
    torch::Tensor indices, torch::Tensor ste_mask,
    torch::Tensor norm_out, torch::Tensor inner_norm_out,
    c10::optional<torch::Tensor> w_hat_save,
    torch::Tensor signs1, torch::Tensor signs2,
    torch::Tensor boundaries_high, torch::Tensor boundaries_low,
    torch::Tensor centroids_high, torch::Tensor centroids_low,
    bool norm_correction) {
  TQ_CHECK_CUDA(x); TQ_CHECK_CUDA(out);
  TQ_CHECK_CUDA(indices); TQ_CHECK_CUDA(ste_mask);
  TQ_CHECK_CUDA(norm_out); TQ_CHECK_CUDA(inner_norm_out);
  TQ_CHECK_CONTIG(x); TQ_CHECK_CONTIG(out);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 512, "x must be [N, 512]");
  TORCH_CHECK(out.sizes() == x.sizes(), "out shape mismatch");
  TORCH_CHECK(indices.size(0) == x.size(0) && indices.size(1) == 512,
              "indices must be [N, 512]");
  TQ_CHECK_DTYPE(indices, torch::kUInt8);
  TQ_CHECK_DTYPE(ste_mask, torch::kUInt8);
  TQ_CHECK_DTYPE(norm_out, torch::kFloat32);
  TQ_CHECK_DTYPE(inner_norm_out, torch::kFloat32);

  __nv_bfloat16* w_hat_ptr = maybe_w_hat_ptr(w_hat_save);

  const int64_t num_rows = x.size(0);
  const int64_t row_stride = x.size(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      x.scalar_type(), "turboquant_kv_fwd",
      AT_DISPATCH_CASE(torch::kFloat32, [&] {
        megatron::turboquant::launch_turboquant_kv_fwd<float>(
            x.data_ptr<float>(), out.data_ptr<float>(),
            indices.data_ptr<uint8_t>(), ste_mask.data_ptr<uint8_t>(),
            norm_out.data_ptr<float>(), inner_norm_out.data_ptr<float>(),
            w_hat_ptr,
            signs1.data_ptr<float>(), signs2.data_ptr<float>(),
            boundaries_high.data_ptr<float>(), boundaries_low.data_ptr<float>(),
            centroids_high.data_ptr<float>(), centroids_low.data_ptr<float>(),
            num_rows, row_stride, norm_correction, stream);
      })
      AT_DISPATCH_CASE(torch::kBFloat16, [&] {
        megatron::turboquant::launch_turboquant_kv_fwd<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            indices.data_ptr<uint8_t>(), ste_mask.data_ptr<uint8_t>(),
            norm_out.data_ptr<float>(), inner_norm_out.data_ptr<float>(),
            w_hat_ptr,
            signs1.data_ptr<float>(), signs2.data_ptr<float>(),
            boundaries_high.data_ptr<float>(), boundaries_low.data_ptr<float>(),
            centroids_high.data_ptr<float>(), centroids_low.data_ptr<float>(),
            num_rows, row_stride, norm_correction, stream);
      })
      AT_DISPATCH_CASE(torch::kFloat16, [&] {
        megatron::turboquant::launch_turboquant_kv_fwd<__half>(
            reinterpret_cast<const __half*>(x.data_ptr()),
            reinterpret_cast<__half*>(out.data_ptr()),
            indices.data_ptr<uint8_t>(), ste_mask.data_ptr<uint8_t>(),
            norm_out.data_ptr<float>(), inner_norm_out.data_ptr<float>(),
            w_hat_ptr,
            signs1.data_ptr<float>(), signs2.data_ptr<float>(),
            boundaries_high.data_ptr<float>(), boundaries_low.data_ptr<float>(),
            centroids_high.data_ptr<float>(), centroids_low.data_ptr<float>(),
            num_rows, row_stride, norm_correction, stream);
      }));
}

void turboquant_kv_bwd(
    torch::Tensor grad_out, torch::Tensor x,
    torch::Tensor indices, torch::Tensor ste_mask,
    torch::Tensor norm_arr, torch::Tensor inner_norm_arr,
    c10::optional<torch::Tensor> w_hat_saved,
    torch::Tensor signs1, torch::Tensor signs2,
    torch::Tensor centroids_high, torch::Tensor centroids_low,
    torch::Tensor grad_x, bool norm_correction) {
  TQ_CHECK_CUDA(grad_out); TQ_CHECK_CUDA(x); TQ_CHECK_CUDA(grad_x);
  TQ_CHECK_CONTIG(grad_out); TQ_CHECK_CONTIG(x); TQ_CHECK_CONTIG(grad_x);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 512, "x must be [N, 512]");
  TORCH_CHECK(grad_out.sizes() == x.sizes(), "grad_out shape mismatch");
  TORCH_CHECK(grad_x.sizes() == x.sizes(), "grad_x shape mismatch");

  const __nv_bfloat16* w_hat_ptr = maybe_w_hat_const(w_hat_saved);

  const int64_t num_rows = x.size(0);
  const int64_t row_stride = x.size(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      x.scalar_type(), "turboquant_kv_bwd",
      AT_DISPATCH_CASE(torch::kFloat32, [&] {
        megatron::turboquant::launch_turboquant_kv_bwd<float>(
            grad_out.data_ptr<float>(), x.data_ptr<float>(),
            indices.data_ptr<uint8_t>(), ste_mask.data_ptr<uint8_t>(),
            norm_arr.data_ptr<float>(), inner_norm_arr.data_ptr<float>(),
            w_hat_ptr,
            signs1.data_ptr<float>(), signs2.data_ptr<float>(),
            centroids_high.data_ptr<float>(), centroids_low.data_ptr<float>(),
            grad_x.data_ptr<float>(),
            num_rows, row_stride, norm_correction, stream);
      })
      AT_DISPATCH_CASE(torch::kBFloat16, [&] {
        megatron::turboquant::launch_turboquant_kv_bwd<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            indices.data_ptr<uint8_t>(), ste_mask.data_ptr<uint8_t>(),
            norm_arr.data_ptr<float>(), inner_norm_arr.data_ptr<float>(),
            w_hat_ptr,
            signs1.data_ptr<float>(), signs2.data_ptr<float>(),
            centroids_high.data_ptr<float>(), centroids_low.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr()),
            num_rows, row_stride, norm_correction, stream);
      })
      AT_DISPATCH_CASE(torch::kFloat16, [&] {
        megatron::turboquant::launch_turboquant_kv_bwd<__half>(
            reinterpret_cast<const __half*>(grad_out.data_ptr()),
            reinterpret_cast<const __half*>(x.data_ptr()),
            indices.data_ptr<uint8_t>(), ste_mask.data_ptr<uint8_t>(),
            norm_arr.data_ptr<float>(), inner_norm_arr.data_ptr<float>(),
            w_hat_ptr,
            signs1.data_ptr<float>(), signs2.data_ptr<float>(),
            centroids_high.data_ptr<float>(), centroids_low.data_ptr<float>(),
            reinterpret_cast<__half*>(grad_x.data_ptr()),
            num_rows, row_stride, norm_correction, stream);
      }));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("turboquant_kv_fwd", &turboquant_kv_fwd, "TurboQuant KV forward");
  m.def("turboquant_kv_bwd", &turboquant_kv_bwd, "TurboQuant KV backward");
}
