// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace megatron {
namespace indexcache {

template <typename scalar_t>
void launch_indexcache_kv_fwd(
    const scalar_t* x, scalar_t* out, float* scale,
    __nv_fp8_e4m3* q_fp8, uint8_t* clip_mask, int32_t* argmax, uint8_t* eps_active,
    float eps, float fp8_max, int64_t num_rows, int64_t row_stride,
    cudaStream_t stream);

template <typename scalar_t>
void launch_indexcache_kv_bwd(
    const scalar_t* grad_y, const scalar_t* x, const float* scale,
    const __nv_fp8_e4m3* q_fp8, const uint8_t* clip_mask,
    const int32_t* argmax, const uint8_t* eps_active,
    scalar_t* grad_x, float fp8_max,
    int64_t num_rows, int64_t row_stride, cudaStream_t stream);

}  // namespace indexcache
}  // namespace megatron

namespace {

#define IC_CHECK_CUDA(t) TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor")
#define IC_CHECK_CONTIG(t) TORCH_CHECK((t).is_contiguous(), #t " must be contiguous")
#define IC_CHECK_DTYPE(t, dt) TORCH_CHECK((t).scalar_type() == (dt), #t " has wrong dtype")

void indexcache_kv_fwd(
    torch::Tensor x, torch::Tensor out,
    torch::Tensor scale, torch::Tensor q_fp8,
    torch::Tensor clip_mask, torch::Tensor argmax, torch::Tensor eps_active,
    double eps, double fp8_max) {
  IC_CHECK_CUDA(x); IC_CHECK_CUDA(out);
  IC_CHECK_CUDA(scale); IC_CHECK_CUDA(q_fp8);
  IC_CHECK_CUDA(clip_mask); IC_CHECK_CUDA(argmax); IC_CHECK_CUDA(eps_active);
  IC_CHECK_CONTIG(x); IC_CHECK_CONTIG(out);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 128, "x must be [N, 128]");
  IC_CHECK_DTYPE(scale, torch::kFloat32);
  IC_CHECK_DTYPE(q_fp8, torch::kFloat8_e4m3fn);
  IC_CHECK_DTYPE(clip_mask, torch::kUInt8);
  IC_CHECK_DTYPE(argmax, torch::kInt32);
  IC_CHECK_DTYPE(eps_active, torch::kUInt8);

  const int64_t num_rows = x.size(0);
  const int64_t row_stride = x.size(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float eps_f = static_cast<float>(eps);
  const float fp8_max_f = static_cast<float>(fp8_max);

  AT_DISPATCH_SWITCH(
      x.scalar_type(), "indexcache_kv_fwd",
      AT_DISPATCH_CASE(torch::kFloat32, [&] {
        megatron::indexcache::launch_indexcache_kv_fwd<float>(
            x.data_ptr<float>(), out.data_ptr<float>(),
            scale.data_ptr<float>(),
            reinterpret_cast<__nv_fp8_e4m3*>(q_fp8.data_ptr()),
            clip_mask.data_ptr<uint8_t>(),
            argmax.data_ptr<int32_t>(),
            eps_active.data_ptr<uint8_t>(),
            eps_f, fp8_max_f, num_rows, row_stride, stream);
      })
      AT_DISPATCH_CASE(torch::kBFloat16, [&] {
        megatron::indexcache::launch_indexcache_kv_fwd<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            scale.data_ptr<float>(),
            reinterpret_cast<__nv_fp8_e4m3*>(q_fp8.data_ptr()),
            clip_mask.data_ptr<uint8_t>(),
            argmax.data_ptr<int32_t>(),
            eps_active.data_ptr<uint8_t>(),
            eps_f, fp8_max_f, num_rows, row_stride, stream);
      })
      AT_DISPATCH_CASE(torch::kFloat16, [&] {
        megatron::indexcache::launch_indexcache_kv_fwd<__half>(
            reinterpret_cast<const __half*>(x.data_ptr()),
            reinterpret_cast<__half*>(out.data_ptr()),
            scale.data_ptr<float>(),
            reinterpret_cast<__nv_fp8_e4m3*>(q_fp8.data_ptr()),
            clip_mask.data_ptr<uint8_t>(),
            argmax.data_ptr<int32_t>(),
            eps_active.data_ptr<uint8_t>(),
            eps_f, fp8_max_f, num_rows, row_stride, stream);
      }));
}

void indexcache_kv_bwd(
    torch::Tensor grad_y, torch::Tensor x,
    torch::Tensor scale, torch::Tensor q_fp8,
    torch::Tensor clip_mask, torch::Tensor argmax, torch::Tensor eps_active,
    torch::Tensor grad_x, double fp8_max) {
  IC_CHECK_CUDA(grad_y); IC_CHECK_CUDA(x); IC_CHECK_CUDA(grad_x);
  IC_CHECK_CONTIG(grad_y); IC_CHECK_CONTIG(x); IC_CHECK_CONTIG(grad_x);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 128, "x must be [N, 128]");

  const int64_t num_rows = x.size(0);
  const int64_t row_stride = x.size(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float fp8_max_f = static_cast<float>(fp8_max);

  AT_DISPATCH_SWITCH(
      x.scalar_type(), "indexcache_kv_bwd",
      AT_DISPATCH_CASE(torch::kFloat32, [&] {
        megatron::indexcache::launch_indexcache_kv_bwd<float>(
            grad_y.data_ptr<float>(), x.data_ptr<float>(),
            scale.data_ptr<float>(),
            reinterpret_cast<const __nv_fp8_e4m3*>(q_fp8.data_ptr()),
            clip_mask.data_ptr<uint8_t>(),
            argmax.data_ptr<int32_t>(),
            eps_active.data_ptr<uint8_t>(),
            grad_x.data_ptr<float>(), fp8_max_f,
            num_rows, row_stride, stream);
      })
      AT_DISPATCH_CASE(torch::kBFloat16, [&] {
        megatron::indexcache::launch_indexcache_kv_bwd<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(grad_y.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            scale.data_ptr<float>(),
            reinterpret_cast<const __nv_fp8_e4m3*>(q_fp8.data_ptr()),
            clip_mask.data_ptr<uint8_t>(),
            argmax.data_ptr<int32_t>(),
            eps_active.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr()), fp8_max_f,
            num_rows, row_stride, stream);
      })
      AT_DISPATCH_CASE(torch::kFloat16, [&] {
        megatron::indexcache::launch_indexcache_kv_bwd<__half>(
            reinterpret_cast<const __half*>(grad_y.data_ptr()),
            reinterpret_cast<const __half*>(x.data_ptr()),
            scale.data_ptr<float>(),
            reinterpret_cast<const __nv_fp8_e4m3*>(q_fp8.data_ptr()),
            clip_mask.data_ptr<uint8_t>(),
            argmax.data_ptr<int32_t>(),
            eps_active.data_ptr<uint8_t>(),
            reinterpret_cast<__half*>(grad_x.data_ptr()), fp8_max_f,
            num_rows, row_stride, stream);
      }));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexcache_kv_fwd", &indexcache_kv_fwd, "IndexCache KV forward");
  m.def("indexcache_kv_bwd", &indexcache_kv_bwd, "IndexCache KV backward");
}
