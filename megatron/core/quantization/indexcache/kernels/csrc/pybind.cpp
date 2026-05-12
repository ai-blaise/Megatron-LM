// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

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

template <typename scalar_t>
void launch_indexcache_nvfp4_fwd(
    const scalar_t* x, scalar_t* out, float* scale, float* q_e2m1,
    uint8_t* clip_mask, int32_t* argmax, uint8_t* eps_active,
    uint8_t* packed_values, int32_t* packed_scales, float eps,
    int64_t num_rows, int64_t row_stride, cudaStream_t stream);

template <typename scalar_t>
void launch_indexcache_nvfp4_bwd(
    const scalar_t* grad_y, const scalar_t* x, const float* scale,
    const float* q_e2m1, const uint8_t* clip_mask, const int32_t* argmax,
    const uint8_t* eps_active, scalar_t* grad_x, float fp4_max,
    int64_t num_rows, int64_t row_stride, cudaStream_t stream);

}  // namespace indexcache
}  // namespace megatron

namespace {

#define IC_CHECK_CUDA(t) TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor")
#define IC_CHECK_CONTIG(t) TORCH_CHECK((t).is_contiguous(), #t " must be contiguous")
#define IC_CHECK_DTYPE(t, dt) TORCH_CHECK((t).scalar_type() == (dt), #t " has wrong dtype")

void ic_check_blackwell() {
  int device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  TORCH_CHECK(
      prop.major >= 10,
      "NVFP4 IndexCache CUDA kernels require Blackwell (SM100+) execution; "
      "use the reference fallback on this device.");
}

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

void indexcache_nvfp4_fwd(
    torch::Tensor x, torch::Tensor out,
    torch::Tensor scale, torch::Tensor q_e2m1,
    torch::Tensor clip_mask, torch::Tensor argmax, torch::Tensor eps_active,
    torch::Tensor packed_values, torch::Tensor packed_scales,
    double eps) {
  IC_CHECK_CUDA(x); IC_CHECK_CUDA(out);
  IC_CHECK_CUDA(scale); IC_CHECK_CUDA(q_e2m1);
  IC_CHECK_CUDA(clip_mask); IC_CHECK_CUDA(argmax); IC_CHECK_CUDA(eps_active);
  IC_CHECK_CUDA(packed_values); IC_CHECK_CUDA(packed_scales);
  IC_CHECK_CONTIG(x); IC_CHECK_CONTIG(out);
  IC_CHECK_CONTIG(scale); IC_CHECK_CONTIG(q_e2m1);
  IC_CHECK_CONTIG(clip_mask); IC_CHECK_CONTIG(argmax); IC_CHECK_CONTIG(eps_active);
  IC_CHECK_CONTIG(packed_values); IC_CHECK_CONTIG(packed_scales);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 128, "x must be [N, 128]");
  TORCH_CHECK(scale.dim() == 2 && scale.size(0) == x.size(0) && scale.size(1) == 4,
              "scale must be [N, 4]");
  TORCH_CHECK(q_e2m1.sizes() == x.sizes(), "q_e2m1 must match x shape");
  TORCH_CHECK(clip_mask.sizes() == x.sizes(), "clip_mask must match x shape");
  TORCH_CHECK(argmax.dim() == 2 && argmax.size(0) == x.size(0) && argmax.size(1) == 4,
              "argmax must be [N, 4]");
  TORCH_CHECK(
      eps_active.dim() == 2 && eps_active.size(0) == x.size(0) && eps_active.size(1) == 4,
      "eps_active must be [N, 4]");
  TORCH_CHECK(
      packed_values.dim() == 2 && packed_values.size(0) == x.size(0) &&
          packed_values.size(1) == 64,
      "packed_values must be [N, 64]");
  TORCH_CHECK(packed_scales.dim() == 1 && packed_scales.size(0) == x.size(0),
              "packed_scales must be [N]");
  IC_CHECK_DTYPE(scale, torch::kFloat32);
  IC_CHECK_DTYPE(q_e2m1, torch::kFloat32);
  IC_CHECK_DTYPE(clip_mask, torch::kUInt8);
  IC_CHECK_DTYPE(argmax, torch::kInt32);
  IC_CHECK_DTYPE(eps_active, torch::kUInt8);
  IC_CHECK_DTYPE(packed_values, torch::kUInt8);
  IC_CHECK_DTYPE(packed_scales, torch::kInt32);
  ic_check_blackwell();

  const int64_t num_rows = x.size(0);
  const int64_t row_stride = x.size(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float eps_f = static_cast<float>(eps);

  AT_DISPATCH_SWITCH(
      x.scalar_type(), "indexcache_nvfp4_fwd",
      AT_DISPATCH_CASE(torch::kFloat32, [&] {
        megatron::indexcache::launch_indexcache_nvfp4_fwd<float>(
            x.data_ptr<float>(), out.data_ptr<float>(),
            scale.data_ptr<float>(), q_e2m1.data_ptr<float>(),
            clip_mask.data_ptr<uint8_t>(), argmax.data_ptr<int32_t>(),
            eps_active.data_ptr<uint8_t>(), packed_values.data_ptr<uint8_t>(),
            packed_scales.data_ptr<int32_t>(), eps_f, num_rows, row_stride, stream);
      })
      AT_DISPATCH_CASE(torch::kBFloat16, [&] {
        megatron::indexcache::launch_indexcache_nvfp4_fwd<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            scale.data_ptr<float>(), q_e2m1.data_ptr<float>(),
            clip_mask.data_ptr<uint8_t>(), argmax.data_ptr<int32_t>(),
            eps_active.data_ptr<uint8_t>(), packed_values.data_ptr<uint8_t>(),
            packed_scales.data_ptr<int32_t>(), eps_f, num_rows, row_stride, stream);
      })
      AT_DISPATCH_CASE(torch::kFloat16, [&] {
        megatron::indexcache::launch_indexcache_nvfp4_fwd<__half>(
            reinterpret_cast<const __half*>(x.data_ptr()),
            reinterpret_cast<__half*>(out.data_ptr()),
            scale.data_ptr<float>(), q_e2m1.data_ptr<float>(),
            clip_mask.data_ptr<uint8_t>(), argmax.data_ptr<int32_t>(),
            eps_active.data_ptr<uint8_t>(), packed_values.data_ptr<uint8_t>(),
            packed_scales.data_ptr<int32_t>(), eps_f, num_rows, row_stride, stream);
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

void indexcache_nvfp4_bwd(
    torch::Tensor grad_y, torch::Tensor x,
    torch::Tensor scale, torch::Tensor q_e2m1,
    torch::Tensor clip_mask, torch::Tensor argmax, torch::Tensor eps_active,
    torch::Tensor grad_x, double fp4_max) {
  IC_CHECK_CUDA(grad_y); IC_CHECK_CUDA(x); IC_CHECK_CUDA(grad_x);
  IC_CHECK_CUDA(scale); IC_CHECK_CUDA(q_e2m1);
  IC_CHECK_CUDA(clip_mask); IC_CHECK_CUDA(argmax); IC_CHECK_CUDA(eps_active);
  IC_CHECK_CONTIG(grad_y); IC_CHECK_CONTIG(x); IC_CHECK_CONTIG(grad_x);
  IC_CHECK_CONTIG(scale); IC_CHECK_CONTIG(q_e2m1);
  IC_CHECK_CONTIG(clip_mask); IC_CHECK_CONTIG(argmax); IC_CHECK_CONTIG(eps_active);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 128, "x must be [N, 128]");
  TORCH_CHECK(grad_y.sizes() == x.sizes(), "grad_y must match x shape");
  TORCH_CHECK(grad_x.sizes() == x.sizes(), "grad_x must match x shape");
  TORCH_CHECK(scale.dim() == 2 && scale.size(0) == x.size(0) && scale.size(1) == 4,
              "scale must be [N, 4]");
  TORCH_CHECK(q_e2m1.sizes() == x.sizes(), "q_e2m1 must match x shape");
  TORCH_CHECK(clip_mask.sizes() == x.sizes(), "clip_mask must match x shape");
  TORCH_CHECK(argmax.dim() == 2 && argmax.size(0) == x.size(0) && argmax.size(1) == 4,
              "argmax must be [N, 4]");
  TORCH_CHECK(
      eps_active.dim() == 2 && eps_active.size(0) == x.size(0) && eps_active.size(1) == 4,
      "eps_active must be [N, 4]");
  IC_CHECK_DTYPE(scale, torch::kFloat32);
  IC_CHECK_DTYPE(q_e2m1, torch::kFloat32);
  IC_CHECK_DTYPE(clip_mask, torch::kUInt8);
  IC_CHECK_DTYPE(argmax, torch::kInt32);
  IC_CHECK_DTYPE(eps_active, torch::kUInt8);
  ic_check_blackwell();

  const int64_t num_rows = x.size(0);
  const int64_t row_stride = x.size(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float fp4_max_f = static_cast<float>(fp4_max);

  AT_DISPATCH_SWITCH(
      x.scalar_type(), "indexcache_nvfp4_bwd",
      AT_DISPATCH_CASE(torch::kFloat32, [&] {
        megatron::indexcache::launch_indexcache_nvfp4_bwd<float>(
            grad_y.data_ptr<float>(), x.data_ptr<float>(),
            scale.data_ptr<float>(), q_e2m1.data_ptr<float>(),
            clip_mask.data_ptr<uint8_t>(), argmax.data_ptr<int32_t>(),
            eps_active.data_ptr<uint8_t>(), grad_x.data_ptr<float>(),
            fp4_max_f, num_rows, row_stride, stream);
      })
      AT_DISPATCH_CASE(torch::kBFloat16, [&] {
        megatron::indexcache::launch_indexcache_nvfp4_bwd<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(grad_y.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            scale.data_ptr<float>(), q_e2m1.data_ptr<float>(),
            clip_mask.data_ptr<uint8_t>(), argmax.data_ptr<int32_t>(),
            eps_active.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr()),
            fp4_max_f, num_rows, row_stride, stream);
      })
      AT_DISPATCH_CASE(torch::kFloat16, [&] {
        megatron::indexcache::launch_indexcache_nvfp4_bwd<__half>(
            reinterpret_cast<const __half*>(grad_y.data_ptr()),
            reinterpret_cast<const __half*>(x.data_ptr()),
            scale.data_ptr<float>(), q_e2m1.data_ptr<float>(),
            clip_mask.data_ptr<uint8_t>(), argmax.data_ptr<int32_t>(),
            eps_active.data_ptr<uint8_t>(),
            reinterpret_cast<__half*>(grad_x.data_ptr()),
            fp4_max_f, num_rows, row_stride, stream);
      }));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexcache_kv_fwd", &indexcache_kv_fwd, "IndexCache KV forward");
  m.def("indexcache_kv_bwd", &indexcache_kv_bwd, "IndexCache KV backward");
  m.def("indexcache_nvfp4_fwd", &indexcache_nvfp4_fwd, "IndexCache NVFP4 forward");
  m.def("indexcache_nvfp4_bwd", &indexcache_nvfp4_bwd, "IndexCache NVFP4 backward");
}
