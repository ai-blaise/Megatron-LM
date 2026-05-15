// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace megatron {
namespace higgs {

template <typename scalar_t>
void launch_higgs_kv_fwd(
    const scalar_t* x, scalar_t* out, uint8_t* indices, uint8_t* ste_mask,
    float* rot_norm_out, float* scale_out,
    __nv_bfloat16* rotated_save, __nv_bfloat16* recon_unit_save,
    const float* codebook, const float* codebook_norm_sq,
    int64_t num_rows, int64_t row_stride, cudaStream_t stream);

template <typename scalar_t>
void launch_higgs_kv_bwd(
    const scalar_t* grad_y, const scalar_t* x,
    const uint8_t* indices, const uint8_t* ste_mask,
    const float* rot_norm_arr, const float* scale_arr,
    const __nv_bfloat16* rotated_save, const __nv_bfloat16* recon_unit_save,
    const float* codebook, const float* codebook_norm_sq,
    scalar_t* grad_x, int64_t num_rows, int64_t row_stride,
    cudaStream_t stream);

}  // namespace higgs
}  // namespace megatron

namespace {

#define HG_CHECK_CUDA(t) TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor")
#define HG_CHECK_CONTIG(t) \
  TORCH_CHECK((t).is_contiguous(), #t " must be contiguous")
#define HG_CHECK_DTYPE(t, dt) \
  TORCH_CHECK((t).scalar_type() == (dt), #t " has wrong dtype")

constexpr int kLatentDim = 512;
constexpr int kNumPairs = 256;
constexpr int kCodebookSize = 16;
constexpr int kPairDim = 2;

__nv_bfloat16* maybe_bf16_ptr(torch::Tensor t) {
  if (!t.defined() || t.numel() == 0) return nullptr;
  HG_CHECK_CUDA(t);
  HG_CHECK_DTYPE(t, torch::kBFloat16);
  return reinterpret_cast<__nv_bfloat16*>(t.data_ptr());
}

const __nv_bfloat16* maybe_bf16_const(torch::Tensor t) {
  if (!t.defined() || t.numel() == 0) return nullptr;
  HG_CHECK_CUDA(t);
  HG_CHECK_DTYPE(t, torch::kBFloat16);
  return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr());
}

void higgs_kv_fwd(
    torch::Tensor x, torch::Tensor out,
    torch::Tensor indices, torch::Tensor ste_mask,
    torch::Tensor rot_norm_out, torch::Tensor scale_out,
    torch::Tensor rotated_save, torch::Tensor recon_unit_save,
    torch::Tensor codebook, torch::Tensor codebook_norm_sq) {
  HG_CHECK_CUDA(x); HG_CHECK_CUDA(out);
  HG_CHECK_CUDA(indices); HG_CHECK_CUDA(ste_mask);
  HG_CHECK_CUDA(rot_norm_out); HG_CHECK_CUDA(scale_out);
  HG_CHECK_CUDA(codebook); HG_CHECK_CUDA(codebook_norm_sq);
  HG_CHECK_CONTIG(x); HG_CHECK_CONTIG(out);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == kLatentDim,
              "x must be [N, kLatentDim=512]");
  TORCH_CHECK(out.sizes() == x.sizes(), "out shape mismatch");
  TORCH_CHECK(indices.size(0) == x.size(0) && indices.size(1) == kNumPairs,
              "indices must be [N, kNumPairs=256]");
  TORCH_CHECK(ste_mask.size(0) == x.size(0) && ste_mask.size(1) == kLatentDim,
              "ste_mask must be [N, kLatentDim=512]");
  HG_CHECK_DTYPE(indices, torch::kUInt8);
  HG_CHECK_DTYPE(ste_mask, torch::kUInt8);
  HG_CHECK_DTYPE(rot_norm_out, torch::kFloat32);
  HG_CHECK_DTYPE(scale_out, torch::kFloat32);
  TORCH_CHECK(codebook.dim() == 2 && codebook.size(0) == kCodebookSize
              && codebook.size(1) == kPairDim,
              "codebook must be [16, 2]");
  TORCH_CHECK(codebook_norm_sq.dim() == 1
              && codebook_norm_sq.size(0) == kCodebookSize,
              "codebook_norm_sq must be [16]");
  HG_CHECK_DTYPE(codebook, torch::kFloat32);
  HG_CHECK_DTYPE(codebook_norm_sq, torch::kFloat32);

  __nv_bfloat16* rot_ptr = maybe_bf16_ptr(rotated_save);
  __nv_bfloat16* ru_ptr = maybe_bf16_ptr(recon_unit_save);

  const int64_t num_rows = x.size(0);
  const int64_t row_stride = x.size(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      x.scalar_type(), "higgs_kv_fwd",
      AT_DISPATCH_CASE(torch::kFloat32, [&] {
        megatron::higgs::launch_higgs_kv_fwd<float>(
            x.data_ptr<float>(), out.data_ptr<float>(),
            indices.data_ptr<uint8_t>(), ste_mask.data_ptr<uint8_t>(),
            rot_norm_out.data_ptr<float>(), scale_out.data_ptr<float>(),
            rot_ptr, ru_ptr,
            codebook.data_ptr<float>(), codebook_norm_sq.data_ptr<float>(),
            num_rows, row_stride, stream);
      })
      AT_DISPATCH_CASE(torch::kBFloat16, [&] {
        megatron::higgs::launch_higgs_kv_fwd<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            indices.data_ptr<uint8_t>(), ste_mask.data_ptr<uint8_t>(),
            rot_norm_out.data_ptr<float>(), scale_out.data_ptr<float>(),
            rot_ptr, ru_ptr,
            codebook.data_ptr<float>(), codebook_norm_sq.data_ptr<float>(),
            num_rows, row_stride, stream);
      })
      AT_DISPATCH_CASE(torch::kFloat16, [&] {
        megatron::higgs::launch_higgs_kv_fwd<__half>(
            reinterpret_cast<const __half*>(x.data_ptr()),
            reinterpret_cast<__half*>(out.data_ptr()),
            indices.data_ptr<uint8_t>(), ste_mask.data_ptr<uint8_t>(),
            rot_norm_out.data_ptr<float>(), scale_out.data_ptr<float>(),
            rot_ptr, ru_ptr,
            codebook.data_ptr<float>(), codebook_norm_sq.data_ptr<float>(),
            num_rows, row_stride, stream);
      }));
}

void higgs_kv_bwd(
    torch::Tensor grad_y, torch::Tensor x,
    torch::Tensor indices, torch::Tensor ste_mask,
    torch::Tensor rot_norm_arr, torch::Tensor scale_arr,
    torch::Tensor rotated_save, torch::Tensor recon_unit_save,
    torch::Tensor codebook, torch::Tensor codebook_norm_sq,
    torch::Tensor grad_x) {
  HG_CHECK_CUDA(grad_y); HG_CHECK_CUDA(x); HG_CHECK_CUDA(grad_x);
  HG_CHECK_CONTIG(grad_y); HG_CHECK_CONTIG(x); HG_CHECK_CONTIG(grad_x);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == kLatentDim,
              "x must be [N, 512]");
  TORCH_CHECK(grad_y.sizes() == x.sizes(), "grad_y shape mismatch");
  TORCH_CHECK(grad_x.sizes() == x.sizes(), "grad_x shape mismatch");
  HG_CHECK_DTYPE(indices, torch::kUInt8);
  HG_CHECK_DTYPE(ste_mask, torch::kUInt8);
  HG_CHECK_DTYPE(rot_norm_arr, torch::kFloat32);
  HG_CHECK_DTYPE(scale_arr, torch::kFloat32);
  HG_CHECK_DTYPE(codebook, torch::kFloat32);
  HG_CHECK_DTYPE(codebook_norm_sq, torch::kFloat32);

  const __nv_bfloat16* rot_ptr = maybe_bf16_const(rotated_save);
  const __nv_bfloat16* ru_ptr = maybe_bf16_const(recon_unit_save);

  const int64_t num_rows = x.size(0);
  const int64_t row_stride = x.size(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      x.scalar_type(), "higgs_kv_bwd",
      AT_DISPATCH_CASE(torch::kFloat32, [&] {
        megatron::higgs::launch_higgs_kv_bwd<float>(
            grad_y.data_ptr<float>(), x.data_ptr<float>(),
            indices.data_ptr<uint8_t>(), ste_mask.data_ptr<uint8_t>(),
            rot_norm_arr.data_ptr<float>(), scale_arr.data_ptr<float>(),
            rot_ptr, ru_ptr,
            codebook.data_ptr<float>(), codebook_norm_sq.data_ptr<float>(),
            grad_x.data_ptr<float>(), num_rows, row_stride, stream);
      })
      AT_DISPATCH_CASE(torch::kBFloat16, [&] {
        megatron::higgs::launch_higgs_kv_bwd<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(grad_y.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            indices.data_ptr<uint8_t>(), ste_mask.data_ptr<uint8_t>(),
            rot_norm_arr.data_ptr<float>(), scale_arr.data_ptr<float>(),
            rot_ptr, ru_ptr,
            codebook.data_ptr<float>(), codebook_norm_sq.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr()),
            num_rows, row_stride, stream);
      })
      AT_DISPATCH_CASE(torch::kFloat16, [&] {
        megatron::higgs::launch_higgs_kv_bwd<__half>(
            reinterpret_cast<const __half*>(grad_y.data_ptr()),
            reinterpret_cast<const __half*>(x.data_ptr()),
            indices.data_ptr<uint8_t>(), ste_mask.data_ptr<uint8_t>(),
            rot_norm_arr.data_ptr<float>(), scale_arr.data_ptr<float>(),
            rot_ptr, ru_ptr,
            codebook.data_ptr<float>(), codebook_norm_sq.data_ptr<float>(),
            reinterpret_cast<__half*>(grad_x.data_ptr()),
            num_rows, row_stride, stream);
      }));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("higgs_kv_fwd", &higgs_kv_fwd, "HIGGS dense 2-bit KV forward");
  m.def("higgs_kv_bwd", &higgs_kv_bwd, "HIGGS dense 2-bit KV backward");
}
