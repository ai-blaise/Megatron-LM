#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void g1_gate_fwd(
    const void* linear_out,
    const void* attn_out,
    void* output,
    void* gate,
    int64_t n,
    cudaStream_t stream);

extern "C" void g1_gate_bwd(
    const void* d_out,
    const void* out_ungated,
    const void* gate,
    void* d_out_ungated,
    void* d_gate_linear,
    int64_t n,
    cudaStream_t stream);

void g1_gate_fwd_torch(
    torch::Tensor linear_out,
    torch::Tensor attn_out,
    torch::Tensor output,
    torch::Tensor gate,
    int64_t n) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  g1_gate_fwd(
      linear_out.data_ptr(),
      attn_out.data_ptr(),
      output.data_ptr(),
      gate.data_ptr(),
      n,
      stream);
}

void g1_gate_bwd_torch(
    torch::Tensor d_out,
    torch::Tensor out_ungated,
    torch::Tensor gate,
    torch::Tensor d_out_ungated,
    torch::Tensor d_gate_linear,
    int64_t n) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  g1_gate_bwd(
      d_out.data_ptr(),
      out_ungated.data_ptr(),
      gate.data_ptr(),
      d_out_ungated.data_ptr(),
      d_gate_linear.data_ptr(),
      n,
      stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("g1_gate_fwd", &g1_gate_fwd_torch, "Fused G1 sigmoid gate forward (BF16)");
  m.def("g1_gate_bwd", &g1_gate_bwd_torch, "Fused G1 sigmoid gate backward (BF16)");
}
