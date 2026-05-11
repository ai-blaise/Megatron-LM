# SM100 (B200) kernel replacements

Adopts the optimized CuTe / launch-config kernel variants from
`ai-blaise/optimization-playground` commit
[`9c124721c`](https://github.com/ai-blaise/optimization-playground/commit/9c124721cb583a46b767dc5d92b3e3e9bcb6d80b)
into Megatron-LM's `flashoptim-kernels` branch. Targets NVIDIA Blackwell B200
(SM100, 148 SMs, 228 KB SMEM/SM, 8 TB/s HBM3e).

## Build prerequisites

Mandatory CUDA compile flags for every kernel below:

```
-std=c++20 -O3 --expt-relaxed-constexpr
-gencode=arch=compute_100,code=sm_100
-DFLASHINFER_ENABLE_BF16
-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
```

Footguns:

* Use `sm_100`, not `sm_100a`. CuTe barrier code crashes with `sm_100a`.
* `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK` is required because `nvcc` ships
  with CCCL 13.2 headers but PyTorch 2.13.0+cu130 ships with 13.0 runtime
  headers; the check rejects the mismatch otherwise.
* If `nvcc` cannot find `cusparse.h`, prepend
  `CPATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/include` —
  PyTorch routes CUDA includes to its bundled `nvidia/cu13/include` path.

## Kernel replacement matrix

| Kernel | Files touched | Bit-exactness | Speedup |
|---|---|---|---|
| G1 Attention Gate | `megatron/core/fusions/fused_g1_gate.{cu,py}` | 0.0 vs Megatron baseline | 1.22×–1.43× at small N vs paper baseline (1.08×–1.19× vs Megatron's existing fast-math baseline) |
| GatedNorm | `megatron/core/fusions/gated_norm.py`, new `gated_norm_cute.cu`, new `gated_norm_cute_wrapper.cpp` | max abs 9.77e-4 (≤1e-3) vs Triton | 1.48×–1.81× vs `torch.mm` (paper 1.05×–1.49×) |
| TurboQuant Dense KV | (no change) | n/a | n/a — see notes |
| IndexCache NSA fused-store | `megatron/core/quantization/indexcache/kernels/csrc/indexcache.cuh`, `indexcache_fwd.cu`, `indexcache_bwd.cu` | 0.0 across all 6 outputs | flat (~5.95 µs) — see notes |

### G1 Attention Gate (`fused_g1_gate.cu`)

`output = attn_out * sigmoid(linear_out)`. Forward kernel ported from
`g1_attention_cute.cuh`:

* Inline-PTX `ex2.approx.ftz.f32` + `rcp.approx.ftz.f32` for fast sigmoid
  (eliminates denormal-handling branches and the Newton–Raphson refine).
* N-adaptive launch geometry: `BLOCK=128 GRIDX=8` for `n*hidden_size <=
  1.5M`, else `BLOCK=256 GRIDX=4`.
* BF16x8 vectorized `__ldg` / `float4`-cast load and store.

The backward kernel is unchanged — the optimization is forward-only and the
analytic backward is unaffected.

The gain over the *paper* baseline (no `--use_fast_math`) reproduces the
upstream 1.23×–1.34× claim. Against Megatron's existing `--use_fast_math`
baseline, much of the arithmetic delta washes at the SASS level (`__expf`
already lowers to `MUFU.EX2`); the residual win at small N comes from the
new launch geometry.

### GatedNorm (`gated_norm_cute.cu`, `gated_norm.py`)

`output = normed * sigmoid(silu(normed @ w_down.T) @ w_up.T)`. New CuTe
SM100 forward replaces the previous Triton fast path:

* Two-pass tensor-core kernel using `mma.sync.aligned.m16n8k16` with
  `cp.async` double-buffer pipeline (prefetch issued *after* the mma into
  the released stage).
* SMEM rows padded by 8 bf16 to break the power-of-2 stride that would
  otherwise cause bank conflicts on `ldmatrix` loads.
* Pass-2 N-axis warp partitioning at low N (`NUM_N_WARPS=4`).
* `R=64 N>=16` returns `cudaErrorInvalidValue` so the launcher falls back
  to `torch.mm` (matches the SMEM-overflow contract of the reference).

The Triton fast path is retained behind `MEGATRON_GATED_NORM_USE_TRITON=1`
and continues to drive the backward pass. Set
`MEGATRON_GATED_NORM_DISABLE_CUTE=1` to bypass the CuTe path entirely.

When the CuTe forward runs in a graph that needs backward, it additionally
recomputes `z = normed @ w_down.T` via `torch.mm` so the saved tensor
matches what the Triton backward expects.

### TurboQuant Dense KV (no change)

Upstream's optimization hoists a 128-byte rope-copy from end-of-kernel to
start-of-kernel so its `LDG`/`STG` transactions overlap with the inverse
FWHT. Megatron's `turboquant_kv_fwd_kernel` operates only on the 512-dim
MLA latent; RoPE is a separate tensor (`k_pos_emb`) processed independently
downstream by `apply_rotary_pos_emb` in `multi_latent_attention.py`. There
is no rope-copy in this kernel to pipeline, so the optimization does not
apply. The kernel is left unchanged.

The broader CuTe rewrite in upstream's diff (cute::Layout descriptors,
swizzled SMEM, register tiling, `cp.async` preloading) is structural and
not separately measured against the rope-copy hoist in upstream's perf
table; per Rule 1 (light + minimalist + upstream-compatible) it is left
out of scope.

### IndexCache NSA fused-store

Upstream's optimization removes `__launch_bounds__(32,1)`, switches to
4-warp blocks, and elides a redundant bounds check. Megatron's port was
already in this configuration:

* No `__launch_bounds__` was ever applied.
* Launch already uses `dim3 block(kHeadDim)` with `kHeadDim=128` (4 warps).
* The Megatron port is a fake-quant on a contiguous `[N, 128]` tensor, not
  a paged-cache scatter; there is no equivalent `PagedCacheLayout` helper
  or per-page bounds check.

Surgical changes applied (bit-exact, dead-store elimination):

* Right-sized `block_reduce_max_128` and `block_reduce_sum_128` SMEM
  scratch from `float[kHeadDim]` (128) to `float[9]` (slots 0–3 partials,
  slot 8 broadcast).
* Removed a trailing `__syncthreads()` from each block-reduce helper —
  the helpers are called once per kernel before any SMEM reuse, so the
  second sync was dead.

The kernel runs at ~5.95 µs at production dims (`nt=32 hd=128
page_size=64`) regardless of `n` in `[32, 2048]` — the launch-overhead-
dominated regime that upstream identified (achieved occupancy 6.35 %, DRAM
0.04 %). The 1.36× upstream speedup came from removing a
`__launch_bounds__(32,1)` annotation that was actively constraining the
compiler; that constraint was never present here, so the optimization
opportunity was already captured.

## Verification

All measurements on a single B200 (SM100, 183 GB HBM3e, driver 580.126.09)
with PyTorch 2.13.0.dev+cu130, CUDA 13.2, fixed seed, median of 500 iters.

Forward bit-exactness (latest re-run from a clean clone of `flashoptim-kernels`):

```
K1 G1 Gate     N=1   ..512  max abs diff 0.0   (vs Megatron pre-port baseline)
K1 G1 Gate     backward N=64  d_lin_max=7.6e-6   d_att_max=1.2e-4
K2 GatedNorm   R=16, N=1..256  max abs diff 9.77e-4 (≤1e-3 budget)
K2 GatedNorm   R=64 N=32  cudaErrorInvalidValue → torch.mm fallback (finite output)
K4 IndexCache  N=32..2048  max abs diff 0.0 across out/scale/q_fp8/clip_mask/argmax/eps_active
```

Existing Megatron tests pass on B200: 12 IndexCache unit tests + 4
`nvfp4_act_eco_compose` tests.

## Runtime knobs

| Env var | Effect | Default |
|---|---|---|
| `MEGATRON_GATED_NORM_USE_TRITON=1` | Force the Triton fast path (also drives the backward pass). | unset (CuTe default) |
| `MEGATRON_GATED_NORM_DISABLE_CUTE=1` | Bypass the CuTe fast path entirely (use `torch.mm` or Triton). | unset |
| `MEGATRON_GATED_NORM_TORCH_MM_MIN_TOKENS=N` | Threshold above which the dispatcher prefers `torch.mm` (cuBLAS) over CuTe. | rank-dependent (R≥64: 256, R≥32: 512, R≥8: 2048, R≥1: 4096) |
| `MEGATRON_GATED_NORM_TORCH_MM_R{1,8,32,64}_MIN_TOKENS=N` | Per-rank-bucket override of the threshold above. | unset |
| `G1_BLOCK128_N_THRESHOLD` | Compile-time C++ macro: switch from BLOCK=128 to BLOCK=256 above this many output elements. | 1500000 |

The IndexCache kernel exposes no runtime knobs — the launch configuration
is hard-coded to `dim3(num_rows) x dim3(kHeadDim)` (4 warps per block).

## Reproduce on a B200

```sh
# 1. CUDA toolchain on PATH (CUDA 13 → nvcc 13.2 supports sm_100)
export PATH=/usr/local/cuda-13/bin:$PATH
# 2. Pick up cusparse.h from PyTorch's bundled include path
export CPATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/include

# 3. Fresh checkout
git clone --branch flashoptim-kernels \
    https://github.com/ai-blaise/Megatron-LM.git Megatron-LM
cd Megatron-LM

# 4. Wipe any stale JIT cache
rm -rf "$HOME/.cache/torch_extensions/py310_cu130"/{g1_gate_cuda,megatron_gated_norm_cute,megatron_indexcache_kv}

# 5. Trigger JIT build for all 3 kernels
PYTHONPATH=$PWD python3 -c "
from megatron.core.fusions.fused_g1_gate import _load_kernel
from megatron.core.fusions.gated_norm import _load_cuda_kernel
from megatron.core.quantization.indexcache import autograd  # builds on import
_load_kernel(); _load_cuda_kernel()
print('all kernels built')
"
```

## Reference

* Upstream commit: `ai-blaise/optimization-playground@9c124721c`
* Upstream docs: `optimization-playground/docs/developer_guide/cute_kernels_b200.md`
