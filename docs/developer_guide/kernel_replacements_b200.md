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
| HISA / DSA sparse attention | `megatron/core/extensions/hisa_indexer`, `megatron/core/transformer/experimental_attention_variant/dsa_triton.py` | see DSA tests | HISA has a tensor-core-shaped candidate scorer; sparse DSA attention backward is currently the larger bottleneck |

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

### HISA / DSA follow-up

HISA selection and sparse DSA attention should not be lumped together as a
single "CUDA path":

* HISA candidate scoring has a real matrix shape: per query row it computes
  `[64 indexer heads, 128 dim] x [128 dim, N candidates]`, then applies
  ReLU, per-head weights, and top-k. This is the plausible CuTe/UMMA target.
  The current production default is the BMM backend because it maps that work
  to large cuBLAS calls and is still faster than the experimental per-row
  packed-NVFP4 cuBLASDx extension.
* Sparse DSA attention is different. Per attention head it is effectively one
  query vector against a different selected key set for every row, followed by
  softmax, value accumulation, and scattered K/V gradients. That shape is not a
  clean dense GEMM. The profile points to the backward atomics and replay count,
  not HISA selection alone.
* Do not use `cute`, `cutlass`, or `mxf8f6f4` as aliases for the current HISA
  selector backend. Those names should be reserved for a true SM100
  blockscaled/UMMA implementation. The existing experimental options are
  explicit: `packed_cublasdx` and `packed_cublasdx_fp8`.
* The production BMM HISA selector now computes causal partial-block means with
  block-local prefix sums instead of rebuilding a full sequence cumsum for every
  query chunk. On the same B200 selector microbench this moved BMM from ~9.5 ms
  to ~2.9 ms at `Q=64`, and from ~15.2 ms to ~8.3 ms at `Q=512`.
* The optimized HISA selector and deferred HISA/indexer-loss path now keep
  selected-token outputs in `int32` instead of widening to `int64`. PyTorch
  `topk` fallback buffers still use `long` where required by the API, but the
  production HISA-to-DSA path feeds `int32` selected indices into the compact
  `int16` DSA top-k buffer and fused DSA kernels.
* The SFT launcher now defaults to `MEGATRON_HISA_SELECTOR_ROW_CHUNK=512` and
  `MEGATRON_HISA_CANDIDATE_SLOT_GROUP=16`. On a B200 HISA selector microbench
  at `Q=512,H=64,D=128,L=32768,topk=1024,block=128,compression=4.0`, this
  reduced exact-FP32 BMM selection from ~7.90 ms (`256/8`) to ~6.63 ms
  (`512/16`) with identical selected indices and scores. Enabling TF32 reduced
  the same case to ~5.48 ms but changed about 1.2% of selected slots, so TF32
  remains disabled for this selector path.
* The `packed_cublasdx` selector now uses cuBLASDx tensor-core tiles for both
  block-rep scoring and selected-token scoring, then performs deterministic
  CTA-local bitonic top-k over the candidate pool. On a B200 selector
  microbench (`Q=64,H=64,D=128,L=32768,topk=1024,block=128,compression=4.0`)
  this reduced the packed cuBLASDx path from ~74.3 ms to ~23.0 ms and removed
  the previous top-k runtime slope. The dense BMM backend is still faster at
  ~2.9 ms after the block-prefix fix, so the packed path remains experimental
  rather than the launcher default. Scaling also favors BMM: at `Q=512`, BMM
  measured ~8.3 ms while the per-row packed cuBLASDx backend measured ~90.4 ms.
  A production packed path would need row/tile-parallel CUTLASS/CuTe scheduling,
  not one CTA walking all candidate tiles for a row.
* A current-shape recheck at
  `Q=512,H=64,D=128,L=32768,topk=1024,block=128,compression=4.0` measured
  `bmm` at ~6.98 ms, scalar packed CUDA at ~372.8 ms, and
  `packed_cublasdx` at ~89.2 ms. The packed paths matched the BMM selected set
  for >99.9% of slots but still lose badly on wall time. That is why the
  launcher stays on exact BMM while we reserve CuTe/UMMA work for a true
  multi-row/tile scheduler.
* `packed_cublasdx_tiled` is the first row/tile-scheduled packed-NVFP4
  selector prototype. It splits the packed-token refine phase into one
  cuBLASDx CTA per query/candidate tile, writes a compact candidate-score
  scratch, then runs a deterministic device top-k reduction. On the same
  `Q=512,H=64,D=128,L=32768,topk=1024,block=128,compression=4.0` B200 check it
  matched the BMM selected sets exactly and moved packed cuBLASDx from
  ~89.2 ms to ~46.0 ms, at ~0.11 GiB peak selector allocation. This is a real
  scheduler improvement, but still not production-default because exact BMM is
  ~8.1 ms on the same shape.
* `MEGATRON_HISA_BMM_FP32_ACCUM_TENSORCORES=1` is an experimental BMM variant
  that keeps BF16/FP16 inputs and requests FP32 output accumulation from cuBLAS.
  It is not a launcher default. At the same `Q=512,H=64,D=128,L=32768,topk=1024`
  shape, it reduced peak selector allocation from ~1.42 GiB to ~1.13 GiB, but
  measured slower (~7.62 ms versus ~6.84 ms) and changed ~1.2% of selected
  slots relative to exact FP32-input BMM. That is a useful diagnostic but not an
  acceptable default for the deterministic HISA selector.

Latest DSA backward tile sweep on one B200 at `q_len=2048`, `kv_len=32768`,
`heads=32`, `qk_dim=192`, `v_dim=128`, `topk=1024`, and BF16 K/V grad atomics:

| `MEGATRON_DSA_TRITON_BLOCK_K_BWD` | `MEGATRON_DSA_TRITON_BWD_NUM_WARPS` | Backward ms | Peak GiB |
|---:|---:|---:|---:|
| 8 | 1 | 18.96 | 2.65 |
| 16 | 2 | 21.48 | 2.65 |
| 32 | 2 | 17.79 | 2.65 |
| 64 | 2 | 22.37 | 2.65 |
| 128 | 4 | 24.55 | 2.65 |

The SFT launcher therefore defaults to `BLOCK_K_BWD=32`, `BWD_NUM_WARPS=2`.
`MEGATRON_DSA_TRITON_BLOCK_Q` was also checked as a scheduler knob on
`q_len=512,bsz=4,heads=8,qk_dim=192,v_dim=128,kv_len=32768,topk=1024`.
`BLOCK_Q=1` remained best (`~11.8 ms` fwd+bwd) versus `BLOCK_Q=2`
(`~23.2 ms`) and `BLOCK_Q=4` (`~51.5 ms`). The reason is structural: selected
K/V sets differ per query row, so increasing `BLOCK_Q` expands gather-shaped
work without giving Triton a dense MMA tile to reuse. A true DSA win needs
selected-key grouping/inverted scheduling, not just larger query tiles.
The launcher keeps compact int16 top-k enabled but leaves
`MEGATRON_DSA_SORT_TOPK_INDICES=0` by default. Sorting is only needed by the
experimental key-block K/V backward binary-search path, which is not a
production default. The production sparse-attention/teacher path consumes
indices, selected indexer scores, and teacher probabilities in the same order,
so selected-token ordering is permutation-invariant. On B200, sorting
`[4,32768,1024]` top-k indices measured ~3.2 ms versus ~0.2 ms for the compact
int16 copy alone.
It also defaults to `MEGATRON_DSA_TEACHER_SCORE_SCRATCH=1`, which stores
selected attention logits during the first sparse-attention QK pass and reuses
them for indexer-loss teacher emission after LSE is known. This avoids the
previous selected-QK recompute in the teacher path. On the same B200 shape with
`topk=1024`, teacher forward measured ~10.38 ms versus ~17.37 ms for the
recompute path, with identical outputs and about +0.27 GiB transient memory at
`q_len=2048`.

### HISA selected-score backward

The trainable HISA indexer-loss backward is separate from sparse DSA attention
backward:

```
selected_score[row, slot] =
    sum_h weight[row, h] * relu(q[row, h] dot k[topk[row, slot]])
```

The previous launcher default used `MEGATRON_HISA_SELECTED_SCORE_BWD_HEAD_GROUP=8`.
That reduced global `grad_k` atomics by summing eight indexer heads before each
write, but the implementation still computed those eight head dots serially
with block-wide reductions.

`MEGATRON_HISA_SELECTED_SCORE_BWD_WARP_GROUPED=1` is now the SFT launcher
default. It maps one warp to each head inside the head group, broadcasts the
lane-0 dot result to the warp before the ReLU branch, and then sums the eight
per-head `grad_k` contributions in shared memory before one global atomic per
selected token/dim. This preserves the selected-logit semantics and the
head-grouped atomic reduction, while removing most of the per-head block syncs.

B200 selected-score backward microbench (`H=64,D=128,L=32768,K=1024`, fp32
inputs, `MEGATRON_HISA_SELECTED_SCORE_BWD_HEAD_GROUP=8`):

| Rows Q | old head-grouped | warp-grouped | speedup |
|---:|---:|---:|---:|
| 512 | 12.09 ms | 5.06 ms | 2.39x |
| 1024 | 22.48 ms | 9.85 ms | 2.28x |
| 4096 | 84.01 ms | 38.74 ms | 2.17x |

For MBS>1, the selected-score autograd path now flattens the microbatch and
concatenates per-batch indexer K rows before calling the same warp-grouped CUDA
backward once. This preserves exactly the same selected-token math as one call
per batch item by adding a per-batch K-row offset to top-k indices. On B200 at
`B=4,H=64,D=128,L=32768,K=1024`, this measured:

| Rows per batch | old per-batch launches | batched launch | speedup |
|---:|---:|---:|---:|
| 512 | 20.45 ms | 19.54 ms | 1.05x |
| 2048 | 77.40 ms | 76.46 ms | 1.01x |

The improvement is modest because the kernel is dominated by selected-token
work and K atomics rather than launch overhead, but it removes repeated
extension calls from the live MBS path without changing semantics.

An experimental cuBLASDx selected-score backward (`64 x {32,64,128} x 128`
tiles) is correctness-covered but is not enabled by default. The 64-slot tile
measured ~100.85 ms at `Q=4096`, slower than both old head-grouped and
warp-grouped. Its dot phase is tensor-core shaped, but the extra tile-level
`grad_q` atomics and shared-memory footprint dominate at the production row
counts.

A q-tile unique K/V backward reducer for sparse DSA attention was also tested
behind `MEGATRON_DSA_TRITON_QTILE_UNIQUE_KV_BWD=1`. It is mathematically
correct, but not production-worthy: at a production-ish BF16 shape
(`q=256,bsz=2,heads=4,kv=32768,topk=1024`) it measured ~44 ms versus ~2.3 ms
for the default row-major fused path. It remains off by default.

`MEGATRON_DSA_TRITON_GROUPED_KV_BWD=1` is a second, narrower sparse-attention
K/V grouping experiment that keeps K/V reduction inside the main row-major
backward launch instead of adding a separate q-tile reducer launch. It groups
duplicate selected keys inside each backward tile before global K/V atomics.
This preserves sparse-attention semantics and passes the sparse-reference
gradient test, but is still slower than the current simple atomic path. On a
B200 BF16/topk=1024 shape (`q=256,bsz=2,heads=4,qk_dim=192,v_dim=128,kv=32768`)
with intentionally correlated top-k rows, the current default measured
~2.22 ms forward+backward. Grouped variants measured ~5.69 ms (`BQ=2,BK=8`),
~10.43 ms (`BQ=2,BK=32`), ~9.59 ms (`BQ=4,BK=16`), and some larger tiles
exceeded SM100 shared-memory limits. This rules out local same-key matrix
grouping as a production default; a useful sparse-attention backward win still
needs a lower-level inverted-edge/block-sparse scheduler rather than more work
inside the existing row-major replay.

`MEGATRON_DSA_CUDA_KV_BWD=1` is the first lower-level CUDA sparse K/V backward
reducer. It leaves the existing fused selected-attention forward and Triton
grad-query replay in place, but moves selected K/V gradient accumulation into a
CUDA edge-tile scheduler. The CUDA path accepts compact `int16` top-k indices
directly, supports `int32`/`int64` top-k for validation, and groups duplicate
selected keys within each CTA before global K/V atomics. It is correctness
covered but intentionally not a launcher default yet: on a B200 BF16 shape
(`q=512,bsz=2,heads=4,qk_dim=192,v_dim=128,kv=32768,topk=1024`) the current
Triton fused row replay measured ~1.97 ms forward+backward, while CUDA K/V
tiles measured ~13.94 ms (`2x4`), ~13.87 ms (`1x8`), ~13.86 ms (`4x2`), and
~10.53 ms (`1x4`/`2x2`). The main issue is structural: the opt-in CUDA path
recomputes selected QK/probability work for K/V while Triton still computes
grad-query. A production replacement needs one lower-level backward scheduler
that owns grad-query and K/V together, or a forward-saved selected-logit path
with an explicit memory budget.

The hot Triton selected-index loads now keep compact top-k values in 32-bit
integer arithmetic instead of widening to `tl.int64`. That preserves support
for validation inputs with `torch.int64`, but avoids paying 64-bit index math in
the production `int16`/`int32` selected-token path.

When `MEGATRON_DSA_TEACHER_SCORE_SCRATCH=1`, the selected-attention forward
stores the fp32 selected logits it already computed for teacher emission. The
Triton backward now reuses that scratch when
`MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH=1` instead of recomputing selected QK
just to recover softmax probabilities. This keeps the existing row-major
backward scheduler and exact gradients, but removes duplicate dot products in
the trainable HISA/indexer-loss path. On a focused B200 BF16 check
(`q=512,bsz=2,heads=4,qk_dim=192,v_dim=128,kv=32768,topk=1024`), fwd+bwd
measured ~3.24 ms with backward recompute and ~2.74 ms with score-scratch
reuse, with the same ~0.82 GiB peak allocation in the benchmark.

`MEGATRON_DSA_CUDA_BWD_FROM_SCORES=1` is a lower-level CUDA attempt at owning
the full sparse backward from the same forward-saved scores. It is correctness
covered, but it is not a launcher default: the current scalar edge-warp
scheduler measured ~10.95-15.56 ms on the smaller B200 teacher shape where the
Triton score-scratch backward measured ~2.74 ms. That confirms the right
direction is score reuse plus a genuinely better block-sparse scheduler, not a
one-warp-per-edge CUDA rewrite.

`MEGATRON_DSA_CUDA_ROW_BWD_FROM_SCORES=1` is the next CUDA scheduler probe. It
maps one CTA to a whole `(query row, batch, head)`, caches the query and
grad-output row in shared memory, computes `dO dot O` once per row, streams over
forward-saved selected logits, and emits exact `dQ/dK/dV`. This is a real
dataflow change from the edge-tile CUDA prototype and is covered by the sparse
reference test, but it is still not a launcher default. On a B200 BF16 shape
(`Q=512,B=2,H=4,D=192,V=128,S=32768,K=1024`, score scratch), the current Triton
score-scratch path measured ~2.19 ms, while row-owned CUDA measured ~12.65 ms
with 4 warps, ~11.43 ms with 8 warps, and ~11.50 ms with 16 warps. The row CTA
removes redundant row loads and `delta` work, but serializes too much top-k work
relative to the existing Triton topk-vectorized replay.

`MEGATRON_DSA_CUDA_SORTED_KV_BWD=1` is an inverted-edge reducer that finally
removes global K/V atomics for exact selected keys. The main Triton backward is
run with K/V emission disabled so it only computes `dQ`; the CUDA reducer builds
selected-edge keys `(batch, head, selected_key)`, sorts them on device, and
reduces every exact key segment into one `dK/dV` write. This is correctness
covered, but it is also not a launcher default. On B200 BF16 with score scratch,
`Q=128,B=2,H=4,K=512` measured ~0.63 ms for the Triton score-scratch path and
~2.03 ms for sorted-segment K/V. `Q=256,B=2,H=4,K=1024` measured ~1.38 ms versus
~5.29 ms. Even with artificial top-k overlap from 25% to 100%, sorted-segment
stayed around ~5.27-5.88 ms while Triton stayed around ~1.27-1.36 ms. The sort
and segment pass dominate more than the current BF16 row-major atomics.

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
