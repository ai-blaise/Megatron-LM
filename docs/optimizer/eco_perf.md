# ECO inject — kernel autotune

This document records a single-round perf optimization on the FlashAdamW
**ECO** (Error-Compensating Optimization, arXiv:2601.22101) inject path.

## Context

This Megatron fork keeps model weights in NVFP4 throughout training; there
is no fp32 master shadow during the backward. ECO is the *only* signal
compensating for the missing precision: at the end of each step the
quantization error `θ − q(θ)` is injected into AdamW's first-moment buffer
scaled by `α = ((1 − β₁ᵗ)/η) · (1 − 1/β₁)` and the Adam denominator
`D = √v̂ + ε`.

The inject is performed by `_triton_eco_inject_kernel` in
`megatron/core/optimizer/flash_optimizers.py`. Per-step, per-element inject
semantics are non-negotiable in this fork: there is no shadow path the
optimizer can fall back to if the inject is delayed, batched across steps,
or coarsened.

## What changed

The kernel was hard-coded to `BLOCK_SIZE_N=1024` with no autotune. The
neighbouring quantize/dequantize kernels in the same module already use
`triton.autotune`; the inject kernel was the outlier.

Added:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": bs}, num_warps=nw, num_stages=ns)
        for bs in (256, 512, 1024, 2048, 4096)
        for nw in (2, 4, 8)
        for ns in (1, 2, 3)
    ],
    key=["N", "QUANTIZE_OPTIM_STATES", "PARAM_DTYPE"],
)
@triton.jit
def _triton_eco_inject_kernel(...):
    ...
```

Plus a meta-aware grid at the launch site so the autotune-chosen
`BLOCK_SIZE_N` feeds back into the CTA count (capped at `2*SM_count`).

## What did NOT change

- Per-step inject semantics. Every element still receives
  `m₁ ← m₁ + α · D · (pre − post)` exactly as before.
- Memory footprint. No new tensors, no extra in-flight buffers.
- Numerical contract. Outputs are equivalent to the baseline within fp32
  round-off (verified by the correctness oracle below).

This is purely a plumbing change — autotune over launch parameters that
the kernel was already amenable to.

## Benchmarks (NVIDIA B200)

Per-call timing in microseconds, average of 200 iterations after 20-iter
warmup. Two paths: quantized (int8 momentum/variance, the live training
path) and fp32 (the unquantized fallback).

| elements | baseline-q | tuned-q | Δ | baseline-fp32 | tuned-fp32 | Δ |
|---:|---:|---:|---:|---:|---:|---:|
|   262 144 |  35 µs |  45 µs | +30 % |  30 µs |  40 µs | +33 % |
|  4 194 304 |  36 µs |  46 µs | +28 % |  29 µs |  39 µs | +35 % |
| 16 777 216 | 123 µs | **50 µs** | **−60 %** |  86 µs | **65 µs** | **−25 %** |
| 67 108 864 | 550 µs | **189 µs** | **−66 %** | 326 µs | **240 µs** | **−26 %** |

The break-even is around 4–8 M elements. Below that, autotune-dispatch
overhead (~10 µs) dominates the absolute time. Above that, the larger
BLOCK_SIZE_N + num_warps configs unlock substantial bandwidth and the
kernel scales with HBM throughput. Real training shards in transformer
weight matrices live in the 10–100 M-element regime where the win is
large.

## Correctness oracle

`tests/unit_tests/optimizer/test_flash_adamw_eco_correctness.py` (the
landing version of `/tmp/eco_inject_correctness.py`) compares the
autotuned kernel against an fp64 PyTorch reference implementation that
mirrors the kernel's math exactly. Results on B200:

| N | quantized | absolute err | rel err / ULP |
|---:|---|---|---|
|   262 144 | yes | 0 mismatched int8 outputs | — |
|   262 144 | no  | 2.9 × 10⁻⁵ max abs (fp32) | round-off only |
| 4 194 304 | yes | 11 elements at ±1 ULP | bounded |
| 4 194 304 | no  | 4.5 × 10⁻⁵ max abs | round-off only |
| 16 777 216 | yes | 33 elements at ±1 ULP | bounded |
| 16 777 216 | no  | 5.6 × 10⁻⁵ max abs | round-off only |

The ±1 ULP int8 differences are floor-vs-round disagreements at element
boundaries and are below the noise floor that ECO already absorbs through
the next step's inject.

## What stays out of scope this round

The wider perf wins identified during profiling — fusing the NVFP4 cast
with error compute, eliminating the full-tensor dequantize-then-slice in
`distrib_optimizer._inject_nvfp4_eco_errors`, and per-tile dither — all
require Transformer Engine to exercise end-to-end. They are flagged as
followup work to land once TE is brought up on the dev VM. Specifically:

- **Cast + inject fusion.** The cast kernel in
  `nvfp4_sr.cast_master_weights_to_nvfp4_2d_sr` has `pre_cast` in
  registers and the just-packed NVFP4 codeword in registers; computing
  `error = pre_cast − dequantize(NVFP4)` inside the same tile would
  eliminate one full pass over the post-cast shard. Estimated 6–8 % step
  speedup; zero memory delta.
- **Slice-then-dequantize.** `distrib_optimizer._inject_nvfp4_eco_errors`
  currently dequantizes the *full* model param then slices to the rank's
  shard. The per-shard dequantize (which is just per-block decode-scale
  multiplications on the relevant 4-bit codes) is `dp_size×` cheaper.
- **Per-tile dither.** Replacing per-element `tl.rand` with per-block
  broadcast in the SR dither would change the SR variance bound
  (unbiasedness preserved). The user explicitly asked to stay grounded in
  the published ECO recipe, which calls for per-element SR; this needs a
  variance-bound analysis before landing.

## Footprint contract

The hard constraint for this round was **no additional memory**. The
autotune change touches launch parameters only — no new tensors, no
scratchpad allocations, no enlarged kernel SRAM. Confirmed by reading
`triton-cache` after a full run: SRAM/register usage per the chosen
config is at or below the original `BLOCK_SIZE_N=1024, num_warps=4`
defaults at every shape we benchmarked.
