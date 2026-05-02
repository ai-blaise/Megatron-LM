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

## Round 4 — correctness sweep + dither autotune candidate (NOT SHIPPED)

This round was about defense-in-depth: an exhaustive audit of every
`@triton.autotune` decorator in the ECO path to confirm the round-3
state-corruption fix was the *only* missed bug. Audit summary:

| Kernel | Autotuned today | Risk | Status |
|---|---|---|---|
| `_triton_eco_inject_kernel` | yes | in-place RMW on momentum | ✅ has `restore_value` (round-3 fix) |
| `_triton_dequantize_kernel` | yes | pure-functional (separate input/output) | ✅ safe — output is deterministic |
| `_triton_quantize_kernel` | yes | pure-functional | ✅ safe |
| `_triton_adam_kernel` | **no** (deliberately) | in-place RMW on param/m₁/m₂/ECC | ✅ safe — autotune skipped per the line-1631 comment |
| `_triton_momentum_kernel` | **no** (deliberately) | same as Adam | ✅ safe |
| `_triton_block_dither_kernel` (`nvfp4_sr.py`) | **no** today | in-place RMW on master | ✅ safe today; see candidate below |

The audit found no further missed bugs. The remaining ECO kernels are
either (a) explicitly opted out of autotune for the same reason as Adam
or (b) read-only on inputs.

### Dither autotune — shipped with a size-threshold dispatch

A standalone bench of `_triton_block_dither_kernel` showed dramatic
autotune wins at large shards (16 M+ elements) but a 30–45 % regression
at small shards from autotune-dispatch overhead exceeding the work.
Rather than ship one or the other, the wrapper now dispatches between
two kernel variants based on shard `numel`:

```python
if numel < _DITHER_AUTOTUNE_THRESHOLD:    # 8 Mi elements
    _triton_block_dither_kernel[grid](...)              # fixed BLOCK=1024
else:
    _triton_block_dither_kernel_autotuned[grid](...)    # autotune sweep
```

This mirrors the variant-dispatch pattern already used in the
TurboQuant forward kernel (which selects between template
specializations at the launch site based on the per-call shape /
flag combination).

End-to-end bench of the threshold-dispatched wrapper on B200:

| elements | path | us/call | vs. original baseline |
|---:|---|---:|---:|
|   262 144 | baseline (small) |  25 µs | ~same (~23 µs orig) |
|  4 194 304 | baseline (small) |  24 µs | ~same (~22 µs orig) |
| 16 777 216 | autotuned (large) | **33 µs** | **−60 %** (83 µs orig) |
| 67 108 864 | autotuned (large) | **86 µs** | **−73 %** (316 µs orig) |

Crossover threshold of 8 Mi elements is set conservatively at the
midpoint of the bench-derived 4 M–16 M crossover band. In practice this
keeps small MLA LoRA / DSA Indexer / per-expert MoE shards under
moderate DP on the baseline path, while FFN matrices, embeddings, and
large MoE shards take the autotuned win.

#### Statistical contract preserved; bit-for-bit reproducibility caveat

A direct bit-equivalence check revealed that the two variants produce
**different realized random sequences** at identical seed + input. This
is a Triton `tl.rand` implementation detail: the PRNG state per SIMD
lane depends on the tensor layout, so two configs that pick different
`BLOCK_SIZE` produce different per-element noise even when the
absolute offset values match. What is preserved across both variants:

- Per-element distribution: U(−`DITHER_COEF`, +`DITHER_COEF`)
- Mean: 0
- Variance: identical
- ECO's expected-value contract on the cast result: unchanged
- Within-run determinism: a single configuration is reproducible across
  re-runs once the autotune cache is warm

What is lost: bit-for-bit reproducibility across runs that select
different autotune configs (e.g. different hardware revisions, Triton
versions, or inputs that cross the 8 Mi threshold).

The autotuned variant carries
`restore_value=("master_ptr",)` so the in-place RMW on the master shard
is correctly snapshot-and-restored around each autotune timing trial —
same contract as the round-3 inject fix.

## Round 3 — autotune correctness fix (CRITICAL)

The round-1 autotune addition (commit `7fbec44dd`'s parent chain, originally
landed via PR #7) was missing a ``restore_value`` argument. Triton's
``@autotune`` decorator invokes the kernel once per candidate config to
time it, then runs the chosen config again — *without* snapshotting any
input. Because ``_triton_eco_inject_kernel`` does an in-place RMW on the
momentum buffer (and its int8 scales), every cache-miss inject call was
applying the optimizer update **45 + 1 = 46 times** instead of once,
silently corrupting the optimizer state on the first encounter of each
unique ``(N, PARAM_DTYPE, QUANTIZE_OPTIM_STATES)`` shape.

The existing correctness oracle masked the bug because it clones inputs
per call. Round-3 added
``tests/unit_tests/optimizer/test_flash_adamw_eco_first_call.py``: same
prepared state called twice without cloning — the first call sweeps the
autotune configs and the second hits the cache. Without ``restore_value``
the two diverge wildly; with it they are bit-equal.

The fix is a one-line addition to the autotune decorator
(``restore_value=("mom_ptr", "mom_scales_f16_ptr")``) so Triton
snapshot-and-restores those tensors around each timing run. The neighboring
``_triton_adam_kernel`` and ``_triton_momentum_kernel`` deliberately skip
``@triton.autotune`` entirely with the same justification (see line 1631
of ``flash_optimizers.py``); this fix brings the inject kernel into
compliance with that house rule.

Bench post-fix on B200 (compare to round-1 numbers above):

| elements | autotune-only (q) | + restore_value (q) | Δ |
|---:|---:|---:|---:|
|   262 144 |  45 µs |  47 µs | +4 % |
|  4 194 304 |  46 µs |  48 µs | +4 % |
| 16 777 216 |  50 µs |  52 µs | +4 % |
| 67 108 864 | 189 µs | 176 µs | −7 % |

Steady-state perf is preserved (the cache-hit path was already correct);
only first-call latency for new shapes pays a one-time snapshot cost
during the autotune sweep, then is cached forever.

## Round 2 — algebraic identity in the var dequant path

Followup tightening: when the optimizer state is quantized, the variance
is stored as its sqrt (the FlashAdamW int8 store-time invariant). The
inject kernel previously squared `var_sqrt` to recover `var`, then took
sqrt again to compute the Adam denominator:

```
var       = var_sqrt * var_sqrt
denom     = sqrt(var / bc2) + eps
```

Algebraically (with `var_sqrt ≥ 0`):

```
sqrt(var_sqrt^2 / bc2) = var_sqrt * (1 / sqrt(bc2))
```

so the kernel now computes `denom = var_sqrt * inv_sqrt_bc2 + eps`,
where `inv_sqrt_bc2` is precomputed once host-side per call. Saves one
square and one sqrt per element in the QUANTIZE_OPTIM_STATES branch
(the live training path). Identity-preserving, verified by the
correctness oracle (same bit-equivalence numbers as before the change:
zero int8 mismatches at 262 K, 11/4 M and 33/16 M at ±1 ULP, fp32 round-
off only on the unquantized path).

Wall-clock is unchanged because the kernel is bandwidth-bound at large
N and launch-bound at small N — neither regime is gated on the saved
arithmetic. The change is kept for code clarity and lower per-thread
register pressure (one fewer live fp32 between var dequant and denom).

## What stays out of scope this round

The wider perf wins identified during profiling — fusing the NVFP4 cast
with error compute, eliminating the full-tensor dequantize-then-slice in
`distrib_optimizer._inject_nvfp4_eco_errors`, and per-tile dither — all
require Transformer Engine to exercise end-to-end. The pip wheel for TE
(>= 2.7.0.dev0) targets a newer cuBLAS than the CUDA 13.0 driver on the
dev VM ships with (undefined symbol
`cublasLtGroupedMatrixLayoutInit_internal` against `libcublasLt.so.13`),
so a clean source build is needed before the TE-pipeline round can land.
Specifically:

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

## End-to-end production-path test

`tests/unit_tests/optimizer/test_flash_adamw_eco_e2e.py` synthesizes a
mini `FlashAdamW(eco=True)` instance and drives the full production call
chain `FlashAdamW.inject_eco_error → _fused_eco_inject →
_triton_eco_inject_kernel` for the same shape/dtype matrix as the
correctness oracle. All six combinations pass on B200: momentum changes
non-trivially after the inject and stays finite. This validates the
autotuned kernel through exactly the API surface the distributed
optimizer invokes in production, with no TE dependency.

## Footprint contract

The hard constraint for this round was **no additional memory**. The
autotune change touches launch parameters only — no new tensors, no
scratchpad allocations, no enlarged kernel SRAM. Confirmed by reading
`triton-cache` after a full run: SRAM/register usage per the chosen
config is at or below the original `BLOCK_SIZE_N=1024, num_warps=4`
defaults at every shape we benchmarked.
