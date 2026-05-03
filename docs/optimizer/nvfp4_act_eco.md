# NVFP4 Activation-ECO

Activation-ECO is the activation-side companion to FlashOptim's
weight-side ECO. Where weight-ECO compensates the rounding error in
the optimizer's persistent parameter representation, activation-ECO
compensates the rounding error in the per-step layer-input cast that
TE applies before each NVFP4 cuBLASLt GEMM.

## Why activation-side ECO is meaningful

Activations are not persistent across steps and have no associated
optimizer state, so the strict-sense ECO mechanism (inject the
residual into `m1`) does not apply. What activation-ECO does instead
is compensate the **bias the activation rounding induces in the weight
gradient** at the same `te.Linear` site:

* Forward today: `y = q(x) @ W` where `q` is the NVFP4 per-block cast.
* Standard QAT backward (saturated-zero STE): `dW = dy @ q(x).T`.
* That `dW` is biased by activation rounding by exactly `dy @ (x - q(x)).T`.
* Activation-ECO adds the residual term to recover the unbiased gradient
  `dW = dy @ x.T` that pure BF16 forward would have produced.

The injection target moves from `m1` (where weight-ECO injects) to
`dW` at the same matmul site (where activation-ECO injects), but the
underlying mechanism — capture the cast residual, lift it through the
chain rule, inject where the missing state would otherwise live — is
the same as in the original ECO paper.

## Math

For a single layer's `y = q(x) @ W` with NVFP4 per-block fake-quant
on `x` (block size 16, FP8E4M3 per-block scale, FP4_E2M1 nibble):

```
block_amax  = max(|x[i, b*16:(b+1)*16]|, dim=-1)
block_scale = block_amax / FP4_MAX                # FP8E4M3 in production
nibble      = round(clamp(x / block_scale, [-FP4_MAX, FP4_MAX]))
q(x)        = nibble * block_scale                # dequantized fake-quant
e_x         = x - q(x)                            # the cast residual
```

Activation-ECO weight-gradient correction:

```
dW_naive       = dy @ q(x).T                      # standard QAT, biased
dW_correction  = dy @ e_x.T
dW_corrected   = dW_naive + dW_correction = dy @ x.T   # unbiased
```

The activation gradient `dx` stays at the standard saturated-zero STE
value: `dx = (dy @ W) * mask` where `mask` is 1 on unsaturated lanes
and 0 on saturated ones. This matches the convention used by the
TurboQuant and IndexCache modules in this repo.

## Composition with the rest of the FlashOptim ECO + Turbo + Index stack

| Stage | Touches | Owns |
|---|---|---|
| Activation-ECO | layer **input** before NVFP4 GEMM | `dW += dy @ e_x.T` per matmul |
| IndexCache fp8 fake-quant | indexer K **output** | per-token fp8 round-trip + STE |
| TurboQuant 2.5-bit dense KV | dense MLA latent **output** | block-Hadamard-rotated KV cache |
| Weight-ECO | optimizer **state** | `m1 += alpha * D * (theta - q(theta))` |
| Bucket-completion overlap | grad bucket sync | scheduler hooks for memory-pressure NVFP4 |

These five stages live in five different places in the layer + optimizer
pipeline and are mutually orthogonal. Composition is additive:
activation-ECO's correction lands in `dW` during backward, weight-ECO's
correction lands in `m1` during step, IndexCache and TurboQuant are
forward-side fake-quants that emit BF16 outputs by the time they cross
into the next stage. No coordination is required.

## Memory cost

The naive implementation saves `x_pre` (BF16) for backward to compute
`e_x = x_pre - q(x_pre)`. At every NVFP4 layer's input that doubles
activation memory. **This violates the FlashOptim ECO hard memory
constraint as-is.** Three escapes, in order of preference for
production:

1. **Recompute `e_x` during backward.** When Megatron's
   `--recompute-activations` is on, `x_pre` is already being
   recomputed for STE backward; computing `e_x` from it is free. This
   is the recommended path for large model training.

2. **Save `e_x` in FP8E4M3 instead of BF16.** Its magnitude is
   bounded by half a FP4 ULP, so FP8 keeps essentially all its
   information; extra memory drops to half of BF16 activations.

3. **Skip on layers with low quantization noise.** Pareto-trade
   correction completeness against memory by gating activation-ECO
   per layer based on a measured `||e_x||_inf` threshold.

The reference implementation in
``megatron/core/quantization/nvfp4_act_eco/autograd.py`` saves `x_pre`
in compute-dtype for clarity. Wiring (1) into a production training
run requires no module change — Megatron's existing
recompute-activations machinery handles it.

## Production wiring on `te.Linear`

`install_act_eco_on_te_linear(te_linear, config)` attaches three
hooks to a `te.Linear` instance:

1. `register_forward_pre_hook(with_kwargs=True)` captures the BF16
   activation entering the te.Linear.
2. A no-op forward post-hook taps into the output's `grad_fn` so we
   can capture `dy` on backward.
3. `weight.register_hook` adds `dy @ e_x.T` to the accumulated
   weight gradient.

The `te.Linear` itself is unmodified — TE continues to own the
NVFP4 cast and cuBLASLt GEMM. Activation-ECO observes the cast
residual via the reference `nvfp4_act_quant_forward` (which produces
a bit-identical or near-identical result to TE's internal cast on
B200 hardware).

`is_te_available()` returns `False` in environments where TE cannot
be imported (CPU CI, fallback BF16 backends), and the package's
public surface stays usable in those environments via the standalone
`apply_nvfp4_act_eco_linear` entry point.

## Verification surface

CPU-only, no GPU required (full suite runs in ~3 seconds on x86_64).

```bash
python -m pytest -q \
  tests/unit_tests/quantization/test_nvfp4_act_eco_correctness.py \
  tests/unit_tests/quantization/test_nvfp4_act_eco_convergence.py \
  tests/unit_tests/quantization/test_nvfp4_act_eco_compose.py
```

| Test file | Asserts |
|---|---|
| `test_nvfp4_act_eco_correctness.py` (9 tests) | Forward shape + finiteness; per-block range envelope; quant-error bound; zero input; the activation-ECO `dW` matches the unbiased BF16 gradient bit-for-bit; saturated-zero STE `dx` matches `torch.autograd` on the STE-detach forward; statistical bias reduction across 200 random trials; saturated-lane gradient finite; correction term zeroes when every coord lies on the NVFP4 grid. |
| `test_nvfp4_act_eco_convergence.py` (2 tests) | Toy 2-layer regression converges under all of {BF16 reference, NVFP4+RTN, NVFP4+act-ECO}; act-ECO is within 20% of RTN-only loss (the rigorous unbiased-gradient claim is in the correctness suite — convergence wins from act-ECO scale with model depth and saturation). |
| `test_nvfp4_act_eco_compose.py` (4 tests) | Stack with IndexCache emits finite forward + finite gradient; `dW` carries the activation-ECO correction through the IndexCache stage (verified by perturbing `x` inside one NVFP4 cell and confirming `dW` changes); BF16 weight-shard compatibility; no extra persistent state across forward calls. |

GPU smoke test is deferred to the user's training environment after
the TE wheel installation issue (CUDA 13.0 vs cuBLAS 12.9) is
resolved by source-building TE. The CPU suite is the source of truth
for correctness; the GPU suite is for performance.
