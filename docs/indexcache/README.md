# IndexCache fake-quant for the DSA indexer (Megatron-LM)

This package adds selected fake-quant methods on the post-rotation DSA
indexer K tensor:

* `fp8_e4m3`, the original IndexerK8 path.
* `nvfp4_e2m1_ue8m0`, matching the Blackwell NVFP4 IndexCache layout in
  `optimization-playground`.

The FP8 path is a direct port of the forward kernel in
`optimization-playground/python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`.
The NVFP4 path matches
`optimization-playground/python/sglang/jit_kernel/csrc/nsa/nvfp4_indexer_quant.cuh`:
128-dim indexer rows split into four 32-dim groups, packed E2M1 values, and a
single packed UE8M0 scale word per row. Megatron adds analytic backward paths
so both methods can be used during training; the SGLang references are
forward-only.

The op is the kernel-level half of the THUDM/IndexCache feature; the
**cross-layer index reuse** (skip_topk) half is a follow-up — it needs the
surrounding decoder-loop to thread `topk_indices` through layers.

## What it does

For each token in the indexer K of shape `[seq, batch, index_head_dim=128]`,
FP8 computes:

```
abs_max = max_lane(|x|)
amax    = max(abs_max, eps)            # eps = 1e-4, matches SGLang
scale   = amax / 448                   # 448 = fp8_e4m3 max
q_fp8   = cast<fp8_e4m3>(clamp(x/scale, ±448))
y       = q_fp8 * scale                # dequantized fake-quant output
```

The output dtype matches the input. Internally the op saves the per-coord
fp8 index, the saturating-clip mask, the per-row scale, the per-row argmax
of |x|, and an eps-active flag so the backward kernel can skip recomputing
them.

NVFP4 computes the same fake-quant round trip with four independent groups:

```
group_abs_max = max_group(|x|)              # four groups of 32 dims
scale         = ceil_ue8m0(max(group_abs_max, eps) / 6)
q_e2m1        = round_e2m1(clamp(x / scale, ±6))
y             = q_e2m1 * scale
```

For layout parity with optimization-playground, the reference path also
materializes `packed_values[N, 64]` and `packed_scales[N]`, where the four
UE8M0 exponents are packed into one int32. The CUDA training path saves the
dequantized E2M1 values needed by the backward.

## Backward (new — SGLang is forward-only)

```
grad_x_j = grad_y_j * clip_mask_j
         + δ_{j, argmax} * sign(x_argmax) * eps_active * (1/fp8_max)
           * Σ_i [grad_y_i * (q_fp8_i - clip_mask_i * x_i / scale)]
```

The first term is the direct STE on the rounding+clip. The second is a rank-1
update that lives only on the argmax-of-|x| coordinate of each row — it
captures the dependency of `scale` on the row's max-magnitude lane. Verified
against `torch.autograd` on the STE-detach forward to fp64 precision.

For NVFP4 the same formula is applied independently to each 32-dim group with
`fp4_max=6`, producing four rank updates per row.

## Files

```
megatron/core/quantization/indexcache/
├── __init__.py            public API
├── codec.py               IndexCacheConfig + method constants
├── reference.py           pure-PyTorch fwd/bwd (gradcheck oracle)
├── autograd.py            torch.autograd.Function dispatching to CUDA or ref
└── kernels/
    ├── __init__.py
    ├── build.py           torch.utils.cpp_extension JIT build
    └── csrc/
        ├── indexcache.cuh         shared device utils (warp/block reduces)
        ├── indexcache_fwd.cu      FP8 forward kernel
        ├── indexcache_bwd.cu      FP8 backward kernel
        ├── indexcache_nvfp4_fwd.cu
        ├── indexcache_nvfp4_bwd.cu
        └── pybind.cpp             pybind11 entry points
```

## Public API

```python
from megatron.core.quantization.indexcache import (
    apply_indexcache_kv,
    build_indexcache_config,
)

cfg = build_indexcache_config(eps=1e-4)
y = apply_indexcache_kv(x, cfg)  # x: [..., 128]; y same shape & dtype

nvfp4_cfg = build_indexcache_config(
    eps=1e-4,
    quantization="nvfp4_e2m1_ue8m0",
)
```

## Configuration

Two new `TransformerConfig` fields:

```python
dsa_indexcache_quant_enabled: bool = False
dsa_indexcache_quantization: str = "disabled"
dsa_indexcache_quant_eps: float = 1e-4
```

CLI flags:

```
--dsa-indexcache-quant-enabled
--dsa-indexcache-quantization {disabled,fp8_e4m3,nvfp4_e2m1_ue8m0}
--dsa-indexcache-quant-eps 1e-4
```

`--dsa-indexcache-quant-enabled` is retained as a backward-compatible alias
for `--dsa-indexcache-quantization fp8_e4m3`. New configs should use the
explicit method selector.

The DSA hook lives at
`megatron/core/transformer/experimental_attention_variant/dsa.py` —
immediately after `rotate_activation(k)`. K only is quantized. Q stays at
full precision (matches the SGLang reference; the bandwidth cost is on the
K-side cache, and Q is recomputed per query anyway).

## Composes with TurboQuant

The dense MLA latent path (`kv_lora_rank=512`) is TurboQuant 2.5-bit; the
indexer K path (`index_head_dim=128`) is IndexCache FP8 or NVFP4. The two
paths share no tensors. They share the `fast_hadamard_transform` library only
because the DSA indexer applies a Hadamard rotation before scoring.

## Parallelism

The op is per-token-local on the last (head) dim; it commutes with every
parallelism strategy that doesn't split that dim:

| Strategy | Why it's safe |
|---|---|
| **CP / SP** | Per-token; sequence-dim sharding is transparent. |
| **TP** | The hook sits on the `[seq, batch, index_head_dim]` tensor that the indexer owns end-to-end on each rank — no cross-rank reduction needed. |
| **EP** | Indexer is independent of MoE FFN. |

## Verification

| Test | Result |
|---|---|
| FP8 reference forward parity vs SGLang `act_quant` math | bit-identical fp32 |
| FP8 analytic backward vs `torch.autograd` STE-detach oracle | fp64 precision |
| NVFP4 packed values/scales vs OP-compatible Python oracle | exact |
| NVFP4 analytic backward vs independent STE oracle | fp64 precision |
| NVFP4 analytic backward vs `torch.autograd` STE-detach oracle | fp64 precision |
| H200 CUDA behavior | NVFP4 extension symbols build; direct execution rejects with SM100+ guard |

## Override note (DeepSeek-V3.2-REAP target)

The published checkpoint
`BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1`
encodes `IndexerK8` — exactly the fp8 e4m3 scheme this op implements. To
finetune under that regime, enable both `--turboquant-kv-enabled` and
`--dsa-indexcache-quantization fp8_e4m3` in the SFT script.

The NVFP4 indexer path is available for experiments with
`DSA_INDEXCACHE_QUANTIZATION=nvfp4_e2m1_ue8m0`. Its CUDA path is Blackwell
gated; H200 validation covers reference correctness and guard behavior only.

## Followups (not in this PR)

- **Cross-layer index reuse** (skip_topk patterns from the THUDM/IndexCache
  patches): requires modifying `TransformerLayer` and the decoder loop to
  thread `topk_indices` between adjacent layers. Deferred to a follow-up
  because it touches code outside the quantization package.
- **Training-aware multi-layer distillation loss** (paper §4.2): the F-layer
  indexer's KL loss aggregated across served S layers. Builds on top of
  skip_topk.
