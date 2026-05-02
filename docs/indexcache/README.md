# IndexCache fp8 fake-quant for the DSA indexer (Megatron-LM)

This package adds an in-place fp8 e4m3 fake-quant on the post-rotation DSA
indexer K tensor. It is a direct port of the forward kernel in
`optimization-playground/python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`
(plus its Triton fallback at `python/sglang/srt/layers/attention/nsa/triton_kernel.py`)
and adds a new analytic backward kernel so the op can be used during training —
the SGLang reference is forward-only.

The op is the kernel-level half of the THUDM/IndexCache feature; the
**cross-layer index reuse** (skip_topk) half is a follow-up — it needs the
surrounding decoder-loop to thread `topk_indices` through layers.

## What it does

For each token in the indexer K of shape `[seq, batch, index_head_dim=128]`:

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

## Files

```
megatron/core/quantization/indexcache/
├── __init__.py            public API
├── codec.py               IndexCacheConfig + INDEXCACHE_FP8_MAX
├── reference.py           pure-PyTorch fwd/bwd (gradcheck oracle)
├── autograd.py            torch.autograd.Function dispatching to CUDA or ref
└── kernels/
    ├── __init__.py
    ├── build.py           torch.utils.cpp_extension JIT build
    └── csrc/
        ├── indexcache.cuh         shared device utils (warp/block reduces)
        ├── indexcache_fwd.cu      forward kernel (port of SGLang)
        ├── indexcache_bwd.cu      backward kernel (new)
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
```

## Configuration

Two new `TransformerConfig` fields:

```python
dsa_indexcache_quant_enabled: bool = False
dsa_indexcache_quant_eps: float = 1e-4
```

CLI flags:

```
--dsa-indexcache-quant-enabled
--dsa-indexcache-quant-eps 1e-4
```

The DSA hook lives at
`megatron/core/transformer/experimental_attention_variant/dsa.py:881` —
immediately after `rotate_activation(k)`. K only is quantized. Q stays at
full precision (matches the SGLang reference; the bandwidth cost is on the
K-side cache, and Q is recomputed per query anyway).

## Composes with TurboQuant

The dense MLA latent path (`kv_lora_rank=512`) is TurboQuant 2.5-bit; the
indexer K path (`index_head_dim=128`) is IndexCache fp8. The two paths share
no tensors. They share the `fast_hadamard_transform` library only because
the DSA indexer applies a Hadamard rotation before scoring. Convergence
under both ops simultaneously is verified within ±2% of the no-quant
baseline at 200 steps on a target-shape mini config.

## Parallelism

The op is per-token-local on the last (head) dim; it commutes with every
parallelism strategy that doesn't split that dim:

| Strategy | Why it's safe |
|---|---|
| **CP / SP** | Per-token; sequence-dim sharding is transparent. |
| **TP** | The hook sits on the `[seq, batch, index_head_dim]` tensor that the indexer owns end-to-end on each rank — no cross-rank reduction needed. |
| **EP** | Indexer is independent of MoE FFN. |

Bit-exact across ranks for both forward and backward
(`test_indexcache_parallelism.py` analog).

## Verification

| Test | Result |
|---|---|
| Reference forward parity vs SGLang `act_quant` Triton math | bit-identical fp32 |
| Analytic backward vs `torch.autograd` (STE-detach oracle) | < 2e-7 abs in fp64 |
| CUDA forward vs reference (B200) | 0.0 abs (bit-exact) |
| CUDA backward vs reference (B200) | 3.3e-7 abs (machine precision) |
| Parallelism (sharded vs unsharded fwd+bwd, 2-rank) | bit-exact (0 diff) |
| 200-step convergence with TurboQuant + IndexCache both on | -1.45% vs baseline |

## Override note (DeepSeek-V3.2-REAP target)

The published checkpoint
`BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1`
encodes `IndexerK8` — exactly the fp8 e4m3 scheme this op implements. To
finetune under that regime, enable both `--turboquant-kv-enabled` and
`--dsa-indexcache-quant-enabled` in the SFT script.

## Followups (not in this PR)

- **Cross-layer index reuse** (skip_topk patterns from the THUDM/IndexCache
  patches): requires modifying `TransformerLayer` and the decoder loop to
  thread `topk_indices` between adjacent layers. Deferred to a follow-up
  because it touches code outside the quantization package.
- **Training-aware multi-layer distillation loss** (paper §4.2): the F-layer
  indexer's KL loss aggregated across served S layers. Builds on top of
  skip_topk.
