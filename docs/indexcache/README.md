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
├── hisa.py                opt-in HISA 4:1 forward selector
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
dsa_indexcache_hisa_enabled: bool = False
dsa_indexcache_hisa_block_size: int = 128
dsa_indexcache_hisa_block_topk: int = 64
dsa_indexcache_hisa_compression_ratio: float = 4.0
```

CLI flags:

```
--dsa-indexcache-quant-enabled
--dsa-indexcache-quantization {disabled,fp8_e4m3,nvfp4_e2m1_ue8m0}
--dsa-indexcache-quant-eps 1e-4
--dsa-indexcache-hisa-enabled
--dsa-indexcache-hisa-block-size 128
--dsa-indexcache-hisa-block-topk 64
--dsa-indexcache-hisa-compression-ratio 4.0
```

`--dsa-indexcache-quant-enabled` is retained as a backward-compatible alias
for `--dsa-indexcache-quantization fp8_e4m3`. New configs should use the
explicit method selector.

The DSA hook lives at
`megatron/core/transformer/experimental_attention_variant/dsa.py` —
immediately after `rotate_activation(k)`. K only is quantized. Q stays at
full precision (matches the SGLang reference; the bandwidth cost is on the
K-side cache, and Q is recomputed per query anyway).

## NVFP4 IndexCache + HISA 4:1 selector

HISA is a configurable selector on top of NVFP4 IndexCache, not a replacement.
Ordinary NVFP4 IndexCache remains the default unless
`--dsa-indexcache-hisa-enabled` is set or the Hugging Face config explicitly
selects it. The selector is valid only with
`--dsa-indexcache-quantization nvfp4_e2m1_ue8m0`.

The strict 4:1 contract is:

- Logical block size `B=128`.
- Eligible block count `M=ceil(t / B)`.
- Selected block count `m=ceil(M / compression_ratio)`, capped by `M`.
- If the selected candidate pool has fewer entries than `index_topk`, the
  sparse DSA path consumes the shorter candidate set via padded indices.
- `compression_ratio=4.0` gives the accepted 4:1 mode.
- The first and last eligible blocks are forced into the selected block set.
  The old `last_minus_one`/`block_count-2` boundary heuristic is not part of
  the default 4:1 path.
- If `t <= index_topk`, top-k selection falls back to ordinary NVFP4
  IndexCache without HISA.
- With indexer loss enabled, HISA still supplies the selected token set for the
  sparse attention path, while the KL objective keeps exact full-candidate
  indexer scores so training semantics remain unchanged.

Hugging Face model-card config path:

```json
{
  "quantization_config": {
    "indexer_quantization": {
      "quant_method": "nvfp4_e2m1_ue8m0",
      "hisa": {
        "enabled": true,
        "mode": "indexcache-hisa",
        "block_size": 128,
        "block_topk": 64,
        "compression_ratio": 4.0,
        "execution_mode": "optimized"
      }
    }
  }
}
```

The converter consumes that block from
`quantization_config.indexer_quantization.hisa`. CLI/config values remain the
runtime source of truth after conversion. The SFT script keeps HISA off by
default; set `DSA_INDEXCACHE_QUANTIZATION=nvfp4_e2m1_ue8m0` and
`DSA_INDEXCACHE_HISA=1` to enable the 4:1 selector.

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
| HISA 4:1 block budget and short-context fallback | exact CPU unit tests |
| HISA 4:1 map-all candidate path at 8192/2048 | exact CPU unit tests |
| H200 CUDA behavior | NVFP4 extension symbols build; direct execution rejects with SM100+ guard |
| Blackwell CUDA behavior | SM103 forward/backward executable path passes 56 fp32/bf16 cases across rows 1..8192 |

The May 2026 Blackwell pass ran on a 2x B300 SXM6 node. The CUDA path matched
the Python/oracle layout exactly for the forward and packed values/scales. The
largest observed backward deltas were `2.384e-7` for fp32 and `2.441e-4` for
bf16, consistent with output dtype rounding. CUDA-event timings for 8192 rows
were about `9.16 us` forward, `8.19 us` backward, and `17.34 us` combined for
bf16. IKP identified forward load/reduce and scale/argmax work as the main
per-warp regions, and backward load plus inner reduction as the main regions.
Warp-local argmax and lane-0 metadata-broadcast variants were correct but did
not improve the 8192-row path, so the baseline kernels remain the selected
implementation.

The backward CUDA path now exposes a packed-value variant used by the autograd
wrapper. It decodes `packed_values[N, 64]` instead of saving and loading
`q_e2m1[N, 128]` as fp32, reducing the NVFP4 backward saved quantized-value
state from 512 bytes/row to 64 bytes/row. On the same B300, paired CUDA-event
timings showed this as a latency tie rather than a speedup at the target 8192
rows: bf16 q-path backward `8.1999 us` vs packed backward `8.2005 us`; fp32
q-path `8.2011 us` vs packed `8.2000 us`. IKP showed the packed decode moves
work from the inner-reduce region into the load/decode region, leaving total
per-warp bf16 time effectively flat (`1.3202 us` baseline vs `1.3203 us`
packed). The change is kept for training memory footprint, not for raw kernel
latency.

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
