# TurboQuant 2.5-bit dense MLA-latent KV (Megatron-LM)

This package adds an in-place 2.5-bit fake-quant on the post-LayerNorm 512-dim
MLA latent. It is a direct port of the forward kernel from the SGLang fork at
`ai-blaise/optimization-playground` and adds a new analytic backward kernel so
the op can be used during training (the SGLang implementation is forward-only).

## What it does

For each token, on the post-`kv_layernorm` MLA latent of shape
`[s, b, kv_lora_rank]`:

1. Normalize: `u = x / ||x||₂`
2. Sign-flip + Walsh–Hadamard rotate: `v = (1/√d) · WHT(u ⊙ s₁) ⊙ s₂`
3. Two-tier scalar quantize against frozen Lloyd–Max codebooks (3-bit on the
   first 32 channels of each 128-group, 2-bit on the remaining 96 — averaging
   to 2.5 bits per coord).
4. Inverse rotate + scale by a norm-correction factor.

`s₁`, `s₂`, codebooks and boundaries are frozen layer-scoped buffers seeded
deterministically from `(seed, layer_idx)` so every TP/SP/CP/EP rank
constructs identical state without collective communication.

The RoPE part of the MLA latent (`qk_pos_emb_head_dim` dim) is passed through
unchanged.

## Files

```
megatron/core/quantization/turboquant/
├── __init__.py            public API
├── codec.py               frozen-buffer construction + Lloyd-Max codebooks
├── reference.py           pure-PyTorch fwd/bwd (gradcheck oracle)
├── autograd.py            torch.autograd.Function dispatching to CUDA or ref
└── kernels/
    ├── __init__.py
    ├── build.py           torch.utils.cpp_extension JIT build
    └── csrc/
        ├── turboquant_kv.cuh         shared device utils (FWHT, reduces)
        ├── turboquant_kv_fwd.cu      forward kernel (port of SGLang)
        ├── turboquant_kv_bwd.cu      backward kernel (new)
        └── pybind.cpp                pybind11 entry points
```

## Public API

```python
from megatron.core.quantization.turboquant import (
    apply_turboquant_kv,
    build_turboquant_buffers,
)

buf = build_turboquant_buffers(
    latent_dim=512,
    preset="latent_2p5bit_nc",
    seed=0,
    layer_idx=layer_number,
    device=device,
    dtype=torch.float32,
)
y = apply_turboquant_kv(x, buf)  # x: [..., 512]; y same shape & dtype
```

## Configuration

Three new `TransformerConfig` fields (in `transformer_config.py`):

```python
turboquant_kv_enabled: bool = False
turboquant_kv_preset: str = "latent_2p5bit_nc"
turboquant_kv_seed: int = 0
```

Three new CLI flags (in `arguments.py`):

```
--turboquant-kv-enabled
--turboquant-kv-preset latent_2p5bit_nc
--turboquant-kv-seed 0
```

The MLA hook lives at
`megatron/core/transformer/multi_latent_attention.py` immediately after
`kv_layernorm` and before `linear_kv_up_proj`.

## Parallelism

The op is per-token-local and commutes with every parallelism strategy that
does not split the latent dimension:

| Strategy | Why it's safe |
|---|---|
| **CP** (context parallel) | Op is per-token; sequence-dim sharding is transparent. |
| **SP** (sequence parallel) | Same as CP — per-token, no cross-token communication. |
| **TP** (tensor parallel) | The hook sits AFTER the gather of `kv_compressed`; the kernel sees the full 512-dim latent regardless of TP shard count. |
| **EP** (expert parallel) | Affects MoE FFN only; orthogonal to attention. |

Frozen buffers are constructed locally on each rank from a deterministic
seed; replication is verified bit-exact by
`tests/unit_tests/quantization/test_turboquant_parallelism.py`.

## Verification

| Test | Result |
|---|---|
| Pure-PyTorch reference vs SGLang round-trip | bit-identical fp32 |
| Analytic backward vs `torch.autograd` (STE) | < 2e-7 abs in fp64 |
| CUDA forward vs reference | < 2e-6 abs in fp32 |
| CUDA backward vs reference | < 2e-6 abs in fp32 |
| Parallelism (sharded vs unsharded fwd+bwd) | bit-exact (0 diff) |
| 200-step mini convergence | TurboQuant within 5% of baseline |

## Benchmarks (NVIDIA B200)

Latent dim 512, fwd + bwd (per-iter, microseconds; 200 iters with 20-iter warmup).
Backward saves `w_hat` from forward to skip the recompute_w_hat region:

| tokens | dtype | fwd | bwd | fwd+bwd | tokens/sec (fwd+bwd) |
|---|---|---|---|---|---|
| 256   | bf16 |  38 us | 156 us | 194 us |  1.3 M |
| 1024  | bf16 |  38 us | 158 us | 195 us |  5.3 M |
| 4096  | bf16 |  38 us | 159 us | 197 us | 20.8 M |
| 16384 | bf16 |  99 us | 131 us | 231 us | 71.1 M |
| 65536 | bf16 | 391 us | 500 us | 891 us | 73.6 M |

Below 16K tokens kernel-launch dispatch dominates. The save-w_hat
optimization shaves 12–14% off the backward at large token counts (151 µs
→ 131 µs at 16K, 581 µs → 500 µs at 65K) at the cost of 1 KB bf16/token
of activation memory.

### Saved-w_hat dtype: bf16 with ECO, fp32 without

`w_hat` is saved in **bf16** (1 KB/token) by default. The bf16 round-trip
introduces ~1e-3 max abs / 1e-4 relative-norm error in `grad_x` vs the
fp32-saved baseline — well below the gradient-noise floor that
**FlashAdamW + ECO** (Error-Compensating Optimization, arXiv:2601.22101)
absorbs by injecting weight-quant error into the momentum buffer each
step. Convergence under plain AdamW (no ECO) on the target-shape
mini-config: TurboQuant +4.79% rel vs no-quant baseline, well within the
10% tolerance.

Callers without ECO can switch back to fp32 by editing
`autograd.py:w_hat_save` to `dtype=torch.float32` and updating the
pybind dtype check; this is a 2-line change. A config-flag toggle is on
the followup track.

## Running training

The example SFT script lives at
`examples/sft/run_sft_deepseek_turboquant.sh`. It is a fork of
`run_sft_deepseek_nvfp4.sh` with `--turboquant-kv-enabled` added; it
overrides the target checkpoint's `KV4` scheme on the dense KV.

```bash
bash examples/sft/run_sft_deepseek_turboquant.sh
```

## IKP profiling (optional)

Set `MEGATRON_TURBOQUANT_IKP=1` and `IKP_ROOT=<path to intra-kernel-profiler>`
before importing the package; the kernel is rebuilt with named regions
(`norm`, `rotate`, `quant`, `parseval`, `inv_rot`, `writeout` for forward;
`recompute_w_hat`, `chain_outputs`, `invert_rotation`, `chain_norm` for
backward) so per-region traces appear in IKP Explorer.

## Tests

```bash
# CPU unit tests (no GPU needed)
pytest tests/unit_tests/quantization/test_turboquant_kv.py

# CPU parity vs SGLang reference (requires SGLang clone)
SGLANG_REF_PATH=/path/to/optimization-playground \
    pytest tests/unit_tests/quantization/test_turboquant_parity_sglang.py

# Multi-GPU parallelism correctness
torchrun --nproc-per-node=2 -m pytest \
    tests/unit_tests/quantization/test_turboquant_parallelism.py
```
