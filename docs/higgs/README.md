# HIGGS 2-bit dense MLA-latent KV (Megatron-LM)

This package adds an in-place 2-bit fake-quant on the post-LayerNorm 512-dim
MLA latent. It is a direct port of the forward kernel from the SGLang fork at
`ai-blaise/optimization-playground` (commit `2e2f51717`) and adds a new
analytic backward kernel so the op can be used during training (the SGLang
implementation is forward-only).

HIGGS replaces TurboQuant when the user opts in; the two paths are mutually
exclusive at config-validate time. Slot layout is 258 B / token, which is
16 B / token smaller than the 2.5-bit TurboQuant slot (274 B / token) — a
~6% reduction in compressed KV memory at the same training-time fake-quant
fidelity.

## What it does

For each token, on the post-`kv_layernorm` MLA latent of shape
`[s, b, kv_lora_rank]` with `kv_lora_rank == 512`:

1. Rotate: `rotated = (1/√d) · WHT(x)` (orthonormal block-Hadamard;
   single 512-wide block per token).
2. Per-token block scale: `s = ‖rotated‖ / √d`, stored as fp16 (2 B).
3. Pair codebook: split `normalized = rotated / s` into 256 pairs of
   dimension 2; pick the nearest codeword from the public AquaKV EDEN2-16
   lattice via `argmax_i (2 ⟨x, G_i⟩ - ‖G_i‖²)`. Pack the 4-bit indices
   two-per-byte (128 B / token).
4. Decode: `rotated_recon = s · G[indices]` → `y = (1/√d) · WHT(rotated_recon)`.

The EDEN2-16 codebook is a 16-entry public 2-D lattice from
[AquaKV](https://github.com/goodevening13/aquakv) (cited in the kernel
source comments). It does **not** require Lloyd-Max calibration, unlike
TurboQuant — so HIGGS has zero per-layer randomness and adopting it is
essentially free at training start-up cost. The codebook is identical
across every layer and every rank.

The RoPE part of the MLA latent (`qk_pos_emb_head_dim` dim) is passed
through unchanged.

Reference: HIGGS-KV (Pletka et al.,
[*Cache Me If You Must* / arXiv:2501.19392](https://arxiv.org/abs/2501.19392)).
Kernel-side optimisations (compile-time XOR partner permutation for the
FWHT butterfly, `1/sqrt(N)` pre-scale fold, fused single-kernel store) are
adopted from `togethercomputer/saw-int4` (BDR — Block-Diagonal Rotation),
documented at the top of `kernels/csrc/higgs_kv.cuh`.

## Slot layout (kSlotBytes = 258)

```
[packed 4-bit pair indices: 128 B]
[per-token fp16 block scale:  2 B]
[bf16 rope passthrough (64):128 B]
                              ----
                               258 B
```

That is 16 B / token smaller than the existing 2.5-bit TurboQuant slot
(`kSlotBytes2p5 = 274`). The savings come from the fact that HIGGS uses a
*uniform 2-bit* code (no high/low channel split) and a single scale per
token.

## Files

```
megatron/core/quantization/higgs/
├── __init__.py            public API
├── codec.py               frozen-buffer construction + EDEN2-16 codebook
├── reference.py           pure-PyTorch fwd/bwd (gradient oracle)
├── autograd.py            torch.autograd.Function dispatching to CUDA or ref
└── kernels/
    ├── __init__.py
    ├── build.py           torch.utils.cpp_extension JIT build
    └── csrc/
        ├── higgs_kv.cuh        shared device utils (FWHT, reduces, codebook NN)
        ├── higgs_kv_fwd.cu     forward kernel (port of SGLang store kernel)
        ├── higgs_kv_bwd.cu     backward kernel (new, STE-detach closed form)
        └── pybind.cpp          pybind11 entry points
```

## Public API

```python
from megatron.core.quantization.higgs import (
    apply_higgs_dense_2bit_kv,
    build_higgs_buffers,
)

buf = build_higgs_buffers(
    latent_dim=512,
    preset="dense_2bit",
    layer_idx=layer_number,   # included for API symmetry; codebook is layer-invariant
    device=device,
    dtype=torch.float32,
)
y = apply_higgs_dense_2bit_kv(x, buf)  # x: [..., 512]; y same shape & dtype
```

## Configuration

Two new `TransformerConfig` fields (in `transformer_config.py`):

```python
enable_higgs_dense_2bit_kv_cache: bool = False
higgs_kv_preset: str = "dense_2bit"
```

Two new CLI flags (in `arguments.py`):

```
--enable-higgs-dense-2bit-kv-cache
--higgs-kv-preset dense_2bit
```

The config validator raises `ValueError` if both
`turboquant_kv_enabled` and `enable_higgs_dense_2bit_kv_cache` are True
at the same time.

## Parallelism

The op is per-token-local on the latent dim, so it commutes with every
parallelism scheme that does not split `kv_lora_rank`:

* **TP**: the MLA latent is gathered on TP-2..N before this op runs
  (matches the TurboQuant integration site). Per-token operation, so no TP
  collective communication is added.
* **SP**: sequence-parallel shards along the seq dim, which has no
  bearing on the per-token op.
* **CP**: context-parallel shards along the seq dim, same as SP.
* **EP**: expert-parallel shards along the MoE expert dim, downstream of
  the MLA layer.

`tests/unit_tests/quantization/test_higgs_parallelism.py` verifies all
three with a 2-GPU torchrun run.

## SGLang parity

`HiggsBuffers` carries the same EDEN2-16 codebook as the SGLang reference
codec, and `reference.py::reference_compress` produces byte-identical
packed slots to the SGLang `HiggsDense2BitCodec.compress` for the same fp32
inputs. The parity test
(`tests/unit_tests/quantization/test_higgs_parity_sglang.py`) is gated on
the `SGLANG_REF_PATH` environment variable pointing at a clone of
`optimization-playground`.

## CUDA kernel structure

* Block-per-token, 512 threads per block (one per latent coord).
* Shared 512-fp32 scratch for the FWHT butterfly + a 64-fp32 scratch for
  block reductions.
* Levels 0..4 of the FWHT live inside a warp via `__shfl_xor_sync`;
  levels 5..8 cross warps via SMEM (no swizzle needed since `len >= 32`
  keeps banks distinct).
* Codebook (16 × 2 fp32 = 192 B) lives in the read-only cache via
  `__ldg`. Per-pair nearest-neighbour is a 16-way `argmax` of
  `2 · ⟨x, G_i⟩ - ‖G_i‖²`.
* The backward saves the rotated latent and per-thread codebook coord as
  bf16 in the forward, removing the two largest FWHT recomputations from
  the backward (same pattern as TurboQuant's `w_hat_save`).

## Performance (sanity-check shapes)

The kernel is structurally analogous to TurboQuant's 2.5-bit forward
(both port from SGLang `store_2p5_kernel`-style kernels with a single
512-wide FWHT, a per-token L2 norm, and a per-coord codebook lookup);
the HIGGS variant runs one **shorter** codebook lookup (16-entry argmax
vs 8+4-entry two-tier searchsorted) plus one **less** norm reduction
(no inner-norm Parseval pass), so wall-clock cost per token is
slightly lower. Concrete numbers from the H200 sanity benchmark are
reported in the commit message.

## Run a full training launch

```
bash examples/sft/run_sft_deepseek_higgs.sh
```

Uses the V3.2-REAP model with TP=8 / EP=1 / PP=1 / SP=on. The dense MLA KV
runs through HIGGS; the DSA Indexer's IndexerK8 path is unaffected. To
swap to NVFP4 weights only (no fake-quant on the latent KV), use
`run_sft_deepseek_nvfp4.sh`. To use TurboQuant 2.5-bit instead, use
`run_sft_deepseek_turboquant.sh`.
