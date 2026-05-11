# DSA + StreamBP Fusion Plan

This note preserves the current reasoning for the DeepSeek V3.2 REAP SFT stack after
context compaction. The training target is W4A4KV4 with IndexCache FP8 on the DSA
indexer K path, TurboQuant on MLA latent KV, StreamBP, FlashAdamW + ECO, G1 gates,
and gated norm.

## Non-negotiable Semantics

- Keep DSA behavior. Do not replace DSA with dense attention for the real run.
- Keep high DSA indexer top-k. Lowering top-k is a diagnostic only, not a training plan.
- Keep IndexCache semantics: FP8 E4M3 fake-quant only on the post-rotation DSA indexer K.
  The indexer Q stays unquantized.
- Keep sparse DSA attention K/V gradient accumulation in FP32 by default. BF16 atomics are
  only an experimental opt-in and are not the training default.
- Preserve StreamBP correctness for MoE and packed DSA replay.

## Current Bottleneck Picture

- Real 16x B200 probes are dominated by StreamBP replay plus DSA/MoE launch volume.
- The previous dtype-correct known-good probe was about 660s/update at GBS 8 with
  `DSA_CHUNK_SIZE=2048`.
- A real-model probe with larger sparse DSA Triton top-k tiles regressed to 1645s/update,
  so tile-width microbenchmarks do not predict the full pipeline. Keep DSA sparse-attention
  tile defaults at the known-good values unless a real probe proves otherwise.
- `DSA_CHUNK_SIZE=4096` was memory-safe but slower:
  - no score/top-k workspace reuse: about 700.0s/update.
  - with score/top-k workspace reuse: about 698.2s/update.
- The 4096 result means allocator reuse is not enough by itself, and larger DSA chunks are
  not automatically better in the full StreamBP/MoE pipeline. Re-test `DSA_CHUNK_SIZE=2048`
  with workspace reuse before keeping the 4096 default.
- Node/pipeline imbalance is visible: downstream PP ranks can remain active while upstream
  ranks are idle or waiting. Full-run validation must use the two-node real model, not only
  single-GPU microbenchmarks.

## Fusion Options

### 1. Macro-fuse DSA indexer top-k chunks

Current DSA top-k generation is chunked by query length. With StreamBP, it was tied to
`STREAMBP_CHUNK_SIZE=2048`, which means 16 indexer-score/top-k phases per 32k sequence per
DSA layer/replay. Sparse DSA attention already runs as one autograd op after the full
`topk_buffer` is built.

Plan:
- Decouple `DSA_CHUNK_SIZE` from `STREAMBP_CHUNK_SIZE`.
- Default DSA top-k generation to a larger chunk, starting at 4096 for StreamBP.
- This halves DSA indexer score/top-k/copy launch count without changing DSA math or sparse
  attention backward.
- Main risk is temporary score tensor memory. For 4096 x 32768 FP32 scores, the temporary is
  about 512 MiB per active chunk, which is small relative to the 150-160 GiB observed peaks.

### 2. Exact fused DSA indexer score + top-k

The ideal kernel would score and select top-k without materializing `[batch, q_chunk, sk]`.
This would remove the score tensor and PyTorch top-k launch.

Constraint:
- Exact top-k 2048 over 32k keys with 64 index heads is too large for a single practical
  Triton program. Keeping a 2048-entry candidate heap per query while looping K blocks would
  blow register/shared-memory pressure.

Practical route:
- Implement a CUDA extension, not a pure Triton toy, with a block-level/top-k merge design.
- Preserve exact top-k and IndexCache FP8-dequantized K values.
- Validate against dense scores/top-k before using in training.

### 3. DSA backward selected-edge reduction

Current sparse DSA backward atomically accumulates K/V gradients from query-selected edges.
This preserves semantics but creates high atomic pressure.

Possible route:
- Build selected edges `(key_idx, q_idx, topk_slot)` and reduce by key/head, using sorted or
  bucketed selected keys.
- This could reduce atomics but likely adds sort/scatter kernels and a large temporary edge
  list. It needs a real CUDA implementation and full-model probes.

### 4. StreamBP replay launch reduction

Profiler evidence showed StreamBP replay overhead and dynamic MoE expert-token shapes causing
repeated compile/autotune and many launches.

Possible route:
- Keep MoE replay chunking but stabilize expert-token shapes with padding/bucketing where it
  does not change routing semantics.
- CUDA graph capture is attractive only after dynamic shapes are stabilized.
- Weighted SwiGLU Triton fusion did not improve the real probe by itself; do not assume
  local fuser speedups transfer to the full model.

### 5. NCCLX / communication overlap

NCCLX remains useful for RoCE/RDMA communication, but it is separate from the current DSA
compute bottleneck. Use NCCL while fusing DSA/StreamBP unless the torchcomms adapter is
validated independently.

## Immediate Next Probe

Isolate the DSA workspace reuse change against the known-good chunk size:

- Keep `TP=4 PP=4 CP=1 EP=4 ETP=1`.
- Keep `MBS=2`, `GBS=8` for comparison to the 660s baseline.
- Set `DSA_CHUNK_SIZE=2048`, `STREAMBP_CHUNK_SIZE=2048`.
- Keep `MEGATRON_DSA_TRITON_BF16_GRAD_ATOMICS=0`.
- Keep `MEGATRON_WEIGHTED_SWIGLU_FUSER=eager`.

If 2048 plus workspace reuse beats the old 660s baseline, keep the reuse patch and revert the
SFT default back to the StreamBP chunk size. If it does not, the next meaningful optimization
is exact CUDA score+top-k or StreamBP/MoE replay launch reduction, not more chunk-size tuning.

## Implemented So Far

- Added `dsa_indexer_scores_triton(..., out=...)` so the fused indexer score kernel can fill
  a caller-provided FP32 score workspace.
- Reused DSA index-score and PyTorch `topk` output buffers inside `chunked_dsa_forward`.
  This preserves exact top-k and IndexCache/DSA dtype semantics while reducing repeated
  large allocations and allocator churn during StreamBP replay.
- Changed the SFT StreamBP default DSA top-k chunk to 4096, still overridable with
  `DSA_CHUNK_SIZE`. Real probe result: memory-safe but slightly slower than the 2048 baseline
  before workspace reuse, so compare again after the workspace reuse change before keeping it.
