# GCP A4 Kernel Fast-Path Audit

This note records the current full-fleet fast-path profile and the available
one-GPU B200 microbench results. The launch profile is authoritative; older
tables are labeled when they came from a previous diagnostic shape.

## Launch Profile

Authoritative launcher: `handoff/gcp-a4/scripts/launch_deepseek_nvfp4.sh`

Current default shape:

```text
TP=4 PP=5 CP=2 EP=8 ETP=1 DP=3
MBS=2 GBS=60 grad_accum=10
SEQ_LENGTH=32768 DSA_INDEXER_TOPK=1024
ENABLE_VPP=0
```

The local query sequence for CP ranks is 16384 tokens per microbatch and the
gathered key/value extent is 32768 tokens. The MBS2 runtime shape therefore has
32768 query rows per rank before tensor/head dimensions are applied.

The launcher now defaults to the fast-memory profile:

```text
FASTPATH_STRICT=1
USE_STREAMBP=0
MEGATRON_CHUNKED_LM_HEAD_LOSS=0
MEGATRON_STREAMBP_FUSED_LCE=0
FINE_GRAINED_ACTIVATION_OFFLOADING=0
MEGATRON_TEMP_ACTIVATION_OFFLOAD=0
MEGATRON_PIPELINE_QUEUE_OFFLOAD=1
MEGATRON_PIPELINE_QUEUE_OFFLOAD_DEPTH=1
RECOMPUTE=0
NVFP4_ACTIVATION_ECO=0
FLASH_ADAMW_ECO=1
EMPTY_UNUSED_MEMORY_LEVEL=0
allocator=expandable_segments:True
MEGATRON_HISA_CANDIDATE_MAX_TEMP_MB=0
MEGATRON_HISA_CANDIDATE_FREE_MEM_RESERVE_MB=0
MEGATRON_HISA_CANDIDATE_TRIM_CACHE=0
```

`FASTPATH_STRICT=1` rejects accidental re-enablement of the old
memory-conservative profile, including StreamBP replay/offload sub-switches,
activation/temp offload module lists, activation ECO, cache trims, HISA scratch
caps/cache trimming, DSA deferred query gradients, and DSA teacher-score
recompute, because those reintroduce extra backward work. Disable strict mode
only for a diagnostic run.

The pipeline queue offload exception is deliberate. Depth-1 boundary queue
offload is enabled; the expensive profile was TE/core-attention activation
offload, which is still disabled.

This only disables activation ECO. Optimizer ECO stays enabled:
`FLASH_ADAMW_ECO=1` remains part of the strict fast path.

## Active Quality-Critical Kernels

These remain enabled and are expected in the launch audit:

```text
USE_HIGGS=1
INDEXCACHE=1
DSA_INDEXCACHE_HISA=1
DSA_INDEXCACHE_QUANTIZATION=nvfp4_e2m1_ue8m0
TURBOQUANT=0
MEGATRON_HISA_SELECTOR_BACKEND=bmm
MEGATRON_HISA_FUSED_INDEXER_LOSS=1
MEGATRON_HISA_FALLBACK_DENSE_IF_SHORT=0
MEGATRON_DSA_TRITON=1
MEGATRON_DSA_SPLIT_QK=1
MEGATRON_DSA_STREAMING_INDEXER_TOPK=1
MEGATRON_DSA_COMPACT_TOPK_INDICES=1
```

The DSA/HISA indexer remains trainable; the launch keeps
`DSA_INDEXER_LOSS_COEFF=0.1` and does not zero the indexer loss.

HISA is not an additional dense DSA indexer pass. The paper describes it as a
drop-in indexer replacement: a block-level coarse filter first retains
candidate blocks, then the original token-level indexer refines only inside
those candidates. The local DSA path follows that structure and fails closed if
HISA cannot produce selected scores for the trainable indexer loss, rather than
falling back to full-prefix DSA index-score KL.

## Chunking State

Current defaults:

```text
DSA_CHUNK_SIZE=2048
MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE=4096
MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD_CHUNK=32768
MEGATRON_DSA_SPLIT_QK_REENTRANT_DEFER_QUERY_GRADS=0
MEGATRON_DSA_CUDA_SPLIT_QK_ROW_QUERY_BWD=1
MEGATRON_DSA_CUDA_SPLIT_QK_ROW_BWD_WARPS=8
MEGATRON_DSA_TEACHER_SCORE_SCRATCH=1
MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH=1
MEGATRON_HISA_SELECTOR_ROW_CHUNK=256
MEGATRON_HISA_TARGET_ROW_CHUNK=256
MEGATRON_HISA_CANDIDATE_MAX_TEMP_MB=0
MEGATRON_HISA_CANDIDATE_TRIM_CACHE=0
MEGATRON_DEEPEP_COMPACT_ROW_CHUNK=131072
```

With CP2, `DSA_CHUNK_SIZE=2048` still chunks the local 16384-token query axis.
That is a current memory-risk compromise, not a proven throughput optimum. The
next PP5/VPP-off profile should tell us whether the chunk can be raised toward
4096/8192/16384 or whether MoE/DSA live activation pressure still requires it.

The DSA split-QK reentrant KV backward chunk was raised from 8192 to 32768 after
the measured speedup below. Query gradients are now emitted by the CUDA row
backward in the same packed K/V pass at the default single-KV-chunk shape.
Strict mode rejects deferred query gradients, changing the split-QK K/V backward
chunk away from the full 32768-token key extent, disabling packed K/V gradients,
or changing the split-QK row backward scheduler away from 8 warps; the measured
topk-1024 K/V-gradient timings were 7.88s at 4 warps, 7.24s at 8 warps, and
10.19s at 16 warps. It also rejects smaller HISA selector row chunks than the
current 256 default, HISA target row chunks other than 256, and HISA selector
backends other than the measured `bmm` path.

`MEGATRON_DSA_TEACHER_SCORE_SCRATCH=1` is the main DSA recompute removal knob
for HISA training. The selected sparse-attention teacher path now saves the
actual attention scores from forward and uses them during backward instead of
recomputing QK scores per selected edge. It is a memory-for-speed tradeoff:
roughly 8 GiB extra per live DSA layer at MBS4/topk1024, or 4 GiB at MBS2.

HISA selector row chunking is currently 256. That was reduced after full-fleet
OOM pressure; it should be revisited once PP5/VPP-off reaches the first full
profile step. `MEGATRON_HISA_CANDIDATE_MAX_TEMP_MB=0` means no artificial
candidate-refine scratch cap, and
`MEGATRON_HISA_CANDIDATE_TRIM_CACHE=0` keeps the selector from calling
`torch.cuda.empty_cache()` in the hot path.

`MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE` only affects the non-HISA streaming QK
top-k fallback used when the indexer loss is disabled. The current HISA
trainable selected-score path does not run that full-prefix key-block loop.
`MEGATRON_DSA_STREAM_TRITON_ATTENTION_CHUNKS=1` is not an additional attention
split in the GCP profile because the query chunk is the whole local CP sequence;
strict mode rejects smaller query chunks.

The CE path is not the StreamBP/chunked LM-head loss path. With
`USE_STREAMBP=0` and `MEGATRON_CHUNKED_LM_HEAD_LOSS=0`, `GPTModel.forward()`
materializes the output logits and calls `vocab_parallel_cross_entropy`. The
`LCE_*_VOCAB_SPLIT_SIZE` values are inactive in this profile; they only affect
the fused linear-CE or StreamBP fused-LCE paths, which strict mode does not
allow for this run profile.

## B200 Microbench Results

The table below is retained from the older one-GPU CP8 diagnostic shape. It is
useful for relative kernel choices, but it is not sufficient proof for the
current CP2/MBS2 full-fleet shape. Regenerate with the script defaults after
this patch for the current launcher-aligned shape:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python tools/bench_deepseek_hot_kernels.py \
  --components higgs,indexcache,hisa,dsa-attn \
  --dsa-use-teacher --warmup 1 --iters 3 --device cuda:0
```

Existing key results at `q=4096`, `sk=32768`, `batch=4`, `topk=1024`:

| Component | Time | Peak Allocated |
| --- | ---: | ---: |
| HIGGS dense 2-bit fwd, rows 131072 | 1.18 ms mean | 0.97 GiB |
| HIGGS dense 2-bit fwd+bwd, rows 131072 | 1.84 ms | 1.09 GiB |
| IndexCache NVFP4 fwd, rows 131072 | 0.22 ms mean | 0.26 GiB |
| IndexCache NVFP4 fwd+bwd, rows 131072 | 0.57 ms | 0.29 GiB |
| Dense DSA indexer score matrix, diagnostic only | 65 ms | 2.29 GiB |
| HISA select-with-scores fwd, `bmm` backend, row chunk 2048, no candidate cap | 0.41 s | 1.18 GiB |
| HISA select-with-scores fwd+bwd, `bmm` backend, row chunk 2048, no candidate cap | 0.565 s | 2.53 GiB |
| HISA select-with-scores fwd+bwd, `bmm` backend, row chunk 512 diagnostic | 0.60 s | 2.53 GiB |
| HISA select-with-scores fwd, `deepgemm` backend | 1.67 s | 1.42 GiB |
| HISA select-with-scores fwd+bwd, `deepgemm` backend | 1.84 s | 2.56 GiB |
| HISA select-with-scores fwd+bwd, `packed_cublasdx_tiled` backend | 1.41 s | 2.52 GiB |
| HISA select-with-scores fwd+bwd, `packed_cublasdx_fp8` backend | 2.85 s | 2.52 GiB |
| HISA full-row selector diagnostic, row chunk 4096, no candidate cap | 0.589 s | 2.53 GiB |
| DSA split-QK attention fwd, packed MLA KV view | 241 ms | 9.84 GiB |
| DSA split-QK teacher fwd, packed MLA KV view | 370 ms | 9.90 GiB |
| DSA split-QK attention fwd+bwd, KV chunk 8192 | 22.36 s | 33.2 GiB |
| DSA split-QK attention fwd+bwd, KV chunk 16384 | 15.28 s | 35.1 GiB |
| DSA split-QK attention fwd+bwd, packed MLA KV view, KV chunk 32768, deferred query pass | 9.62 s | 36.4 GiB |
| DSA split-QK attention fwd+bwd, packed MLA KV view, KV chunk 32768, CUDA row query grads | 9.41 s | 35.9 GiB |
| DSA split-QK teacher fwd+bwd, score recompute, MBS4 | 9.52 s | 35.9 GiB |
| DSA split-QK teacher fwd+bwd, score scratch, MBS4 | 0.895 s | 55.9 GiB |
| DSA split-QK teacher fwd+bwd, score scratch, MBS2 | 0.447 s | 28.0 GiB |
| DSA split-QK attention fwd+bwd, packed MLA KV view, K-noPE/V grads only | 7.24 s | 35.1 GiB |
| DSA split-QK attention fwd+bwd, packed MLA KV view, topk 512 diagnostic | 4.89 s | 35.9 GiB |

Current launcher-aligned spot check on one idle B200, `tp=4`, `cp=2`,
`batch=2`, `q=16384`, `sk=32768`, `topk=1024`, `hisa_row_chunk=256`,
`target_row_chunk=256`, `candidate_slot_group=8`, `warmup=0`, `iters=1`:

| Component | Time | Peak Allocated |
| --- | ---: | ---: |
| IndexCache NVFP4 fwd, rows 65536 | 0.61 ms | 0.11 GiB |
| IndexCache NVFP4 fwd+bwd, rows 65536 | 0.47 ms | 0.14 GiB |
| HISA select-with-scores fwd, `bmm` backend | 456 ms | 1.71 GiB |
| HISA select-with-scores fwd+bwd, `bmm` backend | 760 ms | 4.76 GiB |
| DSA split-QK attention fwd, packed MLA KV view | 124 ms | 3.64 GiB |
| DSA split-QK teacher fwd, packed MLA KV view | 157 ms | 7.77 GiB |
| DSA split-QK teacher fwd+bwd, score scratch | 443 ms | 11.66 GiB |

This says the standalone HISA/DSA kernels are not independently explaining
multi-minute step time at the current TP4/CP2/MBS2 shape. The next full-fleet
profile should focus on pipeline residency, per-layer repetition, MoE dispatch
and grouped GEMMs, communication stalls, and checkpoint/profiler overhead rather
than assuming a single DSA/HISA kernel is the only bottleneck.

HIGGS and IndexCache are not current throughput limiters. HISA selection is also
not the dominant wall-clock issue at topk 1024. The original DSA split-QK
backward bottleneck was score recompute plus selected-edge K/V gradient work.
Saving attention teacher scores removes the QK recompute part and is now the
default fast path for HISA training. The `bmm` HISA backend remains the fastest
validated immediate path at this shape. The DeepGEMM and packed cuBLASDx/CuTe
extension paths load successfully but are currently slower at `q=4096`,
`sk=32768`, `batch=4`, `topk=1024`; strict mode therefore rejects them for the
training fast path. `FASTPATH_STRICT=0 MEGATRON_HISA_SELECTOR_BACKEND=deepgemm`
is still useful for explicit numerics experiments.

The HISA candidate scratch cap is now disabled in the full-fleet profile. Older
runs used a 32 MB cap and an allocator-cache release helper before candidate
scratch allocation; that was a memory-conservative profile from smaller runs.
The cap sweep did not show a speed win, but disabling it is still the cleaner
120-GPU profile because it removes hidden slot serialization and an optional
`empty_cache()` path from the hot selector.

The real training path uses packed MLA K-noPE/V views from a single `kv` tensor.
At 32k sequence, batch 4, and 128 local heads, selected-key stride arithmetic can
exceed 32-bit element offsets. The split-QK Triton kernels now cast selected
indices to int64 before multiplying by K/V sequence strides so the packed view
path does not fault and does not require materializing full contiguous K/V copies.

The CUDA row query-gradient path is covered by
`test_split_qk_reentrant_cuda_row_query_grads_matches_query_only_pass`. It
matches the previous query-only pass and saves roughly 0.23 seconds in the
one-GPU real-shape fwd+bwd benchmark, so the launcher defaults to it. The modest
size of that win confirms that DSA K/V backward, not query-gradient deferral, is
still the main wall-clock target.

The CUDA row backward shared accumulators must be zeroed across the full
`[WARPS][256]` tile, not just `WARPS * D` or `WARPS * P` contiguous elements.
Otherwise dimensions below 256 can leave later warp rows dirty and corrupt
query gradients in an order-dependent way. This is fixed in
`dsa_sparse_kv_bwd.cu` and validated by the row-query and row-from-scores tests.

The split-QK teacher-score scratch path is covered by
`test_split_qk_teacher_score_scratch_backward_matches_recompute`. It matches the
score-recompute path and is the measured fast path for HISA indexer-loss
training.

## Next Kernel Work

The measured hot spots are too slow for 120 B200s. The likely useful work is:

1. Prove full-stage memory with score scratch enabled. If MBS4 overflows, keep
   score scratch on and try MBS2 before falling back to score recompute; the
   recompute path is an order of magnitude slower in the measured DSA layer.
2. Consider a fused CUDA/CuTe HISA selector only after the DSA backward path is
   improved. HISA is visible but not the current blocker: select-with-scores is
   roughly 0.41s forward and 0.56s forward/backward at the launch shape, and the
   available DeepGEMM/cuBLASDx extension backends are slower on this shape.
3. Keep HIGGS and IndexCache as-is until profiling contradicts the current
   microbench; they are sub-millisecond to low-millisecond at the current row
   counts.
