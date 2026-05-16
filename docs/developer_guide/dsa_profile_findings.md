# DSA Profile Findings

Nsight profile: `/home/sjpat/profiles/nsys_mbs2_gbs32_20260510_023036`

Later Nsight profile: `/home/sjpat/profiles/nsys_mbs2_gbs32_20260510_193206`

Run shape:

- `TP=4 PP=4 CP=1 EP=4 ETP=1`
- `MICRO_BATCH_SIZE=2 GLOBAL_BATCH_SIZE=32`
- `SEQ_LENGTH=32768`
- `USE_STREAMBP=1`
- `STREAMBP_CHUNK_SIZE=2048`
- `DSA_INDEXER_TOPK=2048`

Main finding:

- DSA/indexer kernels dominate GPU kernel time on node0.
- `_sparse_dsa_backward_kernel`: 48.5%, 15,872 launches, 5,763.7 aggregate GPU seconds.
- `_sparse_dsa_forward_kernel`: 22.1%, 31,744 launches, 2,628.3 aggregate GPU seconds.
- `_dsa_indexer_scores_kernel`: 9.8%, 126,976 launches, 1,161.2 aggregate GPU seconds.
- NCCL is second-order at roughly 11-12% aggregate kernel time.
- Weighted SwiGLU and generic GEMM kernels are not the current first-order bottleneck.

The later full-profile run was even more concentrated in sparse DSA:

- `_sparse_dsa_backward_kernel`: 70.9%, 528 launches, 5,773.4 aggregate GPU seconds.
- `_sparse_dsa_forward_kernel`: 11.9%, 1,404 launches, 968.7 aggregate GPU seconds.
- `_dsa_indexer_scores_kernel`: 9.0%, 22,464 launches, 729.5 aggregate GPU seconds.
- NCCL send/recv was roughly 5.3% aggregate GPU kernel time.

This means the next useful changes should reduce DSA backward arithmetic,
memory traffic, and atomics. Removing one or two Python/Triton launches is not
material at this point.

Microbench shape:

- `q_len=2048`
- `kv_len=32768`
- `heads=32`
- `qk_dim=192`
- `v_dim=128`
- `topk=2048`

Measured DSA sparse attention microbench results on one B200:

| Knob | Topk dtype | Fwd ms | Bwd ms | Peak mem GB | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `BLOCK_Q=1`, fp32 grad atomics | int64 | 20.8 | 73.6 | 2.66 | Current conservative kernel shape. |
| `BLOCK_Q=2`, fp32 grad atomics | int64 | 198.0 | 298.8 | 2.66 | Much worse at real-ish shape. |
| `BLOCK_Q=4`, fp32 grad atomics | int64 | 140.2 | 388.7 | 2.66 | Much worse at real-ish shape. |
| `BLOCK_Q=1`, fp32 grad atomics | int16 | 24.3 | 67.5 | 2.63 | Small net win, safe index storage change. |
| `BLOCK_Q=1`, bf16 grad atomics | int16 | 31.2 | 54.3 | 1.38 | Faster/lower memory, but changes K/V grad accumulation precision. |

Current decision:

- Keep `MEGATRON_DSA_TRITON_BLOCK_Q=1`.
- Default `MEGATRON_DSA_COMPACT_TOPK_INDICES=1` for the SFT script.
- HISA/IndexCache selectors emit `int32` selected-token buffers, and the DSA
  common path narrows PyTorch `topk` `int64` output to `int32` before it reaches
  the CUDA/Triton attention path. The Triton attention API also narrows direct
  `int64` callers at the kernel boundary, so autograd does not retain large
  selected-token buffers as `int64` for 32k sequence runs.
- Keep `MEGATRON_DSA_SORT_TOPK_INDICES=0` by default. Sparse selected attention
  and the trainable indexer-loss teacher path are permutation-invariant when
  top-k indices, selected indexer scores, and teacher probabilities share the
  same order. Sorting is only required by the experimental key-block K/V
  backward binary-search path, which is not a production default. On B200,
  sorting a full `[4,32768,1024]` top-k tensor costs ~3.2 ms versus ~0.2 ms for
  the compact int16 copy alone, and that work repeats across DSA layers and
  StreamBP replay. A focused blocky-selected-index bench at
  `q=512,bsz=2,heads=4,topk=1024` showed sorted top-k is only ~1.2% faster in
  the selected-attention kernel, which does not repay an explicit sort.
- Library default keeps `MEGATRON_DSA_TRITON_BF16_GRAD_ATOMICS=0`, but the DeepSeek
  SFT script opts into `MEGATRON_DSA_TRITON_BF16_GRAD_ATOMICS=1` by default after
  focused correctness tests and microbenching. Override it to `0` for an fp32
  K/V-gradient-atomic control run.
- Add `MEGATRON_STREAMBP_DSA_FULL_NO_GRAD_FORWARD=1` for the SFT StreamBP path. This
  collapses the initial no-grad checkpoint forward for DSA-backed layers into a
  full forward while keeping backward replay chunked. It covers both the current
  `chunked_packed_dsa` layers and any DSA-backed MoE layers. It is a launch-count
  reduction, not a semantic DSA/topk change.

Next optimization targets:

- Split Q/K and V/O vector widths in the sparse DSA Triton kernels. DeepSeek's
  active shape is `qk_dim=192`, `v_dim=128`; the previous kernel used
  `BLOCK_D=next_power_of_2(max(qk_dim, v_dim))=256` for both paths, so value
  loads/stores/atomics and output accumulation carried masked lanes. The split
  keeps Q/K at 256 lanes and V/O at 128 lanes without changing DSA semantics.
- Use BF16 K/V gradient atomics for the SFT run. On the full DSA kernel shape
  `q=2048, kv=32768, heads=32, qk_dim=192, v_dim=128, topk=2048`, the split-dim
  kernel measured ~17 ms forward and ~424 ms backward with fp32 K/V grad atomics,
  versus ~17 ms forward and ~43.5 ms backward with BF16 K/V grad atomics. Peak
  microbench memory dropped from ~2.67 GiB to ~1.42 GiB. Output and dQ were
  identical in a paired check; dK/dV differed at BF16-scale absolute error.
- Probe the full no-grad DSA forward collapse, ideally with `STREAMBP_CHUNK_SIZE=4096`
  if memory allows, to reduce DSA replay launches further.
- Use `MEGATRON_DSA_TRITON_BLOCK_K_BWD=32` with
  `MEGATRON_DSA_TRITON_BWD_NUM_WARPS=2` for the SFT launcher. A focused B200
  sweep at `q_len=2048`, `kv_len=32768`, `heads=32`, `qk_dim=192`,
  `v_dim=128`, `topk=1024`, and BF16 K/V grad atomics measured backward at
  ~17.8 ms versus ~21.5 ms for the previous `BLOCK_K_BWD=16` default, with the
  same ~2.65 GiB peak allocation.
- The sparse DSA Triton attention path now supports `batch > 1`, so MBS>1 does
  not fall back to the PyTorch sparse attention implementation solely because
  the batch dimension is larger than one. A B200 sanity check at `bsz=4`,
  `q=256`, `kv=4096`, `heads=8`, `qk_dim=192`, `v_dim=128`, and `topk=512`
  measured fused Triton forward+backward at ~2.01 ms versus ~30.0 ms for the
  fallback path, with reference-matching forward and gradients covered by
  `test_forward_backward_supports_microbatch_greater_than_one`.
- Use `MEGATRON_HISA_SELECTED_SCORE_BWD_HEAD_GROUP=8` plus
  `MEGATRON_HISA_SELECTED_SCORE_BWD_WARP_GROUPED=1` for the selected-indexer
  loss backward. The old head-grouped kernel reduced selected-K atomics in the
  indexer-loss path, but still computed the eight head dots serially with
  block-wide reductions. The warp-grouped kernel maps one warp to each head,
  broadcasts the lane-0 dot before the ReLU branch, and keeps the same
  one-atomic-per-group selected-K write. On a B200 microbench
  (`Q=4096,H=64,D=128,L=32768,K=1024`), old head-grouped measured ~84.0 ms and
  warp-grouped measured ~38.7 ms.
- For MBS>1, selected-score autograd now batches the backward call by flattening
  microbatch rows and concatenating per-batch indexer K rows. This changes the
  scheduler, not the math: top-k indices are offset into the concatenated K
  table, so each row still only references its own batch item. On B200 with
  `B=4,H=64,D=128,L=32768,K=1024`, four per-batch launches measured ~20.45 ms
  at `q_len=512` and ~77.40 ms at `q_len=2048`; the batched launch measured
  ~19.54 ms and ~76.46 ms respectively. This is a small enabled win that mostly
  removes launch/Python overhead; it does not solve the larger sparse-attention
  K/V atomic problem.
- `MEGATRON_DSA_TRITON_KEY_BLOCK_KV_BWD=1` adds a correctness-checked key-block
  K/V backward path for sparse DSA attention. It runs the row-major backward for
  `grad_query` only, then accumulates K/V gradients by key block and query tile
  before global writes. The key-block kernel now uses only the binary-search
  steps required by the active topk instead of the original hard-coded 16
  iterations. This improved the prototype but still did not make it a training
  default: with BF16 K/V grad atomics on one B200, a moderate shape (`q=256`,
  `kv=1024`, `heads=4`, `qk_dim=192`, `v_dim=128`, `topk=128`) measured
  ~0.36 ms for default row-major backward versus ~0.65 ms for key-block K/V.
  A larger-topk check (`q=128`, `kv=4096`, `heads=2`, `topk=1024`) measured
  ~0.45 ms default versus ~0.73 ms key-block. Keeping it opt-in avoids
  regressing the real run while preserving a covered baseline for the next
  edge-list/block-sparse scheduler.
- A scratch edge-sorted K/V reducer prototype was also tested before adding it
  to production. It sorted selected edges by key, launched one reducer segment
  per selected key/head, and avoided K/V atomics inside each segment. On a
  late-context shape (`q=512`, `kv=32768`, `heads=8`, `topk=1024`,
  `q_start=32256`), edge preparation measured ~0.90 ms and the edge K/V
  reducer alone measured ~4.57 ms, while the existing fused path measured
  ~2.41 ms for full forward+backward on the same small-head benchmark. This
  rules out a PyTorch-sort plus Triton segment reducer as the next production
  patch; a real improvement needs a lower-level block-sparse/CUTLASS-style
  scheduler.
- A query-tile K/V reducer prototype was also tested. It coalesces exact
  same-slot selected keys across neighboring query rows before global writes,
  but even its best high-overlap setting measured ~3.06 ms versus ~2.24 ms for
  the existing fused path on the same late-context benchmark, and a
  production-ish BF16/topk=1024 check was much worse (~44 ms versus ~2.3 ms).
  It remains off by default.
- A narrower in-kernel grouping prototype
  (`MEGATRON_DSA_TRITON_GROUPED_KV_BWD=1`) keeps duplicate-key K/V grouping
  inside the main sparse-attention backward launch instead of adding a second
  reducer launch. It passes sparse-reference gradient parity, but still loses:
  on a B200 BF16/topk=1024 shape (`q=256,bsz=2,heads=4,qk_dim=192,v_dim=128`,
  `kv=32768`) with intentionally correlated top-k rows, default row-major
  forward+backward measured ~2.22 ms while grouped variants measured ~5.69 ms
  (`BQ=2,BK=8`), ~10.43 ms (`BQ=2,BK=32`), and ~9.59 ms (`BQ=4,BK=16`).
  Larger grouped tiles also hit SM100 shared-memory limits. This reinforces
  that the useful next step is not another Triton loop around the current
  row-major representation, but a lower-level scheduler that changes the
  dataflow enough to reduce atomics and memory traffic.
- Reduce `_sparse_dsa_backward_kernel` atomic pressure without lowering DSA topk.
- Reduce `_dsa_indexer_scores_kernel` plus PyTorch `topk` overhead by replacing the
  score-materialize plus `torch.topk` path with an exact fused block-topk/merge kernel.
- The production BMM HISA selector no longer rebuilds a full sequence cumsum
  for every query chunk. The causal partial-block correction now uses
  block-local prefix sums, which measured ~2.85 ms versus the previous ~9.5 ms
  at `Q=64,H=64,D=128,L=32768,topk=1024`.
- Keep DSA topk at 2048 unless explicitly running an ablation; topk reduction changes
  the long-context behavior we are trying to preserve.

Fusion constraints:

- DSA topk is shared across attention heads. A literal one-program attention
  megakernel that recomputes exact topk per attention head would reduce saved
  buffers but multiply indexer work by local attention-head count, so it is not a
  good direction.
- Exact top-2048 over 32k keys is too large for a single Triton program to hold
  in registers/shared memory while also computing the 64-head indexer score. The
  practical exact path is a small kernel family: block score/topk, merge topk,
  sparse attention, and a redesigned backward.
- Exact selected-attention teacher emission now has an opt-in score-scratch path
  (`MEGATRON_DSA_TEACHER_SCORE_SCRATCH=1`, enabled in the SFT launcher). The
  sparse attention forward stores selected logits during the first QK pass and
  uses those stored logits after LSE is known, so teacher probabilities do not
  recompute selected QK. On a focused B200 shape (`q_len=2048`, `kv_len=32768`,
  `heads=32`, `qk_dim=192`, `v_dim=128`, `topk=1024`), this reduced teacher
  forward time from ~17.37 ms to ~10.38 ms with bitwise-identical output and
  teacher tensors, while increasing transient peak allocation by ~0.27 GiB. At
  full 32k this scratch scales to roughly 4.4 GiB for local `heads=32`,
  `topk=1024`, with the same selected-token semantics.
- The same score scratch is now also used by the Triton backward when
  `MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH=1`. This keeps the current row-major
  sparse DSA backward and exact gradients, but skips the backward selected-QK
  recompute needed only to recover probabilities. On a B200 BF16 focused check
  (`q=512,bsz=2,heads=4,qk_dim=192,v_dim=128,kv=32768,topk=1024`), the DSA
  teacher fwd+bwd path improved from ~3.24 ms to ~2.74 ms with unchanged peak
  allocation in the benchmark.
- The selected-attention teacher path is no longer limited to `batch == 1`.
  For MBS>1, HISA selected-score autograd now stacks per-batch selector results
  and gathers teacher rows from the fused sparse-attention emission in
  batch-major flattened order. This prevents MBS=4 from being forced through the
  older per-batch fused-indexer-loss teacher recompute path. A regression test
  makes the old fallback fail so this dispatch stays covered.
- The current backward is row-major and scatters `grad_key`/`grad_value` with
  atomics. A key-tile/inverted-index backward is the main remaining DSA kernel
  target because it can reduce random atomics and improve K/V reuse without
  changing selected indices.
- Three tempting local rewrites are now ruled out as defaults: q-tile unique
  K/V reduction and in-kernel duplicate-key grouping are correct but slower than
  the simple row-major sparse-attention atomics, and cuBLASDx selected-score
  backward is correct but slower than warp-grouped at real row counts because
  tile-level `grad_q` atomics dominate.
- Sorting selected top-k does not make the existing key-block or grouped K/V
  backward paths competitive. A B200 check at
  `q_len=256,bsz=4,heads=8,qk_dim=192,v_dim=128,kv_len=32768,topk=1024`
  measured default row-major fwd+bwd at ~5.86 ms. Sorted key-block measured
  ~258-274 ms, and sorted grouped variants measured ~18 ms to >200 ms. This
  means the remaining production fix has to change the scheduler or selected
  edge representation, not just sort the current token top-k tensor.
- The key-block K/V backward path relies on sorted selected-token rows for its
  binary search. Since the SFT launcher defaults `MEGATRON_DSA_SORT_TOPK_INDICES`
  to `0`, the key-block path now falls back to the default row-major backward
  when the provided top-k tensor is not actually sorted.
