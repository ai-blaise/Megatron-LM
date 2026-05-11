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
- Reduce `_sparse_dsa_backward_kernel` atomic pressure without lowering DSA topk.
- Reduce `_dsa_indexer_scores_kernel` plus PyTorch `topk` overhead by replacing the
  score-materialize plus `torch.topk` path with an exact fused block-topk/merge kernel.
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
- The current backward is row-major and scatters `grad_key`/`grad_value` with
  atomics. A key-tile/inverted-index backward is the main remaining DSA kernel
  target because it can reduce random atomics and improve K/V reuse without
  changing selected indices.
