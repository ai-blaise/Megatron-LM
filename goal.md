# Corsaire-1 Kernel Port / AWS Handoff Progress

Last updated: 2026-05-21 02:02 UTC

## Current Objective

Prepare the 32xB200 AWS handoff while auditing `optimization-playground` kernels against the active Megatron training stack. Port only kernels or switches that are both correct and measurably useful on our active paths.

## Current Live Finding

The latest `corsaire-1-research-preview` run is intentionally running with broad fine-grained activation offload and temp activation offload disabled. This was done because the offloaded run was extremely slow in the forward path, and we needed to know whether CPU offload was the direct cause.

Result so far:

- offload disabled is wired into the tmux launcher;
- actual Megatron args show `fine_grained_activation_offloading False` and `offload_modules []`;
- early forward timings are effectively unchanged versus the offload-enabled run:
  - PP0 `vmb=0/vp=0`: ~67s with offload, ~66.6s without;
  - PP0 `vmb=4/vp=1`: ~171s with offload, ~170.5s without;
  - PP1 `vmb=3/vp=0`: ~172s with offload, ~170.8s without;
- current memory is not worse with offload disabled and is healthy during early forward.

Conclusion: broad activation/temp offload should not remain part of the production launcher. It does not explain the current forward slowdown. The bottleneck appears tied to specific VPP/layer groups and their active MoE/DSA/HISA paths, not the offload machinery.

Launcher state:

- `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh` now hard-disables `FINE_GRAINED_ACTIVATION_OFFLOADING` and `MEGATRON_TEMP_ACTIVATION_OFFLOAD` for training launches.
- `examples/sft/run_sft_deepseek_nvfp4.sh` also hard-disables those paths, so stale caller env vars cannot silently re-enable offload when bypassing the tmux wrapper.
- The underlying Megatron feature code is still present, but the production launcher no longer exposes default module lists or opt-in defaults for this run path.

## Current Launcher Readiness

The tmux launcher has been made safer for moving to a replacement second node:

- `REMOTE_HOST`/`NODE1_HOST` is now required instead of defaulting to the old GCP node.
- `MASTER_ADDR` auto-detects from the route to the remote host, but should still be set explicitly if the cluster has a preferred RDMA/control-plane address.
- `SSH_KEY` is optional; set it only when a non-default key is needed. `SSH_OPTS` can carry extra SSH flags.
- `GLOBAL_BATCH_SIZE` now defaults to `16`, matching the current survivable diagnostic shape, instead of silently reverting to `128`.
- `DRY_RUN=1` bypasses existing tmux-session attach behavior and prints the resolved launch shape for verification.
- Optional MoE stage progress logging is wired through as `MEGATRON_MOE_STAGE_PROGRESS_LOG`, `MEGATRON_MOE_STAGE_PROGRESS_RANKS`, and `MEGATRON_MOE_STAGE_PROGRESS_SYNC`.

## Active Training Shape To Preserve

- Launcher entrypoint: `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
- Inner script: `examples/sft/run_sft_deepseek_nvfp4.sh`
- W&B/session name convention: `corsaire-1-research-preview`
- Current important shape defaults:
  - `TP=4`, `PP=4`, `CP=1`, `EP=4`
  - `SEQ_LENGTH=16384`
  - `DSA_INDEXER_TOPK=512`
  - `USE_STREAMBP=1`
  - `STREAMBP_MOE_MLP_CHUNKS=1`
  - `STREAMBP_MOE_MLP_BACKWARD_CHUNKS=4`
  - `MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD=1`
  - `MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD_CHUNK=8192`
  - `MEGATRON_HISA_SELECTOR_BACKEND=deepgemm`
  - `MEGATRON_HISA_BMM_CUBLASDX_REFINE=1`
  - `MEGATRON_HISA_FUSED_INDEXER_LOSS=1`
  - `MEGATRON_HISA_TARGET_TRITON=1`
  - `MEGATRON_HISA_KL_GRAD_TRITON=1`

## What Was Ported / Changed

### HIGGS Dense 2-Bit KV Fake Quant

Source inspected:

- `/tmp/optimization-playground/docs/developer_guide/kernel_results/higgs.md`
- `/tmp/optimization-playground/python/sglang/jit_kernel/csrc/quantization/higgs_dense_2bit_kv.cuh`

Megatron files changed:

- `megatron/core/quantization/higgs/kernels/csrc/higgs_kv.cuh`
- `megatron/core/quantization/higgs/kernels/csrc/higgs_kv_fwd.cu`

Implemented:

- static constant EDEN2-16 codebook/norm arrays;
- constant-codebook nearest-code lookup;
- warp-shuffle pair exchange instead of shared-memory pair handoff;
- swizzled shared-memory indexing in the FWHT helper.

Why:

- OP accepted path showed this exact optimization was useful on B200 for HIGGS store-like kernels.
- Our active training path is fake-quant forward/backward rather than OP's compressed store-only path, so this was adapted conservatively without changing API or saved backward tensors.

Validation:

- `test_cuda_forward_matches_reference`
- `test_cuda_backward_matches_reference`
- `test_cuda_backward_matches_reference_under_bf16_noise`
- Result: 3 passed.

Benchmark result, extension-to-extension:

- Forward at 16k rows: 0.161 ms -> 0.147 ms, about 1.10x.
- Forward+backward at 16k rows: 0.316 ms -> 0.305 ms, about 1.04x.
- Forward at 32k rows: 0.293 ms -> 0.268 ms, about 1.09x.
- Forward+backward at 32k rows: 0.572 ms -> 0.549 ms, about 1.04x.
- Peak allocator memory unchanged. This is a small real kernel win, not the OOM fix.

### HISA / NVFP4 IndexCache

Source inspected:

- `/tmp/optimization-playground/docs/developer_guide/kernel_results/nvfp4_hisa_indexcache.md`
- `/tmp/optimization-playground/docs/developer_guide/kernel_results/nvfp4_indexcache_dequant.md`
- `/tmp/optimization-playground/python/sglang/srt/layers/attention/nsa/hisa_tilelang_kernels/hisa.py`
- `/tmp/optimization-playground/python/sglang/jit_kernel/nvfp4_indexer.py`

Megatron files changed:

- `examples/sft/run_sft_deepseek_nvfp4.sh`
- `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
- `artifacts/hisa_selector_backend_bench.py`

Decision:

- Use `MEGATRON_HISA_SELECTOR_BACKEND=deepgemm` by default.
- Keep exact BMM available by overriding `MEGATRON_HISA_SELECTOR_BACKEND=bmm`; when using BMM, `MEGATRON_HISA_BMM_CUBLASDX_REFINE=1` is the preferred exact refine path.

Why:

- OP's fastest HISA path is an SGLang paged/packed TileLang/DeepGEMM setup. The local `deepgemm` backend is the closest active Megatron path to that DeepSeek-style selector.
- This intentionally validates against the FP4-Q HISA score oracle, not the old exact BMM selector. That is now the intended semantic target.
- BMM+CuBLASDx refine remains the safe exact fallback.

Validation:

- `test_hisa_deepgemm_selector_backend_matches_fp4_oracle`
- `test_hisa_deepgemm_selector_production_config_scores_match_fp4_oracle`
- `test_hisa_bmm_dense_cublasdx_refine_matches_bmm`
- `test_hisa_bmm_dense_cublasdx_refine_matches_bmm_production_block`
- Result: passed.

Benchmarks at `sk=16384`, `topk=512`, `heads=64`, `head_dim=128`, `row_chunk=512`:

- rows 512:
  - `bmm`: 9.55 ms, 0.160 GiB peak
  - `bmm_cublasdx_refine`: 6.53 ms, 0.094 GiB peak, exact set match
- rows 1024:
  - `bmm`: 13.95 ms, 0.179 GiB peak
  - `bmm_cublasdx_refine`: 12.57 ms, 0.116 GiB peak, exact set match
- rows 2048:
  - `bmm`: 29.21 ms, 0.209 GiB peak
  - `bmm_cublasdx_refine`: 24.61 ms, 0.151 GiB peak, 0.9995 set match due likely near-tie behavior
- DeepGEMM was 2.1-8.0 ms across these rows but is a semantics tradeoff because it scores FP4-Q, not the exact current selector.

## Pipeline Schedule Options From `nvfp4-indexer`

`origin/nvfp4-indexer` is already an ancestor of current `flashtraining`; its pipeline schedule commits are present locally.

Available selectors:

- `auto`: current behavior, which resolves to non-interleaved or interleaved 1F1B depending on VPP.
- `interleaved_1f1b`: current VPP-style production path.
- `gpipe_fill_drain`: correctness baseline; likely poor for our memory because it runs all forwards before draining backwards.
- `zero_bubble`: non-virtual ZB; currently rejects fine-grained activation offload, CPU offload, Transformer Engine, overlapped grad reduce, overlapped param gather, and MoE EP overlap. Not viable for the current stack without a large compatibility pass.
- `zero_bubble_v`: V-shaped ZB; same compatibility blockers as `zero_bubble`, and requires exactly two virtual stages.
- `dualpipe_v`: conservative sequential DeepSeek-style V topology. It requires exactly two virtual stages and training-only runs with untied embeddings. It rejects P2P overlap and MoE EP overlap, but does not have the same TE/offload rejection as ZB in the current code.

Most plausible later probe:

```bash
PIPELINE_PARALLEL_SCHEDULE=dualpipe_v \
NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK=2 \
ENABLE_VPP=0 \
EVAL_INTERVAL=0
```

This should be treated as a schedule probe, not a default flip. The static verifier for PP=4/VP=2 reports a sequential V topology with entry/loss on rank 0 and bridge on rank 3.

## What Was Not Ported

- OP GatedNorm: active Megatron training already has a CuTe GatedNorm forward and training-specific backward/saved tensor behavior. OP result is forward/inference-focused, not a drop-in.
- OP G1 gate/o_proj fusion: inference-only FP4 projection fusion. Training needs gate backward semantics.
- OP layersplit / flashsampling / warpdecode: serving/inference paths, not active in SFT training.
- OP HISA DeepGEMM as launcher default: too much selector semantic drift without a deliberate quality decision.

## Artifacts

- `artifacts/higgs_fake_quant_bench.py`
- `artifacts/higgs_fake_quant_bench_current.jsonl`
- `artifacts/higgs_fake_quant_bench_current_large.jsonl`
- `artifacts/higgs_fake_quant_bench_baseline_HEAD_cuda.jsonl`
- `artifacts/higgs_fake_quant_bench_baseline_HEAD_cuda_large.jsonl`
- `artifacts/hisa_selector_backend_bench.py`
- `artifacts/hisa_selector_backend_bench_smoke.jsonl`
- `artifacts/hisa_selector_backend_bench_sk16384.jsonl`
- `artifacts/hisa_selector_backend_bench_bmm_variants_sk16384.jsonl`

Do not use `artifacts/higgs_fake_quant_bench_baseline_HEAD.jsonl` as a baseline; that run silently used Python fallback because `ninja` was not on `PATH`.
