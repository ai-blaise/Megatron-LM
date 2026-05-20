# Corsaire-1 Training Progress

Last updated: 2026-05-20 12:31 UTC

## Current Objective

Get the 2-node / 16xB200 DeepSeek V3.2 REAP/NVFP4 SFT run stable and fast enough for the real `corsaire-1-research-preview` training run. Current practical target is still to maximize tokens trained inside the remaining wall-clock budget, with quality-sensitive settings preserved where possible.

## Last Run Outcome

- Session / W&B name: `corsaire-1-research-preview`
- Launcher: `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
- Shape:
  - `TP=4`
  - `PP=4`
  - `CP=1`
  - `DP=1`
  - `EP=4`
  - `ETP=1`
  - `VPP=on`
  - `MBS=4`
  - `GBS=16`
  - `grad_accum=4`
  - `seq_len=16384`
  - `DSA_INDEXER_TOPK=512`
- VPP layout:
  - `Et*5|t*4|t*4|t*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*4|t*3|t*3L`
  - PP layer balance: PP0=16+embed, PP1=16, PP2=15, PP3=14+loss.
- StreamBP / chunking:
  - `USE_STREAMBP=1`
  - local sequence under sequence parallel: `4096`
  - layer StreamBP chunk: `2048`
  - DSA chunk: `2048`
  - MoE forward chunks: `1`
  - MoE backward MLP chunks: `4`
  - MoE attention backward chunk: `2048`
- Activation/offload:
  - activation offload enabled for `expert_fc1 core_attn attn_proj qkv_linear moe_act attn_norm mlp_norm mlp_residual moe_shared`
  - temp offload enabled for `mlp_residual moe_shared`
  - activation ECO enabled, recompute-only, TE backend, BF16 correction.

## Latest Observations

- The MBS=4/GBS=16 run crashed before completing step 1.
- It was not an OOM.
- Root cause was node0 rank3 / local_rank3, during StreamBP MoE split replay:
  - `streambp.py:1951 -> streambp.py:1771 -> transformer_layer.py:928 -> moe_layer.py:531 -> moe_layer.py:419 -> temp_activation_offload.py:63`
  - Failure call was `maybe_temp_cpu_reload(shared_expert_output) -> tensor.to(device, non_blocking=...)`.
  - Error: `torch.AcceleratorError: CUDA error: unspecified launch failure`.
  - Immediately before the traceback, logs showed:
    - `DeepEP timeout check failed: rank = 3, thread = 0, value = 1024`
    - `DeepEP timeout check failed: rank = 3, thread = 1, value = 1024`
    - `DeepEP timeout check failed: rank = 3, thread = 2, value = 1024`
    - `DeepEP timeout check failed: rank = 3, thread = 3, value = 0`
  - Node1 later showed TCPStore broken-pipe and NCCL remote-process-exited errors only because node0 had already died.
- MBS=4 substantially improved memory and backward speed versus MBS=8:
  - It reached backward at ~530s after step start instead of ~1385s.
  - PP2/PP3 early backward VMBs were ~80-170s instead of ~400-800s.
  - Node1 stayed around ~120-137G used with ~45-63G free before node0 died.
- Bottleneck/failure shifted to node0 PP0/PP1:
  - PP0 bwd `vmb=0` took ~687s.
  - PP1 bwd `vmb=0` took ~407s.
  - PP1 bwd `vmb=2` raised PP1 to ~124.3G allocated / ~137.3G reserved.
  - Node0 GPUs 4-7 reached ~142G used with ~40G free before the crash.

## Previous MBS=8/GBS=32 Outcome

- The MBS=8/GBS=32 run crashed before completing step 1.
- Root cause was node1 PP2, ranks 8-11, in StreamBP attention backward into DSA reentrant backward:
  - `streambp.py:1951 -> streambp.py:1831 -> streambp.py:220 -> streambp.py:173 -> dsa_triton.py:4135 -> tensor_parallel/random.py:845`
  - rank11 was first observed failure.
  - rank11 tried to allocate 1.62 GiB on GPU3 with only ~48 MiB free.
  - ranks 8/9 tried 832 MiB with <1 GiB free.
  - PyTorch allocated memory was ~165-166 GiB and reserved-but-unallocated was ~6.8-7.5 GiB.
- Node0 did not cause the crash. Node0 later hit NCCL watchdog/abort because node1 had already OOMed.
- The run had 4 real microbatches and 16 virtual microbatches.
- It reached backward at ~23 minutes after step start.
- Backward entry memory was still healthy:
  - PP0: ~77.8G allocated, ~97.5G reserved.
  - PP1: ~91.3G allocated, ~110.7G reserved.
  - PP3: ~81.7G allocated, ~101.2G reserved.
- Deeper backward was extremely slow:
  - PP3 bwd `vmb=0` took ~411s.
  - PP3 bwd `vmb=1` took ~590s.
  - PP3 bwd `vmb=2` took ~591s.
  - PP3 bwd `vmb=3` took ~383s.
  - PP3 bwd `vmb=4` took ~1863s and raised PP3 to ~144.7G allocated / ~162.8G reserved.
  - PP2 bwd `vmb=0` took ~609s.
  - PP2 bwd `vmb=1` took ~600s.
  - PP2 bwd `vmb=2` took ~807s and raised PP2 to ~152.7G allocated / ~168.6G reserved.
  - PP2 started `bwd vmb=3` at ~152.3G allocated / ~168.6G reserved, then OOMed shortly after.
- Current read from MBS=8: that shape is neither memory-safe nor time-viable without deeper lifecycle fixes. The dominant failure path was PP2 DSA backward under StreamBP replay, with PP3 also showing pathological slow backward VMBs.

## Recent Probe Results

- `MBS=32, GBS=32`
  - Invalid with VPP on.
  - Reason: VPP interleaved scheduler requires `microbatch_group_size_per_vp_stage >= PP` and `<= num_microbatches`; with `DP=1`, `GBS/MBS=1`, so `num_microbatches=1`, while PP=4.
- `MBS=32, GBS=128`
  - Legal with VPP on because `GBS/MBS=4`.
  - Failed during PP0 forward before useful backward data.
  - Failure stack hit MoE forward:
    - `StreamBP -> _forward_mlp -> moe_layer.postprocess -> maybe_temp_cpu_reload(shared_expert_output)`
  - GPU0 on node0 spiked near the ceiling (~171G used, ~11G free).
  - Takeaway: MBS=32 is too large with the current MoE shared-expert temp offload/reload path.
- Earlier `MBS=4, GBS=16`
  - Looked stable in early forward and had very large memory headroom.
  - User requested scaling curiosity probes before it reached backward.

## Important Fixes Already Made

- Fixed StreamBP chunk default for sequence parallel:
  - Old hard default `8192` exceeded local sequence length `4096` and could silently disable useful chunking.
  - Current default computes local sequence and uses chunk `2048`.
- Fixed activation offload forced-release corruption:
  - `untyped_storage().resize_(0)` is skipped for view tensors (`tensor._base is not None`).
  - This prevents StreamBP chunk views from zeroing the full hidden-state base tensor.
- Rebalanced PP layout away from overloaded PP2/PP3:
  - Current gentler balance is PP0=16+embed, PP1=16, PP2=15, PP3=14+loss.

## Current Decision Gate

Do not relaunch the same MBS=8 shape unchanged. It failed at PP2 DSA backward and would be far too slow even if it completed.

Do not relaunch MBS=4 unchanged either. It is much closer, but currently dies in temp offload reload of `shared_expert_output` during StreamBP MoE replay on node0 rank3.

Next useful actions should target the actual combined hot path, not isolated single-tensor edits:

1. First fix or disable the fragile temp offload path for `moe_shared` during StreamBP replay:
   - The crash is at `maybe_temp_cpu_reload(shared_expert_output)`.
   - Options to discuss before relaunch:
     - remove `moe_shared` from `TEMP_ACTIVATION_OFFLOAD_MODULES` only;
     - make `maybe_temp_cpu_reload` synchronous/safe for StreamBP replay;
     - add stream/event synchronization around the host-to-device reload and shared expert combine.
2. Revisit PP layout with the new evidence:
   - MBS=4 moved the fatal issue from node1 PP2 OOM to node0 PP0/PP1 replay/offload fragility.
   - PP0/PP1 are now throughput bottlenecks.
3. Inspect StreamBP attention backward plus DSA reentrant backward as one lifecycle:
   - avoid retaining/reloading tensors across the nested reentrant graph longer than needed;
   - confirm offload finalizers clear both forward and backward caches as soon as the owning VMB is done;
   - identify which tensors survive from PP2 bwd `vmb=2` into `vmb=3`.
4. Next candidate launch shape, when we decide to run:
   - `MBS=4`
   - `GBS=16`
   - This keeps `GBS/MBS=4`, so VPP still has 4 real microbatches / 16 virtual microbatches, but cuts per-microbatch activation pressure roughly in half versus `MBS=8/GBS=32`.
   - It also halves tokens per optimizer step, so wall-clock token throughput only improves if the step becomes more than 2x faster or, at minimum, becomes stable enough to run.
   - MBS=4/GBS=16 had early headroom but was not run to full backward after the newer fixes.
   - MBS=8/GBS=32 is too close to the ceiling in PP2.
5. Avoid raising StreamBP chunk count as the first response unless there is no other path.

## Constraints / Preferences

- Do not disable VPP for real throughput runs.
- Do not turn off indexer training unless explicitly chosen as a quality tradeoff.
- Avoid increasing StreamBP chunk count as the first lever because it heavily hurts step time.
- Prefer fixing tensor lifetimes/offload boundaries and kernel/system paths.
- Keep run name `corsaire-1-research-preview`.
- Use `uv run --no-sync` for Python/tests.
- Use both node logs when diagnosing:
  - Node0: `/home/sjpat/logs/corsaire-1-research-preview_node0.log`
  - Node1: `/home/sjpat/logs/corsaire-1-research-preview_node1.log`
