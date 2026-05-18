# Flashtraining Stability Notes

## Objective

Keep the `corsaire-1-research-preview` training path running without changing the launch shape or sacrificing functionality/model quality. Preserve PP=4/VPP usage, trainable DSA indexer, HISA/IndexCache, StreamBP, activation ECO, HIGGS/SpinQuant, and the existing MBS/GBS unless explicitly directed otherwise.

## Current Launch Shape

- Nodes/GPUs: 2 nodes, 8 GPUs each
- Parallelism: TP=4, PP=4, CP=1, DP=1, EP=4, ETP=1
- Batch: MBS=4, GBS=128
- StreamBP: enabled, MoE MLP chunks=4, MoE backward chunks=16
  - MoE backward chunks now default to 16 after repeated TE backward OOMs.
  - Attention replay chunk default: 4096; backward MLP replay intersects this
    down to 2048-token ranges when using 16 chunks.
- DSA: trainable indexer enabled, topk=1024, fused/split-QK path enabled
- Checkpointing: ZCC enabled, retain latest=1

## Promoted Fixes

- DSA teacher score scratch disabled by default:
  - `MEGATRON_DSA_TEACHER_SCORE_SCRATCH=0`
  - Reason: removed a large retained teacher-score tensor and moved past the split-QK DSA OOM.
- Split-QK DSA query grad scratch uses BF16/FP16 when BF16 grad atomics are enabled:
  - Reason: avoids avoidable FP32 query-grad scratch while preserving fused DSA semantics.
- HISA compact candidate refine is enabled:
  - `MEGATRON_HISA_CANDIDATE_MAX_TEMP_MB=128`
  - `MEGATRON_HISA_COMPACT_CANDIDATE_TOPK=1`
  - Reason: bounds temporary candidate refine memory. The cublasdx refine path exists but remains default-off because BMM was faster in microbench.
- StreamBP MoE replay trims CUDA cache before the autograd backward boundary:
  - Reason: previous run moved past TE grouped-linear wgrad OOM after adding trim immediately before `torch.autograd.backward(...)`.
- MoE FP8/NVFP4 padding now trims CUDA cache before TE `Fp8Padding` allocates the padded expert input:
  - Reason: latest crash was `Fp8Padding.forward` trying to allocate 452 MiB with only 406 MiB free while 4.86 GiB was reserved-but-unallocated.
- Activation-ECO TE hooks now eagerly stash corrections when Megatron
  `main_grad` exists:
  - Files: `megatron/core/quantization/nvfp4_act_eco/te_hook.py`
  - Reason: StreamBP MoE replay runs short-lived grad graphs. Waiting until
    parameter weight hooks to consume captured activation-ECO entries can keep
    replay activations live into TE grouped-linear backward. In the real DDP
    path, `main_grad` exists, so the output-gradient hook can compute and stash
    the same correction immediately, then release the captured activation before
    TE's heavier backward allocations. Non-DDP/test paths keep the old
    weight-hook behavior.
- StreamBP attention replay chunks now default to 4096:
  - Files:
    - `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
    - `examples/sft/run_sft_deepseek_nvfp4.sh`
  - Reason: with 32k sequence length and `--streambp-moe-mlp-backward-chunks
    8`, each MoE MLP replay chunk is 4096 tokens. The previous 2048 attention
    replay chunk forced each MLP replay chunk to keep two attention outputs
    plus a copied/stitched post-attention tensor live. Aligning to 4096 lets
    the existing no-copy fast path return the exact attention tensor, reducing
    replay memory and halving attention replay chunk count for the MoE path.
- MoE router-side padding for quantization is enabled by default:
  - `MOE_ROUTER_PADDING_FOR_QUANTIZATION=1`
  - Reason: the latest run got past TE grouped-linear output allocation and
    failed inside TE `Fp8Unpadding.forward`. Router-side padding aligns expert
    token counts before dispatch, so TE grouped MLP padding/unpadding can
    become a no-op for aligned experts instead of allocating a second dense
    output at the replay peak.
- CUDA allocator garbage collection threshold reduced from 0.8 to 0.6:
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6`
  - Reason: after the router-padding fix, the remaining failure moved into TE
    grouped-linear backward with only tens of MiB free while PyTorch still had
    ~11.8 GiB reserved-but-unallocated. A lower GC threshold should return
    cached blocks earlier and is preferable to increasing StreamBP replay
    chunks because it does not change math or replay granularity.
- StreamBP MoE replay now synchronizes and empties CUDA cache immediately
  before the per-chunk autograd backward call:
  - File: `megatron/core/transformer/streambp.py`
  - Env gate: `MEGATRON_STREAMBP_MOE_REPLAY_TRIM_SYNC=1`
  - Reason: the remaining OOM happens inside TE grouped-linear backward after
    the replay forward has queued work. Synchronizing before cache trim should
    make event-pending cached blocks reclaimable before TE allocates its
    backward quantization and weight-grad staging buffers.
- StreamBP MoE MLP backward chunks increased from 8 to 16:
  - Forward MoE chunking remains 4.
  - Reason: after router-side padding and allocator/cache fixes, the remaining
    failure is a tiny allocation inside TE grouped-linear backward for the
    4096-token replay chunk. Sixteen backward chunks reduce the per-chunk MLP
    replay peak to 2048 tokens while keeping MBS/GBS, PP layout, DSA indexer
    training, HISA, activation ECO, and quantization enabled.
- StreamBP replay now uses TE's saved-quantized-input path inside replay:
  - `MEGATRON_STREAMBP_REPLAY_SAVE_QUANTIZED_TE_INPUTS=1`
  - `MEGATRON_STREAMBP_RELEASE_MLP_INPUT_BEFORE_BACKWARD=1`
  - Reason: expert `fc1` is configured to save original BF16 inputs because
    normal fine-grained offload wants that path. StreamBP suppresses the
    global offload queue during replay, so the original-input setting only
    defers TE's `split_quantize` allocation into backward. The latest OOM was
    exactly that backward `split_quantize` allocation. During replay we can use
    TE's default saved quantized columnwise input path, restore the module
    attribute immediately afterwards, and drop the local BF16 MLP input before
    backward.
- TE GroupedLinear expert wgrad now accumulates directly into `main_grad`:
  - `MEGATRON_TE_GROUPED_LINEAR_FUSE_WGRAD_ACCUM=1`
  - Global `--no-gradient-accumulation-fusion` remains in place because the
    non-TE tensor-parallel linear path would require the missing Apex
    `fused_weight_gradient_mlp_cuda` extension.
  - Reason: after removing the backward `split_quantize` OOM, the next failure
    moved to TE `grouped_linear.py:423`, where TE allocates per-expert BF16
    temporary wgrad tensors when `fuse_wgrad_accumulation=False`. Expert TE
    grouped linear can use TE's own direct accumulation into Megatron
    `main_grad`, avoiding that 56 MiB scratch allocation at the replay peak.
  - DDP now consumes pending activation-ECO corrections even when TE has
    already marked `grad_added_to_main_grad=True`, so activation ECO remains
    active on this fused wgrad path.

## Reverted / Not Promoted

- PP/VPP layout rebalance was reverted.
  - The altered layout confounded run comparisons and was not the right lever for the current OOM investigation.
  - Restored default layout:
    `Et*5|t*5|t*4|t|t*5|t*5|t*5|t|t*5|t*5|t*4|t|t*5|t*5|t*4|tL`

## Latest Failure Before This Note

- Run: `corsaire-1-research-preview`
- Step: step 1, before first completed update
- Layout: restored default PP layout
- Failure: node1 ranks 8/10, GPUs 0/2 OOM
- Stack: StreamBP MoE replay backward -> `_moe_chunk_attention_full_mlp_backward`
  -> `layer._forward_mlp` -> MoE routed experts -> TE `GroupedLinear.forward`
  -> `torch.empty`
- Allocations: 534 MiB on rank8/GPU0 and 362 MiB on rank10/GPU2
- Free at failure: 126 MiB on GPU0 and 322 MiB on GPU2
- PyTorch reserved-but-unallocated: ~11.5 GiB on both failed ranks
- Interpretation: the PP rebalance was not the sole problem. The current
  pressure is StreamBP MoE replay MLP re-forward on pp_rank=2, with real
  live memory pressure plus allocator fragmentation at the TE grouped-linear
  output allocation.

## Next Gate

Fix the StreamBP MoE replay memory pressure without changing launch shape or
disabling functionality. Success criterion for this gate is one completed
training step with no OOM and all major components still enabled.

## Validation Before Relaunch

- `uv run --no-sync python -m py_compile` passed for:
  - `megatron/core/transformer/moe/experts.py`
  - `megatron/core/transformer/streambp.py`
  - `megatron/core/transformer/experimental_attention_variant/dsa_triton.py`
  - `megatron/core/quantization/indexcache/hisa.py`
- `bash -n` passed for:
  - `examples/sft/run_sft_deepseek_nvfp4.sh`
  - `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`

## Validation After Activation-ECO Early-Stash Patch

- `uv run --no-sync pytest tests/unit_tests/transformer/test_streambp.py -k 'act_eco_grouped_hook or act_eco_te_hook' -q`
  - Result: 8 passed, 33 deselected
- `uv run --no-sync python -m py_compile` passed for:
  - `megatron/core/quantization/nvfp4_act_eco/te_hook.py`
  - `megatron/core/transformer/streambp.py`
  - `megatron/core/transformer/moe/experts.py`

## Validation After StreamBP 4096 Alignment

- `uv run --no-sync pytest tests/unit_tests/transformer/test_streambp.py -k 'streambp or act_eco' -q`
  - Result: 39 passed, 2 skipped
- `uv run --no-sync python -m py_compile` passed for:
  - `megatron/core/transformer/streambp.py`
  - `megatron/core/quantization/nvfp4_act_eco/te_hook.py`
  - `megatron/core/transformer/moe/experts.py`
- `bash -n` passed for:
  - `examples/sft/run_sft_deepseek_nvfp4.sh`
  - `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`

## Validation After MoE Router Quantization Padding

- `uv run --no-sync pytest tests/unit_tests/transformer/test_streambp.py -k 'moe_seq_aux_loss_split_mlp_replay or moe_seq_aux_loss_asymmetric_mlp_replay or act_eco_grouped_hook' -q`
  - Result: 5 passed
- `uv run --no-sync pytest tests/unit_tests/transformer/moe/test_token_dispatcher.py -k 'router_padding' -q`
  - Result: 6 skipped because DeepEP/HybridEP are not available in this environment
- `bash -n` passed for:
  - `examples/sft/run_sft_deepseek_nvfp4.sh`
  - `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`

## Validation After Synchronized Replay Trim

- `uv run --no-sync pytest tests/unit_tests/transformer/test_streambp.py -k 'moe_seq_aux_loss_split_mlp_replay or moe_seq_aux_loss_asymmetric_mlp_replay or act_eco_grouped_hook' -q`
  - Result: 5 passed
- `uv run --no-sync python -m py_compile megatron/core/transformer/streambp.py`
  - Result: passed
- `bash -n` passed for:
  - `examples/sft/run_sft_deepseek_nvfp4.sh`
  - `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`

## Latest Failure After Activation-ECO Early-Stash Patch

- Run: `corsaire-1-research-preview`
- Layout: restored default PP layout
- StreamBP settings in command:
  - `--streambp-chunk-size 2048`
  - `--streambp-moe-mlp-backward-chunks 8`
- Failure: node1 pp_rank=2 ranks 9/10/11 OOM
- Stack: StreamBP MoE replay backward -> `_moe_chunk_attention_full_mlp_backward`
  -> `layer._forward_mlp` -> MoE routed experts -> TE `GroupedLinear.forward`
- Allocation sites:
  - rank10/GPU2: TE `GroupedLinear.forward` `torch.empty`, 334 MiB request,
    124 MiB free
  - rank11/GPU3: TE `general_grouped_gemm`, 56 MiB request, 26 MiB free
  - rank9/GPU1: TE `GroupedLinear.forward` `torch.empty`, 576 MiB request,
    486 MiB free
- Interpretation:
  - The eager activation-ECO stash did not break correctness tests and likely
    reduces some retained replay state, but it is not enough by itself.
  - The current failure is still the TE grouped MLP allocation inside
    StreamBP MoE MLP replay, after fused LCE has run.
  - Since MoE backward chunks are already 8, the next preferred fix is to
    remove avoidable replay/copy pressure before increasing chunk counts
    further.

## Failure After StreamBP 4096 Alignment

- Run: `corsaire-1-research-preview`
- StreamBP settings in command:
  - `--streambp-chunk-size 4096`
  - `--streambp-moe-mlp-backward-chunks 8`
- Failure: node1 rank8/GPU0 OOM after fused LCE, inside StreamBP MoE replay.
- Stack: StreamBP MoE replay backward -> `_moe_chunk_attention_full_mlp_backward`
  -> `layer._forward_mlp` -> MoE routed experts -> TE `Fp8Unpadding.forward`.
- Allocation: 582 MiB requested with 416 MiB free.
- Interpretation:
  - The 4096 alignment moved the failure past the earlier TE grouped-linear
    matmul/output allocations, but the replay peak now fails at TE
    unpadding's dense output materialization.
  - Router-side quantization padding is the preferred next fix because it
    should remove the explicit TE unpadding allocation rather than further
    chunking StreamBP.

## Failure After Router Quantization Padding

- Run: `corsaire-1-research-preview`
- Config confirmed: `moe_router_padding_for_quantization=True`
- Progress: reached fused LCE and moved beyond the previous TE unpadding
  allocation failure.
- Failure: node1 pp_rank=2 ranks 8/9 OOM inside TE grouped-linear backward.
- Stack:
  - rank9: `torch.autograd.backward(chunk_output, ...)` ->
    `transformer_engine/pytorch/module/grouped_linear.py:441` ->
    `tex.split_quantize(...)`, 178 MiB request with 138 MiB free.
  - rank8: `grouped_linear.py:423` weight-grad staging allocation, 56 MiB
    request with 42 MiB free.
- Allocator state: PyTorch reported ~11.8 GiB reserved-but-unallocated on the
  failed ranks, so the next fix is allocator GC tuning before adding more
  StreamBP chunks.

## Failure After Allocator GC Tuning

- Run: `corsaire-1-research-preview`
- Failure: same TE grouped-linear backward allocation class as above.
- Allocation: 56 MiB requests with only 42-58 MiB free.
- Allocator state: still ~12 GiB reserved-but-unallocated.
- Interpretation: lowering allocator GC threshold alone was not enough.
  The next fix is a synchronized forced cache trim at the exact StreamBP MoE
  replay autograd boundary.

## Failure After Synchronized Replay Trim

- Run: `corsaire-1-research-preview`
- Failure: same TE grouped-linear backward allocation class after fused LCE.
- Allocation: 178 MiB / 56 MiB requests with only 118 MiB / 22 MiB free.
- Interpretation: allocator cleanup is insufficient. The remaining reliable
  lever is reducing the MoE MLP replay chunk peak from 4096 to 2048 tokens by
  increasing `STREAMBP_MOE_MLP_BACKWARD_CHUNKS` to 16.

## Failure After StreamBP MoE Backward Chunks 16

- Run: `corsaire-1-research-preview`
- Shape confirmed: TP=4, PP=4, CP=1, DP=1, EP=4, ETP=1, MBS=4, GBS=128,
  VPP=4.
- Layout confirmed:
  `Et*5|t*5|t*4|t|t*5|t*5|t*5|t|t*5|t*5|t*4|t|t*5|t*5|t*4|tL`
- StreamBP settings confirmed:
  - `streambp_chunk_size=4096`
  - `streambp_moe_mlp_backward_chunks=16`
- Failure: node1 rank10/GPU2 OOM inside TE grouped-linear backward.
- Stack:
  - `torch.autograd.backward(chunk_output, ...)` ->
    `transformer_engine/pytorch/module/grouped_linear.py:441` ->
    `tex.split_quantize(inp_view, ctx.m_splits, ctx.input_quantizers)`.
- Allocation: 96 MiB request with 48.81 MiB free.
- Allocator state: 161.15 GiB allocated by PyTorch, 11.10 GiB reserved but
  unallocated.
- Interpretation:
  - The OOM is now a very narrow TE backward scratch allocation, not a PP shape
    issue and not missing router padding.
  - Increasing chunks again would likely fit, but it would sacrifice
    throughput. The preferred next fix is to avoid this backward-time
    split-quantize allocation by using TE's saved quantized input path inside
    StreamBP replay.

## Validation After Replay Saved-Quantized-Input Patch

- `uv run --no-sync pytest tests/unit_tests/transformer/test_streambp.py -k 'replay_quantized_te_input_context or moe_seq_aux_loss_split_mlp_replay or moe_seq_aux_loss_asymmetric_mlp_replay or act_eco_grouped_hook' -q`
  - Result: 7 passed
- `uv run --no-sync python -m py_compile` passed for:
  - `megatron/core/transformer/streambp.py`
  - `megatron/core/transformer/moe/experts.py`
  - `megatron/core/quantization/nvfp4_act_eco/te_hook.py`
- `bash -n` passed for:
  - `examples/sft/run_sft_deepseek_nvfp4.sh`
  - `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`

## Failure After Replay Saved-Quantized-Input Patch

- Run: `corsaire-1-research-preview`
- Progress: reached `StreamBP fused LCE active`; survived the previous
  `grouped_linear.py:441 tex.split_quantize(...)` OOM.
- Failure: node1 rank8/GPU0 OOM inside TE grouped-linear backward.
- Stack:
  - `torch.autograd.backward(chunk_output, grad_chunk)` ->
    `transformer_engine/pytorch/module/grouped_linear.py:423` ->
    `torch.empty(w.size(), dtype=ctx.activation_dtype, device=ctx.device)`.
- Allocation: 56 MiB request with 18.81 MiB free.
- Allocator state: 160.72 GiB allocated by PyTorch, 11.30 GiB reserved but
  unallocated.
- Interpretation:
  - The saved-quantized-input patch removed the prior split-quantize scratch
    failure.
  - The next peak is the temporary BF16 wgrad allocation TE creates when
    grouped-linear wgrad accumulation is not fused.
  - Preferred fix is expert-only TE grouped wgrad accumulation into existing
    `main_grad`, not another StreamBP chunk increase.

## Validation After Expert TE Grouped Wgrad Accumulation Patch

- `uv run --no-sync pytest tests/unit_tests/transformer/test_streambp.py -k 'te_grouped_wgrad_accum_env or replay_quantized_te_input_context or moe_seq_aux_loss_split_mlp_replay or moe_seq_aux_loss_asymmetric_mlp_replay or act_eco_grouped_hook' -q`
  - Result: 8 passed
- `uv run --no-sync python -m py_compile` passed for:
  - `megatron/core/transformer/streambp.py`
  - `megatron/core/distributed/distributed_data_parallel.py`
  - `megatron/core/extensions/transformer_engine.py`
- `bash -n` passed for:
  - `examples/sft/run_sft_deepseek_nvfp4.sh`
  - `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`

## Failure After Expert TE Grouped Wgrad Accumulation Patch

- Run: `corsaire-1-research-preview`
- Shape confirmed unchanged:
  TP=4, PP=4, CP=1, DP=1, EP=4, ETP=1, MBS=4, GBS=128, VPP=4.
- Progress: reached `StreamBP fused LCE active`.
- Failure: node1 rank11/GPU3 OOM inside DSA split-QK Triton backward.
- Stack:
  - `streambp.py` `_moe_chunk_attention_full_mlp_backward`
  - `torch.autograd.backward(chunk_output, grad_chunk)`
  - `dsa_triton.py:3300`, `grad_value = torch.zeros(...)`
- Allocation: 832 MiB request with 412.81 MiB free.
- Allocator state: 160.72 GiB allocated by PyTorch, 11.17 GiB reserved but
  unallocated.
- Interpretation:
  - The replay saved-quantized-input and expert grouped-wgrad patches moved the
    failure past both TE split-quantize and TE temporary wgrad allocation.
  - The next peak is DSA backward's dense K/V gradient return allocation.
  - Preferred fix is a DSA-backward-local conditional allocator trim before the
    dense grad buffers are created. This targets cached-but-unallocated memory
    at the exact failure boundary without changing PP shape, StreamBP chunks,
    trainable indexer semantics, or MBS/GBS.

## Validation After DSA Backward Local Trim Patch

- `uv run --no-sync python -m py_compile megatron/core/transformer/experimental_attention_variant/dsa_triton.py`
  - Result: passed
- `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
  - Result: passed
- `uv run --no-sync pytest tests/unit_tests/transformer/experimental_attention_variant/test_attention_variant_dsa.py -q`
  - Result: 48 DSA tests passed before the suite stopped at an existing
    single-process TP=2 test setup failure (`world_size (1) is not divisible by
    2`), unrelated to the trim patch.

## Failure After DSA Backward Local Trim Patch

- Run: `corsaire-1-research-preview`
- Progress: reached fused LCE and passed the previous DSA `grad_value` OOM.
- Failure: node1 rank8/GPU0 OOM in activation-ECO correction:
  - `nvfp4_act_eco/te_hook.py`
  - `activation_eco_bias_correction(...)`
  - `reference.py:157`, `flat_grad.T @ e_x`
- Allocation: 112 MiB request with 42.81 MiB free.
- Allocator state: 160.91 GiB allocated by PyTorch, 11.09 GiB reserved but
  unallocated.
- Interpretation:
  - The DSA-local trim moved the failure past dense DSA grad allocation.
  - The next peak is activation-ECO materializing a full correction tensor.
  - Preferred fix is to accumulate `dy.T @ (x_pre - q_x)` directly into
    `param.main_grad` with `addmm_` when Megatron main-grad buffers exist,
    preserving ECO math while avoiding the dense correction allocation.

## Validation After Activation-ECO Direct Main-Grad Addmm Patch

- `uv run --no-sync python -m py_compile megatron/core/quantization/nvfp4_act_eco/te_hook.py tests/unit_tests/transformer/test_streambp.py`
  - Result: passed
- `uv run --no-sync pytest tests/unit_tests/quantization/test_nvfp4_act_eco_correctness.py tests/unit_tests/quantization/test_nvfp4_act_eco_compose.py -q`
  - Result: 14 passed
- `uv run --no-sync pytest tests/unit_tests/transformer/test_streambp.py -k 'act_eco_grouped_hook or te_grouped_wgrad_accum_env or replay_quantized_te_input_context' -q`
  - Result: 6 passed

## Failure After Activation-ECO Direct Main-Grad Addmm Patch

- Run: `corsaire-1-research-preview`
- Progress: reached fused LCE and passed the prior activation-ECO correction
  allocation site.
- Failure: node1 rank8/GPU0 OOM during replayed MoE MLP forward combine:
  - `moe_layer.py` -> `token_dispatcher.combine_preprocess`
  - `moe_utils.sort_chunks_by_idxs`
  - TE Triton permutation `sort_chunks_by_map`
  - allocation of sorted output buffer `(num_tokens, hidden_size)`
- Allocation: 378 MiB request with 272.81 MiB free.
- Allocator state: 160.49 GiB allocated by PyTorch, 11.28 GiB reserved but
  unallocated.
- Interpretation:
  - The direct activation-ECO addmm removed the previous correction allocation.
  - The next boundary is TE's fused MoE sort output allocation.
  - Preferred fix is a conditional MoE-sort allocator trim immediately before
    fused TE sort chunks, preserving the fused TE path and avoiding chunk/shape
    degradation.

## Validation After MoE Sort Trim Patch

- `uv run --no-sync python -m py_compile megatron/core/transformer/moe/moe_utils.py megatron/core/quantization/nvfp4_act_eco/te_hook.py megatron/core/transformer/experimental_attention_variant/dsa_triton.py`
  - Result: passed
- `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
  - Result: passed
- `uv run --no-sync pytest tests/unit_tests/transformer/test_streambp.py -k 'act_eco_grouped_hook or te_grouped_wgrad_accum_env or replay_quantized_te_input_context' -q`
  - Result: 6 passed

## Failure After MoE Sort Trim Patch

- Run: `corsaire-1-research-preview`
- Progress: reached fused LCE and got past the MoE sort allocation boundary.
- Failure: node1 rank10/GPU2 OOM in DSA split-QK backward:
  - `dsa_triton.py`, `grad_key_nope = torch.zeros(...)`
- Allocation: 832 MiB request with 752.81 MiB free.
- Allocator state: 160.24 GiB allocated by PyTorch, 11.32 GiB reserved but
  unallocated.
- Interpretation:
  - The first DSA trim happened before query-grad buffers were allocated.
  - By the exact K/V dense-grad allocation boundary, free memory had fallen
    below the allocation size while cached memory remained available.
  - Preferred fix is to add additional DSA-local trims immediately before the
    K/V dense grad allocations, not change PP/MBS/GBS.

## Validation After Staged DSA Backward Trims

- `uv run --no-sync python -m py_compile megatron/core/transformer/experimental_attention_variant/dsa_triton.py megatron/core/transformer/moe/moe_utils.py`
  - Result: passed
- `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
  - Result: passed

## Failure After Staged DSA Backward Trims

- Run: `corsaire-1-research-preview`
- Failure returned to TE MoE sort output allocation:
  - `moe_utils.sort_chunks_by_idxs` -> TE Triton `sort_chunks_by_map`
  - 378 MiB request with 132.81 MiB free
  - 11.16 GiB reserved but unallocated
- Interpretation:
  - The first MoE sort trim was too conservative because it only considered the
    imminent output size plus a small safety margin.
  - At this memory pressure level, the sort path should follow the replay trim
    policy: if device free memory is below a large threshold and cached blocks
    exist, release them before the fused TE sort call.

## Validation After MoE Sort High-Water Trim

- `uv run --no-sync python -m py_compile megatron/core/transformer/moe/moe_utils.py megatron/core/transformer/experimental_attention_variant/dsa_triton.py`
  - Result: passed
- `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
  - Result: passed

## Failure After MoE Sort High-Water Trim

- Run: `corsaire-1-research-preview`
- Shape remained `TP=4 PP=4 CP=1 DP=1 EP=4 ETP=1`, `MBS=4`, `GBS=128`.
- Failure: node1 rank8/GPU0 OOM inside StreamBP MoE replay backward:
  - `streambp.py` -> `_moe_chunk_attention_full_mlp_backward`
  - `torch.autograd.backward(chunk_output, grad_chunk)`
  - 64 MiB request with only 76.81 MiB device-free at the failure point
  - 160.69 GiB allocated by PyTorch and 11.27 GiB reserved but unallocated
- Interpretation:
  - The earlier targeted fixes moved the failure past TE split-quantize,
    grouped-linear wgrad, DSA dense grad, activation-ECO correction, and MoE
    sort allocations.
  - This remaining site is the one-shot replay autograd call holding both
    replayed MoE MLP internals and replayed attention graph state at the same
    time.
  - Changing PP count is not the right lever; the next fix should lower live
    tensors inside the replay graph without changing top-level launch shape.

## Patch: Split StreamBP MoE MLP And Attention Replay Backward

- Added `MEGATRON_STREAMBP_SPLIT_MOE_MLP_ATTENTION_BACKWARD=1` default.
- In the chunked MoE replay backward path, run the MLP backward against a
  detached leaf view of `post_attention`, then feed its input gradient into a
  second backward through the replayed attention graph.
- This preserves the exact chain rule but lets the MLP graph release before
  attention backward starts, reducing peak live memory at the latest OOM site.
- Legacy one-shot behavior remains available with
  `MEGATRON_STREAMBP_SPLIT_MOE_MLP_ATTENTION_BACKWARD=0`.

## Validation After Split StreamBP MoE Replay Backward

- `uv run --no-sync python -m py_compile megatron/core/transformer/streambp.py`
  - Result: passed
- `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
  - Result: passed
- `uv run --no-sync pytest tests/unit_tests/transformer/test_streambp.py -q`
  - Result: 42 passed, 2 skipped
- `uv run --no-sync pytest tests/unit_tests/quantization/test_nvfp4_act_eco_correctness.py tests/unit_tests/quantization/test_nvfp4_act_eco_compose.py -q`
  - Result: 14 passed

## Failure After Split StreamBP MoE Replay Backward

- Run: `corsaire-1-research-preview`
- Progress: reached the split attention-only backward, so the previous one-shot
  replay autograd OOM was avoided.
- Failure: node1 rank8/GPU0 OOM in DSA split-QK backward:
  - `streambp.py` second backward: `torch.autograd.backward(post_attention, attention_grad)`
  - `dsa_triton.py`: `grad_value = torch.zeros(...)`
  - 832 MiB request with 760.81 MiB device-free and 11.34 GiB reserved but
    unallocated.
- Interpretation:
  - The DSA trim hook did not synchronize before `empty_cache()`, unlike the
    stricter StreamBP replay trim.
  - The DSA backward also allocated the smaller K-grad buffers before the
    largest V-grad buffer, which increases fragmentation sensitivity at this
    edge.

## Patch: DSA Backward Sync Trim And Largest-First K/V Grad Allocation

- Added `MEGATRON_DSA_BACKWARD_TRIM_SYNC=1` default.
- DSA backward trims now synchronize before `torch.cuda.empty_cache()` when
  near OOM.
- DSA regular and split-QK backward now allocate the largest dense `grad_value`
  buffer before K-grad buffers.

## Validation After DSA Backward Trim Sync Patch

- `uv run --no-sync python -m py_compile megatron/core/transformer/experimental_attention_variant/dsa_triton.py megatron/core/transformer/streambp.py`
  - Result: passed
- `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
  - Result: passed
- `uv run --no-sync pytest tests/unit_tests/transformer/experimental_attention_variant/test_attention_variant_dsa.py -q`
  - Result: 48 passed before the known single-process TP=2 failure:
    `world_size (1) is not divisible by 2`
