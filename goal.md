# Flashtraining Stability Notes

## Objective

Keep the `corsaire-1-research-preview` training path running without changing the launch shape or sacrificing functionality/model quality. Preserve PP=4/VPP usage, trainable DSA indexer, HISA/IndexCache, StreamBP, activation ECO, HIGGS/SpinQuant, and the existing MBS/GBS unless explicitly directed otherwise.

## Current Launch Shape

- Nodes/GPUs: 2 nodes, 8 GPUs each
- Parallelism: TP=4, PP=4, CP=1, DP=1, EP=4, ETP=1
- Batch: MBS=4, GBS=128
- MoE dispatcher: flex dispatcher with DeepEP backend by default.
- StreamBP: enabled, chunk size=8192, logits chunk size=8192, MoE MLP forward
  chunks=1, MoE backward chunks=4.
  - Do not increase replay/chunk counts unless explicitly approved. The current
    optimization target is the StreamBP -> DSA -> TE reentrant backward memory
    lifecycle at the existing chunk shape.
- DSA: trainable indexer enabled, topk=1024, fused/split-QK path enabled
- Checkpointing: ZCC enabled, retain latest=1

## Promoted Fixes

- DeepEP fused MoE dispatch/combine is installed and wired by default:
  - Installed patched DeepEP `v1.2.1+9af0e0d` on both nodes in the uv env.
  - `examples/sft/run_sft_deepseek_nvfp4.sh` and
    `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh` now default
    `MOE_TOKEN_DISPATCHER_TYPE=flex` and
    `MOE_FLEX_DISPATCHER_BACKEND=deepep`.
  - Reason: the previous `alltoall` path separately materialized routing-map
    padding, hidden/prob permutation, all-to-all hidden, all-to-all probs,
    TP gather, expert sort, unsort, TP reduce-scatter, all-to-all combine, and
    unpermute. DeepEP moves the cross-rank dispatch/combine work onto the
    fused flex dispatcher path instead of paying that whole sequence.
  - Validation:
    - Both nodes import `megatron.core.transformer.moe.fused_a2a` with
      `HAVE_DEEP_EP=True`.
    - 8-GPU smoke passed:
      `TestFlexDispatcher::test_forward_backward[deepep-True-4-2]`.
  - Note: current public DeepEP `main`/EPv2 requires NCCL GIN APIs absent from
    our NCCL 2.28.9 headers. The compatible API for this Megatron wrapper is
    DeepEP v1.2.1, with the CUDA 13 CCCL include path patch.
- Forward MoE replay chunking reduced:
  - `STREAMBP_MOE_MLP_CHUNKS=1`
  - `STREAMBP_MOE_MLP_BACKWARD_CHUNKS=4`
  - Reason: forward memory had headroom while forward time was poor. Keeping
    the backward chunk count at 4 preserves the current memory guardrail while
    avoiding avoidable forward replay/dispatch overhead.

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
- StreamBP attention replay chunks were restored to 8192:
  - Files:
    - `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
    - `examples/sft/run_sft_deepseek_nvfp4.sh`
  - Reason: chunking smaller than 8192 was rejected because it directly slows
    the step. Memory fixes must come from better object lifetimes/fusion within
    the existing replay shape.
- Split-QK DSA reentrant backward now defers query-gradient materialization:
  - Env: `MEGATRON_DSA_SPLIT_QK_REENTRANT_DEFER_QUERY_GRADS=1`
  - Reason: the crash scope is StreamBP MoE replay backward -> DSA reentrant
    backward -> TE NVFP4 linear backward. DSA previously allocated full
    `grad_query_nope` and `grad_query_pe` before chunked K/V gradients were
    consumed by the upstream MLA KV-up projection. Those query-grad buffers do
    not feed the TE K/V projection backward, so they now materialize only after
    K/V reentrant backward has finished.
- Split-QK DSA reentrant backward now packs K-noPE and V gradients together:
  - Env: `MEGATRON_DSA_SPLIT_QK_REENTRANT_PACK_KV_GRAD=1`
  - Reason: the MLA KV-up projection produces packed `[K_noPE | V]`. The older
    reentrant path emitted separate key/value chunk grads and let autograd
    reconstruct dense packed grads at the TE boundary. The new path writes
    selected-attention K-noPE/V grads into one packed chunk and re-enters the
    original KV-up output once per chunk. Key-PE stays separate because it comes
    from the RoPE positional path, not the packed KV-up projection.
- TE reentrant NVFP4 grad-output preprocessing is split by usage:
  - File: `tools/patch_te_reentrant_backward.py`
  - Reason: the latest OOM stack landed inside TE `grad_output_preprocess`
    while DSA re-entered a TE Linear backward. For reentrant NVFP4, the patcher
    now quantizes grad-output rowwise for DGRAD first, then recreates the
    columnwise WGRAD tensor from the original dense chunk at the WGRAD boundary.
    This avoids carrying combined row+column grad-output state through the
    whole TE backward at the StreamBP/DSA peak.
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
- StreamBP MoE MLP backward chunks were restored to 4:
  - Forward MoE chunking remains 4.
  - Reason: increasing chunks slowed the step too much. Current work must keep
    the 8192-token replay shape and remove memory pressure inside the
    StreamBP/DSA/TE backward lifecycle instead.
- Split-QK DSA now consumes strided K-noPE from packed MLA KV:
  - Files:
    - `megatron/core/transformer/multi_latent_attention.py`
    - `megatron/core/transformer/experimental_attention_variant/dsa_triton.py`
    - `megatron/core/extensions/hisa_indexer/kernels/csrc/dsa_sparse_kv_bwd.cu`
    - `megatron/core/extensions/hisa_indexer/kernels/csrc/pybind.cpp`
  - Reason: `k_no_pe` is a view into packed MLA `[K_noPE | V]`. The old
    split-QK DSA path forced `key_nope.contiguous()`, retaining a full-prefix
    K-noPE copy in every StreamBP DSA attention graph. The fused Triton
    forward/backward and CUDA row backward now take K-noPE strides and load the
    view directly.
  - Measured saving: representative `sk=32768, local_heads=32,
    qk_head_dim=128, bf16` microbench saved exactly 256 MiB per retained DSA
    attention graph (`268 MiB` old simulated peak delta vs `12 MiB` new).
    Expected training peak benefit is this amount times the number of retained
    StreamBP DSA attention graphs in the hot MoE split-backward window.
- Split-QK DSA now consumes strided Q-noPE from packed MLA Q:
  - Files:
    - `megatron/core/transformer/multi_latent_attention.py`
    - `megatron/core/transformer/experimental_attention_variant/dsa_triton.py`
    - `megatron/core/extensions/hisa_indexer/kernels/csrc/dsa_sparse_kv_bwd.cu`
    - `megatron/core/extensions/hisa_indexer/kernels/csrc/pybind.cpp`
  - Reason: the StreamBP DSA attention replay chunk uses the no-PE query
    slice from packed MLA Q. Keeping that as a view avoids one more retained
    per-replay contiguous copy without changing DSA math.
  - Expected saving at the active `8192` query chunk:
    `8192 * 1 * 32 * 128 * 2 = 64 MiB` per retained DSA graph/rank.

## Current Memory Savings From This Scope

The directly measured and formula-derived DSA copy-removal saving is about
320 MiB per retained DSA attention graph/rank at the active shape:

- K-noPE view instead of contiguous copy: 256 MiB per full-prefix graph/rank.
- Q-noPE view instead of contiguous copy: 64 MiB per 8192-query replay
  graph/rank.

The total hot-window saving is larger because several promoted fixes shorten
lifetimes or remove temporaries that overlap in the same StreamBP -> DSA ->
TE backward region:

- Deferred DSA query gradients: avoids roughly 96 MiB per 8192-query DSA
  graph/rank during the K/V reentrant peak.
- Packed K/V reentrant gradient handoff: removes separate K-noPE/V component
  lifetimes at the TE K/V-up boundary; expected benefit is hundreds of MiB per
  active reentrant chunk/rank depending overlap.
- TE reentrant NVFP4 grad-output split: avoids carrying combined rowwise and
  columnwise grad-output state through the whole reentrant backward; expected
  benefit is hundreds of MiB to low single-digit GiB per hot rank.
- Activation-ECO early correction stash and MLP input release: shortens
  captured activation lifetimes before TE grouped-linear backward; expected
  benefit is stage-dependent and shows up as the failure moving deeper through
  the replay stack rather than as a clean isolated tensor-size formula.
- TE GroupedLinear fused WGRAD accumulation: removes per-expert temporary
  BF16 WGRAD tensors on the expert replay path; the individual failed
  allocation was only tens of MiB, but it occurred at the exact peak and helps
  allocator pressure.

Conservative total estimate for the current hot rank/window is 3-8 GiB saved
versus the pre-scope path, with the hard measured floor at about 320 MiB per
retained DSA graph/rank from Q/K-noPE copy removal alone. The next training
probe is still required because the real peak depends on how many StreamBP
DSA graphs and TE grouped-linear replay chunks overlap on each PP stage.
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

### Current DSA Backward Gate

The latest failure moved from TE grouped MLP replay into the DSA attention
replay boundary:

- Stack: StreamBP MoE replay backward -> attention replay autograd ->
  `SparseDSASplitQKAttentionTriton.backward`.
- Failure site: `megatron/core/transformer/experimental_attention_variant/dsa_triton.py`
  line 3352, allocating dense `grad_key_nope`.
- Interpretation: the active split-QK DSA kernel is the right semantic path,
  but its autograd contract materializes dense K/V gradients (`grad_value`,
  `grad_key_nope`, `grad_key_pe`) at the replay peak before the sparse backward
  math starts.

Deliverables for this gate:

1. Memory fix: fuse or explicitly split the DSA backward boundary so K/V
   gradients are consumed by the upstream MLA KV-up projection path without
   needing all dense K/V grad tensors live at once during StreamBP replay.
2. Speed fix: add a split-QK-specific CUDA/CuTe/CUTLASS-grade backward path
   that is key/block/tile owned rather than the current query-owned
   top-k-loop-plus-atomics Triton kernel.
3. Validation: compare the new path against the current split-QK Triton path
   for forward/backward numerical parity, then benchmark memory and time at
   representative DSA shapes before relaunching training.

### DSA Backward OOM Trace And Reentrant K/V Boundary

What the latest logs prove:

- Latest crash log:
  `/home/sjpat/logs/corsaire-1-research-preview_node1.log:1944-1958`.
- Stack:
  `StreamBP MoE replay backward -> attention replay autograd ->
  SparseDSASplitQKAttentionTriton.backward`.
- Failure:
  `dsa_triton.py` attempted to allocate dense `grad_key_nope` after already
  entering the split-QK DSA backward. The OOM request was 896 MiB with only
  110.81 MiB free.
- Code path:
  - MLA splits Q/K into no-PE and PE tensors in
    `multi_latent_attention.py`, then stores `_dsa_split_qk_parts`.
  - DSA calls `sparse_dsa_attention_split_qk_with_teacher_triton` when HISA
    indexer loss is deferred to the attention path.
  - The normal split-QK backward allocated all dense K/V grad tensors before
    launching the sparse backward:
    `grad_value`, `grad_key_nope`, and `grad_key_pe`.
  - The underlying Triton backward is query-owned: each query/top-k tile
    recomputes selected scores and scatters K/V updates with atomics. That is
    semantically correct, but the autograd contract forced all dense K/V grad
    outputs to stay live until outer autograd consumed them.

Patch implemented:

- Added `MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD`.
- Added `MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD_CHUNK`; unset means full
  sequence component chunks, so the path avoids the all-K/V live peak without
  defaulting to hundreds of tiny replay chunks.
- Split-QK backward now has an opt-in reentrant path:
  - Defer query grads until after K/V component grads have been consumed by
    upstream MLA projection backward.
  - Build `grad_value` and K grads as separate component buffers.
  - Immediately call reentrant autograd on `value`, `key_nope`, and `key_pe`
    slices so upstream MLA projection consumes those grads before the next
    dense component is allocated.
  - Return `None` for K/V grads to the outer custom autograd boundary because
    those grads have already been propagated.
- The launcher now defaults this path on:
  `MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD=1`.

Validation:

- `uv run --no-sync python -m py_compile
  megatron/core/transformer/experimental_attention_variant/dsa_triton.py`
  passed.
- `bash -n examples/sft/run_sft_deepseek_nvfp4.sh
  examples/sft/launch_sft_deepseek_nvfp4_tmux.sh` passed.
- FP32 parity against the old split-QK teacher path passed for both default
  full-component reentrant mode and a non-dividing chunk size of 17:
  - output max diff: 0
  - teacher max diff: 5.96e-08
  - q_nope/q_pe grad max diff: 0
  - k_nope max diff: up to 2.38e-07
  - k_pe max diff: up to 4.17e-07
  - value max diff: 4.77e-07
- q512, kv32768, batch4, heads32, topk1024 benchmark:
  - old split-QK teacher: 22.746 ms, 4.145 GiB synthetic peak
  - reentrant full-component teacher: 29.191 ms, 5.160 GiB synthetic peak
  - reentrant 2048 chunk teacher: 207.300 ms, 5.271 GiB synthetic peak
  - old no-teacher: 19.065 ms, 4.129 GiB synthetic peak
  - reentrant no-teacher: 25.587 ms, 5.145 GiB synthetic peak

Interpretation:

- The memory fix targets the real training OOM boundary, not the synthetic
  leaf-tensor benchmark peak. In the trainer, K/V grads flow into upstream MLA
  projection graphs; reentrant propagation should avoid keeping
  `grad_value + grad_key_nope + grad_key_pe` live at the StreamBP replay peak.
- The synthetic benchmark has leaf K/V tensors, so PyTorch accumulates `.grad`
  buffers on those leaves and does not show the intended upstream-consumption
  memory benefit.
- This path is a safety valve, not the final speed solution. It costs about
  1.28x on the q512 teacher microbench because it still scans selected top-k
  more than once.

Remaining speed work:

- The actual fast target is still a split-QK-specific lower-level CUDA/CuTe or
  CUTLASS-style sparse backward that reduces random atomics and avoids repeated
  top-k scans.
- Existing non-split CUDA K/V reducers in
  `megatron/core/extensions/hisa_indexer/kernels/csrc/dsa_sparse_kv_bwd.cu`
  do not handle split-QK no-PE/PE tensors, so they are not a drop-in fix for
  the active training path.
- The current split-QK Triton backward remains query-owned and atomics-heavy;
  a key/block-owned or tiled split-QK kernel is still required for throughput.

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

## Failure: TE Linear Retained-Graph Reentry

- Latest training run failed without OOM on node1 ranks 13/14/15.
- Failure site:
  - `StreamBP MoE replay backward`
  - `SparseDSASplitQKAttentionTriton.backward`
  - reentrant `torch.autograd.backward(...)` into the shared TE KV-up projection
  - `transformer_engine/pytorch/module/linear.py`
- Error:
  - `TypeError: 'NoneType' object is not iterable`
  - TE had already set `ctx.tensor_objects = None` after the first backward
    restore, but the split-QK reentrant path needs to re-enter the same TE
    Linear backward for value and key chunks from `linear_kv_up_proj`.
- Interpretation:
  - This was not a capacity failure.
  - It is a TE single-backward-lifetime assumption exposed by StreamBP + DSA
    retained-graph reentry.

## Patch: TE Linear Reentrant Metadata Retention

- Added `tools/patch_te_reentrant_backward.py`.
- The launcher now defaults
  `MEGATRON_TE_RETAIN_TENSOR_OBJECTS_FOR_REENTRANT_BACKWARD=1` and runs the
  patch before torchrun on both nodes.
- The patch makes TE's `ctx.tensor_objects = None` cleanup conditional on the
  env flag. Math is unchanged; it only preserves TE tensor-object metadata long
  enough for retained-graph reentry.
- Validation:
  - `uv run --no-sync python -m py_compile tools/patch_te_reentrant_backward.py megatron/core/transformer/experimental_attention_variant/dsa_triton.py`
  - `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
  - local `.venv` TE Linear patch applied and marker verified.

## Split-QK CUDA Row Backward Kernel Status

- Added a split-QK row-owned CUDA DSA backward extension:
  - `dsa_split_qk_bwd_row`
  - supports split query/key tensors, optional positions, and selective query /
    key / value emission.
- Correctness parity passed for the K/V and value emission subpath.
- Follow-up finding:
  - CUDA row query-gradient emission is not numerically equivalent for split-QK
    positional query grads, so training uses CUDA row only for K/V/value chunks
    and emits query grads through the existing Triton query-only pass.
- Benchmark at `q=512, kv=32768, batch=4, heads=32, topk=1024`:
  - current teacher path: `22.764 ms`, `4.145 GiB`
  - reentrant Triton teacher path: `29.203 ms`, `5.160 GiB`
  - new CUDA row teacher path: `34.760 ms`, `5.160 GiB`
  - current no-teacher path: `19.068 ms`, `4.129 GiB`
  - reentrant Triton no-teacher path: `25.588 ms`, `5.145 GiB`
  - new CUDA row no-teacher path: `31.078 ms`, `5.145 GiB`
- Decision:
  - Keep `MEGATRON_DSA_CUDA_SPLIT_QK_ROW_BWD=1` for the K/V/value side only.
  - Keep query-gradient emission on the existing Triton query-only pass until
    the CUDA row query path is fixed or replaced by the Blackwell-grade tiled
    split-QK kernel.

## Failure: TE NVFP4 Saved-Input Amax Missing In Reentrant KV-Up WGRAD

- Relaunched `corsaire-1-research-preview` with the TE retained-metadata patch.
- It passed checkpoint load and entered step 1.
- Failure moved forward to pp_rank 3 during `StreamBP fused LCE` / backward:
  - `StreamBP -> split MoE attention backward`
  - `SparseDSASplitQKAttentionTriton.backward`
  - reentrant backward into TE MLA `linear_kv_up_proj`
  - TE Linear WGRAD GEMM
- Error:
  - `nvte_nvfp4_compute_per_tensor_scale: Assertion failed: amax_A_ptr != nullptr`
- Interpretation:
  - The previous `ctx.tensor_objects=None` crash is fixed.
  - Reusing TE's saved NVFP4 input object for retained-graph reentry is not
    safe for this KV-up projection WGRAD path; the restored quantized object can
    reach GEMM without a valid amax pointer.

## Patch: Save Original MLA KV-Up Input Under FP4/TE

- For FP4 Transformer Engine MLA, set `linear_kv_up_proj.save_original_input`.
- This keeps the BF16 `kv_compressed` input for backward and lets TE quantize
  the WGRAD input freshly on each retained-graph traversal.
- Scope:
  - Only MLA `linear_kv_up_proj`.
  - Does not change launch shape.
  - Does not disable StreamBP, DSA, HISA/IndexCache, activation-ECO, or trainable
    indexer loss.
- Expected tradeoff:
  - Slightly more replay saved-input memory for KV-up, but it avoids invalid TE
    NVFP4 saved-input metadata in the reentrant DSA backward path.
- Validation:
  - `uv run --no-sync python -m py_compile megatron/core/transformer/multi_latent_attention.py megatron/core/transformer/experimental_attention_variant/dsa_triton.py tools/patch_te_reentrant_backward.py`
  - `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`

## Failure: TE Fuser Saved-Tensor Range Cleared During Reentrant DSA Backward

- Relaunched after the MLA KV-up saved-original-input patch.
- The previous TE Linear `tensor_objects=None` and NVFP4 `amax_A_ptr` failures
  were passed.
- New failure on node1 ranks 13/14/15:
  - `StreamBP MoE replay backward`
  - `SparseDSASplitQKAttentionTriton.backward`
  - reentrant backward into TE operation fuser
  - `transformer_engine/pytorch/ops/fuser.py`
- Error:
  - `TypeError: slice() argument after * must be an iterable, not NoneType`
  - TE fuser had set `ctx._saved_tensors_range = None` after the first saved
    tensor restore, then the chunked DSA K/V reentrant path entered the same
    retained graph again.
- Important memory note:
  - Keeping `_saved_tensors_range` does not keep activation tensors alive.
  - It preserves only small range metadata; TE still clears each basic op's
    `ctx.saved_tensors` after the op backward.
  - This keeps the chunked K/V propagation fix intact instead of falling back
    to a full dense K/V grad materialization.

## Patch: TE Fuser Reentrant Range Metadata Retention

- Extended `tools/patch_te_reentrant_backward.py` to also patch
  `transformer_engine.pytorch.ops.fuser`.
- The fuser patch is gated by the same env:
  `MEGATRON_TE_RETAIN_TENSOR_OBJECTS_FOR_REENTRANT_BACKWARD=1`.
- The patch makes `ctx._saved_tensors_range = None` conditional under that flag.
- It does not change DSA math, HISA/IndexCache, StreamBP routing, launch shape,
  trainable indexer loss, or the TE saved tensor cleanup itself.
- Validation:
  - `uv run --no-sync python tools/patch_te_reentrant_backward.py`
  - local TE Linear was already patched
  - local TE fuser was patched
  - `uv run --no-sync python -m py_compile tools/patch_te_reentrant_backward.py megatron/core/transformer/experimental_attention_variant/dsa_triton.py megatron/core/transformer/multi_latent_attention.py`
  - `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`

## Failure: TE RMSNorm Saved Tensor Cleared During Second Reentrant Traversal

- Relaunched after the TE fuser saved-range patch.
- The run entered step 1 and reached `StreamBP fused LCE active`.
- Node1 ranks 12/13/14/15 failed in TE fuser RMSNorm backward:
  - `transformer_engine/pytorch/ops/fuser.py`
  - `transformer_engine/pytorch/ops/basic/rmsnorm.py`
- Error:
  - `RuntimeError: shape '[0]' is invalid for input of size 1048576`
- Interpretation:
  - The fuser range metadata was now preserved correctly.
  - The first reentrant K/V traversal still allowed TE RMSNorm to call
    `clear_tensor_data(x)` and `clear_tensor_data(rstdevs)`.
  - The second reentrant traversal restored the same saved tensor objects, but
    their storage had already been cleared to the TE empty tensor shape `[0]`.

## Patch: Scope TE Retention To Active DSA Reentrant Backward

- Added a short-lived DSA guard:
  - `MEGATRON_TE_REENTRANT_BACKWARD_ACTIVE=1`
  - set only around the nested `torch.autograd.backward(...)` calls in
    `SparseDSASplitQKAttentionTriton.backward`.
- Updated `tools/patch_te_reentrant_backward.py` so TE retention now requires:
  - `MEGATRON_TE_RETAIN_TENSOR_OBJECTS_FOR_REENTRANT_BACKWARD=1`
  - and `MEGATRON_TE_REENTRANT_BACKWARD_ACTIVE=1`
- Updated the TE fuser patch to mark restored fuser saved tensors with
  `_do_not_clear` only inside that active reentrant guard.
- This avoids globally disabling TE saved-tensor cleanup. Normal TE backwards
  still clear tensors; only the retained graph being re-entered by DSA K/V
  propagation is protected.
- Validation:
  - `uv run --no-sync python tools/patch_te_reentrant_backward.py`
    - upgraded local TE Linear and TE fuser patches
  - `uv run --no-sync python -m py_compile tools/patch_te_reentrant_backward.py megatron/core/transformer/experimental_attention_variant/dsa_triton.py megatron/core/transformer/multi_latent_attention.py`
  - `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`

## Failure: TE Linear Saved Quantized Input Cleared During Reentrant DSA WGRAD

- Relaunched after the active-guard fuser patch.
- The previous TE fuser range and RMSNorm `[0]` failures were passed.
- New root failure was on node1 rank 13 during step 1 backward:
  - `StreamBP MoE replay backward`
  - split MoE MLP/attention backward
  - `SparseDSASplitQKAttentionTriton.backward`
  - nested reentrant backward into TE Linear WGRAD
- Error:
  - `Invalid matrix dimensions for GEMM (A=(1,0), transa=0, B=(2048,576), transb=1)`
- Interpretation:
  - TE Linear `ctx.tensor_objects` metadata survived.
  - But the restored saved quantized input storage itself was still allowed to
    be cleared by TE Linear's `clear_tensor_data(inputmat_total)` after the
    first retained-graph traversal.
  - The next DSA K/V reentry restored a quantized input wrapper whose data had
    been cleared, producing the invalid `(1,0)` GEMM input.

## Patch: Protect Restored TE Linear Saved Input Storage During Active Reentry

- Extended `tools/patch_te_reentrant_backward.py` again.
- During active DSA retained-graph reentry only, restored TE Linear `inputmat`
  and its quantized data tensors are marked `_do_not_clear`.
- This preserves the quantized saved input across repeated DSA K/V reentries.
- It does not globally save BF16 inputs and does not globally disable TE cleanup.
- Validation:
  - `uv run --no-sync python tools/patch_te_reentrant_backward.py`
  - `uv run --no-sync python -m py_compile tools/patch_te_reentrant_backward.py megatron/core/transformer/experimental_attention_variant/dsa_triton.py megatron/core/transformer/streambp.py megatron/core/transformer/multi_latent_attention.py`
  - `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`

## Failure: TE Linear Saved-Data Patch Did Not Skip Empty Data Tensor Slots

- Relaunched after the TE Linear saved-data patch.
- The run reached the same StreamBP MoE replay -> DSA split-QK reentrant
  backward region.
- New failure:
  - `AttributeError: 'NoneType' object has no attribute '_do_not_clear'`
  - from the newly inserted TE Linear saved-data patch.
- Interpretation:
  - Some TE quantized tensor wrappers return `None` entries from
    `get_data_tensors()`.
  - The storage protection logic must skip those slots.
- Fix:
  - Added a `data_tensor is None` guard.
  - Made `tools/patch_te_reentrant_backward.py` upgrade the already-installed
    old TE patch rather than adding a duplicate block.
- Validation:
  - `uv run --no-sync python tools/patch_te_reentrant_backward.py`
  - confirmed installed TE Linear now skips `None` data tensors.
  - `uv run --no-sync python -m py_compile tools/patch_te_reentrant_backward.py megatron/core/transformer/experimental_attention_variant/dsa_triton.py megatron/core/transformer/streambp.py megatron/core/transformer/multi_latent_attention.py`
  - `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`

## Current Failure: Allocator Fragmentation At TE MoE Sort During Replay

- Relaunched with the split-QK row CUDA backward path enabled:
  - `MEGATRON_DSA_CUDA_SPLIT_QK_ROW_BWD=1`
  - `MEGATRON_DSA_CUDA_SPLIT_QK_ROW_BWD_WARPS=8`
- The launcher banner confirmed the path was active.
- The run reached StreamBP fused LCE and then failed on node1 rank8/GPU0 inside
  StreamBP MoE MLP replay, before the next DSA split-QK reentrant boundary.
- Failure stack:
  - `streambp.py::_moe_chunk_attention_full_mlp_backward`
  - `moe_layer.py::routed_experts_compute`
  - `token_dispatcher.py::combine_preprocess`
  - `moe_utils.py::sort_chunks_by_idxs`
  - Transformer Engine `triton_permutation.sort_chunks_by_map`
- Error:
  - requested allocation: 334 MiB
  - free device memory: 226.81 MiB
  - allocated by PyTorch: 160.49 GiB
  - reserved but unallocated by PyTorch: 11.33 GiB
- Interpretation:
  - This crash is not the DSA row CUDA kernel itself.
  - The 11.33 GiB reserved/unallocated memory is not reliable model headroom;
    it is allocator-held memory that could not satisfy the contiguous TE sort
    allocation at the replay peak.
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6`
    was already set, so this is not simply a missing expandable-segments flag.
- Patch:
  - `megatron/core/transformer/moe/moe_utils.py` now optionally synchronizes
    before `torch.cuda.empty_cache()` in the MoE sort trim path.
  - Launchers default `MEGATRON_MOE_SORT_TRIM_SYNC=1` and print it in the
    banner.
- Validation:
  - `uv run --no-sync python -m py_compile megatron/core/transformer/moe/moe_utils.py`
  - `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`

## Current Failure: DSA Reentrant Value Backward Peak

- Relaunch with synchronized MoE sort trimming moved past the previous TE MoE
  sort OOM and reached the split-QK DSA reentrant backward boundary.
- Failure site:
  - `megatron/core/transformer/experimental_attention_variant/dsa_triton.py:3676`
  - nested `torch.autograd.backward(value_ref[kv_start:kv_end], grad_value_chunk, ...)`
- OOM requests on node1 ranks 8-11 were 640 MiB to 1.25 GiB with only a few
  hundred MiB truly free and roughly 11.2-11.35 GiB reserved/unallocated.
- Patch:
  - Default `MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD_CHUNK=8192` in both SFT
    launchers.
  - This shrinks the specific K/V reentrant component backward slice instead
    of changing global launch shape or disabling trainable indexer/DSA paths.

## CUTLASS/CuTe DSA Kernel Status

- The current active split-QK row CUDA path is not a true CuTe/CUTLASS/UMMA
  implementation. It is a row-owned CUDA kernel that avoids the Triton fallback
  and supports non-contiguous value strides, but still computes selected dot
  products with scalar CUDA loops and atomically scatters K/V gradients.
- Existing docs and microbenches show why a naive cuBLASDx/CuTe wrapper is not
  enough:
  - DSA selected attention is per-query sparse; each query row has a different
    selected key set.
  - Increasing query tile size made Triton slower because selected K/V gathers
    do not become a clean dense GEMM tile.
  - The previous cuBLASDx selected-score backward was correct but slower at
    production row counts because tile-level gradient atomics dominated.
- The real hardware-aware target remains a key/block-owned or selected-edge
  reduction scheduler for split-QK DSA backward:
  - keep selected-token semantics unchanged;
  - reduce repeated selected-QK recompute;
  - reduce random global atomics into selected K/V;
  - avoid materializing large edge lists unless the memory budget proves safe;
  - validate against the current exact split-QK backward before relaunch.

## Current Failure: pp2 StreamBP MoE Replay fc2 Peak

- Relaunch with finite DSA reentrant K/V chunking moved past the previous DSA
  value backward OOM.
- New failure site:
  - node1 rank8/GPU0
  - `streambp.py::_moe_chunk_attention_full_mlp_backward`
  - `experts.py::TEGroupedMLP.forward`
  - Transformer Engine `grouped_linear.py`, expert `linear_fc2` output
    allocation
- Error:
  - requested allocation: 328 MiB
  - free device memory: 294.81 MiB
  - PyTorch allocated: 160.52 GiB
  - PyTorch reserved but unallocated: 11.23 GiB
- Confirmed allocator:
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6`
    is set in both the tmux-generated node env and `run_sft_deepseek_nvfp4.sh`.
  - Added explicit launcher/per-node logging so future runs show this directly.
- Non-chunk levers applied:
  - Rebalanced PP layout from decoder totals `20/20/17/4` to `20/20/14/7`.
    This moves three MoE decoder layers from pp2 to pp3 while keeping
    TP/PP/CP/EP/ETP, MBS/GBS, DSA top-k, and StreamBP chunks unchanged.
  - Added a targeted low-free cache trim immediately before TE grouped expert
    `linear_fc2` output allocation:
    `MEGATRON_MOE_EXPERT_FC2_TRIM_CACHE=1`,
    `FREE_MB=8192`, `CACHED_MB=256`, `SYNC=1`.
    This does not change math or add replay chunks; it only releases cached
    allocator blocks at the exact allocation boundary that failed.

## Cross-Component Finding: StreamBP -> DSA -> TE Reentrant Peak

- The latest OOM stack is a systems-level lifetime issue, not an isolated bad
  kernel:
  - `streambp.py` split MoE backward keeps `post_attention` and the MLP input
    grad live while it calls `torch.autograd.backward(post_attention, ...)`.
  - DSA split-QK backward then allocates dense query/K/V grad chunks and
    re-enters autograd through K/V slices.
  - Those K/V slices flow into the retained MLA up-projection
    `CheckpointWithoutOutput` graph.
  - TE NVFP4 Linear backward then quantizes a dense `grad_output` and launches
    dgrad/WGRAD work while StreamBP and DSA replay tensors are still resident.
- The previous reentrant guard was too coarse:
  - `MEGATRON_TE_REENTRANT_BACKWARD_ACTIVE=1` forced
    `CheckpointWithoutOutput` and patched TE Linear/fuser paths to retain saved
    state for every reentrant call.
  - For the final K/V chunk there is no later consumer, so retaining at that
    point only extends tensor lifetimes into the worst memory peak.
- Patch:
  - Added `MEGATRON_TE_REENTRANT_BACKWARD_RETAIN`.
  - DSA now sets `RETAIN=1` only for non-final K/V chunks and `RETAIN=0` for
    the final chunk.
  - `CheckpointWithoutOutput` clears recompute state when active but
    `RETAIN=0`.
  - `tools/patch_te_reentrant_backward.py` now patches TE Linear and TE fuser
    retention only when both `ACTIVE=1` and `RETAIN=1`.
  - SFT launchers default `MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD_CHUNK=8192`
    instead of `32768`, so the inner DSA->TE K/V handoff is actually chunked
    without changing the global training shape.
- Validation:
  - `uv run --no-sync python tools/patch_te_reentrant_backward.py`
  - `uv run --no-sync python -m py_compile megatron/core/tensor_parallel/random.py megatron/core/transformer/experimental_attention_variant/dsa_triton.py tools/patch_te_reentrant_backward.py`
  - `bash -n examples/sft/run_sft_deepseek_nvfp4.sh examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
  - Selected DSA backward tests passed:
    `test_forward_backward_matches_torch_sparse_reference[...]`,
    `test_forward_backward_accepts_strided_value_view`,
    `test_cuda_kv_backward_matches_torch_sparse_reference_int16_topk`.
  - Selected StreamBP split MoE replay tests passed:
    `test_streambp_moe_nonchunk_forward_and_backward_chunk_attention_once_for_mlp`,
    `test_streambp_moe_hybrid_can_split_mlp_replay_into_large_chunks`,
    `test_streambp_moe_seq_aux_loss_split_mlp_replay_matches_baseline_gradients`.

## Current Hot Path: StreamBP Replay -> DSA -> HISA Candidate Refinement

- Latest full-run failure moved into PP2 StreamBP replay:
  - `streambp.py::_moe_chunk_attention_full_mlp_backward`
  - DSA `chunked_dsa_forward`
  - `_HISASelectWithScoresBatched`
  - `indexcache/hisa.py::_hisa_grouped_candidate_topk`
  - OOM at `candidate_k = k_rows.index_select(...)`.
- This is a whole-path memory lifetime issue:
  - StreamBP replay already has retained attention/MLP tensors live.
  - DSA reentrant backward is holding split-QK replay state.
  - HISA BMM candidate refinement then materializes gathered candidate K and
    per-head candidate dot products inside that tight replay window.
- Benchmarked production internal HISA shape on B200:
  - rows=512, heads=64, head_dim=128, context=32768, topk=1024.
  - BMM selector, temp cap 128 MiB: ~8.6 ms, ~332 MiB peak.
  - BMM selector, temp cap 32 MiB: ~13.1 ms, ~251 MiB peak.
  - packed cuBLASDx tiled selector: exact selected sets/scores vs BMM on
    sampled rows, ~47.9 ms, ~113 MiB peak.
- Applied immediate safe fix:
  - Keep exact BMM HISA selector semantics by default.
  - Lower default `MEGATRON_HISA_CANDIDATE_MAX_TEMP_MB` to 32.
  - Add live free-memory-aware cap with
    `MEGATRON_HISA_CANDIDATE_FREE_MEM_RESERVE_MB=64`.
  - This reduces candidate-refine scratch at the precise replay allocation
    boundary without changing launch shape, DSA top-k, or trainable indexer
    loss semantics.
- DeepGEMM status:
  - Installed and validated on both nodes with CUDA 13 / SM100.
  - APIs available: `fp8_fp4_mqa_logits`,
    `fp8_fp4_paged_mqa_logits`, `get_paged_mqa_logits_metadata`.
  - Isolated FP4 paged candidate scorer for rows=512,candidates=8192 is very
    fast (~0.07 ms kernel, ~0.14 ms torch topk), matching the optimization
    playground direction.
  - Direct OP-style DeepGEMM candidate scoring quantizes Q to FP4; on a random
    production-shaped comparison it overlapped only ~12% of the current exact
    BMM selected top-k set. Do not default it without either accepting this
    selector semantic change or adding an exact selected-logit recompute /
    custom BF16-Q + FP4-K kernel.

## FP4-Q DeepGEMM HISA Selector

- User accepted the FP4-Q selector contract if correctness is validated.
- Wired `MEGATRON_HISA_SELECTOR_BACKEND=deepgemm`:
  - HISA block selection still uses the existing BF16/FP32 block-score path.
  - Candidate refinement now quantizes Q to FP4 and reads K from the packed
    NVFP4 IndexCache sidecar through DeepGEMM paged MQA logits.
  - Selected-token IDs are still treated as non-differentiable; the downstream
    selected-score backward keeps the existing trainable-indexer semantics for
    the chosen tokens.
- Important bug fixed during validation:
  - DeepGEMM expects each 64-token page as a value plane followed by a scale
    plane, matching its `kv_cache_cast_to_fp4` helper.
  - A per-token `[values, scale]` layout produced huge garbage logits; the new
    unit test caught this before training.
- Launchers now default to `deepgemm` and set
  `DG_JIT_CACHE_DIR=$HOME/.cache/deep_gemm/deepseek_v32_reap_sft`.
- Validation:
  - `test_hisa_deepgemm_selector_backend_matches_fp4_oracle` compares selected
    sets and scores against the matching FP4-Q/NVFP4-K oracle.
  - Existing packed cuBLASDx and NVFP4 sidecar tests still pass.
  - Selected DSA split-QK/HISA tests still pass.

## Current Correction: Return HISA Default To BMM And Shorten Multi-GB Lifetimes

- User asked to go back to
  `MEGATRON_HISA_SELECTOR_BACKEND=bmm` with the new 32 MiB HISA temp cap and
  free-memory cap.
- Launchers now default back to:
  - `MEGATRON_HISA_SELECTOR_BACKEND=bmm`
  - `MEGATRON_HISA_CANDIDATE_MAX_TEMP_MB=32`
  - `MEGATRON_HISA_CANDIDATE_FREE_MEM_RESERVE_MB=64`
  - `MEGATRON_HISA_SELECTOR_ROW_CHUNK=512`
- Why DeepGEMM was slow in trainer:
  - Our integrated path only swapped candidate scoring, not the full OP
    optimized HISA pipeline.
  - It still rebuilt/copy-packed the DeepGEMM K page cache per HISA call or
    row chunk, quantized Q per call, rebuilt page tables/schedules per call,
    used PyTorch `topk`, and kept Python partial-block remapping.
  - OP's accepted path persists packed NVFP4 page/block reps and fuses block
    top-k, page table construction, mask/top-k, and token-id mapping. The
    DeepGEMM matmul itself is only a small part of that full win.
- BMM path issue found and fixed:
  - With `MEGATRON_HISA_SELECTOR_ROW_CHUNK=512`, the old BMM recursion rebuilt
    dense K, block means, and partial-block prefix tensors for every query
    chunk.
  - The BMM path now computes K-side dense/block/prefix tensors once per HISA
    call and reuses them across row chunks.
  - Row-chunk output assembly now preallocates `[rows, topk]` outputs instead
    of collecting chunks and `torch.cat`-ing them at the end.
- Multi-GB MoE lifetime fixes:
  - `MoELayer.dispatch_and_routed_experts_compute()` now dispatches and runs
    routed experts in one frame so all-to-all/gather outputs can be released
    before expert GEMMs allocate padded inputs/fc1/fc2 tensors.
  - `MoEAlltoAllTokenDispatcher.combine_preprocess()` and the allgather
    combine path now support `MEGATRON_MOE_COMBINE_REDUCE_SCATTER_DTYPE`.
    Launchers default it to `input`, avoiding a multi-GB BF16 -> FP32 expert
    output cast caused only by FP32 router-prob dtype before TP reduce-scatter.
    This should improve both peak memory and bandwidth.
- StreamBP replay lifetime fixes:
  - Split MoE replay now clears each attention-output graph reference
    immediately after that attention chunk's backward finishes, instead of
    keeping all DSA/TE graphs in the list until the entire MLP chunk is done.
  - StreamBP LM-head hidden grad uses `empty_like` instead of `zeros_like`;
    every slice is assigned before return, so this removes a full-buffer memset
    without changing peak allocation.
- Validation:
  - `uv run --no-sync python -m py_compile` on modified HISA/MoE/StreamBP files.
  - `git diff --check`.
  - HISA BMM selector parity test passed.
  - Selected DSA split-QK/HISA tests passed.
  - Selected StreamBP MoE replay and fused-LCE tests passed.
  - A helper smoke confirms `MEGATRON_MOE_COMBINE_REDUCE_SCATTER_DTYPE=input`
    chooses BF16 expert-output dtype while `router` preserves old FP32 behavior.

## Current Correction: MoE StreamBP Grad-Readiness Accounting

- Audit conclusion:
  - The PP3 OOM is not just one HISA temp. The hot region is
    `StreamBP MoE backward -> replay attention graphs -> replay MoE MLP ->
    TE grouped fc2`.
  - In the MoE hybrid StreamBP path, attention params are replayed per
    attention chunk while MoE/MLP params are replayed per MLP chunk.
  - The regular StreamBP path marked all layer params as pending across
    chunks before letting DDP register grad-ready, but the MoE hybrid path did
    not. That allowed overlap grad-reduce/param-gather work to begin while
    later chunks of the same layer were still replaying in the peak-memory
    region.
- Fix:
  - Added `_streambp_moe_hybrid_pending_marks()`.
  - Attention-side params are marked with the attention backward chunk count.
  - MLP/MoE-side params are marked with the MLP backward chunk count.
  - Shared/overlapping params are marked only once.
- Throughput/memory classification:
  - This is not a chunk-size increase and does not add replay compute.
  - It may reduce some within-layer communication overlap, but that overlap was
    occurring during the exact allocator peak and could be both a memory and
    correctness problem.
  - Expected effect is memory-positive and correctness-positive; throughput
    should be neutral to slightly positive if it avoids allocator thrash, or
    slightly negative if the previous premature comm overlap was actually
    useful.
- Validation:
  - `uv run --no-sync python -m py_compile megatron/core/transformer/streambp.py`.
  - Existing selected StreamBP tests passed.
  - Added and passed
    `test_streambp_moe_hybrid_marks_attention_and_mlp_ddp_chunks_separately`.
