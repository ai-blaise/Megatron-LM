# Training Stack

## Model

The run is a continued-pretraining/SFT healing run for a DeepSeek V3.2 REAP
checkpoint:

- HF source model:
  `BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4`
- Converted Megatron checkpoint:
  `/home/sjpat/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron`
- Save path:
  `/home/sjpat/checkpoints/sft_deepseek_v32_reap_spinquant_actkv_nvfp4`
- Dataset default in the runner:
  `/home/sjpat/data/sft/blaise-sft-training-mix/nemotron-full-family.jsonl`

The model stack includes:

- DeepSeek V3.2-style MLA.
- REAP-pruned expert structure.
- DSA sparse attention as the experimental attention variant.
- G1 attention output gates: `--attention-output-gate`.
- GatedNorm: `--gated-norm --gated-norm-rank 16`.
- MoE with 128 experts, top-8 router, group top-k, sigmoid score function,
  pre-softmax routing, and expert bias.
- Quantile router bias update:
  `--moe-router-expert-bias-update-method quantile`.

## Quantization / Compression

Enabled by the current launcher:

- BF16 training base dtype.
- NVFP4 recipe:
  `--fp4-format e2m1 --fp4-recipe nvfp4 --fp4-param-gather`
- SpinQuant:
  `--spinquant`, `w/a/k/v` all 4-bit.
- HIGGS dense 2-bit KV:
  `--enable-higgs-dense-2bit-kv-cache`.
- TurboQuant:
  present in code but disabled by default because HIGGS is enabled.
- IndexCache:
  enabled by default for DSA indexer K.
- HISA IndexCache:
  enabled with NVFP4 quantization:
  `DSA_INDEXCACHE_QUANTIZATION=nvfp4_e2m1_ue8m0`,
  block size 128, block top-k 64, compression ratio 4.0.
- Activation ECO:
  `NVFP4_ACTIVATION_ECO=1`,
  modules `all`,
  recompute-only,
  TE backend,
  BF16 correction dtype.
- FlashAdamW ECO:
  enabled with LR floor and gain projection:
  `FLASH_ADAMW_ECO=1`,
  `FLASH_ADAMW_ECO_LR_FLOOR=base`,
  `FLASH_ADAMW_ECO_PROJECTION=gain`.

Important prior finding: naive/paper-style optimizer ECO and gradient clipping
interactions caused loss/grad instability. A confirmed issue was TE/Apex
multi-tensor clipping corrupting BF16 decoupled grad views in the FlashAdamW
NVFP4 path; clipping was changed so non-FP32 grads use native foreach/mul paths.

## DSA / HISA Kernel Paths

Current launcher defaults:

- `MEGATRON_DSA_TRITON=1`
- `MEGATRON_DSA_TRITON_INDEXER=1`
- `MEGATRON_DSA_SPLIT_QK=1`
- `MEGATRON_DSA_STREAMING_INDEXER_TOPK=1`
- `MEGATRON_DSA_COMPACT_TOPK_INDICES=1`
- `MEGATRON_DSA_STREAM_TRITON_ATTENTION_CHUNKS=1`
- `MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH=1`
- `MEGATRON_DSA_TRITON_BWD_NUM_WARPS=2`
- `MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD=1`
- `MEGATRON_DSA_SPLIT_QK_REENTRANT_PACK_KV_GRAD=1`
- `MEGATRON_DSA_CUDA_SPLIT_QK_ROW_BWD=1`
- `MEGATRON_DSA_CUDA_KV_BWD=0`
- `MEGATRON_HISA_SELECTOR_BACKEND=bmm`
- `MEGATRON_HISA_SELECTOR_CUDA=1`
- `MEGATRON_HISA_FUSED_INDEXER_LOSS=1`
- `MEGATRON_HISA_TARGET_TRITON=1`
- `MEGATRON_HISA_KL_GRAD_TRITON=1`
- `MEGATRON_HISA_SELECTED_SCORE_BWD_WARP_GROUPED=1`

Custom extension:

- `megatron_hisa_indexer`
- source:
  `megatron/core/extensions/hisa_indexer/kernels/csrc/`
- includes HISA selector, selected-score backward, DSA sparse KV backward,
  and DSA indexer RoPE CUDA kernels.

The DSA indexer is intended to train:

- `--dsa-indexer-loss-coeff` default is `0.01`
- `--dsa-indexer-topk` current default is `512`

Do not silently set indexer loss to zero unless explicitly making a quality
tradeoff.

## MoE / DeepEP

Current launcher defaults:

- `MOE_TOKEN_DISPATCHER_TYPE=flex`
- `MOE_FLEX_DISPATCHER_BACKEND=deepep`
- `MEGATRON_DEEPEP_COMPACT_LOCAL_PERMUTE=1`
- `MEGATRON_DEEPEP_COMPACT_ROW_CHUNK=131072`
- `MEGATRON_WEIGHTED_SWIGLU_FUSER=triton`
- `--moe-grouped-gemm`
- `--moe-permute-fusion`
- `--moe-router-padding-for-quantization`

Important nuance:

`moe_enable_deepep False` in logs does not mean DeepEP is disabled. That flag is
deprecated. The active path is controlled by:

- `--moe-token-dispatcher-type flex`
- `--moe-flex-dispatcher-backend deepep`

Latest crash diagnosis:

- first hard signal was `DeepEP timeout check failed` on node0 rank3
- the CUDA error surfaced later at `maybe_temp_cpu_reload(shared_expert_output)`
- node1 failures were collateral after node0 died

This means the current DeepEP+StreamBP replay interaction is not fully proven.
The next clean diagnostic is an A/B:

1. Same shape with `MOE_TOKEN_DISPATCHER_TYPE=alltoall`.
2. If that survives, retry flex/deepep with
   `MEGATRON_DEEPEP_COMPACT_LOCAL_PERMUTE=0`.
3. If compact-off still fails, investigate external DeepEP/NVSHMEM path.

## StreamBP / Recompute / Offload

Current runner defaults:

- `USE_STREAMBP=1`
- StreamBP is incompatible with CP in this launcher; CP must stay 1 unless the
  StreamBP code is changed.
- With TP=4 and seq=16384, local sequence is 4096.
- Default `STREAMBP_CHUNK_SIZE` is half local sequence, currently 2048.
- DSA chunk follows StreamBP chunk unless overridden.
- `STREAMBP_MOE_MLP_CHUNKS=1`
- `STREAMBP_MOE_MLP_BACKWARD_CHUNKS=4`
- `MEGATRON_STREAMBP_SPLIT_MOE_MLP_ATTENTION_BACKWARD=1`
- fused / decoupled LM-head CE path enabled:
  `MEGATRON_STREAMBP_FUSED_LCE=1`,
  `MEGATRON_CHUNKED_LM_HEAD_LOSS=1`

Fine-grained activation offload:

- enabled by default
- modules:
  `expert_fc1 core_attn attn_proj qkv_linear moe_act attn_norm mlp_norm mlp_residual moe_shared`

Temporary activation offload:

- enabled by default
- modules:
  `mlp_residual moe_shared`

Known fragile area:

- StreamBP reentrant MoE replay plus DeepEP plus temporary reload of
  `shared_expert_output`.
- A previous bug zeroed base storage through a view during forced offload
  release. The current code skips `untyped_storage().resize_(0)` for view
  tensors.

## Checkpointing

Normal Megatron checkpointing:

- `--ckpt-format torch_dist`
- `--auto-detect-ckpt-format`
- save interval default 50 updates
- retain interval huge by default so only latest normal Megatron checkpoint is
  retained

ZCC:

- `ENABLE_ZCC=1`
- `ZCC_WORKERS_NUM=8`
- `ZCC_DURABLE_INTERVAL=50`
- `ZCC_RETAIN_LATEST=1`
- flash path defaults to `/dev/shm/megatron_zcc/$RUN_NAME`
- durable path defaults under `$SAVE_CKPT/zcc/$RUN_NAME`

Conceptually:

- ZCC is recovery/local durability.
- Normal Megatron checkpoint is the export/convert-to-HF source.

