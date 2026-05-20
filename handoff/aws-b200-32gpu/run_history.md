# Run History And Current Pain Points

This is the condensed context that should survive conversation compaction.

## Original Goal

Train/heal the DeepSeek V3.2 REAP/NVFP4 Corsaire checkpoint on the Blaise SFT
mix, preserving the model-specific features:

- DSA sparse attention.
- Trainable DSA indexer.
- HISA/IndexCache.
- SpinQuant / ActKV / NVFP4 stack.
- G1 attention output gates.
- GatedNorm.
- HIGGS dense 2-bit KV.
- FlashAdamW ECO.
- Activation ECO.
- StreamBP memory savings.

Target was originally 1-2B tokens, later discussed as running through more data
and stopping when quality/time budget says stop.

## Major Things Learned

### 1. Loss/Grad Instability Was Not Just The DSA Indexer

Early scary loss/grad behavior appeared to concentrate in attention-related
buckets such as attention output and KV up/down paths. That motivated enabling
DSA indexer training instead of freezing the indexer and forcing all adaptation
pressure through narrow attention paths.

Confirmed separate issue:

- TE/Apex fused multi-tensor clipping was corrupting BF16 decoupled grad views
  in the FlashAdamW/NVFP4 path.
- Fix: keep fused clipping for FP32 grads, use native foreach/mul for BF16 and
  other non-FP32 grads.

### 2. Naive Optimizer ECO Was Unstable At Scale

Simulations and training probes suggested paper-style ECO injection into Adam
moment state can become unstable as parameter count scales. A warmup-aware /
projection-budgeted ECO path was introduced:

- `FLASH_ADAMW_ECO_LR_FLOOR=base`
- `FLASH_ADAMW_ECO_PROJECTION=gain`

Do not assume the ECO paper behavior is safe for this model scale.

### 3. DSA/HISA Python Reference Paths Were Too Slow

HISA forward/reference selection with CPU/tolist/item loops was a major problem.
We moved to CUDA/Triton and compact HISA/DSA paths:

- CUDA HISA selector pieces.
- Fused selected-score / teacher / KL-related pieces.
- Triton DSA paths.
- Reentrant split-QK DSA backward.
- DSA RoPE fusion and compact top-k handling.

The current launcher still uses `MEGATRON_HISA_SELECTOR_BACKEND=bmm` with CUDA
support enabled; do not assume all experimental DeepGEMM/CuTe variants are
active.

### 4. StreamBP Is Necessary But Expensive

Turning off StreamBP repeatedly made memory worse or impossible. But StreamBP
replay, especially around DSA backward and MoE, has been the dominant time and
memory pain.

Important corrected understanding:

- StreamBP does reduce some activation residency, but pipeline queues and
  retained graph states can still accumulate across virtual microbatches.
- The issue is end-to-end tensor lifecycle across PP/VPP/StreamBP/DSA/MoE, not
  any single isolated tensor.

### 5. MBS/GBS Findings On 16 GPUs

`MBS=8, GBS=32`:

- reached backward but OOMed in node1 PP2 DSA backward.
- PP3 backward was also very slow.
- not memory-safe.

`MBS=4, GBS=16`:

- much better memory and earlier backward.
- latest crash was not OOM.
- first meaningful signal was a DeepEP timeout on node0 rank3 during StreamBP
  MoE replay.

`MBS=32`:

- invalid with `GBS=32` and VPP because only one microbatch.
- with `GBS=128`, legal but OOMed in MoE forward/shared expert path.

## Latest Failure

Most recent run:

- two nodes / 16 GPUs
- `TP=4 PP=4 CP=1 DP=1 EP=4 ETP=1`
- VPP enabled
- `MBS=4 GBS=16` in the latest probe context
- seq 16k
- top-k 512
- StreamBP enabled
- DeepEP flex dispatcher enabled

Failure:

```text
DeepEP timeout check failed: rank = 3, thread = 0, value = 1024
DeepEP timeout check failed: rank = 3, thread = 1, value = 1024
DeepEP timeout check failed: rank = 3, thread = 2, value = 1024
DeepEP timeout check failed: rank = 3, thread = 3, value = 0
```

Trace later surfaced at:

```text
StreamBP MoE replay
  -> layer._forward_mlp
  -> MoE postprocess
  -> maybe_temp_cpu_reload(shared_expert_output)
  -> tensor.to(device)
  -> CUDA error: unspecified launch failure
```

Interpretation:

- `tensor.to(device)` is probably where CUDA synchronized/reported the failure.
- Root is more likely a DeepEP dispatch/combine timeout under StreamBP replay.
- Node1 errors were collateral after node0 rank3 died.

## Best Next Diagnostic

Do not relaunch blindly.

Run a controlled MoE dispatcher isolation:

1. same shape with:
   `MOE_TOKEN_DISPATCHER_TYPE=alltoall`
2. if stable, flex/deepep with:
   `MEGATRON_DEEPEP_COMPACT_LOCAL_PERMUTE=0`
3. if still failing, debug external DeepEP/NVSHMEM path

This is diagnostic, not a final performance choice. The final intended path is
still the fused/flex/DeepEP path if it can be made stable.

## Things To Avoid Relearning

- `moe_enable_deepep False` in logs does not mean DeepEP is disabled.
- The two-node tmux launcher is not an AWS four-node launcher.
- `uv sync` does not currently recreate the live environment.
- CP is blocked while `USE_STREAMBP=1`.
- Raising StreamBP chunking can make memory fit but usually destroys step time.
- Do not freeze DSA indexer as a quick fix unless making an explicit quality
  tradeoff.

