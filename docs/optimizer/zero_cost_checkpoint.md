# Zero Cost Checkpoint

Zero Cost Checkpoint (ZCC) captures a per-rank optimizer-state shadow after
the optimizer update while avoiding the normal synchronous checkpoint path.
It is off by default and is enabled with:

```bash
--enable-zero-cost-checkpoint
```

The implementation follows the ERNIE/PaddleNLP ZCC invariant: parameters and
optimizer state are stable outside the optimizer-update window, so post-update
state can be mirrored to host memory and persisted separately from the normal
checkpoint path.

## Snapshot Scope

ZCC snapshots:

- Model parameter tensors. Transformer Engine FP8/NVFP4 wrapper tensors are
  snapshotted through their persistent raw storage attributes rather than by
  flattening the logical wrapper tensor.
- FlashAdamW optimizer state, including quantized `_MaybeQuantizedTensor`
  storage.
- Error-correction state when present.
- Persistent parameter-side quantization tensors. The built-in registry covers
  known Transformer Engine FP4 tensor payload names, and
  `--zcc-extra-tensor-attrs` / `MEGATRON_ZCC_EXTRA_TENSOR_ATTRS` adds
  attributes for FP8 or custom quantizers.
- Optimizer hyperparameter sidecars.
- Optimizer parameter-scheduler state.
- PyTorch CPU/CUDA RNG state and the NVFP4 stochastic-rounding dither counter
  when the corresponding flags are enabled.

Transient FlashOptim fields such as `_fa_updated_shard` are not walked by the
snapshot planner.

Snapshot tensors are recorded as generic descriptors: name, shape, dtype,
source device, and CPU payload. The planner does not branch on NVFP4, FP8, or
any other quantization scheme. A quantizer participates in ZCC by storing its
persistent restart state in optimizer state or by exposing parameter-side
tensor attributes listed in the registry.

## Fused State Buffer

When ZCC is enabled, FlashAdamW optimizer-state tensors are packed into one
contiguous uint8 CUDA buffer after state initialization. Each
`_MaybeQuantizedTensor` storage field is rebound to a dtype-correct view into
that buffer. `_MaybeQuantizedTensor.set_data()` updates existing matching
storage in place, so optimizer updates preserve the fused backing buffer across
steps.

This gives ZCC a single large DtoH source for optimizer state rather than
thousands of small tensor copies.

## Runtime Flow

The training loop calls the manager around each optimizer update:

1. `sync_before_step()` drains the previous ZCC copy stream and initializes or
   fuses optimizer state if needed.
2. `optimizer.step()` updates parameters normally.
3. `opt_param_scheduler.step()` advances the learning-rate schedule.
4. `snapshot_after_step(iteration + 1, opt_param_scheduler=...)` plans
   persistent tensors, copies them into pinned host mirrors on the ZCC CUDA
   stream, writes tier-1 with atomic rename and checksum, and enqueues tier-2
   durable persistence when due.

The tier-1 path is synchronous and uncompressed because it is the hot-recovery
source. Tier-2 is written by a background dump worker pool and is not on the
training step's critical path. Durable snapshots honor `--zcc-compress`; `zstd`
compression is used when the optional `zstandard` Python module is installed
and otherwise falls back to the same checksum-protected uncompressed envelope.
By default ZCC retains only the newest step directory in each configured root;
set `--zcc-retain-latest 0` to disable pruning, or a larger value to keep more
recent steps.

If `--zcc-flash-stripe` is set, ranks are spread across the configured
tier-1 roots by rank. This balances per-rank flash writes across local devices;
it is not a distributed global checkpoint format.

## Knobs

| CLI | Env | Default |
|---|---|---|
| `--enable-zero-cost-checkpoint` | `MEGATRON_ENABLE_ZCC` | off |
| `--zcc-workers-num` | `MEGATRON_ZCC_WORKERS_NUM` | `1` |
| `--zcc-flash-device` | `MEGATRON_ZCC_FLASH_DEV` | `/dev/shm/megatron_zcc` |
| `--zcc-flash-stripe` | `MEGATRON_ZCC_FLASH_STRIPE` | empty |
| `--zcc-durable-dir` | `MEGATRON_ZCC_DURABLE_DIR` | unset |
| `--zcc-durable-interval` | `MEGATRON_ZCC_DURABLE_INTERVAL` | `10` |
| `--zcc-compress` | `MEGATRON_ZCC_COMPRESS` | `zstd:1` |
| `--zcc-recovery-mode` | `MEGATRON_ZCC_RECOVERY_MODE` | `auto` |
| `--zcc-extra-tensor-attrs` | `MEGATRON_ZCC_EXTRA_TENSOR_ATTRS` | empty |
| `--zcc-retain-latest` | `MEGATRON_ZCC_RETAIN_LATEST` | `1` |

Boolean flags also have environment overrides:
`MEGATRON_ZCC_INCLUDE_RNG`, `MEGATRON_ZCC_INCLUDE_DITHER`,
`MEGATRON_ZCC_BUCKET_HOOK`, `MEGATRON_ZCC_NUMA_PIN`,
`MEGATRON_ZCC_USE_GDS`, and `MEGATRON_ZCC_FAULT_INJECT`.

## Recovery

Use `megatron.core.optimizer.zero_cost_checkpoint.load_zcc_state_dict(path)` to
read a ZCC snapshot. The loader validates the ZCC magic and SHA-256 preamble.
`mode="auto"` tries flash, peer, then durable. When flash and durable roots are
separate, pass `durable_dir` so auto recovery can map the flash step/rank
directory to the matching durable snapshot. Use `restore_zcc_state()` to copy
payload tensors back into the live optimizer and restore scheduler/RNG/dither
metadata. Pass the same extra tensor attributes when restoring, or restore from
an optimizer wrapper that still owns its ZCC manager. Peer recovery is exposed
through `recover_from_peer()` for runs that
already formed a replacement-rank process group.

## Current Boundaries

This implementation provides in-process ZCC for the FlashOptim branch. It does
not replace Megatron's cold-start distributed checkpointing format, and it does
not claim automatic node replacement orchestration. The peer API handles tensor
transfer once the recovery process group exists.

## Verification

The ZCC tests live in `tests/unit_tests/optimizer/test_zcc.py`.

On the b1 H200 VM:

```bash
CUDA_VISIBLE_DEVICES=0 ~/work/zcc-py311/bin/python -m pytest \
  tests/unit_tests/optimizer/test_zcc.py -q
```

Result on this branch: `8 passed`.

A local B200 CUDA smoke also validated quantized FlashAdamW state fusion,
latest-only pruning, ZCC load, and restore of the fused quantized momentum
buffer.

```bash
CUDA_VISIBLE_DEVICES=0 ~/work/zcc-py311/bin/python -m pytest tests/unit_tests/optimizer -q
```

Result: `65 passed`.

The distributed production-style ZCC stress was run on all 8 H200s:

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ~/work/zcc-py311/bin/torchrun --standalone --nproc-per-node=8 \
  /tmp/zcc_dist_stress.py
```

Result: passed on 8 ranks. The stress used FlashAdamW quantized optimizer
state, a synthetic `uint8` `_fp8_weight_cache` parameter-side quantizer tensor,
`--zcc-extra-tensor-attrs`, flash striping across two `/dev/shm` roots, zstd
durable snapshots every 8 steps, deliberate flash corruption, and auto recovery
through a separate durable root.

Transformer Engine 2.14.1 was built in the b1 Python 3.11 environment with
explicit cuDNN/NCCL include paths. H200 FP8 support reports available, and:

```bash
CUDA_VISIBLE_DEVICES=0 ~/work/zcc-py311/bin/python -m pytest \
  tests/unit_tests/test_fp8_utils.py -q
```

Result: `2 passed`.

The quantization/custom-feature smoke coverage used:

```bash
CUDA_VISIBLE_DEVICES=0 ~/work/zcc-py311/bin/python -m pytest \
  tests/unit_tests/quantization/test_turboquant_kv.py \
  tests/unit_tests/quantization/test_turboquant_parallelism.py \
  tests/unit_tests/quantization/test_indexcache.py \
  tests/unit_tests/quantization/test_nvfp4_act_eco_correctness.py \
  tests/unit_tests/quantization/test_nvfp4_act_eco_compose.py -q
```

Result: `34 passed, 3 skipped`; the skipped TurboQuant parallelism tests passed
under a 2-rank `torchrun` invocation.

Known non-ZCC caveats from the same environment:

- `tests/unit_tests/test_fp8_param.py` reaches FP8 setup but requires the Apex
  `fused_weight_gradient_mlp_cuda` extension when
  `gradient_accumulation_fusion=True`.
- The DSA suite progressed after fixing an in-place autograd issue in the
  index-score reference path, then exposed an existing missing-gradient issue
  in `TestDSAttention.test_dsa_backward`.
- `tests/unit_tests/test_sft_dataset.py` imports after installing `pandas`, but
  its fixture currently returns 2 rows while the test expects 1.
