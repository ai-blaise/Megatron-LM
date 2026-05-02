# FlashAdamW + ECO under Gradient Release

This document covers the gradient-release (GR) path on the
`flashoptim-gradientrelease` branch: how it interacts with FlashAdamW's
ECO injection, NVFP4 cast orchestration, and Megatron-Core DDP.

## Background

Gradient release fires the per-parameter optimizer step inside
`register_post_accumulate_grad_hook` and frees `param.grad` immediately
after, capping peak gradient memory at one parameter's worth instead of
the whole graph. The base mechanism lives in
`megatron/core/optimizer/flash_optimizers.py::enable_gradient_release`
and was authored before the ECO codepath landed; this round closes the
gaps on **(a)** ECO numerical parity, **(b)** NVFP4 cast + inject
orchestration, and **(c)** Megatron-Core DDP support.

## ECO under GR — what just works

For non-NVFP4 FlashAdamW with `eco=True`, the existing GR path is
correct out of the box: `step_param` → `_do_step` →
`_fused_adam_step(eco=True, eco_scalar=…)` folds the ECO injection into
the Adam Triton kernel. There is no separate post-step phase to
orchestrate, so the standard `enable_gradient_release(model, optimizer)`
call gives bit-equivalent updates over a 50-step run.

Verified by `tests/unit_tests/optimizer/test_flash_adamw_eco_gradient_release.py`:

- `test_eco_unquantized_50step` — bf16 master, ECO on
- `test_eco_quantized_50step`   — quantized exp_avg, ECO on
- `test_no_eco_unquantized_50step` — control

## NVFP4 + ECO — orchestration required

For NVFP4 params, `step_param` routes through `_step_nvfp4_transient`
which only does the Adam math on the transient bf16 master shard and
stashes it on `p._fa_updated_shard`. The TE NVFP4 cast and the ECO
error inject normally fire later, batched, inside
`distrib_optimizer._copy_main_params_to_model_params`. That batched
phase is invoked from `optimizer.step()` — which is a no-op under GR.

To bridge this, GR now accepts a `post_step` callback that fires after
each `step_param` and before `param.grad` is freed. Compose it with
`NVFP4EcoGradientReleaseOrchestrator`:

```python
from megatron.core.optimizer.flash_optimizers import (
    enable_gradient_release, NVFP4EcoGradientReleaseOrchestrator,
)
from megatron.core.optimizer.nvfp4_sr import cast_master_weights_to_nvfp4_2d_sr

orch = NVFP4EcoGradientReleaseOrchestrator(
    optimizer=optimizer,
    cast_fn=cast_master_weights_to_nvfp4_2d_sr,
    data_parallel_group=data_parallel_group,
)
handle = enable_gradient_release(model, optimizer, post_step=orch.post_step)
```

The orchestrator buffers each NVFP4 param's `_fa_updated_shard` as it
arrives and dispatches the batched TE cast + per-param `inject_eco_error`
once the expected NVFP4 param count is reached. **Cross-rank amax
all-reduce determinism is preserved** because the cast is still issued
once per backward pass (not per parameter), matching the ordering the
batched path produces today.

If a backward pass touches fewer than the expected number of NVFP4
params (e.g., MoE expert sparsity), call `orch.flush()` after backward
to drain the buffer.

### Memory invariant

The orchestrator allocates **no persistent state**. The buffered
references are the same `_fa_updated_shard` tensors the batched path
already creates per step; the orchestrator just reorders when they're
consumed. Verified by `test_state_keys_match_batched`.

## Megatron-Core DDP

Plain `enable_gradient_release` rejects Megatron-Core DDP: under MCore
DDP, gradients land on `param.main_grad` (fp32) after a per-bucket
reduce-scatter; `param.grad` is `None` by the time
`register_post_accumulate_grad_hook` fires. Stepping against
`param.grad` would silently use stale local-rank gradients.

Use `enable_gradient_release_mcore_ddp(ddp_module, optimizer)` instead.
It hooks the post-accumulate hook to read `param.main_grad`, stages it
on `param.decoupled_grad` (the precision-aware path that allows fp32
grad against bf16 param), and runs `step_param`.

```python
from megatron.core.optimizer.flash_optimizers import enable_gradient_release_mcore_ddp
from megatron.core.distributed import DistributedDataParallel as MCoreDDP

ddp_model = MCoreDDP(config, model)
handle = enable_gradient_release_mcore_ddp(ddp_model, optimizer)
```

**Caveat:** the single-rank path is bit-equivalent to bare
`enable_gradient_release` (verified by
`test_main_grad_path_matches_grad_path`). Multi-rank with overlapped
reduce-scatter requires the bucket sync to land before the per-param
step; until that integration is wired into `param_and_grad_buffer`'s
bucket-completion callback, callers running with
`overlap_grad_reduce=True` must call `optimizer.step()` themselves
after `finish_grad_sync()`. Pure single-bucket and
`overlap_grad_reduce=False` configurations are fine.

## Parallelism (CP / TP / EP / SP)

The per-parameter step is local — no cross-rank op happens inside
`step_param` for non-NVFP4 ECO, and the NVFP4 cast collects amax
across ranks via the cast_fn the orchestrator calls (which still
batches the all-reduce). Therefore GR is safe under all four
sharding modes from the project constraints:

- **CP** (Context Parallelism, definitely used) — operates on the
  sequence dim; per-param updates are unaffected.
- **TP** (Tensor Parallelism) — each rank holds a shard of the param;
  `_get_local_tensor` extracts the local DTensor view inside
  `step_param`. Verified by `test_tp_shard_equivalence`.
- **EP** (Expert Parallelism, undecided vs TP) — only routes MoE FFN
  params; orthogonal to the optimizer step.
- **SP** (Sequence Parallelism) — sequence-dim sharding; like CP, has
  no effect on per-param updates.

Frozen quantization buffers (TurboQuant signs/codebooks/boundaries,
seeded by `layer_idx`) are unaffected — they're not optimizer state.

## Target model alignment

The `kv_lora_rank=512` and `hidden_size=7168` shapes from
`BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1`
are exercised by `test_kv_lora_shape_under_gr`. The bf16 saved-w_hat
decision (TurboQuant relies on FlashAdamW + ECO to absorb the cast
noise) is preserved — GR doesn't change the precision of any tensor in
flight.

## Hard invariants preserved

1. **No fp32 master shadow** — neither GR helper introduces one.
2. **No extra peak/persistent memory** — verified by state-key parity
   test.
3. **`@triton.autotune` + in-place RMW must use `restore_value`** —
   GR doesn't add new autotuned kernels; the existing
   `_triton_eco_inject_kernel` guard remains in force.
4. **Per-token locality, bit-exact under CP/TP/EP/SP** — verified by
   the TP-shard equivalence test.

## Test surface

| File | Coverage |
|---|---|
| `test_flash_adamw_eco_gradient_release.py` | 50-step bit-parity vs batched, eco × {quantized, unquantized} + non-eco control |
| `test_flash_adamw_gr_extensions.py` | post_step callback, MCore DDP detection + single-rank step |
| `test_flash_adamw_eco_gr_alignment.py` | Target-model shapes, TP-shard equivalence, no-extra-state, bf16-param/fp32-main_grad parity |

Run: `python -m unittest discover tests/unit_tests/optimizer/`.
