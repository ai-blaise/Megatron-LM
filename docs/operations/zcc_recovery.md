# ZCC Recovery

ZCC writes one snapshot directory per step and rank:

```text
<zcc root>/step_0000250/rank_00000/zcc_snapshot.pt
```

Durable snapshots use the same layout with `zcc_durable.pt`.

## Flash Recovery

Use the flash tier when the local host survived and `/dev/shm` or the configured
flash mount is still available:

```python
from megatron.core.optimizer.zero_cost_checkpoint import load_zcc_state_dict

payload = load_zcc_state_dict(
    "/dev/shm/megatron_zcc/step_0000250/rank_00000/zcc_snapshot.pt",
    mode="flash",
)
```

The loader validates the file magic and checksum before returning the payload.

## Durable Fallback

Use durable mode when the flash tier is missing or failed validation:

```python
payload = load_zcc_state_dict(
    "/checkpoints/run/zcc/step_0000250/rank_00000/zcc_durable.pt",
    mode="durable",
)
```

`mode="auto"` tries flash, peer, then durable. This is the normal mode for a
hot restart: a torn or corrupted flash file fails checksum validation and falls
through to the durable tier.

## Applying a Snapshot

Loading returns a payload. Applying it to a live optimizer is a separate step:

```python
from megatron.core.optimizer.zero_cost_checkpoint import (
    load_zcc_state_dict,
    restore_zcc_state,
)

payload = load_zcc_state_dict(
    rank_snapshot_dir,
    mode="auto",
    durable_dir="/checkpoints/run/zcc",
)
missing, unexpected = restore_zcc_state(
    optimizer,
    payload,
    opt_param_scheduler=opt_param_scheduler,
)
```

Both lists should be empty for an identical restart topology. Non-empty values
mean the live optimizer layout does not match the snapshot and the restart
should fall back to the regular distributed checkpoint path.

If a quantizer stores persistent parameter-side tensors outside the optimizer
state, restore with the same tensor-attribute registry used during snapshot:

```python
missing, unexpected = restore_zcc_state(
    optimizer,
    payload,
    extra_tensor_attrs=("_fp8_weight_cache",),
    opt_param_scheduler=opt_param_scheduler,
)
```

The registry is dtype-agnostic. ZCC copies tensors back to the live tensor's
dtype and device using the saved descriptor and does not need quantizer-specific
restore code.

## Peer Recovery

Peer recovery is for replacement-rank flows where a surviving rank still owns a
host mirror. The training job must already have an initialized recovery process
group containing the donor and replacement ranks.

```python
from megatron.core.optimizer.zero_cost_checkpoint.recovery import recover_from_peer

recover_from_peer(
    donor_rank=1,
    receiver_rank=0,
    tensor=pinned_or_device_mirror,
    group=recovery_group,
)
```

For CPU tensors PyTorch uses the CPU-capable backend configured for the group.
For device tensors the receiver copies the incoming CPU mirror back to the
target tensor.

## Failure Handling

- Torn or partial writes fail checksum validation.
- Missing flash snapshots should fall through to durable with `mode="auto"`.
- Durable corruption raises after all requested sources fail.
- ZCC worker write failures are surfaced on the next `sync_before_step()` or
  `finalize()`.
