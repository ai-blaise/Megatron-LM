<!--
SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Megatron CRIU Hooks

`tools/criu` contains the signal hooks used by `ai-blaise/criu-snapshots`
for flashtraining Megatron jobs. The Kubernetes launcher must add this
directory to `PYTHONPATH` and set `MEGATRON_CRIU_ENABLE=1`; Python then
loads `sitecustomize.py`, installs the handlers, and keeps normal
training entrypoints unchanged.

Signals:
- `SIGRTMIN+5` (`39` on Linux): synchronize CUDA, destroy an initialized
  torch distributed process group, and write
  `${MEGATRON_CRIU_STATE_DIR}/pre_snapshot.ready`.
- `SIGRTMIN+6` (`40` on Linux): rebuild the destroyed process group when
  state was captured, synchronize CUDA, and write
  `${MEGATRON_CRIU_STATE_DIR}/post_restore.done`.

The hooks intentionally do not call a distributed barrier in the
pre-snapshot path. The snapshot controller fans out to all selected rank
Pods concurrently, but single-rank snapshots are still valid and must not
deadlock waiting for ranks that were not selected.

The restore entrypoint is `tools/criu/megatron-criu-entrypoint.sh`. It
execs the original command when no CRIU image is mounted and runs CRIU
restore when `${SNAPSHOT_DIR}/img/inventory.img` is present.

## Smoke Test

Run from the repository root:

```bash
state_dir=$(mktemp -d)
PYTHONPATH=tools/criu \
MEGATRON_CRIU_ENABLE=1 \
MEGATRON_CRIU_STATE_DIR="$state_dir" \
python3 -c 'import megatron_criu_hooks, os, time; print(os.getpid(), flush=True); time.sleep(20)' &
pid=$!
sleep 1
kill -39 "$pid" && test -s "$state_dir/pre_snapshot.ready"
kill -40 "$pid" && test -s "$state_dir/post_restore.done"
kill "$pid"
```
