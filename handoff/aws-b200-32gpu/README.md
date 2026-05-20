# AWS B200 32-GPU Handoff

This folder is the handoff context for continuing `corsaire-1-research-preview`
training on AWS with 4 nodes / 32 B200 GPUs.

The main repo branch is `flashtraining`. At the time this was written:

- `HEAD`: `13a98acec249962431e9f30c96c86ceb77c3319a`
- branch status: ahead of `origin/flashtraining` by 2 commits
- uncommitted files: `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`,
  `examples/sft/run_sft_deepseek_nvfp4.sh`, `goal.md`

## Read Order

1. `training_stack.md`
   Current model, quantization, kernels, optimizer, StreamBP, offload, checkpoint,
   and known failure context.
2. `launcher_and_runtime.md`
   How the current launch scripts work, what is GCP/two-node-specific, and what
   must change for 4 AWS nodes.
3. `aws_32gpu_plan.md`
   Recommended starting parallelism shapes and tradeoffs for 32 GPUs.
4. `uv_sync_and_build.md`
   Critical environment reproducibility notes. Read this before running
   `uv sync`.
5. `run_history.md`
   Condensed history of the failures and why the current choices exist.

## Current Most Important Finding

The live `.venv` is not reproduced by a clean `uv sync` today.

`uv sync --locked --extra dev --extra mlm --check` reports that it would remove
locally installed packages that are required by the current launcher/runtime,
including:

- `deep-ep @ file:///tmp/DeepEP-v1.2.1`
- `deep-gemm @ file:///tmp/DeepGEMM.install`
- `nv-one-logger-*`
- the currently installed TE wheel packages

It would also reinstall `megatron-core` and downgrade/replace several packages.
Do not assume a clean AWS `uv sync` is sufficient until `uv_sync_and_build.md`
is handled.

## Current 16-GPU Runtime Shape

The current two-node launcher defaults to:

- `NNODES=2`, `GPUS_PER_NODE=8`, world size 16
- `TP=4`, `PP=4`, `CP=1`, `DP=1`
- `EP=4`, `ETP=1`, expert DP 1
- VPP enabled with layout:
  `Et*5|t*4|t*4|t*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*4|t*3|t*3L`
- `SEQ_LENGTH=16384`
- current launcher default `MICRO_BATCH_SIZE=4`, `GLOBAL_BATCH_SIZE=128`
- StreamBP enabled in the runner by default
- DSA top-k default `512`
- DeepEP flex MoE dispatcher enabled by default:
  `MOE_TOKEN_DISPATCHER_TYPE=flex`,
  `MOE_FLEX_DISPATCHER_BACKEND=deepep`

The most recent stability experiments often overrode GBS to 16/32. Do not rely
on the launcher defaults as the last tested stable shape.

## Immediate AWS Migration Checklist

- Copy or push this branch and all uncommitted launcher changes.
- Decide how to reproduce the live environment:
  either update `pyproject.toml`/`uv.lock`, or run the documented post-sync
  installs for DeepEP/DeepGEMM/torchcomms.
- Ensure Python 3.12 and CUDA 13/cu13 packages are used.
- Put the `torchcomms` wheel at `/home/sjpat/wheelhouse/...` or change the
  `tool.uv.sources.torchcomms` path.
- Build DeepEP with NVSHMEM available. The current DeepEP source has a local
  setup.py change adding `${CUDA_HOME}/include/cccl`.
- Validate custom extensions on an AWS B200 node:
  `scripts/prebuild_custom_extensions.sh`
- Render the launch command before training:
  `scripts/render_launch_dry_run.sh`
- Run the environment checker:
  `uv run --no-sync python handoff/aws-b200-32gpu/scripts/verify_env.py --expected-gpus 8`
  on each node, then a distributed smoke.

## Scripts

- `scripts/verify_env.py`
  Imports key packages, checks CUDA/SM capability, prints package versions, and
  warns about known missing/fragile pieces.
- `scripts/prebuild_custom_extensions.sh`
  Builds/loads the custom CUDA extensions before a real run.
- `scripts/bootstrap_live_env_after_uv_sync.sh`
  Reinstalls live-only packages after `uv sync` until the lockfile is made
  authoritative.
- `scripts/render_launch_dry_run.sh`
  Prints the runner-resolved command without launching. It defaults to the
  recommended 4-node AWS starting shape, not the old 2-node tmux wrapper shape.
- `scripts/aws_4node_env_example.sh`
  Template environment overrides for a 4-node AWS launch. This is intentionally
  not executable as a finished launcher; fill in AWS host/network details first.
