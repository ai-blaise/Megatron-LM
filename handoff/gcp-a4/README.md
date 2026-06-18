# GCP A4 Multi-Node Handoff

This handoff is for the `flashtraining` branch on GCP `a4-highgpu-8g`
instances named `instance-group-1-*` in `us-east1-b`.

The default full-fleet DeepSeek NVFP4 SFT shape is:

```bash
TP=4 PP=5 CP=2 EP=8 ETP=1 GPUS_PER_NODE=8
MICRO_BATCH_SIZE=2 GLOBAL_BATCH_SIZE=60
SEQ_LENGTH=32768 DSA_INDEXER_TOPK=1024 USE_STREAMBP=0 RECOMPUTE=0
LR=5.0e-5 DSA_INDEXER_LOSS_COEFF=0.1 NUM_WORKERS=16
TRAIN_TOKEN_TARGET=24691703808 SAVE_INTERVAL=10 SAVE_RETAIN_INTERVAL=500
```

The default SFT data path is the two-config materialized mix:

```text
$HOME/data/sft/blaise-sft-training-mix/blaise-sft-training-mix-full.jsonl
```

Build it with `tools/prepare_blaise_sft_mix_jsonl.py`, which combines
`nemotron-full-family` and `nemotron-mixed-sample` and writes the companion
offset index. The current materialized row count is:

```text
nemotron-full-family: 380565
nemotron-mixed-sample: 372966
total: 753531
```

`TRAIN_TOKEN_TARGET=24691703808` is `753531 * 32768`, so the default run covers
the full materialized mix. With the current `GBS=60`, the runner rounds this
to `753540` train samples.

The converted Megatron checkpoint is expected at:

```text
$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron_tp8_pp1_ep8
```

Current full-fleet parity verification:

```text
nodes: 15
checkpoint files per node: 18
chunk manifest entries per node: 2640
chunk manifest sha256: 4a988ddf04d07be8aa55153cd804126b810ff88da49f9cddf45a82839a36b204
```

On the 15-node A4 fleet this is 120 GPUs, `DP=3`, and expert DP 3. The shape
keeps `TP4 x CP2` inside each 8-GPU A4 node and uses five physical pipeline
stages, so each pipeline replica spans five nodes and the full fleet provides
three data-parallel replicas. With `SEQ_LENGTH=32768` and `CP=2`, the local
query sequence for the DSA/HISA hot path is 16384 tokens per microbatch. The
default `MBS=2` and `GBS=60` give ten accumulation microbatches:

```text
GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP) = 60 / (2 * 3) = 10
```

VPP is disabled by default for the current 32k DSA/HISA profile. The PP15/VPP2
and PP5/VPP experiments OOMed from pipeline warmup activation residency before
reaching useful 1F1B steady state. Plain PP5 uses this layout:

```text
Et*13|t*12|t*12|t*12|t*12L
```

Set `ALLOW_VPP_OOM_EXPERIMENT=1 ENABLE_VPP=1` only for a short diagnostic run
that intentionally explores the VPP memory cliff.

The current no-pipeline diagnostic shape uses 12 of the 15 nodes and leaves
three nodes as restart/replacement capacity:

```bash
MAX_NODES=12 TP=8 PP=1 CP=4 EP=8 ETP=1
MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=12
```

This resolves to `world_size=96`, `DP=3`, `expert_dp=12`, and local DSA/HISA
query length `8192`. Use `PP=1` for no pipeline; Megatron does not use `PP=0`.
For this shape, pipeline queue offload is disabled because no pipeline
send/recv queue exists. The point of this diagnostic is to remove the PP warmup
activation residency cliff before reintroducing any targeted memory relief.

## Scripts

- `kernel_fastpath_audit.md`
  Records the current fast-path launcher profile, memory-conservative knobs
  that should stay disabled, and B200 microbench results for HIGGS, IndexCache,
  HISA, and DSA. The earlier one-GPU table was taken at the CP8/32k diagnostic
  shape; the current fleet launch is the PP5/CP2 baseline above.
- `scripts/discover_nodes.sh`
  Lists running `instance-group-1-*` nodes from GCP, sorted by name.
- `scripts/bootstrap_node.sh`
  Idempotent per-node repo/env setup. It clones or updates this repo, installs
  missing system tools (`git`, `rsync`, `tar`, `gcc`, `g++`, `make`, `patch`)
  and RDMA userspace development headers when needed, installs `uv`, hydrates
  the locked build group, prepares the CUDA pip package layout needed by
  source-built extensions, runs the full locked sync, installs documented
  live-only DeepEP/DeepGEMM extras, verifies the runtime, and prebuilds custom
  extensions.
- `scripts/launch_deepseek_nvfp4.sh`
  Discovers the current fleet, selects a valid node set for the current shape,
  computes `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `DP`, and `GLOBAL_BATCH_SIZE`,
  starts the per-node HF checkpoint upload sidecar, then runs
  `examples/sft/run_sft_deepseek_nvfp4.sh`.
- `scripts/launch_conversion_deepseek_nvfp4.sh`
  Uses the same fleet discovery and rank math for HF-to-Megatron conversion.
- `scripts/launch_fleet_tmux.sh`
  Starts one remote tmux training/conversion session per selected node and opens
  a controller dashboard for fleet state, GPU metrics, rank-log tail, and W&B
  writer tail.
- `scripts/setup_slurm.sh`
  Installs/configures Munge + Slurm on the discovered fleet, writes static node
  and GPU config, starts `slurmctld` on the controller, starts `slurmd` on every
  node, and grants the controller Slurm job UID access to runtime paths.
- `scripts/submit_slurm_deepseek_nvfp4.sh`
  Syncs changed repo files, then submits the training or conversion entrypoint
  as one Slurm task per node with all eight GPUs allocated per node.
- `scripts/propagate_auth_env.sh`
  Writes user-level HF/W&B auth env files on the fleet. Tokens are not stored in
  this repo.
- `scripts/fanout_checkpoint_from_node.sh`
  Rsyncs the verified one-node converted Megatron checkpoint from a source node
  to the rest of the fleet with bounded parallelism and per-node logs. Default
  parallelism is 12 target lanes, with append-verify resume for interrupted
  large files.
- `scripts/fanout_sft_data_from_node.sh`
  Rsyncs the materialized SFT JSONL and `.offsets.npy` sidecar from a source
  node to the rest of the fleet with bounded parallelism and per-node logs.
  Default parallelism is 12 target lanes, with append-verify resume for
  interrupted large files.
- `scripts/verify_checkpoint_parity.sh`
  Launches high-parallel chunk hashing in tmux on every discovered node and
  compares the resulting manifests to prove bitwise checkpoint parity across
  the fleet. Defaults to 96 workers per node and 256 MiB chunks; override
  `HASH_WORKERS` when there is spare CPU.
- `scripts/run_on_gcp_nodes.sh`
  Thin SSH fan-out wrapper for running a command on discovered nodes once SSH is
  authorized.
- `scripts/install_nvidia_driver.sh`
  Installs and verifies the pinned host NVIDIA driver for A4/B200 nodes.

## Current Driver State

All 15 discovered `instance-group-1-*` nodes were verified with:

```text
NVIDIA driver: 580.159.03
GPUs per node: 8 x NVIDIA B200
CUDA shown by nvidia-smi: 13.0
```

`580.159.03` is the R580 LTS branch version reported by Google's current A4
installer and satisfies the CUDA 13/cu13 runtime used by `uv.lock`.

The normal remote path is:

```bash
DRIVER_VERSION=580.159.03 EXPECTED_GPUS=8 \
  handoff/gcp-a4/scripts/install_nvidia_driver.sh
```

The Google installer may do this in two passes on a fresh node: first it
installs kernel/development prerequisites and reboots, then a second run
installs the NVIDIA driver. If SSH host keys change after reboot, reconnect
with refreshed host-key handling.

On the controller (`instance-group-1-1jzl`), the driver was installed without
reboot by installing `kernel-devel-$(uname -r)` and running the exact NVIDIA
`580.159.03` runfile directly.

## Remaining Setup Notes

The lockfile also points at a local torchcomms wheel:

```text
/home/sjpat/wheelhouse/torchcomms-0.2.0-cp312-cp312-linux_x86_64.whl
```

That file must exist on every node before `uv sync --locked --extra dev --extra
mlm` can reproduce the locked environment.

The lockfile and `pyproject.toml` also include the CUDA 13 compiler/header
packages used by local source builds (`nvidia-cuda-nvcc`, `nvidia-cuda-cccl`,
`nvidia-cuda-crt`, `nvidia-nvvm`, `nvidia-nvml-dev`) plus `cmake`/`ninja`.
`bootstrap_node.sh` intentionally runs `uv sync --locked --only-group build`
first so those packages exist before compiling Transformer Engine and the other
CUDA extensions.

DeepEP additionally needs the system RDMA userspace development headers,
specifically `/usr/include/infiniband/mlx5dv.h`. On these Rocky/Ctrl IQ images
that comes from `rdma-core-devel`; the bootstrap script installs it via `dnf`
unless `INSTALL_SYSTEM_DEPS=0`.

SSH fan-out currently uses `/home/sjpat/google_compute_engine`. After a node
reboot, use refreshed host-key handling if direct SSH reports stale host keys.

`run_on_gcp_nodes.sh` defaults to `SSH_USER=sjpat` and
prefers `SSH_KEY=$HOME/google_compute_engine`, falling back to
`$HOME/.ssh/google_compute_engine`. The matching public key must be in project
or instance SSH metadata for every selected node.

## Bootstrap One Node

Run on each node:

```bash
cd /home/sjpat/Megatron-LM
handoff/gcp-a4/scripts/bootstrap_node.sh
```

If the host driver/device nodes are not visible yet, bootstrap just the Python
and CUDA user-space environment first:

```bash
EXPECTED_GPUS=0 PREBUILD_EXTENSIONS=0 handoff/gcp-a4/scripts/bootstrap_node.sh
```

This uses `uv.lock` as the authority for Python package versions:

```bash
uv sync --locked --only-group build
uv sync --locked --extra dev --extra mlm
```

Between those commands, the script creates unversioned `lib*.so` symlinks in
the CUDA/NCCL/cuDNN/NVSHMEM pip package directories and adds a compatibility
`cuda_profiler_api.h` shim for Transformer Engine's CUDA 13 build.

It then runs the documented post-sync installer for DeepEP and DeepGEMM because
the launcher defaults still import/use those packages:

```bash
handoff/aws-b200-32gpu/scripts/bootstrap_live_env_after_uv_sync.sh
handoff/aws-b200-32gpu/scripts/prebuild_custom_extensions.sh
```

The live installer defaults to the staged patched DeepEP wheel:

```text
handoff/gcp-a4/wheelhouse/deep_ep-1.2.1+9af0e0d-cp312-cp312-linux_x86_64.whl
```

That wheel exposes the strict expert-major dispatch APIs used by the current
MoE movement fast path. With `REQUIRE_DEEPEP_EXPERT_MAJOR=1`, bootstrap fails
instead of silently building unpatched DeepEP. The script also fails early if
the torchcomms wheel is missing.

## Discover Nodes

```bash
handoff/gcp-a4/scripts/discover_nodes.sh
```

Useful overrides:

```bash
ZONE=us-east1-b INSTANCE_REGEX='^instance-group-1-' handoff/gcp-a4/scripts/discover_nodes.sh
```

## Render A Dry Run

On each participating node:

```bash
cd /home/sjpat/Megatron-LM
DRY_RUN=1 handoff/gcp-a4/scripts/launch_deepseek_nvfp4.sh
```

For the current 15-node fleet, default selection should produce:

```text
selected nodes: 15
world size: 120
DP: 3
GLOBAL_BATCH_SIZE: 60
```

`GLOBAL_BATCH_SIZE=60` uses ten accumulation steps:

```text
GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP) = 60 / (2 * 3) = 10
```

With the default PP5/VPP-off layout, PP0 only needs to hold four warmup forward
microbatches before 1F1B starts. Do not enable VPP for the main 32k run unless
you are intentionally reproducing the warmup-residency OOM profile.

For a lighter communication smoke, override the batch and token target:

```bash
MICRO_BATCH_SIZE=1 GRAD_ACCUM_STEPS=10 TRAIN_TOKEN_TARGET=10000000 \
  DRY_RUN=1 handoff/gcp-a4/scripts/launch_deepseek_nvfp4.sh
```

For the no-pipeline 12-node profile, keep both Slurm and launcher selection at
12 nodes:

```bash
JOB_NAME=gcp-a4-nopp-profile PROFILE_RUN=1 NNODES=12 MAX_NODES=12 \
  TP=8 CP=4 PP=1 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=12 \
  handoff/gcp-a4/scripts/submit_slurm_deepseek_nvfp4.sh
```

If `NNODES` is omitted, the submitter now uses `MAX_NODES` for the Slurm node
request so `MAX_NODES=12` does not accidentally allocate 15 nodes.

Render the fleet tmux launch without starting training:

```bash
DRY_RUN=1 ACTION=start ATTACH=0 handoff/gcp-a4/scripts/launch_fleet_tmux.sh
```

Render the conversion command on the controller:

```bash
DRY_RUN=1 handoff/gcp-a4/scripts/launch_conversion_deepseek_nvfp4.sh
```

## Slurm

For the full multi-day run, prefer Slurm over a pure tmux launch. Slurm gives us
one job ID, clean cancellation/requeue behavior, node state, and an allocation
boundary around all 120 GPUs. The tmux dashboard is still useful for manual
monitoring and quick experiments.

Setup or refresh Slurm from the controller:

```bash
handoff/gcp-a4/scripts/setup_slurm.sh
```

Current verified state:

```text
Slurmctld(primary) at instance-group-1-1jzl is UP
15 nodes idle in partition a4
```

Basic allocation checks:

```bash
sinfo -Nel
srun -N15 --ntasks-per-node=1 bash -lc 'cd /home/sjpat/Megatron-LM && hostname -s'
srun -N15 --ntasks-per-node=1 --gres=gpu:8 \
  bash -lc 'printf "%s gpus=%s auth=%s\n" "$(hostname -s)" "$(nvidia-smi -L | wc -l)" "$([[ -r ~/.config/megatron/auth.env ]] && echo yes || echo no)"'
```

Submit a full-fleet launch dry-run through the same Slurm path:

```bash
DRY_RUN=1 JOB_NAME=gcp-a4-launch-dryrun TIME_LIMIT=00:05:00 \
  handoff/gcp-a4/scripts/submit_slurm_deepseek_nvfp4.sh
```

Submit conversion through Slurm:

```bash
JOB_NAME=gcp-a4-convert TIME_LIMIT=24:00:00 \
ENTRYPOINT=handoff/gcp-a4/scripts/launch_conversion_deepseek_nvfp4.sh \
  handoff/gcp-a4/scripts/submit_slurm_deepseek_nvfp4.sh
```

Submit training through Slurm:

```bash
JOB_NAME=gcp-a4-deepseek-sft \
  handoff/gcp-a4/scripts/submit_slurm_deepseek_nvfp4.sh
```

The `a4` partition is configured with `DefaultTime=NONE` and
`MaxTime=UNLIMITED`, so the training submit path intentionally omits `--time`.
Spot preemption and checkpoint cadence are the effective run boundary.

Because the nodes do not share `/home/sjpat`, Slurm batch stdout/stderr files
live on the job `BatchHost`. Find the host and log paths with:

```bash
scontrol show job <job-id> | rg 'JobState|BatchHost|StdOut|StdErr'
```

## Launch Path

The default launcher keeps the quality-sensitive stack enabled:

```bash
USE_STREAMBP=0
FASTPATH_STRICT=1
RECOMPUTE=0
ENABLE_VPP=0
INDEXCACHE=1
DSA_INDEXCACHE_HISA=1
DSA_INDEXCACHE_QUANTIZATION=nvfp4_e2m1_ue8m0
SPINQUANT=1
USE_HIGGS=1
TURBOQUANT=0
DSA_INDEXER_TOPK=1024
DSA_INDEXER_LOSS_COEFF=0.1
SEQ_LENGTH=32768
MEGATRON_HISA_SELECTOR_BACKEND=bmm
MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD_CHUNK=32768
MEGATRON_DSA_SPLIT_QK_REENTRANT_DEFER_QUERY_GRADS=0
MEGATRON_DSA_CUDA_SPLIT_QK_ROW_BWD=1
MEGATRON_DSA_CUDA_SPLIT_QK_ROW_QUERY_BWD=1
MEGATRON_DSA_CUDA_SPLIT_QK_ROW_BWD_WARPS=8
DSA_CHUNK_SIZE=2048
MEGATRON_HISA_CANDIDATE_SLOT_GROUP=8
MEGATRON_HISA_SELECTOR_ROW_CHUNK=256
MEGATRON_HISA_TARGET_ROW_CHUNK=256
MEGATRON_HISA_COMPACT_CANDIDATE_TOPK=1
MEGATRON_DSA_TEACHER_SCORE_SCRATCH=1
MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH=1
NVFP4_ACTIVATION_ECO=0
FINE_GRAINED_ACTIVATION_OFFLOADING=0
MEGATRON_PIPELINE_QUEUE_OFFLOAD=1  # PP>1 only; PP=1 defaults to 0
MEGATRON_PIPELINE_QUEUE_OFFLOAD_DEPTH=1  # PP>1 only; PP=1 defaults to 0
FLASH_ADAMW_ECO=1
FLASH_ADAMW_ECO_LR_FLOOR=base
FLASH_ADAMW_ECO_PROJECTION=gain
MOE_TOKEN_DISPATCHER_TYPE=flex
MOE_FLEX_DISPATCHER_BACKEND=deepep
SAVE_INTERVAL=10
SAVE_RETAIN_INTERVAL=500
WANDB_ENTITY=blaise-ai
WANDB_PROJECT=corsaire-1
WANDB_EXP_NAME=corsaire-1-research-preview
HF_REPO_ID=BlaiseAI/corsaire-1-research-preview
HF_UPLOAD_INTERVAL=500
HF_UPLOAD_RETAIN=1
```

`FASTPATH_STRICT=1` makes the launcher fail before training if the old
memory-conservative paths are accidentally re-enabled: StreamBP, LM-head
StreamBP, StreamBP replay/offload sub-switches, activation offload, full
recompute, activation ECO, allocator trim passes,
DSA deferred query gradients, smaller split-QK K/V backward chunks,
non-measured DSA row backward warp counts, DSA teacher-score recompute,
TurboQuant, HISA dense fallback, or non-measured HISA selector settings. Set
`FASTPATH_STRICT=0` only for an intentional diagnostic run.

The one queue exception for PP>1 is `MEGATRON_PIPELINE_QUEUE_OFFLOAD=1` with
depth 1. That is boundary queue offload only; it is kept on because the
expensive profile was TE/core-attention activation offload, not the pipeline
send/recv queue. The PP=1/no-pipeline diagnostic disables it because there is
no pipeline queue to offload.

`NVFP4_ACTIVATION_ECO=0` only disables the activation ECO/recompute path.
`FLASH_ADAMW_ECO=1` remains enabled and is part of the strict GCP profile.

The CE path is full-logit vocab-parallel cross entropy. The launcher keeps
`MEGATRON_CHUNKED_LM_HEAD_LOSS=0` and `MEGATRON_STREAMBP_FUSED_LCE=0`, so the
`LCE_*_VOCAB_SPLIT_SIZE` settings are inactive unless a diagnostic run
explicitly enables fused linear-CE or StreamBP fused-LCE.

`MEGATRON_DSA_TEACHER_SCORE_SCRATCH=1` is intentionally a memory-for-speed
choice. At the current 32k/topk1024 shape the one-GPU B200 benchmark moved the
HISA teacher-path DSA fwd+bwd from about 9.5s with score recompute to about
0.9s with score scratch at MBS4. If full training memory is too high, first try
MBS2 with score scratch before disabling the scratch path.

`MEGATRON_HISA_SELECTOR_BACKEND=bmm` is intentional for the current training
shape. On one B200 at `q=4096`, `sk=32768`, `batch=4`, `topk=1024`, the BMM
selector measured about 0.41s forward and 0.56s forward+backward. The available
DeepGEMM and packed cuBLASDx/CuTe extension paths load, but measured slower
for this shape, so strict mode rejects them unless explicitly disabled for a
diagnostic numerics run.

HF upload is shard-native. Every node watches its local `SAVE_CKPT` and uploads
the rank shard files it owns into the same raw MCore checkpoint folder on HF.
The retention owner, node rank 0 by default, waits for one marker per expected
node before promoting `latest_mcore_checkpoint.json` and deleting older remote
checkpoint folders. This avoids gathering a multi-TB checkpoint onto one node.

The old two-node failure was a DeepEP timeout during StreamBP replay. The fleet
path disables StreamBP by default because the larger TP/PP/EP shape should buy
back enough memory to avoid forward replay during backward. For dispatcher
isolation, all-to-all is still available:

```bash
MOE_TOKEN_DISPATCHER_TYPE=alltoall \
DRY_RUN=1 \
handoff/gcp-a4/scripts/launch_deepseek_nvfp4.sh
```

The intended throughput path is flex/DeepEP:

```bash
MOE_TOKEN_DISPATCHER_TYPE=flex \
MOE_FLEX_DISPATCHER_BACKEND=deepep \
MEGATRON_DEEPEP_EXPERT_MAJOR_DISPATCH=1 \
MEGATRON_DEEPEP_EXPERT_MAJOR_COMBINE=0 \
handoff/gcp-a4/scripts/launch_deepseek_nvfp4.sh
```

To launch the full fleet dashboard/training path:

```bash
ACTION=start handoff/gcp-a4/scripts/launch_fleet_tmux.sh
```

To use the same dashboard path for model conversion:

```bash
ACTION=start SESSION=gcp-a4-convert \
  REMOTE_ENTRYPOINT=handoff/gcp-a4/scripts/launch_conversion_deepseek_nvfp4.sh \
  handoff/gcp-a4/scripts/launch_fleet_tmux.sh
```

After conversion, validate selected tensors before training:

```bash
uv run --no-sync python tools/validate_blaise_deepseek_v32_reap_conversion.py \
  --hf-model-id BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4 \
  --checkpoint "$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron_tp8_pp1_ep8"
```

After checkpoint fanout, verify bitwise parity across nodes:

```bash
HASH_WORKERS=128 handoff/gcp-a4/scripts/verify_checkpoint_parity.sh launch
handoff/gcp-a4/scripts/verify_checkpoint_parity.sh status
handoff/gcp-a4/scripts/verify_checkpoint_parity.sh compare
```

## Network Defaults

The current rank-0 node exposes `eth0`, `eth1`, and `gpu0rdma0` through
`gpu7rdma0`, with InfiniBand devices `mlx5_0` through `mlx5_7`.

The GCP launcher defaults to:

```bash
GLOO_SOCKET_IFNAME=eth0
NCCL_SOCKET_IFNAME=eth0
NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

The branch's `DISTRIBUTED_BACKEND=ncclx` path has separate topology
requirements and will set `NCCL_SOCKET_IFNAME=gpu0rdma0` when appropriate.
Stay on plain `DISTRIBUTED_BACKEND=nccl` until the basic NCCL/DeepEP path is
validated on this GCP image.
