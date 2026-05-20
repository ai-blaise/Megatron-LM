# Launcher And Runtime

## Files

- tmux wrapper:
  `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
- actual training runner:
  `examples/sft/run_sft_deepseek_nvfp4.sh`

The tmux wrapper is convenient for the current 2-node GCP setup, but it is not
an AWS 4-node launcher as-is.

## What The Tmux Wrapper Does

The wrapper:

- creates a 2x2 tmux dashboard
- starts node0 locally
- starts node1 through SSH
- syncs changed repo files to one remote host only
- syncs only checkpoint metadata/common files, not full checkpoint shards
- hard-codes current GCP-style defaults unless overridden:
  - `REMOTE_HOST=sjpat@10.180.0.45`
  - `SSH_KEY=$HOME/.ssh/google_compute_engine`
  - `MASTER_ADDR=10.200.0.21`
  - `NCCL_SOCKET_IFNAME=gpu` inside the remote env block
  - `NCCL_IB_HCA=mlx5_0,...,mlx5_7`

For AWS 4 nodes, do not use it unchanged. Either:

- write a new 4-node tmux/SSH wrapper, or
- use Slurm/srun if the AWS cluster provides Slurm, or
- launch `examples/sft/run_sft_deepseek_nvfp4.sh` on all 4 nodes with
  `NNODES=4`, `NODE_RANK=0..3`, same `MASTER_ADDR`, and same env.

## Runner Behavior

The runner is the authoritative script. It supports:

- `torchrun`
- Slurm via `srun` if `SLURM_JOB_ID` is present
- `DISTRIBUTED_BACKEND=nccl` by default
- optional `DISTRIBUTED_BACKEND=ncclx` through TorchComms/NCCLX
- dry run with `DRY_RUN=1`
- prebuild of HISA, HIGGS, IndexCache, TurboQuant extensions
- W&B, tensorboard, ZCC, normal checkpointing

Render the command before training:

```bash
handoff/aws-b200-32gpu/scripts/render_launch_dry_run.sh
```

Or render through the runner directly:

```bash
DRY_RUN=1 \
NNODES=4 NODE_RANK=0 GPUS_PER_NODE=8 \
MASTER_ADDR=<rank0-private-ip> MASTER_PORT=29673 \
TP=4 PP=4 CP=1 EP=4 ETP=1 \
MICRO_BATCH_SIZE=4 GLOBAL_BATCH_SIZE=32 \
examples/sft/run_sft_deepseek_nvfp4.sh
```

## RDMA / Network Notes

Current default backend is plain `nccl`.

The `ncclx` path exists but is not the default. It requires:

- `torchcomms==0.2.0`
- `DISTRIBUTED_BACKEND=ncclx`
- `MEGATRON_USE_TORCHCOMMS=1`
- `MEGATRON_NCCLX_RDMA=1`
- a valid topology file:
  `NCCL_TOPO_FILE_PATH` or `$HOME/ncclx_topology.env`

The current `torchcomms` dependency is a local wheel path:

```toml
torchcomms = { path = "/home/sjpat/wheelhouse/torchcomms-0.2.0-cp312-cp312-linux_x86_64.whl" }
```

On AWS, either copy that wheel to the same path on every node or update
`pyproject.toml`/`uv.lock`.

For AWS EFA/IB/RDMA:

- override the GCP interface defaults
- verify `fi_info`, `ibv_devinfo`, NCCL tests, and DeepEP/NVSHMEM behavior
- make sure `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, and any EFA/NVSHMEM variables
  match the AWS image

Do not blindly carry over:

```bash
NCCL_SOCKET_IFNAME=gpu
NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

unless those are correct on the AWS nodes.

## DeepEP / NVSHMEM

DeepEP is installed in the live `.venv` from:

- `/tmp/DeepEP-v1.2.1`
- upstream remote: `https://github.com/deepseek-ai/DeepEP.git`
- commit: `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`

The local source has an uncommitted setup.py patch:

```diff
- include_dirs = ['csrc/']
+ cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
+ include_dirs = ['csrc/', f'{cuda_home}/include/cccl']
```

Without this, CUDA 13/CCCL builds may fail.

DeepEP should be built after `nvidia-nvshmem-cu13` is present. Confirm:

```bash
uv run --no-sync python - <<'PY'
import deep_ep, nvidia.nvshmem
print("deep_ep", deep_ep.__file__)
print("nvshmem ok")
PY
```

## Checkpoint / Data Placement

The 2-node wrapper only syncs checkpoint metadata/common files. It assumes full
checkpoint shards already exist at the same path on both nodes or on shared
storage.

For 4 AWS nodes:

- either use shared storage with the same mount path on all nodes, or
- pre-stage full checkpoint shards on all nodes, not just metadata
- pre-stage the dataset or put it on fast shared storage
- keep `LOAD_CKPT`, `SAVE_CKPT`, `DATA_PATH`, `TRITON_CACHE_DIR`, and
  `DG_JIT_CACHE_DIR` consistent

The converted Megatron checkpoint should be validated before training:

```bash
uv run --no-sync python tools/validate_blaise_deepseek_v32_reap_conversion.py \
  --hf-model BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4 \
  --megatron-checkpoint "$LOAD_CKPT"
```

Adjust arguments if the validation script interface changed.
