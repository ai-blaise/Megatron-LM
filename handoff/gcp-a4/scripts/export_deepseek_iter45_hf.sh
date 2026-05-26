#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Megatron-LM}"
cd "$REPO_DIR"

BASE="${LOAD_CKPT:-$HOME/checkpoints/sft_deepseek_v32_reap_spinquant_actkv_nvfp4_tp8_pp1_cp4_ep8}"
OUT="${HF_OUTPUT_PATH:-$HOME/checkpoints/corsaire-1-research-preview-hf-iter45}"
if [[ -z "${HF_SOURCE_MODEL_ID:-}" && -d "$HOME/models/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4" ]]; then
  SOURCE_MODEL="$HOME/models/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4"
else
  SOURCE_MODEL="${HF_SOURCE_MODEL_ID:-BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4}"
fi

# Keep rank 0 on the controller node so the HF checkpoint is written locally.
ordered_nodes=(
  instance-group-1-1jzl
  instance-group-1-064k
  instance-group-1-1pks
  instance-group-1-f3kb
  instance-group-1-g9z6
  instance-group-1-gdzt
  instance-group-1-rhwm
  instance-group-1-sqv8
  instance-group-1-t2gg
  instance-group-1-w6tc
  instance-group-1-wk5d
  instance-group-1-zj5n
)

host_short="$(hostname -s)"
node_rank=""
for idx in "${!ordered_nodes[@]}"; do
  if [[ "${ordered_nodes[$idx]}" == "$host_short" ]]; then
    node_rank="$idx"
    break
  fi
done

if [[ -z "$node_rank" ]]; then
  echo "Host $host_short is not in the fixed HF export node list" >&2
  exit 2
fi

export CUDA_HOME="${CUDA_HOME:-$REPO_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

if [[ -f "$HOME/.config/megatron/auth.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.config/megatron/auth.env"
  set +a
fi
if [[ -f "$HOME/.cache/huggingface/token" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

export NNODES="${NNODES:-12}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export WORLD_SIZE="$((NNODES * GPUS_PER_NODE))"
export NODE_RANK="$node_rank"
export SLURM_NODEID="$node_rank"
export MASTER_ADDR="${MASTER_ADDR:-10.142.0.43}"
export MASTER_PORT="${MASTER_PORT:-29847}"

export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export USE_NCCL_GIB="${USE_NCCL_GIB:-0}"
if [[ "$USE_NCCL_GIB" == "1" ]]; then
  if [[ -f /usr/local/gib/scripts/set_nccl_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/gib/scripts/set_nccl_env.sh
    export LD_LIBRARY_PATH="/usr/local/gib/lib64:${LD_LIBRARY_PATH:-}"
  else
    echo "WARNING: USE_NCCL_GIB=1 but /usr/local/gib/scripts/set_nccl_env.sh is missing" >&2
  fi
fi
if [[ -z "${MEGATRON_LOCAL_RANK_NCCL_HCA+x}" ]]; then
  export MEGATRON_LOCAL_RANK_NCCL_HCA=0
fi
export MEGATRON_NCCL_LOCAL_RANK_HCAS="${MEGATRON_NCCL_LOCAL_RANK_HCAS:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
if [[ "$MEGATRON_LOCAL_RANK_NCCL_HCA" == "1" ]]; then
  unset NCCL_IB_HCA
elif [[ "$USE_NCCL_GIB" != "1" ]]; then
  export NCCL_IB_HCA="${NCCL_IB_HCA:-$MEGATRON_NCCL_LOCAL_RANK_HCAS}"
fi

if ! ulimit -l unlimited 2>/dev/null; then
  echo "WARNING: could not raise memlock soft limit to unlimited" >&2
fi

export TP="${TP:-8}"
export PP="${PP:-1}"
export CP="${CP:-4}"
export EP="${EP:-8}"
export ETP="${ETP:-1}"
export USE_TP_PP_DP_MAPPING="${USE_TP_PP_DP_MAPPING:-1}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-auto}"
export ENABLE_VPP=0
export OVERLAP_P2P_COMM_WARMUP_FLUSH=0

export USE_MEGATRON_FSDP=1
export DATA_PARALLEL_SHARDING_STRATEGY="${DATA_PARALLEL_SHARDING_STRATEGY:-optim_grads_params}"
export FSDP_INIT_MODEL_WITH_META_DEVICE="${FSDP_INIT_MODEL_WITH_META_DEVICE:-1}"
export LOAD_CKPT="$BASE"
export SAVE_CKPT="$BASE"
export CKPT_FORMAT="${CKPT_FORMAT:-fsdp_dtensor}"
export CKPT_FULLY_PARALLEL_LOAD="${CKPT_FULLY_PARALLEL_LOAD:-1}"
export CKPT_FULLY_PARALLEL_LOAD_PROCESS_GROUP="${CKPT_FULLY_PARALLEL_LOAD_PROCESS_GROUP:-dp}"
export CKPT_FULLY_PARALLEL_LOAD_EXCHANGE_ALGO="${CKPT_FULLY_PARALLEL_LOAD_EXCHANGE_ALGO:-broadcast}"
export DISABLE_SAVE=1
export ENABLE_ZCC=0
export ENABLE_TENSORBOARD_LOGGING=0
export ENABLE_ONE_LOGGER=0
export WANDB_PROJECT=
export WANDB_FORCE_OFFLINE=1

export MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}"
export MOE_FLEX_DISPATCHER_BACKEND="${MOE_FLEX_DISPATCHER_BACKEND:-alltoall}"
export DSA_INDEXER_TOPK="${DSA_INDEXER_TOPK:-1024}"
export DSA_INDEXCACHE_HISA_COMPRESSION_RATIO="${DSA_INDEXCACHE_HISA_COMPRESSION_RATIO:-8.0}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-12}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-120}"

export TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-$HOME/logs/torchrun/hf_export_iter45_slurm-${SLURM_JOB_ID:-manual}/node${NODE_RANK}}"
export TORCHRUN_REDIRECTS="${TORCHRUN_REDIRECTS:-3}"
export TORCHRUN_TEE="${TORCHRUN_TEE:-0}"
export TORCHRUN_LOCAL_RANKS_FILTER="${TORCHRUN_LOCAL_RANKS_FILTER:-0}"
export EXTRA_MEGATRON_ARGS="--hf-output-path $OUT --hf-source-model-id $SOURCE_MODEL --hf-max-shard-size-gb ${HF_MAX_SHARD_SIZE_GB:-4} --hf-export-load-source-first --hf-export-load-non-strict"

if [[ "$NODE_RANK" == "0" ]]; then
  mkdir -p "$OUT"
fi

echo "[hf-export-entry] host=$host_short node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT out=$OUT" >&2
echo "[hf-export-entry] nccl_socket_ifname=$NCCL_SOCKET_IFNAME nccl_net=${NCCL_NET:-<default>} use_nccl_gib=$USE_NCCL_GIB ib_disable=$NCCL_IB_DISABLE local_rank_hca=$MEGATRON_LOCAL_RANK_NCCL_HCA hcas=$MEGATRON_NCCL_LOCAL_RANK_HCAS parent_NCCL_IB_HCA=${NCCL_IB_HCA:-<unset>} memlock=$(ulimit -Sl)/$(ulimit -Hl)" >&2

cmd="$(DRY_RUN=1 examples/sft/run_sft_deepseek_nvfp4.sh 2>&1 | sed -n '/^uv run --no-sync torchrun /p')"
cmd="${cmd/pretrain_gpt.py/tools\/export_blaise_megatron_to_hf.py}"
cmd="${cmd/ --overlap-p2p-communication-warmup-flush/}"
if [[ -z "$cmd" ]]; then
  echo "Failed to build HF export torchrun command" >&2
  exit 2
fi

eval "$cmd"
