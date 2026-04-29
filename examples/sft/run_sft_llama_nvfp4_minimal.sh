#!/bin/bash
# Multi-GPU NVFP4 + SFT + finetune smoke test
# Single node - direct torchrun (no SLURM)
#
# Usage:
#   OPTIMIZER=adam      ./run_sft_llama_nvfp4_minimal.sh   # TE FusedAdam + precision-aware (default)
#   OPTIMIZER=flash_adamw ./run_sft_llama_nvfp4_minimal.sh # FlashAdamW

set -e

# ======================
# Path Setup
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${SCRIPT_DIR}/../.."
cd ${MEGATRON_DIR}

OPTIMIZER=${OPTIMIZER:-adam}
USE_ECO=${USE_ECO:-1}
PROBE_OPTIMIZER=${PROBE_OPTIMIZER:-0}
if [[ "$PROBE_OPTIMIZER" == "1" ]]; then
    export MEGATRON_OPTIMIZER_STEP_PROBE=1
fi
SPINQUANT=${SPINQUANT:-0}
SPINQUANT_MODE=${SPINQUANT_MODE:-random}
SPINQUANT_ROTATION_PATH=${SPINQUANT_ROTATION_PATH:-}
SPINQUANT_FUSE_WEIGHTS=${SPINQUANT_FUSE_WEIGHTS:-0}

# Suffix so ECC and ECO flash_adamw runs don't share a checkpoint dir.
if [[ "$OPTIMIZER" == "flash_adamw" ]]; then
    RUN_TAG="${OPTIMIZER}_$([[ "$USE_ECO" == "1" ]] && echo eco || echo ecc)"
else
    RUN_TAG="$OPTIMIZER"
fi

CHECKPOINT_PATH=${1:-"$HOME/checkpoints/sft_llama_nvfp4_minimal_${RUN_TAG}"}
TENSORBOARD_LOGS_PATH=${2:-"$HOME/tensorboard_logs/sft_llama_nvfp4_minimal"}

mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$TENSORBOARD_LOGS_PATH"
mkdir -p snapshots

# ======================
# Environment variables for performance tuning
# ======================
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
# Single-node NVLink: override cluster-level NCCL env vars that assume
# multi-node IB fabric.
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_NET="Socket"
export NCCL_SOCKET_IFNAME=eth0
unset NCCL_ALGO
unset NCCL_PROTO
unset NCCL_IB_HCA
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_READ
unset NCCL_IB_GID_INDEX
unset NCCL_IB_SPLIT_DATA_ON_QPS
unset NCCL_IB_QPS_PER_CONNECTION
unset NCCL_IB_AR_THRESHOLD
unset NCCL_IB_PCI_RELAXED_ORDERING
unset NCCL_IB_RETRY_CNT
unset NCCL_IB_TIMEOUT
unset NCCL_CROSS_NIC
unset NCCL_TOPO_DUMP_FILE

# ======================
# Distributed Setup (single node multi-GPU)
# ======================
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ======================
# Model Args (LLaMA 8B)
# ======================
MODEL_ARGS=(
    --use-mcore-models
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --kv-channels 128
    --seq-length 8192
    --max-position-embeddings 8192
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --init-method-std 0.0134
)

# ======================
# Model Parallelism Args
# ======================
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
)

# ======================
# Training Args
# ======================
TRAIN_ITERS=${TRAIN_ITERS:-200}

TRAINING_ARGS=(
    --optimizer $OPTIMIZER
    --micro-batch-size 1
    --global-batch-size 32
    --train-iters $TRAIN_ITERS
    --lr 3e-5
    --min-lr 1e-6
    --lr-decay-style cosine
    --lr-warmup-iters 2
    --seq-length 8192
    --max-position-embeddings 8192
    --bf16
    --log-interval 1
    --eval-iters 1
    --eval-interval 1000
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# ======================
# NVFP4
# ======================
DTYPE_ARGS=(
    --fp4-format e2m1
    --fp4-recipe nvfp4
    --fp4-param-gather
)

# ======================
# SpinQuant
# ======================
SPINQUANT_ARGS=()
if [[ "$SPINQUANT" == "1" ]]; then
    SPINQUANT_ARGS+=(
        --spinquant
        --spinquant-mode "$SPINQUANT_MODE"
        --spinquant-w-bits 4
        --spinquant-a-bits 4
        --spinquant-k-bits 4
        --spinquant-v-bits 4
    )
    if [[ "$SPINQUANT_FUSE_WEIGHTS" == "1" ]]; then
        SPINQUANT_ARGS+=(--spinquant-fuse-weights)
    fi
    if [[ -n "$SPINQUANT_ROTATION_PATH" ]]; then
        SPINQUANT_ARGS+=(--spinquant-rotation-path "$SPINQUANT_ROTATION_PATH")
    fi
fi

# ======================
# Optimizer-specific args
# flash_adamw and --use-precision-aware-optimizer are mutually exclusive.
# ======================
OPTIM_EXTRA_ARGS=()
if [[ "$OPTIMIZER" == "flash_adamw" ]]; then
    if [[ "$USE_ECO" == "1" ]]; then
        # ECO: error-compensating optimization eliminates master weights by
        # feeding FP32→NVFP4 quantization error back through momentum.
        OPTIM_EXTRA_ARGS+=(--flash-adamw-eco)
    fi
else
    OPTIM_EXTRA_ARGS+=(
        --use-precision-aware-optimizer
        --exp-avg-dtype bf16
        --exp-avg-sq-dtype bf16
    )
fi

# ======================
# No SFT flags — our optimizer changes are orthogonal to data format.
# Using standard pretrain with mock data for clean profiling.
# ======================
SFT_ARGS=()

# ======================
# Data Args — wikitext-103 tokenized with Llama 3.2 tokenizer
# (byte-identical BPE to Llama 3.1-8B, non-gated)
# ======================
DATA_PATH=${DATA_PATH:-"$HOME/datasets/wikitext/wikitext103_llama3_text_document"}
DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model meta-llama/Llama-3.2-1B
    --split '999,1,0'
    --num-workers 1
)

# ======================
# Profiling + Logging
# ======================
EVAL_AND_LOGGING_ARGS=(
    --tensorboard-dir "${TENSORBOARD_LOGS_PATH}/${OPTIMIZER}"
    --log-throughput
    --log-memory-to-tensorboard
    --tensorboard-log-interval 1
    --record-memory-history
    --memory-snapshot-path "snapshots/sft_nvfp4_${OPTIMIZER}.pickle"
)

# ======================
# PyTorch Profiler (Chrome trace)
# ======================
PROFILING_ARGS=(
    --profile
    --use-pytorch-profiler
    --profile-step-start 3
    --profile-step-end 5
    --profile-ranks 0
)

# ======================
# Checkpointing
# ======================
CKPT_ARGS=(
    --save "$CHECKPOINT_PATH"
    --save-interval 1000
    --load "$CHECKPOINT_PATH"
)

# ======================
# LAUNCH
# ======================
uv run torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${SPINQUANT_ARGS[@]} \
    ${OPTIM_EXTRA_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${PROFILING_ARGS[@]} \
    ${CKPT_ARGS[@]}
