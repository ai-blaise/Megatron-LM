#!/bin/bash
# Multi-GPU NVFP4 + SFT + finetune smoke test
# Single node - direct torchrun (no SLURM)

set -e

# ======================
# Path Setup
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${SCRIPT_DIR}/../.."
cd ${MEGATRON_DIR}

CHECKPOINT_PATH=${1:-"$HOME/checkpoints/sft_llama_nvfp4_minimal"}
TENSORBOARD_LOGS_PATH=${2:-"$HOME/tensorboard_logs/sft_llama_nvfp4_minimal"}

mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$TENSORBOARD_LOGS_PATH"

# ======================
# Environment variables for performance tuning
# ======================
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# ======================
# Distributed Setup (single node multi-GPU)
# ======================
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}

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
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 32
    --train-iters 5
    --seq-length 8192
    --max-position-embeddings 8192
    --bf16
    --log-interval 10
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# ======================
# NVFP4 + Precision-Aware
# ======================
DTYPE_ARGS=(
    --fp4-format e2m1
    --fp4-recipe nvfp4
    --fp4-param-gather
)

PRECISION_AWARE_ARGS=(
    --use-precision-aware-optimizer
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
)

# ======================
# SFT + finetune
# ======================
SFT_ARGS=(
    --sft
    --finetune
    --sft-tokenizer-prompt-format nemotron-h-aligned
    --sft-mock-dataset-config-json '{"mode": "distribution", "num_samples": 100}'
)

# ======================
# Data Args (mock-data for smoke test)
# Uses mock-data mode to bypass tokenizer issues for quick smoke test
# For real data with tokenizer, use run_sft.sh or run_sft_deepseek_nvfp4.sh
# ======================
DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 128256
    --split '99,1,0'
    --num-workers 1
)

# ======================
# TensorBoard
# ======================
TENSORBOARD_ARGS=(
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --log-throughput
    --log-memory-to-tensorboard
    --tensorboard-log-interval 10
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
torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${PRECISION_AWARE_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TENSORBOARD_ARGS[@]} \
    ${CKPT_ARGS[@]}