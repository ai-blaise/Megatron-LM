#!/bin/bash
# Single GPU NVFP4 + SFT + finetune smoke test
# No SLURM - direct torchrun

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
# Distributed Setup
# ======================
GPUS_PER_NODE=1

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
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
# Training Args
# ======================
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1
    --train-iters 5
    --seq-length 8192
    --max-position-embeddings 8192
    --bf16
    --log-interval 10
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
)

# ======================
# Test Parquet Data (for SFTDataset smoke test)
# Uses mock-data mode to bypass tokenizer issues for quick smoke test
# For real data with tokenizer, use run_sft.sh or run_sft_deepseek_nvfp4.sh
# ======================
TEST_PARQUET_PATH="${MEGATRON_DIR}/tests/unit_tests/test_data/sft_test_conversations.parquet"

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
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${PRECISION_AWARE_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TENSORBOARD_ARGS[@]} \
    ${CKPT_ARGS[@]}