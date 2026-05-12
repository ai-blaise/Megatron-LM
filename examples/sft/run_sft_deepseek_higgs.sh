#!/bin/bash
# DeepSeek-V3.2 MoE SFT with 2-bit HIGGS fake-quant on the dense MLA latent KV.
#
# Override note: the BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-...
# checkpoint encodes a 4-bit token-wise KV scheme (the "KV4" in the model
# name). This script REPLACES that scheme with 2-bit HIGGS fake-quant applied
# at the post-LayerNorm 512-dim MLA latent. SpinQuant K/V bits are left at
# their defaults but the SpinQuant KV path is not enabled, so only HIGGS runs
# on the dense KV. The DSA Indexer's IndexerK8 path is unaffected (out of
# scope per project specification).
#
# Slot layout (258 B / token) is 16 B / token smaller than the 2.5-bit
# TurboQuant slot (274 B / token). HIGGS uses the public AquaKV EDEN2-16
# codebook (4 bits per pair = 2 bits per scalar) plus a single fp16 per-token
# block scale and a 128 B bf16 rope passthrough. The codebook does not require
# Lloyd-Max calibration, so adopting HIGGS is essentially free at training
# start-up cost.
#
# To run baseline NVFP4 weights without HIGGS, use run_sft_deepseek_nvfp4.sh.
# To run 2.5-bit TurboQuant instead, use run_sft_deepseek_turboquant.sh.

#SBATCH --job-name=deepseek_higgs_sft
#SBATCH --nodes=${NNODES:-1}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

set -e

# ======================
# Environment
# ======================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET=TCP
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=3600
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=False

# ======================
# Path Setup
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${SCRIPT_DIR}/../.."
cd ${MEGATRON_DIR}

CHECKPOINT_PATH=${1:-"$HOME/checkpoints/sft_deepseek_higgs"}
TENSORBOARD_LOGS_PATH=${2:-"$HOME/tensorboard_logs/sft_deepseek_higgs"}

mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$TENSORBOARD_LOGS_PATH"

# ======================
# Distributed Setup
# ======================
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST 2>/dev/null | head -1)
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ======================
# Model Args (DeepSeek-V3.2 MoE)
# ======================
MODEL_ARGS=(
    --use-mcore-models
    --num-layers 61
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --kv-channels 128
    --seq-length 4096
    --max-position-embeddings 163840
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

# ======================
# MLA Args (Multi-Latent Attention) — REQUIRED for DSA
# DSA builds ON TOP of MLA, not a replacement
# ======================
MLA_ARGS=(
    --multi-latent-attention
    --kv-lora-rank 512
    --q-lora-rank 1536
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
)

# ======================
# DSA Args (DeepSeek Sparse Attention) — DeepSeek-V3.2 specific
# ======================
DSA_ARGS=(
    --experimental-attention-variant dsa
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk 2048
    --dsa-indexer-loss-coeff 0.01
)

# ======================
# MoE Args
# ======================
MODEL_ARGS+=(
    --num-experts 128
    --moe-router-topk 8
    --moe-layer-freq $(python3 -c "print('[' + ','.join(['0']*3 + ['1']*58) + ']')")
    --moe-ffn-hidden-size 2048
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-aux-loss-coeff 1e-4
    --moe-shared-expert-intermediate-size 2048
    --moe-token-dispatcher-type alltoall
)

# ======================
# Training Args
# ======================
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 32
    --train-samples 32000000
    --lr-decay-samples 31968645
    --lr-warmup-samples 31348
    --lr 5.0e-6
    --min-lr 1.0e-7
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.010
    --log-interval 10
)

# ======================
# Parallelism
# ======================
TP=8
EP=1
PP=1
CP=1

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP
    --expert-tensor-parallel-size 1
    --expert-model-parallel-size $EP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --sequence-parallel
)

# ======================
# NVFP4 + Precision-Aware
# ======================
DTYPE_ARGS=(
    --fp4-format e2m1
    --fp4-recipe nvfp4
    --fp4-param-gather
)

# ======================
# HIGGS — 2-bit fake-quant on the dense MLA-latent KV
# Replaces the model's KV4 scheme. The EDEN2-16 codebook is a public AquaKV
# constant (see ``arXiv:2501.19392``); every rank constructs identical
# buffers locally without collective communication.
# Mutually exclusive with --turboquant-kv-enabled.
# ======================
HIGGS_ARGS=(
    --enable-higgs-dense-2bit-kv-cache
    --higgs-kv-preset dense_2bit
)

PRECISION_AWARE_ARGS=(
    --use-precision-aware-optimizer
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
)

# ======================
# Optimizer
# ======================
TRAINING_ARGS+=(
    --use-distributed-optimizer
    --no-gradient-accumulation-fusion
    --reset-position-ids
    --reset-attention-mask
    --eod-mask-loss
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
# Tokenizer
# ======================
TOKENIZER_MODEL=${3:-"deepseek-ai/DeepSeek-V3.2"}

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
)

# ======================
# Data (Real)
# ======================
DATA_PATH=${HOME}/data/sft/swe_rebench_v2_data

DATA_ARGS=(
    --data-path $DATA_PATH
    --split 100,0,0
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --num-workers 1
    --vocab-size 128256
)

# ======================
# TensorBoard + Profiling
# ======================
TENSORBOARD_ARGS=(
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --log-throughput
    --log-memory-to-tensorboard
    --log-l2-norm-grad-to-tensorboard
    --tensorboard-log-interval 10
)

PROFILING_ARGS=(
    --profile
    --profile-step-start 4
    --profile-step-end 6
)

# ======================
# Checkpointing
# ======================
CKPT_ARGS=(
    --save-interval 500
    --eval-interval 100
    --eval-iters 10
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --distributed-timeout-minutes 60
    --ckpt-format torch_dist
    --auto-detect-ckpt-format
)

# ======================
# LAUNCH (SLURM + torchrun)
# ======================
srun --mpi=pmix -l \
    torchrun ${DISTRIBUTED_ARGS[@]} \
        pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${MLA_ARGS[@]} \
        ${DSA_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${DTYPE_ARGS[@]} \
        ${HIGGS_ARGS[@]} \
        ${PRECISION_AWARE_ARGS[@]} \
        ${SFT_ARGS[@]} \
        ${TOKENIZER_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${TENSORBOARD_ARGS[@]} \
        ${PROFILING_ARGS[@]} \
        ${CKPT_ARGS[@]}
