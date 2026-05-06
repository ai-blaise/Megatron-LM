#!/bin/bash
# DeepSeek-V3.2 REAP NVFP4 W4A4KV4 SFT on 16x B200.
#
# This script expects --load to point at a Megatron torch_dist checkpoint. The
# BlaiseAI Hugging Face checkpoint still needs conversion before training.

#SBATCH --job-name=deepseek_v32_reap_sft
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

set -euo pipefail

# ======================
# Environment
# ======================
USE_MEGATRON_FSDP="${USE_MEGATRON_FSDP:-0}"
if [[ "$USE_MEGATRON_FSDP" == "1" ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
else
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
fi
if [[ "$USE_MEGATRON_FSDP" == "1" && "$CUDA_DEVICE_MAX_CONNECTIONS" == "1" ]]; then
    echo "USE_MEGATRON_FSDP=1 requires CUDA_DEVICE_MAX_CONNECTIONS to be unset or greater than 1" >&2
    exit 1
fi
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MEGATRON_DSA_TRITON="${MEGATRON_DSA_TRITON:-1}"
export MEGATRON_DSA_STREAMING_INDEXER_TOPK="${MEGATRON_DSA_STREAMING_INDEXER_TOPK:-1}"
export MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE="${MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE:-2048}"
export MEGATRON_FLASH_ADAMW_NVFP4_IMMEDIATE_CAST="${MEGATRON_FLASH_ADAMW_NVFP4_IMMEDIATE_CAST:-1}"

# ======================
# Path setup
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${MEGATRON_DIR:-"${SCRIPT_DIR}/../.."}"
cd "$MEGATRON_DIR"

if [[ -z "${CUDA_HOME:-}" && -x "$MEGATRON_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc" ]]; then
    export CUDA_HOME="$MEGATRON_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -z "${CC:-}" && -x /usr/bin/gcc ]]; then
    export CC=/usr/bin/gcc
fi
if [[ -z "${CXX:-}" && -x /usr/bin/g++ ]]; then
    export CXX=/usr/bin/g++
fi

MODEL_ID="${MODEL_ID:-BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$MODEL_ID}"

LOAD_CKPT="${LOAD_CKPT:-"$HOME/checkpoints/deepseek_v32_reap_megatron"}"
SAVE_CKPT="${SAVE_CKPT:-"$HOME/checkpoints/sft_deepseek_v32_reap_nvfp4"}"
DATA_PATH="${DATA_PATH:-"$HOME/data/sft/blaise-sft-training-mix/nemotron-full-family.jsonl"}"
TENSORBOARD_LOGS_PATH="${TENSORBOARD_LOGS_PATH:-"$HOME/tensorboard_logs/sft_deepseek_v32_reap_nvfp4"}"

mkdir -p "$SAVE_CKPT" "$TENSORBOARD_LOGS_PATH"

# ======================
# Distributed setup
# ======================
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${SLURM_NNODES:-${NNODES:-2}}"
NODE_RANK="${SLURM_NODEID:-${NODE_RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-}"
if [[ -z "$MASTER_ADDR" && -n "${SLURM_JOB_NODELIST:-}" ]]; then
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
fi
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# ======================
# Parallelism
# ======================
# Default 16-GPU shape is TP=4 * PP=2 with CP=1, leaving DP=2.
# CP can be overridden, but CP=2 with PP=2 is weight/optimizer-memory heavy for this model.
TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-1}"
EP="${EP:-4}"
ETP="${ETP:-1}"
if [[ -z "${DECODER_FIRST_PIPELINE_NUM_LAYERS:-}" ]]; then
    if [[ "$PP" -eq 4 ]]; then
        DECODER_FIRST_PIPELINE_NUM_LAYERS=16
    else
        DECODER_FIRST_PIPELINE_NUM_LAYERS=31
    fi
fi

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --context-parallel-size "$CP"
    --expert-model-parallel-size "$EP"
    --expert-tensor-parallel-size "$ETP"
)

if [[ "${SEQUENCE_PARALLEL:-auto}" == "1" || ( "${SEQUENCE_PARALLEL:-auto}" == "auto" && "$TP" -gt 1 ) ]]; then
    MODEL_PARALLEL_ARGS+=(--sequence-parallel)
fi

if [[ "$PP" -gt 1 && -n "$DECODER_FIRST_PIPELINE_NUM_LAYERS" ]]; then
    MODEL_PARALLEL_ARGS+=(--decoder-first-pipeline-num-layers "$DECODER_FIRST_PIPELINE_NUM_LAYERS")
fi
if [[ "$PP" -gt 1 && -n "${DECODER_LAST_PIPELINE_NUM_LAYERS:-}" ]]; then
    MODEL_PARALLEL_ARGS+=(--decoder-last-pipeline-num-layers "$DECODER_LAST_PIPELINE_NUM_LAYERS")
fi

# ======================
# Model args
# ======================
SEQ_LENGTH="${SEQ_LENGTH:-32768}"
MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-([0]*3+[1]*58)}"

MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --num-layers 61
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --kv-channels 128
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings 163840
    --position-embedding-type rope
    --rope-type yarn
    --rotary-base 10000
    --rotary-percent 1.0
    --rotary-scaling-factor 40
    --mscale 1.0
    --mscale-all-dim 1.0
    --no-rope-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --init-method-std 0.02
    --attention-backend fused
    --attention-softmax-in-fp32
    --qk-layernorm
    --attention-output-gate
    --gated-norm
    --gated-norm-rank 16
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --bf16
)

# ======================
# MLA + DSA
# ======================
MLA_ARGS=(
    --multi-latent-attention
    --kv-lora-rank 512
    --q-lora-rank 1536
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
)

DSA_ARGS=(
    --experimental-attention-variant dsa
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk "${DSA_INDEXER_TOPK:-2048}"
    --dsa-indexer-loss-coeff "${DSA_INDEXER_LOSS_COEFF:-0.0}"
)

# ======================
# MoE
# ======================
MOE_ARGS=(
    --num-experts 128
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 8
    --moe-router-topk-scaling-factor 2.5
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 1e-4
    --moe-token-dispatcher-type alltoall
)

if [[ "${MOE_GROUPED_GEMM:-1}" == "1" ]]; then
    MOE_ARGS+=(--moe-grouped-gemm)
fi
if [[ "${MOE_PERMUTE_FUSION:-1}" == "1" ]]; then
    MOE_ARGS+=(--moe-permute-fusion)
fi
if [[ "${MOE_PER_LAYER_LOGGING:-1}" == "1" ]]; then
    MOE_ARGS+=(--moe-per-layer-logging)
fi

# ======================
# NVFP4, SpinQuant, TurboQuant, IndexCache
# ======================
DTYPE_ARGS=(
    --fp4-format e2m1
    --fp4-recipe nvfp4
    --fp4-param-gather
)

SPINQUANT_ARGS=()
if [[ "${SPINQUANT:-1}" == "1" ]]; then
    SPINQUANT_ARGS+=(
        --spinquant
        --spinquant-mode "${SPINQUANT_MODE:-random}"
        --spinquant-w-bits 4
        --spinquant-a-bits 4
        --spinquant-k-bits 4
        --spinquant-v-bits 4
    )
    if [[ "${SPINQUANT_FUSE_WEIGHTS:-0}" == "1" ]]; then
        SPINQUANT_ARGS+=(--spinquant-fuse-weights)
    fi
    if [[ -n "${SPINQUANT_ROTATION_PATH:-}" ]]; then
        SPINQUANT_ARGS+=(--spinquant-rotation-path "$SPINQUANT_ROTATION_PATH")
    fi
fi

TURBOQUANT_ARGS=()
if [[ "${TURBOQUANT:-1}" == "1" ]]; then
    TURBOQUANT_ARGS+=(
        --turboquant-kv-enabled
        --turboquant-kv-preset "${TURBOQUANT_KV_PRESET:-latent_2p5bit_nc}"
        --turboquant-kv-seed "${TURBOQUANT_KV_SEED:-0}"
    )
fi

INDEXCACHE_ARGS=()
if [[ "${INDEXCACHE:-1}" == "1" ]]; then
    INDEXCACHE_ARGS+=(
        --dsa-indexcache-quant-enabled
        --dsa-indexcache-quant-eps "${DSA_INDEXCACHE_QUANT_EPS:-1e-4}"
    )
fi

# ======================
# Training
# ======================
TRAINING_ARGS=(
    --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-16}"
    --train-samples "${TRAIN_SAMPLES:-32000000}"
    --lr-decay-samples "${LR_DECAY_SAMPLES:-31968645}"
    --lr-warmup-samples "${LR_WARMUP_SAMPLES:-31348}"
    --lr "${LR:-5.0e-6}"
    --min-lr "${MIN_LR:-1.0e-7}"
    --lr-decay-style cosine
    --clip-grad "${CLIP_GRAD:-1.0}"
    --weight-decay "${WEIGHT_DECAY:-0.0}"
    --adam-beta1 "${ADAM_BETA1:-0.9}"
    --adam-beta2 "${ADAM_BETA2:-0.95}"
    --log-interval "${LOG_INTERVAL:-10}"
    --empty-unused-memory-level "${EMPTY_UNUSED_MEMORY_LEVEL:-1}"
    --rerun-mode "${RERUN_MODE:-disabled}"
    --optimizer flash_adamw
    --flash-adamw-eco
    --use-distributed-optimizer
    --no-gradient-accumulation-fusion
)

if [[ "${OVERLAP_GRAD_REDUCE:-1}" == "1" ]]; then
    TRAINING_ARGS+=(--overlap-grad-reduce)
fi
if [[ "${OVERLAP_PARAM_GATHER:-1}" == "1" ]]; then
    TRAINING_ARGS+=(--overlap-param-gather)
fi
if [[ "${GRAD_REDUCE_IN_BF16:-1}" == "1" ]]; then
    TRAINING_ARGS+=(--grad-reduce-in-bf16)
fi
if [[ "${FLASH_ADAMW_COMPRESS_STATE_DICT:-1}" == "1" ]]; then
    TRAINING_ARGS+=(--flash-adamw-compress-state-dict)
fi

FSDP_ARGS=()
if [[ "$USE_MEGATRON_FSDP" == "1" ]]; then
    FSDP_ARGS+=(
        --use-megatron-fsdp
        --data-parallel-sharding-strategy "${DATA_PARALLEL_SHARDING_STRATEGY:-optim_grads_params}"
    )
    if [[ "${FSDP_DOUBLE_BUFFER:-0}" == "1" ]]; then
        FSDP_ARGS+=(--fsdp-double-buffer)
    fi
    if [[ -n "${SUGGESTED_COMMUNICATION_UNIT_SIZE:-}" ]]; then
        FSDP_ARGS+=(--suggested-communication-unit-size "$SUGGESTED_COMMUNICATION_UNIT_SIZE")
    fi
fi

RECOMPUTE_ARGS=()
if [[ "${RECOMPUTE:-1}" == "1" ]]; then
    RECOMPUTE_ARGS+=(
        --recompute-granularity "${RECOMPUTE_GRANULARITY:-full}"
        --recompute-method "${RECOMPUTE_METHOD:-uniform}"
        --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
    )
fi

# ======================
# SFT + tokenizer
# ======================
SFT_ARGS=(
    --sft
    --finetune
    --sft-tokenizer-prompt-format deepseek-v3.2
)

TOKENIZER_ARGS=(
    --tokenizer-type SFTTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --padded-vocab-size 129280
)

# ======================
# Data
# ======================
DATA_ARGS=(
    --data-path "$DATA_PATH"
    --split 100,0,0
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --num-workers "${NUM_WORKERS:-1}"
)

# ======================
# Logging, profiling, checkpointing
# ======================
TENSORBOARD_ARGS=(
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --log-throughput
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --tensorboard-log-interval "${TENSORBOARD_LOG_INTERVAL:-10}"
)

if [[ "${LOG_TIMERS_TO_TENSORBOARD:-1}" == "1" ]]; then
    TENSORBOARD_ARGS+=(
        --log-timers-to-tensorboard
        --timing-log-level "${TIMING_LOG_LEVEL:-1}"
        --timing-log-option "${TIMING_LOG_OPTION:-minmax}"
    )
fi
if [[ -n "${LOG_MEMORY_INTERVAL:-}" ]]; then
    TENSORBOARD_ARGS+=(--log-memory-interval "$LOG_MEMORY_INTERVAL")
fi
if [[ "${LOG_PARAMS_NORM:-0}" == "1" ]]; then
    TENSORBOARD_ARGS+=(--log-params-norm)
fi
if [[ "${LOG_NUM_ZEROS_IN_GRAD:-0}" == "1" ]]; then
    TENSORBOARD_ARGS+=(--log-num-zeros-in-grad)
fi
if [[ "${LOG_MAX_ATTENTION_LOGIT:-0}" == "1" ]]; then
    TENSORBOARD_ARGS+=(--log-max-attention-logit)
fi
if [[ "${LOG_ENERGY:-0}" == "1" ]]; then
    TENSORBOARD_ARGS+=(--log-energy)
fi

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    WANDB_ARGS+=(
        --wandb-project "$WANDB_PROJECT"
        --wandb-exp-name "${WANDB_EXP_NAME:-sft_deepseek_v32_reap_nvfp4}"
        --wandb-save-dir "${WANDB_SAVE_DIR:-"$SAVE_CKPT/wandb"}"
    )
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    fi
fi

PROFILING_ARGS=()
if [[ "${ENABLE_PROFILING:-0}" == "1" ]]; then
    PROFILING_ARGS+=(
        --profile
        --profile-step-start "${PROFILE_STEP_START:-4}"
        --profile-step-end "${PROFILE_STEP_END:-6}"
    )
fi

if [[ "$USE_MEGATRON_FSDP" == "1" ]]; then
    CKPT_FORMAT_VALUE="${CKPT_FORMAT:-fsdp_dtensor}"
else
    CKPT_FORMAT_VALUE="${CKPT_FORMAT:-torch_dist}"
fi

CKPT_ARGS=(
    --eval-interval "${EVAL_INTERVAL:-100}"
    --eval-iters "${EVAL_ITERS:-10}"
    --load "$LOAD_CKPT"
    --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES:-60}"
    --ckpt-format "$CKPT_FORMAT_VALUE"
    --auto-detect-ckpt-format
)
if [[ "${DISABLE_SAVE:-0}" != "1" ]]; then
    CKPT_ARGS+=(
        --save-interval "${SAVE_INTERVAL:-500}"
        --save "$SAVE_CKPT"
    )
fi
if [[ "${NO_SAVE_OPTIM:-0}" == "1" ]]; then
    CKPT_ARGS+=(--no-save-optim)
fi
if [[ "${NO_SAVE_RNG:-0}" == "1" ]]; then
    CKPT_ARGS+=(--no-save-rng)
fi

CMD=(
    uv run --no-sync torchrun
    "${DISTRIBUTED_ARGS[@]}"
    pretrain_gpt.py
    "${MODEL_ARGS[@]}"
    "${MLA_ARGS[@]}"
    "${DSA_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${MODEL_PARALLEL_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${FSDP_ARGS[@]}"
    "${RECOMPUTE_ARGS[@]}"
    "${DTYPE_ARGS[@]}"
    "${SPINQUANT_ARGS[@]}"
    "${TURBOQUANT_ARGS[@]}"
    "${INDEXCACHE_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${TENSORBOARD_ARGS[@]}"
    "${WANDB_ARGS[@]}"
    "${PROFILING_ARGS[@]}"
    "${CKPT_ARGS[@]}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" && "${USE_SRUN:-1}" == "1" ]]; then
    srun --mpi=pmix -l "${CMD[@]}"
else
    "${CMD[@]}"
fi
