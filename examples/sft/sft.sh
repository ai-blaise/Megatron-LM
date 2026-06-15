#!/bin/bash
# Unified SFT launcher with composable argument profiles.
#
# Examples:
#   DRY_RUN=1 bash examples/sft/sft.sh
#   MODEL_PROFILE=glm4_9b_omp LOAD_CKPT=/ckpts/glm4_9b_omp_mcore DATA_PATH=/data/sft.jsonl bash examples/sft/sft.sh
#   MODEL_PROFILE=glm4_9b_omp MEGATRON_CKPT=/ckpts/glm4_9b_omp_bridge_ckpt DATA_ROOT=/data/my_sft_jsonl bash examples/sft/sft.sh
#   MODEL_PROFILE=deepseek_v32_reap PRECISION_PROFILE=deepseek_nvfp4 PARALLEL_PROFILE=deepseek_tp4_pp2_ep4 bash examples/sft/sft.sh

set -euo pipefail

usage() {
    cat <<'EOF'
examples/sft/sft.sh profiles

Required for a real JSONL run:
  DATA_PATH=/path/to/sft.jsonl

Useful checkpoint/tokenizer variables:
  LOAD_CKPT=/path/to/megatron/checkpoint
  SAVE_CKPT=/path/to/output/checkpoint
  MODEL_ID=hf-or-local-model-id
  TOKENIZER_MODEL=hf-or-local-tokenizer
  TOKENIZER_REVISION=hf-tokenizer-commit

Bridge conversion aliases:
  MEGATRON_CKPT=/path/to/bridge/export/root
  BRIDGE_CKPT=/path/to/bridge/export/root
  DATA_ROOT=/path/to/bridge-style/jsonl/root

  If LOAD_CKPT is unset, MEGATRON_CKPT or BRIDGE_CKPT maps to:
    LOAD_CKPT=$MEGATRON_CKPT/iter_0000000

  If DATA_PATH is unset, DATA_ROOT maps to:
    DATA_PATH=$DATA_ROOT/training.jsonl

Profiles:
  MODEL_PROFILE:
    deepseek_v32_reap     DeepSeek V3.2 REAP MLA/DSA/MoE architecture
    checkpoint_args       Load architecture from checkpoint args
    glm4_9b_omp           BlaiseAI GLM-4-9B OMP checkpoint args + DeepSeek tokenizer
    glm45_air             Alias for glm4_9b_omp unless MODEL_ID is overridden
    llama3_8b             Dense Llama 3/3.1 8B-style model args
    qwen3_8b              Dense Qwen3 8B model args

  TOKENIZER_PROFILE:
    auto                  Pick from MODEL_PROFILE
    deepseek_v32          SFTTokenizer + deepseek-v3.2 prompt format
    sft_default           SFTTokenizer + tokenizer.chat_template
    nemotron_h            SFTTokenizer + nemotron-h-aligned
    identity              SFTTokenizer + identity

  DATA_PROFILE:
    jsonl_messages        JSONL records with a messages list
    hf_blaise_mix         Hugging Face dataset repo passed via DATA_PATH
    mock_sft              SFT mock data

  PRECISION_PROFILE:
    bf16                  BF16 only
    nvfp4                 NVFP4 W4A4 base flags
    nvfp4_spinquant       NVFP4 + SpinQuant W/A/K/V
    nvfp4_higgs           NVFP4 + Higgs dense 2-bit KV
    nvfp4_turboquant      NVFP4 + TurboQuant KV
    deepseek_nvfp4        NVFP4 + SpinQuant + Higgs + DSA IndexCache

  OPTIMIZER_PROFILE:
    adamw                 adam optimizer with distributed optimizer
    flash_adamw           flash_adamw with distributed optimizer
    precision_aware       adamw + precision-aware optimizer states
    flash_adamw_eco       flash_adamw + ECO

  PARALLEL_PROFILE:
    auto                  Small local-safe defaults unless model overrides them
    single_gpu            TP/PP/CP/EP all 1
    single_node_8gpu      8 GPUs, TP 1, PP 1, CP 1, EP 1
    two_node_8gpu         2 nodes, 8 GPUs per node, TP 8, PP 1, CP 1, EP 1
    deepseek_tp4_pp2_ep4  DeepSeek-ish 8 GPU shape
    deepseek_tp8_pp5_ep8  Large DeepSeek shape for multi-node jobs

  RUNTIME_PROFILE:
    local                 torchrun directly
    slurm                 launch through srun

  DEBUG_PROFILE:
    none                  Use TRAIN_ITERS/TRAIN_SAMPLES as configured
    dry_run               Print command and exit
    one_step              Force TRAIN_ITERS=1 and save/eval intervals small
    tokenizer_smoke       Mock SFT data, TRAIN_ITERS=1
EOF
}

if [[ "${LIST_PROFILES:-0}" == "1" || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${MEGATRON_DIR:-"${SCRIPT_DIR}/../.."}"
cd "$MEGATRON_DIR"

MODEL_PROFILE="${MODEL_PROFILE:-deepseek_v32_reap}"
TOKENIZER_PROFILE="${TOKENIZER_PROFILE:-auto}"
DATA_PROFILE="${DATA_PROFILE:-jsonl_messages}"
PRECISION_PROFILE="${PRECISION_PROFILE:-bf16}"
OPTIMIZER_PROFILE="${OPTIMIZER_PROFILE:-adamw}"
PARALLEL_PROFILE="${PARALLEL_PROFILE:-auto}"
RUNTIME_PROFILE="${RUNTIME_PROFILE:-local}"
DEBUG_PROFILE="${DEBUG_PROFILE:-none}"

case "$DEBUG_PROFILE" in
    none) ;;
    dry_run)
        DRY_RUN=1
        ;;
    one_step)
        TRAIN_ITERS="${TRAIN_ITERS:-1}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
        EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
        ;;
    tokenizer_smoke)
        DATA_PROFILE=mock_sft
        TRAIN_ITERS="${TRAIN_ITERS:-1}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
        EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
        ;;
    *)
        echo "Unknown DEBUG_PROFILE=$DEBUG_PROFILE" >&2
        usage >&2
        exit 2
        ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export SUPPRESS_UNBATCHED_P2P_WARN="${SUPPRESS_UNBATCHED_P2P_WARN:-1}"

if [[ -z "${CUDA_HOME:-}" && -x "$MEGATRON_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc" ]]; then
    export CUDA_HOME="$MEGATRON_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi

MODEL_ARGS=()
MLA_ARGS=()
DSA_ARGS=()
MOE_ARGS=()
TOKENIZER_ARGS=()
SFT_ARGS=()
DATA_ARGS=()
DTYPE_ARGS=()
SPINQUANT_ARGS=()
TURBOQUANT_ARGS=()
HIGGS_ARGS=()
INDEXCACHE_ARGS=()
TRAINING_ARGS=()
MODEL_PARALLEL_ARGS=()
CKPT_ARGS=()
TENSORBOARD_ARGS=()
WANDB_ARGS=()
PROFILING_ARGS=()
EXTRA_MEGATRON_ARGS_ARRAY=()

SEQ_LENGTH="${SEQ_LENGTH:-}"
MODEL_ID="${MODEL_ID:-}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-}"
LOAD_CKPT="${LOAD_CKPT:-}"
SAVE_CKPT="${SAVE_CKPT:-"$HOME/checkpoints/sft_${MODEL_PROFILE}"}"
TENSORBOARD_LOGS_PATH="${TENSORBOARD_LOGS_PATH:-"$HOME/tensorboard_logs/sft_${MODEL_PROFILE}"}"

BRIDGE_CKPT_ROOT="${MEGATRON_CKPT:-${BRIDGE_CKPT:-}}"
if [[ -z "$LOAD_CKPT" && -n "$BRIDGE_CKPT_ROOT" ]]; then
    LOAD_CKPT="${BRIDGE_CKPT_ROOT%/}/iter_0000000"
fi
if [[ -z "${DATA_PATH:-}" && -n "${DATA_ROOT:-}" ]]; then
    DATA_PATH="${DATA_ROOT%/}/training.jsonl"
fi

case "$MODEL_PROFILE" in
    deepseek|deepseek_v32|deepseek_v32_reap)
        MODEL_ID="${MODEL_ID:-BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4}"
        TOKENIZER_MODEL="${TOKENIZER_MODEL:-$MODEL_ID}"
        LOAD_CKPT="${LOAD_CKPT:-"$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron_tp8_pp1_ep8"}"
        SEQ_LENGTH="${SEQ_LENGTH:-32768}"
        MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-([0]*3+[1]*58)}"
        MODEL_ARGS+=(
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
        )
        MLA_ARGS+=(
            --multi-latent-attention
            --kv-lora-rank 512
            --q-lora-rank 1536
            --qk-head-dim 128
            --qk-pos-emb-head-dim 64
            --v-head-dim 128
        )
        DSA_ARGS+=(
            --experimental-attention-variant dsa
            --dsa-indexer-n-heads 64
            --dsa-indexer-head-dim 128
            --dsa-indexer-topk "${DSA_INDEXER_TOPK:-1024}"
            --dsa-indexer-loss-coeff "${DSA_INDEXER_LOSS_COEFF:-0.1}"
            --dsa-chunk-size "${DSA_CHUNK_SIZE:-8192}"
        )
        if [[ "${APPLY_ROPE_FUSION:-0}" != "1" ]]; then
            MODEL_ARGS+=(--no-rope-fusion)
        fi
        MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-flex}"
        MOE_ARGS+=(
            --num-experts 128
            --moe-layer-freq "$MOE_LAYER_FREQ"
            --moe-ffn-hidden-size 2048
            --moe-shared-expert-intermediate-size 2048
            --moe-router-load-balancing-type "${MOE_ROUTER_LOAD_BALANCING_TYPE:-seq_aux_loss}"
            --moe-router-topk 8
            --moe-router-topk-scaling-factor 2.5
            --moe-router-num-groups 8
            --moe-router-group-topk 4
            --moe-router-pre-softmax
            --moe-router-score-function sigmoid
            --moe-router-enable-expert-bias
            --moe-router-bias-update-rate "${MOE_ROUTER_BIAS_UPDATE_RATE:-1e-3}"
            --moe-router-expert-bias-update-method "${MOE_ROUTER_EXPERT_BIAS_UPDATE_METHOD:-sign}"
            --moe-router-quantile-bias-iters "${MOE_ROUTER_QUANTILE_BIAS_ITERS:-5}"
            --moe-router-quantile-bias-application "${MOE_ROUTER_QUANTILE_BIAS_APPLICATION:-delayed}"
            --moe-router-quantile-bias-warmup-steps "${MOE_ROUTER_QUANTILE_BIAS_WARMUP_STEPS:-0}"
            --moe-router-dtype fp32
            --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF:-0.0}"
            --moe-token-dispatcher-type "$MOE_TOKEN_DISPATCHER_TYPE"
        )
        if [[ "$MOE_TOKEN_DISPATCHER_TYPE" == "flex" ]]; then
            MOE_ARGS+=(--moe-flex-dispatcher-backend "${MOE_FLEX_DISPATCHER_BACKEND:-deepep}")
        fi
        if [[ "${MOE_GROUPED_GEMM:-1}" == "1" ]]; then
            MOE_ARGS+=(--moe-grouped-gemm)
        fi
        if [[ "${MOE_PERMUTE_FUSION:-1}" == "1" ]]; then
            MOE_ARGS+=(--moe-permute-fusion)
        fi
        if [[ "${MOE_PER_LAYER_LOGGING:-1}" == "1" ]]; then
            MOE_ARGS+=(--moe-per-layer-logging)
        fi
        ;;
    checkpoint_args)
        TOKENIZER_MODEL="${TOKENIZER_MODEL:-$MODEL_ID}"
        SEQ_LENGTH="${SEQ_LENGTH:-4096}"
        MODEL_ARGS+=(
            --use-mcore-models
            --use-checkpoint-args
            --no-use-tokenizer-model-from-checkpoint-args
            --seq-length "$SEQ_LENGTH"
        )
        if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
            MODEL_ARGS+=(--trust-remote-code)
        fi
        ;;
    glm4_9b_omp|glm4_9b|glm45_air|glm|glm4_5_air)
        MODEL_ID="${MODEL_ID:-BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP}"
        TOKENIZER_MODEL="${TOKENIZER_MODEL:-cerebras/DeepSeek-V3.2-REAP-345B-A37B}"
        TOKENIZER_REVISION="${TOKENIZER_REVISION:-4fd8e8c3e08442c4a6dde6dd3fa3dac481a0205b}"
        SEQ_LENGTH="${SEQ_LENGTH:-4096}"
        MODEL_ARGS+=(
            --use-mcore-models
            --use-checkpoint-args
            --no-use-tokenizer-model-from-checkpoint-args
            --seq-length "$SEQ_LENGTH"
        )
        if [[ "${TRUST_REMOTE_CODE:-1}" == "1" ]]; then
            MODEL_ARGS+=(--trust-remote-code)
        fi
        ;;
    llama3_8b|llama3_1_8b|llama_3_8b)
        MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
        TOKENIZER_MODEL="${TOKENIZER_MODEL:-$MODEL_ID}"
        SEQ_LENGTH="${SEQ_LENGTH:-8192}"
        MODEL_ARGS+=(
            --use-mcore-models
            --num-layers 32
            --hidden-size 4096
            --ffn-hidden-size 14336
            --num-attention-heads 32
            --group-query-attention
            --num-query-groups 8
            --kv-channels 128
            --seq-length "$SEQ_LENGTH"
            --max-position-embeddings "${MAX_POSITION_EMBEDDINGS:-8192}"
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
        ;;
    qwen3_8b|qwen_3_8b)
        MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
        TOKENIZER_MODEL="${TOKENIZER_MODEL:-$MODEL_ID}"
        SEQ_LENGTH="${SEQ_LENGTH:-4096}"
        MODEL_ARGS+=(
            --use-mcore-models
            --num-layers 36
            --hidden-size 4096
            --ffn-hidden-size 12288
            --num-attention-heads 32
            --group-query-attention
            --num-query-groups 8
            --kv-channels 128
            --seq-length "$SEQ_LENGTH"
            --max-position-embeddings "${MAX_POSITION_EMBEDDINGS:-40960}"
            --position-embedding-type rope
            --rotary-base 1000000
            --rotary-percent 1.0
            --attention-dropout 0.0
            --hidden-dropout 0.0
            --swiglu
            --normalization RMSNorm
            --norm-epsilon 1e-6
            --attention-backend fused
            --attention-softmax-in-fp32
            --qk-layernorm
            --no-masked-softmax-fusion
            --no-rope-fusion
            --no-bias-swiglu-fusion
            --untie-embeddings-and-output-weights
            --disable-bias-linear
            --make-vocab-size-divisible-by "${MAKE_VOCAB_SIZE_DIVISIBLE_BY:-1187}"
        )
        ;;
    *)
        echo "Unknown MODEL_PROFILE=$MODEL_PROFILE" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ "$TOKENIZER_PROFILE" == "auto" ]]; then
    case "$MODEL_PROFILE" in
        deepseek|deepseek_v32|deepseek_v32_reap|glm4_9b_omp|glm4_9b|glm45_air|glm|glm4_5_air) TOKENIZER_PROFILE=deepseek_v32 ;;
        *) TOKENIZER_PROFILE=sft_default ;;
    esac
fi

SFT_ARGS+=(--sft)
if [[ "${SFT_FINETUNE:-1}" == "1" ]]; then
    SFT_ARGS+=(--finetune)
fi

case "$TOKENIZER_PROFILE" in
    deepseek_v32|deepseek-v32|deepseek-v3.2)
        SFT_ARGS+=(--sft-tokenizer-prompt-format deepseek-v3.2)
        TOKENIZER_ARGS+=(--tokenizer-type SFTTokenizer --tokenizer-model "$TOKENIZER_MODEL")
        if [[ -n "$TOKENIZER_REVISION" ]]; then
            TOKENIZER_ARGS+=(--tokenizer-revision "$TOKENIZER_REVISION")
        fi
        TOKENIZER_ARGS+=(--padded-vocab-size "${PADDED_VOCAB_SIZE:-129280}")
        ;;
    sft_default|default)
        SFT_ARGS+=(--sft-tokenizer-prompt-format default)
        TOKENIZER_ARGS+=(--tokenizer-type SFTTokenizer --tokenizer-model "$TOKENIZER_MODEL")
        if [[ -n "$TOKENIZER_REVISION" ]]; then
            TOKENIZER_ARGS+=(--tokenizer-revision "$TOKENIZER_REVISION")
        fi
        ;;
    nemotron_h|nemotron-h-aligned)
        SFT_ARGS+=(--sft-tokenizer-prompt-format nemotron-h-aligned)
        TOKENIZER_ARGS+=(--tokenizer-type SFTTokenizer --tokenizer-model "$TOKENIZER_MODEL")
        if [[ -n "$TOKENIZER_REVISION" ]]; then
            TOKENIZER_ARGS+=(--tokenizer-revision "$TOKENIZER_REVISION")
        fi
        ;;
    identity)
        SFT_ARGS+=(--sft-tokenizer-prompt-format identity)
        TOKENIZER_ARGS+=(--tokenizer-type SFTTokenizer --tokenizer-model "$TOKENIZER_MODEL")
        if [[ -n "$TOKENIZER_REVISION" ]]; then
            TOKENIZER_ARGS+=(--tokenizer-revision "$TOKENIZER_REVISION")
        fi
        ;;
    *)
        echo "Unknown TOKENIZER_PROFILE=$TOKENIZER_PROFILE" >&2
        usage >&2
        exit 2
        ;;
esac

case "$DATA_PROFILE" in
    jsonl_messages|jsonl)
        DATA_PATH="${DATA_PATH:-"$HOME/data/sft/train.jsonl"}"
        DATA_ARGS+=(
            --data-path "$DATA_PATH"
            --split "${SPLIT:-100,0,0}"
            --dataloader-type "${DATALOADER_TYPE:-cyclic}"
            --no-create-attention-mask-in-dataloader
            --no-mmap-bin-files
            --num-workers "${NUM_WORKERS:-1}"
        )
        ;;
    hf_blaise_mix)
        if [[ -z "${DATA_PATH:-}" ]]; then
            echo "DATA_PROFILE=hf_blaise_mix requires DATA_PATH to be set to a Hugging Face dataset repo" >&2
            exit 2
        fi
        DATA_ARGS+=(
            --data-path "$DATA_PATH"
            --split "${SPLIT:-100,0,0}"
            --dataloader-type "${DATALOADER_TYPE:-cyclic}"
            --no-create-attention-mask-in-dataloader
            --num-workers "${NUM_WORKERS:-1}"
        )
        ;;
    mock_sft|mock)
        DATA_ARGS+=(
            --mock-data
            --sft-mock-dataset-config-json "${SFT_MOCK_DATASET_CONFIG_JSON:-{\"mode\":\"distribution\",\"type\":\"lognormal\",\"min_seq_len\":1024,\"max_seq_len\":$SEQ_LENGTH,\"mean_seq_len\":2048,\"lognormal_sigma\":1.1}}"
            --split "${SPLIT:-100,0,0}"
            --no-create-attention-mask-in-dataloader
            --no-mmap-bin-files
            --num-workers "${NUM_WORKERS:-1}"
        )
        ;;
    *)
        echo "Unknown DATA_PROFILE=$DATA_PROFILE" >&2
        usage >&2
        exit 2
        ;;
esac

case "$PRECISION_PROFILE" in
    bf16)
        MODEL_ARGS+=(--bf16)
        ;;
    nvfp4|nvfp4_spinquant|nvfp4_higgs|nvfp4_turboquant|deepseek_nvfp4)
        MODEL_ARGS+=(--bf16)
        DTYPE_ARGS+=(--fp4-format e2m1 --fp4-recipe nvfp4 --fp4-param-gather)
        if [[ "$PRECISION_PROFILE" == "nvfp4_spinquant" || "$PRECISION_PROFILE" == "deepseek_nvfp4" ]]; then
            SPINQUANT_ARGS+=(
                --spinquant
                --spinquant-mode "${SPINQUANT_MODE:-random}"
                --spinquant-w-bits 4
                --spinquant-a-bits 4
                --spinquant-k-bits 4
                --spinquant-v-bits 4
            )
            if [[ -n "${SPINQUANT_ROTATION_PATH:-}" ]]; then
                SPINQUANT_ARGS+=(--spinquant-rotation-path "$SPINQUANT_ROTATION_PATH")
            fi
        fi
        if [[ "$PRECISION_PROFILE" == "nvfp4_higgs" || "$PRECISION_PROFILE" == "deepseek_nvfp4" ]]; then
            HIGGS_ARGS+=(--enable-higgs-dense-2bit-kv-cache --higgs-kv-preset "${HIGGS_KV_PRESET:-dense_2bit}")
        fi
        if [[ "$PRECISION_PROFILE" == "nvfp4_turboquant" ]]; then
            TURBOQUANT_ARGS+=(--turboquant-kv-enabled --turboquant-kv-preset "${TURBOQUANT_KV_PRESET:-latent_2p5bit_nc}" --turboquant-kv-seed "${TURBOQUANT_KV_SEED:-0}")
        fi
        if [[ "$PRECISION_PROFILE" == "deepseek_nvfp4" ]]; then
            INDEXCACHE_ARGS+=(
                --dsa-indexcache-quantization "${DSA_INDEXCACHE_QUANTIZATION:-nvfp4_e2m1_ue8m0}"
                --dsa-indexcache-quant-eps "${DSA_INDEXCACHE_QUANT_EPS:-1e-4}"
                --dsa-indexcache-hisa-enabled
                --dsa-indexcache-hisa-block-size "${DSA_INDEXCACHE_HISA_BLOCK_SIZE:-128}"
                --dsa-indexcache-hisa-block-topk "${DSA_INDEXCACHE_HISA_BLOCK_TOPK:-64}"
                --dsa-indexcache-hisa-compression-ratio "${DSA_INDEXCACHE_HISA_COMPRESSION_RATIO:-8.0}"
            )
        fi
        ;;
    *)
        echo "Unknown PRECISION_PROFILE=$PRECISION_PROFILE" >&2
        usage >&2
        exit 2
        ;;
esac

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
TRAINING_ARGS+=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
)
if [[ -n "${TRAIN_SAMPLES:-}" ]]; then
    TRAINING_ARGS+=(--train-samples "$TRAIN_SAMPLES")
    TRAINING_ARGS+=(--lr-decay-samples "${LR_DECAY_SAMPLES:-$TRAIN_SAMPLES}")
    TRAINING_ARGS+=(--lr-warmup-samples "${LR_WARMUP_SAMPLES:-0}")
else
    TRAINING_ARGS+=(--train-iters "${TRAIN_ITERS:-10}")
    TRAINING_ARGS+=(--lr-warmup-iters "${LR_WARMUP_ITERS:-0}")
fi
TRAINING_ARGS+=(
    --lr "${LR:-1.0e-5}"
    --min-lr "${MIN_LR:-1.0e-6}"
    --lr-decay-style "${LR_DECAY_STYLE:-cosine}"
    --clip-grad "${CLIP_GRAD:-1.0}"
    --weight-decay "${WEIGHT_DECAY:-0.0}"
    --adam-beta1 "${ADAM_BETA1:-0.9}"
    --adam-beta2 "${ADAM_BETA2:-0.95}"
    --adam-eps "${ADAM_EPS:-1.0e-6}"
    --log-interval "${LOG_INTERVAL:-10}"
    --empty-unused-memory-level "${EMPTY_UNUSED_MEMORY_LEVEL:-0}"
    --rerun-mode "${RERUN_MODE:-disabled}"
)
if [[ "${CALCULATE_PER_TOKEN_LOSS:-1}" == "1" ]]; then
    TRAINING_ARGS+=(--calculate-per-token-loss)
fi

case "$OPTIMIZER_PROFILE" in
    adamw)
        TRAINING_ARGS+=(--optimizer adam --use-distributed-optimizer)
        ;;
    flash_adamw)
        TRAINING_ARGS+=(--optimizer flash_adamw --use-distributed-optimizer --no-gradient-accumulation-fusion)
        ;;
    precision_aware)
        TRAINING_ARGS+=(
            --optimizer adam
            --use-distributed-optimizer
            --use-precision-aware-optimizer
            --exp-avg-dtype bf16
            --exp-avg-sq-dtype bf16
        )
        ;;
    flash_adamw_eco)
        TRAINING_ARGS+=(--optimizer flash_adamw --use-distributed-optimizer --no-gradient-accumulation-fusion --flash-adamw-eco)
        ;;
    *)
        echo "Unknown OPTIMIZER_PROFILE=$OPTIMIZER_PROFILE" >&2
        usage >&2
        exit 2
        ;;
esac
if [[ "${OVERLAP_GRAD_REDUCE:-1}" == "1" ]]; then
    TRAINING_ARGS+=(--overlap-grad-reduce)
fi
if [[ "${OVERLAP_PARAM_GATHER:-1}" == "1" ]]; then
    TRAINING_ARGS+=(--overlap-param-gather)
fi
if [[ "${GRAD_REDUCE_IN_BF16:-1}" == "1" ]]; then
    TRAINING_ARGS+=(--grad-reduce-in-bf16)
fi

case "$PARALLEL_PROFILE" in
    auto)
        GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
        NNODES="${NNODES:-1}"
        TP="${TP:-1}"
        PP="${PP:-1}"
        CP="${CP:-1}"
        EP="${EP:-1}"
        ETP="${ETP:-1}"
        ;;
    single_gpu)
        GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
        NNODES="${NNODES:-1}"
        TP="${TP:-1}"
        PP="${PP:-1}"
        CP="${CP:-1}"
        EP="${EP:-1}"
        ETP="${ETP:-1}"
        ;;
    single_node_8gpu)
        GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
        NNODES="${NNODES:-1}"
        TP="${TP:-1}"
        PP="${PP:-1}"
        CP="${CP:-1}"
        EP="${EP:-1}"
        ETP="${ETP:-1}"
        ;;
    two_node_8gpu)
        GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
        NNODES="${NNODES:-2}"
        TP="${TP:-8}"
        PP="${PP:-1}"
        CP="${CP:-1}"
        EP="${EP:-1}"
        ETP="${ETP:-1}"
        ;;
    deepseek_tp4_pp2_ep4)
        GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
        NNODES="${NNODES:-1}"
        TP="${TP:-4}"
        PP="${PP:-2}"
        CP="${CP:-1}"
        EP="${EP:-4}"
        ETP="${ETP:-1}"
        ;;
    deepseek_tp8_pp5_ep8)
        GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
        NNODES="${SLURM_NNODES:-${NNODES:-5}}"
        TP="${TP:-8}"
        PP="${PP:-5}"
        CP="${CP:-1}"
        EP="${EP:-8}"
        ETP="${ETP:-1}"
        ;;
    *)
        echo "Unknown PARALLEL_PROFILE=$PARALLEL_PROFILE" >&2
        usage >&2
        exit 2
        ;;
esac

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
DISTRIBUTED_BACKEND_ARGS=(--distributed-backend "${DISTRIBUTED_BACKEND:-nccl}")

MODEL_PARALLEL_ARGS+=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --context-parallel-size "$CP"
    --expert-model-parallel-size "$EP"
    --expert-tensor-parallel-size "$ETP"
)
if [[ "${SEQUENCE_PARALLEL:-auto}" == "1" || ( "${SEQUENCE_PARALLEL:-auto}" == "auto" && "$TP" -gt 1 ) ]]; then
    MODEL_PARALLEL_ARGS+=(--sequence-parallel)
fi
if [[ -n "${PIPELINE_MODEL_PARALLEL_LAYOUT:-}" ]]; then
    MODEL_PARALLEL_ARGS+=(--pipeline-model-parallel-layout "$PIPELINE_MODEL_PARALLEL_LAYOUT")
fi
if [[ -n "${DECODER_FIRST_PIPELINE_NUM_LAYERS:-}" ]]; then
    MODEL_PARALLEL_ARGS+=(--decoder-first-pipeline-num-layers "$DECODER_FIRST_PIPELINE_NUM_LAYERS")
fi
if [[ -n "${DECODER_LAST_PIPELINE_NUM_LAYERS:-}" ]]; then
    MODEL_PARALLEL_ARGS+=(--decoder-last-pipeline-num-layers "$DECODER_LAST_PIPELINE_NUM_LAYERS")
fi

if [[ "${RECOMPUTE:-0}" == "1" ]]; then
    TRAINING_ARGS+=(--recompute-granularity "${RECOMPUTE_GRANULARITY:-selective}")
    if [[ "${RECOMPUTE_GRANULARITY:-selective}" == "selective" ]]; then
        # shellcheck disable=SC2206
        RECOMPUTE_MODULE_LIST=(${RECOMPUTE_MODULES:-core_attn})
        TRAINING_ARGS+=(--recompute-modules "${RECOMPUTE_MODULE_LIST[@]}")
    else
        TRAINING_ARGS+=(--recompute-method "${RECOMPUTE_METHOD:-uniform}" --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}")
    fi
fi

if [[ "${ENABLE_TENSORBOARD_LOGGING:-1}" == "1" ]]; then
    TENSORBOARD_ARGS+=(--tensorboard-dir "$TENSORBOARD_LOGS_PATH" --tensorboard-log-interval "${TENSORBOARD_LOG_INTERVAL:-10}")
    if [[ "${LOG_THROUGHPUT:-1}" == "1" ]]; then
        TENSORBOARD_ARGS+=(--log-throughput)
    fi
fi

export WANDB_FORCE_OFFLINE="${WANDB_FORCE_OFFLINE:-1}"
if [[ "$WANDB_FORCE_OFFLINE" == "1" ]]; then
    export WANDB_MODE=offline
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-exp-name "${WANDB_EXP_NAME:-sft_${MODEL_PROFILE}}" --wandb-save-dir "${WANDB_SAVE_DIR:-"$SAVE_CKPT/wandb"}")
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    fi
fi

if [[ "${ENABLE_PROFILING:-0}" == "1" ]]; then
    PROFILING_ARGS+=(--profile --profile-step-start "${PROFILE_STEP_START:-4}" --profile-step-end "${PROFILE_STEP_END:-6}")
    if [[ "${USE_PYTORCH_PROFILER:-0}" == "1" ]]; then
        PROFILING_ARGS+=(--use-pytorch-profiler)
    fi
fi

CKPT_FORMAT_VALUE="${CKPT_FORMAT:-torch_dist}"
CKPT_ARGS+=(
    --eval-interval "${EVAL_INTERVAL:-100}"
    --eval-iters "${EVAL_ITERS:-0}"
    --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES:-60}"
    --ckpt-format "$CKPT_FORMAT_VALUE"
    --auto-detect-ckpt-format
)
if [[ -n "$LOAD_CKPT" ]]; then
    CKPT_ARGS+=(--load "$LOAD_CKPT")
fi
if [[ "${NO_LOAD_OPTIM:-1}" == "1" ]]; then
    CKPT_ARGS+=(--no-load-optim)
fi
if [[ "${NO_LOAD_RNG:-1}" == "1" ]]; then
    CKPT_ARGS+=(--no-load-rng)
fi
if [[ "${DISABLE_SAVE:-0}" != "1" ]]; then
    CKPT_ARGS+=(--save-interval "${SAVE_INTERVAL:-500}" --save "$SAVE_CKPT")
fi
if [[ "${NO_SAVE_OPTIM:-0}" == "1" ]]; then
    CKPT_ARGS+=(--no-save-optim)
fi
if [[ "${NO_SAVE_RNG:-0}" == "1" ]]; then
    CKPT_ARGS+=(--no-save-rng)
fi

if [[ -n "${EXTRA_MEGATRON_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_MEGATRON_ARGS_ARRAY=(${EXTRA_MEGATRON_ARGS})
fi

CMD=(
    uv run --no-sync torchrun
    "${DISTRIBUTED_ARGS[@]}"
    pretrain_gpt.py
    "${DISTRIBUTED_BACKEND_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    "${MLA_ARGS[@]}"
    "${DSA_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${MODEL_PARALLEL_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${DTYPE_ARGS[@]}"
    "${SPINQUANT_ARGS[@]}"
    "${TURBOQUANT_ARGS[@]}"
    "${HIGGS_ARGS[@]}"
    "${INDEXCACHE_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${TENSORBOARD_ARGS[@]}"
    "${WANDB_ARGS[@]}"
    "${PROFILING_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${EXTRA_MEGATRON_ARGS_ARRAY[@]}"
)

printf 'MODEL_PROFILE=%s TOKENIZER_PROFILE=%s DATA_PROFILE=%s PRECISION_PROFILE=%s OPTIMIZER_PROFILE=%s PARALLEL_PROFILE=%s\n' \
    "$MODEL_PROFILE" "$TOKENIZER_PROFILE" "$DATA_PROFILE" "$PRECISION_PROFILE" "$OPTIMIZER_PROFILE" "$PARALLEL_PROFILE"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$SAVE_CKPT" "$TENSORBOARD_LOGS_PATH"

case "$RUNTIME_PROFILE" in
    local)
        "${CMD[@]}"
        ;;
    slurm)
        srun --mpi="${SRUN_MPI:-pmix}" -l "${CMD[@]}"
        ;;
    *)
        echo "Unknown RUNTIME_PROFILE=$RUNTIME_PROFILE" >&2
        usage >&2
        exit 2
        ;;
esac
