#!/usr/bin/env bash
# Launch the DeepSeek-V3.2 REAP NVFP4 SFT run in a 2x2 tmux dashboard:
#   top-left: node 0 training log      top-right: node 1 training log
#   bottom-left: node 0 GPU monitor    bottom-right: node 1 GPU monitor

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${MEGATRON_DIR:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}"
cd "$MEGATRON_DIR"

command -v tmux >/dev/null || {
    echo "tmux is required for this launcher" >&2
    exit 1
}

SESSION="${SESSION:-corsaire_1_research_preview}"
ATTACH="${ATTACH:-1}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    if [[ "$ATTACH" == "1" ]]; then
        echo "tmux session '$SESSION' already exists; attaching." >&2
        exec tmux attach-session -t "$SESSION"
    fi
    echo "tmux session '$SESSION' already exists; leaving it running." >&2
    exit 0
fi

TS="${TS:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="${WANDB_EXP_NAME:-corsaire-1-research-preview}"
LOG_DIR="${LOG_DIR:-"$HOME/logs"}"
LOG0="$LOG_DIR/${RUN_NAME}_node0.log"
LOG1="$LOG_DIR/${RUN_NAME}_node1.log"
mkdir -p "$LOG_DIR"

MODEL_ID="${MODEL_ID:-BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$MODEL_ID}"
LOAD_CKPT="${LOAD_CKPT:-"$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron"}"
SAVE_CKPT="${SAVE_CKPT:-"$HOME/checkpoints/sft_deepseek_v32_reap_spinquant_actkv_nvfp4"}"

REMOTE_HOST="${REMOTE_HOST:-sjpat@10.180.0.45}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/google_compute_engine}"
REMOTE_MEGATRON_DIR="${REMOTE_MEGATRON_DIR:-/home/sjpat/Megatron-LM}"
REMOTE_RUNNER="/tmp/${RUN_NAME}_node1.sh"

MASTER_ADDR="${MASTER_ADDR:-10.200.0.21}"
MASTER_PORT="${MASTER_PORT:-29673}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-"$HOME/.cache/triton/deepseek_v32_reap_sft"}"
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP="${TP:-4}"
PP="${PP:-4}"
CP="${CP:-1}"
EP="${EP:-4}"
ETP="${ETP:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
DENSE_MODEL_PARALLEL_SIZE=$((TP * PP * CP))
EXPERT_MODEL_PIPELINE_PARALLEL_SIZE=$((ETP * EP * PP))
if (( WORLD_SIZE % DENSE_MODEL_PARALLEL_SIZE != 0 )); then
    echo "world_size=$WORLD_SIZE must be divisible by TP*PP*CP=$DENSE_MODEL_PARALLEL_SIZE" >&2
    exit 1
fi
if (( WORLD_SIZE % EXPERT_MODEL_PIPELINE_PARALLEL_SIZE != 0 )); then
    echo "world_size=$WORLD_SIZE must be divisible by ETP*EP*PP=$EXPERT_MODEL_PIPELINE_PARALLEL_SIZE" >&2
    exit 1
fi
DP=$((WORLD_SIZE / DENSE_MODEL_PARALLEL_SIZE))
EXPERT_DP=$((WORLD_SIZE / EXPERT_MODEL_PIPELINE_PARALLEL_SIZE))
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP) != 0 )); then
    echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE*DP=$((MICRO_BATCH_SIZE * DP))" >&2
    exit 1
fi
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP)))

# Full visible Blaise SFT mix target: 753,531 rows rounded down to a full
# GBS=64 update. This runs the local combined JSONL once without dataset loops.
DATA_PATH="${DATA_PATH:-"$HOME/data/sft/blaise-sft-training-mix/blaise-sft-training-mix-full.jsonl"}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-753408}"
LR_DECAY_SAMPLES="${LR_DECAY_SAMPLES:-$TRAIN_SAMPLES}"
LR_WARMUP_SAMPLES="${LR_WARMUP_SAMPLES:-31616}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
DECODER_FIRST_PIPELINE_NUM_LAYERS="${DECODER_FIRST_PIPELINE_NUM_LAYERS:-16}"
DECODER_LAST_PIPELINE_NUM_LAYERS="${DECODER_LAST_PIPELINE_NUM_LAYERS:-15}"
PIPELINE_MODEL_PARALLEL_LAYOUT="${PIPELINE_MODEL_PARALLEL_LAYOUT:-Et*8|t*8|t*8|t*8|t*8|t*7|t*7|t*7L}"
OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-0}"
# Megatron keeps checkpoints whose iteration is divisible by this value and
# deletes the previous non-retained checkpoint after a new save. Pick a value
# above the planned run so only the newest checkpoint remains.
SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-100000}"
if (( SAVE_RETAIN_INTERVAL % SAVE_INTERVAL != 0 )); then
    echo "SAVE_RETAIN_INTERVAL=$SAVE_RETAIN_INTERVAL must be divisible by SAVE_INTERVAL=$SAVE_INTERVAL" >&2
    exit 1
fi

cat <<EOF
Run name:      $RUN_NAME
Node 0 log:    $LOG0
Node 1 log:    $LOG1
Load ckpt:     $LOAD_CKPT
Save ckpt:     $SAVE_CKPT
Train samples: $TRAIN_SAMPLES
World shape:   nodes=$NNODES gpus_per_node=$GPUS_PER_NODE world=$WORLD_SIZE
Parallelism:   TP=$TP PP=$PP CP=$CP DP=$DP EP=$EP ETP=$ETP expert_DP=$EXPERT_DP
Batches:       MBS=$MICRO_BATCH_SIZE GBS=$GLOBAL_BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS
Save every:    $SAVE_INTERVAL updates
Retention:     keep latest normal Megatron checkpoint only
ZCC:           ENABLE_ZCC=${ENABLE_ZCC:-1} durable_interval=${ZCC_DURABLE_INTERVAL:-50} retain_latest=${ZCC_RETAIN_LATEST:-1}
Param gather:  OVERLAP_PARAM_GATHER=$OVERLAP_PARAM_GATHER
PP layout:     ${PIPELINE_MODEL_PARALLEL_LAYOUT:-first=$DECODER_FIRST_PIPELINE_NUM_LAYERS middle=auto last=$DECODER_LAST_PIPELINE_NUM_LAYERS}
StreamBP MoE:  chunk_forward=${STREAMBP_MOE_CHUNK_FORWARD:-0} mlp_chunks=${STREAMBP_MOE_MLP_CHUNKS:-4}
Quant stack:   spinquant=${SPINQUANT:-1} higgs=${USE_HIGGS:-1} turboquant=${TURBOQUANT:-0} indexcache=${INDEXCACHE:-1} indexcache_hisa=${DSA_INDEXCACHE_HISA:-1}
DSA fused:     triton=${MEGATRON_DSA_TRITON:-1} triton_indexer=${MEGATRON_DSA_TRITON_INDEXER:-1} streaming_topk=${MEGATRON_DSA_STREAMING_INDEXER_TOPK:-1} sort_topk=${MEGATRON_DSA_SORT_TOPK_INDICES:-0} validate_topk=${MEGATRON_DSA_VALIDATE_TOPK_INDICES:-0} hisa_slot_group=${MEGATRON_HISA_CANDIDATE_SLOT_GROUP:-16} hisa_selector_backend=${MEGATRON_HISA_SELECTOR_BACKEND:-bmm} hisa_selector_cuda=${MEGATRON_HISA_SELECTOR_CUDA:-1} hisa_selector_row_chunk=${MEGATRON_HISA_SELECTOR_ROW_CHUNK:-512} hisa_assume_sorted=${MEGATRON_HISA_ASSUME_SORTED_POSITIONS:-1} hisa_dense_fallback=${MEGATRON_HISA_FALLBACK_DENSE_IF_SHORT:-0} hisa_fused_loss=${MEGATRON_HISA_FUSED_INDEXER_LOSS:-1} teacher_score_scratch=${MEGATRON_DSA_TEACHER_SCORE_SCRATCH:-1} bwd_score_scratch=${MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH:-1} hisa_target_triton=${MEGATRON_HISA_TARGET_TRITON:-1} hisa_target_block_k=${MEGATRON_HISA_TARGET_BLOCK_K:-64} selected_bwd_head_group=${MEGATRON_HISA_SELECTED_SCORE_BWD_HEAD_GROUP:-8} selected_bwd_warp_grouped=${MEGATRON_HISA_SELECTED_SCORE_BWD_WARP_GROUPED:-1} bwd_block_k=${MEGATRON_DSA_TRITON_BLOCK_K_BWD:-32} bwd_warps=${MEGATRON_DSA_TRITON_BWD_NUM_WARPS:-2} cuda_kv_bwd=${MEGATRON_DSA_CUDA_KV_BWD:-0} cuda_kv_tile=${MEGATRON_DSA_CUDA_KV_BWD_TILE_Q:-2}x${MEGATRON_DSA_CUDA_KV_BWD_TILE_K:-4}
MoE SwiGLU:    weighted_fuser=${MEGATRON_WEIGHTED_SWIGLU_FUSER:-triton}
MoE LB type:   ${MOE_ROUTER_LOAD_BALANCING_TYPE:-seq_aux_loss}
MoE aux coeff: ${MOE_AUX_LOSS_COEFF:-1e-4}
MoE bias upd:  ${MOE_ROUTER_BIAS_UPDATE_RATE:-1e-3}
MoE bias rule: ${MOE_ROUTER_EXPERT_BIAS_UPDATE_METHOD:-sign}
ECO:           enabled=${FLASH_ADAMW_ECO:-1} lr_floor=${FLASH_ADAMW_ECO_LR_FLOOR:-base} projection=${FLASH_ADAMW_ECO_PROJECTION:-gain}
Triton cache:  TRITON_CACHE_AUTOTUNING=1 TRITON_CACHE_DIR=$TRITON_CACHE_DIR
HF upload:     enabled=${HF_UPLOAD_CHECKPOINTS:-1} interval=${HF_UPLOAD_INTERVAL:-100} repo=${HF_REPO_ID:-BlaiseAI/corsaire-1-research-preview} retain=${HF_UPLOAD_RETAIN:-2}
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1 set; not syncing, creating tmux panes, or launching training."
    exit 0
fi

sync_load_checkpoint_metadata() {
    local tracker="$LOAD_CKPT/latest_checkpointed_iteration.txt"
    if [[ ! -f "$tracker" ]]; then
        echo "missing load checkpoint tracker: $tracker" >&2
        exit 1
    fi

    local iteration
    iteration="$(<"$tracker")"
    local iter_dir
    if [[ "$iteration" == "release" ]]; then
        iter_dir="release"
    else
        iter_dir="$(printf 'iter_%07d' "$iteration")"
    fi

    local rel_files=(
        "latest_checkpointed_iteration.txt"
        "latest_train_state.pt"
        "$iter_dir/.metadata"
        "$iter_dir/common.pt"
        "$iter_dir/metadata.json"
        "$iter_dir/train_state.pt"
        "$iter_dir/run_config.yaml"
    )

    local rel
    for rel in "${rel_files[@]}"; do
        if [[ ! -f "$LOAD_CKPT/$rel" ]]; then
            echo "missing load checkpoint metadata file: $LOAD_CKPT/$rel" >&2
            exit 1
        fi
    done

    echo "Syncing load checkpoint metadata/common files to $REMOTE_HOST ..."
    ssh -i "$SSH_KEY" "$REMOTE_HOST" "mkdir -p '$LOAD_CKPT/$iter_dir'"
    tar -C "$LOAD_CKPT" -cf - "${rel_files[@]}" | ssh -i "$SSH_KEY" "$REMOTE_HOST" \
        "tar -C '$LOAD_CKPT' -xf -"
    ssh -i "$SSH_KEY" "$REMOTE_HOST" \
        "test -f '$LOAD_CKPT/latest_checkpointed_iteration.txt' && test -f '$LOAD_CKPT/$iter_dir/.metadata' && test -f '$LOAD_CKPT/$iter_dir/common.pt'"
}

if [[ "${SYNC_LOAD_CKPT_METADATA:-1}" == "1" ]]; then
    sync_load_checkpoint_metadata
fi

if [[ "${SYNC_REMOTE:-1}" == "1" ]]; then
    UPSTREAM_REF=""
    CURRENT_BRANCH="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
    if [[ -n "$CURRENT_BRANCH" ]] && git rev-parse --verify --quiet "origin/$CURRENT_BRANCH" >/dev/null; then
        UPSTREAM_REF="origin/$CURRENT_BRANCH"
    elif git rev-parse --verify --quiet origin/dev-sft >/dev/null; then
        UPSTREAM_REF="origin/dev-sft"
    fi
    mapfile -t SYNC_FILES < <(
        {
            if [[ -n "$UPSTREAM_REF" ]]; then
                git diff --name-only "$UPSTREAM_REF..HEAD"
            fi
            git diff --name-only
            git diff --name-only --cached
            git ls-files --others --exclude-standard
        } | sort -u | grep -Ev '^(artifacts/|\.pytest_cache/|.*__pycache__/|.*\.pyc$)' || true
    )
    if ((${#SYNC_FILES[@]})); then
        echo "Syncing ${#SYNC_FILES[@]} changed/untracked repo files to $REMOTE_HOST ..."
        tar -cf - "${SYNC_FILES[@]}" | ssh -i "$SSH_KEY" "$REMOTE_HOST" \
            "cd '$REMOTE_MEGATRON_DIR' && tar -xf -"
    fi
fi

write_env_block() {
    local node_rank="$1"
    cat <<EOF
export PATH="\$HOME/.local/bin:\$PATH"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_AUTOTUNING="${TRITON_CACHE_AUTOTUNING:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
export MODEL_ID="${MODEL_ID}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL}"
export LOAD_CKPT="${LOAD_CKPT}"
export SAVE_CKPT="${SAVE_CKPT}"
export DATA_PATH="${DATA_PATH}"
export NNODES="${NNODES}"
export GPUS_PER_NODE="${GPUS_PER_NODE}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-gpu}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TP="${TP}"
export PP="${PP}"
export CP="${CP}"
export EP="${EP}"
export ETP="${ETP}"
export DECODER_FIRST_PIPELINE_NUM_LAYERS="${DECODER_FIRST_PIPELINE_NUM_LAYERS}"
export DECODER_LAST_PIPELINE_NUM_LAYERS="${DECODER_LAST_PIPELINE_NUM_LAYERS}"
export PIPELINE_MODEL_PARALLEL_LAYOUT="${PIPELINE_MODEL_PARALLEL_LAYOUT}"
export PIPELINE_PARALLEL_SCHEDULE="${PIPELINE_PARALLEL_SCHEDULE:-}"
export NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE="${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE:-}"
export NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK="${NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK:-}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}"
export GRAD_REDUCE_IN_BF16="${GRAD_REDUCE_IN_BF16:-1}"
export DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-120}"
export WANDB_ENTITY="${WANDB_ENTITY:-blaise-ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-corsaire-1}"
export WANDB_EXP_NAME="${RUN_NAME}"
export LOG_MEMORY_INTERVAL="${LOG_MEMORY_INTERVAL:-10}"
export LOG_NUM_ZEROS_IN_GRAD="${LOG_NUM_ZEROS_IN_GRAD:-1}"
export TRAIN_SAMPLES="${TRAIN_SAMPLES}"
export LR_DECAY_SAMPLES="${LR_DECAY_SAMPLES}"
export LR_WARMUP_SAMPLES="${LR_WARMUP_SAMPLES}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export SAVE_INTERVAL="${SAVE_INTERVAL}"
export SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL}"
export DISABLE_SAVE="${DISABLE_SAVE:-0}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-0}"
export NO_SAVE_RNG="${NO_SAVE_RNG:-0}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export TENSORBOARD_LOG_INTERVAL="${TENSORBOARD_LOG_INTERVAL:-1}"
export OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER}"
export USE_STREAMBP="${USE_STREAMBP:-1}"
export STREAMBP_CHUNK_SIZE="${STREAMBP_CHUNK_SIZE:-2048}"
export STREAMBP_MOE_CHUNK_FORWARD="${STREAMBP_MOE_CHUNK_FORWARD:-0}"
export STREAMBP_MOE_MLP_CHUNKS="${STREAMBP_MOE_MLP_CHUNKS:-4}"
export SPINQUANT="${SPINQUANT:-1}"
export TURBOQUANT="${TURBOQUANT:-0}"
export USE_HIGGS="${USE_HIGGS:-1}"
export HIGGS_KV_PRESET="${HIGGS_KV_PRESET:-dense_2bit}"
export INDEXCACHE="${INDEXCACHE:-1}"
export DSA_INDEXCACHE_QUANTIZATION="${DSA_INDEXCACHE_QUANTIZATION:-nvfp4_e2m1_ue8m0}"
export DSA_INDEXCACHE_QUANT_EPS="${DSA_INDEXCACHE_QUANT_EPS:-1e-4}"
export DSA_INDEXCACHE_HISA="${DSA_INDEXCACHE_HISA:-1}"
export DSA_INDEXCACHE_HISA_BLOCK_SIZE="${DSA_INDEXCACHE_HISA_BLOCK_SIZE:-128}"
export DSA_INDEXCACHE_HISA_BLOCK_TOPK="${DSA_INDEXCACHE_HISA_BLOCK_TOPK:-64}"
export DSA_INDEXCACHE_HISA_COMPRESSION_RATIO="${DSA_INDEXCACHE_HISA_COMPRESSION_RATIO:-4.0}"
export MOE_ROUTER_LOAD_BALANCING_TYPE="${MOE_ROUTER_LOAD_BALANCING_TYPE:-seq_aux_loss}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-1e-4}"
export MOE_ROUTER_BIAS_UPDATE_RATE="${MOE_ROUTER_BIAS_UPDATE_RATE:-1e-3}"
export MOE_ROUTER_EXPERT_BIAS_UPDATE_METHOD="${MOE_ROUTER_EXPERT_BIAS_UPDATE_METHOD:-sign}"
export MOE_ROUTER_QUANTILE_BIAS_ITERS="${MOE_ROUTER_QUANTILE_BIAS_ITERS:-5}"
export MOE_ROUTER_QUANTILE_BIAS_SYNC_SCORES="${MOE_ROUTER_QUANTILE_BIAS_SYNC_SCORES:-1}"
export DSA_CHUNK_SIZE="${DSA_CHUNK_SIZE:-2048}"
export DSA_INDEXER_TOPK="${DSA_INDEXER_TOPK:-1024}"
export DSA_INDEXER_LOSS_COEFF="${DSA_INDEXER_LOSS_COEFF:-0.01}"
export MEGATRON_DSA_TRITON="${MEGATRON_DSA_TRITON:-1}"
export MEGATRON_DSA_TRITON_INDEXER="${MEGATRON_DSA_TRITON_INDEXER:-1}"
export MEGATRON_DSA_STREAMING_INDEXER_TOPK="${MEGATRON_DSA_STREAMING_INDEXER_TOPK:-1}"
export MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE="${MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE:-4096}"
export MEGATRON_DSA_SORT_TOPK_INDICES="${MEGATRON_DSA_SORT_TOPK_INDICES:-0}"
export MEGATRON_DSA_COMPACT_TOPK_INDICES="${MEGATRON_DSA_COMPACT_TOPK_INDICES:-1}"
export MEGATRON_DSA_TRITON_BF16_GRAD_ATOMICS="${MEGATRON_DSA_TRITON_BF16_GRAD_ATOMICS:-1}"
export MEGATRON_DSA_TRITON_BLOCK_K_BWD="${MEGATRON_DSA_TRITON_BLOCK_K_BWD:-32}"
export MEGATRON_DSA_TRITON_BWD_NUM_WARPS="${MEGATRON_DSA_TRITON_BWD_NUM_WARPS:-2}"
export MEGATRON_DSA_CUDA_KV_BWD="${MEGATRON_DSA_CUDA_KV_BWD:-0}"
export MEGATRON_DSA_CUDA_KV_BWD_TILE_Q="${MEGATRON_DSA_CUDA_KV_BWD_TILE_Q:-2}"
export MEGATRON_DSA_CUDA_KV_BWD_TILE_K="${MEGATRON_DSA_CUDA_KV_BWD_TILE_K:-4}"
export MEGATRON_DSA_VALIDATE_TOPK_INDICES="${MEGATRON_DSA_VALIDATE_TOPK_INDICES:-0}"
export MEGATRON_HISA_CANDIDATE_SLOT_GROUP="${MEGATRON_HISA_CANDIDATE_SLOT_GROUP:-16}"
export MEGATRON_HISA_SELECTOR_BACKEND="${MEGATRON_HISA_SELECTOR_BACKEND:-bmm}"
export MEGATRON_HISA_SELECTOR_CUDA="${MEGATRON_HISA_SELECTOR_CUDA:-1}"
export MEGATRON_HISA_SELECTOR_ROW_CHUNK="${MEGATRON_HISA_SELECTOR_ROW_CHUNK:-512}"
export MEGATRON_HISA_SELECTED_SCORE_BWD_HEAD_GROUP="${MEGATRON_HISA_SELECTED_SCORE_BWD_HEAD_GROUP:-8}"
export MEGATRON_HISA_SELECTED_SCORE_BWD_WARP_GROUPED="${MEGATRON_HISA_SELECTED_SCORE_BWD_WARP_GROUPED:-1}"
export MEGATRON_HISA_ASSUME_SORTED_POSITIONS="${MEGATRON_HISA_ASSUME_SORTED_POSITIONS:-1}"
export MEGATRON_HISA_FALLBACK_DENSE_IF_SHORT="${MEGATRON_HISA_FALLBACK_DENSE_IF_SHORT:-0}"
export MEGATRON_HISA_FUSED_INDEXER_LOSS="${MEGATRON_HISA_FUSED_INDEXER_LOSS:-1}"
export MEGATRON_HISA_TARGET_TRITON="${MEGATRON_HISA_TARGET_TRITON:-1}"
export MEGATRON_HISA_TARGET_BLOCK_K="${MEGATRON_HISA_TARGET_BLOCK_K:-64}"
export MEGATRON_HISA_TARGET_ROW_CHUNK="${MEGATRON_HISA_TARGET_ROW_CHUNK:-128}"
export MEGATRON_HISA_KL_GRAD_TRITON="${MEGATRON_HISA_KL_GRAD_TRITON:-1}"
export MEGATRON_DSA_TEACHER_SCORE_SCRATCH="${MEGATRON_DSA_TEACHER_SCORE_SCRATCH:-1}"
export MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH="${MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH:-1}"
export MEGATRON_WEIGHTED_SWIGLU_FUSER="${MEGATRON_WEIGHTED_SWIGLU_FUSER:-triton}"
export MEGATRON_FLASH_ADAMW_NVFP4_IMMEDIATE_CAST="${MEGATRON_FLASH_ADAMW_NVFP4_IMMEDIATE_CAST:-1}"
export FLASH_ADAMW_ECO="${FLASH_ADAMW_ECO:-1}"
export FLASH_ADAMW_ECO_LR_FLOOR="${FLASH_ADAMW_ECO_LR_FLOOR:-base}"
export FLASH_ADAMW_ECO_PROJECTION="${FLASH_ADAMW_ECO_PROJECTION:-gain}"
export FLASH_ADAMW_ECO_PROJECTION_SCALE_BUDGET="${FLASH_ADAMW_ECO_PROJECTION_SCALE_BUDGET:-2.0}"
export FLASH_ADAMW_ECO_PROJECTION_GAIN_BUDGET="${FLASH_ADAMW_ECO_PROJECTION_GAIN_BUDGET:-0.25}"
export FLASH_ADAMW_ECO_PROJECTION_STEPS="${FLASH_ADAMW_ECO_PROJECTION_STEPS:-16}"
export FLASH_ADAMW_COMPRESS_STATE_DICT="${FLASH_ADAMW_COMPRESS_STATE_DICT:-1}"
export ENABLE_ZCC="${ENABLE_ZCC:-1}"
export ZCC_FLASH_DEVICE="${ZCC_FLASH_DEVICE:-/dev/shm/megatron_zcc/$RUN_NAME}"
export ZCC_WORKERS_NUM="${ZCC_WORKERS_NUM:-8}"
export ZCC_DURABLE_INTERVAL="${ZCC_DURABLE_INTERVAL:-50}"
export ZCC_DURABLE_DIR="${ZCC_DURABLE_DIR:-$SAVE_CKPT/zcc/$RUN_NAME}"
export ZCC_COMPRESS="${ZCC_COMPRESS:-zstd:1}"
export ZCC_RETAIN_LATEST="${ZCC_RETAIN_LATEST:-1}"
export MEGATRON_NUMERIC_DEBUG="${MEGATRON_NUMERIC_DEBUG:-0}"
export MEGATRON_NUMERIC_DEBUG_RANKS="${MEGATRON_NUMERIC_DEBUG_RANKS:-all}"
export MEGATRON_NUMERIC_DEBUG_HOOK_RANKS="${MEGATRON_NUMERIC_DEBUG_HOOK_RANKS:-${MEGATRON_NUMERIC_DEBUG_RANKS:-all}}"
export MEGATRON_NUMERIC_DEBUG_ROUTER_RANKS="${MEGATRON_NUMERIC_DEBUG_ROUTER_RANKS:-${MEGATRON_NUMERIC_DEBUG_RANKS:-all}}"
export MEGATRON_NUMERIC_DEBUG_DSA_RANKS="${MEGATRON_NUMERIC_DEBUG_DSA_RANKS:-${MEGATRON_NUMERIC_DEBUG_RANKS:-all}}"
export MEGATRON_NUMERIC_DEBUG_FLASHOPT_RANKS="${MEGATRON_NUMERIC_DEBUG_FLASHOPT_RANKS:-${MEGATRON_NUMERIC_DEBUG_RANKS:-all}}"
export MEGATRON_NUMERIC_DEBUG_PARAM_RANKS="${MEGATRON_NUMERIC_DEBUG_PARAM_RANKS:-${MEGATRON_NUMERIC_DEBUG_RANKS:-all}}"
export MEGATRON_NUMERIC_DEBUG_GRAD_RANKS="${MEGATRON_NUMERIC_DEBUG_GRAD_RANKS:-${MEGATRON_NUMERIC_DEBUG_RANKS:-all}}"
export MEGATRON_NUMERIC_DEBUG_DDP_GRAD_RANKS="${MEGATRON_NUMERIC_DEBUG_DDP_GRAD_RANKS:-${MEGATRON_NUMERIC_DEBUG_RANKS:-all}}"
export MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD_RANKS="${MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD_RANKS:-${MEGATRON_NUMERIC_DEBUG_RANKS:-all}}"
export MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD_RANKS="${MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD_RANKS:-${MEGATRON_NUMERIC_DEBUG_RANKS:-all}}"
export MEGATRON_NUMERIC_DEBUG_START_ITER="${MEGATRON_NUMERIC_DEBUG_START_ITER:-1}"
export MEGATRON_NUMERIC_DEBUG_HOOK_START_ITER="${MEGATRON_NUMERIC_DEBUG_HOOK_START_ITER:-${MEGATRON_NUMERIC_DEBUG_START_ITER:-1}}"
export MEGATRON_NUMERIC_DEBUG_ROUTER_START_ITER="${MEGATRON_NUMERIC_DEBUG_ROUTER_START_ITER:-${MEGATRON_NUMERIC_DEBUG_START_ITER:-1}}"
export MEGATRON_NUMERIC_DEBUG_DSA_START_ITER="${MEGATRON_NUMERIC_DEBUG_DSA_START_ITER:-${MEGATRON_NUMERIC_DEBUG_START_ITER:-1}}"
export MEGATRON_NUMERIC_DEBUG_FLASHOPT_START_ITER="${MEGATRON_NUMERIC_DEBUG_FLASHOPT_START_ITER:-${MEGATRON_NUMERIC_DEBUG_START_ITER:-1}}"
export MEGATRON_NUMERIC_DEBUG_PARAM_START_ITER="${MEGATRON_NUMERIC_DEBUG_PARAM_START_ITER:-${MEGATRON_NUMERIC_DEBUG_START_ITER:-1}}"
export MEGATRON_NUMERIC_DEBUG_GRAD_START_ITER="${MEGATRON_NUMERIC_DEBUG_GRAD_START_ITER:-${MEGATRON_NUMERIC_DEBUG_START_ITER:-1}}"
export MEGATRON_NUMERIC_DEBUG_DDP_GRAD_START_ITER="${MEGATRON_NUMERIC_DEBUG_DDP_GRAD_START_ITER:-${MEGATRON_NUMERIC_DEBUG_START_ITER:-1}}"
export MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD_START_ITER="${MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD_START_ITER:-${MEGATRON_NUMERIC_DEBUG_START_ITER:-1}}"
export MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD_START_ITER="${MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD_START_ITER:-${MEGATRON_NUMERIC_DEBUG_START_ITER:-1}}"
export MEGATRON_NUMERIC_DEBUG_FIRST_N="${MEGATRON_NUMERIC_DEBUG_FIRST_N:-16}"
export MEGATRON_NUMERIC_DEBUG_INTERVAL="${MEGATRON_NUMERIC_DEBUG_INTERVAL:-1}"
export MEGATRON_NUMERIC_DEBUG_MAX_ELEMS="${MEGATRON_NUMERIC_DEBUG_MAX_ELEMS:-262144}"
export MEGATRON_NUMERIC_DEBUG_FULL_FINITE="${MEGATRON_NUMERIC_DEBUG_FULL_FINITE:-0}"
export MEGATRON_NUMERIC_DEBUG_EVENT_LIMIT="${MEGATRON_NUMERIC_DEBUG_EVENT_LIMIT:-16}"
export MEGATRON_NUMERIC_DEBUG_FORWARD_HOOKS="${MEGATRON_NUMERIC_DEBUG_FORWARD_HOOKS:-0}"
export MEGATRON_NUMERIC_DEBUG_HOOK_INPUTS="${MEGATRON_NUMERIC_DEBUG_HOOK_INPUTS:-0}"
export MEGATRON_NUMERIC_DEBUG_HOOK_CLASSES="${MEGATRON_NUMERIC_DEBUG_HOOK_CLASSES:-TransformerLayer,MoELayer,TopKRouter,DSAttention,DSAIndexer,SelfAttention,MLP}"
export MEGATRON_NUMERIC_DEBUG_VERBOSE_HOOKS="${MEGATRON_NUMERIC_DEBUG_VERBOSE_HOOKS:-0}"
export MEGATRON_NUMERIC_DEBUG_ABORT_ON_NONFINITE="${MEGATRON_NUMERIC_DEBUG_ABORT_ON_NONFINITE:-1}"
export MEGATRON_NUMERIC_DEBUG_PARAM_STATS="${MEGATRON_NUMERIC_DEBUG_PARAM_STATS:-0}"
export MEGATRON_NUMERIC_DEBUG_TOPK="${MEGATRON_NUMERIC_DEBUG_TOPK:-16}"
export MEGATRON_NUMERIC_DEBUG_ROUTER="${MEGATRON_NUMERIC_DEBUG_ROUTER:-0}"
export MEGATRON_NUMERIC_DEBUG_ROUTER_FORCE="${MEGATRON_NUMERIC_DEBUG_ROUTER_FORCE:-0}"
export MEGATRON_NUMERIC_DEBUG_ROUTER_LIMIT="${MEGATRON_NUMERIC_DEBUG_ROUTER_LIMIT:-64}"
export MEGATRON_NUMERIC_DEBUG_DSA="${MEGATRON_NUMERIC_DEBUG_DSA:-0}"
export MEGATRON_NUMERIC_DEBUG_DSA_FORCE="${MEGATRON_NUMERIC_DEBUG_DSA_FORCE:-0}"
export MEGATRON_NUMERIC_DEBUG_DSA_LIMIT="${MEGATRON_NUMERIC_DEBUG_DSA_LIMIT:-64}"
export MEGATRON_NUMERIC_DEBUG_FLASHOPT="${MEGATRON_NUMERIC_DEBUG_FLASHOPT:-0}"
export MEGATRON_NUMERIC_DEBUG_FLASHOPT_FORCE="${MEGATRON_NUMERIC_DEBUG_FLASHOPT_FORCE:-0}"
export MEGATRON_NUMERIC_DEBUG_FLASHOPT_LIMIT="${MEGATRON_NUMERIC_DEBUG_FLASHOPT_LIMIT:-32}"
export MEGATRON_NUMERIC_DEBUG_FLASHOPT_ADAM_LIMIT="${MEGATRON_NUMERIC_DEBUG_FLASHOPT_ADAM_LIMIT:-32}"
export MEGATRON_NUMERIC_DEBUG_FLASHOPT_CHECK_ALL="${MEGATRON_NUMERIC_DEBUG_FLASHOPT_CHECK_ALL:-0}"
export MEGATRON_NUMERIC_DEBUG_FLASHOPT_CHECK_PRECAST="${MEGATRON_NUMERIC_DEBUG_FLASHOPT_CHECK_PRECAST:-0}"
export MEGATRON_NUMERIC_DEBUG_FLASHOPT_ABORT="${MEGATRON_NUMERIC_DEBUG_FLASHOPT_ABORT:-${MEGATRON_NUMERIC_DEBUG_ABORT_ON_NONFINITE:-1}}"
export MEGATRON_NUMERIC_DEBUG_GRAD_HOOKS="${MEGATRON_NUMERIC_DEBUG_GRAD_HOOKS:-0}"
export MEGATRON_NUMERIC_DEBUG_GRAD_REGEX="${MEGATRON_NUMERIC_DEBUG_GRAD_REGEX:-shared_experts\\.linear_fc2\\.weight}"
export MEGATRON_NUMERIC_DEBUG_GRAD_ABORT="${MEGATRON_NUMERIC_DEBUG_GRAD_ABORT:-1}"
export MEGATRON_NUMERIC_DEBUG_DDP_GRAD="${MEGATRON_NUMERIC_DEBUG_DDP_GRAD:-0}"
export MEGATRON_NUMERIC_DEBUG_DDP_GRAD_REGEX="${MEGATRON_NUMERIC_DEBUG_DDP_GRAD_REGEX:-shared_experts\\.linear_fc2\\.weight}"
export MEGATRON_NUMERIC_DEBUG_DDP_GRAD_ABORT="${MEGATRON_NUMERIC_DEBUG_DDP_GRAD_ABORT:-1}"
export MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD="${MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD:-0}"
export MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD_REGEX="${MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD_REGEX:-shared_experts\\.linear_fc2\\.weight}"
export MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD_FULL_FINITE="${MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD_FULL_FINITE:-1}"
export MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD_ABORT="${MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD_ABORT:-1}"
export MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD_STAGE_LIMIT="${MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD_STAGE_LIMIT:-8}"
export MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD="${MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD:-0}"
export MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD_REGEX="${MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD_REGEX:-shared_experts\\.linear_fc2\\.weight}"
export MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD_STAGE_LIMIT="${MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD_STAGE_LIMIT:-4}"
export MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD_COPY_LIMIT="${MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD_COPY_LIMIT:-8}"
export MEGATRON_NUMERIC_DEBUG_LOSS="${MEGATRON_NUMERIC_DEBUG_LOSS:-0}"
export MEGATRON_NUMERIC_DEBUG_BATCH="${MEGATRON_NUMERIC_DEBUG_BATCH:-0}"
export MEGATRON_FLASH_ADAMW_NVFP4_USE_MAIN_GRAD="${MEGATRON_FLASH_ADAMW_NVFP4_USE_MAIN_GRAD:-0}"
export MEGATRON_GRAD_OWNERSHIP="${MEGATRON_GRAD_OWNERSHIP:-0}"
export MEGATRON_GRAD_OWNERSHIP_RANKS="${MEGATRON_GRAD_OWNERSHIP_RANKS:-all}"
export MEGATRON_GRAD_OWNERSHIP_START_ITER="${MEGATRON_GRAD_OWNERSHIP_START_ITER:-1}"
export MEGATRON_GRAD_OWNERSHIP_FIRST_N="${MEGATRON_GRAD_OWNERSHIP_FIRST_N:-8}"
export MEGATRON_GRAD_OWNERSHIP_INTERVAL="${MEGATRON_GRAD_OWNERSHIP_INTERVAL:-1}"
export MEGATRON_GRAD_OWNERSHIP_TOP_OWNERS="${MEGATRON_GRAD_OWNERSHIP_TOP_OWNERS:-32}"
export MEGATRON_GRAD_OWNERSHIP_TOP_PARAMS="${MEGATRON_GRAD_OWNERSHIP_TOP_PARAMS:-0}"
export HF_UPLOAD_CHECKPOINTS="${HF_UPLOAD_CHECKPOINTS:-1}"
export HF_REPO_ID="${HF_REPO_ID:-BlaiseAI/corsaire-1-research-preview}"
export HF_UPLOAD_FOLDER_PREFIX="${HF_UPLOAD_FOLDER_PREFIX:-corsaire-1-research-preview}"
export HF_UPLOAD_INTERVAL="${HF_UPLOAD_INTERVAL:-100}"
export HF_UPLOAD_RETAIN="${HF_UPLOAD_RETAIN:-2}"
export HF_UPLOAD_STABLE_SECONDS="${HF_UPLOAD_STABLE_SECONDS:-180}"
export HF_UPLOAD_POLL_SECONDS="${HF_UPLOAD_POLL_SECONDS:-60}"
export HF_UPLOAD_PRIVATE="${HF_UPLOAD_PRIVATE:-1}"
export NODE_RANK="$node_rank"
mkdir -p "\$TRITON_CACHE_DIR"
EOF
}

NODE0_RUNNER="$LOG_DIR/${RUN_NAME}_node0_runner.sh"
NODE1_LOCAL_RUNNER="$LOG_DIR/${RUN_NAME}_node1_runner.sh"
NODE1_WRAPPER="$LOG_DIR/${RUN_NAME}_node1_wrapper.sh"
LOCAL_MONITOR="$LOG_DIR/${RUN_NAME}_node0_gpu_monitor.sh"
REMOTE_MONITOR="$LOG_DIR/${RUN_NAME}_node1_gpu_monitor.sh"

{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    write_env_block 0
    echo "cd '$MEGATRON_DIR'"
    echo "if [[ \"\$HF_UPLOAD_CHECKPOINTS\" == \"1\" ]]; then"
    echo "  mkdir -p '$LOG_DIR'"
    echo "  HF_PRIVATE_ARG=(); [[ \"\$HF_UPLOAD_PRIVATE\" == \"1\" ]] && HF_PRIVATE_ARG=(--private)"
    echo "  uv run --no-sync python tools/upload_mcore_checkpoints_to_hf.py \\"
    echo "    --checkpoint-root \"\$SAVE_CKPT\" \\"
    echo "    --repo-id \"\$HF_REPO_ID\" \\"
    echo "    --folder-prefix \"\$HF_UPLOAD_FOLDER_PREFIX\" \\"
    echo "    --node-name node0 \\"
    echo "    --shard-start 0 --shard-end 7 \\"
    echo "    --upload-interval \"\$HF_UPLOAD_INTERVAL\" \\"
    echo "    --retain \"\$HF_UPLOAD_RETAIN\" \\"
    echo "    --stable-seconds \"\$HF_UPLOAD_STABLE_SECONDS\" \\"
    echo "    --poll-seconds \"\$HF_UPLOAD_POLL_SECONDS\" \\"
    echo "    --delete-before-upload \\"
    echo "    \"\${HF_PRIVATE_ARG[@]}\" > '$LOG_DIR/${RUN_NAME}_hf_upload_node0.log' 2>&1 &"
    echo "  HF_UPLOAD_PID=\$!"
    echo "  trap 'kill \"\$HF_UPLOAD_PID\" 2>/dev/null || true' EXIT"
    echo "fi"
    echo "echo '[node0] starting $RUN_NAME at '\"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
    echo "examples/sft/run_sft_deepseek_nvfp4.sh 2>&1 | tee '$LOG0'"
} > "$NODE0_RUNNER"

{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    write_env_block 1
    echo "cd '$REMOTE_MEGATRON_DIR'"
    echo "if [[ \"\$HF_UPLOAD_CHECKPOINTS\" == \"1\" ]]; then"
    echo "  mkdir -p '$LOG_DIR'"
    echo "  HF_PRIVATE_ARG=(); [[ \"\$HF_UPLOAD_PRIVATE\" == \"1\" ]] && HF_PRIVATE_ARG=(--private)"
    echo "  uv run --no-sync python tools/upload_mcore_checkpoints_to_hf.py \\"
    echo "    --checkpoint-root \"\$SAVE_CKPT\" \\"
    echo "    --repo-id \"\$HF_REPO_ID\" \\"
    echo "    --folder-prefix \"\$HF_UPLOAD_FOLDER_PREFIX\" \\"
    echo "    --node-name node1 \\"
    echo "    --shard-start 8 --shard-end 15 \\"
    echo "    --upload-interval \"\$HF_UPLOAD_INTERVAL\" \\"
    echo "    --retain \"\$HF_UPLOAD_RETAIN\" \\"
    echo "    --stable-seconds \"\$HF_UPLOAD_STABLE_SECONDS\" \\"
    echo "    --poll-seconds \"\$HF_UPLOAD_POLL_SECONDS\" \\"
    echo "    --delete-before-upload \\"
    echo "    \"\${HF_PRIVATE_ARG[@]}\" > '$LOG_DIR/${RUN_NAME}_hf_upload_node1.log' 2>&1 &"
    echo "  HF_UPLOAD_PID=\$!"
    echo "  trap 'kill \"\$HF_UPLOAD_PID\" 2>/dev/null || true' EXIT"
    echo "fi"
    echo "echo '[node1] starting $RUN_NAME at '\"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
    echo "examples/sft/run_sft_deepseek_nvfp4.sh"
} > "$NODE1_LOCAL_RUNNER"

chmod +x "$NODE0_RUNNER" "$NODE1_LOCAL_RUNNER"
ssh -i "$SSH_KEY" "$REMOTE_HOST" "cat > '$REMOTE_RUNNER' && chmod +x '$REMOTE_RUNNER'" < "$NODE1_LOCAL_RUNNER"

cat > "$NODE1_WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ssh -tt -i "$SSH_KEY" "$REMOTE_HOST" "bash '$REMOTE_RUNNER'" 2>&1 | tee "$LOG1"
EOF

cat > "$LOCAL_MONITOR" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if command -v nvtop >/dev/null 2>&1; then
    exec nvtop
fi
exec watch -n 2 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv'
EOF

cat > "$REMOTE_MONITOR" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec ssh -tt -i "$SSH_KEY" "$REMOTE_HOST" 'if command -v nvtop >/dev/null 2>&1; then exec nvtop; fi; exec watch -n 2 "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv"'
EOF

chmod +x "$NODE1_WRAPPER" "$LOCAL_MONITOR" "$REMOTE_MONITOR"

tmux new-session -d -s "$SESSION" -n train
P0="$(tmux display-message -p -t "$SESSION:0" '#{pane_id}')"
tmux send-keys -t "$P0" "bash '$NODE0_RUNNER'; echo; echo '[node0 pane exited]'; exec bash" C-m
tmux split-window -h -t "$P0"
P1="$(tmux display-message -p -t "$SESSION:0" '#{pane_id}')"
tmux send-keys -t "$P1" "bash '$NODE1_WRAPPER'; echo; echo '[node1 pane exited]'; exec bash" C-m
tmux split-window -v -t "$P0" "bash '$LOCAL_MONITOR'"
tmux split-window -v -t "$P1" "bash '$REMOTE_MONITOR'"
tmux select-layout -t "$SESSION:0" tiled >/dev/null
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
if [[ "$ATTACH" == "1" ]]; then
    tmux attach-session -t "$SESSION"
else
    echo "tmux session '$SESSION' started. Attach with: tmux attach -t '$SESSION'"
fi
