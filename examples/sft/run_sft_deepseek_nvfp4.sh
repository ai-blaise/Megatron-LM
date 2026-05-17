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
USE_STREAMBP="${USE_STREAMBP:-1}"
STREAMBP_MOE_MLP_CHUNKS="${STREAMBP_MOE_MLP_CHUNKS:-1}"
OVERLAP_MOE_EXPERT_PARALLEL_COMM="${OVERLAP_MOE_EXPERT_PARALLEL_COMM:-0}"
if [[ -z "${RECOMPUTE+x}" ]]; then
    if [[ "$USE_STREAMBP" == "1" ]]; then
        RECOMPUTE=0
    else
        RECOMPUTE=1
    fi
fi
if [[ "$USE_MEGATRON_FSDP" == "1" ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
elif [[ "$OVERLAP_MOE_EXPERT_PARALLEL_COMM" == "1" ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
if [[ "${ENABLE_TRAINING_DEBUG:-0}" != "1" ]]; then
    export MEGATRON_ACT_ECO_DEBUG_SYNC=0
    export MEGATRON_ACT_ECO_DEBUG_VERBOSE=0
    export MEGATRON_STREAMBP_DEBUG_SYNC=0
    export MEGATRON_STREAMBP_DEBUG_VERBOSE=0
    export MEGATRON_DDP_GRAD_DEBUG_SYNC=0
    export MEGATRON_DDP_GRAD_DEBUG_VERBOSE=0
    export MEGATRON_NUMERIC_DEBUG=0
    export MEGATRON_NUMERIC_DEBUG_FORWARD_HOOKS=0
    export MEGATRON_NUMERIC_DEBUG_HOOK_INPUTS=0
    export MEGATRON_NUMERIC_DEBUG_PARAM_STATS=0
    export MEGATRON_NUMERIC_DEBUG_ROUTER=0
    export MEGATRON_NUMERIC_DEBUG_ROUTER_FORCE=0
    export MEGATRON_NUMERIC_DEBUG_DSA=0
    export MEGATRON_NUMERIC_DEBUG_DSA_FORCE=0
    export MEGATRON_NUMERIC_DEBUG_FLASHOPT=0
    export MEGATRON_NUMERIC_DEBUG_FLASHOPT_FORCE=0
    export MEGATRON_NUMERIC_DEBUG_FLASHOPT_CHECK_ALL=0
    export MEGATRON_NUMERIC_DEBUG_FLASHOPT_CHECK_PRECAST=0
    export MEGATRON_NUMERIC_DEBUG_GRAD_HOOKS=0
    export MEGATRON_NUMERIC_DEBUG_DDP_GRAD=0
    export MEGATRON_NUMERIC_DEBUG_FINALIZE_GRAD=0
    export MEGATRON_NUMERIC_DEBUG_OPTIMIZER_GRAD=0
    export MEGATRON_NUMERIC_DEBUG_LOSS=0
    export MEGATRON_NUMERIC_DEBUG_BATCH=0
    export MEGATRON_GRAD_OWNERSHIP=0
fi
export MEGATRON_STREAMBP_MOE_REPLAY_TRIM_CACHE="${MEGATRON_STREAMBP_MOE_REPLAY_TRIM_CACHE:-1}"
export MEGATRON_STREAMBP_MOE_REPLAY_TRIM_FREE_MB="${MEGATRON_STREAMBP_MOE_REPLAY_TRIM_FREE_MB:-2048}"
export MEGATRON_STREAMBP_MOE_REPLAY_TRIM_CACHED_MB="${MEGATRON_STREAMBP_MOE_REPLAY_TRIM_CACHED_MB:-512}"
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
export MEGATRON_HISA_BMM_FP32_ACCUM_TENSORCORES="${MEGATRON_HISA_BMM_FP32_ACCUM_TENSORCORES:-1}"
export MEGATRON_HISA_SELECTED_SCORE_BWD_HEAD_GROUP="${MEGATRON_HISA_SELECTED_SCORE_BWD_HEAD_GROUP:-8}"
export MEGATRON_HISA_SELECTED_SCORE_BWD_WARP_GROUPED="${MEGATRON_HISA_SELECTED_SCORE_BWD_WARP_GROUPED:-1}"
export MEGATRON_HISA_ASSUME_SORTED_POSITIONS="${MEGATRON_HISA_ASSUME_SORTED_POSITIONS:-1}"
export MEGATRON_HISA_FALLBACK_DENSE_IF_SHORT="${MEGATRON_HISA_FALLBACK_DENSE_IF_SHORT:-0}"
export MEGATRON_HISA_FUSED_INDEXER_LOSS="${MEGATRON_HISA_FUSED_INDEXER_LOSS:-1}"
export MEGATRON_DSA_COMPACT_TOPK_INDICES="${MEGATRON_DSA_COMPACT_TOPK_INDICES:-1}"
export MEGATRON_DSA_STREAM_TRITON_ATTENTION_CHUNKS="${MEGATRON_DSA_STREAM_TRITON_ATTENTION_CHUNKS:-1}"
export MEGATRON_DSA_SP_PROJECT_BEFORE_GATHER="${MEGATRON_DSA_SP_PROJECT_BEFORE_GATHER:-1}"
export MEGATRON_HISA_TARGET_TRITON="${MEGATRON_HISA_TARGET_TRITON:-1}"
export MEGATRON_HISA_TARGET_BLOCK_K="${MEGATRON_HISA_TARGET_BLOCK_K:-64}"
export MEGATRON_HISA_TARGET_ROW_CHUNK="${MEGATRON_HISA_TARGET_ROW_CHUNK:-128}"
export MEGATRON_HISA_KL_GRAD_TRITON="${MEGATRON_HISA_KL_GRAD_TRITON:-1}"
export MEGATRON_PREWARM_PIPELINE_P2P="${MEGATRON_PREWARM_PIPELINE_P2P:-1}"
export MEGATRON_DSA_TEACHER_SCORE_SCRATCH="${MEGATRON_DSA_TEACHER_SCORE_SCRATCH:-1}"
export MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH="${MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH:-1}"
export MEGATRON_WEIGHTED_SWIGLU_FUSER="${MEGATRON_WEIGHTED_SWIGLU_FUSER:-triton}"
export MEGATRON_FLASH_ADAMW_NVFP4_IMMEDIATE_CAST="${MEGATRON_FLASH_ADAMW_NVFP4_IMMEDIATE_CAST:-1}"
if [[ "${FINE_GRAINED_ACTIVATION_OFFLOADING:-1}" == "1" ]]; then
    export NVTE_CPU_OFFLOAD_V1="${NVTE_CPU_OFFLOAD_V1:-1}"
fi
APPLY_ROPE_FUSION="${APPLY_ROPE_FUSION:-1}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-60}"

DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
if [[ "$DISTRIBUTED_BACKEND" == "ncclx" ]]; then
    if ulimit -Sn "$(ulimit -Hn)" 2>/dev/null; then
        :
    fi
    export MEGATRON_USE_TORCHCOMMS="${MEGATRON_USE_TORCHCOMMS:-1}"
    export MEGATRON_NCCLX_MEM_POOL="${MEGATRON_NCCLX_MEM_POOL:-1}"
    export MEGATRON_NCCLX_RDMA="${MEGATRON_NCCLX_RDMA:-1}"
    export MEGATRON_NCCLX_RDMA_PROFILE="${MEGATRON_NCCLX_RDMA_PROFILE:-roce}"
    export MEGATRON_NCCLX_RDMA_BACKENDS="${MEGATRON_NCCLX_RDMA_BACKENDS:-ib,nvl,socket}"
    export TORCHCOMM_TIMEOUT_SECONDS="${TORCHCOMM_TIMEOUT_SECONDS:-$((DISTRIBUTED_TIMEOUT_MINUTES * 60))}"
    if [[ -z "${NCCL_SOCKET_IFNAME:-}" || "$NCCL_SOCKET_IFNAME" == "gpu" ]]; then
        export NCCL_SOCKET_IFNAME="gpu0rdma0"
    fi
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
    export NCCL_IGNORE_TOPO_LOAD_FAILURE="${NCCL_IGNORE_TOPO_LOAD_FAILURE:-0}"
    if [[ -z "${NCCL_TOPO_FILE_PATH:-}" && -f "$HOME/ncclx_topology.env" ]]; then
        export NCCL_TOPO_FILE_PATH="$HOME/ncclx_topology.env"
    fi
    if [[ -z "${NCCL_TOPO_FILE_PATH:-}" || ! -f "$NCCL_TOPO_FILE_PATH" ]]; then
        echo "DISTRIBUTED_BACKEND=ncclx requires NCCL_TOPO_FILE_PATH or $HOME/ncclx_topology.env" >&2
        exit 1
    fi
fi

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

MODEL_ID="${MODEL_ID:-BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$MODEL_ID}"

LOAD_CKPT="${LOAD_CKPT:-"$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron"}"
SAVE_CKPT="${SAVE_CKPT:-"$HOME/checkpoints/sft_deepseek_v32_reap_spinquant_actkv_nvfp4"}"
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

DISTRIBUTED_BACKEND_ARGS=(
    --distributed-backend "$DISTRIBUTED_BACKEND"
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
if [[ "$USE_STREAMBP" == "1" && "$CP" != "1" ]]; then
    echo "USE_STREAMBP=1 requires CP=1; StreamBP does not yet support context parallelism" >&2
    exit 1
fi
if [[ -z "${DECODER_FIRST_PIPELINE_NUM_LAYERS:-}" ]]; then
    if [[ "$PP" -eq 4 ]]; then
        DECODER_FIRST_PIPELINE_NUM_LAYERS=16
    else
        DECODER_FIRST_PIPELINE_NUM_LAYERS=31
    fi
fi

if [[ "$PP" -eq 4 && -z "${PIPELINE_MODEL_PARALLEL_LAYOUT:-}" && -z "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE:-}" && -z "${NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK:-}" ]]; then
    PIPELINE_MODEL_PARALLEL_LAYOUT="Et*9|t*7|t*7|t*7|t*8|t*8|t*8|t*7L"
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
    if [[ -z "${PIPELINE_MODEL_PARALLEL_LAYOUT:-}" ]]; then
        MODEL_PARALLEL_ARGS+=(--decoder-first-pipeline-num-layers "$DECODER_FIRST_PIPELINE_NUM_LAYERS")
    fi
fi
if [[ "$PP" -gt 1 && -n "${DECODER_LAST_PIPELINE_NUM_LAYERS:-}" ]]; then
    if [[ -z "${PIPELINE_MODEL_PARALLEL_LAYOUT:-}" ]]; then
        MODEL_PARALLEL_ARGS+=(--decoder-last-pipeline-num-layers "$DECODER_LAST_PIPELINE_NUM_LAYERS")
    fi
fi
if [[ "$PP" -gt 1 && -n "${PIPELINE_MODEL_PARALLEL_LAYOUT:-}" ]]; then
    MODEL_PARALLEL_ARGS+=(--pipeline-model-parallel-layout "$PIPELINE_MODEL_PARALLEL_LAYOUT")
fi
if [[ "$PP" -gt 1 && -n "${PIPELINE_PARALLEL_SCHEDULE:-}" ]]; then
    MODEL_PARALLEL_ARGS+=(--pipeline-parallel-schedule "$PIPELINE_PARALLEL_SCHEDULE")
fi
if [[ "${OVERLAP_MOE_EXPERT_PARALLEL_COMM:-0}" == "1" ]]; then
    MODEL_PARALLEL_ARGS+=(--overlap-moe-expert-parallel-comm)
fi
if [[ "${HIGH_PRIORITY_A2A_COMM_STREAM:-0}" == "1" ]]; then
    MODEL_PARALLEL_ARGS+=(--high-priority-a2a-comm-stream)
fi
if [[ "${OVERLAP_P2P_COMM_WARMUP_FLUSH:-1}" == "1" ]]; then
    MODEL_PARALLEL_ARGS+=(--overlap-p2p-communication-warmup-flush)
fi
if [[ "${EP_OVERLAP_EARLY_ATTN_MEMORY_RELEASE:-0}" == "1" ]]; then
    MODEL_PARALLEL_ARGS+=(--ep-overlap-early-attn-memory-release)
fi
if [[ "${DELAY_WGRAD_COMPUTE:-0}" == "1" ]]; then
    MODEL_PARALLEL_ARGS+=(--delay-wgrad-compute)
fi
if [[ "${OVERLAP_DISPATCH_BACKWARD_WITH_EXPERTS_WGRAD:-0}" == "1" ]]; then
    echo "ERROR: OVERLAP_DISPATCH_BACKWARD_WITH_EXPERTS_WGRAD is disabled in this SFT launcher: focused parity still shows BF16 delayed expert-wgrad drift/NaNs with non-fused accumulation." >&2
    exit 1
fi
if [[ -n "${MOE_HYBRIDEP_NUM_SMS_PREPROCESSING:-}" ]]; then
    MODEL_PARALLEL_ARGS+=(--moe-hybridep-num-sms-preprocessing "$MOE_HYBRIDEP_NUM_SMS_PREPROCESSING")
fi
if [[ "$PP" -gt 1 && -n "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE:-}" ]]; then
    MODEL_PARALLEL_ARGS+=(
        --num-layers-per-virtual-pipeline-stage "$NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE"
    )
fi
if [[ "$PP" -gt 1 && -n "${NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK:-}" ]]; then
    MODEL_PARALLEL_ARGS+=(
        --num-virtual-stages-per-pipeline-rank "$NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK"
    )
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

if [[ -z "${DSA_CHUNK_SIZE:-}" ]]; then
    if [[ "$USE_STREAMBP" == "1" ]]; then
        DSA_CHUNK_SIZE="${DSA_STREAMBP_CHUNK_SIZE:-4096}"
    else
        DSA_CHUNK_SIZE=256
    fi
fi

DSA_ARGS=(
    --experimental-attention-variant dsa
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk "${DSA_INDEXER_TOPK:-1024}"
    --dsa-indexer-loss-coeff "${DSA_INDEXER_LOSS_COEFF:-0.01}"
    --dsa-chunk-size "$DSA_CHUNK_SIZE"
)
if [[ "$APPLY_ROPE_FUSION" != "1" ]]; then
    MODEL_ARGS+=(--no-rope-fusion)
fi

# ======================
# MoE
# ======================
MOE_ARGS=(
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
    --moe-router-dtype fp32
    --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF:-1e-4}"
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
if [[ "${MOE_ROUTER_PADDING_FOR_QUANTIZATION:-0}" == "1" ]]; then
    MOE_ARGS+=(--moe-router-padding-for-quantization)
fi
if [[ "${MOE_ROUTER_QUANTILE_BIAS_SYNC_SCORES:-1}" == "0" ]]; then
    MOE_ARGS+=(--no-moe-router-quantile-bias-sync-scores)
fi
if [[ -n "${MOE_EXPERT_CAPACITY_FACTOR:-}" ]]; then
    MOE_ARGS+=(--moe-expert-capacity-factor "$MOE_EXPERT_CAPACITY_FACTOR")
fi
if [[ "${MOE_PAD_EXPERT_INPUT_TO_CAPACITY:-0}" == "1" ]]; then
    MOE_ARGS+=(--moe-pad-expert-input-to-capacity)
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
if [[ -z "${TURBOQUANT+x}" && "${USE_HIGGS:-0}" == "1" ]]; then
    TURBOQUANT=0
fi
if [[ "${TURBOQUANT:-0}" == "1" ]]; then
    TURBOQUANT_ARGS+=(
        --turboquant-kv-enabled
        --turboquant-kv-preset "${TURBOQUANT_KV_PRESET:-latent_2p5bit_nc}"
        --turboquant-kv-seed "${TURBOQUANT_KV_SEED:-0}"
    )
fi

HIGGS_ARGS=()
if [[ "${USE_HIGGS:-0}" == "1" ]]; then
    HIGGS_ARGS+=(
        --enable-higgs-dense-2bit-kv-cache
        --higgs-kv-preset "${HIGGS_KV_PRESET:-dense_2bit}"
    )
fi

INDEXCACHE_ARGS=()
if [[ "${INDEXCACHE:-1}" == "1" ]]; then
    INDEXCACHE_QUANT_METHOD="${DSA_INDEXCACHE_QUANTIZATION:-fp8_e4m3}"
    if [[ "${DSA_INDEXCACHE_HISA:-0}" == "1" && -z "${DSA_INDEXCACHE_QUANTIZATION:-}" ]]; then
        INDEXCACHE_QUANT_METHOD="nvfp4_e2m1_ue8m0"
    fi
    INDEXCACHE_ARGS+=(
        --dsa-indexcache-quantization "${INDEXCACHE_QUANT_METHOD}"
        --dsa-indexcache-quant-eps "${DSA_INDEXCACHE_QUANT_EPS:-1e-4}"
    )
    if [[ "${DSA_INDEXCACHE_HISA:-0}" == "1" ]]; then
        INDEXCACHE_ARGS+=(
            --dsa-indexcache-hisa-enabled
            --dsa-indexcache-hisa-block-size "${DSA_INDEXCACHE_HISA_BLOCK_SIZE:-128}"
            --dsa-indexcache-hisa-block-topk "${DSA_INDEXCACHE_HISA_BLOCK_TOPK:-64}"
            --dsa-indexcache-hisa-compression-ratio "${DSA_INDEXCACHE_HISA_COMPRESSION_RATIO:-4.0}"
        )
    fi
fi

# ======================
# Training
# ======================
TRAINING_ARGS=(
    --micro-batch-size "${MICRO_BATCH_SIZE:-4}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-64}"
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
    --use-distributed-optimizer
    --no-gradient-accumulation-fusion
)

if [[ "${FLASH_ADAMW_ECO:-1}" == "1" ]]; then
    TRAINING_ARGS+=(--flash-adamw-eco)
fi
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

STREAMBP_ARGS=()
if [[ "$USE_STREAMBP" == "1" ]]; then
    if [[ "$RECOMPUTE" == "1" ]]; then
        echo "USE_STREAMBP=1 is incompatible with full activation recompute; set RECOMPUTE=0" >&2
        exit 1
    fi
    STREAMBP_ARGS+=(--use-streambp)
    if [[ -n "${STREAMBP_CHUNK_SIZE:-}" ]]; then
        STREAMBP_ARGS+=(--streambp-chunk-size "$STREAMBP_CHUNK_SIZE")
    fi
    if [[ -n "${STREAMBP_LOGITS_CHUNK_SIZE:-}" ]]; then
        STREAMBP_ARGS+=(--streambp-logits-chunk-size "$STREAMBP_LOGITS_CHUNK_SIZE")
    fi
    if [[ -n "${STREAMBP_CHUNK_FORWARD:-}" ]]; then
        if [[ "$STREAMBP_CHUNK_FORWARD" == "1" ]]; then
            STREAMBP_ARGS+=(--streambp-chunk-forward)
        else
            STREAMBP_ARGS+=(--no-streambp-chunk-forward)
        fi
    fi
    if [[ -n "${STREAMBP_MOE_CHUNK_FORWARD:-}" ]]; then
        if [[ "$STREAMBP_MOE_CHUNK_FORWARD" == "1" ]]; then
            STREAMBP_ARGS+=(--streambp-moe-chunk-forward)
        else
            STREAMBP_ARGS+=(--no-streambp-moe-chunk-forward)
        fi
    fi
    if [[ -n "${STREAMBP_MOE_MLP_CHUNKS:-}" ]]; then
        STREAMBP_ARGS+=(--streambp-moe-mlp-chunks "$STREAMBP_MOE_MLP_CHUNKS")
    fi
    if [[ "${STREAMBP_SKIP_MOE:-0}" == "1" ]]; then
        STREAMBP_ARGS+=(--streambp-skip-moe)
    else
        STREAMBP_ARGS+=(--no-streambp-skip-moe)
    fi
    if [[ "${STREAMBP_SKIP_DSA:-0}" == "1" ]]; then
        STREAMBP_ARGS+=(--streambp-skip-dsa)
    else
        STREAMBP_ARGS+=(--no-streambp-skip-dsa)
    fi
    if [[ "${STREAMBP_VALIDATE:-0}" == "1" ]]; then
        STREAMBP_ARGS+=(--streambp-validate)
    fi
    if [[ "${STREAMBP_PROFILE:-0}" == "1" ]]; then
        STREAMBP_ARGS+=(--streambp-profile)
        STREAMBP_ARGS+=(--streambp-profile-rank "${STREAMBP_PROFILE_RANK:-0}")
        STREAMBP_ARGS+=(--streambp-profile-limit "${STREAMBP_PROFILE_LIMIT:-4}")
        if [[ -n "${STREAMBP_PROFILE_DIR:-}" ]]; then
            STREAMBP_ARGS+=(--streambp-profile-dir "$STREAMBP_PROFILE_DIR")
        fi
        if [[ -n "${STREAMBP_PROFILE_FILTER:-}" ]]; then
            STREAMBP_ARGS+=(--streambp-profile-filter "$STREAMBP_PROFILE_FILTER")
        fi
        if [[ "${STREAMBP_PROFILE_RECORD_SHAPES:-0}" == "1" ]]; then
            STREAMBP_ARGS+=(--streambp-profile-record-shapes)
        fi
        if [[ "${STREAMBP_PROFILE_WITH_STACK:-0}" == "1" ]]; then
            STREAMBP_ARGS+=(--streambp-profile-with-stack)
        fi
    fi
fi

ZCC_ARGS=()
if [[ "${ENABLE_ZCC:-0}" == "1" ]]; then
    export MEGATRON_FLASH_FUSE_STATE_BUFFER="${MEGATRON_FLASH_FUSE_STATE_BUFFER:-0}"
    ZCC_ARGS+=(
        --enable-zero-cost-checkpoint
        --zcc-workers-num "${ZCC_WORKERS_NUM:-1}"
        --zcc-flash-device "${ZCC_FLASH_DEVICE:-/dev/shm/megatron_zcc}"
        --zcc-durable-interval "${ZCC_DURABLE_INTERVAL:-10}"
        --zcc-compress "${ZCC_COMPRESS:-zstd:1}"
        --zcc-retain-latest "${ZCC_RETAIN_LATEST:-1}"
    )
    if [[ -n "${ZCC_FLASH_STRIPE:-}" ]]; then
        ZCC_ARGS+=(--zcc-flash-stripe "$ZCC_FLASH_STRIPE")
    fi
    if [[ -n "${ZCC_DURABLE_DIR:-}" ]]; then
        ZCC_ARGS+=(--zcc-durable-dir "$ZCC_DURABLE_DIR")
    fi
    if [[ -n "${ZCC_EXTRA_TENSOR_ATTRS:-}" ]]; then
        ZCC_ARGS+=(--zcc-extra-tensor-attrs "$ZCC_EXTRA_TENSOR_ATTRS")
    fi
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

OFFLOAD_ARGS=()
if [[ "${FINE_GRAINED_ACTIVATION_OFFLOADING:-1}" == "1" ]]; then
    OFFLOAD_MODULES_VALUE="${OFFLOAD_MODULES:-expert_fc1 core_attn attn_proj}"
    # shellcheck disable=SC2206
    OFFLOAD_MODULE_LIST=(${OFFLOAD_MODULES_VALUE})
    if ((${#OFFLOAD_MODULE_LIST[@]} == 0)); then
        echo "FINE_GRAINED_ACTIVATION_OFFLOADING=1 requires OFFLOAD_MODULES to name at least one module" >&2
        exit 1
    fi
    OFFLOAD_ARGS+=(--fine-grained-activation-offloading)
    OFFLOAD_ARGS+=(--offload-modules "${OFFLOAD_MODULE_LIST[@]}")
fi

ACTIVATION_ECO_ARGS=()
if [[ "${NVFP4_ACTIVATION_ECO:-1}" == "1" ]]; then
    ACTIVATION_ECO_ARGS+=(--nvfp4-activation-eco)
    ACT_ECO_MODULES_VALUE="${NVFP4_ACTIVATION_ECO_MODULES:-all}"
    # shellcheck disable=SC2206
    ACT_ECO_MODULE_LIST=(${ACT_ECO_MODULES_VALUE})
    if ((${#ACT_ECO_MODULE_LIST[@]} == 0)); then
        echo "NVFP4_ACTIVATION_ECO=1 requires NVFP4_ACTIVATION_ECO_MODULES to name at least one module" >&2
        exit 1
    fi
    ACTIVATION_ECO_ARGS+=(--nvfp4-activation-eco-modules "${ACT_ECO_MODULE_LIST[@]}")
    if [[ "${NVFP4_ACTIVATION_ECO_RECOMPUTE_ONLY:-1}" == "0" ]]; then
        ACTIVATION_ECO_ARGS+=(--no-nvfp4-activation-eco-recompute-only)
    fi
    ACTIVATION_ECO_ARGS+=(
        --nvfp4-activation-eco-quantizer-backend "${NVFP4_ACTIVATION_ECO_QUANTIZER_BACKEND:-te}"
        --nvfp4-activation-eco-correction-dtype "${NVFP4_ACTIVATION_ECO_CORRECTION_DTYPE:-fp32}"
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
    if [[ "${USE_PYTORCH_PROFILER:-0}" == "1" ]]; then
        PROFILING_ARGS+=(--use-pytorch-profiler)
    fi
    if [[ "${PYTORCH_PROFILER_COLLECT_SHAPES:-0}" == "1" ]]; then
        PROFILING_ARGS+=(--pytorch-profiler-collect-shapes)
    fi
    if [[ "${PYTORCH_PROFILER_COLLECT_CALLSTACK:-0}" == "1" ]]; then
        PROFILING_ARGS+=(--pytorch-profiler-collect-callstack)
    fi
    if [[ "${PYTORCH_PROFILER_COLLECT_CHAKRA:-0}" == "1" ]]; then
        PROFILING_ARGS+=(--pytorch-profiler-collect-chakra)
    fi
    if [[ -n "${PROFILE_RANKS:-}" ]]; then
        # shellcheck disable=SC2206
        PROFILE_RANK_ARGS=(${PROFILE_RANKS//,/ })
        PROFILING_ARGS+=(--profile-ranks "${PROFILE_RANK_ARGS[@]}")
    fi
fi

if [[ "$USE_MEGATRON_FSDP" == "1" ]]; then
    CKPT_FORMAT_VALUE="${CKPT_FORMAT:-fsdp_dtensor}"
else
    CKPT_FORMAT_VALUE="${CKPT_FORMAT:-torch_dist}"
fi

CKPT_ARGS=(
    --eval-interval "${EVAL_INTERVAL:-100}"
    --eval-iters "${EVAL_ITERS:-0}"
    --load "$LOAD_CKPT"
    --distributed-timeout-minutes "$DISTRIBUTED_TIMEOUT_MINUTES"
    --ckpt-format "$CKPT_FORMAT_VALUE"
    --auto-detect-ckpt-format
)
if [[ "${DISABLE_SAVE:-0}" != "1" ]]; then
    CKPT_ARGS+=(
        --save-interval "${SAVE_INTERVAL:-500}"
        --save "$SAVE_CKPT"
    )
    if [[ -n "${SAVE_RETAIN_INTERVAL:-}" ]]; then
        CKPT_ARGS+=(--save-retain-interval "$SAVE_RETAIN_INTERVAL")
    fi
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
    "${DISTRIBUTED_BACKEND_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    "${MLA_ARGS[@]}"
    "${DSA_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${MODEL_PARALLEL_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${STREAMBP_ARGS[@]}"
    "${ZCC_ARGS[@]}"
    "${FSDP_ARGS[@]}"
    "${RECOMPUTE_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
    "${ACTIVATION_ECO_ARGS[@]}"
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
