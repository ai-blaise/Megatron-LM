#!/bin/bash

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
# Single-node NVLink: override cluster-level NCCL env vars that assume
# multi-node IB fabric. Let NCCL auto-select algo/proto for NVLink topology.
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
#export LOG_LEVEL=${LOG_LEVEL:-INFO}
#export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
#export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
#export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
#export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
#export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

CHECKPOINT_PATH=${1:-"checkpoints/llama3_8b_fp4"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/llama3_8b_fp4"}
TOKENIZER_ARG=${3:-"MOCK"} # Path to tokenizer model, or "MOCK"
DATA_ARG=${4:-"MOCK"}     # Data prefix, or "MOCK"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters
TP_SIZE=1
CP_SIZE=1
PP_SIZE=1
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
NUM_LAYERS=32
DTYPE="fp8"
SEQ_LENGTH=8192
MAX_POSITION_EMBEDDINGS=8192

# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="${PWD}/benchmark_cache_llama3_8b_fp4"
mkdir -p "$DATA_CACHE_PATH"
mkdir -p snapshots

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
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
    --no-gradient-accumulation-fusion
)

OPTIMIZER=${OPTIMIZER:-adam}

TRAINING_ARGS=(
    --optimizer $OPTIMIZER
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters 8
    --lr-decay-iters 6
    --lr-warmup-iters 2
    --lr 0.00015
    --min-lr 0.00001
    --decoupled-lr 5.0e-4
    --decoupled-min-lr 4.5e-5
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level 1
)

# Precision args
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
elif [[ "$DTYPE" == "fp4" ]]; then
    DTYPE_ARGS+=(
        "--fp4-format e2m1"
        "--fp4-param-gather"
    )
fi

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --sequence-parallel
)

# Distributed Data Parallel (DDP) arguments
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")


# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 128256"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--tiktoken-pattern v2"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type HuggingFaceTokenizer"
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
        "--vocab-size 128256"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 0
    --eval-interval 1000
    --save-interval 8
    --log-throughput
    --use-pytorch-profiler
    --pytorch-profiler-collect-shapes
    --pytorch-profiler-collect-callstack
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --profile-ranks 0
    --record-memory-history
    --memory-snapshot-path snapshots/llama3_8b_fp4_${OPTIMIZER}.pickle
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --tensorboard-dir "${TENSORBOARD_LOGS_PATH}/${OPTIMIZER}"
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

# Run the training command
uv run torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

set +x
