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

SESSION="${SESSION:-deepseek_sft_real}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; attaching." >&2
    exec tmux attach-session -t "$SESSION"
fi

TS="${TS:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="${WANDB_EXP_NAME:-deepseek-v32-reap-sft-2b-${TS}}"
LOG_DIR="${LOG_DIR:-"$HOME/logs"}"
LOG0="$LOG_DIR/${RUN_NAME}_node0.log"
LOG1="$LOG_DIR/${RUN_NAME}_node1.log"
mkdir -p "$LOG_DIR"

REMOTE_HOST="${REMOTE_HOST:-sjpat@10.180.0.45}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/google_compute_engine}"
REMOTE_MEGATRON_DIR="${REMOTE_MEGATRON_DIR:-/home/sjpat/Megatron-LM}"
REMOTE_RUNNER="/tmp/${RUN_NAME}_node1.sh"

MASTER_ADDR="${MASTER_ADDR:-10.200.0.21}"
MASTER_PORT="${MASTER_PORT:-29673}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-"$HOME/.cache/triton/deepseek_v32_reap_sft"}"

# 2x the 1B-token target: 61056 samples * 32768 tokens/sample ~= 2.0007B tokens.
TRAIN_SAMPLES="${TRAIN_SAMPLES:-61056}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
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
Train samples: $TRAIN_SAMPLES
Save every:    $SAVE_INTERVAL updates
Retention:     keep latest normal Megatron checkpoint only
ZCC:           ENABLE_ZCC=${ENABLE_ZCC:-0} (last successful probe used 0)
Triton cache:  TRITON_CACHE_AUTOTUNING=1 TRITON_CACHE_DIR=$TRITON_CACHE_DIR
EOF

if [[ "${SYNC_REMOTE:-1}" == "1" ]]; then
    mapfile -t SYNC_FILES < <(
        {
            git diff --name-only
            git diff --name-only --cached
            git ls-files --others --exclude-standard
        } | sort -u
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
export NNODES="${NNODES:-2}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-gpu}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TP="${TP:-4}"
export PP="${PP:-4}"
export CP="${CP:-1}"
export EP="${EP:-4}"
export ETP="${ETP:-1}"
export DECODER_FIRST_PIPELINE_NUM_LAYERS="${DECODER_FIRST_PIPELINE_NUM_LAYERS:-16}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export GRAD_REDUCE_IN_BF16="${GRAD_REDUCE_IN_BF16:-1}"
export DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-120}"
export WANDB_ENTITY="${WANDB_ENTITY:-blaise-ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-corsaire-1}"
export WANDB_EXP_NAME="${RUN_NAME}"
export LOG_MEMORY_INTERVAL="${LOG_MEMORY_INTERVAL:-10}"
export LOG_NUM_ZEROS_IN_GRAD="${LOG_NUM_ZEROS_IN_GRAD:-1}"
export TRAIN_SAMPLES="${TRAIN_SAMPLES}"
export SAVE_INTERVAL="${SAVE_INTERVAL}"
export SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export TENSORBOARD_LOG_INTERVAL="${TENSORBOARD_LOG_INTERVAL:-1}"
export USE_STREAMBP="${USE_STREAMBP:-1}"
export STREAMBP_CHUNK_SIZE="${STREAMBP_CHUNK_SIZE:-2048}"
export STREAMBP_MOE_CHUNK_FORWARD="${STREAMBP_MOE_CHUNK_FORWARD:-0}"
export DSA_CHUNK_SIZE="${DSA_CHUNK_SIZE:-2048}"
export DSA_INDEXER_TOPK="${DSA_INDEXER_TOPK:-2048}"
export MEGATRON_DSA_TRITON_BF16_GRAD_ATOMICS="${MEGATRON_DSA_TRITON_BF16_GRAD_ATOMICS:-1}"
export MEGATRON_DSA_TRITON_BWD_NUM_WARPS="${MEGATRON_DSA_TRITON_BWD_NUM_WARPS:-2}"
export FLASH_ADAMW_COMPRESS_STATE_DICT="${FLASH_ADAMW_COMPRESS_STATE_DICT:-1}"
export ENABLE_ZCC="${ENABLE_ZCC:-0}"
export ZCC_RETAIN_LATEST="${ZCC_RETAIN_LATEST:-1}"
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
    echo "echo '[node0] starting $RUN_NAME at '\"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
    echo "examples/sft/run_sft_deepseek_nvfp4.sh 2>&1 | tee '$LOG0'"
} > "$NODE0_RUNNER"

{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    write_env_block 1
    echo "cd '$REMOTE_MEGATRON_DIR'"
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
tmux attach-session -t "$SESSION"
