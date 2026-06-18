#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}"
cd "$REPO_DIR"

command -v tmux >/dev/null || {
  echo "tmux is required on the controller" >&2
  exit 127
}

SSH_USER="${SSH_USER:-sjpat}"
if [[ -z "${SSH_KEY:-}" ]]; then
  if [[ -f "$HOME/google_compute_engine" ]]; then
    SSH_KEY="$HOME/google_compute_engine"
  else
    SSH_KEY="$HOME/.ssh/google_compute_engine"
  fi
fi

ACTION="${ACTION:-start}"
ATTACH="${ATTACH:-1}"
SESSION="${SESSION:-gcp-a4-deepseek}"
REMOTE_SESSION="${REMOTE_SESSION:-${SESSION}-train}"
REMOTE_ENTRYPOINT="${REMOTE_ENTRYPOINT:-handoff/gcp-a4/scripts/launch_deepseek_nvfp4.sh}"
RUN_NAME="${WANDB_EXP_NAME:-corsaire-1-research-preview}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-$REPO_DIR}"
LOG_ROOT="${LOG_ROOT:-$HOME/logs/$RUN_NAME}"
AUTH_PATH="${AUTH_PATH:-$HOME/.config/megatron/auth.env}"
SYNC_REPO="${SYNC_REPO:-1}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP="${TP:-8}"
PP="${PP:-5}"
CP="${CP:-1}"
EP="${EP:-8}"
ETP="${ETP:-1}"
STRICT_ALL_NODES="${STRICT_ALL_NODES:-1}"
MAX_NODES="${MAX_NODES:-}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-10}"
SEQ_LENGTH="${SEQ_LENGTH:-32768}"
DSA_INDEXER_TOPK="${DSA_INDEXER_TOPK:-1024}"
DSA_INDEXER_LOSS_COEFF="${DSA_INDEXER_LOSS_COEFF:-0.1}"
TRAIN_TOKEN_TARGET="${TRAIN_TOKEN_TARGET:-24691703808}"
LR="${LR:-5.0e-5}"
NUM_WORKERS="${NUM_WORKERS:-16}"
ENABLE_VPP="${ENABLE_VPP:-1}"
USE_STREAMBP="${USE_STREAMBP:-0}"
INDEXCACHE="${INDEXCACHE:-1}"
DSA_INDEXCACHE_HISA="${DSA_INDEXCACHE_HISA:-1}"
TURBOQUANT="${TURBOQUANT:-0}"
USE_HIGGS="${USE_HIGGS:-1}"
FLASH_ADAMW_ECO="${FLASH_ADAMW_ECO:-1}"
FLASH_ADAMW_FSDP_ECO_INJECT="${FLASH_ADAMW_FSDP_ECO_INJECT:-0}"
FLASH_ADAMW_ECO_LR_FLOOR="${FLASH_ADAMW_ECO_LR_FLOOR:-base}"
FLASH_ADAMW_ECO_PROJECTION="${FLASH_ADAMW_ECO_PROJECTION:-gain}"
FLASH_ADAMW_ECO_PROJECTION_SCALE_BUDGET="${FLASH_ADAMW_ECO_PROJECTION_SCALE_BUDGET:-2.0}"
FLASH_ADAMW_ECO_PROJECTION_GAIN_BUDGET="${FLASH_ADAMW_ECO_PROJECTION_GAIN_BUDGET:-0.25}"
FLASH_ADAMW_ECO_PROJECTION_STEPS="${FLASH_ADAMW_ECO_PROJECTION_STEPS:-16}"
LOAD_CKPT="${LOAD_CKPT:-$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron_tp8_pp1_ep8}"
SAVE_CKPT="${SAVE_CKPT:-$HOME/checkpoints/sft_deepseek_v32_reap_spinquant_actkv_nvfp4_tp8_pp5_ep8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-500}"
HF_UPLOAD_CHECKPOINTS="${HF_UPLOAD_CHECKPOINTS:-1}"
HF_REPO_ID="${HF_REPO_ID:-BlaiseAI/corsaire-1-research-preview}"
HF_UPLOAD_FOLDER_PREFIX="${HF_UPLOAD_FOLDER_PREFIX:-corsaire-1-research-preview}"
HF_UPLOAD_INTERVAL="${HF_UPLOAD_INTERVAL:-500}"
HF_UPLOAD_RETAIN="${HF_UPLOAD_RETAIN:-1}"
HF_UPLOAD_STABLE_SECONDS="${HF_UPLOAD_STABLE_SECONDS:-180}"
HF_UPLOAD_POLL_SECONDS="${HF_UPLOAD_POLL_SECONDS:-60}"
HF_UPLOAD_FINALIZE_TIMEOUT_SECONDS="${HF_UPLOAD_FINALIZE_TIMEOUT_SECONDS:-7200}"
HF_UPLOAD_PRIVATE="${HF_UPLOAD_PRIVATE:-1}"
HF_UPLOAD_RETENTION_NODE_RANK="${HF_UPLOAD_RETENTION_NODE_RANK:-0}"

MODEL_PARALLEL_SIZE=$((TP * PP * CP))
EXPERT_MODEL_PARALLEL_SIZE=$((ETP * EP * PP))

mapfile -t rows < <(OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | tail -n +2)
if (( ${#rows[@]} == 0 )); then
  echo "No running GCP nodes discovered" >&2
  exit 2
fi

names=()
ips=()
for row in "${rows[@]}"; do
  IFS=, read -r _rank name internal_ip _nat_ip <<<"$row"
  names+=("$name")
  ips+=("$internal_ip")
done

discovered_count="${#names[@]}"
use_count="${MAX_NODES:-$discovered_count}"
if (( use_count > discovered_count )); then
  echo "MAX_NODES=$use_count exceeds discovered node count $discovered_count" >&2
  exit 2
fi

while (( use_count > 0 )); do
  world_size=$((use_count * GPUS_PER_NODE))
  if (( world_size % MODEL_PARALLEL_SIZE == 0 && world_size % EXPERT_MODEL_PARALLEL_SIZE == 0 )); then
    break
  fi
  if [[ "$STRICT_ALL_NODES" == "1" || -n "$MAX_NODES" ]]; then
    cat >&2 <<EOF
Invalid distributed shape:
  nodes=$use_count gpus_per_node=$GPUS_PER_NODE world_size=$world_size
  TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP
  required: world_size divisible by $MODEL_PARALLEL_SIZE and $EXPERT_MODEL_PARALLEL_SIZE
EOF
    exit 2
  fi
  use_count=$((use_count - 1))
done

world_size=$((use_count * GPUS_PER_NODE))
dp=$((world_size / MODEL_PARALLEL_SIZE))
expert_dp=$((world_size / EXPERT_MODEL_PARALLEL_SIZE))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * dp * GRAD_ACCUM_STEPS))}"
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * dp) != 0 )); then
  echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE*DP=$((MICRO_BATCH_SIZE * dp))" >&2
  exit 2
fi

LOG_NODE_RANK="${LOG_NODE_RANK:-0}"
LOG_LOCAL_RANK="${LOG_LOCAL_RANK:-0}"
WANDB_NODE_RANK="${WANDB_NODE_RANK:-$((use_count - 1))}"
WANDB_LOCAL_RANK="${WANDB_LOCAL_RANK:-$((GPUS_PER_NODE - 1))}"

ssh_base=(
  ssh
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o UserKnownHostsFile=/dev/null
  -o StrictHostKeyChecking=no
)

remote_sh() {
  local ip="$1"
  shift
  "${ssh_base[@]}" "$SSH_USER@$ip" "$@"
}

q() {
  printf '%q' "$1"
}

sync_changed_files() {
  local ip="$1"
  mapfile -t sync_files < <(
    {
      git diff --name-only
      git diff --name-only --cached
      git ls-files --others --exclude-standard
    } | sort -u | grep -Ev '^(artifacts/|\.pytest_cache/|.*__pycache__/|.*\.pyc$)' || true
  )
  if (( ${#sync_files[@]} == 0 )); then
    return
  fi
  tar -cf - "${sync_files[@]}" | remote_sh "$ip" "cd $(q "$REMOTE_REPO_DIR") && tar -xf -"
}

write_remote_runner() {
  local node_rank="$1"
  local name="$2"
  local ip="$3"
  local node_log_dir="$LOG_ROOT/nodes/node$(printf '%03d' "$node_rank")-$name"
  local runner="$node_log_dir/runner.sh"

  remote_sh "$ip" "mkdir -p $(q "$node_log_dir")"
  {
    cat <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ -f $(q "$AUTH_PATH") ]]; then
  # shellcheck disable=SC1090
  source $(q "$AUTH_PATH")
fi
export PATH="\$HOME/.local/bin:\$PATH"
export REPO_DIR=$(q "$REMOTE_REPO_DIR")
export NNODES=$(q "$use_count")
export GPUS_PER_NODE=$(q "$GPUS_PER_NODE")
export NODE_RANK=$(q "$node_rank")
export HOSTNAME_SHORT=$(q "$name")
export MASTER_ADDR=$(q "${ips[0]}")
export MASTER_PORT=$(q "${MASTER_PORT:-29673}")
export TP=$(q "$TP")
export PP=$(q "$PP")
export CP=$(q "$CP")
export EP=$(q "$EP")
export ETP=$(q "$ETP")
export ENABLE_VPP=$(q "$ENABLE_VPP")
export USE_STREAMBP=$(q "$USE_STREAMBP")
export MICRO_BATCH_SIZE=$(q "$MICRO_BATCH_SIZE")
export GLOBAL_BATCH_SIZE=$(q "$GLOBAL_BATCH_SIZE")
export SEQ_LENGTH=$(q "$SEQ_LENGTH")
export DSA_INDEXER_TOPK=$(q "$DSA_INDEXER_TOPK")
export DSA_INDEXER_LOSS_COEFF=$(q "$DSA_INDEXER_LOSS_COEFF")
export TRAIN_TOKEN_TARGET=$(q "$TRAIN_TOKEN_TARGET")
export LR=$(q "$LR")
export NUM_WORKERS=$(q "$NUM_WORKERS")
export INDEXCACHE=$(q "$INDEXCACHE")
export DSA_INDEXCACHE_HISA=$(q "$DSA_INDEXCACHE_HISA")
export TURBOQUANT=$(q "$TURBOQUANT")
export USE_HIGGS=$(q "$USE_HIGGS")
export FLASH_ADAMW_ECO=$(q "$FLASH_ADAMW_ECO")
export FLASH_ADAMW_FSDP_ECO_INJECT=$(q "$FLASH_ADAMW_FSDP_ECO_INJECT")
export FLASH_ADAMW_ECO_LR_FLOOR=$(q "$FLASH_ADAMW_ECO_LR_FLOOR")
export FLASH_ADAMW_ECO_PROJECTION=$(q "$FLASH_ADAMW_ECO_PROJECTION")
export FLASH_ADAMW_ECO_PROJECTION_SCALE_BUDGET=$(q "$FLASH_ADAMW_ECO_PROJECTION_SCALE_BUDGET")
export FLASH_ADAMW_ECO_PROJECTION_GAIN_BUDGET=$(q "$FLASH_ADAMW_ECO_PROJECTION_GAIN_BUDGET")
export FLASH_ADAMW_ECO_PROJECTION_STEPS=$(q "$FLASH_ADAMW_ECO_PROJECTION_STEPS")
export LOAD_CKPT=$(q "$LOAD_CKPT")
export SAVE_CKPT=$(q "$SAVE_CKPT")
export SAVE_INTERVAL=$(q "$SAVE_INTERVAL")
export SAVE_RETAIN_INTERVAL=$(q "$SAVE_RETAIN_INTERVAL")
export WANDB_ENTITY="\${WANDB_ENTITY:-blaise-ai}"
export WANDB_PROJECT="\${WANDB_PROJECT:-corsaire-1}"
export WANDB_EXP_NAME=$(q "$RUN_NAME")
export WANDB_SAVE_DIR="\${WANDB_SAVE_DIR:-\$SAVE_CKPT/wandb}"
# Offline is the default for every training run. To intentionally publish live,
# set both WANDB_FORCE_OFFLINE=0 and WANDB_MODE=online at launch.
export WANDB_FORCE_OFFLINE="\${WANDB_FORCE_OFFLINE:-1}"
if [[ "\$WANDB_FORCE_OFFLINE" == "1" ]]; then
  export WANDB_MODE=offline
else
  export WANDB_MODE="\${WANDB_MODE:-offline}"
fi
export HF_UPLOAD_CHECKPOINTS=$(q "$HF_UPLOAD_CHECKPOINTS")
export HF_REPO_ID=$(q "$HF_REPO_ID")
export HF_UPLOAD_FOLDER_PREFIX=$(q "$HF_UPLOAD_FOLDER_PREFIX")
export HF_UPLOAD_INTERVAL=$(q "$HF_UPLOAD_INTERVAL")
export HF_UPLOAD_RETAIN=$(q "$HF_UPLOAD_RETAIN")
export HF_UPLOAD_STABLE_SECONDS=$(q "$HF_UPLOAD_STABLE_SECONDS")
export HF_UPLOAD_POLL_SECONDS=$(q "$HF_UPLOAD_POLL_SECONDS")
export HF_UPLOAD_FINALIZE_TIMEOUT_SECONDS=$(q "$HF_UPLOAD_FINALIZE_TIMEOUT_SECONDS")
export HF_UPLOAD_PRIVATE=$(q "$HF_UPLOAD_PRIVATE")
export HF_UPLOAD_RETENTION_NODE_RANK=$(q "$HF_UPLOAD_RETENTION_NODE_RANK")
export HF_UPLOAD_EXPECTED_NODES=$(q "$use_count")
export TORCHRUN_LOG_DIR=$(q "$node_log_dir/torchrun")
export TORCHRUN_REDIRECTS="\${TORCHRUN_REDIRECTS:-3}"
export LOG_DIR=$(q "$node_log_dir")
export NCCL_DEBUG="\${NCCL_DEBUG:-WARN}"
mkdir -p "\$TORCHRUN_LOG_DIR"
EOF
    if [[ "$node_rank" == "$LOG_NODE_RANK" ]]; then
      printf 'export TORCHRUN_TEE="${TORCHRUN_TEE:-3}"\n'
      printf 'export TORCHRUN_LOCAL_RANKS_FILTER="${TORCHRUN_LOCAL_RANKS_FILTER:-%s}"\n' "$LOG_LOCAL_RANK"
    elif [[ "$node_rank" == "$WANDB_NODE_RANK" ]]; then
      printf 'export TORCHRUN_TEE="${TORCHRUN_TEE:-3}"\n'
      printf 'export TORCHRUN_LOCAL_RANKS_FILTER="${TORCHRUN_LOCAL_RANKS_FILTER:-%s}"\n' "$WANDB_LOCAL_RANK"
    fi
    cat <<'EOF'
cd "$REPO_DIR"
echo "[$(hostname -s)] starting $WANDB_EXP_NAME rank=$NODE_RANK at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
    printf 'exec %q\n' "$REMOTE_ENTRYPOINT"
  } | remote_sh "$ip" "cat > $(q "$runner") && chmod +x $(q "$runner")"
}

write_controller_helpers() {
  local controller_dir="$LOG_ROOT/controller"
  mkdir -p "$controller_dir"
  {
    echo "rank,name,ip,role"
    for idx in $(seq 0 $((use_count - 1))); do
      role="worker"
      [[ "$idx" == "$LOG_NODE_RANK" ]] && role="rank-log"
      [[ "$idx" == "$WANDB_NODE_RANK" ]] && role="wandb-writer"
      printf "%s,%s,%s,%s\n" "$idx" "${names[$idx]}" "${ips[$idx]}" "$role"
    done
  } > "$controller_dir/nodes.csv"

  cat > "$controller_dir/fleet_status.sh" <<EOF
#!/usr/bin/env bash
set +e
echo "run=$RUN_NAME nodes=$use_count world=$world_size TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP DP=$dp expert_DP=$expert_dp"
echo "seq=$SEQ_LENGTH dsa_topk=$DSA_INDEXER_TOPK dsa_loss_coeff=$DSA_INDEXER_LOSS_COEFF streambp=$USE_STREAMBP vpp=$ENABLE_VPP mbs=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE lr=$LR workers=$NUM_WORKERS target_tokens=$TRAIN_TOKEN_TARGET"
echo "higgs=$USE_HIGGS indexcache=$INDEXCACHE hisa=$DSA_INDEXCACHE_HISA turboquant=$TURBOQUANT flash_adamw_eco=$FLASH_ADAMW_ECO flash_adamw_fsdp_eco_inject=$FLASH_ADAMW_FSDP_ECO_INJECT eco_lr_floor=$FLASH_ADAMW_ECO_LR_FLOOR eco_projection=$FLASH_ADAMW_ECO_PROJECTION"
echo "load=$LOAD_CKPT save=$SAVE_CKPT save_interval=$SAVE_INTERVAL retain_interval=$SAVE_RETAIN_INTERVAL"
echo "wandb_entity=blaise-ai wandb_project=corsaire-1 wandb_run=$RUN_NAME wandb_mode=\${WANDB_MODE:-offline} hf_upload=$HF_UPLOAD_CHECKPOINTS hf_repo=$HF_REPO_ID hf_interval=$HF_UPLOAD_INTERVAL hf_retain=$HF_UPLOAD_RETAIN"
echo
tail -n +2 $(q "$controller_dir/nodes.csv") | while IFS=, read -r rank name ip role; do
  printf "%3s %-28s %-12s " "\$rank" "\$name" "\$role"
  ${ssh_base[*]} "$SSH_USER@\$ip" "tmux has-session -t $(q "$REMOTE_SESSION") 2>/dev/null && printf train=running || printf train=stopped; printf ' '; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{u+=\$1; m+=\$2; t+=\$3; n++} END {if (n) printf \"gpu_avg=%d%% mem=%d/%dMiB\", u/n, m, t; else printf \"gpu=na\"}'" 2>/dev/null || printf "ssh=failed"
  echo
done
EOF

  cat > "$controller_dir/fleet_gpu.sh" <<EOF
#!/usr/bin/env bash
set +e
while true; do
  clear
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  tail -n +2 $(q "$controller_dir/nodes.csv") | while IFS=, read -r rank name ip role; do
    printf "\\n[%s] %s %s\\n" "\$rank" "\$name" "\$role"
    ${ssh_base[*]} "$SSH_USER@\$ip" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits" 2>/dev/null | sed 's/^/  /' || echo "  ssh failed"
  done
  sleep "\${GPU_WATCH_INTERVAL:-15}"
done
EOF

  chmod +x "$controller_dir/fleet_status.sh" "$controller_dir/fleet_gpu.sh"
}

start_nodes() {
  mkdir -p "$LOG_ROOT"
  write_controller_helpers
  for idx in $(seq 0 $((use_count - 1))); do
    name="${names[$idx]}"
    ip="${ips[$idx]}"
    echo "[$idx] preparing $name ($ip)" >&2
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      node_log_dir="$LOG_ROOT/nodes/node$(printf '%03d' "$idx")-$name"
      echo "[$idx] would sync changed files and start tmux session $REMOTE_SESSION -> $node_log_dir/train.log" >&2
      continue
    fi
    if [[ "$SYNC_REPO" == "1" ]]; then
      sync_changed_files "$ip"
    fi
    write_remote_runner "$idx" "$name" "$ip"
    node_log_dir="$LOG_ROOT/nodes/node$(printf '%03d' "$idx")-$name"
    runner="$node_log_dir/runner.sh"
    train_log="$node_log_dir/train.log"
    remote_sh "$ip" "tmux kill-session -t $(q "$REMOTE_SESSION") 2>/dev/null || true; tmux new-session -d -s $(q "$REMOTE_SESSION") 'bash $(q "$runner") >> $(q "$train_log") 2>&1'"
  done
}

stop_nodes() {
  for idx in $(seq 0 $((use_count - 1))); do
    remote_sh "${ips[$idx]}" "tmux kill-session -t $(q "$REMOTE_SESSION") 2>/dev/null || true" || true
  done
}

open_dashboard() {
  write_controller_helpers
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" -n overview "watch -n 20 'bash $(q "$LOG_ROOT/controller/fleet_status.sh")'"
  tmux new-window -t "$SESSION" -n gpu "bash $(q "$LOG_ROOT/controller/fleet_gpu.sh")"

  rank_name="${names[$LOG_NODE_RANK]}"
  rank_ip="${ips[$LOG_NODE_RANK]}"
  rank_log="$LOG_ROOT/nodes/node$(printf '%03d' "$LOG_NODE_RANK")-$rank_name/train.log"
  tmux new-window -t "$SESSION" -n rank-log "${ssh_base[*]} -tt $SSH_USER@$rank_ip 'tail -n +1 -F $(q "$rank_log")'"

  wandb_name="${names[$WANDB_NODE_RANK]}"
  wandb_ip="${ips[$WANDB_NODE_RANK]}"
  wandb_log="$LOG_ROOT/nodes/node$(printf '%03d' "$WANDB_NODE_RANK")-$wandb_name/train.log"
  tmux new-window -t "$SESSION" -n wandb "${ssh_base[*]} -tt $SSH_USER@$wandb_ip 'tail -n +1 -F $(q "$wandb_log")'"

  tmux new-window -t "$SESSION" -n control "printf '%s\n' 'run: $RUN_NAME' 'logs: $LOG_ROOT' 'remote session: $REMOTE_SESSION' 'stop: ACTION=stop handoff/gcp-a4/scripts/launch_fleet_tmux.sh' 'nodes: $LOG_ROOT/controller/nodes.csv'; exec bash"
  if [[ "$ATTACH" == "1" ]]; then
    tmux attach-session -t "$SESSION"
  else
    echo "Dashboard ready: tmux attach -t $SESSION" >&2
  fi
}

cat >&2 <<EOF
GCP A4 fleet:
  discovered_nodes=$discovered_count selected_nodes=$use_count world_size=$world_size
  TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP DP=$dp expert_dp=$expert_dp
  MBS=$MICRO_BATCH_SIZE GBS=$GLOBAL_BATCH_SIZE grad_accum=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * dp)))
  seq=$SEQ_LENGTH dsa_topk=$DSA_INDEXER_TOPK dsa_loss_coeff=$DSA_INDEXER_LOSS_COEFF lr=$LR workers=$NUM_WORKERS train_token_target=$TRAIN_TOKEN_TARGET
  streambp=$USE_STREAMBP vpp=$ENABLE_VPP run=$RUN_NAME
  higgs=$USE_HIGGS indexcache=$INDEXCACHE hisa=$DSA_INDEXCACHE_HISA turboquant=$TURBOQUANT flash_adamw_eco=$FLASH_ADAMW_ECO flash_adamw_fsdp_eco_inject=$FLASH_ADAMW_FSDP_ECO_INJECT
  hf_upload=$HF_UPLOAD_CHECKPOINTS hf_repo=$HF_REPO_ID hf_interval=$HF_UPLOAD_INTERVAL hf_retain=$HF_UPLOAD_RETAIN
  entrypoint=$REMOTE_ENTRYPOINT
  load_ckpt=$LOAD_CKPT
  save_ckpt=$SAVE_CKPT save_interval=$SAVE_INTERVAL retain_interval=$SAVE_RETAIN_INTERVAL
EOF

case "$ACTION" in
  start)
    start_nodes
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      open_dashboard
    fi
    ;;
  dashboard)
    open_dashboard
    ;;
  status)
    write_controller_helpers
    bash "$LOG_ROOT/controller/fleet_status.sh"
    ;;
  stop)
    stop_nodes
    ;;
  *)
    echo "Unsupported ACTION=$ACTION. Use start, dashboard, status, or stop." >&2
    exit 2
    ;;
esac
