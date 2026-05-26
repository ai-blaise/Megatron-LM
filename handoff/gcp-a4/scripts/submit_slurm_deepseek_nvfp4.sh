#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}"
cd "$REPO_DIR"

SSH_USER="${SSH_USER:-sjpat}"
if [[ -z "${SSH_KEY:-}" ]]; then
  if [[ -f "$HOME/google_compute_engine" ]]; then
    SSH_KEY="$HOME/google_compute_engine"
  else
    SSH_KEY="$HOME/.ssh/google_compute_engine"
  fi
fi

PARTITION_NAME="${PARTITION_NAME:-a4}"
if [[ -z "${NNODES+x}" && -n "${MAX_NODES:-}" ]]; then
  NNODES="$MAX_NODES"
else
  # The current default training shape is TP8 x CP4 x PP1, which requires a
  # world size divisible by 32. Use 12 A4 nodes by default and leave the rest of
  # the spot fleet as warm redundancy unless the caller explicitly overrides it.
  NNODES="${NNODES:-12}"
fi
if [[ -z "${MAX_NODES+x}" ]]; then
  export MAX_NODES="$NNODES"
fi
SLURM_NODELIST="${SLURM_NODELIST:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
JOB_NAME="${JOB_NAME:-gcp-a4-deepseek-sft}"
TIME_LIMIT="${TIME_LIMIT:-}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
ENTRYPOINT="${ENTRYPOINT:-handoff/gcp-a4/scripts/launch_deepseek_nvfp4.sh}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-$REPO_DIR}"
SYNC_REPO="${SYNC_REPO:-1}"
SUBMIT_DRY_RUN="${SUBMIT_DRY_RUN:-0}"
REQUEUE="${REQUEUE:-1}"

mkdir -p "$LOG_DIR"

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
  mapfile -t sync_files < <(
    {
      git diff --name-only --diff-filter=ACMRT
      git diff --name-only --cached --diff-filter=ACMRT
      git ls-files --others --exclude-standard
    } | sort -u | grep -Ev '^(artifacts/|logs/|\.pytest_cache/|.*__pycache__/|.*\.pyc$)' || true
  )
  if (( ${#sync_files[@]} == 0 )); then
    return
  fi

  mapfile -t rows < <(OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | tail -n +2)
  for row in "${rows[@]}"; do
    IFS=, read -r rank name internal_ip _nat_ip <<<"$row"
    if ! remote_sh "$internal_ip" "test -d $(q "$REMOTE_REPO_DIR")"; then
      echo "[$rank] skipping sync to $name ($internal_ip): missing $REMOTE_REPO_DIR" >&2
      continue
    fi
    echo "[$rank] syncing changed files to $name ($internal_ip)" >&2
    tar -cf - "${sync_files[@]}" | remote_sh "$internal_ip" "cd $(q "$REMOTE_REPO_DIR") && tar -xf -"
  done
}

selected_rows() {
  mapfile -t rows < <(OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | tail -n +2)
  if [[ -z "$SLURM_NODELIST" ]]; then
    printf '%s\n' "${rows[@]}"
    return
  fi

  mapfile -t selected_names < <(scontrol show hostnames "$SLURM_NODELIST")
  declare -A selected=()
  local name
  for name in "${selected_names[@]}"; do
    selected["$name"]=1
  done

  local row rank internal_ip nat_ip
  for row in "${rows[@]}"; do
    IFS=, read -r rank name internal_ip nat_ip <<<"$row"
    if [[ -n "${selected[$name]:-}" ]]; then
      printf '%s\n' "$row"
      unset "selected[$name]"
    fi
  done

  # Fallback for Slurm nodes that are usable but not returned by the GCP
  # discovery filter yet.
  local idx=0
  for name in "${!selected[@]}"; do
    internal_ip="$(scontrol show node "$name" | awk '{for (i=1; i<=NF; i++) if ($i ~ /^NodeAddr=/) {sub(/^NodeAddr=/, "", $i); print $i; exit}}')"
    printf '%s,%s,%s,\n' "$idx" "$name" "${internal_ip:-$name}"
    idx=$((idx + 1))
  done
}

preflight_runtime_paths() {
  local tp="${TP:-8}"
  local pp="${PP:-1}"
  local cp="${CP:-4}"
  local ep="${EP:-8}"
  local save_ckpt="${SAVE_CKPT:-$HOME/checkpoints/sft_deepseek_v32_reap_spinquant_actkv_nvfp4_tp${tp}_pp${pp}_cp${cp}_ep${ep}}"
  local wandb_exp_name="${WANDB_EXP_NAME:-corsaire-1-research-preview}"
  local zcc_run_name="${ZCC_RUN_NAME:-$wandb_exp_name}"
  local zcc_flash_device="${ZCC_FLASH_DEVICE:-/dev/shm/megatron_zcc/$zcc_run_name}"
  local zcc_durable_dir="${ZCC_DURABLE_DIR:-$save_ckpt/zcc/$zcc_run_name}"
  local logs_dir="$LOG_DIR"
  local torchrun_logs_dir="$HOME/logs/torchrun"
  local clean_zcc_durable="${PREFLIGHT_CLEAN_ZCC_DURABLE:-1}"

  mapfile -t rows < <(selected_rows)
  for row in "${rows[@]}"; do
    [[ -n "$row" ]] || continue
    IFS=, read -r rank name internal_ip _nat_ip <<<"$row"
    echo "[$rank] preflighting runtime paths on $name ($internal_ip)" >&2
    remote_sh "$internal_ip" "
      set -euo pipefail
      sudo install -d -m 1777 /dev/shm/megatron_zcc
      sudo rm -rf -- $(q "$zcc_flash_device")
      sudo install -d -m 1777 -- $(q "$zcc_flash_device")
      if [[ $(q "$clean_zcc_durable") == 1 ]]; then
        sudo rm -rf -- $(q "$zcc_durable_dir")
      fi
      sudo install -d -m 0777 -- $(q "$save_ckpt") $(q "$zcc_durable_dir") $(q "$logs_dir") $(q "$torchrun_logs_dir")
      sudo chmod 0777 -- $(q "$save_ckpt") $(q "$zcc_durable_dir") $(q "$logs_dir") $(q "$torchrun_logs_dir")
      test -w /dev/shm/megatron_zcc
      test -w $(q "$zcc_flash_device")
      test -w $(q "$save_ckpt")
      test -w $(q "$zcc_durable_dir")
      test -w $(q "$logs_dir")
      test -w $(q "$torchrun_logs_dir")
    "
  done
}

batch_file="$(mktemp "/tmp/${JOB_NAME}.XXXXXX.sbatch")"
cleanup() {
  rm -f "$batch_file"
}
trap cleanup EXIT

cat > "$batch_file" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
scontrol show hostnames "$SLURM_JOB_NODELIST"

if [[ -f "$HOME/.config/megatron/auth.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.config/megatron/auth.env"
  set +a
fi

export USE_SRUN=0
srun --label --kill-on-bad-exit=1 bash -lc '
  set -euo pipefail
  if [[ -f "$HOME/.config/megatron/auth.env" ]]; then
    set -a
    source "$HOME/.config/megatron/auth.env"
    set +a
  fi
  cd "$REPO_DIR"
  USE_SRUN=0 "$ENTRYPOINT"
'
EOF

sbatch_args=(
  --job-name "$JOB_NAME"
  --partition "$PARTITION_NAME"
  --nodes "$NNODES"
  --ntasks-per-node 1
  --gres "gpu:$GPUS_PER_NODE"
  --exclusive
  --output "$LOG_DIR/%x-%j.out"
  --error "$LOG_DIR/%x-%j.err"
  --export "ALL,REPO_DIR=$REMOTE_REPO_DIR,ENTRYPOINT=$ENTRYPOINT,USE_SRUN=0"
)
if [[ -n "$TIME_LIMIT" ]]; then
  sbatch_args+=(--time "$TIME_LIMIT")
fi
if [[ "$REQUEUE" == "0" ]]; then
  sbatch_args+=(--no-requeue)
fi
if [[ -n "$SLURM_NODELIST" ]]; then
  sbatch_args+=(--nodelist "$SLURM_NODELIST")
fi

cat >&2 <<EOF
Slurm submit:
  job=$JOB_NAME partition=$PARTITION_NAME nodes=$NNODES gpus_per_node=$GPUS_PER_NODE time=${TIME_LIMIT:-partition-default} requeue=$REQUEUE nodelist=${SLURM_NODELIST:-<any>}
  logs=$LOG_DIR/%x-%j.{out,err}
  repo=$REMOTE_REPO_DIR
  entrypoint=$ENTRYPOINT
EOF

if [[ "$SUBMIT_DRY_RUN" == "1" ]]; then
  printf 'sbatch'
  printf ' %q' "${sbatch_args[@]}"
  printf ' %q\n' "$batch_file"
  echo "--- batch script ---"
  cat "$batch_file"
  exit 0
fi

if [[ "$SYNC_REPO" == "1" ]]; then
  sync_changed_files
fi

preflight_runtime_paths

job_id="$(sbatch --parsable "${sbatch_args[@]}" "$batch_file")"
echo "Submitted batch job $job_id"
echo "Inspect after start: scontrol show job $job_id | rg 'JobState|BatchHost|StdOut|StdErr'"
