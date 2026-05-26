#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron_tp8_pp1_ep8}"
TARGETS_FILE="${TARGETS_FILE:-}"
PARALLEL="${PARALLEL:-12}"
SSH_USER="${SSH_USER:-sjpat}"
SSH_KEY="${SSH_KEY:-$HOME/google_compute_engine}"
LOG_DIR="${LOG_DIR:-$HOME/logs/ckpt-fanout-$(basename "$CHECKPOINT_DIR")}"
DELETE_EXTRA="${DELETE_EXTRA:-0}"

if [[ ! -d "$CHECKPOINT_DIR/iter_0000000" ]]; then
  echo "Missing checkpoint directory: $CHECKPOINT_DIR/iter_0000000" >&2
  exit 2
fi
if find "$CHECKPOINT_DIR" -name '*.tmp' -print -quit | grep -q .; then
  echo "Checkpoint still has temp files; refusing to fan out: $CHECKPOINT_DIR" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT_DIR/iter_0000000/.metadata" || ! -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]]; then
  echo "Checkpoint does not look finalized: $CHECKPOINT_DIR" >&2
  exit 2
fi

ssh_opts=(
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o UserKnownHostsFile=/dev/null
  -o StrictHostKeyChecking=no
)

rsync_opts=(-a --partial --append-verify --info=progress2,stats2)
if [[ "$DELETE_EXTRA" == "1" ]]; then
  rsync_opts+=(--delete)
fi

q() {
  printf "%q" "$1"
}

load_rows() {
  if [[ -n "$TARGETS_FILE" ]]; then
    tail -n +2 "$TARGETS_FILE"
  else
    OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | tail -n +2
  fi
}

is_local_node() {
  local name="$1"
  local ip="$2"
  local host_short
  host_short="$(hostname -s)"
  [[ "$name" == "$host_short" ]] && return 0
  hostname -I | tr ' ' '\n' | grep -qxF "$ip"
}

mkdir -p "$LOG_DIR"
mapfile -t rows < <(load_rows)
if (( ${#rows[@]} == 0 )); then
  echo "No fanout targets found" >&2
  exit 2
fi

status=0
for row in "${rows[@]}"; do
  IFS=, read -r rank name internal_ip _nat_ip <<<"$row"
  if is_local_node "$name" "$internal_ip"; then
    echo "[$rank] $name local source; skipping" | tee "$LOG_DIR/${name}.log"
    continue
  fi

  while (( $(jobs -rp | wc -l) >= PARALLEL )); do
    if ! wait -n; then
      status=1
    fi
  done

  (
    set -euo pipefail
    log="$LOG_DIR/${name}.log"
    {
      date -Is
      echo "target=$name ip=$internal_ip"
      echo "source=$CHECKPOINT_DIR"
      ssh "${ssh_opts[@]}" "$SSH_USER@$internal_ip" "mkdir -p $(q "$CHECKPOINT_DIR")"
      RSYNC_RSH="ssh ${ssh_opts[*]}" rsync "${rsync_opts[@]}" \
        "$CHECKPOINT_DIR/" \
        "$SSH_USER@$internal_ip:$(q "$CHECKPOINT_DIR")/"
      date -Is
    } >"$log" 2>&1
  ) &
done

while (( $(jobs -rp | wc -l) > 0 )); do
  if ! wait -n; then
    status=1
  fi
done

echo "$(date -Is) status=$status" > "$LOG_DIR/complete.status"
exit "$status"
