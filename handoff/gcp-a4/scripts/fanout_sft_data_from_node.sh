#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${DATA_PATH:-$HOME/data/sft/blaise-sft-training-mix/blaise-sft-training-mix-full.jsonl}"
OFFSETS_PATH="${OFFSETS_PATH:-$DATA_PATH.offsets.npy}"
SHUFFLE_INDEX_PATH="${SHUFFLE_INDEX_PATH:-$DATA_PATH.shuffle.seed${MEGATRON_SFT_SHUFFLE_SEED:-1234}.npy}"
TARGETS_FILE="${TARGETS_FILE:-}"
SSH_USER="${SSH_USER:-sjpat}"
SSH_KEY="${SSH_KEY:-$HOME/google_compute_engine}"
PARALLEL="${PARALLEL:-12}"
RSYNC_INFO="${RSYNC_INFO:-progress2,stats2}"
LOG_DIR="${LOG_DIR:-$HOME/logs/sft-data-fanout}"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "DATA_PATH does not exist: $DATA_PATH" >&2
  exit 2
fi
if [[ ! -f "$OFFSETS_PATH" ]]; then
  echo "OFFSETS_PATH does not exist: $OFFSETS_PATH" >&2
  exit 2
fi
if [[ -n "$SHUFFLE_INDEX_PATH" && ! -f "$SHUFFLE_INDEX_PATH" ]]; then
  echo "SHUFFLE_INDEX_PATH does not exist: $SHUFFLE_INDEX_PATH" >&2
  exit 2
fi
if compgen -G "$DATA_PATH.*.tmp" >/dev/null || compgen -G "$DATA_PATH.tmp" >/dev/null; then
  echo "Refusing to fan out while tmp data files exist for $DATA_PATH" >&2
  exit 2
fi
if [[ ! -f "$SSH_KEY" ]]; then
  echo "SSH key does not exist: $SSH_KEY" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

ssh_opts=(
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o UserKnownHostsFile=/dev/null
  -o StrictHostKeyChecking=no
)

q() {
  printf '%q' "$1"
}

is_local_node() {
  local name="$1"
  local ip="$2"
  local host_short
  host_short="$(hostname -s)"
  [[ "$name" == "$host_short" ]] && return 0
  hostname -I | tr ' ' '\n' | grep -qxF "$ip"
}

load_rows() {
  if [[ -n "$TARGETS_FILE" ]]; then
    awk -F, 'NF >= 3 && $1 ~ /^[0-9]+$/ {print $1 "," $2 "," $3}' "$TARGETS_FILE"
  else
    OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | tail -n +2 | awk -F, '{print $1 "," $2 "," $3}'
  fi
}

sync_one() {
  local rank="$1"
  local name="$2"
  local ip="$3"
  local log="$LOG_DIR/$name.log"
  local remote_dir
  remote_dir="$(dirname "$DATA_PATH")"

  {
    if is_local_node "$name" "$ip"; then
      echo "[$rank] $name local SFT data source; skipping"
      return 0
    fi

    echo "[$rank] syncing SFT data to $name ($ip)"
    ssh "${ssh_opts[@]}" "$SSH_USER@$ip" "mkdir -p $(q "$remote_dir")"
    files=("$DATA_PATH" "$OFFSETS_PATH")
    if [[ -n "$SHUFFLE_INDEX_PATH" ]]; then
      files+=("$SHUFFLE_INDEX_PATH")
    fi
    rsync -a --partial --append-verify --info="$RSYNC_INFO" \
      -e "ssh ${ssh_opts[*]}" \
      "${files[@]}" \
      "$SSH_USER@$ip:$(q "$remote_dir")/"
  } >"$log" 2>&1
}

mapfile -t rows < <(load_rows)
if (( ${#rows[@]} == 0 )); then
  echo "No target rows found" >&2
  exit 2
fi

active=0
status=0
for row in "${rows[@]}"; do
  IFS=, read -r rank name ip <<<"$row"
  sync_one "$rank" "$name" "$ip" &
  active=$((active + 1))
  if (( active >= PARALLEL )); then
    if ! wait -n; then
      status=1
    fi
    active=$((active - 1))
  fi
done

while (( active > 0 )); do
  if ! wait -n; then
    status=1
  fi
  active=$((active - 1))
done

printf 'status=%s\ncompleted_at=%s\n' "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$LOG_DIR/complete.status"
exit "$status"
