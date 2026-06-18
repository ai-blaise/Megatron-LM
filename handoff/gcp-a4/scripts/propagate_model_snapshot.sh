#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/models/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4}"
SSH_USER="${SSH_USER:-sjpat}"
PARALLEL="${PARALLEL:-12}"
RSYNC_INFO="${RSYNC_INFO:-stats2}"
if [[ -z "${SSH_KEY:-}" ]]; then
  if [[ -f "$HOME/google_compute_engine" ]]; then
    SSH_KEY="$HOME/google_compute_engine"
  else
    SSH_KEY="$HOME/.ssh/google_compute_engine"
  fi
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "MODEL_DIR does not exist: $MODEL_DIR" >&2
  exit 2
fi

mapfile -t rows < <(OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | tail -n +2)
if (( ${#rows[@]} == 0 )); then
  echo "No nodes discovered" >&2
  exit 2
fi

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

sync_one() {
  local rank="$1"
  local name="$2"
  local ip="$3"

  if is_local_node "$name" "$ip"; then
    echo "[$rank] $name local model source: $MODEL_DIR" >&2
    return 0
  fi

  echo "[$rank] propagating model snapshot to $name ($ip)" >&2
  ssh "${ssh_opts[@]}" "$SSH_USER@$ip" "mkdir -p $(q "$MODEL_DIR")"
  rsync -a --partial --append-verify --info="$RSYNC_INFO" \
    -e "ssh ${ssh_opts[*]}" \
    "$MODEL_DIR/" \
    "$SSH_USER@$ip:$(q "$MODEL_DIR")/"
}

active=0
status=0
for row in "${rows[@]}"; do
  IFS=, read -r rank name internal_ip _nat_ip <<<"$row"
  sync_one "$rank" "$name" "$internal_ip" &
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

exit "$status"
