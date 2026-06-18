#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron_tp8_pp1_ep8}"
MODE="${MODE:-both}"
SSH_USER="${SSH_USER:-sjpat}"
if [[ -z "${SSH_KEY:-}" ]]; then
  if [[ -f "$HOME/google_compute_engine" ]]; then
    SSH_KEY="$HOME/google_compute_engine"
  else
    SSH_KEY="$HOME/.ssh/google_compute_engine"
  fi
fi

case "$MODE" in
  gather|fanout|both) ;;
  *)
    echo "Unsupported MODE=$MODE. Use gather, fanout, or both." >&2
    exit 2
    ;;
esac

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
  [[ "$name" == "$host_short" || "$ip" == "$(hostname -I | tr ' ' '\n' | grep -m1 -F "$ip" || true)" ]]
}

mkdir -p "$CHECKPOINT_DIR"

if [[ "$MODE" == "gather" || "$MODE" == "both" ]]; then
  for row in "${rows[@]}"; do
    IFS=, read -r rank name internal_ip _nat_ip <<<"$row"
    if is_local_node "$name" "$internal_ip"; then
      echo "[$rank] $name local checkpoint source: $CHECKPOINT_DIR" >&2
      continue
    fi
    echo "[$rank] gathering checkpoint shards from $name ($internal_ip)" >&2
    rsync -a --ignore-existing \
      -e "ssh ${ssh_opts[*]}" \
      "$SSH_USER@$internal_ip:$(q "$CHECKPOINT_DIR")/" \
      "$CHECKPOINT_DIR/"
  done
fi

if [[ "$MODE" == "fanout" || "$MODE" == "both" ]]; then
  for row in "${rows[@]}"; do
    IFS=, read -r rank name internal_ip _nat_ip <<<"$row"
    if is_local_node "$name" "$internal_ip"; then
      echo "[$rank] $name local checkpoint target: $CHECKPOINT_DIR" >&2
      continue
    fi
    echo "[$rank] propagating aggregate checkpoint to $name ($internal_ip)" >&2
    ssh "${ssh_opts[@]}" "$SSH_USER@$internal_ip" "mkdir -p $(q "$CHECKPOINT_DIR")"
    rsync -a \
      -e "ssh ${ssh_opts[*]}" \
      "$CHECKPOINT_DIR/" \
      "$SSH_USER@$internal_ip:$(q "$CHECKPOINT_DIR")/"
  done
fi

echo "Checkpoint propagation complete: $CHECKPOINT_DIR"
