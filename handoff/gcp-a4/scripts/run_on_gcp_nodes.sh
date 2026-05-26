#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${*:-}"
SSH_USER="${SSH_USER:-sjpat}"
if [[ -z "${SSH_KEY:-}" ]]; then
  if [[ -f "$HOME/google_compute_engine" ]]; then
    SSH_KEY="$HOME/google_compute_engine"
  else
    SSH_KEY="$HOME/.ssh/google_compute_engine"
  fi
fi

if [[ -z "$COMMAND" ]]; then
  echo "Usage: $0 <command to run on each discovered node>" >&2
  exit 2
fi

mapfile -t rows < <(OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | tail -n +2)
if (( ${#rows[@]} == 0 )); then
  echo "No nodes discovered" >&2
  exit 2
fi

status=0
for row in "${rows[@]}"; do
  IFS=, read -r rank name internal_ip _nat_ip <<<"$row"
  echo "[$rank] $name ($internal_ip): $COMMAND" >&2
  if ! ssh \
    -i "$SSH_KEY" \
    -o IdentitiesOnly=yes \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    "$SSH_USER@$internal_ip" \
    "$COMMAND"; then
    echo "[$rank] $name failed" >&2
    status=1
  fi
done

exit "$status"
