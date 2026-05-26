#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_USER="${SSH_USER:-sjpat}"
AUTH_PATH="${AUTH_PATH:-$HOME/.config/megatron/auth.env}"

if [[ -z "${WANDB_API_KEY:-}" || -z "${HF_TOKEN:-}" ]]; then
  echo "WANDB_API_KEY and HF_TOKEN must be set in the environment" >&2
  exit 2
fi

write_auth_file() {
  local path="$1"
  install -d -m 700 "$(dirname "$path")"
  umask 077
  {
    printf 'export WANDB_API_KEY=%q\n' "$WANDB_API_KEY"
    printf 'export HF_TOKEN=%q\n' "$HF_TOKEN"
    printf 'export HUGGING_FACE_HUB_TOKEN=%q\n' "$HF_TOKEN"
  } > "$path"
  chmod 600 "$path"
}

write_auth_file "$AUTH_PATH"

remote_payload="$(mktemp)"
trap 'rm -f "$remote_payload"' EXIT
{
  printf 'set -euo pipefail\n'
  printf 'AUTH_PATH=%q\n' "$AUTH_PATH"
  declare -f write_auth_file
  printf 'WANDB_API_KEY=%q\n' "$WANDB_API_KEY"
  printf 'HF_TOKEN=%q\n' "$HF_TOKEN"
  printf 'write_auth_file "$AUTH_PATH"\n'
} > "$remote_payload"

OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | tail -n +2 | while IFS=, read -r rank name internal_ip _nat_ip; do
  echo "[$rank] writing auth env on $name ($internal_ip)" >&2
  ssh \
    -i "${SSH_KEY:-$HOME/google_compute_engine}" \
    -o IdentitiesOnly=yes \
    -o BatchMode=yes \
    -o UserKnownHostsFile=/dev/null \
    -o StrictHostKeyChecking=no \
    "$SSH_USER@$internal_ip" \
    'bash -s' < "$remote_payload" >/dev/null
done

echo "Auth env written to $AUTH_PATH on discovered nodes." >&2
