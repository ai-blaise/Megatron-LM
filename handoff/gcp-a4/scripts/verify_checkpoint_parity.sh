#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron_tp8_pp1_ep8}"
HASH_WORKERS="${HASH_WORKERS:-96}"
HASH_MODE="${HASH_MODE:-chunk}"
CHUNK_BYTES="${CHUNK_BYTES:-268435456}"
SESSION="${SESSION:-ckpt-hash-verify}"
OUT_DIR="${OUT_DIR:-$HOME/logs/ckpt-hash-verify-$(basename "$CHECKPOINT_DIR")}"
PARALLEL="${PARALLEL:-15}"
SSH_USER="${SSH_USER:-sjpat}"
SSH_KEY="${SSH_KEY:-$HOME/google_compute_engine}"
REMOTE_SCRIPT="$HOME/bin/remote_checkpoint_manifest_hash.sh"

ssh_opts=(
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o UserKnownHostsFile=/dev/null
  -o StrictHostKeyChecking=no
  -o LogLevel=ERROR
)

usage() {
  cat >&2 <<USAGE
Usage: $0 <launch|status|compare|structural>

Environment:
  CHECKPOINT_DIR   checkpoint directory to verify
  HASH_WORKERS     per-node xargs worker count, default 96
  HASH_MODE        chunk or file, default chunk
  CHUNK_BYTES      byte range size for chunk mode, default 268435456
  PARALLEL         controller-side SSH fanout, default 15
  OUT_DIR          remote log/output directory
USAGE
}

q() {
  printf "%q" "$1"
}

load_rows() {
  OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | tail -n +2
}

run_parallel() {
  local status=0
  while IFS= read -r row; do
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do
      if ! wait -n; then
        status=1
      fi
    done
    "$@" "$row" &
  done < <(load_rows)

  while (( $(jobs -rp | wc -l) > 0 )); do
    if ! wait -n; then
      status=1
    fi
  done
  return "$status"
}

launch_one() {
  local row="$1"
  local rank name internal_ip nat_ip
  IFS=, read -r rank name internal_ip nat_ip <<<"$row"

  echo "[$rank] launching hash on $name ($internal_ip)" >&2
  ssh "${ssh_opts[@]}" "$SSH_USER@$internal_ip" "mkdir -p $HOME/bin $(q "$OUT_DIR")" </dev/null
  rsync -a -e "ssh ${ssh_opts[*]}" \
    "$SCRIPT_DIR/remote_checkpoint_manifest_hash.sh" \
    "$SSH_USER@$internal_ip:$(q "$REMOTE_SCRIPT")"
  ssh "${ssh_opts[@]}" "$SSH_USER@$internal_ip" \
    "chmod +x $(q "$REMOTE_SCRIPT") && tmux kill-session -t $(q "$SESSION") 2>/dev/null || true" </dev/null
  ssh "${ssh_opts[@]}" "$SSH_USER@$internal_ip" \
    "tmux new-session -d -s $(q "$SESSION") 'CHECKPOINT_DIR=$(q "$CHECKPOINT_DIR") OUT_DIR=$(q "$OUT_DIR") HASH_WORKERS=$(q "$HASH_WORKERS") HASH_MODE=$(q "$HASH_MODE") CHUNK_BYTES=$(q "$CHUNK_BYTES") $(q "$REMOTE_SCRIPT") > $(q "$OUT_DIR")/run.log 2>&1'" </dev/null
}

status_one() {
  local row="$1"
  local rank name internal_ip nat_ip
  IFS=, read -r rank name internal_ip nat_ip <<<"$row"
  local output

  if output="$(ssh "${ssh_opts[@]}" "$SSH_USER@$internal_ip" \
    "if [[ -f $(q "$OUT_DIR")/status ]]; then tr '\n' ' ' < $(q "$OUT_DIR")/status; echo; else tmux has-session -t $(q "$SESSION") 2>/dev/null && echo status=running_no_status || echo status=no_status; fi" \
    </dev/null)"; then
    printf "[%s] %s %s\n" "$rank" "$name" "$output"
  else
    printf "[%s] %s status=ssh_failed\n" "$rank" "$name"
  fi
}

compare_one() {
  local row="$1"
  local rank name internal_ip nat_ip
  IFS=, read -r rank name internal_ip nat_ip <<<"$row"

  ssh "${ssh_opts[@]}" "$SSH_USER@$internal_ip" \
    "if [[ -f $(q "$OUT_DIR")/status ]]; then awk -v rank=$(q "$rank") -v name=$(q "$name") '
      BEGIN { status=\"\"; mode=\"\"; files=\"\"; entries=\"\"; digest=\"\"; }
      /^status=/ { status=substr(\$0,8); }
      /^mode=/ { mode=substr(\$0,6); }
      /^files=/ { files=substr(\$0,7); }
      /^manifest_entries=/ { entries=substr(\$0,18); }
      /^manifest_sha256=/ { digest=substr(\$0,17); }
      END { printf \"%s,%s,%s,%s,%s,%s,%s\\n\", rank, name, status, mode, files, entries, digest; }
    ' $(q "$OUT_DIR")/status; else echo $(q "$rank,$name,missing_status,,"); fi" </dev/null \
    || echo "$rank,$name,ssh_failed,,,,"
}

structural_one() {
  local row="$1"
  local rank name internal_ip nat_ip
  IFS=, read -r rank name internal_ip nat_ip <<<"$row"
  local output

  if output="$(ssh "${ssh_opts[@]}" "$SSH_USER@$internal_ip" "d=$(q "$CHECKPOINT_DIR"); \
    if [[ ! -d \"\$d\" ]]; then echo missing; exit 0; fi; \
    tracker=missing; [[ -f \"\$d/latest_checkpointed_iteration.txt\" ]] && tracker=\$(tr -d '\n' < \"\$d/latest_checkpointed_iteration.txt\"); \
    meta=no; [[ -f \"\$d/iter_0000000/.metadata\" ]] && meta=yes; \
    files=\$(find \"\$d\" -type f | wc -l); \
    tmp=\$(find \"\$d\" \( -name '*.tmp' -o -name '*.part' \) | wc -l); \
    size=\$(du -sh \"\$d\" | awk '{print \$1}'); \
    echo size=\$size files=\$files tracker=\$tracker metadata=\$meta temp_files=\$tmp" \
    </dev/null)"; then
    printf "[%s] %s %s\n" "$rank" "$name" "$output"
  else
    printf "[%s] %s ssh_failed\n" "$rank" "$name"
  fi
}

cmd="${1:-}"
case "$cmd" in
  launch)
    run_parallel launch_one
    ;;
  status)
    run_parallel status_one
    ;;
  compare)
    echo "rank,name,status,mode,files,manifest_entries,manifest_sha256"
    load_rows | while IFS= read -r row; do
      compare_one "$row"
    done | sort -t, -k1,1n
    ;;
  structural)
    run_parallel structural_one
    ;;
  *)
    usage
    exit 2
    ;;
esac
