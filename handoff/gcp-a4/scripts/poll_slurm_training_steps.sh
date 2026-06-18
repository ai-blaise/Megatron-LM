#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}"
cd "$REPO_DIR"

JOB_ID="${JOB_ID:-${1:-}}"
if [[ -z "$JOB_ID" ]]; then
  JOB_ID="$(squeue -h -n gcp-a4-deepseek-sft -o '%i' | head -n 1 || true)"
fi
if [[ -z "$JOB_ID" ]]; then
  echo "Usage: JOB_ID=<slurm-job-id> $0" >&2
  exit 2
fi

TARGET_STEPS="${TARGET_STEPS:-5}"
POLL_SECONDS="${POLL_SECONDS:-120}"
SSH_USER="${SSH_USER:-sjpat}"
if [[ -z "${SSH_KEY:-}" ]]; then
  if [[ -f "$HOME/google_compute_engine" ]]; then
    SSH_KEY="$HOME/google_compute_engine"
  else
    SSH_KEY="$HOME/.ssh/google_compute_engine"
  fi
fi

ssh_base=(
  ssh
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o UserKnownHostsFile=/dev/null
  -o StrictHostKeyChecking=no
)

resolve_job_paths() {
  local info batch_host
  info="$(scontrol show job "$JOB_ID")"
  batch_host="$(awk -F= '/BatchHost=/{split($2,a," "); print a[1]; exit}' <<<"$info")"
  STDOUT_PATH="$(awk -F= '/StdOut=/{split($2,a," "); print a[1]; exit}' <<<"$info")"
  STDERR_PATH="$(awk -F= '/StdErr=/{split($2,a," "); print a[1]; exit}' <<<"$info")"
  JOB_STATE="$(awk -F= '/JobState=/{split($2,a," "); print a[1]; exit}' <<<"$info")"
  BATCH_IP="$batch_host"
  if [[ "$batch_host" != *.*.*.* ]]; then
    BATCH_IP="$(OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | awk -F, -v h="$batch_host" '$2 == h {print $3; exit}' || true)"
    BATCH_IP="${BATCH_IP:-$batch_host}"
  fi
}

max_iteration_from_text() {
  uv run --no-sync python - <<'PY'
import re
import sys

text = sys.stdin.read()
patterns = [
    r"\biteration\s+([0-9]+)\b",
    r"\biteration\s*[:=]\s*([0-9]+)\b",
    r"\biter\s+([0-9]+)\b",
    r"\biter\s*[:=]\s*([0-9]+)\b",
]
values = []
for pattern in patterns:
    values.extend(int(match.group(1)) for match in re.finditer(pattern, text, re.IGNORECASE))
print(max(values) if values else 0)
PY
}

while true; do
  resolve_job_paths
  stamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "[$stamp] job=$JOB_ID state=$JOB_STATE batch=$BATCH_IP target_steps=$TARGET_STEPS"

  recent="$("${ssh_base[@]}" "$SSH_USER@$BATCH_IP" \
    "grep -E '\\[run_stats\\]|iteration| consumed samples| elapsed time per iteration| lm loss| loss scale|TFLOPs|tokens/s|wandb:|Saving checkpoint|successfully saved|Traceback|Error|Exception|failed|fatal|out of memory|timeout' '$STDERR_PATH' '$STDOUT_PATH' 2>/dev/null | tail -n 120" || true)"
  if [[ -n "$recent" ]]; then
    printf '%s\n' "$recent" | tail -n 40
  else
    echo "No iteration/loss lines yet."
  fi

  max_iter="$(printf '%s\n' "$recent" | max_iteration_from_text)"
  echo "max_iteration_seen=$max_iter"
  if (( max_iter >= TARGET_STEPS )); then
    echo "Reached target: first $TARGET_STEPS iterations observed."
    exit 0
  fi
  if [[ "$JOB_STATE" != "RUNNING" && "$JOB_STATE" != "PENDING" ]]; then
    echo "Job left RUNNING/PENDING before $TARGET_STEPS iterations: $JOB_STATE" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done
