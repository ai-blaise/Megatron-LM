#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}"
cd "$REPO_DIR"

command -v tmux >/dev/null || {
  echo "tmux is required on the controller" >&2
  exit 127
}

JOB_ID="${JOB_ID:-${1:-}}"
if [[ -z "$JOB_ID" ]]; then
  JOB_ID="$(squeue -h -n gcp-a4-deepseek-sft -o '%i' | head -n 1 || true)"
fi
if [[ -z "$JOB_ID" ]]; then
  echo "Usage: JOB_ID=<slurm-job-id> $0" >&2
  exit 2
fi

SSH_USER="${SSH_USER:-sjpat}"
if [[ -z "${SSH_KEY:-}" ]]; then
  if [[ -f "$HOME/google_compute_engine" ]]; then
    SSH_KEY="$HOME/google_compute_engine"
  else
    SSH_KEY="$HOME/.ssh/google_compute_engine"
  fi
fi

SESSION="${SESSION:-gcp-a4-train-${JOB_ID}}"
MONITOR_DIR="${MONITOR_DIR:-$HOME/logs/slurm-monitor-${JOB_ID}}"
SAVE_CKPT="${SAVE_CKPT:-}"
POLL_SECONDS="${POLL_SECONDS:-20}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
TARGET_STEPS="${TARGET_STEPS:-5}"

mkdir -p "$MONITOR_DIR"

job_info="$(scontrol show job "$JOB_ID")"
job_name="$(tr ' ' '\n' <<<"$job_info" | awk -F= '$1 == "JobName" {print $2; exit}' || true)"
job_name="${job_name:-unknown-job}"
batch_host="$(awk -F= '/BatchHost=/{split($2,a," "); print a[1]; exit}' <<<"$job_info")"
stdout_path="$(awk -F= '/StdOut=/{split($2,a," "); print a[1]; exit}' <<<"$job_info")"
stderr_path="$(awk -F= '/StdErr=/{split($2,a," "); print a[1]; exit}' <<<"$job_info")"
if [[ -z "$batch_host" || -z "$stdout_path" || -z "$stderr_path" ]]; then
  echo "Could not resolve BatchHost/StdOut/StdErr for job $JOB_ID" >&2
  exit 2
fi

batch_ip="$batch_host"
if [[ "$batch_host" != *.*.*.* ]]; then
  batch_ip="$(OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | awk -F, -v h="$batch_host" '$2 == h {print $3; exit}' || true)"
  batch_ip="${batch_ip:-$batch_host}"
fi

allocated_nodelist="$(tr ' ' '\n' <<<"$job_info" | awk -F= '$1 == "NodeList" {print $2; exit}' || true)"
mapfile -t allocated_hosts < <(
  if [[ -n "$allocated_nodelist" && "$allocated_nodelist" != "(null)" && "$allocated_nodelist" != "N/A" ]]; then
    scontrol show hostnames "$allocated_nodelist" 2>/dev/null || true
  fi
)

q() {
  printf '%q' "$1"
}

mapfile -t rows < <(OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | tail -n +2)
{
  echo "rank,name,ip"
  for row in "${rows[@]}"; do
    IFS=, read -r rank name internal_ip _nat_ip <<<"$row"
    if (( ${#allocated_hosts[@]} > 0 )); then
      keep=0
      for host in "${allocated_hosts[@]}"; do
        if [[ "$name" == "$host" ]]; then
          keep=1
          break
        fi
      done
      (( keep == 1 )) || continue
    fi
    printf "%s,%s,%s\n" "$rank" "$name" "$internal_ip"
  done
} > "$MONITOR_DIR/nodes.csv"

last_node_rank="$(tail -n +2 "$MONITOR_DIR/nodes.csv" | tail -n 1 | awk -F, '{print $1}')"
last_node_name="$(tail -n +2 "$MONITOR_DIR/nodes.csv" | tail -n 1 | awk -F, '{print $2}')"
last_node_ip="$(tail -n +2 "$MONITOR_DIR/nodes.csv" | tail -n 1 | awk -F, '{print $3}')"
last_local_rank="${LAST_LOCAL_RANK:-7}"

cat > "$MONITOR_DIR/common.sh" <<EOF
#!/usr/bin/env bash
set +e
SCRIPT_DIR=$(q "$SCRIPT_DIR")
REPO_DIR=$(q "$REPO_DIR")
JOB_ID=$(q "$JOB_ID")
JOB_NAME=$(q "$job_name")
SSH_USER=$(q "$SSH_USER")
SSH_KEY=$(q "$SSH_KEY")
BATCH_HOST=$(q "$batch_host")
BATCH_IP=$(q "$batch_ip")
STDOUT_PATH=$(q "$stdout_path")
STDERR_PATH=$(q "$stderr_path")
MONITOR_DIR=$(q "$MONITOR_DIR")
NODES_CSV=$(q "$MONITOR_DIR/nodes.csv")
TORCHRUN_JOB_DIR=$(q "$HOME/logs/torchrun/${job_name}-${JOB_ID}")
LAST_NODE_RANK=$(q "$last_node_rank")
LAST_NODE_NAME=$(q "$last_node_name")
LAST_NODE_IP=$(q "$last_node_ip")
LAST_LOCAL_RANK=$(q "$last_local_rank")
SAVE_CKPT=$(q "$SAVE_CKPT")
POLL_SECONDS=$(q "$POLL_SECONDS")
GPU_POLL_SECONDS=$(q "$GPU_POLL_SECONDS")
TARGET_STEPS=$(q "$TARGET_STEPS")
BENIGN_LOG_RE='FutureWarning|pynvml package is deprecated|Setting OMP_NUM_THREADS|Permanently added|Apex is not installed|Falling back to Torch Norm|Empty sink for dataset|Failed to create scuba file|Could not find table name|TORCHELASTIC_USE_AGENT_STORE is enabled so ignoring|destroy_process_group\\(\\) was not called'
PROGRESS_RE='\\[run_stats\\]|\\[pipeline-progress\\]|\\[moe-stage\\]|\\[hisa_runtime\\]|\\[dsa_runtime\\]|\\[path_audit\\]|iteration| consumed samples| elapsed time per iteration| lm loss| loss scale|learning rate|grad norm|TFLOPs|tokens/s|samples/sec|samples per second|Saving checkpoint|successfully saved|wandb:|W&B|Starting HF checkpoint upload'
CRITICAL_RE='Traceback|RuntimeError|AssertionError|DistBackendError|DistStoreError|ChildFailedError|CUDA error|out of memory|fatal|segmentation fault|core dumped|NCCL.*(ERROR|error|unhandled|internal|system|timeout|connect|connection|abort)|ibv_.*(failed|Cannot|error)|failed \\(exitcode|exited with exit code|terminated|Exception'
ssh_base=(
  ssh
  -i "\$SSH_KEY"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o UserKnownHostsFile=/dev/null
  -o StrictHostKeyChecking=no
  -o ConnectTimeout=6
)
remote_batch() {
  "\${ssh_base[@]}" "\$SSH_USER@\$BATCH_IP" "\$@"
}
remote_node() {
  local ip="\$1"
  shift
  "\${ssh_base[@]}" "\$SSH_USER@\$ip" "\$@"
}
remote_rank0_logs_cat() {
  remote_batch "cat '\$STDERR_PATH' '\$STDOUT_PATH' 2>/dev/null; find '\$TORCHRUN_JOB_DIR/node0' -type f \\( -name stdout.log -o -name stderr.log \\) -path '*/attempt_*/0/*' 2>/dev/null | sort | xargs -r cat 2>/dev/null"
}
remote_last_rank_logs_cat() {
  remote_node "\$LAST_NODE_IP" "find '\$TORCHRUN_JOB_DIR/node'\$LAST_NODE_RANK -type f \\( -name stdout.log -o -name stderr.log \\) -path '*/attempt_*/'\$LAST_LOCAL_RANK'/*' 2>/dev/null | sort | xargs -r cat 2>/dev/null"
}
remote_wrapper_and_rank0_tail() {
  remote_batch "rank0_files=\\\$(find '\$TORCHRUN_JOB_DIR/node0' -type f \\( -name stdout.log -o -name stderr.log \\) -path '*/attempt_*/0/*' 2>/dev/null | sort); tail -n 400 -F '\$STDERR_PATH' '\$STDOUT_PATH' \\\$rank0_files 2>/dev/null"
}
header() {
  clear
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  echo "job=\$JOB_ID batch=\$BATCH_HOST(\$BATCH_IP)"
  echo
}
EOF

cat > "$MONITOR_DIR/status.sh" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
while true; do
  header
  echo "== Slurm =="
  squeue -j "$JOB_ID" -o '%i %t %M %D %R %j'
  echo
  scontrol show job "$JOB_ID" | grep -E 'JobId=|JobState=|Reason=|ExitCode=|RunTime=|TimeLimit=|BatchHost=|NumNodes=|NumCPUs=|StdOut=|StdErr='
  echo
  echo "== Nodes =="
  printf "selected=%s\n" "$(tail -n +2 "$NODES_CSV" | wc -l)"
  scontrol show hostnames "$(squeue -h -j "$JOB_ID" -o '%N' 2>/dev/null | head -n 1)" 2>/dev/null | sed -n '1,20p'
  echo
  echo "== Launch Shape =="
  remote_batch "grep -E 'GCP A4 launch shape|TP=|MBS=|streambp=|profile_run=|tensorboard_logs=|pipeline_progress=|dispatcher=|higgs=|path_audit|memlock_soft=|wandb_entity=|hf_upload=|zcc=|load_ckpt=|save_ckpt=' '$STDERR_PATH' 2>/dev/null | tail -n 32" || true
  sleep "$POLL_SECONDS"
done
EOF

cat > "$MONITOR_DIR/progress.sh" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
while true; do
  header
  echo "== Training Progress =="
  recent="$({
    remote_rank0_logs_cat
    remote_last_rank_logs_cat
  } | grep -E "$PROGRESS_RE" | grep -Ev "$BENIGN_LOG_RE" | tail -n 240 || true)"
  max_iter="$(printf '%s\n' "$recent" | python3 -c 'import re, sys; text=sys.stdin.read(); vals=[int(m.group(1)) for m in re.finditer(r"\biteration\s+([0-9]+)\b", text, re.I)]; vals += [int(m.group(1)) for m in re.finditer(r"\biter\s*[=: ]\s*([0-9]+)\b", text, re.I)]; print(max(vals) if vals else 0)')"
  echo "max_iteration_seen=$max_iter target_initial_steps=$TARGET_STEPS"
  echo "normal Megatron loss log source: node_rank=$LAST_NODE_RANK host=$LAST_NODE_NAME local_rank=$LAST_LOCAL_RANK"
  echo
  latest_iter="$(printf '%s\n' "$recent" | grep -Ei '\\[run_stats\\]|\\[pipeline-progress\\]|\\[moe-stage\\]|\\[hisa_runtime\\]|\\[dsa_runtime\\]|\\[path_audit\\]|iteration|lm loss|TFLOPs|tokens/s|samples' | tail -n 1)"
  if [[ -n "$latest_iter" ]]; then
    echo "latest:"
    printf '%s\n' "$latest_iter"
  else
    echo "No iteration line yet. Showing startup/checkpoint/W&B signals."
  fi
  echo
  printf '%s\n' "$recent" | tail -n 80
  sleep "$POLL_SECONDS"
done
EOF

cat > "$MONITOR_DIR/gpu_summary.sh" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
while true; do
  header
  echo "== GPU Fleet Summary =="
  tmp="$(mktemp)"
  while IFS=, read -r rank name ip; do
    [[ "$rank" == "rank" ]] && continue
    (
      out="$(remote_node "$ip" "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null" 2>/dev/null)"
      if [[ -z "$out" ]]; then
        printf "%03d %-28s ssh/gpu query failed\n" "$rank" "$name"
        exit 0
      fi
      printf '%s\n' "$out" | awk -F, -v r="$rank" -v n="$name" '
        {
          for (i=1; i<=NF; i++) gsub(/^ +| +$/, "", $i)
          util=$1+0; mem=$2+0; total=$3+0; temp=$4+0; power=$5+0
          gpus++; util_sum+=util; mem_sum+=mem; total_sum+=total; power_sum+=power
          if (temp > max_temp) max_temp=temp
          if (util >= 50) busy++
        }
        END {
          if (gpus == 0) {
            printf "%03d %-28s no GPUs reported\n", r, n
          } else {
            printf "%03d %-28s gpus=%d busy=%d avg_util=%3.0f%% mem=%5.0f/%5.0fGiB max_temp=%2.0fC power=%5.0fW\n",
              r, n, gpus, busy, util_sum/gpus, mem_sum/1024, total_sum/1024, max_temp, power_sum
          }
        }'
    ) >> "$tmp" &
  done < "$NODES_CSV"
  wait
  sort "$tmp"
  rm -f "$tmp"
  sleep "$GPU_POLL_SECONDS"
done
EOF

cat > "$MONITOR_DIR/alerts.sh" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
while true; do
  header
  echo "== Critical Alerts =="
  echo "Filtered: benign warnings, OMP notices, known torch warnings, structured-log sink noise."
  echo
  alerts="$(remote_rank0_logs_cat | grep -Ein "$CRITICAL_RE" | grep -Evi "$BENIGN_LOG_RE" | tail -n 120 || true)"
  if [[ -n "$alerts" ]]; then
    printf '%s\n' "$alerts"
  else
    echo "No critical alerts found."
  fi
  sleep "$POLL_SECONDS"
done
EOF

cat > "$MONITOR_DIR/rank0.sh" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
echo "Rank/log signal tail for job $JOB_ID. Ctrl-c closes this tmux pane."
remote_wrapper_and_rank0_tail \
  | grep --line-buffered -E '\\[run_stats\\]|\\[pipeline-progress\\]|(^|[[:space:]])0:|rank=0|iteration|lm loss|TFLOPs|tokens/s|Saving checkpoint|successfully saved|wandb:' \
  | grep --line-buffered -Ev "$BENIGN_LOG_RE" || true
EOF

cat > "$MONITOR_DIR/gpu_detail.sh" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
while true; do
  header
  echo "== Per-GPU Detail =="
  while IFS=, read -r rank name ip; do
    [[ "$rank" == "rank" ]] && continue
    printf "\n[%03d] %s\n" "$rank" "$name"
    remote_node "$ip" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null" 2>/dev/null \
      | awk -F, '{for(i=1;i<=NF;i++) gsub(/^ +| +$/, "", $i); printf "  gpu%s util=%3s%% mem=%7s/%7sMiB temp=%sC power=%sW\n", $1,$2,$3,$4,$5,$6}' \
      || echo "  ssh/gpu query failed"
  done < "$NODES_CSV"
  sleep "$GPU_POLL_SECONDS"
done
EOF

cat > "$MONITOR_DIR/gpu_heatmap.sh" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
while true; do
  header
  echo "== GPU Memory/Util Heatmap =="
  echo "glyph scale low->high: .:-=+*#%@"
  tmp="$(mktemp)"
  while IFS=, read -r rank name ip; do
    [[ "$rank" == "rank" ]] && continue
    (
      out="$(remote_node "$ip" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null" 2>/dev/null)"
      if [[ -z "$out" ]]; then
        printf "%03d %-28s ssh/gpu query failed\n" "$rank" "$name"
        exit 0
      fi
      printf '%s\n' "$out" | awk -F, -v r="$rank" -v n="$name" '
        function glyph(p, idx, chars) {
          chars=" .:-=+*#%@"
          idx=int(p / 12.5) + 1
          if (idx < 1) idx=1
          if (idx > length(chars)) idx=length(chars)
          return substr(chars, idx, 1)
        }
        {
          for (i=1; i<=NF; i++) gsub(/^ +| +$/, "", $i)
          util=$2+0; mem=$3+0; total=$4+0; temp=$5+0
          pct=(total > 0) ? (100.0 * mem / total) : 0
          mem_bar=mem_bar glyph(pct)
          util_bar=util_bar glyph(util)
          if (pct > max_mem_pct) max_mem_pct=pct
          if (util > max_util) max_util=util
          if (temp > max_temp) max_temp=temp
          mem_sum+=mem; total_sum+=total; gpus++
        }
        END {
          if (gpus == 0) {
            printf "%03d %-28s no GPUs reported\n", r, n
          } else {
            printf "%03d %-28s mem[%s] util[%s] max_mem=%5.1f%% max_util=%3.0f%% mem=%5.0f/%5.0fGiB temp=%2.0fC\n",
              r, n, mem_bar, util_bar, max_mem_pct, max_util, mem_sum/1024, total_sum/1024, max_temp
          }
        }'
    ) >> "$tmp" &
  done < "$NODES_CSV"
  wait
  sort "$tmp"
  rm -f "$tmp"
  sleep "$GPU_POLL_SECONDS"
done
EOF

cat > "$MONITOR_DIR/checkpoints.sh" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
while true; do
  header
  active_save="$(remote_batch "grep -oE 'save_ckpt=[^ ]+' '$STDERR_PATH' 2>/dev/null | tail -n 1 | sed 's/^save_ckpt=//'" || true)"
  save_root="${active_save:-$SAVE_CKPT}"
  echo "== MCore Checkpoints and Disk =="
  if [[ -z "$save_root" ]]; then
    echo "save_root=<not observed yet>"
  else
    remote_batch "echo save_root='$save_root'; echo; find '$save_root' -maxdepth 2 -type d -name 'iter_*' 2>/dev/null | sort | tail -n 20; echo; df -h '$save_root' 2>/dev/null" || true
  fi
  echo
  echo "== ZCC Durable Snapshots =="
  if [[ -z "$save_root" ]]; then
    echo "save_root=<not observed yet>"
  else
    remote_batch "find '$save_root/zcc' -maxdepth 4 -type f -name 'zcc_durable.pt*' 2>/dev/null | sed 's#/rank_[0-9][0-9]*/zcc_durable.pt.*##' | sort -u | tail -n 20" || true
  fi
  echo
  echo "== ZCC Flash Snapshot Counts =="
  while IFS=, read -r rank name ip; do
    [[ "$rank" == "rank" ]] && continue
    printf "[%03d] %-28s " "$rank" "$name"
    remote_node "$ip" "find /dev/shm/megatron_zcc -maxdepth 4 -type f -name 'zcc_snapshot.pt*' 2>/dev/null | awk -F/ '{print \$(NF-2)}' | sort | uniq -c | tail -n 3" 2>/dev/null | tr '\n' '; ' || printf "ssh failed"
    echo
  done < "$NODES_CSV"
  echo
  echo "== Save/HF Signals =="
  remote_batch "grep -E 'Saving checkpoint|successfully saved|Starting HF checkpoint upload|uploaded|retention|finalize|HF|ZCC|zero-cost|zcc_' '$STDERR_PATH' '$STDOUT_PATH' 2>/dev/null | tail -n 100" || true
  sleep 60
done
EOF

cat > "$MONITOR_DIR/wandb_hf.sh" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
while true; do
  header
  active_save="$(remote_batch "grep -oE 'save_ckpt=[^ ]+' '$STDERR_PATH' 2>/dev/null | tail -n 1 | sed 's/^save_ckpt=//'" || true)"
  save_root="${active_save:-$SAVE_CKPT}"
  echo "== W&B Lines =="
  remote_batch "grep -Ei 'wandb:|wandb_|Weights & Biases|syncing|Run data is saved' '$STDERR_PATH' '$STDOUT_PATH' 2>/dev/null | tail -n 80" || true
  echo
  echo "== HF Upload Sidecar Logs =="
  while IFS=, read -r rank name ip; do
    [[ "$rank" == "rank" ]] && continue
    printf "\n[%03d] %s\n" "$rank" "$name"
    padded="$(printf '%03d' "$rank")"
    if [[ -n "$save_root" ]]; then
      remote_node "$ip" "tail -n 8 '$save_root/hf_upload_logs/node${padded}.log' 2>/dev/null || true" 2>/dev/null | sed 's/^/  /' || echo "  ssh failed"
    else
      echo "  save_root=<not observed yet>"
    fi
  done < "$NODES_CSV"
  sleep 60
done
EOF

cat > "$MONITOR_DIR/raw_tail.sh" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
echo "Raw Slurm stdout/stderr tail for job $JOB_ID. This will include benign framework warnings."
remote_batch "tail -n 200 -F '$STDERR_PATH' '$STDOUT_PATH' 2>/dev/null" || true
EOF

cat > "$MONITOR_DIR/profile.sh" <<'EOF'
#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
LOCAL_TRACE_DIR="$MONITOR_DIR/profile_traces"
LOCAL_REPORT_HTML="$MONITOR_DIR/profile_report.html"
LOCAL_REPORT_MD="$MONITOR_DIR/profile_report.md"
LOCAL_VALIDATION_JSON="$MONITOR_DIR/profile_validation.json"
mkdir -p "$LOCAL_TRACE_DIR"

profile_dir_from_logs() {
  local tb
  tb="$(remote_batch "grep -oE 'tensorboard_logs=[^ ]+' '$STDERR_PATH' 2>/dev/null | tail -n 1 | sed 's/^tensorboard_logs=//'" 2>/dev/null || true)"
  if [[ -z "$tb" || "$tb" == "<default>" ]]; then
    return 1
  fi
  printf '%s/torch_profile\n' "$(dirname "$tb")"
}

profile_ranks_from_logs() {
  local ranks
  ranks="$(remote_batch "grep -oE 'profile_ranks=[^ ]+' '$STDERR_PATH' 2>/dev/null | tail -n 1 | sed 's/^profile_ranks=//'" 2>/dev/null || true)"
  if [[ -z "$ranks" || "$ranks" == "<unset>" ]]; then
    return 1
  fi
  printf '%s\n' "$ranks"
}

while true; do
  header
  echo "== Profile Artifacts and Live Report =="
  remote_profile_dir="$(profile_dir_from_logs || true)"
  if [[ -z "$remote_profile_dir" ]]; then
    echo "profile_dir=<not observed yet>"
    echo
    echo "Waiting for launch line with tensorboard_logs=..."
    remote_batch "grep -E 'profile_run=|tensorboard_logs=|\\[moe-stage\\]|\\[pipeline-progress\\]|Traceback|out of memory' '$STDERR_PATH' '$STDOUT_PATH' 2>/dev/null | tail -n 80" || true
    sleep "$POLL_SECONDS"
    continue
  fi

  echo "remote_profile_dir=$remote_profile_dir"
  echo "local_trace_dir=$LOCAL_TRACE_DIR"
  echo "report_html=$LOCAL_REPORT_HTML"
  echo "report_md=$LOCAL_REPORT_MD"
  echo "validation_json=$LOCAL_VALIDATION_JSON"
  echo

  while IFS=, read -r rank name ip; do
    [[ "$rank" == "rank" ]] && continue
    printf "[%03d] %-28s " "$rank" "$name"
    remote_node "$ip" "if [[ -d '$remote_profile_dir' ]]; then find '$remote_profile_dir' -maxdepth 1 -type f \\( -name 'rank-*.json' -o -name 'rank-*.json.gz' \\) -printf '%f %s bytes\n' 2>/dev/null | sort | tail -n 8; else echo no_profile_dir; fi" 2>/dev/null \
      | sed '1!s/^/                                 /' || echo "ssh failed"
  done < "$NODES_CSV"

  if command -v rsync >/dev/null 2>&1; then
    while IFS=, read -r rank name ip; do
      [[ "$rank" == "rank" ]] && continue
      mkdir -p "$LOCAL_TRACE_DIR/node-${rank}-${name}"
      rsync -az --ignore-existing \
        -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=6" \
        "$SSH_USER@$ip:$remote_profile_dir/" "$LOCAL_TRACE_DIR/node-${rank}-${name}/" >/dev/null 2>&1 || true
    done < "$NODES_CSV"
  fi

  echo
  echo "== Aggregated Local Trace Summary =="
  trace_count="$(find "$LOCAL_TRACE_DIR" -type f \( -name 'rank-*.json' -o -name 'rank-*.json.gz' \) 2>/dev/null | wc -l)"
  echo "local_trace_count=$trace_count"
  if (( trace_count > 0 )); then
    echo
    echo "== Trace Validity =="
    validate_ranks="${PROFILE_VALIDATE_RANKS:-$(profile_ranks_from_logs || true)}"
    validate_args=(--input "$LOCAL_TRACE_DIR" --json "$LOCAL_VALIDATION_JSON" --require-cuda-events)
    if [[ -n "$validate_ranks" ]]; then
      validate_args+=(--require-ranks "$validate_ranks")
      echo "required_ranks=$validate_ranks"
    else
      echo "required_ranks=<not observed>"
    fi
    (cd "$REPO_DIR" && uv run --no-sync python tools/validate_profile_artifacts.py "${validate_args[@]}") 2>&1 | tail -n 80 || true

    echo
    echo "== Trace Report =="
    (cd "$REPO_DIR" && uv run --no-sync python tools/profile_trace_report.py \
      --input "$LOCAL_TRACE_DIR" \
      --output "$LOCAL_REPORT_HTML" \
      --markdown "$LOCAL_REPORT_MD" \
      --top 24 \
      --text) 2>&1 | tail -n 120
  else
    echo "No trace files copied yet. The profiler writes after its active window finishes."
  fi

  echo
  echo "== Recent Profile/MoE/OOM Signals =="
  remote_batch "grep -E 'profile_run=|tensorboard_logs=|\\[moe-stage\\]|\\[pipeline-progress\\]|Traceback|out of memory|torch_profile|export_chrome_trace' '$STDERR_PATH' '$STDOUT_PATH' 2>/dev/null | tail -n 80" || true
  sleep "$POLL_SECONDS"
done
EOF

cat > "$MONITOR_DIR/control.txt" <<EOF
job: $JOB_ID
batch host: $batch_host ($batch_ip)
stdout: $stdout_path
stderr: $stderr_path
monitor dir: $MONITOR_DIR

attach:
  tmux attach -t $SESSION
detach:
  Ctrl-b d
cancel training:
  scancel $JOB_ID
poll first five steps:
  JOB_ID=$JOB_ID handoff/gcp-a4/scripts/poll_slurm_training_steps.sh

Dashboard panes:
  top-left: Slurm status and launch shape
  top-right: parsed training progress
  bottom-left: compact GPU fleet summary
  bottom-right: filtered critical alerts

Secondary windows:
  rank0     filtered rank/progress/W&B tail
  gpu-detail per-GPU metrics by node
  gpu-heat  compact per-node memory/util heatmap
  ckpt      checkpoints, disk, save/HF signals
  profile   live trace collection plus generated profile_report.html/md
  wandb-hf  W&B lines plus HF sidecar logs
  raw-tail  unfiltered logs, expected to include benign warnings

Profile artifacts:
  live report: $MONITOR_DIR/profile_report.html
  markdown:    $MONITOR_DIR/profile_report.md
  validation:  $MONITOR_DIR/profile_validation.json
  copied traces: $MONITOR_DIR/profile_traces
EOF

chmod +x "$MONITOR_DIR/"*.sh

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n dash "bash $(q "$MONITOR_DIR/status.sh")"
tmux split-window -h -t "$SESSION:dash" "bash $(q "$MONITOR_DIR/progress.sh")"
tmux split-window -v -t "$SESSION:dash.0" "bash $(q "$MONITOR_DIR/gpu_summary.sh")"
tmux split-window -v -t "$SESSION:dash.1" "bash $(q "$MONITOR_DIR/alerts.sh")"
tmux select-layout -t "$SESSION:dash" tiled
tmux new-window -t "$SESSION" -n rank0 "bash $(q "$MONITOR_DIR/rank0.sh")"
tmux new-window -t "$SESSION" -n gpu-detail "bash $(q "$MONITOR_DIR/gpu_detail.sh")"
tmux new-window -t "$SESSION" -n gpu-heat "bash $(q "$MONITOR_DIR/gpu_heatmap.sh")"
tmux new-window -t "$SESSION" -n ckpt "bash $(q "$MONITOR_DIR/checkpoints.sh")"
tmux new-window -t "$SESSION" -n profile "bash $(q "$MONITOR_DIR/profile.sh")"
tmux new-window -t "$SESSION" -n wandb-hf "bash $(q "$MONITOR_DIR/wandb_hf.sh")"
tmux new-window -t "$SESSION" -n raw-tail "bash $(q "$MONITOR_DIR/raw_tail.sh")"
tmux new-window -t "$SESSION" -n control "cat $(q "$MONITOR_DIR/control.txt"); exec bash"
tmux select-window -t "$SESSION:dash"

echo "Monitor ready: tmux attach -t $SESSION"
