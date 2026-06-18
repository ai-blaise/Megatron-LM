#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}"
cd "$REPO_DIR"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP="${TP:-8}"
PP="${PP:-5}"
CP="${CP:-1}"
EP="${EP:-8}"
ETP="${ETP:-1}"
STRICT_ALL_NODES="${STRICT_ALL_NODES:-1}"
MAX_NODES="${MAX_NODES:-}"

MODEL_PARALLEL_SIZE=$((TP * PP * CP))
EXPERT_MODEL_PARALLEL_SIZE=$((ETP * EP * PP))

names=()
ips=()
if [[ -n "${SLURM_JOB_NODELIST:-}" && "${FORCE_GCLOUD_DISCOVERY:-0}" != "1" ]]; then
  mapfile -t names < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  for name in "${names[@]}"; do
    ip="$(scontrol show node "$name" | awk '{for (i=1; i<=NF; i++) if ($i ~ /^NodeAddr=/) {sub(/^NodeAddr=/, "", $i); print $i; exit}}')"
    ips+=("${ip:-$name}")
  done
else
  mapfile -t rows < <(OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | tail -n +2)
  if (( ${#rows[@]} == 0 )); then
    echo "No running nodes matched INSTANCE_REGEX=${INSTANCE_REGEX:-^instance-group-1-} in ZONE=${ZONE:-us-east1-b}" >&2
    exit 2
  fi
  for row in "${rows[@]}"; do
    IFS=, read -r _rank name internal_ip _nat_ip <<<"$row"
    names+=("$name")
    ips+=("$internal_ip")
  done
fi

discovered_count="${#names[@]}"
use_count="${MAX_NODES:-$discovered_count}"
if (( use_count > discovered_count )); then
  echo "MAX_NODES=$use_count exceeds discovered node count $discovered_count" >&2
  exit 2
fi

while (( use_count > 0 )); do
  world_size=$((use_count * GPUS_PER_NODE))
  if (( world_size % MODEL_PARALLEL_SIZE == 0 && world_size % EXPERT_MODEL_PARALLEL_SIZE == 0 )); then
    break
  fi
  if [[ "$STRICT_ALL_NODES" == "1" || -n "$MAX_NODES" ]]; then
    cat >&2 <<EOF
Invalid conversion shape:
  nodes=$use_count gpus_per_node=$GPUS_PER_NODE world_size=$world_size
  TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP
  required: world_size divisible by $MODEL_PARALLEL_SIZE and $EXPERT_MODEL_PARALLEL_SIZE
EOF
    exit 2
  fi
  use_count=$((use_count - 1))
done

host_short="${HOSTNAME_SHORT:-$(hostname -s)}"
node_rank="${NODE_RANK:-}"
if [[ -z "$node_rank" ]]; then
  for idx in $(seq 0 $((use_count - 1))); do
    if [[ "${names[$idx]}" == "$host_short" ]]; then
      node_rank="$idx"
      break
    fi
  done
fi

if [[ -z "$node_rank" ]]; then
  cat >&2 <<EOF
This host ($host_short) is not in the selected conversion node set.
Selected nodes:
$(for idx in $(seq 0 $((use_count - 1))); do printf "  %s %s %s\n" "$idx" "${names[$idx]}" "${ips[$idx]}"; done)
EOF
  exit 2
fi

export NNODES="$use_count"
export GPUS_PER_NODE
export NODE_RANK="$node_rank"
export MASTER_ADDR="${MASTER_ADDR:-${ips[0]}}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export TP PP CP EP ETP
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
if [[ "$PP" -eq 5 ]]; then
  export DECODER_FIRST_PIPELINE_NUM_LAYERS="${DECODER_FIRST_PIPELINE_NUM_LAYERS:-13}"
  export DECODER_LAST_PIPELINE_NUM_LAYERS="${DECODER_LAST_PIPELINE_NUM_LAYERS:-12}"
fi

cat >&2 <<EOF
GCP A4 conversion shape:
  discovered_nodes=$discovered_count selected_nodes=$NNODES node_rank=$NODE_RANK
  master=$MASTER_ADDR:$MASTER_PORT world_size=$((NNODES * GPUS_PER_NODE))
  TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP seq=$SEQ_LENGTH
  first_layers=${DECODER_FIRST_PIPELINE_NUM_LAYERS:-auto} last_layers=${DECODER_LAST_PIPELINE_NUM_LAYERS:-auto}
  output=${LOAD_CKPT:-$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron}
EOF

examples/sft/convert_deepseek_v32_reap.sh
