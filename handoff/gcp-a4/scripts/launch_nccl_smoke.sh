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
  for row in "${rows[@]}"; do
    IFS=, read -r _rank name internal_ip _nat_ip <<<"$row"
    names+=("$name")
    ips+=("$internal_ip")
  done
fi

if (( ${#names[@]} == 0 )); then
  echo "No nodes discovered for NCCL smoke" >&2
  exit 2
fi

NNODES="${NNODES:-${#names[@]}}"
if (( NNODES > ${#names[@]} )); then
  echo "NNODES=$NNODES exceeds discovered node count ${#names[@]}" >&2
  exit 2
fi

host_short="${HOSTNAME_SHORT:-$(hostname -s)}"
NODE_RANK="${NODE_RANK:-}"
if [[ -z "$NODE_RANK" ]]; then
  for idx in $(seq 0 $((NNODES - 1))); do
    if [[ "${names[$idx]}" == "$host_short" ]]; then
      NODE_RANK="$idx"
      break
    fi
  done
fi
if [[ -z "$NODE_RANK" ]]; then
  echo "Host $host_short is not in selected smoke node set" >&2
  exit 2
fi

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
export NNODES GPUS_PER_NODE WORLD_SIZE NODE_RANK
export MASTER_ADDR="${MASTER_ADDR:-${ips[0]}}"
export MASTER_PORT="${MASTER_PORT:-29691}"

export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export USE_NCCL_GIB="${USE_NCCL_GIB:-0}"
if [[ "$USE_NCCL_GIB" == "1" ]]; then
  if [[ -f /usr/local/gib/scripts/set_nccl_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/gib/scripts/set_nccl_env.sh
    export LD_LIBRARY_PATH="/usr/local/gib/lib64:${LD_LIBRARY_PATH:-}"
  else
    echo "WARNING: USE_NCCL_GIB=1 but /usr/local/gib/scripts/set_nccl_env.sh is missing" >&2
  fi
fi
if [[ -z "${MEGATRON_LOCAL_RANK_NCCL_HCA+x}" ]]; then
  export MEGATRON_LOCAL_RANK_NCCL_HCA=0
fi
export MEGATRON_NCCL_LOCAL_RANK_HCAS="${MEGATRON_NCCL_LOCAL_RANK_HCAS:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
if [[ "$MEGATRON_LOCAL_RANK_NCCL_HCA" == "1" ]]; then
  unset NCCL_IB_HCA
elif [[ "$USE_NCCL_GIB" != "1" ]]; then
  export NCCL_IB_HCA="${NCCL_IB_HCA:-$MEGATRON_NCCL_LOCAL_RANK_HCAS}"
fi
export USE_TP_PP_DP_MAPPING="${USE_TP_PP_DP_MAPPING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if ! ulimit -l unlimited 2>/dev/null; then
  echo "WARNING: could not raise memlock soft limit to unlimited" >&2
fi

TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-$HOME/logs/nccl-smoke-${SLURM_JOB_ID:-manual}/node$(printf '%03d' "$NODE_RANK")-$host_short/torchrun}"
mkdir -p "$TORCHRUN_LOG_DIR"

echo "GCP A4 NCCL smoke:"
echo "  host=$host_short node_rank=$NODE_RANK nnodes=$NNODES gpus_per_node=$GPUS_PER_NODE world_size=$WORLD_SIZE"
echo "  master=$MASTER_ADDR:$MASTER_PORT"
echo "  nccl_net=${NCCL_NET:-<default>} use_nccl_gib=$USE_NCCL_GIB local_rank_hca=$MEGATRON_LOCAL_RANK_NCCL_HCA hcas=$MEGATRON_NCCL_LOCAL_RANK_HCAS parent_NCCL_IB_HCA=${NCCL_IB_HCA:-<unset>}"
echo "  memlock_soft=$(ulimit -Sl) memlock_hard=$(ulimit -Hl) nccl_max_ctas=${NCCL_MAX_CTAS:-<unset>} nccl_netdevs_policy=${NCCL_NETDEVS_POLICY:-<unset>} nccl_ib_merge_nics=${NCCL_IB_MERGE_NICS:-<unset>}"
echo "  shape tp=$TP pp=$PP cp=$CP ep=$EP etp=$ETP use_tp_pp_dp_mapping=$USE_TP_PP_DP_MAPPING"
echo "  torchrun_log_dir=$TORCHRUN_LOG_DIR"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

uv run --no-sync torchrun \
  --log-dir "$TORCHRUN_LOG_DIR" \
  --redirects "${TORCHRUN_REDIRECTS:-3}" \
  --tee "${TORCHRUN_TEE:-3}" \
  --nnodes "$NNODES" \
  --nproc_per_node "$GPUS_PER_NODE" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  tools/debug_nccl_fleet.py \
  --timeout "${SMOKE_TIMEOUT_SECONDS:-240}" \
  --iters "${SMOKE_ITERS:-3}" \
  --sizes "${SMOKE_SIZES:-1,1024,1048576}" \
  --p2p-numel "${SMOKE_P2P_NUMEL:-4}" \
  --tp "$TP" \
  --pp "$PP" \
  --cp "$CP" \
  --ep "$EP" \
  --etp "$ETP" \
  $(if [[ "$USE_TP_PP_DP_MAPPING" == "1" ]]; then printf '%s\n' --use-tp-pp-dp-mapping; fi) \
  ${SMOKE_EXTRA_ARGS:-}
