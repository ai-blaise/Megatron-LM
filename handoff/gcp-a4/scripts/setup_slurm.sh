#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}"
cd "$REPO_DIR"

SSH_USER="${SSH_USER:-sjpat}"
if [[ -z "${SSH_KEY:-}" ]]; then
  if [[ -f "$HOME/google_compute_engine" ]]; then
    SSH_KEY="$HOME/google_compute_engine"
  else
    SSH_KEY="$HOME/.ssh/google_compute_engine"
  fi
fi

CLUSTER_NAME="${CLUSTER_NAME:-gcp-a4}"
PARTITION_NAME="${PARTITION_NAME:-a4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CONTROLLER_NAME="${CONTROLLER_NAME:-$(hostname -s)}"
CPUS_PER_NODE="${CPUS_PER_NODE:-$(nproc)}"
REAL_MEMORY_MB="${REAL_MEMORY_MB:-$(awk '/MemTotal/ {print int($2/1024)-1024}' /proc/meminfo)}"
SLURM_UID="${SLURM_UID:-64030}"
SLURM_GID="${SLURM_GID:-64030}"
SLURM_JOB_UID="${SLURM_JOB_UID:-$(id -u)}"
SLURM_JOB_HOME="${SLURM_JOB_HOME:-$HOME}"
SLURM_CONF_LOCAL="${SLURM_CONF_LOCAL:-/tmp/${CLUSTER_NAME}.slurm.conf}"
GRES_CONF_LOCAL="${GRES_CONF_LOCAL:-/tmp/${CLUSTER_NAME}.gres.conf}"

mapfile -t rows < <(OUTPUT=csv "$SCRIPT_DIR/discover_nodes.sh" | tail -n +2)
if (( ${#rows[@]} == 0 )); then
  echo "No running GCP nodes discovered" >&2
  exit 2
fi

names=()
ips=()
controller_ip=""
for row in "${rows[@]}"; do
  IFS=, read -r _rank name internal_ip _nat_ip <<<"$row"
  names+=("$name")
  ips+=("$internal_ip")
  if [[ "$name" == "$CONTROLLER_NAME" ]]; then
    controller_ip="$internal_ip"
  fi
done

if [[ -z "$controller_ip" ]]; then
  echo "Controller $CONTROLLER_NAME was not found in discovered node list" >&2
  exit 2
fi

ssh_base=(
  ssh
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o UserKnownHostsFile=/dev/null
  -o StrictHostKeyChecking=no
)

remote_sh() {
  local ip="$1"
  shift
  "${ssh_base[@]}" "$SSH_USER@$ip" "$@"
}

q() {
  printf '%q' "$1"
}

render_slurm_conf() {
  cat <<EOF
ClusterName=$CLUSTER_NAME
SlurmctldHost=$CONTROLLER_NAME($controller_ip)
SlurmUser=slurm
AuthType=auth/munge
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
SlurmctldPort=6817
SlurmdPort=6818
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
MpiDefault=none
ReturnToService=2
InactiveLimit=0
KillWait=30
MinJobAge=300
SlurmctldTimeout=120
SlurmdTimeout=120
Waittime=0
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
AccountingStorageType=accounting_storage/none
JobAcctGatherType=jobacct_gather/none
GresTypes=gpu
DebugFlags=NO_CONF_HASH

EOF
  for idx in "${!names[@]}"; do
    printf 'NodeName=%s NodeAddr=%s CPUs=%s RealMemory=%s Gres=gpu:%s State=UNKNOWN\n' \
      "${names[$idx]}" "${ips[$idx]}" "$CPUS_PER_NODE" "$REAL_MEMORY_MB" "$GPUS_PER_NODE"
  done
  cat <<EOF
PartitionName=$PARTITION_NAME Nodes=ALL Default=YES MaxTime=INFINITE State=UP
EOF
}

render_gres_conf() {
  for name in "${names[@]}"; do
    printf 'NodeName=%s Name=gpu File=/dev/nvidia[0-%s]\n' \
      "$name" "$((GPUS_PER_NODE - 1))"
  done
}

install_packages() {
  local idx name ip
  for idx in "${!names[@]}"; do
    name="${names[$idx]}"
    ip="${ips[$idx]}"
    echo "[$idx] installing Slurm/Munge packages on $name ($ip)" >&2
    remote_sh "$ip" "sudo dnf install -y epel-release"
    if [[ "$name" == "$CONTROLLER_NAME" ]]; then
      remote_sh "$ip" "sudo dnf install -y munge slurm slurm-slurmctld slurm-slurmd"
    else
      remote_sh "$ip" "sudo dnf install -y munge slurm slurm-slurmd"
    fi
  done
}

ensure_slurm_user() {
  local idx ip
  for idx in "${!names[@]}"; do
    ip="${ips[$idx]}"
    echo "[$idx] ensuring Slurm system user on ${names[$idx]}" >&2
    remote_sh "$ip" "if ! getent group slurm >/dev/null; then sudo groupadd --system --gid $(q "$SLURM_GID") slurm; fi; if ! getent passwd slurm >/dev/null; then sudo useradd --system --uid $(q "$SLURM_UID") --gid slurm --home-dir /var/lib/slurm --shell /sbin/nologin slurm; fi; sudo install -d -m 0755 -o slurm -g slurm /var/lib/slurm"
  done
}

ensure_submit_user_access() {
  local idx ip
  for idx in "${!names[@]}"; do
    ip="${ips[$idx]}"
    echo "[$idx] granting Slurm job UID access on ${names[$idx]}" >&2
    remote_sh "$ip" "uid=$(q "$SLURM_JOB_UID"); home=$(q "$SLURM_JOB_HOME"); sudo mkdir -p \"\$home/.cache\" \"\$home/.config/megatron\" \"\$home/checkpoints\" \"\$home/data\" \"\$home/logs\" \"\$home/tensorboard_logs\"; sudo setfacl -m u:\$uid:--x \"\$home\"; for f in \"\$home/.bash_profile\" \"\$home/.bashrc\" \"\$home/.netrc\" \"\$home/ncclx_topology.env\"; do if [[ -e \"\$f\" ]]; then sudo setfacl -m u:\$uid:r \"\$f\"; fi; done; for d in \"\$home/Megatron-LM\" \"\$home/.cache\" \"\$home/.config/megatron\" \"\$home/checkpoints\" \"\$home/data\" \"\$home/logs\" \"\$home/tensorboard_logs\"; do if [[ -e \"\$d\" ]]; then sudo setfacl -R -m u:\$uid:rwX \"\$d\"; sudo find \"\$d\" -type d -exec setfacl -m d:u:\$uid:rwX {} +; fi; done"
  done
}

install_munge_key() {
  local key_b64
  if [[ "${FORCE_MUNGE_KEY:-0}" == "1" || ! -s /etc/munge/munge.key ]]; then
    sudo install -d -m 0700 -o munge -g munge /etc/munge
    dd if=/dev/urandom bs=1 count=1024 status=none | sudo tee /etc/munge/munge.key >/dev/null
    sudo chown munge:munge /etc/munge/munge.key
    sudo chmod 0400 /etc/munge/munge.key
  fi
  key_b64="$(sudo base64 -w0 /etc/munge/munge.key)"
  for idx in "${!names[@]}"; do
    echo "[$idx] installing Munge key on ${names[$idx]}" >&2
    remote_sh "${ips[$idx]}" "sudo install -d -m 0700 -o munge -g munge /etc/munge; printf %s $(q "$key_b64") | sudo base64 -d > /tmp/munge.key; sudo install -m 0400 -o munge -g munge /tmp/munge.key /etc/munge/munge.key; rm -f /tmp/munge.key"
  done
}

install_configs() {
  render_slurm_conf > "$SLURM_CONF_LOCAL"
  render_gres_conf > "$GRES_CONF_LOCAL"
  for idx in "${!names[@]}"; do
    echo "[$idx] installing Slurm configs on ${names[$idx]}" >&2
    tar -C /tmp -cf - "$(basename "$SLURM_CONF_LOCAL")" "$(basename "$GRES_CONF_LOCAL")" | \
      remote_sh "${ips[$idx]}" "tmpdir=\$(mktemp -d); tar -C \"\$tmpdir\" -xf -; sudo install -d -m 0755 /etc/slurm; sudo install -m 0644 \"\$tmpdir/$(basename "$SLURM_CONF_LOCAL")\" /etc/slurm/slurm.conf; sudo install -m 0644 \"\$tmpdir/$(basename "$GRES_CONF_LOCAL")\" /etc/slurm/gres.conf; rm -rf \"\$tmpdir\""
  done
}

prepare_spools_and_services() {
  for idx in "${!names[@]}"; do
    name="${names[$idx]}"
    ip="${ips[$idx]}"
    echo "[$idx] preparing services on $name" >&2
    remote_sh "$ip" "sudo install -d -m 0755 -o slurm -g slurm /var/spool/slurmd /var/log/slurm; sudo systemctl enable --now munge; sudo systemctl restart munge; sudo systemctl enable slurmd"
    if [[ "$name" == "$CONTROLLER_NAME" ]]; then
      remote_sh "$ip" "sudo install -d -m 0755 -o slurm -g slurm /var/spool/slurmctld; sudo systemctl enable slurmctld; sudo systemctl restart slurmctld"
    fi
    remote_sh "$ip" "sudo systemctl restart slurmd"
  done
}

show_status() {
  sinfo -Nel || true
  scontrol ping || true
}

cat >&2 <<EOF
Slurm setup:
  cluster=$CLUSTER_NAME partition=$PARTITION_NAME controller=$CONTROLLER_NAME($controller_ip)
  nodes=${#names[@]} cpus_per_node=$CPUS_PER_NODE real_memory_mb=$REAL_MEMORY_MB gpus_per_node=$GPUS_PER_NODE
EOF

case "${ACTION:-setup}" in
  setup)
    install_packages
    ensure_slurm_user
    install_munge_key
    install_configs
    prepare_spools_and_services
    ensure_submit_user_access
    show_status
    ;;
  configs)
    render_slurm_conf
    echo "--- gres.conf ---"
    render_gres_conf
    ;;
  status)
    show_status
    ;;
  restart)
    ensure_slurm_user
    install_configs
    prepare_spools_and_services
    ensure_submit_user_access
    show_status
    ;;
  *)
    echo "Unsupported ACTION=${ACTION:-setup}. Use setup, configs, restart, or status." >&2
    exit 2
    ;;
esac
