#!/usr/bin/env bash
set -euo pipefail

DRIVER_VERSION="${DRIVER_VERSION:-580.159.03}"
EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
INSTALLER_URL="${INSTALLER_URL:-https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz}"
INSTALLER_PATH="${INSTALLER_PATH:-/tmp/cuda_installer.pyz}"
LOG_PATH="${LOG_PATH:-/tmp/nvidia-driver-install-${DRIVER_VERSION}.log}"

current_version() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
    | sort -u \
    | tr -d ' '
}

gpu_count() {
  nvidia-smi -L 2>/dev/null | wc -l
}

if command -v nvidia-smi >/dev/null 2>&1; then
  installed="$(current_version || true)"
  count="$(gpu_count || true)"
  if [[ "$installed" == "$DRIVER_VERSION" && "$count" == "$EXPECTED_GPUS" ]]; then
    echo "NVIDIA driver already installed: version=$installed gpus=$count"
    exit 0
  fi
fi

sudo systemctl stop google-cloud-ops-agent 2>/dev/null || true
curl -fsSL "$INSTALLER_URL" --output "$INSTALLER_PATH"

sudo python3 "$INSTALLER_PATH" install_driver \
  --installation-mode=binary \
  --installation-branch=lts \
  --force-version "$DRIVER_VERSION" 2>&1 | tee "$LOG_PATH"

installed="$(current_version)"
count="$(gpu_count)"
echo "NVIDIA driver installed: version=$installed gpus=$count"
[[ "$installed" == "$DRIVER_VERSION" ]]
[[ "$count" == "$EXPECTED_GPUS" ]]
