#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SNAPSHOT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --snapshot-dir=*)
      SNAPSHOT_DIR="${1#*=}"
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "megatron-criu-entrypoint: unknown arg $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${SNAPSHOT_DIR}" && -f "${SNAPSHOT_DIR}/img/inventory.img" ]]; then
  CRIU=${CRIU:-/opt/criu-snapshots/criu}
  PLUGINS=${PLUGINS:-/opt/criu-snapshots/plugins}
  exec "${CRIU}" restore \
    -D "${SNAPSHOT_DIR}/img" \
    -W "${SNAPSHOT_DIR}/logs" \
    -v4 -o restore.log \
    -d --restore-detached \
    --shell-job \
    --tcp-established \
    --ext-unix-sk \
    --file-locks \
    --manage-cgroups full \
    --lazy-pages \
    --lib "${PLUGINS}"
fi

tools_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${tools_dir}${PYTHONPATH:+:${PYTHONPATH}}"
exec "$@"
