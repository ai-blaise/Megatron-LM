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
  mkdir -p "${SNAPSHOT_DIR}/logs"
  PLUGINS=${PLUGINS:-/opt/criu-snapshots/plugins}
  PIDFILE="${SNAPSHOT_DIR}/restore.pid"
  export PATH="/opt/criu-snapshots/bin:${PATH}"

  signal_post_restore() {
    local signal_number=${MEGATRON_CRIU_POST_RESTORE_SIGNAL:-40}
    local state_dir=${MEGATRON_CRIU_STATE_DIR:-/var/run/megatron-criu}
    local ready_file=${MEGATRON_CRIU_POST_RESTORE_READY:-${state_dir}/post_restore.done}
    local timeout=${MEGATRON_CRIU_POST_RESTORE_TIMEOUT:-120}
    local root_pid=""
    local deadline

    [[ "${MEGATRON_CRIU_SIGNAL_POST_RESTORE:-1}" == "1" ]] || return 0
    if [[ -f "${PIDFILE}" ]]; then
      root_pid=$(<"${PIDFILE}")
    fi
    [[ -n "${root_pid}" ]] || {
      echo "megatron-criu-entrypoint: restore pidfile is empty" >&2
      return 1
    }

    kill -s "${signal_number}" "${root_pid}"
    [[ "${timeout}" == "0" ]] && return 0

    deadline=$((SECONDS + timeout))
    while (( SECONDS < deadline )); do
      [[ -s "${ready_file}" ]] && return 0
      sleep 1
    done
    echo "megatron-criu-entrypoint: timed out waiting for ${ready_file}" >&2
    return 1
  }

  restore_args=(
    restore
    -D "${SNAPSHOT_DIR}/img"
    -W "${SNAPSHOT_DIR}/logs"
    --pidfile "${PIDFILE}"
    --root /
    --join-ns mnt:/proc/self/ns/mnt
    -v4 -o restore.log
    -d --restore-detached
    --shell-job
    --tcp-established
    --ext-unix-sk
    --file-locks
    --manage-cgroups=full
    --libdir "${PLUGINS}"
  )

  if [[ "${MEGATRON_CRIU_INHERIT_NETNS:-1}" == "1" ]]; then
    exec {netns_fd}< /proc/self/ns/net
    restore_args+=(--inherit-fd "fd[${netns_fd}]:extRootNetNS")
  fi

  set +e
  "${CRIU}" "${restore_args[@]}"
  status=$?
  set -e
  if [[ ${status} -ne 0 ]]; then
    cat "${SNAPSHOT_DIR}/logs/restore.log" >&2 || true
    exit "${status}"
  fi
  signal_post_restore
  if [[ "${MEGATRON_CRIU_RUN_COMMAND_AFTER_RESTORE:-0}" == "1" && $# -gt 0 ]]; then
    "$@"
    command_status=$?
    if [[ "${MEGATRON_CRIU_EXIT_AFTER_RESTORE_COMMAND:-0}" == "1" ]]; then
      exit "${command_status}"
    fi
  fi
  if [[ "${MEGATRON_CRIU_EXIT_AFTER_RESTORE:-0}" == "1" ]]; then
    exit 0
  fi
  sleep infinity
  exit 0
fi

tools_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${tools_dir}${PYTHONPATH:+:${PYTHONPATH}}"
exec "$@"
