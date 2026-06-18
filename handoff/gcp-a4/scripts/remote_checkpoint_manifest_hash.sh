#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron_tp8_pp1_ep8}"
OUT_DIR="${OUT_DIR:-$HOME/logs/ckpt-hash-verify-$(basename "$CHECKPOINT_DIR")}"
HASH_WORKERS="${HASH_WORKERS:-96}"
HASH_MODE="${HASH_MODE:-chunk}"
CHUNK_BYTES="${CHUNK_BYTES:-268435456}"
DD_BS="${DD_BS:-16M}"

mkdir -p "$OUT_DIR"

status_file="$OUT_DIR/status"
manifest_tmp="$OUT_DIR/manifest.sha256.tmp"
manifest="$OUT_DIR/manifest.sha256"
manifest_sum="$OUT_DIR/manifest.sha256.sum"
tasks_file="$OUT_DIR/hash_tasks.tsv"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  {
    echo "status=missing_checkpoint_dir"
    echo "host=$(hostname -s)"
    echo "checkpoint_dir=$CHECKPOINT_DIR"
  } >"$status_file"
  exit 2
fi

if [[ ! -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]]; then
  {
    echo "status=missing_tracker"
    echo "host=$(hostname -s)"
    echo "checkpoint_dir=$CHECKPOINT_DIR"
  } >"$status_file"
  exit 2
fi

{
  echo "status=running"
  echo "host=$(hostname -s)"
  echo "checkpoint_dir=$CHECKPOINT_DIR"
  echo "workers=$HASH_WORKERS"
  echo "mode=$HASH_MODE"
  echo "chunk_bytes=$CHUNK_BYTES"
  echo "start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$status_file"

cd "$CHECKPOINT_DIR"

case "$HASH_MODE" in
  file)
    find . -type f -print0 \
      | sort -z \
      | xargs -0 -n1 -P "$HASH_WORKERS" sha256sum \
      | LC_ALL=C sort -k2,2 >"$manifest_tmp"
    ;;
  chunk)
    find . -type f -printf '%P\t%s\n' | LC_ALL=C sort >"$tasks_file.files"
    : >"$tasks_file"
    while IFS=$'\t' read -r path size; do
      if (( size == 0 )); then
        printf '%s\t0\t0\t0\n' "$path" >>"$tasks_file"
        continue
      fi

      chunks=$(( (size + CHUNK_BYTES - 1) / CHUNK_BYTES ))
      for ((idx = 0; idx < chunks; idx++)); do
        offset=$((idx * CHUNK_BYTES))
        length=$CHUNK_BYTES
        if (( offset + length > size )); then
          length=$((size - offset))
        fi
        printf '%s\t%s\t%s\t%s\n' "$path" "$idx" "$offset" "$length" >>"$tasks_file"
      done
    done <"$tasks_file.files"

    export DD_BS
    xargs -r -P "$HASH_WORKERS" -n4 bash -c '
      set -euo pipefail
      path="$1"
      idx="$2"
      offset="$3"
      length="$4"
      if [[ "$length" == "0" ]]; then
        digest="$(sha256sum /dev/null | awk "{print \$1}")"
      else
        digest="$(dd if="./$path" bs="$DD_BS" iflag=skip_bytes,count_bytes skip="$offset" count="$length" status=none | sha256sum | awk "{print \$1}")"
      fi
      printf "%s\t%012d\t%s\t%s\t%s\n" "$path" "$idx" "$offset" "$length" "$digest"
    ' _ <"$tasks_file" | LC_ALL=C sort -t $'\t' -k1,1 -k2,2 >"$manifest_tmp"
    ;;
  *)
    echo "Unsupported HASH_MODE=$HASH_MODE" >&2
    exit 2
    ;;
esac

mv "$manifest_tmp" "$manifest"
sha256sum "$manifest" >"$manifest_sum"

{
  echo "status=complete"
  echo "host=$(hostname -s)"
  echo "checkpoint_dir=$CHECKPOINT_DIR"
  echo "workers=$HASH_WORKERS"
  echo "mode=$HASH_MODE"
  echo "chunk_bytes=$CHUNK_BYTES"
  echo "files=$(find . -type f | wc -l)"
  echo "manifest_entries=$(wc -l < "$manifest")"
  echo "manifest_sha256=$(awk '{print $1}' "$manifest_sum")"
  echo "end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$status_file"
