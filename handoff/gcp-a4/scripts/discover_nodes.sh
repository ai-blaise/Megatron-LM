#!/usr/bin/env bash
set -euo pipefail

ZONE="${ZONE:-us-east1-b}"
INSTANCE_REGEX="${INSTANCE_REGEX:-^instance-group-1-}"
STATUS="${STATUS:-RUNNING}"
OUTPUT="${OUTPUT:-table}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required for node discovery" >&2
  exit 127
fi

mapfile -t rows < <(
  gcloud compute instances list \
    --zones="$ZONE" \
    --filter="name~${INSTANCE_REGEX} AND status=${STATUS}" \
    --sort-by=name \
    --format='csv[no-heading](name,networkInterfaces[0].networkIP,networkInterfaces[0].accessConfigs[0].natIP)'
)

case "$OUTPUT" in
  csv)
    echo "rank,name,internal_ip,nat_ip"
    ;;
  table)
    printf "%-5s %-28s %-16s %s\n" "rank" "name" "internal_ip" "nat_ip"
    ;;
  names|ips|count)
    ;;
  *)
    echo "Unsupported OUTPUT=$OUTPUT. Use table, csv, names, ips, or count." >&2
    exit 2
    ;;
esac

if [[ "$OUTPUT" == "count" ]]; then
  echo "${#rows[@]}"
  exit 0
fi

rank=0
for row in "${rows[@]}"; do
  IFS=, read -r name internal_ip nat_ip <<<"$row"
  case "$OUTPUT" in
    csv)
      printf "%s,%s,%s,%s\n" "$rank" "$name" "$internal_ip" "$nat_ip"
      ;;
    table)
      printf "%-5s %-28s %-16s %s\n" "$rank" "$name" "$internal_ip" "$nat_ip"
      ;;
    names)
      printf "%s\n" "$name"
      ;;
    ips)
      printf "%s\n" "$internal_ip"
      ;;
  esac
  rank=$((rank + 1))
done
