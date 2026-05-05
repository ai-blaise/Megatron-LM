#!/bin/bash
# Prepare BlaiseAI DeepSeek-V3.2 SFT JSONL for Megatron SFTDataset.

#SBATCH --job-name=prep_blaise_sft
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${MEGATRON_DIR:-"${SCRIPT_DIR}/../.."}"
cd "$MEGATRON_DIR"

DATASET_ID="${DATASET_ID:-BlaiseAI/blaise-sft-training-mix}"
DATASET_CONFIG="${DATASET_CONFIG:-nemotron-full-family}"
DATA_FILES="${DATA_FILES:-}"
SPLIT="${SPLIT:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-"$HOME/data/sft/blaise-sft-training-mix"}"

if [[ -n "$DATA_FILES" ]]; then
    default_name="$(basename "$DATA_FILES" .parquet)"
else
    default_name="$DATASET_CONFIG"
fi
OUTPUT_FILE="${OUTPUT_FILE:-"$OUTPUT_DIR/${default_name}.jsonl"}"

ARGS=(
    --dataset "$DATASET_ID"
    --split "$SPLIT"
    --output "$OUTPUT_FILE"
)

if [[ -n "$DATA_FILES" ]]; then
    ARGS+=(--data-files "$DATA_FILES")
else
    ARGS+=(--config "$DATASET_CONFIG")
fi

if [[ -n "${MAX_SAMPLES:-}" ]]; then
    ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "${STREAMING:-0}" == "1" ]]; then
    ARGS+=(--streaming)
fi

uv run --no-sync python tools/prepare_blaise_sft.py "${ARGS[@]}"
