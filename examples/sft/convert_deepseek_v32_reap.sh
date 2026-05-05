#!/bin/bash
# Convert the Blaise DeepSeek-V3.2 REAP NVFP4 HF checkpoint to Megatron torch_dist.

#SBATCH --job-name=deepseek_v32_reap_convert
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${MEGATRON_DIR:-"${SCRIPT_DIR}/../.."}"
cd "$MEGATRON_DIR"

MODEL_ID="${MODEL_ID:-BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1}"
LOAD_CKPT="${LOAD_CKPT:-"$HOME/checkpoints/deepseek_v32_reap_megatron"}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${SLURM_NNODES:-${NNODES:-2}}"
NODE_RANK="${SLURM_NODEID:-${NODE_RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-}"
if [[ -z "$MASTER_ADDR" && -n "${SLURM_JOB_NODELIST:-}" ]]; then
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
fi
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29501}"

TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-1}"
EP="${EP:-4}"
ETP="${ETP:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-32768}"
DECODER_FIRST_PIPELINE_NUM_LAYERS="${DECODER_FIRST_PIPELINE_NUM_LAYERS:-31}"

CMD=(
    uv run --no-sync torchrun
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
    tools/convert_blaise_deepseek_v32_reap_to_megatron.py
    --hf-model-id "$MODEL_ID"
    --output "$LOAD_CKPT"
    --seq-length "$SEQ_LENGTH"
    --tp "$TP"
    --pp "$PP"
    --cp "$CP"
    --ep "$EP"
    --etp "$ETP"
    --decoder-first-pipeline-num-layers "$DECODER_FIRST_PIPELINE_NUM_LAYERS"
)

if [[ "${METADATA_ONLY:-0}" == "1" ]]; then
    CMD+=(--metadata-only)
fi
if [[ -n "${VALIDATE_SOURCE_KEY:-}" ]]; then
    CMD+=(--validate-source-key "$VALIDATE_SOURCE_KEY")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" && "${USE_SRUN:-1}" == "1" ]]; then
    srun --mpi=pmix -l "${CMD[@]}"
else
    "${CMD[@]}"
fi
