#!/bin/bash
#SBATCH --job-name=prep_swe_rebench
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

set -e

MEGATRON_DIR=/home/divij/Megatron-LM
OUTPUT_DIR=/path/to/3fs/swe_rebench_v2_data
TOKENIZER_MODEL=deepseek-ai/DeepSeek-V3.2

mkdir -p ${OUTPUT_DIR}

cd ${MEGATRON_DIR}
python tools/prepare_swe_rebench_sft.py \
    --dataset nebius/SWE-rebench-V2 \
    --split train \
    --output-dir ${OUTPUT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --system-prompt "You are an expert software engineer. Read the repository context and issue, then produce the smallest correct patch that resolves the problem."