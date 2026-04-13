# Plan: Restructure `examples/sft/` for NVFP4 + SFT + finetune Support

## Objective

Create comprehensive SFT scripts that:
1. Support NVFP4 + precision-aware optimizer with `--sft --finetune`
2. Enable proper TensorBoard profiling and multi-run comparison
3. Work for both single GPU smoke test and multi-GPU production


@architect: Good understanding of the problem at hand

---

## Final File Structure

```
examples/sft/
├── run_sft.sh.backup                     # Divij's original (NO CHANGES)
├── run_sft_llama_nvfp4_minimal.sh        # Single GPU smoke test (NEW)
└── run_sft_deepseek_nvfp4.sh            # Multi GPU production (NEW)
```

---

## Path Handling (ALL Scripts)

**Principle:** NO hardcoded absolute paths like `/home/divij/` or `/path/to/3fs/`. @architect: important for reproducability

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${SCRIPT_DIR}/../.."
cd ${MEGATRON_DIR}

CHECKPOINT_PATH=${1:-"$HOME/checkpoints/sft_nvfp4"}
TENSORBOARD_LOGS_PATH=${2:-"$HOME/tensorboard_logs/sft_nvfp4"}
```

---

## Script 1: `run_sft_llama_nvfp4_minimal.sh` — Single GPU Smoke Test

### Purpose
Quick verification that NVFP4 + precision-aware optimizer + SFT + finetune works on single GPU.
Also validates that SFTDataset can load parquet files with "conversations" column (from sft-dataset-parquet-and-conversations-support plan).

### Configuration

| Aspect | Value |
|--------|-------|
| GPUs | 1 (`--nproc_per_node=1`) |
| Launch | Direct `torchrun` (NO SLURM) |
| Model | LLaMA 8B full config |
| Flags | `--sft --finetune` |
| Precision | NVFP4 + precision-aware optimizer |
| Data | Test parquet file (validates SFTDataset parquet + conversations loading) |
| Profiling | TensorBoard only |

### Full Script

```bash
#!/bin/bash
# Single GPU NVFP4 + SFT + finetune smoke test
# No SLURM - direct torchrun

set -e

# ======================
# Path Setup
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${SCRIPT_DIR}/../.."
cd ${MEGATRON_DIR}

CHECKPOINT_PATH=${1:-"$HOME/checkpoints/sft_llama_nvfp4_minimal"}
TENSORBOARD_LOGS_PATH=${2:-"$HOME/tensorboard_logs/sft_llama_nvfp4_minimal"}

mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$TENSORBOARD_LOGS_PATH"

# ======================
# Distributed Setup
# ======================
GPUS_PER_NODE=1

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
)

# ======================
# Model Args (LLaMA 8B)
# ======================
MODEL_ARGS=(
    --use-mcore-models
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --kv-channels 128
    --seq-length 8192
    --max-position-embeddings 8192
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --init-method-std 0.0134
)

# ======================
# Training Args
# ======================
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1
    --train-iters 5
    --seq-length 8192
    --max-position-embeddings 8192
    --bf16
    --log-interval 10
)

# ======================
# NVFP4 + Precision-Aware
# ======================
DTYPE_ARGS=(
    --fp4-format e2m1
    --fp4-recipe nvfp4
    --fp4-param-gather
)

PRECISION_AWARE_ARGS=(
    --use-precision-aware-optimizer
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
)

# ======================
# SFT + finetune
# ======================
SFT_ARGS=(
    --sft
    --finetune
    --sft-tokenizer-prompt-format nemotron-h-aligned
)

# ======================
# Test Parquet Data (for SFTDataset smoke test)
# Uses the parquet file created by sft-dataset-parquet-and-conversations-support plan
# This validates end-to-end that parquet loading with "conversations" column works
# ======================
TEST_PARQUET_PATH="${MEGATRON_DIR}/tests/unit_tests/test_data/sft_test_conversations.parquet"

DATA_ARGS=(
    --data-path $TEST_PARQUET_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model deepseek-ai/DeepSeek-V3.2
    --vocab-size 128256
)

# ======================
# TensorBoard
# ======================
TENSORBOARD_ARGS=(
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --log-throughput
    --log-memory-to-tensorboard
    --log-l2-norm-grad-to-tensorboard
    --tensorboard-log-interval 10
)

# ======================
# Checkpointing
# ======================
CKPT_ARGS=(
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
)

# ======================
# LAUNCH
# ======================
torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${PRECISION_AWARE_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TENSORBOARD_ARGS[@]} \
    ${CKPT_ARGS[@]}
```

---

## Script 2: `run_sft_deepseek_nvfp4.sh` — Multi GPU Production Script

### Purpose
Full DeepSeek-V3.2 MoE training with NVFP4 + precision-aware optimizer + DSA (DeepSeek Sparse Attention) + SFT + finetune.

> **Why DSA for DeepSeek but NOT for LLaMA 8B test:** DSA (DeepSeek Sparse Attention) is a specialized attention mechanism for DeepSeek models. The simple LLaMA 8B smoke test is just to verify NVFP4 runs at all — adding DSA complexity would conflate two tests. DSA builds ON TOP of MLA (Multi-Latent Attention), not a replacement.

### Configuration

| Aspect | Value |
|--------|-------|
| GPUs | 8 per node |
| Launch | `srun --mpi=pmix torchrun ...` |
| Model | DeepSeek-V3.2 MoE (61 layers, 128 experts) |
| Flags | `--sft --finetune` |
| Precision | NVFP4 + precision-aware optimizer |
| Attention | MLA + DSA (DeepSeek Sparse Attention) |
| Data | Real data |
| Profiling | TensorBoard + NVTX/nsys |

### Full Script

```bash
#!/bin/bash
# Multi GPU DeepSeek-V3.2 MoE + NVFP4 + SFT + finetune
# SLURM with srun

#SBATCH --job-name=deepseek_nvfp4_sft
#SBATCH --nodes=${NNODES:-1}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

set -e

# ======================
# Environment
# ======================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET=TCP
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=3600
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=False

# ======================
# Path Setup
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${SCRIPT_DIR}/../.."
cd ${MEGATRON_DIR}

CHECKPOINT_PATH=${1:-"$HOME/checkpoints/sft_deepseek_nvfp4"}
TENSORBOARD_LOGS_PATH=${2:-"$HOME/tensorboard_logs/sft_deepseek_nvfp4"}

mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$TENSORBOARD_LOGS_PATH"

# ======================
# Distributed Setup
# ======================
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST 2>/dev/null | head -1)
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ======================
# Model Args (DeepSeek-V3.2 MoE)
# ======================
MODEL_ARGS=(
    --use-mcore-models
    --num-layers 61
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --kv-channels 128
    --seq-length 4096
    --max-position-embeddings 163840
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --normalization RMSNorm
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

# ======================
# MLA Args (Multi-Latent Attention) — REQUIRED for DSA
# DSA builds ON TOP of MLA, not a replacement
# ======================
MLA_ARGS=(
    --multi-latent-attention
    --kv-lora-rank 512
    --q-lora-rank 1536
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
)

# ======================
# DSA Args (DeepSeek Sparse Attention) — DeepSeek-V3.2 specific
# Source: DeepSeek-V3.2-Exp GitHub + tests/functional_tests/test_cases/gpt/gpt3_mcore_te_tp2_pp2_dsa/model_config.yaml
# ======================
DSA_ARGS=(
    --experimental-attention-variant dsa
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk 2048
    --dsa-indexer-loss-coeff 0.01
)

# ======================
# MoE Args
# ======================
MODEL_ARGS+=(
    --num-experts 128
    --moe-router-topk 8
    --moe-layer-freq $(python3 -c "print('[' + ','.join(['0']*3 + ['1']*58) + ']')")
    --moe-ffn-hidden-size 2048
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-aux-loss-coeff 1e-4
    --moe-shared-expert-intermediate-size 2048
    --moe-token-dispatcher-type alltoall
)

# ======================
# Training Args
# ======================
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 32
    --train-samples 32000000
    --lr-decay-samples 31968645
    --lr-warmup-samples 31348
    --lr 5.0e-6
    --min-lr 1.0e-7
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.010
    --log-interval 10
)

# ======================
# Parallelism
# ======================
TP=8
EP=1
PP=1
CP=1

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP
    --expert-tensor-parallel-size 1
    --expert-model-parallel-size $EP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --sequence-parallel
)

# ======================
# NVFP4 + Precision-Aware
# ======================
DTYPE_ARGS=(
    --fp4-format e2m1
    --fp4-recipe nvfp4
    --fp4-param-gather
)

PRECISION_AWARE_ARGS=(
    --use-precision-aware-optimizer
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
)

# ======================
# Optimizer
# ======================
TRAINING_ARGS+=(
    --use-distributed-optimizer
    --no-gradient-accumulation-fusion
    --reset-position-ids
    --reset-attention-mask
    --eod-mask-loss
)

# ======================
# SFT + finetune
# ======================
SFT_ARGS=(
    --sft
    --finetune
    --sft-tokenizer-prompt-format nemotron-h-aligned
)

# ======================
# Tokenizer
# ======================
TOKENIZER_MODEL=${3:-"deepseek-ai/DeepSeek-V3.2"}

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
)

# ======================
# Data (Real)
# ======================
DATA_PATH=${HOME}/data/sft/swe_rebench_v2_data

DATA_ARGS=(
    --data-path $DATA_PATH
    --split 100,0,0
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --num-workers 1
    --vocab-size 128256
)

# ======================
# TensorBoard + Profiling
# ======================
TENSORBOARD_ARGS=(
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --log-throughput
    --log-memory-to-tensorboard
    --log-l2-norm-grad-to-tensorboard
    --tensorboard-log-interval 10
)

PROFILING_ARGS=(
    --profile
    --profile-step-start 4
    --profile-step-end 6
)

# ======================
# Checkpointing
# ======================
CKPT_ARGS=(
    --save-interval 500
    --eval-interval 100
    --eval-iters 10
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --distributed-timeout-minutes 60
    --ckpt-format torch_dist
    --auto-detect-ckpt-format
)

# ======================
# LAUNCH (SLURM + torchrun)
# ======================
srun --mpi=pmix -l \
    torchrun ${DISTRIBUTED_ARGS[@]} \
        pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${MLA_ARGS[@]} \
        ${DSA_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${DTYPE_ARGS[@]} \
        ${PRECISION_AWARE_ARGS[@]} \
        ${SFT_ARGS[@]} \
        ${TOKENIZER_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${TENSORBOARD_ARGS[@]} \
        ${PROFILING_ARGS[@]} \
        ${CKPT_ARGS[@]}
```

---

## Multi-Run Comparison Workflow

To compare different quantization formats (FP4 vs FP8):

```bash
# Run 1: NVFP4
./run_sft_llama_nvfp4_minimal.sh \
    --tensorboard-dir $HOME/tensorboard_logs/sft_nvfp4

# Run 2: FP8 (create similar script with --fp8-format hybrid)
./run_sft_llama_fp8.sh \
    --tensorboard-dir $HOME/tensorboard_logs/sft_fp8

# Compare in TensorBoard:
tensorboard --logdir=$HOME/tensorboard_logs
# In UI: Select multiple runs to overlay loss curves, throughput, memory
```

---

## gcloud SSH TensorBoard Access

```bash
# ===========================================
# TensorBoard Remote Access (for gcloud SSH):
# ===========================================
# 1. Local machine: gcloud compute ssh user@instance -- -L 6006:localhost:6006
# 2. Remote: tensorboard --logdir=$TENSORBOARD_LOGS_PATH --host localhost
# 3. Local browser: http://localhost:6006
```

---

## Files to Create

| File | Action |
|------|--------|
| `examples/sft/run_sft.sh.backup` | Copy current run_sft.sh |
| `examples/sft/run_sft_llama_nvfp4_minimal.sh` | Create new (single GPU) |
| `examples/sft/run_sft_deepseek_nvfp4.sh` | Create new (multi GPU) |

---

## Key Flags Summary

| Flag | Minimal (LLaMA 8B) | DeepSeek V3.2 |
|------|-------------------|----------------|
| `--sft` | ✅ | ✅ |
| `--finetune` | ✅ | ✅ |
| `--fp4-format e2m1` | ✅ | ✅ |
| `--fp4-recipe nvfp4` | ✅ | ✅ |
| `--use-precision-aware-optimizer` | ✅ | ✅ |
| **MLA (Multi-Latent Attention)** | ❌ | ✅ |
| `--multi-latent-attention` | - | ✅ |
| `--q-lora-rank` | - | ✅ (1536) |
| `--kv-lora-rank` | - | ✅ (512) |
| `--qk-head-dim` | - | ✅ (128) |
| **DSA (DeepSeek Sparse Attention)** | ❌ | ✅ |
| `--experimental-attention-variant dsa` | - | ✅ |
| `--dsa-indexer-n-heads` | - | ✅ (64) |
| `--dsa-indexer-head-dim` | - | ✅ (128) |
| `--dsa-indexer-topk` | - | ✅ (2048) |
| `--dsa-indexer-loss-coeff` | - | ✅ (0.01) |
| `--profile` (NVTX) | ❌ | ✅ |
| `--use-pytorch-profiler` | ❌ | ❌ |
| TensorBoard | ✅ | ✅ |
| SLURM/srun | ❌ | ✅ |

---

## Dependencies

- TE >= 2.7.0.dev0 (for NVFP4 support)
- B200 (Blackwell) GPU for NVFP4 scripts
- `datasets` library (SFTDataset parquet support - already done)
- DSA support in Megatron core (`megatron/core/transformer/experimental_attention_variant/dsa.py`)

---

## Out of Scope

- wandb integration (for future consideration)
- Modifying `run_sft.sh.backup`
- PyTorch profiler (Perfetto)
- SFTDataset changes - already implemented
