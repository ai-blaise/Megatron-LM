#!/bin/bash
#SBATCH --job-name=deepseekv3_sft
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

set -e

# RoCE v1 network config
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET=TCP
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=3600
export NCCL_IB_TC=46
export NCCL_IB_GID_INDEX=3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=False
export UCX_NET_DEVICES=eth0

# Cluster setup
GPUS_PER_NODE=8
NNODES=${SLURM_NNODES:-2}
NODE_RANK=${SLURM_NODEID:-0}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST 2>/dev/null | head -1)
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=29500

# Parallelism: TP=8, EP=1, PP=1, CP=1, DP=2
TP=8 EP=1 PP=1 CP=1

# Paths
MEGATRON_DIR=/home/divij/Megatron-LM
LOAD_CKPT=/path/to/3fs/checkpoints/deepseek-v3.2-reap-345b/base  # pretrained checkpoint to load
SAVE_CKPT=/path/to/3fs/checkpoints/deepseek-v3.2-reap-345b/sft    # fine-tuned output
DATA_PATH=/path/to/3fs/swe_rebench_v2_data/nebius__SWE-rebench-V2__train.messages.jsonl
TOKENIZER_MODEL=deepseek-ai/DeepSeek-V3.2

mkdir -p ${SAVE_CKPT}

# Training params
MBS=1 GBS=32 SEQ_LEN=4096 LR=5.0e-6 TRAIN_SAMPLES=32000
MOE_LAYER_FREQ=$(python3 - << 'EOF'
print("[" + ",".join(["0"]*3 + ["1"]*58) + "]")
EOF
)

cd ${MEGATRON_DIR}

srun --mpi=pmix -l \
    torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    pretrain_gpt.py \
    --tensor-model-parallel-size ${TP} \
    --expert-tensor-parallel-size 1 \
    --expert-model-parallel-size ${EP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --sequence-parallel \
    --finetune \
    --bf16 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --sequence-length ${SEQ_LEN} \
    --max-position-embeddings 163840 \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-decay-samples ${TRAIN_SAMPLES} \
    --lr-warmup-samples 100 \
    --lr ${LR} \
    --lr-decay-style cosine \
    --min-lr 1.0e-7 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.010 \
    --use-distributed-optimizer \
    --no-gradient-accumulation-fusion \
    --reset-position-ids \
    --reset-attention-mask \
    --eod-mask-loss \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --num-layers 61 \
    --hidden-size 7168 \
    --ffn-hidden-size 18432 \
    --num-attention-heads 128 \
    --kv-channels 128 \
    --multi-latent-attention \
    --kv-lora-rank 512 \
    --q-lora-rank 1536 \
    --qk-head-dim 128 \
    --qk-layernorm \
    --qk-pos-emb-head-dim 64 \
    --v-head-dim 128 \
    --num-experts 128 \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-ffn-hidden-size 2048 \
    --moe-router-topk 8 \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --moe-router-pre-softmax \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-bias-update-rate 1e-3 \
    --moe-aux-loss-coeff 1e-4 \
    --moe-shared-expert-intermediate-size 2048 \
    --moe-token-dispatcher-type alltoall \
    --split 100,0,0 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-path ${DATA_PATH} \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
    --save ${SAVE_CKPT} \
    --load ${LOAD_CKPT} \
    --distributed-timeout-minutes 60 \
    --auto-detect-ckpt-format "$@"