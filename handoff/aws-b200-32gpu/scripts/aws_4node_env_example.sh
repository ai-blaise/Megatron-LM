# Source or copy this as a starting point for AWS.
# Fill in AWS-specific host/network/storage values before use.

export NNODES=4
export GPUS_PER_NODE=8
export MASTER_ADDR="<rank0-private-ip>"
export MASTER_PORT="${MASTER_PORT:-29673}"

# Safest first 32-GPU shape: preserve current TP/PP/EP and add DP=2.
export TP=4
export PP=4
export CP=1
export EP=4
export ETP=1
export ENABLE_VPP=1
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=32
export SEQ_LENGTH=16384
export DSA_INDEXER_TOPK=512

# Preserve current quality-sensitive stack.
export USE_STREAMBP=1
export INDEXCACHE=1
export DSA_INDEXCACHE_HISA=1
export DSA_INDEXCACHE_QUANTIZATION=nvfp4_e2m1_ue8m0
export SPINQUANT=1
export USE_HIGGS=1
export TURBOQUANT=0
export NVFP4_ACTIVATION_ECO=1
export FLASH_ADAMW_ECO=1

# Current MoE fast path. If DeepEP times out, isolate with alltoall.
export MOE_TOKEN_DISPATCHER_TYPE=flex
export MOE_FLEX_DISPATCHER_BACKEND=deepep
export MEGATRON_DEEPEP_COMPACT_LOCAL_PERMUTE=1

# AWS/EFA values must be set for the actual instance image.
# Do not blindly use the old GCP gpu/mlx5 defaults.
export NCCL_SOCKET_IFNAME="<aws-efa-or-primary-interface>"
export NCCL_IB_HCA="<aws-hca-list-if-needed>"
# export FI_PROVIDER=efa
# export FI_EFA_USE_DEVICE_RDMA=1

export LOAD_CKPT="<shared-or-prestaged>/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron"
export SAVE_CKPT="<shared-or-local>/sft_deepseek_v32_reap_spinquant_actkv_nvfp4"
export DATA_PATH="<shared-or-prestaged>/blaise-sft-training-mix.jsonl"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton/deepseek_v32_reap_sft}"
export DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-$HOME/.cache/deep_gemm/deepseek_v32_reap_sft}"

