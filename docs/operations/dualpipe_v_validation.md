# DualPipeV H200 Validation

This runbook validates DualPipeV with NCCL first. It separates static schedule
checks, distributed schedule smoke, and Megatron training measurements so the
output never reports memory or throughput unless those values were measured.
The current runtime is a conservative sequential V-topology production baseline;
these gates do not validate or claim overlapped zero-bubble timing.

## Setup

```bash
ssh instance-20260415-20260415-235136
cd ~/work/Megatron-LM-pipeline-parallels-integration
export PATH="$HOME/work/pipeline-py312/bin:$PATH"
export PYTHONPATH="$PWD"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
```

## Static Mapping Gate

```bash
python tools/dualpipe_v_schedule_verify.py \
  --mode static \
  --pipeline-model-parallel-size 8 \
  --virtual-pipeline-model-parallel-size 2 \
  --num-microbatches 16 \
  --pretty | tee /tmp/dualpipev_static_pp8.json
```

Pass criteria:

- `support_level` is `verification_harness`;
- `execution_baseline` is `sequential_v_topology`;
- `logical_pipeline_stages` is `16` for PP8;
- rank 0 is both `is_entry_rank` and `is_loss_rank`;
- rank 7 is `is_bridge_rank`;
- `runtime_measurements` is `false`.

## 2/4/8-Rank NCCL Smoke

```bash
for pp in 2 4 8; do
  torchrun --standalone --nproc-per-node="${pp}" \
    tools/dualpipe_v_schedule_verify.py \
    --mode distributed-smoke \
    --backend nccl \
    --pipeline-model-parallel-size "${pp}" \
    --virtual-pipeline-model-parallel-size 2 \
    --num-microbatches "$((2 * pp))" \
    --output "/tmp/dualpipev_nccl_smoke_pp${pp}.json"
done
```

Pass criteria for every PP size:

- the command exits 0;
- `correctness` is `passed`;
- `backend` is `nccl`;
- `world_size` equals the PP size.

## Correctness Gate

Run a longer deterministic path check on all eight H200s:

```bash
torchrun --standalone --nproc-per-node=8 \
  tools/dualpipe_v_schedule_verify.py \
  --mode distributed-smoke \
  --backend nccl \
  --pipeline-model-parallel-size 8 \
  --virtual-pipeline-model-parallel-size 2 \
  --num-microbatches 32 \
  --micro-batch-size 2 \
  --seq-length 2048 \
  --hidden-size 4096 \
  --warmup-steps 3 \
  --iterations 10 \
  --output /tmp/dualpipev_correctness_pp8.json
```

This gate reports correctness only. It intentionally omits memory and
throughput fields because `--measure` is not set.

## Memory And Throughput Gate

```bash
torchrun --standalone --nproc-per-node=8 \
  tools/dualpipe_v_schedule_verify.py \
  --mode distributed-smoke \
  --backend nccl \
  --pipeline-model-parallel-size 8 \
  --virtual-pipeline-model-parallel-size 2 \
  --num-microbatches 32 \
  --micro-batch-size 2 \
  --seq-length 4096 \
  --hidden-size 8192 \
  --warmup-steps 5 \
  --iterations 25 \
  --measure \
  --output /tmp/dualpipev_measure_pp8.json
```

Pass criteria:

- `correctness` is `passed`;
- `measurements.max_rank_peak_memory_bytes` is present;
- `measurements.tokens_per_second` is present;
- no memory or throughput number is copied into release notes without the JSON
  artifact from this command.

## NCCL-First Production Megatron Gate

After the DualPipeV runtime/config lane is merged, run the same model shape with
the baseline pipeline schedule and with DualPipeV. Use mock data first so data
loading cannot mask scheduler failures. The cross-schedule H200 smoke runbook in
`docs/operations/pipeline_schedule_validation.md` records the exact command
shape used for the May 12, 2026 all-8 validation.

When using `--transformer-impl local` without Apex or Transformer Engine, omit
`--sequence-parallel`; local torch LayerNorm does not support it. Keep
`--sequence-parallel` only for an environment with a supported fused layernorm
implementation.

Baseline:

```bash
torchrun --standalone --nproc-per-node=8 pretrain_gpt.py \
  --mock-data \
  --transformer-impl local \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --no-masked-softmax-fusion \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 8 \
  --num-virtual-stages-per-pipeline-rank 2 \
  --untie-embeddings-and-output-weights \
  --tokenizer-type NullTokenizer \
  --vocab-size 32000 \
  --num-layers 16 \
  --hidden-size 4096 \
  --ffn-hidden-size 16384 \
  --num-attention-heads 32 \
  --seq-length 4096 \
  --max-position-embeddings 4096 \
  --micro-batch-size 1 \
  --global-batch-size 32 \
  --train-samples 320 \
  --lr 1.0e-4 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --bf16 \
  --use-distributed-optimizer \
  --timing-log-level 2 \
  --log-interval 1 \
  --save-interval 1000000 \
  --eval-interval 1000000 \
  --no-load-optim \
  --no-load-rng \
  2>&1 | tee /tmp/dualpipev_baseline_pretrain.log
```

DualPipeV:

```bash
torchrun --standalone --nproc-per-node=8 pretrain_gpt.py \
  --mock-data \
  --pipeline-parallel-schedule dualpipe_v \
  --transformer-impl local \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --no-masked-softmax-fusion \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 8 \
  --num-virtual-stages-per-pipeline-rank 2 \
  --untie-embeddings-and-output-weights \
  --tokenizer-type NullTokenizer \
  --vocab-size 32000 \
  --num-layers 16 \
  --hidden-size 4096 \
  --ffn-hidden-size 16384 \
  --num-attention-heads 32 \
  --seq-length 4096 \
  --max-position-embeddings 4096 \
  --micro-batch-size 1 \
  --global-batch-size 32 \
  --train-samples 320 \
  --lr 1.0e-4 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --bf16 \
  --use-distributed-optimizer \
  --timing-log-level 2 \
  --log-interval 1 \
  --save-interval 1000000 \
  --eval-interval 1000000 \
  --no-load-optim \
  --no-load-rng \
  2>&1 | tee /tmp/dualpipev_pretrain.log
```

Production pass criteria:

- loss scale and loss trend match the baseline for the same seed and model;
- all ranks complete without NCCL timeout or unmatched send/recv;
- peak allocated CUDA memory is no worse than the accepted baseline for the
  equivalent batch and recompute policy;
- throughput is recorded after warmup under the same model, batch, dtype,
  optimizer, and NCCL environment, then reviewed as a sequential V-topology
  baseline measurement rather than an overlapped zero-bubble claim;
- if NCCLX is evaluated later, the NCCL result remains the source of truth for
  DualPipeV readiness.
