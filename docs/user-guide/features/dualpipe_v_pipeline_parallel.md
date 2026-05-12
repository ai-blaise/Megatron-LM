# DualPipeV Pipeline Parallelism

DualPipeV is the V-shaped form of DeepSeek DualPipe. It uses two virtual
pipeline stages per physical pipeline rank:

- phase 0 maps logical stages `0..PP-1` from rank 0 to rank `PP-1`;
- phase 1 maps logical stages `2*PP-1..PP` from rank 0 to rank `PP-1`;
- rank `PP-1` bridges phase 0 into phase 1;
- rank 0 owns both the input side and the loss/output side.

The current runtime is a conservative sequential V-topology production
baseline. It preserves the V-shaped rank mapping and NCCL P2P data path, but it
does not claim overlapped zero-bubble timing.

This branch prioritizes DualPipeV first. The validation harness in
`tools/dualpipe_v_schedule_verify.py` is DualPipeV-specific and does not claim
support for other pipeline schedules.

## Configuration Contract

DualPipeV requires:

- `--pipeline-model-parallel-size` greater than 1;
- exactly two virtual stages, for example `--num-virtual-stages-per-pipeline-rank 2`;
- `--untie-embeddings-and-output-weights`;
- at least `2 * pipeline_model_parallel_size` microbatches;
- NCCL for production GPU validation.

The untied-embedding requirement is intentional. Megatron's default tied
embedding/output-weight path assumes the loss stage is on the last physical
pipeline rank. DualPipeV returns the loss stage to rank 0, so tied embeddings
are rejected until the embedding synchronization path is made DualPipeV-aware.

The physical rank owns two local model chunks. The first chunk executes in
forward rank order. The second chunk executes the reverse logical half of the
model so the pipeline returns to rank 0 for loss and output handling.

## Static Mapping

Run this before launching distributed work:

```bash
cd ~/work/Megatron-LM-pipeline-parallels
python tools/dualpipe_v_schedule_verify.py \
  --mode static \
  --pipeline-model-parallel-size 8 \
  --virtual-pipeline-model-parallel-size 2 \
  --num-microbatches 16 \
  --pretty
```

The JSON output reports the logical stage on each rank, phase-neighbor ranks,
entry/bridge/loss ranks, the sequential V-topology baseline label, and the
eight DualPipeV step counts from the reference schedule. It does not include
runtime measurements.

## Distributed Smoke

Use torchrun on the H200 VM to verify NCCL process-group setup, sequential
V-shaped point-to-point movement, and deterministic forward/backward parity:

```bash
cd ~/work/Megatron-LM-pipeline-parallels
source ~/streambp-py312/bin/activate
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

torchrun --standalone --nproc-per-node=2 \
  tools/dualpipe_v_schedule_verify.py \
  --mode distributed-smoke \
  --backend nccl \
  --pipeline-model-parallel-size 2 \
  --virtual-pipeline-model-parallel-size 2 \
  --num-microbatches 4 \
  --output /tmp/dualpipev_nccl_smoke_pp2.json

torchrun --standalone --nproc-per-node=4 \
  tools/dualpipe_v_schedule_verify.py \
  --mode distributed-smoke \
  --backend nccl \
  --pipeline-model-parallel-size 4 \
  --virtual-pipeline-model-parallel-size 2 \
  --num-microbatches 8 \
  --output /tmp/dualpipev_nccl_smoke_pp4.json

torchrun --standalone --nproc-per-node=8 \
  tools/dualpipe_v_schedule_verify.py \
  --mode distributed-smoke \
  --backend nccl \
  --pipeline-model-parallel-size 8 \
  --virtual-pipeline-model-parallel-size 2 \
  --num-microbatches 16 \
  --output /tmp/dualpipev_nccl_smoke_pp8.json
```

Add `--measure` only when you want the harness to report measured peak memory
and synthetic tokens per second:

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

Measurements from this harness are schedule-path measurements, not model
training throughput. Use the production runbook for Megatron model validation.
The cross-schedule H200 smoke runbook is
`docs/operations/pipeline_schedule_validation.md`.
