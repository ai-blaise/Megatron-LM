# Pipeline Schedule H200 Validation

This runbook validates the explicit pipeline schedule selector:

```bash
--pipeline-parallel-schedule auto
--pipeline-parallel-schedule 1f1b
--pipeline-parallel-schedule interleaved_1f1b
--pipeline-parallel-schedule gpipe_fill_drain
--pipeline-parallel-schedule dualpipe_v
--pipeline-parallel-schedule zero_bubble
--pipeline-parallel-schedule zero_bubble_v
```

The selector is intentionally the only switch needed to swap schedules. Schedule
specific validation still rejects unsupported combinations before training.

## Environment

```bash
ssh instance-20260415-20260415-235136
cd ~/work/Megatron-LM-pipeline-parallels-integration

export PATH="$HOME/work/pipeline-py312/bin:$PATH"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
```

The current H200 validation environment uses `--transformer-impl local`.
Without Apex or Transformer Engine, local torch LayerNorm does not support
`--sequence-parallel`, so the production smokes below omit sequence parallel.
ZeroBubble and ZeroBubble-V also reject Transformer Engine until the delayed
WGRAD split path is explicitly integrated.

## Focused Distributed Gate

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -q \
  tests/unit_tests/pipeline_parallel/test_schedules.py \
  tests/unit_tests/pipeline_parallel/test_fill_drain_schedule.py \
  tests/unit_tests/pipeline_parallel/test_dualpipe_v_runtime.py \
  tests/unit_tests/pipeline_parallel/test_zero_bubble_integration_surface.py
```

Expected result: all tests pass. The May 12, 2026 H200 run reported
`105 passed, 28 warnings`.

## Common GPT Smoke Arguments

Use these arguments for all schedule smokes below:

```bash
COMMON_ARGS="\
  --mock-data \
  --transformer-impl local \
  --no-overlap-p2p-communication \
  --no-gradient-accumulation-fusion \
  --no-masked-softmax-fusion \
  --no-persist-layer-norm \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 4 \
  --untie-embeddings-and-output-weights \
  --tokenizer-type NullTokenizer \
  --vocab-size 4096 \
  --hidden-size 512 \
  --ffn-hidden-size 2048 \
  --num-attention-heads 8 \
  --seq-length 128 \
  --max-position-embeddings 128 \
  --micro-batch-size 1 \
  --global-batch-size 8 \
  --train-iters 3 \
  --lr 1.0e-4 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --bf16 \
  --timing-log-level 1 \
  --log-interval 1 \
  --save-interval 1000000 \
  --eval-interval 1000000 \
  --eval-iters 0 \
  --no-load-optim \
  --no-load-rng"
```

Run:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  pretrain_gpt.py $COMMON_ARGS \
  --pipeline-parallel-schedule gpipe_fill_drain \
  --num-layers 8 \
  2>&1 | tee /tmp/pipeline_gpipe_pp4_tp2_train.log

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  pretrain_gpt.py $COMMON_ARGS \
  --pipeline-parallel-schedule zero_bubble \
  --num-layers 8 \
  2>&1 | tee /tmp/pipeline_zero_bubble_pp4_tp2_train.log
```

Pass criteria:

- the command exits 0;
- all three iterations report `lm loss`;
- skipped iterations and NaN iterations remain 0;
- CUDA memory summaries are present for all ranks.

## V-Shaped Schedules

DualPipeV and ZeroBubble-V use rank 0 as both the entry stage and the loss
stage. Their training logs must therefore be emitted from rank 0 rather than
from the last global rank.

DualPipeV:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  pretrain_gpt.py $COMMON_ARGS \
  --pipeline-parallel-schedule dualpipe_v \
  --num-layers 16 \
  --num-virtual-stages-per-pipeline-rank 2 \
  2>&1 | tee /tmp/pipeline_dualpipe_v_pp4_tp2_train.log
```

ZeroBubble-V derives its two virtual stages from the selector and layer count:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  pretrain_gpt.py $COMMON_ARGS \
  --pipeline-parallel-schedule zero_bubble_v \
  --num-layers 16 \
  2>&1 | tee /tmp/pipeline_zero_bubble_v_pp4_tp2_train.log
```

Expected May 12, 2026 V-schedule smoke result:

- both commands completed three iterations;
- both logged `lm loss` values `8.483570`, `8.474913`, and `8.467623`;
- skipped iterations and NaN iterations remained 0.

## Guarded Combinations

The current ZeroBubble and ZeroBubble-V runtime rejects these paths until each
has dedicated correctness work:

- overlapped P2P communication;
- ring-exchange P2P;
- MoE expert-parallel communication overlap;
- Transformer Engine and Transformer Engine delayed WGRAD;
- CPU or fine-grained activation offload;
- variable sequence lengths;
- standalone MTP pipeline exchange;
- asynchronous parameter synchronization;
- partial activation checkpoint windows.
