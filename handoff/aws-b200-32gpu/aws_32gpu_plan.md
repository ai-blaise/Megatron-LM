# AWS 32-GPU Plan

The 32-GPU move changes the available parallelism choices. It does not
automatically fix per-rank memory if we keep the same tensor/pipeline/expert
shape, but it gives us more options.

## Candidate A: Preserve Current Model-Parallel Shape, Add DP

```bash
NNODES=4
GPUS_PER_NODE=8
TP=4
PP=4
CP=1
EP=4
ETP=1
```

World size: 32.

Dense model parallel size: `TP*PP*CP = 16`.

Data parallel size: `32/16 = 2`.

Expert model pipeline parallel size: `ETP*EP*PP = 16`.

Expert DP: `32/16 = 2`.

This is the safest first AWS shape because it keeps the same per-rank model
layout and most code paths. It should improve token throughput through DP=2, but
it will not halve activation memory per rank. If the 2-node blocker was purely
per-rank memory in DSA/MoE replay, this may still fail. If the blocker was
pipeline scheduling plus insufficient global capacity, this may be enough.

Suggested first AWS test:

```bash
TP=4 PP=4 CP=1 EP=4 ETP=1 \
MICRO_BATCH_SIZE=4 GLOBAL_BATCH_SIZE=32 \
SEQ_LENGTH=16384 DSA_INDEXER_TOPK=512
```

With DP=2 this gives grad accumulation:

`GBS / (MBS * DP) = 32 / (4 * 2) = 4`.

That matches the recent MBS=4/GBS=16 two-node accumulation count while using
twice as many GPUs.

## Candidate B: More Pipeline Stages For Memory

```bash
NNODES=4
GPUS_PER_NODE=8
TP=4
PP=8
CP=1
EP=4
ETP=1
```

World size: 32.

Dense model parallel size: `TP*PP*CP = 32`.

Data parallel size: 1.

Expert DP: `32/(1*4*8) = 1`.

This should reduce layers per pipeline rank and may be the better memory shape
if Candidate A still fails. The tradeoff is more pipeline bubbles and the need
for a new PP/VPP layout. The current launcher only has a hand-tuned PP=4 VPP
layout.

A non-VPP first layout for PP=8 could start around:

```bash
PIPELINE_MODEL_PARALLEL_LAYOUT='Et*8|t*8|t*8|t*8|t*8|t*7|t*7|t*7L'
```

This is not validated and should be treated as a starting point only.

For VPP with PP=8, derive a balanced layout explicitly rather than letting the
old PP=4 layout leak through. The final stage contains loss/LM-head work and may
need fewer transformer layers.

## Candidate C: Context Parallelism

The current runner rejects:

```bash
USE_STREAMBP=1
CP!=1
```

So CP is not a quick lever unless StreamBP is disabled or ported to CP. Given
the current memory pressure, turning StreamBP off has repeatedly been unsafe.
Treat CP as a future engineering path, not the first AWS run.

## Candidate D: Higher TP

Higher TP may reduce some per-rank tensor sizes but changes a lot of collective
and sequence-parallel behavior. It also changes StreamBP local sequence length.
Use this only after Candidate A/B data.

## Recommended First AWS Sequence

1. Environment validation on one node.
2. Extension prebuild on all nodes.
3. Single-node import/JIT smoke.
4. 4-node distributed smoke with tiny train samples, no profiling.
5. Candidate A: `TP=4 PP=4 DP=2 EP=4`, `MBS=4 GBS=32`, seq 16k, topk 512.
6. If Candidate A reaches step 1 but is slow, tune GBS upward.
7. If Candidate A OOMs in the same StreamBP/DSA/MoE path, try Candidate B
   `TP=4 PP=8 DP=1 EP=4`.
8. If DeepEP timeout recurs, isolate with:
   - `MOE_TOKEN_DISPATCHER_TYPE=alltoall`
   - then flex/deepep with `MEGATRON_DEEPEP_COMPACT_LOCAL_PERMUTE=0`

## Quality-Sensitive Settings

Avoid changing these unless explicitly making a quality tradeoff:

- DSA enabled.
- DSA indexer loss enabled.
- Indexer top-k not reduced below 512 without a reason.
- HISA/IndexCache enabled.
- SpinQuant/ActKV/NVFP4 stack preserved.
- G1 gates and GatedNorm preserved.

