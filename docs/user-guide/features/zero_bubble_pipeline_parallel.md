<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# ZeroBubble Pipeline Parallel Schedules

Megatron Core accepts ZeroBubble schedules through the unified pipeline
parallel schedule selector:

```bash
--pipeline-parallel-schedule zero_bubble
--pipeline-parallel-schedule zero_bubble_v
```

Aliases such as `zero-bubble`, `zb`, `zero-bubble-v`, and `zbv` normalize to
the canonical selector names. `auto` preserves the existing pipeline schedule
behavior.

## Validation

Both ZeroBubble selectors require pipeline parallelism:

```bash
--pipeline-model-parallel-size 2
```

or larger. They also currently require:

- `--untie-embeddings-and-output-weights`
- `--no-overlap-grad-reduce`
- no `--overlap-param-gather`

The overlap combinations are rejected until the runtime implementation
explicitly supports them.

`zero_bubble` is a non-virtual pipeline schedule. Do not set
`--num-layers-per-virtual-pipeline-stage`,
`--num-virtual-stages-per-pipeline-rank`, or a pipeline layout that creates
virtual stages with this selector.

`zero_bubble_v` derives the V-shaped virtual pipeline layout during argument
validation. It requires exactly two virtual stages per physical pipeline rank
and rejects explicit virtual-stage layout controls. The number of layers per
physical pipeline stage must be even so validation can derive:

```text
virtual_pipeline_model_parallel_size = 2
num_layers_per_virtual_pipeline_stage = layers_per_pipeline_stage / 2
```

For ZBV, the first and final model chunks live on pipeline rank 0. Loss and
per-token normalization are therefore sourced from pipeline rank 0 instead of
the last physical pipeline rank.

## Runtime

Both selectors are runtime-backed. They split activation-gradient backward work
from weight-gradient GEMMs with `WeightGradStore`, then drain the deferred weight
work from explicit W slots after gradient communication. This keeps the runtime
usable with the local Transformer implementation and with
`--no-gradient-accumulation-fusion`; if gradient accumulation fusion is enabled,
the existing fused weight-gradient extension path is used.

The current port intentionally guards features that need separate correctness
work before they can be enabled with ZB/ZBV:

- overlapped P2P communication;
- ring-exchange P2P;
- MoE expert-parallel communication overlap;
- Transformer Engine and Transformer Engine delayed WGRAD paths;
- CPU or fine-grained activation offload;
- variable sequence lengths;
- standalone MTP pipeline exchange;
- asynchronous parameter synchronization;
- partial activation checkpoint windows.

Use `auto`, `1f1b`, `interleaved_1f1b`, `gpipe_fill_drain`, `dualpipe_v`,
`zero_bubble`, or `zero_bubble_v` through the same
`--pipeline-parallel-schedule` option to switch schedules without changing
model code.

For H200 validation commands and pass criteria, see
`docs/operations/pipeline_schedule_validation.md`.
