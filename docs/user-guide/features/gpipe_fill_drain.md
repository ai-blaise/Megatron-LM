# GPipe Fill-Drain Pipeline Schedule

`gpipe_fill_drain` is an explicit pipeline schedule selector for GPipe-style
fill-drain execution. It runs all forward microbatches through the pipeline
before draining backward microbatches in reverse order.

Use:

```bash
--pipeline-parallel-schedule gpipe_fill_drain
```

YAML configs can set the same value with:

```yaml
model_parallel:
  pipeline_parallel_schedule: gpipe_fill_drain
```

Hyphenated aliases such as `gpipe-fill-drain` and `fill-drain` normalize to
`gpipe_fill_drain`.

## Requirements

`gpipe_fill_drain` is runtime-backed for non-interleaved pipeline parallelism
only:

- `--pipeline-model-parallel-size` must be greater than 1.
- Virtual pipeline stages are not supported.
- Overlapped P2P communication is not supported.

Leaving the selector at `auto` preserves the existing Megatron behavior:
no pipelining for PP=1, non-interleaved 1F1B for PP>1 without virtual pipeline
stages, and interleaved 1F1B when virtual pipeline stages are configured.

## Memory and Performance

GPipe fill-drain is useful as a compatibility and correctness baseline for
schedule comparisons. It should not be treated as a throughput improvement over
Megatron's default 1F1B schedules.

Because all forward microbatches complete before backward starts, this schedule
usually keeps more activations live than 1F1B and can require higher activation
memory for the same model shape and microbatch count.

For H200 validation commands and pass criteria, see
`docs/operations/pipeline_schedule_validation.md`.
