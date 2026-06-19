# Blaise Training Tools

This directory documents the Blaise-specific tooling under `Megatron-LM/tools`.
Run commands from the `Megatron-LM` repository root unless noted otherwise.

Most tools assume the project environment is available through `uv run --no-sync`.
Conversion tools that depend on Megatron Bridge also assume the local Bridge
checkout is available at `$HOME/Megatron-Bridge`, or that `MEGATRON_BRIDGE_ROOT`
points to the intended Bridge checkout.

```

```
```
  MODEL_PROFILE=glm4_9b_omp \
  LOAD_CKPT="/home/jon/checkpoints/glm4_9b_omp_init" \
  DATA_PROFILE=hf_blaise_mix \
  DATA_PATH="BlaiseAI/blaise-distillation-mix" \
  PARALLEL_PROFILE=single_node_8gpu \
  MICRO_BATCH_SIZE=1 \
  GLOBAL_BATCH_SIZE=32 \
  TRAIN_SAMPLES=319852 \
  LR=1.0e-5 \
  MIN_LR=1.0e-6 \
  LR_DECAY_STYLE=cosine \
  LR_DECAY_SAMPLES=319852 \
  LR_WARMUP_SAMPLES=16000 \
  SAVE_INTERVAL=2000 \
  NO_SAVE_OPTIM=1 \
  EXTRA_MEGATRON_ARGS="--ffn-hidden-size 14336 --add-qkv-bias" \
  bash examples/sft/sft.sh
```


## Model Lifecycle

The supported checkpoint lifecycle is:

```text
Hugging Face checkpoint
  -> Megatron conversion tool
  -> Megatron torch_dist checkpoint
  -> SFT launcher
  -> trained Megatron torch_dist checkpoint
  -> Hugging Face export tool
```

For the current DeepSeek-V3.2 REAP and GLM4 OMP conversion tools, low-bit
source checkpoints are used as input storage formats. The conversion step
materializes normal Megatron tensors, typically BF16, before SFT. Runtime
low-bit training is selected separately by the SFT precision profile.

## Checkpoint Conversion

### `tools/convert_blaise_deepseek_v32_reap_to_megatron.py`

Converts the Blaise DeepSeek-V3.2 REAP NVFP4 Hugging Face checkpoint into a
Megatron `torch_dist` checkpoint.

The HF checkpoint stores most linear weights as NVFP4 triples:

```text
<name>.weight_packed
<name>.weight_scale
<name>.weight_global_scale
```

The tool exposes those triples as virtual `.weight` tensors and dequantizes
them to `--dequant-dtype`, which defaults to `bf16`.

Default source:

```text
BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4
```

Preflight/metadata only:

```bash
uv run --no-sync python tools/convert_blaise_deepseek_v32_reap_to_megatron.py \
  --metadata-only
```

Check conversion mapping structure without saving:

```bash
uv run --no-sync python tools/convert_blaise_deepseek_v32_reap_to_megatron.py \
  --structure-only
```

Convert to Megatron:

```bash
uv run --no-sync python tools/convert_blaise_deepseek_v32_reap_to_megatron.py \
  --hf-model-id BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4 \
  --output "$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron"
```

Useful options:

```text
--tp / --pp / --cp / --ep / --etp
--seq-length
--decoder-first-pipeline-num-layers
--decoder-last-pipeline-num-layers
--dequant-dtype bf16|fp16|fp32
--dequant-device auto|cpu|cuda
--validate-source-key HF_KEY
--no-trust-remote-code
```

SFT handoff usually uses:

```bash
MODEL_PROFILE=deepseek_v32_reap \
PRECISION_PROFILE=deepseek_nvfp4 \
LOAD_CKPT="$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron" \
bash examples/sft/sft.sh
```

`PRECISION_PROFILE=deepseek_nvfp4` belongs to SFT runtime. It is not a converter
flag and does not cause the converter to save a persistent NVFP4 Megatron
checkpoint.

### `tools/convert_blaise_glm4_9b_omp.py`

Converts and exports the Blaise GLM-4-9B FP8 OMP checkpoint through the local
Megatron Bridge checkout.

Use this tool when working with:

```text
BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP
```

This is a single tool with subcommands:

```bash
uv run --no-sync python tools/convert_blaise_glm4_9b_omp.py preflight
uv run --no-sync python tools/convert_blaise_glm4_9b_omp.py import
uv run --no-sync python tools/convert_blaise_glm4_9b_omp.py export
```

Run these commands from `Megatron-LM`.

#### Bridge Checkout Selection

The tool must use the Megatron Bridge checkout that contains the GLM4 OMP work.
It resolves Bridge in this order:

```text
1. MEGATRON_BRIDGE_ROOT
2. $HOME/Megatron-Bridge
```

For the normal machine layout:

```text
$HOME/
  Megatron-LM/
  Megatron-Bridge/
```

no extra setup is needed. If Bridge lives somewhere else, set:

```bash
export MEGATRON_BRIDGE_ROOT=/path/to/Megatron-Bridge
```

The tool prepends `Megatron-Bridge/src` and `Megatron-LM` to `sys.path` inside
the Python process. It does not clone Bridge, install packages, or modify shell
startup files.

#### Defaults

Defaults can be overridden by CLI args or environment variables. CLI args win.

```text
HF_MODEL / --hf-model
  BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP

IMPORT_OUTPUT / --output
  ~/checkpoints/glm4_9b_omp_init

MEGATRON_CKPT / --megatron-path
  ~/checkpoints/glm4_9b_omp_trained

HF_OUTPUT_PATH or HF_PATH / --hf-output-path
  ~/models/GLM-4-9B-0414-OMP-Finetuned

TORCH_DTYPE / --torch-dtype
  bfloat16

TRUST_REMOTE_CODE / --trust-remote-code or --no-trust-remote-code
  true

USE_GPU_INITIALIZATION / --use-gpu-initialization or --no-use-gpu-initialization
  false

TP_SIZE / --tp-size
  1

PP_SIZE / --pp-size
  1
```

#### Precision Behavior

```text
HF source checkpoint: FP8 weights with scale tensors, OMP fused MLP layout
Megatron import output: normal Megatron tensors, dequantized by Bridge
SFT default: BF16 through the existing glm4_9b_omp SFT profile
HF export output: Hugging Face directory suitable for hf upload
```

The tool does not try to keep persistent FP8 Megatron checkpoints in v1. FP8 is
treated as the Hugging Face source storage format. Bridge owns dequantization
and GLM4 OMP tensor mapping.

#### Preflight

Run preflight before any conversion:

```bash
uv run --no-sync python tools/convert_blaise_glm4_9b_omp.py preflight
```

Preflight validates:

```text
AutoBridge resolves GLM4Bridge
model_type = glm4
architectures = ["Glm4ForCausalLM"]
hidden_size = 4096
intermediate_size = 14336
num_hidden_layers = 40
num_attention_heads = 32
num_key_value_heads = 2
vocab_size = 128815
quantization_config.quant_method = fp8
OMP fused gate_up_proj keys exist
FP8 scale keys exist with common `.scale`, `_scale`, or `scale_inv` naming
```

Use a non-default model or local HF directory:

```bash
uv run --no-sync python tools/convert_blaise_glm4_9b_omp.py preflight \
  --hf-model /path/to/GLM-4-9B-0414-FP8-DeepSeekV32-OMP
```

If you are deliberately testing a config variant, bypass hard failures with:

```bash
uv run --no-sync python tools/convert_blaise_glm4_9b_omp.py preflight \
  --allow-config-mismatch
```

Do not use `--allow-config-mismatch` for production conversion unless the shape
differences are intentional and understood.

#### Import HF to Megatron

Import HF FP8 OMP to Megatron:

```bash
uv run --no-sync python tools/convert_blaise_glm4_9b_omp.py import \
  --hf-model BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP \
  --output "$HOME/checkpoints/glm4_9b_omp_init"
```

The source checkpoint is FP8 with scale tensors. Bridge dequantizes the FP8
source into normal Megatron tensors for the imported checkpoint.

Equivalent environment-variable form:

```bash
HF_MODEL=BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP \
IMPORT_OUTPUT="$HOME/checkpoints/glm4_9b_omp_init" \
uv run --no-sync python tools/convert_blaise_glm4_9b_omp.py import
```

Plain import uses CPU initialization by default to avoid CUDA allocation during
checkpoint conversion. To opt into GPU initialization on a free visible device:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python tools/convert_blaise_glm4_9b_omp.py import \
  --use-gpu-initialization
```

Successful import should create an iteration checkpoint under:

```text
~/checkpoints/glm4_9b_omp_init/iter_0000000
```

The tool prints the SFT handoff command after import.

SFT handoff:

```bash
MODEL_PROFILE=glm4_9b_omp \
MEGATRON_CKPT="$HOME/checkpoints/glm4_9b_omp_init" \
DATA_ROOT="$HOME/data/my_sft_jsonl" \
bash examples/sft/sft.sh
```

`MEGATRON_CKPT` maps to:

```text
LOAD_CKPT=$MEGATRON_CKPT/iter_0000000
```

when `LOAD_CKPT` is not set.

#### Export Megatron to HF

Export a trained Megatron checkpoint back to a Hugging Face directory:

```bash
uv run --no-sync python tools/convert_blaise_glm4_9b_omp.py export \
  --megatron-path "$HOME/checkpoints/glm4_9b_omp_trained" \
  --hf-output-path "$HOME/models/GLM-4-9B-0414-OMP-Finetuned"
```

Equivalent environment-variable form:

```bash
MEGATRON_CKPT="$HOME/checkpoints/glm4_9b_omp_trained" \
HF_OUTPUT_PATH="$HOME/models/GLM-4-9B-0414-OMP-Finetuned" \
uv run --no-sync python tools/convert_blaise_glm4_9b_omp.py export
```

The export command derives critical overrides from the source HF config:

```text
num_query_groups = config.num_key_value_heads
add_qkv_bias = config.attention_bias
```

For the current GLM4 OMP config this means:

```text
num_query_groups = 2
add_qkv_bias = true
```

The exporter verifies that the output directory contains:

```text
config.json
model.safetensors.index.json
model-*.safetensors
```

It also reloads the output config and compares critical shape fields against
the source config. This guards against accidentally exporting a different GLM4
shape.

After export, push with:

```bash
hf upload your-org/your-repo "$HOME/models/GLM-4-9B-0414-OMP-Finetuned" .
```

#### Common Failures

Bridge not found:

```text
Megatron Bridge source tree was not found
```

Set `MEGATRON_BRIDGE_ROOT` or restore the `$HOME/Megatron-Bridge` checkout.

Missing Python packages:

```text
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'transformers'
```

Run from `Megatron-LM` with the project environment:

```bash
uv run --no-sync python tools/convert_blaise_glm4_9b_omp.py preflight
```

Shape/config mismatch:

```text
HF config does not match the GLM4 OMP contract
```

Confirm you are using the intended model:

```text
BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP
```

Only pass `--allow-config-mismatch` for deliberate experiments.

### `tools/export_blaise_megatron_to_hf.py`

Exports a trained Blaise DeepSeek-V3.2 REAP Megatron checkpoint back to
Hugging Face safetensors.

This script is meant to run with the same Megatron model and parallelism
arguments used by SFT. It adds export-specific arguments through Megatron's
argument parser.

Required export argument:

```text
--hf-output-path PATH
```

Common export options:

```text
--hf-source-model-id BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4
--hf-max-shard-size-gb 4
--hf-export-load-source-first
--hf-export-load-non-strict
--hf-export-trust-remote-code
```

Typical shape:

```bash
uv run --no-sync torchrun \
  --nproc_per_node "$GPUS_PER_NODE" \
  tools/export_blaise_megatron_to_hf.py \
  --load "$SAVE_CKPT" \
  --hf-output-path "$HOME/models/deepseek-v32-reap-sft-hf" \
  --hf-source-model-id BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4 \
  ...same model and parallel args used for SFT...
```

The GCP handoff wrapper `handoff/gcp-a4/scripts/export_deepseek_iter45_hf.sh`
shows one concrete export orchestration pattern.

### `tools/validate_blaise_deepseek_v32_reap_conversion.py`

Loads selected tensors from a converted DeepSeek-V3.2 REAP Megatron checkpoint
and compares them against the HF source tensors exposed by the NVFP4-aware
state source.

Default checkpoint:

```text
$LOAD_CKPT or ~/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron
```

Run default validation pairs:

```bash
uv run --no-sync python tools/validate_blaise_deepseek_v32_reap_conversion.py \
  --hf-model-id BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4 \
  --checkpoint "$HOME/checkpoints/deepseek_v32_reap_spinquant_actkv_nvfp4_megatron"
```

Validate a custom pair:

```bash
uv run --no-sync python tools/validate_blaise_deepseek_v32_reap_conversion.py \
  --checkpoint "$LOAD_CKPT" \
  --pair model.layers.0.input_layernorm.weight=decoder.layers.0.input_layernorm.weight
```

## Data Preparation

### `tools/prepare_blaise_sft.py`

Prepares `BlaiseAI/blaise-sft-training-mix` records for Megatron SFT. It
normalizes conversation records, synthesizes assistant tool calls where the
dataset uses an empty assistant placeholder followed by a tool response, and
writes JSONL records with a `messages` list.

Basic usage:

```bash
uv run --no-sync python tools/prepare_blaise_sft.py \
  --output "$HOME/data/blaise_sft/training.jsonl"
```

Useful options:

```text
--dataset BlaiseAI/blaise-sft-training-mix
--config nemotron-full-family
--data-files full_mix_all_sources.parquet
--split train
--max-samples N
--streaming
```

### `tools/prepare_blaise_sft_mix_jsonl.py`

Prepares a JSONL SFT mix from local or dataset-backed inputs. Use this when the
input data is already organized as a mix and needs to be normalized into the
SFT launcher's expected JSONL shape.

Inspect available arguments with:

```bash
uv run --no-sync python tools/prepare_blaise_sft_mix_jsonl.py --help
```

### `tools/sft_data_autopsy.py`

Inspects SFT data/tokenization behavior for debugging malformed examples,
unexpected masking, or sequence length issues.

Inspect available arguments with:

```bash
uv run --no-sync python tools/sft_data_autopsy.py --help
```

## Performance and Diagnostics

### `tools/bench_deepseek_hot_kernels.py`

Microbenchmarks DeepSeek-V3.2 SFT hot kernels at the GCP A4 launch shapes.
It sets the DSA/HISA environment defaults used by the target stack, then runs
CUDA timing loops for selected kernels.

Run from a GPU node:

```bash
uv run --no-sync python tools/bench_deepseek_hot_kernels.py --help
```

Use this for kernel-level timing work, not for checkpoint conversion.

### `tools/report_theoretical_memory.py`

Reports theoretical memory estimates for model/training configurations.

```bash
uv run --no-sync python tools/report_theoretical_memory.py --help
```

### Profile Trace Tools

The profile trace helpers process large PyTorch profiler traces:

```text
tools/profile_trace_line_reduce.py
tools/profile_trace_report.py
tools/profile_trace_deep_dive.py
tools/profile_trace_summarize_reduced.py
tools/validate_profile_artifacts.py
```

Start with:

```bash
uv run --no-sync python tools/profile_trace_report.py --help
```

Use `validate_profile_artifacts.py` before trusting generated summaries:

```bash
uv run --no-sync python tools/validate_profile_artifacts.py --help
```

### ECO and Optimizer Diagnostics

These tools are for optimizer and quantization experiments:

```text
tools/eco_master_equivalence.py
tools/eco_qm_qv_diagnostics.py
tools/eco_stability_sweep.py
tools/eco_stability_visualize.py
```

Inspect each tool's arguments before use:

```bash
uv run --no-sync python tools/eco_stability_sweep.py --help
```

### Fleet and Communication Debugging

Useful for distributed launch and NCCL/TorchComms debugging:

```text
tools/debug_nccl_fleet.py
tools/debug_torchcomms_adapter.py
tools/debug_torchcomms_ncclx.py
```

Inspect each tool's `--help` output on the target cluster before running.

## Tool Selection Quick Reference

| Task | Tool |
| --- | --- |
| Convert DeepSeek-V3.2 REAP HF -> Megatron | `tools/convert_blaise_deepseek_v32_reap_to_megatron.py` |
| Validate DeepSeek conversion | `tools/validate_blaise_deepseek_v32_reap_conversion.py` |
| Export DeepSeek trained Megatron -> HF | `tools/export_blaise_megatron_to_hf.py` |
| Convert or export GLM4 FP8 OMP | `tools/convert_blaise_glm4_9b_omp.py` |
| Prepare Blaise SFT JSONL | `tools/prepare_blaise_sft.py` |
| Inspect SFT data/tokenization | `tools/sft_data_autopsy.py` |
| Benchmark DeepSeek hot kernels | `tools/bench_deepseek_hot_kernels.py` |
| Summarize profiler traces | `tools/profile_trace_report.py` |
| Validate profiler artifacts | `tools/validate_profile_artifacts.py` |
