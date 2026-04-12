# Plan for NVFP4 Support in Precision-Aware Optimizer

## Problem

**What:** When `--use-precision-aware-optimizer` is enabled with `--fp4-format` (NVFP4), `is_float8tensor` checks return False for NVFP4 tensors, causing crashes.
**Why:** NVFP4 falls through to `model_param.view(-1)` which fails on QuantizedTensor.
**Source:** `megatron/core/optimizer/distrib_optimizer.py:2601`

---

## Phase 1: Code Changes

### 1. `megatron/core/optimizer/distrib_optimizer.py`

| Line | Change | Citation |
|------|--------|----------|
| **52** | Add import: `from ..fp4_utils import is_nvfp4tensor` | `megatron/core/fp4_utils.py:46` |
| **365** | `if (is_float8tensor(model_param) or is_nvfp4tensor(model_param)) and config.fp8_recipe != "delayed":` | `megatron/core/optimizer/distrib_optimizer.py:365` |
| **385** | `if is_float8tensor(model_param) or is_nvfp4tensor(model_param):` | `megatron/core/optimizer/distrib_optimizer.py:385` |
| **2154** | `if is_float8tensor(...) or is_nvfp4tensor(...):` | `megatron/core/optimizer/distrib_optimizer.py:2154` |
| **2176** | `if not (is_float8tensor(...) or is_nvfp4tensor(...)):` | `megatron/core/optimizer/distrib_optimizer.py:2176` |
| **2376** | `if is_float8tensor(param) or is_nvfp4tensor(param):` | `megatron/core/optimizer/distrib_optimizer.py:2376` |
| **2391** | `if is_float8tensor(model_param) or is_nvfp4tensor(model_param):` | `megatron/core/optimizer/distrib_optimizer.py:2391` |
| **2489** | `if is_float8tensor(model_param) or is_nvfp4tensor(model_param): continue` | `megatron/core/optimizer/distrib_optimizer.py:2489` |
| **2601** | **`if is_float8tensor(model_param) or is_nvfp4tensor(model_param): continue`** | `megatron/core/optimizer/distrib_optimizer.py:2601` — CRITICAL |

### 2. `megatron/core/distributed/param_and_grad_buffer.py`

| Line | Change | Citation |
|------|--------|----------|
| **22** | Add import: `from ..fp4_utils import is_nvfp4tensor` | `megatron/core/fp4_utils.py:46` |
| **362** | `if is_float8tensor(param) or is_nvfp4tensor(param):` | `megatron/core/distributed/param_and_grad_buffer.py:362` |
| **830** | `if is_float8tensor(param) or is_nvfp4tensor(param):` | `megatron/core/distributed/param_and_grad_buffer.py:830` |

---

## Phase 2: New Unit Tests

### 2.1 `tests/unit_tests/test_fp4_param.py` (NEW)
Mirrors `test_fp8_param.py` structure at `tests/unit_tests/test_fp8_param.py:68-450`

| Component | Description | Citation |
|-----------|-------------|----------|
| `TestFP4Param` class | Main test class | `test_fp8_param.py:68` |
| `setup_method` | Set seq_length=512, micro_batch_size=2 | `test_fp8_param.py:68-71` |
| `teardown_method` | Clean up CUDA graphs | `test_fp8_param.py:74-81` |
| `model_provider` | Creates GPTModel with TE FP4 spec | `test_fp8_param.py:83-106` |
| `create_test_args` | Build test args: `fp4='e2m1'`, `fp4_recipe='nvfp4'`, `fp4_param=True` | `test_fp8_param.py:108-161` |
| `_run_test_helper` | Core runner: forward/backward/optimizer step | `test_fp8_param.py:174-310` |
| `test_nvfp4_scaling` | Test basic NVFP4 | NEW |
| `test_nvfp4_with_precision_aware_optimizer` | **KEY TEST** - NVFP4 + precision-aware | NEW |

Key args:
```python
args.fp4 = "e2m1"  # NOT args.fp8
args.fp4_recipe = "nvfp4"
args.fp4_param = True
args.fp8 = None
```

### 2.2 `tests/unit_tests/test_fp4_utils.py` (NEW)
Mirrors `test_fp8_utils.py` at `tests/unit_tests/test_fp8_utils.py:1-132`

| Component | Description |
|-----------|-------------|
| `TestFP4Padding` class | Test FP4 padding utilities |
| `test_is_nvfp4tensor` | Test is_nvfp4tensor() detection |

### 2.3 `tests/unit_tests/dist_checkpointing/test_fp4.py` (NEW)
Mirrors `test_fp8.py` at `tests/unit_tests/dist_checkpointing/test_fp8.py:1-117`

| Component | Description |
|-----------|-------------|
| `TestFP4` class | Test FP4 checkpoint save/load |
| `test_simple_broadcast` | Broadcast FP4 tensors |
| `test_fp4_save_load` | Save/load FP4 tensors |

### 2.4 Extend `tests/unit_tests/test_optimizer.py`
Add FP4 to precision-aware test at `test_optimizer.py:416-423`:

```python
@pytest.mark.parametrize("precision", ['bf16', 'fp8', 'fp4'])  # Add fp4
```

---

## Phase 3: New Example Scripts

### 3.1 Pretrain: `examples/llama/train_llama3_8b_b200_nvfp4.sh` (NEW)
Based on `examples/llama/train_llama3_8b_h100_fp8.sh` structure

**Key flags difference from FP8 script:**

| Aspect | FP8 Script | NVFP4 Script |
|--------|-----------|--------------|
| Precision format | `--fp8-format hybrid` | `--fp4-format e2m1` |
| Param storage | `--fp8-param-gather` | `--fp4-param-gather` |
| Optimizer | (not specified) | `--use-precision-aware-optimizer` |
| Optimizer states | (defaults) | `--exp-avg-dtype bf16 --exp-avg-sq-dtype bf16` |
| TE version | varies | >= 2.7.0.dev0 |
| GPU | H100 | B200 (Blackwell) |

**Citation:** `examples/llama/train_llama3_8b_h100_fp8.sh:104-113` — FP8 script structure

### 3.2 SFT: `examples/llama/finetune_llama3_8b_b200_nvfp4.sh` (NEW)
Based on pretrain script structure with SFT-specific additions

**Key flags difference from pretrain script:**

| Aspect | Pretrain NVFP4 | SFT NVFP4 |
|--------|----------------|-----------|
| Training mode | (pretrain default) | `--sft` |
| Dataset | GPT dataset | SFT dataset with `--sft-mock-dataset-config-json` |
| Prompt format | N/A | `--sft-tokenizer-prompt-format nemotron-h-aligned` |

**Key SFT args:**
```bash
--sft \
--sft-tokenizer-prompt-format nemotron-h-aligned \
--sft-mock-dataset-config-json '{"mode": "distribution", "num_samples": 100}'
```

**Note:** SFT uses the same `pretrain_gpt.py` entry point with `--sft` flag. Both pretrain and SFT share the same optimizer infrastructure (`DistributedOptimizer`, `_ParamAndGradBuffer`), so Phase 1 fixes apply to both training modes.

---

## Phase 4: Command Lines (Documentation Only)

> **NOTE:** This phase is for documentation/reference. Execution will be done manually.

### 4.1 Pretrain Verification

```bash
torchrun --nproc_per_node=1 pretrain_gpt.py \
    --use-mcore-models \
    --num-layers 2 \
    --hidden-size 128 \
    --ffn-hidden-size 512 \
    --num-attention-heads 4 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters 5 \
    --seq-length 128 \
    --max-position-embeddings 128 \
    --bf16 \
    --fp4-format e2m1 \
    --fp4-recipe nvfp4 \
    --fp4-param-gather \
    --use-precision-aware-optimizer \
    --exp-avg-dtype bf16 \
    --exp-avg-sq-dtype bf16 \
    --use-distributed-optimizer \
    --overlap-param-gather \
    --overlap-grad-reduce \
    --tensor-model-parallel-size 1 \
    --no-sequence-parallel \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 128256 \
    --log-interval 1
```

### 4.2 SFT Verification

```bash
torchrun --nproc_per_node=1 pretrain_gpt.py \
    --use-mcore-models \
    --num-layers 2 \
    --hidden-size 128 \
    --ffn-hidden-size 512 \
    --num-attention-heads 4 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters 5 \
    --seq-length 128 \
    --max-position-embeddings 128 \
    --bf16 \
    --fp4-format e2m1 \
    --fp4-recipe nvfp4 \
    --fp4-param-gather \
    --use-precision-aware-optimizer \
    --exp-avg-dtype bf16 \
    --exp-avg-sq-dtype bf16 \
    --use-distributed-optimizer \
    --overlap-param-gather \
    --overlap-grad-reduce \
    --tensor-model-parallel-size 1 \
    --no-sequence-parallel \
    --sft \
    --sft-tokenizer-prompt-format nemotron-h-aligned \
    --sft-mock-dataset-config-json '{"mode": "distribution", "num_samples": 100}' \
    --tokenizer-type NullTokenizer \
    --vocab-size 128256 \
    --log-interval 1
```

**Requirements:** TE >= 2.7.0.dev0, B200 (Blackwell/sm_90)

**Note:** Both pretrain and SFT use the same code paths through `DistributedOptimizer` and `_ParamAndGradBuffer`. The Phase 1 fixes are shared between both training modes.

---

## What This Accomplishes

1. **Phase 1** - Removes dequant step for NVFP4, adds NVFP4 to skip paths (applies to both pretrain and SFT)
2. **Phase 2** - Comprehensive FP4 test coverage matching FP8 test structure
3. **Phase 3** - Runnable example scripts for NVFP4 + precision-aware optimizer training (both pretrain and SFT)
4. **Phase 4** - Reference command lines for manual verification (execution by user)

---

## Future Work (Not in Scope)

- FSDP path (17+ locations in `megatron/core/distributed/fsdp/`)
- Checkpointing utilities (7+ locations in `megatron/core/dist_checkpointing/`)
- `quantize_param_shard` has no NVFP4 equivalent — but OK since precision-aware early returns bypass it

---

## Key Citations

| Source | Relevance |
|--------|-----------|
| `megatron/core/optimizer/distrib_optimizer.py:2601` | CRITICAL: dequant crash point |
| `megatron/core/fp4_utils.py:46` | is_nvfp4tensor defined but never imported elsewhere |
| `megatron/core/optimizer/distrib_optimizer.py:365,385,2154,2176,2376,2391,2489` | Existing is_float8tensor checks that need NVFP4 handling |
| `megatron/core/distributed/param_and_grad_buffer.py:362,830` | Existing is_float8tensor checks in param buffer |
| `tests/unit_tests/test_fp8_param.py:68-450` | Reference test structure for FP4 tests |
| `tests/unit_tests/test_fp8_utils.py:1-132` | Reference test structure for FP4 utils tests |
| `tests/unit_tests/dist_checkpointing/test_fp8.py:1-117` | Reference test structure for FP4 checkpoint tests |
| `tests/unit_tests/test_optimizer.py:416-423` | Existing precision-aware test to extend with FP4 |
| `examples/llama/train_llama3_8b_h100_fp8.sh:104-113` | FP8 example script structure to mirror |
| `megatron/training/arguments.py:787-789` | FP8/FP4 mutual exclusivity validation |
| `megatron/training/arguments.py:795-797` | TE version >= 2.7.0.dev0 requirement for FP4 |
| `megatron/core/transformer/transformer_config.py:532-536` | FP4 config field (note: type annotation bug for 'nvfp4' vs 'e2m1') |
| `megatron/training/datasets/sft_dataset.py:55-215` | SFT dataset class (SFT uses same optimizer infra as pretrain) |
| `pretrain_gpt.py:301-305` | SFT entry point via `--sft` flag |
