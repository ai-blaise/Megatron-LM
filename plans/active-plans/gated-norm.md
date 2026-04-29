# Plan: GatedNorm Triton Integration

## Objective

Implement paper-faithful GatedNorm in Megatron-Core by:

1. Adding a fused Triton op `apply_gated_norm(normed, W_down, W_up)`
2. Keeping RMSNorm in the existing Megatron/TE normalization modules
3. Applying GatedNorm after residual-stream RMSNorm sites
4. Covering the standard, recompute, HyperConnection, and MoE bypass paths
5. Avoiding Q/K norm, MLA compressed norm, SSM, MTP, cross-attention, and final-layernorm scope in the first patch

@architect: The critical design boundary is `RMSNorm outside, learned GatedNorm inside apply_gated_norm`. The Triton op starts from `normed`, not from raw hidden states.

---

## Paper-Grounded Rule

The paper defines GatedNorm as:

```text
y = RMSNorm(x)
gate = sigmoid(W_up(swish(W_down(y))))
y_prime = gate * y
```

Implementation boundary:

```text
existing RMSNorm module
-> normed
-> apply_gated_norm(normed, W_down, W_up)
-> attention or MLP/MoE
```

Do not implement this boundary:

```text
custom Triton RMSNorm plus GatedNorm
```

@architect: The paper's GatedNorm is learned. The learned parameters are `W_down` and `W_up`. The Triton op should own those learned gate projections, but it should not own RMSNorm.

@architect: Keeping RMSNorm outside the kernel preserves the existing TE/local norm behavior, precision handling, sharded state dict behavior, and recompute hooks.

### Paper Citations

| Source | Relevance |
|--------|-----------|
| `Gated-Norm.pdf`, Sec. 2, p. 3 | Defines the pre-norm residual stream as the main object of study. |
| `Gated-Norm.pdf`, Sec. 3.1, p. 4 | Separates softmax attention normalization from residual normalization layers. |
| `Gated-Norm.pdf`, Sec. 3.4, p. 6 | Defines GatedNorm as low-rank elementwise gating after RMSNorm. |
| `Gated-Norm.pdf`, App. A.2, p. 13 | Associates residual sinks with RMSNorm and GatedNorm. |
| `Gated-Norm.pdf`, App. A.3, p. 14 | Notes GatedNorm overhead is affected by lightweight kernels and launch bubbles. |

---

## Architecture Scope

### Required First-Patch Sites

| Location | Flow | Citation |
|----------|------|----------|
| Input norm | `input_layernorm -> GatedNorm -> self_attention` | `megatron/core/transformer/transformer_layer.py:580-602` |
| Pre-MLP norm | `pre_mlp_layernorm -> GatedNorm -> MLP/MoE` | `megatron/core/transformer/transformer_layer.py:692-707` |
| HyperConnection input norm | same as input norm | `megatron/core/transformer/transformer_layer.py:1426-1444` |
| HyperConnection pre-MLP norm | same as pre-MLP norm | `megatron/core/transformer/transformer_layer.py:1527-1543` |
| EP overlap / CUDA graph MoE bypass | `pre_mlp_layernorm -> GatedNorm -> router/shared experts` | `megatron/core/transformer/transformer_layer.py:1143-1152` |
| Fine-grained MoE callable | call `layer._forward_pre_mlp_layernorm(hidden_states)` so recompute/offload stay centralized | `megatron/core/models/gpt/fine_grained_callables.py:479-495` |

### Explicit Exclusions

| Exclusion | Reason | Citation |
|-----------|--------|----------|
| Q/K norms | Attention-space norms, not residual-stream RMSNorm. Gating these changes attention geometry. | `megatron/core/transformer/attention.py:1494-1498` |
| MLA q/kv compressed norms | Compressed attention-space norms, not residual-stream RMSNorm. | `megatron/core/transformer/multi_latent_attention.py:638-640` |
| Absorbed MLA q/kv norms | Same exclusion as MLA norms. | `megatron/core/transformer/experimental_attention_variant/absorbed_mla.py:473-475` |
| SSM/Mamba norms | Different architecture path. | `megatron/core/ssm/gated_delta_net.py:420-450`, `megatron/core/ssm/mamba_layer.py:91,141` |
| Cross-attention norm | Not part of the first patch. | `megatron/core/transformer/transformer_layer.py:648-653` |
| Final layernorm | Not part of the first patch. | `megatron/core/transformer/transformer_block.py:936-938` |
| MTP norms | Separate path, not part of the first patch. | `megatron/core/transformer/multi_token_prediction.py:901-903`, `995` |

@architect: "Every normalization layer" from the paper should be read as every residual-stream RMSNorm matching `y = RMSNorm(x)`, not every object with `layernorm` in its name.

@architect: Gating Q/K norm would change attention geometry directly. That is not the GatedNorm intervention described in Sec. 3.4.

---

## Current Relevant Repo State

| Existing Feature | Current Behavior | Citation |
|------------------|------------------|----------|
| G1 gate op | Existing custom gate pattern computes `x * sigmoid(gate)` | `megatron/core/fusions/fused_g1_gate.py:104-117` |
| Attention output gate | Uses G1-style function call in attention | `megatron/core/transformer/attention.py:1224-1230` |
| Layernorm recompute | Recomputes input and pre-MLP layernorm outputs | `megatron/core/transformer/transformer_config.py:429-437` |
| TE fused norm+linear | Can hide the insertion point between RMSNorm and linear | `megatron/core/models/gpt/gpt_layer_specs.py:283-302`, `520-533` |
| Sequence-parallel example | Non-TP-aware params are marked for grad sync via `sequence_parallel` | `megatron/core/transformer/hyper_connection.py:162-171` |

### Constraints

1. RMSNorm remains outside the Triton op.
2. `W_down` and `W_up` are learned Torch parameters.
3. First implementation assumes replicated `W_down` and `W_up` across tensor-parallel ranks, with `sequence_parallel` set on those weights when needed so gradients synchronize correctly.
4. `rank = 16` is the default target from the paper and is also good for tensor-core friendliness.
5. The production path should not materialize full-width `gate_logits` in HBM.
6. Do not stage the production kernel behind a v0/v1/v2 ladder. Implement the direct fused op and its backward path as the target design.

@kernel: Sharding the rank dimension would require communication between down projection and up projection. That breaks the clean single-op design, so replicated gate weights are the first implementation target.

---

## Phase 1: Config Surface

### 1.1 Add `TransformerConfig` Fields

**File:** `megatron/core/transformer/transformer_config.py`

Add near existing normalization/gate config:

```python
gated_norm: bool = False
"""Whether to apply GatedNorm after residual-stream normalization layers."""

gated_norm_rank: int = 16
"""Low-rank dimension for GatedNorm gate projections."""
```

**Citation:** `megatron/core/transformer/transformer_config.py:220-243`

@architect: Keep the default off. Rank 16 follows the paper example.

### 1.2 CLI Exposure

No dedicated parser changes should be needed for simple bool/int fields because `TransformerConfig` is exposed by `ArgumentGroupFactory`.

**Citations:**

| Source | Relevance |
|--------|-----------|
| `megatron/training/arguments.py:2295-2298` | Builds CLI args from `TransformerConfig`. |
| `megatron/training/arguments.py:1866-1885` | Copies matching args into core config. |

Expected flags:

```bash
--gated-norm \
--gated-norm-rank 16
```

### 1.3 Validation

Add validation:

```text
if gated_norm:
    require normalization == "RMSNorm"
    require gated_norm_rank > 0
    require gated_norm_rank <= hidden_size
```

@architect: The paper formula is explicitly `y = RMSNorm(x)`. LayerNorm support can be a later extension, but the first implementation should be RMSNorm-only.

---

## Phase 2: Triton Kernel

### 2.1 Production Public API

**File:** `megatron/core/fusions/gated_norm.py`

```python
def apply_gated_norm(
    normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
) -> torch.Tensor:
    return GatedNormFunction.apply(normed, w_down, w_up)
```

The function computes:

```text
y = normed
z = y @ W_down.T
a = silu(z)
logits = a @ W_up.T
gate = sigmoid(logits)
output = y * gate
```

Shape contract:

```text
y:      [tokens, hidden]
W_down: [rank, hidden]
W_up:   [hidden, rank]
z:      [tokens, rank]
output: [tokens, hidden]
```

### 2.2 Forward And Backward

Backward formulas:

```text
dlogits = dout * y * g * (1 - g)
dW_up   = dlogits.T @ a
da      = dlogits @ W_up
dz      = da * silu_grad(z)
dW_down = dz.T @ y
dy      = dout * g + dz @ W_down
```

Implementation plan:

```text
forward:
    compute z = normed @ W_down.T
    compute a = silu(z)
    compute gate = sigmoid(a @ W_up.T)
    output = normed * gate

backward:
    recompute z and a or load saved z
    compute dy
    accumulate dW_up and dW_down
```

@kernel: Keep the production path direct. Use saved rank-sized `z` if it helps, otherwise recompute it. Do not add a staged production path.

### 2.3 Saved Tensors

Save:

```text
y
W_down
W_up
```

Optionally save:

```text
z = y @ W_down.T
```

Do not save:

```text
gate_logits: [tokens, hidden]
gate:        [tokens, hidden]
```

@kernel: Saving rank-sized `z` is probably worthwhile because it is small. Saving full-width gate/logit tensors defeats the memory benefit.

### 2.4 Shape, Dtype, And Layout Contract

Initial contract:

```text
normed.is_cuda
w_down.is_cuda
w_up.is_cuda
normed.shape[-1] == w_down.shape[1]
w_down.shape[0] == w_up.shape[1]
w_up.shape[0] == normed.shape[-1]
rank == w_down.shape[0]
dtype in {bf16, fp16, fp32}
output dtype == normed dtype
```

Require contiguous tensors for the first implementation:

```python
normed = normed.contiguous()
w_down = w_down.contiguous()
w_up = w_up.contiguous()
```

@kernel: Contiguity copies in the hot path are a risk. The first implementation may call `.contiguous()` for correctness, but tests and profiling should verify the caller normally supplies contiguous tensors. A later kernel can support strides if needed.

### 2.5 PyTorch Reference Only For Tests

Avoid adding a production Python fallback implementation.

Allowed in tests:

```python
z = normed @ w_down.T
a = F.silu(z)
gate = torch.sigmoid(a @ w_up.T)
expected = normed * gate
```

@kernel: Keep the production path Triton-first. PyTorch formula code belongs in tests and debugging comparisons, not in the runtime GatedNorm path.

---

## Phase 3: Gate Parameters And Layer Helper

### 3.1 Add Gate Modules

**File:** `megatron/core/transformer/transformer_layer.py`

Existing norm construction:

```text
self.input_layernorm
self.pre_cross_attn_layernorm
self.pre_mlp_layernorm
```

**Citation:** `megatron/core/transformer/transformer_layer.py:306-357`

Add learned modules:

```python
self.input_gated_norm_down
self.input_gated_norm_up
self.pre_mlp_gated_norm_down
self.pre_mlp_gated_norm_up
```

Recommended module form:

```python
torch.nn.Linear(hidden_size, gated_norm_rank, bias=False)
torch.nn.Linear(gated_norm_rank, hidden_size, bias=False)
```

The helper should pass `.weight` into the fused op instead of calling these modules directly.
If `config.sequence_parallel` is enabled, mark the gate weights with `sequence_parallel=True` so gradient synchronization follows the same convention used elsewhere for non-TP-aware parameters.

@architect: Bias should be disabled initially because the paper formula only names `W_down` and `W_up`. If later checkpoint retrofit needs identity-ish initialization, add that as a separate compatibility option.

### 3.2 Parameter Count

For hidden size `d` and rank `r`:

```text
W_down: r * d
W_up:   d * r
per GatedNorm site: 2 * d * r
two required sites per layer: 4 * d * r
```

For `r = 16`, this is small relative to attention and FFN parameters.

@architect: Do not prematurely shard the rank dimension. If `r < tp_size`, a tensor-parallel split creates avoidable edge cases.

### 3.3 Helper Function

Add one helper:

```python
def _apply_norm_with_gated_norm(
    self,
    hidden_states,
    norm_module,
    gate_down,
    gate_up,
):
    normed = apply_module(norm_module)(hidden_states)

    if gate_down is None or gate_up is None:
        return normed

    return apply_gated_norm(normed, gate_down.weight, gate_up.weight)
```

Import:

```python
from megatron.core.fusions.gated_norm import apply_gated_norm
```

**Citation:** `megatron/core/transformer/transformer_layer.py:14-18`

@architect: This helper is the recompute-friendly boundary for `existing norm -> fused learned GatedNorm`.

---

## Phase 4: Call-Site Integration

### 4.1 Standard Input LayerNorm

**Source:** `megatron/core/transformer/transformer_layer.py:580-602`

Current:

```text
hidden_states
-> input_layernorm
-> self_attention
```

Target:

```text
hidden_states
-> input_layernorm
-> apply_gated_norm(normed, input_W_down, input_W_up)
-> self_attention
```

Recompute path should checkpoint the helper:

```python
input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
    lambda x: self._apply_norm_with_gated_norm(
        x,
        self.input_layernorm,
        self.input_gated_norm_down,
        self.input_gated_norm_up,
    ),
    hidden_states,
)
```

Existing recompute hook remains after attention.

**Citation:** `megatron/core/transformer/transformer_layer.py:614-619`

@architect: Update the nearby comments from "input layernorm" to "input norm/gated norm" when `gated_norm` is enabled. The recompute hook still sits after attention.

### 4.2 Standard Pre-MLP LayerNorm

**Source:** `megatron/core/transformer/transformer_layer.py:692-707`

Target:

```text
hidden_states
-> pre_mlp_layernorm
-> apply_gated_norm(normed, mlp_W_down, mlp_W_up)
-> MLP/MoE
```

Recompute path should checkpoint the helper:

```python
pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
    lambda x: self._apply_norm_with_gated_norm(
        x,
        self.pre_mlp_layernorm,
        self.pre_mlp_gated_norm_down,
        self.pre_mlp_gated_norm_up,
    ),
    hidden_states,
)
```

Existing recompute hook remains after MLP output.

**Citation:** `megatron/core/transformer/transformer_layer.py:832-837`

### 4.3 Duplicate Paths

| Path | Required Change | Citation |
|------|-----------------|----------|
| EP overlap / CUDA graph bypass | Replace direct `pre_mlp_layernorm` call with helper | `megatron/core/transformer/transformer_layer.py:1143-1152` |
| HyperConnection input norm | Use same helper/checkpoint pattern as standard input norm | `megatron/core/transformer/transformer_layer.py:1426-1444` |
| HyperConnection pre-MLP norm | Use same helper/checkpoint pattern as standard pre-MLP norm | `megatron/core/transformer/transformer_layer.py:1527-1543` |
| Fine-grained MoE callable | Reuse the layer's pre-MLP norm wrapper so recompute/offload stay centralized | `megatron/core/models/gpt/fine_grained_callables.py:479-495` |

@implementer: These are not extra paper locations. They are alternate code paths that must be covered so the feature is not silently skipped in specific training modes.

@architect: Fine-grained callables use `layer`, so the helper can remain on `TransformerLayer`; do not add another GatedNorm implementation here.

---

## Phase 5: TE Fused Norm+Linear Handling

GatedNorm needs an explicit insertion point:

```text
RMSNorm output
-> GatedNorm
-> linear_qkv or linear_fc1
```

TE fused norm+linear removes that insertion point.

### 5.1 Attention Input Norm

Current TE path can use:

```python
linear_qkv = backend.column_parallel_layer_norm_linear()
```

When `gated_norm=True`, force:

```text
input_layernorm = backend.layer_norm()
linear_qkv = backend.column_parallel_linear()
```

**Citation:** `megatron/core/models/gpt/gpt_layer_specs.py:283-302`

If `use_te_op_fuser` is requested together with `gated_norm=True`, fail fast or route to the unfused path; the fused layernorm-linear op hides the insertion point GatedNorm needs.

### 5.2 Dense MLP Pre-MLP Norm

Current dense MLP path can use fused norm+linear:

```python
linear_fc1 = backend.column_parallel_layer_norm_linear()
```

When `gated_norm=True`, force:

```text
pre_mlp_layernorm = backend.layer_norm()
linear_fc1 = backend.column_parallel_linear()
```

**Citation:** `megatron/core/models/gpt/gpt_layer_specs.py:520-533`

If `use_te_op_fuser` is requested together with `gated_norm=True`, fail fast or route to the unfused path for the same reason.

### 5.3 Provider Rule

Do not change TE provider globally.

**Citation:** `megatron/core/extensions/transformer_engine_spec_provider.py:52-58`

@architect: This should be a spec-selection change gated by `config.gated_norm`, not a global TE behavior change.

---

## Phase 6: Checkpointing And State Dicts

### 6.1 New Parameter Names

Use stable, descriptive names:

```text
layers.N.input_gated_norm_down.weight
layers.N.input_gated_norm_up.weight
layers.N.pre_mlp_gated_norm_down.weight
layers.N.pre_mlp_gated_norm_up.weight
```

### 6.2 Existing Checkpoint Loading

First implementation should expect missing GatedNorm weights when enabling the feature on an old checkpoint.

Plan:

```text
from-scratch training: works normally
loading old checkpoints with gated_norm=True: allow missing gate weights only if existing checkpoint load supports non-strict mode
pretrained retrofit: out of first scope unless explicitly requested
```

@architect: The paper experiments appear to train with the architecture enabled, not patch a trained checkpoint. Avoid identity-initialization compatibility work until needed.

### 6.3 State Dict Expectations

When disabling fused norm+linear for GatedNorm, verify:

```text
input_layernorm keys are emitted as explicit layer params
pre_mlp_layernorm keys are emitted as explicit layer params
new gate keys are included in sharded state dict
```

---

## Phase 7: Tests

### 7.1 Triton Op Tests

**File:** `tests/unit_tests/fusions/test_gated_norm.py`

| Test | Description |
|------|-------------|
| `test_apply_gated_norm_forward_bf16` | Compare Triton output with full PyTorch GatedNorm formula. |
| `test_apply_gated_norm_forward_fp16` | Same for fp16. |
| `test_apply_gated_norm_backward` | Compare `dy`, `dW_down`, and `dW_up` against PyTorch reference. |
| `test_apply_gated_norm_rank16_default` | Verify rank-16 path works. |
| `test_apply_gated_norm_shape_mismatch` | Assert shape mismatch fails clearly. |
| `test_apply_gated_norm_requires_cuda` | Assert CPU tensors fail clearly. |

Reference math:

```python
z = normed @ w_down.T
a = F.silu(z)
gate = torch.sigmoid(a @ w_up.T)
expected = normed * gate
```

### 7.2 Transformer Integration Tests

**File:** `tests/unit_tests/transformer/test_gated_norm.py`

| Test | Description |
|------|-------------|
| `test_gated_norm_modules_created_when_enabled` | Verifies input/pre-MLP gate modules exist. |
| `test_gated_norm_modules_absent_when_disabled` | Confirms default behavior unchanged. |
| `test_gated_norm_called_after_input_layernorm` | Verifies self-attention receives gated norm output. |
| `test_gated_norm_called_after_pre_mlp_layernorm` | Verifies MLP/MoE receives gated norm output. |
| `test_qk_layernorm_not_gated` | Ensures Q/K norms do not create GatedNorm modules. |
| `test_layernorm_recompute_with_gated_norm` | Verifies backward succeeds with `recompute_modules=["layernorm"]`. |
| `test_sequence_parallel_marks_gate_weights` | Verifies gate weights are marked for sequence-parallel gradient sync. |
| `test_te_op_fuser_rejected_with_gated_norm` | Verifies fused norm+linear paths are not used with `gated_norm=True`. |

### 7.3 TE Spec Tests

Verify:

```text
gated_norm=False:
    TE fused norm+linear behavior remains unchanged

gated_norm=True:
    input_layernorm is explicit
    linear_qkv is not layernorm-linear fused
    pre_mlp_layernorm is explicit where applicable
    linear_fc1 is not layernorm-linear fused
```

---

## Phase 8: Smoke Commands

### Minimal Smoke

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
    --normalization RMSNorm \
    --swiglu \
    --gated-norm \
    --gated-norm-rank 16 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 128256 \
    --log-interval 1
```

### Recompute Smoke

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
    --normalization RMSNorm \
    --swiglu \
    --gated-norm \
    --gated-norm-rank 16 \
    --recompute-granularity selective \
    --recompute-modules layernorm \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 128256 \
    --log-interval 1
```

---

## Implementation Order

1. Add `TransformerConfig` fields and validation.
2. Implement `megatron/core/fusions/gated_norm.py` with API `apply_gated_norm(normed, W_down, W_up)`.
3. Add gate modules to `TransformerLayer`.
4. Add `_apply_norm_with_gated_norm(...)`.
5. Replace standard input/pre-MLP norm call sites.
6. Update recompute paths to checkpoint the helper, not raw RMSNorm alone.
7. Cover HyperConnection, EP overlap, and fine-grained MoE bypass paths.
8. Disable TE fused norm+linear only when `gated_norm=True`.
9. Add Triton op tests.
10. Add TransformerLayer integration tests.
11. Add recompute and TE spec tests.
12. Run smoke commands.

---

## Success Criteria

1. `--gated-norm` adds GatedNorm after `input_layernorm` and `pre_mlp_layernorm`.
2. `apply_gated_norm(normed, W_down, W_up)` is the production public kernel API.
3. RMSNorm remains in existing Megatron/TE norm modules.
4. The Triton op implements `sigmoid(W_up(swish(W_down(y)))) * y`.
5. Full-width `gate_logits` and `gate` tensors are not materialized by the caller.
6. Q/K norms, MLA compressed norms, SSM/Mamba norms, cross-attention, and final-layernorm are not gated in the first patch.
7. Gate weights are Torch-owned parameters and are marked for sequence-parallel sync when needed.
8. Layernorm recompute works with GatedNorm enabled.
9. TE fused norm+linear is disabled only where GatedNorm needs an insertion point.
10. Default behavior is unchanged when `gated_norm=False`.

---

## Key Citations

| Source | Relevance |
|--------|-----------|
| `Gated-Norm.pdf`, Sec. 3.4, p. 6 | Defines GatedNorm after RMSNorm. |
| `Gated-Norm.pdf`, App. A.3, p. 14 | Motivates kernel fusion by discussing launch overhead. |
| `megatron/core/fusions/fused_g1_gate.py:104-117` | Existing public fused gate API style. |
| `megatron/core/transformer/transformer_config.py:220-243` | Existing normalization/gate config area. |
| `megatron/core/transformer/transformer_config.py:429-437` | Layernorm recompute documentation. |
| `megatron/training/arguments.py:1866-1885` | Args copied into core config. |
| `megatron/training/arguments.py:2295-2298` | Config fields exposed as CLI args. |
| `megatron/core/transformer/transformer_layer.py:306-357` | Builds residual-stream norm modules. |
| `megatron/core/transformer/transformer_layer.py:580-602` | Input norm before self-attention. |
| `megatron/core/transformer/transformer_layer.py:692-707` | Pre-MLP norm helper. |
| `megatron/core/transformer/transformer_layer.py:1143-1152` | EP overlap/CUDA graph pre-MLP bypass. |
| `megatron/core/transformer/transformer_layer.py:1426-1444` | HyperConnection input norm path. |
| `megatron/core/transformer/transformer_layer.py:1527-1543` | HyperConnection pre-MLP norm path. |
| `megatron/core/models/gpt/fine_grained_callables.py:479-495` | Fine-grained MoE bypass path. |
| `megatron/core/transformer/attention.py:1494-1498` | Q/K norms excluded from first scope. |
| `megatron/core/models/gpt/gpt_layer_specs.py:283-302` | TE attention fused norm+linear path. |
| `megatron/core/models/gpt/gpt_layer_specs.py:520-533` | Dense MLP fused norm+linear path. |
| `megatron/core/extensions/transformer_engine_spec_provider.py:52-58` | TE provider fused norm+linear support. |
| `megatron/core/transformer/hyper_connection.py:162-171` | Example of marking non-TP-aware parameters for sequence-parallel sync. |

---

## Out Of Scope For First Patch

1. Gated Attention implementation
2. Q/K norm gating
3. MLA compressed-space norm gating
4. SSM/Mamba norm gating
5. MTP norm gating
6. Pretrained checkpoint retrofit or identity-preserving initialization
7. Any Triton kernel that computes RMSNorm itself
