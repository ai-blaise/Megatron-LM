# Plan: GatedNorm Triton Integration

## Objective

Add paper-faithful GatedNorm support to Megatron-Core with:

1. A single public Triton-backed function, `apply_gated_norm(...)`
2. GatedNorm applied after residual-stream RMSNorm sites
3. Clean call sites that work with Megatron's layernorm recompute path
4. No gating of Q/K attention-space norms in the first implementation

@architect: The important distinction is that `apply_gated_norm` should be the G1-style kernel launcher, not a second Python implementation of the math. A small `TransformerLayer` helper is still useful because Megatron needs one callable boundary for `RMSNorm -> gate projection -> apply_gated_norm` when layernorm recompute is enabled.

---

## Paper Basis

The paper is `/home/archimedes/Documents/archimedesvault/academic-texts/ML/Gated-Norm.pdf`.

Relevant claims from the paper:

1. The paper focuses on pre-norm transformers and the residual stream `H_i`.
2. It separates attention sinks from residual sinks:
   - attention sinks are tied to softmax normalization in attention
   - residual sinks are tied to RMSNorm on the residual stream
3. GatedNorm is introduced as an explicit residual-stream rescaling mechanism after RMSNorm.
4. The formula is:

```text
y = RMSNorm(x)
gate = sigmoid(W_up(swish(W_down(y))))
y_prime = gate * y
```

**Paper citations:**

| Source | Relevance |
|--------|-----------|
| `Gated-Norm.pdf`, Sec. 2, p. 3 | Defines the pre-norm residual stream as the main object of study. |
| `Gated-Norm.pdf`, Sec. 3.1, p. 4 | Separates transformer normalizations into softmax attention and residual normalization layers. |
| `Gated-Norm.pdf`, Sec. 3.4, p. 6 | Defines GatedNorm as a low-rank elementwise gate after RMSNorm. |
| `Gated-Norm.pdf`, App. A.2, p. 13 | Compares attention sinks and residual sinks; associates residual sinks with RMSNorm and GatedNorm. |

@architect: The phrase "after every normalization layer" in Sec. 3.4 should be implemented as every residual-stream RMSNorm that matches `y = RMSNorm(x)`. It should not automatically include Q/K norms, MLA compressed-space norms, or unrelated SSM norms.

---

## Design Decisions

### 1. Public Kernel API

The public API should mirror the existing G1 gate style:

```python
output = apply_gated_norm(gate_logits, normed)
```

It computes:

```python
output = normed * sigmoid(gate_logits)
```

This matches the existing G1 gate shape:

```python
output = g1_gate_impl(linear_out, attn_out)
```

**Citation:** `megatron/core/fusions/fused_g1_gate.py:104-117`

@kernel: `apply_gated_norm` should launch Triton kernels for forward and backward. The RMSNorm module and the low-rank gate projections stay outside the Triton kernel.

### 2. RMSNorm Stays Existing Megatron Code

Do not implement RMSNorm inside the Triton kernel.

Correct boundary:

```text
existing RMSNorm module
-> gate_down / SiLU / gate_up
-> apply_gated_norm Triton kernel
```

Incorrect boundary:

```text
custom Triton RMSNorm + gate projection + gated multiply
```

@architect: Keeping RMSNorm outside the kernel preserves the existing TE/local norm behavior, precision handling, sharded state dict behavior, and recompute hooks.

### 3. One Helper Method Is Acceptable

We should add one layer-local helper to avoid duplicating this block at every norm site:

```python
normed = apply_module(norm_module)(hidden_states)
gate_logits = gate_up(F.silu(gate_down(normed)))
return apply_gated_norm(gate_logits, normed)
```

This helper is not a second GatedNorm implementation. It is a call-site wrapper around:

```text
existing norm
-> existing PyTorch/TE linears
-> Triton apply_gated_norm
```

@architect: This helper is what makes activation recompute clean. Megatron can checkpoint one callable that contains the existing norm, the gate projections, and the Triton gated multiply.

### 4. First Implementation Scope

Implement:

```text
input_layernorm -> GatedNorm -> self_attention
pre_mlp_layernorm -> GatedNorm -> MLP/MoE
```

Also cover duplicate execution paths that bypass the standard helper.

Do not implement initially:

```text
Q/K layernorm -> GatedNorm
MLA q/kv compressed-space layernorm -> GatedNorm
SSM/Mamba norms -> GatedNorm
```

Optional later:

```text
pre_cross_attn_layernorm -> GatedNorm -> cross_attention
final_layernorm -> GatedNorm -> output head
MTP-specific norms -> GatedNorm
```

---

## Files To Modify

| File | Purpose | Citation |
|------|---------|----------|
| `megatron/core/fusions/gated_norm.py` | New Triton `apply_gated_norm` op with autograd. | New file; mirror public API style from `fused_g1_gate.py:104-117`. |
| `megatron/core/transformer/transformer_config.py` | Add GatedNorm config fields. | `transformer_config.py:220-243`, `transformer_config.py:429-437` |
| `megatron/core/transformer/transformer_layer.py` | Add gate modules, helper, and main call sites. | `transformer_layer.py:306-357`, `580-602`, `692-707` |
| `megatron/core/models/gpt/gpt_layer_specs.py` | Disable fused norm+linear when GatedNorm is enabled. | `gpt_layer_specs.py:283-302`, `520-533` |
| `megatron/core/models/gpt/fine_grained_callables.py` | Cover MoE fine-grained callable bypass. | `fine_grained_callables.py:479-495` |
| `megatron/core/transformer/transformer_block.py` | Optional final-layernorm support later. | `transformer_block.py:383-390`, `936-938` |
| `tests/unit_tests/fusions/test_gated_norm.py` | Unit tests for Triton op forward/backward. | New file. |
| `tests/unit_tests/transformer/test_gated_norm.py` | Integration tests for call-site behavior. | New file. |

---

## Phase 1: Config Surface

### 1.1 Add TransformerConfig Fields

Add fields near normalization / existing gate options:

**Source location:** `megatron/core/transformer/transformer_config.py:220-243`

```python
gated_norm: bool = False
"""Whether to apply GatedNorm after residual-stream normalization layers."""

gated_norm_rank: int = 16
"""Low-rank dimension for GatedNorm gate projections."""

gated_norm_include_cross_attention: bool = False
"""Whether to apply GatedNorm after pre_cross_attn_layernorm when cross-attention is active."""

gated_norm_include_final_layernorm: bool = False
"""Whether to apply GatedNorm after the decoder final layernorm."""
```

@architect: Keep the default off. Rank 16 follows the paper example. Cross-attention and final-layernorm should be explicit because GPT-style first implementation does not need them.

### 1.2 CLI Handling

`TransformerConfig` fields are automatically exposed by `ArgumentGroupFactory`.

**Citation:** `megatron/training/arguments.py:2295-2298`

```python
transformer_factory = ArgumentGroupFactory(TransformerConfig, exclude=exclude)
transformer_group = transformer_factory.build_group(
    parser, "transformer configuration"
)
```

Config construction also copies matching dataclass fields from args:

**Citation:** `megatron/training/arguments.py:1866-1885`

No manual parser changes should be needed for simple `bool` and `int` fields.

Expected CLI:

```bash
--gated-norm \
--gated-norm-rank 16
```

Optional:

```bash
--gated-norm-include-cross-attention \
--gated-norm-include-final-layernorm
```

### 1.3 Validation

Add config validation in `TransformerConfig.__post_init__` or nearby validation logic:

```text
if gated_norm:
    require normalization == "RMSNorm" for paper-faithful mode
    require gated_norm_rank > 0
```

@architect: The paper formula is explicitly `y = RMSNorm(x)`. We can support LayerNorm later, but the first implementation should reject or warn for non-RMSNorm.

---

## Phase 2: Triton Fusion Function

### 2.1 New File

Create:

```text
megatron/core/fusions/gated_norm.py
```

Public function:

```python
def apply_gated_norm(gate_logits: torch.Tensor, normed: torch.Tensor) -> torch.Tensor:
    return GatedNormFunction.apply(gate_logits, normed)
```

Forward math:

```text
gate = sigmoid(gate_logits)
output = normed * gate
```

Backward math:

```text
d_normed = d_output * gate
d_gate_logits = d_output * normed * gate * (1 - gate)
```

@kernel: Save either `gate` and `normed`, or `gate_logits` and `normed`. Saving `gate` avoids recomputing sigmoid in backward but costs one activation tensor. Match G1's pattern first, then optimize if profiling shows pressure.

### 2.2 Shape And Dtype Contract

Initial contract:

```text
gate_logits.shape == normed.shape
gate_logits.is_cuda
normed.is_cuda
dtype in {bf16, fp16, fp32}
output dtype == normed dtype
```

Prefer contiguous tensors inside the autograd function:

```python
gate_logits = gate_logits.contiguous()
normed = normed.contiguous()
```

**Citation:** G1 uses this pattern at `megatron/core/fusions/fused_g1_gate.py:89-100`.

### 2.3 No Separate Python Fallback

Avoid adding a second Python fallback implementation in the production path.

Allowed only in tests:

```python
expected = normed * torch.sigmoid(gate_logits.float()).to(normed.dtype)
```

@kernel: This keeps the production API aligned with the user's requested design: Triton implementation first, then source call sites call `apply_gated_norm`.

---

## Phase 3: Gate Parameters In TransformerLayer

### 3.1 Add Gate Projection Modules

Add gate projection modules after the existing norm modules are constructed.

**Source location:** `megatron/core/transformer/transformer_layer.py:306-357`

Existing modules:

```text
self.input_layernorm
self.pre_cross_attn_layernorm
self.pre_mlp_layernorm
```

Add, gated by config and by whether the corresponding norm is not `IdentityOp`:

```python
self.input_gated_norm_down
self.input_gated_norm_up
self.pre_mlp_gated_norm_down
self.pre_mlp_gated_norm_up
```

Optional:

```python
self.pre_cross_attn_gated_norm_down
self.pre_cross_attn_gated_norm_up
```

@architect: Bias should be disabled initially because the paper formula only names `W_down` and `W_up`. If later checkpoint retrofit needs identity-ish initialization, add that as a separate compatibility option.

### 3.2 Parameter Shape

For hidden size `d` and rank `r`:

```text
gate_down: d -> r
gate_up: r -> d
```

For each gated residual norm:

```text
parameters = d*r + r*d = 2*d*r
```

For the two required sites per layer:

```text
per-layer parameters = 4*d*r
```

With `r = 16`, this is small relative to attention and FFN GEMMs.

### 3.3 Tensor Parallelism Decision

Start with replicated low-rank modules, because:

1. GatedNorm needs the full hidden vector after RMSNorm.
2. Rank 16 is small and awkward to shard across tensor-parallel ranks.
3. LayerNorm parameters are already small replicated parameters in the residual path.

Validation item:

```text
Confirm gradients for replicated gate parameters are synchronized correctly under TP/DP.
```

If replicated parameter handling is not correct under tensor parallelism, revise to match the repo's established non-tensor-parallel parameter treatment for LayerNorm-like modules.

@architect: Do not prematurely shard the rank dimension. If `r < tp_size`, a tensor-parallel split creates avoidable edge cases.

---

## Phase 4: Layer Helper

Add one helper method to `TransformerLayer`:

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

    gate_logits = gate_up(F.silu(gate_down(normed)))
    return apply_gated_norm(gate_logits, normed)
```

Imports needed:

```python
import torch.nn.functional as F
from megatron.core.fusions.gated_norm import apply_gated_norm
```

**Source location for imports:** `megatron/core/transformer/transformer_layer.py:14-18`

@architect: This is the only helper we need in `TransformerLayer`. It exists to keep all call sites consistent and to let checkpointing recompute the full logical unit.

---

## Phase 5: Required Call Sites

### 5.1 Standard Input LayerNorm

**Source:** `megatron/core/transformer/transformer_layer.py:580-602`

Current flow:

```text
hidden_states
-> input_layernorm
-> self_attention
```

Target flow:

```text
hidden_states
-> input_layernorm
-> input GatedNorm
-> self_attention
```

Normal path example:

```python
input_layernorm_output = self._apply_norm_with_gated_norm(
    hidden_states,
    self.input_layernorm,
    self.input_gated_norm_down,
    self.input_gated_norm_up,
)
```

Recompute path example:

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

@architect: Update the nearby comments from "input layernorm" to "input norm/gated norm" when `gated_norm` is enabled. The recompute hook still sits after attention at `transformer_layer.py:614-619`.

### 5.2 Standard Pre-MLP LayerNorm

**Source:** `megatron/core/transformer/transformer_layer.py:692-707`

Current flow:

```text
hidden_states
-> pre_mlp_layernorm
-> MLP/MoE
```

Target flow:

```text
hidden_states
-> pre_mlp_layernorm
-> pre-MLP GatedNorm
-> MLP/MoE
```

Normal path example:

```python
pre_mlp_layernorm_output = self._apply_norm_with_gated_norm(
    hidden_states,
    self.pre_mlp_layernorm,
    self.pre_mlp_gated_norm_down,
    self.pre_mlp_gated_norm_up,
)
```

Recompute path example:

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

The existing discard/recompute registration remains after MLP output.

**Citation:** `megatron/core/transformer/transformer_layer.py:832-837`

---

## Phase 6: Duplicate Execution Paths To Cover

These are not new mathematical locations. They are alternate code paths that currently bypass the standard norm helpers.

### 6.1 EP Overlap / CUDA Graph MoE Bypass

**Source:** `megatron/core/transformer/transformer_layer.py:1143-1152`

Current:

```python
hidden_states = apply_module(self.pre_mlp_layernorm)(residual)
```

Target:

```python
hidden_states = self._apply_norm_with_gated_norm(
    residual,
    self.pre_mlp_layernorm,
    self.pre_mlp_gated_norm_down,
    self.pre_mlp_gated_norm_up,
)
```

### 6.2 HyperConnection Input LayerNorm

**Source:** `megatron/core/transformer/transformer_layer.py:1426-1444`

Target:

```text
hidden_states
-> input_layernorm
-> GatedNorm
-> self_attention
```

Use the same helper and checkpoint pattern as the standard input norm path.

### 6.3 HyperConnection Pre-MLP LayerNorm

**Source:** `megatron/core/transformer/transformer_layer.py:1527-1543`

Target:

```text
hidden_states
-> pre_mlp_layernorm
-> GatedNorm
-> MLP/MoE
```

Use the same helper and checkpoint pattern as the standard pre-MLP path.

### 6.4 Fine-Grained MoE Callable

**Source:** `megatron/core/models/gpt/fine_grained_callables.py:479-495`

Current:

```python
pre_mlp_layernorm_output = apply_module(layer.pre_mlp_layernorm)(hidden_states)
```

Target:

```python
pre_mlp_layernorm_output = layer._apply_norm_with_gated_norm(
    hidden_states,
    layer.pre_mlp_layernorm,
    layer.pre_mlp_gated_norm_down,
    layer.pre_mlp_gated_norm_up,
)
```

@architect: Fine-grained callables use `layer`, so the helper can remain on `TransformerLayer`; do not add another GatedNorm implementation here.

---

## Phase 7: Conditional And Optional Sites

### 7.1 Cross-Attention Norm

**Source:** `megatron/core/transformer/transformer_layer.py:648-653`

Current:

```text
hidden_states
-> pre_cross_attn_layernorm
-> cross_attention
```

Target only when `config.gated_norm_include_cross_attention`:

```text
hidden_states
-> pre_cross_attn_layernorm
-> GatedNorm
-> cross_attention
```

@architect: GPT-style decoder-only models usually have this as `IdentityOp`. Keep it off by default to avoid creating unused parameters.

### 7.2 Final LayerNorm

**Sources:**

| Source | Relevance |
|--------|-----------|
| `megatron/core/transformer/transformer_block.py:383-390` | Builds `final_layernorm`. |
| `megatron/core/transformer/transformer_block.py:936-938` | Applies `final_layernorm`. |
| `megatron/core/models/gpt/fine_grained_callables.py:610-613` | Fine-grained final norm path. |

Target only when `config.gated_norm_include_final_layernorm`:

```text
hidden_states
-> final_layernorm
-> GatedNorm
-> output head
```

@architect: This is paper-literal but not the first priority. It does not feed attention or MLP inside a residual block, so keep it behind an explicit flag.

### 7.3 MTP Norms

**Sources:**

| Source | Relevance |
|--------|-----------|
| `megatron/core/transformer/multi_token_prediction.py:786-796` | Builds `enorm` and `hnorm`. |
| `megatron/core/transformer/multi_token_prediction.py:901-903` | Applies MTP input norms. |
| `megatron/core/transformer/multi_token_prediction.py:995` | Applies MTP final norm. |

Out of first scope. Revisit only after the transformer-layer implementation is validated.

---

## Phase 8: Explicit Exclusions

### 8.1 Do Not Gate Q/K Norms Initially

**Sources:**

| Source | Relevance |
|--------|-----------|
| `megatron/core/transformer/attention.py:1298-1314` | Builds Q/K layernorm modules. |
| `megatron/core/transformer/attention.py:1494-1498` | Applies Q/K layernorm. |

Reason:

```text
Q/K norms operate inside attention projection space.
The paper's GatedNorm targets residual-stream RMSNorm.
Attention-side rescaling is a separate Gated Attention mechanism.
```

@architect: Gating Q/K norm would change attention geometry directly. That is not the GatedNorm intervention described in Sec. 3.4.

### 8.2 Do Not Gate MLA Compressed-Space Norms Initially

**Sources:**

| Source | Relevance |
|--------|-----------|
| `megatron/core/transformer/multi_latent_attention.py:500-510` | Builds MLA q/kv norms. |
| `megatron/core/transformer/multi_latent_attention.py:638-640` | Applies MLA q/kv norms. |
| `megatron/core/transformer/experimental_attention_variant/absorbed_mla.py:298-306` | Builds absorbed MLA q/kv norms. |
| `megatron/core/transformer/experimental_attention_variant/absorbed_mla.py:473-475` | Applies absorbed MLA q/kv norms. |

Reason:

```text
These are not residual-stream RMSNorm outputs feeding a transformer sublayer.
```

### 8.3 Do Not Gate SSM/Mamba Norms Initially

**Sources:**

| Source | Relevance |
|--------|-----------|
| `megatron/core/ssm/gated_delta_net.py:420-450` | Already has a gated norm-like pattern. |
| `megatron/core/ssm/mamba_layer.py:91,141` | Mamba norm construction and call. |

Out of first scope.

---

## Phase 9: TE Fused Norm+Linear Handling

GatedNorm requires an explicit tensor between normalization and the next linear/sublayer:

```text
RMSNorm output
-> GatedNorm
-> linear_qkv or linear_fc1
```

TE fused norm+linear removes that insertion point.

### 9.1 Attention Input Norm

In TE GPT specs, common self-attention currently uses fused layernorm+QKV:

**Source:** `megatron/core/models/gpt/gpt_layer_specs.py:283-302`

```python
linear_qkv=backend.column_parallel_layer_norm_linear()
```

When `gated_norm` is enabled, this must become:

```text
input_layernorm=backend.layer_norm()
linear_qkv=backend.column_parallel_linear()
```

### 9.2 Dense MLP Pre-MLP Norm

Dense MLP selection currently uses fused norm+linear when available:

**Source:** `megatron/core/models/gpt/gpt_layer_specs.py:520-533`

When `gated_norm` is enabled, force:

```text
pre_mlp_layernorm=backend.layer_norm()
linear_fc1=backend.column_parallel_linear()
```

### 9.3 TE Provider

TE provider returns fused norm+linear support:

**Source:** `megatron/core/extensions/transformer_engine_spec_provider.py:52-58`

Do not change TE provider globally. Instead, make GPT spec construction choose unfused modules when `config.gated_norm` is true.

@architect: This avoids regressing non-GatedNorm TE performance. The unfused path should only be selected when the new feature needs the insertion point.

---

## Phase 10: Checkpointing And State Dicts

### 10.1 New Parameters

New parameter names should be stable and descriptive:

```text
layers.N.input_gated_norm_down.weight
layers.N.input_gated_norm_up.weight
layers.N.pre_mlp_gated_norm_down.weight
layers.N.pre_mlp_gated_norm_up.weight
```

Optional:

```text
layers.N.pre_cross_attn_gated_norm_down.weight
layers.N.pre_cross_attn_gated_norm_up.weight
decoder.final_gated_norm_down.weight
decoder.final_gated_norm_up.weight
```

### 10.2 Loading Existing Checkpoints

First implementation should expect missing GatedNorm weights when enabling the feature on an old checkpoint.

Plan:

```text
from-scratch training: works normally
loading old checkpoints with gated_norm=True: allow missing gate weights only if existing checkpoint load supports non-strict mode
pretrained retrofit: out of first scope unless the user requests it
```

@architect: The paper experiments appear to train with the architecture enabled, not patch a trained checkpoint. Avoid identity-initialization compatibility work until needed.

### 10.3 TE Sharded State Dict Key Maps

TE/local specs already contain state dict key maps for fused norm paths.

**Citation:** `megatron/core/models/gpt/gpt_layer_specs.py:442-444`

When disabling fused norm+linear for GatedNorm, verify:

```text
input_layernorm keys are emitted as explicit layer params
pre_mlp_layernorm keys are emitted as explicit layer params
new gate keys are included in sharded state dict
```

---

## Phase 11: Tests

### 11.1 Triton Op Unit Tests

New file:

```text
tests/unit_tests/fusions/test_gated_norm.py
```

Tests:

| Test | Description |
|------|-------------|
| `test_apply_gated_norm_forward_bf16` | Compare Triton output with `normed * sigmoid(gate_logits)` |
| `test_apply_gated_norm_forward_fp16` | Same for fp16 |
| `test_apply_gated_norm_backward` | Compare gradients against PyTorch reference |
| `test_apply_gated_norm_shape_mismatch` | Assert shape mismatch fails clearly |
| `test_apply_gated_norm_requires_cuda` | Assert CPU tensors fail clearly |

Reference math only inside tests:

```python
expected = normed * torch.sigmoid(gate_logits.float()).to(normed.dtype)
```

### 11.2 TransformerLayer Integration Tests

New file:

```text
tests/unit_tests/transformer/test_gated_norm.py
```

Tests:

| Test | Description |
|------|-------------|
| `test_gated_norm_modules_created_when_enabled` | Enables `gated_norm`; verifies input/pre-MLP gate modules exist. |
| `test_gated_norm_modules_absent_when_disabled` | Confirms default behavior unchanged. |
| `test_gated_norm_called_after_input_layernorm` | Monkeypatch or hook `apply_gated_norm`; verify self-attention receives gated tensor. |
| `test_gated_norm_called_after_pre_mlp_layernorm` | Verify MLP/MoE receives gated tensor. |
| `test_qk_layernorm_not_gated` | Enables `qk_layernorm`; verifies no Q/K gate modules are created. |

### 11.3 Recompute Tests

Add a targeted test with:

```bash
--recompute-granularity selective \
--recompute-modules layernorm \
--gated-norm
```

Expected:

```text
forward succeeds
backward succeeds
gate parameters receive gradients
CheckpointWithoutOutput recomputes norm+gate helper without stale activation errors
```

**Citation:** `megatron/core/transformer/transformer_config.py:429-437`

### 11.4 TE Spec Tests

Add or extend spec tests to verify:

```text
gated_norm=False with TE:
    fused norm+linear remains allowed

gated_norm=True with TE:
    input_layernorm is explicit
    linear_qkv is not layernorm-linear fused
    pre_mlp_layernorm is explicit where applicable
    linear_fc1 is not layernorm-linear fused
```

**Citations:**

| Source | Relevance |
|--------|-----------|
| `megatron/core/models/gpt/gpt_layer_specs.py:283-302` | TE attention fused norm+linear path. |
| `megatron/core/models/gpt/gpt_layer_specs.py:520-533` | Dense MLP fused norm+linear path. |

### 11.5 Smoke Commands

Minimal local smoke:

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

Recompute smoke:

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
2. Implement `megatron/core/fusions/gated_norm.py` with Triton forward/backward.
3. Add gate projection modules to `TransformerLayer`.
4. Add `_apply_norm_with_gated_norm(...)`.
5. Replace standard input/pre-MLP norm call sites.
6. Update recompute paths to checkpoint the helper, not just raw RMSNorm.
7. Cover HyperConnection, EP overlap, and fine-grained MoE bypass paths.
8. Disable TE fused norm+linear only when `gated_norm=True`.
9. Add Triton op unit tests.
10. Add TransformerLayer integration tests.
11. Add recompute and TE spec tests.
12. Run smoke commands.

@implementer: Do not start with optional final-layernorm or MTP support. Land the two core residual-stream sites first and keep the patch reviewable.

---

## Success Criteria

1. `--gated-norm` adds GatedNorm after `input_layernorm` and `pre_mlp_layernorm`.
2. `apply_gated_norm(gate_logits, normed)` is the only production public kernel call for the gated multiply.
3. Existing RMSNorm modules still own normalization.
4. Gate projections use the paper's low-rank sigmoid gate:

```text
sigmoid(W_up(swish(W_down(y))))
```

5. Q/K norms are not gated.
6. Layernorm recompute works with GatedNorm enabled.
7. TE fused norm+linear is disabled only where GatedNorm requires an insertion point.
8. Default behavior is unchanged when `gated_norm=False`.

---

## Key Citations

| Source | Relevance |
|--------|-----------|
| `Gated-Norm.pdf`, Sec. 3.4, p. 6 | Defines GatedNorm formula after RMSNorm. |
| `Gated-Norm.pdf`, App. A.2, p. 13 | Associates residual sinks with RMSNorm and GatedNorm. |
| `megatron/core/fusions/fused_g1_gate.py:104-117` | Existing public fused gate API style to mirror. |
| `megatron/core/transformer/transformer_config.py:220-243` | Existing normalization and attention gate config area. |
| `megatron/core/transformer/transformer_config.py:429-437` | Layernorm recompute config documentation. |
| `megatron/training/arguments.py:1866-1885` | Args copied into core transformer config. |
| `megatron/training/arguments.py:2295-2298` | TransformerConfig automatically exposed as CLI args. |
| `megatron/core/transformer/transformer_layer.py:306-357` | Builds input/pre-cross/pre-MLP norm modules. |
| `megatron/core/transformer/transformer_layer.py:580-602` | Standard input norm before self-attention. |
| `megatron/core/transformer/transformer_layer.py:614-619` | Input layernorm recompute hook registration. |
| `megatron/core/transformer/transformer_layer.py:648-653` | Cross-attention norm site. |
| `megatron/core/transformer/transformer_layer.py:692-707` | Standard pre-MLP norm helper. |
| `megatron/core/transformer/transformer_layer.py:832-837` | Pre-MLP layernorm recompute hook registration. |
| `megatron/core/transformer/transformer_layer.py:1143-1152` | EP overlap/CUDA graph pre-MLP norm bypass. |
| `megatron/core/transformer/transformer_layer.py:1426-1444` | HyperConnection input norm path. |
| `megatron/core/transformer/transformer_layer.py:1527-1543` | HyperConnection pre-MLP norm path. |
| `megatron/core/models/gpt/fine_grained_callables.py:479-495` | Fine-grained MoE pre-MLP norm bypass. |
| `megatron/core/transformer/attention.py:1298-1314` | Q/K norm construction, excluded from first GatedNorm scope. |
| `megatron/core/transformer/attention.py:1494-1498` | Q/K norm application, excluded from first GatedNorm scope. |
| `megatron/core/transformer/multi_latent_attention.py:638-640` | MLA q/kv norms, excluded from first GatedNorm scope. |
| `megatron/core/models/gpt/gpt_layer_specs.py:283-302` | TE attention fused norm+linear path that must be disabled for GatedNorm. |
| `megatron/core/models/gpt/gpt_layer_specs.py:520-533` | Dense MLP fused norm+linear path that must be disabled for GatedNorm. |
| `megatron/core/extensions/transformer_engine_spec_provider.py:52-58` | TE provider exposes fused norm+linear support. |
| `megatron/core/transformer/transformer_block.py:383-390` | Final layernorm construction, optional later scope. |
| `megatron/core/transformer/transformer_block.py:936-938` | Final layernorm application, optional later scope. |

---

## Out Of Scope For First Patch

1. Gated Attention implementation.
2. Q/K norm gating.
3. MLA compressed-space norm gating.
4. SSM/Mamba norm gating.
5. MTP norm gating.
6. Final-layernorm gating unless explicitly requested.
7. Pretrained checkpoint retrofit or identity-preserving initialization.
8. Fully fused `RMSNorm + gate projection + gated multiply` Triton kernel.

