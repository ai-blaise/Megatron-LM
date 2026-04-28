# GatedNorm Notes

## Contract

```text
y = normed
z = y @ W_down.T
a = silu(z)
gate = sigmoid(a @ W_up.T)
output = y * gate
```

Shapes:

```text
y:      [tokens, hidden]
W_down: [rank, hidden]
W_up:   [hidden, rank]
z:      [tokens, rank]
output: [tokens, hidden]
```

Boundaries:

- `normed` is already RMSNorm output; GatedNorm does not compute RMSNorm.
- Runtime API: `apply_gated_norm(normed, w_down, w_up)`.
- Runtime path is Triton-first, not a PyTorch fallback.
- Save rank-sized `z`; do not save full-width `gate` or `gate_logits`.

## What Fused Means

Fused here means fewer production kernel launches and less HBM materialization.

- Forward: one Triton kernel for down projection, SiLU, up projection, sigmoid, and multiply.
- Backward: one Triton kernel launch computes `dy`, `dW_down`, and `dW_up`.
- Backward still has two internal hidden-block passes because `dy` needs completed `dz`.
- This is still fused because it avoids separate gradient kernels and avoids writing full-width gate/logit tensors.

Backward formulas:

```text
dlogits = dout * y * g * (1 - g)
dW_up   = dlogits.T @ a
da      = dlogits @ W_up
dz      = da * silu_grad(z)
dW_down = dz.T @ y
dy      = dout * g + dz @ W_down
```

## Validation

Fusion tests:

```bash
uv run --no-sync python -m pytest -q tests/unit_tests/fusions/test_gated_norm.py
```

Transformer integration tests:

```bash
uv run --no-sync python -m pytest -q tests/unit_tests/transformer/test_gated_norm.py
```

Combined:

```bash
uv run --no-sync python -m pytest -q \
    tests/unit_tests/fusions/test_gated_norm.py \
    tests/unit_tests/transformer/test_gated_norm.py
```

Non-CUDA or driver-broken expected result:

```text
9 passed, 7 skipped
```

CUDA-only checks compare Triton forward and gradients for `normed`, `W_down`, and
`W_up` against the PyTorch reference formula. If CUDA skips, structure and integration
are tested, but Triton numerical correctness is not.

## Smoke

Requires a working NVIDIA driver and `torch.cuda.is_available() == True`.

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

Add recompute coverage with:

```bash
--recompute-granularity selective --recompute-modules layernorm
```

CUDA probe:

```bash
uv run --no-sync python - <<'PY'
import torch
import triton

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
print("triton", triton.__version__)
PY
```
