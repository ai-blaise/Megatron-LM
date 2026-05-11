# StreamBP Native Integration

StreamBP is an opt-in training memory feature for decoder-only GPT models. It
reduces peak activation and LM-head logit memory by replaying backward over
sequence chunks instead of storing a full layer graph. For chunk `[i:j]`, the
attention query is `[i:j]` and the key/value prefix is `[:j]`, preserving causal
decoder semantics.

## Enablement

Use the feature only with zero dropout:

```bash
--use-streambp \
--hidden-dropout 0.0 \
--attention-dropout 0.0 \
--streambp-chunk-size 4096 \
--streambp-logits-chunk-size 4096
```

The flags also have environment mirrors:

```bash
MEGATRON_USE_STREAMBP=1
MEGATRON_STREAMBP_CHUNK_SIZE=4096
MEGATRON_STREAMBP_LOGITS_CHUNK_SIZE=4096
MEGATRON_STREAMBP_SKIP_MOE=0
MEGATRON_STREAMBP_SKIP_DSA=0
MEGATRON_STREAMBP_VALIDATE=1
```

If a chunk size is omitted, Megatron uses:

```text
min(seq_len, min(8192, max(2048, seq_len // 4)))
```

For short sequences this resolves to the full sequence and StreamBP becomes a
normal forward/backward path.

## Compatibility

Supported:

- decoder self-attention with causal masks
- tensor parallel, pipeline parallel, and data parallel
- sequence parallel when context parallel is disabled
- FP8/FP4 quantization contexts through the same per-layer context factory used
  by normal MCore forward
- NCCLX/DDP overlap, including delayed gradient-ready registration until the
  final chunk for each parameter
- FlashAdamW gradient release only through the MCore DDP
  `overlap_grad_reduce=True` bucket path; the non-overlap per-parameter hook is
  rejected because it would step after each chunk
- optional DSA layers when `--no-streambp-skip-dsa` is used and the attention
  implementation accepts rectangular query/prefix masks
- MoE transformer layers through exact full-layer StreamBP replay
- activation-ECO TE hooks with multiple pending chunk forwards
- LM-head loss chunking when fused linear cross entropy is disabled

Rejected by config validation or runtime compatibility guards:

- full activation recompute
- CPU activation or weight offloading
- context parallelism
- hyper connections
- fused single-QKV RoPE
- nonzero hidden or attention dropout
- nonpositive StreamBP chunk sizes
- non-overlap FlashAdamW MCore DDP gradient release

MoE layers are enabled by default. Dense layers use sequence-chunked replay;
MoE layers use exact full-layer replay under the same StreamBP autograd wrapper.
This is deliberate: sparse expert parameters may receive tokens in only some
sequence chunks, so the dense-layer per-chunk DDP readiness counter would be
incorrect for expert weights. Full-layer replay also preserves router aux/z
losses, capacity/drop behavior, input jitter, and expert-bias accounting with
the same full-token statistics as normal MoE forward/backward. Use
`--streambp-skip-moe` only to opt out and run MoE layers on the ordinary path.

## Implementation Notes

The native integration lives in `megatron/core/transformer/streambp.py`.
`TransformerBlock` wraps eligible layers with a custom autograd function during
training. The forward pass runs once without saving the full layer graph. The
backward pass replays each chunk with `chunk_range=(start, end)`.
MoE layers take the sibling full-layer replay path: the forward pass still
omits the full layer graph, and backward replays the full MoE layer once with
the saved CPU/CUDA RNG state.

`TransformerLayer` applies layernorm to the prefix and returns only the chunk
residual path. `Attention` computes prefix QKV, slices Q and output gates to the
query chunk, slices RoPE and attention bias, and builds or slices a rectangular
causal mask. Rectangular causal attention temporarily disables the fused causal
softmax path because that kernel assumes square query/key lengths.

The LM-head helper computes per-token loss by chunking hidden states and logits.
Its custom backward replays each logits chunk and accumulates hidden, output
weight, and tied embedding gradients without materializing full-sequence logits.

## Validation

Focused local validation:

```bash
python -m pytest tests/unit_tests/transformer/test_streambp.py
python -m py_compile \
  megatron/core/transformer/streambp.py \
  megatron/core/transformer/attention.py \
  megatron/core/transformer/transformer_layer.py \
  megatron/core/transformer/transformer_block.py \
  megatron/core/models/gpt/gpt_model.py
```

Production acceptance should run the committed A/B verifier on the target GPU
fleet. It compares the same GPT job with and without StreamBP under MCore DDP
overlap, checks loss and every DDP `main_grad`, then fails unless StreamBP
reduces peak CUDA memory and meets the requested throughput gate.

For the production long-sequence comparison against Megatron full activation
recompute:

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 tools/streambp_prod_verify.py \
  --baseline-full-recompute \
  --seq-len 4096 \
  --num-layers 2 \
  --hidden-size 512 \
  --num-attention-heads 8 \
  --ffn-hidden-size 2048 \
  --streambp-chunk-size 2048 \
  --streambp-logits-chunk-size 4096 \
  --warmup-steps 2 \
  --measure-steps 10 \
  --max-memory-ratio 0.98 \
  --min-throughput-ratio 1.0
```

For a MoE production check, add MoE flags to the same verifier. This path
validates StreamBP with MoE enabled, MCore DDP overlap, loss/main-grad parity,
peak CUDA memory, and throughput:

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 tools/streambp_prod_verify.py \
  --baseline-full-recompute \
  --seq-len 4096 \
  --num-layers 4 \
  --hidden-size 512 \
  --num-attention-heads 8 \
  --ffn-hidden-size 2048 \
  --num-moe-experts 4 \
  --moe-layer-freq 2 \
  --moe-router-topk 2 \
  --moe-ffn-hidden-size 2048 \
  --streambp-chunk-size 2048 \
  --streambp-logits-chunk-size 4096 \
  --warmup-steps 2 \
  --measure-steps 6 \
  --max-memory-ratio 0.98 \
  --min-throughput-ratio 1.0
```

On the b1 H200 production node the dense-only version of this gate passed on
all 8 GPUs: loss matched, maximum DDP `main_grad` absolute difference was
`5.33e-05`, peak CUDA memory was `0.6065x` of the full-recompute baseline, and
throughput was `1.0898x` of that baseline.

The MoE gate above also passed on all 8 GPUs with mixed dense/MoE layers:
loss matched exactly, maximum DDP `main_grad` absolute difference was
`1.006e-04`, peak CUDA memory was `0.5192x` of the full-recompute baseline, and
throughput was `1.0190x` of that baseline. A separate 2-GPU production run with
`--moe-aux-loss-coeff 0.01` passed loss/main-grad, memory, and throughput gates,
covering the router auxiliary-loss replay path.

The MoE validation environment used BF16 local layer specs. Transformer Engine,
NVFP4/FP8 TE kernels, and TorchComms/NCCLX were not installed in that fresh
validation venv, so those runtime paths are not claimed by the MoE production
numbers above. The StreamBP/NCCLX socket compatibility and RDMA-gated results
from the earlier NCCLX validation remain separate transport checks.

The same verifier can compare against a no-recompute baseline by omitting
`--baseline-full-recompute`. That is a stricter speed baseline and should not be
used to claim throughput parity unless it passes in the target deployment; the
current b1 run reduced memory but did not meet throughput parity against
no-recompute vanilla backward.
