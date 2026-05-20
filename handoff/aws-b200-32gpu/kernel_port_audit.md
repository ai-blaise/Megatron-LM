# Kernel Port Audit

This note captures the `optimization-playground` kernel audit performed on 2026-05-20 for the `flashtraining` branch. The source repo inspected was:

`https://github.com/ai-blaise/optimization-playground/tree/main/docs/developer_guide/kernel_results`

## Active Megatron Training Paths

The launch scripts currently wire the relevant paths as follows:

- `MEGATRON_HISA_SELECTOR_BACKEND=deepgemm`
- `MEGATRON_HISA_BMM_CUBLASDX_REFINE=1`
- `MEGATRON_HISA_FUSED_INDEXER_LOSS=1`
- `MEGATRON_HISA_TARGET_TRITON=1`
- `MEGATRON_HISA_KL_GRAD_TRITON=1`
- `MEGATRON_DSA_TRITON=1`
- `MEGATRON_DSA_SPLIT_QK=1`
- `MEGATRON_DSA_STREAMING_INDEXER_TOPK=1`
- `MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD=1`
- `MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD_CHUNK=8192`
- `MEGATRON_DSA_CUDA_SPLIT_QK_ROW_BWD=1`
- `MEGATRON_DSA_CUDA_KV_BWD=0`

The HISA/DSA trainable-indexer path is still enabled; the current changes do not disable DSA semantics or indexer training.

## HIGGS Dense 2-Bit KV

OP source reviewed:

- `docs/developer_guide/kernel_results/higgs.md`
- `python/sglang/jit_kernel/csrc/quantization/higgs_dense_2bit_kv.cuh`

Megatron source touched:

- `megatron/core/quantization/higgs/kernels/csrc/higgs_kv.cuh`
- `megatron/core/quantization/higgs/kernels/csrc/higgs_kv_fwd.cu`

Ported:

- static EDEN2-16 codebook and norm-squared tables in constant memory;
- const-codebook nearest lookup;
- warp-shuffle pair exchange instead of the shared-memory pair-index handoff;
- swizzled shared-memory indexing in the FWHT helper.

Reasoning:

OP's accepted HIGGS path was a compressed KV store kernel. Megatron's active path is fake-quant forward/backward, so only the compatible inner-loop pieces were ported. This preserves saved tensors and backward behavior.

Validation:

```bash
CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13" \
CUDA_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13" \
PATH="$PWD/.venv/bin:$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH" \
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib64:$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}" \
uv run --no-sync pytest \
  tests/unit_tests/quantization/test_higgs_kv.py::test_cuda_forward_matches_reference \
  tests/unit_tests/quantization/test_higgs_kv.py::test_cuda_backward_matches_reference \
  tests/unit_tests/quantization/test_higgs_backward_ste.py::test_cuda_backward_matches_reference_under_bf16_noise -q
```

Result: passed.

Bench result:

| Rows | Fwd Baseline | Fwd Current | Fwd Speedup | Fwd+Bwd Baseline | Fwd+Bwd Current | Fwd+Bwd Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16384 | 0.161 ms | 0.147 ms | 1.10x | 0.316 ms | 0.305 ms | 1.04x |
| 32768 | 0.293 ms | 0.268 ms | 1.09x | 0.572 ms | 0.549 ms | 1.04x |

Peak allocator memory was unchanged.

## HISA / NVFP4 IndexCache

OP source reviewed:

- `docs/developer_guide/kernel_results/nvfp4_hisa_indexcache.md`
- `docs/developer_guide/kernel_results/nvfp4_indexcache_dequant.md`
- `python/sglang/srt/layers/attention/nsa/hisa_tilelang_kernels/hisa.py`
- `python/sglang/jit_kernel/nvfp4_indexer.py`

Megatron source touched:

- `examples/sft/run_sft_deepseek_nvfp4.sh`
- `examples/sft/launch_sft_deepseek_nvfp4_tmux.sh`
- `artifacts/hisa_selector_backend_bench.py`

Decision:

Use the DeepGEMM HISA selector by default:

```bash
MEGATRON_HISA_SELECTOR_BACKEND=deepgemm
```

Reasoning:

OP's best path is built around SGLang paged caches and TileLang/DeepGEMM-style FP4 scoring. Our training path is dense Megatron tensors with IndexCache fake-quant sidecars, but the local `deepgemm` backend is the closest active path to the DeepSeek-style selector. It intentionally validates against an FP4-Q score oracle instead of exact BMM. Exact BMM remains available as an override:

```bash
MEGATRON_HISA_SELECTOR_BACKEND=bmm
MEGATRON_HISA_BMM_CUBLASDX_REFINE=1
```

Validation:

```bash
CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13" \
CUDA_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13" \
PATH="$PWD/.venv/bin:$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH" \
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib64:$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}" \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_hisa_probe \
uv run --no-sync pytest \
  tests/unit_tests/quantization/test_indexcache.py::test_hisa_bmm_dense_cublasdx_refine_matches_bmm -q
```

Additional coverage was added:

- `test_hisa_deepgemm_selector_backend_matches_fp4_oracle`
- `test_hisa_deepgemm_selector_production_config_scores_match_fp4_oracle`
- `test_hisa_bmm_dense_cublasdx_refine_matches_bmm_production_block`

Result: all listed tests passed.

Bench result at `sk=16384`, `topk=512`, `heads=64`, `head_dim=128`, `row_chunk=512`:

| Rows | Backend | Time | Peak Alloc | Notes |
| ---: | --- | ---: | ---: | --- |
| 512 | `bmm` | 9.55 ms | 0.160 GiB | exact baseline |
| 512 | `bmm_cublasdx_refine` | 6.53 ms | 0.094 GiB | exact set match |
| 1024 | `bmm` | 13.95 ms | 0.179 GiB | exact baseline |
| 1024 | `bmm_cublasdx_refine` | 12.57 ms | 0.116 GiB | exact set match |
| 2048 | `bmm` | 29.21 ms | 0.209 GiB | exact baseline |
| 2048 | `bmm_cublasdx_refine` | 24.61 ms | 0.151 GiB | 0.9995 set match, likely near-tie |

DeepGEMM numbers from the same bench:

| Rows | Backend | Time | Peak Alloc | Notes |
| ---: | --- | ---: | ---: | --- |
| 512 | `deepgemm` | 2.14 ms | 0.197 GiB | FP4-Q score oracle, default |
| 1024 | `deepgemm` | 3.89 ms | 0.216 GiB | FP4-Q score oracle, default |
| 2048 | `deepgemm` | 7.98 ms | 0.250 GiB | FP4-Q score oracle, default |

## Not Ported Yet

- GatedNorm OP kernel: OP result is forward/inference-oriented. Megatron training already has a CuTe GatedNorm forward plus training-specific saved-tensor/backward behavior.
- G1 gate/o_proj OP kernel: inference-only FP4 projection fusion. Training needs gradient semantics and gate-state preservation.
- Layersplit / flashsampling / warpdecode: serving or inference paths, not active in current SFT.
- ZeroBubble pipeline schedules as defaults: currently incompatible with fine-grained offload, CPU offload, Transformer Engine, overlapped grad reduce, overlapped param gather, and MoE EP overlap.
- DualPipeV as a default: present and potentially worth probing, but currently a conservative sequential V-topology runtime requiring exactly two virtual stages. It is not yet proven better for this stack.

## Useful Artifacts

- `artifacts/higgs_fake_quant_bench.py`
- `artifacts/hisa_selector_backend_bench.py`
- `artifacts/hisa_selector_backend_bench_bmm_variants_sk16384.jsonl`
- `artifacts/hisa_selector_backend_bench_sk16384.jsonl`
- `artifacts/higgs_fake_quant_bench_current_large.jsonl`
- `artifacts/higgs_fake_quant_bench_baseline_HEAD_cuda_large.jsonl`

Ignore `artifacts/higgs_fake_quant_bench_baseline_HEAD.jsonl`; it hit Python fallback because `ninja` was not in `PATH`.
