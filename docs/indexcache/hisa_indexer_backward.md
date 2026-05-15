# NVFP4 HISA 4:1 Indexer Backward

This extension adds a configurable training-time backward for the NVFP4
IndexCache + HISA selector. It reuses `IndexCacheHISAConfig`; the existing
IndexCache path remains the fallback.

## Contract

- Pool block size is `B=128`.
- For each query row, `M` is the number of prefix-visible pool blocks.
- Dynamic HISA keeps `m=ceil(M / compression_ratio)` blocks. The default
  `compression_ratio=4.0` therefore applies at all `t > k`, even when the
  resulting candidate pool has fewer than `topk_tokens`.
- If every row has `t <= k`, the selector returns the dense fallback sentinel so
  callers use ordinary NVFP4 IndexCache behavior.
- Stage-1 masking is prefix-local. Blocks outside the row's visible prefix are
  not eligible even if they are present in the backing K cache.
- The default boundary rule matches the TileLang HISA reference:
  `forced_boundary_blocks=("first", "last")`. `last_minus_one` is still parsed
  as an explicit opt-in variant, not the default 4:1 path.

## Backward

The CUDA kernel implements the analytic Jacobian for the two weighted-ReLU DSA
score formulas and the mean-pool stage:

```text
score(t, s) = sum_h w[t, h] * ReLU(dot(q[t, h], k[s]))
k_block[b] = mean(k[s] for s in block b)
```

Top-k and block selection are straight-through boundaries: the saved selection
masks are treated as constants. The backward returns dense gradients for
`Q`, `K`, and the indexer weights.

Production sparse-MLA backpropagation routes gradients from selected token
scores into the candidate-score tensor only when the forward path ran the
candidate refine stage. When `m * B <= topk_tokens`, the OP inference kernel
uses the map-all path and selected-token scores are not materialized, so the
production candidate-score gradient is zero. The block-top-k boundary is
straight-through, so the production block-score gradient is also zero. The CUDA
path returns early for zero candidate or block score gradients before doing
scatter or mean-pool work. Analytical validation still covers explicit
non-zero candidate and block-score gradients.

## B200 Backward Performance

The accepted comparator is ordinary NVFP4 IndexCache versus NVFP4
IndexCache+HISA. Both arms use K produced by `indexcache_nvfp4_fwd`; dense or
unquantized K baselines are not acceptance comparators.

B200, CUDA 13.0, torch `2.13.0.dev20260511+cu130`, `q_rows=64`, `heads=64`,
`head_dim=128`, `topk=2048`, `B=128`, `compression_ratio=4.0`, 2 warmup and
5 timed iterations:

| Prefix length | Ordinary NVFP4 total bwd | HISA total bwd | HISA speedup |
| ---: | ---: | ---: | ---: |
| 4096 | 1.8004 ms | 0.9881 ms | 1.82x |
| 8192 | 2.1984 ms | 0.9663 ms | 2.28x |
| 16384 | 3.2850 ms | 3.0396 ms | 1.08x |
| 32768 | 6.5273 ms | 3.1218 ms | 2.09x |
| 65536 | 12.9347 ms | 3.4112 ms | 3.79x |

Score-backward medians for the same run were
0.8379/1.6426/3.2470/6.4576/12.9272 ms for ordinary NVFP4 IndexCache and
0.0326/0.0366/2.9659/3.0718/3.3314 ms for HISA. The 4096 and 8192 rows use the
map-all path (`m * B <= topk_tokens`), so production score gradients are empty
sentinels and no score kernel is launched.

Top-k=1024 was checked separately on the B200 with the score+packed-quant
backward gate (`q_rows=64`, `heads=64`, `head_dim=128`, `block_grad=zero`, 8
warmup and 30 timed iterations). The comparator remains ordinary NVFP4
IndexCache versus NVFP4 IndexCache+HISA, with both arms consuming K from
`indexcache_nvfp4_fwd`.

| Prefix length | Ordinary score bwd | HISA score bwd | Score speedup | Ordinary total bwd | HISA total bwd | Total speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 5.9133 ms | 1.1275 ms | 5.24x | 5.9028 ms | 1.1327 ms | 5.21x |
| 8192 | 12.4031 ms | 2.7591 ms | 4.50x | 12.4074 ms | 2.7681 ms | 4.48x |
| 16384 | 25.2160 ms | 6.0337 ms | 4.18x | 25.2263 ms | 6.0468 ms | 4.17x |
| 32768 | 50.6971 ms | 12.5480 ms | 4.04x | 50.7160 ms | 12.5692 ms | 4.03x |
| 65536 | 103.1021 ms | 25.8020 ms | 4.00x | 103.1385 ms | 25.8465 ms | 3.99x |

The post-acceptance stress loop also ran with selected block-score gradients.
HISA stayed ahead of ordinary NVFP4 IndexCache at every tested shape, with
total backward speedups of 4.15x, 3.85x, 3.67x, 3.65x, and 3.60x for
4K/8K/16K/32K/64K respectively.

## Rule 6 Notes

- TileLang `examples/dsa_hisa` is the reference for two-stage HISA orchestration
  and first/last valid block masking.
- The HISA paper and OP implementation define the block-pool then token-refine
  structure; this Megatron path preserves the local 4:1 contract above.
- CUTLASS/CuTe Blackwell NVFP4 material is GEMM/tensor-core focused. This
  backward is row-local reduction/scatter work, so the initial kernel uses
  warp-level SIMT reductions and atomics.
- DeepGEMM's indexer and JIT layout informed the benchmark shape and runtime
  build style.
- IKP should be used for further latency work around candidate-score atomics and
  block-score mean-pool recomputation.
