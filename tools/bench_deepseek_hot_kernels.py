#!/usr/bin/env python3
"""Microbench DeepSeek-V3.2 SFT hot kernels at the GCP A4 launch shapes."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch


def configure_env(args: argparse.Namespace) -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    cuda_home = repo_dir / ".venv/lib/python3.12/site-packages/nvidia/cu13"
    if "CUDA_HOME" not in os.environ and (cuda_home / "bin/nvcc").exists():
        os.environ["CUDA_HOME"] = str(cuda_home)
    if "CUDA_HOME" in os.environ:
        os.environ.setdefault("CUDA_PATH", os.environ["CUDA_HOME"])
        os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:{os.environ.get('PATH', '')}"
        cuda_libs = f"{os.environ['CUDA_HOME']}/lib64:{os.environ['CUDA_HOME']}/lib"
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_libs}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
    os.environ.setdefault("MEGATRON_DSA_TRITON", "1")
    os.environ.setdefault("MEGATRON_DSA_TRITON_INDEXER", "1")
    os.environ.setdefault("MEGATRON_DSA_SPLIT_QK", "1")
    os.environ.setdefault("MEGATRON_DSA_STREAMING_INDEXER_TOPK", "1")
    os.environ.setdefault("MEGATRON_DSA_INDEXER_ROPE_FUSION", "1")
    os.environ.setdefault("MEGATRON_DSA_INDEXER_ROPE_INPLACE", "1")
    os.environ.setdefault("MEGATRON_DSA_INDEXER_TORCH_K_NORM", "1")
    os.environ["MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE"] = str(args.dsa_key_block_size)
    os.environ.setdefault("MEGATRON_DSA_SORT_TOPK_INDICES", "0")
    os.environ.setdefault("MEGATRON_DSA_COMPACT_TOPK_INDICES", "1")
    os.environ.setdefault("MEGATRON_DSA_TRITON_BF16_GRAD_ATOMICS", "1")
    os.environ.setdefault("MEGATRON_DSA_TRITON_BLOCK_K_BWD", "32")
    os.environ.setdefault("MEGATRON_DSA_TRITON_BWD_NUM_WARPS", "2")
    os.environ.setdefault("MEGATRON_DSA_BACKWARD_TRIM_CACHE", "0")
    os.environ["MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD"] = str(args.dsa_reentrant_kv_bwd)
    os.environ["MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD_CHUNK"] = str(
        args.dsa_kv_bwd_chunk
    )
    os.environ.setdefault("MEGATRON_DSA_SPLIT_QK_REENTRANT_DEFER_QUERY_GRADS", "0")
    os.environ.setdefault("MEGATRON_DSA_SPLIT_QK_REENTRANT_PACK_KV_GRAD", "1")
    os.environ.setdefault("MEGATRON_DSA_CUDA_SPLIT_QK_ROW_BWD", "1")
    os.environ.setdefault("MEGATRON_DSA_CUDA_SPLIT_QK_ROW_QUERY_BWD", "1")
    os.environ.setdefault("MEGATRON_DSA_CUDA_SPLIT_QK_ROW_BWD_WARPS", "8")
    os.environ.setdefault("MEGATRON_DSA_CUDA_KV_BWD", "0")
    os.environ.setdefault("MEGATRON_DSA_VALIDATE_TOPK_INDICES", "0")
    os.environ.setdefault("MEGATRON_DSA_STREAM_TRITON_ATTENTION_CHUNKS", "1")
    os.environ.setdefault("MEGATRON_DSA_SP_PROJECT_BEFORE_GATHER", "1")
    os.environ.setdefault("MEGATRON_DSA_TEACHER_SCORE_SCRATCH", "1")
    os.environ.setdefault("MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH", "1")
    os.environ.setdefault("MEGATRON_HISA_CANDIDATE_SLOT_GROUP", "8")
    os.environ.setdefault("MEGATRON_HISA_CANDIDATE_MAX_TEMP_MB", "0")
    os.environ.setdefault("MEGATRON_HISA_CANDIDATE_FREE_MEM_RESERVE_MB", "0")
    os.environ.setdefault("MEGATRON_HISA_CANDIDATE_TRIM_CACHE", "0")
    os.environ.setdefault("MEGATRON_HISA_COMPACT_CANDIDATE_TOPK", "1")
    os.environ["MEGATRON_HISA_SELECTOR_BACKEND"] = args.hisa_backend
    os.environ.setdefault("MEGATRON_HISA_SELECTOR_CUDA", "1")
    os.environ["MEGATRON_HISA_MEGAKERNEL_STREAMING_SCRATCH"] = str(
        args.hisa_megakernel_streaming_scratch
    )
    os.environ["MEGATRON_HISA_SELECTOR_ROW_CHUNK"] = str(args.hisa_selector_row_chunk)
    os.environ.setdefault("MEGATRON_HISA_BMM_FP32_ACCUM_TENSORCORES", "1")
    os.environ.setdefault("MEGATRON_HISA_BMM_CUBLASDX_REFINE", "1")
    os.environ.setdefault("MEGATRON_HISA_SELECTED_SCORE_BWD_HEAD_GROUP", "8")
    os.environ.setdefault("MEGATRON_HISA_SELECTED_SCORE_BWD_WARP_GROUPED", "1")
    os.environ.setdefault("MEGATRON_HISA_SELECTED_SCORE_BWD_CUBLASDX_TILE_N", "32")
    os.environ.setdefault("MEGATRON_HISA_ASSUME_SORTED_POSITIONS", "1")
    os.environ.setdefault("MEGATRON_HISA_FALLBACK_DENSE_IF_SHORT", "0")
    os.environ.setdefault("MEGATRON_HISA_FUSED_INDEXER_LOSS", "1")
    os.environ.setdefault("MEGATRON_HISA_TARGET_TRITON", "1")
    os.environ.setdefault("MEGATRON_HISA_TARGET_BLOCK_K", "64")
    os.environ["MEGATRON_HISA_TARGET_ROW_CHUNK"] = str(args.hisa_target_row_chunk)
    os.environ.setdefault("MEGATRON_HISA_KL_GRAD_TRITON", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")


def mib() -> float:
    return 1024.0 * 1024.0


@contextmanager
def cuda_peak_scope(device: torch.device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    yield
    torch.cuda.synchronize(device)


def summarize(times_ms: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(times_ms),
        "p50_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "max_ms": max(times_ms),
    }


def bench_cuda(
    name: str,
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    torch.cuda.empty_cache()
    for _ in range(warmup):
        out = fn()
        if isinstance(out, torch.Tensor):
            out.detach()
    torch.cuda.synchronize(device)

    times: list[float] = []
    with cuda_peak_scope(device):
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            if isinstance(out, torch.Tensor):
                out.detach()
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
        peak_mb = torch.cuda.max_memory_allocated(device) / mib()
    result = {"name": name, "iters": iters, "peak_allocated_mb": peak_mb}
    result.update(summarize(times))
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def make_tail_topk(
    *,
    batch: int,
    q_len: int,
    sk: int,
    topk: int,
    q_start: int,
    device: torch.device,
) -> torch.Tensor:
    prefix_lens = torch.arange(q_start + 1, q_start + q_len + 1, device=device)
    offsets = torch.arange(topk, device=device)
    rows = (prefix_lens[:, None] - topk + offsets[None, :]).clamp_min(0)
    return rows.to(torch.int32).unsqueeze(0).expand(batch, -1, -1).contiguous()


def bench_higgs(args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    from megatron.core.quantization.higgs.autograd import apply_higgs_dense_2bit_kv
    from megatron.core.quantization.higgs.codec import build_higgs_buffers

    buffers = build_higgs_buffers(device=device, dtype=torch.float32)
    results = []
    for rows in (args.local_rows, args.full_rows):
        x = torch.randn(rows, args.kv_lora_rank, device=device, dtype=torch.bfloat16)
        grad = torch.randn_like(x)

        def fwd():
            return apply_higgs_dense_2bit_kv(x, buffers)

        y = fwd()

        def fwd_bwd():
            x_req = x.detach().clone().requires_grad_(True)
            out = apply_higgs_dense_2bit_kv(x_req, buffers)
            out.backward(grad)
            return x_req.grad

        results.append(
            bench_cuda(
                f"higgs_dense_2bit_fwd rows={rows} dim={args.kv_lora_rank}",
                fwd,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
        )
        del y
        results.append(
            bench_cuda(
                f"higgs_dense_2bit_fwd_bwd rows={rows} dim={args.kv_lora_rank}",
                fwd_bwd,
                device=device,
                warmup=max(1, args.warmup // 2),
                iters=max(1, args.iters // 2),
            )
        )
        del x, grad
    return results


def bench_indexcache(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, object]], torch.Tensor]:
    from megatron.core.quantization.indexcache.autograd import (
        apply_indexcache_kv,
        get_indexcache_nvfp4_packed_tensors,
    )
    from megatron.core.quantization.indexcache.codec import build_indexcache_config

    config = build_indexcache_config(eps=args.indexcache_eps, quantization="nvfp4_e2m1_ue8m0")
    k = torch.randn(args.sk, args.batch, args.indexer_head_dim, device=device, dtype=torch.bfloat16)
    grad = torch.randn_like(k)

    def fwd():
        return apply_indexcache_kv(k, config)

    k_quant = fwd()
    packed = get_indexcache_nvfp4_packed_tensors(k_quant[:, 0])
    if packed is None:
        raise RuntimeError("IndexCache NVFP4 packed sidecar was not attached")

    def fwd_bwd():
        k_req = k.detach().clone().requires_grad_(True)
        out = apply_indexcache_kv(k_req, config)
        out.backward(grad)
        return k_req.grad

    results = [
        bench_cuda(
            f"indexcache_nvfp4_fwd rows={args.full_rows} dim={args.indexer_head_dim}",
            fwd,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        ),
        bench_cuda(
            f"indexcache_nvfp4_fwd_bwd rows={args.full_rows} dim={args.indexer_head_dim}",
            fwd_bwd,
            device=device,
            warmup=max(1, args.warmup // 2),
            iters=max(1, args.iters // 2),
        ),
    ]
    del k, grad
    return results, k_quant


def bench_hisa(
    args: argparse.Namespace,
    device: torch.device,
    k_quant: torch.Tensor,
) -> list[dict[str, object]]:
    from megatron.core.quantization.indexcache.hisa import IndexCacheHISAConfig
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        _HISASelectWithScoresBatched,
    )

    config = IndexCacheHISAConfig(
        enabled=True,
        block_size=args.hisa_block_size,
        block_topk=args.hisa_block_topk,
        compression_ratio=args.hisa_compression_ratio,
        topk_tokens=args.topk,
        execution_mode="optimized",
        fallback_to_dense_if_short=False,
        forced_boundary_blocks=("first", "last"),
    )
    q = torch.randn(
        args.q_len,
        args.batch,
        args.indexer_heads,
        args.indexer_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        args.q_len,
        args.batch,
        args.indexer_heads,
        device=device,
        dtype=torch.bfloat16,
    )
    prefix_lens = torch.arange(
        args.q_start + 1, args.q_start + args.q_len + 1, device=device, dtype=torch.long
    ).clamp(max=args.sk)

    def fwd():
        result = _HISASelectWithScoresBatched.apply(
            q,
            weights,
            k_quant,
            prefix_lens,
            None,
            int(args.topk),
            config,
        )
        return result[1]

    q_req = q.detach().clone().requires_grad_(True)
    w_req = weights.detach().clone().requires_grad_(True)
    grad_selected_scores = torch.randn_like(fwd())

    def fwd_bwd():
        q_req.grad = None
        w_req.grad = None
        _topk, selected_scores = _HISASelectWithScoresBatched.apply(
            q_req,
            w_req,
            k_quant,
            prefix_lens,
            None,
            int(args.topk),
            config,
        )
        selected_scores.backward(grad_selected_scores)
        return q_req.grad

    return [
        bench_cuda(
            (
                "hisa_batched_select_with_scores_fwd "
                f"backend={os.environ.get('MEGATRON_HISA_SELECTOR_BACKEND')} "
                f"q={args.q_len} sk={args.sk} batch={args.batch} "
                f"heads={args.indexer_heads} topk={args.topk} "
                f"row_chunk={os.environ.get('MEGATRON_HISA_SELECTOR_ROW_CHUNK')}"
            ),
            fwd,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        ),
        bench_cuda(
            (
                "hisa_batched_select_with_scores_fwd_bwd "
                f"backend={os.environ.get('MEGATRON_HISA_SELECTOR_BACKEND')} "
                f"q={args.q_len} sk={args.sk} batch={args.batch} "
                f"heads={args.indexer_heads} topk={args.topk} "
                f"row_chunk={os.environ.get('MEGATRON_HISA_SELECTOR_ROW_CHUNK')}"
            ),
            fwd_bwd,
            device=device,
            warmup=1,
            iters=max(1, args.iters // 2),
        ),
    ]


def bench_dsa_indexer(args: argparse.Namespace, device: torch.device, k_quant: torch.Tensor) -> list[dict[str, object]]:
    from megatron.core.transformer.experimental_attention_variant.dsa_triton import (
        dsa_indexer_scores_triton,
    )

    q = torch.randn(
        args.q_len,
        args.batch,
        args.indexer_heads,
        args.indexer_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        args.q_len,
        args.batch,
        args.indexer_heads,
        device=device,
        dtype=torch.bfloat16,
    )
    out = torch.empty(args.batch, args.q_len, args.sk, device=device, dtype=torch.float32)

    def fwd():
        return dsa_indexer_scores_triton(q, weights, k_quant, q_start=args.q_start, out=out)

    return [
        bench_cuda(
            (
                "dsa_indexer_scores_triton "
                f"q={args.q_len} sk={args.sk} batch={args.batch} "
                f"heads={args.indexer_heads} dim={args.indexer_head_dim}"
            ),
            fwd,
            device=device,
            warmup=max(1, args.warmup // 2),
            iters=max(1, args.iters // 2),
        )
    ]


def bench_dsa_attention(args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    from megatron.core.transformer.experimental_attention_variant.dsa_triton import (
        sparse_dsa_attention_split_qk_triton,
        sparse_dsa_attention_split_qk_with_teacher_triton,
    )

    q_nope = torch.randn(
        args.q_len,
        args.batch,
        args.local_attention_heads,
        args.qk_nope_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    q_pe = torch.randn(
        args.q_len,
        args.batch,
        args.local_attention_heads,
        args.qk_pos_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    if args.dsa_packed_kv_ref:
        kv_ref = torch.randn(
            args.sk,
            args.batch,
            args.local_attention_heads,
            args.qk_nope_dim + args.value_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        k_nope = kv_ref[..., : args.qk_nope_dim]
        value = kv_ref[..., args.qk_nope_dim :]
    else:
        kv_ref = None
        k_nope = torch.randn(
            args.sk,
            args.batch,
            args.local_attention_heads,
            args.qk_nope_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        value = torch.randn(
            args.sk,
            args.batch,
            args.local_attention_heads,
            args.value_dim,
            device=device,
            dtype=torch.bfloat16,
        )
    k_pe = torch.randn(args.sk, args.batch, 1, args.qk_pos_dim, device=device, dtype=torch.bfloat16)
    topk = make_tail_topk(
        batch=args.batch,
        q_len=args.q_len,
        sk=args.sk,
        topk=args.topk,
        q_start=args.q_start,
        device=device,
    )
    softmax_scale = 1.0 / math.sqrt(args.qk_nope_dim + args.qk_pos_dim)
    def fwd():
        return sparse_dsa_attention_split_qk_triton(
            q_nope,
            q_pe,
            k_nope,
            k_pe,
            value,
            topk,
            softmax_scale,
            q_start=args.q_start,
            kv_nope_value_ref=kv_ref,
        )

    def fwd_teacher():
        return sparse_dsa_attention_split_qk_with_teacher_triton(
            q_nope,
            q_pe,
            k_nope,
            k_pe,
            value,
            topk,
            softmax_scale,
            q_start=args.q_start,
            kv_nope_value_ref=kv_ref,
        )[0]

    def attention_for_backward(qn, qp, kn, kp, v, kv_ref_req):
        if args.dsa_use_teacher:
            return sparse_dsa_attention_split_qk_with_teacher_triton(
                qn,
                qp,
                kn,
                kp,
                v,
                topk,
                softmax_scale,
                q_start=args.q_start,
                kv_nope_value_ref=kv_ref_req,
            )[0]
        return sparse_dsa_attention_split_qk_triton(
            qn,
            qp,
            kn,
            kp,
            v,
            topk,
            softmax_scale,
            q_start=args.q_start,
            kv_nope_value_ref=kv_ref_req,
        )

    q_nope_req = q_nope.detach().clone().requires_grad_(True)
    q_pe_req = q_pe.detach().clone().requires_grad_(True)
    k_pe_req = k_pe.detach().clone().requires_grad_(True)
    grad_output = torch.randn(
        args.q_len,
        args.batch,
        args.local_attention_heads * args.value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    grad_components = {
        item.strip()
        for item in args.dsa_grad_components.split(",")
        if item.strip()
    }
    need_query_grad = "query" in grad_components
    need_kv_grad = bool(grad_components.intersection({"key_nope", "value", "kv"}))
    need_key_pe_grad = "key_pe" in grad_components
    if args.dsa_packed_kv_ref:
        kv_req = kv_ref.detach().clone().requires_grad_(need_kv_grad)
        k_nope_req = None
        value_req = None
    else:
        kv_req = None
        k_nope_req = k_nope.detach().clone().requires_grad_(
            "key_nope" in grad_components or "kv" in grad_components
        )
        value_req = value.detach().clone().requires_grad_(
            "value" in grad_components or "kv" in grad_components
        )
    q_nope_req.requires_grad_(need_query_grad)
    q_pe_req.requires_grad_(need_query_grad)
    k_pe_req.requires_grad_(need_key_pe_grad)

    def fwd_bwd():
        q_nope_req.grad = None
        q_pe_req.grad = None
        k_pe_req.grad = None
        if args.dsa_packed_kv_ref:
            assert kv_req is not None
            kv_req.grad = None
            kn = kv_req[..., : args.qk_nope_dim]
            v = kv_req[..., args.qk_nope_dim :]
        else:
            assert k_nope_req is not None and value_req is not None
            k_nope_req.grad = None
            value_req.grad = None
            kn = k_nope_req
            v = value_req
        out = attention_for_backward(
            q_nope_req,
            q_pe_req,
            kn,
            k_pe_req,
            v,
            kv_req,
        )
        out.backward(grad_output)
        return q_nope_req.grad if q_nope_req.grad is not None else out.detach()

    return [
        bench_cuda(
            (
                "dsa_split_qk_attention_fwd "
                f"q={args.q_len} sk={args.sk} batch={args.batch} "
                f"heads={args.local_attention_heads} topk={args.topk}"
            ),
            fwd,
            device=device,
            warmup=max(1, args.warmup // 2),
            iters=max(1, args.iters // 2),
        ),
        bench_cuda(
            (
                "dsa_split_qk_attention_teacher_fwd "
                f"q={args.q_len} sk={args.sk} batch={args.batch} "
                f"heads={args.local_attention_heads} topk={args.topk}"
            ),
            fwd_teacher,
            device=device,
            warmup=max(1, args.warmup // 2),
            iters=max(1, args.iters // 2),
        ),
        bench_cuda(
            (
                "dsa_split_qk_attention_fwd_bwd "
                f"q={args.q_len} sk={args.sk} batch={args.batch} "
                f"heads={args.local_attention_heads} topk={args.topk} "
                f"teacher={int(args.dsa_use_teacher)}"
            ),
            fwd_bwd,
            device=device,
            warmup=1,
            iters=max(1, args.iters // 3),
        ),
    ]


def parse_args() -> argparse.Namespace:
    def env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            return default

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-len", type=int, default=env_int("SEQ_LENGTH", 32768))
    parser.add_argument("--tp", type=int, default=env_int("TP", 4))
    parser.add_argument("--cp", type=int, default=env_int("CP", 2))
    parser.add_argument("--batch", type=int, default=env_int("MICRO_BATCH_SIZE", 2))
    parser.add_argument("--q-len", type=int, default=0)
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--indexer-heads", type=int, default=64)
    parser.add_argument("--indexer-head-dim", type=int, default=128)
    parser.add_argument("--total-attention-heads", type=int, default=128)
    parser.add_argument("--qk-nope-dim", type=int, default=128)
    parser.add_argument("--qk-pos-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--kv-lora-rank", type=int, default=2048)
    parser.add_argument("--hisa-block-size", type=int, default=128)
    parser.add_argument("--hisa-block-topk", type=int, default=64)
    parser.add_argument("--hisa-compression-ratio", type=float, default=4.0)
    parser.add_argument(
        "--hisa-selector-row-chunk",
        type=int,
        default=env_int("MEGATRON_HISA_SELECTOR_ROW_CHUNK", 256),
    )
    parser.add_argument(
        "--hisa-target-row-chunk",
        type=int,
        default=env_int("MEGATRON_HISA_TARGET_ROW_CHUNK", 256),
    )
    parser.add_argument("--hisa-backend", default="bmm")
    parser.add_argument(
        "--hisa-megakernel-streaming-scratch",
        type=int,
        default=env_int("MEGATRON_HISA_MEGAKERNEL_STREAMING_SCRATCH", 0),
    )
    parser.add_argument("--dsa-key-block-size", type=int, default=4096)
    parser.add_argument(
        "--dsa-kv-bwd-chunk",
        type=int,
        default=env_int("MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD_CHUNK", 32768),
    )
    parser.add_argument(
        "--dsa-reentrant-kv-bwd",
        type=int,
        choices=(0, 1),
        default=env_int("MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD", 0),
        help="Mirror the launcher split-Q/K DSA reentrant K/V backward setting.",
    )
    parser.add_argument(
        "--no-dsa-packed-kv-ref",
        action="store_false",
        dest="dsa_packed_kv_ref",
        help="Benchmark split-QK DSA with independent K-noPE/V tensors instead of packed MLA KV.",
    )
    parser.set_defaults(dsa_packed_kv_ref=True)
    parser.add_argument(
        "--dsa-grad-components",
        default="query,key_nope,key_pe,value",
        help=(
            "Comma-separated DSA fwd+bwd gradient components to request: "
            "query,key_nope,key_pe,value,kv. With packed KV, key_nope/value are one leaf."
        ),
    )
    parser.add_argument(
        "--dsa-use-teacher",
        action="store_true",
        help="Benchmark DSA fwd+bwd through the with-teacher path used by HISA indexer loss.",
    )
    parser.add_argument("--indexcache-eps", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--components",
        default="higgs,indexcache,hisa,dsa-indexer,dsa-attn",
        help="Comma-separated: higgs,indexcache,hisa,dsa-indexer,dsa-attn",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    args.q_len = args.q_len or (args.seq_len // args.cp)
    args.sk = args.seq_len
    args.q_start = args.sk - args.q_len
    args.local_rows = args.q_len * args.batch
    args.full_rows = args.sk * args.batch
    args.local_attention_heads = args.total_attention_heads // args.tp
    configure_env(args)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(1234)

    meta = {
        "device": torch.cuda.get_device_name(device),
        "seq_len": args.seq_len,
        "tp": args.tp,
        "cp": args.cp,
        "q_len": args.q_len,
        "q_start": args.q_start,
        "sk": args.sk,
        "batch": args.batch,
        "topk": args.topk,
        "indexer_heads": args.indexer_heads,
        "local_attention_heads": args.local_attention_heads,
        "hisa_backend": os.environ.get("MEGATRON_HISA_SELECTOR_BACKEND"),
        "hisa_selector_row_chunk": os.environ.get("MEGATRON_HISA_SELECTOR_ROW_CHUNK"),
        "hisa_megakernel_streaming_scratch": os.environ.get(
            "MEGATRON_HISA_MEGAKERNEL_STREAMING_SCRATCH"
        ),
        "hisa_target_row_chunk": os.environ.get("MEGATRON_HISA_TARGET_ROW_CHUNK"),
        "dsa_key_block_size": os.environ.get("MEGATRON_DSA_INDEXER_KEY_BLOCK_SIZE"),
        "dsa_kv_bwd_chunk": os.environ.get("MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD_CHUNK"),
        "dsa_reentrant_kv_bwd": os.environ.get("MEGATRON_DSA_SPLIT_QK_REENTRANT_KV_BWD"),
        "dsa_defer_query_grads": os.environ.get(
            "MEGATRON_DSA_SPLIT_QK_REENTRANT_DEFER_QUERY_GRADS"
        ),
        "dsa_row_query_bwd": os.environ.get("MEGATRON_DSA_CUDA_SPLIT_QK_ROW_QUERY_BWD"),
        "dsa_row_bwd_warps": os.environ.get("MEGATRON_DSA_CUDA_SPLIT_QK_ROW_BWD_WARPS"),
        "dsa_pe_cublasdx_fwd": os.environ.get("MEGATRON_DSA_CUDA_SPLIT_QK_PE_CUBLASDX_FWD"),
        "dsa_cublasdx_fwd": os.environ.get("MEGATRON_DSA_CUDA_SPLIT_QK_CUBLASDX_FWD"),
        "dsa_teacher_score_scratch": os.environ.get("MEGATRON_DSA_TEACHER_SCORE_SCRATCH"),
        "dsa_bwd_score_scratch": os.environ.get("MEGATRON_DSA_TRITON_BWD_SCORE_SCRATCH"),
        "hisa_candidate_slot_group": os.environ.get("MEGATRON_HISA_CANDIDATE_SLOT_GROUP"),
        "hisa_candidate_max_temp_mb": os.environ.get("MEGATRON_HISA_CANDIDATE_MAX_TEMP_MB"),
        "hisa_candidate_free_mem_reserve_mb": os.environ.get(
            "MEGATRON_HISA_CANDIDATE_FREE_MEM_RESERVE_MB"
        ),
        "hisa_candidate_trim_cache": os.environ.get("MEGATRON_HISA_CANDIDATE_TRIM_CACHE"),
        "hisa_selected_score_bwd_tile_n": os.environ.get(
            "MEGATRON_HISA_SELECTED_SCORE_BWD_CUBLASDX_TILE_N"
        ),
        "dsa_packed_kv_ref": args.dsa_packed_kv_ref,
        "dsa_grad_components": args.dsa_grad_components,
        "dsa_use_teacher": args.dsa_use_teacher,
    }
    print(json.dumps({"meta": meta}, sort_keys=True), flush=True)

    components = {item.strip() for item in args.components.split(",") if item.strip()}
    results: list[dict[str, object]] = []
    k_quant = None
    t0 = time.time()

    if "higgs" in components:
        results.extend(bench_higgs(args, device))
    if components.intersection({"indexcache", "hisa", "dsa-indexer"}):
        index_results, k_quant = bench_indexcache(args, device)
        if "indexcache" in components:
            results.extend(index_results)
    if "hisa" in components:
        assert k_quant is not None
        results.extend(bench_hisa(args, device, k_quant))
    if "dsa-indexer" in components:
        assert k_quant is not None
        results.extend(bench_dsa_indexer(args, device, k_quant))
    if "dsa-attn" in components:
        results.extend(bench_dsa_attention(args, device))

    print(json.dumps({"elapsed_s": time.time() - t0, "results": results}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
