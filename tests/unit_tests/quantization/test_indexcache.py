# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for IndexCache fake-quant.

Covers:
  * FP8 forward output shape, finiteness, and quantization-error bounds
  * eps clamp on all-zero rows
  * FP8 analytic backward vs torch.autograd on the STE-detach forward (the same
    trick we used for TurboQuant — finite differences cannot validate STE on
    a quantized op because the forward is locally flat between fp8 levels)
  * NVFP4 packed value/scale layout and the four-group STE backward
  * SGLang reference parity: when running on a host with fp8_e4m3fn, the
    Megatron forward is bit-identical to the SGLang Triton _act_quant kernel
    output (post-dequantize), modulo cast ordering
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from megatron.core.quantization.indexcache import (  # noqa: E402
    INDEXCACHE_QUANT_DISABLED,
    INDEXCACHE_QUANT_FP8,
    INDEXCACHE_QUANT_NVFP4,
    IndexCacheConfig,
    IndexCacheHISAConfig,
    apply_indexcache_kv,
    build_indexcache_config,
    hisa_block_topk_counts,
    indexcache_hisa_cuda_select_with_scores,
    indexcache_hisa_topk,
    indexcache_hisa_topk_with_scores,
    resolve_indexcache_quantization,
)
from megatron.core.extensions.hisa_indexer.reference import hisa_forward_reference  # noqa: E402
from megatron.core.quantization.indexcache.reference import (  # noqa: E402
    indexcache_backward,
    indexcache_forward,
)


HEAD_DIM = 128


def _make_cfg():
    return build_indexcache_config()


def _make_nvfp4_cfg():
    return build_indexcache_config(quantization=INDEXCACHE_QUANT_NVFP4)


def test_forward_shape_and_finite():
    cfg = _make_cfg()
    torch.manual_seed(0)
    x = torch.randn(16, HEAD_DIM, dtype=torch.float32)
    y = indexcache_forward(x, cfg)
    assert y.shape == x.shape
    assert torch.isfinite(y).all().item()


def test_quantization_error_bounded():
    """fp8 e4m3 has ~3-bit mantissa; relative error per coord under ~5%."""
    cfg = _make_cfg()
    torch.manual_seed(0)
    x = torch.randn(64, HEAD_DIM, dtype=torch.float32)
    y = indexcache_forward(x, cfg)
    err = (y - x).abs() / (x.abs() + 1e-3)
    assert err.mean().item() < 0.10
    assert err.max().item() < 1.0


def test_zero_row_handled_by_eps():
    """All-zero token must not divide by zero — eps clamp guards the scale."""
    cfg = _make_cfg()
    x = torch.zeros(4, HEAD_DIM, dtype=torch.float32)
    y, intermed = indexcache_forward(x, cfg, return_intermediates=True)
    # Scale is amax/fp8_max with amax clamped to eps, so scale = eps/fp8_max.
    expected_scale = cfg.eps * cfg.fp8_max_inv
    assert torch.allclose(intermed["scale"], torch.full_like(intermed["scale"], expected_scale))
    assert torch.isfinite(y).all().item()
    # eps_active should be 0 on every all-zero row.
    assert (intermed["eps_active"] == 0).all().item()


def test_autograd_function_shape_preserves():
    cfg = _make_cfg()
    torch.manual_seed(0)
    for shape in [(HEAD_DIM,), (4, HEAD_DIM), (3, 2, HEAD_DIM), (2, 5, 7, HEAD_DIM)]:
        x = torch.randn(*shape, requires_grad=True)
        y = apply_indexcache_kv(x, cfg)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


def test_autograd_finite_grads():
    cfg = _make_cfg()
    torch.manual_seed(0)
    x = torch.randn(8, HEAD_DIM, requires_grad=True)
    y = apply_indexcache_kv(x, cfg)
    upstream = torch.randn_like(y)
    (y * upstream).sum().backward()
    assert torch.isfinite(x.grad).all().item()
    # Gradient should be non-trivial (most coords pass the STE).
    assert x.grad.abs().sum().item() > 0


def _ste_autograd_forward(x: torch.Tensor, cfg: IndexCacheConfig) -> torch.Tensor:
    """Differentiable forward whose autograd matches our analytic backward.

    Uses ``(quantized - x_clip).detach() + x_clip`` to route gradient through
    the clipped pre-quant value while the forward output equals the quantized
    value. Same idiom we used for TurboQuant's STE oracle.
    """

    abs_max = x.abs().amax(dim=-1)
    amax = abs_max.clamp_min(cfg.eps)
    scale = (amax * cfg.fp8_max_inv)[:, None]
    pre_clip = x / scale
    clipped = pre_clip.clamp(-cfg.fp8_max, cfg.fp8_max)
    if x.is_cuda or hasattr(torch, "float8_e4m3fn"):
        q_fp8 = clipped.to(torch.float8_e4m3fn).to(x.dtype)
    else:
        from megatron.core.quantization.indexcache.reference import (
            _simulate_fp8_e4m3_rounding,
        )
        q_fp8 = _simulate_fp8_e4m3_rounding(clipped)
    ste = (q_fp8 - clipped).detach() + clipped
    return ste * scale


def test_backward_matches_torch_autograd():
    """Analytic backward reproduces autograd on the STE-detach forward."""
    cfg = _make_cfg()
    torch.manual_seed(0)
    x = torch.randn(4, HEAD_DIM, dtype=torch.float64)
    upstream = torch.randn_like(x)

    x_ad = x.clone().requires_grad_(True)
    y_ad = _ste_autograd_forward(x_ad, cfg)
    (y_ad * upstream).sum().backward()
    g_torch = x_ad.grad

    y_ref, intermediates = indexcache_forward(x, cfg, return_intermediates=True)
    g_analytic = indexcache_backward(upstream, intermediates, cfg)

    torch.testing.assert_close(y_ad.detach(), y_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(g_torch, g_analytic, rtol=1e-6, atol=1e-6)


def test_dtype_matrix():
    cfg = _make_cfg()
    torch.manual_seed(0)
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(8, HEAD_DIM, dtype=dtype, requires_grad=True)
        y = apply_indexcache_kv(x, cfg)
        assert y.dtype == dtype
        y.sum().backward()
        assert x.grad.dtype == dtype


def _ceil_to_ue8m0_exp(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF).bool().int())
    return exp.clamp(1, 254).to(torch.uint8)


def _ue8m0_exp_to_float(exp: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (exp.to(torch.int32) << 23).contiguous().view(torch.float32).to(dtype)


def _pack_ue8m0_exp_to_int(exp: torch.Tensor) -> torch.Tensor:
    return exp.contiguous().view(torch.int32).reshape(exp.shape[0])


def _quantize_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    idx = torch.zeros_like(ax, dtype=torch.uint8)
    idx = torch.where(ax > 0.25, torch.ones_like(idx), idx)
    idx = torch.where(ax >= 0.75, torch.full_like(idx, 2), idx)
    idx = torch.where(ax > 1.25, torch.full_like(idx, 3), idx)
    idx = torch.where(ax >= 1.75, torch.full_like(idx, 4), idx)
    idx = torch.where(ax > 2.5, torch.full_like(idx, 5), idx)
    idx = torch.where(ax >= 3.5, torch.full_like(idx, 6), idx)
    idx = torch.where(ax > 5.0, torch.full_like(idx, 7), idx)
    sign = (x < 0) & (idx != 0)
    return idx | (sign.to(torch.uint8) << 3)


def _e2m1_codes_to_values(codes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=dtype,
        device=codes.device,
    )
    values = lut[(codes & 0x7).long()]
    return torch.where((codes & 0x8).bool(), -values, values)


def _ref_indexer_nvfp4(x: torch.Tensor, cfg: IndexCacheConfig):
    compute_dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    xf = x.to(compute_dtype)
    rows, cols = xf.shape
    assert cols == HEAD_DIM
    groups = xf.view(rows, 4, 32)
    exp = _ceil_to_ue8m0_exp(groups.abs().amax(dim=2).clamp_min(cfg.eps) / 6.0)
    scale = _ue8m0_exp_to_float(exp, compute_dtype)
    codes = _quantize_to_e2m1_codes(groups / scale.unsqueeze(-1)).view(rows, cols)
    packed_values = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
    y = (_e2m1_codes_to_values(codes, compute_dtype).view(rows, 4, 32) * scale.unsqueeze(-1))
    return y.reshape_as(xf).to(x.dtype), packed_values.contiguous(), _pack_ue8m0_exp_to_int(exp)


def _nvfp4_backward_oracle(grad_y: torch.Tensor, intermediates: dict, cfg: IndexCacheConfig):
    x = intermediates["x_compute"]
    g = grad_y.to(x.dtype)
    groups = x.view(x.shape[0], 4, 32)
    g_groups = g.view_as(groups)
    q_groups = intermediates["q_e2m1"].view_as(groups)
    mask_groups = intermediates["clip_mask"].view_as(groups)
    scale = intermediates["scale"]
    argmax = intermediates["argmax"].long()
    eps_active = intermediates["eps_active"]

    direct = g_groups * mask_groups
    inner = (g_groups * (q_groups - mask_groups * groups / scale.unsqueeze(-1))).sum(dim=-1)
    sign = torch.gather(groups.sign(), -1, argmax.unsqueeze(-1)).squeeze(-1)
    update = inner * sign * eps_active / 6.0
    rank = torch.zeros_like(direct)
    rank.scatter_(-1, argmax.unsqueeze(-1), update.unsqueeze(-1))
    return (direct + rank).reshape_as(x).to(grad_y.dtype)


def _nvfp4_ste_autograd_forward(x: torch.Tensor, cfg: IndexCacheConfig) -> torch.Tensor:
    groups = x.view(x.shape[0], 4, 32)
    scale_base = groups.abs().amax(dim=-1).clamp_min(cfg.eps) / 6.0
    scale_exp = _ceil_to_ue8m0_exp(scale_base)
    scale_quant = _ue8m0_exp_to_float(scale_exp, x.dtype)
    scale = (scale_quant - scale_base).detach() + scale_base
    pre_clip = groups / scale.unsqueeze(-1)
    clipped = pre_clip.clamp(-6.0, 6.0)
    codes = _quantize_to_e2m1_codes(clipped)
    q = _e2m1_codes_to_values(codes, x.dtype)
    ste = (q - clipped).detach() + clipped
    return (ste * scale.unsqueeze(-1)).reshape_as(x)


def test_disabled_indexcache_is_noop():
    cfg = build_indexcache_config(quantization=INDEXCACHE_QUANT_DISABLED)
    x = torch.randn(2, HEAD_DIM, requires_grad=True)
    y = apply_indexcache_kv(x, cfg)
    assert y is x
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_nvfp4_forward_shape_finite_and_reference_dequant():
    cfg = _make_nvfp4_cfg()
    torch.manual_seed(3)
    x = torch.randn(9, HEAD_DIM, dtype=torch.float32) * 0.75
    y, intermediates = indexcache_forward(x, cfg, return_intermediates=True)
    ref_y, _, _ = _ref_indexer_nvfp4(x, cfg)
    assert y.shape == x.shape
    assert torch.isfinite(y).all().item()
    torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
    assert intermediates["scale"].shape == (x.shape[0], 4)
    assert intermediates["q_e2m1"].shape == x.shape


def test_nvfp4_packed_value_and_scale_layout_matches_oracle():
    cfg = _make_nvfp4_cfg()
    x = torch.stack(
        [
            torch.linspace(-3.25, 3.25, HEAD_DIM),
            torch.arange(HEAD_DIM, dtype=torch.float32).sub(64).div(11),
        ],
        dim=0,
    )
    _, intermediates = indexcache_forward(x, cfg, return_intermediates=True)
    _, ref_values, ref_scales = _ref_indexer_nvfp4(x, cfg)
    torch.testing.assert_close(intermediates["packed_values"], ref_values)
    torch.testing.assert_close(intermediates["packed_scales"], ref_scales)


@pytest.mark.parametrize("case", ["random", "zero_tiny", "e2m1_max", "tie_argmax"])
def test_nvfp4_backward_matches_ste_oracle(case):
    cfg = _make_nvfp4_cfg()
    torch.manual_seed(5)
    x = torch.randn(3, HEAD_DIM, dtype=torch.float64) * 0.4
    if case == "zero_tiny":
        x[0].zero_()
        x[1].fill_(cfg.eps * 0.25)
    elif case == "e2m1_max":
        x[0].zero_()
        x[0, 0:32] = torch.linspace(-6.0, 6.0, 32, dtype=x.dtype)
        x[1].mul_(32.0)
    elif case == "tie_argmax":
        x[0].zero_()
        x[0, 0] = 2.0
        x[0, 7] = -2.0
        x[0, 32] = -3.0
        x[0, 33] = 3.0

    upstream = torch.randn_like(x)
    _, intermediates = indexcache_forward(x, cfg, return_intermediates=True)
    got = indexcache_backward(upstream, intermediates, cfg)
    expected = _nvfp4_backward_oracle(upstream, intermediates, cfg)
    torch.testing.assert_close(got, expected, rtol=1e-7, atol=1e-7)

    if case == "zero_tiny":
        assert (intermediates["eps_active"][0:2] == 0).all().item()
    if case == "tie_argmax":
        assert intermediates["argmax"][0, 0].item() == 0
        assert intermediates["argmax"][0, 1].item() == 0


def test_nvfp4_backward_matches_torch_autograd_ste():
    cfg = _make_nvfp4_cfg()
    torch.manual_seed(11)
    x = torch.randn(4, HEAD_DIM, dtype=torch.float64) * 0.5
    x[0, 0:32] = torch.linspace(-7.5, 8.0, 32, dtype=x.dtype)
    x[1].fill_(cfg.eps * 0.25)
    upstream = torch.randn_like(x)

    x_ad = x.clone().requires_grad_(True)
    y_ad = _nvfp4_ste_autograd_forward(x_ad, cfg)
    (y_ad * upstream).sum().backward()

    y_ref, intermediates = indexcache_forward(x, cfg, return_intermediates=True)
    g_ref = indexcache_backward(upstream, intermediates, cfg)

    torch.testing.assert_close(y_ref, y_ad.detach(), rtol=0, atol=0)
    torch.testing.assert_close(g_ref, x_ad.grad, rtol=1e-7, atol=1e-7)


def test_nvfp4_autograd_shape_and_dtype_preserves():
    cfg = _make_nvfp4_cfg()
    torch.manual_seed(7)
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(2, 3, HEAD_DIM, dtype=dtype, requires_grad=True)
        y = apply_indexcache_kv(x, cfg)
        assert y.shape == x.shape
        assert y.dtype == dtype
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.dtype == dtype


def test_indexcache_config_and_cli_selection():
    from argparse import ArgumentParser

    from megatron.training.arguments import add_megatron_arguments

    assert resolve_indexcache_quantization(
        quantization=INDEXCACHE_QUANT_DISABLED, quant_enabled=True
    ) == INDEXCACHE_QUANT_FP8
    assert resolve_indexcache_quantization(
        quantization=INDEXCACHE_QUANT_NVFP4, quant_enabled=False
    ) == INDEXCACHE_QUANT_NVFP4

    parser = ArgumentParser(allow_abbrev=False)
    add_megatron_arguments(parser)
    args = parser.parse_args(["--dsa-indexcache-quantization", INDEXCACHE_QUANT_NVFP4])
    assert args.dsa_indexcache_quantization == INDEXCACHE_QUANT_NVFP4
    assert not args.dsa_indexcache_quant_enabled

    args = parser.parse_args(["--dsa-indexcache-quant-enabled"])
    assert args.dsa_indexcache_quant_enabled
    assert resolve_indexcache_quantization(
        quantization=args.dsa_indexcache_quantization,
        quant_enabled=args.dsa_indexcache_quant_enabled,
    ) == INDEXCACHE_QUANT_FP8

    args = parser.parse_args(
        [
            "--dsa-indexcache-quantization",
            INDEXCACHE_QUANT_NVFP4,
            "--dsa-indexcache-hisa-enabled",
            "--dsa-indexcache-hisa-compression-ratio",
            "4.0",
        ]
    )
    assert args.dsa_indexcache_quantization == INDEXCACHE_QUANT_NVFP4
    assert args.dsa_indexcache_hisa_enabled
    assert args.dsa_indexcache_hisa_block_size == 128
    assert args.dsa_indexcache_hisa_compression_ratio == 4.0


def test_hisa_4to1_dynamic_block_budget():
    block_counts = torch.tensor([1, 2, 16, 64, 128, 256, 512], dtype=torch.int32)
    selected, max_selected = hisa_block_topk_counts(
        block_counts,
        block_size=128,
        topk_tokens=2048,
        compression_ratio=4.0,
    )
    assert selected.tolist() == [1, 1, 4, 16, 32, 64, 128]
    assert max_selected == 128


def test_hisa_falls_back_when_context_fits_topk():
    config = IndexCacheHISAConfig(enabled=True, compression_ratio=4.0)
    q = torch.randn(1, 1, 2, HEAD_DIM)
    k = torch.randn(2048, 1, HEAD_DIM)
    weights = torch.ones(1, 1, 2)
    assert (
        indexcache_hisa_topk(
            q,
            weights,
            k,
            2048,
            config=config,
            q_start=0,
            is_causal=False,
            mask=None,
            query_positions=None,
            key_positions=None,
        )
        is None
    )


@pytest.mark.parametrize("context_len,topk", [(4096, 1024), (8192, 2048)])
def test_hisa_4to1_map_all_candidate_pool(context_len: int, topk: int):
    config = IndexCacheHISAConfig(enabled=True, compression_ratio=4.0)
    torch.manual_seed(17)
    q = torch.randn(1, 1, 2, HEAD_DIM)
    k = torch.randn(context_len, 1, HEAD_DIM)
    weights = torch.ones(1, 1, 2)
    selected_topk = indexcache_hisa_topk(
        q,
        weights,
        k,
        topk,
        config=config,
        q_start=0,
        is_causal=False,
        mask=None,
        query_positions=None,
        key_positions=None,
    )
    assert selected_topk is not None
    assert selected_topk.shape == (1, 1, topk)
    selected = selected_topk[0, 0]
    assert selected.unique().numel() == topk
    assert int(selected.min().item()) >= 0
    assert int(selected.max().item()) < context_len
    assert bool((selected < 128).any().item())
    assert bool((selected >= context_len - 128).any().item())


def test_hisa_4to1_pads_when_candidate_pool_is_smaller_than_topk():
    config = IndexCacheHISAConfig(enabled=True, compression_ratio=4.0)
    torch.manual_seed(18)
    q = torch.randn(1, 1, 2, HEAD_DIM)
    k = torch.randn(4096, 1, HEAD_DIM)
    weights = torch.ones(1, 1, 2)
    topk = indexcache_hisa_topk(
        q,
        weights,
        k,
        2048,
        config=config,
        q_start=0,
        is_causal=False,
        mask=None,
        query_positions=None,
        key_positions=None,
    )
    assert topk is not None
    assert topk.shape == (1, 1, 2048)
    selected = topk[0, 0]
    valid = selected[selected >= 0]
    assert valid.unique().numel() == 1024
    assert int(valid.min().item()) >= 0
    assert int(valid.max().item()) < 4096
    assert int((selected < 0).sum().item()) == 1024


def test_hisa_optimized_with_scores_matches_reference_selection(monkeypatch):
    config = IndexCacheHISAConfig(
        enabled=True,
        block_size=4,
        compression_ratio=2.0,
        forced_boundary_blocks=("first", "last"),
    )
    torch.manual_seed(20260515)
    sq, bsz, heads, context_len, topk = 3, 2, 2, 24, 5
    q = torch.randn(sq, bsz, heads, HEAD_DIM, requires_grad=True)
    k = torch.randn(context_len, bsz, HEAD_DIM, requires_grad=True)
    weights = (torch.rand(sq, bsz, heads) + 0.1).requires_grad_()

    fast = indexcache_hisa_topk_with_scores(
        q,
        weights,
        k,
        topk,
        config=config,
        q_start=0,
        is_causal=False,
        mask=None,
        query_positions=None,
        key_positions=None,
        return_scores=True,
    )
    assert fast is not None
    fast_indices, fast_scores = fast
    assert fast_scores is not None

    q_flat = q.transpose(0, 1).reshape(bsz * sq, heads, HEAD_DIM).detach()
    w_flat = weights.transpose(0, 1).reshape(bsz * sq, heads).detach()
    prefix_lens = torch.full((bsz * sq,), context_len, dtype=torch.long)
    token_to_batch = torch.repeat_interleave(torch.arange(bsz), sq)
    ref_indices, _ = hisa_forward_reference(
        q_flat,
        [k[:, batch_idx].detach() for batch_idx in range(bsz)],
        w_flat,
        prefix_lens,
        token_to_batch,
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=topk,
        fallback_to_dense_if_short=False,
        forced_boundary_blocks=config.forced_boundary_blocks,
    )
    assert ref_indices is not None

    fast_flat = fast_indices.reshape(bsz * sq, topk)
    for row in range(bsz * sq):
        torch.testing.assert_close(
            fast_flat[row].sort().values.cpu(),
            ref_indices[row].long().sort().values.cpu(),
        )
        batch_idx = row // sq
        q_row = q_flat[row]
        w_row = w_flat[row]
        selected_k = k[:, batch_idx].detach().index_select(0, fast_flat[row].cpu())
        expected_scores = (
            torch.relu((q_row.unsqueeze(0) * selected_k.unsqueeze(1)).sum(-1)) * w_row
        ).sum(-1)
        torch.testing.assert_close(
            fast_scores[row].detach().cpu(),
            expected_scores.cpu(),
            rtol=1e-5,
            atol=1e-5,
        )

    fast_scores.sum().backward()
    for tensor in (q, k, weights):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad.float()).all().item()
        assert tensor.grad.float().abs().sum().item() > 0


def test_hisa_candidate_slot_grouping_is_exact(monkeypatch):
    config = IndexCacheHISAConfig(
        enabled=True,
        block_size=8,
        compression_ratio=2.0,
        forced_boundary_blocks=("first", "last"),
    )
    torch.manual_seed(20260517)
    sq, bsz, heads, context_len, topk = 5, 1, 3, 96, 12
    q = torch.randn(sq, bsz, heads, HEAD_DIM)
    k = torch.randn(context_len, bsz, HEAD_DIM)
    weights = torch.rand(sq, bsz, heads) + 0.1

    grouped = {}
    for slot_group in (1, 2, 4, 8):
        monkeypatch.setenv("MEGATRON_HISA_CANDIDATE_SLOT_GROUP", str(slot_group))
        result = indexcache_hisa_topk_with_scores(
            q,
            weights,
            k,
            topk,
            config=config,
            q_start=0,
            is_causal=False,
            mask=None,
            query_positions=None,
            key_positions=None,
            return_scores=True,
        )
        assert result is not None
        grouped[slot_group] = result

    base_indices, base_scores = grouped[1]
    for slot_group in (2, 4, 8):
        indices, scores = grouped[slot_group]
        torch.testing.assert_close(
            indices.reshape(-1, topk).sort(dim=-1).values,
            base_indices.reshape(-1, topk).sort(dim=-1).values,
        )
        for row in range(sq * bsz):
            base_order = base_indices.reshape(-1, topk)[row]
            row_indices = indices.reshape(-1, topk)[row]
            base_lookup = {
                int(idx.item()): base_scores[row, pos].item()
                for pos, idx in enumerate(base_order)
            }
            expected = torch.tensor(
                [base_lookup[int(idx.item())] for idx in row_indices],
                dtype=scores.dtype,
                device=scores.device,
            )
            torch.testing.assert_close(scores[row], expected, rtol=1e-5, atol=1e-5)


def test_hisa_candidate_slot_grouping_is_exact_for_causal_prefix(monkeypatch):
    config = IndexCacheHISAConfig(
        enabled=True,
        block_size=8,
        compression_ratio=2.0,
        forced_boundary_blocks=("first", "last"),
    )
    torch.manual_seed(20260518)
    sq, bsz, heads, context_len, topk, q_start = 5, 1, 3, 128, 12, 72
    q = torch.randn(sq, bsz, heads, HEAD_DIM)
    k = torch.randn(context_len, bsz, HEAD_DIM)
    weights = torch.rand(sq, bsz, heads) + 0.1

    monkeypatch.setenv("MEGATRON_HISA_CANDIDATE_SLOT_GROUP", "8")
    fast = indexcache_hisa_topk_with_scores(
        q,
        weights,
        k,
        topk,
        config=config,
        q_start=q_start,
        is_causal=True,
        mask=None,
        query_positions=None,
        key_positions=None,
        return_scores=True,
    )
    assert fast is not None
    fast_indices, fast_scores = fast
    assert fast_scores is not None

    q_flat = q.transpose(0, 1).reshape(bsz * sq, heads, HEAD_DIM).detach()
    w_flat = weights.transpose(0, 1).reshape(bsz * sq, heads).detach()
    prefix_lens = torch.arange(q_start + 1, q_start + sq + 1, dtype=torch.long)
    token_to_batch = torch.zeros(sq, dtype=torch.long)
    ref_indices, _ = hisa_forward_reference(
        q_flat,
        [k[:, 0].detach()],
        w_flat,
        prefix_lens,
        token_to_batch,
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=topk,
        fallback_to_dense_if_short=False,
        forced_boundary_blocks=config.forced_boundary_blocks,
    )
    assert ref_indices is not None
    fast_flat = fast_indices.reshape(sq, topk)
    for row in range(sq):
        torch.testing.assert_close(
            fast_flat[row].sort().values.cpu(),
            ref_indices[row].long().sort().values.cpu(),
        )
        assert int(fast_flat[row].max().item()) < q_start + row + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hisa_cuda_selector_matches_reference_selection(monkeypatch):
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("HISA CUDA selector is intended for Blackwell+.")

    import megatron.core.quantization.indexcache.hisa as hisa_module

    ext = hisa_module._try_load_hisa_cuda_ext()
    if ext is None or not hasattr(ext, "hisa_selector_fwd"):
        pytest.skip("HISA CUDA selector extension unavailable.")
    monkeypatch.setenv("MEGATRON_HISA_SELECTOR_CUDA", "1")

    config = IndexCacheHISAConfig(
        enabled=True,
        block_size=4,
        compression_ratio=2.0,
        forced_boundary_blocks=("first", "last"),
    )
    torch.manual_seed(20260516)
    sq, heads, context_len, topk = 8, 4, 32, 4
    q = torch.randn(sq, heads, HEAD_DIM, device="cuda")
    k = torch.randn(context_len, HEAD_DIM, device="cuda")
    weights = torch.rand(sq, heads, device="cuda") + 0.1
    q_start = 20
    prefix_lens = torch.arange(q_start + 1, q_start + sq + 1, device="cuda").clamp(
        0, context_len
    )
    row_block_counts = torch.div(
        prefix_lens + config.block_size - 1,
        config.block_size,
        rounding_mode="floor",
    ).to(torch.int32)
    block_topk_counts, effective_block_topk = hisa_block_topk_counts(
        row_block_counts,
        block_size=config.block_size,
        topk_tokens=topk,
        compression_ratio=config.compression_ratio,
    )

    got = hisa_module._indexcache_hisa_topk_cuda_for_batch(
        q,
        weights,
        k,
        topk,
        config=config,
        prefix_lens=prefix_lens,
        block_topk_counts=block_topk_counts,
        effective_block_topk=effective_block_topk,
        return_scores=True,
    )
    assert got is not None
    got_indices, got_scores = got

    ref_indices, _ = hisa_forward_reference(
        q.detach(),
        [k.detach()],
        weights.detach(),
        prefix_lens,
        torch.zeros(sq, device="cuda", dtype=torch.long),
        block_size=config.block_size,
        compression_ratio=config.compression_ratio,
        topk_tokens=topk,
        fallback_to_dense_if_short=False,
        forced_boundary_blocks=config.forced_boundary_blocks,
    )
    assert ref_indices is not None
    for row in range(sq):
        torch.testing.assert_close(
            got_indices[row].sort().values.cpu(),
            ref_indices[row].long().sort().values.cpu(),
        )

    selected_k = k.index_select(0, got_indices.clamp_min(0).reshape(-1)).view(
        sq, topk, HEAD_DIM
    )
    expected_scores = (
        torch.relu((q.unsqueeze(1) * selected_k.unsqueeze(2)).sum(-1))
        * weights.unsqueeze(1)
    ).sum(-1)
    torch.testing.assert_close(got_scores, expected_scores, rtol=1e-5, atol=1e-5)


def test_chunked_dsa_forward_masks_padded_hisa_candidates(monkeypatch):
    import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

    torch.manual_seed(19)
    q = torch.randn(1, 1, 2, HEAD_DIM)
    index_k = torch.randn(4, 1, HEAD_DIM)
    weights = torch.ones(1, 1, 2)
    query = torch.randn(1, 1, 1, 8)
    key = torch.randn(4, 1, 1, 8)
    value = torch.randn(4, 1, 1, 8)

    expected = dsa_module._sparse_dsa_attention_chunk(
        query,
        key,
        value,
        torch.tensor([[[0, 1]]], dtype=torch.long),
        1.0,
        mask=None,
        q_start=0,
        is_causal=False,
    )

    def fake_hisa_topk(*_args, **_kwargs):
        return torch.tensor([[[0, 1, -1, -1]]], dtype=torch.long)

    monkeypatch.setattr(dsa_module, "indexcache_hisa_topk", fake_hisa_topk)
    got, _ = dsa_module.chunked_dsa_forward(
        q,
        index_k,
        weights,
        query,
        key,
        value,
        softmax_scale=1.0,
        topk=4,
        mask=None,
        is_causal=False,
        loss_coeff=0.0,
        sparse_loss=False,
        pg_collection=None,
        chunk_size=1,
        indexcache_hisa_config=IndexCacheHISAConfig(enabled=True),
    )
    torch.testing.assert_close(got, expected)


def test_chunked_dsa_forward_dispatches_hisa_selector(monkeypatch):
    import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

    calls = []

    def fake_hisa_topk(q, weights, k, topk, **kwargs):
        calls.append(kwargs["config"])
        sq = q.shape[0]
        bsz = q.shape[1]
        return torch.arange(topk, device=q.device).view(1, 1, topk).expand(bsz, sq, topk)

    monkeypatch.setattr(dsa_module, "indexcache_hisa_topk", fake_hisa_topk)

    q = torch.randn(2, 1, 2, HEAD_DIM)
    index_k = torch.randn(8, 1, HEAD_DIM)
    weights = torch.ones(2, 1, 2)
    query = torch.randn(2, 1, 1, 8)
    key = torch.randn(8, 1, 1, 8)
    value = torch.randn(8, 1, 1, 8)
    output, indexer_loss = dsa_module.chunked_dsa_forward(
        q,
        index_k,
        weights,
        query,
        key,
        value,
        softmax_scale=1.0,
        topk=4,
        mask=None,
        is_causal=False,
        loss_coeff=0.0,
        sparse_loss=False,
        pg_collection=None,
        chunk_size=2,
        indexcache_hisa_config=IndexCacheHISAConfig(enabled=True),
    )
    assert calls and calls[0].enabled
    assert output.shape == (2, 1, 8)
    assert indexer_loss is None


def test_chunked_dsa_hisa_short_context_matches_ordinary_nvfp4_indexcache(monkeypatch):
    import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

    monkeypatch.setenv("MEGATRON_DSA_STREAMING_INDEXER_TOPK", "0")
    torch.manual_seed(21)
    q = torch.randn(2, 1, 2, HEAD_DIM)
    raw_index_k = torch.randn(8, 1, HEAD_DIM)
    index_k = apply_indexcache_kv(raw_index_k.reshape(-1, HEAD_DIM), _make_nvfp4_cfg())
    index_k = index_k.reshape_as(raw_index_k).detach()
    weights = torch.ones(2, 1, 2)
    query = torch.randn(2, 1, 1, 8)
    key = torch.randn(8, 1, 1, 8)
    value = torch.randn(8, 1, 1, 8)

    ordinary_output, ordinary_loss = dsa_module.chunked_dsa_forward(
        q,
        index_k,
        weights,
        query,
        key,
        value,
        softmax_scale=1.0,
        topk=8,
        mask=None,
        is_causal=False,
        loss_coeff=0.0,
        sparse_loss=False,
        pg_collection=None,
        chunk_size=2,
        indexcache_hisa_config=None,
    )
    hisa_output, hisa_loss = dsa_module.chunked_dsa_forward(
        q,
        index_k,
        weights,
        query,
        key,
        value,
        softmax_scale=1.0,
        topk=8,
        mask=None,
        is_causal=False,
        loss_coeff=0.0,
        sparse_loss=False,
        pg_collection=None,
        chunk_size=2,
        indexcache_hisa_config=IndexCacheHISAConfig(
            enabled=True,
            block_size=4,
            compression_ratio=4.0,
        ),
    )

    assert ordinary_loss is None
    assert hisa_loss is None
    torch.testing.assert_close(hisa_output, ordinary_output)


def test_chunked_dsa_hisa_path_backpropagates_attention_grads():
    import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

    torch.manual_seed(20260513)
    q = torch.randn(2, 1, 2, HEAD_DIM)
    index_k = torch.randn(32, 1, HEAD_DIM)
    weights = torch.ones(2, 1, 2)
    query = torch.randn(2, 1, 1, 8, requires_grad=True)
    key = torch.randn(32, 1, 1, 8, requires_grad=True)
    value = torch.randn(32, 1, 1, 8, requires_grad=True)

    output, indexer_loss = dsa_module.chunked_dsa_forward(
        q,
        index_k,
        weights,
        query,
        key,
        value,
        softmax_scale=1.0,
        topk=4,
        mask=None,
        is_causal=False,
        loss_coeff=0.0,
        sparse_loss=False,
        pg_collection=None,
        chunk_size=2,
        indexcache_hisa_config=IndexCacheHISAConfig(
            enabled=True,
            block_size=4,
            compression_ratio=4.0,
        ),
    )
    assert indexer_loss is None
    assert output.shape == (2, 1, 8)
    output.float().square().sum().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad.float()).all().item()
        assert tensor.grad.float().abs().sum().item() > 0


def test_hisa_attention_target_probs_row_chunk_matches_full(monkeypatch):
    import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

    torch.manual_seed(20260515)
    q_len, bsz, num_heads, head_dim, topk = 5, 2, 3, 4, 6
    query = torch.randn(q_len, bsz, num_heads, head_dim, dtype=torch.bfloat16)
    key = torch.randn(11, bsz, num_heads, head_dim, dtype=torch.bfloat16)
    topk_indices = torch.tensor(
        [
            [
                [0, 1, 2, 3, -1, -1],
                [1, 3, 5, 7, 9, -1],
                [2, 4, 6, 8, 10, -1],
                [0, 2, 4, 6, 8, 10],
                [1, 2, 3, 4, 5, 6],
            ],
            [
                [10, 8, 6, 4, 2, 0],
                [9, 7, 5, 3, 1, -1],
                [8, 7, 6, 5, -1, -1],
                [3, 4, 5, 6, 7, 8],
                [0, 2, 4, 6, 8, 10],
            ],
        ],
        dtype=torch.long,
    )
    softmax_scale = 0.25

    def full_reference():
        attention_scores = torch.empty((bsz * q_len, topk, num_heads), dtype=torch.float32)
        for batch_idx in range(bsz):
            selected = topk_indices[batch_idx]
            valid = selected >= 0
            safe_selected = selected.clamp_min(0)
            selected_key = key[:, batch_idx].float().index_select(0, safe_selected.reshape(-1))
            selected_key = selected_key.view(q_len, topk, num_heads, head_dim)
            scores = (
                torch.einsum("qhd,qkhd->qkh", query[:, batch_idx].float(), selected_key)
                * softmax_scale
            )
            attention_scores[batch_idx * q_len : (batch_idx + 1) * q_len] = scores.masked_fill(
                ~valid.unsqueeze(-1), float("-inf")
            )
        probs = torch.softmax(attention_scores, dim=1, dtype=torch.float32).sum(dim=2)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)

    monkeypatch.setenv("MEGATRON_HISA_TARGET_ROW_CHUNK", "2")
    got = dsa_module._hisa_attention_target_probs(
        query, key, topk_indices, softmax_scale=softmax_scale, tp_group=None
    )
    torch.testing.assert_close(got, full_reference())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hisa_attention_target_probs_triton_matches_full(monkeypatch):
    import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

    torch.manual_seed(20260516)
    q_len, bsz, num_heads, head_dim, topk = 7, 2, 5, 16, 9
    query = torch.randn(q_len, bsz, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(19, bsz, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    topk_indices = torch.randint(0, key.shape[0], (bsz, q_len, topk), device="cuda")
    topk_indices[:, 1::3, -2:] = -1
    softmax_scale = 0.125

    def full_reference():
        attention_scores = torch.empty(
            (bsz * q_len, topk, num_heads), device="cuda", dtype=torch.float32
        )
        for batch_idx in range(bsz):
            selected = topk_indices[batch_idx]
            valid = selected >= 0
            safe_selected = selected.clamp_min(0)
            selected_key = key[:, batch_idx].float().index_select(0, safe_selected.reshape(-1))
            selected_key = selected_key.view(q_len, topk, num_heads, head_dim)
            scores = (
                torch.einsum("qhd,qkhd->qkh", query[:, batch_idx].float(), selected_key)
                * softmax_scale
            )
            attention_scores[batch_idx * q_len : (batch_idx + 1) * q_len] = scores.masked_fill(
                ~valid.unsqueeze(-1), float("-inf")
            )
        probs = torch.softmax(attention_scores, dim=1, dtype=torch.float32).sum(dim=2)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)

    monkeypatch.setenv("MEGATRON_HISA_TARGET_TRITON", "1")
    monkeypatch.setenv("MEGATRON_HISA_TARGET_BLOCK_K", "4")
    got = dsa_module._hisa_attention_target_probs(
        query, key, topk_indices, softmax_scale=softmax_scale, tp_group=None
    )
    torch.testing.assert_close(got, full_reference(), rtol=1e-4, atol=1e-5)


def test_chunked_dsa_hisa_with_indexer_loss_backpropagates_indexer_grads(monkeypatch):
    import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

    class _ProcessGroup:
        def size(self):
            return 1

    class _ProcessGroups:
        tp = _ProcessGroup()

    def fake_hisa_topk(*_args, **_kwargs):
        return torch.tensor([[[0, 1, -1, -1]]], dtype=torch.long)

    monkeypatch.setattr(dsa_module, "indexcache_hisa_topk", fake_hisa_topk)

    torch.manual_seed(20)
    q = torch.randn(1, 1, 2, HEAD_DIM, requires_grad=True)
    index_k = torch.randn(4, 1, HEAD_DIM, requires_grad=True)
    weights = (torch.rand(1, 1, 2) + 0.1).requires_grad_()
    query = torch.randn(1, 1, 1, 8, requires_grad=True)
    key = torch.randn(4, 1, 1, 8, requires_grad=True)
    value = torch.randn(4, 1, 1, 8, requires_grad=True)

    output, indexer_loss = dsa_module.chunked_dsa_forward(
        q,
        index_k,
        weights,
        query,
        key,
        value,
        softmax_scale=1.0,
        topk=4,
        mask=None,
        is_causal=False,
        loss_coeff=0.1,
        sparse_loss=False,
        pg_collection=_ProcessGroups(),
        chunk_size=1,
        indexcache_hisa_config=IndexCacheHISAConfig(enabled=True),
    )

    assert indexer_loss is not None
    (output.float().sum() + indexer_loss).backward()
    for tensor in (q, index_k, weights):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad.float()).all().item()
        assert tensor.grad.float().abs().sum().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hisa_selected_score_bwd_cuda_matches_autograd(monkeypatch):
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("HISA CUDA selector is intended for Blackwell+.")

    import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

    monkeypatch.setenv("MEGATRON_HISA_SELECTOR_CUDA", "1")
    torch.manual_seed(20260517)
    q_len, num_heads, head_dim, seq_len, topk = 5, 3, HEAD_DIM, 19, 6
    q = torch.randn(q_len, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    weights = (torch.rand(q_len, num_heads, device="cuda", dtype=torch.float32) + 0.1).requires_grad_()
    k = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    prefix_lens = torch.tensor([7, 9, 13, 17, 19], device="cuda", dtype=torch.long)
    config = IndexCacheHISAConfig(
        enabled=True,
        block_size=4,
        compression_ratio=2.0,
        topk_tokens=topk,
        fallback_to_dense_if_short=False,
    )

    result = indexcache_hisa_cuda_select_with_scores(
        q,
        weights,
        k,
        topk,
        config=config,
        prefix_lens=prefix_lens,
    )
    assert result is not None
    topk_i32, selected_scores = result
    valid = topk_i32 >= 0
    selected_k = k.index_select(0, topk_i32.clamp_min(0).reshape(-1)).view(
        q_len, topk, head_dim
    )
    ref_scores = (
        torch.relu(torch.einsum("qhd,qkd->qkh", q, selected_k)) * weights.unsqueeze(1)
    ).sum(dim=-1)
    ref_scores = ref_scores.masked_fill(~valid, float("-inf"))
    torch.testing.assert_close(selected_scores, ref_scores.detach(), rtol=1e-5, atol=2e-5)

    grad_seed = torch.randn_like(ref_scores).masked_fill(~valid, 0)
    (ref_scores.masked_fill(~valid, 0) * grad_seed).sum().backward()
    grad_q, grad_w, grad_k = dsa_module._hisa_selected_score_backward_cuda(
        grad_seed,
        q.detach(),
        weights.detach(),
        k.detach(),
        topk_i32,
    )
    torch.testing.assert_close(grad_q, q.grad, rtol=5e-5, atol=1e-4)
    torch.testing.assert_close(grad_w, weights.grad, rtol=5e-5, atol=1e-4)
    torch.testing.assert_close(grad_k, k.grad, rtol=5e-5, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunked_dsa_hisa_fused_indexer_loss_matches_existing_path(monkeypatch):
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("HISA CUDA selector is intended for Blackwell+.")

    import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module

    class _ProcessGroup:
        def size(self):
            return 1

    class _ProcessGroups:
        tp = _ProcessGroup()

    original_sparse_attention = dsa_module.sparse_dsa_attention_triton

    def run_case(fused: bool):
        monkeypatch.setenv("MEGATRON_HISA_SELECTOR_CUDA", "1")
        monkeypatch.setenv("MEGATRON_HISA_FUSED_INDEXER_LOSS", "1" if fused else "0")
        monkeypatch.setenv("MEGATRON_HISA_KL_GRAD_TRITON", "1")
        torch.manual_seed(20260518)
        q_len, bsz, idx_heads, idx_dim, attn_heads, attn_dim, seq_len, topk = (
            8,
            1,
            2,
            HEAD_DIM,
            2,
            16,
            23,
            6,
        )
        q = torch.randn(q_len, bsz, idx_heads, idx_dim, device="cuda", requires_grad=True)
        index_k = torch.randn(seq_len, bsz, idx_dim, device="cuda", requires_grad=True)
        weights = (torch.rand(q_len, bsz, idx_heads, device="cuda") + 0.1).requires_grad_()
        query = torch.randn(q_len, bsz, attn_heads, attn_dim, device="cuda", requires_grad=True)
        key = torch.randn(seq_len, bsz, attn_heads, attn_dim, device="cuda", requires_grad=True)
        value = torch.randn(seq_len, bsz, attn_heads, attn_dim, device="cuda", requires_grad=True)

        topk_seen = []

        def capture_sparse_attention(query_arg, key_arg, value_arg, topk_arg, *args, **kwargs):
            topk_seen.append(topk_arg.detach().clone())
            return original_sparse_attention(query_arg, key_arg, value_arg, topk_arg, *args, **kwargs)

        monkeypatch.setattr(dsa_module, "sparse_dsa_attention_triton", capture_sparse_attention)
        output, indexer_loss = dsa_module.chunked_dsa_forward(
            q,
            index_k,
            weights,
            query,
            key,
            value,
            softmax_scale=0.25,
            topk=topk,
            mask=None,
            is_causal=True,
            loss_coeff=0.1,
            sparse_loss=False,
            pg_collection=_ProcessGroups(),
            chunk_size=4,
            indexcache_hisa_config=IndexCacheHISAConfig(
                enabled=True,
                block_size=4,
                compression_ratio=2.0,
                topk_tokens=topk,
                fallback_to_dense_if_short=False,
            ),
        )
        assert indexer_loss is not None
        (output.float().sum() + indexer_loss).backward()
        assert topk_seen
        return (
            output.detach(),
            indexer_loss.detach(),
            q.grad,
            index_k.grad,
            weights.grad,
            torch.cat([x.reshape(-1, x.size(-1)) for x in topk_seen], dim=0),
        )

    ref = run_case(False)
    fused = run_case(True)
    torch.testing.assert_close(fused[5], ref[5], rtol=0, atol=0)
    torch.testing.assert_close(fused[0], ref[0], rtol=8e-4, atol=8e-4)
    torch.testing.assert_close(fused[1], ref[1], rtol=2e-3, atol=2e-3)
    for got, expected in zip(fused[2:5], ref[2:5]):
        torch.testing.assert_close(got, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_non_blackwell_cuda_uses_reference_fallback(monkeypatch):
    if torch.cuda.get_device_capability() >= (10, 0):
        pytest.skip("Blackwell should exercise the CUDA NVFP4 extension path.")

    import megatron.core.quantization.indexcache.autograd as indexcache_autograd

    def fail_load_ext():
        raise AssertionError("NVFP4 non-Blackwell path must not load the extension")

    monkeypatch.setattr(indexcache_autograd, "_try_load_cuda_ext", fail_load_ext)
    cfg = _make_nvfp4_cfg()
    x = torch.randn(2, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = apply_indexcache_kv(x, cfg)
    y.float().sum().backward()
    assert y.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad.float()).all().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_blackwell_cuda_packed_backward_matches_reference():
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("NVFP4 packed CUDA backward requires Blackwell.")

    from megatron.core.quantization.indexcache.kernels.build import get_ext

    cfg = _make_nvfp4_cfg()
    try:
        ext = get_ext()
    except Exception as exc:
        pytest.skip(f"NVFP4 CUDA extension unavailable: {exc}")
    for dtype in [torch.float32, torch.bfloat16]:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(20260512)
        x = (
            torch.randn((4, HEAD_DIM), device="cuda", dtype=torch.float32, generator=gen)
            * 0.65
        )
        x[0].zero_()
        x[1, 0] = 2.0
        x[1, 7] = -2.0
        x[1, 32] = -3.0
        x[1, 33] = 3.0
        gy = torch.randn((4, HEAD_DIM), device="cuda", dtype=torch.float32, generator=gen)
        x = x.to(dtype).contiguous()
        gy = gy.to(dtype).contiguous()

        ref_y, ref_inter = indexcache_forward(x, cfg, return_intermediates=True)
        ref_gx = indexcache_backward(gy, ref_inter, cfg)

        out = torch.empty_like(x)
        scale = torch.empty((4, 4), device="cuda", dtype=torch.float32)
        q = torch.empty((4, HEAD_DIM), device="cuda", dtype=torch.float32)
        mask = torch.empty((4, HEAD_DIM), device="cuda", dtype=torch.uint8)
        argmax = torch.empty((4, 4), device="cuda", dtype=torch.int32)
        eps_active = torch.empty((4, 4), device="cuda", dtype=torch.uint8)
        packed_values = torch.empty((4, 64), device="cuda", dtype=torch.uint8)
        packed_scales = torch.empty((4,), device="cuda", dtype=torch.int32)
        gx = torch.empty_like(x)

        ext.indexcache_nvfp4_fwd(
            x, out, scale, q, mask, argmax, eps_active,
            packed_values, packed_scales, cfg.eps,
        )
        ext.indexcache_nvfp4_bwd_packed(
            gy, x, scale, packed_values, mask, argmax, eps_active, gx, cfg.fp4_max,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(out, ref_y, rtol=0, atol=0)
        if dtype is torch.float32:
            torch.testing.assert_close(gx, ref_gx, rtol=1e-6, atol=1e-6)
        else:
            torch.testing.assert_close(gx.float(), ref_gx.float(), rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="fp8 unavailable")
def test_parity_with_sglang_act_quant_math():
    """Match the SGLang Triton _act_quant_kernel formula at the cast level.

    SGLang stores fp8 + per-block scale; we dequantize for fake-quant. The
    underlying scaled-quantized values must be bit-identical when running
    on the same fp32 inputs.
    """
    cfg = _make_cfg()
    torch.manual_seed(0)
    x = torch.randn(64, HEAD_DIM, dtype=torch.float32)

    abs_max = x.abs().amax(dim=-1)
    amax = abs_max.clamp_min(cfg.eps)
    scale = (amax * cfg.fp8_max_inv)[:, None]
    pre_clip = (x / scale).clamp(-cfg.fp8_max, cfg.fp8_max)
    sgl_q = pre_clip.to(torch.float8_e4m3fn).to(torch.float32)
    sgl_dequant = sgl_q * scale

    y_ref = indexcache_forward(x, cfg)
    torch.testing.assert_close(y_ref, sgl_dequant, rtol=0, atol=0)
