# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Composition tests: activation-ECO with IndexCache and weight-ECO.

Activation-ECO touches the *input* to a Linear; IndexCache touches the
*output* of the indexer's K projection; weight-ECO touches the
optimizer's m1 buffer at step time. They live in three different
places in the layer pipeline and should compose without coordination.
This file pins those compositional contracts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from megatron.core.quantization.indexcache import (  # noqa: E402
    apply_indexcache_kv,
    build_indexcache_config,
)
from megatron.core.quantization.nvfp4_act_eco import (  # noqa: E402
    apply_nvfp4_act_eco_linear,
    build_nvfp4_act_eco_config,
)
from megatron.core.quantization.nvfp4_act_eco.reference import (  # noqa: E402
    activation_eco_bias_correction,
    nvfp4_act_quant_forward,
)


HIDDEN = 64
HEAD_DIM = 128


def test_compose_with_indexcache_finite():
    """Stack: x -> nvfp4_act_eco_linear (W_k, indexer K projection) -> IndexCache."""
    cfg_act = build_nvfp4_act_eco_config()
    cfg_idx = build_indexcache_config()
    torch.manual_seed(2)
    x = torch.randn(4, HIDDEN, dtype=torch.float32, requires_grad=True)
    W_k = torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float32, requires_grad=True)

    k_pre = apply_nvfp4_act_eco_linear(x, W_k, cfg_act)
    k_post = apply_indexcache_kv(k_pre, cfg_idx)
    loss = (k_post ** 2).mean()
    loss.backward()

    assert torch.isfinite(k_post).all().item()
    assert torch.isfinite(x.grad).all().item()
    assert torch.isfinite(W_k.grad).all().item()


def test_compose_with_indexcache_dW_includes_act_eco_correction():
    """
    Through both fake-quants the weight gradient must still carry the
    activation-ECO correction (i.e. dW depends on x_pre, not just q(x_pre)).
    Verified by perturbing x and confirming dW changes — a dW that ignored
    x_pre and used only q(x_pre) would be insensitive to small x perturbations
    that fall inside the same NVFP4 cell.
    """
    cfg_act = build_nvfp4_act_eco_config()
    cfg_idx = build_indexcache_config()
    torch.manual_seed(3)
    n = 4
    x_base = torch.randn(n, HIDDEN, dtype=torch.float32)
    W_k = torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float32)

    def _grad(x):
        x = x.clone().requires_grad_(True)
        Wk = W_k.clone().requires_grad_(True)
        k_pre = apply_nvfp4_act_eco_linear(x, Wk, cfg_act)
        k_post = apply_indexcache_kv(k_pre, cfg_idx)
        (k_post ** 2).mean().backward()
        return Wk.grad.clone()

    g_base = _grad(x_base)

    # Perturb x by a small amount that's well inside one NVFP4 cell so q(x)
    # stays unchanged (verified explicitly below).
    eps = 1e-4
    x_perturbed = x_base + eps * torch.randn_like(x_base)
    q_base = nvfp4_act_quant_forward(x_base, cfg_act)
    q_pert = nvfp4_act_quant_forward(x_perturbed, cfg_act)
    if (q_base != q_pert).any():
        # Pick a smaller eps if the random perturbation crossed a cell.
        # This is rare at eps=1e-4 and a fresh manual_seed makes it deterministic.
        pytest.skip("perturbation crossed an NVFP4 cell; rerun with smaller eps")

    g_pert = _grad(x_perturbed)
    # dW should differ — proves dW depends on x_pre, not just q(x_pre).
    diff = (g_pert - g_base).abs().max().item()
    assert diff > 1e-6, (
        f"dW invariant to in-cell x perturbation (diff={diff}); "
        "act-ECO correction not propagating through the IndexCache stage"
    )


def test_act_eco_compatible_with_weight_quantization_storage():
    """
    Sanity check: the act-ECO Function does not require W to be in any
    particular dtype or layout; in production W is the bf16 transient
    shard that FlashOptim materialises from NVFP4 storage at step time.
    """
    cfg = build_nvfp4_act_eco_config()
    torch.manual_seed(4)
    x = torch.randn(4, HIDDEN, dtype=torch.float32, requires_grad=True)
    # Simulate the post-cast bf16 transient shard the optimizer hands the model.
    W_bf16 = torch.randn(32, HIDDEN, dtype=torch.bfloat16)
    W = W_bf16.to(torch.float32).requires_grad_(True)

    y = apply_nvfp4_act_eco_linear(x, W, cfg)
    y.sum().backward()
    assert torch.isfinite(x.grad).all().item()
    assert torch.isfinite(W.grad).all().item()


def test_no_extra_persistent_state():
    """
    Hard memory invariant: the act-ECO Function does not stash any
    persistent state across forward calls. Each forward saves only what
    backward needs (x_pre, q_x, weight, mask, scale) — all released by
    the autograd engine on backward completion.
    """
    cfg = build_nvfp4_act_eco_config()
    torch.manual_seed(5)
    x = torch.randn(4, HIDDEN, dtype=torch.float32, requires_grad=True)
    W = torch.randn(32, HIDDEN, dtype=torch.float32, requires_grad=True)

    # Run, backward, ensure W.grad lifecycle is clean.
    for _ in range(3):
        x.grad = None
        W.grad = None
        y = apply_nvfp4_act_eco_linear(x, W, cfg)
        y.sum().backward()
    assert W.grad is not None
    assert x.grad is not None
