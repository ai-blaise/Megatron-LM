# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parity test: FlashAdamW + ECO under enable_gradient_release vs batched step.

Drives a small model down two paths:
    (A) batched: backward then optimizer.step()   (reference)
    (B) GR    : enable_gradient_release(model, opt); backward triggers
                per-parameter step_param via post_accumulate hooks.

Asserts that final parameters AND optimizer state (momentum, variance)
are bit-identical across both paths. Tested for both quantize=True
(NVFP4-style 2-bit shadow) and quantize=False (full bf16 master), with
eco=True throughout. Run on CUDA.

The test exists because enable_gradient_release was authored before the
ECO codepath landed; this is the first parity gate that covers the
combination.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _build_opt(model, *, eco, quantize):
    from megatron.core.optimizer.flash_optimizers import FlashAdamW

    kwargs = dict(
        lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
        eco=eco, fused=True,
    )
    if quantize:
        kwargs["quantize"] = True
    return FlashAdamW(list(model.parameters()), **kwargs)


def _make_model(seed):
    torch.manual_seed(seed)
    m = nn.Sequential(
        nn.Linear(64, 96, bias=True),
        nn.GELU(),
        nn.Linear(96, 32, bias=True),
    ).cuda().to(torch.bfloat16)
    return m


def _clone_state(opt):
    out = []
    for p in (q for g in opt.param_groups for q in g["params"]):
        st = opt.state.get(p, {})
        entry = {"param": p.detach().clone()}
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st:
                entry[k] = st[k].kernel_tensor.detach().clone()
        if "step" in st:
            entry["step"] = int(st["step"].item())
        out.append(entry)
    return out


def _diff_states(a, b):
    diffs = []
    for ia, ib in zip(a, b):
        for k in ia:
            if k == "step":
                if ia[k] != ib[k]:
                    diffs.append(("step", ia[k], ib[k]))
                continue
            ta = ia[k].float()
            tb = ib[k].float()
            d = (ta - tb).abs().max().item()
            if d > 0.0:
                diffs.append((k, ta.shape, d))
    return diffs


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestFlashAdamWEcoGradientRelease(unittest.TestCase):

    def _run(self, *, eco, quantize, steps):
        from megatron.core.optimizer.flash_optimizers import enable_gradient_release

        torch.manual_seed(0)
        batches = [
            (torch.randn(8, 64, device="cuda", dtype=torch.bfloat16),
             torch.randn(8, 32, device="cuda", dtype=torch.bfloat16))
            for _ in range(steps)
        ]

        # Path A: batched step
        m_a = _make_model(seed=42)
        opt_a = _build_opt(m_a, eco=eco, quantize=quantize)
        for x, y in batches:
            opt_a.zero_grad()
            loss = (m_a(x) - y).pow(2).mean()
            loss.backward()
            opt_a.step()
        torch.cuda.synchronize()
        state_a = _clone_state(opt_a)

        # Path B: gradient release
        m_b = _make_model(seed=42)
        opt_b = _build_opt(m_b, eco=eco, quantize=quantize)
        # Materialize optimizer state by calling step_param once on each
        # param with a zero grad — required because GR hooks expect state
        # to exist before the first backward (some kernels assert step>=1
        # for ECO inject; non-ECO Adam initializes lazily). We don't want
        # to step the params here; just init state. The cleanest portable
        # approach is to issue a real step via tiny dummy fwd/bwd before
        # enabling GR — but that consumes one of our steps. Instead we
        # permit lazy init and run GR over all `steps` batches.
        handle = enable_gradient_release(m_b, opt_b)
        try:
            for x, y in batches:
                # NB: opt.zero_grad() is a no-op under GR; hooks set
                # p.grad=None per-param.
                loss = (m_b(x) - y).pow(2).mean()
                loss.backward()
        finally:
            handle.remove()
        torch.cuda.synchronize()
        state_b = _clone_state(opt_b)

        diffs = _diff_states(state_a, state_b)
        if diffs:
            msg = "; ".join(f"{k}: {v}" for k, *v in diffs[:8])
            self.fail(
                f"GR vs batched divergence (eco={eco}, quantize={quantize}, "
                f"steps={steps}): {msg}"
            )

    def test_eco_unquantized_50step(self):
        self._run(eco=True, quantize=False, steps=50)

    def test_eco_quantized_50step(self):
        self._run(eco=True, quantize=True, steps=50)

    def test_no_eco_unquantized_50step(self):
        self._run(eco=False, quantize=False, steps=50)


if __name__ == "__main__":
    unittest.main()
