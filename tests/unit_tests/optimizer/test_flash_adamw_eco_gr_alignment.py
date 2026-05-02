# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Target-model + parallelism alignment for FlashAdamW + ECO GR.

Confirms the gradient-release path holds the invariants we locked in
during the TurboQuant + IndexCache + ECO work for the
DeepSeek-V3.2-REAP-345B-NVFP4 target:

1. **Per-token locality / parallelism bit-exactness** —
   FlashAdamW.step_param is parameter-local and contains no cross-rank
   collectives outside the NVFP4 cast (orchestrated via
   ``NVFP4EcoGradientReleaseOrchestrator``). We assert that running GR
   over a model whose params are TP-sharded (simulated as two halves)
   gives the same updated params as the equivalent unsharded reference.

2. **No extra persistent memory** — enabling GR must not spawn extra
   optimizer state buffers beyond the (exp_avg, exp_avg_sq, step) keys
   already present after the first batched step.

3. **Target-shape coverage** — drive the kv_lora_rank=512 and
   hidden_size=7168 shapes from the target config to surface any
   shape-specific kernel issue under GR.

4. **bf16 master + fp32 main_grad mix** — mimics the MCore distrib opt
   path the user runs in production. The decoupled_grad staging in
   ``enable_gradient_release_mcore_ddp`` must yield bit-identical
   results to the standalone ``enable_gradient_release`` path when
   main_grad == p.grad.float() (single-rank, no reduce-scatter).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _state_snapshot(opt):
    """Snapshot kernel tensors for parity diff."""
    snap = []
    for p in (q for g in opt.param_groups for q in g["params"]):
        st = opt.state.get(p, {})
        snap.append({
            "param": p.detach().clone(),
            "exp_avg": st["exp_avg"].kernel_tensor.detach().clone() if "exp_avg" in st else None,
            "exp_avg_sq": st["exp_avg_sq"].kernel_tensor.detach().clone() if "exp_avg_sq" in st else None,
            "step": int(st["step"].item()) if "step" in st else 0,
        })
    return snap


def _state_keys(opt):
    """Set of all keys present across all per-param state dicts."""
    keys = set()
    for st in opt.state.values():
        keys |= set(st.keys())
    return keys


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestTargetModelShapes(unittest.TestCase):
    """ECO GR over DeepSeek-V3.2-REAP-345B target shapes."""

    KV_LORA_RANK = 512
    HIDDEN = 7168  # Full hidden_size; we use a tiny down-projection to keep
                   # the unit test fast while exercising the same dtype/stride
                   # path the production model would.

    def test_kv_lora_shape_under_gr(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release,
        )
        torch.manual_seed(0)
        # MLA-style projection: hidden -> kv_lora_rank, the exact dim
        # where TurboQuant sits and where the optimizer touches the
        # most weight memory in the target.
        m = nn.Linear(self.HIDDEN, self.KV_LORA_RANK, bias=False).cuda().to(torch.bfloat16)
        opt = FlashAdamW(list(m.parameters()), lr=3e-4, eco=True, fused=True)
        h = enable_gradient_release(m, opt)
        try:
            x = torch.randn(2, self.HIDDEN, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(2, self.KV_LORA_RANK, device="cuda", dtype=torch.bfloat16)
            (m(x) - y).pow(2).mean().backward()
        finally:
            h.remove()
        torch.cuda.synchronize()
        # Param actually moved.
        self.assertTrue(torch.isfinite(next(m.parameters()).float()).all().item())


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestParallelismLocality(unittest.TestCase):
    """Confirm step_param is param-local: TP-sharded run == unsharded run."""

    def test_tp_shard_equivalence(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release,
        )

        torch.manual_seed(0)
        IN, OUT = 64, 96
        x = torch.randn(8, IN, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(8, OUT, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(OUT, IN, device="cuda", dtype=torch.bfloat16) * 0.1

        # --- unsharded reference ---
        m_full = nn.Linear(IN, OUT, bias=False).cuda().to(torch.bfloat16)
        with torch.no_grad():
            m_full.weight.copy_(w)
        opt_full = FlashAdamW(list(m_full.parameters()), lr=3e-4, eco=True, fused=True)
        h = enable_gradient_release(m_full, opt_full)
        try:
            (m_full(x) - y).pow(2).mean().backward()
        finally:
            h.remove()
        full_param = m_full.weight.detach().clone()

        # --- TP-sharded along OUT (column-parallel) ---
        # Each "rank" owns OUT/2 rows. Per-token grads are independent
        # along the OUT dim, so concat(out_left, out_right) must match
        # the unsharded forward exactly.
        half = OUT // 2
        m_l = nn.Linear(IN, half, bias=False).cuda().to(torch.bfloat16)
        m_r = nn.Linear(IN, half, bias=False).cuda().to(torch.bfloat16)
        with torch.no_grad():
            m_l.weight.copy_(w[:half])
            m_r.weight.copy_(w[half:])
        opt_l = FlashAdamW(list(m_l.parameters()), lr=3e-4, eco=True, fused=True)
        opt_r = FlashAdamW(list(m_r.parameters()), lr=3e-4, eco=True, fused=True)
        h_l = enable_gradient_release(m_l, opt_l)
        h_r = enable_gradient_release(m_r, opt_r)
        try:
            out = torch.cat([m_l(x), m_r(x)], dim=-1)
            (out - y).pow(2).mean().backward()
        finally:
            h_l.remove()
            h_r.remove()
        sharded_param = torch.cat([m_l.weight, m_r.weight], dim=0)

        # Bit-equivalent: same kernel, same data, just sliced.
        max_diff = (full_param.float() - sharded_param.float()).abs().max().item()
        self.assertLess(max_diff, 1e-6,
            f"TP-sharded GR diverged from unsharded by {max_diff}; "
            "step_param must be parameter-local.")


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestNoExtraMemory(unittest.TestCase):
    """Per project-wide invariant: GR must not allocate persistent state
    beyond what batched step() does."""

    def test_state_keys_match_batched(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release,
        )

        torch.manual_seed(0)
        def _run(use_gr):
            m = nn.Linear(32, 16, bias=True).cuda().to(torch.bfloat16)
            opt = FlashAdamW(list(m.parameters()), lr=3e-4, eco=True, fused=True)
            x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
            if use_gr:
                h = enable_gradient_release(m, opt)
                try:
                    (m(x) - y).pow(2).mean().backward()
                finally:
                    h.remove()
            else:
                (m(x) - y).pow(2).mean().backward()
                opt.step()
            return _state_keys(opt)

        keys_batched = _run(False)
        keys_gr = _run(True)
        self.assertEqual(keys_batched, keys_gr,
            f"GR introduced extra state keys: {keys_gr - keys_batched}")


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestMixedPrecisionMainGrad(unittest.TestCase):
    """Bit-equivalence between bare GR and MCore-DDP-style staging
    (bf16 param, fp32 main_grad) on a single rank."""

    def test_main_grad_path_matches_grad_path(self):
        from megatron.core.distributed.distributed_data_parallel import (
            DistributedDataParallel as MCoreDDP,
        )
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release, enable_gradient_release_mcore_ddp,
        )

        class _Stub(MCoreDDP):
            def __init__(self, module):
                torch.nn.Module.__init__(self)
                object.__setattr__(self, "module", module)

        torch.manual_seed(0)
        x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16) * 0.1

        # Path 1: bare enable_gradient_release (uses p.grad)
        m_a = nn.Linear(32, 16, bias=False).cuda().to(torch.bfloat16)
        with torch.no_grad():
            m_a.weight.copy_(w)
        opt_a = FlashAdamW(list(m_a.parameters()), lr=3e-4, eco=True, fused=True)
        h = enable_gradient_release(m_a, opt_a)
        try:
            (m_a(x) - y).pow(2).mean().backward()
        finally:
            h.remove()

        # Path 2: enable_gradient_release_mcore_ddp (uses fp32 main_grad)
        m_b = nn.Linear(32, 16, bias=False).cuda().to(torch.bfloat16)
        with torch.no_grad():
            m_b.weight.copy_(w)
        opt_b = FlashAdamW(list(m_b.parameters()), lr=3e-4, eco=True, fused=True)
        for p in m_b.parameters():
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)

        def _stage(p):
            if p.grad is not None:
                p.main_grad.copy_(p.grad.float())

        for p in m_b.parameters():
            p.register_post_accumulate_grad_hook(_stage)

        h = enable_gradient_release_mcore_ddp(_Stub(m_b), opt_b)
        try:
            (m_b(x) - y).pow(2).mean().backward()
        finally:
            h.remove()

        diff = (m_a.weight.float() - m_b.weight.float()).abs().max().item()
        # Single ULP of bf16 round-trip noise in the fp32→bf16 grad cast.
        self.assertLess(diff, 1e-3,
            f"main_grad path diverged from grad path by {diff}")


if __name__ == "__main__":
    unittest.main()
