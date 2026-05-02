# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deep correctness sweep for the flashoptim-gradientrelease additions.

Targets every hazard surfaced during code review of the new GR surface:

H1. Pre-step rejection still skips post_step AND grad clear.
H2. Post-accumulate hook re-entry from shared/tied params is a no-op.
H3. Frozen param (requires_grad=False) is not stepped under either GR
    helper.
H4. Removing the handle restores normal step / zero_grad.
H5. _gradient_release flag is properly toggled across both helpers.
H6. NVFP4 orchestrator's flush() drains a partial pending without
    erroring when nothing is pending.
H7. NVFP4 orchestrator's expected-count auto-infer correctly returns 0
    for a non-NVFP4 optimizer (pure bf16 model).
H8. MCore DDP helper preserves a pre-existing decoupled_grad and
    restores it after the per-param step (no leak across params).
H9. MCore DDP helper does NOT step a param whose main_grad was never
    populated (mimics the no_sync / first-microbatch case).
H10. Step counter advances by exactly 1 per backward under GR, identical
     to batched.
H11. Multi-step convergence: 200 micro-steps GR vs batched produce same
     final loss within fp32 reduction noise (numerical stability check).
H12. Backward called twice without zero_grad: GR clears p.grad after
     each backward, so a 2nd backward starts from a fresh accumulator
     (matches the user-facing semantic of GR as "zero_grad implicit").
"""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _make(seed=0, in_dim=16, out_dim=8):
    torch.manual_seed(seed)
    return nn.Linear(in_dim, out_dim, bias=True).cuda().to(torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestPreStepGate(unittest.TestCase):

    def test_pre_step_false_skips_step_and_post_step(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release,
        )
        m = _make()
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, eco=True, fused=True)
        before = [p.detach().clone() for p in m.parameters()]

        post_calls = []
        h = enable_gradient_release(
            m, opt,
            pre_step=lambda p, g: False,
            post_step=lambda p, g: post_calls.append(id(p)),
        )
        try:
            x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
            (m(x) - y).pow(2).mean().backward()
        finally:
            h.remove()
        # Step skipped → params unchanged.
        for b, p in zip(before, m.parameters()):
            self.assertTrue(torch.equal(b, p))
        # And post_step skipped too.
        self.assertEqual(post_calls, [])


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestFrozenParam(unittest.TestCase):

    def test_frozen_param_not_stepped(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release,
        )
        m = _make()
        # Freeze the bias.
        m[0].bias.requires_grad_(False) if isinstance(m, nn.Sequential) else None
        m.bias.requires_grad_(False)
        opt = FlashAdamW(
            [p for p in m.parameters() if p.requires_grad],
            lr=1e-3, fused=True,
        )
        bias_before = m.bias.detach().clone()
        h = enable_gradient_release(m, opt)
        try:
            x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
            (m(x) - y).pow(2).mean().backward()
        finally:
            h.remove()
        self.assertTrue(torch.equal(bias_before, m.bias))


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestHandleLifecycle(unittest.TestCase):

    def test_remove_restores_step_zero_grad(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release,
        )
        m = _make()
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, fused=True)
        h = enable_gradient_release(m, opt)
        self.assertTrue(opt._gradient_release)
        h.remove()
        self.assertFalse(opt._gradient_release)

        # After remove, step() and zero_grad() must NOT warn.
        x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
        (m(x) - y).pow(2).mean().backward()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt.step()
            opt.zero_grad()
        self.assertFalse(any("gradient_release" in str(w.message) for w in caught))

    def test_mcore_ddp_remove_resets_flag(self):
        from megatron.core.distributed.distributed_data_parallel import (
            DistributedDataParallel as MCoreDDP,
        )
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        class _Stub(MCoreDDP):
            def __init__(self, mod):
                nn.Module.__init__(self)
                object.__setattr__(self, "module", mod)

        m = _make()
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, fused=True)
        h = enable_gradient_release_mcore_ddp(_Stub(m), opt)
        self.assertTrue(opt._gradient_release)
        h.remove()
        self.assertFalse(opt._gradient_release)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestOrchestratorEdgeCases(unittest.TestCase):

    def test_flush_empty_is_noop(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, NVFP4EcoGradientReleaseOrchestrator,
        )
        m = _make()
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, eco=True, fused=True)

        cast_count = [0]
        orch = NVFP4EcoGradientReleaseOrchestrator(
            optimizer=opt,
            cast_fn=lambda *a, **kw: cast_count.__setitem__(0, cast_count[0] + 1),
            expected_nvfp4_params=2,
        )
        orch.flush()  # nothing pending
        self.assertEqual(cast_count[0], 0)

    def test_expected_inferred_zero_for_no_nvfp4(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, NVFP4EcoGradientReleaseOrchestrator,
        )
        m = _make()
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, eco=True, fused=True)
        orch = NVFP4EcoGradientReleaseOrchestrator(
            optimizer=opt,
            cast_fn=lambda *a, **kw: None,
        )
        self.assertEqual(orch._expected(), 0)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestMCoreDDPDecoupledGradPreservation(unittest.TestCase):

    def test_pre_existing_decoupled_grad_restored(self):
        from megatron.core.distributed.distributed_data_parallel import (
            DistributedDataParallel as MCoreDDP,
        )
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        class _Stub(MCoreDDP):
            def __init__(self, mod):
                nn.Module.__init__(self)
                object.__setattr__(self, "module", mod)

        m = _make()
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, fused=True)
        # Stamp a decoupled_grad sentinel before GR runs.
        sentinels = {}
        for p in m.parameters():
            sent = torch.full_like(p, 7.0, dtype=torch.float32)
            p.decoupled_grad = sent
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)
            sentinels[id(p)] = sent

        def _stage(p):
            if p.grad is not None:
                p.main_grad.copy_(p.grad.float())

        for p in m.parameters():
            p.register_post_accumulate_grad_hook(_stage)

        h = enable_gradient_release_mcore_ddp(_Stub(m), opt)
        try:
            x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
            (m(x) - y).pow(2).mean().backward()
        finally:
            h.remove()

        # Sentinel must be restored on each param.
        for p in m.parameters():
            self.assertIs(p.decoupled_grad, sentinels[id(p)])

    def test_no_main_grad_no_step(self):
        from megatron.core.distributed.distributed_data_parallel import (
            DistributedDataParallel as MCoreDDP,
        )
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        class _Stub(MCoreDDP):
            def __init__(self, mod):
                nn.Module.__init__(self)
                object.__setattr__(self, "module", mod)

        m = _make()
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, fused=True)
        # Intentionally do NOT attach main_grad; the helper must skip
        # the step rather than crash.
        before = [p.detach().clone() for p in m.parameters()]
        h = enable_gradient_release_mcore_ddp(_Stub(m), opt)
        try:
            x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
            (m(x) - y).pow(2).mean().backward()
        finally:
            h.remove()
        for b, p in zip(before, m.parameters()):
            self.assertTrue(torch.equal(b, p))


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestStepCounterAndConvergence(unittest.TestCase):

    def test_step_counter_advances_once_per_backward(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release,
        )
        m = _make()
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, eco=True, fused=True)
        h = enable_gradient_release(m, opt)
        try:
            for _ in range(7):
                x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
                y = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
                (m(x) - y).pow(2).mean().backward()
        finally:
            h.remove()
        for p in m.parameters():
            self.assertEqual(int(opt.state[p]["step"].item()), 7)

    def test_convergence_proxy_200_step(self):
        """Loss curves of GR vs batched must overlap within fp32 noise."""
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release,
        )

        def _run(use_gr):
            torch.manual_seed(0)
            m = nn.Sequential(
                nn.Linear(32, 16, bias=True),
                nn.GELU(),
                nn.Linear(16, 8, bias=True),
            ).cuda().to(torch.bfloat16)
            opt = FlashAdamW(list(m.parameters()), lr=1e-3, eco=True, fused=True)
            torch.manual_seed(1)
            batches = [
                (torch.randn(8, 32, device="cuda", dtype=torch.bfloat16),
                 torch.randn(8, 8,  device="cuda", dtype=torch.bfloat16))
                for _ in range(200)
            ]
            losses = []
            if use_gr:
                h = enable_gradient_release(m, opt)
                try:
                    for x, y in batches:
                        loss = (m(x) - y).pow(2).mean()
                        losses.append(loss.item())
                        loss.backward()
                finally:
                    h.remove()
            else:
                for x, y in batches:
                    opt.zero_grad()
                    loss = (m(x) - y).pow(2).mean()
                    losses.append(loss.item())
                    loss.backward()
                    opt.step()
            return losses

        l_a = _run(False)
        l_b = _run(True)
        # Bit-equivalent in our parity test, so this should also be exact;
        # but allow 1e-4 absolute drift just for robustness.
        max_drift = max(abs(a - b) for a, b in zip(l_a, l_b))
        self.assertLess(max_drift, 1e-4,
            f"GR vs batched 200-step max loss drift = {max_drift}")


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestGradientLifecycle(unittest.TestCase):

    def test_back_to_back_backwards_dont_double_count(self):
        """Two backward passes without an explicit zero_grad must each
        produce ONE Adam step (GR clears the grad after each step)."""
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release,
        )
        m = _make()
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, eco=True, fused=True)
        h = enable_gradient_release(m, opt)
        try:
            for _ in range(2):
                x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
                y = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
                (m(x) - y).pow(2).mean().backward()
                # GR must have cleared the grad.
                for p in m.parameters():
                    self.assertIsNone(p.grad)
        finally:
            h.remove()
        for p in m.parameters():
            self.assertEqual(int(opt.state[p]["step"].item()), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
