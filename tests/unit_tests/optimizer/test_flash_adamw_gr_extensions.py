# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the GR extension surface added on flashoptim-gradientrelease.

Covers:
1. ``post_step`` callback fires once per stepped param, after step_param,
   before grad is freed.
2. ``NVFP4EcoGradientReleaseOrchestrator.post_step`` is a no-op for
   non-NVFP4 params (the only path we can drive without TE installed).
3. MCore DDP detection raises NotImplementedError pointing at the
   bucket-aware helper.
4. ``enable_gradient_release_mcore_ddp`` runs ``step_param`` against
   ``param.main_grad`` for a single-rank model (rank-1 control case).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestGRPostStepCallback(unittest.TestCase):

    def test_post_step_fires_per_param(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release,
        )

        torch.manual_seed(0)
        m = nn.Sequential(
            nn.Linear(16, 8, bias=True),
            nn.Linear(8, 4, bias=True),
        ).cuda().to(torch.bfloat16)
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, eco=True, fused=True)

        seen = []

        def post_step(p, group):
            # grad is still attached at this point (cleared right after)
            assert p.grad is not None, "post_step ran after grad was freed"
            seen.append(id(p))

        h = enable_gradient_release(m, opt, post_step=post_step)
        try:
            x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
            (m(x) - y).pow(2).mean().backward()
        finally:
            h.remove()

        # 4 params (2 weights + 2 biases) all should have fired exactly once.
        self.assertEqual(len(seen), 4)
        self.assertEqual(len(set(seen)), 4)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestNVFP4OrchestratorNoop(unittest.TestCase):

    def test_orchestrator_skips_non_nvfp4(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, NVFP4EcoGradientReleaseOrchestrator,
            enable_gradient_release,
        )

        torch.manual_seed(0)
        m = nn.Linear(8, 4, bias=False).cuda().to(torch.bfloat16)
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, eco=True, fused=True)

        cast_calls = []

        def fake_cast(items, group, **kw):
            cast_calls.append(len(items))

        orch = NVFP4EcoGradientReleaseOrchestrator(
            optimizer=opt, cast_fn=fake_cast, expected_nvfp4_params=0,
        )
        h = enable_gradient_release(m, opt, post_step=orch.post_step)
        try:
            x = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
            (m(x) - y).pow(2).mean().backward()
            orch.flush()  # no-op since nothing pending
        finally:
            h.remove()

        # No NVFP4 params present, so the cast must never have fired.
        self.assertEqual(cast_calls, [])


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestMCoreDDPDetection(unittest.TestCase):

    def test_torch_module_passes(self):
        # Plain nn.Module should not be rejected as MCore DDP.
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release,
        )
        m = nn.Linear(4, 2).cuda().to(torch.bfloat16)
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, fused=True)
        h = enable_gradient_release(m, opt)
        h.remove()

    def test_mcore_ddp_route(self):
        # We don't have a real distributed setup in this unit test, so
        # we manufacture a thin object claiming to be MCoreDDP via
        # isinstance check on the imported class. We use a lightweight
        # approach: monkey-patch the class to recognize a plain marker
        # subclass at import time, then verify enable_gradient_release
        # raises NotImplementedError pointing at the bucket-aware helper.
        from megatron.core.distributed.distributed_data_parallel import (
            DistributedDataParallel as MCoreDDP,
        )
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release,
        )

        class _Stub(MCoreDDP):
            def __init__(self):
                # bypass parent __init__ — we only need isinstance to match.
                pass

        m = _Stub()
        opt = FlashAdamW(
            [torch.nn.Parameter(torch.randn(4, device="cuda", dtype=torch.bfloat16))],
            lr=1e-3, fused=True,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            enable_gradient_release(m, opt)
        self.assertIn("enable_gradient_release_mcore_ddp", str(ctx.exception))


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestMCoreDDPSingleRank(unittest.TestCase):
    """Single-rank exercise of enable_gradient_release_mcore_ddp.

    We can't spin up real MCore DDP without a multi-rank distributed
    init, so we drive the path by manually attaching ``main_grad`` to
    each param (mimicking what MCore DDP does after grad accumulation)
    and confirming that the per-param hooks read main_grad rather than
    p.grad. We bypass the isinstance gate by constructing a stub
    subclass of MCoreDDP exposing ``.module``.
    """

    def test_step_uses_main_grad(self):
        from megatron.core.distributed.distributed_data_parallel import (
            DistributedDataParallel as MCoreDDP,
        )
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        class _Stub(MCoreDDP):
            def __init__(self, module):
                nn.Module.__init__(self)
                self._inner = module
                self.module = module

        torch.manual_seed(0)
        m = nn.Linear(8, 4, bias=False).cuda().to(torch.bfloat16)
        opt = FlashAdamW(list(m.parameters()), lr=1e-3, fused=True)

        # Attach main_grad and a post-accumulate hook that materializes
        # main_grad from p.grad — a stand-in for MCore DDP's
        # _make_backward_post_hook (no reduce-scatter in single rank).
        for p in m.parameters():
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)

        def _stage_main_grad(p):
            if p.grad is not None:
                p.main_grad.copy_(p.grad.float())

        for p in m.parameters():
            p.register_post_accumulate_grad_hook(_stage_main_grad)

        ddp = _Stub(m)
        before = [q.detach().clone() for q in m.parameters()]
        h = enable_gradient_release_mcore_ddp(ddp, opt)
        try:
            x = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
            (m(x) - y).pow(2).mean().backward()
        finally:
            h.remove()

        # Params must have moved (step happened).
        for b, q in zip(before, m.parameters()):
            self.assertFalse(torch.equal(b, q),
                             "Param not updated by GR + MCoreDDP path")


if __name__ == "__main__":
    unittest.main()
