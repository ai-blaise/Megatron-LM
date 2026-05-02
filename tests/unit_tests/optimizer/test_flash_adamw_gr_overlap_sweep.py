# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hazard sweep for the bucket-completion overlap path.

Targets the subtle behaviours that the first-pass tests don't reach:

S1. Re-entry guard: installing twice on the same DDP raises a clear
    error. remove() restores cleanly so a subsequent install works.
S2. Graph-capture passthrough: the wrapped start_grad_sync is a no-op
    (no event recording, no opt-stream switch) when CUDA graph capture
    is in flight. The original RS dispatch still runs.
S3. finalize() makes the optimizer-stream updates visible without a
    global torch.cuda.synchronize().
S4. Exception in on_bucket_complete is contained and surfaced on
    remove(); subsequent buckets still drain.
S5. Step counter under overlap matches non-overlap exactly.
S6. Multiple successive backwards under overlap each step exactly
    once per param (no double-counting on bucket reuse).
S7. expert_parallel_bucket_groups path also gets wrapped.
"""

from __future__ import annotations

import contextlib
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _make_param(numel, dtype=torch.bfloat16, value=0.1):
    p = nn.Parameter(torch.full((numel,), value, dtype=dtype, device="cuda"))
    p.main_grad = torch.full((numel,), 1e-3, dtype=torch.float32, device="cuda")
    return p


class _MockBucket:
    def __init__(self, params):
        self.params_list = params
        self.grad_data = torch.ones(
            sum(p.numel() for p in params), dtype=torch.float32, device="cuda"
        )


class _MockBG:
    def __init__(self, buckets):
        self.buckets = buckets
        self.communication_stream = None
        self.dispatched = 0

    def start_grad_sync(self, force_all_reduce=False):
        self.dispatched += 1


def _mock_ddp(bucket_groups, *, expert=(), ddp_config=None):
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel as MCoreDDP,
    )

    class _Stub(MCoreDDP):
        def __init__(self):
            nn.Module.__init__(self)

    d = _Stub()
    object.__setattr__(d, "bucket_groups", list(bucket_groups))
    object.__setattr__(d, "expert_parallel_bucket_groups", list(expert))
    object.__setattr__(d, "ddp_config", ddp_config)
    object.__setattr__(d, "module", nn.Module())
    return d


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestReentryGuard(unittest.TestCase):

    def test_double_install_rejected(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )
        p = _make_param(8)
        opt = FlashAdamW([p], lr=1e-3, fused=True)
        ddp = _mock_ddp([_MockBG([_MockBucket([p])])])

        h1 = enable_gradient_release_mcore_ddp(
            ddp, opt, overlap_grad_reduce=True,
        )
        with self.assertRaises(RuntimeError) as ctx:
            enable_gradient_release_mcore_ddp(
                ddp, opt, overlap_grad_reduce=True,
            )
        self.assertIn("already has a scheduler attached", str(ctx.exception))
        h1.remove()

        # After remove, install must succeed again (sentinel cleared).
        h2 = enable_gradient_release_mcore_ddp(
            ddp, opt, overlap_grad_reduce=True,
        )
        h2.remove()


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestGraphCapturePassthrough(unittest.TestCase):

    def test_skip_during_graph_capture(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        p = _make_param(8)
        opt = FlashAdamW([p], lr=1e-3, fused=True)
        bg = _MockBG([_MockBucket([p])])
        ddp = _mock_ddp([bg])

        h = enable_gradient_release_mcore_ddp(
            ddp, opt, overlap_grad_reduce=True,
        )
        try:
            before = p.detach().clone()
            # Patch is_graph_capturing on the cuda_graphs module to
            # claim we're capturing.
            import megatron.core.transformer.cuda_graphs as cg
            orig = cg.is_graph_capturing
            cg.is_graph_capturing = lambda: True
            try:
                bg.start_grad_sync()
                torch.cuda.synchronize()
            finally:
                cg.is_graph_capturing = orig
            # Original RS dispatch ran (counted), but the optimizer
            # step did NOT — param unchanged.
            self.assertEqual(bg.dispatched, 1)
            self.assertTrue(torch.equal(before, p))
            self.assertNotIn(p, opt.state)  # state never initialised
        finally:
            h.remove()


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestFinalize(unittest.TestCase):

    def test_finalize_makes_updates_visible(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        p = _make_param(8)
        opt = FlashAdamW([p], lr=1e-3, fused=True)
        bg = _MockBG([_MockBucket([p])])
        ddp = _mock_ddp([bg])

        h = enable_gradient_release_mcore_ddp(
            ddp, opt, overlap_grad_reduce=True,
        )
        try:
            before = p.detach().clone()
            bg.start_grad_sync()
            # No torch.cuda.synchronize() — instead, finalize()
            # should sync just the optimizer stream.
            scheduler = h._hooks[0]
            scheduler.finalize()
            # Now reading the param on the default stream must see the
            # update.
            self.assertFalse(torch.equal(before, p))
        finally:
            h.remove()

    def test_finalize_surfaces_pending_exception(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        p = _make_param(8)
        opt = FlashAdamW([p], lr=1e-3, fused=True)
        bg = _MockBG([_MockBucket([p])])
        ddp = _mock_ddp([bg])
        boom = RuntimeError("on_bucket_complete failure")

        def bad_complete(_b):
            raise boom

        h = enable_gradient_release_mcore_ddp(
            ddp, opt, overlap_grad_reduce=True,
            on_bucket_complete=bad_complete,
        )
        bg.start_grad_sync()
        torch.cuda.synchronize()
        scheduler = h._hooks[0]
        with self.assertRaises(RuntimeError) as ctx:
            scheduler.finalize()
        self.assertIs(ctx.exception, boom)
        h.remove()  # must not re-raise (already drained)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestExceptionContainmentAcrossBuckets(unittest.TestCase):

    def test_failed_bucket_doesnt_block_subsequent(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        pa, pb = _make_param(8), _make_param(8)
        opt = FlashAdamW([pa, pb], lr=1e-3, fused=True)
        bgA = _MockBG([_MockBucket([pa])])
        bgB = _MockBG([_MockBucket([pb])])
        ddp = _mock_ddp([bgA, bgB])

        first = [True]
        def bad_complete(_b):
            if first[0]:
                first[0] = False
                raise RuntimeError("first bucket boom")
            # second bucket completes cleanly

        before_b = pb.detach().clone()
        h = enable_gradient_release_mcore_ddp(
            ddp, opt, overlap_grad_reduce=True,
            on_bucket_complete=bad_complete,
        )
        bgA.start_grad_sync()
        bgB.start_grad_sync()
        torch.cuda.synchronize()
        # Bucket B's param still got stepped despite bucket A's
        # callback failure.
        self.assertFalse(torch.equal(before_b, pb))
        with self.assertRaises(RuntimeError):
            h.remove()


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestStepSemantics(unittest.TestCase):

    def test_step_counter_matches_non_overlap(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        torch.manual_seed(0)
        N = 4
        params = [_make_param(8) for _ in range(N)]
        opt = FlashAdamW(params, lr=1e-3, fused=True)
        ddp = _mock_ddp([_MockBG([_MockBucket([p])]) for p in params])

        h = enable_gradient_release_mcore_ddp(
            ddp, opt, overlap_grad_reduce=True,
        )
        try:
            for _ in range(5):
                # refresh main_grad each iteration (mimics backward
                # populating it)
                for p in params:
                    p.main_grad.fill_(1e-3)
                for bg in ddp.bucket_groups:
                    bg.start_grad_sync()
                torch.cuda.synchronize()
        finally:
            h.remove()
        for p in params:
            self.assertEqual(int(opt.state[p]["step"].item()), 5)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestExpertParallelGroupsWrapped(unittest.TestCase):

    def test_expert_bucket_groups_also_wrapped(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        p_dense, p_expert = _make_param(8), _make_param(8)
        opt = FlashAdamW([p_dense, p_expert], lr=1e-3, fused=True)
        bg_dense = _MockBG([_MockBucket([p_dense])])
        bg_expert = _MockBG([_MockBucket([p_expert])])
        ddp = _mock_ddp([bg_dense], expert=[bg_expert])

        before_e = p_expert.detach().clone()
        h = enable_gradient_release_mcore_ddp(
            ddp, opt, overlap_grad_reduce=True,
        )
        try:
            bg_dense.start_grad_sync()
            bg_expert.start_grad_sync()  # must trigger step too
            torch.cuda.synchronize()
            self.assertFalse(torch.equal(before_e, p_expert),
                "expert bucket group not wrapped — expert param not stepped")
        finally:
            h.remove()


if __name__ == "__main__":
    unittest.main(verbosity=2)
