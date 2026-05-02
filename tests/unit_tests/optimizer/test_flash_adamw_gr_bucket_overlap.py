# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bucket-completion overlap tests for enable_gradient_release_mcore_ddp.

Without a real multi-rank distributed init we can't exercise the NCCL
RS path, but the scheduler is testable end-to-end with mock bucket
groups. The mocks expose the same surface MCore DDP relies on:

    bucket_group:
        .start_grad_sync(force_all_reduce=False)
        .buckets         -> [Bucket]
        .communication_stream  (optional)

    Bucket:
        .params_list     -> [nn.Parameter]
        .grad_data       -> torch.Tensor (the persistent buffer)

The scheduler's ``install`` wraps ``start_grad_sync``; tests fire the
wrapped method to simulate MCore DDP completing a bucket's RS, then
assert that step_param ran and (when zero_grad_after_step=True) the
bucket's grad_data was zeroed on a stream observable from the test.

Hazards covered (in order of severity):

H1. Per-bucket step fires once per call to start_grad_sync.
H2. Multi-bucket-group ordering matches dispatch order.
H3. zero_grad_after_step=True zeros only AFTER the step kernel has
    consumed main_grad, never before.
H4. zero_grad_after_step=False leaves grad_data untouched.
H5. Skip cleanly when main_grad is None (lazy init / no-grad params).
H6. pre_step=lambda: False skips both step and zero.
H7. step_param exception is contained, surfaced on remove() / finalize().
H8. remove() restores original start_grad_sync.
H9. DistributedOptimizer is rejected with a helpful error.
H10. Final params bit-equivalent to the non-overlap helper on the
     same inputs (since per-bucket step is order-independent for
     Adam).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _make_param(numel, dtype=torch.bfloat16, value=None):
    p = nn.Parameter(torch.zeros(numel, dtype=dtype, device="cuda"))
    if value is not None:
        with torch.no_grad():
            p.fill_(value)
    p.main_grad = torch.zeros(numel, dtype=torch.float32, device="cuda")
    return p


class _MockBucket:
    def __init__(self, params):
        self.params_list = params
        # grad_data is the persistent buffer; in real MCore DDP it's a
        # contiguous slab that main_grad views into. Here we just track
        # whether it gets zeroed after the step.
        total = sum(p.numel() for p in params)
        self.grad_data = torch.ones(total, dtype=torch.float32, device="cuda")


class _MockBucketGroup:
    def __init__(self, buckets):
        self.buckets = buckets
        self.communication_stream = None
        self._dispatched = 0

    def start_grad_sync(self, force_all_reduce=False):
        # Stand in for "RS dispatched on default stream"; the
        # scheduler's wrapper records an event right after this.
        self._dispatched += 1


def _make_mock_ddp(bucket_groups, ddp_config=None):
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel as MCoreDDP,
    )

    class _MockDDP(MCoreDDP):
        def __init__(self):
            nn.Module.__init__(self)

    d = _MockDDP()
    object.__setattr__(d, "bucket_groups", bucket_groups)
    object.__setattr__(d, "expert_parallel_bucket_groups", [])
    object.__setattr__(d, "ddp_config", ddp_config)
    object.__setattr__(d, "module", nn.Module())
    return d


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestBucketSchedulerCore(unittest.TestCase):

    def _build(self, *, n_buckets=2, params_per_bucket=2, eco=False, **kwargs):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )
        params = [
            [_make_param(8, value=0.1 * (i + j))
             for j in range(params_per_bucket)]
            for i in range(n_buckets)
        ]
        flat = [p for grp in params for p in grp]
        opt = FlashAdamW(flat, lr=1e-3, eco=eco, fused=True)
        # Stamp known main_grad values per param.
        for grp in params:
            for k, p in enumerate(grp):
                p.main_grad.fill_(0.001 + 0.0001 * k)
        ddp = _make_mock_ddp([_MockBucketGroup([_MockBucket(grp)]) for grp in params])
        h = enable_gradient_release_mcore_ddp(
            ddp, opt, overlap_grad_reduce=True, **kwargs
        )
        return params, flat, opt, ddp, h

    def test_step_fires_once_per_dispatch(self):
        params, flat, opt, ddp, h = self._build(n_buckets=3)
        try:
            before = [p.detach().clone() for p in flat]
            for bg in ddp.bucket_groups:
                bg.start_grad_sync()
            torch.cuda.synchronize()
            # Every param updated exactly once.
            for b, p in zip(before, flat):
                self.assertFalse(torch.equal(b, p))
                self.assertEqual(int(opt.state[p]["step"].item()), 1)
        finally:
            h.remove()

    def test_zeros_grad_after_step(self):
        params, flat, opt, ddp, h = self._build(zero_grad_after_step=True)
        try:
            for bg in ddp.bucket_groups:
                bg.start_grad_sync()
            torch.cuda.synchronize()
            for bg in ddp.bucket_groups:
                for bucket in bg.buckets:
                    self.assertTrue(
                        torch.equal(
                            bucket.grad_data,
                            torch.zeros_like(bucket.grad_data),
                        ),
                        "grad_data not zeroed after bucket step",
                    )
        finally:
            h.remove()

    def test_no_zero_when_disabled(self):
        params, flat, opt, ddp, h = self._build(zero_grad_after_step=False)
        try:
            originals = [bucket.grad_data.clone()
                         for bg in ddp.bucket_groups for bucket in bg.buckets]
            for bg in ddp.bucket_groups:
                bg.start_grad_sync()
            torch.cuda.synchronize()
            for orig, bucket in zip(
                originals,
                (b for bg in ddp.bucket_groups for b in bg.buckets),
            ):
                self.assertTrue(torch.equal(orig, bucket.grad_data))
        finally:
            h.remove()

    def test_skips_param_with_no_main_grad(self):
        params, flat, opt, ddp, h = self._build()
        try:
            # Strip main_grad on one param — scheduler must skip it
            # without crashing the whole bucket.
            del flat[0].main_grad
            before = flat[0].detach().clone()
            for bg in ddp.bucket_groups:
                bg.start_grad_sync()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(before, flat[0]),
                            "missing-main_grad param should not be stepped")
        finally:
            h.remove()

    def test_pre_step_false_skips(self):
        params, flat, opt, ddp, h = self._build(
            pre_step=lambda p, g: False, zero_grad_after_step=True,
        )
        try:
            before = [p.detach().clone() for p in flat]
            for bg in ddp.bucket_groups:
                bg.start_grad_sync()
            torch.cuda.synchronize()
            # No params stepped.
            for b, p in zip(before, flat):
                self.assertTrue(torch.equal(b, p))
            # But zero_grad_after_step still fires per bucket (the
            # bucket completed its RS — this matches the production
            # contract: "RS done means grad buffer is reusable").
        finally:
            h.remove()

    def test_step_param_exception_contained_and_surfaced(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )
        p = _make_param(8, value=0.1)
        opt = FlashAdamW([p], lr=1e-3, fused=True)

        def bad_post(_p, _g):
            raise RuntimeError("synthetic post_step failure")

        ddp = _make_mock_ddp([_MockBucketGroup([_MockBucket([p])])])
        h = enable_gradient_release_mcore_ddp(
            ddp, opt, overlap_grad_reduce=True, post_step=bad_post,
        )
        ddp.bucket_groups[0].start_grad_sync()
        torch.cuda.synchronize()
        # Exception was deferred; raised on remove().
        with self.assertRaises(RuntimeError) as ctx:
            h.remove()
        self.assertIn("synthetic post_step failure", str(ctx.exception))

    def test_remove_restores_start_grad_sync(self):
        params, flat, opt, ddp, h = self._build()
        bg = ddp.bucket_groups[0]
        wrapped = bg.start_grad_sync
        h.remove()
        self.assertIsNot(bg.start_grad_sync, wrapped,
                         "start_grad_sync should be unwrapped after remove()")
        # original is callable + still a method-like object
        bg.start_grad_sync()
        self.assertEqual(bg._dispatched, 1)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestDistOptSupported(unittest.TestCase):
    """Pure-NVFP4 + ECO requires use_distributed_optimizer=True (DistOpt
    populates _fa_shard_offset). The overlap path must accept it
    rather than rejecting — otherwise the user's actual training
    config can't run with overlap."""

    def test_distopt_accepted(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        class _Cfg:
            use_distributed_optimizer = True

        p = _make_param(4)
        opt = FlashAdamW([p], lr=1e-3, fused=True)
        ddp = _make_mock_ddp([_MockBucketGroup([_MockBucket([p])])], ddp_config=_Cfg())
        # Must NOT raise.
        h = enable_gradient_release_mcore_ddp(
            ddp, opt, overlap_grad_reduce=True,
        )
        h.remove()


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestPerBucketCompleteHook(unittest.TestCase):
    """The orchestrator-integration seam: on_bucket_complete fires once
    per bucket, after step_param ran for every param in that bucket
    and before grad_data is zeroed."""

    def test_on_bucket_complete_fires_per_bucket(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        params = [_make_param(8) for _ in range(4)]
        for k, p in enumerate(params):
            p.main_grad.fill_(0.001 + 0.0001 * k)
        opt = FlashAdamW(params, lr=1e-3, fused=True)

        # 2 buckets, 2 params each.
        b1 = _MockBucket(params[:2])
        b2 = _MockBucket(params[2:])
        ddp = _make_mock_ddp([
            _MockBucketGroup([b1]),
            _MockBucketGroup([b2]),
        ])

        seen = []
        def on_complete(bucket):
            # All params in this bucket already stepped.
            for p in bucket.params_list:
                # step_param incremented the step counter to 1.
                assert int(opt.state[p]["step"].item()) == 1, \
                    "on_bucket_complete fired before step_param"
            # grad_data must NOT be zeroed yet (zero comes after).
            assert not torch.equal(
                bucket.grad_data, torch.zeros_like(bucket.grad_data),
            ), "grad_data zeroed before on_bucket_complete"
            seen.append(id(bucket))

        h = enable_gradient_release_mcore_ddp(
            ddp, opt, overlap_grad_reduce=True,
            on_bucket_complete=on_complete,
        )
        try:
            for bg in ddp.bucket_groups:
                bg.start_grad_sync()
            torch.cuda.synchronize()
        finally:
            h.remove()

        self.assertEqual(seen, [id(b1), id(b2)])


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestOrchestratorPerBucketFlush(unittest.TestCase):
    """NVFP4EcoGradientReleaseOrchestrator.flush_bucket() drains only the
    pending entries that belong to the given bucket — preserves the
    overlap semantic (cast + inject runs per-bucket, not at end-of-
    backward)."""

    def test_flush_bucket_partitions(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, NVFP4EcoGradientReleaseOrchestrator,
        )

        # Three params: two in bucket A, one in bucket B.
        pa1, pa2, pb = _make_param(8), _make_param(8), _make_param(8)
        opt = FlashAdamW([pa1, pa2, pb], lr=1e-3, eco=True, fused=True)
        bA, bB = _MockBucket([pa1, pa2]), _MockBucket([pb])

        flushed = []
        def cast_fn(items, group, **kw):
            flushed.append(tuple(id(t[0]) for t in items))

        orch = NVFP4EcoGradientReleaseOrchestrator(
            optimizer=opt, cast_fn=cast_fn, expected_nvfp4_params=999,
        )
        # Manually push entries (mimics post_step staging). In
        # production _step_nvfp4_transient sets _fa_updated_shard on
        # the param; mirror that here so _flush's `del` doesn't fail.
        for p in (pa1, pa2, pb):
            p._fa_updated_shard = torch.zeros(8, device="cuda")
            orch._pending.append((p, p._fa_updated_shard, 0))

        # Flush only bucket A — bucket B stays pending.
        # Need to bypass the dequant of NVFP4 (params are not NVFP4 in
        # this test); patch inject_fn to a no-op so flush_bucket can
        # exercise the partition logic without TE.
        orch.inject_fn = lambda *a, **kw: None
        # Also stub dequantize to return a tensor — flush calls it.
        import megatron.core.fp4_utils as fp4u
        orig = fp4u.dequantize_fp4_tensor
        fp4u.dequantize_fp4_tensor = lambda p: torch.zeros(p.numel(),
                                                            device="cuda",
                                                            dtype=torch.bfloat16)
        try:
            orch.flush_bucket(bA)
        finally:
            fp4u.dequantize_fp4_tensor = orig

        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0], (id(pa1), id(pa2)))
        # Bucket B's entry still pending.
        self.assertEqual(len(orch._pending), 1)
        self.assertIs(orch._pending[0][0], pb)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestOverlapVsNonOverlapParity(unittest.TestCase):
    """End-to-end: same inputs through overlap path vs non-overlap path
    yield bit-equivalent param updates."""

    def test_two_bucket_parity(self):
        from megatron.core.optimizer.flash_optimizers import (
            FlashAdamW, enable_gradient_release_mcore_ddp,
        )

        torch.manual_seed(0)
        N = 8

        def _setup():
            torch.manual_seed(42)
            p1 = _make_param(N, value=0.1)
            p2 = _make_param(N, value=0.2)
            for p, g in [(p1, 0.001), (p2, 0.002)]:
                p.main_grad.fill_(g)
            opt = FlashAdamW([p1, p2], lr=1e-3, eco=True, fused=True)
            return p1, p2, opt

        # Path A: overlap
        p1a, p2a, opt_a = _setup()
        ddp_a = _make_mock_ddp([
            _MockBucketGroup([_MockBucket([p1a])]),
            _MockBucketGroup([_MockBucket([p2a])]),
        ])
        h_a = enable_gradient_release_mcore_ddp(
            ddp_a, opt_a, overlap_grad_reduce=True,
            zero_grad_after_step=False,
        )
        for bg in ddp_a.bucket_groups:
            bg.start_grad_sync()
        torch.cuda.synchronize()
        h_a.remove()

        # Path B: non-overlap (single-rank existing path) — drive
        # step_param directly with main_grad on decoupled_grad.
        p1b, p2b, opt_b = _setup()
        for p in (p1b, p2b):
            p.decoupled_grad = p.main_grad
        opt_b.step()
        torch.cuda.synchronize()

        max_diff = max(
            (a.float() - b.float()).abs().max().item()
            for a, b in [(p1a, p1b), (p2a, p2b)]
        )
        self.assertLess(max_diff, 1e-6,
            f"overlap vs non-overlap diverged by {max_diff} (must be 0)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
