# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression test for the autotune state-corruption bug in
``_triton_eco_inject_kernel``.

The other correctness oracle clones inputs per call and so does NOT detect
state corruption from autotune sweeping multiple in-place configs over
the same momentum buffer. This test does the opposite: it does NOT clone
between calls, then verifies that the **first** call for a fresh
(N, dtype, quantize) shape produces the same momentum as the **second**
call (which hits the autotune cache and runs the kernel exactly once).

Without ``restore_value`` on the autotune decorator, the first call would
sweep all 45 configs in-place, applying the inject 45 extra times before
the real call — momentum after the first call would diverge wildly from
the second call's. With ``restore_value`` set, both calls produce the
same one-update-worth-of-momentum.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestEcoInjectAutotuneRestoreValue(unittest.TestCase):

    def _drive_first_call_vs_cached(self, N: int, *, quantize: bool):
        """Call the kernel once on fresh state (autotune sweeps), then again
        on identically-prepared fresh state (autotune cache hit, single
        kernel run). The momenta after each call must be identical."""

        from megatron.core.optimizer.flash_optimizers import _fused_eco_inject

        torch.manual_seed(N)
        if quantize:
            mom_seed = torch.randint(-127, 127, (N,), dtype=torch.int8, device="cuda")
            mom_scales_seed = torch.full(
                (N // 32,), 1e-3, dtype=torch.float16, device="cuda"
            )
        else:
            mom_seed = torch.randn(N, dtype=torch.float32, device="cuda") * 1e-3
            mom_scales_seed = torch.empty(0, dtype=torch.float16, device="cuda")
        var = (torch.randn(N, dtype=torch.float32, device="cuda") * 1e-3) ** 2
        if quantize:
            var_q = (var.sqrt() * 1e3).clamp(0, 255).to(torch.uint8)
            var_scales = torch.full((N // 32,), 1e-3, dtype=torch.float16, device="cuda")
        else:
            var_q = var
            var_scales = torch.empty(0, dtype=torch.float16, device="cuda")
        pre = torch.randn(N, dtype=torch.bfloat16, device="cuda") * 1e-3
        post = pre + (torch.randn_like(pre) * 1e-5)

        # First call: fresh state, autotune may sweep (cache miss).
        mom_a = mom_seed.clone()
        mom_scales_a = mom_scales_seed.clone()
        _fused_eco_inject(
            mom=mom_a, mom_scales_f16=mom_scales_a,
            var=var_q, var_scales_f16=var_scales,
            pre_cast=pre, post_cast=post,
            eco_scalar=0.7, eps=1e-8, bc2=0.999,
            quantize_optim_states=quantize,
        )
        torch.cuda.synchronize()

        # Second call: fresh state again, autotune cache hit (one run only).
        mom_b = mom_seed.clone()
        mom_scales_b = mom_scales_seed.clone()
        _fused_eco_inject(
            mom=mom_b, mom_scales_f16=mom_scales_b,
            var=var_q, var_scales_f16=var_scales,
            pre_cast=pre, post_cast=post,
            eco_scalar=0.7, eps=1e-8, bc2=0.999,
            quantize_optim_states=quantize,
        )
        torch.cuda.synchronize()

        if quantize:
            self.assertTrue(torch.equal(mom_a, mom_b),
                "first-call vs cache-hit momentum diverge at N={}, quantize=True".format(N))
            self.assertTrue(torch.equal(mom_scales_a, mom_scales_b),
                "first-call vs cache-hit mom_scales diverge at N={}".format(N))
        else:
            torch.testing.assert_close(mom_a, mom_b, rtol=0, atol=0)

    # Cover three sizes that hit different autotune configs.
    def test_quant_small(self):
        self._drive_first_call_vs_cached(262_144, quantize=True)

    def test_quant_medium(self):
        self._drive_first_call_vs_cached(4_194_304, quantize=True)

    def test_quant_large(self):
        self._drive_first_call_vs_cached(16_777_216, quantize=True)

    def test_fp32_small(self):
        self._drive_first_call_vs_cached(262_144, quantize=False)


if __name__ == "__main__":
    unittest.main()
