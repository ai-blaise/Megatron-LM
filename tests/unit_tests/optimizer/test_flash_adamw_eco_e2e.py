# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end test driving the production FlashAdamW + ECO call chain.

Exercises the same path the distributed optimizer uses in production:

    distrib_optimizer._inject_nvfp4_eco_errors
        -> FlashAdamW.inject_eco_error(state_key, pre_cast, post_cast)
            -> _fused_eco_inject(...)
                -> _triton_eco_inject_kernel  (the autotuned kernel)

We synthesize a mini FlashAdamW instance with one parameter, drive a
single optimizer step to materialize state, then issue an inject call
with synthetic pre_cast / post_cast tensors that mimic what the
distributed optimizer passes after the NVFP4 cast. The test asserts
that the momentum buffer is non-trivially modified and remains finite
after the inject — the contract that the live training loop depends on.

Runs on CUDA only; gated to skip on CPU-only hosts.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestFlashAdamWEcoEndToEnd(unittest.TestCase):

    def _build(self, N: int, *, quantize: bool):
        from megatron.core.optimizer.flash_optimizers import FlashAdamW

        p = torch.nn.Parameter(
            torch.randn(N, dtype=torch.bfloat16, device="cuda") * 1e-3
        )
        kwargs = dict(
            lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
            eco=True, fused=True,
        )
        if quantize:
            kwargs["quantize"] = True
        opt = FlashAdamW([p], **kwargs)
        # Materialize optimizer state via a single .step()
        p.grad = torch.randn_like(p)
        opt.step()
        p.grad = None
        return opt, p

    def _drive_inject(self, N: int, quantize: bool):
        opt, p = self._build(N, quantize=quantize)
        state = opt.state[p]
        mom_before = state["exp_avg"].kernel_tensor.clone()
        pre = torch.randn(N, dtype=torch.bfloat16, device="cuda") * 1e-3
        post = pre + (torch.randn_like(pre) * 1e-5)
        opt.inject_eco_error(p, pre, post)
        torch.cuda.synchronize()
        mom_after = state["exp_avg"].kernel_tensor
        self.assertEqual(mom_after.shape, mom_before.shape)
        self.assertTrue(torch.isfinite(mom_after.float()).all().item(),
            "non-finite momentum after inject (N={}, quantize={})".format(N, quantize))
        self.assertGreater((mom_after != mom_before).sum().item(), 0,
            "momentum unchanged after inject (N={}, quantize={})".format(N, quantize))

    def test_quant_small(self):
        self._drive_inject(262_144, quantize=True)

    def test_quant_medium(self):
        self._drive_inject(4_194_304, quantize=True)

    def test_quant_large(self):
        self._drive_inject(16_777_216, quantize=True)

    def test_fp32_small(self):
        self._drive_inject(262_144, quantize=False)

    def test_fp32_medium(self):
        self._drive_inject(4_194_304, quantize=False)

    def test_fp32_large(self):
        self._drive_inject(16_777_216, quantize=False)


if __name__ == "__main__":
    unittest.main()
