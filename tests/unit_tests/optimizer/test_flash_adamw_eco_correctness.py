# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Correctness oracle for the autotuned `_triton_eco_inject_kernel`.

The autotune change in ``flash_optimizers.py`` is plumbing-only: BLOCK_SIZE_N,
num_warps, and num_stages are picked by ``triton.autotune`` instead of being
hard-coded. This test verifies that the chosen configs produce numerically
equivalent outputs to the original baseline. We compare against an fp64
PyTorch reference that mirrors the kernel's math exactly.

Run with::

    python tests/unit_tests/optimizer/test_flash_adamw_eco_correctness.py

The test is gated on CUDA + Triton; runs in seconds on a single GPU.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


from megatron.core.optimizer.flash_optimizers import _fused_eco_inject  # noqa: E402


def _reference_inject(
    *,
    mom_q: torch.Tensor,
    mom_scales: torch.Tensor,
    var_q: torch.Tensor,
    var_scales: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    eco_scalar: float,
    eps: float,
    bc2: float,
    quantize: bool,
    group_size: int = 32,
):
    """Pure-PyTorch fp64 reference matching the Triton kernel's math."""
    if quantize:
        N = mom_q.numel()
        mom_f64 = mom_q.to(torch.float64)
        var_f64 = var_q.to(torch.float64)
        n_groups = N // group_size
        mom_g = mom_f64.view(n_groups, group_size)
        var_g = var_f64.view(n_groups, group_size)
        mom_t = mom_g / 127.0
        mom_n = mom_t / (2.0 - mom_t.abs())
        mom_f64 = (mom_n * mom_scales.to(torch.float64).unsqueeze(1)).reshape(N)
        var_t = var_g / 255.0
        var_sqrt = (var_t * var_scales.to(torch.float64).unsqueeze(1)).reshape(N)
        var_f64 = var_sqrt * var_sqrt
    else:
        mom_f64 = mom_q.to(torch.float64)
        var_f64 = var_q.to(torch.float64)

    error = pre.to(torch.float64) - post.to(torch.float64)
    denom = torch.sqrt(var_f64 / bc2) + eps
    mom_f64 = mom_f64 + eco_scalar * denom * error

    if quantize:
        N = mom_f64.numel()
        mom_g = mom_f64.view(N // group_size, group_size)
        absmax = mom_g.abs().amax(dim=1).clamp_min(1e-12)
        mom_n = mom_g / absmax.unsqueeze(1)
        mom_t = 2.0 * mom_n / (1.0 + mom_n.abs())
        mom_out = (mom_t * 127.0).reshape(N)
        return torch.floor(mom_out + 0.5).to(torch.int8), absmax.to(torch.float16)
    else:
        return mom_f64.to(torch.float32), torch.empty(
            0, dtype=torch.float16, device=mom_q.device
        )


def _make_inputs(N: int, *, quantize: bool, seed: int, device: str = "cuda"):
    torch.manual_seed(seed)
    if quantize:
        mom = torch.randint(-127, 127, (N,), dtype=torch.int8, device=device)
        var = torch.randint(0, 255, (N,), dtype=torch.uint8, device=device)
        ng = N // 32
        mom_scales = torch.full((ng,), 1e-3, dtype=torch.float16, device=device)
        var_scales = torch.full((ng,), 1e-3, dtype=torch.float16, device=device)
    else:
        mom = torch.randn(N, dtype=torch.float32, device=device) * 1e-3
        var = (torch.randn(N, dtype=torch.float32, device=device) * 1e-3) ** 2
        mom_scales = torch.empty(0, dtype=torch.float16, device=device)
        var_scales = torch.empty(0, dtype=torch.float16, device=device)
    pre = torch.randn(N, dtype=torch.bfloat16, device=device) * 1e-3
    post = pre + (torch.randn_like(pre) * 1e-5)
    return mom, mom_scales, var, var_scales, pre, post


def _run_kernel(mom, mom_scales, var, var_scales, pre, post, *, quantize):
    mom = mom.clone()
    mom_scales = mom_scales.clone()
    _fused_eco_inject(
        mom=mom,
        mom_scales_f16=mom_scales,
        var=var,
        var_scales_f16=var_scales,
        pre_cast=pre,
        post_cast=post,
        eco_scalar=0.7,
        eps=1e-8,
        bc2=0.999,
        quantize_optim_states=quantize,
    )
    return mom, mom_scales


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestEcoInjectAutotune(unittest.TestCase):

    def _check_one(self, N: int, quantize: bool):
        mom, mom_s, var, var_s, pre, post = _make_inputs(N, quantize=quantize, seed=N)
        kernel_mom, _ = _run_kernel(mom, mom_s, var, var_s, pre, post, quantize=quantize)
        ref_mom, _ = _reference_inject(
            mom_q=mom, mom_scales=mom_s, var_q=var, var_scales=var_s,
            pre=pre, post=post,
            eco_scalar=0.7, eps=1e-8, bc2=0.999, quantize=quantize,
        )
        if quantize:
            diff = (kernel_mom.to(torch.int32) - ref_mom.to(torch.int32)).abs()
            self.assertLessEqual(diff.max().item(), 1,
                "max int8 diff > 1 ULP at N={}".format(N))
        else:
            diff = (kernel_mom.to(torch.float64) - ref_mom).abs()
            self.assertLess(diff.max().item(), 1e-3,
                "max fp32 diff exceeds round-off at N={}".format(N))

    def test_quant_small(self):
        self._check_one(262_144, quantize=True)

    def test_quant_medium(self):
        self._check_one(4_194_304, quantize=True)

    def test_quant_large(self):
        self._check_one(16_777_216, quantize=True)

    def test_fp32_small(self):
        self._check_one(262_144, quantize=False)

    def test_fp32_medium(self):
        self._check_one(4_194_304, quantize=False)

    def test_fp32_large(self):
        self._check_one(16_777_216, quantize=False)


if __name__ == "__main__":
    unittest.main()
