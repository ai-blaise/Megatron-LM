#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Monte Carlo stress test for FlashAdamW ECO scaling.

This is intentionally a toy model, not a trainer.  It simulates independent
coordinates under AdamW, an NVFP4-like block cast, and the same ECO injection
shape used by ``megatron.core.optimizer.flash_optimizers``:

    m += alpha_t * denom_t * (pre_cast - post_cast)

The point is to separate three questions:

* Does the paper ``1 / lr`` scale create extreme first-moment values?
* Does rowwise INT8 moment/variance quantization amplify those extremes?
* Does the expected first bad coordinate scale with virtual parameter count?

The script samples a fixed number of coordinates and extrapolates the per-step
bad-coordinate rate to virtual model sizes.  That makes billion-parameter
sweeps cheap enough to run on CPU.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class SweepResult:
    mode: str
    state: str
    virtual_params: int
    first_expected_bad_step: int | None
    first_p50_bad_step: int | None
    final_loss_proxy: float
    final_max_update: float
    final_p999_update: float
    final_bad_frac: float
    final_m_absmax: float
    final_v_min_nonzero: float


def _parse_int_list(value: str) -> list[int]:
    out = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        multiplier = 1
        if item.endswith("k"):
            multiplier = 1_000
            item = item[:-1]
        elif item.endswith("m"):
            multiplier = 1_000_000
            item = item[:-1]
        elif item.endswith("b"):
            multiplier = 1_000_000_000
            item = item[:-1]
        out.append(int(float(item) * multiplier))
    return out


def _lr_at_step(step: int, *, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, step / warmup_steps)


def _eco_alpha(
    mode: str,
    *,
    bc1: float,
    lr: float,
    base_lr: float,
    beta1: float,
) -> float:
    if mode == "off":
        return 0.0
    factor = 1.0 - 1.0 / beta1
    if mode == "paper":
        return (bc1 / lr) * factor if lr != 0.0 else 0.0
    if mode == "lr_floor":
        return (bc1 / max(lr, base_lr)) * factor
    if mode == "no_inv_lr":
        return bc1 * factor
    raise ValueError(f"unknown ECO mode: {mode}")


def _quantize_moment_softsign(m: torch.Tensor, group_size: int) -> torch.Tensor:
    view = m.view(-1, group_size)
    scale = view.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = (view / scale).clamp(-1.0, 1.0)
    transformed = 2.0 * normalized / (1.0 + normalized.abs())
    packed = torch.round(transformed * 127.0).clamp(-127.0, 127.0)
    unpacked = packed / 127.0
    recovered = unpacked / (2.0 - unpacked.abs()).clamp_min(1e-12)
    return (recovered * scale).reshape_as(m)


def _quantize_var_sqrt(v: torch.Tensor, group_size: int) -> torch.Tensor:
    view = v.clamp_min(0.0).sqrt().view(-1, group_size)
    scale = view.amax(dim=1, keepdim=True).clamp_min(1e-30)
    packed = torch.round((view / scale).clamp(0.0, 1.0) * 255.0).clamp(0.0, 255.0)
    recovered_sqrt = packed / 255.0 * scale
    return recovered_sqrt.square().reshape_as(v)


def _quantize_optimizer_state(
    m: torch.Tensor, v: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return _quantize_moment_softsign(m, group_size), _quantize_var_sqrt(v, group_size)


def _fake_nvfp4_block_cast(
    x: torch.Tensor,
    *,
    group_size: int,
    stochastic: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    """Approximate NVFP4 e2m1 block quantization.

    This deliberately models the error scale, not the exact TE bit encoding.
    Blocks use a shared max scale and half-step normalized levels in [-6, 6].
    """

    view = x.view(-1, group_size)
    scale = (view.abs().amax(dim=1, keepdim=True) / 6.0).clamp_min(1e-12)
    normalized = (view / scale).clamp(-6.0, 6.0)
    scaled = normalized * 2.0
    if stochastic:
        floor = torch.floor(scaled)
        prob_up = (scaled - floor).clamp(0.0, 1.0)
        rnd = torch.rand(prob_up.shape, generator=generator, device=prob_up.device)
        rounded = floor + (rnd < prob_up).to(scaled.dtype)
    else:
        rounded = torch.round(scaled)
    return (rounded.clamp(-12.0, 12.0) / 2.0 * scale).reshape_as(x)


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.detach().float(), q).item())


def _run_one(
    *,
    mode: str,
    quantized_state: bool,
    virtual_params: int,
    device: str,
    sample_elems: int,
    steps: int,
    base_lr: float,
    warmup_steps: int,
    beta1: float,
    beta2: float,
    eps: float,
    grad_noise_std: float,
    curvature: float,
    init_std: float,
    group_size: int,
    bad_update_threshold: float,
    stochastic_cast: bool,
    seed: int,
) -> SweepResult:
    if sample_elems % group_size != 0:
        raise ValueError("--sample-elems must be divisible by --group-size")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    theta = (
        torch.randn(sample_elems, generator=gen, device=device, dtype=torch.float32)
        * init_std
    )
    theta = _fake_nvfp4_block_cast(
        theta, group_size=group_size, stochastic=False, generator=gen
    )
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)

    first_expected_bad_step: int | None = None
    first_p50_bad_step: int | None = None
    final_bad_frac = 0.0
    final_update_abs = torch.zeros_like(theta)

    for step in range(1, steps + 1):
        lr = _lr_at_step(step, base_lr=base_lr, warmup_steps=warmup_steps)
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step

        grad = curvature * theta
        if grad_noise_std > 0.0:
            grad = (
                grad
                + torch.randn(sample_elems, generator=gen, device=device) * grad_noise_std
            )

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad.square()
        if quantized_state:
            m, v = _quantize_optimizer_state(m, v, group_size)

        denom = torch.sqrt(v / bc2).add_(eps)
        update = m / bc1 / denom
        final_update_abs = (lr * update).abs()
        pre_cast = theta - lr * update
        post_cast = _fake_nvfp4_block_cast(
            pre_cast, group_size=16, stochastic=stochastic_cast, generator=gen
        )
        error = pre_cast - post_cast

        alpha = _eco_alpha(mode, bc1=bc1, lr=lr, base_lr=base_lr, beta1=beta1)
        if alpha != 0.0:
            m = m + alpha * denom * error
            if quantized_state:
                m, v = _quantize_optimizer_state(m, v, group_size)

        theta = post_cast

        bad = (
            ~torch.isfinite(theta)
            | ~torch.isfinite(m)
            | ~torch.isfinite(v)
            | (final_update_abs > bad_update_threshold)
        )
        final_bad_frac = float(bad.float().mean().item())
        if final_bad_frac > 0.0:
            expected_bad = virtual_params * final_bad_frac
            p_any = 1.0 - math.exp(-min(700.0, expected_bad))
            if first_expected_bad_step is None and expected_bad >= 1.0:
                first_expected_bad_step = step
            if first_p50_bad_step is None and p_any >= 0.5:
                first_p50_bad_step = step

    nonzero_v = v[v > 0]
    final_v_min_nonzero = float(nonzero_v.min().item()) if nonzero_v.numel() else 0.0
    return SweepResult(
        mode=mode,
        state="int8-row" if quantized_state else "fp32",
        virtual_params=virtual_params,
        first_expected_bad_step=first_expected_bad_step,
        first_p50_bad_step=first_p50_bad_step,
        final_loss_proxy=float(0.5 * theta.square().mean().item()),
        final_max_update=float(final_update_abs.max().item()),
        final_p999_update=_safe_quantile(final_update_abs, 0.999),
        final_bad_frac=final_bad_frac,
        final_m_absmax=float(m.abs().max().item()),
        final_v_min_nonzero=final_v_min_nonzero,
    )


def _format_step(step: int | None) -> str:
    return "-" if step is None else str(step)


def _print_table(results: Iterable[SweepResult]) -> None:
    header = (
        "mode",
        "state",
        "virtual_N",
        "E[bad]>=1",
        "P(any)>=.5",
        "loss_proxy",
        "max_update",
        "p999_update",
        "bad_frac",
        "max|m|",
        "min_nonzero_v",
    )
    print("\t".join(header))
    for r in results:
        print(
            "\t".join(
                [
                    r.mode,
                    r.state,
                    f"{r.virtual_params:.3e}",
                    _format_step(r.first_expected_bad_step),
                    _format_step(r.first_p50_bad_step),
                    f"{r.final_loss_proxy:.3e}",
                    f"{r.final_max_update:.3e}",
                    f"{r.final_p999_update:.3e}",
                    f"{r.final_bad_frac:.3e}",
                    f"{r.final_m_absmax:.3e}",
                    f"{r.final_v_min_nonzero:.3e}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sample-elems", type=int, default=262_144)
    parser.add_argument(
        "--virtual-params",
        type=_parse_int_list,
        default=_parse_int_list("800m,8b,70b,345b"),
        help="comma list with optional k/m/b suffixes",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="off,paper,lr_floor,no_inv_lr",
        help="comma list: off,paper,lr_floor,no_inv_lr",
    )
    parser.add_argument("--base-lr", type=float, default=5e-6)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=980,
        help="31348 warmup samples / GBS32 is about 980 optimizer steps",
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-noise-std", type=float, default=1e-3)
    parser.add_argument("--curvature", type=float, default=1e-2)
    parser.add_argument("--init-std", type=float, default=2e-2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bad-update-threshold", type=float, default=1e-2)
    parser.add_argument("--deterministic-cast", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results: list[SweepResult] = []
    for quantized_state in (False, True):
        for mode in modes:
            for virtual_params in args.virtual_params:
                results.append(
                    _run_one(
                        mode=mode,
                        quantized_state=quantized_state,
                        virtual_params=virtual_params,
                        device=args.device,
                        sample_elems=args.sample_elems,
                        steps=args.steps,
                        base_lr=args.base_lr,
                        warmup_steps=args.warmup_steps,
                        beta1=args.beta1,
                        beta2=args.beta2,
                        eps=args.eps,
                        grad_noise_std=args.grad_noise_std,
                        curvature=args.curvature,
                        init_std=args.init_std,
                        group_size=args.group_size,
                        bad_update_threshold=args.bad_update_threshold,
                        stochastic_cast=not args.deterministic_cast,
                        seed=args.seed,
                    )
                )

    _print_table(results)


if __name__ == "__main__":
    main()
