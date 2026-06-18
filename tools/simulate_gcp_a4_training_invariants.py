#!/usr/bin/env python3
"""Fast invariants for the GCP A4 DeepSeek SFT training shape.

This intentionally avoids loading the model or launching NCCL. It replays the
index math and scaling factors that must be true before another expensive
multi-node run is useful.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    _chunk_size_factor_from_storage_shapes,
    build_data_parallel_buffer_index,
)


@dataclass(frozen=True)
class ShapeCase:
    name: str
    logical: tuple[int, int]


@dataclass(frozen=True)
class IterMetric:
    path: Path
    iteration: int
    elapsed_ms: float
    train_tokens: int
    train_tokens_per_s: float
    lm_loss: float
    seq_load_balancing_loss: float
    indexer_loss: float
    grad_norm: float


@dataclass(frozen=True)
class OptimizerGradMetric:
    path: Path
    rank: int
    local_rank: int
    iteration: int
    stage: str
    name: str
    numel: int
    sample_numel: int
    rms: float
    absmax: float
    abs_p50: float | None
    abs_p99: float | None
    abs_p999: float | None

    @property
    def sample_l2_estimate(self) -> float:
        # numeric_debug reports RMS over a deterministic sample, not the full
        # tensor. Multiplying by sqrt(numel) is therefore a sampled estimate of
        # the local full-shard L2, useful for ranking but not an exact norm.
        return self.rms * math.sqrt(max(self.numel, 1))


@dataclass(frozen=True)
class MoeDispatchMetric:
    path: Path
    rank: int
    valid_edges: int
    expert_edges_nonzero: int
    tokens_per_expert_sum: int
    tokens_per_expert_max: int
    tokens_per_expert_max_over_mean: float
    tokens_per_expert_top: str


FLOAT_RE = r"(?:nan|inf|-inf|[0-9]+(?:\.[0-9]*)?(?:[Ee][+-]?[0-9]+)?)"
ITER_RE = re.compile(
    rf"iteration\s+(?P<iteration>\d+)/.*?"
    rf"elapsed time per iteration \(ms\):\s*(?P<elapsed>{FLOAT_RE}).*?"
    rf"train tokens:\s*(?P<tokens>\d+).*?"
    rf"train tokens/s:\s*(?P<tps>{FLOAT_RE}).*?"
    rf"lm loss:\s*(?P<lm>{FLOAT_RE}).*?"
    rf"seq_load_balancing_loss:\s*(?P<seq>{FLOAT_RE}).*?"
    rf"indexer loss:\s*(?P<idx>{FLOAT_RE}).*?"
    rf"grad norm:\s*(?P<grad>{FLOAT_RE})",
)
OWNER_RE = re.compile(
    rf"\[grad_ownership\.owner\]\s+rank=(?P<rank>\d+)\s+"
    rf"iter=(?P<iteration>\d+).*?owner=(?P<owner>\S+)\s+"
    rf"norm=(?P<norm>{FLOAT_RE}).*?nonfinite=(?P<nonfinite>\d+)"
)
MEM_RE = re.compile(
    rf"\[Rank\s+(?P<rank>\d+)\]\s+\(after\s+(?P<iteration>\d+)\s+iterations\)\s+"
    rf"memory \(MB\).*?allocated:\s*(?P<allocated>{FLOAT_RE}).*?"
    rf"max allocated:\s*(?P<max_allocated>{FLOAT_RE}).*?"
    rf"reserved:\s*(?P<reserved>{FLOAT_RE}).*?"
    rf"max reserved:\s*(?P<max_reserved>{FLOAT_RE})"
)
LOSS_DIAG_MARKER = "[loss_diag] "
OPTIMIZER_GRAD_RE = re.compile(
    rf"\[numeric-debug\]\[rank=(?P<rank>\d+)\s+local=(?P<local>\d+)\s+"
    rf"iter=(?P<iteration>\d+)\s+phase=(?P<phase>\S+)\]\s+"
    rf"optimizer_grad\.(?P<stage>"
    rf"after_clip_grad_norm_{FLOAT_RE}|"
    rf"after_prepare_grads|before_grad_norm|after_grad_norm|"
    rf"after_count_zeros|before_inner_step"
    rf")\.(?P<name>.+?)\.grad:\s+"
    rf".*?numel=(?P<numel>\d+)\s+sample_numel=(?P<sample_numel>\d+)"
    rf".*?rms=(?P<rms>{FLOAT_RE})\s+absmax=(?P<absmax>{FLOAT_RE})"
    rf"(?:.*?abs_p50=(?P<abs_p50>{FLOAT_RE})"
    rf"\s+abs_p99=(?P<abs_p99>{FLOAT_RE})"
    rf"\s+abs_p999=(?P<abs_p999>{FLOAT_RE}))?"
)
MOE_DISPATCH_RE = re.compile(
    rf"\[moe_dispatch_stats\].*?"
    rf"rank=(?P<rank>\d+).*?"
    rf"valid_edges=(?P<valid_edges>\d+).*?"
    rf"expert_edges_nonzero=(?P<expert_edges_nonzero>\d+).*?"
    rf"tokens_per_expert_max=(?P<tokens_per_expert_max>\d+).*?"
    rf"tokens_per_expert_sum=(?P<tokens_per_expert_sum>\d+).*?"
    rf"tokens_per_expert_max_over_mean=(?P<max_over_mean>{FLOAT_RE}).*?"
    rf"tokens_per_expert_top=(?P<top>\S+)"
)


PRODUCTION_SHAPES = [
    ShapeCase("dsa.indexer.linear_wk", (128, 7168)),
    ShapeCase("dsa.indexer.linear_wq", (128, 7168)),
    ShapeCase("mla.q_down", (1536, 7168)),
    ShapeCase("mla.q_up", (24576, 1536)),
    ShapeCase("mla.kv_down", (576, 7168)),
    ShapeCase("mla.kv_up", (32768, 512)),
    ShapeCase("mla.output_proj", (7168, 16384)),
    ShapeCase("moe.expert.fc1", (4096, 7168)),
    ShapeCase("moe.expert.fc2", (7168, 2048)),
    ShapeCase("moe.shared.fc1", (4096, 7168)),
    ShapeCase("moe.shared.fc2", (7168, 2048)),
    ShapeCase("dense_mlp.fc1", (36864, 7168)),
    ShapeCase("dense_mlp.fc2", (7168, 18432)),
    ShapeCase("gated_norm.low_rank", (16, 7168)),
]


def parse_float(value: str) -> float:
    return float(value)


def check_close(
    label: str,
    actual: float,
    expected: float,
    *,
    rel_tol: float = 1.0e-5,
    abs_tol: float = 1.0e-7,
) -> int:
    ok = math.isfinite(actual) and math.isfinite(expected) and math.isclose(
        actual,
        expected,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    status = "PASS" if ok else "FAIL"
    diff = actual - expected
    print(
        f"  {status:4s} {label:44s} actual={actual:.9g} "
        f"expected={expected:.9g} diff={diff:.3e}"
    )
    return 0 if ok else 1


def fp4_storage_shape(shape: tuple[int, int]) -> torch.Size:
    rows, cols = shape
    if cols % 2:
        raise ValueError(f"NVFP4 packed storage requires even trailing dim, got {shape}")
    return torch.Size((rows, cols // 2))


def item_slice(
    storage_shape: torch.Size,
    rank: int,
    dp: int,
    chunk_size_factor: int,
) -> tuple[int, int, int, int]:
    cfg = SimpleNamespace(data_parallel_sharding_strategy="optim_grads_params")
    item_index_map, bucket_index, shard_index = build_data_parallel_buffer_index(
        [storage_shape],
        rank,
        dp,
        True,
        cfg,
        bucket_id=0,
        chunk_size_factor=chunk_size_factor,
    )
    item = item_index_map[0]
    item_start = item.global_data_index
    item_end = item_start + item.size
    shard_start = shard_index.bucket_data_index
    shard_end = shard_start + shard_index.size
    start = max(item_start, shard_start) - item_start
    end = min(item_end, shard_end) - item_start
    if start == end:
        start = 0
        end = 0
    return start, end, bucket_index.size, shard_index.size


def check_nvfp4_packed_shards(dp: int) -> int:
    failures = 0
    print(f"\n[nvfp4-shards] dp={dp}")
    for case in PRODUCTION_SHAPES:
        logical = torch.Size(case.logical)
        storage = fp4_storage_shape(case.logical)
        logical_cf = logical[1:].numel()
        raw_cf = _chunk_size_factor_from_storage_shapes([storage], logical_cf)
        old_bad = 0
        new_bad = 0
        for rank in range(dp):
            main_start, main_end, _, _ = item_slice(logical, rank, dp, logical_cf)
            expected = (main_start // 2, (main_end + 1) // 2)
            old_slice = item_slice(storage, rank, dp, logical_cf)[:2]
            new_slice = item_slice(storage, rank, dp, raw_cf)[:2]
            old_bad += int(old_slice != expected)
            new_bad += int(new_slice != expected)
        status = "ok" if new_bad == 0 else "FAIL"
        print(
            f"  {status:4s} {case.name:24s} logical={tuple(logical)!s:16s} "
            f"raw={tuple(storage)!s:15s} old_bad={old_bad}/{dp} new_bad={new_bad}/{dp} "
            f"old_cf={logical_cf} new_cf={raw_cf}"
        )
        if new_bad:
            failures += 1
    return failures


def grad_norm_accounting_bounds(
    tp: int,
    cp: int,
    pp: int,
    dp: int,
    ep: int,
    etp: int,
    grad_accum: int,
    num_dist_opt_instances: int,
    dsa_scale_mode: str,
) -> None:
    world = tp * cp * pp * dp
    mp = tp * cp * pp
    expert_dp = world // (etp * ep * pp)
    intra_dist_opt_size = world // num_dist_opt_instances
    dp_cp = dp * cp
    print("\n[grad-norm-accounting]")
    print(
        f"  world={world} mp={mp} dp={dp} cp={cp} dp_cp={dp_cp} "
        f"ep={ep} etp={etp} expert_dp={expert_dp} grad_accum={grad_accum}"
    )
    print(f"  sqrt(mp)={math.sqrt(mp):.3f} sqrt(world)={math.sqrt(world):.3f}")
    print(f"  fsdp_dense_reduce_scatter_group_size=dp*cp={dp_cp}")
    print(
        "  grad_stats_parallel_group_size=intra_dist_opt="
        f"{intra_dist_opt_size} for num_dist_opt_instances={num_dist_opt_instances}"
    )
    print(f"  missing_dp_average_factor={dp}")
    print(f"  missing_microbatch_average_factor={grad_accum}")
    if dsa_scale_mode == "token_mean":
        print(
            "  dsa_autoscale_after_fsdp_average=(1/grad_accum)/(dp*cp)="
            f"1/(dp*cp*grad_accum)=1/{dp * cp * grad_accum}"
        )
    else:
        print(
            "  dsa_autoscale_after_fsdp_average=(cp/grad_accum)/(dp*cp)="
            f"1/(dp*grad_accum)=1/{dp * grad_accum}"
        )
    print(
        "  note: jobs 217 and 218 used the same launch-side normalization, so a "
        "100x swing is not explained by topology-only norm reduction."
    )


def megatron_local_loss_scalar(
    losses: torch.Tensor,
    loss_mask: torch.Tensor,
    num_microbatches: int,
) -> tuple[float, int, float]:
    """Replay pretrain_gpt.loss_func plus schedule loss scaling for one microbatch."""

    flat_losses = losses.view(-1).float()
    flat_mask = loss_mask.view(-1).float()
    active = flat_mask > 0
    masked_losses = torch.where(active, flat_losses, torch.zeros_like(flat_losses))
    loss_sum = torch.sum(masked_losses * flat_mask)
    num_tokens = int(flat_mask.sum().item())
    scaled = loss_sum / max(num_tokens, 1) / num_microbatches
    reported_lm = loss_sum / max(num_tokens, 1)
    return float(scaled.item()), num_tokens, float(reported_lm.item())


def megatron_local_loss_tensor(
    losses: torch.Tensor,
    loss_mask: torch.Tensor,
    num_microbatches: int,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Tensor form of the Megatron SFT loss path, preserving autograd."""

    flat_losses = losses.view(-1).float()
    flat_mask = loss_mask.view(-1).float()
    active = flat_mask > 0
    masked_losses = torch.where(active, flat_losses, torch.zeros_like(flat_losses))
    loss_sum = torch.sum(masked_losses * flat_mask)
    num_tokens = int(flat_mask.sum().item())
    scaled = loss_sum / max(num_tokens, 1) / num_microbatches
    reported_lm = loss_sum.detach() / max(num_tokens, 1)
    return scaled, num_tokens, reported_lm


def simulate_loss_and_lm_grad_scale(
    seq_len: int,
    cp: int,
    grad_accum: int,
    seed: int,
) -> None:
    """Show how sparse SFT labels affect loss values and LM grad scale."""

    local_tokens = seq_len // cp
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    print("\n[loss-and-lm-grad-scale]")
    print(
        f"  local_tokens_per_microbatch={local_tokens} num_microbatches={grad_accum} "
        "loss path=sum(active losses)/active_tokens/num_microbatches"
    )
    print("  inactive NaNs are zeroed before masking, matching pretrain_gpt.py")

    rows = []
    for active_frac in (0.005, 0.0091, 0.02, 0.0367, 0.10, 1.0):
        active_tokens = max(1, int(round(local_tokens * active_frac)))
        mask = torch.zeros(local_tokens, dtype=torch.float32)
        mask[:active_tokens] = 1.0
        losses = torch.normal(mean=14.5, std=4.0, size=(local_tokens,), generator=gen)
        # Put garbage in inactive positions to verify the patched masking behavior.
        if active_tokens < local_tokens:
            losses[active_tokens : min(local_tokens, active_tokens + 128)] = float("nan")
        scaled_loss, num_tokens, reported_lm = megatron_local_loss_scalar(
            losses, mask, grad_accum
        )
        # Cross entropy d(logits) is bounded by O(1) per active token. With
        # Megatron's normalization, the per-microbatch LM-head grad norm scale is
        # proportional to sqrt(active_tokens)/(active_tokens * grad_accum).
        lm_grad_unit_norm = math.sqrt(num_tokens) / (num_tokens * grad_accum)
        rows.append((active_frac, num_tokens, reported_lm, scaled_loss, lm_grad_unit_norm))

    baseline = rows[1][-1]  # observed mask-on sample fraction from local tokenizer sample.
    for active_frac, num_tokens, reported_lm, scaled_loss, lm_grad_unit_norm in rows:
        print(
            f"  active_frac={active_frac:7.4f} active_tokens={num_tokens:4d} "
            f"reported_lm~{reported_lm:7.3f} scaled_loss~{scaled_loss:8.5f} "
            f"lm_grad_unit_norm={lm_grad_unit_norm:9.6f} "
            f"ratio_vs_0.0091={lm_grad_unit_norm / baseline:6.2f}x"
        )


def simulate_lm_loss_autograd(
    seq_len: int,
    cp: int,
    grad_accum: int,
) -> int:
    """Use actual backward calls to validate LM loss values and grad norms."""

    failures = 0
    local_tokens = seq_len // cp
    print("\n[lm-loss-autograd]")
    print(
        "  one scalar parameter per local token; inactive token losses are NaN, "
        "then zeroed by the patched pretrain_gpt.py loss path"
    )
    print(
        "  expected_grad_norm uses d(14 + p^2)/dp = 2p with p=1 on active tokens"
    )

    baseline_tokens = max(1, int(round(local_tokens * 0.0091)))
    baseline_norm = 2.0 * math.sqrt(baseline_tokens) / (baseline_tokens * grad_accum)
    for active_frac in (0.005, 0.0091, 0.02, 0.0367, 0.10, 1.0):
        active_tokens = max(1, int(round(local_tokens * active_frac)))
        mask = torch.zeros(local_tokens, dtype=torch.float32)
        mask[:active_tokens] = 1.0
        params = torch.ones(local_tokens, dtype=torch.float32, requires_grad=True)
        finite_losses = 14.0 + params.square()
        losses = torch.where(mask > 0, finite_losses, torch.full_like(finite_losses, float("nan")))
        scaled_loss, num_tokens, reported_lm = megatron_local_loss_tensor(
            losses, mask, grad_accum
        )
        scaled_loss.backward()
        grad_norm = float(params.grad.norm().item())
        expected = 2.0 * math.sqrt(num_tokens) / (num_tokens * grad_accum)
        ratio = grad_norm / baseline_norm if baseline_norm else float("nan")
        print(
            f"  active_frac={active_frac:7.4f} active_tokens={num_tokens:4d} "
            f"reported_lm={float(reported_lm):7.3f} "
            f"scaled_loss={float(scaled_loss.detach()):8.5f} "
            f"grad_norm={grad_norm:9.6f} expected={expected:9.6f} "
            f"ratio_vs_0.0091={ratio:6.2f}x"
        )
        failures += check_close(
            f"lm_autograd_grad_norm active={active_tokens}",
            grad_norm,
            expected,
        )
    return failures


def simulate_cross_entropy_logits_autograd(
    seq_len: int,
    cp: int,
    grad_accum: int,
    vocab_size: int,
) -> int:
    """Run real CE on logits to validate masked-token loss/grad magnitudes."""

    failures = 0
    local_tokens = seq_len // cp
    print("\n[cross-entropy-logits-autograd]")
    print(
        f"  logits=[{local_tokens},{vocab_size}] on CPU with uniform logits; "
        "this tests the same masked CE normalization using real CE backward"
    )
    per_token_grad_norm = math.sqrt(1.0 - 1.0 / vocab_size)
    baseline_tokens = max(1, int(round(local_tokens * 0.0091)))
    baseline_norm = per_token_grad_norm / (math.sqrt(baseline_tokens) * grad_accum)
    for active_frac in (0.005, 0.0091, 0.02, 0.0367, 0.10):
        active_tokens = max(1, int(round(local_tokens * active_frac)))
        logits = torch.zeros(local_tokens, vocab_size, dtype=torch.float32, requires_grad=True)
        labels = torch.arange(local_tokens, dtype=torch.long) % vocab_size
        per_token_loss = F.cross_entropy(logits, labels, reduction="none")
        mask = torch.zeros(local_tokens, dtype=torch.float32)
        mask[:active_tokens] = 1.0
        scaled_loss, num_tokens, reported_lm = megatron_local_loss_tensor(
            per_token_loss,
            mask,
            grad_accum,
        )
        scaled_loss.backward()
        grad_norm = float(logits.grad.norm().item())
        expected_loss = math.log(vocab_size)
        expected_grad_norm = per_token_grad_norm / (math.sqrt(num_tokens) * grad_accum)
        print(
            f"  active_frac={active_frac:7.4f} active_tokens={num_tokens:4d} "
            f"reported_lm={float(reported_lm):8.5f} expected_lm={expected_loss:8.5f} "
            f"logits_grad_norm={grad_norm:9.6f} "
            f"ratio_vs_0.0091={grad_norm / baseline_norm:6.2f}x"
        )
        failures += check_close(
            f"ce_reported_lm active={num_tokens}",
            float(reported_lm),
            expected_loss,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        )
        failures += check_close(
            f"ce_logits_grad_norm active={num_tokens}",
            grad_norm,
            expected_grad_norm,
            rel_tol=1.0e-3,
            abs_tol=1.0e-7,
        )
    return failures


def simulate_dsa_aux_loss_scale(
    seq_len: int,
    cp: int,
    grad_accum: int,
    logged_indexer_loss: float,
    scale_mode: str,
) -> None:
    """Compare the DSA autoscaled aux path to the ordinary LM loss path."""

    local_tokens = seq_len // cp
    raw_layer_loss_estimate = logged_indexer_loss * grad_accum
    if scale_mode == "token_mean":
        autoscale_backward = 1.0 / grad_accum
    elif scale_mode == "moe_cp":
        autoscale_backward = cp / grad_accum
    else:
        raise ValueError(f"unsupported DSA scale mode: {scale_mode!r}")
    legacy_backward = cp / grad_accum
    print("\n[dsa-indexer-loss-scale]")
    print(
        "  launcher uses MEGATRON_DSA_INDEXER_AUX_LOSS_AUTOSCALE=1, so DSA "
        "indexer loss is attached through DSAIndexerLossAutoScaler, not through "
        "loss_func's num-token denominator."
    )
    print(
        f"  logged_indexer_loss={logged_indexer_loss:.6e}; "
        f"raw_per_layer_estimate~=logged*grad_accum={raw_layer_loss_estimate:.6e}"
    )
    print(
        f"  DSA scale mode={scale_mode}; production backward multiplier="
        f"{autoscale_backward:.6f}; legacy_moe_cp={legacy_backward:.6f}"
    )
    for active_frac in (0.005, 0.0091, 0.02, 0.0367, 0.10):
        active_tokens = max(1, int(round(local_tokens * active_frac)))
        lm_scalar_multiplier = 1.0 / (active_tokens * grad_accum)
        print(
            f"  active_frac={active_frac:7.4f} active_tokens={active_tokens:4d} "
            f"lm_token_multiplier={lm_scalar_multiplier:.6e} "
            f"dsa_vs_lm_scalar_multiplier={autoscale_backward / lm_scalar_multiplier:8.1f}x"
        )
    print(
        "  interpretation: the tiny logged indexer scalar cannot be read like LM "
        "loss. Its gradient path is intentionally much less diluted than token CE."
    )


def simulate_dsa_autoscaler_autograd(
    seq_len: int,
    cp: int,
    grad_accum: int,
    scale_mode: str,
) -> int:
    """Compare real backward scaling for explicit vs autoscaled DSA aux loss."""

    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
    )

    failures = 0
    local_tokens = seq_len // cp
    if scale_mode == "token_mean":
        auto_scale = 1.0 / float(grad_accum)
        expected_ratio_factor = 1
    elif scale_mode == "moe_cp":
        auto_scale = float(cp) / float(grad_accum)
        expected_ratio_factor = cp
    else:
        raise ValueError(f"unsupported DSA scale mode: {scale_mode!r}")
    print("\n[dsa-autoscaler-autograd]")
    print(
        "  explicit path = indexer_loss added to pretrain_gpt.loss_func, then "
        "divided by active_tokens and num_microbatches"
    )
    print(
        "  autoscale path = DSAIndexerLossAutoScaler backward multiplier "
        f"{scale_mode}={auto_scale:.6f}"
    )

    for active_frac in (0.005, 0.0091, 0.02, 0.0367, 0.10):
        active_tokens = max(1, int(round(local_tokens * active_frac)))

        explicit_param = torch.tensor([0.25, -0.5, 1.0, -1.5], requires_grad=True)
        explicit_indexer_loss = explicit_param.square().mean()
        explicit_loss = explicit_indexer_loss / active_tokens / grad_accum
        explicit_loss.backward()
        explicit_grad_norm = float(explicit_param.grad.norm().item())

        auto_param = torch.tensor([0.25, -0.5, 1.0, -1.5], requires_grad=True)
        auto_indexer_loss = auto_param.square().mean()
        dummy_output = torch.zeros((), dtype=torch.float32, requires_grad=True)
        attached = DSAIndexerLossAutoScaler.apply(dummy_output, auto_indexer_loss)
        DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(auto_scale))
        attached.backward()
        auto_grad_norm = float(auto_param.grad.norm().item())

        ratio = auto_grad_norm / explicit_grad_norm
        expected_ratio = expected_ratio_factor * active_tokens
        print(
            f"  active_frac={active_frac:7.4f} active_tokens={active_tokens:4d} "
            f"explicit_grad_norm={explicit_grad_norm:.6e} "
            f"autoscaled_grad_norm={auto_grad_norm:.6e} "
            f"ratio={ratio:8.1f}x expected={expected_ratio:8.1f}x"
        )
        failures += check_close(
            f"dsa_autoscale_ratio active={active_tokens}",
            ratio,
            float(expected_ratio),
            rel_tol=1.0e-6,
            abs_tol=1.0e-5,
        )
    return failures


def simulate_grad_norm_ratios(
    dp: int,
    cp: int,
    grad_accum: int,
    seed: int,
    shard_elems: int,
    reference_norm: float,
    observed_norm: float,
) -> None:
    """Fast random-vector simulation of normalization-factor mistakes."""

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 17)
    raw_sum = torch.zeros(shard_elems, dtype=torch.float32)
    for _ in range(grad_accum):
        raw_sum += torch.randn(shard_elems, generator=gen)
    base_norm = float(raw_sum.norm().item())
    correct = base_norm / (grad_accum * dp)
    no_dp_avg = base_norm / grad_accum
    no_grad_accum_avg = base_norm / dp
    no_dp_no_grad_accum = base_norm
    no_dp_cp_no_grad_accum = base_norm * cp

    ratio = observed_norm / reference_norm
    print("\n[grad-norm-normalization-sim]")
    print(
        f"  simulated_shard_elems={shard_elems} reference_job217={reference_norm:.3f} "
        f"observed_job218={observed_norm:.3f} ratio={ratio:.2f}x"
    )
    print(
        f"  random raw_sum_norm={base_norm:.3f}; relative factors are independent of "
        "the random vector values."
    )
    cases = [
        ("correct: /grad_accum/dp", correct, 1.0),
        ("missing dp average", no_dp_avg, no_dp_avg / correct),
        ("missing grad_accum average", no_grad_accum_avg, no_grad_accum_avg / correct),
        ("missing dp and grad_accum average", no_dp_no_grad_accum, no_dp_no_grad_accum / correct),
        (
            "missing dp, cp, and grad_accum average",
            no_dp_cp_no_grad_accum,
            no_dp_cp_no_grad_accum / correct,
        ),
    ]
    for label, norm, factor in cases:
        print(f"  {label:39s} norm={norm:10.3f} factor={factor:8.1f}x")
    max_simple_factor = dp * cp * grad_accum
    if ratio > max_simple_factor * 1.5:
        print(
            f"  observed ratio {ratio:.1f}x is larger than the simple topology "
            f"omission envelope ({max_simple_factor}x). A missing DP/CP/"
            "microbatch average alone is not enough."
        )
    else:
        print(
            f"  observed ratio {ratio:.1f}x is within the simple topology "
            f"omission envelope (max {max_simple_factor}x)."
        )


def simulate_grad_norm_reduction_autograd(
    dp: int,
    grad_accum: int,
    seed: int,
    param_elems: int,
) -> int:
    """Autograd-backed DP/grad-accum norm simulation without launching torchrun."""

    failures = 0
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 31)
    target_grads = torch.randn(dp, grad_accum, param_elems, generator=gen)
    raw_sum = target_grads.sum(dim=(0, 1))
    correct = raw_sum / (dp * grad_accum)
    missing_dp_avg = raw_sum / grad_accum
    missing_grad_accum_avg = raw_sum / dp
    missing_both = raw_sum

    # Validate one row with real autograd: loss=sum(param * target_grad) has
    # gradient target_grad, so accumulating losses replays arbitrary grad sums.
    param = torch.zeros(param_elems, requires_grad=True)
    loss = torch.zeros((), dtype=torch.float32)
    for dp_rank in range(dp):
        for microbatch in range(grad_accum):
            loss = loss + torch.dot(param, target_grads[dp_rank, microbatch])
    (loss / (dp * grad_accum)).backward()
    autograd_norm = float(param.grad.norm().item())

    print("\n[grad-norm-reduction-autograd]")
    print(
        f"  dp={dp} grad_accum={grad_accum} param_elems={param_elems} "
        "using dot-loss gradients to replay arbitrary microbatch gradients"
    )
    print(
        f"  correct_norm={float(correct.norm().item()):.6f} "
        f"autograd_norm={autograd_norm:.6f}"
    )
    print(
        f"  missing_dp_avg_factor={float(missing_dp_avg.norm() / correct.norm()):.1f}x "
        f"missing_grad_accum_avg_factor={float(missing_grad_accum_avg.norm() / correct.norm()):.1f}x "
        f"missing_both_factor={float(missing_both.norm() / correct.norm()):.1f}x"
    )
    failures += check_close(
        "grad_reduction_autograd_norm",
        autograd_norm,
        float(correct.norm().item()),
    )
    failures += check_close(
        "grad_reduction_missing_both_factor",
        float(missing_both.norm() / correct.norm()),
        float(dp * grad_accum),
    )
    return failures


def _adam_update(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU Adam update matching FlashAdamW's no-weight-decay math."""

    exp_avg = exp_avg * beta1 + grad * (1.0 - beta1)
    exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
    corrected_avg = exp_avg / (1.0 - beta1**step)
    corrected_sq = exp_avg_sq / (1.0 - beta2**step)
    update = corrected_avg / (corrected_sq.sqrt() + eps)
    return param - lr * update, exp_avg, exp_avg_sq


def _global_clip(grad: torch.Tensor, clip_grad: float) -> tuple[torch.Tensor, float, float]:
    norm = float(grad.norm().item())
    coeff = clip_grad / (norm + 1.0e-6)
    if coeff < 1.0:
        grad = grad * coeff
    return grad, norm, min(float(coeff), 1.0)


def simulate_cold_adam_clip_math(
    elems: int,
    seed: int,
    lr: float,
    production_first_nonzero_lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    clip_grad: float,
) -> None:
    """Show that Adam can turn a globally clipped grad into an lr-sized sign step."""

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 101)
    print("\n[cold-adam-global-clip-math]")
    print(
        "  key fact: global clipping limits grad L2, but Adam divides by sqrt(v). "
        "With cold moments and tiny eps, nonzero coordinates take an update close "
        "to lr*sign(grad), independent of the clipped grad magnitude."
    )

    for active_frac in (1.0, 0.25, 0.05, 0.01, 0.001):
        active = max(1, int(round(elems * active_frac)))
        grad = torch.zeros(elems, dtype=torch.float32)
        grad[:active] = torch.randn(active, generator=gen)
        grad = grad[torch.randperm(elems, generator=gen)]
        clipped, raw_norm, clip_coeff = _global_clip(grad, clip_grad)

        param = torch.zeros(elems, dtype=torch.float32)
        exp_avg = torch.zeros_like(param)
        exp_avg_sq = torch.zeros_like(param)
        updated, _, _ = _adam_update(
            param,
            clipped,
            exp_avg,
            exp_avg_sq,
            step=1,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        adam_delta = updated - param
        sgd_delta = -lr * clipped
        nonzero = clipped != 0
        median_abs = float(adam_delta[nonzero].abs().median().item()) if active else 0.0
        p95_abs = float(adam_delta[nonzero].abs().quantile(0.95).item()) if active else 0.0
        print(
            f"  active_frac={active_frac:7.4f} active_elems={active:8d} "
            f"raw_norm={raw_norm:9.3f} clip_coeff={clip_coeff:.3e} "
            f"adam_delta_l2={float(adam_delta.norm()):9.6f} "
            f"sgd_delta_l2={float(sgd_delta.norm()):9.6f} "
            f"median_abs_delta={median_abs:.3e} p95_abs_delta={p95_abs:.3e}"
        )

    # Profile/loss-autopsy runs usually warm up over one global batch, so the
    # first logged LR can be max LR even though step 1 itself used lr=0. The
    # production full-token run warms much slower. Show both lr scales.
    print(
        f"  full-lr sign-step estimate for {elems} local elems: "
        f"lr*sqrt(elems)={lr * math.sqrt(elems):.6f}"
    )
    print(
        "  production-warmup first-nonzero-lr estimate for the same local elems: "
        f"{production_first_nonzero_lr:.3e}*sqrt(elems)="
        f"{production_first_nonzero_lr * math.sqrt(elems):.6f}"
    )


def _round_to_nvfp4_grid_cpu(x: torch.Tensor) -> torch.Tensor:
    """Round to signed NVFP4 E2M1 magnitudes used by the reference kernels."""

    ax = x.abs().clamp_max(6.0)
    idx = torch.zeros_like(ax, dtype=torch.long)
    idx = torch.where(ax > 0.25, torch.ones_like(idx), idx)
    idx = torch.where(ax >= 0.75, torch.full_like(idx, 2), idx)
    idx = torch.where(ax > 1.25, torch.full_like(idx, 3), idx)
    idx = torch.where(ax >= 1.75, torch.full_like(idx, 4), idx)
    idx = torch.where(ax > 2.5, torch.full_like(idx, 5), idx)
    idx = torch.where(ax >= 3.5, torch.full_like(idx, 6), idx)
    idx = torch.where(ax > 5.0, torch.full_like(idx, 7), idx)
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=x.dtype,
        device=x.device,
    )
    values = lut[idx]
    return torch.where(x < 0, -values, values)


def _nvfp4_fake_quant_cpu(x: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Approximate rowwise NVFP4 E2M1/UE8M0 quant-dequant on CPU."""

    if x.numel() == 0:
        return x.clone()
    flat = x.flatten().to(torch.float32)
    pad = (-flat.numel()) % group_size
    if pad:
        flat_padded = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    else:
        flat_padded = flat
    groups = flat_padded.view(-1, group_size)
    absmax = groups.abs().amax(dim=-1).clamp_min(1.0e-12)
    # Matches the reference IndexCache description: UE8M0 power-of-two scale
    # for ceil(max(abs(x)) / 6).
    scale = torch.pow(2.0, torch.ceil(torch.log2(absmax / 6.0))).clamp_min(2.0**-126)
    q = _round_to_nvfp4_grid_cpu((groups / scale[:, None]).clamp(-6.0, 6.0))
    out = (q * scale[:, None]).reshape(-1)[: flat.numel()]
    return out.view_as(x)


def _quantize_momentum_cpu(
    x: torch.Tensor, group_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = x.flatten().to(torch.float32)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    groups = flat.view(-1, group_size)
    absmax = groups.abs().amax(dim=-1).clamp_min(1.0e-12)
    normalized = groups / absmax[:, None]
    transformed = 2.0 * normalized / (1.0 + normalized.abs())
    q = torch.round(transformed * 127.0).clamp(-127, 127).to(torch.int8)
    return q, absmax


def _dequantize_momentum_cpu(
    q: torch.Tensor, scale: torch.Tensor, original_numel: int
) -> torch.Tensor:
    transformed = q.to(torch.float32) / 127.0
    normalized = transformed / (2.0 - transformed.abs()).clamp_min(1.0e-12)
    out = normalized * scale[:, None]
    return out.reshape(-1)[:original_numel]


def _quantize_variance_cpu(
    x: torch.Tensor, group_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = x.flatten().to(torch.float32).clamp_min(0.0).sqrt()
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    groups = flat.view(-1, group_size)
    absmax = groups.abs().amax(dim=-1).clamp_min(1.0e-12)
    q = torch.round(groups / absmax[:, None] * 255.0).clamp(0, 255).to(torch.uint8)
    return q, absmax


def _dequantize_variance_cpu(
    q: torch.Tensor, scale: torch.Tensor, original_numel: int
) -> torch.Tensor:
    sqrt_v = q.to(torch.float32) / 255.0 * scale[:, None]
    return sqrt_v.reshape(-1)[:original_numel].square()


def simulate_flashadamw_state_quant_error(
    elems: int,
    seed: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    clip_grad: float,
    steps: int,
) -> None:
    """Compare FP32 Adam states to FlashAdamW's grouped INT8 state codec."""

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 151)
    param_ref = torch.zeros(elems, dtype=torch.float32)
    param_q = torch.zeros_like(param_ref)
    m_ref = torch.zeros_like(param_ref)
    v_ref = torch.zeros_like(param_ref)
    m_q = torch.zeros_like(param_ref)
    v_q = torch.zeros_like(param_ref)

    print("\n[flashadamw-quantized-state-sim]")
    print(
        "  CPU replay of Adam with grouped int8 exp_avg and uint8 sqrt(exp_avg_sq) "
        "quantization after each step."
    )
    for step in range(1, steps + 1):
        grad = torch.randn(elems, generator=gen)
        grad, raw_norm, clip_coeff = _global_clip(grad, clip_grad)

        param_ref, m_ref, v_ref = _adam_update(
            param_ref,
            grad,
            m_ref,
            v_ref,
            step=step,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )

        param_q, m_q, v_q = _adam_update(
            param_q,
            grad,
            m_q,
            v_q,
            step=step,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        mq, ms = _quantize_momentum_cpu(m_q)
        vq, vs = _quantize_variance_cpu(v_q)
        m_q = _dequantize_momentum_cpu(mq, ms, elems)
        v_q = _dequantize_variance_cpu(vq, vs, elems)

        delta = param_q - param_ref
        denom = max(float(param_ref.norm().item()), 1.0e-12)
        print(
            f"  step={step:2d} raw_norm={raw_norm:9.3f} clip={clip_coeff:.3e} "
            f"param_ref_l2={float(param_ref.norm()):9.6f} "
            f"state_quant_delta_l2={float(delta.norm()):9.6f} "
            f"rel={float(delta.norm()) / denom:.3e}"
        )


def simulate_nvfp4_forward_visibility(
    elems: int,
    seed: int,
    lr: float,
    production_first_nonzero_lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    clip_grad: float,
    steps: int,
) -> None:
    """Estimate when FP32-master Adam updates become visible after NVFP4 cast-back."""

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 211)
    master = torch.randn(elems, generator=gen, dtype=torch.float32) * 0.02
    visible = _nvfp4_fake_quant_cpu(master)
    exp_avg = torch.zeros_like(master)
    exp_avg_sq = torch.zeros_like(master)

    print("\n[nvfp4-forward-visible-update-sim]")
    print(
        "  Approximate CPU NVFP4 cast-back. FP32 master weights persist, but the "
        "next forward only sees the quantized/dequantized model shard."
    )
    for step in range(1, steps + 1):
        grad = torch.randn(elems, generator=gen)
        grad, _, _ = _global_clip(grad, clip_grad)
        # Show production warmup scale on the first nonzero production update;
        # use full lr for the profile/loss-autopsy style path.
        step_lr = production_first_nonzero_lr if step == 1 else lr
        master, exp_avg, exp_avg_sq = _adam_update(
            master,
            grad,
            exp_avg,
            exp_avg_sq,
            step=step,
            lr=step_lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        new_visible = _nvfp4_fake_quant_cpu(master)
        changed = new_visible != visible
        visible_delta = new_visible - visible
        master_residual = master - new_visible
        print(
            f"  step={step:2d} lr={step_lr:.3e} "
            f"visible_changed={float(changed.float().mean()):7.4%} "
            f"visible_delta_l2={float(visible_delta.norm()):9.6f} "
            f"master_residual_rms={float(master_residual.square().mean().sqrt()):.3e}"
        )
        visible = new_visible


def _parse_optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _adam_first_step_factor(abs_grad: float, eps: float) -> float:
    """Magnitude fraction of a cold first Adam step relative to lr."""

    value = abs(float(abs_grad))
    return value / (value + eps) if value > 0.0 else 0.0


def _owner_for_optimizer_grad_name(name: str) -> str:
    if "indexer" in name or "indexcache" in name or "hisa" in name:
        return "dsa_hisa_indexer"
    if ".experts.linear_fc1." in name:
        return "moe_routed_expert_fc1"
    if ".experts.linear_fc2." in name:
        return "moe_routed_expert_fc2"
    if ".shared_experts." in name:
        return "moe_shared_expert"
    if ".router" in name or "router." in name:
        return "moe_router"
    if "self_attention" in name:
        return "attention"
    if "gated_norm" in name or "norm" in name:
        return "norm"
    if ".mlp." in name or "linear_fc" in name:
        return "dense_mlp"
    return "other"


def _read_optimizer_grad_metrics(log_roots: list[Path]) -> list[OptimizerGradMetric]:
    metrics: list[OptimizerGradMetric] = []
    for root in log_roots:
        for path in iter_log_files(root):
            try:
                handle = path.open("r", encoding="utf-8", errors="replace")
            except OSError:
                continue
            with handle:
                for line in handle:
                    match = OPTIMIZER_GRAD_RE.search(line)
                    if not match:
                        continue
                    metrics.append(
                        OptimizerGradMetric(
                            path=path,
                            rank=int(match.group("rank")),
                            local_rank=int(match.group("local")),
                            iteration=int(match.group("iteration")),
                            stage=match.group("stage"),
                            name=match.group("name"),
                            numel=int(match.group("numel")),
                            sample_numel=int(match.group("sample_numel")),
                            rms=parse_float(match.group("rms")),
                            absmax=parse_float(match.group("absmax")),
                            abs_p50=_parse_optional_float(match.group("abs_p50")),
                            abs_p99=_parse_optional_float(match.group("abs_p99")),
                            abs_p999=_parse_optional_float(match.group("abs_p999")),
                        )
                    )
    return metrics


def _read_moe_dispatch_metrics(log_roots: list[Path]) -> tuple[list[MoeDispatchMetric], int]:
    metrics: list[MoeDispatchMetric] = []
    router_bias_sign_logs = 0
    for root in log_roots:
        for path in iter_log_files(root):
            try:
                handle = path.open("r", encoding="utf-8", errors="replace")
            except OSError:
                continue
            with handle:
                for line in handle:
                    if "router_bias.sign." in line:
                        router_bias_sign_logs += 1
                    match = MOE_DISPATCH_RE.search(line)
                    if not match:
                        continue
                    metrics.append(
                        MoeDispatchMetric(
                            path=path,
                            rank=int(match.group("rank")),
                            valid_edges=int(match.group("valid_edges")),
                            expert_edges_nonzero=int(match.group("expert_edges_nonzero")),
                            tokens_per_expert_sum=int(match.group("tokens_per_expert_sum")),
                            tokens_per_expert_max=int(match.group("tokens_per_expert_max")),
                            tokens_per_expert_max_over_mean=parse_float(
                                match.group("max_over_mean")
                            ),
                            tokens_per_expert_top=match.group("top"),
                        )
                    )
    return metrics, router_bias_sign_logs


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * q)))
    return sorted_values[idx]


def scan_moe_routing_skew(log_roots: list[Path]) -> None:
    """Parse MoE dispatch logs for expert overload and bias-update evidence."""

    metrics, router_bias_sign_logs = _read_moe_dispatch_metrics(log_roots)
    print("\n[moe-routing-skew]")
    print(f"  router_bias.sign log lines={router_bias_sign_logs}")
    if not metrics:
        print("  no moe_dispatch_stats logs found in selected roots")
        return

    ratios = sorted(m.tokens_per_expert_max_over_mean for m in metrics)
    nonzero = sorted(float(m.expert_edges_nonzero) for m in metrics)
    n = len(metrics)
    ge4 = sum(1 for v in ratios if v >= 4.0)
    ge8 = sum(1 for v in ratios if v >= 8.0)
    ge12 = sum(1 for v in ratios if v >= 12.0)
    print(
        f"  dispatch_events={n} max_over_mean "
        f"min={ratios[0]:.3f} p50={_percentile(ratios, 0.50):.3f} "
        f"p90={_percentile(ratios, 0.90):.3f} p99={_percentile(ratios, 0.99):.3f} "
        f"mean={sum(ratios) / n:.3f} max={ratios[-1]:.3f}"
    )
    print(
        f"  overload_counts ge4={ge4}/{n} ge8={ge8}/{n} ge12={ge12}/{n}"
    )
    print(
        "  local_experts_with_edges "
        f"min={int(nonzero[0])} p50={_percentile(nonzero, 0.50):.0f} "
        f"p10={_percentile(nonzero, 0.10):.0f} max={int(nonzero[-1])}"
    )

    print("  most skewed dispatch events:")
    for metric in sorted(
        metrics,
        key=lambda m: (m.tokens_per_expert_max_over_mean, m.tokens_per_expert_max),
        reverse=True,
    )[:8]:
        rel = metric.path
        try:
            rel = metric.path.relative_to(Path("/home/sjpat/logs"))
        except ValueError:
            pass
        print(
            f"    ratio={metric.tokens_per_expert_max_over_mean:6.3f} "
            f"rank={metric.rank:3d} nonzero={metric.expert_edges_nonzero:2d} "
            f"max={metric.tokens_per_expert_max:5d} sum={metric.tokens_per_expert_sum:5d} "
            f"top={metric.tokens_per_expert_top} path={rel}"
        )


def scan_optimizer_update_grads(
    log_roots: list[Path],
    *,
    lr: float,
    production_first_nonzero_lr: float,
    adam_eps: float,
) -> None:
    """Parse optimizer grad debug logs into update-path estimates."""

    metrics = _read_optimizer_grad_metrics(log_roots)
    print("\n[optimizer-grad-update-path]")
    if not metrics:
        print("  no optimizer_grad tensor logs found in selected roots")
        return

    after_prepare = [m for m in metrics if m.stage == "after_prepare_grads"]
    after_clip = [m for m in metrics if m.stage.startswith("after_clip_grad_norm_")]
    before_inner = [m for m in metrics if m.stage == "before_inner_step"]

    print(
        f"  parsed optimizer grad tensors: total={len(metrics)} "
        f"after_prepare={len(after_prepare)} after_clip={len(after_clip)} "
        f"before_inner={len(before_inner)}"
    )
    print(
        "  note: L2 estimates multiply sampled RMS by sqrt(full numel); use for "
        "ranking, not as exact full norms."
    )

    top_prepare = sorted(after_prepare, key=lambda m: m.sample_l2_estimate, reverse=True)[:12]
    if top_prepare:
        print("  top after_prepare_grads sampled local L2 estimates:")
        for metric in top_prepare:
            owner = _owner_for_optimizer_grad_name(metric.name)
            print(
                f"    l2_est={metric.sample_l2_estimate:9.3f} "
                f"rms={metric.rms:.3e} absmax={metric.absmax:.3e} "
                f"rank={metric.rank:3d} owner={owner:22s} name={metric.name}"
            )

    by_owner_sq: dict[str, float] = {}
    by_owner_count: dict[str, int] = {}
    for metric in after_prepare:
        owner = _owner_for_optimizer_grad_name(metric.name)
        by_owner_sq[owner] = by_owner_sq.get(owner, 0.0) + metric.sample_l2_estimate**2
        by_owner_count[owner] = by_owner_count.get(owner, 0) + 1
    if by_owner_sq:
        total_sq = max(sum(by_owner_sq.values()), 1.0e-30)
        print("  sampled after_prepare owner share:")
        for owner, sq in sorted(by_owner_sq.items(), key=lambda kv: kv[1], reverse=True):
            print(
                f"    owner={owner:22s} l2_est={math.sqrt(sq):9.3f} "
                f"pct_sq={100.0 * sq / total_sq:7.3f}% tensors={by_owner_count[owner]}"
            )

    top_clip = sorted(after_clip, key=lambda m: m.sample_l2_estimate, reverse=True)[:12]
    if top_clip:
        print("  top after_clip grads and cold-Adam first-step factors:")
        for metric in top_clip:
            p50 = metric.abs_p50 if metric.abs_p50 is not None else metric.rms
            p99 = metric.abs_p99 if metric.abs_p99 is not None else metric.absmax
            p999 = metric.abs_p999 if metric.abs_p999 is not None else metric.absmax
            factor_p50 = _adam_first_step_factor(p50, adam_eps)
            factor_p99 = _adam_first_step_factor(p99, adam_eps)
            factor_p999 = _adam_first_step_factor(p999, adam_eps)
            factor_absmax = _adam_first_step_factor(metric.absmax, adam_eps)
            owner = _owner_for_optimizer_grad_name(metric.name)
            print(
                f"    rank={metric.rank:3d} owner={owner:22s} "
                f"rms={metric.rms:.3e} abs_p50={p50:.3e} abs_p99={p99:.3e} "
                f"absmax={metric.absmax:.3e} "
                f"adam_factor[p50,p99,p999,max]="
                f"{factor_p50:.3f},{factor_p99:.3f},{factor_p999:.3f},{factor_absmax:.3f} "
                f"full_lr_delta[p50,p99]={lr * factor_p50:.3e},{lr * factor_p99:.3e} "
                f"prod_lr_delta[p99]={production_first_nonzero_lr * factor_p99:.3e} "
                f"name={metric.name}"
            )

    if before_inner and after_clip:
        # Verify the clipped values were what the optimizer was about to consume.
        clip_by_key = {
            (m.rank, m.iteration, m.name): m for m in after_clip
        }
        ratios = []
        for metric in before_inner:
            clipped = clip_by_key.get((metric.rank, metric.iteration, metric.name))
            if clipped is None:
                continue
            denom = max(clipped.rms, 1.0e-30)
            ratios.append(metric.rms / denom)
        if ratios:
            ratios.sort()
            print(
                "  before_inner_step / after_clip RMS ratio: "
                f"p50={ratios[len(ratios)//2]:.6f} "
                f"min={ratios[0]:.6f} max={ratios[-1]:.6f} n={len(ratios)}"
            )


def default_log_roots() -> list[Path]:
    return [
        Path("/home/sjpat/logs/collected-204"),
        Path("/home/sjpat/logs/torchrun/corsaire-1-dsa-kgrad-boundary-tpgroup-gbs96-217"),
        Path("/home/sjpat/logs/torchrun/corsaire-1-fsdp-nvfp4-shardfix-gbs96-218"),
        Path("/home/sjpat/logs/torchrun/gcp-a4-loss-update-delta-32k-244"),
        Path("/home/sjpat/logs/torchrun/gcp-a4-loss-autopsy-32k-update-delta-246"),
        Path("/home/sjpat/logs/torchrun/gcp-a4-loss-autopsy-32k-update-delta-fix-247"),
    ]


def iter_log_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.log"))


def scan_training_logs(log_roots: list[Path]) -> None:
    """Parse prior run logs into numeric facts used by the loss/grad audit."""

    print("\n[observed-log-numbers]")
    for root in log_roots:
        files = iter_log_files(root)
        if not files:
            print(f"  missing {root}")
            continue

        iterations: dict[tuple[int, int], IterMetric] = {}
        owner_nonfinite_max: dict[tuple[int, str], int] = {}
        owner_norms: dict[tuple[int, str], float] = {}
        mem_by_iter: dict[int, list[tuple[float, float, float, float]]] = {}
        loss_diag_count = 0
        loss_diag_zero_token = 0
        loss_diag_max_loss = float("-inf")
        loss_diag_max_mean = float("-inf")
        loss_diag_max_tokens = 0
        loss_diag_nonfinite = 0

        for path in files:
            try:
                handle = path.open("r", encoding="utf-8", errors="replace")
            except OSError:
                continue
            with handle:
                for line in handle:
                    iter_match = ITER_RE.search(line)
                    if iter_match:
                        metric = IterMetric(
                            path=path,
                            iteration=int(iter_match.group("iteration")),
                            elapsed_ms=parse_float(iter_match.group("elapsed")),
                            train_tokens=int(iter_match.group("tokens")),
                            train_tokens_per_s=parse_float(iter_match.group("tps")),
                            lm_loss=parse_float(iter_match.group("lm")),
                            seq_load_balancing_loss=parse_float(iter_match.group("seq")),
                            indexer_loss=parse_float(iter_match.group("idx")),
                            grad_norm=parse_float(iter_match.group("grad")),
                        )
                        # Deduplicate across mirrored rank logs when present.
                        rank_hint = int(path.parent.name) if path.parent.name.isdigit() else -1
                        iterations[(metric.iteration, rank_hint)] = metric

                    owner_match = OWNER_RE.search(line)
                    if owner_match:
                        key = (
                            int(owner_match.group("iteration")),
                            owner_match.group("owner"),
                        )
                        owner_nonfinite_max[key] = max(
                            owner_nonfinite_max.get(key, 0),
                            int(owner_match.group("nonfinite")),
                        )
                        owner_norms[key] = parse_float(owner_match.group("norm"))

                    mem_match = MEM_RE.search(line)
                    if mem_match:
                        iteration = int(mem_match.group("iteration"))
                        mem_by_iter.setdefault(iteration, []).append(
                            (
                                parse_float(mem_match.group("allocated")),
                                parse_float(mem_match.group("max_allocated")),
                                parse_float(mem_match.group("reserved")),
                                parse_float(mem_match.group("max_reserved")),
                            )
                        )

                    if LOSS_DIAG_MARKER in line:
                        try:
                            payload = json.loads(line.split(LOSS_DIAG_MARKER, 1)[1])
                        except (IndexError, json.JSONDecodeError):
                            continue
                        loss_diag_count += 1
                        num_tokens = int(payload.get("num_tokens", 0))
                        loss_diag_zero_token += int(num_tokens == 0)
                        loss_diag_max_tokens = max(loss_diag_max_tokens, num_tokens)
                        loss_diag_nonfinite += int(payload.get("nonfinite_count", 0))
                        if "masked_max" in payload:
                            loss_diag_max_loss = max(
                                loss_diag_max_loss,
                                float(payload["masked_max"]),
                            )
                        if "masked_mean" in payload:
                            loss_diag_max_mean = max(
                                loss_diag_max_mean,
                                float(payload["masked_mean"]),
                            )

        print(f"  root={root}")
        if iterations:
            by_iter: dict[int, IterMetric] = {}
            for (_, _rank_hint), metric in iterations.items():
                by_iter.setdefault(metric.iteration, metric)
            for iteration in sorted(by_iter):
                metric = by_iter[iteration]
                grad = "nan" if math.isnan(metric.grad_norm) else f"{metric.grad_norm:.3f}"
                print(
                    f"    iter={iteration} elapsed_s={metric.elapsed_ms / 1000.0:.1f} "
                    f"tokens/s={metric.train_tokens_per_s:.1f} lm={metric.lm_loss:.5f} "
                    f"seq_lb={metric.seq_load_balancing_loss:.5f} "
                    f"indexer={metric.indexer_loss:.6e} grad={grad}"
                )
            if 1 in by_iter and 2 in by_iter:
                first = by_iter[1]
                second = by_iter[2]
                print(
                    f"    iter2/iter1 lm_loss_ratio={second.lm_loss / first.lm_loss:.3f} "
                    f"tokens_per_s_ratio={second.train_tokens_per_s / first.train_tokens_per_s:.3f}"
                )
        else:
            print("    no standard Megatron iteration metrics found")

        if mem_by_iter:
            for iteration in sorted(mem_by_iter):
                vals = mem_by_iter[iteration]
                max_alloc = max(v[1] for v in vals)
                max_reserved = max(v[3] for v in vals)
                avg_alloc = sum(v[0] for v in vals) / len(vals)
                print(
                    f"    mem iter={iteration} ranks={len(vals)} avg_alloc={avg_alloc / 1024.0:.1f}GiB "
                    f"max_alloc={max_alloc / 1024.0:.1f}GiB "
                    f"max_reserved={max_reserved / 1024.0:.1f}GiB"
                )

        nonfinite = [
            (iteration, owner, count, owner_norms.get((iteration, owner), float("nan")))
            for (iteration, owner), count in owner_nonfinite_max.items()
            if count > 0
        ]
        if nonfinite:
            for iteration, owner, count, norm in sorted(nonfinite):
                norm_str = "nan" if math.isnan(norm) else ("inf" if math.isinf(norm) else f"{norm:.3e}")
                print(
                    f"    nonfinite owner iter={iteration} owner={owner} "
                    f"max_nonfinite={count} norm={norm_str}"
                )
        elif owner_nonfinite_max:
            print("    grad ownership logged no nonfinite owner counts")

        if loss_diag_count:
            max_loss = "n/a" if loss_diag_max_loss == float("-inf") else f"{loss_diag_max_loss:.2f}"
            max_mean = "n/a" if loss_diag_max_mean == float("-inf") else f"{loss_diag_max_mean:.2f}"
            print(
                f"    loss_diag count={loss_diag_count} zero_token={loss_diag_zero_token} "
                f"max_tokens={loss_diag_max_tokens} max_masked_mean={max_mean} "
                f"max_masked_loss={max_loss} nonfinite_loss_count={loss_diag_nonfinite}"
            )


def run(args: argparse.Namespace) -> int:
    failures = 0
    failures += check_nvfp4_packed_shards(args.dp)
    grad_norm_accounting_bounds(
        args.tp,
        args.cp,
        args.pp,
        args.dp,
        args.ep,
        args.etp,
        args.grad_accum,
        args.num_dist_opt_instances,
        args.dsa_scale_mode,
    )
    simulate_loss_and_lm_grad_scale(args.seq_len, args.cp, args.grad_accum, args.seed)
    failures += simulate_lm_loss_autograd(args.seq_len, args.cp, args.grad_accum)
    failures += simulate_cross_entropy_logits_autograd(
        args.seq_len,
        args.cp,
        args.grad_accum,
        args.ce_vocab_size,
    )
    simulate_dsa_aux_loss_scale(
        args.seq_len,
        args.cp,
        args.grad_accum,
        args.logged_indexer_loss,
        args.dsa_scale_mode,
    )
    failures += simulate_dsa_autoscaler_autograd(
        args.seq_len,
        args.cp,
        args.grad_accum,
        args.dsa_scale_mode,
    )
    simulate_grad_norm_ratios(
        args.dp,
        args.cp,
        args.grad_accum,
        args.seed,
        args.grad_shard_elems,
        args.reference_grad_norm,
        args.observed_grad_norm,
    )
    failures += simulate_grad_norm_reduction_autograd(
        args.dp,
        args.grad_accum,
        args.seed,
        args.grad_sim_elems,
    )
    simulate_cold_adam_clip_math(
        args.optimizer_sim_elems,
        args.seed,
        args.lr,
        args.production_first_nonzero_lr,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_eps,
        args.clip_grad,
    )
    simulate_flashadamw_state_quant_error(
        args.optimizer_sim_elems,
        args.seed,
        args.lr,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_eps,
        args.clip_grad,
        args.optimizer_sim_steps,
    )
    simulate_nvfp4_forward_visibility(
        args.optimizer_sim_elems,
        args.seed,
        args.lr,
        args.production_first_nonzero_lr,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_eps,
        args.clip_grad,
        args.optimizer_sim_steps,
    )
    if not args.no_log_scan:
        log_roots = [Path(p) for p in args.log_root] if args.log_root else default_log_roots()
        scan_training_logs(log_roots)
        scan_optimizer_update_grads(
            log_roots,
            lr=args.lr,
            production_first_nonzero_lr=args.production_first_nonzero_lr,
            adam_eps=args.adam_eps,
        )
        scan_moe_routing_skew(log_roots)
    if failures:
        print(f"\n[result] FAIL: {failures} invariant groups failed")
        return 1
    print("\n[result] PASS: simulated invariants hold for the patched path")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--cp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=3)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--num-dist-opt-instances", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--grad-shard-elems", type=int, default=917504)
    parser.add_argument("--grad-sim-elems", type=int, default=32768)
    parser.add_argument("--ce-vocab-size", type=int, default=1024)
    parser.add_argument("--reference-grad-norm", type=float, default=119.83364868164062)
    parser.add_argument("--observed-grad-norm", type=float, default=10032.130859375)
    parser.add_argument("--logged-indexer-loss", type=float, default=8.792946e-4)
    parser.add_argument("--dsa-scale-mode", choices=("token_mean", "moe_cp"), default="token_mean")
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--production-first-nonzero-lr", type=float, default=1.272265e-7)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--optimizer-sim-elems", type=int, default=200_000)
    parser.add_argument("--optimizer-sim-steps", type=int, default=5)
    parser.add_argument("--log-root", action="append", default=[])
    parser.add_argument("--no-log-scan", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
