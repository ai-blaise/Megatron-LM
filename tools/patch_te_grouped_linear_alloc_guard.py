#!/usr/bin/env python3
"""Patch Transformer Engine GroupedLinear with narrow allocation guards.

The A4 HISA512 diagnostic OOMed inside TE grouped-linear forward after input and
weight quantization, immediately before allocating the grouped output tensor.
Megatron-side MoE cache trims run before those TE-internal allocations, which is
too early for this failure mode. Later profiles also OOMed in TE's grouped
``split_quantize`` input path. This patch adds opt-in guards at both exact
allocation frontiers and optional low-rate shape logging for the MoE fc2 fusion
audit.

The A4 32k FSDP/EP profile also showed third-forward OOMs in MoE ``fc2`` even
after the HISA scratch fix. TE's default grouped-linear autograd save set stores
both quantized weights and the original raw weights. In the normal fp8/fp4 path
the raw weights are only read by the high-precision backward override, so this
patch makes that save opt-out by default to avoid pinning FSDP-gathered expert
weights through the whole forward.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "# Megatron grouped-linear allocation guard"
OLD_MARKER = "# Megatron grouped-linear pre-output allocation guard"


HELPER_SOURCE = f'''
{MARKER}
_MEGATRON_GROUPED_LINEAR_TRUE_VALUES = ("1", "true", "yes", "on")


def _megatron_grouped_linear_env_flag(name, default="0"):
    return os.getenv(name, default).lower() in _MEGATRON_GROUPED_LINEAR_TRUE_VALUES


def _megatron_grouped_linear_rank_selected(rank):
    ranks = os.getenv("MEGATRON_TE_GROUPED_LINEAR_ALLOC_AUDIT_RANKS", "")
    ranks = ranks.strip()
    if not ranks:
        return True
    rank = str(rank)
    return rank in {{item.strip() for item in ranks.split(",") if item.strip()}}


def _megatron_grouped_linear_alloc_guard(
    stage,
    m_splits,
    feature_size,
    activation_dtype,
    device,
    trim_env,
    free_env,
    reserve_env,
    cached_env,
    sync_env,
    default_free_mb,
    default_reserve_mb,
    default_cached_mb,
):
    trim_enabled = _megatron_grouped_linear_env_flag(trim_env)
    rank = os.getenv("RANK", "?")
    audit_enabled = _megatron_grouped_linear_env_flag(
        "MEGATRON_TE_GROUPED_LINEAR_ALLOC_AUDIT_LOG"
    ) and _megatron_grouped_linear_rank_selected(rank)
    if not trim_enabled and not audit_enabled:
        return
    if not torch.cuda.is_available():
        return
    try:
        mib = 1024 * 1024
        split_values = [int(split) for split in m_splits]
        total_rows = sum(split_values)
        allocation_bytes = (
            total_rows
            * int(feature_size)
            * torch.empty((), dtype=activation_dtype, device="meta").element_size()
        )
        free_mb = int(os.getenv(free_env, str(default_free_mb)))
        reserve_mb = int(os.getenv(reserve_env, str(default_reserve_mb)))
        cached_mb = int(os.getenv(cached_env, str(default_cached_mb)))
        free_threshold = max(free_mb * mib, allocation_bytes + reserve_mb * mib)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        cached = max(0, reserved - allocated)
        should_trim = trim_enabled and free_bytes < free_threshold and cached >= cached_mb * mib
        old_free = free_bytes
        critical_sync_ran = False
        if should_trim:
            if _megatron_grouped_linear_env_flag(sync_env):
                torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            cached = max(0, reserved - allocated)
        critical_sync = (
            trim_enabled
            and _megatron_grouped_linear_env_flag(f"{{trim_env}}_CRITICAL_SYNC")
            and not _megatron_grouped_linear_env_flag(sync_env)
            and free_bytes < allocation_bytes + reserve_mb * mib
            and cached > 0
        )
        if critical_sync:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            cached = max(0, reserved - allocated)
            critical_sync_ran = True
        legacy_log = _megatron_grouped_linear_env_flag(
            "MEGATRON_TE_GROUPED_LINEAR_PRE_OUTPUT_TRIM_LOG"
        ) and _megatron_grouped_linear_rank_selected(rank)
        if audit_enabled or legacy_log:
            local_rank = os.getenv("LOCAL_RANK", "?")
            top_splits = sorted(split_values, reverse=True)[:8]
            print(
                "[megatron-te-grouped-linear-alloc] "
                f"rank={{rank}} local_rank={{local_rank}} "
                f"stage={{stage}} rows={{total_rows}} feature={{int(feature_size)}} "
                f"dtype={{str(activation_dtype)}} alloc_mib={{allocation_bytes / mib:.1f}} "
                f"free_mib={{old_free / mib:.1f}} -> {{free_bytes / mib:.1f}} "
                f"allocated_mib={{allocated / mib:.1f}} cached_mib={{cached / mib:.1f}} "
                f"threshold_mib={{free_threshold / mib:.1f}} trimmed={{int(should_trim)}} "
                f"critical_sync={{int(critical_sync_ran)}} "
                f"top_splits={{top_splits}}",
                flush=True,
            )
    except Exception as exc:  # pragma: no cover - defensive runtime guard.
        if _megatron_grouped_linear_env_flag("MEGATRON_TE_GROUPED_LINEAR_ALLOC_AUDIT_LOG"):
            print(f"[megatron-te-alloc-guard] skipped after error: {{exc}}", flush=True)

'''


def _patch_grouped_linear() -> None:
    spec = importlib.util.find_spec("transformer_engine.pytorch.module.grouped_linear")
    if spec is None or spec.origin is None:
        raise SystemExit("Could not locate transformer_engine.pytorch.module.grouped_linear")

    path = Path(spec.origin)
    text = path.read_text()
    original_text = text

    if "import os\n" not in text:
        text = text.replace("import functools\n", "import functools\nimport os\n", 1)

    helper_start = -1
    for marker in (MARKER, OLD_MARKER):
        marker_pos = text.find(marker)
        if marker_pos >= 0:
            helper_start = marker_pos
            break
    if helper_start >= 0:
        class_pos = text.find("\nclass _GroupedLinear", helper_start)
        if class_pos < 0:
            raise SystemExit(f"Could not find helper replacement end in {path}")
        text = text[:helper_start] + HELPER_SOURCE + text[class_pos + 1 :]
    elif HELPER_SOURCE.strip() not in text:
        insert_after = '__all__ = ["GroupedLinear"]\n'
        if insert_after not in text:
            raise SystemExit(f"Could not find helper insertion site in {path}")
        text = text.replace(insert_after, insert_after + HELPER_SOURCE, 1)

    input_call = (
        f"            {MARKER}: input quantization frontier\n"
        "            _megatron_grouped_linear_alloc_guard(\n"
        '                "input_quantize",\n'
        "                m_splits,\n"
        "                in_features,\n"
        "                activation_dtype,\n"
        "                device,\n"
        '                "MEGATRON_TE_GROUPED_LINEAR_PRE_INPUT_TRIM",\n'
        '                "MEGATRON_TE_GROUPED_LINEAR_PRE_INPUT_TRIM_FREE_MB",\n'
        '                "MEGATRON_TE_GROUPED_LINEAR_PRE_INPUT_TRIM_RESERVE_MB",\n'
        '                "MEGATRON_TE_GROUPED_LINEAR_PRE_INPUT_TRIM_CACHED_MB",\n'
        '                "MEGATRON_TE_GROUPED_LINEAR_PRE_INPUT_TRIM_SYNC",\n'
        "                2048,\n"
        "                512,\n"
        "                256,\n"
        "            )\n"
    )
    input_alloc = (
        "            # Disable bulk allocation when CPU offloading is active: offloading skips small\n"
        "            # tensors (like scales), but bulk allocation shares storage across all tensors,\n"
        "            # so if scales can't be offloaded, nothing in the group can be offloaded.\n"
        "            inputmats = tex.split_quantize(\n"
    )
    if input_call not in text:
        if input_alloc not in text:
            raise SystemExit(f"Could not find grouped-linear input quantization site in {path}")
        text = text.replace(input_alloc, input_call + input_alloc, 1)

    output_call = (
        f"        {MARKER}: output allocation frontier\n"
        "        _megatron_grouped_linear_alloc_guard(\n"
        '            "output",\n'
        "            m_splits,\n"
        "            weights_fp8[0].size(0),\n"
        "            activation_dtype,\n"
        "            device,\n"
        '            "MEGATRON_TE_GROUPED_LINEAR_PRE_OUTPUT_TRIM",\n'
        '            "MEGATRON_TE_GROUPED_LINEAR_PRE_OUTPUT_TRIM_FREE_MB",\n'
        '            "MEGATRON_TE_GROUPED_LINEAR_PRE_OUTPUT_TRIM_RESERVE_MB",\n'
        '            "MEGATRON_TE_GROUPED_LINEAR_PRE_OUTPUT_TRIM_CACHED_MB",\n'
        '            "MEGATRON_TE_GROUPED_LINEAR_PRE_OUTPUT_TRIM_SYNC",\n'
        "            2048,\n"
        "            1024,\n"
        "            256,\n"
        "        )\n"
    )
    output_alloc = (
        "        # Initialize output tensor\n"
        "        out = torch.empty(\n"
        "            [sum(m_splits), weights_fp8[0].size(0)],\n"
        "            dtype=activation_dtype,\n"
        "            device=device,\n"
        "        )\n"
    )
    old_call = (
        f"        {OLD_MARKER}\n"
        "        _megatron_grouped_linear_pre_output_trim(\n"
        "            m_splits,\n"
        "            weights_fp8[0].size(0),\n"
        "            activation_dtype,\n"
        "            device,\n"
        "        )\n"
    )
    if old_call in text:
        text = text.replace(old_call, output_call, 1)
    elif output_call not in text:
        if output_alloc not in text:
            raise SystemExit(f"Could not find grouped-linear output allocation site in {path}")
        text = text.replace(output_alloc, output_call + output_alloc, 1)

    old_save_block = (
        "            tensors_to_save, tensor_objects = prepare_for_saving(\n"
        "                *inputmats,\n"
        "                *weights_fp8,\n"
        "                *weights,\n"
        "                *biases,\n"
        "            )\n"
    )
    new_save_block = (
        "            saved_weights_for_backward = (\n"
        "                weights\n"
        '                if backward_override == "high_precision"\n'
        "                or not _megatron_grouped_linear_env_flag(\n"
        '                    "MEGATRON_TE_GROUPED_LINEAR_SKIP_RAW_WEIGHT_SAVE", "1"\n'
        "                )\n"
        "                else [None] * num_gemms\n"
        "            )\n"
        "            tensors_to_save, tensor_objects = prepare_for_saving(\n"
        "                *inputmats,\n"
        "                *weights_fp8,\n"
        "                *saved_weights_for_backward,\n"
        "                *biases,\n"
        "            )\n"
    )
    if new_save_block not in text:
        if old_save_block not in text:
            raise SystemExit(f"Could not find grouped-linear autograd save site in {path}")
        text = text.replace(old_save_block, new_save_block, 1)

    if text == original_text:
        print(f"TE grouped-linear allocation guard already patched: {path}")
        return

    path.write_text(text)
    print(f"Patched TE grouped-linear allocation guard: {path}")


def main() -> None:
    _patch_grouped_linear()


if __name__ == "__main__":
    main()
