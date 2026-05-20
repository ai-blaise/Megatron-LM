#!/usr/bin/env python3
"""Patch Transformer Engine backward paths for retained-graph reentry.

StreamBP + split-QK DSA can re-enter the same TE ``Linear`` backward while
propagating memory-split K/V gradients. Current TE releases
``ctx.tensor_objects`` after the first restore, which is fine for ordinary
single-pass backward but breaks retained-graph reentry.

The same retained-graph issue can appear through TE's operation fuser. Fuser
stores per-basic-op saved tensor ranges and clears those ranges after the first
backward restore. TE basic ops can also call ``clear_tensor_data`` on restored
saved tensors, which is unsafe if the retained graph will be re-entered again.

The patch is enabled by ``MEGATRON_TE_RETAIN_TENSOR_OBJECTS_FOR_REENTRANT_BACKWARD``,
but the extra retention is active only while
``MEGATRON_TE_REENTRANT_BACKWARD_ACTIVE=1`` and
``MEGATRON_TE_REENTRANT_BACKWARD_RETAIN=1``. DSA sets those guards only around
nested backward calls that still have later K/V chunks to consume, so normal TE
backwards and the final reentrant chunk still release their saved state.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


FLAG = "MEGATRON_TE_RETAIN_TENSOR_OBJECTS_FOR_REENTRANT_BACKWARD"
ACTIVE_FLAG = "MEGATRON_TE_REENTRANT_BACKWARD_ACTIVE"
RETAIN_FLAG = "MEGATRON_TE_REENTRANT_BACKWARD_RETAIN"
LINEAR_MARKER = "# Megatron StreamBP/DSA retained-graph reentry patch"
LINEAR_SAVED_DATA_MARKER = "# Megatron StreamBP/DSA retained-graph saved-data patch"
LINEAR_WGRAD_INPUT_MARKER = "# Megatron StreamBP/DSA reentrant NVFP4 wgrad input patch"
LINEAR_WGRAD_GRAD_OUTPUT_MARKER = (
    "# Megatron StreamBP/DSA reentrant NVFP4 wgrad grad-output patch"
)
LINEAR_GRAD_OUTPUT_USAGE_MARKER = (
    "# Megatron StreamBP/DSA reentrant NVFP4 grad-output preprocess usage patch"
)
FUSER_MARKER = "# Megatron StreamBP/DSA retained-graph fuser-range patch"
ENABLED_VALUES = '(\"1\", \"true\", \"yes\", \"on\")'


def _active_condition(indent: str) -> str:
    return (
        f"{indent}os.getenv(\"{FLAG}\", \"0\").lower() in {ENABLED_VALUES}\n"
        f"{indent}and os.getenv(\"{ACTIVE_FLAG}\", \"0\").lower() in {ENABLED_VALUES}"
        f"\n{indent}and os.getenv(\"{RETAIN_FLAG}\", \"1\").lower() in {ENABLED_VALUES}"
    )


def _upgrade_legacy_active_conditions(text: str) -> str:
    """Upgrade installed TE patches that predate the final-chunk retain guard."""

    for indent in ("                ", "                        "):
        legacy = (
            f"{indent}os.getenv(\"{FLAG}\", \"0\").lower() in {ENABLED_VALUES}\n"
            f"{indent}and os.getenv(\"{ACTIVE_FLAG}\", \"0\").lower() in {ENABLED_VALUES}"
        )
        upgraded = (
            legacy
            + f"\n{indent}and os.getenv(\"{RETAIN_FLAG}\", \"1\").lower() in {ENABLED_VALUES}"
        )
        if upgraded not in text:
            text = text.replace(legacy, upgraded)
    return text


def _retention_helper_source() -> str:
    return '''
def _megatron_mark_reentrant_tensor_state(tensor):
    if tensor is None:
        return
    tensor._do_not_clear = True
    if hasattr(tensor, "get_data_tensors"):
        for data_tensor in tensor.get_data_tensors():
            if data_tensor is not None:
                data_tensor._do_not_clear = True
    for attr in (
        "_rowwise_data",
        "_columnwise_data",
        "_rowwise_scale_inv",
        "_columnwise_scale_inv",
        "_amax_rowwise",
        "_amax_columnwise",
        "_data",
        "_transpose",
        "_scale_inv",
        "_amax",
    ):
        value = getattr(tensor, attr, None)
        if value is not None:
            value._do_not_clear = True

'''


def _ensure_retention_helper(text: str) -> str:
    if "def _megatron_mark_reentrant_tensor_state" in text:
        return text
    if "import os\n" not in text:
        raise SystemExit("Could not find import os site for Megatron retention helper")
    return text.replace("import os\n", "import os\n" + _retention_helper_source(), 1)


def _patch_linear() -> None:
    spec = importlib.util.find_spec("transformer_engine.pytorch.module.linear")
    if spec is None or spec.origin is None:
        raise SystemExit("Could not locate transformer_engine.pytorch.module.linear")

    path = Path(spec.origin)
    text = path.read_text()
    original_text = text

    if "import os\n" not in text:
        text = text.replace(
            "from operator import mul as multiply_op\n",
            "from operator import mul as multiply_op\nimport os\n",
        )
    text = _upgrade_legacy_active_conditions(text)
    text = _ensure_retention_helper(text)

    new = (
        f"            {LINEAR_MARKER}\n"
        "            if not (\n"
        f"{_active_condition('                ')}\n"
        "            ):\n"
        "                ctx.tensor_objects = None\n"
    )
    old_unpatched = "            ctx.tensor_objects = None\n"
    old_global_patch = (
        f"            {LINEAR_MARKER}\n"
        f"            if os.getenv(\"{FLAG}\", \"0\").lower() not in {ENABLED_VALUES}:\n"
        "                ctx.tensor_objects = None\n"
    )
    if new in text:
        print(f"TE linear tensor_objects cleanup already patched: {path}")
    elif old_global_patch in text:
        text = text.replace(old_global_patch, new, 1)
    elif old_unpatched in text:
        text = text.replace(old_unpatched, new, 1)
    else:
        raise SystemExit(f"Could not find TE tensor_objects cleanup site in {path}")

    saved_data_patch = (
        f"            {LINEAR_SAVED_DATA_MARKER}\n"
        "            if (\n"
        f"{_active_condition('                ')}\n"
        "            ):\n"
        "                for tensor in (inputmat,):\n"
        "                    _megatron_mark_reentrant_tensor_state(tensor)\n"
    )
    old_saved_data_patch = (
        f"            {LINEAR_SAVED_DATA_MARKER}\n"
        "            if (\n"
        f"{_active_condition('                ')}\n"
        "            ):\n"
        "                for tensor in (inputmat,):\n"
        "                    if tensor is None:\n"
        "                        continue\n"
        "                    tensor._do_not_clear = True\n"
        "                    if hasattr(tensor, \"get_data_tensors\"):\n"
        "                        for data_tensor in tensor.get_data_tensors():\n"
        "                            data_tensor._do_not_clear = True\n"
    )
    old_saved_data_patch_with_none_check = (
        f"            {LINEAR_SAVED_DATA_MARKER}\n"
        "            if (\n"
        f"{_active_condition('                ')}\n"
        "            ):\n"
        "                for tensor in (inputmat,):\n"
        "                    if tensor is None:\n"
        "                        continue\n"
        "                    tensor._do_not_clear = True\n"
        "                    if hasattr(tensor, \"get_data_tensors\"):\n"
        "                        for data_tensor in tensor.get_data_tensors():\n"
        "                            if data_tensor is None:\n"
        "                                continue\n"
        "                            data_tensor._do_not_clear = True\n"
    )
    restore_site = (
        "            inputmat, weight_fp8, weight, bias = (  # pylint: disable=unbalanced-tuple-unpacking\n"
        "                restore_from_saved(ctx.tensor_objects, saved_tensors)\n"
        "            )\n"
    )
    if saved_data_patch in text:
        print(f"TE linear saved-data retention already patched: {path}")
    elif old_saved_data_patch_with_none_check in text:
        text = text.replace(old_saved_data_patch_with_none_check, saved_data_patch, 1)
    elif old_saved_data_patch in text:
        text = text.replace(old_saved_data_patch, saved_data_patch, 1)
    elif restore_site in text:
        text = text.replace(restore_site, restore_site + saved_data_patch, 1)
    else:
        raise SystemExit(f"Could not find TE restore_from_saved site in {path}")

    grad_output_usage_old = (
        "            if ctx.grad_output_quantizer is not None:\n"
        "                quantizer = ctx.grad_output_quantizer\n"
        "                quantizer.set_usage(rowwise=True, columnwise=True)\n"
        "                if ctx.ub_overlap_ag:\n"
        "                    # Userbuffers only supports communication for one\n"
        "                    # tensor usage at a time. Configure quantizer with\n"
        "                    # usage for only dgrad GEMM.\n"
        "                    quantizer.set_usage(columnwise=False)\n"
    )
    grad_output_usage_new = (
        "            if ctx.grad_output_quantizer is not None:\n"
        "                quantizer = ctx.grad_output_quantizer\n"
        "                quantizer.set_usage(rowwise=True, columnwise=True)\n"
        f"                {LINEAR_GRAD_OUTPUT_USAGE_MARKER}\n"
        "                if (\n"
        f"{_active_condition('                    ')}\n"
        "                    and type(quantizer).__name__ == \"NVFP4Quantizer\"\n"
        "                ):\n"
        "                    quantizer.set_usage(columnwise=False)\n"
        "                if ctx.ub_overlap_ag:\n"
        "                    # Userbuffers only supports communication for one\n"
        "                    # tensor usage at a time. Configure quantizer with\n"
        "                    # usage for only dgrad GEMM.\n"
        "                    quantizer.set_usage(columnwise=False)\n"
    )
    if grad_output_usage_new in text:
        print(f"TE linear grad-output preprocess usage already patched: {path}")
    elif grad_output_usage_old in text:
        text = text.replace(grad_output_usage_old, grad_output_usage_new, 1)
    elif LINEAR_GRAD_OUTPUT_USAGE_MARKER not in text:
        raise SystemExit(f"Could not find TE grad-output usage site in {path}")

    wgrad_input_old = (
        "                if ctx.fp8 or ctx.debug:\n"
        "                    if isinstance(inputmat_total, QuantizedTensorStorage):\n"
        "                        inputmat_total.update_usage(columnwise_usage=True)\n"
        "                    else:\n"
        "                        ctx.input_quantizer.set_usage(rowwise=False, columnwise=True)\n"
        "                        inputmat_total = ctx.input_quantizer(inputmat_total)\n"
    )
    wgrad_input_new = (
        "                if ctx.fp8 or ctx.debug:\n"
        f"                    {LINEAR_WGRAD_INPUT_MARKER}\n"
        "                    if (\n"
        f"{_active_condition('                        ')}\n"
        "                        and type(ctx.input_quantizer).__name__ == \"NVFP4Quantizer\"\n"
        "                        and not isinstance(inputmat_total, QuantizedTensorStorage)\n"
        "                    ):\n"
        "                        inputmat_total = cast_if_needed(\n"
        "                            inputmat_total, ctx.activation_dtype\n"
        "                        )\n"
        "                        reentrant_input_quantizer = ctx.input_quantizer.copy()\n"
        "                        reentrant_input_quantizer.set_usage(\n"
        "                            rowwise=False, columnwise=True\n"
        "                        )\n"
        "                        inputmat_total = reentrant_input_quantizer(\n"
        "                            inputmat_total.contiguous()\n"
        "                        )\n"
        "                    elif isinstance(inputmat_total, QuantizedTensorStorage):\n"
        "                        _megatron_mark_reentrant_tensor_state(inputmat_total)\n"
        "                        inputmat_total.update_usage(columnwise_usage=True)\n"
        "                    else:\n"
        "                        ctx.input_quantizer.set_usage(rowwise=False, columnwise=True)\n"
        "                        inputmat_total = ctx.input_quantizer(inputmat_total)\n"
    )
    wgrad_input_old_reentrant = (
        "                if ctx.fp8 or ctx.debug:\n"
        f"                    {LINEAR_WGRAD_INPUT_MARKER}\n"
        "                    if (\n"
        f"{_active_condition('                        ')}\n"
        "                        and type(ctx.input_quantizer).__name__ == \"NVFP4Quantizer\"\n"
        "                    ):\n"
        "                        if isinstance(inputmat_total, QuantizedTensorStorage):\n"
        "                            inputmat_total = inputmat_total.dequantize(\n"
        "                                dtype=ctx.activation_dtype\n"
        "                            )\n"
        "                        else:\n"
        "                            inputmat_total = cast_if_needed(\n"
        "                                inputmat_total, ctx.activation_dtype\n"
        "                            )\n"
        "                        reentrant_input_quantizer = ctx.input_quantizer.copy()\n"
        "                        reentrant_input_quantizer.set_usage(\n"
        "                            rowwise=False, columnwise=True\n"
        "                        )\n"
        "                        inputmat_total = reentrant_input_quantizer(\n"
        "                            inputmat_total.contiguous()\n"
        "                        )\n"
        "                    elif isinstance(inputmat_total, QuantizedTensorStorage):\n"
        "                        inputmat_total.update_usage(columnwise_usage=True)\n"
        "                    else:\n"
        "                        ctx.input_quantizer.set_usage(rowwise=False, columnwise=True)\n"
        "                        inputmat_total = ctx.input_quantizer(inputmat_total)\n"
    )
    if wgrad_input_new in text:
        print(f"TE linear reentrant wgrad input already patched: {path}")
    elif wgrad_input_old_reentrant in text:
        text = text.replace(wgrad_input_old_reentrant, wgrad_input_new, 1)
    elif wgrad_input_old in text:
        text = text.replace(wgrad_input_old, wgrad_input_new, 1)
    elif LINEAR_WGRAD_INPUT_MARKER not in text:
        raise SystemExit(f"Could not find TE wgrad input preparation site in {path}")

    wgrad_grad_output_old = (
        "                if ctx.fp8 or ctx.debug:\n"
        "                    if isinstance(grad_output, QuantizedTensorStorage):\n"
        "                        grad_output.update_usage(columnwise_usage=True)\n"
        "                    else:\n"
        "                        ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)\n"
        "                        grad_output = ctx.grad_output_quantizer(grad_output)\n"
    )
    wgrad_grad_output_new = (
        "                if ctx.fp8 or ctx.debug:\n"
        f"                    {LINEAR_WGRAD_GRAD_OUTPUT_MARKER}\n"
        "                    if (\n"
        f"{_active_condition('                        ')}\n"
        "                        and type(ctx.grad_output_quantizer).__name__ == \"NVFP4Quantizer\"\n"
        "                    ):\n"
        "                        reentrant_grad_output_source = grad_output_arg\n"
        "                        if isinstance(reentrant_grad_output_source, QuantizedTensorStorage):\n"
        "                            reentrant_grad_output_source = reentrant_grad_output_source.dequantize(\n"
        "                                dtype=ctx.activation_dtype\n"
        "                            )\n"
        "                        else:\n"
        "                            reentrant_grad_output_source = cast_if_needed(\n"
        "                                reentrant_grad_output_source, ctx.activation_dtype\n"
        "                            )\n"
        "                        reentrant_grad_output_quantizer = ctx.grad_output_quantizer.copy()\n"
        "                        reentrant_grad_output_quantizer.set_usage(\n"
        "                            rowwise=False, columnwise=True\n"
        "                        )\n"
        "                        grad_output = reentrant_grad_output_quantizer(\n"
        "                            reentrant_grad_output_source.contiguous()\n"
        "                        )\n"
        "                    elif isinstance(grad_output, QuantizedTensorStorage):\n"
        "                        _megatron_mark_reentrant_tensor_state(grad_output)\n"
        "                        grad_output.update_usage(columnwise_usage=True)\n"
        "                    else:\n"
        "                        ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)\n"
        "                        grad_output = ctx.grad_output_quantizer(grad_output)\n"
    )
    wgrad_grad_output_old_reentrant = (
        "                if ctx.fp8 or ctx.debug:\n"
        f"                    {LINEAR_WGRAD_GRAD_OUTPUT_MARKER}\n"
        "                    if (\n"
        f"{_active_condition('                        ')}\n"
        "                        and type(ctx.grad_output_quantizer).__name__ == \"NVFP4Quantizer\"\n"
        "                    ):\n"
        "                        if isinstance(grad_output, QuantizedTensorStorage):\n"
        "                            grad_output = grad_output.dequantize(dtype=ctx.activation_dtype)\n"
        "                        else:\n"
        "                            grad_output = cast_if_needed(grad_output, ctx.activation_dtype)\n"
        "                        reentrant_grad_output_quantizer = ctx.grad_output_quantizer.copy()\n"
        "                        reentrant_grad_output_quantizer.set_usage(\n"
        "                            rowwise=False, columnwise=True\n"
        "                        )\n"
        "                        grad_output = reentrant_grad_output_quantizer(\n"
        "                            grad_output.contiguous()\n"
        "                        )\n"
        "                    elif isinstance(grad_output, QuantizedTensorStorage):\n"
        "                        grad_output.update_usage(columnwise_usage=True)\n"
        "                    else:\n"
        "                        ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)\n"
        "                        grad_output = ctx.grad_output_quantizer(grad_output)\n"
    )
    wgrad_grad_output_current_legacy = (
        "                if ctx.fp8 or ctx.debug:\n"
        f"                    {LINEAR_WGRAD_GRAD_OUTPUT_MARKER}\n"
        "                    if (\n"
        f"{_active_condition('                        ')}\n"
        "                        and type(ctx.grad_output_quantizer).__name__ == \"NVFP4Quantizer\"\n"
        "                        and not isinstance(grad_output, QuantizedTensorStorage)\n"
        "                    ):\n"
        "                        grad_output = cast_if_needed(grad_output, ctx.activation_dtype)\n"
        "                        reentrant_grad_output_quantizer = ctx.grad_output_quantizer.copy()\n"
        "                        reentrant_grad_output_quantizer.set_usage(\n"
        "                            rowwise=False, columnwise=True\n"
        "                        )\n"
        "                        grad_output = reentrant_grad_output_quantizer(\n"
        "                            grad_output.contiguous()\n"
        "                        )\n"
        "                    elif isinstance(grad_output, QuantizedTensorStorage):\n"
        "                        _megatron_mark_reentrant_tensor_state(grad_output)\n"
        "                        grad_output.update_usage(columnwise_usage=True)\n"
        "                    else:\n"
        "                        ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)\n"
        "                        grad_output = ctx.grad_output_quantizer(grad_output)\n"
    )
    if wgrad_grad_output_new in text:
        print(f"TE linear reentrant wgrad grad-output already patched: {path}")
    elif wgrad_grad_output_current_legacy in text:
        text = text.replace(wgrad_grad_output_current_legacy, wgrad_grad_output_new, 1)
    elif wgrad_grad_output_old_reentrant in text:
        text = text.replace(wgrad_grad_output_old_reentrant, wgrad_grad_output_new, 1)
    elif wgrad_grad_output_old in text:
        text = text.replace(wgrad_grad_output_old, wgrad_grad_output_new, 1)
    elif LINEAR_WGRAD_GRAD_OUTPUT_MARKER not in text:
        raise SystemExit(f"Could not find TE wgrad grad-output preparation site in {path}")

    path.write_text(text)
    print(f"Patched TE linear retained-graph reentry support: {path}")


def _patch_fuser() -> None:
    spec = importlib.util.find_spec("transformer_engine.pytorch.ops.fuser")
    if spec is None or spec.origin is None:
        raise SystemExit("Could not locate transformer_engine.pytorch.ops.fuser")

    path = Path(spec.origin)
    text = path.read_text()
    original_text = text

    if "import os\n" not in text:
        text = text.replace(
            "from typing import Any, Optional\nimport itertools\n",
            "from typing import Any, Optional\nimport itertools\nimport os\n",
        )
    text = _upgrade_legacy_active_conditions(text)
    text = _ensure_retention_helper(text)

    new = (
        "            ctx.saved_tensors = saved_tensors[slice(*ctx._saved_tensors_range)]\n"
        f"            {FUSER_MARKER}\n"
        "            if (\n"
        f"{_active_condition('                ')}\n"
        "            ):\n"
        "                for tensor in ctx.saved_tensors:\n"
        "                    _megatron_mark_reentrant_tensor_state(tensor)\n"
        "            else:\n"
        "                ctx._saved_tensors_range = None\n"
    )
    old_unpatched = (
        "            ctx.saved_tensors = saved_tensors[slice(*ctx._saved_tensors_range)]\n"
        "            ctx._saved_tensors_range = None\n"
    )
    old_global_patch = (
        "            ctx.saved_tensors = saved_tensors[slice(*ctx._saved_tensors_range)]\n"
        f"            {FUSER_MARKER}\n"
        f"            if os.getenv(\"{FLAG}\", \"0\").lower() not in {ENABLED_VALUES}:\n"
        "                ctx._saved_tensors_range = None\n"
    )
    old_fuser_data_patch = (
        "            ctx.saved_tensors = saved_tensors[slice(*ctx._saved_tensors_range)]\n"
        f"            {FUSER_MARKER}\n"
        "            if (\n"
        f"{_active_condition('                ')}\n"
        "            ):\n"
        "                for tensor in ctx.saved_tensors:\n"
        "                    if tensor is None:\n"
        "                        continue\n"
        "                    tensor._do_not_clear = True\n"
        "                    if hasattr(tensor, \"get_data_tensors\"):\n"
        "                        for data_tensor in tensor.get_data_tensors():\n"
        "                            data_tensor._do_not_clear = True\n"
        "            else:\n"
        "                ctx._saved_tensors_range = None\n"
    )
    if new in text:
        print(f"TE fuser already patched: {path}")
        if text != original_text:
            path.write_text(text)
            print(f"Upgraded TE fuser retained-graph guard: {path}")
        return None
    if old_fuser_data_patch in text:
        text = text.replace(old_fuser_data_patch, new, 1)
    elif old_global_patch in text:
        text = text.replace(old_global_patch, new, 1)
    elif old_unpatched in text:
        text = text.replace(old_unpatched, new, 1)
    else:
        raise SystemExit(f"Could not find TE fuser saved-tensor range cleanup site in {path}")

    path.write_text(text)
    print(f"Patched TE fuser retained-graph reentry support: {path}")


def main() -> None:
    _patch_linear()
    _patch_fuser()


if __name__ == "__main__":
    main()
