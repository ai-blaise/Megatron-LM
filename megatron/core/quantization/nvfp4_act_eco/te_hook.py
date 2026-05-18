# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional Transformer Engine `te.Linear` hook for activation-ECO.

Production NVFP4 training uses `te.Linear` for the matmul; activation-ECO
needs to observe the same activation cast TE applies. The hook in this
module wraps a `te.Linear` instance so that:

  * forward: capture the pre-cast BF16 activation, run TE's NVFP4 forward
    as today, then return TE's output unchanged.
  * backward: in addition to TE's standard STE-through-cast gradient,
    add the activation-ECO correction
    ``dW += dy.T @ (x_pre - q(x_pre))``
    to TE's accumulated weight gradient.

TE is GPU-only and must be installed separately (Blackwell + matching
cuBLAS). This module imports lazily so non-TE environments (CPU CI,
fallback BF16 backends) can still import the rest of the
``nvfp4_act_eco`` package.
"""

from __future__ import annotations

from collections import deque
import os
import warnings

import torch

from megatron.core.quantization.nvfp4_act_eco.codec import Nvfp4ActEcoConfig
from megatron.core.quantization.nvfp4_act_eco.reference import (
    activation_eco_bias_correction,
    nvfp4_act_quant_forward,
)

ACT_ECO_PENDING_GRAD_CORRECTION_ATTR = "_nvfp4_act_eco_pending_grad_correction"


def _try_import_te():
    try:
        import transformer_engine.pytorch as te

        return te
    except Exception:
        return None


def is_te_available() -> bool:
    """Return True if Transformer Engine is importable in this environment."""
    return _try_import_te() is not None


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.getenv("RANK", "0"))


def _rank_selected(spec: str, rank: int) -> bool:
    spec = spec.strip()
    if not spec or spec == "all":
        return True
    return rank in {int(item) for item in spec.split(",") if item.strip()}


def _debug_sync(label: str) -> None:
    """Synchronize at activation-ECO debug boundaries.

    CUDA illegal-address failures are asynchronous. This opt-in fence turns
    later allocator/NCCL crashes into a failure at the nearest activation-ECO
    phase, while rank filtering keeps distributed logs readable.
    """

    if os.getenv("MEGATRON_ACT_ECO_DEBUG_SYNC", "0").lower() not in ("1", "true", "yes", "on"):
        return
    if not torch.cuda.is_available():
        return
    rank = _distributed_rank()
    if not _rank_selected(os.getenv("MEGATRON_ACT_ECO_DEBUG_RANKS", "all"), rank):
        return
    if os.getenv("MEGATRON_ACT_ECO_DEBUG_VERBOSE", "0").lower() in ("1", "true", "yes", "on"):
        print(f"[rank{rank}] act-eco sync: {label}", flush=True)
    torch.cuda.synchronize()


def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output, None
    if isinstance(output, tuple):
        for idx, item in enumerate(output):
            if isinstance(item, torch.Tensor):
                return item, ("tuple", idx)
    if isinstance(output, list):
        for idx, item in enumerate(output):
            if isinstance(item, torch.Tensor):
                return item, ("list", idx)
    return None, None


def _replace_first_tensor(output, replacement, location):
    if location is None:
        return replacement
    kind, idx = location
    if kind == "tuple":
        items = list(output)
        items[idx] = replacement
        return tuple(items)
    if kind == "list":
        items = list(output)
        items[idx] = replacement
        return items
    raise AssertionError(f"unexpected output location {location!r}")


def _forced_release_enabled_for_group(name: str | None) -> bool:
    if name is None:
        return True
    try:
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            fine_grained_offloading_forced_release_enabled,
        )

        return fine_grained_offloading_forced_release_enabled(name)
    except Exception:
        return True


def _maybe_clone_captured_input(
    x_pre: torch.Tensor,
    clone_captured_input: bool,
    forced_release_group: str | None,
) -> torch.Tensor:
    if clone_captured_input and _forced_release_enabled_for_group(forced_release_group):
        return x_pre.clone()
    return x_pre


def _te_nvfp4_quant_dequant(x_flat: torch.Tensor, config: Nvfp4ActEcoConfig):
    """Quantize/dequantize with TE's NVFP4 tensor path when it can handle the shape.

    TE's current NVFP4 quantizer requires the flattened leading dimension and hidden
    dimension to be multiples of 16. The SFT StreamBP chunks are normally aligned,
    but this function returns ``None`` rather than changing semantics for tails or
    small unit tests; callers then use the reference fake-quant path.
    """

    if not x_flat.is_cuda:
        return None
    if x_flat.dim() != 2 or x_flat.size(0) % config.block_size != 0:
        return None
    if x_flat.size(1) % config.block_size != 0:
        return None
    te = _try_import_te()
    if te is None or not hasattr(te, "NVFP4Quantizer"):
        return None
    try:
        quantizer = te.NVFP4Quantizer(rowwise=True, columnwise=False)
        q_tensor = quantizer.quantize(x_flat.contiguous())
        return q_tensor.dequantize(dtype=x_flat.dtype)
    except Exception as exc:
        if not getattr(_te_nvfp4_quant_dequant, "_warned", False):
            warnings.warn(
                "Falling back to reference NVFP4 activation-ECO quantizer because "
                f"Transformer Engine NVFP4 quantization failed: {exc}",
                RuntimeWarning,
            )
            _te_nvfp4_quant_dequant._warned = True
        return None


def _quant_dequant_activation(
    x_flat: torch.Tensor,
    config: Nvfp4ActEcoConfig,
    *,
    backend: str,
) -> torch.Tensor:
    if backend not in {"te", "reference"}:
        raise ValueError(f"unknown activation-ECO quantizer backend {backend!r}")
    if backend == "te":
        q_te = _te_nvfp4_quant_dequant(x_flat, config)
        if q_te is not None:
            return q_te
    return nvfp4_act_quant_forward(x_flat.to(torch.float32), config).to(x_flat.dtype)


def _tensor_model_parallel_rank_size() -> tuple[int, int]:
    try:
        from megatron.core import parallel_state

        if not torch.distributed.is_initialized():
            return 0, 1
        return (
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_tensor_model_parallel_world_size(),
        )
    except Exception:
        return 0, 1


def _align_activation_and_grad_rows(
    x_flat: torch.Tensor,
    dy_flat: torch.Tensor,
    *,
    module_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match captured activation rows to output-gradient rows.

    Some TE sequence-parallel linears expose a gathered activation to the
    forward hook but only this TP rank's sequence shard reaches the output
    gradient hook; other column-parallel paths expose the inverse. The
    weight-gradient correction is local in rows, so when the row-count ratio
    exactly matches TP size, use this rank's shard of the gathered side.
    """

    x_rows = x_flat.size(0)
    dy_rows = dy_flat.size(0)
    if x_rows == dy_rows:
        return x_flat, dy_flat

    if dy_rows > 0 and x_rows % dy_rows == 0:
        ratio = x_rows // dy_rows
        tp_rank, tp_size = _tensor_model_parallel_rank_size()
        if ratio == tp_size and 0 <= tp_rank < tp_size:
            start = tp_rank * dy_rows
            return x_flat[start : start + dy_rows], dy_flat

    if x_rows > 0 and dy_rows % x_rows == 0:
        ratio = dy_rows // x_rows
        tp_rank, tp_size = _tensor_model_parallel_rank_size()
        if ratio == tp_size and 0 <= tp_rank < tp_size:
            start = tp_rank * x_rows
            return x_flat, dy_flat[start : start + x_rows]

    raise RuntimeError(
        f"Activation-ECO {module_name} hook cannot align activation rows "
        f"{tuple(x_flat.shape)} with grad rows {tuple(dy_flat.shape)}"
    )


def _stash_or_return_corrected_grad(
    param: torch.nn.Parameter,
    grad: torch.Tensor,
    correction: torch.Tensor,
) -> torch.Tensor:
    correction = correction.to(grad.dtype).reshape(grad.shape).detach()

    # Megatron DDP consumes ``param.grad`` into ``param.main_grad`` from an
    # AccumulateGrad post-hook.  Returning a replacement grad from this tensor
    # hook makes that short-lived allocation the object DDP later clears, which
    # has been fragile with TE/NVFP4/StreamBP.  Keep the original grad object in
    # that path and let DDP consume the correction explicitly.
    if getattr(param, "main_grad", None) is not None:
        pending = getattr(param, ACT_ECO_PENDING_GRAD_CORRECTION_ATTR, None)
        if pending is None:
            setattr(param, ACT_ECO_PENDING_GRAD_CORRECTION_ATTR, correction)
        else:
            if pending.shape == correction.shape and pending.dtype == correction.dtype:
                pending.add_(correction)
            else:
                setattr(param, ACT_ECO_PENDING_GRAD_CORRECTION_ATTR, pending + correction)
        return grad

    return grad + correction


def _stash_main_grad_correction(
    param: torch.nn.Parameter,
    correction: torch.Tensor,
) -> bool:
    """Stash activation-ECO correction for Megatron DDP main-grad consumption.

    In Megatron's distributed optimizer path, parameters already own a
    ``main_grad`` buffer.  That lets activation-ECO add its correction through
    the DDP post-hook without waiting for the parameter's regular grad hook.
    Doing this from the output-gradient hook releases the captured activation
    before TE's grouped-linear backward allocates its larger work tensors.
    """

    main_grad = getattr(param, "main_grad", None)
    if main_grad is None:
        return False

    correction = correction.to(main_grad.dtype).reshape(param.shape).detach()
    pending = getattr(param, ACT_ECO_PENDING_GRAD_CORRECTION_ATTR, None)
    if pending is None:
        setattr(param, ACT_ECO_PENDING_GRAD_CORRECTION_ATTR, correction)
    else:
        if pending.shape == correction.shape and pending.dtype == correction.dtype:
            pending.add_(correction)
        else:
            setattr(param, ACT_ECO_PENDING_GRAD_CORRECTION_ATTR, pending + correction)
    return True


def _add_activation_eco_bias_correction_to_main_grad_(
    param: torch.nn.Parameter,
    grad_y: torch.Tensor,
    x_pre: torch.Tensor,
    q_x: torch.Tensor,
) -> bool:
    """Accumulate activation-ECO correction directly into ``param.main_grad``.

    The correction is ``grad_y.T @ (x_pre - q_x)``.  Materializing that full
    matrix is expensive at 345B scale and caused near-OOM failures during
    StreamBP replay.  When Megatron has already allocated ``main_grad``, write
    the GEMM result there directly with ``addmm_``.
    """

    main_grad = getattr(param, "main_grad", None)
    if main_grad is None:
        return False

    flat_grad = grad_y.reshape(-1, grad_y.shape[-1])
    flat_x = x_pre.reshape(-1, x_pre.shape[-1])
    flat_qx = q_x.reshape(-1, q_x.shape[-1])
    if flat_grad.shape[0] != flat_x.shape[0] or flat_x.shape != flat_qx.shape:
        raise RuntimeError(
            "Activation-ECO correction shape mismatch: "
            f"grad={tuple(flat_grad.shape)}, x={tuple(flat_x.shape)}, qx={tuple(flat_qx.shape)}"
        )

    target = main_grad.reshape(param.shape)
    expected_shape = (flat_grad.shape[-1], flat_x.shape[-1])
    if tuple(target.shape) != expected_shape:
        raise RuntimeError(
            "Activation-ECO correction target shape mismatch: "
            f"main_grad={tuple(target.shape)}, expected={expected_shape}"
        )

    compute_dtype = target.dtype
    grad_mat = flat_grad.to(compute_dtype)
    err = flat_x.to(compute_dtype)
    if err.data_ptr() == flat_x.data_ptr():
        err = err.clone()
    err.sub_(flat_qx.to(compute_dtype))
    target.addmm_(grad_mat.transpose(0, 1), err)
    return True


def _remove_pending_entry(entries: deque, entry: dict) -> deque:
    """Remove a captured activation-ECO entry by identity.

    ``deque.remove`` compares dictionaries by value, which compares tensor
    values and can raise "Boolean value of Tensor is ambiguous".
    """

    return deque(candidate for candidate in entries if candidate is not entry)


def pop_act_eco_grad_correction(param: torch.nn.Parameter) -> torch.Tensor | None:
    correction = getattr(param, ACT_ECO_PENDING_GRAD_CORRECTION_ATTR, None)
    if correction is not None:
        delattr(param, ACT_ECO_PENDING_GRAD_CORRECTION_ATTR)
    return correction


def install_act_eco_on_te_linear(
    te_linear: torch.nn.Module,
    config: Nvfp4ActEcoConfig,
    *,
    capture_recompute_only: bool = True,
    clone_captured_input: bool = False,
    forced_release_group: str | None = None,
    quantizer_backend: str = "te",
    correction_dtype: torch.dtype = torch.float32,
    module_label: str = "te_linear",
) -> None:
    """Attach activation-ECO bias correction to a `te.Linear` instance.

    Implementation strategy: register a forward_pre_hook that stashes
    `x_pre` on the module, then a backward hook on the module's
    weight that adds ``dy.T @ e_x`` to the accumulated gradient. We
    reconstruct ``q(x_pre)`` using TE's NVFP4 quantizer when possible,
    falling back to the reference fake-quant path for CPU tests and
    unsupported tail shapes.

    ``capture_recompute_only`` keeps the hook memory-aware for StreamBP:
    the no-grad forward pass does not hold BF16 activations, while the
    backward replay pass captures only the current chunk until its
    corresponding weight gradient is produced.

    ``clone_captured_input`` is needed only when fine-grained activation
    offload may call ``untyped_storage().resize_(0)`` on the linear input
    after forward. A detached view would share that storage and become
    invalid before the weight hook runs.

    No-op when te_linear has no ``weight`` attribute.
    """
    if not hasattr(te_linear, "weight"):
        return
    if getattr(te_linear, "_act_eco_installed", False):
        return

    cell = {"entries": deque(), "unbound": deque()}

    def _pre(_module, args, _kwargs):
        if not args:
            return None
        x = args[0]
        if capture_recompute_only and not torch.is_grad_enabled():
            return None
        if not isinstance(x, torch.Tensor) or not x.requires_grad:
            return None
        x_pre = _maybe_clone_captured_input(
            x.detach(), clone_captured_input, forced_release_group
        )
        entry = {"x_pre": x_pre, "dy": None}
        cell["entries"].append(entry)
        cell["unbound"].append(entry)
        return None

    def _post(_module, _args, output):
        # Tap into the output's grad_fn so we can capture dy in backward.
        y, location = _first_tensor(output)
        if y is None or not y.requires_grad:
            return output
        if not cell["unbound"]:
            return output
        entry = cell["unbound"].pop()

        class _CaptureGrad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y):
                return y

            @staticmethod
            def backward(ctx, dy):
                param = te_linear.weight
                if getattr(param, "main_grad", None) is not None:
                    x_pre = entry["x_pre"]
                    x_flat = x_pre.reshape(-1, x_pre.shape[-1])
                    dy_flat = dy.reshape(-1, dy.shape[-1])
                    _debug_sync(f"{module_label}:before_align")
                    x_flat, dy_flat = _align_activation_and_grad_rows(
                        x_flat,
                        dy_flat,
                        module_name=te_linear.__class__.__name__,
                    )
                    _debug_sync(f"{module_label}:after_align")
                    q_x = _quant_dequant_activation(
                        x_flat,
                        config,
                        backend=quantizer_backend,
                    )
                    _debug_sync(f"{module_label}:after_quant_dequant")
                    if _add_activation_eco_bias_correction_to_main_grad_(
                        param,
                        dy_flat.to(correction_dtype),
                        x_flat.to(correction_dtype),
                        q_x.to(correction_dtype),
                    ):
                        cell["entries"] = _remove_pending_entry(cell["entries"], entry)
                        _debug_sync(f"{module_label}:after_grad_addmm")
                        return dy
                    correction = activation_eco_bias_correction(
                        dy_flat.to(correction_dtype),
                        x_flat.to(correction_dtype),
                        q_x.to(correction_dtype),
                    )
                    _debug_sync(f"{module_label}:after_correction_matmul")
                    if _stash_main_grad_correction(param, correction):
                        cell["entries"] = _remove_pending_entry(cell["entries"], entry)
                        _debug_sync(f"{module_label}:after_grad_stash")
                        return dy
                entry["dy"] = dy.detach()
                return dy

        return _replace_first_tensor(output, _CaptureGrad.apply(y), location)

    def _weight_hook(grad):
        if not cell["entries"]:
            return grad
        ready = [entry for entry in cell["entries"] if entry["dy"] is not None]
        if not ready:
            return grad
        cell["entries"] = deque(entry for entry in cell["entries"] if entry["dy"] is None)

        param = te_linear.weight
        corrected_grad = grad
        for entry in ready:
            x_pre = entry["x_pre"]
            dy = entry["dy"]
            x_flat = x_pre.reshape(-1, x_pre.shape[-1])
            dy_flat = dy.reshape(-1, dy.shape[-1])
            _debug_sync(f"{module_label}:before_align")
            x_flat, dy_flat = _align_activation_and_grad_rows(
                x_flat,
                dy_flat,
                module_name=te_linear.__class__.__name__,
            )
            _debug_sync(f"{module_label}:after_align")
            q_x = _quant_dequant_activation(
                x_flat,
                config,
                backend=quantizer_backend,
            )
            _debug_sync(f"{module_label}:after_quant_dequant")
            if getattr(param, "main_grad", None) is not None:
                _add_activation_eco_bias_correction_to_main_grad_(
                    param,
                    dy_flat.to(correction_dtype),
                    x_flat.to(correction_dtype),
                    q_x.to(correction_dtype),
                )
                _debug_sync(f"{module_label}:after_grad_addmm")
            else:
                correction = activation_eco_bias_correction(
                    dy_flat.to(correction_dtype),
                    x_flat.to(correction_dtype),
                    q_x.to(correction_dtype),
                )
                _debug_sync(f"{module_label}:after_correction_matmul")
                corrected_grad = _stash_or_return_corrected_grad(
                    param, corrected_grad, correction
                )
            _debug_sync(f"{module_label}:after_grad_add")
        return corrected_grad

    te_linear.register_forward_pre_hook(_pre, with_kwargs=True)
    te_linear.register_forward_hook(_post)
    te_linear.weight.register_hook(_weight_hook)
    te_linear._act_eco_installed = True


def install_act_eco_on_te_grouped_linear(
    te_grouped_linear: torch.nn.Module,
    config: Nvfp4ActEcoConfig,
    *,
    num_gemms: int,
    capture_recompute_only: bool = True,
    clone_captured_input: bool = False,
    forced_release_group: str | None = None,
    quantizer_backend: str = "te",
    correction_dtype: torch.dtype = torch.float32,
    module_label: str = "te_grouped_linear",
) -> None:
    """Attach activation-ECO correction to a TE GroupedLinear-like module.

    GroupedLinear receives a single packed activation tensor plus per-expert row
    counts. The correction is computed per expert:
    ``dW_i += dY_i.T @ (X_i - q_nvfp4(X_i))``.
    """

    if num_gemms <= 0:
        return
    if getattr(te_grouped_linear, "_act_eco_installed", False):
        return
    if not hasattr(te_grouped_linear, "weight0"):
        return

    cell = {"entries": deque(), "unbound": deque()}

    def _pre(_module, args, kwargs):
        if len(args) < 2 and "m_splits" not in kwargs:
            return None
        x = args[0] if args else kwargs.get("x")
        m_splits = args[1] if len(args) >= 2 else kwargs["m_splits"]
        if capture_recompute_only and not torch.is_grad_enabled():
            return None
        if not isinstance(x, torch.Tensor) or not x.requires_grad:
            return None
        x_pre = _maybe_clone_captured_input(
            x.detach(), clone_captured_input, forced_release_group
        )
        if isinstance(m_splits, torch.Tensor):
            m_splits = m_splits.detach().cpu().to(torch.long).tolist()
        counts = [int(v) for v in m_splits]
        if len(counts) != num_gemms:
            raise RuntimeError(
                f"Activation-ECO grouped TE hook expected {num_gemms} row counts, "
                f"got {len(counts)}"
            )
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        entry = {
            "x_pre": x_pre,
            "dy": None,
            "offsets": offsets,
            "pending_experts": {idx for idx, count in enumerate(counts) if count > 0},
        }
        cell["entries"].append(entry)
        cell["unbound"].append(entry)
        return None

    def _post(_module, _args, output):
        y, location = _first_tensor(output)
        if y is None or not y.requires_grad:
            return output
        if not cell["unbound"]:
            return output
        entry = cell["unbound"].pop()

        class _CaptureGrad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y):
                return y

            @staticmethod
            def backward(ctx, dy):
                pending_experts = list(entry["pending_experts"])
                if pending_experts and all(
                    getattr(getattr(te_grouped_linear, f"weight{idx}"), "main_grad", None)
                    is not None
                    for idx in pending_experts
                ):
                    dy_flat_all = dy.reshape(-1, dy.shape[-1])
                    for expert_idx in pending_experts:
                        offsets = entry["offsets"]
                        start = offsets[expert_idx]
                        end = offsets[expert_idx + 1]
                        if end == start:
                            continue
                        param = getattr(te_grouped_linear, f"weight{expert_idx}")
                        x_flat = entry["x_pre"][start:end].reshape(
                            -1, entry["x_pre"].shape[-1]
                        )
                        dy_flat = dy_flat_all[start:end]
                        _debug_sync(
                            f"{module_label}.expert{expert_idx}:before_quant_dequant"
                        )
                        q_x = _quant_dequant_activation(
                            x_flat,
                            config,
                            backend=quantizer_backend,
                        )
                        _debug_sync(
                            f"{module_label}.expert{expert_idx}:after_quant_dequant"
                        )
                        if _add_activation_eco_bias_correction_to_main_grad_(
                            param,
                            dy_flat.to(correction_dtype),
                            x_flat.to(correction_dtype),
                            q_x.to(correction_dtype),
                        ):
                            _debug_sync(f"{module_label}.expert{expert_idx}:after_grad_addmm")
                        else:
                            correction = activation_eco_bias_correction(
                                dy_flat.to(correction_dtype),
                                x_flat.to(correction_dtype),
                                q_x.to(correction_dtype),
                            )
                            _debug_sync(
                                f"{module_label}.expert{expert_idx}:after_correction_matmul"
                            )
                            _stash_main_grad_correction(param, correction)
                            _debug_sync(f"{module_label}.expert{expert_idx}:after_grad_stash")
                    entry["pending_experts"].clear()
                    cell["entries"] = _remove_pending_entry(cell["entries"], entry)
                    return dy
                entry["dy"] = dy.detach()
                return dy

        return _replace_first_tensor(output, _CaptureGrad.apply(y), location)

    def _make_weight_hook(expert_idx: int):
        def _weight_hook(grad):
            if not cell["entries"]:
                return grad
            ready = [
                entry
                for entry in cell["entries"]
                if entry["dy"] is not None and expert_idx in entry["pending_experts"]
            ]
            if not ready:
                return grad

            param = getattr(te_grouped_linear, f"weight{expert_idx}")
            corrected_grad = grad
            for entry in ready:
                offsets = entry["offsets"]
                start = offsets[expert_idx]
                end = offsets[expert_idx + 1]
                entry["pending_experts"].remove(expert_idx)
                if end != start:
                    x_flat = entry["x_pre"][start:end].reshape(-1, entry["x_pre"].shape[-1])
                    dy_flat = entry["dy"][start:end].reshape(-1, entry["dy"].shape[-1])
                    _debug_sync(f"{module_label}.expert{expert_idx}:before_quant_dequant")
                    q_x = _quant_dequant_activation(
                        x_flat,
                        config,
                        backend=quantizer_backend,
                    )
                    _debug_sync(f"{module_label}.expert{expert_idx}:after_quant_dequant")
                    if getattr(param, "main_grad", None) is not None:
                        _add_activation_eco_bias_correction_to_main_grad_(
                            param,
                            dy_flat.to(correction_dtype),
                            x_flat.to(correction_dtype),
                            q_x.to(correction_dtype),
                        )
                        _debug_sync(f"{module_label}.expert{expert_idx}:after_grad_addmm")
                    else:
                        correction = activation_eco_bias_correction(
                            dy_flat.to(correction_dtype),
                            x_flat.to(correction_dtype),
                            q_x.to(correction_dtype),
                        )
                        _debug_sync(f"{module_label}.expert{expert_idx}:after_correction_matmul")
                        corrected_grad = _stash_or_return_corrected_grad(
                            param, corrected_grad, correction
                        )
                    _debug_sync(f"{module_label}.expert{expert_idx}:after_grad_add")

            cell["entries"] = deque(
                entry for entry in cell["entries"] if entry["pending_experts"]
            )
            return corrected_grad

        return _weight_hook

    te_grouped_linear.register_forward_pre_hook(_pre, with_kwargs=True)
    te_grouped_linear.register_forward_hook(_post)
    for idx in range(num_gemms):
        getattr(te_grouped_linear, f"weight{idx}").register_hook(_make_weight_hook(idx))
    te_grouped_linear._act_eco_installed = True
