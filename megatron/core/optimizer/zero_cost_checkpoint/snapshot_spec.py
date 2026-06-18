# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from megatron.core.fp4_utils import is_nvfp4tensor
from megatron.core.fp8_utils import is_float8tensor


DEFAULT_PARAM_TENSOR_ATTRS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
    "_amax_rowwise",
    "_amax_columnwise",
)


@dataclass(frozen=True)
class TensorSnapshotSpec:
    name: str
    tensor: torch.Tensor
    persistent: bool = True


def iter_torch_optimizers(megatron_optimizer: Any) -> list[torch.optim.Optimizer]:
    if isinstance(megatron_optimizer, torch.optim.Optimizer):
        return [megatron_optimizer]
    if hasattr(megatron_optimizer, "chained_optimizers"):
        result = []
        for child in megatron_optimizer.chained_optimizers:
            result.extend(iter_torch_optimizers(child))
        return result
    optimizer = getattr(megatron_optimizer, "optimizer", None)
    if isinstance(optimizer, torch.optim.Optimizer):
        return [optimizer]
    return []


class SnapshotPlanner:
    def __init__(self, extra_tensor_attrs: tuple[str, ...] = ()):
        self.param_tensor_attrs = tuple(
            dict.fromkeys(DEFAULT_PARAM_TENSOR_ATTRS + tuple(extra_tensor_attrs))
        )

    def plan(self, megatron_optimizer: Any) -> list[TensorSnapshotSpec]:
        specs: list[TensorSnapshotSpec] = []
        seen: set[int] = set()
        covered_storages: set[tuple[str, int]] = set()
        for opt_idx, optimizer in enumerate(iter_torch_optimizers(megatron_optimizer)):
            fused = getattr(optimizer, "_zcc_fused_state_buffer", None)
            if isinstance(fused, torch.Tensor):
                self._append_tensor(
                    specs,
                    seen,
                    covered_storages,
                    f"optimizer{opt_idx}.fused_state",
                    fused,
                    cover_storage=True,
                )
            for group_idx, group in enumerate(optimizer.param_groups):
                for param_idx, param in enumerate(group["params"]):
                    prefix = f"optimizer{opt_idx}.group{group_idx}.param{param_idx}"
                    self._append_param_specs(specs, seen, covered_storages, prefix, param)
                    state = optimizer.state.get(param, {})
                    for key, value in state.items():
                        self._append_state_specs(
                            specs,
                            seen,
                            covered_storages,
                            f"{prefix}.state.{key}",
                            value,
                        )
        return specs

    def plan_bucket(self, megatron_optimizer: Any, bucket: Any) -> list[TensorSnapshotSpec]:
        bucket_params = {id(param) for param in getattr(bucket, "params_list", [])}
        if not bucket_params:
            return []
        specs: list[TensorSnapshotSpec] = []
        seen: set[int] = set()
        covered_storages: set[tuple[str, int]] = set()
        for opt_idx, optimizer in enumerate(iter_torch_optimizers(megatron_optimizer)):
            fused = getattr(optimizer, "_zcc_fused_state_buffer", None)
            if isinstance(fused, torch.Tensor):
                self._append_tensor(
                    specs,
                    seen,
                    covered_storages,
                    f"optimizer{opt_idx}.fused_state",
                    fused,
                    cover_storage=True,
                )
            for group_idx, group in enumerate(optimizer.param_groups):
                for param_idx, param in enumerate(group["params"]):
                    if id(param) not in bucket_params:
                        continue
                    prefix = f"optimizer{opt_idx}.group{group_idx}.param{param_idx}"
                    self._append_param_specs(specs, seen, covered_storages, prefix, param)
                    state = optimizer.state.get(param, {})
                    for key, value in state.items():
                        self._append_state_specs(
                            specs,
                            seen,
                            covered_storages,
                            f"{prefix}.state.{key}",
                            value,
                        )
        return specs

    def metadata(
        self,
        megatron_optimizer: Any,
        *,
        include_dither: bool,
        include_rng: bool = True,
        opt_param_scheduler: Any | None = None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {"optimizers": []}
        for optimizer in iter_torch_optimizers(megatron_optimizer):
            meta["optimizers"].append(
                {
                    "type": type(optimizer).__name__,
                    "param_groups": [
                        {k: v for k, v in group.items() if k != "params"}
                        for group in optimizer.param_groups
                    ],
                }
            )
        if include_dither:
            try:
                from megatron.core.optimizer import nvfp4_sr

                meta["dither_step_counter"] = int(nvfp4_sr._DITHER_STEP_COUNTER[0])
            except (ImportError, AttributeError, IndexError, TypeError, ValueError):
                pass
        if opt_param_scheduler is not None and hasattr(opt_param_scheduler, "state_dict"):
            meta["opt_param_scheduler"] = opt_param_scheduler.state_dict()
        if include_rng:
            meta["torch_rng_state"] = torch.get_rng_state()
            if torch.cuda.is_available():
                meta["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return meta

    def _append_param_specs(
        self,
        specs: list[TensorSnapshotSpec],
        seen: set[int],
        covered_storages: set[tuple[str, int]],
        prefix: str,
        param: torch.Tensor,
    ) -> None:
        if not (is_nvfp4tensor(param) or is_float8tensor(param) or self._has_param_components(param)):
            self._append_tensor(specs, seen, covered_storages, f"{prefix}.data", param)
        for attr in self.param_tensor_attrs:
            tensor = getattr(param, attr, None)
            if isinstance(tensor, torch.Tensor):
                self._append_tensor(
                    specs,
                    seen,
                    covered_storages,
                    f"{prefix}.{attr}",
                    tensor,
                )

    def _append_state_specs(
        self,
        specs: list[TensorSnapshotSpec],
        seen: set[int],
        covered_storages: set[tuple[str, int]],
        name: str,
        value: Any,
    ) -> None:
        if isinstance(value, torch.Tensor):
            self._append_tensor(
                specs,
                seen,
                covered_storages,
                name,
                value,
                skip_covered_storage=True,
            )
            return
        if all(hasattr(value, attr) for attr in ("is_quantized", "numel")):
            for attr in ("_quantized", "_scales", "_data"):
                tensor = getattr(value, attr, None)
                if isinstance(tensor, torch.Tensor):
                    self._append_tensor(
                        specs,
                        seen,
                        covered_storages,
                        f"{name}.{attr}",
                        tensor,
                        skip_covered_storage=True,
                    )

    def _append_tensor(
        self,
        specs: list[TensorSnapshotSpec],
        seen: set[int],
        covered_storages: set[tuple[str, int]],
        name: str,
        tensor: torch.Tensor,
        *,
        cover_storage: bool = False,
        skip_covered_storage: bool = False,
    ) -> None:
        if id(tensor) in seen or name.split(".")[-1].startswith("_fa_"):
            return
        storage_key = self._storage_key(tensor)
        if (
            skip_covered_storage
            and storage_key is not None
            and storage_key in covered_storages
        ):
            return
        seen.add(id(tensor))
        if cover_storage and storage_key is not None:
            covered_storages.add(storage_key)
        specs.append(TensorSnapshotSpec(name, tensor))

    @staticmethod
    def _storage_key(tensor: torch.Tensor) -> tuple[str, int] | None:
        if tensor.numel() == 0:
            return None
        try:
            return (str(tensor.device), tensor.untyped_storage().data_ptr())
        except RuntimeError as exc:
            if "invalid python storage" in str(exc):
                return None
            raise

    def _has_param_components(self, tensor: torch.Tensor) -> bool:
        return any(
            isinstance(getattr(tensor, attr, None), torch.Tensor)
            for attr in self.param_tensor_attrs
        )
