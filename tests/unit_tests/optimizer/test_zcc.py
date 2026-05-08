# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


class _Wrapper:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.config = None
        self.init_state_fn = self._init_state

    @staticmethod
    def _init_state(optimizer, config=None):
        del config
        for group in optimizer.param_groups:
            for param in group["params"]:
                if len(optimizer.state[param]) == 0:
                    optimizer._ensure_state_initialized(param, hparams=group)


class _Scheduler:
    def __init__(self):
        self.value = 0

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


class TestZeroCostCheckpoint(unittest.TestCase):

    def _make_optimizer(self):
        from megatron.core.optimizer.flash_optimizers import FlashAdamW

        param = torch.nn.Parameter(torch.ones(8, dtype=torch.float32))
        optimizer = FlashAdamW(
            [param],
            lr=1e-3,
            quantize=False,
            fused=False,
            master_weight_bits=None,
        )
        return param, optimizer

    def test_config_env(self):
        from megatron.core.optimizer.zero_cost_checkpoint.config import (
            ZeroCostCheckpointConfig,
        )

        class Config:
            enable_zero_cost_checkpoint = True
            zcc_workers_num = 1
            zcc_flash_device = "/tmp/zcc"
            zcc_flash_stripe = ""
            zcc_durable_dir = None
            zcc_durable_interval = 3
            zcc_compress = "none"
            zcc_include_rng = True
            zcc_include_dither = True
            zcc_bucket_hook = True
            zcc_numa_pin = False
            zcc_use_gds = False
            zcc_fault_inject = False
            zcc_recovery_mode = "auto"
            zcc_extra_tensor_attrs = (
                "_fp8_weight_cache, _custom_quant_payload, _fp8_weight_cache"
            )
            zcc_retain_latest = 1

        config = ZeroCostCheckpointConfig.from_optimizer_config(Config())
        self.assertTrue(config.enabled)
        self.assertEqual(config.durable_interval, 3)
        self.assertEqual(config.retain_latest, 1)
        self.assertEqual(
            config.extra_tensor_attrs,
            ("_fp8_weight_cache", "_custom_quant_payload"),
        )
        config.validate()

    def test_env_enable_installs_manager(self):
        from megatron.core.optimizer import _maybe_enable_zero_cost_checkpoint

        class Config:
            enable_zero_cost_checkpoint = False
            zcc_workers_num = 1
            zcc_flash_device = "/tmp/zcc"
            zcc_flash_stripe = ""
            zcc_durable_dir = None
            zcc_durable_interval = 3
            zcc_compress = "none"
            zcc_include_rng = True
            zcc_include_dither = True
            zcc_bucket_hook = True
            zcc_numa_pin = False
            zcc_use_gds = False
            zcc_fault_inject = False
            zcc_recovery_mode = "auto"
            zcc_extra_tensor_attrs = ""
            zcc_retain_latest = 1

        _, optimizer = self._make_optimizer()
        wrapper = _Wrapper(optimizer)
        with mock.patch.dict("os.environ", {"MEGATRON_ENABLE_ZCC": "1"}):
            result = _maybe_enable_zero_cost_checkpoint(wrapper, Config())
        self.assertIs(result, wrapper)
        self.assertTrue(hasattr(wrapper, "zero_cost_checkpoint_manager"))
        wrapper.zero_cost_checkpoint_manager.close()

    def test_fused_state_buffer_rebases_qtensor_storage(self):
        from megatron.core.optimizer.zero_cost_checkpoint.arena import (
            FusedOptimizerStateBuffer,
        )

        param, optimizer = self._make_optimizer()
        optimizer._ensure_state_initialized(param, hparams=optimizer.param_groups[0])
        buffer = FusedOptimizerStateBuffer(optimizer).fuse()
        self.assertIsNotNone(buffer)
        data_ptr = optimizer.state[param]["exp_avg"].data.untyped_storage().data_ptr()
        base_ptr = buffer.untyped_storage().data_ptr()
        self.assertGreaterEqual(data_ptr, base_ptr)
        self.assertLess(data_ptr, base_ptr + buffer.numel())
        param.grad = torch.full_like(param, 0.5)
        optimizer.step()
        after_ptr = optimizer.state[param]["exp_avg"].data.untyped_storage().data_ptr()
        self.assertEqual(after_ptr, data_ptr)

    def test_snapshot_round_trip_file(self):
        from megatron.core.optimizer.zero_cost_checkpoint.config import (
            ZeroCostCheckpointConfig,
        )
        from megatron.core.optimizer.zero_cost_checkpoint.manager import (
            ZeroCostCheckpointManager,
        )
        from megatron.core.optimizer.zero_cost_checkpoint.recovery import (
            load_zcc_state_dict,
            restore_zcc_state,
        )

        param, optimizer = self._make_optimizer()
        wrapper = _Wrapper(optimizer)
        scheduler = _Scheduler()
        with tempfile.TemporaryDirectory() as tmp:
            config = ZeroCostCheckpointConfig(
                enabled=True,
                workers_num=2,
                flash_device=tmp,
                durable_dir=tmp,
                durable_interval=1,
                compress="zstd:1",
            )
            manager = ZeroCostCheckpointManager(wrapper, config)
            manager.sync_before_step()
            param.grad = torch.full_like(param, 0.25)
            optimizer.step()
            scheduler.value = 7
            expected_param = param.detach().clone()
            expected_exp_avg = optimizer.state[param]["exp_avg"].data.clone()
            torch.manual_seed(1234)
            rng_state = torch.get_rng_state()
            manager.snapshot_after_step(1, opt_param_scheduler=scheduler)
            manager.finalize()
            path = Path(tmp) / "step_0000001" / "rank_00000" / "zcc_snapshot.pt"
            payload = load_zcc_state_dict(str(path), mode="flash")
            self.assertEqual(payload["step"], 1)
            self.assertEqual(payload["metadata"]["opt_param_scheduler"]["value"], 7)
            names = {item["name"] for item in payload["tensors"]}
            self.assertIn("optimizer0.group0.param0.data", names)
            self.assertIn("optimizer0.fused_state", names)
            self.assertNotIn("optimizer0.group0.param0.state.exp_avg._data", names)
            durable_path = Path(tmp) / "step_0000001" / "rank_00000" / "zcc_durable.pt"
            durable_payload = load_zcc_state_dict(str(durable_path), mode="durable")
            self.assertEqual(durable_payload["step"], 1)
            param.data.zero_()
            optimizer.state[param]["exp_avg"].set_data(torch.zeros_like(param))
            scheduler.value = 0
            torch.manual_seed(5678)
            missing, unexpected = restore_zcc_state(
                wrapper,
                payload,
                opt_param_scheduler=scheduler,
            )
            self.assertEqual(missing, [])
            self.assertEqual(unexpected, [])
            self.assertTrue(torch.equal(param, expected_param))
            self.assertTrue(
                torch.equal(optimizer.state[param]["exp_avg"].data, expected_exp_avg)
            )
            self.assertEqual(scheduler.value, 7)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_state))
            manager.close()

    def test_auto_falls_back_to_durable_after_flash_corruption(self):
        from megatron.core.optimizer.zero_cost_checkpoint.config import (
            ZeroCostCheckpointConfig,
        )
        from megatron.core.optimizer.zero_cost_checkpoint.manager import (
            ZeroCostCheckpointManager,
        )
        from megatron.core.optimizer.zero_cost_checkpoint.recovery import (
            load_zcc_state_dict,
        )

        param, optimizer = self._make_optimizer()
        wrapper = _Wrapper(optimizer)
        with tempfile.TemporaryDirectory() as tmp:
            flash_dir = Path(tmp) / "flash"
            durable_dir = Path(tmp) / "durable"
            config = ZeroCostCheckpointConfig(
                enabled=True,
                flash_device=str(flash_dir),
                durable_dir=str(durable_dir),
                durable_interval=1,
                compress="none",
            )
            manager = ZeroCostCheckpointManager(wrapper, config)
            manager.sync_before_step()
            param.grad = torch.full_like(param, 0.125)
            optimizer.step()
            manager.snapshot_after_step(3)
            manager.finalize()
            rank_dir = flash_dir / "step_0000003" / "rank_00000"
            with open(rank_dir / "zcc_snapshot.pt", "r+b") as handle:
                handle.seek(12)
                handle.write(b"corrupt")
            payload = load_zcc_state_dict(
                str(rank_dir),
                mode="auto",
                durable_dir=str(durable_dir),
            )
            self.assertEqual(payload["step"], 3)
            manager.close()

    def test_planner_keeps_shared_storage_parameter_views(self):
        from megatron.core.optimizer.zero_cost_checkpoint.snapshot_spec import (
            SnapshotPlanner,
        )

        base = torch.arange(8, dtype=torch.float32)
        param0 = torch.nn.Parameter(base[:4])
        param1 = torch.nn.Parameter(base[4:])
        optimizer = torch.optim.SGD([param0, param1], lr=1e-3)

        names = {spec.name for spec in SnapshotPlanner().plan(optimizer)}

        self.assertIn("optimizer0.group0.param0.data", names)
        self.assertIn("optimizer0.group0.param1.data", names)

    def test_extra_tensor_attrs_snapshot_quantization_side_tensor(self):
        from megatron.core.optimizer.zero_cost_checkpoint.config import (
            ZeroCostCheckpointConfig,
        )
        from megatron.core.optimizer.zero_cost_checkpoint.manager import (
            ZeroCostCheckpointManager,
        )
        from megatron.core.optimizer.zero_cost_checkpoint.recovery import (
            load_zcc_state_dict,
            restore_zcc_state,
        )

        param, optimizer = self._make_optimizer()
        param._fp8_weight_cache = torch.arange(8, dtype=torch.uint8)
        wrapper = _Wrapper(optimizer)
        with tempfile.TemporaryDirectory() as tmp:
            config = ZeroCostCheckpointConfig(
                enabled=True,
                flash_device=tmp,
                compress="none",
                extra_tensor_attrs=("_fp8_weight_cache",),
            )
            manager = ZeroCostCheckpointManager(wrapper, config)
            manager.sync_before_step()
            param.grad = torch.full_like(param, 0.0625)
            optimizer.step()
            manager.snapshot_after_step(5)
            path = Path(tmp) / "step_0000005" / "rank_00000" / "zcc_snapshot.pt"
            payload = load_zcc_state_dict(str(path), mode="flash")
            tensors = {item["name"]: item for item in payload["tensors"]}
            self.assertIn("optimizer0.group0.param0._fp8_weight_cache", tensors)
            self.assertEqual(
                tensors["optimizer0.group0.param0._fp8_weight_cache"]["dtype"],
                "torch.uint8",
            )

            param._fp8_weight_cache.fill_(0)
            missing, unexpected = restore_zcc_state(
                wrapper,
                payload,
                extra_tensor_attrs=("_fp8_weight_cache",),
            )
            self.assertEqual(missing, [])
            self.assertEqual(unexpected, [])
            self.assertTrue(
                torch.equal(
                    param._fp8_weight_cache,
                    torch.arange(8, dtype=torch.uint8),
                )
            )
            manager.close()

    def test_retain_latest_prunes_old_step_directories(self):
        from megatron.core.optimizer.zero_cost_checkpoint.config import (
            ZeroCostCheckpointConfig,
        )
        from megatron.core.optimizer.zero_cost_checkpoint.manager import (
            ZeroCostCheckpointManager,
        )

        param, optimizer = self._make_optimizer()
        wrapper = _Wrapper(optimizer)
        with tempfile.TemporaryDirectory() as tmp:
            config = ZeroCostCheckpointConfig(
                enabled=True,
                flash_device=tmp,
                durable_interval=1,
                compress="none",
                retain_latest=1,
            )
            manager = ZeroCostCheckpointManager(wrapper, config)
            manager.sync_before_step()
            for step in (1, 2):
                param.grad = torch.full_like(param, 0.03125)
                optimizer.step()
                manager.snapshot_after_step(step)

            self.assertFalse((Path(tmp) / "step_0000001").exists())
            self.assertTrue((Path(tmp) / "step_0000002").exists())
            manager.close()


if __name__ == "__main__":
    unittest.main()
