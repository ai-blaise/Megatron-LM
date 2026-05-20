#!/usr/bin/env python3
"""Sanity-check the Corsaire/Megatron AWS runtime environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def _import(module: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
        return True, getattr(mod, "__file__", "<namespace>")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        return False, repr(exc)


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out.strip()
    except subprocess.CalledProcessError as exc:
        return exc.returncode, exc.output.strip()
    except FileNotFoundError as exc:
        return 127, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--expected-sm-major", type=int, default=10)
    parser.add_argument("--check-uv-sync", action="store_true")
    args = parser.parse_args()

    checks: list[Check] = []
    repo_dir = Path(__file__).resolve().parents[3]
    venv_cuda = repo_dir / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"
    if not os.environ.get("CUDA_HOME") and (venv_cuda / "bin" / "nvcc").exists():
        os.environ["CUDA_HOME"] = str(venv_cuda)
        os.environ.setdefault("CUDA_PATH", str(venv_cuda))
        os.environ["PATH"] = str(venv_cuda / "bin") + os.pathsep + os.environ.get("PATH", "")
        os.environ["LD_LIBRARY_PATH"] = (
            str(venv_cuda / "lib64")
            + os.pathsep
            + str(venv_cuda / "lib")
            + os.pathsep
            + os.environ.get("LD_LIBRARY_PATH", "")
        )

    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks.append(Check("python", py == "3.12", sys.version.replace("\n", " ")))

    for module in [
        "torch",
        "transformer_engine",
        "triton",
        "deep_ep",
        "deep_gemm",
        "torchcomms",
        "nvidia.nvshmem",
        "cutlass",
        "wandb",
        "datasets",
    ]:
        ok, detail = _import(module)
        checks.append(Check(f"import {module}", ok, detail))

    print("Package versions:")
    for dist in [
        "torch",
        "triton",
        "transformer-engine",
        "transformer-engine-torch",
        "deep_ep",
        "deep_gemm",
        "torchcomms",
        "nvidia-nccl-cu13",
        "nvidia-nvshmem-cu13",
        "nvidia-cutlass-dsl",
        "nvidia-mathdx",
        "wandb",
        "datasets",
    ]:
        print(f"  {dist}: {_version(dist)}")
    print()

    torch_ok = False
    try:
        import torch

        torch_ok = True
        count = torch.cuda.device_count()
        checks.append(
            Check(
                "cuda device count",
                count == args.expected_gpus,
                f"found={count} expected={args.expected_gpus}",
            )
        )
        if count:
            caps = [torch.cuda.get_device_capability(i) for i in range(count)]
            checks.append(
                Check(
                    "cuda sm capability",
                    all(cap[0] >= args.expected_sm_major for cap in caps),
                    str(caps),
                )
            )
            checks.append(Check("torch cuda version", True, str(torch.version.cuda)))
            checks.append(Check("current gpu", True, torch.cuda.get_device_name(0)))
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        checks.append(Check("torch cuda check", False, repr(exc)))

    cuda_home = os.environ.get("CUDA_HOME")
    checks.append(Check("CUDA_HOME available", bool(cuda_home), str(cuda_home)))
    if cuda_home:
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        rc, out = _run([nvcc, "--version"])
        checks.append(Check("nvcc", rc == 0, out.splitlines()[-1] if out else ""))

    if args.check_uv_sync:
        rc, out = _run(["uv", "sync", "--locked", "--extra", "dev", "--extra", "mlm", "--check"])
        checks.append(Check("uv sync locked check", rc == 0, out[-2000:]))

    print("Checks:")
    failed = 0
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        print(f"  [{status}] {check.name}: {check.detail}")
        failed += 0 if check.ok else 1

    print()
    print("Important env:")
    for key in [
        "CUDA_HOME",
        "CUDA_PATH",
        "LD_LIBRARY_PATH",
        "TORCH_CUDA_ARCH_LIST",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "FI_PROVIDER",
        "FI_EFA_USE_DEVICE_RDMA",
        "NVSHMEM_DIR",
        "TRITON_CACHE_DIR",
        "DG_JIT_CACHE_DIR",
    ]:
        print(f"  {key}={os.environ.get(key, '')}")

    if torch_ok and failed == 0:
        print("\nEnvironment looks ready for extension prebuild smoke.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
