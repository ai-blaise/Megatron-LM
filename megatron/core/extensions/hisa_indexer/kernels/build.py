# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""JIT build glue for the HISA backward CUDA kernels.

Mirrors ``megatron/core/quantization/indexcache/kernels/build.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


_THIS_DIR = Path(__file__).resolve().parent
_CSRC = _THIS_DIR / "csrc"
_REPO_ROOT = _THIS_DIR.parents[4]
_VENV_SITE_PACKAGES = _REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
_MATHDX_INCLUDE = _VENV_SITE_PACKAGES / "nvidia" / "mathdx" / "include"
_CUTLASS_INCLUDE = _VENV_SITE_PACKAGES / "nvidia" / "mathdx" / "external" / "cutlass" / "include"


def _ensure_cuda_build_env() -> None:
    """Mirror the CUDA/toolchain env exported by the SFT launcher."""

    launcher_cuda = _REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"
    if not os.environ.get("CUDA_HOME") and (launcher_cuda / "bin" / "nvcc").exists():
        os.environ["CUDA_HOME"] = str(launcher_cuda)

    if os.environ.get("CUDA_HOME"):
        cuda_home = Path(os.environ["CUDA_HOME"])
        os.environ.setdefault("CUDA_PATH", str(cuda_home))
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        cuda_bin = str(cuda_home / "bin")
        if cuda_bin not in path_parts:
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
        ld_parts = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        for lib_dir in (cuda_home / "lib", cuda_home / "lib64"):
            lib_str = str(lib_dir)
            if lib_dir.exists() and lib_str not in ld_parts:
                os.environ["LD_LIBRARY_PATH"] = lib_str + os.pathsep + os.environ.get(
                    "LD_LIBRARY_PATH", ""
                )

    if not os.environ.get("CC") and Path("/usr/bin/gcc").exists():
        os.environ["CC"] = "/usr/bin/gcc"
    if not os.environ.get("CXX") and Path("/usr/bin/g++").exists():
        os.environ["CXX"] = "/usr/bin/g++"

    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 10:
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


def _build_ext():
    _ensure_cuda_build_env()

    import torch.utils.cpp_extension as cpp_extension

    if os.environ.get("CUDA_HOME") and cpp_extension.CUDA_HOME is None:
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]
    load = cpp_extension.load

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-gencode=arch=compute_100,code=sm_100",
        "-DFLASHINFER_ENABLE_BF16",
        "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
    if "10.0" in arch_list or "100" in arch_list:
        extra_cuda_cflags.append("-DHISA_INDEXER_BLACKWELL=1")
    extra_cflags = ["-O3", "-std=c++20"]
    extra_include_paths = [str(_CSRC)]
    if _MATHDX_INCLUDE.exists():
        extra_include_paths.append(str(_MATHDX_INCLUDE))
    if _CUTLASS_INCLUDE.exists():
        extra_include_paths.append(str(_CUTLASS_INCLUDE))

    sources = [
        str(_CSRC / "pybind.cpp"),
        str(_CSRC / "hisa_selector_fwd.cu"),
        str(_CSRC / "hisa_score_bwd.cu"),
        str(_CSRC / "hisa_selected_score_bwd.cu"),
        str(_CSRC / "dsa_sparse_kv_bwd.cu"),
    ]
    fused_path = _CSRC / "hisa_score_bwd_fused.cu"
    if fused_path.exists():
        sources.append(str(fused_path))

    return load(
        name="megatron_hisa_indexer",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
        verbose=os.environ.get("MEGATRON_HISA_VERBOSE", "0") == "1",
    )


_EXT = None


def get_ext():
    global _EXT
    if _EXT is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "HISA backward CUDA kernels require CUDA. Use the reference "
                "implementation in ``reference.py`` for CPU-only paths."
            )
        _EXT = _build_ext()
    return _EXT
