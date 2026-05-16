# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""JIT build glue for the IndexCache CUDA kernels.

Mirrors ``megatron/core/quantization/turboquant/kernels/build.py``. The
extension is compiled lazily at first import on a CUDA host. Subsequent
imports hit ``torch.utils.cpp_extension``'s cache for zero overhead.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_CSRC = _THIS_DIR / "csrc"
_REPO_ROOT = _THIS_DIR.parents[4]


def _ensure_launcher_cuda_build_env() -> None:
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


def _build_ext():
    _ensure_launcher_cuda_build_env()

    import torch.utils.cpp_extension as cpp_extension

    if os.environ.get("CUDA_HOME") and cpp_extension.CUDA_HOME is None:
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]
    load = cpp_extension.load

    extra_cuda_cflags = ["-O3", "--use_fast_math", "-std=c++17"]
    extra_cflags = ["-O3", "-std=c++17"]
    extra_include_paths = [str(_CSRC)]

    sources = [
        str(_CSRC / "pybind.cpp"),
        str(_CSRC / "indexcache_fwd.cu"),
        str(_CSRC / "indexcache_bwd.cu"),
        str(_CSRC / "indexcache_nvfp4_fwd.cu"),
        str(_CSRC / "indexcache_nvfp4_bwd.cu"),
    ]

    return load(
        name="megatron_indexcache_kv",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
        verbose=os.environ.get("MEGATRON_INDEXCACHE_VERBOSE", "0") == "1",
    )


_EXT = None


def get_ext():
    global _EXT
    if _EXT is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "IndexCache CUDA kernels require CUDA. Use the reference "
                "implementation in ``reference.py`` for CPU-only paths."
            )
        _EXT = _build_ext()
    return _EXT
