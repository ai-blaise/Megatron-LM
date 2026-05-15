# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""JIT build glue for the HIGGS dense 2-bit CUDA kernels.

Mirrors ``megatron/core/quantization/turboquant/kernels/build.py``: the
extension is compiled lazily at first import on a CUDA host and cached in
``torch.utils.cpp_extension``'s build cache so subsequent imports are
zero-cost.

When ``MEGATRON_HIGGS_IKP=1`` is set the build links against the IKP
runtime header (https://github.com/yao-jz/intra-kernel-profiler) so the
named kernel regions populate the IKP Explorer.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


_THIS_DIR = Path(__file__).resolve().parent
_CSRC = _THIS_DIR / "csrc"


def _build_ext():
    from torch.utils.cpp_extension import load

    extra_cuda_cflags = ["-O3", "--use_fast_math", "-std=c++17"]
    extra_cflags = ["-O3", "-std=c++17"]
    extra_include_paths = [str(_CSRC)]

    if os.environ.get("MEGATRON_HIGGS_IKP", "0") == "1":
        ikp_root = os.environ.get("IKP_ROOT")
        if ikp_root is None:
            raise RuntimeError(
                "MEGATRON_HIGGS_IKP=1 set but IKP_ROOT is not. "
                "Point IKP_ROOT at the cloned intra-kernel-profiler repo."
            )
        extra_cuda_cflags.append("-DIKP_ENABLED")
        extra_include_paths.append(str(Path(ikp_root) / "include"))

    sources = [
        str(_CSRC / "pybind.cpp"),
        str(_CSRC / "higgs_kv_fwd.cu"),
        str(_CSRC / "higgs_kv_bwd.cu"),
    ]

    return load(
        name="megatron_higgs_kv",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
        verbose=os.environ.get("MEGATRON_HIGGS_VERBOSE", "0") == "1",
    )


_EXT = None


def get_ext():
    """Return the lazily-built extension module."""

    global _EXT
    if _EXT is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "HIGGS CUDA kernels require CUDA. Use the reference "
                "implementation in ``reference.py`` for CPU-only paths."
            )
        _EXT = _build_ext()
    return _EXT
