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


def _build_ext():
    from torch.utils.cpp_extension import load

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
