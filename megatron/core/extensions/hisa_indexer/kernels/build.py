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


def _build_ext():
    from torch.utils.cpp_extension import load

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-std=c++17"]
    if "10.0" in arch_list or "100" in arch_list:
        extra_cuda_cflags.append("-DHISA_INDEXER_BLACKWELL=1")
    extra_cflags = ["-O3", "-std=c++17"]
    extra_include_paths = [str(_CSRC)]

    sources = [
        str(_CSRC / "pybind.cpp"),
        str(_CSRC / "hisa_score_bwd.cu"),
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
