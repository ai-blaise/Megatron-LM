# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA kernels for the 2-bit HIGGS dense MLA-KV fake-quant.

The forward kernel ports the SGLang store kernel
(``optimization-playground/python/sglang/jit_kernel/csrc/quantization/
higgs_dense_2bit_kv.cuh``) into a Megatron-style fp-quant-dequant op that
materialises the dequantized fake-quant output and saves the intermediates
the backward needs (codebook indices + per-token scale + STE mask). The
backward kernel implements the closed-form STE derivation in
``../reference.py`` (``higgs_backward``).
"""
