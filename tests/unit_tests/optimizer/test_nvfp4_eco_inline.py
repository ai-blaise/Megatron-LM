# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for NVFP4 kernels")
def test_inline_nvfp4_eco_decode_matches_te_dequantized_shard():
    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    from megatron.core.optimizer.flash_optimizers import _fused_eco_inject_from_nvfp4_rowwise

    torch.manual_seed(123)
    src = (torch.randn((128, 64), device="cuda", dtype=torch.float32) * 0.25).contiguous()
    quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=False,
        with_2d_quantization=True,
    )
    nvfp4_param = quantizer.make_empty(src.shape, dtype=torch.float32, device=torch.device("cuda"))
    nvfp4_param.quantize_(src)

    # Use an odd offset to cover both low- and high-nibble decode paths.
    shard_offset = 17
    shard_size = 4096
    pre_cast = src.view(-1)[shard_offset : shard_offset + shard_size].contiguous()
    dense_post = nvfp4_param.dequantize(dtype=torch.float32).view(-1)[
        shard_offset : shard_offset + shard_size
    ]

    mom = torch.zeros(shard_size, device="cuda", dtype=torch.float32)
    var = torch.ones(shard_size, device="cuda", dtype=torch.float32)
    dummy_scales = torch.empty(1, device="cuda", dtype=torch.float16)

    _fused_eco_inject_from_nvfp4_rowwise(
        mom=mom,
        mom_scales_f16=dummy_scales,
        var=var,
        var_scales_f16=dummy_scales,
        model_param=nvfp4_param,
        pre_cast=pre_cast,
        shard_offset=shard_offset,
        eco_scalar=1.0,
        eps=0.0,
        bc2=1.0,
        quantize_optim_states=False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(mom, pre_cast - dense_post, rtol=0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for NVFP4 kernels")
def test_inline_nvfp4_eco_quantized_state_matches_dense_path():
    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    from megatron.core.optimizer.flash_optimizers import (
        _fused_eco_inject,
        _fused_eco_inject_from_nvfp4_rowwise,
    )

    torch.manual_seed(456)
    src = (torch.randn((128, 64), device="cuda", dtype=torch.float32) * 0.25).contiguous()
    quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=False,
        with_2d_quantization=True,
    )
    nvfp4_param = quantizer.make_empty(src.shape, dtype=torch.float32, device=torch.device("cuda"))
    nvfp4_param.quantize_(src)

    shard_offset = 17
    shard_size = 4096
    pre_cast = src.view(-1)[shard_offset : shard_offset + shard_size].contiguous()
    dense_post = nvfp4_param.dequantize(dtype=torch.float32).view(-1)[
        shard_offset : shard_offset + shard_size
    ].contiguous()

    mom = torch.randint(-127, 127, (shard_size,), device="cuda", dtype=torch.int8)
    var = torch.randint(0, 255, (shard_size,), device="cuda", dtype=torch.uint8)
    mom_scales = torch.full((shard_size // 32,), 1e-3, device="cuda", dtype=torch.float16)
    var_scales = torch.full((shard_size // 32,), 1e-3, device="cuda", dtype=torch.float16)

    dense_mom = mom.clone()
    dense_mom_scales = mom_scales.clone()
    _fused_eco_inject(
        mom=dense_mom,
        mom_scales_f16=dense_mom_scales,
        var=var,
        var_scales_f16=var_scales,
        pre_cast=pre_cast,
        post_cast=dense_post,
        eco_scalar=0.7,
        eps=1e-8,
        bc2=0.999,
        quantize_optim_states=True,
    )

    inline_mom = mom.clone()
    inline_mom_scales = mom_scales.clone()
    _fused_eco_inject_from_nvfp4_rowwise(
        mom=inline_mom,
        mom_scales_f16=inline_mom_scales,
        var=var,
        var_scales_f16=var_scales,
        model_param=nvfp4_param,
        pre_cast=pre_cast,
        shard_offset=shard_offset,
        eco_scalar=0.7,
        eps=1e-8,
        bc2=0.999,
        quantize_optim_states=True,
    )
    torch.cuda.synchronize()

    diff = (inline_mom.to(torch.int32) - dense_mom.to(torch.int32)).abs()
    assert diff.max().item() <= 1
    torch.testing.assert_close(inline_mom_scales, dense_mom_scales, rtol=1e-3, atol=1e-6)
