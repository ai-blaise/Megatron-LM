# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from contextlib import nullcontext

import pytest
import torch

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.common.model_chunk_schedule_plan import TransformerLayerSchedulePlan
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    DummyState,
    build_data,
    compare_captures,
    deterministic_mode,
    get_test_config,
    get_valid_fp8_flags,
    get_valid_token_dispatcher_types,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils


def run_transformer_layer_ref_with_capture(model, input_tensors, iterations):
    """
    Runs the model in reference mode and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each iteration.
        iterations: Number of iterations to run the model.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """
    transformer_layer = model.decoder.layers[0]
    output_tensors = []
    for i in range(iterations):
        output = transformer_layer(input_tensors[i].clone())[0]
        output_tensors.append(output)
        output.backward(torch.ones_like(output))

    capture = {"outputs": output_tensors}
    for name, param in transformer_layer.named_parameters():
        capture[name] = param.grad

    return capture


def run_transformer_layer_a2a_overlap_with_capture(model, input_tensors, microbatches):
    """
    Runs the model with all-to-all overlap optimization and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each microbatch.
        microbatches: Number of microbatches to process.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """
    transformer_layer = model.decoder.layers[0]
    for i in range(len(input_tensors)):
        input_tensors[i] = input_tensors[i].clone()

    event = torch.cuda.Event()
    comp_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream(device="cuda")
    state = DummyState()
    state.is_mtp = False
    state.model = model
    layers = [
        TransformerLayerSchedulePlan(
            transformer_layer,
            event,
            state,
            comp_stream,
            comm_stream,
            extra_args={"is_moe": True, "enable_deepep": False},
        )
        for _ in range(microbatches)
    ]
    output_tensors = []

    # forward for 1st microbatch
    output, _ = TransformerLayerSchedulePlan.run(
        layers[0], None, f_input=input_tensors[0], b_grad=None
    )
    output_tensors.append(output)
    torch.cuda.synchronize()
    # overlapped forward and backward
    for i in range(1, microbatches):
        f_input, b_grad = TransformerLayerSchedulePlan.run(
            layers[i], layers[i - 1], f_input=input_tensors[i], b_grad=torch.ones_like(output)
        )
        output_tensors.append(f_input)
        torch.cuda.synchronize()
    # backward for last microbatch
    TransformerLayerSchedulePlan.run(None, layers[-1], f_input=None, b_grad=torch.ones_like(output))
    torch.cuda.synchronize()
    capture = {"outputs": output_tensors}
    for name, param in transformer_layer.named_parameters():
        capture[name] = param.grad

    return capture


def assert_delayed_expert_wgrad_capture_close(capture_ref, capture_a2a_overlap):
    """
    Compare delayed expert-wgrad captures.

    Delayed TE GroupedLinear wgrad computes and accumulates BF16 expert gradients
    outside the normal autograd edge, so the final expert-weight accumulation can
    differ by a small BF16 rounding quantum. Outputs and non-expert tensors still
    require exact equality.
    """

    for name, value in capture_ref.items():
        assert name in capture_a2a_overlap, f"gradient name mismatch, '{name}' missing"
        other = capture_a2a_overlap[name]
        assert type(value) is type(other), f"{name}: value type mismatch"
        if value is None:
            continue
        if isinstance(value, list):
            assert len(value) == len(other), f"{name}: outputs length mismatch"
            for idx, (ref_tensor, overlap_tensor) in enumerate(zip(value, other)):
                torch.testing.assert_close(
                    overlap_tensor, ref_tensor, rtol=0.0, atol=0.0, msg=f"{name}[{idx}]"
                )
        elif isinstance(value, torch.Tensor):
            if name.startswith("mlp.experts.") and ".weight" in name:
                diff = (other.float() - value.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                assert max_diff <= 5.0, f"{name}: max BF16 delayed-wgrad diff {max_diff}"
                assert mean_diff <= 1.0, f"{name}: mean BF16 delayed-wgrad diff {mean_diff}"
            else:
                torch.testing.assert_close(other, value, rtol=0.0, atol=0.0, msg=name)
        else:
            raise AssertionError(f"{name}: unsupported value type {type(value)}")


def run_mtp_layer_ref_with_capture(
    model,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    attention_mask,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    microbatches,
):
    """
    Runs the model in reference mode and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each iteration.
        iterations: Number of iterations to run the model.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """
    mtp_block = model.mtp

    output_tensors = []
    for i in range(microbatches):
        output = mtp_block(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states[i].clone(),
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            embedding=model.embedding,
        )
        output_tensors.append(output)
        output.backward(torch.ones_like(output))

    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


def run_mtp_layer_a2a_overlap_with_capture(
    model,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    attention_mask,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    microbatches,
):
    """
    Runs the model with all-to-all overlap optimization and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each microbatch.
        microbatches: Number of microbatches to process.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """
    for i in range(len(hidden_states)):
        hidden_states[i] = hidden_states[i].clone()

    comp_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream(device="cuda")
    layers = []
    for _ in range(microbatches):
        state = DummyState()
        state.mtp_labels = labels
        state.input_ids = input_ids
        state.position_ids = position_ids
        state.attention_mask = attention_mask
        state.rotary_pos_emb = rotary_pos_emb
        state.rotary_pos_cos = rotary_pos_cos
        state.rotary_pos_sin = rotary_pos_sin
        state.model = model
        state.is_mtp = True
        event = torch.cuda.Event()
        layers.append(
            TransformerLayerSchedulePlan(
                model.mtp.layers[0],
                event,
                state,
                comp_stream,
                comm_stream,
                extra_args={
                    "is_moe": True,
                    "enable_deepep": False,
                    "is_first_layer": True,
                    "is_last_layer": True,
                },
            )
        )
    output_tensors = []
    # forward for 1st microbatch
    f_input, _ = TransformerLayerSchedulePlan.run(
        layers[0], None, f_input=hidden_states[0], b_grad=None
    )
    output_tensors.append(f_input)
    torch.cuda.synchronize()
    # overlapped forward and backward
    for i in range(1, microbatches):
        f_input, b_grad = TransformerLayerSchedulePlan.run(
            layers[i], layers[i - 1], f_input=hidden_states[i], b_grad=torch.ones_like(f_input)
        )
        output_tensors.append(f_input)
        torch.cuda.synchronize()
    # backward for last microbatch
    TransformerLayerSchedulePlan.run(
        None, layers[-1], f_input=None, b_grad=torch.ones_like(f_input)
    )
    torch.cuda.synchronize()
    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


class TestA2AOverlap:
    """
    Test class for all-to-all overlap optimization in transformer models.

    This class contains tests to verify that the all-to-all overlap optimization
    produces the same results as the reference implementation.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    def test_transformer_layer_overlap_dense(self):
        """
        Verifies all-to-all overlap optimization in dense transformer layer produces
        the same results as the reference implementation.
        """
        extra_kwargs = {"moe_token_dispatcher_type": "alltoall"}
        config = get_test_config(num_moe_experts=None, extra_kwargs=extra_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=config, use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(config, 0) if config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(
                    gpt_model, input_tensors, microbatches
                )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                gpt_model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    def test_transformer_layer_overlap_shared_expert(self):
        """
        Verifies all-to-all overlap optimization in transformer layer with shared expert produces
        the same results as the reference implement
        ation.
        """
        extra_kwargs = {
            "moe_token_dispatcher_type": "alltoall",
            "moe_shared_expert_intermediate_size": 512,
        }
        overlap_config = get_test_config(extra_kwargs=extra_kwargs)
        extra_kwargs["moe_shared_expert_overlap"] = False
        ref_config = get_test_config(extra_kwargs=extra_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=ref_config, use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=ref_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(ref_config, 0) if ref_config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(
                    gpt_model, input_tensors, microbatches
                )
            del gpt_model

            gpt_model = GPTModel(
                config=overlap_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                gpt_model, input_tensors, microbatches
            )
            assert_delayed_expert_wgrad_capture_close(capture_ref, capture_a2a_overlap)

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    def test_transformer_layer_overlap_early_attn_memory_release(self):
        """
        Verifies all-to-all overlap optimization in transformer layer with early attn memory release
        produces the same results as the reference implementation.
        """
        extra_kwargs = {
            "moe_token_dispatcher_type": "alltoall",
            "ep_overlap_early_attn_memory_release": True,
            "overlap_moe_expert_parallel_comm": True,
        }
        overlap_config = get_test_config(extra_kwargs=extra_kwargs)
        ref_config = get_test_config(extra_kwargs=extra_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=ref_config, use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=ref_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(ref_config, 0) if ref_config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(
                    gpt_model, input_tensors, microbatches
                )
            del gpt_model

            gpt_model = GPTModel(
                config=overlap_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                gpt_model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

    @pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
    @pytest.mark.xfail(
        reason=(
            "Known unsafe optional path: TE delayed GroupedLinear expert wgrad with "
            "non-fused accumulation no longer drops whole microbatches, but still shows "
            "BF16 drift and intermittent NaNs for the delayed dispatch-backward overlap mode."
        ),
        strict=True,
    )
    def test_transformer_layer_overlap_dispatch_backward_with_experts_wgrad(self):
        """
        Verifies delayed expert-wgrad overlap produces the same layer outputs and gradients as
        the reference MoE layer path.
        """
        ref_kwargs = {"moe_token_dispatcher_type": "alltoall"}
        overlap_kwargs = {
            **ref_kwargs,
            "overlap_dispatch_backward_with_experts_wgrad": True,
        }
        ref_config = get_test_config(extra_kwargs=ref_kwargs)
        overlap_config = get_test_config(extra_kwargs=overlap_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=ref_config, use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=ref_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(ref_config, 0) if ref_config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(
                    gpt_model, input_tensors, microbatches
                )
            del gpt_model

            gpt_model = GPTModel(
                config=overlap_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                gpt_model, input_tensors, microbatches
            )
            assert_delayed_expert_wgrad_capture_close(capture_ref, capture_a2a_overlap)

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    def test_transformer_layer_overlap(self, dispatcher_type, fp8_flag):
        """
        Verifies all-to-all overlap optimization in transformer layer produces
        the same results as the reference implementation.
        """

        extra_kwargs = {"moe_token_dispatcher_type": dispatcher_type}
        if dispatcher_type == "flex":
            extra_kwargs["moe_flex_dispatcher_backend"] = "deepep"
            extra_kwargs["moe_router_dtype"] = "fp32"
        if fp8_flag is not None:
            extra_kwargs["fp8"] = fp8_flag[0]
            extra_kwargs["fp8_recipe"] = fp8_flag[1]
        config = get_test_config(extra_kwargs=extra_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=config, use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(config, 0) if config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(
                    gpt_model, input_tensors, microbatches
                )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                gpt_model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    def test_mtp_layer_overlap(self, dispatcher_type, fp8_flag):
        """
        Verifies all-to-all overlap optimization in MTP layer produces
        the same results as the reference implementation.
        """

        extra_kwargs = {
            "moe_token_dispatcher_type": dispatcher_type,
            "mtp_num_layers": 1,
            "mtp_loss_scaling_factor": 1.1,
        }
        if dispatcher_type == "flex":
            extra_kwargs["moe_flex_dispatcher_backend"] = "deepep"
            extra_kwargs["moe_router_dtype"] = "fp32"
        if fp8_flag is not None:
            extra_kwargs["fp8_recipe"] = fp8_flag[1]
            extra_kwargs["fp8"] = fp8_flag[0]
        config = get_test_config(extra_kwargs=extra_kwargs)
        microbatches = 1
        seq_len = 32
        with deterministic_mode():
            # init models
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=16,
                moe_grouped_gemm=True,
                qk_layernorm=True,
                multi_latent_attention=True,
            )
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, True)
            if mtp_block_spec is None:
                # only last rank has mtp block
                assert True
                return
            gpt_model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                mtp_block_spec=mtp_block_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            gpt_model.decoder.final_layernorm = None
            gpt_model.cuda()
            params = reset_model(gpt_model)

            # build input data
            data = list(range(seq_len))
            hidden_states = [build_data(seq_len) for _ in range(microbatches)]
            input_ids = torch.tensor(data, dtype=torch.int64).repeat((1, 1)).cuda()
            labels = torch.tensor(data, dtype=torch.int64).repeat((1, 1)).cuda()
            position_ids = torch.tensor(data, dtype=torch.int64).repeat((1, 1)).cuda()
            attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool).cuda()
            # get rotary pos emb
            _, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, _, _padding_mask = (
                gpt_model._preprocess(input_ids, position_ids)
            )
            # reset model
            params = reset_model(gpt_model)

            # run reference implementation
            capture_ref = run_mtp_layer_ref_with_capture(
                model=gpt_model,
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                microbatches=microbatches,
            )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_mtp_layer_a2a_overlap_with_capture(
                model=gpt_model,
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                microbatches=microbatches,
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"
