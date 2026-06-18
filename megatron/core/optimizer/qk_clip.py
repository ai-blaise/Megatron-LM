# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core import parallel_state
from megatron.core.utils import get_attr_wrapped_model


def _iter_decoder_layers(model_chunk):
    """Yield decoder layers from a potentially wrapped model chunk."""
    try:
        chunk_with_decoder = get_attr_wrapped_model(
            model_chunk, 'decoder', allow_none=False, return_model_obj=True
        )
    except RuntimeError:
        return

    decoder = getattr(chunk_with_decoder, 'decoder', None)
    if decoder is None:
        return

    layers = getattr(decoder, 'layers', None)
    if layers is None:
        return

    yield from layers


def clip_qk(model, log_max_only=False) -> float:
    """
    Clip the QK attention logits to the threshold, recommended for Muon optimizer.

    Args:
        model: The model to clip the QK attention logits, a list of model chunks.
        log_only: Whether to only log the max attention logit, without updating the weights.

    Returns:
        The maximum attention logit, a float.
    """

    with torch.no_grad():
        log_max_attention_logit = 0
        for model_chunk in model:
            for transformer_layer in _iter_decoder_layers(model_chunk):
                if hasattr(transformer_layer.self_attention, 'clip_qk'):
                    current_max_attn_logits = getattr(
                        transformer_layer.self_attention.core_attention,
                        'current_max_attn_logits',
                        None,
                    )
                    if current_max_attn_logits is None:
                        continue
                    torch.distributed.all_reduce(
                        current_max_attn_logits,
                        op=torch.distributed.ReduceOp.MAX,
                        group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                    )
                    log_max_attention_logit = max(
                        log_max_attention_logit,
                        torch.max(current_max_attn_logits).item(),
                    )
                    if not log_max_only:
                        transformer_layer.self_attention.clip_qk()

    return log_max_attention_logit
