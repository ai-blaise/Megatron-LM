# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-PyTorch reference forward and backward for the 2-bit HIGGS dense MLA-KV
fake-quant.

The math here mirrors the SGLang reference in
``optimization-playground/python/sglang/srt/layers/quantization/higgs_dense_2bit_kv.py``
(commit ``2e2f51717``):

    rotated     = FWHT_512(x)                          # orthonormal
    scale       = ||rotated|| / sqrt(N)                # per-token fp16
    normalized  = rotated / scale
    indices     = argmax_i (2 * pair . G[i] - ||G[i]||^2)
    recon       = scale * G[indices]
    y           = FWHT_512(recon)                      # involutory

The backward uses a saturating straight-through estimator: the codebook step
``argmax`` is non-differentiable, so we approximate ``d recon / d normalized``
with ``mask`` (1 where the chosen codeword is the same after a small
perturbation; treated as the identity for the gradient pass, consistent with
TurboQuant's STE choice). Every other operator (FWHT and the per-token L2
scale) is differentiable and contributes its exact analytic Jacobian.

Functions in this module are written without ``torch.autograd`` so they can
be composed directly inside the ``HiggsKVFn`` autograd.Function and so the
math is visible without framework magic in the way.
"""

from __future__ import annotations

import math

import torch

from megatron.core.quantization.higgs.codec import (
    HIGGS_INV_SQRT_LATENT_DIM,
    HIGGS_LATENT_DIM,
    HIGGS_PAIR_DIM,
    HiggsBuffers,
)


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Fast Walsh-Hadamard transform on the trailing axis.

    Operates on the last axis of an n-d float tensor whose trailing size is a
    power of two. Returns the transform scaled by ``1/sqrt(n)`` so the
    operator is its own inverse (``H/sqrt(n)`` is involutory: ``H @ H == n *
    I``). Written functionally so the input is never aliased between stages
    --- the naive in-place version silently miscompiles for ``n >= 8`` after
    PyTorch normalises strides (see SGLang reference for the same fix).
    """

    n = x.shape[-1]
    if n <= 1:
        return x.clone()
    if n & (n - 1):
        raise ValueError(f"FWHT requires power-of-2 dim, got {n}")
    *batch_shape, _ = x.shape
    y = x.contiguous().view(-1, n).clone()
    h = 1
    while h < n:
        view = y.view(-1, n // (2 * h), 2, h)
        a = view[:, :, 0, :]
        b = view[:, :, 1, :]
        nxt = torch.empty_like(view)
        nxt[:, :, 0, :] = a + b
        nxt[:, :, 1, :] = a - b
        y = nxt.reshape(-1, n)
        h *= 2
    return (y / math.sqrt(n)).view(*batch_shape, n)


def _nearest_pair_index(
    normalized_pairs: torch.Tensor, buffers: HiggsBuffers
) -> torch.Tensor:
    """Map each 2-D pair to its nearest EDEN2-16 codeword index.

    Args:
      normalized_pairs: ``(N, num_pairs, 2)`` tensor.
      buffers: HIGGS buffers carrying ``codebook`` and ``codebook_norm_sq``.

    Returns:
      ``(N, num_pairs)`` ``int64`` index tensor in [0, 15].
    """

    # nearest neighbour in 2-D <=> argmax(2 * x . G^T - ||G||^2).
    scores = (
        2.0 * torch.matmul(normalized_pairs, buffers.codebook.T)
        - buffers.codebook_norm_sq
    )
    return torch.argmax(scores, dim=-1)


def higgs_forward(
    x: torch.Tensor,
    buffers: HiggsBuffers,
    *,
    return_intermediates: bool = False,
):
    """Apply the HIGGS dense 2-bit fake-quant round-trip to ``x``.

    Args:
      x: ``(N, latent_dim)`` float tensor (any floating dtype).
      buffers: codebook + norms produced by ``build_higgs_buffers``.
      return_intermediates: when True, returns a dict of internal tensors for
        the backward pass to reuse.

    Returns:
      ``(N, latent_dim)`` fake-quant output with the same dtype as ``x``.
    """

    if x.dim() != 2 or x.shape[-1] != buffers.latent_dim:
        raise ValueError(
            f"higgs_forward expects x of shape [N, {buffers.latent_dim}]; "
            f"got {tuple(x.shape)}."
        )

    orig_dtype = x.dtype
    compute_dtype = (
        x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    )
    xf = x.to(compute_dtype)
    codebook = buffers.codebook.to(compute_dtype)
    codebook_norm_sq = buffers.codebook_norm_sq.to(compute_dtype)

    # Forward orthonormal FWHT.
    rotated = fwht(xf)

    # Per-token block scale = ||rotated|| / sqrt(N). EDEN2-16 is calibrated
    # for per-coordinate inputs ~N(0, 1); after this scaling each coord of
    # ``normalized`` is approximately N(0, 1).
    rot_norm = torch.linalg.vector_norm(rotated, dim=-1).clamp_min(1e-8)
    scale = rot_norm * HIGGS_INV_SQRT_LATENT_DIM
    normalized = rotated / scale[:, None]

    # Codebook nearest-neighbour.
    pairs = normalized.reshape(
        -1, buffers.num_pairs, HIGGS_PAIR_DIM
    )
    indices = _nearest_pair_index(
        pairs, HiggsBuffers(
            latent_dim=buffers.latent_dim,
            pair_dim=buffers.pair_dim,
            codebook_size=buffers.codebook_size,
            bits_per_scalar=buffers.bits_per_scalar,
            codebook=codebook,
            codebook_norm_sq=codebook_norm_sq,
        )
    )                                          # (N, num_pairs)

    # STE mask is 1 everywhere by construction: HIGGS uses a lattice, not a
    # bounded codebook, so there is no "outside the codebook" region. We
    # still track a unit mask tensor for parity with the TurboQuant
    # interface and so future variants (e.g. clipped codebook) can switch
    # the saturation behaviour without changing the surface.
    ste_mask = torch.ones_like(normalized)

    # Codebook lookup -> rotated reconstruction (un-scaled).
    rotated_recon_unit = codebook[indices].reshape(
        normalized.shape
    )                                          # (N, latent_dim)

    # Apply per-token scale to recover the original magnitude.
    rotated_recon = rotated_recon_unit * scale[:, None]

    # Inverse FWHT -> output. FWHT/sqrt(N) is involutory.
    y = fwht(rotated_recon)
    y_out = y.to(orig_dtype)

    if return_intermediates:
        return y_out, {
            "x_compute": xf,
            "rotated": rotated,
            "rot_norm": rot_norm,
            "scale": scale,
            "normalized": normalized,
            "indices": indices,
            "rotated_recon_unit": rotated_recon_unit,
            "rotated_recon": rotated_recon,
            "ste_mask": ste_mask,
        }
    return y_out


def higgs_backward(
    grad_y: torch.Tensor,
    intermediates: dict,
    buffers: HiggsBuffers,
) -> torch.Tensor:
    """Closed-form gradient through the HIGGS fake-quant.

    Given ``dL/dy`` and the intermediates saved by ``higgs_forward``, return
    ``dL/dx``. The straight-through estimator replaces the non-differentiable
    ``argmax`` with the identity on the pair input:

        d recon_unit / d normalized = ste_mask  (1 everywhere for EDEN2-16)

    The outer FWHT, the codebook scaling, and the per-token L2-scale are
    fully differentiable. The full chain (per token; subscripts elided):

        rotated         = FWHT(x)
        rot_norm        = ||rotated||
        scale           = rot_norm / sqrt(N)
        normalized      = rotated / scale
        recon_unit      = codebook[argmax(... pairs of normalized ...)]
        rotated_recon   = scale * recon_unit
        y               = FWHT(rotated_recon)

    Reverse chain (FWHT is its own adjoint under the orthonormal 1/sqrt(N)
    normalisation):

        dL/d rotated_recon = FWHT(dL/dy)
        dL/d recon_unit    = scale * dL/d rotated_recon
        dL/d scale         = sum_j (recon_unit_j * dL/d rotated_recon_j)

    STE on codebook step:
        dL/d normalized_i  = mask_i * dL/d recon_unit_i

    ``normalized`` depends on ``rotated`` both directly (rotated/scale) and
    indirectly through ``scale = ||rotated||/sqrt(N)``. Combining both
    paths the total gradient is

        dL/d rotated_i = (1/scale) * dL/d normalized_i
                       + (∂scale/∂rotated_i) * [ dL/d scale
                                               - (1/scale^2) *
                                                  sum_j (rotated_j * dL/d normalized_j) ]

    where ``∂scale/∂rotated_i = rotated_i / (sqrt(N) * rot_norm)``. The
    bracketed term is the "implicit-through-scale" correction without which
    the gradient would only match torch.autograd in the (incorrect) limit
    where ``scale`` is treated as constant.

    Final step is the input-side FWHT adjoint:
        dL/d x = FWHT(dL/d rotated)
    """

    g = grad_y.to(torch.float32)
    rotated = intermediates["rotated"]
    rot_norm = intermediates["rot_norm"]
    scale = intermediates["scale"]
    rotated_recon_unit = intermediates["rotated_recon_unit"]
    ste_mask = intermediates["ste_mask"]

    # 1. Adjoint of outer FWHT (orthonormal => self-adjoint).
    grad_rotated_recon = fwht(g)

    # 2. Split the product rotated_recon = scale * rotated_recon_unit.
    grad_rotated_recon_unit = grad_rotated_recon * scale[:, None]
    grad_scale = (grad_rotated_recon * rotated_recon_unit).sum(dim=-1)

    # 3. STE on the codebook step.
    grad_normalized = grad_rotated_recon_unit * ste_mask

    # 4. Split rotated -> (normalized, scale).
    inv_scale = 1.0 / scale.clamp_min(1e-12)
    # First term: explicit (∂normalized/∂rotated_i, scale held fixed).
    grad_rotated_explicit = grad_normalized * inv_scale[:, None]
    # The implicit-through-scale term combines both the direct
    # (dL/d scale) and the implicit Jacobian feedback from normalised:
    #   correction = dL/d scale - (1/scale^2) * sum_j (rotated_j * dL/d normalized_j)
    proj = (rotated * grad_normalized).sum(dim=-1)
    correction = grad_scale - proj * inv_scale * inv_scale
    # d scale / d rotated_i = rotated_i / (sqrt(N) * rot_norm).
    d_scale_d_rotated_factor = (
        HIGGS_INV_SQRT_LATENT_DIM / rot_norm.clamp_min(1e-12)
    )
    grad_rotated_implicit = (correction * d_scale_d_rotated_factor)[:, None] * rotated
    grad_rotated = grad_rotated_explicit + grad_rotated_implicit

    # 5. Adjoint of input-side FWHT.
    grad_x = fwht(grad_rotated)

    return grad_x.to(grad_y.dtype)


# ---------------------------------------------------------------------------
# Reference compress / decompress helpers (parity with SGLang's codec).
# These are not on the autograd path; they let the parity test build a
# byte-exact packed tensor from the same intermediates the kernel produces.
# ---------------------------------------------------------------------------


def reference_compress(
    latent: torch.Tensor,
    rope: torch.Tensor,
    buffers: HiggsBuffers,
) -> torch.Tensor:
    """Encode ``(N, latent_dim)`` BF16 latent + ``(N, rope_dim)`` BF16 rope
    into ``(N, 1, 258)`` ``uint8`` packed slots.

    This is the SGLang ``compress`` path, replicated here so parity tests
    can compare byte-for-byte against the kernel.
    """

    from megatron.core.quantization.higgs.codec import (
        HIGGS_NORM_BYTES,
        HIGGS_ROPE_DIM,
        HIGGS_SLOT_BYTES,
        pack_higgs_2bit_indices,
    )

    n = latent.shape[0]
    flat = latent.reshape(n, buffers.latent_dim).to(torch.float32)
    rotated = fwht(flat)
    scale_f = (
        torch.linalg.vector_norm(rotated, dim=-1).clamp_min(1e-8)
        * HIGGS_INV_SQRT_LATENT_DIM
    )
    normalized = rotated / scale_f[:, None]
    pairs = normalized.reshape(n, buffers.num_pairs, HIGGS_PAIR_DIM)
    indices = _nearest_pair_index(pairs, buffers).to(torch.uint8)
    packed_idx = pack_higgs_2bit_indices(indices)

    scale = (
        scale_f.to(torch.float16)
        .contiguous()
        .view(torch.uint8)
        .reshape(n, HIGGS_NORM_BYTES)
    )
    latent_bytes = torch.cat((packed_idx, scale), dim=-1)

    rope_bytes = (
        rope.reshape(n, HIGGS_ROPE_DIM)
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
    )
    return torch.cat((latent_bytes, rope_bytes), dim=-1).reshape(n, 1, HIGGS_SLOT_BYTES)


def reference_decompress(
    compressed: torch.Tensor,
    buffers: HiggsBuffers,
    dst_dtype: torch.dtype,
) -> torch.Tensor:
    """Decode ``(N, 1, 258)`` packed slots back to ``(N, 1, latent+rope)``."""

    from megatron.core.quantization.higgs.codec import (
        HIGGS_PACKED_BYTES,
        HIGGS_ROPE_DIM,
        HIGGS_SLOT_BYTES,
        unpack_higgs_2bit_indices,
    )

    n = compressed.shape[0]
    flat = compressed.reshape(n, HIGGS_SLOT_BYTES)
    packed = flat[:, :HIGGS_PACKED_BYTES]
    scale_bytes = flat[:, HIGGS_PACKED_BYTES : HIGGS_PACKED_BYTES + 2]
    indices = unpack_higgs_2bit_indices(packed, buffers.num_pairs).long()
    values = buffers.codebook[indices].reshape(n, buffers.latent_dim).to(torch.float32)
    scale = (
        scale_bytes.contiguous()
        .view(torch.float16)
        .reshape(n)
        .to(torch.float32)
    )
    rotated = values * scale[:, None]
    latent = fwht(rotated).to(dst_dtype)
    rope = (
        flat[:, HIGGS_PACKED_BYTES + 2 :]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(n, HIGGS_ROPE_DIM)
        .to(dst_dtype)
    )
    return torch.cat((latent, rope), dim=-1).reshape(
        n, 1, buffers.latent_dim + HIGGS_ROPE_DIM
    )
