"""Jensen-bias correction for quantized cached keys (arXiv 2605.26266).

Quantization adds approximately zero-mean noise to a cached key, so the
pre-softmax score ``s_i`` stays unbiased.  The exponential does not: by
convexity ``E[e^d] >= 1``, so every quantized cached score contributes more to
the softmax partition sum than its unquantized counterpart, and the cached
block takes attention mass from the unquantized current chunk.  Subtracting

    b_i = (1 / 24d) * sum_j Delta_{i,j}^2 * ||q_j||^2

from each cached score removes that inflation in expectation.  ``Delta`` is the
step size TempoKV already stores, so the correction needs no extra state.

Two properties make this cheap to apply here rather than needing a fused
attention kernel:

**It is a rank-G bilinear form.**  ``b`` pairs a per-query vector
``(||q_j||^2)_j`` with a per-key vector ``(Delta_{i,j}^2)_j``, both of length
``G = d/g`` -- two entries at the production ``d=128, g=64``.  So it folds into
``G`` extra Q/K channels and the ordinary attention call computes it, whatever
backend that call uses.  :func:`augment` builds those channels; the caller must
then pass the *original* ``1/sqrt(d)`` softmax scale, since the augmented head
dimension is no longer the one the scores are normalized by.

**Per-group query norms are RoPE-invariant.**  Wan rotates adjacent channel
pairs and ``g`` is even, so no rotary pair straddles a group boundary and
``||q_j||^2`` is the same before and after RoPE.  The correction may therefore
use the post-RoPE query against pre-RoPE step sizes, which is exactly what a
pre-RoPE cache has on hand.

Keys that were never quantized -- the current chunk, and any BF16 cache chunk
-- carry a zero step size and so receive a zero correction with no masking.
"""

from __future__ import annotations

import math

import torch


def _work_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Accumulate at least in float32, but never round a float64 caller down."""
    return tensor.dtype if tensor.dtype is torch.float64 else torch.float32


def query_channels(query: torch.Tensor, group_size: int) -> torch.Tensor:
    """The ``G`` extra query channels, for ``query`` shaped ``[..., D]``."""
    dim = int(query.shape[-1])
    if dim % group_size:
        raise ValueError("group_size must divide the head dimension")
    work = query.to(_work_dtype(query))
    norms = work.pow(2).reshape(*query.shape[:-1], -1, group_size).sum(-1)
    return -norms / (24.0 * math.sqrt(dim))


def key_channels(step_sizes: torch.Tensor) -> torch.Tensor:
    """The ``G`` extra key channels: the squared step size of each group."""
    return step_sizes.to(_work_dtype(step_sizes)).pow(2)


def augment(
    query: torch.Tensor,
    key: torch.Tensor,
    step_sizes: torch.Tensor,
    *,
    group_size: int,
    pad_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Fold the correction into extra channels of ``query`` and ``key``.

    ``query`` and ``key`` are ``[..., S, D]`` with matching leading axes;
    ``step_sizes`` is ``[..., S_key, D/g]`` and aligns with ``key``.  Returns
    the augmented pair and the softmax scale the caller must pass through,
    which stays ``1/sqrt(D)`` of the *original* head dimension.

    ``pad_to`` widens the augmented head dimension with zero channels, for
    backends that only accept particular head dimensions.
    """
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must share a head dimension")
    if step_sizes.shape[:-1] != key.shape[:-1]:
        raise ValueError(
            f"step sizes {tuple(step_sizes.shape)} do not align with keys "
            f"{tuple(key.shape)}"
        )
    dim = int(query.shape[-1])
    groups = dim // group_size
    if step_sizes.shape[-1] != groups:
        raise ValueError(
            f"step sizes carry {step_sizes.shape[-1]} groups, expected {groups}"
        )
    extra_query = query_channels(query, group_size).to(query.dtype)
    extra_key = key_channels(step_sizes).to(key.dtype)
    width = dim + groups if pad_to is None else pad_to
    if width < dim + groups:
        raise ValueError(f"pad_to={pad_to} cannot hold {dim + groups} channels")
    padding = width - dim - groups
    parts_query = [query, extra_query]
    parts_key = [key, extra_key]
    if padding:
        parts_query.append(query.new_zeros((*query.shape[:-1], padding)))
        parts_key.append(key.new_zeros((*key.shape[:-1], padding)))
    return (
        torch.cat(parts_query, dim=-1),
        torch.cat(parts_key, dim=-1),
        1.0 / math.sqrt(dim),
    )


def bias(query: torch.Tensor, step_sizes: torch.Tensor, group_size: int) -> torch.Tensor:
    """Materialize ``b`` as ``[..., Q, K]``.  A reference, not the fast path.

    Attention never needs this tensor -- :func:`augment` lets the attention
    kernel compute the same values without materializing a score-sized buffer.
    It exists so tests can check the folded form against the definition.
    """
    dim = int(query.shape[-1])
    work = _work_dtype(query)
    norms = query.to(work).pow(2).reshape(*query.shape[:-1], -1, group_size).sum(-1)
    return torch.einsum(
        "...qg,...kg->...qk", norms, step_sizes.to(work).pow(2)
    ) / (24.0 * dim)


__all__ = ["augment", "bias", "key_channels", "query_channels"]
