"""Jensen-bias correction for quantized cached keys (arXiv 2605.26266).

By convexity quantization noise inflates exp(score) for cached keys.  Each
cached score gets ``b_i = (1/24d) * sum_j Delta_{i,j}^2 * ||q_j||^2``
subtracted, with ``Delta`` the stored step size.  ``b`` is a rank-G bilinear
form, so :func:`augment` folds it into G extra Q/K channels (pass the original
``1/sqrt(d)`` scale).  Per-group query norms are RoPE-invariant, and
unquantized keys have zero step, so they get no correction.
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

    ``step_sizes`` is ``[..., S_key, D/g]``.  Returns the augmented pair and the
    original ``1/sqrt(D)`` scale; ``pad_to`` zero-pads the head dimension.
    """
    dim = int(query.shape[-1])
    groups = dim // group_size
    extra_query = query_channels(query, group_size).to(query.dtype)
    extra_key = key_channels(step_sizes).to(key.dtype)
    width = dim + groups if pad_to is None else pad_to
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
    """Materialize ``b`` as ``[..., Q, K]``; a test reference, not the fast path."""
    dim = int(query.shape[-1])
    work = _work_dtype(query)
    norms = query.to(work).pow(2).reshape(*query.shape[:-1], -1, group_size).sum(-1)
    return torch.einsum(
        "...qg,...kg->...qk", norms, step_sizes.to(work).pow(2)
    ) / (24.0 * dim)


__all__ = ["augment", "bias", "key_channels", "query_channels"]
