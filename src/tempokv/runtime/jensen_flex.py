"""Opt-in fused Jensen scores without widening the attention head.

Inputs use the Self-Forcing BSHD layout. Squared steps must already have the
legacy cache rounding: cast Delta to the cache dtype, then square. Query
coefficients likewise round to the query dtype before FP32 score arithmetic.
No score-sized bias is materialized. Compilation is lazy and CUDA-only.
"""

from __future__ import annotations

from functools import lru_cache
import os

import torch

from .jensen import query_channels


def jensen_backend() -> str:
    backend = os.environ.get("TEMPOKV_JENSEN_BACKEND", "augmented").strip().lower()
    if backend not in {"augmented", "flex"}:
        raise ValueError("TEMPOKV_JENSEN_BACKEND must be 'augmented' or 'flex'")
    return backend


def score_terms(query, step_squared, group_size):
    """Return contiguous [G,B,H,Q/K] FP32 planes with legacy rounding."""
    if query.ndim != 4 or step_squared.ndim != 4:
        raise ValueError("query and squared steps must have BSHD/BSHG layout")
    dim = query.shape[-1]
    if dim <= 0 or group_size <= 0 or dim % group_size:
        raise ValueError("group_size must divide the head dimension")
    if (query.shape[0], query.shape[2], dim // group_size) != (
        step_squared.shape[0], step_squared.shape[2], step_squared.shape[3]
    ):
        raise ValueError("squared steps must match query batch, heads and groups")
    q_term = query_channels(query, group_size).to(query.dtype).float()
    return (
        q_term.permute(3, 0, 2, 1).contiguous(),
        step_squared.float().permute(3, 0, 2, 1).contiguous(),
    )


def make_score_mod(query_terms, key_terms, scale):
    # Slice outside score_mod: mixed constant/group and batched tensor indexing
    # is unsupported by some supported FlexAttention tracing versions.
    q_planes = query_terms.unbind(0)
    k_planes = key_terms.unbind(0)
    groups = len(q_planes)

    def score_mod(score, b, h, q_idx, kv_idx):
        correction = q_planes[0][b, h, q_idx] * k_planes[0][b, h, kv_idx]
        for group in range(1, groups):
            correction = correction + (
                q_planes[group][b, h, q_idx] * k_planes[group][b, h, kv_idx]
            )
        # q_planes already contain the negative coefficient / (24 sqrt(D)).
        return score + correction * scale

    return score_mod


@lru_cache(maxsize=1)
def _compiled_attention():
    from torch.nn.attention.flex_attention import flex_attention

    # Dynamic sequence lengths avoid deliberate per-length specialization;
    # Inductor may still guard/recompile, so growing-cache runs must measure it.
    # No CUDA graphs: scale tensors are refreshed on every cache read.
    return torch.compile(
        flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs"
    )


def corrected_attention(query, key, value, step_squared, *, group_size):
    """Unmasked attention over an already selected sink/recent window.

    Matches the cached Self-Forcing call (causal=False, dropout=0, equal heads).
    This is not a general replacement for masked training attention. Failure
    is explicit: no silent fallback to a dense score matrix or widened head.
    """
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("Q/K/V must be rank-4 BSHD tensors, with matching K/V")
    if min(*query.shape, *key.shape) <= 0:
        raise ValueError("Q/K dimensions must be non-empty")
    if (query.shape[0], query.shape[2], query.shape[3]) != (
        key.shape[0], key.shape[2], key.shape[3]
    ):
        raise ValueError("Q/K must have matching batch, heads and head dimension")
    if group_size <= 0 or query.shape[-1] % group_size:
        raise ValueError("group_size must divide the head dimension")
    if step_squared.shape != (*key.shape[:-1], query.shape[-1] // group_size):
        raise ValueError("squared steps must align token-for-token with K")
    if any(t.device != query.device for t in (key, value, step_squared)):
        raise ValueError("Q/K/V and squared steps must share a device")
    if query.device.type != "cuda":
        raise RuntimeError("fused Jensen attention requires CUDA and torch.compile")
    if query.dtype not in (torch.bfloat16, torch.float16) or any(
        t.dtype != query.dtype for t in (key, value, step_squared)
    ):
        raise ValueError("fused Jensen Q/K/V and squared steps must share a half dtype")
    q_terms, k_terms = score_terms(query, step_squared, group_size)
    scale = query.shape[-1] ** -0.5
    out = _compiled_attention()(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
        score_mod=make_score_mod(q_terms, k_terms, scale),
        scale=scale,
    )
    return out.permute(0, 2, 1, 3).contiguous()


__all__ = ["corrected_attention", "jensen_backend"]
