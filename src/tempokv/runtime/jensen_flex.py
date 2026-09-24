"""Opt-in fused Jensen scores via FlexAttention (BSHD, CUDA only)."""

from __future__ import annotations

from functools import lru_cache
import os

import torch

from .jensen import query_channels


def jensen_backend() -> str:
    backend = os.environ.get("TEMPOKV_JENSEN_BACKEND", "augmented").strip().lower()
    if backend not in {"augmented", "flex", "triton"}:
        raise ValueError("TEMPOKV_JENSEN_BACKEND must be 'augmented', 'flex' or 'triton'")
    return backend


def score_terms(query, step_squared, group_size):
    """Return contiguous [G,B,H,Q/K] FP32 planes with legacy rounding."""
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

    # Dynamic lengths; no CUDA graphs since scales change every read.
    return torch.compile(
        flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs"
    )


def corrected_attention(query, key, value, step_squared, *, group_size):
    """Unmasked attention over a selected cache window (no dense-score fallback)."""
    if query.device.type != "cuda":
        raise RuntimeError("fused Jensen attention requires CUDA and torch.compile")
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
