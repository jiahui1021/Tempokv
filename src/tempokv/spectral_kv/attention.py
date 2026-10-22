"""Reference and grouped attention for independent Spectral KV experiments.

The portable fast path tiles queries and keys and accumulates softmax statistics
in float32 (float64 for double inputs), so it never constructs the full attention
matrix.  A fused group backend can be injected, but FlashAttention execution is
not assumed or claimed here.  Scalar multiplicity is merged through each group's
log normalizer; token-dependent multiplicity uses the tiled implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AttentionGroup:
    k: torch.Tensor
    v: torch.Tensor
    log_count: float | torch.Tensor = 0.0


@dataclass(frozen=True)
class KVView:
    groups: list[AttentionGroup]

    @property
    def token_count(self) -> int:
        return sum(group.k.shape[1] for group in self.groups)


def _validate(q: torch.Tensor, view: KVView) -> list[AttentionGroup]:
    if not isinstance(q, torch.Tensor) or q.ndim != 4 or not q.is_floating_point():
        raise ValueError("q must be a real floating tensor with shape [B,Q,H,D]")
    if q.shape[0] < 1 or q.shape[2] < 1 or q.shape[3] < 1:
        raise ValueError("q batch, heads and head dimension must be positive")
    if not isinstance(view, KVView) or not isinstance(view.groups, (list, tuple)):
        raise ValueError("view must be a KVView containing attention groups")
    active = []
    for group in view.groups:
        if not isinstance(group, AttentionGroup):
            raise ValueError("view groups must be AttentionGroup instances")
        k, v = group.k, group.v
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor) or k.ndim != 4 or v.shape != k.shape:
            raise ValueError("group k and v must have the same [B,S,H,D] shape")
        if not k.is_floating_point() or not v.is_floating_point():
            raise ValueError("group k and v must be real floating tensors")
        if k.shape[0] != q.shape[0] or k.shape[2:] != q.shape[2:]:
            raise ValueError("group batch, heads and head dimension must match q")
        if k.device != q.device or v.device != q.device or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("group k, v and q must have matching dtype and device")
        _count_bias(group, q, torch.float64 if q.dtype == torch.float64 else torch.float32)
        if k.shape[1]:
            active.append(group)
    if not active:
        raise ValueError("attention requires at least one nonempty group")
    return active


def _count_bias(group: AttentionGroup, q: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Canonical additive bias [B,H,1,S]; accepts scalar, [S], [B,S]."""
    count = group.log_count
    if isinstance(count, torch.Tensor):
        if count.dtype == torch.bool or count.is_complex() or not (count.is_floating_point() or count.dtype in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        )):
            raise ValueError("log_count must be real and finite")
        if count.device != q.device:
            raise ValueError("log_count tensor must be on the same device as q")
        count = count.to(dtype=dtype)
    elif isinstance(count, Real) and not isinstance(count, bool):
        count = torch.tensor(float(count), dtype=dtype, device=q.device)
    else:
        raise ValueError("log_count must be a real scalar or tensor")
    if not torch.isfinite(count).all():
        raise ValueError("log_count must be finite; masks are not accepted as multiplicities")
    b, _, h, _ = q.shape
    s = group.k.shape[1]
    if count.ndim == 0:
        count = count.reshape(1, 1, 1, 1)
    elif count.shape == (s,):
        count = count.reshape(1, 1, 1, s)
    elif count.shape == (b, s):
        count = count.reshape(b, 1, 1, s)
    elif count.shape == (b, h, s):
        count = count.unsqueeze(-2)
    elif count.shape != (b, h, 1, s):
        raise ValueError("log_count must be scalar, [S], [B,S], [B,H,S] or [B,H,1,S]")
    return count.expand(b, h, 1, s)


def spectral_attention_ref(q: torch.Tensor, view: KVView) -> torch.Tensor:
    """Concatenate groups and use SDPA with their additive log multiplicity."""
    groups = _validate(q, view)
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    k = torch.cat([group.k for group in groups], dim=1)
    v = torch.cat([group.v for group in groups], dim=1)
    bias = torch.cat([_count_bias(group, q, dtype) for group in groups], dim=-1)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias
    ).transpose(1, 2)


def _tiled_group_attention(
    q: torch.Tensor,
    group: AttentionGroup,
    query_chunk_size: int,
    key_chunk_size: int,
    *,
    token_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    qt = q.transpose(1, 2).to(dtype=dtype)
    kt = group.k.transpose(1, 2)
    vt = group.v.transpose(1, 2)
    bias = _count_bias(group, q, dtype) if token_bias else None
    outputs, log_normalizers = [], []
    for q0 in range(0, q.shape[1], query_chunk_size):
        qc = qt[:, :, q0:q0 + query_chunk_size]
        maximum = torch.full((*qc.shape[:-1], 1), -torch.inf, dtype=dtype, device=q.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(qc)
        for k0 in range(0, group.k.shape[1], key_chunk_size):
            kc = kt[:, :, k0:k0 + key_chunk_size].to(dtype=dtype)
            vc = vt[:, :, k0:k0 + key_chunk_size].to(dtype=dtype)
            logits = torch.matmul(qc, kc.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
            if bias is not None:
                logits = logits + bias[..., k0:k0 + key_chunk_size]
            next_maximum = torch.maximum(maximum, logits.amax(dim=-1, keepdim=True))
            old_weight = torch.exp(maximum - next_maximum)
            weights = torch.exp(logits - next_maximum)
            numerator = numerator * old_weight + torch.matmul(weights, vc)
            denominator = denominator * old_weight + weights.sum(dim=-1, keepdim=True)
            maximum = next_maximum
        outputs.append(numerator / denominator)
        log_normalizers.append((maximum + denominator.log()).squeeze(-1))
    if q.shape[1] == 0:
        return q.to(dtype=dtype), torch.empty(q.shape[0], q.shape[2], 0, dtype=dtype, device=q.device)
    return torch.cat(outputs, dim=2).transpose(1, 2), torch.cat(log_normalizers, dim=2)


def _backend_result(
    result: tuple[torch.Tensor, torch.Tensor], q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("group_attention must return (output, lse)")
    out, lse = result
    if not isinstance(out, torch.Tensor) or out.shape != q.shape or not out.is_floating_point() or out.device != q.device:
        raise ValueError("group_attention output must have q's [B,Q,H,D] shape and device")
    expected = (q.shape[0], q.shape[2], q.shape[1])
    if not isinstance(lse, torch.Tensor) or lse.shape != expected or not lse.is_floating_point() or lse.device != q.device:
        raise ValueError("group_attention lse must have shape [B,H,Q] on q's device")
    if not torch.isfinite(out).all() or not torch.isfinite(lse).all():
        raise ValueError("group_attention output and lse must be finite")
    return out, lse


def spectral_attention_fast(
    q: torch.Tensor,
    view: KVView,
    *,
    group_attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    query_chunk_size: int = 32,
    key_chunk_size: int = 1024,
) -> torch.Tensor:
    """Merge group outputs exactly via LSE, with bounded-memory default kernels.

    An injected backend receives ``(q,k,v)`` and returns output ``[B,Q,H,D]``
    plus natural-log softmax normalizer ``[B,H,Q]``, with the usual ``1/sqrt(D)``
    scale.  Token-dependent ``log_count`` uses the portable tiled kernel because
    that backend contract has no additive-mask input.  Actual fused CUDA backend
    behavior requires separate verification on the intended GPU.
    """
    for size, name in ((query_chunk_size, "query_chunk_size"), (key_chunk_size, "key_chunk_size")):
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
            raise ValueError(f"{name} must be a positive integer")
    groups = _validate(q, view)
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    merged_numerator, merged_maximum, merged_denominator = None, None, None
    for group in groups:
        scalar_count = not isinstance(group.log_count, torch.Tensor) or group.log_count.ndim == 0
        if group_attention is not None and scalar_count:
            out, lse = _backend_result(group_attention(q, group.k, group.v), q)
        else:
            out, lse = _tiled_group_attention(q, group, query_chunk_size, key_chunk_size, token_bias=not scalar_count)
        out, lse = out.to(dtype=dtype), lse.to(dtype=dtype)
        if scalar_count:
            lse = lse + torch.as_tensor(group.log_count, device=q.device, dtype=dtype)
        if merged_numerator is None:
            merged_numerator = out
            merged_maximum = lse
            merged_denominator = torch.ones_like(lse)
            continue
        # Keep the maximum and relative mass separately: folding a tiny log
        # normalizer into a huge LSE can erase the accumulated group count.
        maximum = torch.maximum(merged_maximum, lse)
        w0, w1 = (merged_maximum - maximum).exp(), (lse - maximum).exp()
        merged_numerator = (
            merged_numerator * w0.transpose(1, 2)[..., None]
            + out * w1.transpose(1, 2)[..., None]
        )
        merged_denominator = merged_denominator * w0 + w1
        merged_maximum = maximum
    return (merged_numerator / merged_denominator.transpose(1, 2)[..., None]).to(dtype=q.dtype)
