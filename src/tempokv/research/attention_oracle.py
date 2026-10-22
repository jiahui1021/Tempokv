"""Independent full-window attention oracles for ``research.v1``.

Native attention always receives the original dtype and every selected head.
Only queries are batched.  The FP32 score calculations below are bounded by
query/key chunks and are used for masses and count-bias normalizers, never as
a replacement for the production attention callback.
"""
from __future__ import annotations

from dataclasses import replace
import operator
from typing import Iterable, Sequence

import torch

from tempokv.spectral.metrics import relative_error
from .context import FrozenAttention


NATIVE_PROTOCOL = "native_attention"
COUNT_PROTOCOL = "native_group_attention_fp32_lse"
_KEY_CHUNK_SIZE = 1024


def _positive_size(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_tensors(window: FrozenAttention, k: torch.Tensor, v: torch.Tensor) -> None:
    q = window.q_post
    if (k.ndim != 4 or k.shape != v.shape or k.shape[0] != q.shape[0]
            or k.shape[2:] != q.shape[2:]):
        raise ValueError("candidate K/V must be aligned [B,Seq,Heads,D] tensors")
    if k.shape[1] == 0:
        raise ValueError("attention requires at least one key token")
    if any(x.dtype != q.dtype or x.device != q.device for x in (k, v)):
        raise ValueError("candidate Q/K/V dtype and device must agree")


def _native_output(window: FrozenAttention, q: torch.Tensor,
                   k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = window.attention_fn(q, k, v)
    if not isinstance(out, torch.Tensor) or out.shape != q.shape:
        raise ValueError("attention callback must return [B,Q,Heads,D]")
    return out


def _batched_native(window: FrozenAttention, k: torch.Tensor, v: torch.Tensor,
                    query_batch_size: int) -> torch.Tensor:
    _positive_size(query_batch_size, "query_batch_size")
    _validate_tensors(window, k, v)
    if window.q_post.shape[1] == 0:
        return window.q_post.clone()
    return torch.cat([
        _native_output(window, window.q_post[:, start:start + query_batch_size], k, v)
        for start in range(0, window.q_post.shape[1], query_batch_size)
    ], dim=1)


@torch.no_grad()
def reference(window: FrozenAttention, query_batch_size: int = 64) -> torch.Tensor:
    """Run the production callback on Q slices and the complete original K/V."""
    return _batched_native(window, window.k_post, window.v, query_batch_size)


def _log_normalizer(q: torch.Tensor, k: torch.Tensor,
                    key_chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (max logit, centered log partition), each FP32 [B,H,Q].

    Keeping the two terms separate also preserves small count biases when the
    common QK offset is so large that adding that bias directly would round it.
    """
    query = q.float().transpose(1, 2)
    maximum = torch.full(query.shape[:-1], -torch.inf, dtype=torch.float32, device=q.device)
    centered = torch.full_like(maximum, -torch.inf)
    scale = q.shape[-1] ** -0.5
    for start in range(0, k.shape[1], key_chunk_size):
        key = k[:, start:start + key_chunk_size].float().permute(0, 2, 3, 1)
        scores = torch.matmul(query, key) * scale
        chunk_max = scores.amax(dim=-1)
        chunk_centered = torch.logsumexp(scores - chunk_max.unsqueeze(-1), dim=-1)
        next_max = torch.maximum(maximum, chunk_max)
        centered = torch.logaddexp(centered + (maximum - next_max),
                                   chunk_centered + (chunk_max - next_max))
        maximum = next_max
    return maximum, centered


def _weighted_output(window: FrozenAttention, k: torch.Tensor, v: torch.Tensor,
                     log_count: torch.Tensor, query_batch_size: int) -> torch.Tensor:
    # Every equal-count group uses the same native attention kernel.  Adding the
    # group's log-count to its streamed LSE gives its contribution to the single
    # global softmax denominator.  The online merge avoids retaining group QK.
    if window.q_post.shape[1] == 0:
        return window.q_post.clone()
    values = torch.unique(log_count)
    groups = [(bias, torch.nonzero(log_count == bias, as_tuple=False).flatten()) for bias in values]
    outputs = []
    for start in range(0, window.q_post.shape[1], query_batch_size):
        q = window.q_post[:, start:start + query_batch_size]
        merged = total_max = total_centered = None
        output_dtype = None
        for bias, indices in groups:
            group_k, group_v = k.index_select(1, indices), v.index_select(1, indices)
            group_out = _native_output(window, q, group_k, group_v)
            group_max, group_centered = _log_normalizer(q, group_k, _KEY_CHUNK_SIZE)
            group_centered = group_centered + bias.float()
            output_dtype = group_out.dtype
            group_out = group_out.float()
            if total_max is None:
                total_max, total_centered, merged = group_max, group_centered, group_out
            else:
                next_max = torch.maximum(total_max, group_max)
                old_lse = total_centered + (total_max - next_max)
                group_lse = group_centered + (group_max - next_max)
                next_centered = torch.logaddexp(old_lse, group_lse)
                old_weight = torch.exp(old_lse - next_centered).transpose(1, 2).unsqueeze(-1)
                new_weight = torch.exp(group_lse - next_centered).transpose(1, 2).unsqueeze(-1)
                merged = old_weight * merged + new_weight * group_out
                total_max, total_centered = next_max, next_centered
        outputs.append(merged.to(output_dtype))
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def evaluate(window: FrozenAttention, *, k: torch.Tensor | None = None,
             v: torch.Tensor | None = None, log_count: torch.Tensor | None = None,
             query_batch_size: int = 64) -> dict:
    """Compare candidate K/V against the original native full-window output.

    ``log_count`` is an additive per-key logit bias. Uniform biases cancel and
    therefore retain the unmodified callback path. Nonuniform biases use native
    equal-count-group outputs merged with streamed FP32 log-normalizers; that
    mixed protocol is explicitly identified in the result.
    """
    _positive_size(query_batch_size, "query_batch_size")
    candidate_k = window.k_post if k is None else k
    candidate_v = window.v if v is None else v
    _validate_tensors(window, candidate_k, candidate_v)
    if log_count is not None:
        if (not isinstance(log_count, torch.Tensor) or log_count.ndim != 1
                or log_count.numel() != candidate_k.shape[1]
                or not log_count.is_floating_point()
                or not bool(torch.isfinite(log_count).all())):
            raise ValueError("log_count must be a finite floating-point tensor [key_tokens]")
        log_count = log_count.to(device=candidate_k.device)
    baseline = reference(window, query_batch_size)
    weighted = log_count is not None and not bool(torch.all(log_count == log_count[0]))
    if weighted:
        output = _weighted_output(window, candidate_k, candidate_v, log_count, query_batch_size)
    elif candidate_k is window.k_post and candidate_v is window.v:
        output = baseline
    else:
        output = _batched_native(window, candidate_k, candidate_v, query_batch_size)
    return {"reference_output": baseline, "output": output,
            "relative_error": relative_error(baseline, output),
            "backend_protocol": COUNT_PROTOCOL if weighted else NATIVE_PROTOCOL}


def _indices(window: FrozenAttention, values: Iterable[int], name: str) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype == torch.bool or values.is_floating_point():
            raise ValueError(f"{name} must contain integer sequence indices")
        values = values.detach().cpu().tolist()
    result = []
    for value in values:
        try:
            if isinstance(value, bool):
                raise TypeError
            index = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain integer sequence indices") from exc
        if not 0 <= index < window.k_post.shape[1]:
            raise ValueError(f"{name} contains an index outside the original attention window")
        result.append(index)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate sequence indices")
    if result:
        mapped = torch.tensor(result, dtype=torch.long, device=window.context.protected_mask.device)
        if bool(window.context.protected_mask.index_select(0, mapped).any()):
            raise ValueError(f"{name} includes protected sink/recent/current tokens")
    return tuple(result)


@torch.no_grad()
def drop_window(window: FrozenAttention, indices: Iterable[int] = ()) -> FrozenAttention:
    """Remove K and V together, validating protection on original sequence indices."""
    removed = _indices(window, indices, "removed_indices")
    if not removed:
        return window
    if len(removed) == window.k_post.shape[1]:
        raise ValueError("cannot remove every attention token")
    keep = torch.ones(window.k_post.shape[1], dtype=torch.bool, device=window.k_post.device)
    keep[torch.tensor(removed, dtype=torch.long, device=keep.device)] = False
    kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
    context = replace(window.context, key_abs_positions=window.context.key_abs_positions.index_select(
        0, kept_indices.to(window.context.key_abs_positions.device)))
    return replace(window, context=context,
                   k_post=window.k_post.index_select(1, kept_indices),
                   v=window.v.index_select(1, kept_indices),
                   k_pre=None if window.k_pre is None else window.k_pre.index_select(1, kept_indices),
                   pre_valid_mask=None if window.pre_valid_mask is None else window.pre_valid_mask.index_select(
                       0, kept_indices.to(window.pre_valid_mask.device)))


def _attention_masses(window: FrozenAttention, blocks: Sequence[tuple[int, ...]], *,
                      query_batch_size: int, block_batch_size: int,
                      key_chunk_size: int) -> list[float]:
    _positive_size(query_batch_size, "query_batch_size")
    _positive_size(block_batch_size, "block_batch_size")
    _positive_size(key_chunk_size, "key_chunk_size")
    if not blocks:
        return []
    if window.q_post.shape[1] == 0:
        return [0.0] * len(blocks)
    sums = [0.0] * len(blocks)
    count = window.q_post.shape[0] * window.q_post.shape[1] * window.q_post.shape[2]
    for qstart in range(0, window.q_post.shape[1], query_batch_size):
        q = window.q_post[:, qstart:qstart + query_batch_size]
        denominator_max, denominator_centered = _log_normalizer(q, window.k_post, key_chunk_size)
        for bstart in range(0, len(blocks), block_batch_size):
            for offset, indices in enumerate(blocks[bstart:bstart + block_batch_size]):
                if not indices:
                    continue
                # Index-select only one key chunk at a time, including for large
                # candidate blocks; neither numerator nor denominator is dense.
                numerator_max = torch.full_like(denominator_max, -torch.inf)
                numerator_centered = torch.full_like(denominator_centered, -torch.inf)
                for kstart in range(0, len(indices), key_chunk_size):
                    selected = torch.tensor(indices[kstart:kstart + key_chunk_size],
                                            dtype=torch.long, device=window.k_post.device)
                    part_max, part_centered = _log_normalizer(q, window.k_post.index_select(1, selected), key_chunk_size)
                    next_max = torch.maximum(numerator_max, part_max)
                    numerator_centered = torch.logaddexp(numerator_centered + (numerator_max - next_max),
                                                         part_centered + (part_max - next_max))
                    numerator_max = next_max
                log_mass = ((numerator_max - denominator_max)
                            + (numerator_centered - denominator_centered))
                sums[bstart + offset] += float(torch.exp(log_mass).sum())
    return [value / count for value in sums]


@torch.no_grad()
def leave_one_out(window: FrozenAttention, blocks: Sequence[Iterable[int]],
                  query_batch_size: int = 64, block_batch_size: int = 4,
                  key_chunk_size: int = 1024) -> list[dict]:
    """Measure each removal independently against the unchanged original window."""
    _positive_size(query_batch_size, "query_batch_size")
    _positive_size(block_batch_size, "block_batch_size")
    _positive_size(key_chunk_size, "key_chunk_size")
    candidates = [_indices(window, block, "block_indices") for block in blocks]
    if not candidates:
        return []
    baseline = reference(window, query_batch_size)
    masses = _attention_masses(window, candidates, query_batch_size=query_batch_size,
                               block_batch_size=block_batch_size, key_chunk_size=key_chunk_size)
    rows = []
    for start in range(0, len(candidates), block_batch_size):
        for offset, indices in enumerate(candidates[start:start + block_batch_size]):
            output = baseline if not indices else reference(drop_window(window, indices), query_batch_size)
            candidate_index = start + offset
            rows.append({"candidate_index": candidate_index, "attention_mass": masses[candidate_index],
                         "local_attention_relative_error": relative_error(baseline, output)})
    return rows


@torch.no_grad()
def conditional_drop(window: FrozenAttention, target_indices: Iterable[int],
                     conditioning_removed: Iterable[int] = (), *,
                     query_batch_size: int = 64, block_batch_size: int = 4,
                     key_chunk_size: int = 1024) -> dict:
    """Measure target's marginal impact after an explicit, disjoint condition drop.

    Both sets refer to the original window. The reference drops only the
    conditioning set; the candidate drops their union, including both K and V.
    """
    _positive_size(query_batch_size, "query_batch_size")
    _positive_size(block_batch_size, "block_batch_size")
    _positive_size(key_chunk_size, "key_chunk_size")
    target = _indices(window, target_indices, "target_indices")
    condition = _indices(window, conditioning_removed, "conditioning_removed")
    if set(target).intersection(condition):
        raise ValueError("target_indices and conditioning_removed must be disjoint")
    conditioned = drop_window(window, condition)
    baseline = reference(conditioned, query_batch_size)
    output = baseline if not target else reference(drop_window(window, (*condition, *target)), query_batch_size)
    removed_set = set(condition)
    local_indices = {original: local for local, original in enumerate(
        index for index in range(window.k_post.shape[1]) if index not in removed_set)}
    mass = _attention_masses(conditioned, [tuple(local_indices[index] for index in target)],
                             query_batch_size=query_batch_size, block_batch_size=block_batch_size,
                             key_chunk_size=key_chunk_size)[0]
    error = relative_error(baseline, output)
    return {"reference_output": baseline, "output": output, "relative_error": error,
            "local_attention_relative_error": error, "attention_mass": mass,
            "backend_protocol": NATIVE_PROTOCOL, "target_indices": target,
            "conditioning_removed": condition}
