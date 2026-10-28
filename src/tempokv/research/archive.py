"""Causal E5 archive selection over the complete available historical window.

This module selects native K/V pairs and never reencodes them. Logical costs
describe a hypothetical gathered native K/V payload plus one int64 absolute
position per retained token. Selection alone leaves the full input backing
allocations in place; their bytes are reported separately as compute-only.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, localcontext
import math
from numbers import Real
from typing import Any

import torch

from .context import FrozenAttention


POLICIES = ("recent_only", "uniform_archive", "attention_archive", "diversity_archive", "oracle_archive")
INDEX_ELEMENT_BYTES = 8


def _backing_bytes(*values: torch.Tensor) -> int:
    total, seen = 0, set()
    for value in values:
        storage = value.untyped_storage()
        identity = (str(value.device), storage.data_ptr(), storage.nbytes())
        if identity not in seen:
            seen.add(identity)
            total += storage.nbytes()
    return total


def _spread_order(count: int) -> list[int]:
    if count < 2:
        return list(range(count))
    result, intervals = [0, count - 1], [(0, count - 1)]
    offset = 0
    while offset < len(intervals):
        left, right = intervals[offset]
        offset += 1
        middle = (left + right) // 2
        if middle > left:
            result.append(middle)
            intervals.extend(((left, middle), (middle, right)))
    return result


def _scores(values, count: int, eligible: list[int], name: str) -> dict[int, float]:
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must align with the complete original token window")
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must align with the complete original token window") from exc
    if len(values) != count:
        raise ValueError(f"{name} must align with the complete original token window")
    result = {}
    for index in eligible:
        value = values[index]
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} needs one finite scalar per eligible token for this head")
        result[index] = float(value)
    return result


def _diversity(window: FrozenAttention, eligible: list[int], budget: int) -> list[int]:
    """Oldest seed, then exact farthest-point squared Euclidean V priorities."""
    # Only historical eligible rows are read. Neither current/recent/sink V nor
    # future observations affect initialization, normalization, or distances.
    values = window.v[:, eligible].detach().permute(1, 0, 2, 3).reshape(len(eligible), -1).double()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("diversity_archive requires finite available historical V")
    maximum = values.abs().amax()
    if float(maximum) > 0:
        values = values / maximum  # Common scaling preserves Euclidean ordering and avoids overflow.
    chosen = []
    minimum_distance = torch.full((len(eligible),), torch.inf, dtype=torch.float64, device=values.device)
    next_index = 0
    for _ in range(budget):
        chosen.append(next_index)
        distance = (values - values[next_index]).square().sum(-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        minimum_distance[chosen] = -torch.inf
        if len(chosen) < budget:
            next_index = int(torch.argmax(minimum_distance))  # Absolute-position order breaks ties.
    return [eligible[index] for index in chosen]


def select_archive(window: FrozenAttention, *, policy: str, retention: float,
                   attention_scores=None, oracle_scores=None, online: bool = True,
                   token_budget: int | None = None) -> dict[str, Any]:
    """Keep an exact half-up fraction of *all* eligible historical K/V pairs.

    Sink, configured recent history and current-chunk pairs are mandatory. Keys
    and queries outside the current available chunk boundary are rejected,
    including in oracle diagnostics. Unlike sampled-block selection, every
    unprotected historical token participates in this budget.

    ``recent_only`` ranks newest first; ``uniform_archive`` uses coarse-to-fine
    absolute-position priorities; ``attention_archive`` ranks supplied current-
    or past-generated attention scores. Such causal provenance is the caller's
    contract. ``diversity_archive`` uses deterministic farthest-point V sampling
    with the oldest eligible token as seed. ``oracle_archive`` explicitly uses
    future-aware scores, requires ``online=False`` even for null controls, and
    is a diagnostic ranking rather than a claimed joint subset optimum.

    Score arrays align with every original token. Ranked eligible entries must
    be finite; protected entries are ignored. Budgets 0/full need no scores.
    Equal scores/distances favor earlier absolute positions. No global RNG,
    input tensor, coordinate, or cache state is changed.
    """
    if not isinstance(window, FrozenAttention):
        raise TypeError("window must be FrozenAttention")
    if window.k_post.shape[2] != 1:
        raise ValueError("archive selection requires a single frozen head")
    if not isinstance(policy, str) or policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if type(online) is not bool:
        raise ValueError("online must be a boolean")
    if policy == "oracle_archive" and online:
        raise ValueError("oracle_archive uses future-aware scores and requires online=False")
    if isinstance(retention, bool) or not isinstance(retention, Real) or not math.isfinite(retention) or not 0 <= retention <= 1:
        raise ValueError("retention must be a finite real number in [0,1]")
    context = window.context
    available_end = context.current_chunk_start + context.grid_t * context.frame_tokens
    if bool((context.key_abs_positions >= available_end).any()) or bool((context.query_abs_positions >= available_end).any()):
        raise ValueError("window contains future positions beyond the current available chunk")
    eligible = context.eligible_mask.nonzero().flatten().detach().cpu().tolist()
    positions = context.key_abs_positions.detach().cpu().tolist()
    eligible.sort(key=lambda index: positions[index])
    with localcontext() as decimal_context:
        decimal_context.prec = max(28, len(str(len(eligible))) + len(str(retention)) + 8)
        budget = int((Decimal(len(eligible)) * Decimal(str(retention))).to_integral_value(rounding=ROUND_HALF_UP))
    if token_budget is not None:
        if type(token_budget) is not int or not 0<=token_budget<=len(eligible):
            raise ValueError("token_budget must be an integer within the available eligible history")
        budget=token_budget
    if budget == len(eligible):
        selected = eligible
    elif budget == 0:
        selected = []
    elif policy == "recent_only":
        selected = list(reversed(eligible))[:budget]
    elif policy == "uniform_archive":
        selected = [eligible[index] for index in _spread_order(len(eligible))[:budget]]
    elif policy == "diversity_archive":
        selected = _diversity(window, eligible, budget)
    else:
        name = "oracle_scores" if policy == "oracle_archive" else "attention_scores"
        values = _scores(oracle_scores if policy == "oracle_archive" else attention_scores,
                         window.k_post.shape[1], eligible, name)
        selected = sorted(eligible, key=lambda index: (-values[index], positions[index]))[:budget]
    chosen = set(selected)
    removed = tuple(index for index in eligible if index not in chosen)
    removed_set = set(removed)
    kept = tuple(index for index in range(window.k_post.shape[1]) if index not in removed_set)
    protected = int(context.protected_mask.sum())
    if len(chosen) != budget or len(kept) != protected + budget:
        raise RuntimeError("archive selection failed the exact eligible-token budget")
    # One coordinate vector is shared by the selected batch/head payload.
    elements_per_token = window.k_post.shape[0] * window.k_post.shape[2] * window.k_post.shape[3]
    payload = len(kept) * elements_per_token * (window.k_post.element_size() + window.v.element_size())
    index_bytes = len(kept) * INDEX_ELEMENT_BYTES
    backing = _backing_bytes(window.k_post, window.v)
    diagnostic = policy == "oracle_archive"
    return {
        "policy": policy, "retention": float(retention), "kept_indices": kept, "removed_indices": removed,
        "actual_retained_tokens": len(kept), "eligible_kept_tokens": len(chosen),
        "eligible_tokens": len(eligible), "protected_tokens": protected,
        "actual_retention": len(kept) / window.k_post.shape[1] if window.k_post.shape[1] else 1.,
        "payload_bytes": payload, "index_bytes": index_bytes, "logical_bytes": payload + index_bytes,
        "logical_payload_bytes": payload, "logical_index_bytes": index_bytes,
        "index_format": "one_int64_absolute_position_per_retained_token",
        "compute_only_full_backing_bytes": backing, "physical_kv_bytes": backing,
        "physical_storage_kind": "unchanged_full_input_backing_compute_only",
        "packed_or_gathered_storage_allocated": False, "reencoding": False,
        "causal_scope": "future_aware_scores_on_current_available_history" if diagnostic else "current_available_history_only",
        "score_provenance": "future_aware_diagnostic" if diagnostic else "current_or_past_attention" if policy == "attention_archive" else "available_history_only",
        "diagnostic_only": diagnostic, "online": online,
        "selection_scope": "complete_eligible_history_tokens",
        "oracle_interpretation": "future-aware ranking diagnostic; not a joint subset optimum" if diagnostic else None,
        "available_chunk_end_exclusive": available_end, "budget_cohort": f"tokens:{len(kept)}",
    }
