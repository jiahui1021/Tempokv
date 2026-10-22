"""Pure, per-head history selection with exact candidate-token budgets.

Selection returns original full-window indices and never computes attention,
changes a tensor, or removes an unlisted token. In particular, a sampled
candidate reservoir is not treated as the complete historical cache. Oracle
priorities are leave-one-out diagnostic scores, not a joint subset optimum.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_HALF_UP, localcontext
import math
from numbers import Integral, Real
import random
from typing import Any

import torch

from .context import FrozenAttention


POLICIES = ("random", "uniform", "similarity", "attention", "frequency", "frequency_qk", "oracle")


def _blocks(window: FrozenAttention, candidates: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("candidates must be a sequence of original-index blocks")
    positions = window.context.key_abs_positions.detach().cpu().tolist()
    eligible = window.context.eligible_mask.detach().cpu().tolist()
    seen = set()
    result = []
    for block in candidates:
        if isinstance(block, torch.Tensor):
            if block.ndim != 1:
                raise ValueError("each candidate block must be a one-dimensional integer sequence")
            values = block.detach().cpu().tolist()
        else:
            if isinstance(block, (str, bytes)):
                raise ValueError("each candidate block must be an integer sequence")
            try:
                values = list(block)
            except TypeError as exc:
                raise ValueError("each candidate block must be an integer sequence") from exc
        if not values or any(isinstance(index, bool) or not isinstance(index, Integral) for index in values):
            raise ValueError("candidate blocks must be nonempty and contain integer indices")
        values = [int(index) for index in values]
        if any(index < 0 or index >= len(positions) for index in values):
            raise ValueError("candidate index lies outside the original attention window")
        if len(set(values)) != len(values) or seen.intersection(values):
            raise ValueError("candidate blocks must be disjoint, without duplicate indices")
        if any(not eligible[index] for index in values):
            raise ValueError("candidates must be unprotected historical tokens; sink/recent/current tokens are forbidden")
        seen.update(values)
        result.append(tuple(sorted(values, key=lambda index: positions[index])))
    return tuple(result)


def _spread_order(count: int) -> list[int]:
    """Coarse-to-fine priorities over absolute-position-sorted blocks."""
    if count < 2:
        return list(range(count))
    result = [0, count - 1]
    intervals = [(0, count - 1)]
    offset = 0
    while offset < len(intervals):
        left, right = intervals[offset]
        offset += 1
        middle = (left + right) // 2
        if middle > left:
            result.append(middle)
            intervals.extend(((left, middle), (middle, right)))
    return result


def _score_values(values: Any, count: int, name: str) -> list[float]:
    if values is None:
        return [math.nan] * count
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} needs one scalar per candidate block for the current head")
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} needs one scalar per candidate block for the current head") from exc
    if len(raw) != count:
        raise ValueError(f"{name} must align with every candidate block for the current head")
    result = []
    for value in raw:
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            value = value.item()
        if value is None:
            result.append(math.nan)
        elif isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} needs one scalar per candidate block for the current head; do not share head arrays")
        else:
            result.append(float(value))
    return result


def select_history(window: FrozenAttention, candidates: Sequence[Sequence[int]], *, policy: str,
                   retention: float, scores: Mapping[str, Sequence], oracle_errors=None,
                   seed: int = 0, online: bool = False) -> dict[str, Any]:
    """Select an exact number of candidate tokens while retaining everything else.

    The candidate budget is ``round_half_up(candidate_tokens * retention)`` in
    exact decimal arithmetic. Whole blocks are taken in priority order; only
    the last block can be partial. Its tokens follow absolute-position order.
    Returned kept/removed indices both refer to the original full window.

    Similarity keeps low scores (less redundant blocks); attention, frequency,
    frequency_qk and oracle keep high scores. Nonfinite/missing scores sort last,
    with absolute positions breaking ties. When ranking is necessary, an all-
    missing score vector raises instead of silently substituting random scores.
    Retention 0/1 and empty reservoirs do not need score values. Oracle is always
    forbidden online, including null controls. Random uses an isolated local
    generator. No global random state or cross-head score cache is consulted.

    ``protected_tokens`` counts sink/recent/current tokens in the full window;
    other noncandidate history is also retained. ``partial_block_tokens`` is
    the number kept from the single partially retained block, or zero.
    """
    if not isinstance(window, FrozenAttention):
        raise TypeError("window must be FrozenAttention")
    if window.k_post.shape[2] != 1:
        raise ValueError("select_history requires a single-head FrozenAttention to prevent sharing head scores")
    if not isinstance(policy, str) or policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if type(online) is not bool:
        raise ValueError("online must be a boolean")
    if policy == "oracle" and online:
        raise ValueError("oracle LOO priorities are diagnostic only and cannot be used online")
    if isinstance(retention, bool) or not isinstance(retention, Real) or not math.isfinite(retention) or not 0 <= retention <= 1:
        raise ValueError("retention must be a finite real number in [0,1]")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if not isinstance(scores, Mapping):
        raise TypeError("scores must map score names to current-head candidate sequences")
    blocks = _blocks(window, candidates)
    candidate = {index for block in blocks for index in block}
    candidate_count = len(candidate)
    with localcontext() as context:
        context.prec = max(28, len(str(candidate_count)) + len(str(retention)) + 8)
        budget = int((Decimal(candidate_count) * Decimal(str(retention))).to_integral_value(rounding=ROUND_HALF_UP))

    positions = window.context.key_abs_positions.detach().cpu().tolist()
    chronological = sorted(range(len(blocks)), key=lambda i: tuple(positions[index] for index in blocks[i]))
    score_scope = "current_head_sampled_candidate_blocks"
    partial = 0
    kept_candidate: list[int] = []
    finite_scores = None
    if budget == candidate_count:
        kept_candidate = sorted(candidate)
    elif budget > 0:
        if policy == "random":
            order = list(chronological)
            random.Random(int(seed)).shuffle(order)
        elif policy == "uniform":
            order = [chronological[i] for i in _spread_order(len(blocks))]
        else:
            values = _score_values(oracle_errors if policy == "oracle" else scores.get(policy), len(blocks), policy)
            finite_scores = sum(math.isfinite(value) for value in values)
            if finite_scores == 0:
                raise ValueError(f"not measured: no finite {policy} scores for the current head's candidate blocks")
            position_rank = {index: rank for rank, index in enumerate(chronological)}
            order = sorted(range(len(blocks)), key=lambda i: (
                not math.isfinite(values[i]),
                (values[i] if policy == "similarity" else -values[i]) if math.isfinite(values[i]) else 0.,
                position_rank[i],
            ))
        for index in order:
            remaining = budget - len(kept_candidate)
            if remaining == 0:
                break
            take = min(remaining, len(blocks[index]))
            kept_candidate.extend(blocks[index][:take])
            if take < len(blocks[index]):
                partial = take

    selected = set(kept_candidate)
    removed = tuple(sorted(candidate - selected))
    removed_set = set(removed)
    kept = tuple(index for index in range(window.k_post.shape[1]) if index not in removed_set)
    if len(selected) != budget:
        raise RuntimeError("selection did not meet the exact candidate-token budget")
    return {
        "policy": policy, "kept_indices": kept, "removed_indices": removed,
        "kept_candidate_indices": tuple(sorted(selected)), "kept_candidate_tokens": len(selected),
        "candidate_tokens": candidate_count, "actual_retained_tokens": len(kept),
        "protected_tokens": int(window.context.protected_mask.sum()), "retention": float(retention),
        "partial_block_tokens": partial, "score_scope": score_scope,
        "diagnostic_only": policy == "oracle", "finite_score_blocks": finite_scores,
        "oracle_interpretation": "LOO ranking diagnostic; not a joint subset optimum" if policy == "oracle" else None,
        "untouched_tokens": window.k_post.shape[1] - candidate_count,
        "actual_candidate_retention": len(selected) / candidate_count if candidate_count else 1.,
        "actual_retention": len(kept) / window.k_post.shape[1] if window.k_post.shape[1] else 1.,
        "budget_cohort": f"tokens:{len(kept)}",
        "selection_scope": "sampled_candidate_tokens_only",
    }
