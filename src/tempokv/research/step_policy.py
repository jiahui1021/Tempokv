"""Exact denoising-step token budgets and callback-measured schedule comparisons.

Counts refer only to denoising calls. Clean refresh calls are outside the
schedule. The evaluator never constructs clean error by adding local or
per-step errors: a callback must replay the whole supplied schedule and return
its final clean-latent error against the unchanged native reference.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
from numbers import Real
from typing import Any


def _nonnegative_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _counts(values: Sequence[int], name: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"{name} must be a nonempty sequence of nonnegative integers")
    return tuple(_nonnegative_integer(value, name) for value in values)


def _levels(values: Sequence[float]) -> tuple[Fraction, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("retention_levels must be a nonempty sequence")
    levels = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("retention levels must be finite real numbers in [0,1]")
        levels.add(Fraction(str(value)))
    return tuple(sorted(levels))


def _rounded_keep(history: int, level: Fraction) -> int:
    # Exact decimal/rational multiplication, with .5 rounded upward.
    numerator = history * level.numerator
    return (2 * numerator + level.denominator) // (2 * level.denominator)


def _policy_id(counts: tuple[int, ...], protected: tuple[int, ...], kept: tuple[int, ...]) -> str:
    payload = {"schema": "denoise-step-policy.v1", "phase": "denoise",
               "step_token_counts": counts, "protected_counts": protected, "kept_history_tokens": kept}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"step-{digest[:20]}"


def enumerate_step_policies(step_token_counts: Sequence[int], retention_levels=(.25, .5, .75, 1), *,
                            protected_counts: Sequence[int] | None = None, total_budget: int | None = None,
                            max_states: int = 256) -> list[dict[str, Any]]:
    """Enumerate every unique feasible vector of integer history-token budgets.

    ``step_token_counts`` is each denoise call's *complete* native window size;
    ``protected_counts`` is its mandatory sink/recent/current subset. Thus the
    adjustable history count is their difference. Retention is applied only to
    that difference, with exact half-up rounding, and protected tokens are then
    added back. ``total_budget`` caps the sum across denoise calls only.

    Retention labels producing the same integer budget at a step are merged
    before taking the Cartesian product. The product of those distinct choices
    must not exceed ``max_states``, even when a budget would reject most states.
    Different per-step vectors with the same total cost are retained. No greedy
    or sampled policy is returned on overflow. An infeasible budget raises.

    If a vector is attainable by a constant requested retention, it is labelled
    a uniform control. A canonical uniform label is preferred over mixed aliases;
    full retention uses label 1 whenever available. Stable IDs depend on actual
    counts, not level ordering or the optional budget cap.
    """
    counts = _counts(step_token_counts, "step_token_counts")
    protected = (0,) * len(counts) if protected_counts is None else _counts(protected_counts, "protected_counts")
    if len(protected) != len(counts) or any(p > n for p, n in zip(protected, counts)):
        raise ValueError("protected_counts must match the steps and cannot exceed complete token counts")
    if total_budget is not None:
        _nonnegative_integer(total_budget, "total_budget")
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    levels = _levels(retention_levels)
    history = tuple(n - p for n, p in zip(counts, protected))
    choices = []
    aliases = []
    for count in history:
        buckets: dict[int, set[Fraction]] = {}
        for level in levels:
            buckets.setdefault(_rounded_keep(count, level), set()).add(level)
        aliases.append(buckets)
        choices.append(sorted(buckets))
    states = math.prod(len(choice) for choice in choices)
    if states > max_states:
        raise ValueError(f"distinct actual-budget state space ({states}) exceeds max_states={max_states}; no approximate policy returned")

    policies = []
    for kept in product(*choices):
        actual = tuple(p + k for p, k in zip(protected, kept))
        total = sum(actual)
        if total_budget is not None and total > total_budget:
            continue
        common = set.intersection(*(aliases[i][count] for i, count in enumerate(kept)))
        if common:
            uniform = Fraction(1) if Fraction(1) in common else min(common)
            retentions = (float(uniform),) * len(counts)
        else:
            retentions = tuple(float(Fraction(1) if Fraction(1) in aliases[i][count] else min(aliases[i][count]))
                               for i, count in enumerate(kept))
        policies.append({
            "policy_id": _policy_id(counts, protected, kept), "retentions": retentions,
            "kept_history_tokens": tuple(kept), "actual_tokens_per_step": actual,
            "total_token_accesses": total, "requested_budget": total_budget,
            "phase": "denoise", "step_indices": tuple(range(len(counts))),
            "step_token_counts": counts, "protected_counts": protected,
            "policy_family": "uniform" if common else "mixed_step",
            "uniform_retention_levels": tuple(float(level) for level in sorted(common)),
            "is_null": tuple(kept) == history,
        })
    if not policies:
        raise ValueError(f"no feasible denoise-step policy within total_budget={total_budget}")
    return sorted(policies, key=lambda policy: (policy["total_token_accesses"], policy["kept_history_tokens"], policy["policy_id"]))


def _validated_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, Mapping):
        raise TypeError("each policy must be a mapping")
    if policy.get("phase", "denoise") != "denoise":
        raise ValueError("step policies target denoise calls only; clean_refresh is never scheduled")
    required = ("policy_id", "retentions", "kept_history_tokens", "actual_tokens_per_step", "total_token_accesses",
                "step_token_counts", "protected_counts")
    if any(key not in policy for key in required):
        raise ValueError("policy is missing exact token-budget fields")
    counts = _counts(policy["step_token_counts"], "step_token_counts")
    protected = _counts(policy["protected_counts"], "protected_counts")
    kept = _counts(policy["kept_history_tokens"], "kept_history_tokens")
    actual = _counts(policy["actual_tokens_per_step"], "actual_tokens_per_step")
    _levels(policy["retentions"])
    length = len(counts)
    if any(len(values) != length for values in (protected, kept, actual, policy["retentions"])):
        raise ValueError("all policy vectors must match the denoise-step count")
    if any(p > n or k > n - p or a != p + k for n, p, k, a in zip(counts, protected, kept, actual)):
        raise ValueError("policy changes protected counts or has inconsistent actual token costs")
    if any(_rounded_keep(n - p, Fraction(str(level))) != k
           for n, p, k, level in zip(counts, protected, kept, policy["retentions"])):
        raise ValueError("retention labels disagree with exact kept history counts")
    total = _nonnegative_integer(policy["total_token_accesses"], "total_token_accesses")
    if total != sum(actual):
        raise ValueError("total_token_accesses disagrees with exact per-step costs")
    if not isinstance(policy["policy_id"], str) or not policy["policy_id"]:
        raise ValueError("policy_id must be a nonempty string")
    indices = policy.get("step_indices", tuple(range(length)))
    if (not isinstance(indices, Sequence) or any(type(i) is not int for i in indices)
            or tuple(indices) != tuple(range(length))):
        raise ValueError("step_indices must contain every in-schedule denoise index exactly once")
    result = deepcopy(dict(policy))
    result.update({"phase": "denoise", "step_indices": tuple(indices), "step_token_counts": counts,
                   "protected_counts": protected, "kept_history_tokens": kept, "actual_tokens_per_step": actual,
                   "policy_family": "uniform" if len(set(policy["retentions"])) == 1 else "mixed_step",
                   "is_null": all(k == n - p for n, p, k in zip(counts, protected, kept))})
    return result


def build_schedule(policy: Mapping[str, Any], denoise_step_count: int) -> dict[int, int]:
    """Return a fresh ``0-based denoise step -> kept history count`` mapping.

    The caller retains each step's protected subset separately. This function
    creates no persistent flags and accepts neither clean-refresh events nor
    indices outside the exact denoising schedule.
    """
    if type(denoise_step_count) is not int or denoise_step_count < 1:
        raise ValueError("denoise_step_count must be a positive integer")
    validated = _validated_policy(policy)
    if len(validated["kept_history_tokens"]) != denoise_step_count:
        raise ValueError("policy indices fall outside or do not cover the denoise schedule")
    return dict(zip(validated["step_indices"], validated["kept_history_tokens"]))


def evaluate_step_policies(policies: Sequence[Mapping[str, Any]], measure_callback: Callable, *,
                           budget: int | None = None) -> dict[str, Any]:
    """Measure complete schedules, then return exact-cost rows and their frontier.

    The callback receives a private policy copy and must return a mapping with
    finite nonnegative ``clean_latent_relative_error`` from a complete replay.
    Optional model/horizon measurements are retained. No local-error sum,
    sensitivity proxy, interpolation, or predicted clean error is calculated.
    A callback cannot replace policy IDs or costs. Infeasible schedules are not
    evaluated. The oracle is scoped to the supplied feasible schedules only.
    """
    if not isinstance(policies, Sequence) or isinstance(policies, (str, bytes)) or not policies:
        raise ValueError("policies must be a nonempty sequence")
    if not callable(measure_callback):
        raise TypeError("measure_callback must be callable")
    if budget is not None:
        _nonnegative_integer(budget, "budget")
    validated = [_validated_policy(policy) for policy in policies]
    if len({policy["policy_id"] for policy in validated}) != len(validated):
        raise ValueError("policies must have unique policy_id values")
    feasible = [policy for policy in validated if budget is None or policy["total_token_accesses"] <= budget]
    if not feasible:
        raise ValueError(f"no feasible supplied policy within budget={budget}")

    rows = []
    for policy in feasible:
        measured = measure_callback(deepcopy(policy))
        if not isinstance(measured, Mapping) or "clean_latent_relative_error" not in measured:
            raise ValueError("callback must return measured clean_latent_relative_error from a complete replay")
        error = measured["clean_latent_relative_error"]
        if isinstance(error, bool) or not isinstance(error, Real) or not math.isfinite(error) or error < 0:
            raise ValueError("clean_latent_relative_error must be a finite nonnegative real scalar")
        for key in policy.keys() & measured.keys():
            if measured[key] != policy[key]:
                raise ValueError(f"callback cannot replace exact policy field {key!r}")
        row = {**policy, **deepcopy(dict(measured)), "clean_latent_relative_error": float(error),
               "objective_kind": "measured_clean_latent_relative_error",
               "measurement_protocol": "complete_schedule_callback",
               "comparison_group": "uniform_control" if policy.get("policy_family") == "uniform" else "mixed_step",
               "oracle_scope": "supplied_feasible_schedules", "is_budget_oracle": False}
        rows.append(row)
    rows.sort(key=lambda row: (row["total_token_accesses"], row["clean_latent_relative_error"], row["policy_id"]))
    frontier = [row for row in rows if not any(
        other["total_token_accesses"] <= row["total_token_accesses"]
        and other["clean_latent_relative_error"] <= row["clean_latent_relative_error"]
        and (other["total_token_accesses"] < row["total_token_accesses"]
             or other["clean_latent_relative_error"] < row["clean_latent_relative_error"])
        for other in rows)]
    best = min(rows, key=lambda row: (row["clean_latent_relative_error"], row["total_token_accesses"], row["policy_id"]))
    best["is_budget_oracle"] = True
    errors = {row["clean_latent_relative_error"] for row in rows}
    return {"rows": rows, "frontier": frontier, "uniform_rows": [row for row in rows if row["comparison_group"] == "uniform_control"],
            "oracle": best, "budget": budget, "states_evaluated": len(rows),
            "objective_kind": "measured_clean_latent_relative_error", "oracle_scope": "supplied_feasible_schedules",
            "evidence_kind": "constant_objective_no_quality_separation" if len(errors) == 1 else "measured_clean_latent_differences"}
