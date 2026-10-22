"""Independent, bounded calibration of one frozen research intervention.

Local attention error and relative KV RMSE are separate protocols. The local
search retains the closest *measured* candidate and makes no monotonicity claim.
Its stored seed/target_nmse identify the fixed direction and scale to reuse on
the reference branch; a rollout must not recalibrate that direction.
"""
from __future__ import annotations

import math
from numbers import Real

import torch

from . import attention_oracle as oracle
from . import interventions as ops


_KINDS = {"gaussian", "low_rank", "temporal_noise", "quantization_like", "zero"}
_MAX_STRENGTH = 1e4
_MIN_GRID_STRENGTH = 1e-8


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_kind(kind):
    if kind not in _KINDS:
        raise ValueError("unsupported calibration intervention kind")


def _role_errors(window, variant, indices, role):
    """Return actual, selected-role errors with no epsilon-created denominator."""
    error_sum = energy_sum = 0.
    nmse = {"k": 0., "v": 0.}
    zero_roles = []
    for name, source, changed in (("k", window.k_post, variant.k_post), ("v", window.v, variant.v)):
        if name not in role:
            continue
        original = source[:, indices].double()
        energy = float(original.square().sum())
        error = float((changed[:, indices].double() - original).square().sum())
        error_sum += error; energy_sum += energy
        if energy == 0:
            nmse[name] = None
            zero_roles.append(name)
        else:
            nmse[name] = error / energy
    rmse = math.sqrt(error_sum / energy_sum) if energy_sum > 0 else None
    return rmse, nmse, zero_roles


def _common(window, variant, indices, kind, role, seed, scale):
    rmse, nmse, zero_roles = _role_errors(window, variant, indices, role)
    return {"variant": variant, "kind": kind, "role": role, "seed": int(seed), "indices": list(indices),
            "calibration_scale": float(scale), "target_nmse": float(scale * scale),
            "relative_kv_rmse": rmse, "nmse_k": nmse["k"], "nmse_v": nmse["v"],
            "zero_energy_roles": zero_roles, "rmse_scope": "selected_tokens_and_selected_roles"}


@torch.no_grad()
def relative_rmse_intervention(window, indices, strength, kind="gaussian", role="kv", seed=0) -> dict:
    """Request per-role NMSE=strength² and report actual dtype-rounded errors.

    No attention error is fitted or declared matched by this protocol. A zero
    ablation is discrete and reports its actual RMSE regardless of strength.
    """
    strength = _nonnegative(strength, "strength")
    if not math.isfinite(strength * strength):
        raise ValueError("strength squared must be finite")
    _validate_kind(kind)
    variant = ops.perturb(window, indices, kind=kind, role=role, target_nmse=strength * strength, seed=seed)
    # Use ops' normalized indices so tensor sequences and Python sequences have
    # exactly the same selection and metadata representation.
    indices = variant.details["indices"]
    result = _common(window, variant, indices, kind, role, seed, strength)
    active = [result[f"nmse_{name}"] for name in ("k", "v") if name in role]
    target_nmse = strength * strength
    matched = bool(indices) and all(value is not None and
        (value == 0. if target_nmse == 0 else abs(value - target_nmse) <= .05 * target_nmse) for value in active)
    if not indices:
        status = "no_targets"
    elif result["zero_energy_roles"]:
        status = "zero_input_energy"
    elif matched:
        status = "exact_null" if strength == 0 else "matched"
    elif kind == "zero":
        status = "unmatched_discrete_ablation"
    else:
        status = "unmatched_dtype_rounding"
    result.update({"protocol": "relative_kv_rmse", "matched_quantity": "relative_kv_rmse",
                   "requested_relative_kv_rmse": strength, "requested_local_error": None,
                   "achieved_local_error": None, "matched": matched, "status": status, "evaluations": 0})
    return result


def _next_strength(points, target):
    """Refine adjacent measured points, allowing every observed crossing."""
    measured = sorted((point for point in points if point["error"] is not None), key=lambda point: point["scale"])
    existing = {point["scale"] for point in points}
    intervals = list(zip(measured, measured[1:]))
    intervals.sort(key=lambda pair: (
        0 if (pair[0]["error"] - target) * (pair[1]["error"] - target) < 0 else 1,
        min(abs(pair[0]["error"] - target), abs(pair[1]["error"] - target)),
        pair[1]["scale"] - pair[0]["scale"]))
    for left, right in intervals:
        lo, hi, a, b = left["scale"], right["scale"], left["error"], right["error"]
        crossing = (a - target) * (b - target) < 0
        if crossing and a != b:
            candidate = lo + (target - a) * (hi - lo) / (b - a)
        else:
            candidate = math.sqrt(lo * hi) if lo > 0 else hi * .5
        if candidate <= lo or candidate >= hi or candidate in existing:
            candidate = (lo + hi) * .5
        if lo < candidate < hi and candidate not in existing:
            return candidate
    return None


@torch.no_grad()
def calibrate_local_error(window, indices, *, kind="gaussian", role="kv", target_error,
                          seed, query_batch_size=64, max_evaluations=16, tolerance=.05) -> dict:
    """Fit local attention error on the unchanged full reference window.

    Search a bounded logarithmic strength grid, then refine adjacent measured
    intervals. Multiple crossings and nonmonotonic responses are allowed. The
    budget counts oracle.evaluate calls, including the null measurement.
    """
    target = _nonnegative(target_error, "target_error")
    tolerance = _nonnegative(tolerance, "tolerance")
    _positive_integer(query_batch_size, "query_batch_size")
    _positive_integer(max_evaluations, "max_evaluations")
    _validate_kind(kind)
    null = ops.perturb(window, indices, kind=kind, role=role, target_nmse=0., seed=seed)
    indices = null.details["indices"]
    points = []
    variants = {}
    evaluations = 0

    def measure(scale, variant=None):
        nonlocal evaluations
        if variant is None:
            try:
                variant = ops.perturb(window, indices, kind=kind, role=role, target_nmse=scale * scale, seed=seed)
            except ValueError as error:
                if "overflowed" not in str(error):
                    raise
                points.append({"scale": float(scale), "target_nmse": float(scale * scale), "error": None,
                               "status": "dtype_overflow"})
                return None
        evaluated = oracle.evaluate(window, k=variant.k_post, v=variant.v,
                                    log_count=variant.log_count, query_batch_size=query_batch_size)
        evaluations += 1
        if not bool(torch.isfinite(evaluated["reference_output"]).all()) or not bool(torch.isfinite(evaluated["output"]).all()):
            raise ValueError("native attention must return finite calibration outputs")
        error = float(evaluated["relative_error"])
        points.append({"scale": float(scale), "target_nmse": float(scale * scale), "error": error, "status": "measured"})
        variants[float(scale)] = variant
        return evaluated

    baseline = measure(0., null)
    zero_reference = float(baseline["reference_output"].double().square().sum()) == 0.

    def is_match(error):
        return error == 0. if target == 0 else abs(error - target) <= tolerance * target

    def finish(status=None):
        valid = [point for point in points if point["error"] is not None]
        best = min(valid, key=lambda point: (abs(point["error"] - target), point["scale"]))
        result = _common(window, variants[best["scale"]], indices, kind, role, seed, best["scale"])
        matched = not zero_reference and bool(indices) and is_match(best["error"])
        achieved = None if zero_reference else best["error"]
        if status is None:
            positive = [point["error"] for point in valid if point["error"] > 0]
            if matched:
                status = "exact_null" if target == 0 else "matched"
            elif window.k_post.dtype in (torch.bfloat16, torch.float16) and positive and target < min(positive):
                status = "unmatched_dtype_floor"
            elif all(point["error"] == 0 for point in valid):
                status = "unmatched_unresponsive"
            else:
                status = "unmatched_search_budget"
        result.update({"protocol": "local_d_att", "matched_quantity": "local_d_att",
                       "requested_local_error": target, "achieved_local_error": achieved,
                       "requested_relative_kv_rmse": None, "matched": matched, "status": status,
                       "evaluations": evaluations, "attempts": len(points), "max_evaluations": max_evaluations,
                       "tolerance": tolerance, "evaluated_points": points,
                       "search_method": "bounded_log_strength_grid_then_measured_bracket_refinement",
                       "search_strength_bounds": [0., _MAX_STRENGTH], "direction_frozen_on": "reference_window"})
        return result

    if zero_reference:
        return finish("zero_reference")
    if not indices:
        return finish("no_targets")
    if target == 0:
        return finish()
    _, _, zero_roles = _role_errors(window, null, indices, role)
    if len(zero_roles) == len(role):
        return finish("zero_input_energy")
    if max_evaluations == 1:
        return finish("unmatched_evaluation_limit")
    if kind == "zero":
        measure(1.)
        return finish(None if any(is_match(point["error"]) for point in points) else "unmatched_discrete_ablation")

    budget = max_evaluations - 1
    grid_count = min(9, max(2, (budget + 1) // 2)) if budget > 1 else 1
    grid = [1.] if grid_count == 1 else [
        10 ** (math.log10(_MIN_GRID_STRENGTH) +
               i * math.log10(_MAX_STRENGTH / _MIN_GRID_STRENGTH) / (grid_count - 1)) for i in range(grid_count)]
    for scale in grid:
        if len(points) >= max_evaluations:
            break
        evaluated = measure(scale)
        if evaluated is not None and is_match(points[-1]["error"]):
            return finish()
    while len(points) < max_evaluations:
        strength = _next_strength(points, target)
        if strength is None:
            break
        evaluated = measure(strength)
        if evaluated is not None and is_match(points[-1]["error"]):
            return finish()
    return finish()


def _serializable(value):
    if isinstance(value, (torch.Tensor, ops.KVVariant)):
        return None
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()
                if not isinstance(item, (torch.Tensor, ops.KVVariant))}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value if not isinstance(item, (torch.Tensor, ops.KVVariant))]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def calibration_metadata(result: dict) -> dict:
    """Return JSON-safe calibration/branch metadata, excluding tensor payloads."""
    metadata = _serializable(result)
    variant = result.get("variant")
    if isinstance(variant, ops.KVVariant):
        metadata["intervention_details"] = _serializable(variant.details)
    return metadata
