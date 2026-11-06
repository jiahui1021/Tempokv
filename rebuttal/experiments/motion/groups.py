"""Frozen BF16-only quantile grouping, preserving magnitude ties."""
from __future__ import annotations
from typing import Any
from rebuttal.common.statistics import percentile


def assign_groups(rows: list[dict[str, Any]], frozen: dict[str, Any] | None = None) -> dict[str, Any]:
    thresholds = {} if frozen is None else dict(frozen["thresholds"])
    cohorts: dict[str, list[dict[str, Any]]] = {}
    seen = set()
    for row in rows:
        # Multiple candidate methods must share exactly one BF16 observation.
        key = (row["model"], str(row["prompt_id"]), row.get("seed", 0), row.get("sample_id", 0))
        if key not in seen:
            cohorts.setdefault(row["model"], []).append(row)
            seen.add(key)
    for model, members in cohorts.items():
        formats = {(r.get("width"), r.get("height"), r.get("fps")) for r in members if all(r.get(k) is not None for k in ("width", "height", "fps"))}
        if len(formats) > 1:
            raise ValueError(f"raw px/frame motion grouping requires one resolution/fps per model cohort: {model}")
        if model not in thresholds:
            values = [r["motion_magnitude"] for r in members]
            thresholds[model] = {"low": percentile(values, 1 / 3), "high": percentile(values, 2 / 3), "n_reference_videos": len(values)}
        bounds = thresholds[model]
        if bounds["low"] is None or bounds["high"] is None:
            raise ValueError(f"cannot group empty motion cohort: {model}")
    assignments = []
    for row in rows:
        low, high = thresholds[row["model"]]["low"], thresholds[row["model"]]["high"]
        magnitude = row["motion_magnitude"]
        row["motion_group"] = "low" if magnitude <= low else "medium" if magnitude <= high else "high"
        assignments.append({k: row.get(k) for k in ("model", "prompt_id", "seed", "sample_id", "motion_magnitude", "motion_group")})
    return {"schema_version": "tempokv.rebuttal.motion_groups.v1", "source": "bf16_only",
            "quantiles": [1 / 3, 2 / 3], "thresholds": thresholds, "assignments": assignments,
            "ties_policy": "equal_magnitudes_stay_together_group_sizes_may_differ"}
