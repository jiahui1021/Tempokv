"""Prompt-clustered descriptive statistics without optional scientific packages."""
from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import mean
from typing import Any, Callable, Iterable


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values: Iterable[float], q: float) -> float | None:
    data = sorted(float(x) for x in values if finite(x))
    if not 0 <= q <= 1:
        raise ValueError("quantile must be in [0,1]")
    if not data:
        return None
    position = (len(data) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    return data[lo] + (data[hi] - data[lo]) * (position - lo)


def bootstrap_mean(rows: Iterable[dict[str, Any]], key: str, *, cluster: str = "prompt_id", samples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Equal prompt weights; all seeds/rows stay inside their original cluster."""
    if samples < 0:
        raise ValueError("bootstrap samples must be nonnegative")
    values: dict[str, list[float]] = defaultdict(list)
    missing = 0
    for row in rows:
        if finite(row.get(key)):
            values[str(row[cluster])].append(float(row[key]))
        else:
            missing += 1
    prompt_means = [mean(x) for x in values.values()]
    if not prompt_means:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_prompts": 0, "n_values": 0, "n_missing": missing}
    estimates: list[float] = []
    if len(prompt_means) > 1 and samples:
        rng = random.Random(seed)
        estimates = [mean(rng.choices(prompt_means, k=len(prompt_means))) for _ in range(samples)]
    return {
        "mean": mean(prompt_means), "ci_low": percentile(estimates, 0.025), "ci_high": percentile(estimates, 0.975),
        "n_prompts": len(prompt_means), "n_values": sum(map(len, values.values())), "n_missing": missing,
        "aggregation": "equal_prompt_mean", "bootstrap_unit": cluster, "bootstrap_samples": samples,
    }


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    lo = 0
    while lo < len(ordered):
        hi = lo + 1
        while hi < len(ordered) and values[ordered[hi]] == values[ordered[lo]]:
            hi += 1
        rank = (lo + hi - 1) / 2.0 + 1
        for index in ordered[lo:hi]:
            ranks[index] = rank
        lo = hi
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    pairs = [(float(a), float(b)) for a, b in zip(x, y, strict=True) if finite(a) and finite(b)]
    if len(pairs) < 3:
        return None
    rx, ry = average_ranks([p[0] for p in pairs]), average_ranks([p[1] for p in pairs])
    ax, ay = mean(rx), mean(ry)
    sx = sum((v - ax) ** 2 for v in rx)
    sy = sum((v - ay) ** 2 for v in ry)
    if sx == 0 or sy == 0:
        return None
    return sum((a - ax) * (b - ay) for a, b in zip(rx, ry, strict=True)) / math.sqrt(sx * sy)


def correlation(rows: Iterable[dict[str, Any]], x: str, y: str, *, samples: int = 1000, seed: int = 0, cluster: str = "prompt_id") -> dict[str, Any]:
    valid = [r for r in rows if finite(r.get(x)) and finite(r.get(y))]
    rho = spearman([r[x] for r in valid], [r[y] for r in valid])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        groups[str(row[cluster])].append(row)
    intervals = []
    if len(groups) > 2 and rho is not None and samples:
        rng = random.Random(seed)
        keys = list(groups)
        for _ in range(samples):
            drawn = [r for k in rng.choices(keys, k=len(keys)) for r in groups[k]]
            value = spearman([r[x] for r in drawn], [r[y] for r in drawn])
            if value is not None:
                intervals.append(value)
    return {"x": x, "y": y, "rho": rho, "n": len(valid), "n_prompts": len(groups),
            "ci_low": percentile(intervals, 0.025), "ci_high": percentile(intervals, 0.975),
            "method": "spearman_average_ties", "bootstrap_unit": cluster,
            "missing_reason": None if rho is not None else "fewer_than_three_pairs_or_constant_variable"}


def grouped_summary(rows: Iterable[dict[str, Any]], group_keys: list[str], metrics: list[str], *, samples: int = 1000, seed: int = 0) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in group_keys)].append(row)
    result = []
    for group, members in sorted(grouped.items(), key=lambda item: str(item[0])):
        entry = dict(zip(group_keys, group, strict=True))
        entry["n_videos"] = len({(str(r["prompt_id"]), r.get("seed", 0), r.get("sample_id", 0)) for r in members})
        entry["metrics"] = {k: bootstrap_mean(members, k, samples=samples, seed=seed) for k in metrics}
        result.append(entry)
    return result
