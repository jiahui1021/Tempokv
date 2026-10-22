"""Metrics and prompt/seed-clustered uncertainty, never head-level significance."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from tempokv.spectral.metrics import nmse, relative_error, score_metrics, standardize


def clustered_summary(rows, metric, *, group_fields=("experiment_id", "strategy"), seed=0, bootstrap=1000):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        try: value = float(row[metric])
        except (KeyError, TypeError, ValueError): continue
        if not math.isfinite(value) or row.get("status", "measured") != "measured": continue
        key = tuple(row.get(x) for x in group_fields)
        unit = (row.get("prompt_index"), row.get("seed"))
        groups[key][unit].append(value)
    result = []
    for key, clusters in sorted(groups.items(), key=lambda x: str(x[0])):
        values = np.asarray([np.mean(v) for v in clusters.values()])
        low = high = None
        if len(values) >= 2:
            rng = np.random.default_rng(seed)
            means = values[rng.integers(len(values), size=(bootstrap, len(values)))].mean(axis=1)
            low, high = [float(x) for x in np.quantile(means, [.025, .975])]
        result.append({**dict(zip(group_fields, key)), "metric": metric, "mean": float(values.mean()),
            "median": float(np.median(values)), "std": float(values.std()), "ci_low": low, "ci_high": high,
            "independent_prompt_seed_units": len(values), "raw_observations": sum(map(len, clusters.values())),
            "aggregation": "within_prompt_seed_mean_then_cluster_bootstrap"})
    return result


def comparable_correlations(scores, errors, topk_fraction=.25):
    names = sorted(scores)
    arrays = {key: np.asarray(scores[key], dtype=float) for key in names}
    target = np.asarray(errors, dtype=float)
    common = np.isfinite(target)
    for values in arrays.values():
        if values.shape != target.shape: raise ValueError("scores and oracle need the same sampled candidates")
        common &= np.isfinite(values)
    return [{"score_type": key, **score_metrics(np.where(common, value, np.nan), target, topk_fraction),
             "sampled_candidates": len(target), "excluded_missing_candidates": int((~common).sum())}
            for key, value in arrays.items()]
