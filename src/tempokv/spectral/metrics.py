from __future__ import annotations

import math
import numpy as np
import torch


def relative_error(reference: torch.Tensor, actual: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.vector_norm(reference.float() - actual.float()) /
                 (torch.linalg.vector_norm(reference.float()) + eps))


def nmse(reference: torch.Tensor, actual: torch.Tensor, eps: float = 1e-12) -> float:
    return float((reference.float() - actual.float()).square().sum() /
                 (reference.float().square().sum() + eps))


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2
        i = j
    return ranks


def score_metrics(score, oracle, fraction: float = 0.25) -> dict:
    """Pairwise finite candidates only; recall never claims a global candidate set."""
    score, oracle = np.asarray(score, dtype=float), np.asarray(oracle, dtype=float)
    valid = np.isfinite(score) & np.isfinite(oracle)
    score, oracle = score[valid], oracle[valid]
    n = len(score)
    rho = recall = float("nan")
    if n >= 2:
        sr, yr = _ranks(score), _ranks(oracle)
        if np.std(sr) > 0 and np.std(yr) > 0:
            rho = float(np.corrcoef(sr, yr)[0, 1])
        # Ties make a top-k boundary arbitrary; do not silently report tie-break recall.
        k = max(1, math.ceil(n * fraction))
        ss, ys = np.sort(score)[::-1], np.sort(oracle)[::-1]
        if k == n or (ss[k - 1] > ss[k] and ys[k - 1] > ys[k]):
            si = set(np.argsort(-score, kind="stable")[:k])
            yi = set(np.argsort(-oracle, kind="stable")[:k])
            recall = len(si & yi) / k
    return {"spearman": rho, "topk_recall_sampled": recall, "finite_candidates": n,
            "topk_fraction": fraction, "candidate_scope": "sampled_candidate_set"}


def standardize(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    result = np.full_like(x, np.nan)
    if valid.any():
        std = x[valid].std()
        result[valid] = (x[valid] - x[valid].mean()) / std if std > 0 else 0.0
    return result
