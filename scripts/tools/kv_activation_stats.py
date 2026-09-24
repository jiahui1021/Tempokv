"""Shared KV-cache activation statistics for video (AR diffusion) and LLM caches.

Computed from a per-(layer, head) token matrix ``[n_tokens, head_dim]``:
heterogeneity (channel/norm spread, outliers, kurtosis, dissimilarity) and
redundancy (neighbour similarity, low-rank energy, autocorrelation).
"""

from __future__ import annotations


import numpy as np


def token_stats(x: np.ndarray) -> dict[str, float]:
    """Stats for one (layer, head) token matrix [n, d]."""
    n, d = x.shape
    mean = x.mean(axis=0)
    centered = x - mean
    std = centered.std(axis=0) + 1e-12
    rms = float(np.sqrt((std ** 2).mean()))
    scale = std / rms
    cv = float(scale.std() / scale.mean())
    kurt = float(((centered / std) ** 4).mean())

    norms = np.linalg.norm(centered, axis=-1)
    norm_cv = float(norms.std() / (norms.mean() + 1e-12))
    xn = centered / (norms[:, None] + 1e-12)

    rng = np.random.default_rng(0)
    adj = float((xn[1:] * xn[:-1]).sum(-1).mean())
    far_pairs = rng.integers(0, n, size=(min(8192, n), 2))
    far_pairs = far_pairs[np.abs(far_pairs[:, 0] - far_pairs[:, 1]) > max(16, n // 8)]
    far = float((xn[far_pairs[:, 0]] * xn[far_pairs[:, 1]]).sum(-1).mean()) if len(far_pairs) else float("nan")
    diff_energy = float(((centered[1:] - centered[:-1]) ** 2).mean() / max(1e-12, (centered ** 2).mean()))

    out: dict[str, float] = {
        "cv_channel_std": cv,
        "outlier_max_z": float(np.abs(centered / std).max(axis=0).mean()),
        "kurtosis": kurt,
        "norm_cv": norm_cv,
        "adj_sim": adj,
        "far_sim": far,
        "temporal_diff_energy": diff_energy,
        "rms": rms,
    }
    for lag in (1, 2, 5, 10, 25):
        if n > lag:
            out[f"autocorr{lag}"] = float((xn[:-lag] * xn[lag:]).sum(-1).mean())
    sample = x[: min(4096, n)]
    sv = np.linalg.svd(sample - sample.mean(0), compute_uv=False)
    ev = sv ** 2
    total = float(ev.sum()) + 1e-12
    for k in (1, 5, 10, 20):
        out[f"rank_top{k}_energy"] = float(ev[:k].sum() / total)
    # effective rank (participation ratio of the spectrum) as a fraction of head_dim
    out["participation_ratio"] = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12) / d)
    return out


def frame_stats(x: np.ndarray, frames: int, tokens_per_frame: int) -> dict[str, float]:
    """Spatial/temporal split for video token matrices grouped by frame."""
    n = x.shape[0]
    out: dict[str, float] = {"same_unit_sim": float("nan"), "temporal_adj_lag1_sim": float("nan"),
                             "temporal_adj_lag5_sim": float("nan"), "outlier_frame_jaccard": float("nan")}
    usable = (n // tokens_per_frame) * tokens_per_frame if tokens_per_frame > 1 else 0
    if usable < 2 * tokens_per_frame:
        return out
    f = usable // tokens_per_frame
    v = x[:usable].reshape(f, tokens_per_frame, -1)
    v = v - x[:usable].mean(axis=0)
    norms = np.linalg.norm(v, axis=-1) + 1e-12
    vn = v / norms[..., None]
    i_a, i_b = np.triu_indices(tokens_per_frame, 1)
    same = np.stack([(vn[t, i_a] * vn[t, i_b]).sum(-1) for t in range(f)])
    out["same_unit_sim"] = float(same.mean())
    for lag, key in ((1, "temporal_adj_lag1_sim"), (5, "temporal_adj_lag5_sim")):
        if f > lag:
            out[key] = float(((vn[:-lag] * vn[lag:]).sum(-1)).mean())
    # channel-role stability: top-8 outlier channels per frame, Jaccard between
    # consecutive frames (are the "important dims" the same dims over time?)
    per_tok = (v ** 2).mean(axis=1)  # (f, D) mean energy per channel within frame
    top = np.argsort(-per_tok, axis=1)[:, :8]
    jac = []
    for t in range(f - 1):
        a, b = set(top[t].tolist()), set(top[t + 1].tolist())
        jac.append(len(a & b) / len(a | b))
    out["outlier_frame_jaccard"] = float(np.mean(jac))
    return out


def head_vector_dissim(head_means: np.ndarray) -> float:
    """1 - mean pairwise cosine of per-head token-mean vectors [H, d]."""
    p = head_means / (np.linalg.norm(head_means, axis=1, keepdims=True) + 1e-12)
    g = p @ p.T
    iu = np.triu_indices(len(p), 1)
    return float(1.0 - g[iu].mean())


def profile_dissim(profiles: np.ndarray) -> float:
    """1 - mean pairwise cosine of per-(layer or head) scale profiles."""
    p = profiles.reshape(profiles.shape[0], -1)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)
    g = p @ p.T
    iu = np.triu_indices(len(p), 1)
    return float(1.0 - g[iu].mean())


def aggregate(per_lh: list[dict]) -> dict:
    keys = sorted({k for s in per_lh for k in s} - {"rms", "_n"})
    out = {k: float(np.nanmean([s.get(k, np.nan) for s in per_lh])) for k in keys}
    out["n_tokens_per_lh"] = float(np.mean([s["_n"] for s in per_lh]))
    return out
