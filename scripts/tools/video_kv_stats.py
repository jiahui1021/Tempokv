"""Compute KV activation stats from the stored clean video traces (CPU).

Usage: WORKSPACE=... python video_kv_stats.py OUT_JSON
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kv_activation_stats import (aggregate, frame_stats, head_vector_dissim,
                                 profile_dissim, token_stats)

ROOT = Path(os.environ["WORKSPACE"]) / "Tempokv/data"


def per_layer_head(arr: np.ndarray, model: str):
    """Yield (layer, head, frame_token_matrix[F, S, D])."""
    if model == "longcat":  # (L, H, T, S, D)
        for l in range(arr.shape[0]):
            for h in range(arr.shape[1]):
                yield l, h, arr[l, h]
    else:  # self-forcing (L, F, B, sH, sW, H, D)
        L, F, _, sH, sW, H, D = arr.shape
        for l in range(L):
            for h in range(H):
                yield l, h, arr[l, :, 0, :, :, h, :].reshape(F, sH * sW, D)


def run_npz(path: Path, model: str) -> dict:
    z = np.load(path)
    k, v = z["keys"], z["values"]
    if model == "longcat":
        L, H, T, S, D = k.shape
    else:
        L, T, _, sH, sW, H, D = k.shape
        S = sH * sW
    out = {"model": path.parent.parent.name if False else "", "shape": dict(L=L, H=H, T=T, S=S, D=D),
           "trace": str(path)}
    stats = {}
    scales = {}
    for role, arr in (("k", k), ("v", v)):
        per = []
        sc = np.zeros((L, H, D))
        for l, h, t in per_layer_head(arr, "longcat" if model == "longcat" else "sf"):
            flat = t.reshape(-1, D).astype(np.float64)
            s = token_stats(flat)
            s["_n"] = flat.shape[0]
            s.update(frame_stats(flat, T, S))
            per.append(s)
            sc[l, h] = t.reshape(-1, D).std(axis=0)
        stats[role] = aggregate(per)
        scales[role] = sc
    out["layers"] = stats
    out["heterogeneity"] = {
        f"{axis}_profile_dissim_{role}": profile_dissim(
            scales[role].reshape(L, -1) if axis == "layer" else np.concatenate(
                [scales[role][:, h] for h in range(H)], axis=0
            )
        )
        for role in ("k", "v")
        for axis in ("layer", "head")
    }
    # per-layer table for the figure
    layer_rows = {role: [] for role in ("k", "v")}
    for role, arr in (("k", k), ("v", v)):
        acc: dict[int, list] = {}
        hmeans: dict[int, list] = {}
        for l, h, t in per_layer_head(arr, "longcat" if model == "longcat" else "sf"):
            flat = t.reshape(-1, D).astype(np.float64)
            s = token_stats(flat)
            s["_n"] = flat.shape[0]
            s.update(frame_stats(flat, T, S))
            acc.setdefault(l, []).append(s)
            hmeans.setdefault(l, []).append(flat.mean(axis=0))
        layer_rows[role] = [aggregate(acc[l]) for l in range(L)]
        out.setdefault("heterogeneity", {})[f"head_vector_dissim_{role}"] = float(
            np.mean([head_vector_dissim(np.stack(hmeans[l])) for l in range(L)]))
    return out


if __name__ == "__main__":
    jobs = [
        ("self-forcing-wan2.1-1.3b", ROOT / "self-forcing/trace_legacy/clean_trace.npz", "sf"),
        ("longcat-video", ROOT / "longcat/trace/clean_trace.npz", "longcat"),
    ]
    res = []
    for name, p, model in jobs:
        print("processing", p, flush=True)
        r = run_npz(p, model)
        r["model"] = name
        res.append(r)
    Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    print("wrote", sys.argv[1])
