"""Calibration-data low-rank regression; SVD is initialization only."""
from __future__ import annotations

from dataclasses import replace
import torch

from rebuttal.adapters.experimental_codec import ResearchCross, group_count, group_index, encode_tensor


@torch.no_grad()
def fit_low_rank_regression(xx: torch.Tensor, xy: torch.Tensor, *, rank: int,
                            ridge: float = 0.1, steps: int = 30):
    """Fit A B against sufficient statistics with a dense-map ridge prior.

    The objective is tr(W' XX W)-2 tr(W' XY)+lambda||W-I||².
    Unlike truncating W, alternating updates account for the calibration input
    covariance. The objective log omits the target-energy constant.
    """
    dimension = xx.shape[0]
    if not 1 <= rank <= dimension or steps < 1 or ridge < 0:
        raise ValueError("invalid low-rank regression controls")
    penalty = ridge * xx.diagonal().mean().clamp_min(1e-20)
    system = xx + torch.eye(dimension, device=xx.device, dtype=xx.dtype) * penalty
    target = xy + torch.eye(dimension, device=xx.device, dtype=xx.dtype) * penalty
    dense = torch.linalg.lstsq(system, target).solution
    u, singular, vh = torch.linalg.svd(dense, full_matrices=False)
    a = u[:, :rank] * singular[:rank].sqrt()[None]
    b = singular[:rank].sqrt()[:, None] * vh[:rank]

    def objective(a, b):
        w = a @ b
        return float((w * (system @ w)).sum() - 2 * (w * target).sum())

    log = [{"step": 0, "objective_without_target_constant": objective(a, b), "initialization": "dense_svd"}]
    for step in range(steps):
        b = torch.linalg.lstsq(a.T @ system @ a, a.T @ target).solution
        # Solve system A (B B') = target B' without forming a Kronecker matrix.
        a = torch.linalg.lstsq(system, target @ b.T).solution
        a = torch.linalg.lstsq(b @ b.T, a.T).solution.T
        value = objective(a, b)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("non-finite low-rank fit")
        log.append({"step": step + 1, "objective_without_target_constant": value})
    return a.float(), b.float(), log


@torch.no_grad()
def fit_cross_architecture(reconstructed_keys: torch.Tensor, target: torch.Tensor,
                           baseline: ResearchCross, config, *, kind: str, rank: int | None = None,
                           rounds: int = 2, ridge: float = 0.1, als_steps: int = 30):
    """Change only the current-K map; hold V-history coefficients fixed.

    Closed-loop V history is recomputed for each variant and fit round. Current
    projection and bias share exactly the production layer/head grouping.
    """
    if kind not in {"dense", "diagonal", "low_rank"}:
        raise ValueError(f"unsupported cross architecture {kind}")
    if rounds < 1 or ridge < 0:
        raise ValueError("invalid cross fit controls")
    layers, _, frames, heads, dimension = target.shape
    count = group_count(baseline.sharing, layers, heads)
    if kind == "low_rank" and (rank is None or not 1 <= rank <= dimension):
        raise ValueError("low rank must be in [1, D]")
    current = baseline
    cfg = replace(config, stride=1)
    logs = []
    for round_index in range(rounds):
        rebuilt_v = torch.empty_like(target)
        for layer in range(layers):
            _, result = encode_tensor(target[layer].permute(0, 2, 1, 3), current, cfg, bits=cfg.v_bits,
                                      layer=layer, source=reconstructed_keys[layer].permute(0, 2, 1, 3))
            rebuilt_v[layer] = result.permute(0, 2, 1, 3).float()
        sx = torch.zeros(count, dimension, dtype=torch.float64, device=target.device)
        sy = torch.zeros_like(sx)
        xx = torch.zeros(count, dimension, dimension, dtype=torch.float64, device=target.device)
        xy = torch.zeros_like(xx)
        counts = [0] * count
        for layer in range(layers):
            y = target[layer, :, 1:].double().clone()
            for lag in range(1, baseline.history_length + 1):
                if frames <= lag:
                    break
                previous = rebuilt_v[layer, :, :frames - lag].double()
                hw = baseline.history_weights[layer, lag - 1].double()
                if baseline.history_diagonal:
                    y[:, lag - 1:] -= previous * hw[None, None]
                else:
                    y[:, lag - 1:] -= torch.einsum("nthi,hij->nthj", previous, hw)
            for head in range(heads):
                group = group_index(baseline.sharing, layer, head, heads)
                x = reconstructed_keys[layer, :, 1:, head].reshape(-1, dimension).double()
                yy = y[:, :, head].reshape(-1, dimension)
                sx[group] += x.sum(0); sy[group] += yy.sum(0)
                xx[group] += x.T @ x; xy[group] += x.T @ yy; counts[group] += x.shape[0]
        weights, biases, factor_a, factor_b = [], [], [], []
        for group in range(count):
            n = counts[group]
            if not n:
                raise ValueError("projection group has no calibration samples")
            covariance = xx[group] - torch.outer(sx[group], sx[group]) / n
            cross = xy[group] - torch.outer(sx[group], sy[group]) / n
            penalty = ridge * covariance.diagonal().mean().clamp_min(1e-20)
            if kind == "low_rank":
                a, b, log = fit_low_rank_regression(covariance, cross, rank=rank, ridge=ridge, steps=als_steps)
                factor_a.append(a); factor_b.append(b)
                w = a.double() @ b.double()
                logs.append({"round": round_index + 1, "group": group, "rank": rank, "optimization": log})
            elif kind == "diagonal":
                diagonal = (cross.diagonal() + penalty) / (covariance.diagonal() + penalty).clamp_min(1e-20)
                weights.append(diagonal.float()); w = torch.diag(diagonal)
            else:
                w = torch.linalg.lstsq(covariance + torch.eye(dimension, device=target.device) * penalty,
                                      cross + torch.eye(dimension, device=target.device) * penalty).solution
                weights.append(w.float())
            biases.append((sy[group] / n - sx[group] / n @ w).float())
        current = ResearchCross(kind, baseline.sharing, torch.stack(biases), baseline.history_weights.clone(), layers, heads,
                                weight=torch.stack(weights) if weights else None,
                                factor_a=torch.stack(factor_a) if factor_a else None,
                                factor_b=torch.stack(factor_b) if factor_b else None)
    return current, logs
