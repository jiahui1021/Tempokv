"""Independent temporal connectivity ablation with matched sharing/warm-up."""
from __future__ import annotations

import torch

from rebuttal.adapters.experimental_codec import ResearchTemporal, group_count, group_index, encode_tensor


def identity_predictor(layers: int, heads: int, dimension: int, history: int,
                       sharing: str, *, device="cpu", kind="identity") -> ResearchTemporal:
    if not 1 <= history <= 8:
        raise ValueError("history_length must be between 1 and 8")
    count = group_count(sharing, layers, heads)
    tail = (dimension, dimension) if kind == "dense_linear" else ((dimension,) if kind == "channel_linear" else (1,))
    weights = torch.zeros(count, history, history, *tail, device=device)
    for order in range(history):
        if kind == "dense_linear":
            weights[:, order, 0] = torch.eye(dimension, device=device)
        else:
            weights[:, order, 0] = 1
    return ResearchTemporal(kind, sharing, weights, torch.zeros(count, history, dimension, device=device), layers, heads)


def _members(predictor: ResearchTemporal, group: int):
    return [(layer, head) for layer in range(predictor.layers) for head in range(predictor.heads)
            if group_index(predictor.sharing, layer, head, predictor.heads) == group]


@torch.no_grad()
def fit_temporal_architecture(target: torch.Tensor, initial: ResearchTemporal, config,
                              *, rounds: int = 2, ridge: float = 1.0, max_gain: float = 1.0):
    """Fit scalar/dense maps against decoder-visible history, never clean history.

    Sufficient statistics accumulate per layer/head, bounding host memory by
    a single stream plus the normal equations. Bias is per output channel for
    both maps; scalar only ties the lag coefficients across channels.
    """
    from dataclasses import replace
    predictor = initial
    if predictor.kind == "identity":
        return predictor, []
    if predictor.kind not in {"scalar_linear", "dense_linear"}:
        raise ValueError("channel baseline must use the production fitter")
    if rounds < 1 or ridge < 0 or not 0 < max_gain <= 1:
        raise ValueError("invalid temporal fit controls")
    layers, _, frames, heads, dimension = target.shape
    if predictor.history_length >= frames:
        raise ValueError("history_length must be smaller than calibration trace")
    cfg = replace(config, stride=1)
    history_length = predictor.history_length
    logs = []
    for round_index in range(rounds):
        rebuilt = torch.empty_like(target)
        for layer in range(layers):
            _, x = encode_tensor(target[layer].permute(0, 2, 1, 3), predictor, cfg, bits=cfg.k_bits, layer=layer)
            rebuilt[layer] = x.permute(0, 2, 1, 3).float()
        weights, biases = torch.zeros_like(predictor.weights), torch.zeros_like(predictor.bias)
        for group in range(weights.shape[0]):
            members = _members(predictor, group)
            for order in range(1, history_length + 1):
                stops = [order] if order < history_length else list(range(order, frames))
                feature_count = order if predictor.kind == "scalar_linear" else order * dimension
                sx = torch.zeros(order, dimension, dtype=torch.float64, device=target.device)
                sy = torch.zeros(dimension, dtype=torch.float64, device=target.device)
                xx = torch.zeros(feature_count, feature_count, dtype=torch.float64, device=target.device)
                xy = torch.zeros((feature_count,) if predictor.kind == "scalar_linear" else (feature_count, dimension),
                                 dtype=torch.float64, device=target.device)
                count = 0
                for layer, head in members:
                    for frame in stops:
                        x = torch.stack([rebuilt[layer, :, frame - lag, head] for lag in range(1, order + 1)], 1).double()
                        y = target[layer, :, frame, head].double()
                        count += y.shape[0]; sx += x.sum(0); sy += y.sum(0)
                        if predictor.kind == "scalar_linear":
                            xx += torch.einsum("npd,nqd->pq", x, x)
                            xy += torch.einsum("npd,nd->p", x, y)
                        else:
                            flat = x.flatten(1)
                            xx += flat.T @ flat; xy += flat.T @ y
                if count == 0:
                    raise ValueError("warm-up order has no calibration samples")
                if predictor.kind == "scalar_linear":
                    xx -= sx @ sx.T / count
                    xy -= sx @ (sy / count)
                    prior = xx.new_zeros(order); prior[0] = 1
                else:
                    flat_sum = sx.flatten()
                    xx -= torch.outer(flat_sum, flat_sum) / count
                    xy -= flat_sum[:, None] * sy[None] / count
                    prior = xx.new_zeros(feature_count, dimension); prior[:dimension] = torch.eye(dimension, device=target.device)
                penalty = ridge * (xx.diagonal().mean().clamp_min(1e-20))
                system = xx + torch.eye(feature_count, device=target.device, dtype=torch.float64) * penalty
                rhs = xy + prior * penalty
                coefficient = torch.linalg.lstsq(system, rhs).solution
                if predictor.kind == "scalar_linear":
                    coefficient *= min(1.0, max_gain / max(float(coefficient.abs().sum()), 1e-20))
                    weights[group, order - 1, :order, 0] = coefficient.float()
                    bias = sy / count - (sx / count * coefficient[:, None]).sum(0)
                else:
                    matrices = coefficient.reshape(order, dimension, dimension)
                    # Per-output absolute gain generalizes the shipped diagonal
                    # constraint: for diagonal maps it is exactly sum_lag |w|.
                    # A sum of spectral norms would impose a stricter, different
                    # constraint solely on dense variants.
                    gain = matrices.abs().sum(dim=(0, 1)).clamp_min(1e-20)
                    matrices *= torch.clamp(max_gain / gain, max=1.0)[None, None, :]
                    weights[group, order - 1, :order] = matrices.float()
                    bias = sy / count - (sx.flatten() / count) @ matrices.reshape(feature_count, dimension)
                biases[group, order - 1] = bias.float()
        predictor = ResearchTemporal(predictor.kind, predictor.sharing, weights, biases, layers, heads)
        logs.append({"round": round_index + 1, "history_source": "reconstructed", "sample_frames": frames,
                     "ridge": ridge, "max_gain": max_gain})
    return predictor, logs
