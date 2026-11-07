"""Matched diagonal/full temporal maps, with no current-K or CrossKV inputs."""
from __future__ import annotations

import torch

from rebuttal.adapters.experimental_codec import ResearchTemporal
from rebuttal.experiments.architecture.temporal import identity_predictor


def parameter_bytes(model):
    return sum(t.numel()*t.element_size() for t in (model.weights, model.bias))


@torch.no_grad()
def replay(target, model, codec, *, bits, diagnostics=True):
    """Reconstruct [records,frames,heads,dim], using quantized causal history."""
    from tempokv.fitting import _anchor_roundtrip, _quantization_roundtrip
    if target.ndim != 4 or target.shape[-2] != model.heads or model.layers != 1:
        raise ValueError("temporal matrix replay expects one [N,T,H,D] layer")
    dtype = getattr(torch, codec.reconstruction_dtype)
    rebuilt = torch.empty_like(target, dtype=dtype)
    rebuilt[:, 0] = _anchor_roundtrip(target[:, 0], codec).to(dtype)
    frames = []
    for frame in range(1, target.shape[1]):
        history = [rebuilt[:, frame-lag].unsqueeze(-2) for lag in range(1, min(frame, model.history_length)+1)]
        prediction = model.predict(history, layer=0).squeeze(-2)
        residual = target[:, frame].float()-prediction
        residual_q = _quantization_roundtrip(residual, bits=bits, config=codec,
                                            symmetric=codec.residual_mode == "symmetric")
        rebuilt[:, frame] = (prediction+residual_q).to(dtype)
        if diagnostics:
            frames.append({"frame": frame, "signal_sse": target[:, frame].double().square().sum().item(),
                           "prediction_sse": residual.double().square().sum().item(),
                           "quantization_sse": (residual.double()-residual_q.double()).square().sum().item(),
                           "error_sse": (target[:, frame].double()-rebuilt[:, frame].double()).square().sum().item(),
                           "numel": target[:, frame].numel()})
    return rebuilt, frames


def _solve(chol_input, rhs):
    return torch.cholesky_solve(rhs, torch.linalg.cholesky(chol_input))


@torch.no_grad()
def fit_round(reconstructed, target, initial, *, ridge, max_gain):
    """Same centered SSE + persistence ridge + gain constraint in both families.

    One head-level ridge scale is used for both structures. On a diagonal map,
    the full incoming per-output L1 constraint reduces exactly to sum over lags.
    """
    if initial.kind not in {"channel_linear", "dense_linear"} or ridge <= 0 or max_gain <= 0:
        raise ValueError("expected channel/dense temporal with positive ridge and gain cap")
    if reconstructed.shape != target.shape or target.ndim != 4:
        raise ValueError("fit needs matching reconstructed/target [N,T,H,D]")
    frames, heads, dim = target.shape[1:]
    history = initial.history_length
    if frames <= history:
        raise ValueError("temporal history must be shorter than trace")
    weights, biases = torch.zeros_like(initial.weights), torch.zeros_like(initial.bias)
    log = []
    for order in range(1, history+1):
        stop = order+1 if order < history else frames
        x = torch.stack([reconstructed[:, order-lag:stop-lag] for lag in range(1, order+1)], dim=-2)
        x = x.reshape(-1, heads, order, dim).permute(1, 0, 2, 3).double()
        y = target[:, order:stop].reshape(-1, heads, dim).permute(1, 0, 2).double()
        xm, ym = x.mean(1), y.mean(1)
        xc, yc = x-xm[:, None], y-ym[:, None]
        penalty = ridge*xc.square().sum(dim=1).mean(dim=(1, 2)).clamp_min(1e-20)
        if initial.kind == "channel_linear":
            gram = torch.einsum("hnpd,hnqd->hdpq", xc, xc)
            rhs = torch.einsum("hnpd,hnd->hdp", xc, yc)
            prior = x.new_zeros(order); prior[0] = 1
            system = gram+penalty[:, None, None, None]*torch.eye(order, device=x.device)
            coefficient = _solve(system, (rhs+penalty[:, None, None]*prior).unsqueeze(-1)).squeeze(-1)
            gain = coefficient.abs().sum(-1).clamp_min(1e-30)
            coefficient *= torch.clamp(max_gain/gain, max=1.0)[..., None]
            matrices = coefficient.permute(0, 2, 1)
            weights[:, order-1, :order] = matrices.float()
            bias = ym-(xm*matrices).sum(1)
        else:
            flat = xc.flatten(2)
            gram, rhs = flat.transpose(-2, -1)@flat, flat.transpose(-2, -1)@yc
            prior = x.new_zeros(order*dim, dim); prior[:dim] = torch.eye(dim, device=x.device)
            system = gram+penalty[:, None, None]*torch.eye(order*dim, device=x.device)
            coefficient = _solve(system, rhs+penalty[:, None, None]*prior)
            matrices = coefficient.reshape(heads, order, dim, dim)
            gain = matrices.abs().sum(dim=(1, 2)).clamp_min(1e-30)
            matrices *= torch.clamp(max_gain/gain, max=1.0)[:, None, None, :]
            weights[:, order-1, :order] = matrices.float()
            bias = ym-(xm.flatten(1).unsqueeze(1)@matrices.reshape(heads, order*dim, dim)).squeeze(1)
        biases[:, order-1] = bias.float()
        log.append({"order": order, "tokens_per_head": x.shape[1], "ridge_penalty_per_head": penalty.cpu().tolist()})
    return ResearchTemporal(initial.kind, initial.sharing, weights, biases, 1, heads), log


@torch.no_grad()
def fit(target, codec, *, kind, history, bits, rounds=2, ridge=1.0, max_gain=1.0):
    if rounds < 1:
        raise ValueError("temporal fitting needs at least one round")
    model = identity_predictor(1, target.shape[-2], target.shape[-1], history,
                               "layer_head", device=target.device, kind=kind)
    logs = []
    for step in range(rounds):
        rebuilt, _ = replay(target, model, codec, bits=bits, diagnostics=False)
        model, control = fit_round(rebuilt, target, model, ridge=ridge, max_gain=max_gain)
        logs.append({"round": step+1, "history_source": "reconstructed", "warmup": control})
    return model, logs


def model_state(model):
    return {"kind": model.kind, "sharing": model.sharing, "weights": model.weights.cpu(),
            "bias": model.bias.cpu(), "layers": model.layers, "heads": model.heads}


def load_model(state, device):
    return ResearchTemporal(state["kind"], state["sharing"], state["weights"].to(device),
                            state["bias"].to(device), state["layers"], state["heads"])
