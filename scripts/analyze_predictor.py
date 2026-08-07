#!/usr/bin/env python3
"""Analyze the TempoKV predictor on a clean trace.

Reproduces the codec's closed-loop K/V reconstruction (matching
fitting._reconstruct_*_trace) while capturing per-frame prediction,
residual, and rebuilt tensors, then reports per-layer:

  * prediction energy ratio     ||pred||^2 / ||orig||^2      (frame >= 1)
  * residual energy ratio       ||orig - pred||^2 / ||orig||^2
  * prediction-explained share  1 - residual_ratio
  * reconstruction error        ||rebuilt - orig||^2 / ||orig||^2  (all frames)
  * attention output difference (causal softmax, clean K used as query proxy)

The trace carries no queries, so attention uses the clean K as the query
stand-in; Q is therefore unaffected by reconstruction and only K/V errors
propagate into the output, which is the quantity of interest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.config import CodecConfig  # noqa: E402
from tempokv.fitting import _quantization_roundtrip  # noqa: E402
from tempokv.parameters import CrossKVParameters, TemporalParameters  # noqa: E402


def _load_config(path: Path) -> CodecConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "codec" in payload:
        if len(payload) != 1:
            raise ValueError("'codec' must be the only top-level config object")
        payload = payload["codec"]
    return CodecConfig(**payload)


def reconstruct_k(
    keys: torch.Tensor, temporal: TemporalParameters, config: CodecConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-loop K reconstruction; returns (pred, residual, rebuilt)."""
    L, S, T, H, D = keys.shape
    kh = temporal.history_length
    pred = torch.zeros_like(keys)
    resid = torch.zeros_like(keys)
    rebuilt = torch.empty_like(keys)
    rebuilt[:, :, 0] = _quantization_roundtrip(
        keys[:, :, 0], bits=config.anchor_bits, config=config, symmetric=True
    )
    for frame in range(1, T):
        order = min(frame, kh)
        p = (
            temporal.history_bias[:, order - 1]
            .view(L, 1, H, D)
            .expand_as(keys[:, :, frame])
            .clone()
        )
        selected = temporal.history_weights[:, order - 1, :order]
        for lag in range(1, order + 1):
            coeff = selected[:, lag - 1].view(L, 1, H, D)
            p = p + rebuilt[:, :, frame - lag] * coeff
        r = keys[:, :, frame].float() - p
        r_rebuilt = _quantization_roundtrip(
            r,
            bits=config.k_bits,
            config=config,
            symmetric=config.residual_mode == "symmetric",
        )
        pred[:, :, frame] = p
        resid[:, :, frame] = r
        rebuilt[:, :, frame] = p + r_rebuilt
    return pred, resid, rebuilt


def reconstruct_v(
    keys: torch.Tensor,
    values: torch.Tensor,
    k_rebuilt: torch.Tensor,
    cross: CrossKVParameters,
    config: CodecConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-loop V reconstruction; returns (pred, residual, rebuilt)."""
    L, S, T, H, D = values.shape
    vh = cross.history_length
    pred = torch.zeros_like(values)
    resid = torch.zeros_like(values)
    rebuilt = torch.empty_like(values)
    rebuilt[:, :, 0] = _quantization_roundtrip(
        values[:, :, 0], bits=config.anchor_bits, config=config, symmetric=True
    )
    weight = cross.weight
    hist_weights = cross.history_weights
    bias = cross.bias.view(L, 1, H, D)
    for frame in range(1, T):
        p = torch.einsum("lshi,lhij->lshj", k_rebuilt[:, :, frame].float(), weight)
        order = min(frame, vh)
        for lag in range(1, order + 1):
            p = p + torch.einsum(
                "lshi,lhij->lshj",
                rebuilt[:, :, frame - lag].float(),
                hist_weights[:, lag - 1],
            )
        p = p + bias
        r = values[:, :, frame].float() - p
        r_rebuilt = _quantization_roundtrip(
            r,
            bits=config.v_bits,
            config=config,
            symmetric=config.residual_mode == "symmetric",
        )
        pred[:, :, frame] = p
        resid[:, :, frame] = r
        rebuilt[:, :, frame] = p + r_rebuilt
    return pred, resid, rebuilt


def energy_ratio(num: torch.Tensor, den: torch.Tensor) -> float:
    return float((num.square().sum().item() / den.square().sum().item()))


def per_layer_ratio(num: torch.Tensor, den: torch.Tensor) -> list[float]:
    n = num.reshape(num.shape[0], -1).square().sum(dim=1)
    d = den.reshape(den.shape[0], -1).square().sum(dim=1)
    return (n / d).tolist()


def attention_output_diff(
    clean_keys: torch.Tensor,
    clean_values: torch.Tensor,
    rebuilt_keys: torch.Tensor,
    rebuilt_values: torch.Tensor,
) -> tuple[float, list[float]]:
    """Relative output-energy error of causal softmax attention.

    Clean K is used as the query for both branches so only K/V reconstruction
    errors propagate into the output.
    """
    L, S, T, H, D = clean_keys.shape
    flat = lambda t: t.reshape(-1, T, H, D).transpose(0, 2).reshape(L * S * H, T, D)  # noqa: E731
    Q = flat(clean_keys)
    Kc, Vc = flat(clean_keys), flat(clean_values)
    Kr, Vr = flat(rebuilt_keys), flat(rebuilt_values)
    scale = float(D) ** 0.5
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=Q.device), diagonal=1)

    scores_c = torch.bmm(Q, Kc.transpose(1, 2)) / scale
    scores_c = scores_c.masked_fill(mask.unsqueeze(0), -float("inf"))
    Pc = scores_c.softmax(dim=-1)
    Oc = torch.bmm(Pc, Vc)

    scores_r = torch.bmm(Q, Kr.transpose(1, 2)) / scale
    scores_r = scores_r.masked_fill(mask.unsqueeze(0), -float("inf"))
    Pr = scores_r.softmax(dim=-1)
    Or = torch.bmm(Pr, Vr)

    O_f = Oc.reshape(L, S * H, T, D)
    Or_f = Or.reshape(L, S * H, T, D)
    err = (O_f - Or_f).reshape(L, -1).square().sum(dim=1)
    base = O_f.reshape(L, -1).square().sum(dim=1)
    per_layer = (err / base).tolist()
    pooled = float(err.sum().item() / base.sum().item())
    return pooled, per_layer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("parameters/self_forcing_trace.npz"))
    parser.add_argument("--config", type=Path, default=Path("configs/int2.json"))
    parser.add_argument("--k-params", type=Path, default=Path("parameters/k_temporal.npz"))
    parser.add_argument("--v-params", type=Path, default=Path("parameters/v_cross_kv.npz"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.trace, allow_pickle=False) as archive:
        keys = torch.from_numpy(np.array(archive["keys"], copy=True))
        values = torch.from_numpy(np.array(archive["values"], copy=True))
    config = _load_config(args.config)
    temporal = TemporalParameters.load(args.k_params)
    cross = CrossKVParameters.load(args.v_params)

    k_pred, k_res, k_rebuilt = reconstruct_k(keys, temporal, config)
    v_pred, v_res, v_rebuilt = reconstruct_v(
        keys, values, k_rebuilt, cross, config
    )

    # Prediction/residual stats on frames >= 1 (frame 0 is an anchor).
    k_pred_ratio = per_layer_ratio(k_pred[:, :, 1:], keys[:, :, 1:])
    k_res_ratio = per_layer_ratio(k_res[:, :, 1:], keys[:, :, 1:])
    k_recon_ratio = per_layer_ratio(k_rebuilt - keys, keys)
    v_pred_ratio = per_layer_ratio(v_pred[:, :, 1:], values[:, :, 1:])
    v_res_ratio = per_layer_ratio(v_res[:, :, 1:], values[:, :, 1:])
    v_recon_ratio = per_layer_ratio(v_rebuilt - values, values)

    attn_diff, attn_per_layer = attention_output_diff(
        keys, values, k_rebuilt, v_rebuilt
    )

    report = {
        "config": {k: getattr(config, k) for k in (
            "k_bits", "v_bits", "anchor_bits", "group_size", "stride",
            "residual_mode", "k_history_length", "v_history_length",
        )},
        "trace": {
            "shape": list(keys.shape),
            "layers": keys.shape[0],
            "samples": keys.shape[1],
            "frames": keys.shape[2],
            "heads": keys.shape[3],
            "dim": keys.shape[4],
        },
        "k_history_length": temporal.history_length,
        "v_history_length": cross.history_length,
        "stats": {
            "k": {
                "pred_energy_ratio": k_pred_ratio,
                "residual_energy_ratio": k_res_ratio,
                "prediction_explained": [1.0 - r for r in k_res_ratio],
                "reconstruction_error": k_recon_ratio,
                "pooled": {
                    "pred_energy_ratio": energy_ratio(k_pred[:, :, 1:], keys[:, :, 1:]),
                    "residual_energy_ratio": energy_ratio(k_res[:, :, 1:], keys[:, :, 1:]),
                    "prediction_explained": 1.0 - energy_ratio(k_res[:, :, 1:], keys[:, :, 1:]),
                    "reconstruction_error": energy_ratio(k_rebuilt - keys, keys),
                },
            },
            "v": {
                "pred_energy_ratio": v_pred_ratio,
                "residual_energy_ratio": v_res_ratio,
                "prediction_explained": [1.0 - r for r in v_res_ratio],
                "reconstruction_error": v_recon_ratio,
                "pooled": {
                    "pred_energy_ratio": energy_ratio(v_pred[:, :, 1:], values[:, :, 1:]),
                    "residual_energy_ratio": energy_ratio(v_res[:, :, 1:], values[:, :, 1:]),
                    "prediction_explained": 1.0 - energy_ratio(v_res[:, :, 1:], values[:, :, 1:]),
                    "reconstruction_error": energy_ratio(v_rebuilt - values, values),
                },
            },
        },
        "attention_output": {
            "note": "clean K used as query proxy; causal softmax, mean over frames/samples/heads",
            "relative_output_error": attn_diff,
            "per_layer": attn_per_layer,
        },
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    def row(label: str, vals: list[float]) -> str:
        return f"{label:22s} mean={sum(vals)/len(vals):7.4f}  [{" ".join(f"{v:6.3f}" for v in vals[:15])}...]"

    print(rendered)
    print("\n--- K (layers, frames>=1) ---")
    print(row("pred energy ratio", k_pred_ratio))
    print(row("residual ratio", k_res_ratio))
    print(row("prediction explained", [1 - r for r in k_res_ratio]))
    print(row("reconstruction error", k_recon_ratio))
    print("\n--- V (layers, frames>=1) ---")
    print(row("pred energy ratio", v_pred_ratio))
    print(row("residual ratio", v_res_ratio))
    print(row("prediction explained", [1 - r for r in v_res_ratio]))
    print(row("reconstruction error", v_recon_ratio))
    print("\n--- attention output relative error ---")
    print(row("attn output error", attn_per_layer))


if __name__ == "__main__":
    main()
