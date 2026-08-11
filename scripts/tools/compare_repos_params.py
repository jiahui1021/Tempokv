#!/usr/bin/env python3
"""Compare qvg (S2++) vs tempokv legacy parameters on the SAME legacy trace.

Loads both parameter files directly (native npz, bypassing the incompatible
loaders) and runs a closed-loop K + V reconstruction on the tempokv legacy
trace using a faithful reimplementation of the shared legacy math, reporting
per-component error so we can see where qvg wins.

K:  shared [P,P,D] temporal weights + [P,D] bias (both repos identical shape)
V:  cross [L,H,D,D] + bias (+ optional diagonal/innovation history)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tempokv.config import CodecConfig  # noqa: E402
from tempokv.quantization import (  # noqa: E402
    dequantize_asymmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    quantize_symmetric,
)


def _roundtrip(v: torch.Tensor, bits: int, group_size: int, scale_dtype, symmetric: bool):
    """Quantize/dequantize a [.., H, D] slice, mirroring tempokv._quantization_roundtrip."""
    original = v.shape
    head, dim = v.shape[-2], v.shape[-1]
    packed_layout = v.float().reshape(-1, head, 1, dim)
    if symmetric:
        q, scale = quantize_symmetric(packed_layout, bits, group_size, scale_dtype)
        return dequantize_symmetric(q, scale, bits, group_size, dim, torch.float32).reshape(original)
    q, scale, zp = quantize_asymmetric(packed_layout, bits, group_size, scale_dtype)
    return dequantize_asymmetric(q, scale, zp, bits, group_size, dim, torch.float32).reshape(original)


def reconstruct_k(keys, weights, bias, config, device):
    """Closed-loop K: anchor + per-unit temporal prediction + residual QDQ."""
    L, E, B, P, F, H, D = keys.shape
    stride = config.stride
    seq = P * F  # tokens across frames
    k = keys.permute(2, 0, 1, 5, 3, 4, 6).reshape(B, L, E, H, F, P, D)
    # -> [B, L, E, H, F, P, D]; concatenate F -> [B,L,E,H, seq, D]
    kf = k.permute(0, 1, 2, 3, 4, 5, 6).reshape(B, L, E, H, seq, D).float()
    rebuilt = torch.empty_like(kf)
    anchor = _roundtrip(kf[..., :stride, :], config.anchor_bits, config.group_size,
                        getattr(torch, config.scale_dtype), True)
    rebuilt[..., :stride, :] = anchor.to(torch.bfloat16).float()
    hist = [anchor]
    for offset in range(stride, seq, stride):
        unit = min(stride, seq - offset)
        order = min(len(hist), int(weights.shape[0]))
        pred = torch.zeros_like(kf[..., offset:offset + unit, :], dtype=torch.float32)
        for lag in range(order):
            pred = pred + hist[lag][..., :unit, :].float() * weights[order - 1, lag].to(device)
        pred = pred + bias[order - 1].to(device)
        resid = kf[..., offset:offset + unit, :].float() - pred
        resid_rebuilt = _roundtrip(resid, config.v_bits, config.group_size,
                                   getattr(torch, config.scale_dtype), False)
        cur = (pred + resid_rebuilt).to(torch.bfloat16).float()
        rebuilt[..., offset:offset + unit, :] = cur
        hist = [cur, *hist[: order - 1]]
    # back to [L,E,B,P,F,H,D]
    return rebuilt.reshape(B, L, E, H, F, P, D).permute(1, 2, 0, 5, 3, 4, 6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--tempokv-k", required=True)
    parser.add_argument("--tempokv-v", required=True)
    parser.add_argument("--qvg-k", required=True)
    parser.add_argument("--qvg-v", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--events", type=int, default=4)
    args = parser.parse_args()

    with np.load(args.trace, allow_pickle=False) as a:
        keys = torch.from_numpy(np.array(a["keys"], copy=True)).float()
        values = torch.from_numpy(np.array(a["values"], copy=True)).float()
    keys = keys[: args.layers, : args.events]
    values = values[: args.layers, : args.events]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys, values = keys.to(device), values.to(device)

    def load_k(path):
        d = np.load(path, allow_pickle=False)
        return (torch.from_numpy(d["history_weights"]).float().to(device),
                torch.from_numpy(d["history_bias"]).float().to(device))

    def load_v(path):
        d = np.load(path, allow_pickle=False)
        w = torch.from_numpy(d["cross_weight"] if "cross_weight" in d else d["weight"]).float().to(device)
        b = torch.from_numpy(d["cross_bias"] if "cross_bias" in d else d["bias"]).float().to(device)
        gamma = torch.from_numpy(d["innovation_gamma"]).float().to(device) if "innovation_gamma" in d else None
        return w, b, gamma

    tk, tb = load_k(args.tempokv_k)
    qk, qb = load_k(args.qvg_k)
    tv, tvb, tvg = load_v(args.tempokv_v)
    qv, qvb, qvg = load_v(args.qvg_v)

    # tempokv legacy K weights: shape [P,P,D]; qvg affine: [P,P,D] but check
    print(f"tempokv K weights: {tuple(tk.shape)}  qvg K weights: {tuple(qk.shape)}")
    print(f"tempokv V weight: {tuple(tv.shape)}  qvg V weight: {tuple(qv.shape)}")
    if tuple(tk.shape) != tuple(qk.shape):
        print("K weight shapes differ; only comparing prediction on matching prefix")
    P = min(tk.shape[0], qk.shape[0])

    def k_pred_error(w, b):
        # open-loop K prediction residual energy over frames >= 1
        L, E, B, P_tok, F, H, D = keys.shape
        kf = keys.permute(0, 1, 2, 5, 3, 4, 6).reshape(L, E, B, H, F, P_tok, D).float()
        num, den = 0.0, 0.0
        for frame in range(1, F):
            order = min(frame, P)
            pred = torch.zeros_like(kf[..., frame, :, :], dtype=torch.float32)
            for lag in range(order):
                pred = pred + kf[..., frame - 1 - lag, :, :].float() * w[order - 1, lag].to(device)
            pred = pred + b[order - 1].to(device)
            num += (kf[..., frame, :, :].float() - pred).square().sum()
            den += kf[..., frame, :, :].float().square().sum()
        return float(num / den)

    def v_cross_error(w, b, k_rebuilt):
        # V = K_rebuilt @ W + b  (no history), frames >= 1, open loop on K
        L, E, B, P_tok, F, H, D = values.shape
        vf = values.permute(0, 1, 2, 5, 3, 4, 6).reshape(L, E, B, H, F, P_tok, D).float()
        num, den = 0.0, 0.0
        for layer in range(L):
            for frame in range(1, F):
                k = k_rebuilt[layer][..., frame, :, :].float()  # [E,B,H,P,D]
                pred = torch.einsum("...hpd,hdj->...hpj", k, w[layer])
                pred = pred + b[layer].reshape(1, 1, b[layer].shape[0], 1, -1)
                num += (vf[layer][..., frame, :, :] - pred).square().sum()
                den += vf[layer][..., frame, :, :].square().sum()
        return float(num / den)

    k_tk = k_pred_error(tk, tb)
    k_qv = k_pred_error(qk, qb)
    print(f"\n=== K 预测残差能量 (越低越好) ===")
    print(f"  tempokv legacy: {k_tk:.5f}")
    print(f"  qvg affine:     {k_qv:.5f}")

    # V cross error with clean K as source (isolate V fit quality)
    L, E, B, P_tok, F, H, D = keys.shape
    kf = keys.permute(0, 1, 2, 5, 3, 4, 6).reshape(L, E, B, H, F, P_tok, D).float()
    v_tk = v_cross_error(tv, tvb, kf)
    v_qv = v_cross_error(qv, qvb, kf)
    print(f"\n=== V Cross 预测残差能量 (clean K source, 隔离 V 拟合质量) ===")
    print(f"  tempokv legacy: {v_tk:.5f}")
    print(f"  qvg cross:      {v_qv:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
