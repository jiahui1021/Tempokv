#!/usr/bin/env python3
"""Compare legacy V decode: torch path vs triton path on the same packed tensor.

Fits nothing; loads fitted params + legacy trace, encodes one V cache, then
decodes it twice (legacy_v_decode=torch vs triton) and reports per-unit error
growth so the non-bit-exact source is visible.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tempokv.codec import TempoKVCodec, decode_tensor  # noqa: E402
from tempokv.config import CodecConfig  # noqa: E402
from tempokv.parameters import CrossKVParameters, TemporalParameters  # noqa: E402


def _load_params(path: str) -> dict:
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.keys():
            arr = z[k]
            if arr.dtype.kind in "fiu":
                out[k] = torch.from_numpy(arr)
            else:
                out[k] = arr
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--k-params", required=True)
    parser.add_argument("--v-params", required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--events", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as fh:
        payload = json.load(fh)
    payload = payload.get("codec", payload)
    base_fields = {**asdict(CodecConfig()), **payload}

    with np.load(args.trace, allow_pickle=False) as a:
        keys = torch.from_numpy(np.array(a["keys"], copy=True)).float()
        values = torch.from_numpy(np.array(a["values"], copy=True)).float()

    # Legacy layout [L,E,B,P,F,H,D]; pick a small slice.
    if keys.ndim == 7:
        keys = keys[: args.layers, : args.events].cuda()
        values = values[: args.layers, : args.events].cuda()
    else:
        keys = keys[: args.layers].cuda()
        values = values[: args.layers].cuda()

    kp = _load_params(args.k_params)
    vp = _load_params(args.v_params)
    kh = int(kp["history_length"]) if "history_length" in kp else int(payload["k_history_length"])
    temporal = TemporalParameters(
        kp["history_weights"], kp["history_bias"], stride=int(kp["stride"]),
        codec_signature=str(kp["codec_signature"]),
    )
    cross = CrossKVParameters(
        vp["weight"], vp["bias"],
        history_weights=vp["history_weights"],
        stride=int(vp["stride"]),
        codec_signature=str(vp["codec_signature"]),
    )
    print("history_diagonal:", cross.history_diagonal)

    config = CodecConfig(**base_fields)
    codec = TempoKVCodec(temporal, cross, config)

    # Encode a V cache from one layer/event. Legacy record is [B,F,P,H,D]:
    # concatenate F frames along the sequence axis -> [B, F*P, H, D], so each
    # P-token frame is one unit and the cross-unit V history recursion fires.
    layer = 0
    if keys.ndim == 7:
        rec_k = keys[layer][0]  # [B,F,P,H,D]
        rec_v = values[layer][0]
        b, f, p, h, d = rec_v.shape
        kk = rec_k.permute(0, 1, 2, 3, 4).reshape(b, f * p, h, d).contiguous()
        vv = rec_v.reshape(b, f * p, h, d).contiguous()
    else:
        kk = keys[layer].unsqueeze(0)
        vv = values[layer].unsqueeze(0)
    # kk/vv now [N, S, H, D] with S = F*P tokens, stride = P
    k_rebuilt = codec.encode_key(kk, layer=layer)[1]
    packed = codec.encode_value(vv, k_rebuilt, layer=layer)

    # Decode via torch (legacy_v_decode=torch) and triton (triton).
    torch_cfg = CodecConfig(**{**base_fields, "legacy_v_decode": "torch", "use_triton": False})
    triton_cfg = CodecConfig(**{**base_fields, "legacy_v_decode": "triton", "use_triton": True})

    # Force each path by re-creating codec with matching config for decode.
    # decode_tensor uses config passed via packed... simpler: call decode_tensor with use_triton flag.
    v_torch = decode_tensor(packed, source=k_rebuilt, use_triton=False, strict_triton=False)
    v_triton = decode_tensor(packed, source=k_rebuilt, use_triton=True, strict_triton=False)

    print(f"shape: v_torch={tuple(v_torch.shape)} v_triton={tuple(v_triton.shape)}")
    diff = (v_torch.float() - v_triton.float()).abs()
    print(f"max abs diff: {diff.max().item():.6f}")
    print(f"mean abs diff: {diff.mean().item():.6f}")
    # Per-frame relative diff (frame axis is -3? check layout)
    # v_torch/v_triton layout [N, seq, H, D]; find the frame/unit boundaries via stride
    stride = packed.stride
    seq = v_torch.shape[1]
    print(f"seq={seq} stride={stride} units={seq // stride}")
    for u in range(0, seq, stride):
        sl = slice(u, u + stride)
        a, b = v_torch[:, sl], v_triton[:, sl]
        rel = (a.float() - b.float()).norm() / (a.float().norm() + 1e-12)
        print(f"  unit {u//stride}: rel diff={rel:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
