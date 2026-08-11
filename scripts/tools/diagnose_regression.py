"""Localize a TempoKV quality regression to one stage boundary.

Runs the real codec on a real trace and reports, per layer:

  A. predictor residual energy   -- are the fitted parameters any good?
  B. encoder closed-loop error   -- the codec's intrinsic quality
  C. decoder parity              -- does decode() reproduce the encoder exactly?
  D. error attribution           -- quantization vs decoder mismatch

Read the stages in order.  The first one that looks wrong is where the bug is;
later stages inherit the earlier stage's damage and are not independent evidence.

    python scripts/tools/diagnose_regression.py \
        --trace runs/.../self_forcing_trace.npz \
        --config configs/int4.json \
        --k-params <params>/k_temporal.npz \
        --v-params <params>/v_cross_kv.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tempokv.codec import (
    TempoKVCodec,
    decode_tensor,
    predict_cross_kv,
    predict_temporal,
)
from tempokv.config import CodecConfig
from tempokv.parameters import CrossKVParameters, TemporalParameters


def _relative_mse(estimate: torch.Tensor, target: torch.Tensor) -> float:
    return (
        (estimate.float() - target.float()).pow(2).mean()
        / target.float().pow(2).mean()
    ).item()


def _restride(parameters, stride):
    """Rebuild parameters for the trace's tokens-per-frame stride.

    The trace subsamples tokens per frame, so a faithful replay uses one unit
    per sampled frame rather than the production 1560-token stride.  Identity
    stamps are dropped with it: they describe the production config, and
    keeping them would only make the codec reject its own parameters.
    """
    if isinstance(parameters, TemporalParameters):
        return TemporalParameters(
            parameters.history_weights,
            parameters.history_bias,
            None,
            parameters.layer_ids,
            None,
            None,
            parameters.coordinate_system,
            parameters.maximum_history_length,
            parameters.active_history_length,
        )
    return CrossKVParameters(
        parameters.weight,
        parameters.bias,
        None,
        parameters.layer_ids,
        None,
        None,
        parameters.coordinate_system,
        parameters.history_weights,
        parameters.maximum_history_length,
        parameters.active_history_length,
    )


def _as_cache(array: np.ndarray, device: str) -> torch.Tensor:
    """[N,T,H,D] sampled trace -> [1,H,T*N,D] frame-major cache."""
    tensor = torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)
    samples, frames, heads, dim = tensor.shape
    return (
        tensor.permute(1, 0, 2, 3)
        .reshape(frames * samples, heads, dim)
        .permute(1, 0, 2)
        .unsqueeze(0)
        .contiguous()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--k-params", required=True)
    parser.add_argument("--v-params", required=True)
    parser.add_argument("--layers", default="0,15,29")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keys-name", default="keys")
    parser.add_argument("--values-name", default="values")
    args = parser.parse_args()

    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    payload = payload.get("codec", payload)
    temporal = TemporalParameters.load(args.k_params)
    cross_kv = CrossKVParameters.load(args.v_params)

    print("=== parameter identity ===")
    print(f"  K history_mode      : {temporal.history_mode}")
    print(f"  K weights shape     : {tuple(temporal.history_weights.shape)}")
    print(f"  K history_length    : {temporal.history_length}"
          f" (max {temporal.max_history_length})")
    print(f"  K per_head          : {temporal.per_head}")
    print(f"  V history_mode      : {cross_kv.history_mode}")
    print(f"  V history_length    : {cross_kv.history_length}"
          f" (max {cross_kv.max_history_length})")
    print(f"  coordinate_system   : {temporal.coordinate_system}")
    print(f"  codec_signature     : {temporal.codec_signature}")
    if temporal.shared_history:
        print("  !! shared history: prediction collapses to w*mean(history)+b,")
        print("     which has no legacy counterpart and loses per-lag structure.")
        print("     Refit without --k-max-history-length for a legacy comparison.")

    # The fitter caps each channel's L1 gain across lags at --max-gain. A cap
    # of 1.0 forbids extrapolation (lag-1 above 1 against negative higher
    # lags), which is exactly what a smoothly drifting cache wants. When the
    # cap binds, adding history makes prediction *worse* rather than better.
    # Independent weights are [P,P,D], [P,P,H,D] or [L,P,P,H,D]; the lag axis is
    # -3 whenever a head axis is present and -2 otherwise.  `per_head` cannot be
    # used here because it only reads correctly after a layer is selected.
    weights = temporal.history_weights
    if temporal.shared_history:
        gains = weights.abs()  # one coefficient per channel, no lag axis
    else:
        gains = weights.abs().sum(dim=-3 if weights.ndim >= 4 else -2)
    for threshold in (1.0,):
        at_cap = (gains >= threshold - 1e-6).float().mean().item()
        print(f"  fitted L1 gain      : max {gains.max().item():.3f}, "
              f"{at_cap:.1%} of channels at/above {threshold:.1f}")
        if at_cap > 0.05 and gains.max().item() <= threshold + 1e-6:
            print(f"     !! the --max-gain {threshold:.1f} cap is binding. Refit with a "
                  "larger --max-gain")
            print("        and re-read A.K: a 10x drop there means this was the "
                  "regression.")

    with np.load(args.trace, allow_pickle=False) as data:
        keys = data[args.keys_name]
        values = data[args.values_name]
        layer_ids = (
            [int(x) for x in data["layer_ids"]] if "layer_ids" in data else None
        )

    # Trace layout is [L, N, T, H, D]: N independent sampled tokens per frame,
    # T frames.  Replayed frame-major, one codec unit is one frame of N tokens.
    stride = int(keys.shape[1])
    frames = int(keys.shape[2])
    config = CodecConfig(
        **{**payload, "stride": stride, "use_triton": False, "strict_triton": False}
    )
    codec = TempoKVCodec(_restride(temporal, stride), _restride(cross_kv, stride), config)

    triton_config = CodecConfig(
        **{**payload, "stride": stride, "use_triton": True, "strict_triton": True}
    )

    requested = [int(x) for x in args.layers.split(",") if x.strip()]
    needed = max(temporal.history_length, cross_kv.history_length) + 1
    if frames < needed:
        raise SystemExit(
            f"trace has {frames} frames but this predictor needs {needed} to be "
            "probed at its full history order"
        )
    print(f"\ntrace {tuple(keys.shape)}  tokens/frame={stride} frames={frames}  "
          f"k_bits={config.k_bits} v_bits={config.v_bits} "
          f"kh={config.k_history_length} vh={config.v_history_length}")

    header = (
        f"\n{'layer':>5} {'A.K pred':>10} {'A.V pred':>10} "
        f"{'B.K recon':>11} {'B.V recon':>11} "
        f"{'C.K trit':>10} {'C.V trit':>10}"
    )
    print(header)
    print("-" * len(header))

    for layer in requested:
        index = layer_ids.index(layer) if layer_ids else layer
        key = _as_cache(keys[index], args.device)
        value = _as_cache(values[index], args.device)

        packed_key, key_recon = codec.encode_key(key, layer=layer)
        packed_value = codec.encode_value(value, key_recon, layer=layer)

        # --- A. predictor quality, no quantization in the loop -------------
        # Probe the first unit that has the predictor's full history available,
        # so a kh3 parameter set is measured at order 3 rather than at warm-up.
        selected_k = codec._temporal_for_layer(key.device, layer)
        selected_v = codec._cross_for_layer(value.device, layer)
        order = selected_k.history_length

        def unit(tensor: torch.Tensor, index: int) -> torch.Tensor:
            return tensor[:, :, index * stride: (index + 1) * stride]

        # Probe with the same predictor arithmetic the codec itself runs, so
        # A never disagrees with B for a reason the report cannot show.
        legacy = config.fit_mode == "legacy"
        k_history = [unit(key, order - 1 - lag) for lag in range(order)]
        k_pred = predict_temporal(k_history, selected_k, legacy=legacy)
        a_k = _relative_mse(k_pred, unit(key, order))

        v_history = [
            unit(value, order - 1 - lag)
            for lag in range(selected_v.history_length)
        ]
        v_pred = predict_cross_kv(
            unit(key, order),
            selected_v,
            v_history or None,
            matmul_precision=config.cross_matmul_precision,
            legacy=legacy,
        )
        a_v = _relative_mse(v_pred, unit(value, order))

        # --- B. encoder closed-loop reconstruction error -------------------
        b_k = _relative_mse(key_recon, key)
        value_recon = decode_tensor(
            packed_value, source=key_recon, use_triton=False
        )
        b_v = _relative_mse(value_recon, value)

        # --- C. decoder parity against the encoder's own reconstruction ----
        reference_k = decode_tensor(packed_key, use_triton=False)
        exact_k = torch.equal(reference_k, key_recon)
        if not exact_k:
            raise SystemExit(
                f"layer {layer}: PyTorch decode does not reproduce the encoder "
                "reconstruction bit-exactly; this is a codec bug, stop here"
            )
        c_k = c_v = float("nan")
        if torch.cuda.is_available():
            try:
                triton_key = decode_tensor(
                    packed_key, use_triton=True, strict_triton=True
                )
                triton_value = decode_tensor(
                    packed_value,
                    source=triton_key,
                    use_triton=True,
                    strict_triton=True,
                )
                c_k = (triton_key.float() - key_recon.float()).abs().max().item()
                c_v = (triton_value.float() - value_recon.float()).abs().max().item()
            except Exception as error:  # noqa: BLE001 - reported, not raised
                print(f"  layer {layer}: Triton decode unavailable: {error}")

        print(
            f"{layer:>5} {a_k:>10.4f} {a_v:>10.4f} "
            f"{b_k:>11.5f} {b_v:>11.5f} {c_k:>10.2e} {c_v:>10.2e}"
        )

    print(
        """
How to read this
  A.* predictor residual energy, relative to the signal. This is what gets
      quantized. If A is much worse than the sweep_history report claimed,
      the parameter files are wrong (stale, wrong mode, wrong coordinates).
  B.* closed-loop reconstruction error. This is the number to compare against
      the legacy predictor at the same bit width. If A is fine and B is bad, the bug is in the
      quantizer or the closed loop, not the predictor.
  C.* max |Triton decode - encoder reconstruction|. C.K must be 0.0: the K
      path is asserted bit-exact in the test suite. C.V is the CrossKV path,
      where the encoder uses cuBLAS TF32 and the decoder uses Triton tl.dot.
      Any nonzero C.V is error added to V that the residual never absorbed.
      Compare it against B.V: if C.V is a large fraction of B.V, the
      encoder/decoder matmul mismatch is your regression."""
    )


if __name__ == "__main__":
    main()
