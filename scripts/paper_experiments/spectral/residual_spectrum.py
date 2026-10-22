#!/usr/bin/env python3
"""E5 on an explicit clean-grid NPZ; no codec or quantizer is executed.

Input: k_pre and v, both [F,H,W,heads,D]; scalar layer, stride=H*W,
coordinate_system ('pre', 'pre_rope', or 'pre_rope_normalized'), and grid=[F,H,W].
Provide predicted_k/predicted_v arrays or reuse fitted --k-params/--v-params.
Predictor reuse uses actual clean per-frame histories, with batch dimension 1
for TempoKV's [B,heads,tokens,D] API, and skips the full-order warm-up frames.
This is a predictor-only residual study, not a quantized codec round trip.
"""
from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path
import sys

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tempokv.codec import predict_cross_kv, predict_temporal
from tempokv.parameters import CrossKVParameters, TemporalParameters
from tempokv.spectral_kv.residual import ResidualSpectrum


def _scalar(data, name, default=None):
    if name not in data:
        if default is not None:
            return default
        raise ValueError(f"input requires scalar {name}")
    value = np.asarray(data[name])
    if value.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return value.item()


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _feature(data, name, shape=None):
    if name not in data:
        raise ValueError(f"input requires {name} [F,H,W,heads,D]")
    array = np.asarray(data[name])
    if array.ndim != 5 or min(array.shape) < 1 or array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite real [F,H,W,heads,D] array")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} does not match the runtime feature grid")
    return torch.from_numpy(array.copy()).float()


def _parameter(parameters, *, layer, stride, coordinate, heads, dim):
    if parameters.stride is None or parameters.stride != stride:
        raise ValueError("parameter stride must explicitly match runtime stride=H*W")
    if parameters.coordinate_system != coordinate:
        raise ValueError("parameter coordinate_system must explicitly match the input pre-RoPE contract")
    if parameters.layered and layer not in parameters.layer_ids:
        raise ValueError(f"parameter archive has no layer {layer}")
    parameters = parameters.for_layer(layer)
    if isinstance(parameters, TemporalParameters):
        if parameters.history_weights.shape[-1] != dim:
            raise ValueError("K parameter feature dimension does not match the runtime grid")
        if parameters.per_head and parameters.history_weights.shape[-2] != heads:
            raise ValueError("K parameter head count does not match the runtime grid")
        tensors = (parameters.history_weights, parameters.history_bias)
    else:
        if parameters.weight.shape != (heads, dim, dim):
            raise ValueError("V parameter heads/features do not match the runtime grid")
        tensors = (parameters.weight, parameters.bias, parameters.history_weights)
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("predictor parameters must be finite")
    return parameters


def analyze_residual(input_path: str | Path, output_dir: str | Path, *, k_params=None, v_params=None, bins=12) -> dict:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    with np.load(input_path, allow_pickle=False) as data:
        k = _feature(data, "k_pre")
        v = _feature(data, "v", tuple(k.shape))
        frames, h, w, heads, dim = k.shape
        layer = _integer(_scalar(data, "layer"), "layer")
        stride = _integer(_scalar(data, "stride"), "stride")
        if stride != h * w:
            raise ValueError("runtime stride must equal H*W for this per-frame predictor study")
        grid_key = next((key for key in ("grid", "runtime_grid", "grid_sizes") if key in data), None)
        if grid_key is None:
            raise ValueError("input requires runtime grid=[F,H,W]")
        grid = np.asarray(data[grid_key])
        if grid.shape == (1, 3):
            grid = grid[0]
        if grid.shape != (3,) or grid.dtype.kind not in "iu" or tuple(grid.tolist()) != (frames, h, w):
            raise ValueError("runtime grid must exactly match k_pre/v [F,H,W]")
        coordinate = str(_scalar(data, "coordinate_system"))
        if coordinate not in ("pre", "pre_rope", "pre_rope_normalized"):
            raise ValueError("input coordinate_system must explicitly identify pre-RoPE K")
        frame_idx0 = _integer(_scalar(data, "frame_idx0", 0), "frame_idx0")
        sample_id = _integer(_scalar(data, "sample_id", 0), "sample_id")
        supplied = {role: _feature(data, f"predicted_{role}", tuple(k.shape))
                    for role in ("k", "v") if f"predicted_{role}" in data}
        identity = {name: _scalar(data, name) for name in ("prompt_id", "seed") if name in data}
    kp = _parameter(TemporalParameters.load(k_params), layer=layer, stride=stride, coordinate=coordinate, heads=heads, dim=dim) if k_params else None
    vp = _parameter(CrossKVParameters.load(v_params), layer=layer, stride=stride, coordinate=coordinate, heads=heads, dim=dim) if v_params else None
    if kp is not None and "k" in supplied or vp is not None and "v" in supplied:
        raise ValueError("choose supplied predictions or parameters separately for each role")
    if vp is not None and vp.projection_source == "innovation" and kp is None:
        raise ValueError("innovation-source V parameters require --k-params to form actual K innovations")
    if not supplied and kp is None and vp is None:
        raise ValueError("provide predicted_k/predicted_v arrays or fitted predictor parameter files")
    # The raw feature grid is never passed through an encoder or quantizer.
    def frame(x, index):
        return x[index].reshape(h * w, heads, dim).permute(1, 0, 2).unsqueeze(0)

    def grid(prediction):
        return prediction.squeeze(0).permute(1, 0, 2).reshape(1, h, w, heads, dim)

    k_predictions = {}
    if kp is not None and kp.history_length > 0:
        for index in range(kp.history_length, frames):
            history = [frame(k, index - lag - 1) for lag in range(kp.history_length)]
            k_predictions[index] = predict_temporal(history, kp)
    observations = []
    skipped = []
    for role, actual, parameters in (("k", k, kp), ("v", v, vp)):
        if role in supplied:
            observations.append((role, actual, supplied[role], frame_idx0, "supplied_prediction"))
            continue
        if parameters is None:
            skipped.append({"role": role, "reason": "no_prediction_or_parameters", "frames": frames})
            continue
        if role == "k" and parameters.history_length == 0:
            skipped.append({"role": role, "reason": "zero_order_K_has_no_history_prediction", "frames": frames})
            continue
        warmup = max(1, parameters.history_length) if role == "k" else parameters.history_length
        if role == "v" and parameters.projection_source == "innovation":
            warmup = max(warmup, kp.history_length, 1)
        if warmup:
            skipped.append({"role": role, "reason": "full_order_history_warmup", "frames": min(frames, warmup),
                            "frame_idx_start": frame_idx0, "frame_idx_stop_exclusive": frame_idx0 + min(frames, warmup)})
        for index in range(warmup, frames):
            if role == "k":
                prediction = k_predictions[index]
            else:
                source = frame(k, index)
                if parameters.projection_source == "innovation":
                    if index not in k_predictions:
                        skipped.append({"role": "v", "reason": "K_innovation_unavailable", "frame_idx": frame_idx0 + index})
                        continue
                    source = source - k_predictions[index]
                history = [frame(v, index - lag - 1) for lag in range(parameters.history_length)]
                prediction = predict_cross_kv(source, parameters, history)
            observations.append((role, actual[index:index + 1], grid(prediction), frame_idx0 + index,
                                 "tempokv_predictor_clean_history_no_quantization"))
    rows_needed = max(1, sum(item[1].shape[0] for item in observations) * heads * bins * 2)
    spectrum = ResidualSpectrum(bins=bins, max_rows=rows_needed)
    for role, actual, prediction, index, predictor in observations:
        spectrum.update(layer, actual, prediction, role=role, frame_idx0=index, sample_id=sample_id,
            coordinate="pre", coordinate_system=coordinate, predictor=predictor, runtime_grid=[frames, h, w], stride=stride, **identity)
    metadata = {"suite_id": "B_SPECTRAL_KV", "experiment_id": "B_E5_RESIDUAL", "input": str(input_path),
        "layer": layer, "grid": [frames, h, w], "heads": heads, "head_dim": dim, "stride": stride,
        "coordinate_system": coordinate, "source_contract": "pre_rope_clean_features",
        "k_params": str(k_params) if k_params else None, "v_params": str(v_params) if v_params else None,
        "observed_frame_role_pairs": sum(item[1].shape[0] for item in observations),
        "rows": len(spectrum.rows), "skipped": skipped, "codec_or_quantizer_executed": False,
        "result_scope": "spectral measurements only; no generation quality or research conclusion", **identity}
    destination.mkdir(parents=True, exist_ok=False)
    experiment = destination / "B_E5_RESIDUAL"
    experiment.mkdir()
    with (experiment / "records.jsonl").open("x") as handle:
        for row in spectrum.rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    with (experiment / "stats.npz").open("xb") as handle:
        np.savez_compressed(handle, **spectrum.state_dict())
    with (destination / "metadata.json").open("x") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    return metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--k-params", help="existing TemporalParameters NPZ")
    parser.add_argument("--v-params", help="existing CrossKVParameters NPZ")
    parser.add_argument("--bins", type=int, default=12)
    args = parser.parse_args(argv)
    print(json.dumps(analyze_residual(args.input, args.output, k_params=args.k_params, v_params=args.v_params, bins=args.bins), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
