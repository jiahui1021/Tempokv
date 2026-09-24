#!/usr/bin/env python3
"""Evaluate one history-ablation parameter set on a clean K/V trace."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "scripts" / "tools"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from analyze_predictor import (  # noqa: E402
    reconstruct_k,
    reconstruct_temporal,
    reconstruct_v,
)
from tempokv.config import (  # noqa: E402
    load_config,
    is_temporal_v_predictor,
)
from tempokv.parameters import CrossKVParameters, TemporalParameters  # noqa: E402




def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _nmse(error: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = error.float().square().sum().item()
    denominator = reference.float().square().sum().item()
    return float(numerator / max(float(denominator), 1e-12))


def _metric_block(
    prediction: torch.Tensor,
    rebuilt: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | int]:
    later = target[..., 1:, :, :]
    prediction_nmse = _nmse(prediction[..., 1:, :, :] - later, later)
    reconstruction_nmse = _nmse(rebuilt - target, target)
    reconstruction_nmse_predicted = _nmse(
        rebuilt[..., 1:, :, :] - later, later
    )
    return {
        "prediction_nmse": prediction_nmse,
        "prediction_share": 1.0 - prediction_nmse,
        "reconstruction_nmse": reconstruction_nmse,
        "reconstruction_nmse_predicted_frames": reconstruction_nmse_predicted,
        "prediction_frames": int(later.shape[-3]),
        "all_frames": int(target.shape[-3]),
    }


def evaluate(
    trace_path: Path,
    config_path: Path,
    k_params_path: Path,
    v_params_path: Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = _device(device_name)
    config = load_config(config_path)
    with np.load(trace_path, allow_pickle=False) as archive:
        missing = [name for name in ("keys", "values") if name not in archive]
        if missing:
            raise ValueError("trace is missing arrays: " + ", ".join(missing))
        keys = torch.from_numpy(np.array(archive["keys"], copy=True)).to(
            device=device, dtype=torch.float32
        )
        values = torch.from_numpy(np.array(archive["values"], copy=True)).to(
            device=device, dtype=torch.float32
        )
    if keys.shape != values.shape or keys.ndim < 4:
        raise ValueError(
            "history-ablation cache evaluation expects matching [L, ..., T, H, D] "
            f"arrays, got {tuple(keys.shape)} and {tuple(values.shape)}"
        )
    if keys.ndim > 5:
        layer_count = keys.shape[0]
        frame_count = keys.shape[-3]
        head_count = keys.shape[-2]
        dim = keys.shape[-1]
        sample_count = int(np.prod(keys.shape[1:-3]))
        keys = keys.reshape(layer_count, sample_count, frame_count, head_count, dim)
        values = values.reshape(layer_count, sample_count, frame_count, head_count, dim)

    temporal = TemporalParameters.load(k_params_path).to(device)
    if is_temporal_v_predictor(config.v_predictor_mode):
        value_parameters: TemporalParameters | CrossKVParameters = (
            TemporalParameters.load(v_params_path).to(device)
        )
    else:
        value_parameters = CrossKVParameters.load(v_params_path).to(device)
    expected_signature = config.format_signature()
    for name, parameters in (("K", temporal), ("V", value_parameters)):
        if (
            parameters.codec_signature is not None
            and parameters.codec_signature != expected_signature
        ):
            raise ValueError(f"{name} parameters do not match the codec config")
    if (
        temporal.parameter_set_id is not None
        and value_parameters.parameter_set_id is not None
        and temporal.parameter_set_id != value_parameters.parameter_set_id
    ):
        raise ValueError("K and V parameters were not fitted together")

    k_prediction, _, rebuilt_keys = reconstruct_k(keys, temporal, config)
    if is_temporal_v_predictor(config.v_predictor_mode):
        assert isinstance(value_parameters, TemporalParameters)
        v_prediction, _, rebuilt_values = reconstruct_temporal(
            values,
            value_parameters,
            config,
            bits=config.v_bits,
        )
    else:
        assert isinstance(value_parameters, CrossKVParameters)
        v_prediction, _, rebuilt_values, _, _ = reconstruct_v(
            keys,
            values,
            rebuilt_keys,
            value_parameters,
            config,
        )

    return {
        "schema_version": 1,
        "trace": str(trace_path.resolve()),
        "config": str(config_path.resolve()),
        "parameters": {
            "k": str(k_params_path.resolve()),
            "v": str(v_params_path.resolve()),
            "parameter_set_id": temporal.parameter_set_id,
        },
        "device": str(device),
        "codec": asdict(config),
        "trace_shape": list(keys.shape),
        "k": _metric_block(k_prediction, rebuilt_keys, keys),
        "v": _metric_block(v_prediction, rebuilt_values, values),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--k-params", type=Path, required=True)
    parser.add_argument("--v-params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        args.trace,
        args.config,
        args.k_params,
        args.v_params,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
