#!/usr/bin/env python3
"""Measure closed-loop K/V reconstruction NMSE at each span position."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.config import config_from_mapping, is_temporal_v_predictor  # noqa: E402
from tempokv.fitting import (  # noqa: E402
    _reconstruct_cross_kv_trace,
    _reconstruct_temporal_trace,
)
from tempokv.parameters import (  # noqa: E402
    CrossKVParameters,
    TemporalParameters,
)


def _config(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return config_from_mapping(payload.get("codec", payload))


def _windows(tensor: torch.Tensor, span_frames: int) -> torch.Tensor:
    count = int(tensor.shape[-3]) // span_frames
    if count < 1:
        raise ValueError(
            f"span {span_frames} exceeds trace length {tensor.shape[-3]}"
        )
    pieces = [
        tensor[..., index * span_frames : (index + 1) * span_frames, :, :]
        for index in range(count)
    ]
    # The extra axis is an independent sample axis.  Reconstruction only treats
    # -3 as temporal, so history resets at every stacked window.
    return torch.stack(pieces, dim=1).contiguous()


def _energy_by_position(
    clean: torch.Tensor, rebuilt: torch.Tensor
) -> list[dict[str, float | int | None]]:
    if clean.shape != rebuilt.shape:
        raise ValueError("clean and rebuilt tensors must have identical shapes")
    rows = []
    for position in range(int(clean.shape[-3])):
        reference = clean[..., position, :, :].double()
        error = rebuilt[..., position, :, :].double() - reference
        reference_energy = float(reference.square().sum().item())
        error_energy = float(error.square().sum().item())
        rows.append(
            {
                "position": position,
                "error_energy": error_energy,
                "reference_energy": reference_energy,
                "nmse": (
                    error_energy / reference_energy
                    if reference_energy > 0.0
                    else None
                ),
                "values": int(reference.numel()),
            }
        )
    return rows


def evaluate(
    trace: Path,
    config_path: Path,
    k_params: Path,
    v_params: Path,
    span_frames: int,
    *,
    device: str = "cpu",
) -> dict[str, object]:
    config = _config(config_path)
    if span_frames < 2:
        raise ValueError("span_frames must be at least 2")
    with np.load(trace, allow_pickle=False) as archive:
        keys = torch.from_numpy(np.array(archive["keys"], copy=True))
        values = torch.from_numpy(np.array(archive["values"], copy=True))
    keys = _windows(keys, span_frames).to(device)
    values = _windows(values, span_frames).to(device)
    temporal = TemporalParameters.load(k_params).to(device)
    if temporal.codec_signature != config.format_signature():
        raise ValueError("K parameters do not match the requested codec config")
    rebuilt_keys = _reconstruct_temporal_trace(keys, temporal, config)
    if is_temporal_v_predictor(config.v_predictor_mode):
        value_parameters = TemporalParameters.load(v_params).to(device)
        rebuilt_values = _reconstruct_temporal_trace(
            values, value_parameters, config, bits=config.v_bits
        )
    else:
        value_parameters = CrossKVParameters.load(v_params).to(device)
        rebuilt_values = _reconstruct_cross_kv_trace(
            rebuilt_keys, values, value_parameters, config
        )
    if value_parameters.codec_signature != config.format_signature():
        raise ValueError("V parameters do not match the requested codec config")
    return {
        "schema_version": 1,
        "trace": str(trace.resolve()),
        "config": str(config_path.resolve()),
        "span_frames": span_frames,
        "k_bits": config.k_bits,
        "v_bits": config.v_bits,
        "anchor_bits": config.anchor_bits,
        "group_size": config.group_size,
        "stride": config.stride,
        "k": _energy_by_position(keys, rebuilt_keys),
        "v": _energy_by_position(values, rebuilt_values),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--k-params", type=Path, required=True)
    parser.add_argument("--v-params", type=Path, required=True)
    parser.add_argument("--span-frames", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = evaluate(
        args.trace,
        args.config,
        args.k_params,
        args.v_params,
        args.span_frames,
        device=args.device,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "bits",
                "anchor_bits",
                "span_frames",
                "tensor",
                "position",
                "nmse",
                "error_energy",
                "reference_energy",
                "values",
            ),
        )
        writer.writeheader()
        for tensor_name in ("k", "v"):
            for row in result[tensor_name]:
                writer.writerow(
                    {
                        "bits": result["k_bits"],
                        "anchor_bits": result["anchor_bits"],
                        "span_frames": result["span_frames"],
                        "tensor": tensor_name.upper(),
                        **row,
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
