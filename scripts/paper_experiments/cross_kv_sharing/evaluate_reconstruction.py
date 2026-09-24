#!/usr/bin/env python3
"""Evaluate production closed-loop K/V reconstruction on a trace archive."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.config import load_config  # noqa: E402
from tempokv.fitting import (  # noqa: E402
    _reconstruct_cross_kv_trace,
    _reconstruct_temporal_trace,
)
from tempokv.parameters import CrossKVParameters, TemporalParameters  # noqa: E402




def _energy(clean: torch.Tensor, rebuilt: torch.Tensor) -> dict[str, float | int | None]:
    reference = clean.double()
    error = rebuilt.double() - reference
    reference_energy = float(reference.square().sum().item())
    error_energy = float(error.square().sum().item())
    return {
        "error_energy": error_energy,
        "reference_energy": reference_energy,
        "nmse": error_energy / reference_energy if reference_energy > 0.0 else None,
        "values": int(reference.numel()),
    }


def _breakdown(
    clean: torch.Tensor,
    rebuilt: torch.Tensor,
    *,
    layer_ids: tuple[int, ...],
) -> dict[str, list[dict[str, Any]]]:
    layers: list[dict[str, Any]] = []
    heads: list[dict[str, Any]] = []
    layer_heads: list[dict[str, Any]] = []
    layer_count = int(clean.shape[0])
    head_count = int(clean.shape[-2])
    if len(layer_ids) != layer_count:
        raise ValueError("layer IDs do not match trace layer count")
    for layer_index, layer_id in enumerate(layer_ids):
        layers.append(
            {
                "layer_index": layer_index,
                "layer_id": layer_id,
                **_energy(clean[layer_index], rebuilt[layer_index]),
            }
        )
        for head in range(head_count):
            layer_heads.append(
                {
                    "layer_index": layer_index,
                    "layer_id": layer_id,
                    "head": head,
                    **_energy(
                        clean[layer_index, ..., head, :],
                        rebuilt[layer_index, ..., head, :],
                    ),
                }
            )
    for head in range(head_count):
        heads.append(
            {
                "head": head,
                **_energy(clean[..., head, :], rebuilt[..., head, :]),
            }
        )
    return {"per_layer": layers, "per_head": heads, "per_layer_head": layer_heads}


def _rows(result: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for tensor_name in ("k", "v"):
        tensor = result[tensor_name]
        yield {
            "tensor": tensor_name.upper(),
            "scope": "pooled",
            "layer_index": "",
            "layer_id": "",
            "head": "",
            **tensor["pooled"],
        }
        for scope in ("per_layer", "per_head", "per_layer_head"):
            for row in tensor[scope]:
                yield {
                    "tensor": tensor_name.upper(),
                    "scope": scope,
                    "layer_index": row.get("layer_index", ""),
                    "layer_id": row.get("layer_id", ""),
                    "head": row.get("head", ""),
                    **{
                        name: row[name]
                        for name in ("nmse", "error_energy", "reference_energy", "values")
                    },
                }


def evaluate(
    trace: Path,
    config_path: Path,
    k_params_path: Path,
    v_params_path: Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Replay the production reconstruction helpers and summarize NMSE."""
    config = load_config(config_path)
    with np.load(trace, allow_pickle=False) as archive:
        missing = [name for name in ("keys", "values") if name not in archive]
        if missing:
            raise ValueError(f"trace is missing arrays: {', '.join(missing)}")
        keys = torch.from_numpy(np.array(archive["keys"], copy=True)).to(device)
        values = torch.from_numpy(np.array(archive["values"], copy=True)).to(device)
        trace_layer_ids = (
            tuple(int(item) for item in np.asarray(archive["layer_ids"]).reshape(-1))
            if "layer_ids" in archive
            else None
        )
    if keys.shape != values.shape or keys.ndim < 4:
        raise ValueError("trace keys and values must share [L,...,T,H,D] shape")

    k_parameters = TemporalParameters.load(k_params_path).to(device)
    v_parameters = CrossKVParameters.load(v_params_path).to(device)
    signature = config.format_signature()
    if k_parameters.codec_signature != signature:
        raise ValueError("K parameters do not match the requested codec config")
    if v_parameters.codec_signature != signature:
        raise ValueError("V parameters do not match the requested codec config")
    if (
        not k_parameters.parameter_set_id
        or not v_parameters.parameter_set_id
        or k_parameters.parameter_set_id != v_parameters.parameter_set_id
    ):
        raise ValueError("K/V parameters must share the same parameter_set_id")
    if config.v_predictor_mode != "cross_kv":
        raise ValueError("this ablation requires v_predictor_mode=cross_kv")

    rebuilt_keys = _reconstruct_temporal_trace(keys, k_parameters, config)
    rebuilt_values = _reconstruct_cross_kv_trace(
        rebuilt_keys, values, v_parameters, config
    )
    layer_ids = (
        trace_layer_ids
        or v_parameters.layer_ids
        or tuple(range(int(keys.shape[0])))
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "trace": str(trace.resolve()),
        "config": str(config_path.resolve()),
        "cross_kv_weight_mode": config.cross_kv_weight_mode,
        "shape": list(keys.shape),
        "device": str(device),
    }
    for name, clean, rebuilt in (
        ("k", keys, rebuilt_keys),
        ("v", values, rebuilt_values),
    ):
        result[name] = {
            "pooled": _energy(clean, rebuilt),
            **_breakdown(clean, rebuilt, layer_ids=tuple(layer_ids)),
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--k-params", type=Path, required=True)
    parser.add_argument("--v-params", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate(
        args.trace,
        args.config,
        args.k_params,
        args.v_params,
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
                "tensor",
                "scope",
                "layer_index",
                "layer_id",
                "head",
                "nmse",
                "error_energy",
                "reference_energy",
                "values",
            ),
        )
        writer.writeheader()
        writer.writerows(_rows(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
