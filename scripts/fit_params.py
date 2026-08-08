#!/usr/bin/env python3
"""Fit TempoKV parameters from a clean K/V trace archive."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch


# Keep ``python scripts/fit_params.py`` usable from an uninstalled checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.config import CodecConfig  # noqa: E402
from tempokv.fitting import fit_predictors  # noqa: E402


CODEC_FIELDS = {field.name for field in fields(CodecConfig)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit temporal K and joint K/V-history predictors from a clean .npz trace."
    )
    parser.add_argument("trace", type=Path, help=".npz file containing keys and values")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fit_output"),
        help="destination directory (default: fit_output)",
    )
    parser.add_argument("--config", type=Path, help="JSON file containing CodecConfig fields")
    parser.add_argument("--keys-name", default="keys", help="K array name in the trace")
    parser.add_argument("--values-name", default="values", help="V array name in the trace")
    k_history = parser.add_mutually_exclusive_group()
    k_history.add_argument("--k-history-length", type=int)
    k_history.add_argument(
        "--history-length",
        type=int,
        help="legacy K history override; prefer k_history_length in the config",
    )
    parser.add_argument(
        "--k-max-history-length",
        type=int,
        help="fit one shared K parameter set usable from 1 through this length",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--temporal-ridge", type=float, default=0.0)
    parser.add_argument("--cross-ridge", type=float, default=0.0)
    parser.add_argument("--max-gain", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="auto",
        help="fitting device: auto, cpu, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--coordinate-system",
        default="unspecified",
        help="K coordinate semantics recorded in both parameter files",
    )

    parser.add_argument("--k-bits", type=int)
    parser.add_argument("--v-bits", type=int)
    parser.add_argument("--anchor-bits", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--v-history-length", type=int, choices=(0, 1, 2))
    parser.add_argument(
        "--v-max-history-length",
        type=int,
        choices=(0, 1, 2),
        help="fit one shared V parameter set usable from 0 through this length",
    )
    parser.add_argument("--residual-mode", choices=("symmetric", "asymmetric"))
    parser.add_argument("--scale-dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument(
        "--reconstruction-dtype", choices=("float32", "float16", "bfloat16")
    )
    parser.add_argument(
        "--use-triton", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--strict-triton", action=argparse.BooleanOptionalAction, default=None
    )
    return parser


def _read_json_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read config {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"config is not valid JSON: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError("config JSON must contain an object")
    if "codec" in loaded:
        if len(loaded) != 1 or not isinstance(loaded["codec"], dict):
            raise ValueError("when present, 'codec' must be the only top-level config object")
        loaded = loaded["codec"]
    unknown = sorted(set(loaded) - CODEC_FIELDS)
    if unknown:
        raise ValueError(f"unknown CodecConfig fields: {', '.join(unknown)}")
    return loaded


def _codec_config(args: argparse.Namespace) -> CodecConfig:
    values = asdict(CodecConfig())
    values.update(_read_json_config(args.config))
    for name in CODEC_FIELDS:
        override = getattr(args, name, None)
        if override is not None:
            values[name] = override
    try:
        return CodecConfig(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid codec config: {error}") from error


def _load_trace(
    path: Path,
    *,
    keys_name: str,
    values_name: str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...] | None]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [name for name in (keys_name, values_name) if name not in archive]
            if missing:
                raise ValueError(f"trace is missing arrays: {', '.join(missing)}")
            keys_array = np.array(archive[keys_name], copy=True)
            values_array = np.array(archive[values_name], copy=True)
            layer_ids_array = (
                np.array(archive["layer_ids"], copy=True) if "layer_ids" in archive else None
            )
    except OSError as error:
        raise ValueError(f"cannot read trace {path}: {error}") from error

    keys = torch.from_numpy(keys_array)
    values = torch.from_numpy(values_array)
    if keys.shape != values.shape or keys.ndim < 4:
        raise ValueError(
            "keys and values must have matching [L, ..., T, H, D] shapes, got "
            f"{tuple(keys.shape)} and {tuple(values.shape)}"
        )
    if not keys.is_floating_point() or not values.is_floating_point():
        raise ValueError("keys and values must use floating-point dtypes")

    layer_ids = None
    if layer_ids_array is not None:
        if layer_ids_array.ndim != 1 or layer_ids_array.shape[0] != keys.shape[0]:
            raise ValueError("layer_ids must be one-dimensional with one entry per layer")
        if not np.issubdtype(layer_ids_array.dtype, np.integer):
            raise ValueError("layer_ids must contain integers")
        layer_ids = tuple(int(item) for item in layer_ids_array)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("layer_ids must be unique")
    return keys, values, layer_ids


def _fit_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError) as error:
        raise ValueError("fit device must be auto, cpu, cuda, or cuda:N") from error
    if device.type not in ("cpu", "cuda"):
        raise ValueError("fit device must be auto, cpu, cuda, or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA fitting was requested but CUDA is unavailable")
    if (
        device.type == "cuda"
        and device.index is not None
        and not 0 <= device.index < torch.cuda.device_count()
    ):
        raise ValueError(f"CUDA device index is unavailable: {device.index}")
    return device


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _codec_config(args)
    keys, values, layer_ids = _load_trace(
        args.trace,
        keys_name=args.keys_name,
        values_name=args.values_name,
    )
    device = _fit_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fit_started = time.perf_counter()
    if device.type == "cuda":
        # Avoid carrying float64 trace storage onto the GPU.  All CUDA fitting
        # equations and closed-loop reconstruction use float32 inputs.
        keys = keys.to(device=device, dtype=torch.float32)
        values = values.to(device=device, dtype=torch.float32)
    else:
        keys = keys.to(device=device)
        values = values.to(device=device)
    temporal, cross = fit_predictors(
        keys,
        values,
        config=config,
        history_length=(
            args.history_length
            if args.history_length is not None
            else (
                config.k_max_history_length
                if config.k_max_history_length is not None
                else config.k_history_length
            )
        ),
        iterations=args.iterations,
        temporal_ridge=args.temporal_ridge,
        cross_ridge=args.cross_ridge,
        max_gain=args.max_gain,
        layer_ids=layer_ids,
        coordinate_system=args.coordinate_system,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_fit_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_fit_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_fit_allocated = None
        peak_fit_reserved = None
    fit_seconds = time.perf_counter() - fit_started

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal_path = output_dir / "k_temporal.npz"
    cross_path = output_dir / "v_cross_kv.npz"
    report_path = output_dir / "fit_report.json"
    parameter_tensors = (
        temporal.history_weights,
        temporal.history_bias,
        cross.weight,
        cross.history_weights,
        cross.bias,
    )
    parameter_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in parameter_tensors
    )
    temporal.save(temporal_path)
    cross.save(cross_path)
    report = {
        "status": "fit_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trace": str(args.trace.resolve()),
        "input": {
            "shape": list(keys.shape),
            "keys_dtype": str(keys.dtype).removeprefix("torch."),
            "values_dtype": str(values.dtype).removeprefix("torch."),
            "layer_ids": list(temporal.layer_ids or ()),
        },
        "codec_config": asdict(config),
        "fit": {
            # Keep the old report key while exposing the clearer config name.
            "history_length": temporal.max_history_length,
            "k_history_length": temporal.with_history_length(
                config.k_history_length
            ).history_length,
            "k_max_history_length": temporal.max_history_length,
            "k_history_mode": temporal.history_mode,
            "iterations": int(args.iterations),
            "temporal_ridge": float(args.temporal_ridge),
            "cross_ridge": float(args.cross_ridge),
            "v_history_length": cross.with_history_length(
                config.v_history_length
            ).history_length,
            "v_max_history_length": cross.max_history_length,
            "v_history_mode": cross.history_mode,
            "max_gain": float(args.max_gain),
            "coordinate_system": args.coordinate_system,
            "device": str(device),
            "compute_dtype": "float32" if device.type == "cuda" else "float64",
            "seconds": fit_seconds,
            # Keep the original name as an alias for allocated memory.
            "peak_gpu_bytes": peak_fit_allocated,
            "peak_gpu_allocated_bytes": peak_fit_allocated,
            "peak_gpu_reserved_bytes": peak_fit_reserved,
        },
        "parameters": {
            "temporal_weight_shape": list(temporal.history_weights.shape),
            "temporal_bias_shape": list(temporal.history_bias.shape),
            "cross_weight_shape": list(cross.weight.shape),
            "cross_history_weight_shape": list(cross.history_weights.shape),
            "cross_bias_shape": list(cross.bias.shape),
            "bytes": parameter_bytes,
            "mib": parameter_bytes / (1024**2),
        },
        "artifacts": {
            "k_temporal": temporal_path.name,
            "v_cross_kv": cross_path.name,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
