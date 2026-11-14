#!/usr/bin/env python3
"""Compare CrossKV and temporal-only V reconstruction over closed-loop frames."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.config import CodecConfig, config_from_mapping  # noqa: E402
from tempokv.fitting import (  # noqa: E402
    _reconstruct_cross_kv_trace,
    _reconstruct_temporal_trace,
    fit_predictors,
)
from tempokv.parameters import CrossKVParameters, TemporalParameters  # noqa: E402

METHODS = ("cross_kv_history", "temporal_only")
EPS = torch.finfo(torch.float64).tiny


def split_trace_ids(sample_count: int, holdout_step: int) -> tuple[np.ndarray, np.ndarray]:
    if holdout_step < 3:
        raise ValueError("holdout_step must be at least 3")
    eval_ids = np.arange(holdout_step - 1, sample_count, holdout_step, dtype=int)
    train_mask = np.ones(sample_count, dtype=bool)
    train_mask[eval_ids] = False
    train_ids = np.flatnonzero(train_mask)
    if eval_ids.size < 1 or train_ids.size < 4:
        raise ValueError("trace split is too small")
    return train_ids, eval_ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trace(
    path: Path,
    *,
    frames: int,
    requested_layers: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], dict[str, Any]]:
    if frames < 3:
        raise ValueError("frames must be at least 3")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in ("keys", "values") if name not in archive]
        if missing:
            raise ValueError(f"trace is missing arrays: {', '.join(missing)}")
        key_shape = tuple(int(size) for size in archive["keys"].shape)
        value_shape = tuple(int(size) for size in archive["values"].shape)
        if key_shape != value_shape or len(key_shape) != 5:
            raise ValueError(
                "keys and values must have matching [L,S,T,H,D] shapes, got "
                f"{key_shape} and {value_shape}"
            )
        if key_shape[2] < frames:
            raise ValueError(
                f"trace has only {key_shape[2]} frames; {frames} real frames are "
                "required (short traces are never repeated or extrapolated)"
            )
        trace_layer_ids = (
            tuple(int(item) for item in np.asarray(archive["layer_ids"]))
            if "layer_ids" in archive
            else tuple(range(key_shape[0]))
        )
        if len(trace_layer_ids) != key_shape[0] or len(set(trace_layer_ids)) != key_shape[0]:
            raise ValueError("trace layer_ids must uniquely identify every layer")
        layer_ids = (
            trace_layer_ids
            if requested_layers is None
            else tuple(int(item) for item in requested_layers)
        )
        missing_layers = sorted(set(layer_ids) - set(trace_layer_ids))
        if missing_layers:
            raise ValueError(f"trace does not contain layers {missing_layers}")
        positions = [trace_layer_ids.index(layer) for layer in layer_ids]
        keys = np.array(archive["keys"][positions, :, :frames], copy=True)
        values = np.array(archive["values"][positions, :, :frames], copy=True)
        stride_name = "trace_stride" if "trace_stride" in archive else "stride"
        metadata: dict[str, Any] = {
            "source_shape": list(key_shape),
            "evaluated_shape": list(keys.shape),
            "stride": (
                int(np.asarray(archive[stride_name]).item())
                if stride_name in archive
                else 1560
            ),
            "coordinate_system": (
                str(np.asarray(archive["coordinate_system"]).item())
                if "coordinate_system" in archive
                else None
            ),
            "cross_source_kind": (
                str(np.asarray(archive["cross_source_kind"]).item())
                if "cross_source_kind" in archive
                else None
            ),
        }
    if not np.issubdtype(keys.dtype, np.floating) or not np.issubdtype(
        values.dtype, np.floating
    ):
        raise ValueError("trace keys and values must be floating point")
    if not np.isfinite(keys).all() or not np.isfinite(values).all():
        raise ValueError("trace contains NaN or Inf")
    if metadata["cross_source_kind"] not in (None, "normalized"):
        raise ValueError("trace CrossKV source must be normalized")
    return keys, values, layer_ids, metadata


def read_base_config(path: Path | None, *, stride: int, dim: int) -> CodecConfig:
    values = asdict(CodecConfig())
    if path is None:
        values.update(
            {
                "fit_mode": "legacy",
                "k_predictor_mode": "shared",
                "cross_matmul_precision": "ieee",
            }
        )
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values.update(payload.get("codec", payload))
    values.update(
        {
            "stride": stride,
            "group_size": 64 if dim % 64 == 0 else dim,
            "use_triton": False,
            "strict_triton": False,
            "k_history_length": 2,
            "v_history_length": 1,
        }
    )
    return config_from_mapping(values)


def candidate_config(base: CodecConfig, method: str) -> CodecConfig:
    return replace(
        base,
        v_predictor_mode=("cross_kv" if method == "cross_kv_history" else "temporal"),
        v_history_length=1,
    )


def reconstruction_metrics(
    clean: torch.Tensor,
    rebuilt: torch.Tensor,
    *,
    start_frame: int = 1,
) -> dict[str, Any]:
    if clean.shape != rebuilt.shape or clean.ndim != 5:
        raise ValueError("metrics require matching [L,S,T,H,D] tensors")
    clean64 = clean[:, :, start_frame:].to(torch.float64)
    error2 = (rebuilt[:, :, start_frame:].to(torch.float64) - clean64).square()
    signal2 = clean64.square()

    def ratio(reduce_dims: tuple[int, ...]) -> torch.Tensor:
        return error2.sum(dim=reduce_dims) / signal2.sum(dim=reduce_dims).clamp_min(EPS)

    per_layer = ratio((1, 2, 3, 4))
    per_head = ratio((0, 1, 2, 4))
    per_frame = ratio((0, 1, 3, 4))
    per_layer_head = ratio((1, 2, 4))
    frame_values = per_frame.detach().cpu()
    edge = min(10, int(frame_values.numel()))
    return {
        "pooled_nmse": float(error2.sum().div(signal2.sum().clamp_min(EPS)).item()),
        "per_layer_nmse": per_layer.detach().cpu().tolist(),
        "per_head_nmse": per_head.detach().cpu().tolist(),
        "per_layer_head_nmse": per_layer_head.detach().cpu().tolist(),
        "per_frame": [
            {"frame": start_frame + index, "nmse": float(value)}
            for index, value in enumerate(frame_values.tolist())
        ],
        "distribution": {
            "layer_median": float(per_layer.median().item()),
            "layer_p95": float(torch.quantile(per_layer, 0.95).item()),
            "layer_max": float(per_layer.max().item()),
            "first_10_frame_mean": float(frame_values[:edge].mean().item()),
            "last_10_frame_mean": float(frame_values[-edge:].mean().item()),
        },
    }


def parameter_count(parameters: TemporalParameters | CrossKVParameters) -> int:
    if isinstance(parameters, TemporalParameters):
        return int(parameters.history_weights.numel() + parameters.history_bias.numel())
    return int(
        parameters.weight.numel()
        + parameters.bias.numel()
        + parameters.history_weights.numel()
    )


@torch.no_grad()
def fit_candidate(
    method: str,
    *,
    train_keys: torch.Tensor,
    train_values: torch.Tensor,
    eval_keys: torch.Tensor,
    eval_values: torch.Tensor,
    base_config: CodecConfig,
    layer_ids: tuple[int, ...],
    coordinate_system: str | None,
    iterations: int,
    output_dir: Path | None,
) -> tuple[dict[str, Any], TemporalParameters]:
    config = candidate_config(base_config, method)
    started = time.perf_counter()
    key_parameters, value_parameters = fit_predictors(
        train_keys,
        train_values,
        config=config,
        history_length=2,
        iterations=iterations,
        temporal_ridge=1.0 if config.fit_mode == "legacy" else 0.1,
        cross_ridge=0.1,
        v_history_ridge=0.0,
        max_gain=1.0,
        v_history_mode="diagonal",
        layer_ids=layer_ids,
        coordinate_system=coordinate_system,
        fit_history_source="reconstructed",
    )
    fit_seconds = time.perf_counter() - started
    evaluation_started = time.perf_counter()
    rebuilt_keys = _reconstruct_temporal_trace(eval_keys, key_parameters, config)
    if method == "cross_kv_history":
        assert isinstance(value_parameters, CrossKVParameters)
        rebuilt_values = _reconstruct_cross_kv_trace(
            rebuilt_keys, eval_values, value_parameters, config
        )
    else:
        assert isinstance(value_parameters, TemporalParameters)
        rebuilt_values = _reconstruct_temporal_trace(
            eval_values, value_parameters, config, bits=config.v_bits
        )
    result = {
        "label": (
            "CrossKV + V-history(1)"
            if method == "cross_kv_history"
            else "temporal-only V-history(1)"
        ),
        "v_parameter_count": parameter_count(value_parameters),
        "k_parameter_count": parameter_count(key_parameters),
        "v_reconstruction": reconstruction_metrics(eval_values, rebuilt_values),
        "k_reconstruction": reconstruction_metrics(eval_keys, rebuilt_keys),
        "fit_seconds": fit_seconds,
        "evaluation_seconds": time.perf_counter() - evaluation_started,
    }
    if output_dir is not None:
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        key_parameters.save(method_dir / "k_temporal.npz")
        value_parameters.save(
            method_dir
            / ("v_cross_kv.npz" if method == "cross_kv_history" else "v_temporal.npz")
        )
    return result, key_parameters.to("cpu")


def compare_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cross = float(results["cross_kv_history"]["v_reconstruction"]["pooled_nmse"])
    temporal = float(results["temporal_only"]["v_reconstruction"]["pooled_nmse"])
    cross_layers = np.asarray(
        results["cross_kv_history"]["v_reconstruction"]["per_layer_nmse"]
    )
    temporal_layers = np.asarray(
        results["temporal_only"]["v_reconstruction"]["per_layer_nmse"]
    )
    return {
        "cross_kv_nmse": cross,
        "temporal_nmse": temporal,
        "temporal_minus_cross_kv": temporal - cross,
        "cross_kv_relative_error_reduction": 1.0 - cross / max(temporal, float(EPS)),
        "lower_error_method": "cross_kv_history" if cross < temporal else "temporal_only",
        "cross_kv_layer_wins": int(np.count_nonzero(cross_layers < temporal_layers)),
        "temporal_layer_wins": int(np.count_nonzero(temporal_layers < cross_layers)),
        "ties": int(np.count_nonzero(temporal_layers == cross_layers)),
    }


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("layers must be non-empty and unique")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    keys_np, values_np, layer_ids, metadata = load_trace(
        args.trace, frames=args.frames, requested_layers=args.layers
    )
    train_ids, eval_ids = split_trace_ids(keys_np.shape[1], args.holdout_step)
    train_keys = torch.from_numpy(np.ascontiguousarray(keys_np[:, train_ids])).to(device)
    train_values = torch.from_numpy(np.ascontiguousarray(values_np[:, train_ids])).to(device)
    eval_keys = torch.from_numpy(np.ascontiguousarray(keys_np[:, eval_ids])).to(device)
    eval_values = torch.from_numpy(np.ascontiguousarray(values_np[:, eval_ids])).to(device)
    del keys_np, values_np
    base_config = read_base_config(
        args.config, stride=int(metadata["stride"]), dim=int(train_keys.shape[-1])
    )
    results: dict[str, dict[str, Any]] = {}
    key_parameters: dict[str, TemporalParameters] = {}
    for method in METHODS:
        print(f"[reconstruction] fitting {method}", file=sys.stderr, flush=True)
        result, fitted_key = fit_candidate(
            method,
            train_keys=train_keys,
            train_values=train_values,
            eval_keys=eval_keys,
            eval_values=eval_values,
            base_config=base_config,
            layer_ids=layer_ids,
            coordinate_system=metadata["coordinate_system"],
            iterations=args.iterations,
            output_dir=args.output_dir,
        )
        results[method] = result
        key_parameters[method] = fitted_key
        print(
            f"[reconstruction] {method}: V-NMSE="
            f"{result['v_reconstruction']['pooled_nmse']:.8g}",
            file=sys.stderr,
            flush=True,
        )
    comparison = compare_results(results)
    if not all(
        math.isfinite(float(value))
        for value in comparison.values()
        if isinstance(value, float)
    ):
        raise ValueError("comparison produced a non-finite metric")
    report = {
        "experiment": "v_predictor_closed_loop_reconstruction",
        "status": "complete",
        "scope": {
            "video_generation": False,
            "closed_loop": True,
            "evaluated_frames": list(range(1, args.frames)),
            "anchor_frame_excluded": True,
        },
        "trace": {
            "path": str(args.trace.resolve()),
            "sha256": sha256(args.trace),
            **metadata,
            "layer_ids": list(layer_ids),
            "train_sample_ids": train_ids.tolist(),
            "eval_sample_ids": eval_ids.tolist(),
        },
        "protocol": {
            "frames": args.frames,
            "k_history_length": 2,
            "v_history_length": 1,
            "k_bits": base_config.k_bits,
            "v_bits": base_config.v_bits,
            "fit_iterations": args.iterations,
            "fit_history_source": "reconstructed",
            "fit_mode": base_config.fit_mode,
            "k_predictor_mode": base_config.k_predictor_mode,
            "device": str(device),
        },
        "matched_k_fit": {
            "max_abs_weight_difference": float(
                (
                    key_parameters["cross_kv_history"].history_weights
                    - key_parameters["temporal_only"].history_weights
                ).abs().max().item()
            ),
            "max_abs_bias_difference": float(
                (
                    key_parameters["cross_kv_history"].history_bias
                    - key_parameters["temporal_only"].history_bias
                ).abs().max().item()
            ),
        },
        "methods": results,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "legacy.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--holdout-step", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--layers", type=parse_layers)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.frames < 3 or args.iterations < 1:
        parser.error("--frames must be at least 3 and --iterations must be positive")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
