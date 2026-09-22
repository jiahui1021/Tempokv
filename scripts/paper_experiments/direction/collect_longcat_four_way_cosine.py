#!/usr/bin/env python3
"""Collect the four-way closed-loop cosine study on a LongCat trace.

The four measured series are:

* K temporal: fitted two-frame temporal K predictor;
* K + aligned V: reconstructed V plus one-frame K history predicts K;
* V temporal only: fitted one-frame temporal V predictor;
* V + aligned K: reconstructed K plus one-frame V history predicts V.

Both merged ``clean_trace.npz`` and raw ``event_*/layer_*.npz`` LongCat traces
are supported.  Raw traces are processed one layer at a time.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.fitting import (  # noqa: E402
    _reconstruct_cross_kv_trace,
    _reconstruct_temporal_trace,
    fit_predictors,
)
from tempokv.parameters import CrossKVParameters, TemporalParameters  # noqa: E402

from scripts.paper_experiments.direction.collect_k_aligned_v_cosine import (  # noqa: E402
    cross_predictions,
    reverse_config,
    split_trace_ids,
    summarize_cosine,
    temporal_predictions,
)
from scripts.paper_experiments.direction.collect_longcat_k_identity_mean_cosine import (  # noqa: E402
    LongCatTrace,
)


K_TEMPORAL = "k_temporal"
K_PLUS_V = "k_reverse_cross_vk_history"
V_TEMPORAL = "v_temporal_only"
V_PLUS_K = "v_cross_kv_history"
METHODS = (K_TEMPORAL, K_PLUS_V, V_TEMPORAL, V_PLUS_K)


def cosine_values(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(
        target[:, :, 1:].float(),
        prediction[:, :, 1:].float(),
        dim=-1,
        eps=1e-8,
    ).cpu()


def write_csvs(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    summary_path = output.with_name(f"{output.stem}_summary.csv")
    layer_path = output.with_name(f"{output.stem}_per_layer.csv")
    frame_path = output.with_name(f"{output.stem}_per_frame.csv")
    comparison = report["cosine_comparison"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "mean_cosine"))
        writer.writeheader()
        for method in METHODS:
            writer.writerow(
                {"method": method, "mean_cosine": comparison["mean_cosine"][method]}
            )
    with layer_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("layer_id", *METHODS))
        writer.writeheader()
        writer.writerows(comparison["per_layer"])
    with frame_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("frame", *METHODS))
        writer.writeheader()
        writer.writerows(comparison["per_frame"])
    return summary_path, layer_path, frame_path


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    trace = LongCatTrace(args.trace, frames=args.frames)
    trace.validate_longcat_shape(allow_dynamic=args.allow_dynamic_shape)
    layer_ids = trace.layer_ids if args.layers is None else args.layers
    missing = sorted(set(layer_ids) - set(trace.layer_ids))
    if missing:
        raise ValueError(f"trace does not contain layers {missing}")
    train_ids, eval_ids = split_trace_ids(trace.samples, args.holdout_step)

    reverse = reverse_config(args.config, stride=trace.stride, dim=trace.dim)
    forward = replace(reverse, predictor_direction="k_to_v")
    temporal_only = replace(forward, v_predictor_mode="temporal")
    metric_values: dict[str, list[torch.Tensor]] = {method: [] for method in METHODS}
    max_k_weight_difference = 0.0
    max_k_bias_difference = 0.0
    fit_seconds = 0.0
    evaluation_seconds = 0.0
    started = time.perf_counter()

    for layer, keys_np, values_np in trace.kv_layers(layer_ids):
        expected = (trace.samples, args.frames, trace.heads, trace.dim)
        if keys_np.shape != expected or values_np.shape != expected:
            raise ValueError(f"layer {layer} has unexpected K/V shape")
        if not np.isfinite(keys_np).all() or not np.isfinite(values_np).all():
            raise ValueError(f"layer {layer} contains NaN or Inf")
        train_keys = torch.from_numpy(np.ascontiguousarray(keys_np[train_ids]))[
            None
        ].to(device)
        train_values = torch.from_numpy(np.ascontiguousarray(values_np[train_ids]))[
            None
        ].to(device)
        eval_keys = torch.from_numpy(np.ascontiguousarray(keys_np[eval_ids]))[
            None
        ].to(device)
        eval_values = torch.from_numpy(np.ascontiguousarray(values_np[eval_ids]))[
            None
        ].to(device)

        fit_started = time.perf_counter()
        forward_k, forward_v = fit_predictors(
            train_keys,
            train_values,
            config=forward,
            history_length=2,
            iterations=args.iterations,
            temporal_ridge=0.1,
            cross_ridge=0.1,
            v_history_ridge=0.0,
            max_gain=1.0,
            v_history_mode="diagonal",
            layer_ids=(layer,),
            coordinate_system=trace.coordinate_system,
            fit_history_source="reconstructed",
        )
        temporal_k, temporal_v = fit_predictors(
            train_keys,
            train_values,
            config=temporal_only,
            history_length=2,
            iterations=args.iterations,
            temporal_ridge=0.1,
            cross_ridge=0.1,
            v_history_ridge=0.0,
            max_gain=1.0,
            v_history_mode="diagonal",
            layer_ids=(layer,),
            coordinate_system=trace.coordinate_system,
            fit_history_source="reconstructed",
        )
        reverse_v, reverse_k = fit_predictors(
            train_keys,
            train_values,
            config=reverse,
            history_length=2,
            iterations=args.iterations,
            temporal_ridge=0.1,
            cross_ridge=0.1,
            v_history_ridge=0.0,
            max_gain=1.0,
            v_history_mode="diagonal",
            layer_ids=(layer,),
            coordinate_system=trace.coordinate_system,
            fit_history_source="reconstructed",
        )
        fit_seconds += time.perf_counter() - fit_started
        if not isinstance(forward_v, CrossKVParameters):
            raise TypeError("K -> V fit did not return CrossKV V parameters")
        if not isinstance(temporal_v, TemporalParameters):
            raise TypeError("temporal-only fit did not return temporal V parameters")
        if not isinstance(reverse_k, CrossKVParameters):
            raise TypeError("V -> K fit did not return CrossKV K parameters")
        max_k_weight_difference = max(
            max_k_weight_difference,
            float(
                (forward_k.history_weights - temporal_k.history_weights)
                .abs()
                .max()
                .item()
            ),
        )
        max_k_bias_difference = max(
            max_k_bias_difference,
            float(
                (forward_k.history_bias - temporal_k.history_bias)
                .abs()
                .max()
                .item()
            ),
        )

        evaluation_started = time.perf_counter()
        rebuilt_k = _reconstruct_temporal_trace(eval_keys, forward_k, forward)
        prediction_k = temporal_predictions(eval_keys, rebuilt_k, forward_k)
        rebuilt_v_cross = _reconstruct_cross_kv_trace(
            rebuilt_k, eval_values, forward_v, forward
        )
        prediction_v_cross = cross_predictions(
            rebuilt_k, rebuilt_v_cross, forward_v, forward
        )
        rebuilt_v_temporal = _reconstruct_temporal_trace(
            eval_values, temporal_v, temporal_only, bits=temporal_only.v_bits
        )
        prediction_v_temporal = temporal_predictions(
            eval_values, rebuilt_v_temporal, temporal_v
        )
        rebuilt_v_primary = _reconstruct_temporal_trace(
            eval_values, reverse_v, reverse
        )
        rebuilt_k_cross = _reconstruct_cross_kv_trace(
            rebuilt_v_primary, eval_keys, reverse_k, reverse
        )
        prediction_k_cross = cross_predictions(
            rebuilt_v_primary, rebuilt_k_cross, reverse_k, reverse
        )
        evaluation_seconds += time.perf_counter() - evaluation_started

        layer_cosines = {
            K_TEMPORAL: cosine_values(eval_keys, prediction_k),
            K_PLUS_V: cosine_values(eval_keys, prediction_k_cross),
            V_TEMPORAL: cosine_values(eval_values, prediction_v_temporal),
            V_PLUS_K: cosine_values(eval_values, prediction_v_cross),
        }
        for method, values in layer_cosines.items():
            metric_values[method].append(values)
        print(
            f"[LongCat cosine] layer {layer:02d}: "
            + " ".join(
                f"{method}={values.mean().item():.8f}"
                for method, values in layer_cosines.items()
            ),
            file=sys.stderr,
            flush=True,
        )

    metrics = {
        method: summarize_cosine(torch.cat(values, dim=0).double())
        for method, values in metric_values.items()
    }
    per_layer = [
        {
            "layer_id": int(layer),
            **{
                method: metrics[method]["per_layer_cosine"][index]
                for method in METHODS
            },
        }
        for index, layer in enumerate(layer_ids)
    ]
    per_frame = [
        {
            "frame": frame,
            **{
                method: metrics[method]["per_frame"][frame - 1]["cosine"]
                for method in METHODS
            },
        }
        for frame in range(1, args.frames)
    ]
    comparison = {
        "definition": "mean cosine over every [layer,sample,head] D-vector",
        "mean_cosine": {
            method: metrics[method]["mean_cosine"] for method in METHODS
        },
        "k_cross_minus_temporal": (
            metrics[K_PLUS_V]["mean_cosine"] - metrics[K_TEMPORAL]["mean_cosine"]
        ),
        "v_cross_minus_temporal": (
            metrics[V_PLUS_K]["mean_cosine"] - metrics[V_TEMPORAL]["mean_cosine"]
        ),
        "per_layer": per_layer,
        "per_frame": per_frame,
    }
    report = {
        "experiment": "longcat_four_way_closed_loop_cosine",
        "status": "complete",
        "model": "longcat",
        "trace": {
            "path": str(trace.path),
            "fingerprint": trace.fingerprint,
            "fingerprint_kind": (
                "sha256" if trace.path.is_file() else "relative_path_and_size_sha256"
            ),
            "source_format": trace.source_format,
            "source_shape": trace.source_shape,
            "evaluated_shape": [
                len(layer_ids),
                len(eval_ids),
                args.frames,
                trace.heads,
                trace.dim,
            ],
            "layer_ids": list(layer_ids),
            "train_sample_ids": train_ids.tolist(),
            "eval_sample_ids": eval_ids.tolist(),
            "trace_stride": trace.stride,
            "coordinate_system": trace.coordinate_system,
        },
        "protocol": {
            "frames": args.frames,
            "k_history_length": 2,
            "cross_target_history_length": 1,
            "k_bits": forward.k_bits,
            "v_bits": forward.v_bits,
            "fit_iterations": args.iterations,
            "fit_history_source": "reconstructed",
            "device": str(device),
        },
        "matched_k_fit": {
            "max_abs_weight_difference": max_k_weight_difference,
            "max_abs_bias_difference": max_k_bias_difference,
        },
        "methods": metrics,
        "cosine_comparison": comparison,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_paths = write_csvs(report, args.output)
    report["csv_files"] = [str(path.resolve()) for path in csv_paths]
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("layers must be non-empty and unique")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", type=Path, required=True, help="LongCat clean trace or raw root"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "native.json"
    )
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--holdout-step", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--layers", type=parse_layers)
    parser.add_argument("--allow-dynamic-shape", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frames < 3 or args.iterations < 1:
        raise SystemExit("--frames must be at least 3 and --iterations must be positive")
    if args.threads is not None and args.threads < 1:
        raise SystemExit("--threads must be positive")
    report = run(args)
    for method in METHODS:
        print(f"{method}: {report['cosine_comparison']['mean_cosine'][method]:.8f}")


if __name__ == "__main__":
    main()
