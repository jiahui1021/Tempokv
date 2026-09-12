#!/usr/bin/env python3
"""Collect K + aligned V cosine data for the asymmetry figure.

This is the measured reverse of the production CrossKV direction.  V is the
temporally reconstructed primary tensor, and the same-position reconstructed V
plus K history predicts K.  The script writes standalone metrics and can merge
them into an existing cosine report without overwriting that report.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.codec import _cuda_matmul_precision  # noqa: E402
from tempokv.config import CodecConfig, config_from_mapping  # noqa: E402
from tempokv.fitting import (  # noqa: E402
    _reconstruct_cross_kv_trace,
    _reconstruct_temporal_trace,
    fit_predictors,
)
from tempokv.parameters import CrossKVParameters, TemporalParameters  # noqa: E402


METRIC_KEY = "k_reverse_cross_vk_history"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_backing_csvs(result: dict[str, Any], output: Path) -> tuple[Path, Path]:
    """Write the exact rows consumed by figure panels (b) and (c)."""
    layer_path = output.with_name(f"{output.stem}_per_layer.csv")
    frame_path = output.with_name(f"{output.stem}_per_frame.csv")
    layer_path.parent.mkdir(parents=True, exist_ok=True)
    with layer_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("layer_id", "cosine"))
        writer.writeheader()
        writer.writerows(
            {"layer_id": layer, "cosine": cosine}
            for layer, cosine in zip(
                result["trace"]["layer_ids"],
                result["k_aligned_v"]["per_layer_cosine"],
                strict=True,
            )
        )
    with frame_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("frame", "cosine"))
        writer.writeheader()
        writer.writerows(result["k_aligned_v"]["per_frame"])
    return layer_path, frame_path


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


def load_trace(
    path: Path,
    *,
    frames: int,
    requested_layers: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], dict[str, Any]]:
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
                f"trace has only {key_shape[2]} frames; {frames} real frames are required"
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
        metadata = {
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
    if not np.isfinite(keys).all() or not np.isfinite(values).all():
        raise ValueError("trace contains NaN or Inf")
    if metadata["cross_source_kind"] not in (None, "normalized"):
        raise ValueError("trace CrossKV source must be normalized")
    return keys, values, layer_ids, metadata


def reverse_config(path: Path, *, stride: int, dim: int) -> CodecConfig:
    values = asdict(CodecConfig())
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
            "fit_mode": "native",
            "k_predictor_mode": "layer_head",
            "cross_kv_weight_mode": "layer_head",
            "v_predictor_mode": "cross_kv",
            "predictor_direction": "v_to_k",
        }
    )
    return config_from_mapping(values)


def view_term(value: torch.Tensor, target_ndim: int, *, layered: bool) -> torch.Tensor:
    if layered:
        return value.view(
            value.shape[0],
            *([1] * (target_ndim - 3)),
            value.shape[-2],
            value.shape[-1],
        )
    if value.ndim == 2:
        return value.view(*([1] * (target_ndim - 2)), *value.shape)
    return value.view(*([1] * (target_ndim - 1)), value.shape[-1])


def temporal_predictions(
    target: torch.Tensor,
    rebuilt: torch.Tensor,
    parameters: TemporalParameters,
) -> torch.Tensor:
    predictions = torch.zeros_like(target)
    weights = parameters.history_weights.to(target.device)
    bias = parameters.history_bias.to(target.device)
    for frame in range(1, target.shape[-3]):
        order = min(frame, parameters.history_length)
        current_bias = bias[:, order - 1] if parameters.layered else bias[order - 1]
        prediction = view_term(
            current_bias,
            target[..., frame, :, :].ndim,
            layered=parameters.layered,
        )
        for lag in range(order):
            previous = rebuilt[..., frame - 1 - lag, :, :]
            current_weight = (
                weights[:, order - 1, lag]
                if parameters.layered
                else weights[order - 1, lag]
            )
            prediction = prediction + previous * view_term(
                current_weight, previous.ndim, layered=parameters.layered
            )
        predictions[..., frame, :, :] = prediction
    return predictions


def cross_predictions(
    rebuilt_source: torch.Tensor,
    rebuilt_target: torch.Tensor,
    parameters: CrossKVParameters,
    config: CodecConfig,
) -> torch.Tensor:
    predictions = torch.zeros_like(rebuilt_target)
    weight = parameters.weight.to(rebuilt_target.device)
    bias = parameters.bias.to(rebuilt_target.device)
    history_weights = parameters.history_weights.to(rebuilt_target.device)
    diagonal_ndim = 4 if parameters.layered else 3
    with _cuda_matmul_precision(rebuilt_target.device, config.cross_matmul_precision):
        for frame in range(1, rebuilt_target.shape[-3]):
            source = rebuilt_source[..., frame, :, :].float()
            if parameters.layered:
                prediction = torch.einsum("l...hi,lhij->l...hj", source, weight)
            else:
                prediction = torch.einsum("...hi,hij->...hj", source, weight)
            prediction = prediction + view_term(
                bias, prediction.ndim, layered=parameters.layered
            )
            order = min(frame, parameters.history_length)
            for lag in range(order):
                history_weight = (
                    history_weights[:, lag]
                    if parameters.layered
                    else history_weights[lag]
                )
                previous = rebuilt_target[..., frame - 1 - lag, :, :].float()
                if history_weights.ndim == diagonal_ndim:
                    contribution = previous * view_term(
                        history_weight, previous.ndim, layered=parameters.layered
                    )
                elif parameters.layered:
                    contribution = torch.einsum(
                        "l...hi,lhij->l...hj", previous, history_weight
                    )
                else:
                    contribution = torch.einsum(
                        "...hi,hij->...hj", previous, history_weight
                    )
                prediction = prediction + contribution
            predictions[..., frame, :, :] = prediction
    return predictions


def cosine_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    start_frame: int = 1,
) -> dict[str, Any]:
    cosine = F.cosine_similarity(
        target[:, :, start_frame:].float(),
        prediction[:, :, start_frame:].float(),
        dim=-1,
        eps=1e-8,
    ).to(torch.float64)
    return summarize_cosine(cosine, start_frame=start_frame)


def summarize_cosine(
    cosine: torch.Tensor,
    *,
    start_frame: int = 1,
) -> dict[str, Any]:
    """Summarize cosine values shaped [L,S,T,H]."""
    if cosine.ndim != 4:
        raise ValueError("cosine values must have [L,S,T,H] shape")
    per_layer = cosine.mean(dim=(1, 2, 3))
    per_head = cosine.mean(dim=(0, 1, 2))
    per_frame = cosine.mean(dim=(0, 1, 3))
    per_layer_head = cosine.mean(dim=(1, 2))
    frame_values = per_frame.detach().cpu()
    edge = min(10, int(frame_values.numel()))
    return {
        "mean_cosine": float(cosine.mean().item()),
        "per_layer_cosine": per_layer.detach().cpu().tolist(),
        "per_head_cosine": per_head.detach().cpu().tolist(),
        "per_layer_head_cosine": per_layer_head.detach().cpu().tolist(),
        "per_frame": [
            {"frame": start_frame + index, "cosine": float(value)}
            for index, value in enumerate(frame_values.tolist())
        ],
        "distribution": {
            "vector_median": float(cosine.median().item()),
            "vector_p05": float(torch.quantile(cosine, 0.05).item()),
            "vector_min": float(cosine.min().item()),
            "first_10_frame_mean": float(frame_values[:edge].mean().item()),
            "last_10_frame_mean": float(frame_values[-edge:].mean().item()),
        },
    }


def validate_base_report(
    report: dict[str, Any],
    *,
    trace_sha256: str,
    evaluated_shape: list[int],
    train_ids: np.ndarray,
    eval_ids: np.ndarray,
    frames: int,
) -> None:
    expected = {
        "trace.sha256": (report["trace"].get("sha256"), trace_sha256),
        "trace.evaluated_shape": (
            report["trace"].get("evaluated_shape"),
            evaluated_shape,
        ),
        "trace.train_sample_ids": (
            report["trace"].get("train_sample_ids"),
            train_ids.tolist(),
        ),
        "trace.eval_sample_ids": (
            report["trace"].get("eval_sample_ids"),
            eval_ids.tolist(),
        ),
        "protocol.frames": (report["protocol"].get("frames"), frames),
    }
    mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
    if mismatches:
        raise ValueError(
            "base report does not match this run: " + ", ".join(mismatches)
        )


def merge_report(
    base: dict[str, Any],
    *,
    key_metrics: dict[str, Any],
    source_metrics: dict[str, Any],
    fit_seconds: float,
    evaluation_seconds: float,
    layer_ids: Sequence[int],
) -> dict[str, Any]:
    merged = deepcopy(base)
    merged["methods"]["reverse_cross_vk_history"] = {
        "label": "K + aligned V (reverse CrossVK + K-history(1))",
        "key": key_metrics,
        "value_temporal_source": source_metrics,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
    }
    comparison = merged["cosine_comparison"]
    comparison["mean_cosine"][METRIC_KEY] = key_metrics["mean_cosine"]
    for index, row in enumerate(comparison["per_layer"]):
        if int(row["layer_id"]) != int(layer_ids[index]):
            raise ValueError("base report layer order does not match reverse metrics")
        row[METRIC_KEY] = key_metrics["per_layer_cosine"][index]
    for index, row in enumerate(comparison["per_frame"]):
        metric_row = key_metrics["per_frame"][index]
        if int(row["frame"]) != int(metric_row["frame"]):
            raise ValueError("base report frame order does not match reverse metrics")
        row[METRIC_KEY] = metric_row["cosine"]
    k_temporal = comparison["mean_cosine"]["k_temporal"]
    v_cross = comparison["mean_cosine"]["v_cross_kv_history"]
    comparison["k_reverse_cross_minus_temporal"] = (
        key_metrics["mean_cosine"] - k_temporal
    )
    comparison["cross_direction_difference_v_to_k_minus_k_to_v"] = (
        key_metrics["mean_cosine"] - v_cross
    )
    merged["protocol"]["reverse_cross_direction"] = (
        "current reconstructed V -> K + K-history(1)"
    )
    return merged


def raw_layer_ids(path: Path) -> tuple[int, ...] | None:
    """Return layer ids for an L{n}_k/L{n}_v archive, or None."""
    with np.load(path, allow_pickle=False) as archive:
        if "keys" in archive or "values" in archive:
            return None
        ids = sorted(
            int(name[1:-2])
            for name in archive.files
            if name.startswith("L") and name.endswith("_k") and name[1:-2].isdigit()
        )
        if not ids:
            return None
        missing = [layer for layer in ids if f"L{layer}_v" not in archive]
        if missing:
            raise ValueError(f"raw archive is missing V arrays for layers {missing}")
        return tuple(ids)


def reshape_raw_layer(array: np.ndarray, *, frames: int) -> np.ndarray:
    """Convert [B,T*S,H,D] time-major raw cache data to [B*S,T,H,D]."""
    if array.ndim != 4:
        raise ValueError(f"raw layer must have [B,tokens,H,D] shape, got {array.shape}")
    batch, tokens, heads, dim = array.shape
    if tokens % frames:
        raise ValueError(f"raw token count {tokens} is not divisible by {frames} frames")
    positions = tokens // frames
    return (
        array.reshape(batch, frames, positions, heads, dim)
        .transpose(0, 2, 1, 3, 4)
        .reshape(batch * positions, frames, heads, dim)
    )


@torch.no_grad()
def run_raw_archive(
    args: argparse.Namespace,
    available_layers: tuple[int, ...],
) -> dict[str, Any]:
    """Run one layer at a time for the 180-frame raw-layer archive."""
    layer_ids = available_layers if args.layers is None else args.layers
    missing = sorted(set(layer_ids) - set(available_layers))
    if missing:
        raise ValueError(f"raw archive does not contain layers {missing}")

    with np.load(args.trace, allow_pickle=False) as archive:
        first = np.asarray(archive[f"L{layer_ids[0]}_k"])
        first_trace = reshape_raw_layer(first, frames=args.frames)
        sample_count, _, heads, dim = first_trace.shape
        first_trace_shape = tuple(first_trace.shape)
        raw_shape = list(first.shape)
        del first, first_trace
    train_ids, eval_ids = split_trace_ids(sample_count, args.holdout_step)
    evaluated_shape = [len(layer_ids), sample_count, args.frames, heads, dim]
    trace_digest = sha256(args.trace)
    base = None
    if args.base_report is not None:
        base = json.loads(args.base_report.read_text(encoding="utf-8"))
        # The base report names the derived stacked archive, while this input is
        # its layer-wise raw source.  Shape and exact split still have to match.
        validate_base_report(
            base,
            trace_sha256=base["trace"].get("sha256"),
            evaluated_shape=evaluated_shape,
            train_ids=train_ids,
            eval_ids=eval_ids,
            frames=args.frames,
        )
    stride = int(base["trace"].get("stride", 1560)) if base is not None else 1560
    config = reverse_config(args.config, stride=stride, dim=dim)

    key_cosines: list[torch.Tensor] = []
    source_cosines: list[torch.Tensor] = []
    fit_seconds = 0.0
    evaluation_seconds = 0.0
    with np.load(args.trace, allow_pickle=False) as archive:
        for layer in layer_ids:
            keys_np = reshape_raw_layer(
                np.array(archive[f"L{layer}_k"], copy=True), frames=args.frames
            )
            values_np = reshape_raw_layer(
                np.array(archive[f"L{layer}_v"], copy=True), frames=args.frames
            )
            if keys_np.shape != first_trace_shape or values_np.shape != first_trace_shape:
                raise ValueError(f"layer {layer} shape does not match layer {layer_ids[0]}")
            if not np.isfinite(keys_np).all() or not np.isfinite(values_np).all():
                raise ValueError(f"layer {layer} contains NaN or Inf")

            train_keys = torch.from_numpy(np.ascontiguousarray(keys_np[train_ids]))[
                None
            ].to(args.device)
            train_values = torch.from_numpy(np.ascontiguousarray(values_np[train_ids]))[
                None
            ].to(args.device)
            eval_keys = torch.from_numpy(np.ascontiguousarray(keys_np[eval_ids]))[
                None
            ].to(args.device)
            eval_values = torch.from_numpy(np.ascontiguousarray(values_np[eval_ids]))[
                None
            ].to(args.device)
            del keys_np, values_np

            fit_started = time.perf_counter()
            temporal_v, cross_k = fit_predictors(
                train_keys,
                train_values,
                config=config,
                history_length=2,
                iterations=args.iterations,
                temporal_ridge=0.1,
                cross_ridge=0.1,
                v_history_ridge=0.0,
                max_gain=1.0,
                v_history_mode="diagonal",
                layer_ids=(layer,),
                coordinate_system="pre_rope_normalized",
                fit_history_source="reconstructed",
            )
            if not isinstance(cross_k, CrossKVParameters):
                raise TypeError("reverse fit did not return CrossKV parameters for K")
            fit_seconds += time.perf_counter() - fit_started

            evaluation_started = time.perf_counter()
            rebuilt_values = _reconstruct_temporal_trace(eval_values, temporal_v, config)
            value_prediction = temporal_predictions(
                eval_values, rebuilt_values, temporal_v
            )
            rebuilt_keys = _reconstruct_cross_kv_trace(
                rebuilt_values, eval_keys, cross_k, config
            )
            key_prediction = cross_predictions(
                rebuilt_values, rebuilt_keys, cross_k, config
            )
            key_cosine = F.cosine_similarity(
                eval_keys[:, :, 1:].float(), key_prediction[:, :, 1:].float(), dim=-1
            ).cpu()
            source_cosine = F.cosine_similarity(
                eval_values[:, :, 1:].float(),
                value_prediction[:, :, 1:].float(),
                dim=-1,
            ).cpu()
            evaluation_seconds += time.perf_counter() - evaluation_started
            key_cosines.append(key_cosine)
            source_cosines.append(source_cosine)
            print(
                f"[K+V] layer {layer:02d}: cosine={key_cosine.mean().item():.8f}",
                file=sys.stderr,
                flush=True,
            )
            del train_keys, train_values, eval_keys, eval_values
            del rebuilt_values, value_prediction, rebuilt_keys, key_prediction

    key_metrics = summarize_cosine(torch.cat(key_cosines, dim=0).double())
    source_metrics = summarize_cosine(torch.cat(source_cosines, dim=0).double())
    result = {
        "experiment": "k_aligned_v_closed_loop_cosine",
        "status": "complete",
        "trace": {
            "path": str(args.trace.resolve()),
            "sha256": trace_digest,
            "source_format": "layerwise_raw_time_major",
            "raw_layer_shape": raw_shape,
            "evaluated_shape": evaluated_shape,
            "stride": stride,
            "coordinate_system": "pre_rope_normalized",
            "cross_source_kind": "normalized",
            "layer_ids": list(layer_ids),
            "train_sample_ids": train_ids.tolist(),
            "eval_sample_ids": eval_ids.tolist(),
            "base_report_trace_sha256": (
                base["trace"].get("sha256") if base is not None else None
            ),
        },
        "protocol": {
            "frames": args.frames,
            "k_history_length": 1,
            "v_primary_history_length": 2,
            "k_bits": config.k_bits,
            "v_bits": config.v_bits,
            "fit_iterations": args.iterations,
            "fit_history_source": "reconstructed",
            "direction": "current reconstructed V -> K + K-history(1)",
            "device": str(args.device),
        },
        "k_aligned_v": key_metrics,
        "v_temporal_source": source_metrics,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_backing_csvs(result, args.output)
    if args.merged_output is not None:
        if base is None:
            raise ValueError("--merged-output requires --base-report")
        merged = merge_report(
            base,
            key_metrics=key_metrics,
            source_metrics=source_metrics,
            fit_seconds=fit_seconds,
            evaluation_seconds=evaluation_seconds,
            layer_ids=layer_ids,
        )
        args.merged_output.parent.mkdir(parents=True, exist_ok=True)
        args.merged_output.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    raw_layers = raw_layer_ids(args.trace)
    if raw_layers is not None:
        return run_raw_archive(args, raw_layers)

    keys_np, values_np, layer_ids, metadata = load_trace(
        args.trace, frames=args.frames, requested_layers=args.layers
    )
    train_ids, eval_ids = split_trace_ids(keys_np.shape[1], args.holdout_step)
    trace_digest = sha256(args.trace)
    if args.base_report is not None:
        base = json.loads(args.base_report.read_text(encoding="utf-8"))
        validate_base_report(
            base,
            trace_sha256=trace_digest,
            evaluated_shape=metadata["evaluated_shape"],
            train_ids=train_ids,
            eval_ids=eval_ids,
            frames=args.frames,
        )
    else:
        base = None

    train_keys = torch.from_numpy(np.ascontiguousarray(keys_np[:, train_ids])).to(device)
    train_values = torch.from_numpy(np.ascontiguousarray(values_np[:, train_ids])).to(device)
    eval_keys = torch.from_numpy(np.ascontiguousarray(keys_np[:, eval_ids])).to(device)
    eval_values = torch.from_numpy(np.ascontiguousarray(values_np[:, eval_ids])).to(device)
    del keys_np, values_np

    config = reverse_config(
        args.config, stride=int(metadata["stride"]), dim=int(train_keys.shape[-1])
    )
    fit_started = time.perf_counter()
    temporal_v, cross_k = fit_predictors(
        train_keys,
        train_values,
        config=config,
        history_length=2,
        iterations=args.iterations,
        temporal_ridge=0.1,
        cross_ridge=0.1,
        v_history_ridge=0.0,
        max_gain=1.0,
        v_history_mode="diagonal",
        layer_ids=layer_ids,
        coordinate_system=metadata["coordinate_system"],
        fit_history_source="reconstructed",
    )
    if not isinstance(cross_k, CrossKVParameters):
        raise TypeError("reverse fit did not return CrossKV parameters for K")
    fit_seconds = time.perf_counter() - fit_started

    evaluation_started = time.perf_counter()
    rebuilt_values = _reconstruct_temporal_trace(eval_values, temporal_v, config)
    value_prediction = temporal_predictions(eval_values, rebuilt_values, temporal_v)
    rebuilt_keys = _reconstruct_cross_kv_trace(
        rebuilt_values, eval_keys, cross_k, config
    )
    key_prediction = cross_predictions(rebuilt_values, rebuilt_keys, cross_k, config)
    key_metrics = cosine_metrics(eval_keys, key_prediction)
    source_metrics = cosine_metrics(eval_values, value_prediction)
    evaluation_seconds = time.perf_counter() - evaluation_started

    result = {
        "experiment": "k_aligned_v_closed_loop_cosine",
        "status": "complete",
        "trace": {
            "path": str(args.trace.resolve()),
            "sha256": trace_digest,
            **metadata,
            "layer_ids": list(layer_ids),
            "train_sample_ids": train_ids.tolist(),
            "eval_sample_ids": eval_ids.tolist(),
        },
        "protocol": {
            "frames": args.frames,
            "k_history_length": 1,
            "v_primary_history_length": 2,
            "k_bits": config.k_bits,
            "v_bits": config.v_bits,
            "fit_iterations": args.iterations,
            "fit_history_source": "reconstructed",
            "direction": "current reconstructed V -> K + K-history(1)",
            "device": str(device),
        },
        "k_aligned_v": key_metrics,
        "v_temporal_source": source_metrics,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_backing_csvs(result, args.output)

    if args.merged_output is not None:
        if base is None:
            raise ValueError("--merged-output requires --base-report")
        merged = merge_report(
            base,
            key_metrics=key_metrics,
            source_metrics=source_metrics,
            fit_seconds=fit_seconds,
            evaluation_seconds=evaluation_seconds,
            layer_ids=layer_ids,
        )
        args.merged_output.parent.mkdir(parents=True, exist_ok=True)
        args.merged_output.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def parse_layers(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("layers must be non-empty and unique")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--base-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "native.json")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--holdout-step", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--layers", type=parse_layers)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frames < 3 or args.iterations < 1:
        raise SystemExit("--frames must be at least 3 and --iterations must be positive")
    if args.threads is not None and args.threads < 1:
        raise SystemExit("--threads must be positive")
    result = run(args)
    print(f"K + aligned V mean cosine: {result['k_aligned_v']['mean_cosine']:.8f}")
    print(f"per-layer points: {len(result['k_aligned_v']['per_layer_cosine'])}")
    print(f"per-frame points: {len(result['k_aligned_v']['per_frame'])}")


if __name__ == "__main__":
    main()
