#!/usr/bin/env python3
"""Held-out ablation of Cross, temporal, and K-innovation V sources.

K always uses one shared temporal predictor.  Each V candidate is fitted from
scratch with exactly the requested source terms and one shared affine bias.
The default innovation follows the experiment definition ``K_origin-K_pred``;
this is an offline oracle because a decoder only observes the quantized K
residual.  ``--innovation-source reconstructed`` runs the deployable analogue.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.codec import anchor_roundtrip  # noqa: E402
from tempokv.config import CodecConfig, config_from_mapping  # noqa: E402
from tempokv.fitting import (  # noqa: E402
    _quantization_roundtrip,
    _reconstruct_direct_trace,
    _reconstruct_temporal_trace,
    fit_legacy_shared_temporal,
)
from tempokv.parameters import TemporalParameters  # noqa: E402


TERM_SETS: dict[str, tuple[str, ...]] = {
    "cross": ("cross",),
    "temporal": ("temporal",),
    "innovation": ("innovation",),
    "cross_temporal": ("cross", "temporal"),
    "cross_innovation": ("cross", "innovation"),
    "temporal_innovation": ("temporal", "innovation"),
}
EPS = 1e-12


@dataclass(frozen=True, slots=True)
class JointVParameters:
    """Per-layer/head dense maps for an experimental set of V sources."""

    terms: tuple[str, ...]
    weights: dict[str, torch.Tensor]
    bias: torch.Tensor
    weight_mode: str

    @property
    def parameter_count(self) -> int:
        return int(self.bias.numel() + sum(value.numel() for value in self.weights.values()))


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


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not layers or len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layers must be non-empty and unique")
    return layers


def load_trace(
    path: Path,
    *,
    frames: int,
    requested_layers: Sequence[int] | None,
    holdout_step: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    if frames < 3:
        raise ValueError("frames must be at least 3")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in ("keys", "values") if name not in archive]
        if missing:
            raise ValueError("trace is missing arrays: " + ", ".join(missing))
        key_shape = tuple(int(size) for size in archive["keys"].shape)
        value_shape = tuple(int(size) for size in archive["values"].shape)
        if key_shape != value_shape or len(key_shape) != 5:
            raise ValueError(
                "keys and values must have matching [L,S,T,H,D] shapes, got "
                f"{key_shape} and {value_shape}"
            )
        if key_shape[2] < frames:
            raise ValueError(f"trace has {key_shape[2]} frames, requested {frames}")
        trace_layer_ids = (
            tuple(int(item) for item in np.asarray(archive["layer_ids"]))
            if "layer_ids" in archive
            else tuple(range(key_shape[0]))
        )
        layer_ids = trace_layer_ids if requested_layers is None else tuple(requested_layers)
        missing_layers = sorted(set(layer_ids) - set(trace_layer_ids))
        if missing_layers:
            raise ValueError(f"trace does not contain layers {missing_layers}")
        positions = [trace_layer_ids.index(layer) for layer in layer_ids]
        train_ids, eval_ids = split_trace_ids(key_shape[1], holdout_step)

        key_array = np.asarray(archive["keys"])[positions, :, :frames]
        train_keys = torch.from_numpy(np.ascontiguousarray(key_array[:, train_ids]))
        eval_keys = torch.from_numpy(np.ascontiguousarray(key_array[:, eval_ids]))
        del key_array
        value_array = np.asarray(archive["values"])[positions, :, :frames]
        train_values = torch.from_numpy(np.ascontiguousarray(value_array[:, train_ids]))
        eval_values = torch.from_numpy(np.ascontiguousarray(value_array[:, eval_ids]))
        del value_array

        stride_name = "trace_stride" if "trace_stride" in archive else "stride"
        metadata = {
            "source_shape": list(key_shape),
            "evaluated_shape": [len(layer_ids), key_shape[1], frames, key_shape[3], key_shape[4]],
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
    for name, tensor in (
        ("train keys", train_keys),
        ("eval keys", eval_keys),
        ("train values", train_values),
        ("eval values", eval_values),
    ):
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain finite floating-point values")
    return (
        train_keys,
        train_values,
        eval_keys,
        eval_values,
        layer_ids,
        train_ids,
        eval_ids,
        metadata,
    )


def read_config(path: Path, *, stride: int, dim: int) -> CodecConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "codec" in payload:
        payload = payload["codec"]
    payload = {
        **payload,
        "stride": stride,
        "group_size": 64 if dim % 64 == 0 else dim,
        "use_triton": False,
        "strict_triton": False,
        "k_history_length": 2,
        "v_history_length": 1,
    }
    config = config_from_mapping(payload)
    if config.fit_mode != "legacy" or config.k_predictor_mode != "shared":
        raise ValueError("this diagnostic requires legacy shared temporal K")
    return config


def temporal_predictions(
    rebuilt: torch.Tensor,
    parameters: TemporalParameters,
) -> torch.Tensor:
    """Return the exact K predictions used before residual quantization."""
    predictions = torch.zeros_like(rebuilt, dtype=torch.float32)
    weights = parameters.history_weights.to(rebuilt.device)
    bias = parameters.history_bias.to(rebuilt.device)
    prefix_dimensions = rebuilt.ndim - 4
    channel_shape = (1, *(1 for _ in range(prefix_dimensions + 1)), rebuilt.shape[-1])
    for frame in range(1, rebuilt.shape[-3]):
        order = min(frame, parameters.history_length)
        prediction = bias[order - 1].view(channel_shape).expand_as(
            rebuilt[..., frame, :, :]
        ).clone()
        for lag in range(1, order + 1):
            coefficient = weights[order - 1, lag - 1].view(channel_shape)
            prediction = prediction + rebuilt[..., frame - lag, :, :].float() * coefficient
        predictions[..., frame, :, :] = prediction
    return predictions


def key_innovation(
    clean_key: torch.Tensor,
    rebuilt_key: torch.Tensor,
    key_prediction: torch.Tensor,
    *,
    source: str,
) -> torch.Tensor:
    """Construct the requested clean oracle or its decoder-visible analogue."""
    if source == "clean":
        innovation = clean_key.float() - key_prediction.float()
    elif source == "reconstructed":
        innovation = rebuilt_key.float() - key_prediction.float()
    else:
        raise ValueError("innovation source must be clean or reconstructed")
    innovation = innovation.clone()
    innovation[..., 0, :, :] = 0.0
    return innovation


def temporal_source(rebuilt_value: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(rebuilt_value, dtype=torch.float32)
    result[..., 1:, :, :] = rebuilt_value[..., :-1, :, :].float()
    return result


@torch.no_grad()
def fit_joint_dense_predictor(
    terms: tuple[str, ...],
    sources: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    *,
    ridge: float,
    weight_mode: str = "layer_shared",
) -> JointVParameters:
    """Joint ridge fit with one dense D-by-D map per source and one bias."""
    if not terms or any(term not in sources for term in terms):
        raise ValueError("every requested term must have a source tensor")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    if weight_mode not in {"layer_shared", "layer_head"}:
        raise ValueError("weight_mode must be layer_shared or layer_head")
    if any(sources[term].shape != target.shape for term in terms):
        raise ValueError("all sources must match the [L,S,T,H,D] target shape")
    layer_count, _, _, head_count, dim = target.shape
    feature_dim = len(terms) * dim
    identity = torch.eye(feature_dim, dtype=torch.float32, device=target.device)
    fitted: dict[str, list[torch.Tensor]] = {term: [] for term in terms}
    biases: list[torch.Tensor] = []

    def solve(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_mean = x.mean(dim=1)
        y_mean = y.mean(dim=1)
        xc = x - x_mean[:, None]
        yc = y - y_mean[:, None]
        gram = torch.matmul(xc.transpose(-2, -1), xc)
        rhs = torch.matmul(xc.transpose(-2, -1), yc)
        scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
        system = gram + ridge * scale[:, None, None] * identity
        if ridge:
            weight = torch.linalg.solve(system, rhs)
        else:
            weight = torch.matmul(torch.linalg.pinv(system, rtol=1e-6), rhs)
        bias = y_mean - torch.matmul(x_mean.unsqueeze(-2), weight).squeeze(-2)
        return weight, bias

    if weight_mode == "layer_shared":
        x = torch.cat(
            [sources[term][:, :, 1:].float() for term in terms], dim=-1
        ).reshape(-1, head_count, feature_dim).permute(1, 0, 2)
        y = target[:, :, 1:].float().reshape(-1, head_count, dim).permute(1, 0, 2)
        weight, bias = solve(x, y)
        return JointVParameters(
            terms=terms,
            weights={
                term: weight[:, index * dim : (index + 1) * dim]
                for index, term in enumerate(terms)
            },
            bias=bias,
            weight_mode=weight_mode,
        )

    for layer in range(layer_count):
        x = torch.cat(
            [sources[term][layer, :, 1:].float() for term in terms], dim=-1
        ).reshape(-1, head_count, feature_dim).permute(1, 0, 2)
        y = target[layer, :, 1:].float().reshape(-1, head_count, dim).permute(1, 0, 2)
        weight, bias = solve(x, y)
        biases.append(bias)
        for index, term in enumerate(terms):
            fitted[term].append(weight[:, index * dim : (index + 1) * dim])
    return JointVParameters(
        terms=terms,
        weights={term: torch.stack(parts) for term, parts in fitted.items()},
        bias=torch.stack(biases),
        weight_mode=weight_mode,
    )


def predict_terms(
    parameters: JointVParameters,
    frame_sources: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Apply each selected term and add the joint bias exactly once."""
    first = frame_sources[parameters.terms[0]]
    if parameters.weight_mode == "layer_shared":
        prediction = parameters.bias.view(
            *([1] * (first.ndim - 2)),
            parameters.bias.shape[-2],
            parameters.bias.shape[-1],
        ).expand_as(first).clone()
    else:
        prediction = parameters.bias.view(
            parameters.bias.shape[0],
            *([1] * (first.ndim - 3)),
            parameters.bias.shape[-2],
            parameters.bias.shape[-1],
        ).expand_as(first).clone()
    for term in parameters.terms:
        if parameters.weight_mode == "layer_shared":
            prediction = prediction + torch.einsum(
                "...hi,hij->...hj",
                frame_sources[term].float(),
                parameters.weights[term],
            )
        else:
            prediction = prediction + torch.einsum(
                "l...hi,lhij->l...hj",
                frame_sources[term].float(),
                parameters.weights[term],
            )
    return prediction


@torch.no_grad()
def rollout_value(
    clean_value: torch.Tensor,
    rebuilt_key: torch.Tensor,
    innovation: torch.Tensor,
    parameters: JointVParameters,
    config: CodecConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    rebuilt = torch.empty_like(clean_value, dtype=torch.float32)
    predictions = torch.zeros_like(clean_value, dtype=torch.float32)
    anchor = anchor_roundtrip(clean_value[..., 0, :, :], config)
    rebuilt[..., 0, :, :] = anchor.to(torch.bfloat16).float()
    for frame in range(1, clean_value.shape[-3]):
        available = {
            "cross": rebuilt_key[..., frame, :, :],
            "innovation": innovation[..., frame, :, :],
            "temporal": rebuilt[..., frame - 1, :, :],
        }
        prediction = predict_terms(parameters, available)
        residual = clean_value[..., frame, :, :].float() - prediction
        residual_rebuilt = _quantization_roundtrip(
            residual,
            bits=config.v_bits,
            config=config,
            symmetric=config.residual_mode == "symmetric",
        )
        rebuilt[..., frame, :, :] = (prediction + residual_rebuilt).to(torch.bfloat16).float()
        predictions[..., frame, :, :] = prediction
    return rebuilt, predictions


def vector_metrics(
    clean: torch.Tensor,
    candidate: torch.Tensor,
    *,
    start_frame: int = 1,
) -> dict[str, Any]:
    """MSE/NMSE and mean vector cosine over non-anchor frames."""
    if clean.shape != candidate.shape or clean.ndim != 5:
        raise ValueError("metrics require matching [L,S,T,H,D] tensors")
    reference = clean[:, :, start_frame:].double()
    estimate = candidate[:, :, start_frame:].double()
    error = estimate - reference
    error2 = error.square()
    signal2 = reference.square()
    cosine = F.cosine_similarity(reference, estimate, dim=-1, eps=1e-12)
    layer_error = error2.sum(dim=(1, 2, 3, 4))
    layer_signal = signal2.sum(dim=(1, 2, 3, 4)).clamp_min(EPS)
    layer_cosine = cosine.mean(dim=(1, 2, 3))
    frame_error = error2.sum(dim=(0, 1, 3, 4))
    frame_signal = signal2.sum(dim=(0, 1, 3, 4)).clamp_min(EPS)
    frame_cosine = cosine.mean(dim=(0, 1, 3))
    return {
        "pooled_mse": float(error2.mean().item()),
        "pooled_nmse": float(error2.sum().div(signal2.sum().clamp_min(EPS)).item()),
        "pooled_cosine_similarity": float(cosine.mean().item()),
        "per_layer_nmse": (layer_error / layer_signal).cpu().tolist(),
        "per_layer_cosine_similarity": layer_cosine.cpu().tolist(),
        "per_frame": [
            {
                "frame": start_frame + index,
                "nmse": float((frame_error[index] / frame_signal[index]).item()),
                "cosine_similarity": float(frame_cosine[index].item()),
            }
            for index in range(frame_error.numel())
        ],
    }


@torch.no_grad()
def fit_temporal_key(
    clean_key: torch.Tensor,
    config: CodecConfig,
    *,
    iterations: int,
) -> tuple[TemporalParameters, torch.Tensor, torch.Tensor]:
    rebuilt = _reconstruct_direct_trace(clean_key, config)
    parameters: TemporalParameters | None = None
    for _ in range(iterations):
        parameters = fit_legacy_shared_temporal(
            rebuilt,
            clean_key,
            history_length=2,
            ridge=1.0,
            max_gain=1.0,
            stride=config.stride,
        )
        rebuilt = _reconstruct_temporal_trace(clean_key, parameters, config)
    assert parameters is not None
    return parameters, rebuilt, temporal_predictions(rebuilt, parameters)


@torch.no_grad()
def fit_v_candidate(
    terms: tuple[str, ...],
    *,
    clean_value: torch.Tensor,
    rebuilt_key: torch.Tensor,
    innovation: torch.Tensor,
    config: CodecConfig,
    iterations: int,
    ridge: float,
    weight_mode: str,
) -> JointVParameters:
    rebuilt_value = _reconstruct_direct_trace(clean_value, config, bits=config.v_bits)
    parameters: JointVParameters | None = None
    fit_iterations = iterations if "temporal" in terms else 1
    for _ in range(fit_iterations):
        sources = {
            "cross": rebuilt_key,
            "innovation": innovation,
            "temporal": temporal_source(rebuilt_value),
        }
        parameters = fit_joint_dense_predictor(
            terms,
            sources,
            clean_value,
            ridge=ridge,
            weight_mode=weight_mode,
        )
        rebuilt_value, _ = rollout_value(
            clean_value, rebuilt_key, innovation, parameters, config
        )
    assert parameters is not None
    return parameters


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    loaded = load_trace(
        args.trace,
        frames=args.frames,
        requested_layers=args.layers,
        holdout_step=args.holdout_step,
    )
    (
        train_keys,
        train_values,
        eval_keys,
        eval_values,
        layer_ids,
        train_ids,
        eval_ids,
        metadata,
    ) = loaded
    train_keys = train_keys.to(device)
    train_values = train_values.to(device)
    eval_keys = eval_keys.to(device)
    eval_values = eval_values.to(device)
    config = read_config(args.config, stride=metadata["stride"], dim=train_keys.shape[-1])

    started = time.perf_counter()
    key_parameters, rebuilt_train_key, train_key_prediction = fit_temporal_key(
        train_keys, config, iterations=args.iterations
    )
    rebuilt_eval_key = _reconstruct_temporal_trace(eval_keys, key_parameters, config)
    eval_key_prediction = temporal_predictions(rebuilt_eval_key, key_parameters)
    train_innovation = key_innovation(
        train_keys,
        rebuilt_train_key,
        train_key_prediction,
        source=args.innovation_source,
    )
    eval_innovation = key_innovation(
        eval_keys,
        rebuilt_eval_key,
        eval_key_prediction,
        source=args.innovation_source,
    )
    key_seconds = time.perf_counter() - started

    methods: dict[str, dict[str, Any]] = {}
    for name, terms in TERM_SETS.items():
        print(f"[V source ablation] fitting {name}", file=sys.stderr, flush=True)
        fit_started = time.perf_counter()
        parameters = fit_v_candidate(
            terms,
            clean_value=train_values,
            rebuilt_key=rebuilt_train_key,
            innovation=train_innovation,
            config=config,
            iterations=args.iterations,
            ridge=args.ridge,
            weight_mode=args.weight_mode,
        )
        fit_seconds = time.perf_counter() - fit_started
        eval_started = time.perf_counter()
        rebuilt_value, prediction = rollout_value(
            eval_values, rebuilt_eval_key, eval_innovation, parameters, config
        )
        methods[name] = {
            "terms": list(terms),
            "parameter_count": parameters.parameter_count,
            "expanded_parameter_count": (
                parameters.parameter_count * len(layer_ids)
                if parameters.weight_mode == "layer_shared"
                else parameters.parameter_count
            ),
            "pre_quant_value_prediction": vector_metrics(eval_values, prediction),
            "post_quant_value_reconstruction": vector_metrics(eval_values, rebuilt_value),
            "fit_seconds": fit_seconds,
            "evaluation_seconds": time.perf_counter() - eval_started,
        }
        metric = methods[name]["post_quant_value_reconstruction"]
        print(
            f"[V source ablation] {name}: NMSE={metric['pooled_nmse']:.8g}, "
            f"cos={metric['pooled_cosine_similarity']:.8g}",
            file=sys.stderr,
            flush=True,
        )

    ranking = sorted(
        (
            {
                "method": name,
                "nmse": result["post_quant_value_reconstruction"]["pooled_nmse"],
                "cosine_similarity": result["post_quant_value_reconstruction"][
                    "pooled_cosine_similarity"
                ],
            }
            for name, result in methods.items()
        ),
        key=lambda item: item["nmse"],
    )
    oracle = args.innovation_source == "clean"
    report = {
        "schema_version": 1,
        "experiment": "v_source_term_ablation",
        "status": "complete",
        "trace": {
            "path": str(args.trace.resolve()),
            "sha256": sha256(args.trace),
            **metadata,
            "layer_ids": list(layer_ids),
            "train_sample_ids": train_ids.tolist(),
            "eval_sample_ids": eval_ids.tolist(),
        },
        "protocol": {
            "k_predictor": "shared temporal history(2)",
            "k_shared_across_candidates": True,
            "innovation_definition": (
                "K_origin - K_prediction"
                if oracle
                else "K_reconstructed - K_prediction"
            ),
            "innovation_source": (
                "clean_k_minus_prediction"
                if oracle
                else "reconstructed_k_minus_prediction"
            ),
            "oracle": oracle,
            "deployable": not oracle,
            "temporal_definition": "previous reconstructed V",
            "joint_fit": True,
            "single_bias_per_candidate": True,
            "term_projection": (
                "per-head dense D-by-D, shared across layers"
                if args.weight_mode == "layer_shared"
                else "per-layer, per-head dense D-by-D"
            ),
            "weight_mode": args.weight_mode,
            "anchor_frame_excluded_from_fit_and_metrics": True,
            "frames": args.frames,
            "k_bits": config.k_bits,
            "v_bits": config.v_bits,
            "anchor_bits": config.anchor_bits,
            "group_size": config.group_size,
            "iterations": args.iterations,
            "ridge": args.ridge,
            "device": str(device),
            "threads": torch.get_num_threads(),
        },
        "fixed_key": {
            "parameter_count": int(
                key_parameters.history_weights.numel() + key_parameters.history_bias.numel()
            ),
            "fit_seconds": key_seconds,
            "post_quant_reconstruction": vector_metrics(eval_keys, rebuilt_eval_key),
        },
        "methods": methods,
        "ranking_by_reconstruction_nmse": ranking,
        "best_method": ranking[0]["method"],
        "total_seconds": time.perf_counter() - started,
    }
    numeric_values = [
        value
        for result in methods.values()
        for metric in (
            result["pre_quant_value_prediction"],
            result["post_quant_value_reconstruction"],
        )
        for value in (
            metric["pooled_mse"],
            metric["pooled_nmse"],
            metric["pooled_cosine_similarity"],
        )
    ]
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("ablation produced a non-finite metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "legacy.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--holdout-step", type=int, default=5)
    parser.add_argument("--layers", type=parse_layers)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument(
        "--weight-mode",
        choices=("layer_shared", "layer_head"),
        default="layer_shared",
    )
    parser.add_argument("--innovation-source", choices=("clean", "reconstructed"), default="clean")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=8)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
