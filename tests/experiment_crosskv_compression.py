#!/usr/bin/env python3
"""Offline CrossKV parameter-compression experiment.

The experiment never loads a video model and never generates video. It keeps
the fitted bias and V-history maps unchanged, compresses only the dense K->V
weight bank, and evaluates each approximation at three levels:

1. relative Frobenius error of the weight matrices;
2. relative error of the CrossKV term on held-out decoder-visible K;
3. closed-loop V reconstruction NMSE after the existing residual quantizer.

Parameters can be loaded from an existing fitted pair or fitted from the clean
trace with the same native KH2/VH1 protocol used by the repository's held-out
CleanTrace analysis. Factorized candidates are reconstructed to dense weights
for this experiment; their reported storage is the factor payload that a later
runtime implementation would store.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

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


DEFAULT_METHODS = (
    "fp16",
    "int8",
    "int4",
    "svd",
    "diag_svd",
    "shared_tucker",
)
DEFAULT_RANKS = (8, 16, 24, 32, 48, 64, 96)
MODEL_SHAPES = {
    "self_forcing": (30, 12, 128),
    "longcat": (48, 32, 128),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    weight: torch.Tensor
    stored_weight_bytes: int
    stored_scalars: int
    rank: int | None = None
    bits: int | None = None


def _csv_strings(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("the comma-separated list cannot be empty")
    return items


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        items = tuple(int(item) for item in _csv_strings(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if any(item < 0 for item in items) or len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("integers must be unique and non-negative")
    return items


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_trace_ids(sample_count: int, holdout_step: int) -> tuple[np.ndarray, np.ndarray]:
    """Match the existing CleanTrace split: hold out step-1, 2*step-1, ..."""
    if holdout_step < 3:
        raise ValueError("holdout_step must be at least 3")
    eval_ids = np.arange(holdout_step - 1, sample_count, holdout_step, dtype=int)
    train_mask = np.ones(sample_count, dtype=bool)
    train_mask[eval_ids] = False
    train_ids = np.flatnonzero(train_mask)
    if eval_ids.size < 1 or train_ids.size < 4:
        raise ValueError("trace split is too small")
    return train_ids, eval_ids


def _load_trace(
    path: Path,
    requested_layer_ids: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in ("keys", "values") if name not in archive]
        if missing:
            raise ValueError(f"trace is missing arrays: {', '.join(missing)}")
        shape = tuple(int(size) for size in archive["keys"].shape)
        if shape != tuple(int(size) for size in archive["values"].shape):
            raise ValueError("trace keys and values must have identical shapes")
        if len(shape) != 5:
            raise ValueError(f"expected trace [L,S,T,H,D], got {shape}")
        trace_layer_ids = (
            tuple(int(item) for item in np.asarray(archive["layer_ids"]))
            if "layer_ids" in archive
            else tuple(range(shape[0]))
        )
        if len(trace_layer_ids) != shape[0] or len(set(trace_layer_ids)) != shape[0]:
            raise ValueError("trace layer_ids must uniquely identify every layer")
        selected_ids = (
            trace_layer_ids
            if requested_layer_ids is None
            else tuple(int(item) for item in requested_layer_ids)
        )
        missing_layers = sorted(set(selected_ids) - set(trace_layer_ids))
        if missing_layers:
            raise ValueError(f"trace does not contain layers {missing_layers}")
        positions = [trace_layer_ids.index(layer) for layer in selected_ids]
        keys = np.array(archive["keys"][positions], copy=True)
        values = np.array(archive["values"][positions], copy=True)
        stride_name = "trace_stride" if "trace_stride" in archive else "stride"
        metadata = {
            "source_shape": list(shape),
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
    return keys, values, selected_ids, metadata


def _read_config(path: Path) -> CodecConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "codec" in payload:
        if len(payload) != 1:
            raise ValueError("'codec' must be the only top-level config object")
        payload = payload["codec"]
    payload = dict(payload)
    payload["use_triton"] = False
    payload["strict_triton"] = False
    return config_from_mapping(payload)


def _config_from_signature(signature: str) -> CodecConfig:
    try:
        _, encoded = signature.split(":", 1)
        payload = json.loads(encoded)
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError("parameter codec signature is not valid JSON") from error
    payload.pop("quantizer_version", None)
    values = asdict(CodecConfig())
    values.update(payload)
    values["use_triton"] = False
    values["strict_triton"] = False
    return config_from_mapping(values)


def _fit_config(args: argparse.Namespace, *, stride: int, dim: int) -> CodecConfig:
    group_size = args.group_size
    if group_size is None:
        group_size = 64 if dim % 64 == 0 else dim
    return CodecConfig(
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        anchor_bits=args.anchor_bits,
        group_size=group_size,
        stride=stride,
        residual_mode=args.residual_mode,
        scale_dtype=args.scale_dtype,
        reconstruction_dtype=args.reconstruction_dtype,
        use_triton=False,
        strict_triton=False,
        k_history_length=args.k_history,
        v_history_length=args.v_history,
        cross_matmul_precision="tf32",
        k_predictor_mode="layer_head",
        v_predictor_mode="cross_kv",
        fit_mode="native",
    )


def _subset_temporal(
    parameters: TemporalParameters, layer_ids: Sequence[int]
) -> TemporalParameters:
    if not parameters.layered:
        if tuple(layer_ids) not in ((0,), (-1,)):
            raise ValueError("non-layered K parameters cannot select requested layers")
        return parameters
    assert parameters.layer_ids is not None
    missing = sorted(set(layer_ids) - set(parameters.layer_ids))
    if missing:
        raise ValueError(f"K parameters do not contain layers {missing}")
    indices = [parameters.layer_ids.index(int(layer)) for layer in layer_ids]
    return TemporalParameters(
        history_weights=parameters.history_weights[indices],
        history_bias=parameters.history_bias[indices],
        stride=parameters.stride,
        layer_ids=tuple(layer_ids),
        codec_signature=parameters.codec_signature,
        parameter_set_id=parameters.parameter_set_id,
        coordinate_system=parameters.coordinate_system,
    )


def _subset_cross(
    parameters: CrossKVParameters, layer_ids: Sequence[int]
) -> CrossKVParameters:
    if not parameters.layered:
        if tuple(layer_ids) not in ((0,), (-1,)):
            raise ValueError("non-layered V parameters cannot select requested layers")
        return parameters
    assert parameters.layer_ids is not None
    missing = sorted(set(layer_ids) - set(parameters.layer_ids))
    if missing:
        raise ValueError(f"V parameters do not contain layers {missing}")
    indices = [parameters.layer_ids.index(int(layer)) for layer in layer_ids]
    return CrossKVParameters(
        weight=parameters.weight[indices],
        bias=parameters.bias[indices],
        stride=parameters.stride,
        layer_ids=tuple(layer_ids),
        codec_signature=parameters.codec_signature,
        parameter_set_id=parameters.parameter_set_id,
        coordinate_system=parameters.coordinate_system,
        history_weights=parameters.history_weights[indices],
    )


def _replace_weight(
    parameters: CrossKVParameters, weight: torch.Tensor
) -> CrossKVParameters:
    return CrossKVParameters(
        weight=weight,
        bias=parameters.bias,
        stride=parameters.stride,
        layer_ids=parameters.layer_ids,
        codec_signature=parameters.codec_signature,
        parameter_set_id=parameters.parameter_set_id,
        coordinate_system=parameters.coordinate_system,
        history_weights=parameters.history_weights,
    )


def _symmetric_quantized_weight(
    weight: torch.Tensor, bits: int
) -> tuple[torch.Tensor, int, int]:
    """Quantize each output channel across its input dimension."""
    qmax = (1 << (bits - 1)) - 1
    scale = weight.abs().amax(dim=-2, keepdim=True) / qmax
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    quant = torch.round(weight / scale).clamp(-qmax, qmax)
    rebuilt = quant * scale
    packed_bytes = math.ceil(weight.numel() * bits / 8)
    scale_count = scale.numel()
    stored_bytes = packed_bytes + scale_count * 2  # FP16 scales
    return rebuilt, stored_bytes, weight.numel() + scale_count


def _matrix_count_and_dim(weight: torch.Tensor) -> tuple[int, int]:
    if weight.ndim < 3 or weight.shape[-1] != weight.shape[-2]:
        raise ValueError("weight bank must end in square matrices")
    return weight.numel() // (weight.shape[-1] ** 2), int(weight.shape[-1])


def candidate_stream(
    weight: torch.Tensor,
    *,
    methods: Sequence[str],
    ranks: Sequence[int],
) -> Iterable[Candidate]:
    """Yield one reconstructed candidate at a time to bound memory use."""
    matrix_count, dim = _matrix_count_and_dim(weight)
    dense_bytes = weight.numel() * 4
    yield Candidate("dense_fp32", "dense", weight, dense_bytes, weight.numel())

    selected = set(methods)
    unknown = sorted(selected - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")

    if "fp16" in selected:
        yield Candidate(
            "fp16",
            "fp16",
            weight.to(torch.float16).float(),
            weight.numel() * 2,
            weight.numel(),
            bits=16,
        )
    for bits in (8, 4):
        family = f"int{bits}"
        if family in selected:
            rebuilt, stored_bytes, stored_scalars = _symmetric_quantized_weight(
                weight, bits
            )
            yield Candidate(
                family,
                family,
                rebuilt,
                stored_bytes,
                stored_scalars,
                bits=bits,
            )

    valid_ranks = tuple(rank for rank in ranks if 0 < rank <= dim)
    if not valid_ranks:
        return
    flattened = weight.reshape(matrix_count, dim, dim)

    if "svd" in selected:
        u, s, vh = torch.linalg.svd(flattened, full_matrices=False)
        for rank in valid_ranks:
            stored_scalars = matrix_count * 2 * dim * rank
            if stored_scalars >= weight.numel():
                continue
            rebuilt = torch.matmul(
                u[:, :, :rank] * s[:, None, :rank], vh[:, :rank, :]
            ).reshape_as(weight)
            yield Candidate(
                f"svd_r{rank}",
                "svd",
                rebuilt,
                stored_scalars * 4,
                stored_scalars,
                rank=rank,
            )
        del u, s, vh

    if "diag_svd" in selected:
        diagonal = torch.diagonal(flattened, dim1=-2, dim2=-1).clone()
        residual = flattened - torch.diag_embed(diagonal)
        u, s, vh = torch.linalg.svd(residual, full_matrices=False)
        for rank in valid_ranks:
            stored_scalars = matrix_count * (dim + 2 * dim * rank)
            if stored_scalars >= weight.numel():
                continue
            rebuilt = torch.diag_embed(diagonal) + torch.matmul(
                u[:, :, :rank] * s[:, None, :rank], vh[:, :rank, :]
            )
            yield Candidate(
                f"diag_svd_r{rank}",
                "diag_svd",
                rebuilt.reshape_as(weight),
                stored_scalars * 4,
                stored_scalars,
                rank=rank,
            )
        del residual, u, s, vh

    if "shared_tucker" in selected:
        left_covariance = torch.einsum("nij,nkj->ik", flattened, flattened)
        right_covariance = torch.einsum("nij,nil->jl", flattened, flattened)
        _, left_vectors = torch.linalg.eigh(left_covariance)
        _, right_vectors = torch.linalg.eigh(right_covariance)
        for rank in valid_ranks:
            stored_scalars = 2 * dim * rank + matrix_count * rank * rank
            if stored_scalars >= weight.numel():
                continue
            left = left_vectors[:, -rank:]
            right = right_vectors[:, -rank:]
            core = torch.einsum("ia,nij,jb->nab", left, flattened, right)
            rebuilt = torch.einsum("ia,nab,jb->nij", left, core, right)
            yield Candidate(
                f"shared_tucker_r{rank}",
                "shared_tucker",
                rebuilt.reshape_as(weight),
                stored_scalars * 4,
                stored_scalars,
                rank=rank,
            )


def _ratio_summary(
    difference: torch.Tensor,
    reference: torch.Tensor,
    *,
    reduce_dims: Sequence[int],
    root: bool = False,
) -> dict[str, float]:
    numerator = difference.square().sum(dim=tuple(reduce_dims), dtype=torch.float64)
    denominator = reference.square().sum(dim=tuple(reduce_dims), dtype=torch.float64)
    tiny = torch.finfo(torch.float64).tiny
    ratios = numerator / denominator.clamp_min(tiny)
    pooled = numerator.sum() / denominator.sum().clamp_min(tiny)
    if root:
        ratios = ratios.sqrt()
        pooled = pooled.sqrt()
    flat = ratios.flatten().cpu()
    return {
        "pooled": float(pooled.item()),
        "median": float(torch.quantile(flat, 0.5).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
    }


def _relative_error_increase_summary(
    candidate_error: torch.Tensor,
    dense_error: torch.Tensor,
    *,
    reduce_dims: Sequence[int],
) -> dict[str, float]:
    candidate_sse = candidate_error.square().sum(
        dim=tuple(reduce_dims), dtype=torch.float64
    )
    dense_sse = dense_error.square().sum(
        dim=tuple(reduce_dims), dtype=torch.float64
    )
    tiny = torch.finfo(torch.float64).tiny
    increases = candidate_sse / dense_sse.clamp_min(tiny) - 1.0
    pooled = candidate_sse.sum() / dense_sse.sum().clamp_min(tiny) - 1.0
    flat = increases.flatten().cpu()
    return {
        "pooled": float(pooled.item()),
        "median": float(torch.quantile(flat, 0.5).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
        "min": float(flat.min().item()),
    }


def _weight_storage_bytes(
    family: str,
    *,
    matrix_count: int,
    dim: int,
    rank: int | None,
    bits: int | None,
) -> int:
    if family == "dense":
        return matrix_count * dim * dim * 4
    if family == "fp16":
        return matrix_count * dim * dim * 2
    if family in {"int8", "int4"}:
        assert bits is not None
        return math.ceil(matrix_count * dim * dim * bits / 8) + matrix_count * dim * 2
    if family == "svd":
        assert rank is not None
        return matrix_count * 2 * dim * rank * 4
    if family == "diag_svd":
        assert rank is not None
        return matrix_count * (dim + 2 * dim * rank) * 4
    if family == "shared_tucker":
        assert rank is not None
        return (2 * dim * rank + matrix_count * rank * rank) * 4
    raise ValueError(f"unsupported storage family {family!r}")


def _projected_storage(candidate: Candidate) -> dict[str, dict[str, float | int]]:
    projected: dict[str, dict[str, float | int]] = {}
    for name, (layers, heads, dim) in MODEL_SHAPES.items():
        matrix_count = layers * heads
        dense_bytes = matrix_count * dim * dim * 4
        stored_bytes = _weight_storage_bytes(
            candidate.family,
            matrix_count=matrix_count,
            dim=dim,
            rank=candidate.rank,
            bits=candidate.bits,
        )
        projected[name] = {
            "layers": layers,
            "heads": heads,
            "dim": dim,
            "dense_weight_bytes": dense_bytes,
            "dense_weight_mib": dense_bytes / (1024**2),
            "stored_weight_bytes": stored_bytes,
            "stored_weight_mib": stored_bytes / (1024**2),
            "weight_compression_ratio": dense_bytes / stored_bytes,
        }
    return projected


def evaluate_candidates(
    *,
    eval_keys: torch.Tensor,
    eval_values: torch.Tensor,
    temporal: TemporalParameters | None,
    cross: CrossKVParameters,
    config: CodecConfig,
    methods: Sequence[str],
    ranks: Sequence[int],
) -> list[dict[str, object]]:
    if tuple(eval_keys.shape) != tuple(eval_values.shape) or eval_keys.ndim != 5:
        raise ValueError("evaluation trace must have matching [L,S,T,H,D] K/V")
    if cross.weight.shape[:2] != (eval_keys.shape[0], eval_keys.shape[-2]):
        raise ValueError("CrossKV parameters do not match evaluation layers/heads")
    if cross.weight.shape[-1] != eval_keys.shape[-1]:
        raise ValueError("CrossKV parameters do not match evaluation head dimension")

    if temporal is None:
        rebuilt_keys = eval_keys
        key_source = "clean"
    else:
        rebuilt_keys = _reconstruct_temporal_trace(eval_keys, temporal, config)
        key_source = "closed_loop_reconstructed"

    dense_cross_term = torch.einsum(
        "lsthi,lhij->lsthj", rebuilt_keys[:, :, 1:].float(), cross.weight
    )
    dense_rebuilt_v = _reconstruct_cross_kv_trace(
        rebuilt_keys, eval_values, cross, config
    )
    auxiliary_bytes = (cross.bias.numel() + cross.history_weights.numel()) * 4
    dense_weight_bytes = cross.weight.numel() * 4
    results: list[dict[str, object]] = []

    for candidate in candidate_stream(cross.weight, methods=methods, ranks=ranks):
        started = time.perf_counter()
        weight_difference = candidate.weight - cross.weight
        weight_error = _ratio_summary(
            weight_difference,
            cross.weight,
            reduce_dims=(-2, -1),
            root=True,
        )
        if candidate.family == "dense":
            candidate_cross_term = dense_cross_term
            candidate_rebuilt_v = dense_rebuilt_v
        else:
            candidate_cross_term = torch.einsum(
                "lsthi,lhij->lsthj",
                rebuilt_keys[:, :, 1:].float(),
                candidate.weight,
            )
            candidate_rebuilt_v = _reconstruct_cross_kv_trace(
                rebuilt_keys,
                eval_values,
                _replace_weight(cross, candidate.weight),
                config,
            )
        cross_error = _ratio_summary(
            candidate_cross_term - dense_cross_term,
            dense_cross_term,
            reduce_dims=(1, 2, 4),
        )
        reconstruction_error = _ratio_summary(
            candidate_rebuilt_v[:, :, 1:] - eval_values[:, :, 1:],
            eval_values[:, :, 1:],
            reduce_dims=(1, 2, 3, 4),
        )
        relative_reconstruction_increase_by_layer = _relative_error_increase_summary(
            candidate_rebuilt_v[:, :, 1:] - eval_values[:, :, 1:],
            dense_rebuilt_v[:, :, 1:] - eval_values[:, :, 1:],
            reduce_dims=(1, 2, 3, 4),
        )
        induced_reconstruction_error = _ratio_summary(
            candidate_rebuilt_v[:, :, 1:] - dense_rebuilt_v[:, :, 1:],
            dense_rebuilt_v[:, :, 1:],
            reduce_dims=(1, 2, 3, 4),
        )
        total_dense_bytes = dense_weight_bytes + auxiliary_bytes
        total_stored_bytes = candidate.stored_weight_bytes + auxiliary_bytes
        results.append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "rank": candidate.rank,
                "bits": candidate.bits,
                "stored_scalars": candidate.stored_scalars,
                "stored_weight_bytes": candidate.stored_weight_bytes,
                "weight_compression_ratio": dense_weight_bytes
                / candidate.stored_weight_bytes,
                "total_predictor_compression_ratio": total_dense_bytes
                / total_stored_bytes,
                "weight_relative_frobenius_error": weight_error,
                "cross_output_nmse_vs_dense": cross_error,
                "v_reconstruction_nmse": reconstruction_error,
                "v_reconstruction_relative_increase_by_layer": (
                    relative_reconstruction_increase_by_layer
                ),
                "v_reconstruction_nmse_vs_dense": induced_reconstruction_error,
                "projected_weight_storage": _projected_storage(candidate),
                "evaluation_seconds": time.perf_counter() - started,
                "key_source": key_source,
            }
        )
        if candidate.family != "dense":
            del candidate_cross_term, candidate_rebuilt_v
    return results


def _recommend(
    results: Sequence[dict[str, object]],
    *,
    max_relative_reconstruction_increase: float,
    max_cross_output_nmse: float,
) -> dict[str, object]:
    next(item for item in results if item["family"] == "dense")
    evaluated: list[dict[str, object]] = []
    for item in results:
        if item["family"] == "dense":
            item["relative_reconstruction_increase"] = 0.0
            item["meets_thresholds"] = True
            continue
        relative_increase = float(
            item["v_reconstruction_relative_increase_by_layer"]["pooled"]  # type: ignore[index]
        )
        cross_nmse = float(item["cross_output_nmse_vs_dense"]["pooled"])  # type: ignore[index]
        meets = (
            float(item["weight_compression_ratio"]) > 1.0
            and relative_increase <= max_relative_reconstruction_increase
            and cross_nmse <= max_cross_output_nmse
        )
        item["relative_reconstruction_increase"] = relative_increase
        item["meets_thresholds"] = meets
        if meets:
            evaluated.append(item)
    if not evaluated:
        return {
            "selected": None,
            "reason": "no compressed candidate met both explicit thresholds",
            "max_relative_reconstruction_increase": max_relative_reconstruction_increase,
            "max_cross_output_nmse": max_cross_output_nmse,
        }
    selected = max(
        evaluated,
        key=lambda item: (
            float(item["weight_compression_ratio"]),
            -float(item["v_reconstruction_nmse"]["pooled"]),  # type: ignore[index]
        ),
    )
    return {
        "selected": selected["name"],
        "reason": "highest weight compression ratio meeting both explicit thresholds",
        "max_relative_reconstruction_increase": max_relative_reconstruction_increase,
        "max_cross_output_nmse": max_cross_output_nmse,
    }


def _print_table(results: Sequence[dict[str, object]]) -> None:
    print(
        f"{'candidate':<22} {'weight x':>9} {'W relF':>11} "
        f"{'KW NMSE':>11} {'V NMSE':>11} {'V delta':>10}"
    )
    for item in results:
        v_delta = float(item.get("relative_reconstruction_increase", 0.0))
        print(
            f"{str(item['name']):<22} "
            f"{float(item['weight_compression_ratio']):9.3f} "
            f"{float(item['weight_relative_frobenius_error']['pooled']):11.4g} "  # type: ignore[index]
            f"{float(item['cross_output_nmse_vs_dense']['pooled']):11.4g} "  # type: ignore[index]
            f"{float(item['v_reconstruction_nmse']['pooled']):11.4g} "  # type: ignore[index]
            f"{v_delta:10.3%}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--k-params", type=Path)
    parser.add_argument("--v-params", type=Path)
    parser.add_argument(
        "--fit",
        action="store_true",
        help="fit K and CrossKV parameters from the training split before evaluation",
    )
    parser.add_argument("--save-params-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=_csv_ints)
    parser.add_argument("--holdout-step", type=int, default=5)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--methods", type=_csv_strings, default=DEFAULT_METHODS)
    parser.add_argument("--ranks", type=_csv_ints, default=DEFAULT_RANKS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--k-history", type=int, default=2)
    parser.add_argument("--v-history", type=int, default=1)
    parser.add_argument("--v-history-mode", choices=("dense", "diagonal"), default="diagonal")
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument("--anchor-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--residual-mode", choices=("symmetric", "asymmetric"), default="asymmetric")
    parser.add_argument("--scale-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--reconstruction-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-relative-reconstruction-increase", type=float, default=0.01)
    parser.add_argument("--max-cross-output-nmse", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.fit and (args.k_params or args.v_params):
        parser.error("--fit cannot be combined with --k-params/--v-params")
    if not args.fit and args.v_params is None:
        parser.error("provide --fit or --v-params")
    if args.k_params is not None and args.v_params is None:
        parser.error("--k-params requires --v-params")
    if args.max_eval_samples is not None and args.max_eval_samples <= 0:
        parser.error("--max-eval-samples must be positive")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.ridge < 0.0:
        parser.error("--ridge must be non-negative")
    if args.max_relative_reconstruction_increase < 0.0:
        parser.error("--max-relative-reconstruction-increase must be non-negative")
    if args.max_cross_output_nmse < 0.0:
        parser.error("--max-cross-output-nmse must be non-negative")
    return args


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    loaded_temporal: TemporalParameters | None = None
    loaded_cross: CrossKVParameters | None = None
    requested_layers = args.layers
    if not args.fit:
        loaded_cross = CrossKVParameters.load(args.v_params)
        if args.k_params is not None:
            loaded_temporal = TemporalParameters.load(args.k_params)
        if requested_layers is None:
            requested_layers = loaded_cross.layer_ids
            if requested_layers is None:
                requested_layers = (0,)

    print("[crosskv] loading clean trace", file=sys.stderr, flush=True)
    keys_np, values_np, layer_ids, metadata = _load_trace(args.trace, requested_layers)
    train_ids, eval_ids = split_trace_ids(keys_np.shape[1], args.holdout_step)
    if args.max_eval_samples is not None:
        eval_ids = eval_ids[: args.max_eval_samples]
    if not len(eval_ids):
        raise ValueError("evaluation split is empty")

    fit_seconds: float | None = None
    if args.fit:
        config = _read_config(args.config) if args.config else _fit_config(
            args, stride=int(metadata["stride"]), dim=int(keys_np.shape[-1])
        )
        train_keys = torch.from_numpy(
            np.ascontiguousarray(keys_np[:, train_ids])
        ).to(device)
        train_values = torch.from_numpy(
            np.ascontiguousarray(values_np[:, train_ids])
        ).to(device)
        print(
            f"[crosskv] fitting {len(layer_ids)} layers on {len(train_ids)} traces",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        temporal, cross = fit_predictors(
            train_keys,
            train_values,
            config=config,
            history_length=config.k_history_length,
            iterations=args.iterations,
            temporal_ridge=args.ridge,
            cross_ridge=args.ridge,
            v_history_ridge=0.0,
            max_gain=1.0,
            v_history_mode=args.v_history_mode,
            layer_ids=layer_ids,
            coordinate_system=metadata["coordinate_system"],
            fit_history_source="reconstructed",
        )
        fit_seconds = time.perf_counter() - started
        del train_keys, train_values
        if args.save_params_dir is not None:
            args.save_params_dir.mkdir(parents=True, exist_ok=True)
            temporal.save(args.save_params_dir / "k_temporal.npz")
            cross.save(args.save_params_dir / "v_cross_kv.npz")
    else:
        assert loaded_cross is not None
        cross = _subset_cross(loaded_cross, layer_ids).to(device)
        temporal = (
            _subset_temporal(loaded_temporal, layer_ids).to(device)
            if loaded_temporal is not None
            else None
        )
        if args.config:
            config = _read_config(args.config)
        elif cross.codec_signature:
            config = _config_from_signature(cross.codec_signature)
        else:
            config = _fit_config(
                args, stride=int(metadata["stride"]), dim=int(keys_np.shape[-1])
            )

    if cross.stride is not None and cross.stride != config.stride:
        raise ValueError("CrossKV stride does not match evaluation config")
    if temporal is not None:
        if temporal.stride is not None and temporal.stride != config.stride:
            raise ValueError("K-parameter stride does not match evaluation config")
        if (
            temporal.parameter_set_id
            and cross.parameter_set_id
            and temporal.parameter_set_id != cross.parameter_set_id
        ):
            raise ValueError("K and V parameters are not a fitted pair")
    if cross.coordinate_system and metadata["coordinate_system"]:
        if cross.coordinate_system != metadata["coordinate_system"]:
            raise ValueError("trace and CrossKV parameters use different coordinates")

    eval_keys = torch.from_numpy(np.ascontiguousarray(keys_np[:, eval_ids])).to(device)
    eval_values = torch.from_numpy(np.ascontiguousarray(values_np[:, eval_ids])).to(device)
    del keys_np, values_np
    print(
        f"[crosskv] evaluating {len(args.methods)} method families on "
        f"{len(layer_ids)} layers x {len(eval_ids)} held-out traces",
        file=sys.stderr,
        flush=True,
    )
    experiment_started = time.perf_counter()
    results = evaluate_candidates(
        eval_keys=eval_keys,
        eval_values=eval_values,
        temporal=temporal,
        cross=cross,
        config=config,
        methods=args.methods,
        ranks=args.ranks,
    )
    recommendation = _recommend(
        results,
        max_relative_reconstruction_increase=args.max_relative_reconstruction_increase,
        max_cross_output_nmse=args.max_cross_output_nmse,
    )
    report: dict[str, object] = {
        "experiment": "crosskv_weight_compression_reconstruction_only",
        "scope": {
            "compressed": "CrossKV dense K_to_V weight only",
            "unchanged": ["bias", "V history weights"],
            "video_generation": False,
            "factorized_runtime_implemented": False,
        },
        "trace": {
            "path": str(args.trace.resolve()),
            "sha256": _sha256(args.trace),
            **metadata,
            "selected_layer_ids": list(layer_ids),
            "train_trace_ids": train_ids.tolist() if args.fit else None,
            "eval_trace_ids": eval_ids.tolist(),
            "evaluation_shape": list(eval_keys.shape),
        },
        "parameters": {
            "source": "fitted_from_trace" if args.fit else str(args.v_params.resolve()),
            "weight_shape": list(cross.weight.shape),
            "weight_parameters": cross.weight.numel(),
            "weight_fp32_mib": cross.weight.numel() * 4 / (1024**2),
            "bias_parameters": cross.bias.numel(),
            "history_parameters": cross.history_weights.numel(),
            "history_mode": cross.history_mode,
            "fit_seconds": fit_seconds,
        },
        "config": asdict(config),
        "protocol": {
            "holdout_step": args.holdout_step,
            "methods": list(args.methods),
            "ranks": list(args.ranks),
            "primary_metric": "held-out frame>=1 closed-loop V relative SSE",
            "matrix_metric": "per-matrix relative Frobenius norm",
            "cross_metric": "held-out K@W_hat relative SSE versus K@W_dense",
            "storage_scope": "weight payload only; factors use FP32 and quantizer scales use FP16",
        },
        "results": results,
        "recommendation": recommendation,
        "elapsed_seconds": time.perf_counter() - experiment_started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_table(results)
    print(f"recommendation: {recommendation['selected']}")
    print(f"report: {args.output.resolve()}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
