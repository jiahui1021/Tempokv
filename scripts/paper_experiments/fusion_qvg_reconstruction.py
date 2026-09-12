#!/usr/bin/env python3
"""Compare TempoKV reconstruction with a QVG-style residual-VQ fusion."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.config import config_from_mapping  # noqa: E402
from tempokv.fitting import (  # noqa: E402
    _reconstruct_cross_kv_trace,
    _reconstruct_direct_trace,
    _reconstruct_temporal_trace,
    fit_predictors,
)
from tempokv.parameters import CrossKVParameters  # noqa: E402
from tempokv.quantization import roundtrip  # noqa: E402
from tempokv.residual_vq import (  # noqa: E402
    ResidualVQCodebook,
    fit_residual_vq_codebook,
    residual_vq_roundtrip,
)


SCALE_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class ResidualCollector:
    def __init__(self, reconstruct: Callable[[torch.Tensor], torch.Tensor]):
        self.reconstruct = reconstruct
        self.frames: list[torch.Tensor] = []

    def __call__(self, residual: torch.Tensor) -> torch.Tensor:
        self.frames.append(residual.detach().cpu())
        return self.reconstruct(residual)

    def stacked(self) -> torch.Tensor:
        if not self.frames:
            raise RuntimeError("no residuals were collected")
        return torch.stack(self.frames, dim=2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--train-contexts", type=int, default=48)
    parser.add_argument("--eval-contexts", type=int, default=16)
    parser.add_argument("--bits", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--anchor-bits", type=int, default=4, choices=range(1, 17))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--k-history", type=int, default=2)
    parser.add_argument("--v-history", type=int, default=1)
    parser.add_argument("--fit-iterations", type=int, default=2)
    parser.add_argument("--codebook-size", type=int, default=32)
    parser.add_argument("--codebook-iterations", type=int, default=4)
    parser.add_argument("--codebook-max-points", type=int, default=2048)
    parser.add_argument("--codebook-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", default="cpu")
    return parser


def _load_trace(
    path: Path,
    *,
    max_layers: int,
    total_contexts: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        for name in ("keys", "values"):
            if name not in archive:
                raise ValueError(f"trace is missing {name}")
        keys_source = archive["keys"]
        if keys_source.ndim != 5:
            raise ValueError(
                f"screening experiment expects [L,N,T,H,D], got {keys_source.shape}"
            )
        layers = min(max_layers, int(keys_source.shape[0]))
        contexts = min(total_contexts, int(keys_source.shape[1]))
        keys = torch.from_numpy(np.array(keys_source[:layers, :contexts], copy=True))
        del keys_source
        values_source = archive["values"]
        values = torch.from_numpy(np.array(values_source[:layers, :contexts], copy=True))
        layer_ids = (
            tuple(
                int(item)
                for item in np.asarray(archive["layer_ids"]).reshape(-1)[:layers]
            )
            if "layer_ids" in archive
            else tuple(range(layers))
        )
        metadata: dict[str, object] = {}
        for name in ("stride", "coordinate_system", "cross_source_kind"):
            if name in archive:
                metadata[name] = np.asarray(archive[name]).item()
    if keys.shape != values.shape:
        raise ValueError("trace K/V shapes do not match")
    return keys, values, layer_ids, metadata


def _scalar_roundtrip(config, bits: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def apply(residual: torch.Tensor) -> torch.Tensor:
        return roundtrip(
            residual,
            bits=bits,
            group_size=config.group_size,
            scale_dtype=SCALE_DTYPES[config.scale_dtype],
            symmetric=config.residual_mode == "symmetric",
        )

    return apply


def _vq_roundtrip(
    config,
    bits: int,
    codebook: ResidualVQCodebook,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def apply(residual: torch.Tensor) -> torch.Tensor:
        return residual_vq_roundtrip(
            residual,
            codebook.to(residual.device),
            bits=bits,
            group_size=config.group_size,
            scale_dtype=SCALE_DTYPES[config.scale_dtype],
            symmetric=config.residual_mode == "symmetric",
        )

    return apply


def _fit_codebook(samples: torch.Tensor, args, *, seed_offset: int) -> ResidualVQCodebook:
    return fit_residual_vq_codebook(
        samples,
        num_centroids=args.codebook_size,
        iterations=args.codebook_iterations,
        max_points=args.codebook_max_points,
        seed=args.seed + seed_offset,
        storage_dtype=torch.bfloat16,
    )


def _fit_fusion_codebooks(
    keys: torch.Tensor,
    values: torch.Tensor,
    temporal,
    cross: CrossKVParameters,
    config,
    args,
) -> tuple[ResidualVQCodebook, ResidualVQCodebook]:
    scalar_k = _scalar_roundtrip(config, config.k_bits)
    scalar_v = _scalar_roundtrip(config, config.v_bits)
    collect_k = ResidualCollector(scalar_k)
    rebuilt_k = _reconstruct_temporal_trace(
        keys, temporal, config, residual_roundtrip=collect_k
    )
    collect_v = ResidualCollector(scalar_v)
    _reconstruct_cross_kv_trace(
        rebuilt_k, values, cross, config, residual_roundtrip=collect_v
    )
    k_codebook = _fit_codebook(collect_k.stacked(), args, seed_offset=0)
    v_codebook = _fit_codebook(collect_v.stacked(), args, seed_offset=1_000_000)

    for round_index in range(1, args.codebook_rounds):
        collect_k = ResidualCollector(_vq_roundtrip(config, config.k_bits, k_codebook))
        rebuilt_k = _reconstruct_temporal_trace(
            keys, temporal, config, residual_roundtrip=collect_k
        )
        collect_v = ResidualCollector(_vq_roundtrip(config, config.v_bits, v_codebook))
        _reconstruct_cross_kv_trace(
            rebuilt_k, values, cross, config, residual_roundtrip=collect_v
        )
        k_codebook = _fit_codebook(
            collect_k.stacked(), args, seed_offset=10_000 * round_index
        )
        v_codebook = _fit_codebook(
            collect_v.stacked(), args, seed_offset=1_000_000 + 10_000 * round_index
        )
    return k_codebook, v_codebook


def _energy(clean: torch.Tensor, rebuilt: torch.Tensor) -> dict[str, float | int]:
    reference = clean.double()
    error = rebuilt.double() - reference
    error_energy = float(error.square().sum().item())
    reference_energy = float(reference.square().sum().item())
    return {
        "mse": float(error.square().mean().item()),
        "nmse": error_energy / reference_energy,
        "relative_l2": (error_energy / reference_energy) ** 0.5,
        "max_abs": float(error.abs().max().item()),
        "values": int(reference.numel()),
    }


def _method_metrics(
    keys: torch.Tensor,
    values: torch.Tensor,
    rebuilt_k: torch.Tensor,
    rebuilt_v: torch.Tensor,
    layer_ids: tuple[int, ...],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for role, clean, rebuilt in (
        ("k", keys, rebuilt_k),
        ("v", values, rebuilt_v),
    ):
        result[role] = {
            "pooled": _energy(clean, rebuilt),
            "per_layer": [
                {"layer_id": layer_id, **_energy(clean[index], rebuilt[index])}
                for index, layer_id in enumerate(layer_ids)
            ],
        }
    return result


def _physical_bits(
    *,
    frames: int,
    dim: int,
    residual_bits: int,
    anchor_bits: int,
    group_size: int,
    with_vq_index: bool,
) -> float:
    anchor = anchor_bits + 16.0 / group_size
    residual = residual_bits + 16.0 / group_size + 8.0 / group_size
    if with_vq_index:
        residual += 8.0 / dim
    return (anchor + (frames - 1) * residual) / frames


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.train_contexts < 1 or args.eval_contexts < 1:
        raise ValueError("train/eval contexts must be positive")
    if args.codebook_rounds < 1:
        raise ValueError("codebook_rounds must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    started = time.perf_counter()
    keys, values, layer_ids, trace_metadata = _load_trace(
        args.trace,
        max_layers=args.max_layers,
        total_contexts=args.train_contexts + args.eval_contexts,
    )
    if int(keys.shape[1]) < args.train_contexts + args.eval_contexts:
        raise ValueError("trace does not contain the requested train/eval contexts")
    keys = keys.to(device)
    values = values.to(device)
    keys_train = keys[:, : args.train_contexts]
    values_train = values[:, : args.train_contexts]
    keys_eval = keys[:, args.train_contexts : args.train_contexts + args.eval_contexts]
    values_eval = values[:, args.train_contexts : args.train_contexts + args.eval_contexts]
    stride = int(trace_metadata.get("stride", 1560))
    config = config_from_mapping(
        {
            "k_bits": args.bits,
            "v_bits": args.bits,
            "anchor_bits": args.anchor_bits,
            "group_size": args.group_size,
            "stride": stride,
            "k_history_length": args.k_history,
            "v_history_length": args.v_history,
            "k_predictor_mode": "shared",
            "v_predictor_mode": "cross_kv",
            "cross_kv_weight_mode": "layer_shared",
            "cross_matmul_precision": "ieee",
            "fit_mode": "legacy",
            "use_triton": False,
            "strict_triton": False,
        }
    )

    fit_started = time.perf_counter()
    temporal, value_parameters = fit_predictors(
        keys_train,
        values_train,
        config=config,
        history_length=args.k_history,
        iterations=args.fit_iterations,
        temporal_ridge=1.0,
        cross_ridge=0.1,
        v_history_ridge=0.0,
        max_gain=1.0,
        v_history_mode="diagonal",
        layer_ids=layer_ids,
        coordinate_system=str(trace_metadata.get("coordinate_system", "unspecified")),
        fit_history_source="reconstructed",
    )
    if not isinstance(value_parameters, CrossKVParameters):
        raise TypeError("fusion experiment requires CrossKV V parameters")
    fit_seconds = time.perf_counter() - fit_started

    near_rate_config = replace(
        config,
        anchor_bits=min(16, args.anchor_bits + 1),
    )
    near_fit_started = time.perf_counter()
    near_temporal, near_value_parameters = fit_predictors(
        keys_train,
        values_train,
        config=near_rate_config,
        history_length=args.k_history,
        iterations=args.fit_iterations,
        temporal_ridge=1.0,
        cross_ridge=0.1,
        v_history_ridge=0.0,
        max_gain=1.0,
        v_history_mode="diagonal",
        layer_ids=layer_ids,
        coordinate_system=str(trace_metadata.get("coordinate_system", "unspecified")),
        fit_history_source="reconstructed",
    )
    if not isinstance(near_value_parameters, CrossKVParameters):
        raise TypeError("near-rate baseline requires CrossKV V parameters")
    near_fit_seconds = time.perf_counter() - near_fit_started

    scalar_started = time.perf_counter()
    tempo_k = _reconstruct_temporal_trace(keys_eval, temporal, config)
    tempo_v = _reconstruct_cross_kv_trace(
        tempo_k, values_eval, value_parameters, config
    )
    scalar_seconds = time.perf_counter() - scalar_started

    near_started = time.perf_counter()
    near_k = _reconstruct_temporal_trace(
        keys_eval,
        near_temporal,
        near_rate_config,
    )
    near_v = _reconstruct_cross_kv_trace(
        near_k,
        values_eval,
        near_value_parameters,
        near_rate_config,
    )
    near_seconds = time.perf_counter() - near_started

    codebook_started = time.perf_counter()
    k_codebook, v_codebook = _fit_fusion_codebooks(
        keys_train,
        values_train,
        temporal,
        value_parameters,
        config,
        args,
    )
    codebook_seconds = time.perf_counter() - codebook_started

    fusion_started = time.perf_counter()
    fused_k = _reconstruct_temporal_trace(
        keys_eval,
        temporal,
        config,
        residual_roundtrip=_vq_roundtrip(config, config.k_bits, k_codebook),
    )
    fused_v = _reconstruct_cross_kv_trace(
        fused_k,
        values_eval,
        value_parameters,
        config,
        residual_roundtrip=_vq_roundtrip(config, config.v_bits, v_codebook),
    )
    fusion_seconds = time.perf_counter() - fusion_started

    raw_k_codebook = _fit_codebook(
        keys_train[..., 1:, :, :], args, seed_offset=2_000_000
    )
    raw_v_codebook = _fit_codebook(
        values_train[..., 1:, :, :], args, seed_offset=3_000_000
    )
    qvg_started = time.perf_counter()
    qvg_k = _reconstruct_direct_trace(
        keys_eval,
        config,
        residual_roundtrip=_vq_roundtrip(config, config.k_bits, raw_k_codebook),
    )
    qvg_v = _reconstruct_direct_trace(
        values_eval,
        config,
        bits=config.v_bits,
        residual_roundtrip=_vq_roundtrip(config, config.v_bits, raw_v_codebook),
    )
    qvg_seconds = time.perf_counter() - qvg_started

    methods = {
        "tempokv_scalar": _method_metrics(
            keys_eval, values_eval, tempo_k, tempo_v, layer_ids
        ),
        "tempokv_near_rate": _method_metrics(
            keys_eval, values_eval, near_k, near_v, layer_ids
        ),
        "qvg_style_fixed_prq": _method_metrics(
            keys_eval, values_eval, qvg_k, qvg_v, layer_ids
        ),
        "tempokv_qvg_fusion": _method_metrics(
            keys_eval, values_eval, fused_k, fused_v, layer_ids
        ),
    }
    for role in ("k", "v"):
        baseline = methods["tempokv_scalar"][role]["pooled"]["nmse"]
        near_rate = methods["tempokv_near_rate"][role]["pooled"]["nmse"]
        fused = methods["tempokv_qvg_fusion"][role]["pooled"]["nmse"]
        methods["tempokv_qvg_fusion"][role]["vs_tempokv_nmse_ratio"] = fused / baseline
        methods["tempokv_qvg_fusion"][role]["vs_near_rate_nmse_ratio"] = (
            fused / near_rate
        )

    frames = int(keys.shape[-3])
    dim = int(keys.shape[-1])
    scalar_bits = _physical_bits(
        frames=frames,
        dim=dim,
        residual_bits=args.bits,
        anchor_bits=args.anchor_bits,
        group_size=args.group_size,
        with_vq_index=False,
    )
    fusion_bits = _physical_bits(
        frames=frames,
        dim=dim,
        residual_bits=args.bits,
        anchor_bits=args.anchor_bits,
        group_size=args.group_size,
        with_vq_index=True,
    )
    near_rate_bits = _physical_bits(
        frames=frames,
        dim=dim,
        residual_bits=args.bits,
        anchor_bits=near_rate_config.anchor_bits,
        group_size=args.group_size,
        with_vq_index=False,
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "scope": "trace-level reconstruction; QVG-style fixed one-stage residual VQ",
        "trace": str(args.trace.resolve()),
        "trace_metadata": trace_metadata,
        "shape_used": list(keys.shape),
        "layer_ids": list(layer_ids),
        "split": {
            "train_contexts": args.train_contexts,
            "eval_contexts": args.eval_contexts,
            "train_indices": [0, args.train_contexts],
            "eval_indices": [
                args.train_contexts,
                args.train_contexts + args.eval_contexts,
            ],
        },
        "config": {
            "bits": args.bits,
            "anchor_bits": args.anchor_bits,
            "group_size": args.group_size,
            "k_history": args.k_history,
            "v_history": args.v_history,
            "fit_iterations": args.fit_iterations,
            "codebook_size": args.codebook_size,
            "codebook_iterations": args.codebook_iterations,
            "codebook_max_points": args.codebook_max_points,
            "codebook_rounds": args.codebook_rounds,
            "seed": args.seed,
            "device": str(device),
        },
        "rate": {
            "tempokv_physical_cache_bits_per_value": scalar_bits,
            "tempokv_near_rate_anchor_bits": near_rate_config.anchor_bits,
            "tempokv_near_rate_physical_cache_bits_per_value": near_rate_bits,
            "fusion_physical_cache_bits_per_value": fusion_bits,
            "vq_physical_index_bits_per_residual_value": k_codebook.physical_index_bits_per_value(),
            "vq_logical_index_bits_per_residual_value": k_codebook.logical_index_bits_per_value(),
            "fixed_codebook_parameter_bytes": {
                "fusion_k": k_codebook.parameter_bytes(),
                "fusion_v": v_codebook.parameter_bytes(),
                "qvg_raw_k": raw_k_codebook.parameter_bytes(),
                "qvg_raw_v": raw_v_codebook.parameter_bytes(),
            },
        },
        "timing_seconds": {
            "predictor_fit": fit_seconds,
            "near_rate_predictor_fit": near_fit_seconds,
            "fusion_codebook_fit": codebook_seconds,
            "tempokv_eval": scalar_seconds,
            "tempokv_near_rate_eval": near_seconds,
            "qvg_style_eval": qvg_seconds,
            "fusion_eval": fusion_seconds,
            "total": time.perf_counter() - started,
        },
        "methods": methods,
        "limitations": [
            "This is a held-out trace reconstruction screen, not end-to-end video quality.",
            "The QVG baseline uses offline fixed codebooks so the same train/eval split applies; it is not original online per-span PRQ.",
            "The fusion adds one UINT8 centroid index per residual vector and is not exactly rate-matched to scalar TempoKV.",
            "The near-rate TempoKV baseline adds one anchor bit; its rate is close to, but not identical to, the fusion rate.",
        ],
    }
    return result


def _write_outputs(result: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_layer.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "tensor", "layer_id", "mse", "nmse", "relative_l2", "max_abs", "values"),
        )
        writer.writeheader()
        for method_name, method in result["methods"].items():
            for tensor_name in ("k", "v"):
                for row in method[tensor_name]["per_layer"]:
                    writer.writerow(
                        {"method": method_name, "tensor": tensor_name.upper(), **row}
                    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args)
    _write_outputs(result, args.output_dir)
    summary = {
        method: {
            role: metrics[role]["pooled"]["nmse"]
            for role in ("k", "v")
        }
        for method, metrics in result["methods"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
