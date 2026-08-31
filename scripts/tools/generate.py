#!/usr/bin/env python3
"""Quantize and reconstruct exported K/V caches with TempoKV."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv import (
    CodecConfig,
    CrossKVParameters,
    TempoKVCodec,
    TemporalParameters,
    configure_triton_cache,
)
from tempokv.config import config_from_mapping, is_temporal_v_predictor


CONFIG_FIELDS = tuple(CodecConfig.__dataclass_fields__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Encode exported K/V caches, decode them through the same closed loop, "
            "and save reconstructed caches plus a compression report."
        )
    )
    parser.add_argument("input", type=Path, help="NPZ containing keys and values")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--config", type=Path, required=True, help="Codec JSON file")
    parser.add_argument("--k-params", type=Path, required=True)
    parser.add_argument("--v-params", type=Path, required=True)
    parser.add_argument("--k-key", default="keys")
    parser.add_argument("--v-key", default="values")
    parser.add_argument(
        "--layer-ids",
        default="",
        help="Comma-separated model layer IDs; defaults to parameter metadata",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--triton-cache", type=Path)
    parser.add_argument("--k-bits", type=int, choices=(2, 4, 8))
    parser.add_argument("--v-bits", type=int, choices=(2, 4, 8))
    parser.add_argument("--anchor-bits", type=int, choices=(2, 4, 8))
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--k-history-length", type=int)
    parser.add_argument("--v-history-length", type=int, choices=range(0, 6))
    parser.add_argument(
        "--residual-mode", choices=("symmetric", "asymmetric")
    )
    parser.add_argument(
        "--scale-dtype", choices=("float32", "float16", "bfloat16")
    )
    parser.add_argument(
        "--reconstruction-dtype",
        choices=("float32", "float16", "bfloat16"),
    )
    triton = parser.add_mutually_exclusive_group()
    triton.add_argument("--use-triton", action="store_true", default=None)
    triton.add_argument("--no-triton", action="store_false", dest="use_triton")
    strict = parser.add_mutually_exclusive_group()
    strict.add_argument("--strict-triton", action="store_true", default=None)
    strict.add_argument(
        "--allow-triton-fallback", action="store_false", dest="strict_triton"
    )
    return parser


def _load_config(path: Path, args: argparse.Namespace) -> CodecConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("codec config must be a JSON object")
    unknown = sorted(set(payload) - set(CONFIG_FIELDS))
    if unknown:
        raise ValueError(f"unknown codec config fields: {unknown}")
    config = config_from_mapping(payload)
    if config.fit_mode != "legacy":
        raise ValueError("only fit_mode=legacy is supported")
    if config.cross_kv_weight_mode != "layer_shared":
        raise ValueError(
            "production CrossKV parameters must use "
            "cross_kv_weight_mode=layer_shared"
        )
    overrides = {
        field: getattr(args, field)
        for field in CONFIG_FIELDS
        if hasattr(args, field) and getattr(args, field) is not None
    }
    return replace(config, **overrides)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _normalize_cache(array: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    value = np.asarray(array)
    if value.ndim == 4:
        return value[None], True
    if value.ndim == 5:
        return value, False
    raise ValueError(
        f"{name} must have shape [B,H,S,D] or [L,B,H,S,D], got {value.shape}"
    )


def _layer_ids(
    text: str,
    layers: int,
    temporal: TemporalParameters,
    value_parameters: CrossKVParameters | TemporalParameters,
) -> tuple[int, ...]:
    if text.strip():
        result = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    else:
        candidates = []
        if temporal.layered:
            candidates.append(temporal.layer_ids)
        if value_parameters.layered:
            candidates.append(value_parameters.layer_ids)
        if candidates and any(item != candidates[0] for item in candidates[1:]):
            raise ValueError("K and V parameter layer IDs do not match")
        result = candidates[0] if candidates else tuple(range(layers))
    if result is None or len(result) != layers or len(set(result)) != layers:
        raise ValueError(f"expected {layers} unique layer IDs, got {result}")
    return tuple(result)


def _tensor_bytes(value: torch.Tensor | None) -> int:
    return 0 if value is None else value.numel() * value.element_size()


def _packed_bytes(packed: Any) -> int:
    return sum(
        _tensor_bytes(getattr(packed, name))
        for name in (
            "anchor_quant",
            "anchor_scale",
            "residual_quant",
            "residual_scale",
            "residual_zero_point",
        )
    )


def _nmse(reference: torch.Tensor, rebuilt: torch.Tensor) -> float:
    difference = rebuilt.float() - reference.float()
    denominator = reference.float().square().sum().clamp_min(1e-12)
    return float(difference.square().sum().div(denominator).item())


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config, args)
    device = _device(args.device)
    if config.use_triton and device.type == "cuda":
        configure_triton_cache(args.triton_cache)

    temporal = TemporalParameters.load(args.k_params)
    value_parameters: CrossKVParameters | TemporalParameters
    if is_temporal_v_predictor(config.v_predictor_mode):
        value_parameters = TemporalParameters.load(args.v_params)
    else:
        value_parameters = CrossKVParameters.load(args.v_params)
    with np.load(args.input, allow_pickle=False) as data:
        if args.k_key not in data or args.v_key not in data:
            raise KeyError(
                f"input must contain {args.k_key!r} and {args.v_key!r}"
            )
        keys, single_layer = _normalize_cache(data[args.k_key], args.k_key)
        values, value_single_layer = _normalize_cache(data[args.v_key], args.v_key)
    if single_layer != value_single_layer or keys.shape != values.shape:
        raise ValueError("keys and values must have the same layout and shape")

    layers = int(keys.shape[0])
    layer_ids = _layer_ids(args.layer_ids, layers, temporal, value_parameters)
    codec = TempoKVCodec(temporal, value_parameters, config)
    rebuilt_keys = []
    rebuilt_values = []
    per_layer = []
    input_bytes = 0
    packed_bytes = 0
    for position, layer_id in enumerate(layer_ids):
        key = torch.from_numpy(np.asarray(keys[position])).to(device=device)
        value = torch.from_numpy(np.asarray(values[position])).to(device=device)
        if not key.is_floating_point() or not value.is_floating_point():
            raise TypeError("keys and values must use floating-point dtypes")
        packed = codec.encode(key, value, layer=layer_id)
        key_rebuilt, value_rebuilt = codec.decode(packed)
        layer_input_bytes = (
            key.numel() * key.element_size() + value.numel() * value.element_size()
        )
        layer_packed_bytes = _packed_bytes(packed.key) + _packed_bytes(packed.value)
        input_bytes += layer_input_bytes
        packed_bytes += layer_packed_bytes
        per_layer.append(
            {
                "layer_id": layer_id,
                "key_nmse": _nmse(key, key_rebuilt),
                "value_nmse": _nmse(value, value_rebuilt),
                "input_bytes": layer_input_bytes,
                "packed_cache_bytes": layer_packed_bytes,
            }
        )
        rebuilt_keys.append(key_rebuilt.float().cpu().numpy())
        rebuilt_values.append(value_rebuilt.float().cpu().numpy())

    key_output = np.stack(rebuilt_keys)
    value_output = np.stack(rebuilt_values)
    if single_layer:
        key_output = key_output[0]
        value_output = value_output[0]
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "reconstructed.npz",
        keys=key_output,
        values=value_output,
        layer_ids=np.asarray(layer_ids, dtype=np.int32),
    )
    report = {
        "status": "complete",
        "input": str(args.input.resolve()),
        "config": asdict(config),
        "device": str(device),
        "layer_ids": list(layer_ids),
        "input_bytes": input_bytes,
        "packed_cache_bytes": packed_bytes,
        "compression_ratio": input_bytes / max(1, packed_bytes),
        "layers": per_layer,
    }
    with (args.output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
