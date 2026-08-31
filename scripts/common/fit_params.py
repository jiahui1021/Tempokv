#!/usr/bin/env python3
"""Fit TempoKV parameters from a clean K/V trace archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch


# Keep this entry point usable from an uninstalled checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.config import (  # noqa: E402
    CodecConfig,
    FIXED_TEMPORAL_V_PREDICTOR_MODES,
    TEMPORAL_V_PREDICTOR_MODES,
    config_from_mapping,
    is_temporal_v_predictor,
)
from tempokv.fitting import fit_predictors, refine_attention  # noqa: E402
from tempokv.parameters import CrossKVParameters, TemporalParameters  # noqa: E402


CODEC_FIELDS = {field.name for field in fields(CodecConfig)}

# The legacy fitting protocol collects two 24-frame records per prompt.  Keep
# these in step with `Self-Forcing/tempokv_adapter/prepare_trace.py`, which
# writes the metadata checked below.
LEGACY_PROMPT_COUNT = 12
LEGACY_TRACE_PROTOCOL = f"direct_v_{LEGACY_PROMPT_COUNT}p_51f_8t_24x2_v1"
# The attention objective needs the rotated Q and K the model actually used.
ATTENTION_ARRAYS = ("q_post_rope", "k_post_rope")


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
        "--v-history-mode",
        choices=("dense", "diagonal"),
        default="dense",
        help="V-history map: dense fits a full DxD matrix per lag so the "
        "history can mix channels; diagonal fits one coefficient per output "
        "channel and cannot represent cross-channel recurrence",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--fit-history-source",
        choices=("reconstructed", "clean"),
        default="reconstructed",
        help=(
            "history visible during fitting: reconstructed matches deployment; "
            "clean is a teacher-forced ablation"
        ),
    )
    parser.add_argument("--temporal-ridge", type=float)
    parser.add_argument("--cross-ridge", type=float)
    parser.add_argument("--v-history-ridge", type=float)
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
    parser.add_argument(
        "--span-frames",
        type=int,
        help=(
            "split every calibration event into independent spans of this "
            "many temporal units before fitting"
        ),
    )
    parser.add_argument("--v-history-length", type=int, choices=range(0, 6))
    parser.add_argument("--residual-mode", choices=("symmetric", "asymmetric"))
    parser.add_argument("--scale-dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument(
        "--reconstruction-dtype", choices=("float32", "float16", "bfloat16")
    )
    parser.add_argument("--cross-matmul-precision", choices=("ieee", "tf32"))
    parser.add_argument(
        "--k-predictor-mode", choices=("shared", "direct")
    )
    parser.add_argument(
        "--v-predictor-mode",
        choices=(
            "cross_kv",
            "temporal",
            "identity",
            "taylor",
            "taylor2",
            "innovation",
            "direct",
        ),
        help=(
            "V predictor: cross_kv, temporal, identity, taylor, taylor2, "
            "innovation, or direct"
        ),
    )
    parser.add_argument(
        "--predictor-direction",
        choices=("k_to_v", "v_to_k"),
        help=(
            "which tensor carries the cross dependency: k_to_v predicts V from "
            "reconstructed K (production), v_to_k mirrors the roles so K is "
            "projected from reconstructed V"
        ),
    )
    parser.add_argument(
        "--attention-steps",
        type=int,
        help="maximum Adam steps per phase; any positive value enables refinement",
    )
    parser.add_argument(
        "--attention-optimizer",
        choices=("adam", "sign_gradient"),
        default="adam",
        help="refinement optimizer; sign_gradient reproduces the historical scan",
    )
    parser.add_argument(
        "--attention-lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--attention-patience",
        type=int,
        default=10,
        help="stop a phase after this many steps without material improvement",
    )
    parser.add_argument(
        "--attention-max-tokens",
        type=int,
        help="deterministically cap flattened sampled token tracks during refinement",
    )
    parser.add_argument(
        "--attention-min-relative-improvement",
        type=float,
        default=1e-4,
        help="relative score improvement that resets patience (default: 1e-4)",
    )
    parser.add_argument(
        "--attention-layer-chunk",
        type=int,
        default=4,
        help="layers per refinement chunk; bounds memory without changing results",
    )
    parser.add_argument(
        "--attention-reconstruction-weight",
        type=float,
        default=0.5,
        help="normalized own-reconstruction objective weight (default: 0.5)",
    )
    parser.add_argument(
        "--attention-output-weight",
        type=float,
        default=0.5,
        help="normalized attention-output objective weight (default: 0.5)",
    )
    parser.add_argument(
        "--attention-refine-tensors",
        choices=("key", "value", "key_value"),
        default="key_value",
        help="run only K, only V, or sequential K+V refinement",
    )
    parser.add_argument(
        "--attention-sign-step-sizes",
        type=float,
        nargs="+",
        default=(0.01,),
        help="candidate step sizes for the historical sign-gradient scan",
    )
    parser.add_argument("--fit-mode", choices=("legacy",))
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
    values = _read_json_config(args.config)
    for name in CODEC_FIELDS:
        override = getattr(args, name, None)
        if override is not None:
            values[name] = override
    # CrossKV sharing is irrelevant when V predicts from its own history, but
    # keep the single repository-wide mode in the resolved identity.
    if values.get("v_predictor_mode") in (
        (TEMPORAL_V_PREDICTOR_MODES - {"direct"}) | {"innovation"}
    ):
        values["cross_kv_weight_mode"] = "layer_shared"
    try:
        config = config_from_mapping(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid codec config: {error}") from error
    if config.fit_mode != "legacy":
        raise ValueError("only fit_mode=legacy is supported")
    return config


def _load_trace(
    path: Path,
    *,
    keys_name: str,
    values_name: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...] | None,
    dict[str, object],
    dict[str, torch.Tensor],
]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [name for name in (keys_name, values_name) if name not in archive]
            if missing:
                raise ValueError(f"trace is missing arrays: {', '.join(missing)}")
            keys_array = np.array(archive[keys_name], copy=True)
            values_array = np.array(archive[values_name], copy=True)
            # Present only in traces collected for the attention objective.
            rotated = {
                name: torch.from_numpy(np.array(archive[name], copy=True))
                for name in ATTENTION_ARRAYS
                if name in archive
            }
            layer_ids_array = (
                np.array(archive["layer_ids"], copy=True) if "layer_ids" in archive else None
            )
            metadata: dict[str, object] = {}
            for name in (
                "temporal_units",
                "num_layers",
                "num_heads",
                "head_dim",
                "coordinate_system",
                "cross_source_kind",
                "trace_layout",
                "trace_event_count",
                "prompt_count",
                "frames_per_event",
                "tokens_per_frame",
                "legacy_trace_protocol",
                "trace_role",
            ):
                if name in archive:
                    metadata[name] = np.asarray(archive[name]).item()
            for name in (
                "token_indices",
                "event_tokens_start",
                "event_tokens_end",
            ):
                if name in archive:
                    array = np.asarray(archive[name])
                    if array.ndim != 1 or not np.issubdtype(
                        array.dtype, np.integer
                    ):
                        raise ValueError(
                            f"trace {name} metadata must be a one-dimensional "
                            "integer array"
                        )
                    metadata[name] = tuple(int(item) for item in array)
            named_trace_stride = (
                np.asarray(archive["trace_stride"]).item()
                if "trace_stride" in archive
                else None
            )
            legacy_stride = (
                np.asarray(archive["stride"]).item()
                if "stride" in archive
                else None
            )
            if (
                named_trace_stride is not None
                and legacy_stride is not None
                and int(named_trace_stride) != int(legacy_stride)
            ):
                raise ValueError("trace contains conflicting trace stride metadata")
            trace_stride = (
                named_trace_stride
                if named_trace_stride is not None
                else legacy_stride
            )
            if trace_stride is not None:
                trace_stride = int(trace_stride)
                if trace_stride <= 0:
                    raise ValueError("trace_stride metadata must be positive")
                metadata["trace_stride"] = trace_stride
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
    return keys, values, layer_ids, metadata, rotated


def _fit_device(name: str) -> torch.device:
    """Resolve the fitting device, defaulting to the GPU.

    `auto` prefers CUDA and falls back to the CPU, but says so loudly: CPU
    fitting also runs its equations in float64 rather than float32, so an
    unnoticed fallback is far slower than it looks, not marginally so.
    Pass `cuda` to make an unavailable GPU an error instead of a fallback.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print(
            "[fit] CUDA is unavailable; fitting on the CPU in float64, which is "
            "much slower. Set --device cuda (FIT_DEVICE=cuda) to fail instead "
            "of falling back.",
            file=sys.stderr,
        )
        return torch.device("cpu")
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


def _reset_cuda_peak_memory_stats(device: torch.device) -> None:
    """Initialize the selected CUDA context before resetting its counters."""
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)


def _limit_attention_tracks(
    tensor: torch.Tensor, max_tokens: int | None
) -> torch.Tensor:
    """Flatten non-layer sample axes and keep a deterministic prefix."""
    if max_tokens is None:
        return tensor
    if max_tokens <= 0:
        raise ValueError("attention_max_tokens must be positive")
    if tensor.ndim < 5:
        raise ValueError(
            "attention token limiting needs [L, ..., F, H, D] with a sample axis"
        )
    tracks = int(np.prod(tensor.shape[1:-3]))
    flattened = tensor.reshape(
        tensor.shape[0], tracks, tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]
    )
    return flattened[:, : min(max_tokens, tracks)]


def _split_calibration_spans(
    tensor: torch.Tensor, span_frames: int, *, legacy: bool
) -> torch.Tensor:
    """Reset fitting history at each non-overlapping calibration span."""
    if span_frames < 2:
        raise ValueError("span_frames must be at least 2")
    frame_count = int(tensor.shape[-3])
    windows = frame_count // span_frames
    if windows < 1:
        raise ValueError(
            f"span_frames={span_frames} exceeds trace length {frame_count}"
        )
    pieces = [
        tensor[..., index * span_frames : (index + 1) * span_frames, :, :]
        for index in range(windows)
    ]
    if legacy:
        # [L,E,B,P,F,H,D] -> [L,E*windows,B,P,span,H,D]
        return torch.cat(pieces, dim=1).contiguous()
    # Preserve every pre-temporal sample axis and introduce a window axis.
    return torch.stack(pieces, dim=1).contiguous()


def _pad_temporal_history(
    parameters: TemporalParameters, target: int
) -> TemporalParameters:
    current = parameters.history_length
    if current == target:
        return parameters
    if not 0 < current < target:
        raise ValueError("invalid temporal history padding request")
    weights = parameters.history_weights
    bias = parameters.history_bias
    if weights.ndim == 5:
        padded_weights = weights.new_zeros(
            weights.shape[0], target, target, *weights.shape[-2:]
        )
        padded_bias = bias.new_zeros(bias.shape[0], target, *bias.shape[-2:])
        padded_weights[:, :current, :current] = weights
        padded_bias[:, :current] = bias
    else:
        padded_weights = weights.new_zeros(target, target, *weights.shape[2:])
        padded_bias = bias.new_zeros(target, *bias.shape[1:])
        padded_weights[:current, :current] = weights
        padded_bias[:current] = bias
    return TemporalParameters(
        padded_weights,
        padded_bias,
        parameters.stride,
        parameters.layer_ids,
        coordinate_system=parameters.coordinate_system,
    )


def _pad_cross_history(
    parameters: CrossKVParameters, target: int
) -> CrossKVParameters:
    current = parameters.history_length
    if current == target:
        return parameters
    if not 0 <= current < target:
        raise ValueError("invalid CrossKV history padding request")
    history = parameters.history_weights
    if parameters.layered:
        padded = history.new_zeros(
            history.shape[0], target, *history.shape[2:]
        )
        padded[:, :current] = history
    else:
        padded = history.new_zeros(target, *history.shape[1:])
        padded[:current] = history
    return CrossKVParameters(
        parameters.weight,
        parameters.bias,
        parameters.stride,
        parameters.layer_ids,
        coordinate_system=parameters.coordinate_system,
        history_weights=padded,
        projection_source=parameters.projection_source,
    )


def _bind_requested_config(
    temporal: TemporalParameters,
    value_parameters: CrossKVParameters | TemporalParameters,
    config: CodecConfig,
    coordinate_system: str,
) -> tuple[TemporalParameters, CrossKVParameters | TemporalParameters]:
    """Bind padded, span-specific fits to the production codec identity."""
    signature = config.format_signature()
    identity = hashlib.sha256(signature.encode("utf-8"))
    identity.update(coordinate_system.encode("utf-8"))
    tensors = [temporal.history_weights, temporal.history_bias]
    if isinstance(value_parameters, TemporalParameters):
        tensors.extend(
            [value_parameters.history_weights, value_parameters.history_bias]
        )
    else:
        tensors.extend(
            [
                value_parameters.weight,
                value_parameters.history_weights,
                value_parameters.bias,
            ]
        )
    for tensor in tensors:
        identity.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    parameter_set_id = identity.hexdigest()
    rebound_temporal = TemporalParameters(
        temporal.history_weights,
        temporal.history_bias,
        temporal.stride,
        temporal.layer_ids,
        signature,
        parameter_set_id,
        coordinate_system,
    )
    if isinstance(value_parameters, TemporalParameters):
        rebound_value = TemporalParameters(
            value_parameters.history_weights,
            value_parameters.history_bias,
            value_parameters.stride,
            value_parameters.layer_ids,
            signature,
            parameter_set_id,
            coordinate_system,
        )
    else:
        rebound_value = CrossKVParameters(
            value_parameters.weight,
            value_parameters.bias,
            value_parameters.stride,
            value_parameters.layer_ids,
            signature,
            parameter_set_id,
            coordinate_system,
            value_parameters.history_weights,
            value_parameters.projection_source,
        )
    return rebound_temporal, rebound_value


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _codec_config(args)
    if config.v_predictor_mode == "innovation" and args.attention_steps is not None:
        raise ValueError(
            "attention refinement does not support v_predictor_mode=innovation; "
            "fit innovation parameters without --attention-steps"
        )
    if (
        config.v_predictor_mode in FIXED_TEMPORAL_V_PREDICTOR_MODES
        and args.attention_steps is not None
    ):
        raise ValueError(
            f"attention refinement does not support fixed {config.v_predictor_mode} V; "
            "fixed predictor coefficients are not fitted"
        )
    temporal_ridge = (
        1.0
        if args.temporal_ridge is None and config.fit_mode == "legacy"
        else 0.0 if args.temporal_ridge is None else args.temporal_ridge
    )
    cross_ridge = (
        0.1
        if args.cross_ridge is None and config.fit_mode == "legacy"
        else 0.0 if args.cross_ridge is None else args.cross_ridge
    )
    v_history_ridge = (
        0.0 if args.v_history_ridge is None else args.v_history_ridge
    )
    v_history_mode = (
        "diagonal"
        if config.fit_mode == "legacy" and config.v_history_length
        else args.v_history_mode
    )
    config_sha256 = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    keys, values, layer_ids, trace_metadata, rotated = _load_trace(
        args.trace,
        keys_name=args.keys_name,
        values_name=args.values_name,
    )
    trace_shape_metadata = {
        "num_layers": int(keys.shape[0]),
        "num_heads": int(keys.shape[-2]),
        "head_dim": int(keys.shape[-1]),
        "temporal_units": int(keys.shape[-3]),
    }
    for name, actual in trace_shape_metadata.items():
        recorded = trace_metadata.get(name)
        if recorded is not None and int(recorded) != actual:
            raise ValueError(
                f"trace {name} metadata is {int(recorded)}, but the array has {actual}"
            )
    if config.fit_mode == "legacy":
        if trace_metadata.get("trace_role", "calibration") != "calibration":
            raise ValueError(
                "legacy fitting rejects diagnostic traces; use the calibration trace"
            )
        trace_layout = trace_metadata.get("trace_layout")
        if trace_layout is not None and trace_layout != "legacy_event_bpfhd_v1":
            raise ValueError(f"unsupported legacy trace layout {trace_layout!r}")
        if trace_layout == "legacy_event_bpfhd_v1":
            events = 2 * LEGACY_PROMPT_COUNT
            expected_trace_metadata: dict[str, object] = {
                "trace_layout": "legacy_event_bpfhd_v1",
                "trace_event_count": events,
                "prompt_count": LEGACY_PROMPT_COUNT,
                "frames_per_event": 24,
                "tokens_per_frame": 8,
                "legacy_trace_protocol": LEGACY_TRACE_PROTOCOL,
                "token_indices": (0, 223, 445, 668, 891, 1114, 1336, 1559),
                "event_tokens_start": (0, 37440) * LEGACY_PROMPT_COUNT,
                "event_tokens_end": (37440, 74880) * LEGACY_PROMPT_COUNT,
                "trace_stride": 1560,
            }
            for name, expected in expected_trace_metadata.items():
                actual = trace_metadata.get(name)
                if actual != expected:
                    raise ValueError(
                        f"legacy fitting requires trace {name}={expected!r}, "
                        f"got {actual!r}; recollect the legacy-protocol trace"
                    )
            expected_shape = (30, events, 1, 8, 24, 12, 128)
            if tuple(keys.shape) != expected_shape:
                raise ValueError(
                    f"legacy trace must use {list(expected_shape)} layout, got "
                    f"{tuple(keys.shape)}"
                )
            expected_layer_ids = tuple(range(30))
        else:
            # LongCat writes the regular [L,S,T,H,D] fitting layout.  The
            # legacy equations are shape-independent, so validate the model
            # dimensions from the archive instead of imposing the 30-layer,
            # 12-head event protocol used by the two Forcing trace bridges.
            if keys.ndim != 5:
                raise ValueError(
                    "legacy standard traces must use [L,S,T,H,D] layout, got "
                    f"{tuple(keys.shape)}"
                )
            expected_layer_ids = tuple(range(int(keys.shape[0])))
            trace_stride = trace_metadata.get("trace_stride")
            if trace_stride is None:
                raise ValueError("legacy standard traces must record trace_stride")
            if int(trace_stride) != int(config.stride):
                raise ValueError(
                    f"legacy trace stride {int(trace_stride)} does not match "
                    f"codec stride {int(config.stride)}"
                )
        if keys.dtype != torch.float32 or values.dtype != torch.float32:
            raise ValueError("legacy trace K/V arrays must use float32")
        if layer_ids != expected_layer_ids:
            raise ValueError(
                "legacy trace layer_ids must cover every layer in order, got "
                f"{layer_ids!r}"
            )
    trace_coordinate_system = trace_metadata.get("coordinate_system")
    coordinate_system = args.coordinate_system
    if trace_coordinate_system is not None:
        trace_coordinate_system = str(trace_coordinate_system)
        if coordinate_system == "unspecified":
            coordinate_system = trace_coordinate_system
        elif coordinate_system != trace_coordinate_system:
            raise ValueError(
                f"trace coordinate system {trace_coordinate_system!r} does not match "
                f"requested {coordinate_system!r}"
            )
    if trace_metadata.get("cross_source_kind") not in (None, "normalized"):
        raise ValueError("trace CrossKV source must be normalized for TempoKV")
    requested_k_history = (
        args.history_length
        if args.history_length is not None
        else config.k_history_length if config.k_history_length is not None else 3
    )
    fit_config = config
    if args.span_frames is not None:
        keys = _split_calibration_spans(
            keys, args.span_frames, legacy=config.fit_mode == "legacy"
        )
        values = _split_calibration_spans(
            values, args.span_frames, legacy=config.fit_mode == "legacy"
        )
        rotated = {
            name: _split_calibration_spans(
                tensor, args.span_frames, legacy=config.fit_mode == "legacy"
            )
            for name, tensor in rotated.items()
        }
        reachable = args.span_frames - 1
        fit_config = replace(
            config,
            k_history_length=min(int(requested_k_history), reachable),
            v_history_length=min(int(config.v_history_length), reachable),
        )
    device = _fit_device(args.device)
    print(f"[fit] device={device}", file=sys.stderr)
    if device.type == "cuda":
        _reset_cuda_peak_memory_stats(device)
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
        config=fit_config,
        history_length=(
            fit_config.k_history_length
            if fit_config.k_history_length is not None
            else int(requested_k_history)
        ),
        iterations=args.iterations,
        temporal_ridge=temporal_ridge,
        cross_ridge=cross_ridge,
        v_history_ridge=v_history_ridge,
        max_gain=args.max_gain,
        v_history_mode=v_history_mode,
        layer_ids=layer_ids,
        coordinate_system=coordinate_system,
        fit_history_source=args.fit_history_source,
    )
    ls_temporal, ls_cross = temporal, cross
    refinement = None
    if args.attention_steps is not None:
        missing = [name for name in ATTENTION_ARRAYS if name not in rotated]
        if missing:
            raise ValueError(
                f"attention refinement needs {', '.join(missing)} in the trace; "
                "recollect it with the current trace bridge"
            )
        refinement_tensors = tuple(
            _limit_attention_tracks(tensor, args.attention_max_tokens)
            for tensor in (
                keys,
                values,
                rotated["q_post_rope"].to(device=device, dtype=keys.dtype),
                rotated["k_post_rope"].to(device=device, dtype=keys.dtype),
            )
        )
        temporal, cross, refinement = refine_attention(
            *refinement_tensors,
            temporal,
            cross,
            fit_config,
            steps=args.attention_steps,
            learning_rate=args.attention_lr,
            patience=args.attention_patience,
            min_relative_improvement=args.attention_min_relative_improvement,
            max_gain=args.max_gain,
            layer_chunk=args.attention_layer_chunk,
            optimizer_name=args.attention_optimizer,
            reconstruction_weight=args.attention_reconstruction_weight,
            attention_weight=args.attention_output_weight,
            refine_tensors=(
                ("key", "value")
                if args.attention_refine_tensors == "key_value"
                else (args.attention_refine_tensors,)
            ),
            sign_step_sizes=tuple(args.attention_sign_step_sizes),
        )
    if temporal.history_length != int(requested_k_history):
        temporal = _pad_temporal_history(temporal, int(requested_k_history))
    if is_temporal_v_predictor(config.v_predictor_mode):
        if cross.history_length != int(config.v_history_length):
            cross = _pad_temporal_history(cross, int(config.v_history_length))
    elif cross.history_length != int(config.v_history_length):
        cross = _pad_cross_history(cross, int(config.v_history_length))
    temporal, cross = _bind_requested_config(
        temporal, cross, config, coordinate_system
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
    # The file names state which tensor each parameter set describes and how
    # it is predicted, so a mirrored set can never be mistaken on disk for a
    # production one.  Under v_to_k the temporal parameters describe V and the
    # projection parameters describe K, which is the reverse of every other
    # configuration.
    temporal_v = is_temporal_v_predictor(config.v_predictor_mode)
    if config.predictor_direction == "v_to_k":
        temporal_stem = "v_temporal_primary"
        v_artifact_stem = "k_cross_from_v"
    else:
        temporal_stem = "k_temporal"
        v_artifact_stem = {
            "temporal": "v_temporal",
            "cross_kv": "v_cross_kv",
            "innovation": "v_innovation",
            # Direct V keeps an empty temporal map, so it shares the temporal file.
            "direct": "v_temporal",
            "identity": "v_temporal",
            "taylor": "v_temporal",
            "taylor2": "v_temporal",
        }[config.v_predictor_mode]
    temporal_path = output_dir / f"{temporal_stem}.npz"
    cross_path = output_dir / f"{v_artifact_stem}.npz"
    ls_temporal_path = output_dir / f"{temporal_stem}_ls.npz"
    ls_cross_path = output_dir / f"{v_artifact_stem}_ls.npz"
    report_path = output_dir / "fit_report.json"
    parameter_tensors = (
        (temporal.history_weights, temporal.history_bias,
         cross.history_weights, cross.history_bias)
        if temporal_v
        else (
            temporal.history_weights,
            temporal.history_bias,
            cross.weight,
            cross.history_weights,
            cross.bias,
        )
    )
    parameter_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in parameter_tensors
    )
    logical_parameter_bytes = parameter_bytes
    cross_weight_sharing = None
    if not temporal_v and config.cross_kv_weight_mode != "layer_head":
        # The experiment deliberately keeps the deployed [L,H,...] tensors so
        # every existing Torch/Triton generation path stays unchanged.  Report
        # the actual artifact size and the logical size at the selected sharing
        # granularity.
        expanded_cross_bytes = (
            cross.weight.numel() * cross.weight.element_size()
            + cross.bias.numel() * cross.bias.element_size()
        )
        dim = int(cross.weight.shape[-1])
        layers = int(cross.weight.shape[0])
        heads = int(cross.weight.shape[1])
        unique_copies = {
            "layer_shared": heads,
            "head_shared": layers,
            "shared": 1,
        }[config.cross_kv_weight_mode]
        unique_weight_shape = {
            "layer_shared": [heads, dim, dim],
            "head_shared": [layers, dim, dim],
            "shared": [dim, dim],
        }[config.cross_kv_weight_mode]
        unique_bias_shape = unique_weight_shape[:-2] + [dim]
        unique_cross_bytes = unique_copies * (
            dim * dim * cross.weight.element_size()
            + dim * cross.bias.element_size()
        )
        logical_parameter_bytes = (
            parameter_bytes - expanded_cross_bytes + unique_cross_bytes
        )
        cross_weight_sharing = {
            "mode": config.cross_kv_weight_mode,
            "unique_copies": unique_copies,
            "expanded_copies": layers * heads,
            "unique_weight_shape": unique_weight_shape,
            "unique_bias_shape": unique_bias_shape,
            "logical_bytes": unique_cross_bytes,
            "materialized_bytes": expanded_cross_bytes,
        }
    temporal.save(temporal_path)
    cross.save(cross_path)
    if refinement is not None:
        ls_temporal.save(ls_temporal_path)
        ls_cross.save(ls_cross_path)
    report = {
        "status": "fit_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trace": str(args.trace.resolve()),
        "input": {
            "shape": list(keys.shape),
            "keys_dtype": str(keys.dtype).removeprefix("torch."),
            "values_dtype": str(values.dtype).removeprefix("torch."),
            "layer_ids": list(layer_ids or ()),
            "trace_stride": trace_metadata.get("trace_stride"),
            "runtime_stride": config.stride,
        },
        "codec_config": asdict(config),
        "codec_config_sha256": config_sha256,
        "fit": {
            # Keep the old report key while exposing the clearer config name.
            "history_length": temporal.history_length,
            "span_frames": args.span_frames,
            "reachable_k_history_length": fit_config.k_history_length,
            "reachable_v_history_length": fit_config.v_history_length,
            "k_history_length": temporal.with_history_length(
                config.k_history_length
            ).history_length,
            "k_predictor_mode": config.k_predictor_mode,
            "iterations": int(args.iterations),
            "requested_iterations": int(args.iterations),
            "fit_history_source": args.fit_history_source,
            "effective_reconstruction_rounds": (
                int(args.iterations)
                if args.fit_history_source == "reconstructed"
                else 0
            ),
            "effective_regression_solves": (
                int(args.iterations)
                if args.fit_history_source == "reconstructed"
                else 1
            ),
            "v_regression_solves": (
                0
                if config.v_predictor_mode in FIXED_TEMPORAL_V_PREDICTOR_MODES
                else (
                    int(args.iterations)
                    if args.fit_history_source == "reconstructed"
                    else 1
                )
            ),
            "k_history_input": args.fit_history_source,
            "v_history_input": (
                None
                if config.v_predictor_mode in FIXED_TEMPORAL_V_PREDICTOR_MODES
                else args.fit_history_source
            ),
            "cross_k_input": (
                args.fit_history_source
                if config.v_predictor_mode == "cross_kv"
                else None
            ),
            "innovation_k_input": (
                args.fit_history_source
                if config.v_predictor_mode == "innovation"
                else None
            ),
            "temporal_ridge": float(temporal_ridge),
            "cross_ridge": float(cross_ridge),
            "v_history_ridge": float(v_history_ridge),
            "v_history_length": cross.with_history_length(
                config.v_history_length
            ).history_length,
            # The temporal V predictor has no dense/diagonal distinction; it
            # reads its own reconstructed V history instead of a Cross(K) map.
            "v_history_mode": (
                (
                    "fixed_" + config.v_predictor_mode
                    if config.v_predictor_mode in FIXED_TEMPORAL_V_PREDICTOR_MODES
                    else "temporal"
                )
                if temporal_v
                else cross.history_mode
            ),
            "v_predictor_fitting": (
                "fixed"
                if config.v_predictor_mode in FIXED_TEMPORAL_V_PREDICTOR_MODES
                else "regression"
            ),
            "max_gain": float(args.max_gain),
            "coordinate_system": coordinate_system,
            "attention_max_tokens": args.attention_max_tokens,
            "trace_metadata": trace_metadata,
            "device": str(device),
            "compute_dtype": (
                "float64_stats"
                if config.fit_mode == "legacy"
                else "float32" if device.type == "cuda" else "float64"
            ),
            "seconds": fit_seconds,
            # Keep the original name as an alias for allocated memory.
            "peak_gpu_bytes": peak_fit_allocated,
            "peak_gpu_allocated_bytes": peak_fit_allocated,
            "peak_gpu_reserved_bytes": peak_fit_reserved,
        },
        "attention_refinement": refinement,
        "parameters": {
            "v_predictor_mode": config.v_predictor_mode,
            "cross_kv_weight_mode": config.cross_kv_weight_mode,
            "temporal_weight_shape": list(temporal.history_weights.shape),
            "temporal_bias_shape": list(temporal.history_bias.shape),
            **(
                {
                    "v_temporal_weight_shape": list(cross.history_weights.shape),
                    "v_temporal_bias_shape": list(cross.history_bias.shape),
                }
                if temporal_v
                else {
                    (
                        "innovation_weight_shape"
                        if config.v_predictor_mode == "innovation"
                        else "cross_weight_shape"
                    ): list(cross.weight.shape),
                    (
                        "innovation_history_weight_shape"
                        if config.v_predictor_mode == "innovation"
                        else "cross_history_weight_shape"
                    ): list(cross.history_weights.shape),
                    (
                        "innovation_bias_shape"
                        if config.v_predictor_mode == "innovation"
                        else "cross_bias_shape"
                    ): list(cross.bias.shape),
                }
            ),
            "bytes": parameter_bytes,
            "mib": parameter_bytes / (1024**2),
            "logical_bytes": logical_parameter_bytes,
            "logical_mib": logical_parameter_bytes / (1024**2),
            "cross_weight_sharing": cross_weight_sharing,
        },
        "artifacts": {
            temporal_stem: temporal_path.name,
            v_artifact_stem: cross_path.name,
            **(
                {
                    f"{temporal_stem}_ls": ls_temporal_path.name,
                    f"{v_artifact_stem}_ls": ls_cross_path.name,
                }
                if refinement is not None
                else {}
            ),
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
