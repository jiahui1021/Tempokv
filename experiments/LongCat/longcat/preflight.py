#!/usr/bin/env python3
"""Check the bundled LongCat runtime before loading a large checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


REQUIRED_CHECKPOINT_SUBDIRECTORIES = (
    "tokenizer",
    "text_encoder",
    "vae",
    "scheduler",
    "dit",
)
REQUIRED_LONGCAT_FILES = (
    "experiments/LongCat/inference.py",
    "experiments/LongCat/LICENSE",
    "experiments/LongCat/longcat_video/modules/longcat_video_dit.py",
    "experiments/LongCat/longcat_video/modules/attention.py",
    "experiments/LongCat/longcat_video/modules/autoencoder_kl_wan.py",
    "experiments/LongCat/longcat_video/modules/scheduling_flow_match_euler_discrete.py",
    "experiments/LongCat/longcat_video/pipeline_longcat_video.py",
    "experiments/LongCat/longcat_video/context_parallel/context_parallel_util.py",
)
LONGCAT_SOURCE_CONTRACT = {
    "experiments/LongCat/longcat_video/modules/longcat_video_dit.py": (
        "class LongCatVideoTransformer3DModel",
        "def quantize_kv_cache",
        "_onload_kv_cache",
    ),
    "experiments/LongCat/longcat_video/modules/attention.py": (
        "class Attention",
        "def forward_with_kv_cache",
        "uncompress_kv_cache",
    ),
    "experiments/LongCat/longcat_video/pipeline_longcat_video.py": (
        "def _cache_clean_latents",
        "def _get_kv_cache_dict",
    ),
}
REQUIRED_MODULES = (
    "torch",
    "triton",
    "numpy",
    "PIL",
    "tqdm",
    "einops",
    "diffusers",
    "transformers",
    "safetensors",
    "ftfy",
    "regex",
    "loguru",
    "av",
    "torchvision",
    "flash_attn",
)
EXPECTED_LONGCAT_MODEL = {
    "depth": 48,
    "num_heads": 32,
    "hidden_size": 4096,
    "head_dim": 128,
}


def _check(ok: bool, detail: str, **extra: object) -> dict[str, object]:
    return {"ok": bool(ok), "detail": detail, **extra}


def _module_check(name: str) -> dict[str, object]:
    available = importlib.util.find_spec(name) is not None
    return _check(available, "available" if available else "module is not installed")


def _read_codec_config(path: Path) -> tuple[dict[str, Any] | None, dict[str, object]]:
    if not path.is_file():
        return None, _check(False, f"file does not exist: {path}", path=str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, _check(False, f"cannot read JSON: {error}", path=str(path))
    if not isinstance(payload, dict):
        return None, _check(False, "config must contain a JSON object", path=str(path))
    if "codec" in payload:
        if len(payload) != 1 or not isinstance(payload.get("codec"), dict):
            return None, _check(
                False,
                "when present, config.codec must be the only top-level object",
                path=str(path),
            )
        payload = payload["codec"]
    if not isinstance(payload, dict):
        return None, _check(False, "config.codec must contain a JSON object", path=str(path))
    try:
        from tempokv.config import CodecConfig

        config = CodecConfig(**payload)
    except (ImportError, TypeError, ValueError) as error:
        return None, _check(False, f"invalid TempoKV codec configuration: {error}", path=str(path))
    # CodecConfig only rejects a non-positive stride, so a float or bool would
    # survive it and then fail much later as a signature mismatch.
    declared_stride = payload.get("stride")
    if declared_stride is not None and (
        isinstance(declared_stride, bool) or not isinstance(declared_stride, int)
    ):
        return None, _check(
            False, "config stride must be an integer", path=str(path)
        )
    # Report the resolved settings rather than the literal file: a config only
    # states what gives it an identity, so anything it omits comes from a
    # CodecConfig default and that default is what the runtime uses.
    resolved = {
        **payload,
        "stride": int(config.stride),
        "v_predictor_mode": config.v_predictor_mode,
    }
    return resolved, _check(
        True,
        "valid codec configuration",
        path=str(path),
        stride=int(config.stride),
    )


def _longcat_runtime_check(project_root: Path) -> dict[str, object]:
    missing_files = [
        relative
        for relative in REQUIRED_LONGCAT_FILES
        if not (project_root / relative).is_file()
    ]
    missing_symbols: dict[str, list[str]] = {}
    for relative, symbols in LONGCAT_SOURCE_CONTRACT.items():
        path = project_root / relative
        if not path.is_file():
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        absent = [symbol for symbol in symbols if symbol not in source]
        if absent:
            missing_symbols[relative] = absent
    ok = not missing_files and not missing_symbols
    return _check(
        ok,
        "LongCat runtime hooks found"
        if ok
        else "LongCat runtime source is incomplete or has changed",
        project_root=str(project_root),
        missing_files=missing_files,
        missing_symbols=missing_symbols,
    )


def _model_shape_check(checkpoint: Path) -> dict[str, object]:
    config_path = checkpoint / "dit" / "config.json"
    if not config_path.is_file():
        return _check(
            False,
            f"missing model config: {config_path}",
            path=str(config_path),
        )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return _check(False, f"cannot read model config: {error}", path=str(config_path))
    if not isinstance(payload, dict):
        return _check(False, "model config must contain a JSON object", path=str(config_path))
    actual = {
        "depth": payload.get("depth"),
        "num_heads": payload.get("num_heads"),
        "hidden_size": payload.get("hidden_size"),
    }
    if (
        all(isinstance(value, int) and not isinstance(value, bool) for value in actual.values())
        and actual["num_heads"] > 0
        and actual["hidden_size"] > 0
        and actual["hidden_size"] % actual["num_heads"] == 0
    ):
        actual["head_dim"] = actual["hidden_size"] // actual["num_heads"]
    else:
        actual["head_dim"] = None
    matches = actual == EXPECTED_LONGCAT_MODEL
    return _check(
        matches,
        "LongCat model shape is 48 layers / 32 heads / 128 head dim"
        if matches
        else f"expected {EXPECTED_LONGCAT_MODEL}, got {actual}",
        path=str(config_path),
        expected=EXPECTED_LONGCAT_MODEL,
        actual=actual,
    )


def _parameter_file_check(
    path: Path | None,
    *,
    expected_kind: str,
    expected_stride: int | None,
) -> dict[str, object]:
    if path is None:
        return _check(False, "parameter path was not supplied")
    if not path.is_file():
        return _check(False, "parameter file does not exist", path=str(path))
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            if "kind" not in data:
                return _check(False, "parameter file is missing kind metadata", path=str(path))
            kind = str(np.asarray(data["kind"]).item())
            if kind != expected_kind:
                return _check(
                    False,
                    f"expected {expected_kind} parameters, got {kind}",
                    path=str(path),
                )
            if "stride" not in data:
                return _check(False, "parameter file is missing stride metadata", path=str(path))
            stride = int(np.asarray(data["stride"]).item())
            if expected_stride is not None and stride != expected_stride:
                return _check(
                    False,
                    f"parameter stride {stride} does not match expected {expected_stride}",
                    path=str(path),
                    stride=stride,
                    expected_stride=expected_stride,
                )
            if "coordinate_system" not in data:
                return _check(
                    False,
                    "parameter file is missing coordinate_system metadata",
                    path=str(path),
                )
            coordinate_system = str(np.asarray(data["coordinate_system"]).item())
            if coordinate_system != "pre_rope_normalized":
                return _check(
                    False,
                    "LongCat parameters must use pre_rope_normalized coordinates",
                    path=str(path),
                    coordinate_system=coordinate_system,
                )
            history_mode = None
            if "history_mode" in data:
                history_mode = str(np.asarray(data["history_mode"]).item())
            if "layer_ids" in data and data["layer_ids"].size:
                layer_ids_array = np.asarray(data["layer_ids"])
                if layer_ids_array.ndim != 1:
                    return _check(
                        False,
                        "parameter layer_ids must be one-dimensional",
                        path=str(path),
                    )
                layer_ids = [int(item) for item in layer_ids_array]
                expected_layers = list(range(EXPECTED_LONGCAT_MODEL["depth"]))
                if layer_ids != expected_layers:
                    return _check(
                        False,
                        f"parameter layer_ids must be {expected_layers[0]} through "
                        f"{expected_layers[-1]}",
                        path=str(path),
                        layer_ids=layer_ids,
                    )
            if expected_kind == "temporal":
                weights_name = "history_weights"
                if weights_name not in data or "history_bias" not in data:
                    return _check(
                        False,
                        "temporal parameter file is missing weight or bias arrays",
                        path=str(path),
                    )
                weights = data[weights_name]
                bias = data["history_bias"]
                if weights.ndim == 5:
                    shape_ok = (
                        weights.shape[0] == EXPECTED_LONGCAT_MODEL["depth"]
                        and weights.shape[-2:] == (
                            EXPECTED_LONGCAT_MODEL["num_heads"],
                            EXPECTED_LONGCAT_MODEL["head_dim"],
                        )
                        and bias.shape
                        == (
                            weights.shape[0],
                            weights.shape[1],
                            weights.shape[-2],
                            weights.shape[-1],
                        )
                    )
                elif weights.ndim == 4:
                    shape_ok = (
                        weights.shape[-2:] == (
                            EXPECTED_LONGCAT_MODEL["num_heads"],
                            EXPECTED_LONGCAT_MODEL["head_dim"],
                        )
                        and bias.shape
                        == (weights.shape[0], weights.shape[-2], weights.shape[-1])
                    )
                elif weights.ndim in (2, 3):
                    shape_ok = (
                        weights.shape[-2:] == (
                            EXPECTED_LONGCAT_MODEL["num_heads"],
                            EXPECTED_LONGCAT_MODEL["head_dim"],
                        )
                        and bias.shape == weights.shape
                    )
                else:
                    shape_ok = False
                if history_mode is not None:
                    shape_ok = shape_ok and (
                        history_mode == "shared"
                        or (history_mode == "independent" and weights.ndim == 5)
                    )
                if weights.ndim == 2 and "layer_ids" in data and data["layer_ids"].size:
                    shape_ok = False
            else:
                if "weight" not in data or "bias" not in data:
                    return _check(
                        False,
                        "CrossKV parameter file is missing weight or bias arrays",
                        path=str(path),
                    )
                weight = data["weight"]
                bias = data["bias"]
                shape_ok = (
                    weight.ndim in (3, 4)
                    and weight.shape[-3] == EXPECTED_LONGCAT_MODEL["num_heads"]
                    and weight.shape[-2:] == (
                        EXPECTED_LONGCAT_MODEL["head_dim"],
                        EXPECTED_LONGCAT_MODEL["head_dim"],
                    )
                    and bias.shape == weight.shape[:-2] + (weight.shape[-1],)
                )
                if history_mode is not None:
                    shape_ok = shape_ok and (
                        history_mode in ("shared_diagonal", "shared")
                        or (history_mode in ("dense", "diagonal") and weight.ndim == 4)
                    )
                if weight.ndim == 3 and "layer_ids" in data and data["layer_ids"].size:
                    shape_ok = False
            if not shape_ok:
                return _check(
                    False,
                    "parameter arrays do not match LongCat 32-head / 128-dim layout",
                    path=str(path),
                )
    except (ImportError, OSError, ValueError, TypeError, KeyError, EOFError) as error:
        return _check(False, f"cannot read parameter file: {error}", path=str(path))
    return _check(
        True,
        "parameter metadata and shape match LongCat",
        path=str(path),
        kind=expected_kind,
        stride=stride,
        coordinate_system=coordinate_system,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    checks: dict[str, object] = {}
    checks["longcat_runtime"] = _longcat_runtime_check(PROJECT_ROOT)

    checkpoint = args.checkpoint_dir.expanduser().resolve()
    missing_subdirectories = [
        name for name in REQUIRED_CHECKPOINT_SUBDIRECTORIES if not (checkpoint / name).is_dir()
    ]
    checks["checkpoint"] = _check(
        checkpoint.is_dir() and not missing_subdirectories,
        "required LongCat subdirectories found"
        if checkpoint.is_dir() and not missing_subdirectories
        else "missing checkpoint subdirectories: " + ", ".join(missing_subdirectories),
        path=str(checkpoint),
        missing_subdirectories=missing_subdirectories,
    )
    checks["model_shape"] = _model_shape_check(checkpoint)

    module_checks = {name: _module_check(name) for name in REQUIRED_MODULES}
    checks["python_modules"] = {
        "ok": all(item["ok"] for item in module_checks.values()),
        "modules": module_checks,
    }

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        checks["cuda"] = _check(
            cuda_available,
            "CUDA is available" if cuda_available else "CUDA is unavailable",
            torch_version=torch.__version__,
        )
    except ImportError as error:
        checks["cuda"] = _check(False, f"PyTorch is unavailable: {error}")

    codec_config: dict[str, Any] | None = None
    if args.mode == "tempokv":
        required_paths = {
            "config": args.config,
            "k_parameters": args.k_parameters,
            "v_parameters": args.v_parameters,
        }
        checks["tempokv_files"] = {
            name: _check(path is not None and path.is_file(),
                         "file found" if path is not None and path.is_file() else "file does not exist",
                         path=str(path) if path is not None else None)
            for name, path in required_paths.items()
        }
        if args.config is not None:
            codec_config, config_check = _read_codec_config(args.config)
            checks["codec_config"] = config_check
        # The parameter files must have been fitted for the stride this run will
        # use, which is the override when one was given.
        expected_stride = args.stride or (
            int(codec_config["stride"]) if codec_config is not None else None
        )
        resolved_v_predictor = args.v_predictor_mode or (
            codec_config.get("v_predictor_mode")
            if codec_config is not None
            else None
        )
        expected_v_kind = (
            "temporal" if resolved_v_predictor == "temporal" else "cross_kv"
        )
        parameter_checks = {
            "k_parameters": _parameter_file_check(
                args.k_parameters,
                expected_kind="temporal",
                expected_stride=expected_stride,
            ),
            "v_parameters": _parameter_file_check(
                args.v_parameters,
                expected_kind=expected_v_kind,
                expected_stride=expected_stride,
            ),
        }
        checks["parameter_files"] = parameter_checks
    elif args.stride is not None:
        checks["stride"] = _check(args.stride > 0, "stride is positive", stride=args.stride)

    failed = []
    for name, value in checks.items():
        if name == "python_modules":
            if not value["ok"]:
                failed.append(name)
        elif name == "tempokv_files":
            if not all(item["ok"] for item in value.values()):
                failed.append(name)
        elif name == "parameter_files":
            if not all(item["ok"] for item in value.values()):
                failed.append(name)
        elif isinstance(value, dict) and not value.get("ok", True):
            failed.append(name)
    return {
        "status": "ready" if not failed else "blocked",
        "mode": args.mode,
        "context_parallel_size": args.context_parallel_size,
        "checks": checks,
        "failed_checks": failed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check LongCat and optional TempoKV prerequisites without loading the model."
    )
    parser.add_argument(
        "--mode", choices=("tempokv", "bf16", "trace"), default="tempokv"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--k-parameters", type=Path)
    parser.add_argument("--v-parameters", type=Path)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--stride", type=int)
    # The V predictor decides which kind of parameter file to expect, and it
    # is a command-line override, so preflight has to be told about it too.
    parser.add_argument(
        "--v-predictor-mode", choices=("cross_kv", "temporal")
    )
    args = parser.parse_args(argv)
    if args.context_parallel_size <= 0:
        parser.error("--context-parallel-size must be positive")
    if args.stride is not None and args.stride <= 0:
        parser.error("--stride must be positive")
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
