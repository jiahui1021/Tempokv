#!/usr/bin/env python3
"""Generate Self-Forcing videos with TempoKV or native bf16 KV caches."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.runtime import persistent_units_per_launch  # noqa: E402

RUNTIME_ROOT = PROJECT_ROOT / "experiments" / "Self-Forcing"
REQUIRED_RUNTIME_FILES = (
    "inference.py",
    "kv_cache.py",
    "pipeline/__init__.py",
    "pipeline/causal_inference.py",
    "utils/wan_wrapper.py",
    "wan/modules/causal_model.py",
)
EXPECTED_LAYER_IDS = tuple(range(30))
EXPECTED_COORDINATE_SYSTEM = "pre_rope_normalized"
VIDEO_FRAMES_PER_LATENT = 4


def latent_frames_for(video_frames: int) -> int:
    if video_frames < 1:
        raise ValueError("video frame count must be positive")
    shifted = video_frames + VIDEO_FRAMES_PER_LATENT - 1
    if shifted % (VIDEO_FRAMES_PER_LATENT * 3):
        raise ValueError(
            "Self-Forcing FRAMES must map exactly to 3-frame latent blocks; "
            "valid values satisfy (FRAMES + 3) % 12 == 0"
        )
    return shifted // VIDEO_FRAMES_PER_LATENT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Self-Forcing videos with TempoKV or bf16 KV caches."
    )
    parser.add_argument(
        "--mode", choices=("tempokv", "bf16"), default="tempokv"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--k-params", type=Path)
    parser.add_argument("--v-params", type=Path)
    parser.add_argument(
        "--k-history-length",
        type=int,
        choices=range(1, 9),
        help="number of reconstructed K frames used by the predictor",
    )
    parser.add_argument(
        "--v-history-length",
        type=int,
        choices=range(0, 6),
        help="number of reconstructed V frames used by the predictor",
    )
    # Overrides so one config per model covers every bit width and decoder,
    # instead of one committed config per combination.
    parser.add_argument(
        "--v-predictor-mode",
        choices=("cross_kv", "temporal"),
        help="predict V from reconstructed K (cross_kv) or from its own history",
    )
    parser.add_argument("--k-bits", type=int, choices=(2, 4, 8))
    parser.add_argument("--v-bits", type=int, choices=(2, 4, 8))
    parser.add_argument(
        "--torch-decoder",
        action="store_true",
        help="decode with the PyTorch reference path instead of Triton",
    )
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frames",
        type=int,
        default=717,
        help="final video frame count; converted to latent frames internally (1 latent = 4 video frames)",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--local-attn-size", type=int)
    parser.add_argument("--triton-cache", type=Path)
    parser.add_argument(
        "--allow-triton-fallback",
        action="store_true",
        help="use the config's Triton flags instead of enforcing strict acceleration",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="record a per-section CUDA-event runtime breakdown for this run",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="require a fresh, complete generation with no skipped samples",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_config(path: Path):
    from tempokv.config import config_from_mapping

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("codec config must be a JSON object")
    if "codec" in payload:
        if len(payload) != 1 or not isinstance(payload["codec"], dict):
            raise ValueError("'codec' must be the only top-level config object")
        payload = payload["codec"]
    return config_from_mapping(payload)


def _validate_parameters(
    k_path: Path, v_path: Path, config
) -> dict[str, object]:
    from tempokv.codec import TempoKVCodec
    from tempokv.parameters import CrossKVParameters, TemporalParameters

    temporal = TemporalParameters.load(k_path)
    temporal_v = config.v_predictor_mode == "temporal"
    cross = (
        TemporalParameters.load(v_path)
        if temporal_v
        else CrossKVParameters.load(v_path)
    )
    expected_signature = config.format_signature()
    if (
        temporal.codec_signature != expected_signature
        or cross.codec_signature != expected_signature
    ):
        raise ValueError(
            "Self-Forcing parameters were not fitted with the current codec config"
        )
    if (
        not temporal.parameter_set_id
        or not cross.parameter_set_id
        or temporal.parameter_set_id != cross.parameter_set_id
    ):
        raise ValueError(
            "Self-Forcing K/V parameters must share a current parameter_set_id"
        )
    codec = TempoKVCodec(temporal, cross, config)
    if config.k_predictor_mode == "shared":
        if temporal.layer_ids is not None:
            raise ValueError("legacy K parameters must be shared across layers")
    elif temporal.layer_ids != EXPECTED_LAYER_IDS:
        raise ValueError("K parameters must contain layers 0 through 29")
    # Both fitting families produce temporal V now: legacy pools every layer and
    # head into one coefficient set, native keeps one per layer and head.
    legacy_family = config.fit_mode == "legacy"
    if temporal_v and legacy_family:
        if cross.layer_ids is not None:
            raise ValueError(
                "legacy temporal V parameters must be shared across layers"
            )
    elif cross.layer_ids != EXPECTED_LAYER_IDS:
        raise ValueError("V parameters must contain layers 0 through 29")
    if temporal.stride != config.stride or cross.stride != config.stride:
        raise ValueError("parameter stride does not match the codec config")
    if (
        temporal.coordinate_system != EXPECTED_COORDINATE_SYSTEM
        or cross.coordinate_system != EXPECTED_COORDINATE_SYSTEM
    ):
        raise ValueError(
            "Self-Forcing parameters must use pre_rope_normalized coordinates"
        )
    if config.k_predictor_mode == "shared":
        expected_k_history_shape = (
            config.k_history_length,
            config.k_history_length,
            128,
        )
        expected_k_bias_shape = (config.k_history_length, 128)
        if (
            tuple(temporal.history_weights.shape) != expected_k_history_shape
            or tuple(temporal.history_bias.shape) != expected_k_bias_shape
        ):
            raise ValueError(
                "legacy K parameters must be shared across all layers "
                "and heads"
            )
    elif config.k_history_length is not None:
        expected_k_history_shape = (
            30,
            config.k_history_length,
            config.k_history_length,
            12,
            128,
        )
        if tuple(temporal.history_weights.shape) != expected_k_history_shape:
            raise ValueError(
                "K history parameters must match 30 layers, the configured history "
                "length, 12 heads, and dimension 128"
            )
    elif tuple(temporal.history_weights.shape[-2:]) != (12, 128):
        raise ValueError("K parameters must contain 12 heads with dimension 128")
    if temporal_v:
        order = config.v_history_length
        if legacy_family:
            # Legacy pools every layer and head, so the parameters carry
            # neither axis.
            expected_v_shape = (order, order, 128)
            expected_v_bias_shape = (order, 128)
        else:
            expected_v_shape = (30, order, order, 12, 128)
            expected_v_bias_shape = (30, order, 12, 128)
        if (
            tuple(cross.history_weights.shape) != expected_v_shape
            or tuple(cross.history_bias.shape) != expected_v_bias_shape
        ):
            raise ValueError(
                f"temporal V parameters must have shape {expected_v_shape} with "
                f"bias {expected_v_bias_shape}, got "
                f"{tuple(cross.history_weights.shape)} and "
                f"{tuple(cross.history_bias.shape)}"
            )
    elif tuple(cross.weight.shape[-3:]) != (12, 128, 128):
        raise ValueError("V parameters must contain 12 heads with 128x128 weights")
    elif config.v_history_length:
        expected_v_history_shape = (30, config.v_history_length, 12, 128)
        if tuple(cross.history_weights.shape) != expected_v_history_shape:
            raise ValueError(
                "Self-Forcing requires fast diagonal V history parameters matching "
                "30 layers, the configured history length, 12 heads, and dimension "
                "128; refit v_cross_kv.npz with the current fitter"
            )
    parameter_tensors = (
        (
            temporal.history_weights,
            temporal.history_bias,
            cross.history_weights,
            cross.history_bias,
        )
        if temporal_v
        else (
            temporal.history_weights,
            temporal.history_bias,
            cross.weight,
            cross.bias,
            cross.history_weights,
        )
    )
    predictor_parameter_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in parameter_tensors
    )
    v_history_parameter_bytes = (
        cross.history_weights.numel() * cross.history_weights.element_size()
    )
    return {
        "k_params": str(k_path),
        "v_params": str(v_path),
        "layers": 30,
        "heads": 12,
        "head_dimension": 128,
        "history_length": codec.temporal.history_length,
        "k_history_length": codec.temporal.history_length,
        "k_predictor_mode": config.k_predictor_mode,
        "v_history_length": codec.v_parameters.history_length,
        "v_history_mode": (
            "temporal"
            if temporal_v
            else cross.history_mode
        ),
        "v_history_diagonal": (
            False
            if temporal_v
            else cross.history_diagonal
        ),
        "predictor_parameter_bytes": predictor_parameter_bytes,
        "predictor_parameter_mib": predictor_parameter_bytes / (1024**2),
        "v_history_parameter_bytes": v_history_parameter_bytes,
        "v_history_parameter_mib": v_history_parameter_bytes / (1024**2),
        "stride": config.stride,
        "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
    }


def _generation_command(
    args: argparse.Namespace, paths: dict[str, Path]
) -> list[str]:
    latent_frames = latent_frames_for(args.frames)
    local_attn_size = (
        latent_frames
        if args.local_attn_size is None
        else args.local_attn_size
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.num_gpus}",
        str(Path(__file__).with_name("worker.py")),
        "--mode",
        args.mode,
    ]
    if args.profile:
        command.extend(("--profile", str(paths["profile"])))
    command += [
        "--",
        "--config_path",
        str(paths["model_config"]),
        "--checkpoint_path",
        str(paths["checkpoint"]),
        "--data_path",
        str(paths["prompts"]),
        "--output_folder",
        str(paths["videos"]),
        "--num_output_frames",
        str(latent_frames),
        "--num_samples",
        str(args.samples),
        "--seed",
        str(args.seed),
        "--start_index",
        str(args.start_index),
        "--end_index",
        str(args.end_index),
        "--save_with_index",
        "--use_ema",
        "--local_attn_size",
        str(local_attn_size),
    ]
    if args.skip_existing:
        command.append("--skip_existing")
    return command


def _load_worker_metrics(
    metrics_dir: Path, expected_ranks: int
) -> dict[str, object]:
    rank_metrics: list[dict[str, object]] = []
    for expected_rank in range(expected_ranks):
        path = metrics_dir / f"rank-{expected_rank}.json"
        if not path.is_file():
            raise RuntimeError(f"missing metrics for rank {expected_rank}: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise RuntimeError(
                f"invalid metrics for rank {expected_rank}: {path}"
            ) from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"metrics for rank {expected_rank} must be an object")
        required = {
            "rank",
            "wall_seconds",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "gpu_name",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise RuntimeError(
                f"metrics for rank {expected_rank} are missing: {', '.join(missing)}"
            )
        if payload["rank"] != expected_rank:
            raise RuntimeError(
                f"metrics rank mismatch in {path}: expected {expected_rank}"
            )
        try:
            wall_seconds = float(payload["wall_seconds"])
            peak_allocated = int(payload["peak_allocated_bytes"])
            peak_reserved = int(payload["peak_reserved_bytes"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"metrics for rank {expected_rank} contain invalid numbers"
            ) from error
        if wall_seconds < 0 or peak_allocated < 0 or peak_reserved < 0:
            raise RuntimeError(
                f"metrics for rank {expected_rank} must not be negative"
            )
        rank_metrics.append(
            {
                **payload,
                "wall_seconds": wall_seconds,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
            }
        )
    peak_allocated_bytes = max(
        metric["peak_allocated_bytes"] for metric in rank_metrics
    )
    peak_reserved_bytes = max(
        metric["peak_reserved_bytes"] for metric in rank_metrics
    )
    decode_routes: dict[str, int] = {}
    for metric in rank_metrics:
        routes = metric.get("decode_routes", {})
        if not isinstance(routes, dict):
            raise RuntimeError("worker decode_routes must be an object")
        for name, count in routes.items():
            if not isinstance(name, str) or not isinstance(count, int) or count < 0:
                raise RuntimeError("worker decode_routes contains an invalid entry")
            decode_routes[name] = decode_routes.get(name, 0) + count
    return {
        "max_worker_seconds": max(
            metric["wall_seconds"] for metric in rank_metrics
        ),
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_allocated_gib": peak_allocated_bytes / (1024**3),
        "peak_reserved_gib": peak_reserved_bytes / (1024**3),
        "decode_routes": decode_routes,
        "ranks": rank_metrics,
    }


def _load_worker_profiles(
    profile_dir: Path, expected_ranks: int
) -> list[dict[str, object]]:
    """Read the per-rank runtime breakdown written by an instrumented worker.

    A rank that produced no profile is an error rather than an empty entry: the
    Stage 0 decision gate compares section shares, and a silently missing rank
    would move every share.
    """
    profiles: list[dict[str, object]] = []
    for expected_rank in range(expected_ranks):
        path = profile_dir / f"rank-{expected_rank}.json"
        if not path.is_file():
            raise RuntimeError(
                f"missing runtime profile for rank {expected_rank}: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "sections" not in payload:
            raise RuntimeError(f"invalid runtime profile for rank {expected_rank}")
        profiles.append(payload)
    return profiles


def _video_summary_entries(videos_dir: Path) -> list[dict[str, object]]:
    summary_path = videos_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Self-Forcing did not write the current run summary")
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError("Self-Forcing run summary is invalid") from error
    entries: list[dict[str, object]] = []
    for rank in payload.get("rank_summaries") or []:
        if not isinstance(rank, dict):
            continue
        for video in rank.get("videos") or []:
            if isinstance(video, dict):
                entries.append(video)
    return entries


def _batch_summary_entries(videos_dir: Path) -> list[dict[str, object]]:
    """Read prompt-level timing records from the current run summary."""
    summary_path = videos_dir / "summary.json"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError("Self-Forcing run summary is invalid") from error
    entries: list[dict[str, object]] = []
    for rank in payload.get("rank_summaries") or []:
        if not isinstance(rank, dict):
            continue
        for batch in rank.get("batches") or []:
            if isinstance(batch, dict):
                entries.append(batch)
    return entries


def _end_to_end_latency(
    batch_entries: list[dict[str, object]],
    video_entries: list[dict[str, object]],
) -> float | None:
    """Average prompt-batch latency, excluding model load.

    New summaries measure one generation and one save interval per prompt. A
    legacy video-only summary is accepted as a compatibility fallback, with
    at most one sample counted for each prompt.
    """
    candidates: list[float] = []
    for batch in batch_entries:
        if int(batch.get("generated_samples", 0) or 0) <= 0:
            continue
        total = batch.get("total_seconds")
        if isinstance(total, (int, float)) and total > 0:
            candidates.append(float(total))
    if not candidates:
        seen_prompts: set[object] = set()
        for video in video_entries:
            if bool(video.get("skipped")):
                continue
            prompt_index = video.get("prompt_index")
            if prompt_index in seen_prompts:
                continue
            seen_prompts.add(prompt_index)
            total = video.get("total_seconds")
            if isinstance(total, (int, float)) and total > 0:
                candidates.append(float(total))
    if not candidates:
        return None
    return float(sum(candidates) / len(candidates))


def run(args: argparse.Namespace) -> dict[str, object]:
    inputs = {
        "checkpoint": args.checkpoint.expanduser().resolve(),
        "prompts": args.prompts.expanduser().resolve(),
    }
    model_config = args.model_config
    if model_config is None:
        model_config = RUNTIME_ROOT / "configs" / "self_forcing_dmd.yaml"
    paths = {
        **inputs,
        "model_config": model_config.expanduser().resolve(),
        "run": args.output_dir.expanduser().resolve(),
        "videos": args.output_dir.expanduser().resolve() / "videos",
        "profile": args.output_dir.expanduser().resolve() / "profile",
        "runtime_config": args.output_dir.expanduser().resolve()
        / "runtime_config.json",
    }
    if args.mode == "tempokv":
        missing_arguments = [
            name
            for name, value in (
                ("--config", args.config),
                ("--k-params", args.k_params),
                ("--v-params", args.v_params),
            )
            if value is None
        ]
        if missing_arguments:
            raise ValueError(
                f"mode=tempokv requires {', '.join(missing_arguments)}"
            )
        paths.update(
            {
                "config": args.config.expanduser().resolve(),
                "k_params": args.k_params.expanduser().resolve(),
                "v_params": args.v_params.expanduser().resolve(),
            }
        )
    if paths["run"] == Path(paths["run"].anchor):
        raise ValueError("output-dir must not be the filesystem root")
    required_files = ["checkpoint", "prompts", "model_config"]
    if args.mode == "tempokv":
        required_files.extend(("config", "k_params", "v_params"))
    for name in required_files:
        if not paths[name].is_file():
            raise ValueError(f"{name} file does not exist: {paths[name]}")
    for relative in REQUIRED_RUNTIME_FILES:
        if not (RUNTIME_ROOT / relative).is_file():
            raise ValueError(f"bundled Self-Forcing runtime is missing: {relative}")

    if args.samples < 1:
        raise ValueError("samples must be positive")
    if args.benchmark and args.skip_existing:
        raise ValueError("benchmark mode does not allow --skip-existing")
    if args.num_gpus < 1:
        raise ValueError("num-gpus must be positive")
    if args.local_attn_size is not None and (
        args.local_attn_size == 0 or args.local_attn_size < -1
    ):
        raise ValueError("local-attn-size must be -1 or a positive integer")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        visible_count = len(
            [item for item in visible_devices.split(",") if item.strip()]
        )
        if args.num_gpus > visible_count:
            raise ValueError(
                f"num-gpus={args.num_gpus} exceeds "
                f"CUDA_VISIBLE_DEVICES ({visible_count})"
            )
    latent_frames = latent_frames_for(args.frames)
    if (
        args.local_attn_size not in (None, -1)
        and args.local_attn_size < latent_frames
    ):
        raise ValueError(
            "local-attn-size must cover the complete generated latent sequence "
            f"({latent_frames} frames) so BF16 and TempoKV use identical "
            "Self-Forcing attention semantics"
        )
    config = None
    parameters = None
    if args.mode == "tempokv":
        config = _load_config(paths["config"])
        overrides = {
            name: value
            for name, value in (
                ("k_history_length", args.k_history_length),
                ("v_history_length", args.v_history_length),
                ("k_bits", args.k_bits),
                ("v_bits", args.v_bits),
                ("v_predictor_mode", args.v_predictor_mode),
                ("use_triton", False if args.torch_decoder else None),
            )
            if value is not None
        }
        if overrides:
            config = replace(config, **overrides)
        if config.stride != 1560:
            raise ValueError("Self-Forcing requires stride=1560")
        if 128 % config.group_size:
            raise ValueError("Self-Forcing group_size must divide 128")
        if config.reconstruction_dtype != "bfloat16":
            raise ValueError("Self-Forcing requires reconstruction_dtype=bfloat16")
        if latent_frames < 27:
            raise ValueError(
                "Self-Forcing needs at least 27 latent frames "
                "(about 105 video frames) to use TempoKV"
            )
        parameters = _validate_parameters(
            paths["k_params"], paths["v_params"], config
        )

    generation_command = _generation_command(args, paths)
    report: dict[str, object] = {
        "status": "dry_run" if args.dry_run else "running",
        "mode": args.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_command": generation_command,
        "output_dir": str(paths["run"]),
    }
    if args.mode == "tempokv":
        report["parameters"] = parameters
        report["runtime_config"] = str(paths["runtime_config"])
        requested_persistent_units = persistent_units_per_launch()
        report["persistent_units_per_launch"] = {
            "requested": requested_persistent_units,
            "k": (
                min(requested_persistent_units, 8)
                if parameters["k_history_length"] > 4
                else requested_persistent_units
            ),
            # Both Cross routes unroll a K projection per unit, so they cap the
            # launch batch where the reference implementation does.
            "v": (
                min(requested_persistent_units, 8)
                if config.v_predictor_mode == "cross_kv"
                else requested_persistent_units
            ),
            "v_backend": "triton",
        }
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    paths["run"].mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if args.mode == "tempokv":
        runtime_payload = asdict(config)
        if not args.allow_triton_fallback and not args.torch_decoder:
            runtime_payload["use_triton"] = True
            runtime_payload["strict_triton"] = True
        paths["runtime_config"].write_text(
            json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        environment.update(
            {
                "TEMPOKV_CONFIG_PATH": str(paths["runtime_config"]),
                "TEMPOKV_K_PARAMS_PATH": str(paths["k_params"]),
                "TEMPOKV_V_PARAMS_PATH": str(paths["v_params"]),
                "TEMPOKV_NUM_FRAMES": str(latent_frames_for(args.frames)),
            }
        )
        # The attention window normally spans every generated frame, because
        # TempoKV compresses the whole cache and no longer needs the model's
        # 21-frame default.  Overriding it back to a smaller window is what
        # makes a run comparable against a baseline that kept that default,
        # since the window decides how many quantized frames reach the output.
        attention_frames = os.environ.get("TEMPOKV_ATTENTION_FRAMES")
        if attention_frames:
            environment["TEMPOKV_NUM_FRAMES"] = str(int(attention_frames))
        if args.triton_cache is not None:
            environment["TEMPOKV_TRITON_CACHE"] = str(
                args.triton_cache.expanduser().resolve()
            )
    else:
        for name in tuple(environment):
            if name.startswith("TEMPOKV_"):
                environment.pop(name)
        if args.mode == "bf16":
            # The legacy runtime also rotates dense cached K in complex64.  Keep the
            # BF16 reference on the same arithmetic path as its quantized run.
            environment["SELF_FORCING_CACHE_ROPE_PRECISION"] = "complex64"
    paths["videos"].mkdir(parents=True, exist_ok=True)
    metrics_dir = paths["run"] / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(args.num_gpus):
        (metrics_dir / f"rank-{rank}.json").unlink(missing_ok=True)
    environment["SELF_FORCING_METRICS_DIR"] = str(metrics_dir)
    if args.profile:
        paths["profile"].mkdir(parents=True, exist_ok=True)
        for rank in range(args.num_gpus):
            (paths["profile"] / f"rank-{rank}.json").unlink(missing_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        generation_command,
        cwd=RUNTIME_ROOT,
        env=environment,
        check=False,
    )
    generation_seconds = time.perf_counter() - started
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode, generation_command
        )
    performance = _load_worker_metrics(metrics_dir, args.num_gpus)
    if args.profile:
        report["runtime_profile"] = _load_worker_profiles(
            paths["profile"], args.num_gpus
        )
    video_entries = _video_summary_entries(paths["videos"])
    batch_entries = _batch_summary_entries(paths["videos"])
    if not video_entries:
        raise RuntimeError("Self-Forcing current run contains no requested videos")
    videos: list[Path] = []
    for entry in video_entries:
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            raise RuntimeError("Self-Forcing run summary contains no video path")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Self-Forcing video is missing or empty: {path}")
        videos.append(path)
    generated_entries = [
        entry for entry in video_entries if not bool(entry.get("skipped"))
    ]
    requested_samples = len(video_entries)
    generated_samples = len(generated_entries)
    skipped_samples = requested_samples - generated_samples
    benchmark_valid = generated_samples > 0 and skipped_samples == 0
    if args.benchmark and not benchmark_valid:
        raise RuntimeError(
            "benchmark run must generate every requested sample without skips"
        )
    if (
        args.mode == "tempokv"
        and not args.allow_triton_fallback
        and not args.torch_decoder
        and generated_entries
    ):
        required_routes = (
            ("temporal_persistent_direct",)
            if config.v_predictor_mode == "temporal"
            else ("temporal_persistent_direct", "cross_kv_persistent_direct")
        )
        missing_routes = [
            route
            for route in required_routes
            if int(performance["decode_routes"].get(route, 0)) <= 0
        ]
        if missing_routes:
            raise RuntimeError(
                "TempoKV direct persistent Triton decode was not used: "
                + ", ".join(missing_routes)
            )
        # "declined:" entries are not routes; each names the gate that sent a
        # span to the Torch decoder, so report them as the reason alongside
        # whichever fallback route actually ran.
        declines = {
            name.split(":", 1)[1]: int(count)
            for name, count in performance["decode_routes"].items()
            if name.startswith("declined:") and int(count) > 0
        }
        report["triton_declines"] = declines
        unexpected_routes = {
            name: int(count)
            for name, count in performance["decode_routes"].items()
            if int(count) > 0
            and name not in required_routes
            and not name.startswith("declined:")
        }
        if unexpected_routes:
            reason = (
                "; declined because: "
                + ", ".join(
                    f"{name}={count}" for name, count in sorted(declines.items())
                )
                if declines
                else ""
            )
            raise RuntimeError(
                "TempoKV used a non-direct or non-persistent decode route: "
                + ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(unexpected_routes.items())
                )
                + reason
            )
    end_to_end = _end_to_end_latency(batch_entries, video_entries)
    if end_to_end is not None:
        generation_seconds = end_to_end
    elif generated_samples == 0:
        generation_seconds = None
    status = "complete" if generated_samples else "all_skipped"
    report.update(
        {
            "status": status,
            "generation_seconds": generation_seconds,
            "timing_scope": "prompt_batch",
            "requested_samples": requested_samples,
            "generated_samples": generated_samples,
            "skipped_samples": skipped_samples,
            "benchmark_valid": benchmark_valid,
            "performance": performance,
            "videos": [str(path) for path in videos],
        }
    )
    (paths["run"] / "pipeline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
