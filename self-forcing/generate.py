#!/usr/bin/env python3
"""Generate Self-Forcing videos with TempoKV or native bf16 KV caches."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

REQUIRED_MODEL_FILES = (
    "experiments/Self-Forcing/inference.py",
    "experiments/Self-Forcing/pipeline/__init__.py",
    "experiments/Self-Forcing/pipeline/causal_inference.py",
    "experiments/Self-Forcing/utils/wan_wrapper.py",
    "experiments/Self-Forcing/wan/modules/causal_model.py",
)
EXPECTED_LAYER_IDS = tuple(range(30))
EXPECTED_COORDINATE_SYSTEM = "pre_rope_normalized"
VIDEO_FRAMES_PER_LATENT = 4


def latent_frames_for(video_frames: int) -> int:
    return (video_frames + VIDEO_FRAMES_PER_LATENT - 1) // VIDEO_FRAMES_PER_LATENT


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
    parser.add_argument("--model-repo", type=Path, required=True)
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
    parser.add_argument("--local-attn-size", type=int, default=-1)
    parser.add_argument("--triton-cache", type=Path)
    parser.add_argument(
        "--allow-triton-fallback",
        action="store_true",
        help="use the config's Triton flags instead of enforcing strict acceleration",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_config(path: Path):
    from tempokv.config import CodecConfig

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("codec config must be a JSON object")
    if "codec" in payload:
        if len(payload) != 1 or not isinstance(payload["codec"], dict):
            raise ValueError("'codec' must be the only top-level config object")
        payload = payload["codec"]
    return CodecConfig(**payload)


def _validate_parameters(
    k_path: Path, v_path: Path, config
) -> dict[str, object]:
    from tempokv.codec import TempoKVCodec
    from tempokv.parameters import CrossKVParameters, TemporalParameters

    temporal = TemporalParameters.load(k_path)
    cross = CrossKVParameters.load(v_path)
    TempoKVCodec(temporal, cross, config)
    if temporal.layer_ids != EXPECTED_LAYER_IDS:
        raise ValueError("K parameters must contain layers 0 through 29")
    if cross.layer_ids != EXPECTED_LAYER_IDS:
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
    if config.k_history_length is not None:
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
    if tuple(cross.weight.shape[-3:]) != (12, 128, 128):
        raise ValueError("V parameters must contain 12 heads with 128x128 weights")
    expected_v_history_shape = (30, config.v_history_length, 12, 128, 128)
    if tuple(cross.history_weights.shape) != expected_v_history_shape:
        raise ValueError(
            "V history parameters must match 30 layers, the configured history "
            "length, 12 heads, and 128x128 weights"
        )
    return {
        "k_params": str(k_path),
        "v_params": str(v_path),
        "layers": 30,
        "heads": 12,
        "head_dimension": 128,
        "history_length": temporal.history_length,
        "k_history_length": temporal.history_length,
        "v_history_length": cross.history_length,
        "stride": config.stride,
        "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
    }


def _generation_command(
    args: argparse.Namespace, paths: dict[str, Path]
) -> list[str]:
    latent_frames = latent_frames_for(args.frames)
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.num_gpus}",
        str(Path(__file__).with_name("worker.py")),
        "--mode",
        args.mode,
        "--model-repo",
        str(paths["model_repo"]),
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
        str(args.local_attn_size),
        "--quant_type",
        "none",
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


def _end_to_end_latency(videos_dir: Path) -> float | None:
    """Read Self-Forcing's authoritative per-video timing from summary.json.

    `videos[i].total_seconds` covers generation + video save and excludes
    model loading, checkpoint loading, and Triton compilation, so it is the
    meaningful end-to-end generation latency.
    """
    summary_path = videos_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    candidates: list[float] = []
    for rank in payload.get("rank_summaries") or []:
        for video in rank.get("videos") or []:
            total = video.get("total_seconds")
            if isinstance(total, (int, float)) and total > 0:
                candidates.append(float(total))
    if not candidates:
        return None
    return float(sum(candidates) / len(candidates))


def run(args: argparse.Namespace) -> dict[str, object]:
    inputs = {
        "model_repo": args.model_repo.expanduser().resolve(),
        "checkpoint": args.checkpoint.expanduser().resolve(),
        "prompts": args.prompts.expanduser().resolve(),
    }
    model_config = args.model_config
    if model_config is None:
        model_config = (
            inputs["model_repo"]
            / "experiments"
            / "Self-Forcing"
            / "configs"
            / "self_forcing_dmd.yaml"
        )
    paths = {
        **inputs,
        "model_config": model_config.expanduser().resolve(),
        "run": args.output_dir.expanduser().resolve(),
        "videos": args.output_dir.expanduser().resolve() / "videos",
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
    if not paths["model_repo"].is_dir():
        raise ValueError(f"model repository does not exist: {paths['model_repo']}")
    for relative in REQUIRED_MODEL_FILES:
        if not (paths["model_repo"] / relative).is_file():
            raise ValueError(f"model repository is missing: {relative}")

    if args.samples < 1:
        raise ValueError("samples must be positive")
    if args.num_gpus < 1:
        raise ValueError("num-gpus must be positive")
    config = None
    parameters = None
    if args.mode == "tempokv":
        config = _load_config(paths["config"])
        if config.stride != 1560:
            raise ValueError("Self-Forcing requires stride=1560")
        if 128 % config.group_size:
            raise ValueError("Self-Forcing group_size must divide 128")
        if config.reconstruction_dtype != "bfloat16":
            raise ValueError("Self-Forcing requires reconstruction_dtype=bfloat16")
        if latent_frames_for(args.frames) < 27:
            raise ValueError(
                "Self-Forcing needs at least 108 video frames (27 latent frames) "
                "to use TempoKV"
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
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    paths["run"].mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if args.mode == "tempokv":
        runtime_payload = asdict(config)
        if not args.allow_triton_fallback:
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
        if args.triton_cache is not None:
            environment["TEMPOKV_TRITON_CACHE"] = str(
                args.triton_cache.expanduser().resolve()
            )
    else:
        for name in tuple(environment):
            if name.startswith("TEMPOKV_"):
                environment.pop(name)
    paths["videos"].mkdir(parents=True, exist_ok=True)
    metrics_dir = paths["run"] / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    environment["SELF_FORCING_METRICS_DIR"] = str(metrics_dir)
    started = time.perf_counter()
    completed = subprocess.run(
        generation_command,
        cwd=paths["model_repo"],
        env=environment,
        check=False,
    )
    generation_seconds = time.perf_counter() - started
    performance = _load_worker_metrics(metrics_dir, args.num_gpus)
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode, generation_command
        )
    if (
        args.mode == "tempokv"
        and not args.allow_triton_fallback
        and not args.skip_existing
    ):
        required_routes = (
            "temporal_persistent_direct",
            "cross_kv_persistent_direct",
        )
        missing_routes = [
            route
            for route in required_routes
            if performance["decode_routes"].get(route, 0) <= 0
        ]
        if missing_routes:
            raise RuntimeError(
                "TempoKV direct decode fast path was not used: "
                + ", ".join(missing_routes)
            )
    videos = sorted(paths["videos"].rglob("*.mp4"))
    if not videos or any(path.stat().st_size == 0 for path in videos):
        raise RuntimeError("Self-Forcing completed without a non-empty MP4")
    end_to_end = _end_to_end_latency(paths["videos"])
    if end_to_end is not None:
        generation_seconds = end_to_end
    report.update(
        {
            "status": "complete",
            "generation_seconds": generation_seconds,
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
