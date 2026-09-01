#!/usr/bin/env python3
"""Fit TempoKV parameters and launch an integrated video generator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
FIT_SCRIPT = ROOT / "scripts" / "common" / "fit_params.py"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit predictor parameters, pass them to a TempoKV-integrated model, "
            "and verify that the model produced videos."
        )
    )
    parser.add_argument("--trace", type=Path, help="clean K/V .npz trace")
    parser.add_argument("--config", type=Path, required=True, help="codec JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--history-length",
        type=int,
        help="legacy K history override; prefer k_history_length in the config",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--temporal-ridge", type=float, default=0.1)
    parser.add_argument("--cross-ridge", type=float, default=0.1)
    parser.add_argument("--max-gain", type=float, default=1.0)
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--k-params", type=Path)
    parser.add_argument("--v-params", type=Path)
    parser.add_argument("--triton-cache", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "generator",
        nargs=argparse.REMAINDER,
        help="generator command after '--'; supports {config}, {k_params}, "
        "{v_params}, {video_dir}, and {run_dir}",
    )
    return parser


def _resolve_generator(tokens: list[str], paths: dict[str, Path]) -> list[str]:
    if tokens and tokens[0] == "--":
        tokens = tokens[1:]
    if not tokens:
        raise ValueError("a video generator command is required after '--'")
    replacements = {name: str(path) for name, path in paths.items()}
    try:
        return [token.format(**replacements) for token in tokens]
    except KeyError as error:
        raise ValueError(f"unknown generator placeholder: {error.args[0]}") from error


def _run(command: list[str], *, env: dict[str, str] | None = None) -> float:
    resolved_env = os.environ.copy() if env is None else env.copy()
    source_root = str(ROOT / "src")
    existing_pythonpath = resolved_env.get("PYTHONPATH")
    resolved_env["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else source_root + os.pathsep + existing_pythonpath
    )
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, env=resolved_env, check=True)
    return time.perf_counter() - started


def _video_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def _v_parameter_filename(config_path: Path) -> str:
    """Return the V artifact name selected by a codec config."""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "codec" in payload:
        payload = payload["codec"]
    mode = payload.get("v_predictor_mode", "cross_kv")
    if mode in {"temporal", "identity", "taylor", "taylor2", "direct"}:
        return "v_temporal.npz"
    if mode in {"cross_kv_identity", "cross_kv_taylor", "cross_kv_taylor2"}:
        return "v_cross_kv.npz"
    if mode == "innovation":
        return "v_innovation.npz"
    return "v_cross_kv.npz"


def run(args: argparse.Namespace) -> dict[str, object]:
    config = args.config.resolve()
    if not config.is_file():
        raise ValueError(f"config file does not exist: {config}")
    run_dir = args.output_dir.resolve()
    parameter_dir = run_dir / "parameters"
    video_dir = run_dir / "videos"

    if args.skip_fit:
        if args.trace is not None:
            raise ValueError("--trace cannot be combined with --skip-fit")
        if args.k_params is None or args.v_params is None:
            raise ValueError("--skip-fit requires --k-params and --v-params")
        k_params = args.k_params.resolve()
        v_params = args.v_params.resolve()
    else:
        if args.trace is None:
            raise ValueError("--trace is required unless --skip-fit is used")
        if args.k_params is not None or args.v_params is not None:
            raise ValueError("--k-params/--v-params require --skip-fit")
        trace = args.trace.resolve()
        if not trace.is_file():
            raise ValueError(f"trace file does not exist: {trace}")
        k_params = parameter_dir / "k_temporal.npz"
        v_params = parameter_dir / _v_parameter_filename(config)

    for path in (k_params, v_params):
        if args.skip_fit and not path.is_file():
            raise ValueError(f"parameter file does not exist: {path}")

    paths = {
        "config": config,
        "k_params": k_params,
        "v_params": v_params,
        "video_dir": video_dir,
        "run_dir": run_dir,
    }
    generator = _resolve_generator(args.generator, paths)
    fit_command = None
    if not args.skip_fit:
        fit_command = [
            sys.executable,
            str(FIT_SCRIPT),
            str(args.trace.resolve()),
            "--config",
            str(config),
            "--output-dir",
            str(parameter_dir),
            "--iterations",
            str(args.iterations),
            "--temporal-ridge",
            str(args.temporal_ridge),
            "--cross-ridge",
            str(args.cross_ridge),
            "--max-gain",
            str(args.max_gain),
        ]
        if args.history_length is not None:
            fit_command.extend(("--history-length", str(args.history_length)))

    report: dict[str, object] = {
        "status": "dry_run" if args.dry_run else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config),
        "fit_command": fit_command,
        "generator_command": generator,
        "parameters": {"k": str(k_params), "v": str(v_params)},
        "video_dir": str(video_dir),
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    run_dir.mkdir(parents=True, exist_ok=True)
    durations: dict[str, float] = {}
    if fit_command is not None:
        durations["fit_seconds"] = _run(fit_command)

    video_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "TEMPOKV_CONFIG_PATH": str(config),
            "TEMPOKV_K_PARAMS_PATH": str(k_params),
            "TEMPOKV_V_PARAMS_PATH": str(v_params),
            "TEMPOKV_VIDEO_OUTPUT_DIR": str(video_dir),
        }
    )
    if args.triton_cache is not None:
        environment["TEMPOKV_TRITON_CACHE"] = str(args.triton_cache.resolve())
    durations["generation_seconds"] = _run(generator, env=environment)

    videos = _video_files(video_dir)
    if not videos:
        raise RuntimeError(f"generator completed but produced no video under {video_dir}")
    report.update(
        {
            "status": "complete",
            "durations": durations,
            "videos": [str(path) for path in videos],
        }
    )
    (run_dir / "pipeline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
