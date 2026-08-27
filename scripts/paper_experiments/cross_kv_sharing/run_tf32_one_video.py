#!/usr/bin/env python3
"""Fit layer_shared+TF32, generate one video, and compare saved latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _report_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    return resolved / "pipeline_report.json" if resolved.is_dir() else resolved


def _profile_total(report: dict[str, Any], section_name: str) -> float:
    total = 0.0
    found = False
    for profile in report.get("runtime_profile", []) or []:
        for section in profile.get("sections", []) or []:
            if section.get("name") == section_name:
                total += float(section["total_ms"])
                found = True
    if not found or total <= 0.0:
        raise ValueError(f"pipeline report has no positive {section_name} profile")
    return total


def _route_count(report: dict[str, Any]) -> int:
    routes = report.get("performance", {}).get("decode_routes", {})
    return sum(
        int(count)
        for name, count in routes.items()
        if name.startswith("cross_kv_persistent")
    )


def _validate_report(report: dict[str, Any], label: str) -> None:
    if report.get("benchmark_valid") is not True:
        raise ValueError(f"{label} is not a fresh benchmark run")
    if int(report.get("generated_samples", 0)) != 1:
        raise ValueError(f"{label} must contain exactly one generated video")
    if float(report.get("generation_seconds") or 0.0) <= 0.0:
        raise ValueError(f"{label} has no positive generation_seconds")
    _profile_total(report, "decode_v")


def _validate_codec_identity(
    report: dict[str, Any], label: str, expected_precision: str
) -> None:
    runtime_config = report.get("runtime_config")
    if not isinstance(runtime_config, str):
        raise ValueError(f"{label} does not record runtime_config")
    config = _read_json(Path(runtime_config))
    if config.get("cross_kv_weight_mode") != "layer_shared":
        raise ValueError(f"{label} is not a layer_shared run")
    if config.get("cross_matmul_precision") != expected_precision:
        raise ValueError(
            f"{label} must use cross_matmul_precision={expected_precision}"
        )


def _command_option(command: list[str], name: str) -> str | None:
    return command[command.index(name) + 1] if name in command else None


def _command_assignment(command: list[str], prefix: str) -> str | None:
    return next((item for item in command if item.startswith(prefix)), None)


def _validate_fairness(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> None:
    before = baseline.get("generation_command")
    after = candidate.get("generation_command")
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("both reports must record generation_command")
    shared_options = (
        "--checkpoint_path",
        "--data_path",
        "--num_output_frames",
        "--num_samples",
        "--seed",
        "--start_index",
        "--end_index",
        "--config_path",
        "--local_attn_size",
        "--sink_size",
    )
    mismatches = [
        name
        for name in shared_options
        if _command_option(before, name) != _command_option(after, name)
    ]
    if mismatches:
        raise ValueError(
            "baseline and TF32 candidate use different generation settings: "
            + ", ".join(mismatches)
        )
    if _command_assignment(before, "--nproc_per_node=") != _command_assignment(
        after, "--nproc_per_node="
    ):
        raise ValueError("baseline and TF32 candidate use different GPU counts")
    if baseline.get("span_frames") != candidate.get("span_frames"):
        raise ValueError("baseline and TF32 candidate use different span_frames")


def compare_reports(
    baseline_path: Path, candidate_path: Path
) -> dict[str, Any]:
    baseline = _read_json(_report_path(baseline_path))
    candidate = _read_json(_report_path(candidate_path))
    _validate_report(baseline, "layer_shared baseline")
    _validate_report(candidate, "layer_shared_tf32 candidate")
    _validate_codec_identity(baseline, "layer_shared baseline", "ieee")
    _validate_codec_identity(candidate, "layer_shared_tf32 candidate", "tf32")
    _validate_fairness(baseline, candidate)

    baseline_e2e = float(baseline["generation_seconds"])
    candidate_e2e = float(candidate["generation_seconds"])
    baseline_decode = _profile_total(baseline, "decode_v")
    candidate_decode = _profile_total(candidate, "decode_v")
    return {
        "schema_version": 1,
        "baseline": "layer_shared_ieee",
        "candidate": "layer_shared_tf32",
        "videos_generated_by_candidate": 1,
        "generation_seconds": {
            "baseline": baseline_e2e,
            "candidate": candidate_e2e,
            "speedup": baseline_e2e / candidate_e2e,
            "reduced": candidate_e2e < baseline_e2e,
        },
        "decode_v_ms": {
            "baseline": baseline_decode,
            "candidate": candidate_decode,
            "speedup": baseline_decode / candidate_decode,
            "reduced": candidate_decode < baseline_decode,
        },
        "cross_kv_triton_calls": {
            "baseline": _route_count(baseline),
            "candidate": _route_count(candidate),
        },
        "latency_reduced": candidate_e2e < baseline_e2e,
        "baseline_report": str(_report_path(baseline_path)),
        "candidate_report": str(_report_path(candidate_path)),
    }


def _candidate_config(base_config: Path, output: Path) -> Path:
    payload = _read_json(base_config)
    codec = payload.get("codec", payload)
    if not isinstance(codec, dict):
        raise ValueError("base config must contain a codec object")
    if codec.get("fit_mode", "legacy") != "legacy":
        raise ValueError("base config must use fit_mode=legacy")
    candidate = {
        **codec,
        "v_predictor_mode": "cross_kv",
        "cross_kv_weight_mode": "layer_shared",
        "cross_matmul_precision": "tf32",
    }
    path = output / "config" / "layer_shared_tf32.json"
    _write_json(path, {"codec": candidate})
    return path


def _fit_command(args: argparse.Namespace, config: Path, params: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/common/fit_params.py"),
        str(args.calibration_trace.resolve()),
        "--config",
        str(config.resolve()),
        "--output-dir",
        str(params.resolve()),
        "--fit-history-source",
        "reconstructed",
        "--v-history-mode",
        "diagonal",
        "--device",
        args.fit_device,
    ]


def _generation_command(
    args: argparse.Namespace, config: Path, params: Path, run: Path
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/Self-Forcing/tempokv_adapter/generate.py"),
        "--mode",
        "tempokv",
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--prompts",
        str(args.prompts.resolve()),
        "--output-dir",
        str(run.resolve()),
        "--config",
        str(config.resolve()),
        "--k-params",
        str((params / "k_temporal.npz").resolve()),
        "--v-params",
        str((params / "v_cross_kv.npz").resolve()),
        "--frames",
        str(args.frames),
        "--samples",
        "1",
        "--seed",
        str(args.seed),
        "--start-index",
        str(args.prompt_index),
        "--end-index",
        str(args.prompt_index + 1),
        "--num-gpus",
        str(args.num_gpus),
        "--benchmark",
        "--profile",
    ]
    if args.model_config is not None:
        command.extend(("--model-config", str(args.model_config.resolve())))
    if args.local_attn_size is not None:
        command.extend(("--local-attn-size", str(args.local_attn_size)))
    if args.span_frames is not None:
        command.extend(("--span-frames", str(args.span_frames)))
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=717)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--fit-device", default="auto")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--local-attn-size", type=int)
    parser.add_argument("--span-frames", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prompt_index < 0 or args.num_gpus < 1:
        raise ValueError("prompt-index must be non-negative and num-gpus positive")
    output = args.output.resolve()
    config = _candidate_config(args.base_config.resolve(), output)
    params = output / "params"
    run = output / "run"
    fit_command = _fit_command(args, config, params)
    generation_command = _generation_command(args, config, params, run)
    if args.dry_run:
        _write_json(
            output / "plan.json",
            {
                "videos_to_generate": 1,
                "method": "layer_shared_tf32",
                "fit_command": fit_command,
                "generation_command": generation_command,
            },
        )
        return 0

    subprocess.run(fit_command, cwd=PROJECT_ROOT, check=True)
    subprocess.run(generation_command, cwd=PROJECT_ROOT, check=True)
    candidate_report = run / "pipeline_report.json"
    comparison = compare_reports(args.baseline_report, candidate_report)
    _write_json(output / "latency_comparison.json", comparison)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
