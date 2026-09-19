#!/usr/bin/env python3
"""Generation-quality half of the K->V direction study (Self-Forcing).

Fits and generates one Self-Forcing run per arm against a shared BF16
reference, holding the trace, prompts, seed, frame count, bit widths and
history lengths fixed so the predictor direction is the only difference.

    CHECKPOINT=/models/self_forcing_dmd.pt \\
    WAN_MODEL_DIR=/models/Wan2.1-T2V-1.3B \\
    python scripts/paper_experiments/direction/run_quality.py \\
        --checkpoint $CHECKPOINT --wan-model-dir $WAN_MODEL_DIR \\
        --prompts assets/moviegen128.txt \\
        --trace data/self-forcing/trace/clean_trace.npz \\
        --output-root results/direction/quality

Set ``--dry-run`` to record the full command matrix without fitting or
generating; that is the way to check the wiring before spending GPU time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from protocol import ARMS, arm as get_arm  # noqa: E402


FIT_SCRIPT = PROJECT_ROOT / "scripts" / "Self-Forcing" / "fit_params.sh"
GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "Self-Forcing" / "generate.sh"
QUALITY_SCRIPT = PROJECT_ROOT / "scripts" / "evaluation" / "evaluate_quality.py"

# The metrics evaluate_quality.py reports; recorded by name so a missing one is
# visible in the summary instead of silently absent.
QUALITY_METRICS = ("psnr", "ssim", "lpips", "fid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _command_record(command, environment: dict[str, str]) -> dict[str, Any]:
    return {"command": list(command), "environment": dict(sorted(environment.items()))}


def _run(command, environment: dict[str, str], log: Path, *, dry_run: bool) -> int:
    printable = " ".join(f"{k}={v}" for k, v in sorted(environment.items()))
    print(f"\n$ {printable} {' '.join(str(item) for item in command)}", flush=True)
    if dry_run:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [str(item) for item in command],
            env={**os.environ, **environment},
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return completed.returncode


def _complete_generation(run_dir: Path) -> bool:
    videos = run_dir / "videos"
    return videos.is_dir() and any(videos.iterdir())


def _paths(root: Path, name: str) -> dict[str, Path]:
    return {
        "params": root / "params" / name,
        "run": root / "runs" / name,
        "fit_log": root / "logs" / "fit" / f"{name}.log",
        "generation_log": root / "logs" / "generation" / f"{name}.log",
        "quality_log": root / "logs" / "quality" / f"{name}.log",
        "quality": root / "quality" / f"{name}.json",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wan-model-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/legacy.json"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=[item.name for item in ARMS])
    parser.add_argument("--frames", type=int, default=105)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--k-history-length", type=int, default=2)
    parser.add_argument("--v-history-length", type=int, default=1)
    parser.add_argument("--k-bits", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--v-bits", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--fit-device", default="auto")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--quality-device", default="auto")
    parser.add_argument("--max-quality-frames", type=int)
    parser.add_argument("--local-attn-size", type=int)
    parser.add_argument(
        "--include-bf16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--quality", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--quality-fid", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.k_bits != args.v_bits:
        # The mirrored arm exchanges the tensor roles wholesale, so unequal
        # widths would move bits between K and V and confound the comparison.
        raise ValueError(
            "the direction study requires --k-bits == --v-bits so the mirrored "
            "arm does not also reallocate bit width"
        )
    if args.quality and not args.include_bf16:
        raise ValueError("quality comparison needs the BF16 reference")
    if args.end_index <= args.start_index:
        raise ValueError("--end-index must be greater than --start-index")
    for name in args.arms:
        get_arm(name)
    required = (args.checkpoint, args.wan_model_dir, args.prompts, args.trace, args.config)
    missing = [str(path) for path in required if not Path(path).expanduser().exists()]
    if missing:
        raise FileNotFoundError("missing inputs: " + ", ".join(missing))


def _generation_env(args: argparse.Namespace) -> dict[str, str]:
    environment = {
        "CHECKPOINT": str(args.checkpoint.expanduser().resolve()),
        "WAN_MODEL_DIR": str(args.wan_model_dir.expanduser().resolve()),
        "PROMPTS": str(args.prompts.expanduser().resolve()),
        "FRAMES": str(args.frames),
        "SEED": str(args.seed),
        "START_INDEX": str(args.start_index),
        "END_INDEX": str(args.end_index),
        "NUM_SAMPLES": str(args.samples),
        "OVERWRITE": "1",
        "DRY_RUN": "0",
    }
    if args.model_config is not None:
        environment["MODEL_CONFIG"] = str(args.model_config.expanduser().resolve())
    if args.local_attn_size is not None:
        environment["LOCAL_ATTN_SIZE"] = str(args.local_attn_size)
    return environment


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate(args)
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    arms = [get_arm(name) for name in args.arms]
    generation_env = _generation_env(args)
    commands: list[dict[str, Any]] = []
    trace = args.trace.expanduser().resolve()
    prompts = args.prompts.expanduser().resolve()

    manifest = {
        "schema": "tempokv.direction_quality.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if args.dry_run else "running",
        # Everything below is held identical across arms on purpose; the
        # predictor direction is the only thing allowed to move.
        "fairness_controls": {
            "same_trace": str(trace),
            "trace_sha256": None if args.dry_run else _sha256(trace),
            "same_prompts": str(prompts),
            "prompts_sha256": _sha256(prompts),
            "same_seed": args.seed,
            "same_frames": args.frames,
            "same_samples": args.samples,
            "same_prompt_range": [args.start_index, args.end_index],
            "same_k_bits": args.k_bits,
            "same_v_bits": args.v_bits,
            "same_k_history_length": args.k_history_length,
            "same_v_history_length": args.v_history_length,
            "same_cross_weight_mode": "layer_shared",
        },
        "include_bf16": args.include_bf16,
        "quality": {
            "enabled": args.quality,
            "fid_enabled": args.quality_fid,
            "max_frames_per_video": args.max_quality_frames,
            "reference": str(root / "runs/bf16/videos"),
        },
        "arms": [asdict(item) for item in arms],
        "commands": commands,
    }
    manifest_path = root / "manifest.json"
    if args.resume and manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing is None or existing.get("schema") != manifest["schema"]:
            raise ValueError(
                f"cannot resume invalid direction manifest: {manifest_path}"
            )
        existing_controls = existing.get("fairness_controls") or {}
        existing_bits = (
            existing_controls.get("same_k_bits"),
            existing_controls.get("same_v_bits"),
        )
        if existing_bits != (args.k_bits, args.v_bits):
            raise ValueError(
                "cannot resume direction study with different K/V bit widths: "
                f"existing K{existing_bits[0]}/V{existing_bits[1]}, "
                f"requested K{args.k_bits}/V{args.v_bits}"
            )
    elif args.resume and any(root.iterdir()):
        raise ValueError(
            f"cannot resume without the existing manifest: {manifest_path}"
        )
    _write_json(manifest_path, manifest)

    failure: str | None = None
    bf16_run = root / "runs" / "bf16"
    if args.include_bf16 and not (args.resume and _complete_generation(bf16_run)):
        environment = {**generation_env, "MODE": "bf16", "OUTPUT_DIR": str(bf16_run)}
        command = ("bash", str(GENERATE_SCRIPT))
        commands.append(_command_record(command, environment))
        code = _run(
            command,
            environment,
            root / "logs" / "generation" / "bf16.log",
            dry_run=args.dry_run,
        )
        if code:
            failure = f"bf16 generation failed with exit code {code}"

    for item in arms:
        if failure is not None:
            break
        paths = _paths(root, item.name)
        shared = {
            "CONFIG": str(args.config.expanduser().resolve()),
            "KH": str(args.k_history_length),
            "VH": str(args.v_history_length),
            "K_BITS": str(args.k_bits),
            "V_BITS": str(args.v_bits),
            "V_PREDICTOR": item.v_predictor_mode,
            "DIRECTION": item.predictor_direction,
        }
        if not (args.resume and (paths["params"] / "fit_report.json").is_file()):
            fit_environment = {
                **shared,
                "TRACE_FILE": str(trace),
                "OUTPUT_DIR": str(paths["params"]),
                "FIT_DEVICE": args.fit_device,
                "ITERATIONS": str(args.iterations),
                "FIT_HISTORY_SOURCE": "reconstructed",
                "OVERWRITE": "1",
                "DRY_RUN": "0",
            }
            command = ("bash", str(FIT_SCRIPT))
            commands.append(_command_record(command, fit_environment))
            code = _run(command, fit_environment, paths["fit_log"], dry_run=args.dry_run)
            if code:
                failure = f"{item.name} fitting failed with exit code {code}"
                break

        if not (args.resume and _complete_generation(paths["run"])):
            environment = {
                **generation_env,
                **shared,
                "MODE": "tempokv",
                "PARAM_DIR": str(paths["params"]),
                "OUTPUT_DIR": str(paths["run"]),
            }
            command = ("bash", str(GENERATE_SCRIPT))
            commands.append(_command_record(command, environment))
            code = _run(
                command, environment, paths["generation_log"], dry_run=args.dry_run
            )
            if code:
                failure = f"{item.name} generation failed with exit code {code}"
                break

    quality: dict[str, dict[str, Any]] = {}
    if args.quality and failure is None:
        reference = bf16_run / "videos"
        if not args.dry_run and not reference.is_dir():
            for item in arms:
                quality[item.name] = {
                    "status": "not_run",
                    "reason": f"BF16 reference directory is missing: {reference}",
                }
        else:
            for item in arms:
                paths = _paths(root, item.name)
                if not args.dry_run and args.resume and paths["quality"].is_file():
                    quality[item.name] = {"status": "complete"}
                    continue
                command = [
                    str(Path(sys.executable).resolve()),
                    str(QUALITY_SCRIPT),
                    str(reference),
                    str(paths["run"] / "videos"),
                    "--output",
                    str(paths["quality"]),
                    "--device",
                    args.quality_device,
                    "--allow-missing",
                ]
                if not args.quality_fid:
                    command.append("--no-fid")
                if args.max_quality_frames is not None:
                    command.extend(("--max-frames", str(args.max_quality_frames)))
                commands.append(_command_record(command, {}))
                code = _run(command, {}, paths["quality_log"], dry_run=args.dry_run)
                if args.dry_run:
                    quality[item.name] = {"status": "dry_run"}
                elif code == 0 and paths["quality"].is_file():
                    quality[item.name] = {"status": "complete"}
                else:
                    quality[item.name] = {
                        "status": "failed",
                        "reason": f"exit code {code}",
                    }

    results = []
    for item in arms:
        paths = _paths(root, item.name)
        report = _read_json(paths["quality"])
        # Never invent a metric value: an arm whose evaluation did not run is
        # recorded as not_run rather than as a missing key.
        metrics = (
            {name: report.get(name) for name in QUALITY_METRICS}
            if report is not None
            else None
        )
        results.append(
            {
                "arm": item.name,
                "v_predictor_mode": item.v_predictor_mode,
                "predictor_direction": item.predictor_direction,
                "interpretation": item.interpretation,
                "quality_status": quality.get(item.name, {}).get("status", "not_run"),
                "quality_metrics": metrics,
                "artifacts": {
                    "parameters": str(paths["params"]),
                    "videos": str(paths["run"] / "videos"),
                    "quality_report": str(paths["quality"]),
                },
            }
        )

    manifest["status"] = (
        "dry_run" if args.dry_run else "failed" if failure else "complete"
    )
    if failure:
        manifest["failure"] = failure
    _write_json(manifest_path, manifest)
    summary = {
        "schema": "tempokv.direction_quality.v1",
        "status": manifest["status"],
        "failure": failure,
        "reference": str(bf16_run / "videos"),
        "arms": results,
    }
    _write_json(root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"\nstatus: {summary['status']}")
    for item in summary["arms"]:
        metrics = item["quality_metrics"]
        rendered = (
            "  ".join(
                f"{name}={value:.4f}" if isinstance(value, (int, float)) else f"{name}=-"
                for name, value in metrics.items()
            )
            if metrics
            else f"({item['quality_status']})"
        )
        print(f"  {item['arm']:<16}{rendered}")
    if summary["failure"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
