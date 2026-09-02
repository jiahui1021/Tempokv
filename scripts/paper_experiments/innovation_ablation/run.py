#!/usr/bin/env python3
"""Run a small paired Self-Forcing V-predictor comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIT_SCRIPT = PROJECT_ROOT / "scripts" / "Self-Forcing" / "fit_params.sh"
GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "Self-Forcing" / "generate.sh"
QUALITY_SCRIPT = PROJECT_ROOT / "scripts" / "evaluation" / "evaluate_quality.py"


@dataclass(frozen=True)
class Variant:
    name: str
    v_predictor_mode: str
    v_history_length: int
    interpretation: str

    @property
    def v_artifact(self) -> str:
        return {
            "temporal": "v_temporal.npz",
            "cross_kv": "v_cross_kv.npz",
            "innovation": "v_innovation.npz",
        }[self.v_predictor_mode]


VARIANTS = (
    Variant("temporal_vh1", "temporal", 1, "only temporal"),
    Variant("cross_temporal_vh1", "cross_kv", 1, "cross + temporal"),
    Variant(
        "innovation_temporal_vh1",
        "innovation",
        1,
        "innovation + temporal",
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wan-model-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/legacy.json")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results/self_forcing/innovation_ablation_105f",
    )
    parser.add_argument("--frames", type=int, default=105)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--k-history-length", type=int, default=2)
    parser.add_argument("--k-bits", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--v-bits", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--fit-device", default="auto")
    parser.add_argument("--quality-device", default="auto")
    parser.add_argument("--max-quality-frames", type=int)
    parser.add_argument("--local-attn-size", type=int)
    parser.add_argument(
        "--include-bf16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--quality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-fid", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _command_record(command: Sequence[str], environment: dict[str, str]) -> dict[str, Any]:
    return {
        "command": list(command),
        "environment": dict(sorted(environment.items())),
        "rendered": shlex.join(("env", *(f"{k}={v}" for k, v in environment.items()), *command)),
    }


def _run_command(
    command: Sequence[str],
    environment: dict[str, str],
    log_path: Path,
    *,
    dry_run: bool,
) -> int:
    print(_command_record(command, environment)["rendered"])
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    merged.update(environment)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            env=merged,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode


def _validate(args: argparse.Namespace) -> None:
    if args.frames < 105 or (args.frames + 3) % 12:
        raise ValueError(
            "--frames must be at least 105 and satisfy (frames + 3) % 12 == 0"
        )
    if args.samples < 1 or args.iterations < 1:
        raise ValueError("--samples and --iterations must be positive")
    if not 1 <= args.k_history_length <= 8:
        raise ValueError("--k-history-length must be in [1, 8]")
    if args.max_quality_frames is not None and args.max_quality_frames < 1:
        raise ValueError("--max-quality-frames must be positive")
    output_root = args.output_root.expanduser().resolve()
    if output_root == Path(output_root.anchor):
        raise ValueError("--output-root must not be the filesystem root")
    if args.quality and not args.include_bf16:
        raise ValueError("quality comparison requires --include-bf16")
    if args.dry_run:
        return
    required_files = (args.checkpoint, args.prompts, args.trace, args.config)
    if args.model_config is not None:
        required_files += (args.model_config,)
    for path in required_files:
        if not path.expanduser().is_file():
            raise ValueError(f"required file does not exist: {path}")
    if not args.wan_model_dir.expanduser().is_dir():
        raise ValueError(f"Wan model directory does not exist: {args.wan_model_dir}")


def _base_generation_env(args: argparse.Namespace) -> dict[str, str]:
    environment = {
        "CHECKPOINT": str(args.checkpoint.expanduser().resolve()),
        "WAN_MODEL_DIR": str(args.wan_model_dir.expanduser().resolve()),
        "PROMPTS": str(args.prompts.expanduser().resolve()),
        "FRAMES": str(args.frames),
        "NUM_SAMPLES": str(args.samples),
        "SEED": str(args.seed),
        "START_INDEX": str(args.start_index),
        "END_INDEX": str(args.end_index),
        "NUM_GPUS": "1",
        "SKIP_EXISTING": "0",
        "BENCHMARK": "1",
        "DRY_RUN": "0",
    }
    if args.model_config is not None:
        environment["MODEL_CONFIG"] = str(args.model_config.expanduser().resolve())
    if args.local_attn_size is not None:
        environment["LOCAL_ATTN_SIZE"] = str(args.local_attn_size)
    return environment


def _paths(root: Path, variant: Variant) -> dict[str, Path]:
    return {
        "params": root / "params" / variant.name,
        "run": root / "runs" / variant.name,
        "fit_log": root / "logs" / "fit" / f"{variant.name}.log",
        "generation_log": root / "logs" / "generation" / f"{variant.name}.log",
        "quality_log": root / "logs" / "quality" / f"{variant.name}.log",
        "quality": root / "quality" / f"{variant.name}.json",
    }


def _complete_generation(path: Path) -> bool:
    report = _read_json(path / "pipeline_report.json")
    return bool(report and report.get("status") == "complete")


def _variant_summary(root: Path, variant: Variant, *, dry_run: bool) -> dict[str, Any]:
    paths = _paths(root, variant)
    fit = _read_json(paths["params"] / "fit_report.json")
    generation = _read_json(paths["run"] / "pipeline_report.json")
    quality = _read_json(paths["quality"])
    performance = (generation or {}).get("performance") or {}
    quality_metrics = None
    quality_status = "not_run"
    if quality is not None:
        quality_status = "complete"
        quality_metrics = {
            key: quality.get(key)
            for key in ("videos", "frames", "psnr_db", "ssim", "lpips", "fid")
        }
    return {
        **asdict(variant),
        "status": (
            "dry_run"
            if dry_run
            else (generation or {}).get("status", "not_run")
        ),
        "fit_status": (fit or {}).get("status", "not_run"),
        "generation_seconds": (generation or {}).get("generation_seconds"),
        "peak_allocated_bytes": performance.get("peak_allocated_bytes"),
        "peak_reserved_bytes": performance.get("peak_reserved_bytes"),
        "predictor_parameter_bytes": ((generation or {}).get("parameters") or {}).get(
            "predictor_parameter_bytes"
        ),
        "quality_status": quality_status,
        "quality_metrics": quality_metrics,
        "paths": {
            "params": str(paths["params"]),
            "generation_report": str(paths["run"] / "pipeline_report.json"),
            "quality_report": str(paths["quality"]),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate(args)
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    generation_env = _base_generation_env(args)
    commands: list[dict[str, Any]] = []
    manifest = {
        "schema": "tempokv.innovation_ablation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if args.dry_run else "running",
        "fairness_controls": {
            "same_trace": str(args.trace.expanduser().resolve()),
            "trace_sha256": _sha256(args.trace.expanduser().resolve()),
            "same_prompts": str(args.prompts.expanduser().resolve()),
            "prompts_sha256": _sha256(args.prompts.expanduser().resolve()),
            "same_seed": args.seed,
            "same_frames": args.frames,
            "same_samples": args.samples,
            "same_prompt_range": [args.start_index, args.end_index],
            "same_k_bits": args.k_bits,
            "same_v_bits": args.v_bits,
            "same_k_history_length": args.k_history_length,
            "same_v_history_length": 1,
            "same_cross_weight_mode": "layer_shared",
        },
        "include_bf16": args.include_bf16,
        "quality": {
            "enabled": args.quality,
            "fid_enabled": args.quality_fid,
            "max_frames_per_video": args.max_quality_frames,
            "reference": str(root / "runs/bf16/videos"),
        },
        "variants": [asdict(variant) for variant in VARIANTS],
        "commands": commands,
    }
    _write_json(root / "manifest.json", manifest)

    failure: str | None = None
    bf16_run = root / "runs/bf16"
    if args.include_bf16 and not (args.resume and _complete_generation(bf16_run)):
        environment = {**generation_env, "MODE": "bf16", "OUTPUT_DIR": str(bf16_run)}
        command = ("bash", str(GENERATE_SCRIPT))
        commands.append(_command_record(command, environment))
        code = _run_command(
            command,
            environment,
            root / "logs/generation/bf16.log",
            dry_run=args.dry_run,
        )
        if code:
            failure = f"bf16 generation failed with exit code {code}"

    for variant in VARIANTS:
        if failure is not None:
            break
        paths = _paths(root, variant)
        if not (args.resume and (paths["params"] / "fit_report.json").is_file()):
            fit_environment = {
                "CONFIG": str(args.config.expanduser().resolve()),
                "TRACE_FILE": str(args.trace.expanduser().resolve()),
                "OUTPUT_DIR": str(paths["params"]),
                "KH": str(args.k_history_length),
                "VH": str(variant.v_history_length),
                "K_BITS": str(args.k_bits),
                "V_BITS": str(args.v_bits),
                "V_PREDICTOR": variant.v_predictor_mode,
                "FIT_DEVICE": args.fit_device,
                "ITERATIONS": str(args.iterations),
                "FIT_HISTORY_SOURCE": "reconstructed",
                "OVERWRITE": "0",
                "DRY_RUN": "0",
            }
            command = ("bash", str(FIT_SCRIPT))
            commands.append(_command_record(command, fit_environment))
            code = _run_command(
                command, fit_environment, paths["fit_log"], dry_run=args.dry_run
            )
            if code:
                failure = f"{variant.name} fitting failed with exit code {code}"
                break

        if not (args.resume and _complete_generation(paths["run"])):
            environment = {
                **generation_env,
                "MODE": "tempokv",
                "CONFIG": str(args.config.expanduser().resolve()),
                "PARAM_DIR": str(paths["params"]),
                "OUTPUT_DIR": str(paths["run"]),
                "KH": str(args.k_history_length),
                "VH": str(variant.v_history_length),
                "K_BITS": str(args.k_bits),
                "V_BITS": str(args.v_bits),
                "V_PREDICTOR": variant.v_predictor_mode,
            }
            command = ("bash", str(GENERATE_SCRIPT))
            commands.append(_command_record(command, environment))
            code = _run_command(
                command,
                environment,
                paths["generation_log"],
                dry_run=args.dry_run,
            )
            if code:
                failure = f"{variant.name} generation failed with exit code {code}"
                break

    quality_results: dict[str, dict[str, str]] = {}
    if args.quality and failure is None:
        reference = bf16_run / "videos"
        if not args.dry_run and not reference.is_dir():
            for variant in VARIANTS:
                quality_results[variant.name] = {
                    "status": "not_run",
                    "reason": f"BF16 reference directory is missing: {reference}",
                }
        else:
            for variant in VARIANTS:
                paths = _paths(root, variant)
                if not args.dry_run and args.resume and paths["quality"].is_file():
                    quality_results[variant.name] = {"status": "complete"}
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
                code = _run_command(
                    command, {}, paths["quality_log"], dry_run=args.dry_run
                )
                quality_results[variant.name] = (
                    {"status": "dry_run"}
                    if args.dry_run
                    else {"status": "complete"}
                    if code == 0 and paths["quality"].is_file()
                    else {
                        "status": "not_run",
                        "reason": (
                            "quality dependencies, BF16 pairing, or video decoding "
                            f"were unavailable; see {paths['quality_log']}"
                        ),
                    }
                )

    rows = [_variant_summary(root, variant, dry_run=args.dry_run) for variant in VARIANTS]
    for row in rows:
        quality_result = quality_results.get(row["name"])
        if quality_result is not None and quality_result["status"] != "complete":
            row["quality_status"] = "not_run"
            row["quality_reason"] = quality_result["reason"]
    summary = {
        "schema": "tempokv.innovation_ablation.summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if failure else "dry_run" if args.dry_run else "complete",
        "failure": failure,
        "manifest": str(root / "manifest.json"),
        "bf16": {
            "included": args.include_bf16,
            "generation_report": str(bf16_run / "pipeline_report.json"),
            "status": (
                "dry_run"
                if args.dry_run and args.include_bf16
                else (_read_json(bf16_run / "pipeline_report.json") or {}).get(
                    "status", "not_run"
                )
            ),
        },
        "variants": rows,
    }
    manifest["status"] = summary["status"]
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failure:
        raise RuntimeError(failure)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
