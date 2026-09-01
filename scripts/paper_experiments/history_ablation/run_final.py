#!/usr/bin/env python3
"""Run the final four-GPU Self-Forcing K/V history ablation.

The runner keeps the experiment axes orthogonal:

* K axis: ``h_K=0..5`` with CrossKV V history fixed at ``h_V=1``;
* V axis: CrossKV ``h_V=0,1,2,4,5`` with K history fixed at ``h_K=3``;
* no-Cross control: temporal-only V with ``h_V=1`` and ``h_K=3``.

The shared ``h_K=3, h_V=1`` CrossKV run is used by both the K and V tables,
so the same video directory is never generated twice.  Four visible GPUs are
required; each subprocess is restricted to one GPU and the runner schedules
jobs in groups of four.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.config import CodecConfig, config_from_mapping  # noqa: E402


FIT_SCRIPT = PROJECT_ROOT / "scripts" / "Self-Forcing" / "fit_params.sh"
RUN_TEMPOKV = PROJECT_ROOT / "scripts" / "Self-Forcing" / "run_tempokv.sh"
RUN_BF16 = PROJECT_ROOT / "scripts" / "Self-Forcing" / "run_bf16.sh"
CACHE_EVALUATOR = (
    PROJECT_ROOT / "scripts" / "paper_experiments" / "history_ablation" / "evaluate_cache.py"
)
QUALITY_EVALUATOR = PROJECT_ROOT / "scripts" / "evaluation" / "evaluate_quality.py"

K_HISTORY_VALUES = (0, 1, 2, 3, 4, 5)
V_HISTORY_VALUES = (0, 1, 2, 4, 5)
SUMMARY_FIELDS = (
    "name",
    "axis",
    "k_history_length",
    "k_predictor_mode",
    "v_history_length",
    "v_predictor_mode",
    "k_prediction_nmse",
    "k_reconstruction_nmse",
    "v_prediction_nmse",
    "v_reconstruction_nmse",
    "psnr_db",
    "ssim",
    "lpips",
    "fid",
    "videos",
    "generation_seconds",
    "status",
)


@dataclass(frozen=True)
class Variant:
    name: str
    axis: str
    k_history_length: int
    v_history_length: int
    v_predictor_mode: str
    k_predictor_mode: str


@dataclass(frozen=True)
class Job:
    name: str
    phase: str
    command: tuple[str, ...]
    environment: dict[str, str]
    log_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _codec_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if "codec" in payload:
        if len(payload) != 1 or not isinstance(payload["codec"], dict):
            raise ValueError("'codec' must be the only top-level config object")
        payload = payload["codec"]
    return dict(payload)


def _parse_gpus(raw: str | None) -> tuple[str, ...]:
    selected = raw or os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1,2,3"
    values = tuple(item.strip() for item in selected.split(",") if item.strip())
    if len(values) != 4:
        raise ValueError(
            "the final history ablation requires exactly four visible GPUs; "
            f"got {values!r}"
        )
    if len(set(values)) != 4:
        raise ValueError(f"GPU list contains duplicates: {values!r}")
    return values


def _prompt_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _variants(base: CodecConfig, *, k_baseline: int, v_baseline: int) -> tuple[Variant, ...]:
    if base.fit_mode != "legacy":
        raise ValueError("history ablation requires fit_mode=legacy")
    if base.cross_kv_weight_mode != "layer_shared":
        raise ValueError(
            "history ablation requires cross_kv_weight_mode=layer_shared"
        )
    if k_baseline not in K_HISTORY_VALUES:
        raise ValueError(f"k baseline must be one of {K_HISTORY_VALUES}")
    if v_baseline not in V_HISTORY_VALUES:
        raise ValueError(f"v baseline must be one of {V_HISTORY_VALUES}")
    positive_k_mode = base.k_predictor_mode
    if positive_k_mode == "direct":
        positive_k_mode = "shared"
    variants = [
        Variant(
            name=f"k_h{history}_cross_vh{v_baseline}",
            axis="k_history",
            k_history_length=history,
            v_history_length=v_baseline,
            v_predictor_mode="cross_kv",
            k_predictor_mode="direct" if history == 0 else positive_k_mode,
        )
        for history in K_HISTORY_VALUES
    ]
    for history in V_HISTORY_VALUES:
        if history == v_baseline:
            continue
        variants.append(
            Variant(
                name=f"v_h{history}_cross_kh{k_baseline}",
                axis="v_history",
                k_history_length=k_baseline,
                v_history_length=history,
                v_predictor_mode="cross_kv",
                k_predictor_mode=(
                    "direct" if k_baseline == 0 else positive_k_mode
                ),
            )
        )
    variants.append(
        Variant(
            name=f"v_h1_no_cross_kh{k_baseline}",
            axis="without_cross",
            k_history_length=k_baseline,
            v_history_length=1,
            v_predictor_mode="temporal",
            k_predictor_mode=(
                "direct" if k_baseline == 0 else positive_k_mode
            ),
        )
    )
    # Full V-history sweep with the temporal (no-CrossKV) predictor.  The
    # temporal predictor needs h_V >= 1, and h_V=1 is already the `without_cross`
    # control above, so this adds h_V = 2, 4, 5.
    for history in V_HISTORY_VALUES:
        if history <= 1:
            continue
        variants.append(
            Variant(
                name=f"v_h{history}_no_cross_kh{k_baseline}",
                axis="v_history_no_cross",
                k_history_length=k_baseline,
                v_history_length=history,
                v_predictor_mode="temporal",
                k_predictor_mode=(
                    "direct" if k_baseline == 0 else positive_k_mode
                ),
            )
        )
    return tuple(variants)


def _variant_config(base_payload: dict[str, Any], variant: Variant) -> CodecConfig:
    payload = dict(base_payload)
    payload.update(
        {
            "k_bits": 2,
            "v_bits": 2,
            "k_history_length": variant.k_history_length,
            "v_history_length": variant.v_history_length,
            "k_predictor_mode": variant.k_predictor_mode,
            "v_predictor_mode": variant.v_predictor_mode,
        }
    )
    if variant.v_predictor_mode in {"temporal", "identity", "taylor", "taylor2"}:
        payload["cross_kv_weight_mode"] = "layer_shared"
    return config_from_mapping(payload)


def _variant_paths(root: Path, variant: Variant) -> dict[str, Path]:
    return {
        "config": root / "configs" / f"{variant.name}.json",
        "params": root / "params" / variant.name,
        "run": root / "runs" / variant.name,
        "cache": root / "cache" / f"{variant.name}.json",
        "quality": root / "runs" / variant.name / "quality.json",
    }


def _v_parameter_name(config: CodecConfig) -> str:
    if config.v_predictor_mode in {
        "temporal", "identity", "taylor", "taylor2", "direct"
    }:
        return "v_temporal.npz"
    if config.v_predictor_mode in {
        "cross_kv_identity", "cross_kv_taylor", "cross_kv_taylor2"
    }:
        return "v_cross_kv.npz"
    if config.v_predictor_mode == "innovation":
        return "v_innovation.npz"
    return "v_cross_kv.npz"


def _generation_complete(path: Path, expected_videos: int) -> bool:
    report_path = path / "pipeline_report.json"
    if not report_path.is_file():
        return False
    try:
        report = _read_json(report_path)
    except ValueError:
        return False
    return (
        report.get("status") == "complete"
        and int(report.get("requested_samples", -1)) == expected_videos
        and int(report.get("generated_samples", -1)) == expected_videos
        and int(report.get("skipped_samples", -1)) == 0
    )


def _base_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = {
        "PYTHON_BIN": str(Path(sys.executable).resolve()),
        "CHECKPOINT": str(args.checkpoint.resolve()),
        "WAN_MODEL_DIR": str(args.wan_model_dir.resolve()),
        "PROMPTS": str(args.prompts.resolve()),
        "FRAMES": str(args.frames),
        "NUM_SAMPLES": "1",
        "SEED": str(args.seed),
        "START_INDEX": str(args.start_index),
        "END_INDEX": str(args.end_index),
        "NUM_GPUS": "1",
        "SKIP_EXISTING": "1" if args.resume else "0",
        "BENCHMARK": "0" if args.resume else "1",
        "PROFILE": "1" if args.profile else "0",
        "TORCH_DECODER": "1" if args.torch_decoder else "0",
        "ALLOW_TRITON_FALLBACK": "1" if args.allow_triton_fallback else "0",
    }
    if args.model_config is not None:
        environment["MODEL_CONFIG"] = str(args.model_config.resolve())
    return environment


def _fit_jobs(
    args: argparse.Namespace,
    root: Path,
    variants: Iterable[Variant],
    configs: dict[str, CodecConfig],
) -> list[Job]:
    jobs = []
    for variant in variants:
        paths = _variant_paths(root, variant)
        if args.resume and (paths["params"] / "fit_report.json").is_file():
            continue
        jobs.append(
            Job(
                name=variant.name,
                phase="fit",
                command=("bash", str(FIT_SCRIPT)),
                environment={
                    "CONFIG": str(paths["config"]),
                    "TRACE_FILE": str(args.trace.resolve()),
                    "OUTPUT_DIR": str(paths["params"]),
                    "FIT_HISTORY_SOURCE": args.fit_history_source,
                    "FIT_DEVICE": args.fit_device,
                    "OVERWRITE": "0",
                },
                log_path=root / "logs" / "fit" / f"{variant.name}.log",
            )
        )
    return jobs


def _cache_jobs(
    args: argparse.Namespace,
    root: Path,
    variants: Iterable[Variant],
    configs: dict[str, CodecConfig],
) -> list[Job]:
    jobs = []
    for variant in variants:
        paths = _variant_paths(root, variant)
        if args.resume and paths["cache"].is_file():
            continue
        jobs.append(
            Job(
                name=variant.name,
                phase="cache",
                command=(
                    str(Path(sys.executable).resolve()),
                    str(CACHE_EVALUATOR),
                    "--trace",
                    str(args.trace.resolve()),
                    "--config",
                    str(paths["config"]),
                    "--k-params",
                    str(paths["params"] / "k_temporal.npz"),
                    "--v-params",
                    str(paths["params"] / _v_parameter_name(configs[variant.name])),
                    "--output",
                    str(paths["cache"]),
                    "--device",
                    args.cache_device,
                ),
                environment={},
                log_path=root / "logs" / "cache" / f"{variant.name}.log",
            )
        )
    return jobs


def _generation_jobs(
    args: argparse.Namespace,
    root: Path,
    variants: Iterable[Variant],
    configs: dict[str, CodecConfig],
) -> list[Job]:
    jobs: list[Job] = []
    bf16_run = root / "runs" / "bf16"
    if not args.resume or not _generation_complete(bf16_run, args.videos):
        jobs.append(
            Job(
                name="bf16",
                phase="generation",
                command=("bash", str(RUN_BF16)),
                environment={
                    **_base_environment(args),
                    "OUTPUT_DIR": str(bf16_run),
                },
                log_path=root / "logs" / "generation" / "bf16.log",
            )
        )
    for variant in variants:
        paths = _variant_paths(root, variant)
        if args.resume and _generation_complete(paths["run"], args.videos):
            continue
        jobs.append(
            Job(
                name=variant.name,
                phase="generation",
                command=("bash", str(RUN_TEMPOKV)),
                environment={
                    **_base_environment(args),
                    "CONFIG": str(paths["config"]),
                    "PARAM_DIR": str(paths["params"]),
                    "OUTPUT_DIR": str(paths["run"]),
                },
                log_path=root / "logs" / "generation" / f"{variant.name}.log",
            )
        )
    return jobs


def _quality_jobs(
    args: argparse.Namespace,
    root: Path,
    variants: Iterable[Variant],
) -> list[Job]:
    reference = root / "runs" / "bf16" / "videos"
    jobs = []
    for variant in variants:
        paths = _variant_paths(root, variant)
        if args.resume and paths["quality"].is_file():
            continue
        command = [
            str(Path(sys.executable).resolve()),
            str(QUALITY_EVALUATOR),
            str(reference),
            str(paths["run"] / "videos"),
            "--output",
            str(paths["quality"]),
            "--device",
            args.quality_device,
            "--allow-missing",
        ]
        if args.max_quality_frames is not None:
            command.extend(("--max-frames", str(args.max_quality_frames)))
        if args.quality_frame_stride != 1:
            command.extend(("--frame-stride", str(args.quality_frame_stride)))
        jobs.append(
            Job(
                name=variant.name,
                phase="quality",
                command=tuple(command),
                environment={},
                log_path=root / "logs" / "quality" / f"{variant.name}.log",
            )
        )
    return jobs


def _render_command(command: Iterable[str]) -> str:
    import shlex

    return shlex.join(tuple(command))


def _run_jobs(jobs: list[Job], gpus: tuple[str, ...], *, dry_run: bool) -> None:
    if not jobs:
        return
    for batch_start in range(0, len(jobs), len(gpus)):
        batch = jobs[batch_start : batch_start + len(gpus)]
        if dry_run:
            for slot, job in enumerate(batch):
                print(
                    f"[dry-run] phase={job.phase} gpu={gpus[slot]} "
                    f"name={job.name}: {_render_command(job.command)}"
                )
            continue
        processes: list[tuple[Job, str, subprocess.Popen[bytes]]] = []
        for slot, job in enumerate(batch):
            gpu = gpus[slot]
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(job.environment)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            if job.phase == "generation":
                cache_path = job.log_path.parents[2] / "triton_cache" / f"gpu-{gpu}"
                environment["TRITON_CACHE"] = str(cache_path)
            log = job.log_path.open("wb")
            process = subprocess.Popen(
                list(job.command),
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.close()
            processes.append((job, gpu, process))
            print(f"[{job.phase}] gpu={gpu} name={job.name}")
        failures: list[str] = []
        for job, gpu, process in processes:
            return_code = process.wait()
            if return_code:
                failures.append(
                    f"{job.name} on GPU {gpu} exited {return_code}; log={job.log_path}"
                )
        if failures:
            raise RuntimeError("\n".join(failures))


def _load_quality(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    required = ("videos", "psnr_db", "ssim", "lpips")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"quality report is missing {missing}: {path}")
    return payload


def _load_cache(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    for tensor in ("k", "v"):
        if tensor not in payload or not isinstance(payload[tensor], dict):
            raise ValueError(f"cache report is missing {tensor}: {path}")
        for metric in ("prediction_nmse", "reconstruction_nmse"):
            if metric not in payload[tensor]:
                raise ValueError(f"cache report is missing {tensor}.{metric}: {path}")
    return payload


def _load_generation(path: Path, expected_videos: int) -> dict[str, Any]:
    report = _read_json(path / "pipeline_report.json")
    if report.get("status") != "complete":
        raise ValueError(f"generation is not complete: {path}")
    if (
        int(report.get("requested_samples", -1)) != expected_videos
        or int(report.get("generated_samples", -1)) != expected_videos
        or int(report.get("skipped_samples", -1)) != 0
    ):
        raise ValueError(f"generation does not contain exactly {expected_videos} fresh videos: {path}")
    return report


def _summary_row(
    variant: Variant,
    config: CodecConfig,
    cache: dict[str, Any],
    quality: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": variant.name,
        "axis": variant.axis,
        "k_history_length": variant.k_history_length,
        "k_predictor_mode": config.k_predictor_mode,
        "v_history_length": variant.v_history_length,
        "v_predictor_mode": config.v_predictor_mode,
        "k_prediction_nmse": cache["k"]["prediction_nmse"],
        "k_reconstruction_nmse": cache["k"]["reconstruction_nmse"],
        "v_prediction_nmse": cache["v"]["prediction_nmse"],
        "v_reconstruction_nmse": cache["v"]["reconstruction_nmse"],
        "psnr_db": quality["psnr_db"],
        "ssim": quality["ssim"],
        "lpips": quality["lpips"],
        "fid": quality.get("fid"),
        "videos": quality["videos"],
        "generation_seconds": generation.get("generation_seconds"),
        "status": "complete",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in SUMMARY_FIELDS} for row in rows)


def _find_row(rows: list[dict[str, Any]], *, k: int, v: int, mode: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row["k_history_length"] == k
            and row["v_history_length"] == v
            and row["v_predictor_mode"] == mode
        ),
        None,
    )


def _delta(after: dict[str, Any], before: dict[str, Any], key: str) -> float:
    return float(after[key]) - float(before[key])


def _aggregate(
    args: argparse.Namespace,
    root: Path,
    variants: tuple[Variant, ...],
    configs: dict[str, CodecConfig],
) -> dict[str, Any]:
    bf16_generation = _load_generation(root / "runs" / "bf16", args.videos)
    del bf16_generation
    rows = []
    details = []
    for variant in variants:
        paths = _variant_paths(root, variant)
        cache = _load_cache(paths["cache"])
        quality = _load_quality(paths["quality"])
        generation = _load_generation(paths["run"], args.videos)
        if int(quality["videos"]) != args.videos:
            raise ValueError(
                f"quality report contains {quality['videos']} videos, expected {args.videos}: "
                f"{paths['quality']}"
            )
        row = _summary_row(variant, configs[variant.name], cache, quality, generation)
        rows.append(row)
        details.append(
            {
                "name": variant.name,
                "axis": variant.axis,
                "config": asdict(configs[variant.name]),
                "paths": {key: str(value.resolve()) for key, value in paths.items()},
                "cache": cache,
                "quality": {
                    key: quality.get(key)
                    for key in (
                        "videos",
                        "frames",
                        "mse",
                        "psnr_db",
                        "ssim",
                        "lpips",
                        "lpips_network",
                        "frame_stride",
                        "max_frames_per_video",
                    )
                },
                "generation": {
                    key: generation.get(key)
                    for key in (
                        "generation_seconds",
                        "requested_samples",
                        "generated_samples",
                        "skipped_samples",
                        "performance",
                    )
                },
            }
        )

    k_rows = [row for row in rows if row["axis"] == "k_history"]
    v_rows = [
        row
        for row in rows
        if row["k_history_length"] == args.k_baseline
        and row["v_predictor_mode"] == "cross_kv"
    ]
    v_rows.sort(key=lambda row: int(row["v_history_length"]))
    no_cross = [row for row in rows if row["axis"] == "without_cross"]
    v_temporal_rows = [
        row for row in rows if row["v_predictor_mode"] == "temporal"
    ]
    v_temporal_rows.sort(key=lambda row: int(row["v_history_length"]))
    cross_baseline = _find_row(
        rows, k=args.k_baseline, v=args.v_baseline, mode="cross_kv"
    )
    no_cross_baseline = no_cross[0] if no_cross else None
    cross_comparison = None
    if cross_baseline is not None and no_cross_baseline is not None:
        cross_comparison = {
            "definition": "CrossKV+V-history minus independently fitted temporal-only V",
            "v_prediction_nmse_delta": _delta(
                cross_baseline, no_cross_baseline, "v_prediction_nmse"
            ),
            "v_reconstruction_nmse_delta": _delta(
                cross_baseline, no_cross_baseline, "v_reconstruction_nmse"
            ),
            "psnr_db_delta": _delta(cross_baseline, no_cross_baseline, "psnr_db"),
            "ssim_delta": _delta(cross_baseline, no_cross_baseline, "ssim"),
            "lpips_delta": _delta(cross_baseline, no_cross_baseline, "lpips"),
        }

    summary = {
        "schema_version": 1,
        "status": "complete",
        "study": "self_forcing_history_ablation",
        "protocol": {
            "model": "Self-Forcing",
            "bits": 2,
            "anchor_bits": configs[variants[0].name].anchor_bits,
            "videos_per_variant": args.videos,
            "frames_per_video": args.frames,
            "prompt_range": [args.start_index, args.end_index],
            "seed": args.seed,
            "fit_history_source": args.fit_history_source,
            "gpus": list(args.gpus),
            "k_history_values": list(K_HISTORY_VALUES),
            "v_history_values": list(V_HISTORY_VALUES),
            "k_baseline_for_v_axis": args.k_baseline,
            "v_baseline_for_k_axis": args.v_baseline,
        },
        "inputs": {
            "trace": str(args.trace.resolve()),
            "trace_sha256": _sha256(args.trace),
            "base_config": str(args.base_config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "wan_model_dir": str(args.wan_model_dir.resolve()),
            "prompts": str(args.prompts.resolve()),
            "prompts_sha256": _sha256(args.prompts),
        },
        "baseline": {
            "method": "bf16",
            "run": str((root / "runs" / "bf16").resolve()),
        },
        "tables": {
            "k_history": k_rows,
            "v_history_cross_kv": v_rows,
            "without_cross": no_cross,
            "v_history_no_cross": v_temporal_rows,
        },
        "cross_comparison": cross_comparison,
        "rows": rows,
        "variants": details,
    }
    _write_json(root / "summary.json", summary)
    _write_csv(root / "summary.csv", rows)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "legacy.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wan-model-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        help="exactly four GPU ids, e.g. 0,1,2,3; defaults to CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--videos", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=717)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-baseline", type=int, default=3)
    parser.add_argument("--v-baseline", type=int, default=1)
    parser.add_argument("--fit-device", default="cuda")
    parser.add_argument("--cache-device", default="cpu")
    parser.add_argument("--quality-device", default="cuda")
    parser.add_argument(
        "--fit-history-source",
        choices=("reconstructed", "clean"),
        default="reconstructed",
    )
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--max-quality-frames", type=int)
    parser.add_argument("--quality-frame-stride", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--torch-decoder", action="store_true")
    parser.add_argument("--allow-triton-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    for name, path in (
        ("trace", args.trace),
        ("base-config", args.base_config),
        ("checkpoint", args.checkpoint),
        ("wan-model-dir", args.wan_model_dir),
        ("prompts", args.prompts),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if not args.wan_model_dir.is_dir():
        raise ValueError(f"wan-model-dir is not a directory: {args.wan_model_dir}")
    if args.videos <= 0:
        raise ValueError("videos must be positive")
    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")
    if args.frames <= 0 or (args.frames + 3) % 12:
        raise ValueError("frames must satisfy (frames + 3) % 12 == 0")
    if args.quality_frame_stride <= 0:
        raise ValueError("quality-frame-stride must be positive")
    if args.max_quality_frames is not None and args.max_quality_frames <= 0:
        raise ValueError("max-quality-frames must be positive")
    args.gpus = _parse_gpus(args.gpus)
    prompt_count = _prompt_count(args.prompts)
    args.end_index = args.start_index + args.videos
    if prompt_count < args.end_index:
        raise ValueError(
            f"prompts contains {prompt_count} non-empty lines; "
            f"{args.end_index} are required"
        )

    base_payload = _codec_payload(args.base_config)
    base = config_from_mapping(base_payload)
    variants = _variants(
        base,
        k_baseline=args.k_baseline,
        v_baseline=args.v_baseline,
    )
    configs = {
        variant.name: _variant_config(base_payload, variant) for variant in variants
    }
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        paths = _variant_paths(root, variant)
        _write_json(paths["config"], {"codec": asdict(configs[variant.name])})

    fit_jobs = _fit_jobs(args, root, variants, configs)
    cache_jobs = _cache_jobs(args, root, variants, configs)
    generation_jobs = _generation_jobs(args, root, variants, configs)
    quality_jobs = _quality_jobs(args, root, variants)
    manifest = {
        "schema_version": 1,
        "study": "self_forcing_history_ablation",
        "protocol": {
            "bits": 2,
            "videos_per_variant": args.videos,
            "frames_per_video": args.frames,
            "prompt_range": [args.start_index, args.end_index],
            "seed": args.seed,
            "gpus": list(args.gpus),
            "fit_history_source": args.fit_history_source,
        },
        "variants": [
            {
                "name": variant.name,
                "axis": variant.axis,
                "config": str(_variant_paths(root, variant)["config"].resolve()),
                "params": str(_variant_paths(root, variant)["params"].resolve()),
                "run": str(_variant_paths(root, variant)["run"].resolve()),
                "k_history_length": variant.k_history_length,
                "v_history_length": variant.v_history_length,
                "k_predictor_mode": variant.k_predictor_mode,
                "v_predictor_mode": variant.v_predictor_mode,
            }
            for variant in variants
        ],
        "commands": {
            phase: [
                {
                    "name": job.name,
                    "gpu_slot": index % len(args.gpus),
                    "command": list(job.command),
                    "log": str(job.log_path.resolve()),
                }
                for index, job in enumerate(jobs)
            ]
            for phase, jobs in (
                ("fit", fit_jobs),
                ("cache", cache_jobs),
                ("generation", generation_jobs),
                ("quality", quality_jobs),
            )
        },
    }
    _write_json(root / "manifest.json", manifest)

    if args.dry_run:
        for phase, jobs in (
            ("fit", fit_jobs),
            ("cache", cache_jobs),
            ("generation", generation_jobs),
            ("quality", quality_jobs),
        ):
            print(f"\n--- {phase} ({len(jobs)} jobs, four-GPU batches) ---")
            _run_jobs(jobs, args.gpus, dry_run=True)
        print(f"\n[dry-run] manifest: {root / 'manifest.json'}")
        return None

    _run_jobs(fit_jobs, args.gpus, dry_run=False)
    _run_jobs(cache_jobs, args.gpus, dry_run=False)

    # Streaming quality evaluation: run generation in batches of 4 GPUs,
    # then immediately run quality eval for each completed batch.
    bf16_generation_jobs = [job for job in generation_jobs if job.name == "bf16"]
    variant_generation_jobs = [job for job in generation_jobs if job.name != "bf16"]

    _run_jobs(bf16_generation_jobs, args.gpus, dry_run=False)

    # Process variant generation in batches of 4, with quality eval after each batch
    batch_size = len(args.gpus)
    for batch_start in range(0, len(variant_generation_jobs), batch_size):
        batch_gen_jobs = variant_generation_jobs[batch_start : batch_start + batch_size]
        _run_jobs(batch_gen_jobs, args.gpus, dry_run=False)

        # Immediately run quality eval for this batch's variants
        batch_variant_names = {job.name for job in batch_gen_jobs}
        batch_qual_jobs = [job for job in quality_jobs if job.name in batch_variant_names]
        if batch_qual_jobs:
            _run_jobs(batch_qual_jobs, args.gpus, dry_run=False)

    # On --resume, variants whose generation was already complete are not part of
    # any batch, so their quality eval was skipped above. Run any still-missing
    # quality jobs now so the aggregate does not crash on absent quality.json.
    remaining_qual = [
        job for job in quality_jobs
        if not (root / "runs" / job.name / "quality.json").is_file()
    ]
    if remaining_qual:
        _run_jobs(remaining_qual, args.gpus, dry_run=False)

    return _aggregate(args, root, variants, configs)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(args)
    if report is not None:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n完成：{args.output.expanduser().resolve() / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
