#!/usr/bin/env python3
"""Run the 60-prompt Self-Forcing K/V predictor ablation on selected GPUs.

The runner varies K and V temporal predictors together and treats CrossKV as an
independent V-side switch.  BF16 and the existing production TempoKV videos are
supplied as reference directories and are never regenerated here.  Generation
runs one variant per selected GPU in each batch.  As soon as a generation batch
finishes, quality workers start in another GPU batch and append one row per
video to the CSV while the video frames are being decoded.

Typical invocation::

    python scripts/paper_experiments/predictor_ablation/run.py \
        --checkpoint /models/self_forcing_dmd.pt \
        --wan-model-dir /models/Wan2.1-T2V-1.3B \
        --trace data/self-forcing/trace/clean_trace.npz \
        --bf16-videos results/self_forcing/bf16/videos \
        --tempokv-videos results/self_forcing/legacy-kh2-vh1-int4-layershared/videos \
        --output-root results/self_forcing/predictor_ablation_60p \
        --gpus 0,1
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

EVALUATION_ROOT = PROJECT_ROOT / "scripts" / "evaluation"
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from tempokv.config import CodecConfig, config_from_mapping  # noqa: E402


FIT_SCRIPT = PROJECT_ROOT / "scripts" / "Self-Forcing" / "fit_params.sh"
GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "Self-Forcing" / "generate.sh"
PROMPTS_DEFAULT = PROJECT_ROOT / "assets" / "moviegen128.txt"
VIDEO_COUNT = 60
START_INDEX = 0
SCHEMA = "tempokv.predictor_ablation.v2"


def _shell_binary() -> str:
    configured = os.environ.get("BASH_BIN")
    if configured:
        return configured
    return "/bin/bash" if Path("/bin/bash").is_file() else "bash"


BASH_BIN = _shell_binary()


@dataclass(frozen=True)
class Variant:
    """One K/V predictor pair and CrossKV choice to generate and score."""

    name: str
    k_predictor_mode: str
    k_history_length: int
    v_temporal_predictor_mode: str
    v_history_length: int
    use_cross_kv: bool
    interpretation: str

    @property
    def v_predictor_mode(self) -> str:
        if not self.use_cross_kv:
            return self.v_temporal_predictor_mode
        if self.v_temporal_predictor_mode == "temporal":
            return "cross_kv"
        return f"cross_kv_{self.v_temporal_predictor_mode}"

    @property
    def v_artifact(self) -> str:
        return "v_cross_kv.npz" if self.use_cross_kv else "v_temporal.npz"

    @property
    def k_coefficient_source(self) -> str:
        return (
            "fixed"
            if self.k_predictor_mode in {"identity", "taylor", "taylor2"}
            else "regression"
        )

    @property
    def v_coefficient_source(self) -> str:
        return (
            "fixed"
            if self.v_temporal_predictor_mode in {"identity", "taylor", "taylor2"}
            else "regression"
        )

    def record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "v_predictor_mode": self.v_predictor_mode,
            "v_artifact": self.v_artifact,
            "k_coefficient_source": self.k_coefficient_source,
            "v_coefficient_source": self.v_coefficient_source,
            "cross_kv_fitting": "regression" if self.use_cross_kv else "disabled",
        }


VARIANTS: tuple[Variant, ...] = (
    Variant(
        "learned_no_cross_vh2",
        "shared",
        2,
        "temporal",
        2,
        False,
        "learned K + learned V temporal predictors without CrossKV",
    ),
    Variant(
        "learned_cross_vh2",
        "shared",
        2,
        "temporal",
        2,
        True,
        "learned K + learned V temporal predictors with CrossKV",
    ),
    Variant(
        "identity_no_cross_vh2",
        "identity",
        2,
        "identity",
        2,
        False,
        "fixed identity K/V predictors without CrossKV",
    ),
    Variant(
        "identity_cross_vh2",
        "identity",
        2,
        "identity",
        2,
        True,
        "fixed identity K/V predictors with CrossKV",
    ),
    Variant(
        "taylor_no_cross_vh2",
        "taylor",
        2,
        "taylor",
        2,
        False,
        "fixed first-order Taylor K/V predictors without CrossKV",
    ),
    Variant(
        "taylor_cross_vh2",
        "taylor",
        2,
        "taylor",
        2,
        True,
        "fixed first-order Taylor K/V predictors with CrossKV",
    ),
    Variant(
        "taylor2_no_cross_vh3",
        "taylor2",
        3,
        "taylor2",
        3,
        False,
        "fixed second-order Taylor K/V predictors without CrossKV",
    ),
    Variant(
        "taylor2_cross_vh3",
        "taylor2",
        3,
        "taylor2",
        3,
        True,
        "fixed second-order Taylor K/V predictors with CrossKV",
    ),
)


def _select_variants(raw: str | None) -> tuple[Variant, ...]:
    """Return the requested subset of ``VARIANTS``, or the whole matrix."""
    if not raw:
        return VARIANTS
    wanted = tuple(item.strip() for item in raw.split(",") if item.strip())
    known = tuple(variant.name for variant in VARIANTS)
    unknown = tuple(name for name in wanted if name not in known)
    if unknown:
        raise ValueError(f"unknown variant(s) {list(unknown)}; choose from {list(known)}")
    selected = set(wanted)
    return tuple(variant for variant in VARIANTS if variant.name in selected)


@dataclass(frozen=True)
class Job:
    name: str
    kind: str
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


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Publish a valid partial/final report after every completed video."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt_digest(path: Path) -> str:
    """Hash exactly the first 60 prompt lines passed to Self-Forcing."""
    prompts = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(prompts) < VIDEO_COUNT:
        raise ValueError(
            f"prompt file contains {len(prompts)} lines; "
            f"{VIDEO_COUNT} are required"
        )
    selected = prompts[:VIDEO_COUNT]
    if any(not prompt.strip() for prompt in selected):
        raise ValueError("the first 60 prompt lines must all be non-empty")
    digest = hashlib.sha256()
    digest.update("\n".join(selected).encode("utf-8"))
    return digest.hexdigest()


def _codec_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if "codec" in payload:
        if len(payload) != 1 or not isinstance(payload["codec"], dict):
            raise ValueError("'codec' must be the only top-level config object")
        payload = payload["codec"]
    return dict(payload)


def _parse_gpus(raw: str | None) -> tuple[str, ...]:
    selected = raw or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    values = tuple(item.strip() for item in selected.split(",") if item.strip())
    if not values:
        raise ValueError("predictor ablation requires at least one GPU id")
    if len(set(values)) != len(values):
        raise ValueError(f"GPU list contains duplicates: {values!r}")
    return values


def _resolve_bits(args: argparse.Namespace, base: CodecConfig) -> tuple[int, int]:
    if args.bits is not None and (
        args.k_bits is not None or args.v_bits is not None
    ):
        raise ValueError("--bits cannot be combined with --k-bits or --v-bits")
    if args.bits is not None:
        return args.bits, args.bits
    return (
        base.k_bits if args.k_bits is None else args.k_bits,
        base.v_bits if args.v_bits is None else args.v_bits,
    )


def _variant_config(
    base_payload: dict[str, Any],
    base: CodecConfig,
    variant: Variant,
    *,
    k_bits: int,
    v_bits: int,
) -> CodecConfig:
    payload = dict(base_payload)
    payload.update(
        {
            "k_bits": k_bits,
            "v_bits": v_bits,
            "k_history_length": variant.k_history_length,
            "v_history_length": variant.v_history_length,
            "k_predictor_mode": variant.k_predictor_mode,
            "v_predictor_mode": variant.v_predictor_mode,
            "cross_kv_weight_mode": "layer_shared",
            "fit_mode": "legacy",
        }
    )
    config = config_from_mapping(payload)
    if config.fit_mode != "legacy":
        raise ValueError("predictor ablation requires fit_mode=legacy")
    # ``base`` is used to make the validation intent explicit and to catch a
    # malformed config before any job is scheduled.
    if base.group_size != config.group_size:
        raise ValueError("variant config changed the base group_size")
    return config


def _paths(root: Path, variant: Variant) -> dict[str, Path]:
    return {
        "config": root / "configs" / f"{variant.name}.json",
        "params": root / "params" / variant.name,
        "run": root / "runs" / variant.name,
        "quality": root / "quality" / variant.name,
        "generation_log": root / "logs" / "generation" / f"{variant.name}.log",
        "fit_log": root / "logs" / "fit" / f"{variant.name}.log",
    }


def _generation_complete(path: Path) -> bool:
    report_path = path / "pipeline_report.json"
    if not report_path.is_file():
        return False
    try:
        report = _read_json(report_path)
        return (
            report.get("status") == "complete"
            and int(report.get("requested_samples", -1)) == VIDEO_COUNT
            and int(report.get("generated_samples", -1)) == VIDEO_COUNT
            and int(report.get("skipped_samples", -1)) == 0
        )
    except (TypeError, ValueError):
        return False


def _fit_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        report = _read_json(path)
    except ValueError:
        return False
    return report.get("status") == "fit_complete"


def _quality_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
        return (
            payload.get("status") == "complete"
            and int(payload.get("videos", -1)) == VIDEO_COUNT
        )
    except (TypeError, ValueError):
        return False


def _video_sort_key(path: Path, root: Path) -> tuple[int, int, str]:
    """Sort ``0_0.mp4, 1_0.mp4, ...`` by prompt index, not text order."""
    relative = path.relative_to(root)
    prefix = relative.name.split("_", 1)[0]
    try:
        return (0, int(prefix), relative.as_posix())
    except ValueError:
        return (1, 0, relative.as_posix())


def _base_generation_environment(args: argparse.Namespace) -> dict[str, str]:
    return {
        "PYTHON_BIN": sys.executable,
        "CHECKPOINT": str(args.checkpoint.expanduser().resolve()),
        "WAN_MODEL_DIR": str(args.wan_model_dir.expanduser().resolve()),
        "PROMPTS": str(args.prompts.expanduser().resolve()),
        "FRAMES": str(args.frames),
        "NUM_SAMPLES": "1",
        "SEED": str(args.seed),
        "START_INDEX": str(START_INDEX),
        "END_INDEX": str(VIDEO_COUNT),
        "NUM_GPUS": "1",
        "SKIP_EXISTING": "1" if args.resume else "0",
        "BENCHMARK": "0" if args.resume else "1",
        "PROFILE": "1" if args.profile else "0",
        "TORCH_DECODER": "1" if args.torch_decoder else "0",
        "ALLOW_TRITON_FALLBACK": "1" if args.allow_triton_fallback else "0",
    }


def _fit_jobs(
    args: argparse.Namespace,
    root: Path,
    *,
    k_bits: int,
    v_bits: int,
) -> list[Job]:
    jobs: list[Job] = []
    for variant in VARIANTS:
        paths = _paths(root, variant)
        if args.resume and _fit_complete(paths["params"] / "fit_report.json"):
            continue
        jobs.append(
            Job(
                name=variant.name,
                kind="fit",
                command=(BASH_BIN, str(FIT_SCRIPT)),
                environment={
                    "PYTHON_BIN": sys.executable,
                    "CONFIG": str(paths["config"].resolve()),
                    "TRACE_FILE": str(args.trace.expanduser().resolve()),
                    "OUTPUT_DIR": str(paths["params"].resolve()),
                    "KH": str(variant.k_history_length),
                    "VH": str(variant.v_history_length),
                    "K_BITS": str(k_bits),
                    "V_BITS": str(v_bits),
                    "V_PREDICTOR": variant.v_predictor_mode,
                    "FIT_HISTORY_SOURCE": args.fit_history_source,
                    "FIT_DEVICE": args.fit_device,
                    "ITERATIONS": str(args.iterations),
                    "OVERWRITE": "1" if args.resume else "0",
                    "DRY_RUN": "0",
                },
                log_path=paths["fit_log"],
            )
        )
    return jobs


def _generation_jobs(
    args: argparse.Namespace,
    root: Path,
    configs: dict[str, CodecConfig],
) -> list[Job]:
    jobs: list[Job] = []
    base_environment = _base_generation_environment(args)
    for variant in VARIANTS:
        paths = _paths(root, variant)
        if args.resume and _generation_complete(paths["run"]):
            continue
        environment = {
            **base_environment,
            "MODE": "tempokv",
            "CONFIG": str(paths["config"].resolve()),
            "PARAM_DIR": str(paths["params"].resolve()),
            "OUTPUT_DIR": str(paths["run"].resolve()),
            "KH": str(configs[variant.name].k_history_length),
            "VH": str(variant.v_history_length),
            "K_BITS": str(configs[variant.name].k_bits),
            "V_BITS": str(configs[variant.name].v_bits),
            "V_PREDICTOR": variant.v_predictor_mode,
        }
        if args.model_config is not None:
            environment["MODEL_CONFIG"] = str(
                args.model_config.expanduser().resolve()
            )
        if args.local_attn_size is not None:
            environment["LOCAL_ATTN_SIZE"] = str(args.local_attn_size)
        jobs.append(
            Job(
                name=variant.name,
                kind="generation",
                command=(BASH_BIN, str(GENERATE_SCRIPT)),
                environment=environment,
                log_path=paths["generation_log"],
            )
        )
    return jobs


def _quality_jobs(
    args: argparse.Namespace,
    root: Path,
    references: dict[str, Path],
) -> list[Job]:
    jobs: list[Job] = []
    for variant in VARIANTS:
        paths = _paths(root, variant)
        for reference_name, reference_dir in references.items():
            output_json = paths["quality"] / f"vs_{reference_name}.json"
            output_csv = paths["quality"] / f"vs_{reference_name}.csv"
            if args.resume and _quality_complete(output_json):
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--quality-worker",
                "--reference-dir",
                str(reference_dir.resolve()),
                "--candidate-dir",
                str((paths["run"] / "videos").resolve()),
                "--output-json",
                str(output_json.resolve()),
                "--output-csv",
                str(output_csv.resolve()),
                "--quality-device",
                args.quality_device,
                "--quality-frame-stride",
                str(args.quality_frame_stride),
                "--fid-batch-size",
                str(args.fid_batch_size),
                "--lpips-network",
                args.lpips_network,
            ]
            if args.max_quality_frames is not None:
                command.extend(("--max-quality-frames", str(args.max_quality_frames)))
            if args.quality_fid:
                command.append("--quality-fid")
            jobs.append(
                Job(
                    name=f"{variant.name}_vs_{reference_name}",
                    kind="quality",
                    command=tuple(command),
                    environment={},
                    log_path=paths["quality"] / f"{reference_name}.log",
                )
            )
    return jobs


def _job_record(job: Job, gpu_slot: int) -> dict[str, Any]:
    return {
        "name": job.name,
        "kind": job.kind,
        "gpu_slot": gpu_slot,
        "command": list(job.command),
        "environment": dict(sorted(job.environment.items())),
        "log": str(job.log_path.resolve()),
    }


def _run_jobs(
    jobs: list[Job],
    gpus: tuple[str, ...],
    *,
    root: Path,
    dry_run: bool,
) -> None:
    """Run jobs in batches sized to the selected GPU count."""
    for batch_start in range(0, len(jobs), len(gpus)):
        batch = jobs[batch_start : batch_start + len(gpus)]
        if dry_run:
            for slot, job in enumerate(batch):
                environment = {
                    **job.environment,
                    "CUDA_VISIBLE_DEVICES": gpus[slot],
                }
                print(
                    f"[dry-run] kind={job.kind} gpu={gpus[slot]} "
                    f"name={job.name}: "
                    + shlex.join(
                        tuple(
                            f"{key}={value}"
                            for key, value in sorted(environment.items())
                        )
                        + job.command
                    )
                )
            continue

        processes: list[tuple[Job, str, subprocess.Popen[Any]]] = []
        for slot, job in enumerate(batch):
            gpu = gpus[slot]
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(job.environment)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONUNBUFFERED"] = "1"
            if job.kind == "generation":
                environment["TRITON_CACHE"] = str(
                    root / "triton_cache" / f"gpu-{gpu}"
                )
            log = job.log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                list(job.command),
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log.close()
            processes.append((job, gpu, process))
            print(f"[{job.kind}] gpu={gpu} name={job.name}", flush=True)

        failures: list[str] = []
        for job, gpu, process in processes:
            return_code = process.wait()
            if return_code:
                failures.append(
                    f"{job.name} on GPU {gpu} exited {return_code}; "
                    f"log={job.log_path}"
                )
        if failures:
            raise RuntimeError("\n".join(failures))


def _video_quality_worker(args: argparse.Namespace) -> int:
    """Stream one reference/candidate comparison and persist partial output."""
    try:
        import evaluate_quality as quality

        reference_dir = args.reference_dir.expanduser().resolve()
        candidate_dir = args.candidate_dir.expanduser().resolve()
        references = quality._video_paths(reference_dir, quality.DEFAULT_EXTENSIONS)
        quality._video_paths(candidate_dir, quality.DEFAULT_EXTENSIONS)
        references.sort(key=lambda path: _video_sort_key(path, reference_dir))
        if len(references) < VIDEO_COUNT:
            raise ValueError(
                f"reference directory has {len(references)} videos; "
                f"{VIDEO_COUNT} are required: {reference_dir}"
            )
        references = references[:VIDEO_COUNT]
        missing = [
            path.relative_to(reference_dir).as_posix()
            for path in references
            if not (
                candidate_dir / path.relative_to(reference_dir)
            ).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"candidate directory is missing {len(missing)} of the first "
                f"{VIDEO_COUNT} reference videos: {', '.join(missing[:10])}"
            )

        backend = quality.QualityBackend(
            device=args.quality_device,
            network=args.lpips_network,
        )
        fid_backend = (
            quality.FidBackend(
                device=args.quality_device,
                batch_size=args.fid_batch_size,
            )
            if args.quality_fid
            else None
        )
        output_json = args.output_json.expanduser().resolve()
        output_csv = args.output_csv.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            "video",
            "frames",
            "mse",
            "psnr_db",
            "ssim",
            "lpips",
        )
        rows: list[dict[str, Any]] = []
        total_error = 0.0
        total_values = 0
        total_frames = 0
        total_ssim = 0.0
        total_lpips = 0.0
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            handle.flush()
            for reference in references:
                relative = reference.relative_to(reference_dir)
                candidate = candidate_dir / relative
                metrics = quality.compare_video(
                    reference,
                    candidate,
                    max_frames=args.max_quality_frames,
                    backend=backend,
                    fid_backend=fid_backend,
                    frame_stride=args.quality_frame_stride,
                )
                row = {
                    "video": relative.as_posix(),
                    "frames": metrics["frames"],
                    "mse": metrics["mse"],
                    "psnr_db": metrics["psnr_db"],
                    "ssim": metrics["ssim"],
                    "lpips": metrics["lpips"],
                }
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                total_error += float(metrics["squared_error"])
                total_values += int(metrics["values"])
                total_frames += int(metrics["frames"])
                total_ssim += float(metrics["ssim"]) * int(metrics["frames"])
                total_lpips += float(metrics["lpips"]) * int(metrics["frames"])
                _write_json_atomic(
                    output_json,
                    _quality_report(
                        status="running",
                        reference_dir=reference_dir,
                        candidate_dir=candidate_dir,
                        rows=rows,
                        missing=missing,
                        total_error=total_error,
                        total_values=total_values,
                        total_frames=total_frames,
                        total_ssim=total_ssim,
                        total_lpips=total_lpips,
                        fid=None,
                        args=args,
                    ),
                )
                print(
                    f"[quality] {candidate_dir.name} {len(rows)}/{VIDEO_COUNT} "
                    f"video={relative.name} psnr={metrics['psnr_db']}",
                    flush=True,
                )
        fid = fid_backend.compute() if fid_backend is not None else None
        _write_json_atomic(
            output_json,
            _quality_report(
                status="complete",
                reference_dir=reference_dir,
                candidate_dir=candidate_dir,
                rows=rows,
                missing=missing,
                total_error=total_error,
                total_values=total_values,
                total_frames=total_frames,
                total_ssim=total_ssim,
                total_lpips=total_lpips,
                fid=fid,
                args=args,
            ),
        )
        return 0
    except Exception as error:
        print(f"quality worker failed: {error}", file=sys.stderr)
        return 2


def _quality_report(
    *,
    status: str,
    reference_dir: Path,
    candidate_dir: Path,
    rows: list[dict[str, Any]],
    missing: list[str],
    total_error: float,
    total_values: int,
    total_frames: int,
    total_ssim: float,
    total_lpips: float,
    fid: float | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pooled_mse = total_error / total_values if total_values else None
    psnr = (
        None
        if pooled_mse is None or pooled_mse == 0
        else 10.0 * math.log10((255.0**2) / pooled_mse)
    )
    return {
        "schema": "tempokv.predictor_ablation.quality.v1",
        "status": status,
        "streaming": True,
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "expected_videos": VIDEO_COUNT,
        "videos": len(rows),
        "frames": total_frames,
        "missing": missing,
        "mse": pooled_mse,
        "psnr_db": psnr,
        "ssim": total_ssim / total_frames if total_frames else None,
        "lpips": total_lpips / total_frames if total_frames else None,
        "lpips_network": args.lpips_network,
        "fid": fid,
        "fid_feature": 2048 if args.quality_fid else None,
        "fid_frames": (
            sum(int(row["frames"]) for row in rows)
            if args.quality_fid
            else 0
        ),
        "frame_stride": args.quality_frame_stride,
        "max_frames_per_video": args.max_quality_frames,
        "device": args.quality_device,
        "per_video": rows,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _summary_rows(
    root: Path,
    references: dict[str, Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        paths = _paths(root, variant)
        generation = _read_json(paths["run"] / "pipeline_report.json")
        for reference_name in references:
            quality_path = paths["quality"] / f"vs_{reference_name}.json"
            quality = _read_json(quality_path)
            rows.append(
                {
                    **variant.record(),
                    "reference": reference_name,
                    "videos": quality.get("videos"),
                    "frames": quality.get("frames"),
                    "psnr_db": quality.get("psnr_db"),
                    "ssim": quality.get("ssim"),
                    "lpips": quality.get("lpips"),
                    "fid": quality.get("fid"),
                    "generation_seconds": generation.get("generation_seconds"),
                    "status": quality.get("status"),
                }
            )
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "variant",
        "k_predictor_mode",
        "k_history_length",
        "v_temporal_predictor_mode",
        "v_predictor_mode",
        "v_history_length",
        "use_cross_kv",
        "k_coefficient_source",
        "v_coefficient_source",
        "cross_kv_fitting",
        "interpretation",
        "reference",
        "videos",
        "frames",
        "psnr_db",
        "ssim",
        "lpips",
        "fid",
        "generation_seconds",
        "status",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _validate_main(args: argparse.Namespace) -> tuple[CodecConfig, int, int]:
    if not args.dry_run:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
        if not args.wan_model_dir.is_dir():
            raise FileNotFoundError(
                f"Wan model directory does not exist: {args.wan_model_dir}"
            )
    for name, path in (
        ("config", args.config),
        ("prompts", args.prompts),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if not args.dry_run and not args.trace.is_file():
        raise FileNotFoundError(f"trace does not exist: {args.trace}")
    if args.bf16_videos is None and args.quality and not args.dry_run:
        raise ValueError("--bf16-videos is required when quality evaluation is enabled")
    if args.bf16_videos is not None and not args.bf16_videos.is_dir() and not args.dry_run:
        raise FileNotFoundError(f"BF16 video directory does not exist: {args.bf16_videos}")
    if args.tempokv_videos is not None and not args.tempokv_videos.is_dir() and not args.dry_run:
        raise FileNotFoundError(
            f"TempoKV video directory does not exist: {args.tempokv_videos}"
        )
    if args.frames < 105 or (args.frames + 3) % 12:
        raise ValueError("--frames must be at least 105 and satisfy (frames + 3) % 12 == 0")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.local_attn_size is not None and (
        args.local_attn_size == 0 or args.local_attn_size < -1
    ):
        raise ValueError("--local-attn-size must be -1 or positive")
    if args.quality_frame_stride <= 0:
        raise ValueError("--quality-frame-stride must be positive")
    if args.max_quality_frames is not None and args.max_quality_frames <= 0:
        raise ValueError("--max-quality-frames must be positive")
    args.gpus = _parse_gpus(args.gpus)
    _prompt_digest(args.prompts)
    base_payload = _codec_payload(args.config)
    base = config_from_mapping(base_payload)
    if base.fit_mode != "legacy":
        raise ValueError("predictor ablation requires fit_mode=legacy")
    if base.cross_kv_weight_mode != "layer_shared":
        raise ValueError("predictor ablation requires cross_kv_weight_mode=layer_shared")
    k_bits, v_bits = _resolve_bits(args, base)
    # CodecConfig performs the bit-width validation; constructing every variant
    # here also catches Taylor history mistakes before any job starts.
    for variant in VARIANTS:
        _variant_config(
            base_payload,
            base,
            variant,
            k_bits=k_bits,
            v_bits=v_bits,
        )
    return base, k_bits, v_bits


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    global VARIANTS
    VARIANTS = _select_variants(args.variants)
    base, k_bits, v_bits = _validate_main(args)
    root = args.output_root.expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError("--output-root must not be the filesystem root")
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"output root is not empty: {root}; use --resume or choose another root"
        )
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if args.resume and manifest_path.is_file():
        existing_manifest = _read_json(manifest_path)
        if existing_manifest.get("schema") != SCHEMA:
            raise ValueError(
                f"cannot resume manifest schema {existing_manifest.get('schema')!r}; "
                f"expected {SCHEMA!r}"
            )
        existing_protocol = existing_manifest.get("protocol") or {}
        existing_bits = (
            existing_protocol.get("k_bits"),
            existing_protocol.get("v_bits"),
        )
        if existing_bits != (k_bits, v_bits):
            raise ValueError(
                "cannot resume predictor ablation with different K/V bit widths: "
                f"existing K{existing_bits[0]}/V{existing_bits[1]}, "
                f"requested K{k_bits}/V{v_bits}"
            )
    elif args.resume and any(root.iterdir()):
        raise ValueError(
            f"cannot resume without the existing manifest: {manifest_path}"
        )

    base_payload = _codec_payload(args.config)
    configs: dict[str, CodecConfig] = {}
    for variant in VARIANTS:
        config = _variant_config(
            base_payload,
            base,
            variant,
            k_bits=k_bits,
            v_bits=v_bits,
        )
        configs[variant.name] = config
        paths = _paths(root, variant)
        _write_json(paths["config"], {"codec": asdict(config)})

    references: dict[str, Path] = {}
    if args.bf16_videos is not None:
        references["bf16"] = args.bf16_videos.expanduser().resolve()
    if args.tempokv_videos is not None:
        references["tempokv"] = args.tempokv_videos.expanduser().resolve()

    fit_jobs = _fit_jobs(
        args,
        root,
        k_bits=k_bits,
        v_bits=v_bits,
    )
    generation_jobs = _generation_jobs(args, root, configs)
    quality_jobs = _quality_jobs(args, root, references) if args.quality else []
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if args.dry_run else "running",
        "protocol": {
            "model": "Self-Forcing",
            "prompt_file": str(args.prompts.expanduser().resolve()),
            "prompt_file_sha256": _sha256(args.prompts.expanduser().resolve()),
            "prompt_first60_sha256": _prompt_digest(args.prompts.expanduser().resolve()),
            "prompt_count": VIDEO_COUNT,
            "prompt_range": [START_INDEX, VIDEO_COUNT],
            "frames_per_video": args.frames,
            "samples_per_prompt": 1,
            "seed": args.seed,
            "k_bits": k_bits,
            "v_bits": v_bits,
            "k_history_length_by_variant": {
                variant.name: variant.k_history_length for variant in VARIANTS
            },
            "v_history_length_by_variant": {
                variant.name: variant.v_history_length for variant in VARIANTS
            },
            "cross_kv_by_variant": {
                variant.name: variant.use_cross_kv for variant in VARIANTS
            },
            "gpus": list(args.gpus),
            "generation_parallelism": len(args.gpus),
            "quality_streaming": bool(args.quality),
            "quality_frame_stride": args.quality_frame_stride,
            "rerun_bf16": False,
            "rerun_tempokv": False,
        },
        "references": {name: str(path) for name, path in references.items()},
        "variants": [
            {
                **variant.record(),
                "config": str(_paths(root, variant)["config"].resolve()),
                "params": str(_paths(root, variant)["params"].resolve()),
                "run": str(_paths(root, variant)["run"].resolve()),
            }
            for variant in VARIANTS
        ],
        "jobs": {
            "fit": [
                _job_record(job, i % len(args.gpus))
                for i, job in enumerate(fit_jobs)
            ],
            "generation": [
                _job_record(job, i % len(args.gpus))
                for i, job in enumerate(generation_jobs)
            ],
            "quality": [
                _job_record(job, i % len(args.gpus))
                for i, job in enumerate(quality_jobs)
            ],
        },
    }
    _write_json(manifest_path, manifest)

    if args.dry_run:
        for title, jobs in (
            ("fit", fit_jobs),
            ("generation", generation_jobs),
            ("quality", quality_jobs),
        ):
            print(
                f"\n--- {title} ({len(jobs)} jobs, "
                f"{len(args.gpus)}-GPU batches) ---"
            )
            _run_jobs(jobs, args.gpus, root=root, dry_run=True)
        print(f"\n[dry-run] manifest: {manifest_path}")
        return None

    _run_jobs(fit_jobs, args.gpus, root=root, dry_run=False)

    # Generation uses one predictor job per selected GPU. Once all generation
    # batches finish, quality workers stream one video at a time and do not
    # accumulate decoded frames in memory.
    _run_jobs(generation_jobs, args.gpus, root=root, dry_run=False)
    _run_jobs(quality_jobs, args.gpus, root=root, dry_run=False)

    rows = _summary_rows(root, references)
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "protocol": manifest["protocol"],
        "references": manifest["references"],
        "variants": manifest["variants"],
        "rows": rows,
    }
    _write_json(root / "summary.json", summary)
    _write_summary_csv(root / "summary.csv", rows)
    manifest["status"] = "complete"
    _write_json(manifest_path, manifest)
    print(f"[complete] summary: {root / 'summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wan-model-dir", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/legacy.json")
    parser.add_argument("--prompts", type=Path, default=PROMPTS_DEFAULT)
    parser.add_argument("--bf16-videos", type=Path)
    parser.add_argument("--tempokv-videos", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results/self_forcing/predictor_ablation_60p",
    )
    parser.add_argument(
        "--gpus",
        help="comma-separated GPU ids; defaults to CUDA_VISIBLE_DEVICES or GPU 0",
    )
    parser.add_argument(
        "--variants",
        help="comma-separated subset of the matrix to run (default: every variant)",
    )
    parser.add_argument("--frames", type=int, default=717)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bits", type=int, choices=range(1, 9))
    parser.add_argument("--k-bits", type=int, choices=range(1, 9))
    parser.add_argument("--v-bits", type=int, choices=range(1, 9))
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--fit-device", default="cuda")
    parser.add_argument("--quality-device", default="cuda")
    parser.add_argument("--quality-frame-stride", type=int, default=1)
    parser.add_argument("--max-quality-frames", type=int)
    parser.add_argument("--quality-fid", action="store_true")
    parser.add_argument(
        "--quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stream RGB quality metrics after the generation batches",
    )
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--lpips-network", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument(
        "--fit-history-source",
        choices=("reconstructed", "clean"),
        default="reconstructed",
    )
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--local-attn-size", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--torch-decoder", action="store_true")
    parser.add_argument("--allow-triton-fallback", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--quality-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--reference-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--candidate-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-csv", type=Path, help=argparse.SUPPRESS)
    return parser


def _prepare_worker_namespace(args: argparse.Namespace) -> argparse.Namespace:
    if args.fid_batch_size <= 0:
        raise ValueError("--fid-batch-size must be positive")
    if args.quality_frame_stride <= 0:
        raise ValueError("--quality-frame-stride must be positive")
    if args.reference_dir is None or args.candidate_dir is None:
        raise ValueError("quality worker requires reference and candidate directories")
    if args.output_json is None or args.output_csv is None:
        raise ValueError("quality worker requires output JSON and CSV paths")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quality_worker:
        try:
            return _video_quality_worker(_prepare_worker_namespace(args))
        except (OSError, ValueError) as error:
            print(f"quality worker setup failed: {error}", file=sys.stderr)
            return 2
    if args.checkpoint is None or args.wan_model_dir is None or args.trace is None:
        print(
            "--checkpoint, --wan-model-dir, and --trace are required for the "
            "orchestrator",
            file=sys.stderr,
        )
        return 2
    try:
        run(args)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"predictor ablation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
