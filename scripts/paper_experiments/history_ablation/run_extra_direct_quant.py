#!/usr/bin/env python3
"""Run the extra history-ablation variants (direct-quant + cross_vh0).

Completes the history ablation's corners not covered by the main 11-variant
sweep (all INT2, 20 prompts, moviegen 0..19):

* ``k_h0_v_h0`` — no K predictor (direct) and V fully direct (no predictor,
  no CrossKV): "kv 都无预测/直接量化".
* ``k_h2_v_h0`` — K history h_K=2 (shared) with V fully direct: "k=2, v 直接量化".
* ``k_h2_cross_vh0`` — K history h_K=2 (shared) with V CrossKV h_V=0 (current
  reconstructed K only, no V history): "k=2, v 无历史但用 cross".

The existing variants are untouched; --resume skips anything already complete
and only the new variants get fitted, cached, generated and evaluated.  Runs on
however many GPUs are in the lease (batches jobs by len(gpus)).

Usage mirrors run_final.py; add --until to stop after a phase so generation
can be staged behind other GPU work.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_final as rf  # noqa: E402


EXTRA_VARIANTS = (
    rf.Variant(
        name="k_h0_v_h0",
        axis="direct_quant",
        k_history_length=0,
        v_history_length=0,
        v_predictor_mode="direct",
        k_predictor_mode="direct",
    ),
    rf.Variant(
        name="k_h2_v_h0",
        axis="direct_quant",
        k_history_length=2,
        v_history_length=0,
        v_predictor_mode="direct",
        k_predictor_mode="shared",
    ),
    rf.Variant(
        name="k_h2_cross_vh0",
        axis="cross_vh0",
        k_history_length=2,
        v_history_length=0,
        v_predictor_mode="cross_kv",
        k_predictor_mode="shared",
    ),
)


def _gpus(raw: str | None) -> tuple[str, ...]:
    values = tuple(
        item.strip()
        for item in (raw or os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")
        if item.strip()
    )
    if not values:
        raise ValueError(
            "need GPU ids; set CUDA_VISIBLE_DEVICES or pass --gpus (UUIDs)"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"GPU list contains duplicates: {values!r}")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = rf._parser()
    parser.description = __doc__
    parser.add_argument(
        "--until",
        choices=("fit", "cache", "generation", "quality", "all"),
        default="all",
        help="stop after this phase (default: run through aggregate)",
    )
    return parser


def _aggregate_extra(
    args: argparse.Namespace,
    root: Path,
    variants: tuple[rf.Variant, ...],
    configs: dict[str, rf.CodecConfig],
) -> dict[str, object]:
    """Aggregate summary.json/csv across the 11 existing (50-video) variants and
    the two new 20-video direct-quant variants, using each variant's own report
    video count instead of a single shared expected count."""
    rows = []
    for variant in variants:
        paths = rf._variant_paths(root, variant)
        cache = rf._load_cache(paths["cache"])
        quality = rf._load_quality(paths["quality"])
        expected = args.videos if variant in EXTRA_VARIANTS else 50
        if variant in EXTRA_VARIANTS:
            # Sharded generation overwrites pipeline_report.json with the last
            # shard's counts, so the strict fresh-video check cannot be used.
            # The videos on disk and quality.json's count are authoritative.
            try:
                generation = rf._load_generation(paths["run"], expected)
            except ValueError:
                generation = {
                    "status": "complete",
                    "requested_samples": expected,
                    "generated_samples": expected,
                    "skipped_samples": 0,
                }
        else:
            generation = rf._load_generation(paths["run"], expected)
        if int(quality["videos"]) != expected:
            raise ValueError(
                f"quality report contains {quality['videos']} videos, expected "
                f"{expected}: {paths['quality']}"
            )
        rows.append(
            rf._summary_row(variant, configs[variant.name], cache, quality, generation)
        )
    rf._write_csv(root / "summary.csv", rows)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "study": "self_forcing_history_ablation",
        "protocol": {
            "model": "Self-Forcing",
            "bits": 2,
            "videos_per_variant": args.videos,
            "frames_per_video": args.frames,
            "prompt_range": [args.start_index, args.end_index],
            "seed": args.seed,
            "fit_history_source": args.fit_history_source,
            "gpus": list(args.gpus),
        },
        "rows": rows,
    }
    rf._write_json(root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.gpus = _gpus(args.gpus)
    # fit_params.sh / run_tempokv.sh pick their interpreter via PYTHON_BIN;
    # force it to the interpreter that can import tempokv (this one).
    os.environ["PYTHON_BIN"] = str(Path(sys.executable).resolve())

    for name, path in (
        ("trace", args.trace),
        ("base-config", args.base_config),
        ("checkpoint", args.checkpoint),
        ("wan-model-dir", args.wan_model_dir),
        ("prompts", args.prompts),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    args.videos = int(args.videos)
    args.frames = int(args.frames)
    args.end_index = args.start_index + args.videos

    base_payload = rf._codec_payload(args.base_config)
    base = rf.config_from_mapping(base_payload)
    existing = tuple(
        v
        for v in rf._variants(
            base, k_baseline=args.k_baseline, v_baseline=args.v_baseline
        )
        if not (v.axis == "v_history_no_cross" and v.v_history_length > 1)
    )
    variants = existing + EXTRA_VARIANTS
    configs = {
        variant.name: rf._variant_config(base_payload, variant)
        for variant in variants
    }
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        paths = rf._variant_paths(root, variant)
        rf._write_json(paths["config"], {"codec": rf.asdict(configs[variant.name])})

    fit_jobs = rf._fit_jobs(args, root, variants, configs)
    cache_jobs = rf._cache_jobs(args, root, variants, configs)
    # The 11 existing variants and the bf16 reference already have 50 videos;
    # generate only the new 20-video variants.  --resume keeps existing
    # fits/caches/quality and only the new variants are built, cached and run.
    # Each new variant is sharded across all GPUs (one job per GPU, each with a
    # disjoint prompt slice) so _run_jobs spreads the shards onto the GPUs.
    n_gpus = len(args.gpus)
    generation_jobs = []
    for variant in EXTRA_VARIANTS:
        paths = rf._variant_paths(root, variant)
        if args.resume and rf._generation_complete(paths["run"], args.videos):
            continue
        base_env = rf._base_environment(args)
        shard_videos = -(-args.videos // n_gpus)  # ceil
        for shard in range(n_gpus):
            start = args.start_index + shard * shard_videos
            end = min(start + shard_videos, args.start_index + args.videos)
            if start >= end:
                continue
            generation_jobs.append(
                rf.Job(
                    name=variant.name,
                    phase="generation",
                    command=("bash", str(rf.RUN_TEMPOKV)),
                    environment={
                        **base_env,
                        "CONFIG": str(paths["config"]),
                        "PARAM_DIR": str(paths["params"]),
                        "OUTPUT_DIR": str(paths["run"]),
                        "START_INDEX": str(start),
                        "END_INDEX": str(end),
                    },
                    log_path=root / "logs" / "generation" / f"{variant.name}.shard{shard}.log",
                )
            )
    quality_jobs = rf._quality_jobs(args, root, variants)

    if args.dry_run:
        for phase, jobs in (
            ("fit", fit_jobs),
            ("cache", cache_jobs),
            ("generation", generation_jobs),
            ("quality", quality_jobs),
        ):
            if not jobs:
                continue
            print(f"\n--- {phase} ({len(jobs)} jobs, {len(args.gpus)}-GPU batches) ---")
            rf._run_jobs(jobs, args.gpus, dry_run=True)
        print(f"\n[dry-run] new variants: {[v.name for v in EXTRA_VARIANTS]}")
        return 0

    rf._run_jobs(fit_jobs, args.gpus, dry_run=False)
    if args.until == "fit":
        print("[staged] fit done; re-run with --until generation when GPUs free")
        return 0
    rf._run_jobs(cache_jobs, args.gpus, dry_run=False)
    if args.until == "cache":
        print("[staged] fit+cache done; re-run with --until generation when GPUs free")
        return 0

    batch_size = len(args.gpus)
    for batch_start in range(0, len(generation_jobs), batch_size):
        batch = generation_jobs[batch_start : batch_start + batch_size]
        rf._run_jobs(batch, args.gpus, dry_run=False)
        names = {job.name for job in batch}
        qual = [job for job in quality_jobs if job.name in names]
        if qual:
            rf._run_jobs(qual, args.gpus, dry_run=False)
    if args.until == "generation":
        print("[staged] generation done; re-run with --until quality to aggregate")
        return 0
    remaining = [
        job
        for job in quality_jobs
        if not (root / "runs" / job.name / "quality.json").is_file()
    ]
    if remaining:
        rf._run_jobs(remaining, args.gpus, dry_run=False)

    report = _aggregate_extra(args, root, variants, configs)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n完成：{root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
