#!/usr/bin/env python3
"""Minimal complete run of the long-horizon reconstruction study.

Runs the measurement and the figure end to end on CPU, in seconds, against a
short synthetic calibration trace and a longer synthetic horizon trace, and
checks the collection script's frame arithmetic without a GPU.

    python scripts/paper_experiments/long_horizon/run_minimal.py \\
        --output-root /tmp/long-horizon-minimal

The numbers describe the synthetic generator and say nothing about TempoKV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from methods import METHODS  # noqa: E402


LAYERS, RECORDS, HEADS, DIM, STRIDE = 3, 6, 2, 128, 4
CALIBRATION_FRAMES = 12
HORIZON_FRAMES = 60


def synthesize(path: Path, frames: int, *, seed: int) -> Path:
    """A trace whose statistics drift, so the horizon has something to show."""
    rng = np.random.default_rng(seed)
    shape = (LAYERS, RECORDS, HEADS, DIM)
    keys = np.zeros((LAYERS, RECORDS, frames, HEADS, DIM), dtype=np.float32)
    values = np.zeros_like(keys)
    keys[:, :, 0] = rng.normal(size=shape)
    values[:, :, 0] = rng.normal(size=shape)
    for frame in range(1, frames):
        drift = 1.0 + 0.002 * frame
        keys[:, :, frame] = (
            0.9 * keys[:, :, frame - 1] + 0.1 * drift * rng.normal(size=shape)
        )
        values[:, :, frame] = (
            0.6 * keys[:, :, frame]
            + 0.3 * values[:, :, frame - 1]
            + 0.1 * drift * rng.normal(size=shape)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        keys=keys,
        values=values,
        layer_ids=np.arange(LAYERS, dtype=np.int32),
        trace_stride=np.asarray(STRIDE, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
        cross_source_kind=np.asarray("normalized"),
        trace_role=np.asarray("calibration"),
    )
    return path


def _run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    if subprocess.run(command, text=True).returncode != 0:
        raise SystemExit(f"step failed: {' '.join(command)}")


def check_collection_arithmetic() -> dict[str, int]:
    """Dry-run the collector so its frame conversion is checked without a GPU."""
    horizons = {}
    for frames in (105, 357, 1401):
        completed = subprocess.run(
            ["bash", str(HERE / "collect_trace.sh")],
            env={
                **dict(__import__("os").environ),
                "CHECKPOINT": "/nonexistent",
                "WAN_MODEL_DIR": "/nonexistent",
                "PROMPTS": str(PROJECT_ROOT / "assets" / "moviegen128_first30.txt"),
                "HORIZON_FRAMES": str(frames),
                "TRACE_DIR": "/tmp/long-horizon-dryrun",
                "DRY_RUN": "1",
            },
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"collection dry run failed for {frames} frames:\n{completed.stderr}"
            )
        latent = None
        for line in completed.stdout.splitlines():
            if "TRACE_FRAMES=" in line:
                latent = int(line.split("TRACE_FRAMES=")[1].split()[0])
        if latent is None:
            raise SystemExit(f"collection dry run did not report TRACE_FRAMES:\n{completed.stdout}")
        horizons[frames] = latent
    # The documented conversion, restated independently as a cross-check.
    for frames, latent in horizons.items():
        expected = (frames + 3) // 4
        if latent != expected:
            raise SystemExit(
                f"{frames} video frames mapped to {latent} latent frames, expected {expected}"
            )
    return horizons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.output_root.expanduser().resolve()

    print("== collection: frame arithmetic (dry run, no GPU) ==")
    horizons = check_collection_arithmetic()
    for frames, latent in sorted(horizons.items()):
        print(f"  {frames:>5} video frames -> {latent:>4} latent frames")

    calibration = synthesize(
        root / "traces" / "calibration.npz", CALIBRATION_FRAMES, seed=args.seed
    )
    horizon = synthesize(
        root / "traces" / "horizon.npz", HORIZON_FRAMES, seed=args.seed + 1
    )

    report_dir = root / "reconstruction"
    _run(
        [args.python_bin, str(HERE / "run_reconstruction.py"),
         "--calibration-trace", str(calibration),
         "--horizon-trace", str(horizon),
         "--output-dir", str(report_dir),
         "--methods", *[item.name for item in METHODS]]
    )
    _run(
        [args.python_bin, str(HERE / "plot_figure.py"),
         "--report", str(report_dir / "long_horizon_reconstruction.json"),
         "--output-dir", str(root / "figure")]
    )

    expected = [
        report_dir / "long_horizon_reconstruction.json",
        root / "figure" / "reconstruction_by_frame.csv",
        root / "figure" / "reconstruction_summary.csv",
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise SystemExit(
            "the run finished but did not produce: "
            + ", ".join(str(path) for path in missing)
        )
    payload = json.loads(expected[0].read_text(encoding="utf-8"))
    if payload["frames"] != HORIZON_FRAMES:
        raise SystemExit(f"expected {HORIZON_FRAMES} frames, got {payload['frames']}")
    covered = {item["method"] for item in payload["methods"]}
    if covered != {item.name for item in METHODS}:
        raise SystemExit(f"report is missing methods: {covered}")

    print("\nminimal long-horizon study complete")
    for path in expected:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
