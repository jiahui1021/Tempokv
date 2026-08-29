#!/usr/bin/env python3
"""Reconstruction half of the K->V direction study.

Fits all three arms on one calibration trace and scores them on a held-out
trace, reporting K and V normalized reconstruction MSE separately plus the
comparisons the design argument rests on.

    python scripts/paper_experiments/direction/run_reconstruction.py \\
        --calibration-trace data/self-forcing/trace/clean_trace.npz \\
        --evaluation-trace data/self-forcing/trace/heldout_trace.npz \\
        --output-dir results/direction/reconstruction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from protocol import ARMS, SCHEMA_VERSION, arm, measure_arm, summarize  # noqa: E402


def _load_trace(path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Read a fitting trace as [L,...,T,H,D] float32 K/V plus its stride."""
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in ("keys", "values") if name not in archive]
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        keys = torch.from_numpy(np.array(archive["keys"], copy=True)).float()
        values = torch.from_numpy(np.array(archive["values"], copy=True)).float()
        stride = None
        for name in ("trace_stride", "stride"):
            if name in archive:
                stride = int(np.asarray(archive[name]).item())
                break
    if keys.shape != values.shape:
        raise ValueError(f"{path} K and V must share a shape")
    if stride is None:
        raise ValueError(f"{path} does not record a trace stride")
    return keys, values, stride


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument(
        "--evaluation-trace",
        type=Path,
        help=(
            "held-out trace; defaults to the calibration trace, which measures "
            "fit rather than generalization and is only useful for smoke runs"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[item.name for item in ARMS],
        help="subset of arms to run",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--temporal-ridge", type=float, default=1.0)
    parser.add_argument("--cross-ridge", type=float, default=0.1)
    parser.add_argument("--model", default="self-forcing")
    parser.add_argument("--report-name", default="direction_reconstruction.json")
    return parser


def run(args: argparse.Namespace) -> dict:
    calibration = _load_trace(args.calibration_trace)
    evaluation_path = args.evaluation_trace or args.calibration_trace
    evaluation = _load_trace(evaluation_path)
    if calibration[2] != evaluation[2]:
        raise ValueError(
            f"calibration stride {calibration[2]} does not match evaluation "
            f"stride {evaluation[2]}; the two traces are from different runs"
        )
    stride = calibration[2]
    results = []
    for name in args.arms:
        arm_spec = arm(name)
        print(f"[direction] {arm_spec.name}", file=sys.stderr, flush=True)
        results.append(
            measure_arm(
                arm_spec,
                calibration[:2],
                evaluation[:2],
                stride,
                iterations=args.iterations,
                temporal_ridge=args.temporal_ridge,
                cross_ridge=args.cross_ridge,
            )
        )
    payload = {
        "module": "direction_reconstruction",
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "calibration_trace": str(args.calibration_trace.resolve()),
        "evaluation_trace": str(Path(evaluation_path).resolve()),
        "held_out": str(Path(evaluation_path).resolve())
        != str(args.calibration_trace.resolve()),
        "stride": stride,
        "trace_shape": list(calibration[0].shape),
        "arms": results,
        **summarize(results),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / args.report_name
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    if not payload["held_out"]:
        print(
            "warning: evaluated on the calibration trace, so these numbers "
            "measure fit rather than generalization",
            file=sys.stderr,
        )
    print(f"{'arm':<16}{'K NMSE':>12}{'V NMSE':>12}{'total':>12}")
    for item in payload["arms"]:
        total = item["k_nmse"] + item["v_nmse"]
        print(
            f"{item['arm']:<16}{item['k_nmse']:>12.6f}"
            f"{item['v_nmse']:>12.6f}{total:>12.6f}"
        )
    comparison = payload["direction_comparison"]
    if comparison is not None:
        print(
            f"\nmirror penalty (v_to_k vs k_to_v, total NMSE): "
            f"{comparison['mirror_penalty']:+.2%}"
        )
        print(f"preferred direction: {comparison['preferred_direction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
