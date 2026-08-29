#!/usr/bin/env python3
"""Module 1 — how many calibration prompts a TempoKV predictor needs.

Self-Forcing, INT2, one fixed quantizer and anchor schedule.  The predictor is
refit from N in {1,2,4,8,16,32} calibration prompts and scored on a fixed
held-out set of 32 prompts that no fit ever sees.  K and V normalized
reconstruction MSE are reported separately, because the two predictors have
different structure and saturate at different points.

Every size is measured over several disjoint draws from the calibration pool,
so the curve reports the spread across prompt choices rather than one lucky
sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol import (  # noqa: E402
    PoolTrace,
    PromptGroups,
    add_common_arguments,
    provenance,
    sweep_sizes,
    write_report,
)


DEFAULT_SIZES = (1, 2, 4, 8, 16, 32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--model", default="self-forcing")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="calibration prompt counts to measure",
    )
    parser.add_argument("--calibration-group", default="calibration")
    parser.add_argument("--heldout-group", default="heldout")
    parser.add_argument(
        "--report-name",
        default="calibration_size.json",
        help="report file written inside --output-dir",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    groups = PromptGroups.load(args.prompt_groups)
    pool = PoolTrace.load(args.pool_trace, groups.prompt_count)
    sweep = sweep_sizes(
        pool,
        groups,
        args.sizes,
        calibration_group=args.calibration_group,
        heldout_group=args.heldout_group,
        output_dir=args.output_dir,
        repeats=args.repeats,
        seed=args.seed,
        fit_device=args.fit_device,
        eval_device=args.eval_device,
        iterations=args.iterations,
        python_bin=args.python_bin,
        keep_artifacts=args.keep_artifacts,
    )
    payload = {
        "module": "calibration_size",
        **provenance(pool, groups, model=args.model),
        **sweep,
    }
    write_report(args.output_dir / args.report_name, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    for point in payload["points"]:
        print(
            f"N={point['calibration_size']:>3}  "
            f"K nmse={point['k_nmse']['mean']:.6f}  "
            f"V nmse={point['v_nmse']['mean']:.6f}  "
            f"(n={point['k_nmse']['samples']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
