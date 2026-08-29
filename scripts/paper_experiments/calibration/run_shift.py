#!/usr/bin/env python3
"""Module 2 — does the calibration prompt distribution matter?

The pool is split into two deliberately dissimilar halves: A (human / action)
and B (scene / object).  Each half is split again into calibration and
evaluation prompts, so all four cells of the 2x2

    A->A   A->B
    B->A   B->B

fit and evaluate on strictly disjoint prompts.  Every cell calibrates on the
same number of prompts, so a difference between cells cannot come from
calibration size.

Reported as the AWQ-style table: K and V normalized reconstruction MSE per
cell, plus the relative error increase a cross-distribution calibration set
costs against the matched-distribution set on the same evaluation prompts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol import (  # noqa: E402
    PoolTrace,
    PromptGroups,
    add_common_arguments,
    draw_subsets,
    measure,
    provenance,
    summarize,
    write_codec_config,
    write_report,
)


DISTRIBUTIONS = ("action", "scene")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--model", default="self-forcing")
    parser.add_argument(
        "--calibration-size",
        type=int,
        help=(
            "prompts per calibration set; defaults to the largest size both "
            "distributions can supply, so the cells stay comparable"
        ),
    )
    parser.add_argument(
        "--distributions",
        nargs=2,
        default=list(DISTRIBUTIONS),
        metavar=("A", "B"),
        help="group name stems; each needs a _calibration and a _heldout group",
    )
    parser.add_argument("--report-name", default="calibration_shift.json")
    return parser


def run(args: argparse.Namespace) -> dict:
    groups = PromptGroups.load(args.prompt_groups)
    pool = PoolTrace.load(args.pool_trace, groups.prompt_count)
    first, second = args.distributions
    if first == second:
        raise ValueError("the two distributions must be different groups")
    calibration_groups = {name: f"{name}_calibration" for name in (first, second)}
    heldout_groups = {name: f"{name}_heldout" for name in (first, second)}
    for name in (first, second):
        # A cell is only evidence about distribution shift if its own prompts
        # never leak between fitting and scoring.
        groups.require_disjoint(calibration_groups[name], heldout_groups[name])
    groups.require_disjoint(calibration_groups[first], heldout_groups[second])
    groups.require_disjoint(calibration_groups[second], heldout_groups[first])

    available = min(len(groups.group(calibration_groups[name])) for name in (first, second))
    size = int(args.calibration_size) if args.calibration_size else available
    if size > available:
        raise ValueError(
            f"calibration size {size} exceeds the {available} prompts the "
            "smaller distribution can supply"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = write_codec_config(args.output_dir / "codec.json", pool.stride)

    heldout_traces = {}
    for name in (first, second):
        path = args.output_dir / f"heldout_{name}.npz"
        pool.write_slice(path, groups.group(heldout_groups[name]))
        heldout_traces[name] = path

    cells: list[dict] = []
    try:
        for calibration_name in (first, second):
            draws = draw_subsets(
                groups.group(calibration_groups[calibration_name]),
                size,
                args.repeats,
                seed=args.seed,
            )
            for evaluation_name in (first, second):
                measurements = []
                for index, calibration in enumerate(draws):
                    print(
                        f"[shift] {calibration_name}->{evaluation_name} "
                        f"draw {index + 1}/{len(draws)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    measurements.append(
                        measure(
                            pool,
                            calibration,
                            heldout_traces[evaluation_name],
                            config,
                            args.output_dir
                            / "work"
                            / f"{calibration_name}_{evaluation_name}_r{index}",
                            fit_device=args.fit_device,
                            eval_device=args.eval_device,
                            iterations=args.iterations,
                            python_bin=args.python_bin,
                            keep_artifacts=args.keep_artifacts,
                        )
                    )
                cells.append(
                    {
                        "calibration": calibration_name,
                        "evaluation": evaluation_name,
                        "matched": calibration_name == evaluation_name,
                        "draws": [item.as_dict() for item in measurements],
                        "k_nmse": summarize(item.k_nmse for item in measurements),
                        "v_nmse": summarize(item.v_nmse for item in measurements),
                    }
                )
    finally:
        if not args.keep_artifacts:
            for path in heldout_traces.values():
                path.unlink(missing_ok=True)
            shutil.rmtree(args.output_dir / "work", ignore_errors=True)

    payload = {
        "module": "calibration_shift",
        **provenance(pool, groups, model=args.model),
        "distributions": [first, second],
        "calibration_size": size,
        "repeats": args.repeats,
        "seed": args.seed,
        "cells": cells,
        "error_increase": _error_increase(cells, first, second),
    }
    write_report(args.output_dir / args.report_name, payload)
    return payload


def _error_increase(cells: list[dict], first: str, second: str) -> list[dict]:
    """Relative cost of calibrating off-distribution, per evaluation set.

    Held-out sets differ in intrinsic difficulty, so a cross-distribution cell
    is only meaningful against the matched cell on the *same* evaluation
    prompts.  That ratio is the number the 2x2 table is read for.
    """
    by_cell = {(cell["calibration"], cell["evaluation"]): cell for cell in cells}
    rows = []
    for evaluation in (first, second):
        matched = by_cell[(evaluation, evaluation)]
        crossed = by_cell[(second if evaluation == first else first, evaluation)]
        row = {
            "evaluation": evaluation,
            "matched_calibration": matched["calibration"],
            "crossed_calibration": crossed["calibration"],
        }
        for tensor in ("k_nmse", "v_nmse"):
            baseline = matched[tensor]["mean"]
            row[f"{tensor}_matched"] = baseline
            row[f"{tensor}_crossed"] = crossed[tensor]["mean"]
            row[f"{tensor}_increase"] = (
                crossed[tensor]["mean"] / baseline - 1.0 if baseline > 0.0 else None
            )
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    print(f"calibration size per cell: {payload['calibration_size']}")
    for cell in payload["cells"]:
        print(
            f"{cell['calibration']:>7} -> {cell['evaluation']:<7} "
            f"K nmse={cell['k_nmse']['mean']:.6f}  "
            f"V nmse={cell['v_nmse']['mean']:.6f}"
        )
    for row in payload["error_increase"]:
        k = row["k_nmse_increase"]
        v = row["v_nmse_increase"]
        print(
            f"eval={row['evaluation']:<7} cross-distribution increase  "
            f"K={k:+.2%}  V={v:+.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
