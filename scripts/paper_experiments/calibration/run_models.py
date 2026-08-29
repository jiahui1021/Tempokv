#!/usr/bin/env python3
"""Module 3 — does the calibration law hold on other AR video models?

Predictor parameters are deliberately *not* transferred between models.  KV
representations differ in layer count, head count and head dimension, so a
transferred matrix would measure the mismatch, not the calibration protocol.
What is transferred is the protocol: each of Self-Forcing, Causal-Forcing and
LongCat-Video refits its own predictor from N in {2,4,8,32} of its own
calibration prompts and is scored on its own held-out prompts.

Absolute NMSE is not comparable across models, so the reported curve is

    MSE(N) / MSE(N_max)

per model and per tensor.  A shared shape in that normalized curve is the
claim: the saturation point is a property of the method, not of Self-Forcing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol import (  # noqa: E402
    PoolTrace,
    PromptGroups,
    provenance,
    sweep_sizes,
    write_report,
)


DEFAULT_SIZES = (2, 4, 8, 32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=POOL_TRACE:PROMPT_GROUPS",
        help=(
            "one model per flag, e.g. "
            "--model self-forcing=data/self-forcing/pool.npz:configs/groups.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--calibration-group", default="calibration")
    parser.add_argument("--heldout-group", default="heldout")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--fit-device", default="auto")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--report-name", default="calibration_across_models.json")
    return parser


def _parse_model(spec: str) -> tuple[str, Path, Path]:
    name, _, rest = spec.partition("=")
    pool, _, groups = rest.partition(":")
    if not name or not pool or not groups:
        raise ValueError(
            f"model spec {spec!r} must be NAME=POOL_TRACE:PROMPT_GROUPS"
        )
    return name, Path(pool), Path(groups)


def _normalize(points: list[dict], reference_size: int) -> list[dict]:
    """Rescale each tensor's curve by its own value at the reference size."""
    baseline = {
        tensor: next(
            point[tensor]["mean"]
            for point in points
            if point["calibration_size"] == reference_size
        )
        for tensor in ("k_nmse", "v_nmse")
    }
    rows = []
    for point in points:
        row = {"calibration_size": point["calibration_size"]}
        for tensor in ("k_nmse", "v_nmse"):
            reference = baseline[tensor]
            row[f"{tensor}_mean"] = point[tensor]["mean"]
            row[f"{tensor}_ratio"] = (
                point[tensor]["mean"] / reference if reference > 0.0 else None
            )
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> dict:
    sizes = sorted({int(size) for size in args.sizes})
    reference_size = sizes[-1]
    models: list[dict] = []
    for spec in args.model:
        name, pool_path, groups_path = _parse_model(spec)
        print(f"[across-models] {name}", file=sys.stderr, flush=True)
        groups = PromptGroups.load(groups_path)
        pool = PoolTrace.load(pool_path, groups.prompt_count)
        sweep = sweep_sizes(
            pool,
            groups,
            sizes,
            calibration_group=args.calibration_group,
            heldout_group=args.heldout_group,
            output_dir=args.output_dir / name,
            repeats=args.repeats,
            seed=args.seed,
            fit_device=args.fit_device,
            eval_device=args.eval_device,
            iterations=args.iterations,
            python_bin=args.python_bin,
            keep_artifacts=args.keep_artifacts,
        )
        models.append(
            {
                **provenance(pool, groups, model=name),
                **sweep,
                "normalized": _normalize(sweep["points"], reference_size),
            }
        )
    payload = {
        "module": "calibration_across_models",
        "schema_version": 1,
        "sizes": sizes,
        "reference_size": reference_size,
        "models": models,
    }
    write_report(args.output_dir / args.report_name, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    reference = payload["reference_size"]
    for model in payload["models"]:
        print(f"\n{model['model']}  (ratios against N={reference})")
        for row in model["normalized"]:
            print(
                f"  N={row['calibration_size']:>3}  "
                f"K {row['k_nmse_ratio']:.3f}x  V {row['v_nmse_ratio']:.3f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
