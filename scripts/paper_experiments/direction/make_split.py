#!/usr/bin/env python3
"""Split one collected trace into disjoint calibration and held-out traces.

The direction study needs two traces that share a collection run but no
prompts: fitting on one and scoring on the other is what makes the numbers
about generalization rather than about fit.  A trace bridge produces a single
archive, so the split happens here.

The slicing itself is the calibration study's ``PoolTrace``, imported rather
than reimplemented -- a second copy of "which records belong to which prompt"
is exactly the kind of thing that drifts.

Two ways to say how the prompts divide:

    # named groups, from calibration/build_groups.py
    python scripts/paper_experiments/direction/make_split.py \\
        --trace data/calibration/self-forcing_pool.npz \\
        --prompt-groups data/calibration/prompt_groups.json \\
        --output-dir data/direction

    # a plain positional split, for a trace collected without a groups file
    python scripts/paper_experiments/direction/make_split.py \\
        --trace data/self-forcing/trace/clean_trace.npz \\
        --prompt-count 12 --calibration-prompts 8 \\
        --output-dir data/direction
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
CALIBRATION_PROTOCOL = HERE.parent / "calibration" / "protocol.py"


def _calibration_protocol():
    spec = importlib.util.spec_from_file_location(
        "direction_calibration_protocol", CALIBRATION_PROTOCOL
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {CALIBRATION_PROTOCOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    groups = parser.add_argument_group("prompt division")
    groups.add_argument(
        "--prompt-groups",
        type=Path,
        help="manifest from calibration/build_groups.py",
    )
    groups.add_argument("--calibration-group", default="calibration")
    groups.add_argument("--heldout-group", default="heldout")
    groups.add_argument(
        "--prompt-count",
        type=int,
        help="number of prompts in the trace, for a positional split",
    )
    groups.add_argument(
        "--calibration-prompts",
        type=int,
        help="leading prompts used for calibration; the rest are held out",
    )
    parser.add_argument("--calibration-name", default="calibration.npz")
    parser.add_argument("--heldout-name", default="heldout.npz")
    return parser


def run(args: argparse.Namespace) -> dict:
    protocol = _calibration_protocol()
    if args.prompt_groups is not None:
        if args.prompt_count is not None or args.calibration_prompts is not None:
            raise ValueError(
                "use either --prompt-groups or --prompt-count/--calibration-prompts, "
                "not both"
            )
        groups = protocol.PromptGroups.load(args.prompt_groups)
        groups.require_disjoint(args.calibration_group, args.heldout_group)
        calibration = groups.group(args.calibration_group)
        heldout = groups.group(args.heldout_group)
        prompt_count = groups.prompt_count
        source = str(args.prompt_groups)
    else:
        if args.prompt_count is None or args.calibration_prompts is None:
            raise ValueError(
                "a positional split needs both --prompt-count and "
                "--calibration-prompts"
            )
        if not 0 < args.calibration_prompts < args.prompt_count:
            raise ValueError(
                "--calibration-prompts must leave at least one held-out prompt"
            )
        prompt_count = args.prompt_count
        calibration = tuple(range(args.calibration_prompts))
        heldout = tuple(range(args.calibration_prompts, prompt_count))
        source = "positional"

    pool = protocol.PoolTrace.load(args.trace, prompt_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = pool.write_slice(
        args.output_dir / args.calibration_name, calibration
    )
    heldout_path = pool.write_slice(args.output_dir / args.heldout_name, heldout)
    report = {
        "source_trace": str(args.trace.resolve()),
        "prompt_division": source,
        "prompt_count": prompt_count,
        "records_per_prompt": pool.rows_per_prompt,
        "stride": pool.stride,
        "calibration": {
            "path": str(calibration_path),
            "prompts": list(calibration),
        },
        "heldout": {"path": str(heldout_path), "prompts": list(heldout)},
    }
    (args.output_dir / "split.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"calibration  {report['calibration']['path']}")
    print(f"             {len(report['calibration']['prompts'])} prompts")
    print(f"held out     {report['heldout']['path']}")
    print(f"             {len(report['heldout']['prompts'])} prompts")
    print(f"records per prompt: {report['records_per_prompt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
