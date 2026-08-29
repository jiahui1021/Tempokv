#!/usr/bin/env python3
"""Minimal complete run of the calibration study.

This executes all three modules and the figure end to end, on CPU, in seconds,
against small synthetic pool traces.  It is the real pipeline -- the shipped
fitter, the shipped closed-loop reconstruction evaluator and the shipped
plotting code -- driven at a size that needs no GPU, no checkpoint and no
collected trace.

Its purpose is to prove the wiring before a real run spends GPU hours: if this
passes, the only thing a production run changes is which traces are read.  The
numbers it prints are properties of the synthetic generator and say nothing
about TempoKV.

    python scripts/paper_experiments/calibration/run_minimal.py \
        --output-root /tmp/calibration-minimal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from protocol import PromptGroups  # noqa: E402


# Three stand-in models with deliberately different KV geometry, because the
# point of module 3 is that the protocol does not depend on the shape.
MODELS = {
    "self-forcing": {"layers": 4, "frames": 6, "heads": 2, "dim": 128, "stride": 1560,
                     "records_per_prompt": 4},
    "causal-forcing": {"layers": 3, "frames": 6, "heads": 2, "dim": 128, "stride": 1560,
                       "records_per_prompt": 4},
    "longcat": {"layers": 2, "frames": 6, "heads": 4, "dim": 128, "stride": 1024,
                "records_per_prompt": 2},
}

POOL_PROMPTS = 16
CALIBRATION_PROMPTS = 8
SIZE_GRID = (1, 2, 4, 8)
MODEL_SIZE_GRID = (2, 4, 8)

# Two prompt "distributions" with different temporal dynamics, so module 2 has
# a real shift to detect rather than two relabelled copies of one process.
DISTRIBUTION_DYNAMICS = {
    "action": {"decay": 0.90, "noise": 0.10, "cross": 0.50, "recurrence": 0.40},
    "scene": {"decay": 0.55, "noise": 0.35, "cross": 0.20, "recurrence": 0.70},
}


def distribution_for(position: int) -> str:
    """Pool positions alternate between the two distributions."""
    return "action" if position % 2 == 0 else "scene"


def synthesize_pool(spec: dict[str, int], path: Path, *, seed: int) -> Path:
    """Write one synthetic pool trace in the plain [L,S,T,H,D] fitting layout."""
    rng = np.random.default_rng(seed)
    layers, frames = spec["layers"], spec["frames"]
    heads, dim = spec["heads"], spec["dim"]
    records = spec["records_per_prompt"]
    rows = POOL_PROMPTS * records
    keys = np.zeros((layers, rows, frames, heads, dim), dtype=np.float32)
    values = np.zeros_like(keys)
    for prompt in range(POOL_PROMPTS):
        dynamics = DISTRIBUTION_DYNAMICS[distribution_for(prompt)]
        block = slice(prompt * records, (prompt + 1) * records)
        shape = (layers, records, heads, dim)
        keys[:, block, 0] = rng.normal(size=shape)
        values[:, block, 0] = rng.normal(size=shape)
        for frame in range(1, frames):
            keys[:, block, frame] = (
                dynamics["decay"] * keys[:, block, frame - 1]
                + dynamics["noise"] * rng.normal(size=shape)
            )
            values[:, block, frame] = (
                dynamics["cross"] * keys[:, block, frame]
                + dynamics["recurrence"] * values[:, block, frame - 1]
                + dynamics["noise"] * rng.normal(size=shape)
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        keys=keys,
        values=values,
        layer_ids=np.arange(layers, dtype=np.int32),
        trace_stride=np.asarray(spec["stride"], dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
        cross_source_kind=np.asarray("normalized"),
        trace_role=np.asarray("calibration"),
    )
    return path


def write_groups(path: Path) -> Path:
    """Mirror the layout build_groups.py produces, for the synthetic pool."""
    calibration = list(range(CALIBRATION_PROMPTS))
    heldout = list(range(CALIBRATION_PROMPTS, POOL_PROMPTS))
    groups = {"calibration": calibration, "heldout": heldout}
    for name in ("action", "scene"):
        groups[f"{name}_calibration"] = [
            p for p in calibration if distribution_for(p) == name
        ]
        groups[f"{name}_heldout"] = [
            p for p in heldout if distribution_for(p) == name
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "prompts_file": "<synthetic>",
                "prompt_count": POOL_PROMPTS,
                "prompt_indices": list(range(POOL_PROMPTS)),
                "groups": groups,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}", flush=True)
    completed = subprocess.run(command, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"step failed: {printable}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.output_root.expanduser().resolve()
    traces = root / "traces"
    reports = root / "reports"

    groups_path = write_groups(root / "prompt_groups.json")
    # Reject a malformed manifest here rather than inside the first fit.
    PromptGroups.load(groups_path)
    pools = {
        name: synthesize_pool(spec, traces / f"{name}.npz", seed=args.seed + index)
        for index, (name, spec) in enumerate(MODELS.items())
    }
    print(f"synthetic pools: {', '.join(str(path) for path in pools.values())}")

    shared = [
        "--prompt-groups", str(groups_path),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--fit-device", "cpu",
        "--eval-device", "cpu",
        "--python-bin", args.python_bin,
    ]

    _run(
        [args.python_bin, str(HERE / "run_size.py"),
         "--pool-trace", str(pools["self-forcing"]),
         "--output-dir", str(reports / "size"),
         "--sizes", *[str(size) for size in SIZE_GRID],
         *shared]
    )
    _run(
        [args.python_bin, str(HERE / "run_shift.py"),
         "--pool-trace", str(pools["self-forcing"]),
         "--output-dir", str(reports / "shift"),
         *shared]
    )
    _run(
        [args.python_bin, str(HERE / "run_models.py"),
         "--output-dir", str(reports / "models"),
         "--sizes", *[str(size) for size in MODEL_SIZE_GRID],
         *[f"--model={name}={path}:{groups_path}" for name, path in pools.items()],
         # run_models.py takes the groups per model inside each --model spec,
         # so it does not accept the shared --prompt-groups flag.
         "--repeats", str(args.repeats),
         "--seed", str(args.seed),
         "--fit-device", "cpu",
         "--eval-device", "cpu",
         "--python-bin", args.python_bin]
    )
    _run(
        [args.python_bin, str(HERE / "plot_figure.py"),
         "--size-report", str(reports / "size" / "calibration_size.json"),
         "--shift-report", str(reports / "shift" / "calibration_shift.json"),
         "--models-report", str(reports / "models" / "calibration_across_models.json"),
         "--output-dir", str(root / "figure")]
    )

    expected = [
        reports / "size" / "calibration_size.json",
        reports / "shift" / "calibration_shift.json",
        reports / "models" / "calibration_across_models.json",
        root / "figure" / "panel_a_calibration_size.csv",
        root / "figure" / "panel_b_distribution_shift.csv",
        root / "figure" / "panel_c_across_models.csv",
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise SystemExit(
            "the run finished but did not produce: "
            + ", ".join(str(path) for path in missing)
        )
    print("\nminimal calibration study complete")
    for path in expected:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
