#!/usr/bin/env python3
"""Render the three-panel calibration figure and its backing CSVs.

    (a) calibration size on Self-Forcing
    (b) calibration distribution shift on Self-Forcing
    (c) calibration efficiency across models

The CSVs are always written.  The figure needs matplotlib and is skipped, with
a message, when it is not installed, so a headless fitting host can still
produce the numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


TENSORS = ("k_nmse", "v_nmse")
TENSOR_LABELS = {"k_nmse": "K", "v_nmse": "V"}


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return payload


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fields))
        writer.writeheader()
        writer.writerows(rows)


def size_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for point in report["points"]:
        row: dict[str, Any] = {"calibration_size": point["calibration_size"]}
        for tensor in TENSORS:
            row[f"{tensor}_mean"] = point[tensor]["mean"]
            row[f"{tensor}_std"] = point[tensor]["std"]
            row[f"{tensor}_samples"] = point[tensor]["samples"]
        rows.append(row)
    return rows


def shift_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "calibration": cell["calibration"],
            "evaluation": cell["evaluation"],
            "matched": cell["matched"],
            **{f"{tensor}_mean": cell[tensor]["mean"] for tensor in TENSORS},
            **{f"{tensor}_std": cell[tensor]["std"] for tensor in TENSORS},
        }
        for cell in report["cells"]
    ]


def model_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in report["models"]:
        for row in model["normalized"]:
            rows.append({"model": model["model"], **row})
    return rows


def _panel_a(axis, report: dict[str, Any]) -> None:
    sizes = [point["calibration_size"] for point in report["points"]]
    for tensor in TENSORS:
        means = [point[tensor]["mean"] for point in report["points"]]
        errors = [point[tensor]["std"] for point in report["points"]]
        axis.errorbar(
            sizes, means, yerr=errors, marker="o", capsize=3, label=TENSOR_LABELS[tensor]
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(sizes)
    axis.set_xticklabels([str(size) for size in sizes])
    axis.set_xlabel("calibration prompts $N$")
    axis.set_ylabel("held-out normalized reconstruction MSE")
    axis.set_title("(a) Calibration size (Self-Forcing, INT2)")
    axis.legend(title="tensor")
    axis.grid(alpha=0.3)


def _panel_b(axis, report: dict[str, Any]) -> None:
    """Draw the 2x2 as the relative cost of calibrating off-distribution.

    Colouring by raw NMSE would mostly show which held-out set is intrinsically
    harder, which is a property of the prompts and not of calibration.  Each
    cell is therefore shown against the matched-distribution cell in its own
    evaluation column, so the diagonal is zero by construction and the
    off-diagonal reads directly as the error increase shift costs.
    """
    first, second = report["distributions"]
    order = (first, second)
    by_cell = {(c["calibration"], c["evaluation"]): c for c in report["cells"]}
    increases = []
    labels = []
    for calibration in order:
        increase_row = []
        label_row = []
        for evaluation in order:
            cell = by_cell[(calibration, evaluation)]
            matched = by_cell[(evaluation, evaluation)]
            baseline = matched["k_nmse"]["mean"]
            increase = (
                cell["k_nmse"]["mean"] / baseline - 1.0 if baseline > 0.0 else 0.0
            )
            increase_row.append(increase)
            label_row.append(
                f"K {cell['k_nmse']['mean']:.4f}\n"
                f"V {cell['v_nmse']['mean']:.4f}\n"
                f"{increase:+.1%}"
            )
        increases.append(increase_row)
        labels.append(label_row)

    limit = max(abs(value) for row in increases for value in row) or 1.0
    image = axis.imshow(increases, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(order)), [f"eval {name}" for name in order])
    axis.set_yticks(range(len(order)), [f"calib {name}" for name in order])
    for i, row in enumerate(labels):
        for j, label in enumerate(row):
            axis.text(j, i, label, ha="center", va="center", fontsize=8)
    axis.set_title(
        f"(b) Calibration distribution shift (Self-Forcing)\n"
        f"$N={report['calibration_size']}$ per cell",
        fontsize=10,
    )
    axis.figure.colorbar(
        image, ax=axis, fraction=0.046, label="K NMSE increase vs matched calibration"
    )


def _panel_c(axis, report: dict[str, Any]) -> None:
    for model in report["models"]:
        sizes = [row["calibration_size"] for row in model["normalized"]]
        ratios = [row["k_nmse_ratio"] for row in model["normalized"]]
        axis.plot(sizes, ratios, marker="o", label=model["model"])
    axis.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    axis.set_xscale("log", base=2)
    if report["models"]:
        sizes = [row["calibration_size"] for row in report["models"][0]["normalized"]]
        axis.set_xticks(sizes)
        axis.set_xticklabels([str(size) for size in sizes])
    axis.set_xlabel("calibration prompts $N$")
    axis.set_ylabel(f"K NMSE$(N)$ / K NMSE$({report['reference_size']})$")
    axis.set_title("(c) Calibration efficiency across models")
    axis.legend()
    axis.grid(alpha=0.3)


def render(
    size_report: dict[str, Any],
    shift_report: dict[str, Any],
    models_report: dict[str, Any],
    output: Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the host env
        raise RuntimeError("plotting requires matplotlib") from error

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    _panel_a(axes[0], size_report)
    _panel_b(axes[1], shift_report)
    _panel_c(axes[2], models_report)
    figure.tight_layout()
    written = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        path = output.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=220)
        written.append(path)
    plt.close(figure)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-report", type=Path, required=True)
    parser.add_argument("--shift-report", type=Path, required=True)
    parser.add_argument("--models-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-name", default="calibration")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    size_report = _load(args.size_report)
    shift_report = _load(args.shift_report)
    models_report = _load(args.models_report)

    _write_csv(
        args.output_dir / "panel_a_calibration_size.csv",
        size_rows(size_report),
        ["calibration_size"]
        + [f"{t}_{s}" for t in TENSORS for s in ("mean", "std", "samples")],
    )
    _write_csv(
        args.output_dir / "panel_b_distribution_shift.csv",
        shift_rows(shift_report),
        ["calibration", "evaluation", "matched"]
        + [f"{t}_{s}" for t in TENSORS for s in ("mean", "std")],
    )
    _write_csv(
        args.output_dir / "panel_c_across_models.csv",
        model_rows(models_report),
        ["model", "calibration_size"]
        + [f"{t}_{s}" for t in TENSORS for s in ("mean", "ratio")],
    )
    try:
        written = render(
            size_report,
            shift_report,
            models_report,
            args.output_dir / args.figure_name,
            formats=args.formats,
        )
    except RuntimeError as error:
        print(f"[plot] {error}; CSVs were still written", flush=True)
        return 0
    for path in written:
        print(f"[plot] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
