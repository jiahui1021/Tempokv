#!/usr/bin/env python3
"""Render the long-horizon reconstruction figure and its backing CSVs.

    (a) K NMSE against frame index
    (b) V NMSE against frame index

Each panel draws the per-frame curve and the cumulative curve, because the two
answer different questions: the per-frame line shows where a method starts to
degrade, and the cumulative line ends at the pooled number a table would quote.

CSVs are always written; the figure needs matplotlib and is skipped with a
message when it is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


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
        writer = csv.DictWriter(handle, fieldnames=tuple(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def curve_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in report["methods"]:
        for tensor in ("k", "v"):
            curve = item[tensor]
            cumulative = {row["frame"]: row["nmse"] for row in curve["cumulative"]}
            for row in curve["per_frame"]:
                rows.append(
                    {
                        "method": item["method"],
                        "tensor": tensor.upper(),
                        "frame": row["frame"],
                        "nmse": row["nmse"],
                        "cumulative_nmse": cumulative.get(row["frame"]),
                    }
                )
    return rows


def summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in report["methods"]:
        for tensor in ("k", "v"):
            rows.append(
                {
                    "method": item["method"],
                    "tensor": tensor.upper(),
                    **item["summary"][tensor],
                }
            )
    return rows


# Methods that share a K predictor produce *identical* K curves, so those lines
# sit exactly on top of one another.  Cycling dash patterns keeps every method
# visible instead of letting the last one drawn hide the rest.
_DASHES = ((), (6, 2), (2, 2), (6, 2, 1, 2), (1, 1.6))


def _panel(
    axis, report: dict[str, Any], tensor: str, *, cumulative: bool, letter: str
) -> None:
    for index, item in enumerate(report["methods"]):
        curve = item[tensor]
        source = curve["cumulative"] if cumulative else curve["per_frame"]
        frames = [row["frame"] for row in source]
        values = [row["nmse"] for row in source]
        axis.plot(
            frames,
            values,
            label=item["label"],
            linewidth=1.6,
            dashes=_DASHES[index % len(_DASHES)],
            alpha=0.9,
        )
    axis.set_xlabel("latent frame index")
    axis.set_ylabel("normalized reconstruction MSE")
    kind = "cumulative" if cumulative else "per frame"
    axis.set_title(f"({letter}) {tensor.upper()}, {kind}")
    axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)


def render(
    report: dict[str, Any], output: Path, *, formats: Sequence[str] = ("pdf", "png")
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the host env
        raise RuntimeError("plotting requires matplotlib") from error

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    _panel(axes[0][0], report, "k", cumulative=False, letter="a")
    _panel(axes[0][1], report, "v", cumulative=False, letter="b")
    _panel(axes[1][0], report, "k", cumulative=True, letter="c")
    _panel(axes[1][1], report, "v", cumulative=True, letter="d")
    figure.suptitle(
        f"Reconstruction error over {report['frames']} latent frames "
        f"(INT{report['bits']}, {report['model']})"
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        path = output.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=220)
        written.append(path)
    plt.close(figure)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-name", default="long_horizon")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = _load(args.report)
    _write_csv(
        args.output_dir / "reconstruction_by_frame.csv",
        curve_rows(report),
        ["method", "tensor", "frame", "nmse", "cumulative_nmse"],
    )
    _write_csv(
        args.output_dir / "reconstruction_summary.csv",
        summary_rows(report),
        [
            "method",
            "tensor",
            "first_frame_nmse",
            "last_frame_nmse",
            "max_frame_nmse",
            "head_mean_nmse",
            "tail_mean_nmse",
            "tail_over_head",
            "pooled_nmse",
        ],
    )
    try:
        written = render(report, args.output_dir / args.figure_name, formats=args.formats)
    except RuntimeError as error:
        print(f"[plot] {error}; CSVs were still written")
        return 0
    for path in written:
        print(f"[plot] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
