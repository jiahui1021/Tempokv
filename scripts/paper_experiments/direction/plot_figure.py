#!/usr/bin/env python3
"""Render the direction-study figure and its backing CSVs.

    (a) held-out reconstruction NMSE per arm, K and V shown separately
    (b) generation quality against the shared BF16 reference

The CSVs are always written.  The figure needs matplotlib and is skipped, with
a message, when it is missing, so a headless fitting host still produces the
numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


ARM_ORDER = ("temporal_only", "k_to_v", "v_to_k")
ARM_LABELS = {
    "temporal_only": "temporal only",
    "k_to_v": "$K_t \\rightarrow V_t$",
    "v_to_k": "$V_t \\rightarrow K_t$",
}
# Higher is better for PSNR and SSIM, lower for LPIPS; the axis label says so
# rather than leaving the reader to remember.
QUALITY_METRICS = (("psnr", "PSNR (dB, higher better)"), ("ssim", "SSIM (higher better)"),
                   ("lpips", "LPIPS (lower better)"))


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


def reconstruction_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    by_arm = {row["arm"]: row for row in report["per_arm"]}
    rows = []
    for item in report["arms"]:
        summary = by_arm.get(item["arm"], {})
        rows.append(
            {
                "arm": item["arm"],
                "predictor_direction": item["predictor_direction"],
                "v_predictor_mode": item["v_predictor_mode"],
                "k_nmse": item["k_nmse"],
                "v_nmse": item["v_nmse"],
                "total_nmse": item["k_nmse"] + item["v_nmse"],
                "k_nmse_vs_temporal_only": summary.get("k_nmse_vs_temporal_only"),
                "v_nmse_vs_temporal_only": summary.get("v_nmse_vs_temporal_only"),
            }
        )
    return rows


def quality_rows(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if report is None:
        return []
    rows = []
    for item in report["arms"]:
        metrics = item.get("quality_metrics") or {}
        rows.append(
            {
                "arm": item["arm"],
                "predictor_direction": item["predictor_direction"],
                "quality_status": item["quality_status"],
                **{name: metrics.get(name) for name, _ in QUALITY_METRICS},
            }
        )
    return rows


def _ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {name: position for position, name in enumerate(ARM_ORDER)}
    return sorted(rows, key=lambda row: index.get(row["arm"], len(index)))


def _panel_reconstruction(axis, rows: list[dict[str, Any]]) -> None:
    rows = _ordered(rows)
    labels = [ARM_LABELS.get(row["arm"], row["arm"]) for row in rows]
    positions = range(len(rows))
    width = 0.38
    axis.bar(
        [p - width / 2 for p in positions],
        [row["k_nmse"] for row in rows],
        width,
        label="K",
    )
    axis.bar(
        [p + width / 2 for p in positions],
        [row["v_nmse"] for row in rows],
        width,
        label="V",
    )
    axis.set_xticks(list(positions), labels)
    axis.set_ylabel("held-out normalized reconstruction MSE")
    axis.set_title("(a) Reconstruction error by dependency direction")
    axis.legend(title="tensor")
    axis.grid(axis="y", alpha=0.3)


def _panel_quality(axis, rows: list[dict[str, Any]]) -> None:
    rows = _ordered([row for row in rows if row.get("psnr") is not None])
    if not rows:
        axis.text(
            0.5,
            0.5,
            "generation quality not run\n(needs a GPU run of run_quality.py)",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        axis.set_title("(b) Generation quality vs BF16")
        return
    labels = [ARM_LABELS.get(row["arm"], row["arm"]) for row in rows]
    positions = list(range(len(rows)))
    # PSNR carries the axis; SSIM and LPIPS are annotated so one panel holds
    # all three without three incompatible scales.
    axis.bar(positions, [row["psnr"] for row in rows], 0.5)
    for position, row in zip(positions, rows):
        annotation = []
        if row.get("ssim") is not None:
            annotation.append(f"SSIM {row['ssim']:.3f}")
        if row.get("lpips") is not None:
            annotation.append(f"LPIPS {row['lpips']:.3f}")
        if annotation:
            axis.text(
                position,
                row["psnr"],
                "\n".join(annotation),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("PSNR vs BF16 (dB, higher better)")
    axis.set_title("(b) Generation quality vs BF16")
    axis.grid(axis="y", alpha=0.3)


def render(
    reconstruction: list[dict[str, Any]],
    quality: list[dict[str, Any]],
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

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    _panel_reconstruction(axes[0], reconstruction)
    _panel_quality(axes[1], quality)
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
    parser.add_argument("--reconstruction-report", type=Path, required=True)
    parser.add_argument(
        "--quality-report",
        type=Path,
        help="summary.json from run_quality.py; panel (b) says so when absent",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-name", default="direction")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reconstruction = reconstruction_rows(_load(args.reconstruction_report))
    quality = quality_rows(
        _load(args.quality_report) if args.quality_report else None
    )
    _write_csv(
        args.output_dir / "panel_a_reconstruction.csv",
        _ordered(reconstruction),
        [
            "arm",
            "predictor_direction",
            "v_predictor_mode",
            "k_nmse",
            "v_nmse",
            "total_nmse",
            "k_nmse_vs_temporal_only",
            "v_nmse_vs_temporal_only",
        ],
    )
    if quality:
        _write_csv(
            args.output_dir / "panel_b_quality.csv",
            _ordered(quality),
            ["arm", "predictor_direction", "quality_status"]
            + [name for name, _ in QUALITY_METRICS],
        )
    try:
        written = render(
            reconstruction, quality, args.output_dir / args.figure_name,
            formats=args.formats,
        )
    except RuntimeError as error:
        print(f"[plot] {error}; CSVs were still written")
        return 0
    for path in written:
        print(f"[plot] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
