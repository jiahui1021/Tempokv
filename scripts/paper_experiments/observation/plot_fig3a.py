#!/usr/bin/env python3
"""Render Fig 3(a) and the section 3.2 tier table from a decompose report.

The figure is one stacked bar per tensor, split into what each modelling step
removes: per-channel DC, the temporal predictor, the aligned-K term (V only),
and the residual that still has to be coded.  The CSVs are always written; the
figure needs matplotlib and is skipped with a message when it is missing, so a
headless fitting host still produces the numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


SEGMENTS = (
    ("dc", "per-channel DC"),
    ("temporal", "temporally predictable"),
    ("cross", "cross ($K_t \\rightarrow V_t$)"),
    ("residual", "residual"),
)
# Colour-blind safe and printable in greyscale: the residual is the darkest
# segment because it is the part the reader should be looking at.
SEGMENT_COLORS = {
    "dc": "#c6dbef",
    "temporal": "#6baed6",
    "cross": "#fdae6b",
    "residual": "#4a4a4a",
}
TIER_LABELS = {
    "K0": "tier 0 direct",
    "K1": "tier 1 +DC",
    "K2": "tier 2 +temporal",
    "V0": "tier 0 direct",
    "V1": "tier 1 +DC",
    "V2": "tier 2 +temporal",
    "V3": "tier 3 +aligned K",
}


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tempokv.observation-decompose.v1":
        raise ValueError(f"{path} is not an observation decompose report")
    return payload


def table_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The paper's table 1, one row per tier and loop."""
    rows: list[dict[str, Any]] = []
    for loop, tiers in report["pooled"].items():
        for tier, cell in sorted(tiers.items()):
            rows.append(
                {
                    "loop": loop,
                    "tensor": tier[0],
                    "tier": tier,
                    "label": TIER_LABELS.get(tier, tier),
                    "energy_ratio": cell["energy_ratio"],
                    "energy_equivalent_bits": cell["energy_equivalent_bits"],
                    "bits_saved": cell["bits_saved_median_over_heads"],
                    "range_over_std": cell["range_over_std_median_over_heads"],
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_figure(report: dict[str, Any], loop: str, output: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; wrote the CSVs and skipped the figure")
        return False
    shares = report["ladder_shares"][loop]
    figure, axes = plt.subplots(figsize=(4.2, 3.2))
    positions = [0, 1]
    bottoms = [0.0, 0.0]
    for name, label in SEGMENTS:
        heights = [float(shares["k"][name]), float(shares["v"][name])]
        axes.bar(
            positions,
            heights,
            bottom=bottoms,
            width=0.55,
            label=label,
            color=SEGMENT_COLORS[name],
            edgecolor="white",
            linewidth=0.6,
        )
        for position, height, bottom in zip(positions, heights, bottoms):
            if height >= 0.06:
                axes.text(
                    position,
                    bottom + height / 2,
                    f"{height * 100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if name == "residual" else "black",
                )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
    axes.set_xticks(positions)
    axes.set_xticklabels(["K (pre-RoPE)", "V"])
    axes.set_ylim(0.0, 1.0)
    axes.set_ylabel("share of original energy")
    axes.set_title(
        "open-loop probe" if loop == "clean" else "closed loop (INT4 anchor)",
        fontsize=9,
    )
    axes.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("results/observation/decompose.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/observation"))
    parser.add_argument(
        "--loop",
        default="clean",
        choices=("clean", "closed_loop"),
        help="which pass the figure shows; the table always carries both",
    )
    args = parser.parse_args(argv)
    try:
        report = load_report(args.report)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.loop not in report["ladder_shares"]:
        parser.error(f"the report has no {args.loop!r} pass")
    rows = table_rows(report)
    _write_csv(
        args.output_dir / "table1_tiers.csv",
        rows,
        (
            "loop",
            "tensor",
            "tier",
            "label",
            "energy_ratio",
            "energy_equivalent_bits",
            "bits_saved",
            "range_over_std",
        ),
    )
    _write_csv(
        args.output_dir / "fig3a_segments.csv",
        [
            {"loop": loop, "tensor": tensor, **{name: shares[tensor][name] for name, _ in SEGMENTS}}
            for loop, shares in report["ladder_shares"].items()
            for tensor in ("k", "v")
        ],
        ("loop", "tensor", *(name for name, _ in SEGMENTS)),
    )
    render_figure(report, args.loop, args.output_dir / f"fig3a_{args.loop}.pdf")
    for row in rows:
        print(
            f"{row['loop']:11s} {row['tier']:3s} {row['label']:20s} "
            f"energy={row['energy_ratio']:7.4f}  "
            f"equiv_bits={row['energy_equivalent_bits']:6.3f}  "
            f"bits_saved={row['bits_saved']:6.3f}  "
            f"range/std={row['range_over_std']:6.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
