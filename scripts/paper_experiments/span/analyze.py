#!/usr/bin/env python3
"""Collect span-study outputs and render the paper/appendix figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable


SPANS = (2, 4, 8, 12, 18)
BITS = (2, 4)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fields))
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report must contain an object: {path}")
    return payload


def _real_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    quality_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    nmse_rows: list[dict[str, Any]] = []
    for params_dir in sorted((root / "params").glob("int*_anchor*_span*")):
        words = params_dir.name.split("_")
        bits = int(words[0].removeprefix("int"))
        anchor_bits = int(words[1].removeprefix("anchor"))
        span = int(words[2].removeprefix("span"))
        fit = _load_json(params_dir / "fit_report.json")
        if (fit.get("fit") or {}).get("span_frames") != span:
            raise ValueError(f"fit span mismatch in {params_dir}")
        nmse_path = params_dir / "nmse_by_position.csv"
        with nmse_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["span_frames"]) != span:
                    raise ValueError(f"NMSE span mismatch in {nmse_path}")
                nmse_rows.append(dict(row))
        for pixel_frames in (717, 1401):
            run = root / "runs" / params_dir.name / f"pixel_{pixel_frames}"
            report_path = run / "pipeline_report.json"
            quality_path = run / "quality.json"
            if not report_path.is_file() and pixel_frames == 1401:
                continue
            report = _load_json(report_path)
            quality = _load_json(quality_path)
            batches = report.get("prompt_batches") or []
            memory = (batches[0].get("kv_cache_memory") or {}) if batches else {}
            compressed = memory.get("total_kv_cache_bytes")
            uncompressed = memory.get("total_bf16_kv_bytes")
            if not isinstance(compressed, int) or not isinstance(uncompressed, int):
                raise ValueError(f"cache byte accounting missing in {report_path}")
            if compressed <= 0 or uncompressed <= 0:
                raise ValueError(f"invalid cache byte accounting in {report_path}")
            row = {
                "bits": bits,
                "anchor_bits": anchor_bits,
                "span_frames": span,
                "pixel_frames": pixel_frames,
                "bf16_bytes": uncompressed,
                "packed_bytes": compressed,
                "compression_ratio": uncompressed / compressed,
                "psnr_db": quality.get("psnr_db"),
                "lpips": quality.get("lpips"),
                "e2e_seconds": report.get("generation_seconds"),
            }
            if pixel_frames == 717:
                quality_rows.append(row)
            else:
                long_rows.append(row)
    return quality_rows, long_rows, nmse_rows


def _synthetic_rows(seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    quality: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    nmse: list[dict[str, Any]] = []
    for bits in BITS:
        for anchor_bits in ((2, 4) if bits == 2 else (4,)):
            for span in SPANS:
                ratio = (16.0 / bits) * (1.0 - 0.55 / span)
                damage = (5 - bits) * 0.12 * math.log2(span)
                if anchor_bits == 2:
                    damage += 0.18 / math.sqrt(span)
                    ratio *= 1.0 + 0.04 / span
                base = {
                    "bits": bits,
                    "anchor_bits": anchor_bits,
                    "span_frames": span,
                    "bf16_bytes": 1_000_000,
                    "packed_bytes": round(1_000_000 / ratio),
                    "compression_ratio": ratio,
                    "psnr_db": 39.5 - damage + rng.uniform(-0.03, 0.03),
                    "lpips": 0.025 + damage / 120 + rng.uniform(-0.0002, 0.0002),
                }
                quality.append({**base, "pixel_frames": 717, "e2e_seconds": 120.0})
                if anchor_bits == 4 and span in (2, 18):
                    long_rows.append(
                        {
                            **base,
                            "pixel_frames": 1401,
                            "psnr_db": base["psnr_db"] - 0.15 * math.log2(span),
                            "lpips": base["lpips"] + 0.001 * math.log2(span),
                            "e2e_seconds": 265.0,
                        }
                    )
                for tensor, scale in (("K", 1.0), ("V", 1.25)):
                    for position in range(span):
                        value = scale * (0.0005 + position * (0.0002 + span / 200000))
                        nmse.append(
                            {
                                "bits": bits,
                                "anchor_bits": anchor_bits,
                                "span_frames": span,
                                "tensor": tensor,
                                "position": position,
                                "nmse": value,
                                "error_energy": value * 1000,
                                "reference_energy": 1000,
                                "values": 100,
                            }
                        )
    return quality, long_rows, nmse


def _plots(
    output: Path,
    quality: list[dict[str, Any]],
    long_rows: list[dict[str, Any]],
    nmse: list[dict[str, Any]],
    *,
    synthetic: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("plotting requires matplotlib") from error
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for bits in BITS:
        main = sorted(
            (
                row
                for row in quality
                if int(row["bits"]) == bits and int(row["anchor_bits"]) == 4
            ),
            key=lambda row: int(row["span_frames"]),
        )
        x = [int(row["span_frames"]) for row in main]
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
        for left, metric, label, color in (
            (axes[0], "psnr_db", "PSNR (dB)", "#dc2626"),
            (axes[1], "lpips", "LPIPS (lower is better)", "#16a34a"),
        ):
            right = left.twinx()
            left.plot(x, [float(row["compression_ratio"]) for row in main], "o-", color="#2563eb", label="Compression")
            right.plot(x, [float(row[metric]) for row in main], "s-", color=color, label=label.split(" (")[0])
            left.set(xlabel=r"$L_{span}$ (latent frames)", ylabel="Compression ratio (×)")
            right.set_ylabel(label)
            left.grid(alpha=0.25)
            handles = left.lines + right.lines
            left.legend(handles, [line.get_label() for line in handles], fontsize=8)
        if synthetic:
            fig.suptitle("SYNTHETIC — NOT PAPER RESULTS", color="#b91c1c")
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(figures / f"compression_quality_int{bits}.{suffix}", dpi=220)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=False)
        for axis, tensor in zip(axes, ("K", "V")):
            for span in SPANS:
                rows = sorted(
                    (
                        row
                        for row in nmse
                        if int(row["bits"]) == bits
                        and int(row["anchor_bits"]) == 4
                        and int(row["span_frames"]) == span
                        and row["tensor"] == tensor
                    ),
                    key=lambda row: int(row["position"]),
                )
                axis.plot(
                    [int(row["position"]) for row in rows],
                    [float(row["nmse"]) for row in rows],
                    marker="o",
                    markersize=2.5,
                    label=f"L={span}",
                )
            axis.set(xlabel="Distance from anchor", ylabel=f"{tensor} reconstruction NMSE", title=tensor)
            axis.grid(alpha=0.25)
        axes[1].legend(fontsize=7, ncol=2)
        if synthetic:
            fig.suptitle("SYNTHETIC — NOT PAPER RESULTS", color="#b91c1c")
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(figures / f"nmse_by_position_int{bits}.{suffix}", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for bits in BITS:
        rows = sorted(
            (row for row in long_rows if int(row["bits"]) == bits),
            key=lambda row: int(row["span_frames"]),
        )
        axes[0].plot([int(r["span_frames"]) for r in rows], [float(r["psnr_db"]) for r in rows], "o-", label=f"INT{bits}")
        axes[1].plot([int(r["span_frames"]) for r in rows], [float(r["lpips"]) for r in rows], "o-", label=f"INT{bits}")
    axes[0].set(xlabel=r"$L_{span}$", ylabel="PSNR (dB)")
    axes[1].set(xlabel=r"$L_{span}$", ylabel="LPIPS")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    title = "Long-horizon validation (1401 computed pixel frames)"
    if synthetic:
        title += " — SYNTHETIC"
    fig.suptitle(title, color="#b91c1c" if synthetic else None)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures / f"long_horizon.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.synthetic:
        quality, long_rows, nmse = _synthetic_rows(args.seed)
    else:
        quality, long_rows, nmse = _real_rows(args.study_root)
    tables = args.study_root / "tables"
    _write_csv(
        tables / "compression_quality.csv",
        quality,
        ("bits", "anchor_bits", "span_frames", "pixel_frames", "bf16_bytes", "packed_bytes", "compression_ratio", "psnr_db", "lpips", "e2e_seconds"),
    )
    _write_csv(
        tables / "long_horizon.csv",
        long_rows,
        ("bits", "anchor_bits", "span_frames", "pixel_frames", "bf16_bytes", "packed_bytes", "compression_ratio", "psnr_db", "lpips", "e2e_seconds"),
    )
    _write_csv(
        tables / "nmse_by_position.csv",
        nmse,
        ("bits", "anchor_bits", "span_frames", "tensor", "position", "nmse", "error_energy", "reference_energy", "values"),
    )
    _plots(args.study_root, quality, long_rows, nmse, synthetic=args.synthetic)
    summary = {
        "schema_version": 1,
        "synthetic": args.synthetic,
        "compression_quality_rows": len(quality),
        "long_horizon_rows": len(long_rows),
        "nmse_rows": len(nmse),
        "long_selection_rule": (
            "per bit width: order 717-frame quality by higher PSNR, then lower "
            "LPIPS, then shorter span; select both ends"
        ),
    }
    selection_path = args.study_root / "long_selection.json"
    if selection_path.is_file() and not args.synthetic:
        summary["long_selection"] = _load_json(selection_path)
    (args.study_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
