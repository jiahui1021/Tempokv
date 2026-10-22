#!/usr/bin/env python3
"""Merge rank/sample spectral reports without inventing missing measurements."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid

import numpy as np


def read_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def number(row, name):
    try:
        return float(row[name])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def aggregate(rows, keys, values):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        row = dict(zip(keys, key))
        row["rows"] = len(group)
        for value in values:
            finite = [number(item, value) for item in group if math.isfinite(number(item, value))]
            row[value + "_n"] = len(finite)
            row[value + "_mean"] = float(np.mean(finite)) if finite else float("nan")
            row[value + "_std"] = float(np.std(finite)) if finite else float("nan")
        result.append(row)
    return result


def write_csv(path, rows, columns):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _columns(keys, values):
    return [*keys, "rows", *(name + suffix for name in values for suffix in ("_n", "_mean", "_std"))]


def _empty(ax, label):
    ax.text(0.5, 0.5, "not measured", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(label)


def analyze(input_dir: Path, output_dir: Path) -> Path:
    roots = []
    metadata = []
    for path in sorted(input_dir.rglob("metadata.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if "coordinates" in item and "config" in item and item.get("suite_id", "A_FREQUENCY_ORACLE") == "A_FREQUENCY_ORACLE":
            roots.append(path.parent)
            metadata.append(item)
    if not roots:
        raise ValueError(f"no spectral sample metadata found under {input_dir}")
    # A new analysis never overwrites an earlier report.
    if output_dir.exists():
        output_dir = output_dir / f"analysis-{uuid.uuid4().hex[:12]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    spatial = read_rows([r / "spectrum/spatial_energy.csv" for r in roots])
    blocks = read_rows([r / "sensitivity/block_scores.csv" for r in roots])
    correlations = read_rows([r / "sensitivity/correlation.csv" for r in roots])
    budgets = read_rows([r / "compression/error_budget.csv" for r in roots])
    measured_budgets = [row for row in budgets if row["status"] == "measured"]
    frequency_keys, frequency_values = ("layer", "head", "role", "cutoff"), ("ratio",)
    correlation_keys = ("layer", "head", "score_name", "combination_alpha")
    correlation_values = ("spearman", "topk_recall_sampled", "finite_candidates")
    budget_keys = ("policy", "combination_alpha", "retention_ratio")
    budget_values = ("relative_attention_output_error", "actual_history_retention", "actual_retained_tokens", "kept_candidate_tokens")
    frequency = aggregate(spatial, frequency_keys, frequency_values)
    corr = aggregate(correlations, correlation_keys, correlation_values)
    budget_summary = aggregate(measured_budgets, budget_keys, budget_values)
    write_csv(output_dir / "frequency_summary.csv", frequency, _columns(frequency_keys, frequency_values))
    write_csv(output_dir / "correlation_summary.csv", corr, _columns(correlation_keys, correlation_values))
    write_csv(output_dir / "budget_summary.csv", budget_summary, _columns(budget_keys, budget_values))
    temporal = []
    for root, meta in zip(roots, metadata):
        with np.load(root / "spectrum/temporal_energy.npz", allow_pickle=False) as data:
            for key in data.files:
                match = re.fullmatch(r"layer_(\d+)_(k|v)_segment_(\d+)_low_ratio", key)
                if not match:
                    continue
                prefix = key.removesuffix("_low_ratio")
                for ci, cutoff in enumerate(data["cutoffs"]):
                    for hi, head in enumerate(data["heads"]):
                        temporal.append({"prompt_id": meta["prompt_id"], "sample_id": meta["sample_id"],
                                         "rank": meta["rank"], "inference_id": meta["inference_id"],
                                         "layer": match[1], "role": match[2], "segment": match[3],
                                         "head": int(head), "cutoff": float(cutoff),
                                         "frames": len(data[prefix + "_frames"]),
                                         "ratio": float(data[key][ci, hi])})
    write_csv(output_dir / "temporal_summary.csv", temporal,
              ["prompt_id", "sample_id", "rank", "inference_id", "layer", "role", "segment", "head", "cutoff", "frames", "ratio"])
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tempokv-spectral-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, role in zip(axes, ("k", "v")):
        rows = [row for row in frequency if row["role"] == role and math.isfinite(row["ratio_mean"])]
        for cutoff in sorted({row["cutoff"] for row in rows}, key=float):
            for head in sorted({row["head"] for row in rows}, key=int):
                grouped = [r for r in rows if r["cutoff"] == cutoff and r["head"] == head]
                grouped.sort(key=lambda r: int(r["layer"]))
                ax.plot([int(r["layer"]) for r in grouped], [r["ratio_mean"] for r in grouped],
                        marker="o", label=f"head={head}, cutoff={cutoff}")
        if rows:
            ax.legend(fontsize=7)
            ax.set_title(f"Clean {role.upper()} spatial low-frequency energy")
        else:
            _empty(ax, f"Clean {role.upper()} spatial spectrum")
        ax.set(xlabel="Layer (frames averaged, heads separate)", ylabel="Low-energy ratio", ylim=(-0.02, 1.02))
    fig.savefig(output_dir / "frequency_by_layer.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, role in zip(axes, ("k", "v")):
        points = [(number(r, f"freq_{role}_high_ratio"), number(r, "oracle_error")) for r in blocks]
        points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
        if points:
            x, y = zip(*points)
            ax.scatter(x, y, s=10, alpha=0.4)
            ax.set_title(f"{role.upper()}: sampled candidate blocks")
        else:
            _empty(ax, f"{role.upper()} frequency vs importance")
        ax.set(xlabel="Clean local high-frequency energy ratio", ylabel="Leave-one-block-out relative error")
    fig.savefig(output_dir / "frequency_vs_importance.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    if budget_summary:
        for policy, alpha in sorted({(r["policy"], r["combination_alpha"]) for r in budget_summary}):
            rows = sorted([r for r in budget_summary if r["policy"] == policy and r["combination_alpha"] == alpha], key=lambda r: float(r["retention_ratio"]))
            label = policy + (f" a={alpha}" if alpha else "")
            ax.errorbar([r["actual_history_retention_mean"] for r in rows],
                        [r["relative_attention_output_error_mean"] for r in rows],
                        yerr=[r["relative_attention_output_error_std"] for r in rows], marker="o", label=label)
        ax.legend(fontsize=7)
        ax.set_title("Joint deletion on the same sampled candidate sets")
    else:
        _empty(ax, "Compression error vs retention")
    ax.set(xlabel="Actual retained history tokens / all history tokens", ylabel="Relative attention output error")
    fig.savefig(output_dir / "error_vs_retention.png", dpi=160)
    plt.close(fig)
    synthetic = any("synthetic" in str(meta.get("source", "")).lower() for meta in metadata)
    lines = ["# Spectral experiment summary", "",
             "Synthetic CPU smoke data; not research evidence." if synthetic else "Results from the supplied spectral report files.", "",
             f"- Samples: {len(roots)}; failed samples: {sum(m['status'] != 'completed' for m in metadata)}.",
             f"- Spatial rows: {len(spatial)} ({'measured' if spatial else 'not measured'}).",
             f"- Temporal ratio rows: {len(temporal)} ({'measured' if temporal else 'not measured'}).",
             f"- Block oracle rows: {len(blocks)} ({'measured' if blocks else 'not measured'}).",
             f"- Compression rows: {len(measured_budgets)} measured, {len(budgets) - len(measured_budgets)} not measured.", "",
             "Correlation tables average per-call/layer/head statistics; all compared scores use the same common-finite subset of sampled candidates. Tied top-k boundaries and constant-score Spearman are NaN (not measured). Correlation does not establish causation.", "",
             "Retention budgets apply to sampled candidate tokens. All unsampled history, current chunk and protected recent frames remain in the full-window softmax. Every policy uses the same exact token budget; the boundary block may be partially retained. The figure uses actual total-history retention. Error bars show standard deviation across records, including random seeds.", "",
             "No video-quality, throughput gain, quantized-path or multi-resolution measurements are inferred from these tables.", "",
             "| Policy | Alpha | Candidate retention | Mean error | Actual history retention |", "|---|---:|---:|---:|---:|"]
    for row in budget_summary:
        lines.append(f"| {row['policy']} | {row['combination_alpha']} | {row['retention_ratio']} | {row['relative_attention_output_error_mean']:.6g} | {row['actual_history_retention_mean']:.6g} |")
    if not budget_summary:
        lines.append("| not measured | | | | |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(analyze(args.input, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
