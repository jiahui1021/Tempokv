#!/usr/bin/env python3
"""Summarize the token pruning pilot (reports/pruning.md).

Experiment 1 reads the per-token statistics of the unpruned TempoKV run
(``runs/none/prune_stats``); experiment 2 reads the paired quality reports of
every criterion x ratio run.  Everything goes to ``<root>/analysis``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ranks(values: np.ndarray) -> np.ndarray:
    """Within-frame percentile in [0, 1) along the last axis."""
    return values.argsort(-1).argsort(-1) / values.shape[-1]


def _load_stats(directory: Path) -> dict[str, np.ndarray]:
    """Stack every video's [L, frames, tokens] arrays, dropping frame 0."""
    stats: dict[str, list[np.ndarray]] = {"attention": [], "residual": [], "similarity": []}
    for path in sorted(directory.glob("*.npz")):
        data = np.load(path)
        valid = ~np.isnan(data["residual"][0, :, 0])
        for key in stats:
            stats[key].append(data[key][:, valid])
    if not stats["attention"]:
        raise FileNotFoundError(f"no pruning statistics under {directory}")
    return {key: np.concatenate(value, axis=1) for key, value in stats.items()}


def _within_frame_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation per (layer, frame), shape [L, frames]."""
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean(-1, keepdims=True)
    rb = rb - rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / np.sqrt((ra**2).sum(-1) * (rb**2).sum(-1))


def experiment1(stats: dict[str, np.ndarray], out: Path) -> dict:
    attention, residual = stats["attention"], stats["residual"]
    # Redundancy increases with similarity, so negate it to share the
    # "low = prune first" direction of A and E.
    redundancy = -stats["similarity"]
    ra, re = _ranks(attention), _ranks(residual)
    low_a, low_e = ra < 0.5, re < 0.5
    quadrants = {
        "low_residual_low_attention": float((low_e & low_a).mean()),
        "low_residual_high_attention": float((low_e & ~low_a).mean()),
        "high_residual_low_attention": float((~low_e & low_a).mean()),
        "high_residual_high_attention": float((~low_e & ~low_a).mean()),
    }
    rho = {
        "attention_vs_residual": _within_frame_spearman(attention, residual),
        "attention_vs_neg_similarity": _within_frame_spearman(attention, redundancy),
        "residual_vs_neg_similarity": _within_frame_spearman(residual, redundancy),
    }
    overlap = {}
    for ratio in (0.1, 0.25, 0.5):
        a_set, e_set = ra < ratio, re < ratio
        jaccard = (a_set & e_set).sum(-1) / np.maximum((a_set | e_set).sum(-1), 1)
        overlap[str(ratio)] = {
            "jaccard_attention_residual": float(jaccard.mean()),
            "independent_expectation": ratio / (2 - ratio),
        }
    result = {
        "tokens": int(attention.size),
        "layers": int(attention.shape[0]),
        "frames": int(attention.shape[1]),
        "median_split_quadrants": quadrants,
        "spearman_within_frame_mean": {k: float(np.nanmean(v)) for k, v in rho.items()},
        "spearman_attention_vs_residual_per_layer": np.nanmean(
            rho["attention_vs_residual"], axis=1
        ).round(4).tolist(),
        "bottom_set_overlap": overlap,
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    pick = rng.choice(attention.size, size=min(attention.size, 400_000), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].hexbin(
        np.log10(residual.ravel()[pick] + 1e-12),
        np.log10(attention.ravel()[pick] + 1e-12),
        gridsize=80, bins="log", cmap="viridis",
    )
    axes[0].set(xlabel="log10 prediction residual E", ylabel="log10 attention A",
                title="Raw (all layers)")
    density, _, _ = np.histogram2d(re.ravel()[pick], ra.ravel()[pick], bins=20,
                                   range=[[0, 1], [0, 1]], density=True)
    image = axes[1].imshow(density.T, origin="lower", extent=[0, 1, 0, 1],
                           cmap="RdBu_r", vmin=0, vmax=2)
    axes[1].axhline(0.5, color="k", lw=0.8)
    axes[1].axvline(0.5, color="k", lw=0.8)
    for (x, y), key in zip(
        ((0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)), quadrants
    ):
        axes[1].text(x, y, f"{quadrants[key]:.1%}", ha="center", va="center")
    axes[1].set(xlabel="E percentile (within frame)", ylabel="A percentile",
                title="Joint density (1 = independent)")
    fig.colorbar(image, ax=axes[1])
    axes[2].bar(range(attention.shape[0]), result["spearman_attention_vs_residual_per_layer"])
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set(xlabel="layer", ylabel="Spearman(A, E)", title="Per-layer correlation")
    fig.tight_layout()
    fig.savefig(out / "exp1_residual_attention.png", dpi=150)
    plt.close(fig)
    return result


def experiment2(root: Path, out: Path) -> dict:
    rows = []
    for stats_dir in sorted((root / "runs").glob("*/prune_stats")):
        name = stats_dir.parent.name
        reports = [json.loads(p.read_text()) for p in stats_dir.glob("*.json")]
        if not reports:
            continue
        row = {
            "run": name,
            "criterion": reports[0]["criterion"],
            "ratio": reports[0]["ratio"],
            "retention": float(np.mean([r["retention"] for r in reports])),
        }
        for reference in ("bf16", "none"):
            path = root / "quality" / f"vs_{reference}" / f"{name}.json"
            if path.is_file():
                quality = json.loads(path.read_text())
                for metric in ("psnr_db", "ssim", "lpips", "fid"):
                    row[f"{metric}_vs_{reference}"] = quality.get(metric)
        rows.append(row)
    rows.sort(key=lambda r: (r["ratio"], r["criterion"]))

    # Winner per ratio by LPIPS against the unpruned TempoKV run, which isolates
    # the pruning damage from the quantization error both sides share.
    winners = {}
    for ratio in sorted({r["ratio"] for r in rows if r["criterion"] != "none"}):
        scored = [r for r in rows if r["ratio"] == ratio and r.get("lpips_vs_none") is not None]
        if scored:
            winners[str(ratio)] = min(scored, key=lambda r: r["lpips_vs_none"])["criterion"]

    columns = ["criterion", "ratio", "retention", "psnr_db_vs_none", "ssim_vs_none",
               "lpips_vs_none", "psnr_db_vs_bf16", "lpips_vs_bf16", "fid_vs_bf16"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append("-" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    (out / "exp2_table.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for criterion in sorted({r["criterion"] for r in rows} - {"none"}):
        series = [r for r in rows if r["criterion"] == criterion]
        for axis, metric in zip(axes, ("psnr_db_vs_none", "ssim_vs_none", "lpips_vs_none")):
            points = [(r["ratio"], r.get(metric)) for r in series if r.get(metric) is not None]
            if points:
                axis.plot(*zip(*points), marker="o", label=criterion)
                axis.set(xlabel="pruning ratio", ylabel=metric)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "exp2_criteria.png", dpi=150)
    plt.close(fig)
    return {"rows": rows, "best_criterion_by_lpips_vs_none": winners}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="pruning pilot OUTPUT_ROOT")
    args = parser.parse_args()
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment1": experiment1(_load_stats(args.root / "runs/none/prune_stats"), out),
        "experiment2": experiment2(args.root, out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["experiment1"], indent=2))
    print((out / "exp2_table.md").read_text())
    print(json.dumps(summary["experiment2"]["best_criterion_by_lpips_vs_none"], indent=2))


if __name__ == "__main__":
    main()
