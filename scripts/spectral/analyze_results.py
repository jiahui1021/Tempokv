#!/usr/bin/env python3
"""Merge rank/sample spectral reports without inventing missing measurements."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
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


def _experiment_identity(meta):
    """Identity of measurements that may share an aggregate, without paths."""
    config = dict(meta["config"])
    config.pop("output_dir", None)
    windows = meta.get("effective_windows", [])
    runtime = {
        json.dumps({key: window[key] for key in (
            "dtype", "attention_backend", "softmax_scale", "mask", "rope_precision"
        ) if key in window}, sort_keys=True, separators=(",", ":"))
        for window in windows
    }
    identity = {
        "config": config,
        "model": {key: meta[key] for key in (
            "model", "checkpoint_sha256", "model_sha256", "weight_id", "model_config", "git_commit"
        ) if key in meta},
        "source": meta.get("source"),
        "coordinates": meta["coordinates"],
        "effective_grids": meta.get("effective_grids", {}),
        "runtime": sorted(runtime),
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _select_samples(input_dir):
    """Exclude unfinished attempts and retain one completed retry per identity."""
    discovered = []
    for path in sorted(input_dir.rglob("metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if "coordinates" in meta and "config" in meta and meta.get("suite_id", "A_FREQUENCY_ORACLE") == "A_FREQUENCY_ORACLE":
            discovered.append((path, meta))
    if not discovered:
        raise ValueError(f"no spectral sample metadata found under {input_dir}")
    selection = {
        "discovered_samples": len(discovered), "retained_samples": 0,
        "excluded_failed_samples": 0, "excluded_incomplete_samples": 0,
        "excluded_retry_samples": 0, "unknown_generation_seed_samples": 0,
        "excluded": [], "retained_metadata": [],
    }
    completed = []
    for path, meta in discovered:
        if meta.get("status") != "completed":
            reason = "failed" if meta.get("status") == "failed" else "incomplete_or_unknown_status"
            selection["excluded_failed_samples" if reason == "failed" else "excluded_incomplete_samples"] += 1
            selection["excluded"].append({"metadata": str(path), "reason": reason})
        else:
            completed.append((path, meta))
    signatures = {_experiment_identity(meta) for _, meta in completed}
    if len(signatures) > 1:
        raise ValueError(
            "completed samples have different experiment configurations or model/runtime identities; "
            "analyze each configuration/model separately (output_dir alone may differ)"
        )
    signature = next(iter(signatures)) if signatures else None
    selection["experiment_identity"] = json.loads(signature) if signature else None
    selection["experiment_fingerprint"] = hashlib.sha256(signature.encode()).hexdigest()[:16] if signature else None
    batch_sizes = defaultdict(int)
    for path, _ in discovered:
        batch_sizes[path.parent.parent] += 1
    kept = {}
    for path, meta in sorted(completed, key=lambda item: (item[0].stat().st_mtime_ns, str(item[0]))):
        # Rank and output path are assignment details, not independent samples.
        # Missing generation seeds cannot safely be replaced by the analysis
        # seed: legacy reports retain their inference identity instead.
        generation_seed = meta.get("generation_seed")
        if generation_seed is None:
            selection["unknown_generation_seed_samples"] += 1
            sampling = {"inference_id": meta.get("inference_id", str(path.parent.parent))}
        else:
            sampling = {"generation_seed": generation_seed}
        identity = json.dumps({
            "prompt_id": meta.get("prompt_id"), "sample_id": meta.get("sample_id"),
            "batch_size": batch_sizes[path.parent.parent], **sampling,
            "prompt_identity": {key: meta[key] for key in (
                "prompt", "prompt_text", "prompt_sha256", "prompt_file", "prompts_path", "dataset"
            ) if key in meta},
        }, sort_keys=True, separators=(",", ":"))
        previous = kept.get(identity)
        if previous is not None:
            selection["excluded_retry_samples"] += 1
            selection["excluded"].append({
                "metadata": str(previous[0]), "reason": "completed_retry",
                "retained_metadata": str(path),
            })
        kept[identity] = (path, meta)
    retained = sorted(kept.values(), key=lambda item: str(item[0]))
    selection["retained_samples"] = len(retained)
    selection["retained_metadata"] = [str(path) for path, _ in retained]
    return [path.parent for path, _ in retained], [meta for _, meta in retained], selection


def analyze(input_dir: Path, output_dir: Path) -> Path:
    roots, metadata, selection = _select_samples(input_dir)
    # A new analysis never overwrites an earlier report.
    if output_dir.exists():
        output_dir = output_dir / f"analysis-{uuid.uuid4().hex[:12]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "analysis_selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
             f"- Samples retained: {len(roots)} / {selection['discovered_samples']} discovered.",
             f"- Excluded failed samples: {selection['excluded_failed_samples']}; incomplete/unknown status: {selection['excluded_incomplete_samples']}; completed retries: {selection['excluded_retry_samples']}.",
             f"- Samples without generation_seed: {selection['unknown_generation_seed_samples']} (kept distinct by inference_id; no seed inferred).",
             "- Selection and exclusions are recorded in analysis_selection.json. Completed retries use the newest metadata modification time, with the path as a deterministic tie-break.",
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
