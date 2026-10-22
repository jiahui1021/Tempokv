"""Rebuild research.v1 tables, clustered summaries and figures from saved CSVs.

This module reads no model tensors and performs no generation.  Cohorts keep
configuration, data origin, model, checkpoint, attention window and source
separate throughout aggregation and plotting.
"""
from __future__ import annotations

from collections import Counter, defaultdict
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

from .config import EXPERIMENTS
from .io import SCHEMA, atomic_json, safe, is_real_origin
from .metrics import clustered_summary


QUALITY_METRICS = ("PSNR", "SSIM", "LPIPS", "FID", "VBench")
COHORT_FIELDS = ("config_signature", "data_origin", "model", "checkpoint", "window", "source")
GROUP_FIELDS = ("cohort_id", "experiment_id", "strategy", "role", "score_type", "metric_scope", "cutoff",
                "temporal_window", "temporal_control", "retention", "k_bits", "v_bits", "strength",
                "horizon_chunks", "conditioning_set", "budget_cohort", "intervention", "phase",
                "chunk_index", "denoise_step_index", "denoise_timestep", "requested_strength",
                "calibration_protocol", "comparison_cohort", "coordinate", "comparison_reference", "statistic", "topk_fraction")
METRICS = ("score", "low_energy_ratio", "total_energy", "attention_mass", "local_attention_relative_error",
           "model_output_relative_error", "clean_latent_relative_error", "nmse_k", "nmse_v", "spearman",
           "topk_recall_sampled", "interaction", "interaction_error", "generation_relative_error",
           "reappearance_score", "matched_reappearance_identity_cosine", "matched_reappearance_identity_relative_l2",
           "matched_reappearance_state_cosine", "matched_reappearance_state_relative_l2", "sensitivity", "error", *QUALITY_METRICS)
_TEXT_FIELDS = {"schema_version", "experiment_id", "event_id", "data_origin", "phase", "block_id", "score_type",
                "strategy", "intervention", "role", "calibration_protocol", "budget_cohort", "paired_run_id",
                "status", "reason", "config_signature", "model", "checkpoint", "window", "source", "cohort_id",
                "manifest_path", "csv_path", "metric_scope", "coordinate", "temporal_control"}
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
FIGURES = {
    "S1": ("frequency_energy_by_layer_head", "spatial_vs_temporal"),
    "S2": ("score_vs_attention_oracle",), "E1": ("local_vs_generation_importance",),
    "E3": ("conditional_substitutability",), "E4": ("kv_sensitivity_and_bit_budget",),
    "E2": ("error_propagation_heatmap",), "S3": ("error_vs_budget",),
    "E6": ("step_sensitivity",), "E7": ("quant_prune_interaction",),
    "E5": ("long_term_reappearance",),
}


def _canonical(value):
    return json.dumps(safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _decode(value, name=""):
    if not isinstance(value, str):
        return safe(value)
    value = value.strip()
    if not value:
        return None
    if name in _TEXT_FIELDS:
        return value
    if value.lower() in {"nan", "inf", "-inf", "none", "null"}:
        return None
    if value.startswith(("[", "{")) or value in {"true", "false"}:
        try:
            return safe(json.loads(value))
        except (ValueError, TypeError):
            pass
    if _NUMBER.fullmatch(value):
        return float(value) if any(x in value.lower() for x in (".", "e")) else int(value)
    return value


def _cohort(manifest, row):
    config = manifest.get("config") or {}
    metadata = manifest.get("model_metadata") or {}
    signature = manifest.get("config_signature") or hashlib.sha256(_canonical(config).encode()).hexdigest()
    window = row.get("window") or row.get("window_signature") or manifest.get("window")
    if window is None:
        sampling = config.get("sampling") or {}
        window = {key: config.get(key) for key in ("local_attn_size", "sink_size")}
        window["protect_recent_latent_frames"] = sampling.get("protect_recent_latent_frames")
        # An explicitly supplied effective window remains part of its cohort.
        for key in ("window_size", "effective_window_tokens", "grid_h", "grid_w", "key_token_count"):
            if row.get(key) is not None:
                window[key] = row[key]
    result = {
        "config_signature": str(signature),
        "data_origin": manifest.get("data_origin") or row.get("data_origin") or "unknown",
        "model": row.get("model") or manifest.get("model") or manifest.get("model_id") or metadata.get("model") or "unknown",
        "checkpoint": row.get("checkpoint") or manifest.get("checkpoint") or manifest.get("checkpoint_path") or metadata.get("checkpoint") or "unknown",
        "window": window if isinstance(window, str) else _canonical(window),
        "source": row.get("source") or manifest.get("source") or config.get("source") or "unknown",
    }
    result["cohort_id"] = hashlib.sha256(_canonical(result).encode()).hexdigest()[:16]
    return result


def _load(input_root):
    root = Path(input_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    manifests = [root] if root.is_file() and root.name == "manifest.json" else sorted(root.rglob("manifest.json"))
    rows, seen, valid_runs = [], set(), []
    diagnostics = Counter()
    for path in manifests:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            diagnostics["invalid_manifests"] += 1
            continue
        if not isinstance(manifest, dict) or manifest.get("schema_version") != "research.v1":
            diagnostics["excluded_non_research_manifests"] += 1
            continue
        valid_runs.append(str(path))
        for part in sorted(path.parent.glob("*/events-*.csv")):
            with part.open(newline="", encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    if raw.get("schema_version") != "research.v1" or raw.get("experiment_id") not in EXPERIMENTS:
                        diagnostics["excluded_non_research_rows"] += 1
                        continue
                    row = {key: _decode(value, key) for key, value in raw.items() if key is not None}
                    details = row.get("details_json") or {}
                    if isinstance(details, str):
                        try:
                            details = json.loads(details)
                        except ValueError:
                            details = {}
                            diagnostics["invalid_details_json"] += 1
                    if not isinstance(details, dict):
                        details = {}
                        diagnostics["invalid_details_json"] += 1
                    row["details_json"] = safe(details)
                    for key, value in details.items():
                        if key not in {"schema_version", "experiment_id", "event_id"} and row.get(key) is None:
                            row[key] = _decode(value, key)
                    for key in ("prompt_index", "seed"):
                        if row.get(key) is None:
                            row[key] = _decode(manifest.get(key), key)
                    row.update(_cohort(manifest, row))
                    event = row.get("event_id")
                    identity = (row["cohort_id"], _canonical(row.get("prompt_index")), _canonical(row.get("seed")), event)
                    if event and identity in seen:
                        diagnostics["duplicate_event_ids"] += 1
                        continue
                    if event:
                        seen.add(identity)
                    row["manifest_path"], row["csv_path"] = str(path), str(part)
                    rows.append(row)
    return rows, {"input_root": str(root), "manifests": valid_runs, "ingestion": dict(diagnostics)}


def load_rows(input_root):
    """Decode all research.v1 event rows, deduplicating only within a cohort/run unit."""
    return _load(input_root)[0]


def metric_names(rows):
    """Return measured metric columns; identifiers and dimensions are excluded."""
    names = set(METRICS)
    for row in rows:
        for key, value in row.items():
            if key.endswith(("_relative_error", "_energy_ratio", "_interaction", "_reappearance")):
                names.add(key)
    return sorted(name for name in names if any(_number(row.get(name)) is not None for row in rows))


def _synthetic(row):
    return str(row.get("data_origin", "")).lower().startswith("synthetic")


def _real(row):
    return is_real_origin(str(row.get("data_origin", "unknown")).lower())


def summarize_rows(rows, *, bootstrap=1000, seed=0):
    """Aggregate within prompt+seed, then bootstrap those independent units.

    Layer/head observations remain within their prompt/seed cluster. No
    confidence interval is reported when a group has fewer than two units.
    """
    prepared = []
    for row in rows:
        item = dict(row)
        for field in GROUP_FIELDS:
            if isinstance(item.get(field), (dict, list, tuple)):
                item[field] = _canonical(item[field])
        for metric in QUALITY_METRICS:
            if item.get(metric) is None:
                item[metric] = item.get(metric.lower())
        prepared.append(item)
    statistics = []
    for metric in metric_names(prepared):
        statistics.extend(clustered_summary(prepared, metric, group_fields=GROUP_FIELDS, bootstrap=bootstrap, seed=seed))
    measured = [row for row in prepared if row.get("status") == "measured"]
    completed = sorted({row["experiment_id"] for row in measured if _real(row)})
    synthetic_completed = sorted({row["experiment_id"] for row in measured if _synthetic(row)})
    cohorts = []
    for cohort_id in sorted({row["cohort_id"] for row in rows}):
        cohort_rows = [row for row in prepared if row["cohort_id"] == cohort_id]
        actual = [row for row in cohort_rows if row.get("status") == "measured"]
        quality = {name: None for name in QUALITY_METRICS}
        for name in QUALITY_METRICS:
            entries = clustered_summary([row for row in actual if _real(row)], name, group_fields=("cohort_id",), bootstrap=bootstrap, seed=seed)
            if entries:
                quality[name] = entries[0]
        cohorts.append({"cohort_id": cohort_id, **{name: cohort_rows[0].get(name) for name in COHORT_FIELDS},
                        "row_count": len(cohort_rows), "measured_row_count": len(actual),
                        "measured_experiments": sorted({row["experiment_id"] for row in actual}),
                        "independent_prompt_seed_units": len({(_canonical(row.get("prompt_index")), _canonical(row.get("seed"))) for row in actual}),
                        "quality_metrics": quality})
    fields = {experiment: sorted({name for name in metric_names(measured)
                                 if any(row["experiment_id"] == experiment and _number(row.get(name)) is not None for row in measured)})
              for experiment in EXPERIMENTS}
    quality = {name: ({cohort["cohort_id"]: cohort["quality_metrics"][name] for cohort in cohorts
                       if cohort["quality_metrics"][name] is not None} or None) for name in QUALITY_METRICS}
    return {"schema_version": "research.v1", "row_count": len(rows), "measured_row_count": len(measured),
            "completed_experiments": completed, "synthetic_completed_experiments": synthetic_completed,
            "unrun_experiments": [name for name in EXPERIMENTS if name not in completed],
            "quality_metrics": quality, "cohorts": cohorts, "statistics": statistics,
            "measured_fields": fields, "status_counts": dict(Counter(row.get("status") or "unknown" for row in rows)),
            "uncertainty_unit": "prompt_index+seed; layers and heads are repeated observations",
            "limitations": ["Synthetic records validate execution only and support no scientific conclusion.",
                            "Weighted merging of nonidentical keys/values is approximate, not lossless.",
                            "Missing generation and quality measurements remain null; local error is not generation quality."]}


def _write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(SCHEMA) + sorted({key for row in rows for key in row} - set(SCHEMA))
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _canonical(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items() if key in fields})


def _plt():
    # Import lazily: rebuilding tables alone does not initialize a GUI backend.
    config_dir = Path(tempfile.gettempdir()) / "tempokv-research-matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 120, "pdf.fonttype": 42})
    return plt


def _measured(rows):
    return [row for row in rows if row.get("status") == "measured"]


def _note(ax, message):
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", wrap=True)
    ax.set_axis_off()


def _pairs(rows, x, y):
    return [(float(row[x]), float(row[y]), row) for row in rows
            if _number(row.get(x)) is not None and _number(row.get(y)) is not None]


def _scatter(ax, rows, x, y, label="strategy"):
    values = _pairs(rows, x, y)
    groups = defaultdict(list)
    for a, b, row in values:
        groups[str(row.get(label) or "measured")].append((a, b))
    for name, points in sorted(groups.items()):
        arr = np.asarray(points)
        ax.scatter(arr[:, 0], arr[:, 1], s=20, alpha=.7, label=name.replace("_", " "))
    if values:
        ax.set_xlabel(x.replace("_", " ")); ax.set_ylabel(y.replace("_", " "))
        ax.grid(alpha=.2)
        if len(groups) <= 10:
            ax.legend(fontsize=7, loc="best")
    return bool(values)


def _categories(ax, rows, metric, fields):
    valid = [row for row in rows if _number(row.get(metric)) is not None]
    if not valid:
        return False
    # Use the same prompt/seed cluster estimator for plotted group means.
    copied = []
    for row in valid:
        item = dict(row)
        item["plot_group"] = " | ".join(str(row.get(field)) for field in fields if row.get(field) is not None) or "measured"
        copied.append(item)
    stats = clustered_summary(copied, metric, group_fields=("plot_group",), bootstrap=1000)
    labels = [entry["plot_group"].replace("_", " ") for entry in stats]
    xs = np.arange(len(stats))
    ax.scatter(xs, [entry["mean"] for entry in stats], s=28)
    for x, entry in zip(xs, stats):
        if entry["ci_low"] is not None:
            ax.vlines(x, entry["ci_low"], entry["ci_high"], linewidth=1)
    # Keep all measured points while avoiding unreadable overlapping labels in
    # large precision grids. Full labels and individual records remain in CSV.
    ticks = np.unique(np.linspace(0, len(stats) - 1, min(12, len(stats))).round().astype(int))
    tick_labels = [labels[index] if len(labels[index]) <= 45 else labels[index][:42] + "..." for index in ticks]
    ax.set_xticks(ticks, tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(metric.replace("_", " ")); ax.grid(axis="y", alpha=.2)
    return True


def _first_pair(rows, xs, ys):
    for x in xs:
        for y in ys:
            if _pairs(rows, x, y):
                return x, y
    return None


def _draw(plt, experiment, stem, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), squeeze=False)
    a, b = axes[0]
    used = False
    if experiment == "S1":
        energy = [row for row in rows if "low_energy" in str(row.get("score_type", "")) or row.get("low_energy_ratio") is not None]
        if stem == "frequency_energy_by_layer_head":
            spatial = [row for row in energy if "spatial" in str(row.get("score_type", "")) or row.get("metric_scope") == "clean_dense_spatial"]
            for ax, role in ((a, "k"), (b, "v")):
                selected = [row for row in spatial if row.get("role") == role]
                metric = "score" if any(_number(row.get("score")) is not None for row in selected) else "low_energy_ratio"
                ok = _categories(ax, selected, metric, ("layer", "head", "cutoff"))
                used |= ok
                if not ok: _note(ax, f"{role.upper()} clean spatial energy: not measured")
                else: ax.set_ylim(0, 1); ax.set_ylabel("Low-frequency energy ratio")
                ax.set_title(f"{role.upper()}: layer | head | cutoff")
        else:
            for ax, domain in ((a, "spatial"), (b, "temporal")):
                selected = [row for row in energy if domain in str(row.get("score_type", ""))]
                metric = "score" if any(_number(row.get("score")) is not None for row in selected) else "low_energy_ratio"
                ok = _categories(ax, selected, metric, ("role", "cutoff", "temporal_window", "temporal_control"))
                used |= ok
                if not ok: _note(ax, f"Clean {domain} energy: not measured")
                else: ax.set_ylim(0, 1); ax.set_ylabel("Low-frequency energy ratio")
                ax.set_title(f"Clean {domain} low-frequency energy")
    elif experiment == "S2":
        ok = _scatter(a, rows, "score", "local_attention_relative_error", "score_type")
        if not ok: _note(a, "Paired candidate scores and attention oracle: not measured")
        corr = _categories(b, rows, "spearman", ("score_type",))
        if not corr: _note(b, "Candidate rank correlations: not measured")
        a.set_title("Sampled candidates"); b.set_title("Prompt/seed clustered rank correlation")
        used = ok or corr
    elif experiment == "E1":
        pair = _first_pair(rows, ("local_attention_relative_error",), ("model_output_relative_error", "clean_latent_relative_error", "generation_relative_error"))
        ok = _scatter(a, rows, *pair) if pair else False
        if not ok: _note(a, "Generation importance: not measured\nLocal errors do not establish generation quality")
        local = _scatter(b, rows, "score", "local_attention_relative_error", "score_type")
        if not local: local = _categories(b, rows, "local_attention_relative_error", ("strategy", "intervention"))
        if not local: _note(b, "Local attention importance: not measured")
        a.set_title("Local vs generation importance"); b.set_title("Measured local attention importance")
        used = ok or local
    elif experiment == "E3":
        regular = [row for row in rows if row.get("strategy") != "conditional_drop"]
        conditional = [row for row in rows if row.get("strategy") == "conditional_drop"]
        left = _categories(a, regular, "local_attention_relative_error", ("strategy", "budget_cohort"))
        right = _categories(b, conditional, "local_attention_relative_error", ("conditioning_set",))
        if not left: _note(a, "Replacement / merge comparisons: not measured")
        if not right: _note(b, "Conditioned marginal impact: not measured")
        a.set_title("Local errors at recorded budgets"); b.set_title("Marginal error vs conditioned reference")
        used = left or right
    elif experiment == "E4":
        left = _scatter(a, rows, "strength", "local_attention_relative_error", "role")
        quantized = [row for row in rows if row.get("k_bits") is not None and row.get("v_bits") is not None]
        right = _categories(b, quantized, "local_attention_relative_error", ("k_bits", "v_bits", "budget_cohort"))
        if not right:
            pair = _first_pair(rows, ("allocated_candidate_bytes", "logical_bytes"), ("local_attention_relative_error",))
            right = _scatter(b, rows, *pair) if pair else False
        if not left: _note(a, "Matched K/V sensitivity: not measured")
        if not right: _note(b, "K/V bit-budget comparisons: not measured")
        a.set_title("K / V / KV perturbation sensitivity"); b.set_title("K bits | V bits | recorded budget")
        used = left or right
    elif experiment == "E2":
        pair = _first_pair(rows, ("horizon_chunks",), ("model_output_relative_error", "clean_latent_relative_error", "local_attention_relative_error"))
        if pair and any(_number(row.get("layer")) is not None for row in rows):
            x, y = pair
            subset = [row for row in rows if _number(row.get(x)) is not None and _number(row.get(y)) is not None and _number(row.get("layer")) is not None]
            stats = clustered_summary(subset, y, group_fields=("layer", x))
            layers = sorted({entry["layer"] for entry in stats}); horizons = sorted({entry[x] for entry in stats})
            matrix = np.full((len(layers), len(horizons)), np.nan)
            for entry in stats: matrix[layers.index(entry["layer"]), horizons.index(entry[x])] = entry["mean"]
            image = a.imshow(np.ma.masked_invalid(matrix), aspect="auto", interpolation="nearest")
            fig.colorbar(image, ax=a, label=y.replace("_", " "))
            a.set_xticks(range(len(horizons)), horizons); a.set_yticks(range(len(layers)), layers)
            a.set_xlabel("Horizon chunks"); a.set_ylabel("Layer"); used = True
        elif pair: used = _scatter(a, rows, *pair)
        if not used: _note(a, "Error propagation: not measured")
        _note(b, "Blank heatmap cells are unmeasured.\nUncertainty units are prompt + seed.")
    else:
        fields = {
            "S3": (("logical_bytes", "actual_retention", "retention"), ("local_attention_relative_error", "model_output_relative_error", "error")),
            "E6": (("denoise_step_index", "denoise_timestep"), ("model_output_relative_error", "clean_latent_relative_error", "local_attention_relative_error", "sensitivity")),
            "E7": (("retention", "actual_retention", "k_bits"), ("interaction", "interaction_error", "local_attention_relative_error")),
            "E5": (("horizon_chunks", "reappearance_frame", "frame_age", "lag"), ("reappearance_score", "clean_latent_relative_error", "local_attention_relative_error")),
        }
        xs, ys = fields[experiment]
        pair = _first_pair(rows, xs, ys)
        used = _scatter(a, rows, *pair) if pair else False
        if not used: _note(a, "Required paired fields: not measured")
        _note(b, "Only recorded measured fields are shown.\nNo missing values are filled with zero.")
    fig.suptitle(f"{experiment} — {stem.replace('_', ' ')}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    return fig, used


def propagation_partitions(rows):
    """Never pool distinct error doses or unmatched calibration in a heatmap."""
    groups={}
    fields=("intervention","role","requested_strength","calibration_protocol","denoise_step_index","head")
    for row in rows:
        if row.get("calibration_matched") is False or str(row.get("comparison_cohort","")).startswith("unmatched"):
            continue
        key=tuple(row.get(field) for field in fields)
        groups.setdefault(key,[]).append(row)
    return [(hashlib.sha256(_canonical(key).encode()).hexdigest()[:10],part) for key,part in sorted(groups.items(),key=lambda item:_canonical(item[0]))]


def write_figures(rows, output_dir):
    """Write separate PNG/PDF figures per cohort; report missing plots explicitly."""
    rows = list(rows)
    if not rows:
        return []
    plt = _plt()
    destination = Path(output_dir)
    results = []
    for cohort_id in sorted({row["cohort_id"] for row in rows}):
        cohort = [row for row in rows if row["cohort_id"] == cohort_id]
        for experiment in EXPERIMENTS:
            actual = _measured([row for row in cohort if row["experiment_id"] == experiment])
            figure_inputs=[(stem,actual) for stem in FIGURES[experiment]]
            if experiment=="E2" and actual:
                figure_inputs=[(f"{FIGURES[experiment][0]}-{identity}",part) for identity,part in propagation_partitions(actual)]
                if not figure_inputs:
                    results.append({"cohort_id":cohort_id,"experiment_id":experiment,"name":FIGURES[experiment][0],
                        "status":"not_measured","reason":"no comparable matched calibration cohort"})
            for stem,plot_rows in figure_inputs:
                entry = {"cohort_id": cohort_id, "experiment_id": experiment, "name": stem}
                if experiment=="E2" and plot_rows:
                    entry["plot_protocol"]={field:plot_rows[0].get(field) for field in
                        ("intervention","role","requested_strength","calibration_protocol","denoise_step_index","head")}
                if not actual:
                    results.append({**entry, "status": "not_measured", "reason": "no measured rows"})
                    continue
                fig, used = _draw(plt, experiment, stem, plot_rows)
                try:
                    if used:
                        fig.text(.5, .925, f"{cohort[0]['data_origin']} | source: {cohort[0]['source']} | cohort: {cohort_id}",
                                 ha="center", va="center", fontsize=8, color="dimgray")
                        if experiment=="E2":
                            fig.text(.5,.89," | ".join(f"{key}={value}" for key,value in entry["plot_protocol"].items()),
                                     ha="center",fontsize=6,color="dimgray")
                        folder = destination / "figures" / f"cohort-{cohort_id}"
                        folder.mkdir(parents=True, exist_ok=True)
                        paths = []
                        for suffix in ("png", "pdf"):
                            path = folder / f"{stem}.{suffix}"
                            if path.exists(): raise FileExistsError(path)
                            fig.savefig(path, dpi=180, bbox_inches="tight")
                            paths.append(str(path.relative_to(destination)))
                        results.append({**entry, "status": "measured", "paths": paths})
                    else:
                        results.append({**entry, "status": "not_measured", "reason": "required numeric fields absent"})
                finally:
                    plt.close(fig)
    return results


def _report(summary):
    real = ", ".join(summary["completed_experiments"]) or "none"
    synthetic = ", ".join(summary["synthetic_completed_experiments"]) or "none"
    unrun = ", ".join(summary["unrun_experiments"]) or "none"
    lines = ["# Research summary", "", f"Saved records: {summary['row_count']}; measured records: {summary['measured_row_count']}; separate cohorts: {len(summary['cohorts'])}.",
             "", f"Real measured experiments: {real}. Synthetic execution checks: {synthetic}.",
             f"Real experiments not run: {unrun}.", "", "| Experiment | Recorded measured fields |", "|---|---|"]
    for experiment in EXPERIMENTS:
        names = summary["measured_fields"][experiment]
        lines.append(f"| {experiment} | {', '.join(names) if names else 'not measured'} |")
    lines += ["", "Means, medians and standard deviations summarize prompt/seed means. Bootstrap intervals resample prompt/seed units; heads and layers are repeated observations. Different configurations, origins, models, checkpoints, windows and sources remain separate.",
              "", "Zero and negative observations are retained. Missing clean data, generation outcomes and quality scores remain unmeasured; local attention error alone does not establish generation importance.",
              "", "Synthetic records validate execution and support no scientific conclusion. Weighted merging is exact for duplicate keys/values with multiplicity, but merging nonidentical keys/values is not lossless.",
              "", "The CSVs preserve individual records. summary.json contains cohort-level statistics, missing-figure reasons and all figure paths."]
    return "\n".join(lines) + "\n"


def analyze(input_root, output_dir=None) -> Path:
    """Create a fresh analysis child from saved research.v1 events, without overwrites."""
    rows, ingestion = _load(input_root)
    root = Path(input_root).expanduser().resolve()
    base = Path(output_dir).expanduser().resolve() if output_dir is not None else (root.parent if root.is_file() else root) / "analysis"
    base.mkdir(parents=True, exist_ok=True)
    destination = base / f"analysis-{uuid.uuid4().hex[:12]}"
    destination.mkdir(exist_ok=False)
    _write_csv(destination / "merged.csv", rows)
    for experiment in EXPERIMENTS:
        _write_csv(destination / f"{experiment}.csv", (row for row in rows if row["experiment_id"] == experiment))
    summary = {**summarize_rows(rows), **ingestion}
    _write_csv(destination / "clustered_statistics.csv", summary["statistics"],
               fields=[*GROUP_FIELDS, "metric", "mean", "median", "std", "ci_low", "ci_high",
                       "independent_prompt_seed_units", "raw_observations", "aggregation"])
    summary["figures"] = write_figures(rows, destination)
    atomic_json(destination / "summary.json", summary)
    with (destination / "research_summary.md").open("x", encoding="utf-8") as handle:
        handle.write(_report(summary))
    return destination
