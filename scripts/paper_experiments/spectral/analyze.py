#!/usr/bin/env python3
"""Analyze suite B only, one labeled report per experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from tempokv.spectral_kv.config import EXPERIMENTS, SUITE_ID
from tempokv.spectral_kv.stats import fit_distance_decay


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def write_csv(path, rows):
    columns = sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()} for row in rows)


def analyze(input_dir, output_dir, experiment=None, _metadata=None):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    metadata = [] if _metadata is None else _metadata
    paths = ([input_dir] if input_dir.is_file() else sorted(input_dir.rglob("metadata.json"))) if _metadata is None else []
    for path in paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("suite_id") == SUITE_ID:
            metadata.append((path.parent, item))
    if not metadata:
        raise ValueError("no suite B metadata found; suite A is deliberately excluded")
    if output_dir.exists():
        output_dir = output_dir / ("analysis-" + uuid.uuid4().hex[:12])
    output_dir.mkdir(parents=True, exist_ok=False)
    cohorts = defaultdict(list)
    for root, item in metadata:
        key = (item.get("profile", "offline"), item.get("runtime", "offline"), item.get("local_attn_size", "unknown"),
               item.get("sink_size", 0), item.get("trajectory_changed", False))
        cohorts[key].append((root, item))
    if len(cohorts) > 1:
        index = ["# B_SPECTRAL_KV — separate settings", "", "Runtime, window, sink, profile and trajectory mode are never averaged together.", ""]
        for key, group in sorted(cohorts.items(), key=lambda pair: str(pair[0])):
            label = "setting-" + hashlib.sha256(str(key).encode()).hexdigest()[:10]
            analyze(input_dir, output_dir / label, experiment, _metadata=group)
            index.append(f"- [{key}]({label}/summary.md)")
        (output_dir / "summary.md").write_text("\n".join(index) + "\n", encoding="utf-8")
        return output_dir
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tempokv-spectral-kv-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [experiment] if experiment else sorted(EXPERIMENTS)
    index = ["# B_SPECTRAL_KV — separate experiment reports", "",
             "Suite A means frequency / sensitivity / token selection; none of its outputs are merged here.", "",
             "Synthetic inputs, when present, are workflow checks and are not research evidence.", "",
             "| Experiment | Rows | Status |", "|---|---:|---|"]
    for exp in selected:
        destination = output_dir / exp
        destination.mkdir()
        rows = []
        for root, meta in metadata:
            source = root / exp / "records.jsonl"
            if source.is_file():
                for line in source.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    if row.get("suite_id", SUITE_ID) != SUITE_ID or row.get("experiment_id", exp) != exp:
                        raise ValueError("conflicting experiment identity in records")
                    rows.append({**row, "source_run": str(root), "profile": meta.get("profile", "offline"),
                                 "trajectory_changed": meta.get("trajectory_changed", False)})
        write_csv(destination / "records.csv", rows)
        measured = [r for r in rows if r.get("status", "ok") in {"ok", "measured"}]
        status = "measured" if measured else "not measured"
        index.append(f"| [{exp}]({exp}/summary.md) | {len(rows)} | {status} |")
        summary = [f"# {exp}", "", f"Suite: {SUITE_ID}; records: {len(rows)}; status: {status}.", ""]
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        points = False
        if exp in {"B_E1_SPECTRUM", "B_E5_RESIDUAL"}:
            groups = defaultdict(dict)
            for row in rows:
                if row.get("record_kind", "spatial") != "spatial" or not finite(row.get("r95")):
                    continue
                key = (row.get("profile"), row.get("role"), row.get("coordinate"), row.get("head"), row.get("source_kind", "original"))
                groups[key].setdefault(row.get("layer", 0), []).append(row["r95"])
            for key, values in groups.items():
                layers = sorted(values)
                ax.plot(layers, [np.mean(values[layer]) for layer in layers], marker="o", label=str(key))
                points = True
            ax.set(xlabel="Layer (heads and coordinates separate)", ylabel="r95 after removing DC", ylim=(-.02, 1.02))
            summary += ["r95 uses radial cycles/token divided by sqrt(0.5). DC-free results are undefined when no non-DC energy exists.", ""]
            dc_groups = defaultdict(list)
            for row in rows:
                if finite(row.get("dc_ratio")):
                    dc_groups[(row.get("layer"), row.get("head"), row.get("role"), row.get("coordinate"), row.get("source_kind"))].append(row["dc_ratio"])
            write_csv(destination / "dc_summary.csv", [{"layer": k[0], "head": k[1], "role": k[2], "coordinate": k[3],
                      "source_kind": k[4], "dc_ratio": float(np.mean(v))} for k, v in dc_groups.items()])
            temporal = [r for r in rows if r.get("record_kind") == "temporal"]
            write_csv(destination / "joint_temporal_spatial.csv", temporal)
        elif exp == "B_E4_DISTANCE":
            radius, fits = [], []
            for source_run in sorted({r["source_run"] for r in rows}):
                chosen = [r for r in rows if r["source_run"] == source_run]
                threshold = next((m.get("config", {}).get("correlation_threshold", .9) for root, m in metadata if str(root) == source_run), .9)
                rr, ff = fit_distance_decay(chosen, threshold)
                radius += [{**r, "source_run": source_run} for r in rr]
                fits += [{**r, "source_run": source_run} for r in ff]
            write_csv(destination / "rstar.csv", radius); write_csv(destination / "powerlaw_fits.csv", fits)
            grouped = defaultdict(list)
            for row in radius:
                if finite(row.get("rstar")) and row["rstar"] > 0:
                    grouped[(row.get("source_run"), row.get("layer"), row.get("head"), row.get("role"))].append(row)
            for key, values in grouped.items():
                values.sort(key=lambda r: r["lag"])
                ax.plot([r["lag"] for r in values], [r["rstar"] for r in values], marker="o", label=str(key[1:]))
                points = True
            ax.set(xlabel="Frame distance", ylabel="Stable frequency radius")
            summary += ["r*(distance) uses the contiguous low-frequency prefix above the configured correlation threshold. b is fitted only with at least two measured positive radii. No convergence conclusion is assumed.", ""]
        elif exp == "B_E2_CHUNK_AXIS":
            groups = defaultdict(list)
            for row in rows:
                value = row.get("reuse_rel_l2")
                if finite(value):
                    groups[(row.get("spatial_band", row.get("band", 0)), row.get("temporal_band", 0))].append(value)
            if groups:
                ns, nt = max(k[0] for k in groups) + 1, max(k[1] for k in groups) + 1
                heat = np.full((ns, nt), np.nan)
                for (s, t), values in groups.items(): heat[s, t] = np.mean(values)
                fig.colorbar(ax.imshow(heat, origin="lower", aspect="auto"), ax=ax, label="Mean reuse relative L2")
                points = True
            ax.set(xlabel="Temporal frequency band", ylabel="Spatial frequency band")
            summary += ["Same denoising step and adjacent chunk only. Warm-up, mismatched shapes, gaps and missing linear extrapolation history remain unmeasured.", ""]
        elif exp == "B_E2B_REFRESH":
            groups = defaultdict(list)
            for row in rows:
                value = row.get("relative_error", row.get("rel_l2"))
                if finite(value): groups[(row.get("role"), row.get("band", 0))].append(value)
            for role in sorted({key[0] for key in groups}):
                bands = sorted(key[1] for key in groups if key[0] == role)
                ax.plot(bands, [np.mean(groups[role, b]) for b in bands], marker="o", label=str(role)); points = True
            ax.set(xlabel="Spatial frequency band", ylabel="Last-denoise vs clean relative error")
            summary += ["KVSKIP_raw/lin change the trajectory and require a separate paired rollout comparison. Skipped clean passes cannot supply clean-reference spectra.", ""]
        else:
            groups = defaultdict(list)
            for row in rows:
                if finite(row.get("relative_attention_output_error")) and finite(row.get("tokens_attended")):
                    groups[row.get("method_spec", "unknown")].append(row)
            for method, values in groups.items():
                ax.scatter([r["tokens_attended"] for r in values], [r["relative_attention_output_error"] for r in values], label=method, s=18); points = True
            ax.set(xlabel="Actual tokens attended", ylabel="Relative attention output error")
            summary += ["E3-off candidates share the same BF16 window and sampled queries. E3-on modifies the rollout. The prototype retains native BF16 backing; resident bytes include it and are not a production memory-saving result.", ""]
        if points:
            if exp != "B_E2_CHUNK_AXIS": ax.legend(fontsize=6, loc="best")
        else:
            ax.text(.5, .5, "not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(exp)
        fig.savefig(destination / "results.png", dpi=140); plt.close(fig)
        summary += ["VBench, PSNR, LPIPS, FID and GPU throughput: not measured by this analysis. See separately supplied paired-rollout/evaluation files for actual measurements.", ""]
        (destination / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    args = parser.parse_args(argv)
    print(analyze(args.input, args.output, args.experiment))
    return 0


if __name__ == "__main__": raise SystemExit(main())
