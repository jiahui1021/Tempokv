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
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _cohort_spec(metadata):
    """Keep replicate identities out, but preserve measurement definitions.

    Seeds and row-retention limits identify repetitions/storage rather than a
    different measurement. All remaining config fields are conservative cohort
    boundaries, including sampled layers/heads/steps, band counts, channels,
    context limits, methods, precision and calibration paths.
    """
    config = metadata.get("config", {})
    config = config if isinstance(config, dict) else {"unparsed_config": config}
    excluded = {"profile", "seed", "max_stat_rows", "experiments"}
    spec = {
        "profile": metadata.get("profile", "offline"),
        "runtime": metadata.get("runtime", "offline"),
        "local_attn_size": metadata.get("local_attn_size", "unknown"),
        "sink_size": metadata.get("sink_size", 0),
        "trajectory_changed": metadata.get("trajectory_changed", False),
        "measurement_config": {key: value for key, value in config.items() if key not in excluded},
    }
    # Paths/identities are explicit: model and checkpoint must not collapse to
    # the runtime family. Unknown or absent provenance stays visible as such.
    for key in ("model", "checkpoint", "checkpoint_path", "model_root", "model_config", "runtime_config",
                "git_commit", "source", "coordinate_system", "source_contract", "grid", "effective_grids",
                "heads", "head_dim", "stride", "latent_frames", "k_params", "v_params"):
        spec[key] = metadata.get(key)
    terminal = metadata.get("terminal_clean_collection", {})
    spec["terminal_clean_policy"] = {key: terminal.get(key) for key in
        ("retain_final_cache_effective", "additional_clean_requested")}
    return spec


def _provenance(root, metadata):
    cohort = _cohort_spec(metadata)
    return {
        "source_run": str(root),
        "source_id": hashlib.sha256(str(root).encode()).hexdigest()[:16],
        "cohort_id": hashlib.sha256(_canonical(cohort).encode()).hexdigest()[:16],
        "profile": metadata.get("profile", "offline"),
        "model": metadata.get("model"),
        "checkpoint": metadata.get("checkpoint", metadata.get("checkpoint_path")),
        "model_root": metadata.get("model_root"),
        "source": metadata.get("source", "not specified"),
        "trajectory_changed": metadata.get("trajectory_changed", False),
    }


def _spectrum_observations(rows):
    """One common metric row per observed frame/window, independent of bands."""
    identity = ("source_run", "prompt_id", "sample_id", "rank", "layer", "head", "role", "coordinate",
                "source_kind", "record_kind", "dc_definition", "frame_idx", "frame_idx0", "chunk_idx", "step_idx",
                "phase", "window_start_frame", "window_end_frame", "window_frames", "window_size", "predictor")
    metrics = ("dc_ratio", "r95", "r95_excl_dc", "r95_incl_dc", "r95_cycles", "total_power",
               "non_dc_power", "zero_signal", "zero_non_dc", "valid", "status")
    provenance = ("source_id", "cohort_id", "profile", "model", "checkpoint", "model_root", "source", "trajectory_changed")
    observations = {}
    for source in rows:
        if not any(key in source for key in metrics):
            continue
        row = dict(source)
        kind = row.get("record_kind", "temporal" if row.get("statistic") == "temporal_spectrum" else "spatial")
        row["record_kind"] = kind
        row["dc_definition"] = row.get("dc_definition", "joint_t0_h0_w0" if kind == "temporal" else "spatial_h0_w0")
        key = _canonical([row.get(field) for field in identity])
        projected = {field: row.get(field) for field in (*identity, *metrics, *provenance)}
        if key in observations:
            previous = observations[key]
            if any(previous[field] != projected[field] for field in metrics):
                raise ValueError("inconsistent common spectrum metrics across repeated frequency bands")
        else:
            observations[key] = projected
    return list(observations.values())


def _dc_summary(observations):
    fields = ("record_kind", "dc_definition", "layer", "head", "role", "coordinate", "source_kind",
              "window_frames", "window_size", "model", "checkpoint", "cohort_id")
    groups = defaultdict(list)
    for row in observations:
        if finite(row.get("dc_ratio")) and row.get("valid", True) is not False:
            groups[tuple(row.get(field) for field in fields)].append(row)
    return [{**dict(zip(fields, key)), "dc_ratio": float(np.mean([r["dc_ratio"] for r in values])),
             "observation_count": len(values), "source_runs": sorted({r["source_run"] for r in values})}
            for key, values in sorted(groups.items(), key=lambda item: str(item[0]))]


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
    excluded=[];selected=[]
    for root,item in metadata:
        state=item.get("status")
        if state is not None and state!="completed":
            excluded.append({"source_run":str(root),"status":state})
        else:selected.append((root,item))
    metadata=selected
    if output_dir.exists():
        output_dir = output_dir / ("analysis-" + uuid.uuid4().hex[:12])
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir/"analysis_selection.json").open("x",encoding="utf-8") as handle:
        json.dump({"excluded_incomplete_or_failed":excluded,"retained_runs":len(metadata),
                   "legacy_unknown_status_runs":sum(item.get("status") is None for _,item in metadata)},handle,indent=2)
        handle.write("\n")
    cohorts = defaultdict(list)
    for root, item in metadata:
        key = _canonical(_cohort_spec(item))
        cohorts[key].append((root, item))
    sources = [{**_provenance(root, item), "cohort": _cohort_spec(item), "metadata": item} for root, item in metadata]
    with (output_dir / "sources.json").open("x", encoding="utf-8") as handle:
        json.dump(sources, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    if len(cohorts) > 1:
        index = ["# B_SPECTRAL_KV — separate settings", "",
                 "Model/checkpoint, measurement configuration, runtime, window, sink and trajectory mode are never averaged across cohorts.",
                 "Full source metadata and configuration are preserved in [sources.json](sources.json).", ""]
        for key, group in sorted(cohorts.items(), key=lambda pair: str(pair[0])):
            label = "setting-" + hashlib.sha256(str(key).encode()).hexdigest()[:10]
            analyze(input_dir, output_dir / label, experiment, _metadata=group)
            spec = _cohort_spec(group[0][1])
            index.append(f"- [{label}: model={spec['model']}; checkpoint={spec['checkpoint']}; runtime={spec['runtime']}; profile={spec['profile']}]({label}/summary.md)")
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
             "Full source metadata and configuration: [sources.json](sources.json).", "",
             "| Experiment | Rows | Status |", "|---|---:|---|"]
    for exp in selected:
        destination = output_dir / exp
        destination.mkdir()
        rows = []
        for root, meta in metadata:
            source = root / exp / "records.jsonl"
            if source.is_file():
                provenance = _provenance(root, meta)
                for line in source.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    if row.get("suite_id", SUITE_ID) != SUITE_ID or row.get("experiment_id", exp) != exp:
                        raise ValueError("conflicting experiment identity in records")
                    rows.append({**row, **provenance})
        write_csv(destination / "records.csv", rows)
        measured = [r for r in rows if r.get("status", "ok") in {"ok", "measured"}]
        status = "measured" if measured else "not measured"
        index.append(f"| [{exp}]({exp}/summary.md) | {len(rows)} | {status} |")
        summary = [f"# {exp}", "", f"Suite: {SUITE_ID}; records: {len(rows)}; status: {status}.", ""]
        summary += ["Source model/checkpoint and complete measurement settings are preserved in [sources.json](../sources.json).", ""]
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        points = False
        if exp in {"B_E1_SPECTRUM", "B_E5_RESIDUAL"}:
            observations = _spectrum_observations(rows)
            write_csv(destination / "spectrum_observations.csv", observations)
            groups = defaultdict(dict)
            for row in observations:
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
            dc_summary = _dc_summary(observations)
            write_csv(destination / "dc_summary.csv", dc_summary)
            write_csv(destination / "spatial_dc_summary.csv", [r for r in dc_summary if r["record_kind"] == "spatial"])
            write_csv(destination / "joint_dc_summary.csv", [r for r in dc_summary if r["record_kind"] == "temporal"])
            summary += ["Spatial DC and joint temporal/spatial DC are summarized separately. Common frame/window metrics are counted once, independent of repeated frequency-band rows. Joint-window lengths remain separate during warm-up.", ""]
            temporal = [r for r in rows if r.get("record_kind") == "temporal"]
            write_csv(destination / "joint_temporal_spatial.csv", temporal)
        elif exp == "B_E4_DISTANCE":
            radius, fits = [], []
            for source_run in sorted({r["source_run"] for r in rows}):
                chosen = [r for r in rows if r["source_run"] == source_run]
                threshold = next((m.get("config", {}).get("correlation_threshold", .9) for root, m in metadata if str(root) == source_run), .9)
                rr, ff = fit_distance_decay(chosen, threshold)
                provenance = {key: chosen[0][key] for key in _provenance(source_run, {})}
                radius += [{**r, **provenance} for r in rr]
                fits += [{**r, **provenance} for r in ff]
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
