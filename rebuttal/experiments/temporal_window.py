"""Paired fidelity by real temporal windows; no generation is hidden here."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from rebuttal.common.statistics import grouped_summary
from rebuttal.common.video_metrics import (
    aggregate_frames, iter_pair_metrics, make_quality_backend, pair_key, read_csv,
    resolve_pairs, write_csv, write_json,
)


def summarize(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    windows = read_csv(output_dir / "windows.csv")
    videos = read_csv(output_dir / "videos.csv")
    group = ["model", "method", "comparison_id"]
    settings = {"samples": int(config.get("bootstrap_samples", 1000)), "seed": int(config.get("bootstrap_seed", 0))}
    metrics = ["psnr_db", "ssim", "lpips"]
    summary = {
        "schema_version": "tempokv.rebuttal.temporal_window.v1", "status": "complete",
        "metric_scope": "end_to_end_video_fidelity", "window_seconds": config.get("window_seconds", 15),
        "n_video_pairs": len(videos), "n_windows": len(windows),
        "windows": grouped_summary(windows, [*group, "window_index"], metrics, **settings),
        "videos": grouped_summary(videos, group, metrics, **settings),
        "zero_error_windows": sum(bool(r["zero_error"]) for r in windows),
        "psnr_zero_error_policy": "null_with_zero_error_true_means_infinity_excluded_from_finite_means",
        "timing_policy": "actual_pts_with_pyav; explicit_cfr_assumption_with_opencv",
    }
    by_video: dict[tuple, list[dict[str, Any]]] = {}
    for row in windows:
        key = tuple(row.get(k) for k in ("model", "method", "comparison_id", "prompt_id", "seed", "sample_id"))
        by_video.setdefault(key, []).append(row)
    changes = []
    for key, rows in by_video.items():
        rows.sort(key=lambda r: r["window_index"])
        entry = dict(zip(("model", "method", "comparison_id", "prompt_id", "seed", "sample_id"), key, strict=True))
        for metric in metrics:
            entry[f"{metric}_last_minus_first"] = rows[-1][metric] - rows[0][metric] if rows[-1].get(metric) is not None and rows[0].get(metric) is not None else None
        changes.append(entry)
    write_csv(output_dir / "first_last_changes.csv", changes)
    summary["first_last_changes"] = grouped_summary(changes, group, [f"{k}_last_minus_first" for k in metrics], **settings)
    write_json(output_dir / "summary.json", summary)
    flattened = []
    for row in summary["windows"]:
        for metric, values in row["metrics"].items():
            flattened.append({**{k: v for k, v in row.items() if k != "metrics"}, "metric": metric, **values})
    write_csv(output_dir / "summary.csv", flattened)
    if config.get("plots", True):
        _plot(summary, windows, output_dir)
    return summary


def _plot(summary: dict[str, Any], windows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("temporal-window plots require matplotlib; use plots=false to explicitly omit figures") from exc
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in summary["windows"]:
        grouped.setdefault((row["model"], row["method"], row["comparison_id"]), []).append(row)
    for key, rows in grouped.items():
        rows.sort(key=lambda r: r["window_index"])
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for axis, metric in zip(axes, ("psnr_db", "ssim", "lpips"), strict=True):
            valid = [r for r in rows if r["metrics"][metric]["mean"] is not None]
            xs = [(int(r["window_index"]) * float(summary["window_seconds"])) for r in valid]
            means = [r["metrics"][metric]["mean"] for r in valid]
            axis.plot(xs, means, marker="o")
            cis = [(i, r["metrics"][metric]) for i, r in enumerate(valid) if r["metrics"][metric]["ci_low"] is not None]
            for i, value in cis:
                axis.vlines(xs[i], value["ci_low"], value["ci_high"], alpha=0.6)
            axis.set(xlabel="Window start (s)", ylabel=metric)
            axis.grid(alpha=0.2)
        durations = sorted({round(float(w["duration_seconds"]), 4) for w in windows if (w["model"], w["method"], w["comparison_id"]) == key})
        fig.suptitle(f"{' / '.join(map(str,key))}; actual window durations: {durations} s", fontsize=9)
        fig.tight_layout()
        name = "_".join(str(x).replace("/", "_") for x in key)
        fig.savefig(output_dir / f"temporal_{name}.png", dpi=180)
        plt.close(fig)


def run(config: dict[str, Any], stage: str, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage == "plan":
        try:
            pairs, reason = resolve_pairs(config), None
        except (ValueError, FileNotFoundError) as exc:
            pairs, reason = [], str(exc)
        result = {"experiment": "temporal_window", "status": "planned", "pairs": pairs,
                  "missing_input": reason, "metrics": config.get("metrics", ["psnr", "ssim", "lpips"]),
                  "window_seconds": config.get("window_seconds", 15), "supported_stages": ["plan", "evaluate", "summarize"]}
        write_json(output_dir / "plan.json", result)
        return result
    if stage == "summarize":
        return summarize(output_dir, config)
    if stage != "evaluate":
        raise ValueError("temporal_window supports plan/evaluate/summarize; generation uses the model adapter")
    pairs = resolve_pairs(config)
    backend = make_quality_backend(config)
    windows, videos = [], []
    write_json(output_dir / "video_pairs.json", {"pairs": pairs, "config": config})
    frames_path = output_dir / "frames.csv"
    with frames_path.open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for pair in pairs:
            rows = []
            for row in iter_pair_metrics(pair, config, backend=backend):
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
            stream.flush()
            current = aggregate_frames(rows)
            for row in current:
                row["partial_window"] = row["duration_seconds"] + 1e-6 < float(config.get("window_seconds", 15))
            windows.extend(current)
            videos.extend(aggregate_frames(rows, group_by_window=False))
            write_csv(output_dir / "windows.csv", windows)
            write_csv(output_dir / "videos.csv", videos)
    return summarize(output_dir, config)
