"""Small durable tables and plots for fixed-budget evidence."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def write_capacity(report: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "capacity.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = report["rows"]
    if rows:
        with (output_dir / "capacity.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        report["plot_status"] = "matplotlib_unavailable; raw_table_written"
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in dict.fromkeys(row["method"] for row in rows):
        values = sorted((r for r in rows if r["method"] == method and r["history_frames"] is not None),
                        key=lambda r: r["budget_bytes"])
        ax.plot([r["budget_gib"] for r in values], [r["history_frames"] for r in values], "o-", label=method)
    ax.set(xlabel="KV budget (GiB)", ylabel="Maximum history (latent frames)",
           title="Independent-span capacity (shape accounting)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "capacity.png", dpi=180)
    plt.close(fig)


def summarize_generation(output_dir: Path) -> dict:
    rows = []
    for path in sorted((output_dir / "generation").rglob("job.json")):
        job = json.loads(path.read_text())
        report_path = path.parent / "memory_report.json"
        metrics = json.loads(report_path.read_text()) if report_path.is_file() else {}
        rows.append({"name": job["name"], "method": job["method"],
                     "comparison": job["comparison"], "history_frames": job["history_frames"],
                     "budget_bytes": job["budget_bytes"], "status": job["status"],
                     "measured_peak_bytes": metrics.get("peak_checked_bytes"),
                     "final_kv_owned_bytes": metrics.get("kv_owned_bytes"),
                     "parameter_bytes": metrics.get("parameter_bytes"),
                     "memory_report": str(report_path) if report_path.is_file() else None})
    report = {"schema_version": "tempokv.rebuttal.memory.summary.v1", "rows": rows,
              "status": "summarized" if rows else "no_generation_results",
              "quality_status": "requires_paired_reference_video_metrics_and_vbench_outputs"}
    (output_dir / "generation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if rows:
        with (output_dir / "generation_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report
