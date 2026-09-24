#!/usr/bin/env python3
"""Run the fixed-717 and generation-horizon latency protocols.

Every runtime is an adapter command supplied by the manifest.  The harness
owns ordering, warmups, repeat identity, validation, aggregation, and figures;
the adapter owns keeping the model loaded and writing one canonical timing
report to ``$PAPER_EXPERIMENT_REPORT``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_experiments.common import (  # noqa: E402
    COMPILATION_SCHEMA,
    FIXED_FRAMES,
    FROZEN_KEYS,
    HORIZONS,
    MANIFEST_SCHEMA,
    METHODS,
    REPEATS,
    RUN_SCHEMA,
    TIMING_KEYS,
    ContractError,
    read_json,
    summary,
    validate_compilation_report,
    validate_run_report,
    write_json,
)


SUPPORTED_SESSION_SCOPE = "coordinator_session_multiple_processes"
FORBIDDEN_SESSION_SCOPE = "single_process"


def _manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ContractError("manifest must be a JSON object")
    required = {
        "schema_version",
        "experiment_id",
        "output_dir",
        "horizons",
        "fixed_frames",
        "repeats",
        "session_scope",
        "frozen",
        "methods",
    }
    missing = required - set(payload)
    extra = set(payload) - required
    if missing or extra:
        raise ContractError(
            f"manifest fields differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if payload["schema_version"] != MANIFEST_SCHEMA:
        raise ContractError(f"schema_version must be {MANIFEST_SCHEMA!r}")
    horizons = payload["horizons"]
    if not isinstance(horizons, list) or not horizons:
        raise ContractError("horizons must be a non-empty list")
    if any(frame not in HORIZONS for frame in horizons):
        raise ContractError(f"each horizon must be drawn from {list(HORIZONS)}")
    if horizons != sorted(horizons):
        raise ContractError(f"horizons must be sorted ascending: {horizons}")
    if horizons != [h for h in HORIZONS if h in horizons]:
        raise ContractError(f"horizons must keep the protocol order: {list(HORIZONS)}")
    if payload["fixed_frames"] != FIXED_FRAMES:
        raise ContractError(f"fixed_frames must be {FIXED_FRAMES}")
    if payload["repeats"] != REPEATS:
        raise ContractError(f"repeats must be exactly {REPEATS}")
    scope = payload["session_scope"]
    if scope == FORBIDDEN_SESSION_SCOPE:
        raise ContractError(
            "this harness starts adapter processes and cannot claim single_process; "
            f"use {SUPPORTED_SESSION_SCOPE!r}"
        )
    if scope != SUPPORTED_SESSION_SCOPE:
        raise ContractError(
            f"session_scope must be {SUPPORTED_SESSION_SCOPE!r}; got {scope!r}"
        )
    frozen = payload["frozen"]
    if not isinstance(frozen, dict) or set(frozen) != set(FROZEN_KEYS):
        raise ContractError(f"frozen must contain exactly {list(FROZEN_KEYS)}")
    if frozen["gpu_name"] != "NVIDIA A100-SXM4-80GB":
        raise ContractError("the formal manifest is fixed to NVIDIA A100-SXM4-80GB")
    methods = payload["methods"]
    if not isinstance(methods, list) or not methods:
        raise ContractError("manifest must configure at least one method")
    names = [item.get("name") if isinstance(item, dict) else None for item in methods]
    if any(name not in METHODS for name in names):
        raise ContractError(f"methods must be drawn from {list(METHODS)}")
    if tuple(names) != tuple(dict.fromkeys(names)):
        raise ContractError(f"methods must not repeat: {names}")
    if tuple(names) != tuple(m for m in METHODS if m in names):
        raise ContractError(f"methods must keep the protocol order: {list(METHODS)}")
    for item in methods:
        if set(item) != {"name", "bit_width", "command"}:
            raise ContractError(
                f"method {item.get('name')!r} must contain name, bit_width, command"
            )
        command = item["command"]
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command
        ):
            raise ContractError(f"method {item['name']} command must be a non-empty argv list")
        expected_bits = None if item["name"] == "BF16" else int(item["name"].rsplit("INT", 1)[1])
        if item["bit_width"] != expected_bits:
            raise ContractError(
                f"method {item['name']} bit_width must be {expected_bits!r}"
            )
    return payload


def _expand_command(command: Sequence[str], context: Mapping[str, Any]) -> list[str]:
    try:
        expanded = [part.format_map(context) for part in command]
    except KeyError as error:
        raise ContractError(f"unknown command placeholder: {error.args[0]}") from error
    unresolved = [part for part in expanded if "REPLACE_ME" in part]
    if unresolved:
        raise ContractError("replace every REPLACE_ME command token before a real run")
    return expanded


def _run_adapter(
    method: Mapping[str, Any],
    *,
    phase: str,
    frames: int,
    repeat: int,
    report_path: Path,
    output_dir: Path,
    session_id: str,
    session_scope: str,
    runtime_instance_id: str,
    frozen: Mapping[str, Any],
    dry_run: bool,
) -> list[str]:
    context = {
        "method": method["name"],
        "phase": phase,
        "frames": frames,
        "repeat": repeat,
        "seed": frozen["seed"],
        "report_path": str(report_path),
        "output_dir": str(output_dir),
        "session_id": session_id,
        "session_scope": session_scope,
        "runtime_instance_id": runtime_instance_id,
    }
    command = _expand_command(method["command"], context) if not dry_run else [
        part.format_map(context) for part in method["command"]
    ]
    if dry_run:
        return command
    environment = os.environ.copy()
    environment.update(
        {
            "PAPER_EXPERIMENT_METHOD": method["name"],
            "PAPER_EXPERIMENT_PHASE": phase,
            "PAPER_EXPERIMENT_FRAMES": str(frames),
            "PAPER_EXPERIMENT_REPEAT": str(repeat),
            "PAPER_EXPERIMENT_SEED": str(frozen["seed"]),
            "PAPER_EXPERIMENT_REPORT": str(report_path),
            "PAPER_EXPERIMENT_OUTPUT": str(output_dir),
            "PAPER_EXPERIMENT_SESSION_ID": session_id,
            "PAPER_EXPERIMENT_SESSION_SCOPE": session_scope,
            "PAPER_EXPERIMENT_RUNTIME_INSTANCE_ID": runtime_instance_id,
            "PAPER_EXPERIMENT_FROZEN_JSON": json.dumps(frozen, sort_keys=True),
        }
    )
    report_path.unlink(missing_ok=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"adapter failed ({method['name']} {phase}, frames={frames}, "
            f"repeat={repeat}): exit {completed.returncode}"
        )
    if not report_path.is_file():
        raise ContractError(f"adapter did not write {report_path}")
    return command


def _run_adapter_pair(
    method: Mapping[str, Any],
    *,
    frames: int,
    repeat: int,
    warmup_report_path: Path,
    measure_report_path: Path,
    output_dir: Path,
    session_id: str,
    session_scope: str,
    runtime_instance_id: str,
    frozen: Mapping[str, Any],
    dry_run: bool,
) -> list[str]:
    """Run warmup and measurement in one loaded runtime process."""
    context = {
        "method": method["name"],
        "phase": "pair",
        "frames": frames,
        "repeat": repeat,
        "seed": frozen["seed"],
        "report_path": str(measure_report_path),
        "warmup_report_path": str(warmup_report_path),
        "measure_report_path": str(measure_report_path),
        "output_dir": str(output_dir),
        "session_id": session_id,
        "session_scope": session_scope,
        "runtime_instance_id": runtime_instance_id,
    }
    command = _expand_command(method["command"], context) if not dry_run else [
        part.format_map(context) for part in method["command"]
    ]
    if dry_run:
        return command
    environment = os.environ.copy()
    environment.update(
        {
            "PAPER_EXPERIMENT_METHOD": method["name"],
            "PAPER_EXPERIMENT_PHASE": "pair",
            "PAPER_EXPERIMENT_FRAMES": str(frames),
            "PAPER_EXPERIMENT_REPEAT": str(repeat),
            "PAPER_EXPERIMENT_SEED": str(frozen["seed"]),
            # Keep REPORT as a compatibility alias for the measured report.
            "PAPER_EXPERIMENT_REPORT": str(measure_report_path),
            "PAPER_EXPERIMENT_WARMUP_REPORT": str(warmup_report_path),
            "PAPER_EXPERIMENT_MEASURE_REPORT": str(measure_report_path),
            "PAPER_EXPERIMENT_OUTPUT": str(output_dir),
            "PAPER_EXPERIMENT_SESSION_ID": session_id,
            "PAPER_EXPERIMENT_SESSION_SCOPE": session_scope,
            "PAPER_EXPERIMENT_RUNTIME_INSTANCE_ID": runtime_instance_id,
            "PAPER_EXPERIMENT_FROZEN_JSON": json.dumps(frozen, sort_keys=True),
        }
    )
    warmup_report_path.unlink(missing_ok=True)
    measure_report_path.unlink(missing_ok=True)
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, env=environment, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            f"adapter failed ({method['name']} pair, frames={frames}, "
            f"repeat={repeat}): exit {completed.returncode}"
        )
    missing = [
        str(path)
        for path in (warmup_report_path, measure_report_path)
        if not path.is_file()
    ]
    if missing:
        raise ContractError(
            "paired adapter did not write both reports: " + ", ".join(missing)
        )
    return command


def _synthetic_timings(method: str, frames: int, repeat: int) -> dict[str, float]:
    # Deliberately right-skewed repeat factors exercise the requested robust
    # summaries.  These are plumbing fixtures and are stamped synthetic.
    factors = (0.985, 0.997, 1.010, 1.045, 1.115)
    factor = factors[repeat]
    scale = frames / FIXED_FRAMES
    vae = 7.2 * scale
    base_denoise = 43.0 * scale
    method_factor = {
        "BF16": 1.00,
        "RTN_INT2": 0.92,
        "RTN_INT4": 0.95,
        "QVG_INT2": 0.94,
        "QVG_INT4": 0.97,
        "TempoKV_INT2": 0.90,
        "TempoKV_INT4": 0.93,
    }[method]
    encode_coeff = {
        "BF16": 0.0,
        "RTN_INT2": 0.55,
        "RTN_INT4": 0.48,
        "QVG_INT2": 0.80,
        "QVG_INT4": 0.72,
        "TempoKV_INT2": 1.35,
        "TempoKV_INT4": 1.10,
    }[method]
    decode_coeff = {
        "BF16": 0.0,
        "RTN_INT2": 0.42,
        "RTN_INT4": 0.35,
        "QVG_INT2": 1.25,
        "QVG_INT4": 1.00,
        "TempoKV_INT2": 1.85,
        "TempoKV_INT4": 1.50,
    }[method]
    encode = encode_coeff * scale
    decode = decode_coeff * scale**1.28
    denoising = base_denoise * method_factor
    values = {
        "cache_encode": encode * factor,
        "cache_decode": decode * factor,
        "vae_decode": vae * factor,
        "denoising_other": denoising * factor,
    }
    values["e2e"] = sum(values.values())
    return values


def _canonical_run(
    *,
    method: str,
    phase: str,
    repeat: int,
    frames: int,
    session_id: str,
    session_scope: str,
    runtime_instance_id: str,
    frozen: Mapping[str, Any],
    timings: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA,
        "method": method,
        "phase": phase,
        "repeat": repeat,
        "requested_frames": frames,
        "effective_frames": frames,
        "session_id": session_id,
        "session_scope": session_scope,
        "runtime_instance_id": runtime_instance_id,
        "frozen": dict(frozen),
        "timings_s": dict(timings),
        "synchronization": {
            "api": "torch.cuda.synchronize",
            "before_each_segment": True,
            "after_each_segment": True,
        },
        "exclusions": {"model_loading": True, "compilation": True},
    }


def execute(
    manifest: Mapping[str, Any],
    *,
    dry_run: bool,
    synthetic: bool,
    pilot: bool,
) -> Path:
    output_dir = Path(manifest["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = "synthetic-" + manifest["experiment_id"] if synthetic else str(uuid.uuid4())
    scope = manifest["session_scope"]
    frozen = manifest["frozen"]
    resolved = dict(manifest)
    resolved.update(
        {
            "resolved_output_dir": str(output_dir),
            "coordinator_session_id": session_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "synthetic": synthetic,
            "pilot": pilot,
        }
    )
    write_json(output_dir / "resolved_manifest.json", resolved)
    raw_dir = output_dir / "raw"
    plan: list[dict[str, Any]] = []
    compilation: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for method in manifest["methods"]:
        name = method["name"]
        report_path = raw_dir / name / "compilation.json"
        command = _run_adapter(
            method,
            phase="compilation",
            frames=FIXED_FRAMES,
            repeat=-1,
            report_path=report_path,
            output_dir=output_dir,
            session_id=session_id,
            session_scope=scope,
            runtime_instance_id=f"{session_id}:{name}:compilation",
            frozen=frozen,
            dry_run=dry_run or synthetic,
        )
        plan.append({"method": name, "phase": "compilation", "command": command})
        if synthetic:
            report = {
                "schema_version": COMPILATION_SCHEMA,
                "method": name,
                "one_time_compilation_s": 0.0 if name == "BF16" else 18.0 + METHODS.index(name),
                "session_id": session_id,
                "session_scope": scope,
                "frozen": dict(frozen),
            }
            write_json(report_path, report)
        if not dry_run:
            compilation.append(
                validate_compilation_report(
                    read_json(report_path),
                    expected_method=name,
                    expected_session_id=session_id,
                    expected_session_scope=scope,
                    expected_frozen=frozen,
                )
            )

    # Interleave all methods within a repeat and horizon.  BF16 therefore has
    # the same repeat identity as every paired comparison, without duplicating
    # baseline runs in the table.
    for frames in manifest["horizons"]:
        for repeat in range(REPEATS):
            for method in manifest["methods"]:
                name = method["name"]
                method_dir = raw_dir / name / f"frames-{frames}" / f"repeat-{repeat}"
                pair_runtime_instance = f"{session_id}:{name}:{frames}:{repeat}"
                warmup_path = method_dir / "warmup.json"
                measure_path = method_dir / "measure.json"
                command = _run_adapter_pair(
                    method,
                    frames=frames,
                    repeat=repeat,
                    warmup_report_path=warmup_path,
                    measure_report_path=measure_path,
                    output_dir=output_dir,
                    session_id=session_id,
                    session_scope=scope,
                    runtime_instance_id=pair_runtime_instance,
                    frozen=frozen,
                    dry_run=dry_run or synthetic,
                )
                plan.append(
                    {
                        "method": name,
                        "phase": "pair",
                        "frames": frames,
                        "repeat": repeat,
                        "warmup_report": str(warmup_path),
                        "measure_report": str(measure_path),
                        "command": command,
                    }
                )
                if synthetic:
                    for phase, path in (
                        ("warmup", warmup_path),
                        ("measure", measure_path),
                    ):
                        write_json(
                            path,
                            _canonical_run(
                                method=name,
                                phase=phase,
                                repeat=repeat,
                                frames=frames,
                                session_id=session_id,
                                session_scope=scope,
                                runtime_instance_id=pair_runtime_instance,
                                frozen=frozen,
                                timings=_synthetic_timings(name, frames, repeat),
                            ),
                        )
                if not dry_run:
                    warmup = validate_run_report(
                        read_json(warmup_path),
                        expected_method=name,
                        expected_phase="warmup",
                        expected_repeat=repeat,
                        expected_frames=frames,
                        expected_session_id=session_id,
                        expected_session_scope=scope,
                        expected_frozen=frozen,
                        expected_runtime_instance_id=pair_runtime_instance,
                        formal=not pilot,
                    )
                    measured = validate_run_report(
                        read_json(measure_path),
                        expected_method=name,
                        expected_phase="measure",
                        expected_repeat=repeat,
                        expected_frames=frames,
                        expected_session_id=session_id,
                        expected_session_scope=scope,
                        expected_frozen=frozen,
                        expected_runtime_instance_id=warmup["runtime_instance_id"],
                        formal=not pilot,
                    )
                    records.append(measured)

    write_json(output_dir / "execution_plan.json", plan)
    if dry_run:
        return output_dir
    write_json(output_dir / "compilation.json", compilation)
    write_json(output_dir / "measurements.json", records)
    aggregate(
        output_dir,
        compilation,
        records,
        methods=[item["name"] for item in manifest["methods"]],
        horizons=manifest["horizons"],
        synthetic=synthetic,
    )
    return output_dir


def _stats(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    return summary([float(record["timings_s"][key]) for record in records])


def _flatten(prefix: str, stats: Mapping[str, float], target: dict[str, Any]) -> None:
    for statistic in ("median", "p10", "p90"):
        target[f"{prefix}_{statistic}"] = stats[statistic]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ContractError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    output_dir: Path,
    compilation: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    horizons: Sequence[int],
    synthetic: bool,
) -> None:
    expected = len(methods) * len(horizons) * REPEATS
    if len(records) != expected:
        raise ContractError(f"expected {expected} measured reports, got {len(records)}")
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    by_pair: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for record in records:
        key = (str(record["method"]), int(record["effective_frames"]))
        grouped.setdefault(key, []).append(record)
        pair_key = (str(record["method"]), int(record["effective_frames"]), int(record["repeat"]))
        if pair_key in by_pair:
            raise ContractError(f"duplicate measurement: {pair_key}")
        by_pair[pair_key] = record

    rows: list[dict[str, Any]] = []
    for frames in horizons:
        baselines = grouped[("BF16", frames)]
        if sorted(int(item["repeat"]) for item in baselines) != list(range(REPEATS)):
            raise ContractError(f"BF16 frames={frames} has incomplete repeat identities")
        for method in methods:
            members = grouped.get((method, frames), [])
            if sorted(int(item["repeat"]) for item in members) != list(range(REPEATS)):
                raise ContractError(f"{method} frames={frames} has incomplete repeats")
            row: dict[str, Any] = {
                "method": method,
                "bit_width": "BF16" if method == "BF16" else method.rsplit("INT", 1)[1],
                "requested_frames": frames,
                "effective_frames": frames,
                "repeats": REPEATS,
                "synthetic": synthetic,
            }
            for key in TIMING_KEYS:
                _flatten(f"{key}_s", _stats(members, key), row)
            shares = [
                (item["timings_s"]["cache_encode"] + item["timings_s"]["cache_decode"])
                / item["timings_s"]["e2e"]
                * 100.0
                for item in members
            ]
            _flatten("cache_overhead_share_pct", summary(shares), row)
            paired_seconds = []
            paired_percent = []
            for repeat in range(REPEATS):
                current = by_pair[(method, frames, repeat)]["timings_s"]["e2e"]
                baseline = by_pair[("BF16", frames, repeat)]["timings_s"]["e2e"]
                paired_seconds.append(current - baseline)
                paired_percent.append((current - baseline) / baseline * 100.0)
            _flatten("paired_bf16_overhead_s", summary(paired_seconds), row)
            _flatten("paired_bf16_overhead_pct", summary(paired_percent), row)
            rows.append(row)

    horizon_path = output_dir / "horizon.csv"
    _write_csv(horizon_path, rows)
    fixed_rows = [row for row in rows if row["effective_frames"] == FIXED_FRAMES]
    _write_csv(output_dir / "fixed_717.csv", fixed_rows)
    compile_rows = [
        {
            "method": item["method"],
            "one_time_compilation_s": item["one_time_compilation_s"],
            "synthetic": synthetic,
        }
        for item in compilation
    ]
    _write_csv(output_dir / "compilation.csv", compile_rows)
    _svg_plot(
        output_dir / "horizon_absolute_log.svg",
        rows,
        value_key="e2e_s_median",
        title="Absolute end-to-end latency vs. generation horizon",
        y_label="E2E latency (s, log scale)",
        logarithmic=True,
    )
    _svg_plot(
        output_dir / "horizon_overhead_share.svg",
        rows,
        value_key="cache_overhead_share_pct_median",
        title="Cache encode + decode share vs. generation horizon",
        y_label="Cache overhead share of E2E (%)",
        logarithmic=False,
    )


def _svg_plot(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    title: str,
    y_label: str,
    logarithmic: bool,
) -> None:
    """Write a dependency-free SVG, keeping the two scientific questions separate."""
    width, height = 1000, 620
    left, right, top, bottom = 105, 220, 65, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = ("#222222", "#D55E00", "#E69F00", "#0072B2", "#56B4E9", "#009E73", "#CC79A7")
    frames_present = sorted({int(row["effective_frames"]) for row in rows})
    values = [float(row[value_key]) for row in rows]
    transformed = [math.log10(value) if logarithmic else value for value in values]
    low, high = min(transformed), max(transformed)
    padding = max((high - low) * 0.08, 0.03 if logarithmic else 0.5)
    low -= padding
    high += padding

    def x_pos(frame: int) -> float:
        return left + (frame - min(frames_present)) / (max(frames_present) - min(frames_present)) * plot_w

    def y_pos(value: float) -> float:
        adjusted = math.log10(value) if logarithmic else value
        return top + (high - adjusted) / (high - low) * plot_h

    esc = lambda value: str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1.4}.grid{stroke:#ddd;stroke-width:1}.series{fill:none;stroke-width:2.4}.point{stroke:white;stroke-width:1}</style>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-size="20" font-weight="bold">{esc(title)}</text>',
    ]
    for tick in range(6):
        y = top + tick / 5 * plot_h
        transformed_value = high - tick / 5 * (high - low)
        label = 10**transformed_value if logarithmic else transformed_value
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-size="13">{label:.2f}</text>')
    lines.extend(
        (
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
        )
    )
    for frame in frames_present:
        x = x_pos(frame)
        lines.append(f'<text x="{x:.2f}" y="{top+plot_h+26}" text-anchor="middle" font-size="13">{frame}</text>')
    lines.append(f'<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle" font-size="15">Generated video frames</text>')
    lines.append(f'<text transform="translate(25 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-size="15">{esc(y_label)}</text>')
    plotted = [m for m in METHODS if any(row["method"] == m for row in rows)]
    for index, method in enumerate(plotted):
        members = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["effective_frames"]),
        )
        points = " ".join(
            f"{x_pos(int(row['effective_frames'])):.2f},{y_pos(float(row[value_key])):.2f}"
            for row in members
        )
        lines.append(f'<polyline class="series" stroke="{colors[index]}" points="{points}"/>')
        for row in members:
            lines.append(
                f'<circle class="point" fill="{colors[index]}" cx="{x_pos(int(row["effective_frames"])):.2f}" '
                f'cy="{y_pos(float(row[value_key])):.2f}" r="4"/>'
            )
        legend_y = top + 22 + index * 32
        lines.append(f'<line x1="{width-right+30}" y1="{legend_y}" x2="{width-right+58}" y2="{legend_y}" stroke="{colors[index]}" stroke-width="3"/>')
        lines.append(f'<text x="{width-right+68}" y="{legend_y+5}" font-size="14">{esc(method.replace("_", " "))}</text>')
    if any(bool(row["synthetic"]) for row in rows):
        lines.append(f'<text x="{left+plot_w/2}" y="{top+plot_h/2}" text-anchor="middle" font-size="44" fill="#cc0000" opacity="0.16" transform="rotate(-24 {left+plot_w/2} {top+plot_h/2})">SYNTHETIC — NOT PAPER RESULTS</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="write the complete command plan without executing it")
    mode.add_argument("--synthetic", action="store_true", help="exercise aggregation and plotting with marked fake data")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="admit asynchronously profiled legacy reports; outputs are not protocol-valid",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = _manifest(args.manifest.expanduser().resolve())
        output = execute(
            manifest,
            dry_run=args.dry_run,
            synthetic=args.synthetic,
            pilot=args.pilot,
        )
    except (ContractError, OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"latency experiment rejected: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
