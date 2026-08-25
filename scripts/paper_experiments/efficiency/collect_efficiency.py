#!/usr/bin/env python3
"""Validate Self-Forcing efficiency records and render paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence

try:
    from efficiency_record import largest_grid_horizon
except ModuleNotFoundError:  # pragma: no cover - direct script/test import
    _record_dir = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "Self-Forcing"
        / "tempokv_adapter"
    )
    if str(_record_dir) not in sys.path:
        sys.path.insert(0, str(_record_dir))
    from efficiency_record import largest_grid_horizon


GIB = 1024 ** 3
FROZEN_FIELDS = (
    "model",
    "width",
    "height",
    "frames",
    "prompts_sha256",
    "prompt_indices",
    "seed",
    "num_gpus",
    "gpu_uuid",
)
METHOD_ORDER = {
    "BF16": (0, 0),
    "KIVI": (1, 0),
    "QuaRot-KV": (1, 1),
    "RTN": (1, 2),
    "QVG": (1, 3),
    "QVG-Pro": (1, 4),
    "TempoKV": (1, 5),
}
FOOTNOTE = (
    "BF16 was measured once, in the TempoKV repository. The vs-BF16 ratios for "
    "QVG and KVQuant4Video rows therefore use that baseline and include any "
    "difference between the three Self-Forcing copies, not method overhead alone."
)
LOCAL_NOTE = (
    "This is a standalone repository summary. Resident compression is computed "
    "from each row's BF16-equivalent bytes; cross-repository cache reduction is "
    "left blank until the TempoKV BF16 record is collected."
)


def load_records(output_root: Path) -> list[dict]:
    """Load every method-level ``efficiency.json`` below ``output_root``."""
    root = Path(output_root).expanduser()
    paths = sorted(root.glob("*/efficiency.json"))
    records: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"record is not an object: {path}")
        payload.setdefault("_record_path", str(path))
        records.append(payload)
    return records


def _record_name(record: dict) -> str:
    return str(record.get("method", record.get("repo", "<unknown>")))


def check_consistency(records: list[dict]) -> list[str]:
    """Check the frozen experiment identity across every run."""
    if not records:
        return ["no efficiency.json records were found"]
    reference = records[0]
    failures: list[str] = []
    reference_run = reference.get("run", {})
    for record in records[1:]:
        run = record.get("run", {})
        for field in FROZEN_FIELDS:
            expected = reference_run.get(field)
            observed = run.get(field)
            if expected != observed:
                failures.append(
                    f"{_record_name(record)} disagrees with "
                    f"{_record_name(reference)} on {field}: "
                    f"{observed!r} != {expected!r}"
                )
    return failures


def check_byte_accounting(records: list[dict]) -> list[str]:
    """Apply the two exact `resident_analytic.v1` cross-repository equalities."""
    failures: list[str] = []
    baseline = next(
        (record for record in records if _record_name(record) == "BF16"), None
    )
    if baseline is None:
        return ["no BF16 row: every reduction figure needs it as the denominator"]
    try:
        baseline_cache = baseline["kv_cache"]
        baseline_resident = int(baseline_cache["peak_resident_bytes"])
        baseline_equivalent = int(baseline_cache["peak_bf16_equivalent_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        return [f"BF16 record has invalid KV byte fields: {error}"]
    if baseline_resident != baseline_equivalent:
        failures.append(
            f"BF16 resident bytes {baseline_resident} != its own BF16 equivalent "
            f"{baseline_equivalent}"
        )
    for record in records:
        name = _record_name(record)
        try:
            cache = record["kv_cache"]
            definition = cache.get("definition")
            observed = int(cache["peak_bf16_equivalent_bytes"])
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{name}: invalid KV byte fields: {error}")
            continue
        if definition != "resident_analytic.v1":
            failures.append(
                f"{name}: unsupported KV definition {definition!r}; expected "
                "resident_analytic.v1"
            )
        if observed != baseline_resident:
            failures.append(
                f"{name} BF16-equivalent {observed} != BF16 resident "
                f"{baseline_resident} (differs by {abs(observed - baseline_resident)} bytes)"
            )
    return failures


def check_decomposition(record: dict) -> str | None:
    """Return why one row cannot appear in the latency table, if anything."""
    timing = record.get("timing", {})
    segments = timing.get("segments_ms", {})
    names = ("e2e", "cache_encode", "cache_decode", "vae_decode")
    missing = [name for name in names if segments.get(name) is None]
    if missing:
        return (
            f"{_record_name(record)}: unmeasured segment(s) {missing}; "
            "denoising_other is a residual and cannot absorb them"
        )
    if timing.get("profile_available") is False:
        return f"{_record_name(record)}: profile_available is false"
    try:
        e2e = float(segments["e2e"])
        codec = sum(float(segments[name]) for name in names[1:])
    except (KeyError, TypeError, ValueError) as error:
        return f"{_record_name(record)}: invalid latency segments: {error}"
    residual = e2e - codec
    if residual < 0:
        return (
            f"{_record_name(record)}: denoising_other is {residual:.1f} ms; "
            "segments overlap or e2e brackets the wrong interval"
        )
    return None


def _sort_key(row: dict) -> tuple:
    method = str(row.get("method", ""))
    group, order = METHOD_ORDER.get(method, (9, 9))
    bits = row.get("bits")
    return (group, 0 if bits is None else int(bits), order, method)


def _final_bf16_equivalent_bytes(record: dict) -> int | None:
    """BF16 bytes for the same full-sequence state as ``final_resident_bytes``.

    The two peak fields are independent maxima and must not be divided.  At the
    final sample, all target methods retain the complete sequence, so the
    equivalent size is reconstructed from the recorded cache shape and the
    run's latent-frame count.
    """
    try:
        run = record["run"]
        cache = record["kv_cache"]
        latent_frames = int(run["latent_frames"])
        num_layers = int(cache["num_layers"])
        heads = int(cache["num_heads"])
        head_dim = int(cache["head_dim"])
        frame_seq_length = int(cache["frame_seq_length"])
        batch_size = int(cache["batch_size"])
    except (KeyError, TypeError, ValueError):
        return None
    if min(
        latent_frames,
        num_layers,
        heads,
        head_dim,
        frame_seq_length,
        batch_size,
    ) <= 0:
        return None
    return (
        batch_size
        * latent_frames
        * frame_seq_length
        * heads
        * head_dim
        * 2
        * 2
        * num_layers
    )


def efficiency_rows(records: list[dict]) -> list[dict]:
    """Build one cache/peak/time row for each run."""
    baseline = next(
        (record for record in records if _record_name(record) == "BF16"), None
    )
    baseline_cache = (
        float(baseline["kv_cache"]["peak_resident_bytes"])
        if baseline is not None
        else None
    )
    baseline_seconds = (
        baseline["timing"].get("generation_seconds_median")
        if baseline is not None
        else None
    )
    baseline_seconds = (
        float(baseline_seconds) if baseline_seconds is not None else None
    )
    rows: list[dict] = []
    for record in records:
        cache_data = record["kv_cache"]
        cache = int(cache_data["peak_resident_bytes"])
        final_resident = int(
            cache_data.get("final_resident_bytes", cache)
        )
        final_equivalent = _final_bf16_equivalent_bytes(record)
        seconds_value = record["timing"].get("generation_seconds_median")
        seconds = float(seconds_value) if seconds_value is not None else None
        resident_compression = (
            final_equivalent / final_resident
            if final_equivalent is not None and final_resident
            else None
        )
        rows.append(
            {
                "method": _record_name(record),
                "bits": record.get("bits"),
                "kv_cache_gb": cache / GIB,
                # Cross-repository reduction uses the single TempoKV BF16
                # row. It is unavailable for a standalone QVG/KVQuant run.
                "cache_reduction": (
                    baseline_cache / cache
                    if baseline_cache is not None and cache
                    else None
                ),
                "final_bf16_equivalent_bytes": final_equivalent,
                "final_resident_bytes": final_resident,
                # This ratio uses one real final sample, so its numerator and
                # denominator refer to the same resident cache state.
                "resident_compression_ratio": resident_compression,
                # Keep a short alias for scripts that call this quantity
                # simply "compression_ratio".
                "compression_ratio": resident_compression,
                "peak_gpu_gb": int(record["gpu_memory"]["peak_allocated_bytes"]) / GIB,
                "generation_s": seconds,
                "vs_bf16": (
                    seconds / baseline_seconds
                    if seconds is not None and baseline_seconds
                    else None
                ),
                "predictor_parameter_bytes": int(
                    record.get("predictor_parameter_bytes", 0)
                ),
                "fake_quant": bool(record["kv_cache"].get("fake_quant", False)),
            }
        )
    return sorted(rows, key=_sort_key)


def latency_rows(records: list[dict]) -> list[dict]:
    """Build latency rows, skipping only rows whose decomposition is invalid."""
    rows: list[dict] = []
    for record in records:
        reason = check_decomposition(record)
        if reason is not None:
            print(f"skipping latency row: {reason}", file=sys.stderr)
            continue
        segments = record["timing"]["segments_ms"]
        residual = float(segments["e2e"]) - sum(
            float(segments[name])
            for name in ("cache_encode", "cache_decode", "vae_decode")
        )
        rows.append(
            {
                "method": _record_name(record),
                "bits": record.get("bits"),
                "e2e_ms": float(segments["e2e"]),
                "cache_encode_ms": float(segments["cache_encode"]),
                "cache_decode_ms": float(segments["cache_decode"]),
                "vae_decode_ms": float(segments["vae_decode"]),
                "denoising_other_ms": residual,
            }
        )
    return sorted(rows, key=_sort_key)


def parse_main_compression_ratios(path: Path) -> dict[str, float]:
    """Best-effort parser for method compression ratios in a TeX table."""
    text = Path(path).read_text(encoding="utf-8")
    ratios: dict[str, float] = {}
    aliases = {
        "KIVI": "KIVI",
        "QuaRot-KV": "QuaRot-KV",
        "QuaRot\\mbox{-}KV": "QuaRot-KV",
        "RTN": "RTN",
        "QVG": "QVG",
        "QVG-Pro": "QVG-Pro",
        "TempoKV": "TempoKV",
    }
    for line in text.splitlines():
        for alias, method in aliases.items():
            if alias not in line:
                continue
            matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:\\times|x)", line)
            if matches:
                ratios[method] = float(matches[-1])
    return ratios


def check_compression_scale(
    records: list[dict], main_ratios: dict[str, float] | None
) -> list[str]:
    """Warn when resident reduction is not lower than the main-table ratio."""
    if not main_ratios:
        return []
    baseline = next((r for r in records if _record_name(r) == "BF16"), None)
    if baseline is None:
        return []
    baseline_bytes = int(baseline["kv_cache"]["peak_resident_bytes"])
    warnings: list[str] = []
    for record in records:
        method = _record_name(record)
        expected = main_ratios.get(method)
        if expected is None:
            continue
        resident = int(record["kv_cache"]["peak_resident_bytes"])
        reduction = baseline_bytes / resident if resident else float("inf")
        if reduction >= expected:
            warnings.append(
                f"{method}: resident cache reduction {reduction:.2f}x is not "
                f"lower than the main-table compression ratio {expected:.2f}x; "
                "check that dense windows and metadata were counted"
            )
    return warnings


def _method_label(row: dict) -> str:
    bits = row.get("bits")
    return str(row["method"]) if bits is None else f"{row['method']} INT{bits}"


def render_efficiency_tex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & KV cache (GB) & Final compression & Cache reduction & Peak GPU (GB) & Generation (s) \\",
        r"\midrule",
    ]
    previous_group = None
    for row in sorted(rows, key=_sort_key):
        group = 0 if row.get("bits") is None else int(row["bits"])
        if previous_group is not None and group != previous_group:
            lines.append(r"\midrule")
        reduction = (
            "--"
            if row["cache_reduction"] is None
            else f"{row['cache_reduction']:.2f}$\\times$"
        )
        resident_compression = (
            "--"
            if row["resident_compression_ratio"] is None
            else f"{row['resident_compression_ratio']:.2f}$\\times$"
        )
        generation = (
            "--" if row["generation_s"] is None else f"{row['generation_s']:.2f}"
        )
        lines.append(
            f"{_method_label(row)} & {row['kv_cache_gb']:.2f} & "
            f"{resident_compression} & {reduction} & {row['peak_gpu_gb']:.2f} & "
            f"{generation} \\\\"
        )
        previous_group = group
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_latency_tex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & E2E (ms) & Cache encode (ms) & Cache decode (ms) & VAE (ms) & Other (ms) \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=_sort_key):
        lines.append(
            f"{_method_label(row)} & {row['e2e_ms']:.2f} & "
            f"{row['cache_encode_ms']:.2f} & {row['cache_decode_ms']:.2f} & "
            f"{row['vae_decode_ms']:.2f} & {row['denoising_other_ms']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def check_horizon_config(records: list[dict]) -> list[str]:
    """Check the global-window and exclusive-device horizon prerequisites."""
    failures: list[str] = []
    settings = {
        record.get("budget", {}).get("alloc_conf") for record in records
    }
    if len(settings) > 1:
        failures.append(f"horizon runs used different alloc_conf values: {settings}")
    for record in records:
        name = _record_name(record)
        budget = record.get("budget", {})
        if budget.get("exclusive_device") is not True:
            failures.append(f"{name}: card was not exclusive during the horizon probe")
        run = record.get("run", {})
        window = run.get("local_attn_size")
        latent = run.get("latent_frames")
        if window != latent:
            failures.append(
                f"{name}: local_attn_size {window} != requested latent frames "
                f"{latent}; a bounded window is not the requested experiment"
            )
    return failures


def predicted_horizon(efficiency_record: dict, device_total_bytes: int) -> int:
    """Predict a horizon from the 717-frame peak allocation and KV slope."""
    latent = int(efficiency_record["run"]["latent_frames"])
    kv_bytes = int(efficiency_record["kv_cache"]["peak_resident_bytes"])
    peak = int(efficiency_record["gpu_memory"]["peak_allocated_bytes"])
    slope = kv_bytes / latent if latent else 0.0
    base = peak - kv_bytes
    if slope <= 0:
        return 0
    predicted_latent = int((int(device_total_bytes) - base) // slope)
    return largest_grid_horizon(predicted_latent * 4 - 3)


def horizon_rows(
    horizon_records: list[dict], efficiency_records: list[dict]
) -> list[dict]:
    """Attach measured-vs-extrapolated horizon information to each method."""
    by_method = {_record_name(record): record for record in efficiency_records}
    baseline = next(
        (record for record in horizon_records if _record_name(record) == "BF16"),
        None,
    )
    rows: list[dict] = []
    for record in horizon_records:
        name = _record_name(record)
        measured = int(record["max_feasible_frames"])
        source = by_method.get(name)
        total = int(record.get("budget", {}).get("device_total_bytes", 0))
        predicted = predicted_horizon(source, total) if source and total else None
        warning = None
        if predicted:
            drift = abs(measured - predicted) / predicted
            if drift > 0.10:
                warning = (
                    f"{name}: measured {measured} frames but the 717-frame "
                    f"extrapolation predicts {predicted} ({drift:.0%} apart, "
                    "threshold 10%); non-KV memory or fragmentation may set the limit"
                )
        rows.append(
            {
                "method": name,
                "bits": record.get("bits"),
                "max_feasible_frames": measured,
                "is_lower_bound": not bool(record.get("oom_reached", False)),
                "probe_mode": record.get("probe_mode"),
                "vs_bf16": (
                    measured / int(baseline["max_feasible_frames"])
                    if baseline and int(baseline["max_feasible_frames"])
                    else None
                ),
                "predicted_frames": predicted,
                "extrapolation_warning": warning,
            }
        )
    return sorted(rows, key=_sort_key)


def render_horizon_tex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Max feasible frames & vs. BF16 & Probe \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=_sort_key):
        marker = r"$\dagger$" if row["is_lower_bound"] else ""
        ratio = "--" if row["vs_bf16"] is None else f"{row['vs_bf16']:.2f}$\\times$"
        lines.append(
            f"{_method_label(row)} & {row['max_feasible_frames']}{marker} & "
            f"{ratio} & {row['probe_mode']} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\smallskip",
        r"\noindent $\dagger$ Probe horizon reached without an OOM; value is a lower bound.",
    ])
    return "\n".join(lines) + "\n"


def headline_sentence(rows: list[dict], efficiency_rows_data: list[dict]) -> str:
    """Render the design-spec headline using 717-frame timing only."""
    by_method = {row["method"]: row for row in rows}
    by_cost = {row["method"]: row for row in efficiency_rows_data}
    baseline = by_method.get("BF16")
    if baseline is None:
        return ""
    others = ", ".join(
        f"{row['method']} {row['max_feasible_frames']}"
        for row in rows
        if row["method"] not in {"BF16", "TempoKV"}
    )
    tempokv = by_method.get("TempoKV")
    time_ratio = by_cost.get("TempoKV", {}).get("vs_bf16")
    if tempokv is None:
        return f"BF16 reaches {baseline['max_feasible_frames']} frames."
    tail = f"; {others} respectively" if others else ""
    ratio = "unknown" if tempokv["vs_bf16"] is None else f"{tempokv['vs_bf16']:.2f}x"
    time_text = "unknown" if time_ratio is None else f"{time_ratio:.0%}"
    return (
        f"On one A100 80GB under a global attention window, BF16 generates at most "
        f"{baseline['max_feasible_frames']} frames; TempoKV reaches "
        f"{tempokv['max_feasible_frames']} ({ratio}){tail}. At the shared 717-frame "
        f"horizon TempoKV's per-frame generation time is {time_text} of BF16's."
    )


def latency_sentence(rows: list[dict]) -> str:
    """Summarize the measured latency decomposition without inventing gaps."""
    if not rows:
        return "Latency decomposition is unavailable because no row passed the profile gate."
    labels = ", ".join(
        f"{row['method']}: encode {row['cache_encode_ms']:.1f} ms, "
        f"decode {row['cache_decode_ms']:.1f} ms, VAE {row['vae_decode_ms']:.1f} ms, "
        f"other {row['denoising_other_ms']:.1f} ms"
        for row in rows
    )
    return (
        "At the shared 717-frame horizon, the async-event latency decomposition "
        f"is {labels}."
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--main-table", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help=(
            "Collect one repository's rows without requiring the TempoKV BF16 "
            "cross-repository baseline. Resident compression remains available."
        ),
    )
    args = parser.parse_args(argv)

    output_root = args.output_root.expanduser()
    output_dir = (args.output_dir or output_root).expanduser()
    try:
        records = load_records(output_root)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"could not load efficiency records: {error}", file=sys.stderr)
        return 2
    consistency_failures = check_consistency(records)
    byte_failures = [] if args.local_only else check_byte_accounting(records)
    if consistency_failures or byte_failures:
        for message in (*consistency_failures, *byte_failures):
            print(f"collector gate failed: {message}", file=sys.stderr)
        return 2

    main_ratios: dict[str, float] = {}
    if args.main_table:
        try:
            main_ratios = parse_main_compression_ratios(args.main_table)
        except OSError as error:
            print(f"warning: cannot read main table: {error}", file=sys.stderr)
    for warning in check_compression_scale(records, main_ratios):
        print(f"WARNING: {warning}", file=sys.stderr)
    if not main_ratios:
        print(
            "WARNING: main-table compression ratios were not supplied; "
            "the magnitude gate was not evaluated",
            file=sys.stderr,
        )

    efficiency_data = efficiency_rows(records)
    latency_data = latency_rows(records)
    _write_csv(output_dir / "efficiency.csv", efficiency_data)
    (output_dir / "efficiency.tex").write_text(
        render_efficiency_tex(efficiency_data), encoding="utf-8"
    )
    _write_csv(output_dir / "latency.csv", latency_data)
    (output_dir / "latency.tex").write_text(
        render_latency_tex(latency_data), encoding="utf-8"
    )
    note = LOCAL_NOTE if args.local_only else FOOTNOTE
    (output_dir / "efficiency_plan.md").write_text(
        "# Self-Forcing efficiency outputs\n\n"
        + note
        + "\n\n"
        + latency_sentence(latency_data)
        + "\n",
        encoding="utf-8",
    )
    print(note)

    horizon_paths = sorted(output_root.glob("*/horizon.json"))
    if horizon_paths:
        horizon_records = [
            json.loads(path.read_text(encoding="utf-8")) for path in horizon_paths
        ]
        horizon_failures = check_horizon_config(horizon_records)
        if horizon_failures:
            for message in horizon_failures:
                print(f"horizon gate failed: {message}", file=sys.stderr)
            return 2
        horizon_data = horizon_rows(horizon_records, records)
        _write_csv(output_dir / "horizon.csv", horizon_data)
        (output_dir / "horizon.tex").write_text(
            render_horizon_tex(horizon_data), encoding="utf-8"
        )
        sentence = headline_sentence(horizon_data, efficiency_data)
        with (output_dir / "efficiency_plan.md").open("a", encoding="utf-8") as handle:
            handle.write("\n" + sentence + "\n")
        print(sentence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
