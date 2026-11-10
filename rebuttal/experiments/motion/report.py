"""BF16 motion groups joined to paired quality and identity-matched cache data."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rebuttal.common.statistics import correlation, finite, grouped_summary
from rebuttal.common.video_metrics import IDENTITY_FIELDS, VideoReader, quality_contract, read_csv, resolve_pairs, write_csv, write_json
from rebuttal.experiments import temporal_window
from .camera import camera_proxy, forward_backward_proxy
from .cases import export_paired_clip, select_candidates
from .flow import make_flow_backend
from .groups import assign_groups


def _identity(row: dict[str, Any]) -> tuple:
    return tuple(str(row.get(k, 0 if k in {"seed", "sample_id"} else "")) for k in ("model", "prompt_id", "seed", "sample_id"))


def analyze_reference(pair: dict[str, Any], config: dict[str, Any], backend: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    reader = VideoReader(Path(pair["reference"]), config.get("video_decoder", "pyav"))
    previous = None
    rows = []
    cuts = config.get("cuts", {})
    threshold = float(cuts.get("mean_absolute_rgb_threshold", 60.0))
    if threshold <= 0:
        raise ValueError("cut threshold must be positive")
    try:
        for frame in reader:
            if previous is None:
                previous = frame
                continue
            if frame.image.shape != previous.image.shape:
                raise ValueError("BF16 reference changes frame shape")
            if frame.timestamp <= previous.timestamp:
                raise ValueError("BF16 reference has nonmonotonic PTS")
            score = float(np.abs(frame.image.astype(np.float32) - previous.image.astype(np.float32)).mean())
            cut = score >= threshold
            magnitude = None
            if not cut:
                flow = backend(previous.image, frame.image)
                if flow.shape != (*previous.image.shape[:2], 2) or not np.isfinite(flow).all():
                    raise ValueError("flow backend returned invalid original-resolution vectors")
                magnitude = float(np.linalg.norm(flow, axis=-1).mean())
            row = {k: pair.get(k) for k in ("model", "prompt_id", "seed", "sample_id")}
            row.update(frame_index=previous.index, next_frame_index=frame.index, timestamp=previous.timestamp,
                       end_timestamp=frame.timestamp, motion_magnitude=magnitude,
                       motion_diagonal_per_second=magnitude * reader.fps / math.hypot(reader.width, reader.height) if magnitude is not None else None,
                       is_cut=cut, cut_score=score, flow_valid=not cut, global_motion=None, residual_motion=None,
                       camera_valid=False, camera_failure="scene_cut" if cut else None,
                       fb_inconsistent_fraction=None, fb_valid_fraction=None)
            if not cut:
                row.update(camera_proxy(flow, config.get("camera", {})))
                if config.get("flow", {}).get("forward_backward", True):
                    reverse = backend(frame.image, previous.image)
                    row.update(forward_backward_proxy(flow, reverse))
            rows.append(row)
            previous = frame
    finally:
        reader.close()
    if not rows:
        raise ValueError("motion analysis needs at least two decoded frames")
    total_frames = len(rows) + 1
    expected = pair.get("expected_frames", config.get("expected_frames"))
    if expected is not None and total_frames != int(expected):
        raise ValueError(f"motion reference expected {expected} frames, decoded {total_frames}")
    continuous = [r for r in rows if not r["is_cut"]]
    camera_valid = [r for r in continuous if r["camera_valid"]]
    def average(members, name):
        values = [r[name] for r in members if finite(r.get(name))]
        return sum(values) / len(values) if values else None
    summary = {k: pair.get(k) for k in ("model", "prompt_id", "seed", "sample_id")}
    summary.update(reference=pair["reference"], frames=total_frames, flow_pairs=len(rows), fps=reader.fps,
                   duration_seconds=previous.timestamp + 1 / reader.fps, width=reader.width, height=reader.height,
                   motion_magnitude=average(continuous, "motion_magnitude"), motion_diagonal_per_second=average(continuous, "motion_diagonal_per_second"),
                   motion_basis="non_cut_adjacent_frames", motion_missing_reason=None if continuous else "no_non_cut_frame_pairs",
                   motion_continuous=average(continuous, "motion_magnitude"), cut_pairs=len(rows)-len(continuous),
                   cut_fraction=(len(rows)-len(continuous))/len(rows), global_motion=average(camera_valid, "global_motion"),
                   residual_motion=average(camera_valid, "residual_motion"), camera_valid_pairs=len(camera_valid),
                   camera_failed_pairs=len(continuous)-len(camera_valid), fb_inconsistent_fraction=average(continuous, "fb_inconsistent_fraction"),
                   timestamp_policy=reader.timestamp_policy)
    return rows, summary


def _load_events(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("events", payload.get("cache_events", [payload]))


def _interval(event: dict[str, Any]) -> tuple[int, int] | None:
    start = event.get("pixel_start_frame", event.get("pixel_start"))
    end = event.get("pixel_end_frame", event.get("pixel_end"))
    if start is None or end is None or not float(start).is_integer() or not float(end).is_integer() or int(end) <= int(start):
        return None
    return int(start), int(end)


def _cache_records(pair: dict[str, Any], events: list[dict[str, Any]], scope: str) -> tuple[list[dict[str, Any]], str | None]:
    """Resolve experiment identity as well as video identity before summing."""
    candidates = [r for r in events if _identity(r) == _identity(pair) and r.get("metric_scope") == scope]
    if not candidates:
        return [], "no_matching_identity_and_scope"
    selectors = {"method": pair.get("cache_method", pair.get("method")),
                 "variant": pair.get("cache_variant", pair.get("variant", pair.get("method"))),
                 "comparison_id": pair.get("cache_comparison_id", pair.get("comparison_id", "default"))}
    for field, expected in selectors.items():
        present = [r.get(field) is not None for r in candidates]
        # Legacy observer files have no experiment fields: their explicitly
        # declared per-pair file binds the configuration. Do not combine such
        # unlabelled rows with labelled configurations in one event stream.
        if any(present) and not all(present):
            return [], f"incomplete_cache_{field}_identity"
        if any(present):
            candidates = [r for r in candidates if str(r[field]) == str(expected)]
            if not candidates:
                return [], f"no_matching_cache_{field}"
    return candidates, None


def join_cache(pair: dict[str, Any], start_frame: int, end_frame: int, scope: str = "bf16_trace_replay", *, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """No guessed latent→pixel mapping; no mean-of-NMSE reconstruction."""
    result = {"k_reconstruction_nmse": None, "v_reconstruction_nmse": None,
              "k_prediction_nmse": None, "v_prediction_nmse": None, "cache_metric_scope": scope,
              "cache_missing_reason": "no_cache_events"}
    filename = pair.get("cache_events") or pair.get("cache_metrics")
    if not filename:
        return result
    rows = _load_events(Path(filename)) if events is None else events
    matched, reason = _cache_records(pair, rows, scope)
    if reason:
        result["cache_missing_reason"] = reason
        return result
    result["cache_identity_validation"] = "event_fields_and_declared_file" if any(any(r.get(k) is not None for k in ("method", "variant", "comparison_id")) for r in matched) else "declared_single_configuration_file"
    if any(_interval(r) is None for r in matched):
        result["cache_missing_reason"] = "pixel_time_mapping_missing"
        return result
    for role in ("k", "v"):
        members = [r for r in matched if str(r.get("role", "")).lower() == role and _interval(r)[0] >= start_frame and _interval(r)[1] <= end_frame]
        coverage = set(i for r in members for i in range(*_interval(r)))
        if coverage != set(range(start_frame, end_frame)):
            result["cache_missing_reason"] = "cache_and_quality_coverage_differ"
            continue
        fields = ("error_sse", "signal_sse", "prediction_sse")
        if any(not finite(r.get("error_sse")) or not finite(r.get("signal_sse")) for r in members):
            result["cache_missing_reason"] = "cache_energy_sums_missing"
            continue
        denominator = sum(float(r["signal_sse"]) for r in members)
        if denominator <= 0:
            result["cache_missing_reason"] = "zero_signal_energy"
            continue
        result[f"{role}_reconstruction_nmse"] = sum(float(r["error_sse"]) for r in members) / denominator
        predicted = [r for r in members if not r.get("is_anchor", False)]
        pred_denominator = sum(float(r["signal_sse"]) for r in predicted)
        if pred_denominator > 0 and all(finite(r.get("prediction_sse")) for r in predicted):
            result[f"{role}_prediction_nmse"] = sum(float(r["prediction_sse"]) for r in predicted) / pred_denominator
    if result["k_reconstruction_nmse"] is not None and result["v_reconstruction_nmse"] is not None:
        result["cache_missing_reason"] = None
    return result


def cache_motion_intervals(pair: dict[str, Any], flow_rows: list[dict[str, Any]], scope: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate flow into the explicit cache pixel intervals, never vice versa."""
    relevant, _ = _cache_records(pair, events, scope)
    by_interval = {}
    unmapped = []
    for event in relevant:
        interval = _interval(event)
        if interval is not None:
            by_interval.setdefault(interval, []).append(event)
        else:
            unmapped.append(event)
    by_frame = {}
    identity = _identity(pair)
    for position, row in enumerate(flow_rows):
        if _identity(row) == identity:
            by_frame.setdefault(row['frame_index'], []).append((position, row))
    result = []
    for start, end in sorted(by_interval):
        selected = [item for frame in range(start, end) for item in by_frame.get(frame, [])
                    if item[1]['next_frame_index'] < end]
        members = [row for _, row in sorted(selected, key=lambda item: item[0])]
        if not members or len(members) != end - start - 1:
            continue
        valid_camera = all(r.get("camera_valid") and not r.get("is_cut") for r in members)
        row = {k: pair.get(k) for k in (*IDENTITY_FIELDS,)}
        row.update(pixel_start_frame=start, pixel_end_frame=end, timestamp=members[0]["timestamp"], end_timestamp=members[-1]["end_timestamp"],
                   is_cut=any(r["is_cut"] for r in members), camera_valid=valid_camera,
                   motion_magnitude=sum(r["motion_magnitude"] for r in members) / len(members) if not any(r["is_cut"] for r in members) else None,
                   global_motion=sum(r["global_motion"] for r in members) / len(members) if valid_camera else None,
                   residual_motion=sum(r["residual_motion"] for r in members) / len(members) if valid_camera else None)
        # Include every contained event, preserving join_cache's interval
        # semantics even when the input contains nested interval lengths.
        contained = [event for (a,b), group in by_interval.items()
                     if a >= start and b <= end for event in group]
        contained.extend(unmapped)
        row.update(join_cache(pair, start, end, scope, events=contained))
        result.append(row)
    return result


def validate_quality_reuse(directory: Path, pairs: list[dict[str, Any]], config: dict[str, Any]) -> None:
    completion_path = directory / temporal_window.QUALITY_COMPLETION_FILE
    if not completion_path.is_file():
        raise ValueError("quality reuse requires a successful completion record; recompute quality")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (completion.get("schema_version") != temporal_window.QUALITY_COMPLETION_SCHEMA or
            completion.get("status") != "complete"):
        raise ValueError("quality reuse completion record is invalid; recompute quality")
    try:
        actual_outputs = temporal_window.quality_output_fingerprints(directory)
    except FileNotFoundError as exc:
        raise ValueError("quality reuse completed output file is missing; recompute quality") from exc
    if completion.get("outputs") != actual_outputs:
        raise ValueError("quality reuse completed output fingerprint differs; recompute quality")
    path = directory / "video_pairs.json"
    if not path.is_file():
        raise ValueError("quality reuse requires video_pairs.json with captured input fingerprints; recompute quality")
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected_contract = quality_contract(config)
    for pair in pairs:
        old = [r for r in saved.get("pairs", []) if all(str(r.get(k)) == str(pair.get(k)) for k in IDENTITY_FIELDS)]
        if len(old) != 1:
            raise ValueError("quality reuse requires exactly one saved pair for each full identity")
        if old[0].get("quality_contract") != expected_contract:
            raise ValueError("quality reuse metric/decoder contract differs or is missing; recompute quality")
        for role in ("reference", "candidate"):
            actual = pair.get("video_fingerprints", {}).get(role)
            previous = old[0].get("video_fingerprints", {}).get(role)
            if actual is None or previous is None or actual != previous:
                raise ValueError(f"quality reuse {role} input fingerprint differs or is missing; recompute quality")


def case_frames(pair: dict[str, Any], case: dict[str, Any], frame_rows: list[dict[str, Any]],
                events: list[dict[str, Any]], scope: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use one actual RGB interval for both cache sums and video fidelity."""
    ordered = sorted(frame_rows, key=lambda r: int(r["frame_index"]))
    selected = [r for r in ordered if case["start_seconds"] <= float(r["timestamp"]) < case["end_seconds"]]
    info = {"requested_start_seconds": case["start_seconds"], "requested_end_seconds": case["end_seconds"],
            "cache_interval_alignment": "no_complete_cache_interval"}
    if not selected:
        return [], info
    start, end = int(selected[0]["frame_index"]), int(selected[-1]["frame_index"]) + 1
    relevant, reason = _cache_records(pair, events, scope)
    if reason or any(_interval(r) is None for r in relevant):
        return selected, info
    boundaries = []
    for role in ("k", "v"):
        intervals = {_interval(r) for r in relevant if str(r.get("role", "")).lower() == role}
        boundaries.append({point for interval in intervals for point in interval})
    shared = boundaries[0] & boundaries[1]
    lower = [point for point in shared if int(ordered[0]["frame_index"]) <= point <= start]
    upper = [point for point in shared if end <= point <= int(ordered[-1]["frame_index"]) + 1]
    if not lower or not upper:
        return selected, info
    aligned_start, aligned_end = max(lower), min(upper)
    aligned = [r for r in ordered if aligned_start <= int(r["frame_index"]) < aligned_end]
    if [int(r["frame_index"]) for r in aligned] != list(range(aligned_start, aligned_end)):
        return selected, info
    metrics = join_cache(pair, aligned_start, aligned_end, scope, events=events)
    if metrics["cache_missing_reason"] is not None:
        return selected, info
    info["cache_interval_alignment"] = "expanded_to_shared_explicit_cache_boundaries"
    return aligned, info


def summarize(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    observations = read_csv(output_dir / "motion_videos.csv")
    samples, seed = int(config.get("bootstrap_samples", 1000)), int(config.get("bootstrap_seed", 0))
    metric_names = ["motion_magnitude", "motion_continuous", "global_motion", "residual_motion", "psnr_db", "ssim", "lpips",
                    "k_reconstruction_nmse", "v_reconstruction_nmse", "k_prediction_nmse", "v_prediction_nmse"]
    groups = grouped_summary(observations, ["model", "method", "comparison_id", "motion_group"], metric_names, samples=samples, seed=seed)
    for entry in groups:
        members = [r for r in observations if all(r.get(k) == entry[k] for k in ("model", "method", "comparison_id", "motion_group"))]
        entry["n_cache"] = len({(r["prompt_id"], r.get("seed", 0), r.get("sample_id", 0)) for r in members if r.get("cache_missing_reason") is None})
    cohorts = {}
    for row in observations:
        cohorts.setdefault(tuple(row.get(k) for k in ("model", "method", "comparison_id")), []).append(row)
    correlations = []
    for cohort, members in cohorts.items():
        for x in ("motion_magnitude", "global_motion", "residual_motion"):
            for y in ("lpips", "k_reconstruction_nmse", "v_reconstruction_nmse"):
                entry = correlation(members, x, y, samples=samples, seed=seed)
                entry.update(zip(("model", "method", "comparison_id"), cohort, strict=True))
                # Camera proxies exclude invalid/cut pairs; their cache joins
                # must cover the exact same intervals, otherwise no correlation.
                if x in {"global_motion", "residual_motion"} and y.endswith("reconstruction_nmse"):
                    entry["rho"] = None
                    entry["ci_low"] = entry["ci_high"] = None
                    entry["missing_reason"] = "camera_proxy_cache_correlation_requires_pair_aligned_intervals_use_motion_cache_pairs_csv"
                correlations.append(entry)
    pair_cache_path = output_dir / "motion_cache_pairs.csv"
    if pair_cache_path.exists():
        paired = read_csv(pair_cache_path)
        for cohort in cohorts:
            members = [r for r in paired if tuple(r.get(k) for k in ("model", "method", "comparison_id")) == cohort and not r.get("is_cut") and r.get("camera_valid")]
            for x in ("global_motion", "residual_motion"):
                for y in ("k_reconstruction_nmse", "v_reconstruction_nmse"):
                    entry = correlation(members, x, y, samples=samples, seed=seed)
                    entry.update(zip(("model", "method", "comparison_id"), cohort, strict=True))
                    entry["unit"] = "explicit_cache_pixel_interval_clustered_by_prompt"
                    correlations.append(entry)
    summary = {"schema_version": "tempokv.rebuttal.motion.v1", "status": "complete", "n_video_pairs": len(observations),
               "groups": groups, "correlations": correlations,
               "flow_unit": "original_resolution_pixels_per_frame", "object_motion_interpretation": "residual_affine_flow_proxy_not_ground_truth",
               "cut_interpretation": "BF16_mean_absolute_difference_candidate_not_scene_change_ground_truth",
               "cache_missing": [{k: r.get(k) for k in ("model", "prompt_id", "seed", "method", "cache_missing_reason")} for r in observations if r.get("cache_missing_reason")]}
    write_json(output_dir / "summary.json", summary)
    flattened = []
    for row in groups:
        for metric, values in row["metrics"].items():
            flattened.append({**{k: v for k, v in row.items() if k != "metrics"}, "metric": metric, **values})
    write_csv(output_dir / "summary.csv", flattened)
    write_csv(output_dir / "correlations.csv", correlations)
    if config.get("plots", True):
        _plots(observations, output_dir)
    return summary


def _plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("motion scatter plots require matplotlib; select plots=false to explicitly omit") from exc
    cohorts = {}
    for row in rows:
        cohorts.setdefault(tuple(row.get(k) for k in ("model", "method", "comparison_id")), []).append(row)
    for cohort, members in cohorts.items():
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
        for axis, metric in zip(axes, ("lpips", "k_reconstruction_nmse", "v_reconstruction_nmse"), strict=True):
            valid = [r for r in members if finite(r.get(metric)) and finite(r.get("motion_magnitude"))]
            for group, color in (("low", "#4378bf"), ("medium", "#55a583"), ("high", "#d97b54")):
                current = [r for r in valid if r["motion_group"] == group]
                axis.scatter([r["motion_magnitude"] for r in current], [r[metric] for r in current], label=f"{group} (n={len(current)})", color=color, s=22, alpha=0.8)
            if not valid:
                axis.text(0.5, 0.5, "No aligned observations", transform=axis.transAxes, ha="center")
            axis.set(xlabel="BF16 motion (px/frame)", ylabel=metric)
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize=7)
        fig.suptitle(" / ".join(map(str, cohort)), fontsize=9)
        fig.tight_layout()
        name = "_".join(str(x).replace("/", "_") for x in cohort)
        fig.savefig(output_dir / f"motion_scatter_{name}.png", dpi=180)
        plt.close(fig)
    aligned_path = output_dir / "motion_cache_pairs.csv"
    if not aligned_path.exists():
        return
    intervals = read_csv(aligned_path)
    for cohort in cohorts:
        members = [r for r in intervals if tuple(r.get(k) for k in ("model", "method", "comparison_id")) == cohort and r.get("camera_valid") and not r.get("is_cut")]
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.3))
        for column, x in enumerate(("global_motion", "residual_motion")):
            for line, y in enumerate(("k_reconstruction_nmse", "v_reconstruction_nmse")):
                axis = axes[line, column]
                valid = [r for r in members if finite(r.get(x)) and finite(r.get(y))]
                axis.scatter([r[x] for r in valid], [r[y] for r in valid], s=14, alpha=0.65)
                if not valid:
                    axis.text(0.5, 0.5, "No aligned observations", transform=axis.transAxes, ha="center")
                axis.set(xlabel=f"{x} (px/frame)", ylabel=y, title=f"Aligned intervals: {len(valid)}")
                axis.grid(alpha=0.2)
        fig.suptitle(" / ".join(map(str, cohort)) + "; prompt-clustered intervals", fontsize=9)
        fig.tight_layout()
        name = "_".join(str(x).replace("/", "_") for x in cohort)
        fig.savefig(output_dir / f"motion_camera_scatter_{name}.png", dpi=180)
        plt.close(fig)


def run(config: dict[str, Any], stage: str, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage == "plan":
        try:
            pairs, reason = resolve_pairs(config), None
        except (ValueError, FileNotFoundError) as exc:
            pairs, reason = [], str(exc)
        result = {"experiment": "motion", "status": "planned", "pairs": pairs, "missing_input": reason,
                  "flow": config.get("flow", {"backend": "raft"}), "supported_stages": ["plan", "evaluate", "summarize"]}
        write_json(output_dir / "plan.json", result)
        return result
    if stage == "summarize":
        return summarize(output_dir, config)
    if stage != "evaluate":
        raise ValueError("motion supports plan/evaluate/summarize; generation uses the model adapter")
    pairs = resolve_pairs(config)
    # Reuse frame scores when supplied, otherwise evaluate with the same backend.
    quality_dir = output_dir / "quality"
    if config.get("quality_output_dir"):
        quality_dir = Path(config["quality_output_dir"])
        validate_quality_reuse(quality_dir, pairs, config)
    else:
        temporal_window.run(config, "evaluate", quality_dir)
    quality = read_csv(quality_dir / "videos.csv")
    frame_quality = read_csv(quality_dir / "frames.csv")
    backend = make_flow_backend(config.get("flow", {}))
    write_json(output_dir / "flow_backend.json", backend.metadata)
    unique, all_flow_rows, all_motion = {}, [], {}
    for pair in pairs:
        identity = _identity(pair)
        if identity in unique and unique[identity] != pair["reference"]:
            raise ValueError("same BF16 video identity points to different reference files")
        if identity not in unique:
            flow_rows, motion = analyze_reference(pair, config, backend)
            unique[identity] = pair["reference"]
            all_motion[identity] = motion
            all_flow_rows.extend(flow_rows)
            write_csv(output_dir / "motion_pairs.csv", all_flow_rows)
    joined, pair_cache, loaded_cache, event_files = [], [], {}, {}
    for pair in pairs:
        motion = all_motion[_identity(pair)]
        matched = [r for r in quality if all(str(r.get(k)) == str(pair.get(k)) for k in IDENTITY_FIELDS)]
        if len(matched) != 1:
            raise ValueError("quality records must match exactly one declared video pair identity")
        if int(matched[0]["frames"]) != motion["frames"]:
            raise ValueError("flow and quality frame coverage differ")
        cache_filename = pair.get("cache_events") or pair.get("cache_metrics")
        events = []
        if cache_filename:
            path = Path(cache_filename).resolve()
            if path not in event_files:
                indexed = {}
                for event in _load_events(path):
                    indexed.setdefault(_identity(event), []).append(event)
                event_files[path] = indexed
            events = event_files[path].get(_identity(pair), [])
        loaded_cache[tuple(str(pair.get(k)) for k in IDENTITY_FIELDS)] = events
        scope = config.get("cache_metric_scope", "bf16_trace_replay")
        row = {**motion, **matched[0], **join_cache(pair, 0, motion["frames"], scope, events=events)}
        joined.append(row)
        pair_cache.extend(cache_motion_intervals(pair, all_flow_rows, scope, events))
    frozen = json.loads(Path(config["motion_groups"]).read_text()) if config.get("motion_groups") else None
    grouping = assign_groups(joined, frozen)
    write_json(output_dir / "motion_groups.json", grouping)
    write_csv(output_dir / "motion_videos.csv", joined)
    write_csv(output_dir / "motion_cache_pairs.csv", pair_cache)
    candidates = select_candidates(all_flow_rows, pairs, config, output_dir)
    case_rows = []
    for case in candidates:
        pair = next(p for p in pairs if _identity(p) == _identity(case) and p["candidate"] == case["candidate"])
        events = loaded_cache[tuple(str(pair.get(k)) for k in IDENTITY_FIELDS)]
        all_frames = [r for r in frame_quality if all(str(r.get(k)) == str(pair.get(k)) for k in IDENTITY_FIELDS)]
        frames, alignment = case_frames(pair, case, all_frames, events, config.get("cache_metric_scope", "bf16_trace_replay"))
        if not frames:
            continue
        start, end = int(frames[0]["frame_index"]), int(frames[-1]["frame_index"]) + 1
        cache = join_cache(pair, start, end, config.get("cache_metric_scope", "bf16_trace_replay"), events=events)
        result = {**case, **alignment, **aggregate_case_quality(frames), **cache,
                  "start_seconds": frames[0]["timestamp"], "end_seconds": float(frames[-1]["timestamp"]) + float(frames[-1]["frame_duration"]),
                  "actual_start_seconds": frames[0]["timestamp"], "actual_end_seconds": float(frames[-1]["timestamp"]) + float(frames[-1]["frame_duration"])}
        if config.get("difficult_cases", {}).get("export_clips", True):
            result["clip"] = export_paired_clip(result, output_dir / "cases" / f"{case['case_id']}.mp4", decoder=config.get("video_decoder", "pyav"))
        case_rows.append(result)
    write_json(output_dir / "difficult_cases.json", {"cases": case_rows, "verified_case_count": sum(bool(r["manual_verified"]) for r in case_rows),
                                                     "candidate_only_case_count": sum(not r["manual_verified"] for r in case_rows)})
    return summarize(output_dir, config)


def aggregate_case_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from rebuttal.common.video_metrics import aggregate_frames
    aggregated = aggregate_frames(rows, group_by_window=False)[0]
    return {k: aggregated[k] for k in ("frames", "psnr_db", "zero_error", "ssim", "lpips", "sse", "pixel_count")}
