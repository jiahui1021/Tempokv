"""Regressions for experiment identity and shared measurement intervals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rebuttal.common.video_metrics import (
    VideoFrame, aggregate_frames, read_csv, resolve_pairs, write_csv, write_json,
)
from rebuttal.experiments import temporal_window
from rebuttal.experiments.motion import report
from rebuttal.experiments.motion.groups import assign_groups


def pair(**updates):
    return {"model": "self_forcing", "prompt_id": "0_0", "seed": 0, "sample_id": 0,
            "method": "int4", "comparison_id": "full", "reference": "bf16.mp4",
            "candidate": "int4.mp4", "cache_events": "events.jsonl", **updates}


def cache_events(p, *, error=1.0, intervals=((0, 4),)):
    return [{**p, "role": role, "metric_scope": "bf16_trace_replay", "pixel_start": start,
             "pixel_end": end, "signal_sse": 10.0, "error_sse": error}
            for role in ("k", "v") for start, end in intervals]


def test_csv_preserves_opaque_ids_without_merging_prompts(tmp_path):
    rows = [pair(prompt_id=prompt, psnr_db=20.0, frames=4) for prompt in ("0_0", "0001", "1", "1e2", "True")]
    path = tmp_path / "scores.csv"
    write_csv(path, rows)
    loaded = read_csv(path)
    assert [r["prompt_id"] for r in loaded] == [r["prompt_id"] for r in rows]
    assert all(r["seed"] == 0 and r["frames"] == 4 for r in loaded)
    from rebuttal.common.statistics import grouped_summary
    assert grouped_summary(loaded, ["model", "method"], ["psnr_db"], samples=0)[0]["metrics"]["psnr_db"]["n_prompts"] == 5


def test_cache_join_separates_method_variant_and_context():
    p = pair(cache_variant="production")
    rows = (cache_events({**p, "variant": "production"}) +
            cache_events({**p, "method": "int2", "variant": "production"}, error=9) +
            cache_events({**p, "variant": "previous_frame"}, error=7) +
            cache_events({**p, "variant": "production", "comparison_id": "short"}, error=5))
    result = report.join_cache(p, 0, 4, events=rows)
    assert result["k_reconstruction_nmse"] == pytest.approx(0.1)
    assert result["v_reconstruction_nmse"] == pytest.approx(0.1)
    assert result["cache_missing_reason"] is None
    flow = [{**p, "frame_index": i, "next_frame_index": i + 1, "timestamp": i / 16,
             "end_timestamp": (i + 1) / 16, "camera_valid": True, "is_cut": False,
             "motion_magnitude": 1, "global_motion": 1, "residual_motion": 0} for i in range(3)]
    assert report.cache_motion_intervals(p, flow, "bf16_trace_replay", rows)[0]["k_reconstruction_nmse"] == pytest.approx(0.1)


def test_cache_variant_only_replay_requires_explicit_mapping():
    p = pair()
    rows = cache_events(p)
    for row in rows:
        row.pop("method")
        row.pop("comparison_id")
        row["variant"] = "production"
    assert report.join_cache(p, 0, 4, events=rows)["cache_missing_reason"] == "no_matching_cache_variant"
    assert report.join_cache({**p, "cache_variant": "production"}, 0, 4, events=rows)["k_reconstruction_nmse"] == pytest.approx(0.1)
    rows[0].pop("variant")
    assert report.join_cache({**p, "cache_variant": "production"}, 0, 4, events=rows)["cache_missing_reason"] == "incomplete_cache_variant_identity"


def saved_quality(tmp_path):
    reference, candidate = tmp_path / "bf16.mp4", tmp_path / "int4.mp4"
    reference.write_bytes(b"reference video bytes")
    candidate.write_bytes(b"candidate video bytes")
    p = pair(reference=str(reference), candidate=str(candidate))
    config = {"pairs": [p], "metrics": ["psnr"], "plots": False}
    pairs = resolve_pairs(config)
    output = tmp_path / "quality"
    write_json(output / "video_pairs.json", {"pairs": pairs, "config": config})
    # This fixture exercises identity validation, not video score calculation.
    for name in ("frames.csv", "windows.csv", "videos.csv"):
        write_csv(output / name, [{"fixture": "completed scores"}])
    write_json(output / temporal_window.QUALITY_COMPLETION_FILE,
               {"schema_version": temporal_window.QUALITY_COMPLETION_SCHEMA, "status": "complete",
                "outputs": temporal_window.quality_output_fingerprints(output)})
    return output, config, candidate


def test_quality_reuse_checks_bytes_paths_and_metric_contract(tmp_path):
    output, config, candidate = saved_quality(tmp_path)
    report.validate_quality_reuse(output, resolve_pairs(config), config)
    candidate.write_bytes(b"replacement video bytes")
    with pytest.raises(ValueError, match="candidate input fingerprint"):
        report.validate_quality_reuse(output, resolve_pairs(config), config)
    candidate.unlink()
    with pytest.raises(ValueError, match="candidate input fingerprint"):
        report.validate_quality_reuse(output, resolve_pairs(config), config)
    changed = {**config, "metrics": ["psnr", "lpips"]}
    with pytest.raises(ValueError, match="metric/decoder contract"):
        report.validate_quality_reuse(output, resolve_pairs(changed), changed)


def test_quality_reuse_rejects_legacy_unfingerprinted_pairs(tmp_path):
    output, config, _ = saved_quality(tmp_path)
    old = json.loads((output / "video_pairs.json").read_text())
    old["pairs"][0].pop("video_fingerprints")
    write_json(output / "video_pairs.json", old)
    write_json(output / temporal_window.QUALITY_COMPLETION_FILE,
               {"schema_version": temporal_window.QUALITY_COMPLETION_SCHEMA, "status": "complete",
                "outputs": temporal_window.quality_output_fingerprints(output)})
    with pytest.raises(ValueError, match="fingerprint"):
        report.validate_quality_reuse(output, resolve_pairs(config), config)


def test_difficult_case_uses_same_expanded_cache_and_video_interval():
    p = pair()
    intervals = [(0, 1), *[(i, i + 4) for i in range(1, 65, 4)]]
    events = cache_events(p, intervals=intervals)
    frames = [{**p, "frame_index": i, "timestamp": i / 16, "frame_duration": 1 / 16,
               "window_index": 0, "sse": float(i + 1), "pixel_count": 1} for i in range(65)]
    case = {"start_seconds": 6 / 16, "end_seconds": 54 / 16}
    selected, alignment = report.case_frames(p, case, frames, events, "bf16_trace_replay")
    assert [r["frame_index"] for r in selected] == list(range(5, 57))
    assert alignment["requested_start_seconds"] == 6 / 16
    assert alignment["cache_interval_alignment"] == "expanded_to_shared_explicit_cache_boundaries"
    result = report.join_cache(p, 5, 57, events=events)
    assert result["cache_missing_reason"] is None
    assert result["k_reconstruction_nmse"] == pytest.approx(0.1)
    assert aggregate_frames(selected, group_by_window=False)[0]["sse"] == sum(range(6, 58))


def test_motion_main_score_excludes_cuts_and_all_cut_video_is_unclassified(monkeypatch):
    class Reader:
        fps = 16
        width = height = 16
        timestamp_policy = "synthetic_cfr"
        values = [0, 0, 255, 255]
        def __init__(self, *args):
            pass
        def __iter__(self):
            for i, value in enumerate(self.values):
                yield VideoFrame(i, i / 16, np.full((16, 16, 3), value, dtype=np.uint8))
        def close(self):
            pass
    calls = []
    def flow(a, b):
        calls.append(1)
        assert a[0, 0, 0] == b[0, 0, 0], "flow must never evaluate cross-cut pairs"
        value = np.zeros((16, 16, 2), dtype=np.float32)
        value[..., 0] = 2
        return value
    monkeypatch.setattr(report, "VideoReader", Reader)
    monkeypatch.setattr(report, "camera_proxy", lambda *args: {"camera_valid": False})
    rows, summary = report.analyze_reference(pair(), {"flow": {"forward_backward": False}}, flow)
    assert len(calls) == 2
    assert rows[1]["is_cut"] and rows[1]["motion_magnitude"] is None
    assert summary["motion_magnitude"] == 2
    Reader.values = [0, 255, 0]
    _, all_cuts = report.analyze_reference(pair(prompt_id="cut"), {"flow": {"forward_backward": False}}, flow)
    grouped = [summary, all_cuts]
    thresholds = assign_groups(grouped)
    assert thresholds["thresholds"]["self_forcing"]["n_reference_videos"] == 1
    assert all_cuts["motion_group"] == "unclassified"
    assert all_cuts["motion_missing_reason"] == "no_non_cut_frame_pairs"
    assert assign_groups([all_cuts])["thresholds"]["self_forcing"]["low"] is None


def test_filename_fallback_and_verified_quality_reuse_with_real_videos(tmp_path):
    av = pytest.importorskip("av")
    pytest.importorskip("cv2")
    from fractions import Fraction
    from rebuttal.experiments import temporal_window
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    base = np.random.default_rng(8).integers(0, 200, (64, 64, 3), dtype=np.uint8)
    for folder, difference in ((reference, 0), (candidate, 5)):
        folder.mkdir()
        with av.open(str(folder / "0_0.mp4"), "w") as container:
            stream = container.add_stream("mpeg4", rate=16)
            stream.width = stream.height = 64
            stream.pix_fmt = "yuv420p"
            for index in range(9):
                frame = av.VideoFrame.from_ndarray(np.roll(base, index, axis=1) + difference, format="rgb24")
                frame.pts, frame.time_base = index, Fraction(1, 16)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    config = {"reference_dir": str(reference), "candidate_dir": str(candidate), "model": "self_forcing",
              "method": "int4", "metrics": ["psnr"], "expected_frames": 9, "plots": False,
              "flow": {"backend": "farneback", "forward_backward": False},
              "camera": {"grid_step": 8}, "bootstrap_samples": 0,
              "difficult_cases": {"clips_per_category": 1, "export_clips": False}}
    temporal_window.run(config, "evaluate", tmp_path / "quality")
    result = report.run({**config, "quality_output_dir": str(tmp_path / "quality")}, "evaluate", tmp_path / "motion")
    assert result["status"] == "complete"
    assert result["n_video_pairs"] == 1
    assert read_csv(tmp_path / "motion" / "motion_videos.csv")[0]["prompt_id"] == "0_0"
