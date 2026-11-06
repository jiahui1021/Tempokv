from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from rebuttal.common.statistics import average_ranks, bootstrap_mean, correlation, spearman
from rebuttal.common.video_metrics import VideoFrame, aggregate_frames, iter_pair_metrics, resolve_pairs, window_index
from rebuttal.experiments.motion.groups import assign_groups
from rebuttal.experiments.motion.report import cache_motion_intervals, join_cache


def pair():
    return {"model": "test", "prompt_id": "p0", "seed": 0, "sample_id": 0, "method": "tempokv", "comparison_id": "matched", "reference": "left.mp4", "candidate": "right.mp4"}


class Reader:
    count = 1401
    fps = 16.0
    width = height = 2
    timestamp_policy = "synthetic_cfr"
    closed = []

    def __init__(self, path, decoder):
        self.path = path
        self.declared_frames = self.count

    def __iter__(self):
        for i in range(self.count):
            value = 0 if str(self.path).startswith("left") else 1
            yield VideoFrame(i, i / self.fps, np.full((2, 2, 3), value, dtype=np.uint8))

    def close(self):
        self.closed.append(str(self.path))


def test_real_15_second_windows_keep_partial_1401_frame_tail():
    rows = list(iter_pair_metrics(pair(), {"metrics": ["psnr"], "expected_frames": 1401}, reader_factory=Reader))
    windows = aggregate_frames(rows)
    assert [r["frames"] for r in windows] == [240, 240, 240, 240, 240, 201]
    assert windows[-1]["duration_seconds"] == pytest.approx(12.5625)
    assert sum(r["sse"] for r in windows) == 1401 * 12
    assert windows[0]["psnr_db"] == pytest.approx(48.1308036087)
    assert rows[240]["window_index"] == 1
    assert window_index(15, 15) == 1


def test_psnr_is_from_pooled_pixels_not_average_decibels():
    rows = [{**pair(), "window_index": 0, "timestamp": i, "frame_duration": 1, "sse": value, "pixel_count": 1} for i, value in enumerate((1, 9))]
    actual = aggregate_frames(rows)[0]
    assert actual["mse"] == 5
    assert actual["psnr_db"] == pytest.approx(10 * np.log10(255**2 / 5))
    rows[0]["sse"] = rows[1]["sse"] = 0
    assert aggregate_frames(rows)[0]["psnr_db"] is None
    assert aggregate_frames(rows)[0]["zero_error"] is True


def test_paired_video_mismatch_raises_and_readers_close():
    class Unequal(Reader):
        count = 3
        def __iter__(self):
            for frame in super().__iter__():
                if str(self.path).startswith("right") and frame.index == 2:
                    break
                yield frame
    Unequal.closed = []
    with pytest.raises(ValueError, match="counts differ"):
        list(iter_pair_metrics(pair(), {"metrics": ["psnr"]}, reader_factory=Unequal))
    assert len(Unequal.closed) == 2


def test_pts_alignment_and_variable_rate_are_checked():
    class BadPTS(Reader):
        count = 3
        def __iter__(self):
            for frame in super().__iter__():
                if frame.index == 2:
                    frame.timestamp += 0.02
                yield frame
    with pytest.raises(ValueError, match="variable-rate"):
        list(iter_pair_metrics(pair(), {"metrics": ["psnr"]}, reader_factory=BadPTS))


def test_prompt_bootstrap_does_not_weight_repeated_frames_as_prompts():
    rows = [{"prompt_id": "a", "x": 0.0}] * 99 + [{"prompt_id": "b", "x": 10.0}]
    result = bootstrap_mean(rows, "x", samples=20)
    assert result["mean"] == 5
    assert result["n_prompts"] == 2
    assert average_ranks([1, 1, 3]) == [1.5, 1.5, 3]
    assert spearman([1, 1, 3], [2, 2, 6]) == pytest.approx(1)
    assert correlation([{"prompt_id": str(i), "x": 0, "y": i} for i in range(4)], "x", "y")["rho"] is None


def test_motion_groups_preserve_ties_and_share_bf16_between_methods():
    rows = [{"model": "a", "prompt_id": str(i), "motion_magnitude": value, "method": method} for i, value in enumerate((0, 0, 0)) for method in ("int2", "int4")]
    groups = assign_groups(rows)
    assert groups["thresholds"]["a"]["n_reference_videos"] == 3
    assert {r["motion_group"] for r in rows} == {"low"}


def test_missing_pixel_mapping_does_not_guess_from_latent_frame(tmp_path):
    p = pair()
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({**p, "role": "k", "absolute_latent_frame": 3, "metric_scope": "bf16_trace_replay", "error_sse": 2, "signal_sse": 10}) + "\n")
    p["cache_events"] = str(path)
    result = join_cache(p, 0, 4)
    assert result["cache_missing_reason"] == "pixel_time_mapping_missing"
    assert result["k_reconstruction_nmse"] is None


def test_cache_join_uses_energy_sums_and_exact_identity_and_coverage(tmp_path):
    p = pair()
    events = []
    for role in ("k", "v"):
        for start, end, error, signal in ((0, 1, 1, 1), (1, 4, 9, 99)):
            events.append({**p, "role": role, "metric_scope": "bf16_trace_replay", "pixel_start_frame": start, "pixel_end_frame": end,
                           "error_sse": error, "signal_sse": signal, "prediction_sse": signal / 2, "is_anchor": start == 0})
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events))
    p["cache_events"] = str(path)
    result = join_cache(p, 0, 4)
    assert result["cache_missing_reason"] is None
    assert result["k_reconstruction_nmse"] == pytest.approx(0.1)
    assert result["k_prediction_nmse"] == pytest.approx(0.5)
    assert join_cache(p, 0, 3)["cache_missing_reason"] == "cache_and_quality_coverage_differ"
    other = {**p, "seed": 99}
    assert join_cache(other, 0, 4)["cache_missing_reason"] == "no_matching_identity_and_scope"


def test_camera_cache_motion_uses_explicit_pixel_interval(tmp_path):
    p = pair()
    events = [{**p, "role": role, "metric_scope": "bf16_trace_replay", "pixel_start_frame": 0, "pixel_end_frame": 4,
               "error_sse": 1, "signal_sse": 10, "prediction_sse": 2} for role in ("k", "v")]
    p["cache_events"] = "declared.jsonl"
    flow = [{**p, "frame_index": i, "next_frame_index": i+1, "timestamp": i/16, "end_timestamp": (i+1)/16,
             "motion_magnitude": i+1, "global_motion": 3, "residual_motion": 1, "camera_valid": True, "is_cut": False} for i in range(3)]
    joined = cache_motion_intervals(p, flow, "bf16_trace_replay", events)
    assert len(joined) == 1
    assert joined[0]["motion_magnitude"] == 2
    assert joined[0]["k_reconstruction_nmse"] == 0.1
    assert joined[0]["camera_valid"] is True


def test_generation_metadata_must_match_when_sidecars_provided(tmp_path):
    p = pair()
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps({"seed": 0}))
    right.write_text(json.dumps({"seed": 1}))
    with pytest.raises(ValueError, match="metadata mismatch"):
        resolve_pairs({"pairs": [{**p, "reference_metadata": str(left), "candidate_metadata": str(right)}], "matched_metadata_fields": ["seed"]})


def test_directory_pairing_uses_generation_noise_seed_not_filename_guess(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    for directory in (reference, candidate):
        directory.mkdir()
        (directory / "0_0.mp4").touch()
        (directory / "summary.json").write_text(json.dumps({"rank_summaries": [{"videos": [
            {"path": "/former/server/videos/0_0.mp4", "prompt_index": 38, "sample_index": 2, "noise_seed": 38000114}
        ]}]}))
    result = resolve_pairs({"reference_dir": str(reference), "candidate_dir": str(candidate), "model": "self_forcing"})[0]
    assert result["prompt_id"] == "38"
    assert result["sample_id"] == 2
    assert result["seed"] == 38000114
    assert result["identity_validation"] == "generation_indices_matched_not_full_generation_settings_verified"
    changed = {"rank_summaries": [{"videos": [{"path": "0_0.mp4", "prompt_index": 38, "sample_index": 2, "noise_seed": 1}]}]}
    (candidate / "summary.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="generation identity differs"):
        resolve_pairs({"reference_dir": str(reference), "candidate_dir": str(candidate)})


def test_full_motion_report_keeps_method_identity_and_explicit_cache_intervals(tmp_path, monkeypatch):
    from rebuttal.experiments import temporal_window
    from rebuttal.experiments.motion import report
    class ShortReader(Reader):
        count = 8
        fps = 4.0
    class Flow:
        metadata = {"backend": "synthetic_test_only"}
        def __call__(self, a, b):
            return np.ones((*a.shape[:2], 2), dtype=np.float32)
    original = iter_pair_metrics
    monkeypatch.setattr(temporal_window, "iter_pair_metrics", lambda p, c, backend=None: original(p, c, backend=backend, reader_factory=ShortReader))
    monkeypatch.setattr(report, "VideoReader", ShortReader)
    monkeypatch.setattr(report, "make_flow_backend", lambda cfg: Flow())
    monkeypatch.setattr(report, "camera_proxy", lambda f, cfg: {"global_motion": 1.0, "residual_motion": 0.5, "camera_valid": True, "camera_failure": None})
    pairs = []
    for method in ("int2", "int4"):
        p = {**pair(), "method": method}
        path = tmp_path / f"events_{method}.jsonl"
        events = [{**p, "role": role, "metric_scope": "bf16_trace_replay", "pixel_start_frame": start, "pixel_end_frame": start+4,
                   "error_sse": 1, "signal_sse": 10, "prediction_sse": 2, "is_anchor": False} for role in ("k", "v") for start in (0, 4)]
        path.write_text("\n".join(json.dumps(r) for r in events))
        p["cache_events"] = str(path)
        pairs.append(p)
    result = report.run({"pairs": pairs, "metrics": ["psnr"], "window_seconds": 1,
                         "flow": {"forward_backward": False}, "difficult_cases": {"export_clips": False},
                         "bootstrap_samples": 0, "plots": False}, "evaluate", tmp_path / "output")
    assert result["n_video_pairs"] == 2
    assert {r["method"] for r in result["groups"]} == {"int2", "int4"}
    assert all(r["n_cache"] == 1 for r in result["groups"])
    assert (tmp_path / "output" / "motion_cache_pairs.csv").is_file()
    cases = json.loads((tmp_path / "output" / "difficult_cases.json").read_text())
    assert cases["verified_case_count"] == 0


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="opencv optional dependency unavailable")
def test_flow_resize_restores_anisotropic_original_pixel_units():
    from rebuttal.experiments.motion.flow import restore_flow_units
    flow = np.ones((8, 10, 2), dtype=np.float32)
    restored = restore_flow_units(flow, 32, 20)
    assert restored.shape == (32, 20, 2)
    assert np.allclose(restored[..., 0], 2)
    assert np.allclose(restored[..., 1], 4)


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="opencv optional dependency unavailable")
def test_robust_affine_separates_constant_camera_translation():
    from rebuttal.experiments.motion.camera import camera_proxy
    flow = np.zeros((64, 64, 2), dtype=np.float32)
    flow[..., 0], flow[..., 1] = 3, 4
    result = camera_proxy(flow, {"grid_step": 8})
    assert result["camera_valid"] is True
    assert result["global_motion"] == pytest.approx(5, abs=1e-5)
    assert result["residual_motion"] == pytest.approx(0, abs=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None or importlib.util.find_spec("av") is None, reason="video integration dependencies unavailable")
def test_real_pyav_videos_farneback_and_paired_case_exports(tmp_path):
    import av
    from fractions import Fraction
    from rebuttal.experiments.motion.report import run
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    base = np.random.default_rng(17).integers(0, 220, (64, 64, 3), dtype=np.uint8)
    for directory, offset in ((reference, 0), (candidate, 6)):
        directory.mkdir()
        container = av.open(str(directory / "0_0.mp4"), "w")
        stream = container.add_stream("mpeg4", rate=16)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(35):
            image = np.roll(base, index, axis=1) + offset
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 16)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        (directory / "summary.json").write_text(json.dumps({"rank_summaries": [{"videos": [{"path": "0_0.mp4", "prompt_index": 0, "sample_index": 0, "noise_seed": 19}]}]}))
    out = tmp_path / "motion"
    result = run({"reference_dir": str(reference), "candidate_dir": str(candidate), "model": "self_forcing", "method": "int4",
                  "metrics": ["psnr"], "expected_frames": 35, "window_seconds": 1, "video_decoder": "pyav",
                  "flow": {"backend": "farneback", "resize_width": 32, "forward_backward": True},
                  "camera": {"grid_step": 8}, "bootstrap_samples": 10,
                  "difficult_cases": {"clips_per_category": 1, "clip_seconds": 0.5, "export_clips": True}, "plots": True}, "evaluate", out)
    assert result["n_video_pairs"] == 1
    assert result["groups"][0]["n_videos"] == 1
    video = json.loads((out / "summary.json").read_text())
    assert video["cache_missing"][0]["cache_missing_reason"] == "no_cache_events"
    from rebuttal.common.video_metrics import read_csv
    windows = read_csv(out / "quality" / "windows.csv")
    assert [r["frames"] for r in windows] == [16, 16, 3]
    assert read_csv(out / "motion_videos.csv")[0]["flow_pairs"] == 34
    assert (out / "motion_scatter_self_forcing_int4_default.png").is_file()
    cases = json.loads((out / "difficult_cases.json").read_text())
    assert cases["candidate_only_case_count"] >= 1
    assert all(Path(r["clip"]["path"]).is_file() for r in cases["cases"])
