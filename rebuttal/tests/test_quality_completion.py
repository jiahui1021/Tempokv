"""Interrupted quality runs must never certify stale or partial score tables."""
from __future__ import annotations

import json

import pytest

from rebuttal.common.video_metrics import IDENTITY_FIELDS, read_csv, resolve_pairs, write_json
from rebuttal.experiments import temporal_window
from rebuttal.experiments.motion.report import validate_quality_reuse


@pytest.fixture
def quality_run(tmp_path, monkeypatch):
    reference, candidate = tmp_path / "reference.mp4", tmp_path / "candidate.mp4"
    reference.write_bytes(b"reference video")
    candidate.write_bytes(b"original candidate video")
    config = {"pairs": [{"model": "self_forcing", "prompt_id": "0001", "seed": 0,
                         "sample_id": 0, "method": "tempokv", "comparison_id": "full",
                         "reference": str(reference), "candidate": str(candidate)}],
              "metrics": ["psnr"], "plots": False, "bootstrap_samples": 0}

    def scores(pair, config, **kwargs):
        for index in range(3):
            yield {**{key: pair[key] for key in IDENTITY_FIELDS}, "frame_index": index,
                   "timestamp": index / 16, "frame_duration": 1 / 16, "window_index": 0,
                   "sse": float(index + 1), "pixel_count": 12, "ssim": None, "lpips": None}

    monkeypatch.setattr(temporal_window, "iter_pair_metrics", scores)
    output = tmp_path / "quality"
    result = temporal_window.run(config, "evaluate", output)
    assert result["status"] == "complete"
    return output, config, candidate, scores


def test_completed_quality_is_reusable_after_plan_and_summarize(quality_run):
    output, config, _, _ = quality_run
    completion_path = output / temporal_window.QUALITY_COMPLETION_FILE
    completed = completion_path.read_bytes()
    validate_quality_reuse(output, resolve_pairs(config), config)
    temporal_window.run(config, "plan", output)
    temporal_window.run(config, "summarize", output)
    assert completion_path.read_bytes() == completed
    validate_quality_reuse(output, resolve_pairs(config), config)


@pytest.mark.parametrize("yield_partial_frame", [False, True])
def test_failed_rerun_cannot_certify_old_scores_for_changed_video(quality_run, monkeypatch, yield_partial_frame):
    output, config, candidate, scores = quality_run
    old_video_scores = (output / "videos.csv").read_bytes()
    old_fingerprints = json.loads((output / "video_pairs.json").read_text())["pairs"][0]["video_fingerprints"]
    candidate.write_bytes(b"different candidate video")

    def interrupted(pair, config, **kwargs):
        if yield_partial_frame:
            yield next(scores(pair, config))
        raise RuntimeError("decoder interrupted")

    monkeypatch.setattr(temporal_window, "iter_pair_metrics", interrupted)
    with pytest.raises(RuntimeError, match="decoder interrupted"):
        temporal_window.run(config, "evaluate", output)
    current_fingerprints = json.loads((output / "video_pairs.json").read_text())["pairs"][0]["video_fingerprints"]
    assert current_fingerprints != old_fingerprints
    assert (output / "videos.csv").read_bytes() == old_video_scores
    assert not (output / temporal_window.QUALITY_COMPLETION_FILE).exists()
    with pytest.raises(ValueError, match="successful completion"):
        validate_quality_reuse(output, resolve_pairs(config), config)
    # Summarization of retained diagnostic tables cannot recreate authentication.
    temporal_window.run(config, "summarize", output)
    with pytest.raises(ValueError, match="successful completion"):
        validate_quality_reuse(output, resolve_pairs(config), config)
    monkeypatch.setattr(temporal_window, "iter_pair_metrics", scores)
    temporal_window.run(config, "evaluate", output)
    validate_quality_reuse(output, resolve_pairs(config), config)


def test_failed_setup_invalidates_previous_completion(quality_run, monkeypatch):
    output, config, _, _ = quality_run

    def unavailable(config):
        raise RuntimeError("quality backend unavailable")

    monkeypatch.setattr(temporal_window, "make_quality_backend", unavailable)
    with pytest.raises(RuntimeError, match="backend unavailable"):
        temporal_window.run(config, "evaluate", output)
    with pytest.raises(ValueError, match="successful completion"):
        validate_quality_reuse(output, resolve_pairs(config), config)


@pytest.mark.parametrize("name", temporal_window.QUALITY_OUTPUT_FILES)
def test_changed_completed_output_is_rejected(quality_run, name):
    output, config, _, _ = quality_run
    path = output / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="completed output fingerprint"):
        validate_quality_reuse(output, resolve_pairs(config), config)


def test_missing_completed_output_is_rejected(quality_run):
    output, config, _, _ = quality_run
    (output / "frames.csv").unlink()
    with pytest.raises(ValueError, match="completed output file is missing"):
        validate_quality_reuse(output, resolve_pairs(config), config)


def test_missing_or_incomplete_completion_is_rejected(quality_run):
    output, config, _, _ = quality_run
    path = output / temporal_window.QUALITY_COMPLETION_FILE
    path.unlink()
    with pytest.raises(ValueError, match="successful completion"):
        validate_quality_reuse(output, resolve_pairs(config), config)
    write_json(path, {"schema_version": temporal_window.QUALITY_COMPLETION_SCHEMA, "status": "running"})
    with pytest.raises(ValueError, match="completion record is invalid"):
        validate_quality_reuse(output, resolve_pairs(config), config)


def test_failed_summary_does_not_publish_completion(quality_run, monkeypatch):
    output, config, _, _ = quality_run

    def fail(*args):
        raise RuntimeError("summary interrupted")

    monkeypatch.setattr(temporal_window, "summarize", fail)
    with pytest.raises(RuntimeError, match="summary interrupted"):
        temporal_window.run(config, "evaluate", output)
    assert len(read_csv(output / "videos.csv")) == 1
    with pytest.raises(ValueError, match="successful completion"):
        validate_quality_reuse(output, resolve_pairs(config), config)
