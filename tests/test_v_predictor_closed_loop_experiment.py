from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCTION_SCRIPT = ROOT / "tests" / "experiment_v_reconstruction_closed_loop.py"
COSINE_SCRIPT = ROOT / "tests" / "experiment_predictor_cosine_closed_loop.py"


def _load_experiment(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reconstruction = _load_experiment(RECONSTRUCTION_SCRIPT, "v_reconstruction_experiment")
cosine = _load_experiment(COSINE_SCRIPT, "predictor_cosine_experiment")


def _write_trace(path: Path, *, frames: int, samples: int = 10) -> None:
    rng = np.random.default_rng(17)
    keys = rng.normal(size=(2, samples, frames, 2, 64)).astype(np.float32)
    values = np.empty_like(keys)
    values[:, :, 0] = 0.65 * keys[:, :, 0]
    for frame in range(1, frames):
        values[:, :, frame] = (
            0.65 * keys[:, :, frame]
            + 0.3 * values[:, :, frame - 1]
            + rng.normal(scale=0.03, size=values[:, :, frame].shape)
        )
    np.savez_compressed(
        path,
        keys=keys,
        values=values,
        layer_ids=np.arange(2, dtype=np.int32),
        stride=np.asarray(8, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
        cross_source_kind=np.asarray("normalized"),
    )


def test_short_trace_is_rejected_instead_of_repeated(tmp_path: Path) -> None:
    trace = tmp_path / "short.npz"
    _write_trace(trace, frames=21)
    with pytest.raises(ValueError, match="only 21 frames; 180 real frames are required"):
        reconstruction.load_trace(trace, frames=180, requested_layers=None)
    with pytest.raises(ValueError, match="only 21 frames; 180 real frames are required"):
        cosine.load_trace(trace, frames=180, requested_layers=None)


def test_metrics_cover_every_closed_loop_frame() -> None:
    clean = torch.ones(2, 3, 180, 2, 4)
    rebuilt = clean.clone()
    rebuilt[:, :, 1:] += torch.arange(1, 180).view(1, 1, 179, 1, 1) / 1000
    metrics = reconstruction.reconstruction_metrics(clean, rebuilt, start_frame=1)

    assert len(metrics["per_frame"]) == 179
    assert metrics["per_frame"][0]["frame"] == 1
    assert metrics["per_frame"][-1]["frame"] == 179
    assert metrics["distribution"]["last_10_frame_mean"] > metrics["distribution"][
        "first_10_frame_mean"
    ]


def test_cosine_similarity_uses_each_head_vector_and_all_frames() -> None:
    target = torch.zeros(1, 2, 180, 2, 3)
    target[..., 0] = 1.0
    prediction = target.clone()
    prediction[:, :, 90:, 0] = -target[:, :, 90:, 0]
    prediction[:, :, 90:, 1, 0] = 0.0
    prediction[:, :, 90:, 1, 1] = 1.0

    metrics = cosine.cosine_similarity_metrics(
        target, prediction, start_frame=1
    )

    assert len(metrics["per_frame"]) == 179
    assert metrics["per_frame"][0]["cosine"] == pytest.approx(1.0)
    assert metrics["per_frame"][-1]["cosine"] == pytest.approx(-0.5)
    assert -1.0 <= metrics["mean_cosine"] <= 1.0


@pytest.mark.parametrize(
    "script",
    (RECONSTRUCTION_SCRIPT, COSINE_SCRIPT),
)
def test_separate_experiment_entrypoints_are_runnable(script: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--frames" in completed.stdout


def test_full_180_frame_fit_and_rollout(tmp_path: Path) -> None:
    trace = tmp_path / "trace_180.npz"
    reconstruction_output = tmp_path / "reconstruction.json"
    cosine_output = tmp_path / "cosine.json"
    _write_trace(trace, frames=180)
    common = dict(
        trace=trace,
        config=None,
        output_dir=None,
        frames=180,
        holdout_step=5,
        iterations=2,
        layers=None,
        device="cpu",
        threads=1,
    )
    reconstruction_report = reconstruction.run(
        argparse.Namespace(output=reconstruction_output, **common)
    )
    cosine_report = cosine.run(argparse.Namespace(output=cosine_output, **common))

    assert reconstruction_output.exists() and cosine_output.exists()
    persisted_reconstruction = json.loads(
        reconstruction_output.read_text(encoding="utf-8")
    )
    persisted_cosine = json.loads(cosine_output.read_text(encoding="utf-8"))
    assert persisted_reconstruction["protocol"]["frames"] == 180
    assert persisted_cosine["protocol"]["frames"] == 180
    assert persisted_reconstruction["scope"]["evaluated_frames"] == list(range(1, 180))
    assert persisted_cosine["scope"]["evaluated_frames"] == list(range(1, 180))
    assert "comparison" in persisted_reconstruction
    assert "cosine_comparison" not in persisted_reconstruction
    assert "cosine_comparison" in persisted_cosine
    assert "comparison" not in persisted_cosine
    assert set(persisted_cosine["cosine_comparison"]["mean_cosine"]) == {
        "k_temporal",
        "v_temporal_only",
        "v_cross_kv_history",
        "k_reverse_cross_vk_history",
    }
    reverse = persisted_cosine["methods"]["reverse_cross_vk_history"]
    assert len(reverse["key"]["per_frame"]) == 179
    assert len(reverse["value_temporal_source"]["per_frame"]) == 179
    assert len(persisted_cosine["cosine_comparison"]["per_frame"]) == 179
    assert reconstruction_report["matched_k_fit"]["max_abs_weight_difference"] == pytest.approx(0.0)
    assert cosine_report["matched_k_fit"]["max_abs_weight_difference"] == pytest.approx(0.0)
    for method in reconstruction.METHODS:
        metric = persisted_reconstruction["methods"][method]["v_reconstruction"]
        assert metric["pooled_nmse"] >= 0.0
        assert len(metric["per_frame"]) == 179
