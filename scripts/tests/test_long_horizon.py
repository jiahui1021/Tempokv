from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


LONG_HORIZON = Path(__file__).parents[1] / "paper_experiments" / "long_horizon"
if str(LONG_HORIZON) not in sys.path:
    sys.path.insert(0, str(LONG_HORIZON))


def _module(name: str):
    path = LONG_HORIZON / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"long_horizon_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


methods = _module("methods")
plot_figure = _module("plot_figure")
run_reconstruction = _module("run_reconstruction")


# --------------------------------------------------------------------------
# The comparison is about predictors, not about the quantizer
# --------------------------------------------------------------------------


def test_every_method_shares_one_quantizer() -> None:
    variable = {
        "k_predictor_mode",
        "v_predictor_mode",
        "k_history_length",
        "v_history_length",
        "predictor_direction",
    }
    mappings = {item.name: item.codec_mapping(1560, 2) for item in methods.METHODS}
    reference = mappings["tempokv"]
    for name, mapping in mappings.items():
        assert {k: v for k, v in mapping.items() if k not in variable} == {
            k: v for k, v in reference.items() if k not in variable
        }, name


def test_rtn_is_the_shipped_zero_predictor_path() -> None:
    """A zero-length temporal map is direct quantization; not a reimplementation."""
    rtn = methods.method("rtn")
    assert rtn.k_predictor_mode == "direct"
    assert rtn.v_predictor_mode == "direct"
    assert rtn.k_history_length == 0
    assert rtn.v_history_length == 0
    assert not rtn.has_cross_term


def test_unknown_methods_name_what_is_implemented() -> None:
    with pytest.raises(ValueError, match="tempokv"):
        methods.method("qvg")


@pytest.mark.parametrize("name", [item.name for item in methods.METHODS])
def test_every_method_builds_a_valid_codec(name: str) -> None:
    config = methods.method(name).codec(1560, 2)
    assert config.k_bits == config.v_bits == 2


# --------------------------------------------------------------------------
# Frame curves
# --------------------------------------------------------------------------


def _curve(errors: list[float]) -> dict:
    clean = torch.ones(1, 1, len(errors), 1, 1)
    rebuilt = clean.clone()
    for index, value in enumerate(errors):
        rebuilt[0, 0, index, 0, 0] = 1.0 + value
    return methods.frame_curve(clean, rebuilt)


def test_the_cumulative_curve_ends_at_the_pooled_number() -> None:
    """A table quoting the pooled figure cannot disagree with the plotted curve."""
    curve = _curve([0.1, 0.2, 0.3, 0.4])
    assert curve["cumulative"][-1]["nmse"] == pytest.approx(curve["pooled_nmse"])


def test_per_frame_error_is_reported_at_every_frame() -> None:
    curve = _curve([0.1, 0.0, 0.5])
    assert curve["frames"] == 3
    assert [row["frame"] for row in curve["per_frame"]] == [0, 1, 2]
    assert curve["per_frame"][1]["nmse"] == pytest.approx(0.0)


def test_a_late_degrading_method_is_distinguished_from_a_uniform_one() -> None:
    """This separation is the reason the study exists."""
    late = methods.summarize_curve(_curve([0.1] * 6 + [1.0] * 2), tail_fraction=0.25)
    uniform = methods.summarize_curve(_curve([0.25] * 8), tail_fraction=0.25)
    assert uniform["tail_over_head"] == pytest.approx(1.0)
    assert late["tail_over_head"] > 5.0


def test_the_head_window_is_everything_before_the_tail() -> None:
    """It once took the first `tail_length` frames instead, shrinking the head."""
    summary = methods.summarize_curve(
        _curve([0.1] * 6 + [1.0] * 2), tail_fraction=0.25
    )
    assert summary["tail_start_frame"] == 6
    # 0.1 relative error on a unit reference is 0.01 NMSE, over all six frames.
    assert summary["head_mean_nmse"] == pytest.approx(0.01, rel=1e-6)


def test_a_full_width_tail_compares_against_itself() -> None:
    """With no frames outside the tail the ratio is 1, not a slice of the tail."""
    summary = methods.summarize_curve(_curve([0.2] * 4), tail_fraction=1.0)
    assert summary["tail_start_frame"] == 0
    assert summary["tail_over_head"] == pytest.approx(1.0)


def test_summarize_reports_the_tail_window_it_used() -> None:
    summary = methods.summarize_curve(_curve([0.1] * 8), tail_fraction=0.5)
    assert summary["tail_start_frame"] == 4


# --------------------------------------------------------------------------
# Trace loading
# --------------------------------------------------------------------------


def _write_trace(path: Path, *, frames: int, records: int = 4, layers: int = 2):
    rng = np.random.default_rng(0)
    shape = (layers, records, frames, 2, 128)
    np.savez(
        path,
        keys=rng.normal(size=shape).astype(np.float32),
        values=rng.normal(size=shape).astype(np.float32),
        layer_ids=np.arange(layers, dtype=np.int32),
        trace_stride=np.asarray(4, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
    )
    return path


def test_max_records_trims_records_and_never_frames(tmp_path: Path) -> None:
    """Memory control must not silently shorten the curve being measured."""
    trace = _write_trace(tmp_path / "trace.npz", frames=20, records=8)
    keys, _, _ = run_reconstruction._load_trace(trace, max_records=3)
    assert keys.shape[1] == 3
    assert keys.shape[2] == 20


def test_a_mismatched_stride_is_refused(tmp_path: Path) -> None:
    """Two traces with different strides come from different runs."""
    horizon = _write_trace(tmp_path / "horizon.npz", frames=12)
    calibration = tmp_path / "calibration.npz"
    rng = np.random.default_rng(1)
    shape = (2, 4, 6, 2, 128)
    np.savez(
        calibration,
        keys=rng.normal(size=shape).astype(np.float32),
        values=rng.normal(size=shape).astype(np.float32),
        layer_ids=np.arange(2, dtype=np.int32),
        trace_stride=np.asarray(8, dtype=np.int32),
        coordinate_system=np.asarray("pre_rope_normalized"),
    )
    args = run_reconstruction.build_parser().parse_args(
        [
            "--calibration-trace", str(calibration),
            "--horizon-trace", str(horizon),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="different runs"):
        run_reconstruction.run(args)


# --------------------------------------------------------------------------
# Collection arithmetic
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "video_frames,latent_frames",
    [(105, 27), (177, 45), (357, 90), (717, 180), (1053, 264), (1401, 351)],
)
def test_the_collector_converts_video_frames_to_latent_frames(
    video_frames: int, latent_frames: int, tmp_path: Path
) -> None:
    """1401 is a video frame count; the trace is collected in latent frames."""
    import os

    completed = subprocess.run(
        ["bash", str(LONG_HORIZON / "collect_trace.sh")],
        env={
            **os.environ,
            "CHECKPOINT": "/nonexistent",
            "WAN_MODEL_DIR": "/nonexistent",
            "PROMPTS": str(LONG_HORIZON.parents[2] / "assets" / "moviegen128_first30.txt"),
            "HORIZON_FRAMES": str(video_frames),
            "TRACE_DIR": str(tmp_path),
            "DRY_RUN": "1",
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert f"TRACE_FRAMES={latent_frames}" in completed.stdout
    # The long protocol cannot express these lengths, so it must be off.
    assert "LEGACY_PROTOCOL=0" in completed.stdout


@pytest.mark.parametrize("video_frames", [100, 106, 1400])
def test_frame_counts_that_do_not_map_to_latent_blocks_are_refused(
    video_frames: int, tmp_path: Path
) -> None:
    import os

    completed = subprocess.run(
        ["bash", str(LONG_HORIZON / "collect_trace.sh")],
        env={
            **os.environ,
            "CHECKPOINT": "/nonexistent",
            "WAN_MODEL_DIR": "/nonexistent",
            "PROMPTS": str(LONG_HORIZON.parents[2] / "assets" / "moviegen128_first30.txt"),
            "HORIZON_FRAMES": str(video_frames),
            "TRACE_DIR": str(tmp_path),
            "DRY_RUN": "1",
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "FRAMES+3" in completed.stderr


# --------------------------------------------------------------------------
# Figure inputs
# --------------------------------------------------------------------------


def test_curve_rows_carry_both_views_per_frame() -> None:
    report = {
        "methods": [
            {
                "method": "tempokv",
                "label": "TempoKV",
                "k": _curve([0.1, 0.2]),
                "v": _curve([0.3, 0.4]),
            }
        ]
    }
    rows = plot_figure.curve_rows(report)
    assert {row["tensor"] for row in rows} == {"K", "V"}
    assert all(row["cumulative_nmse"] is not None for row in rows)
