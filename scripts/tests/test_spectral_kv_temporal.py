"""Synthetic temporal-collector checks; no real-model result is asserted."""

import io
import json

import numpy as np
import pytest
import torch

from tempokv.spectral_kv.temporal import OnlineTemporalSpectrum


def test_constant_joint_dc_and_bounded_warmup():
    stats = OnlineTemporalSpectrum(window=3, spatial_bins=2, temporal_bins=2)
    rows = stats.update(4, torch.ones(5, 8, 8, 1, 2), frame_idx0=10)
    assert len(rows) == 4 * 2 * 2
    assert sorted({row["window_frames"] for row in rows}) == [2, 3]
    assert {row["frame_idx"] for row in rows} == {11, 12, 13, 14}
    assert all(row["dc_ratio"] == pytest.approx(1) and row["zero_non_dc"] for row in rows)
    assert all(row["power_excl_dc"] == pytest.approx(0) for row in rows)
    assert all(row["dc_definition"] == "joint_t0_h0_w0" for row in rows)
    assert len(stats._history[(0, 4, "k", "pre")]) == 3
    assert rows[-1]["window_start_frame"] == 12


def test_joint_frequency_is_not_flattened_and_temporal_dc_plane_remains():
    # A time-constant spatial checkerboard is temporal DC, but is not joint DC.
    row = (-1.0) ** torch.arange(8)
    checker = (row[:, None] * row[None, :])[None, :, :, None, None].expand(4, -1, -1, 1, 1)
    stats = OnlineTemporalSpectrum(window=4, spatial_bins=4, temporal_bins=3)
    rows = [r for r in stats.update(0, checker) if r["window_frames"] == 4]
    measured = [r for r in rows if r["fraction_excl_dc"] > 0.99]
    assert len(measured) == 1
    assert measured[0]["spatial_band"] == 3
    assert measured[0]["temporal_band"] == 0
    assert measured[0]["dc_ratio"] == pytest.approx(0)
    assert sum(r["fraction_excl_dc"] for r in rows) == pytest.approx(1)


def test_temporal_nyquist_is_separate_from_spatial_dc():
    alternating = ((-1.0) ** torch.arange(4))[:, None, None, None, None].expand(4, 8, 8, 1, 1)
    rows = OnlineTemporalSpectrum(window=4, spatial_bins=4, temporal_bins=3).update(0, alternating)
    last = [r for r in rows if r["window_frames"] == 4]
    measured = [r for r in last if r["fraction_excl_dc"] > 0.99]
    assert len(measured) == 1
    assert measured[0]["spatial_band"] == 0 and measured[0]["temporal_band"] == 2


def test_gaps_shapes_and_samples_do_not_join():
    x = torch.randn(1, 8, 8, 1, 2, generator=torch.Generator().manual_seed(7))
    stats = OnlineTemporalSpectrum(window=4)
    assert not stats.update(0, x, frame_idx0=0)
    assert not stats.update(0, x, frame_idx0=5)
    rows = stats.update(0, x, frame_idx0=6)
    assert rows and all(r["window_start_frame"] == 5 and r["window_frames"] == 2 for r in rows)
    assert not stats.update(0, x, frame_idx0=7, sample_id=2)
    assert not stats.update(0, x, frame_idx0=7, coordinate="post")
    assert not stats.update(0, x[:, :6], frame_idx0=7)


def test_explicit_denoising_pass_is_ignored_without_inference_from_timestep():
    stats = OnlineTemporalSpectrum()
    x = torch.ones(1, 4, 4, 1, 1)
    assert not stats.update(0, x, frame_idx0=0, is_clean_pass=False, timestep=0)
    assert not stats._history
    assert not stats.update(0, x, frame_idx0=0, timestep=123)
    assert stats.update(0, x, frame_idx0=1, timestep=123)


def test_nonfinite_frames_reset_instead_of_fabricating_continuity():
    stats = OnlineTemporalSpectrum(window=4)
    x = torch.ones(1, 4, 4, 1, 1)
    stats.update(0, x, frame_idx0=0)
    bad = x.clone()
    bad[..., 0, 0] = float("nan")
    rows = stats.update(0, bad, frame_idx0=1)
    assert len(rows) == 1 and rows[0]["status"] == "nonfinite_input"
    assert not stats.update(0, x, frame_idx0=2)
    rows = stats.update(0, x, frame_idx0=3)
    assert all(r["window_start_frame"] == 2 for r in rows)
    json.dumps(rows, allow_nan=False)


def test_sampled_channels_are_consistent_and_inputs_are_unchanged():
    stats = OnlineTemporalSpectrum(channels=1)
    x = torch.ones(3, 4, 4, 1, 2)
    x[..., 1] = torch.arange(3)[:, None, None, None] * 100
    saved = x.clone()
    rows = stats.update(0, x)
    assert all(r["dc_ratio"] == pytest.approx(1) and r["sampled_channels"] == 1 for r in rows)
    assert torch.equal(saved, x)


def test_state_is_safe_npz_and_merge_does_not_join_history():
    stats = OnlineTemporalSpectrum(window=3, max_rows=3)
    emitted = stats.update(0, torch.ones(3, 4, 4, 1, 1))
    assert len(stats.rows) == 3 and stats.dropped_rows == len(emitted) - 3
    buffer = io.BytesIO()
    np.savez(buffer, **stats.state_dict())
    buffer.seek(0)
    restored = OnlineTemporalSpectrum(window=3)
    with np.load(buffer, allow_pickle=False) as data:
        restored.merge(data)
        assert json.loads(str(data["config_json"].item()))["window"] == 3
    assert restored.rows == stats.rows
    assert not restored._history
    assert not restored.update(0, torch.ones(1, 4, 4, 1, 1), frame_idx0=3)


def test_active_stream_bound_and_validation():
    stats = OnlineTemporalSpectrum(max_streams=2)
    for sample in range(5):
        stats.update(0, torch.ones(1, 4, 4, 1, 1), sample_id=sample)
    assert len(stats._history) == 2
    with pytest.raises(ValueError, match="at least 2"):
        OnlineTemporalSpectrum(window=1)
    with pytest.raises(ValueError, match="eps"):
        OnlineTemporalSpectrum(eps=float("nan"))
