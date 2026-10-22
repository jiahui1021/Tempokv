"""Synthetic checks for suite B statistics; these are not research results."""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest
import torch

from tempokv.spectral_kv.stats import (
    ChunkAxisStats, DistanceDecay, KVRefreshDiff, OnlineSpectrum, fit_distance_decay,
)


def _noise(frames=3, h=8, w=8, heads=2, dim=3):
    return torch.randn(frames, h, w, heads, dim, generator=torch.Generator().manual_seed(411))


def _assert_plain_finite(rows):
    # Serializing with allow_nan=False also checks that no Tensor is retained.
    json.dumps(rows, allow_nan=False)


def test_spectrum_constant_distinguishes_dc_and_zero_ac():
    rows = OnlineSpectrum(bins=4).update(2, torch.ones(2, 8, 8, 2, 3), frame_idx0=7)
    assert len(rows) == 2 * 2 * 4
    for row in rows:
        assert row["valid"]
        assert row["dc_ratio"] == pytest.approx(1)
        assert row["r95_incl_dc"] == 0
        assert row["r95_excl_dc"] is None
        assert row["zero_non_dc"]
        assert not row["zero_signal"]
        assert row["power_excl_dc"] == pytest.approx(0)
    assert {row["frame_idx"] for row in rows} == {7, 8}
    _assert_plain_finite(rows)


def test_spectrum_known_sinusoid_radius_and_dc_removal():
    # A single horizontal sinusoid has radius (2/16)/sqrt(0.5).
    sinusoid = torch.cos(torch.arange(16) * (2 * math.pi * 2 / 16))
    x = sinusoid[None, None, :, None, None].expand(1, 16, 16, 1, 1) + 100
    rows = OnlineSpectrum(bins=8).update(0, x, role="v", coordinate="post", sample_id=9)
    expected = (2 / 16) / math.sqrt(0.5)
    assert rows[0]["r95_incl_dc"] == 0
    assert rows[0]["r95_excl_dc"] == pytest.approx(expected, abs=1e-6)
    assert rows[0]["r95_cycles"] == pytest.approx(2 / 16, abs=1e-6)
    assert sum(row["fraction_excl_dc"] for row in rows) == pytest.approx(1)
    assert sum(row["fraction_incl_dc"] for row in rows) == pytest.approx(1)
    assert rows[0]["role"] == "v" and rows[0]["coordinate"] == "post"


def test_spectrum_zero_and_nonfinite_are_explicit():
    x = torch.zeros(1, 8, 8, 2, 2)
    x[..., 1, 0] = float("nan")
    rows = OnlineSpectrum(bins=3).update(0, x)
    assert all(row["zero_signal"] and row["r95"] is None for row in rows if row["head"] == 0)
    assert all(not row["valid"] and row["status"] == "nonfinite_input"
               for row in rows if row["head"] == 1)
    _assert_plain_finite(rows)


def test_finite_input_overflow_does_not_escape_as_nan_measurement():
    rows = OnlineSpectrum(bins=2).update(0, torch.full((1, 8, 8, 1, 1), 1e35))
    assert all(not row["valid"] for row in rows)
    _assert_plain_finite(rows)


def test_row_retention_state_roundtrip_and_merge_without_pickle():
    spectrum = OnlineSpectrum(bins=2, max_rows=3)
    emitted = spectrum.update(0, _noise(frames=2, heads=1))
    assert len(emitted) == 4
    assert len(spectrum.rows) == 3 and spectrum.dropped_rows == 1
    buffer = io.BytesIO()
    np.savez(buffer, **spectrum.state_dict())
    buffer.seek(0)
    merged = OnlineSpectrum(bins=2, max_rows=10)
    with np.load(buffer, allow_pickle=False) as state:
        merged.merge(state)
    assert merged.rows == spectrum.rows
    assert merged.dropped_rows == 1
    merged.merge(spectrum)
    assert len(merged.rows) == 6 and merged.dropped_rows == 2
    with pytest.raises(TypeError):
        merged.merge(DistanceDecay())


def test_distance_decay_identical_frames_and_bounded_ring():
    feature = _noise(frames=1, h=16, w=16, heads=1)
    decay = DistanceDecay(max_lag=2, bins=4, channels=2)
    rows = decay.update(4, feature.expand(5, -1, -1, -1, -1), frame_idx0=20)
    assert {row["lag"] for row in rows} == {1, 2}
    assert all(row["correlation"] == pytest.approx(1, abs=1e-6) for row in rows)
    assert len(decay._history[(0, 4, "k", "pre")]) == 2
    assert all(row["frame_idx"] - row["previous_frame_idx"] == row["lag"] for row in rows)
    _assert_plain_finite(rows)


def test_distance_does_not_compare_across_gaps_samples_or_coordinates():
    x = _noise(frames=1, heads=1)
    decay = DistanceDecay(max_lag=4, bins=2)
    assert not decay.update(0, x, frame_idx0=0)
    assert not decay.update(0, x, frame_idx0=3)
    rows = decay.update(0, x, frame_idx0=4)
    assert rows and {r["previous_frame_idx"] for r in rows} == {3}
    assert not decay.update(0, x, frame_idx0=5, sample_id=1)
    assert not decay.update(0, x, frame_idx0=5, coordinate="post")
    assert not decay.update(0, _noise(frames=1, h=10, heads=1), frame_idx0=5)


def test_distance_active_stream_limit_and_nan_rows():
    decay = DistanceDecay(max_lag=2, bins=2, max_streams=2)
    for sample in range(5):
        decay.update(0, _noise(frames=1), sample_id=sample)
    assert len(decay._history) == 2
    x = _noise(frames=2, heads=1)
    x[1, 0, 0, 0, 0] = float("nan")
    rows = decay.update(0, x, sample_id=8)
    assert rows and all(not r["valid"] for r in rows)
    _assert_plain_finite(rows)


def _synthetic_radius_rows():
    # Exact, constructed radii 1, 1/2, 1/4 at lags 1, 2, 4 give a=1,b=1.
    rows = []
    for lag, stable in ((1, 8), (2, 4), (4, 2)):
        for band in range(8):
            rows.append(dict(sample_id=0, layer=1, head=2, role="k", coordinate="pre",
                             lag=lag, band=band, bins=8, radius_high=(band + 1) / 8,
                             correlation=0.99 if band < stable else 0.2, valid=True))
    return rows


def test_distance_known_synthetic_radius_and_powerlaw():
    radii, fits = fit_distance_decay(_synthetic_radius_rows(), threshold=0.9)
    assert [r["rstar"] for r in radii] == [1, 0.5, 0.25]
    assert fits[0]["valid"]
    assert fits[0]["a"] == pytest.approx(1)
    assert fits[0]["b"] == pytest.approx(1)
    assert fits[0]["r2"] == pytest.approx(1)


def test_distance_radius_must_be_contiguous_from_low_and_missing_is_unmeasured():
    rows = _synthetic_radius_rows()
    # A high-frequency island cannot restore the stable radius.
    next(r for r in rows if r["lag"] == 2 and r["band"] == 7)["correlation"] = 1
    radii, _ = fit_distance_decay(rows)
    assert next(r for r in radii if r["lag"] == 2)["rstar"] == 0.5
    missing = [r for r in rows if not (r["lag"] == 1 and r["band"] == 0)]
    radii, _ = fit_distance_decay(missing)
    assert radii[0]["rstar"] is None and not radii[0]["valid"]
    zero_low = [dict(r, correlation=0.1) for r in rows if r["lag"] == 1]
    radii, fits = fit_distance_decay(zero_low)
    assert radii[0]["rstar"] == 0
    assert fits[0]["a"] is None and not fits[0]["valid"]


def test_chunk_axis_same_step_alignment_and_linear_extrapolation():
    x = _noise(frames=4, h=16, w=16, heads=1)
    stats = ChunkAxisStats(spatial_bins=3, temporal_bins=3)
    warm = stats.update(2, x, chunk_idx=0, step_idx=3)
    assert all(row["reuse_rel_l2"] is None for row in warm)
    # An unrelated denoising step cannot become the previous chunk for step 3.
    stats.update(2, x * 100, chunk_idx=1, step_idx=2)
    second = stats.update(2, x * 2, chunk_idx=1, step_idx=3)
    nonempty = [row for row in second if row["frequency_count"]]
    assert all(row["cosine"] == pytest.approx(1, abs=1e-6) for row in nonempty)
    assert all(row["reuse_rel_l2"] == pytest.approx(0.5, abs=1e-6) for row in nonempty)
    assert all(row["linear_rel_l2"] is None for row in second)
    third = stats.update(2, x * 3, chunk_idx=2, step_idx=3)
    assert all(row["linear_rel_l2"] == pytest.approx(0, abs=2e-6)
               for row in third if row["frequency_count"])
    assert len(stats._history[(0, 2, 3, "feature", "pre")]) == 2
    _assert_plain_finite(third)


def test_chunk_axis_gap_and_shape_changes_reset_comparisons():
    stats = ChunkAxisStats()
    x = _noise(frames=3, heads=1)
    stats.update(0, x, chunk_idx=0, step_idx=0)
    gap = stats.update(0, x, chunk_idx=4, step_idx=0)
    assert all(row["reuse_rel_l2"] is None for row in gap)
    assert any(row["status"] == "nonadjacent_chunk" for row in gap)
    shape = stats.update(0, x[:2], chunk_idx=5, step_idx=0)
    assert all(row["linear_rel_l2"] is None and row["reuse_rel_l2"] is None for row in shape)
    assert any(row["status"] == "shape_change" for row in shape)


def test_chunk_axis_zero_and_nonfinite_do_not_emit_nan():
    stats = ChunkAxisStats(spatial_bins=2, temporal_bins=2)
    zero = torch.zeros(2, 8, 8, 1, 2)
    stats.update(0, zero, chunk_idx=0, step_idx=1)
    rows = stats.update(0, zero, chunk_idx=1, step_idx=1)
    assert all(row["cosine"] is None for row in rows)
    bad = zero.clone()
    bad[0, 0, 0, 0, 0] = float("nan")
    rows = stats.update(0, bad, chunk_idx=2, step_idx=1)
    assert all(not row["valid"] for row in rows)
    _assert_plain_finite(rows)


def test_refresh_explicit_stash_exact_inputs_and_pop():
    stats = KVRefreshDiff(bins=3)
    k, v = _noise(frames=2), _noise(frames=2) * 3
    missing = stats.update(4, k, v, frame_idx0=10)
    assert all(row["status"] == "missing_stash" and row["relative_error"] is None for row in missing)
    stats.stash(4, k, v, frame_idx0=10)
    rows = stats.update(4, k, v, frame_idx0=10)
    assert all(row["relative_error"] == pytest.approx(0) for row in rows if row["frequency_count"])
    assert not stats._stashes
    missing_again = stats.update(4, k, v, frame_idx0=10)
    assert all(not row["valid"] for row in missing_again)
    _assert_plain_finite(rows)


def test_refresh_copies_inputs_and_measures_role_and_head_separately():
    stats = KVRefreshDiff(bins=2)
    k, v = _noise(frames=1), _noise(frames=1)
    clean_k, clean_v = k.clone() * 2, v.clone()
    stats.stash(0, k, v, frame_idx0=0)
    k.zero_()  # Stash must not alias a mutable attention/cache tensor.
    rows = stats.update(0, clean_k, clean_v, frame_idx0=0)
    assert all(row["rel_l2"] == pytest.approx(0.5, abs=1e-6)
               for row in rows if row["role"] == "k" and row["frequency_count"])
    assert all(row["rel_l2"] == pytest.approx(0, abs=1e-6)
               for row in rows if row["role"] == "v" and row["frequency_count"])


def test_refresh_mismatch_missing_nonfinite_zero_and_stash_bound():
    stats = KVRefreshDiff(bins=2, roles=("k",), max_stashes=2)
    zero = torch.zeros(1, 8, 8, 1, 2)
    for sample_id in range(4):
        stats.stash(0, zero, zero, frame_idx0=3, sample_id=sample_id)
    assert len(stats._stashes) == 2
    mismatch = stats.update(0, zero, zero, frame_idx0=4, sample_id=3)
    assert all(row["status"] == "stash_mismatch" for row in mismatch)
    assert (3, 0) not in stats._stashes
    stats.stash(0, zero, zero, frame_idx0=0)
    rows = stats.update(0, zero, zero, frame_idx0=0)
    assert all(row["relative_error"] == 0 and row["zero_clean"] for row in rows)
    bad = zero.clone()
    bad[0, 0, 0, 0, 0] = float("nan")
    stats.stash(0, zero, zero, frame_idx0=0)
    rows = stats.update(0, bad, zero, frame_idx0=0)
    assert all(not row["valid"] for row in rows)
    _assert_plain_finite(rows)


@pytest.mark.parametrize("factory", [OnlineSpectrum, DistanceDecay, ChunkAxisStats, KVRefreshDiff])
def test_bad_dimensions_are_rejected(factory):
    obj = factory()
    with pytest.raises(ValueError, match="shape"):
        if isinstance(obj, KVRefreshDiff):
            obj.update(0, torch.zeros(2, 2), torch.zeros(2, 2), frame_idx0=0)
        elif isinstance(obj, ChunkAxisStats):
            obj.update(0, torch.zeros(2, 2), chunk_idx=0, step_idx=0)
        else:
            obj.update(0, torch.zeros(2, 2))


@pytest.mark.parametrize("factory", [OnlineSpectrum, DistanceDecay, ChunkAxisStats, KVRefreshDiff])
def test_epsilon_must_be_finite_and_positive(factory):
    with pytest.raises(ValueError, match="eps"):
        factory(eps=float("nan"))
    with pytest.raises(ValueError, match="eps"):
        factory(eps=-1)


def test_statistics_do_not_modify_inputs():
    x = _noise(frames=3)
    saved = x.clone()
    OnlineSpectrum().update(0, x)
    DistanceDecay(max_lag=2).update(0, x)
    ChunkAxisStats().update(0, x, chunk_idx=0, step_idx=0)
    refresh = KVRefreshDiff()
    refresh.stash(0, x, x, frame_idx0=0)
    refresh.update(0, x, x, frame_idx0=0)
    assert torch.equal(x, saved)
