"""Suite B frame-policy contracts and protected-frame behavior."""

import pytest

from tempokv.spectral_kv.policy import (
    DistanceSpectralPolicy, ExternalSelectorPolicy, FrameMeta, FullPolicy,
    HistoryPlan, Mode, OracleTopKPolicy, WindowSinkPolicy, target_grid,
)


def _meta(count=12):
    return [FrameMeta(i, (9, 15), (9, 15), is_current=i >= count - 2, is_sink=i == 0) for i in range(count)]


def test_full_policy_retains_every_frame_and_is_order_independent():
    meta = _meta()
    expected = {frame.frame_idx: (Mode.FULL, None) for frame in meta}
    assert FullPolicy().plan(0, 3, meta).entries == expected
    assert FullPolicy().plan(7, 3, list(reversed(meta))).entries == expected
    assert FullPolicy().plan(0, 0, []).entries == {}


def test_window_sink_retains_latest_and_explicit_protected_frames():
    meta = _meta()
    result = WindowSinkPolicy(3, sink=2).plan(0, 4, meta).entries
    assert {idx for idx, (mode, _) in result.items() if mode == Mode.FULL} == {0, 1, 7, 8, 9, 10, 11}
    result = WindowSinkPolicy(0, sink=0).plan(0, 4, meta).entries
    assert {idx for idx, (mode, _) in result.items() if mode == Mode.FULL} == {0, 10, 11}


def test_window_uses_latest_existing_frames_even_with_gaps():
    meta = [FrameMeta(i, (3, 5), (3, 5)) for i in (2, 5, 12, 19)]
    result = WindowSinkPolicy(2, sink=0).plan(0, 0, meta).entries
    assert result[2][0] == result[5][0] == Mode.DROP
    assert result[12][0] == result[19][0] == Mode.FULL


def test_oracle_topk_is_additional_to_protected_and_ties_prefer_newer():
    meta = _meta()
    scores = {i: 1.0 for i in range(12)}
    scores[3], scores[5] = 8, 8
    policy = OracleTopKPolicy(2, others="drop", recent=3, sink=2)
    result = policy.plan(1, 3, meta, scores).entries
    kept = {idx for idx, (mode, _) in result.items() if mode == Mode.FULL}
    assert kept == {0, 1, 3, 5, 7, 8, 9, 10, 11}
    tied = OracleTopKPolicy(1, others="drop", recent=0, sink=0).plan(0, 1, meta, {i: 1. for i in range(12)})
    assert tied.entries[9][0] == Mode.FULL
    assert tied.entries[8][0] == Mode.DROP


def test_oracle_summary_ratio_and_original_coordinate_target():
    meta = [FrameMeta(2, (2, 3), (9, 15)), FrameMeta(9, (9, 15), (9, 15), is_current=True)]
    result = OracleTopKPolicy(0, ratio=4, recent=0, sink=0).plan(0, 3, meta).entries
    assert result[2] == (Mode.SUMMARY, (4, 8))
    assert result[9] == (Mode.FULL, None)
    assert OracleTopKPolicy(0, ratio=1, recent=0, sink=0).plan(0, 0, _meta()).entries[5] == (Mode.FULL, None)


@pytest.mark.parametrize("scores", [None, {}, {i: float("nan") for i in range(12)}, {i: True for i in range(12)}])
def test_oracle_requires_finite_candidate_scores(scores):
    with pytest.raises(ValueError, match="q_probe"):
        OracleTopKPolicy(2, recent=0).plan(0, 0, _meta(), scores)


def test_distance_is_monotonic_and_preserves_recent_current_and_sink():
    meta = _meta(20)
    result = DistanceSpectralPolicy(0.7, recent=3, sink=2).plan(0, 7, meta).entries
    for idx in (0, 1, 15, 16, 17, 18, 19):
        assert result[idx] == (Mode.FULL, None)
    grids = [result[i][1] for i in range(2, 15)]
    assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(grids, grids[1:]))
    assert all(1 <= grid[0] <= 9 and 1 <= grid[1] <= 15 for grid in grids)
    assert DistanceSpectralPolicy(0, recent=0, sink=0).plan(0, 1, meta).entries[4] == (Mode.FULL, None)


def test_distance_target_uses_ceil_and_original_grid():
    meta = [FrameMeta(2, (1, 2), (9, 15)), FrameMeta(6, (9, 15), (9, 15), is_current=True)]
    plan = DistanceSpectralPolicy(1, r0=1, r_min=0.125, recent=0, sink=0).plan(0, 4, meta)
    assert plan.entries[2] == (Mode.SUMMARY, (3, 4))


def test_external_selector_gets_context_and_returns_validated_plan():
    meta = _meta(4)
    received = []
    def selector(layer, chunk_idx, frames, q_probe):
        received.append((layer, chunk_idx, frames, q_probe))
        return HistoryPlan({f.frame_idx: (Mode.FULL, None) if f.is_sink or f.is_current else (Mode.SUMMARY, (2, 3)) for f in frames})
    result = ExternalSelectorPolicy(selector).plan(5, 7, meta, {1: 0.5})
    assert result.entries[1] == (Mode.SUMMARY, (2, 3))
    assert received == [(5, 7, meta, {1: 0.5})]


@pytest.mark.parametrize("entry", [(Mode.DROP, None), (Mode.SUMMARY, (1, 1))])
def test_external_selector_cannot_remove_protected_frames(entry):
    meta = _meta(4)
    def selector(*args):
        return {f.frame_idx: entry for f in meta}
    with pytest.raises(ValueError, match="protected"):
        ExternalSelectorPolicy(selector).plan(0, 0, meta)


@pytest.mark.parametrize("entry", [(7, None), (True, None), (Mode.SUMMARY, None), (Mode.SUMMARY, (10, 15)),
                                   (Mode.SUMMARY, (0, 3)), (Mode.FULL, (2, 3)), (Mode.DROP, (2, 3))])
def test_external_invalid_modes_and_grids_fail(entry):
    meta = [FrameMeta(5, (3, 5), (9, 15))]
    with pytest.raises(ValueError):
        ExternalSelectorPolicy(lambda *args: {5: entry}).plan(0, 0, meta)


def test_external_allows_original_based_resolution_and_requires_complete_mapping():
    meta = [FrameMeta(5, (1, 2), (9, 15))]
    assert ExternalSelectorPolicy(lambda *args: {5: (1, (3, 5))}).plan(0, 0, meta).entries[5] == (Mode.SUMMARY, (3, 5))
    for result in ({}, {5: (Mode.FULL, None), 6: (Mode.DROP, None)}, None):
        with pytest.raises(ValueError, match="every frame"):
            ExternalSelectorPolicy(lambda *args: result).plan(0, 0, meta)


@pytest.mark.parametrize("hw,ratio,expected", [((9, 15), 4, (4, 8)), ((3, 5), 16, (1, 1)), ((3, 5), 1, (3, 5)), ((1, 2), 64, (1, 1))])
def test_target_grid_ratio_rounding(hw, ratio, expected):
    assert target_grid(hw, ratio) == expected


@pytest.mark.parametrize("factory", [lambda: WindowSinkPolicy(-1), lambda: OracleTopKPolicy(-1),
    lambda: OracleTopKPolicy(1, others="unknown"), lambda: OracleTopKPolicy(1, ratio=0),
    lambda: DistanceSpectralPolicy(-1), lambda: DistanceSpectralPolicy(float("nan")),
    lambda: DistanceSpectralPolicy(1, r_min=0), lambda: DistanceSpectralPolicy(1, r_min=1.1),
    lambda: DistanceSpectralPolicy(1, kind="fft"), lambda: ExternalSelectorPolicy(None),
    lambda: target_grid((3, 5), True), lambda: FrameMeta(-1, (3, 5), (3, 5)),
    lambda: FrameMeta(1, (4, 5), (3, 5)), lambda: FrameMeta(1, (3.5, 5), (3, 5))])
def test_invalid_policy_arguments_fail(factory):
    with pytest.raises(ValueError):
        factory()


def test_duplicate_metadata_fails():
    with pytest.raises(ValueError, match="duplicate"):
        FullPolicy().plan(0, 0, [_meta()[0], _meta()[0]])
