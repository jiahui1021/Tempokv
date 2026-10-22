"""Synthetic CPU validation of dense clean capture and bounded history."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from tempokv.research.capture import CleanBank, stable_seed
from tempokv.research.config import ResearchConfig
from tempokv.research.context import ProbeContext


def config(**updates):
    data = {"selection": {"layers": [0], "heads": [0], "chunk_indices": [0, 1, 2, 3], "denoise_step_indices": [0]},
            "sampling": {"spatial_block_hw": [4, 4], "protect_recent_latent_frames": 0,
                         "clean_blocks_per_stream": 5, "max_history_blocks_per_event": 5},
            "spectral": {"cutoffs": [0., .15, .4], "temporal_windows": [4, 8], "temporal_positions": 8}}
    for name, value in updates.items():
        data[name] = {**data.get(name, {}), **value} if isinstance(value, dict) else value
    return ResearchConfig.from_dict(data)


def context(positions, *, h=4, w=6, current_frame=8, sink_frames=0, recent_frames=0, sample=0, layer=0, head=0):
    start = current_frame * h * w
    return ProbeContext(0, 3, sample, layer, head, current_frame, "denoise", 0, 0, start, 1, h, w,
                        torch.arange(start, start + h * w), torch.tensor(positions, dtype=torch.int64), sink_frames, recent_frames)


def add(bank, frame, k, v, *, sample=0, layer=0, head=0, seed=3):
    return bank.add(sample=sample, layer=layer, head=head, frame=frame, k=k, v=v, prompt="synthetic", seed=seed)


def independent_spatial_ratio(value, cutoff):
    h, w, _ = value.shape
    power = torch.fft.fft2(value.float(), dim=(0, 1), norm="ortho").abs().square().sum(-1)
    mask = torch.tensor([[(min(y, h-y) / h) ** 2 + (min(x, w-x) / w) ** 2 <= cutoff ** 2
                          for x in range(w)] for y in range(h)])
    return float(power[mask].sum() / power.sum()), float(power.sum())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dense_30x52_stats_and_exact_full_block_reservoir(dtype):
    cfg = config()
    bank = CleanBank(cfg)
    h, w, d = 30, 52, 3
    spatial = torch.arange(h * w).reshape(h, w).float()
    k = torch.stack((spatial / 100, (spatial % 2) * 3, torch.sin(spatial / 10)), -1).to(dtype)
    # The eight temporal trace positions contain no K energy. Dense spatial
    # statistics must still see every other token in the actual 30x52 grid.
    trace = np.linspace(0, h * w - 1, 8).round().astype(int)
    k.reshape(h*w, d)[trace] = 0
    v = (k.float() * .7 + 2).to(dtype)
    original_k, original_v = k.clone(), v.clone()
    state_torch = torch.get_rng_state().clone(); state_numpy = np.random.get_state()
    records = add(bank, 0, k, v)
    assert torch.equal(state_torch, torch.get_rng_state())
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == state_numpy[0] and np.array_equal(actual_numpy[1], state_numpy[1]) and actual_numpy[2:] == state_numpy[2:]
    spectral_records = [row for row in records if row["score_type"] == "spatial_low_energy"]
    pooling_records = [row for row in records if row["score_type"] == "uniform_pooling_nmse"]
    assert len(spectral_records) == 2 * len(cfg.spectral.cutoffs)
    assert len(pooling_records) == 2 and all(row["score"] >= 0 for row in pooling_records)
    for row in spectral_records:
        source = original_k if row["role"] == "k" else original_v
        ratio, total = independent_spatial_ratio(source, row["cutoff"])
        assert row["score"] == pytest.approx(ratio, rel=3e-6, abs=2e-7)
        assert row["total_energy"] == pytest.approx(total, rel=3e-6)
        assert row["metric_scope"] == "clean_dense_spatial" and not row["near_zero_energy"]
        assert row["coordinate"] == ("pre_rope" if row["role"] == "k" else "raw")
    entries = list(bank.blocks[(0, 0, 0)].values())
    assert len(entries) == cfg.sampling.clean_blocks_per_stream
    assert any(entry.k_pre.shape[1] > 8 for entry in entries)
    for entry in entries:
        assert torch.equal(entry.k_pre, original_k.reshape(-1, d)[list(entry.spatial_indices)].reshape(1, -1, 1, d))
        assert torch.equal(entry.v, original_v.reshape(-1, d)[list(entry.spatial_indices)].reshape(1, -1, 1, d))
        assert entry.k_pre.dtype == entry.v.dtype == dtype
        assert entry.bytes == 2 * len(entry.spatial_indices) * d * original_k.element_size()
    assert bank.bytes == sum(entry.bytes for entry in entries)
    k.zero_(); v.zero_()
    assert all(torch.equal(entry.k_pre, original_k.reshape(-1, d)[list(entry.spatial_indices)].reshape(1, -1, 1, d)) for entry in entries)


def test_seeded_priority_reservoir_is_reproducible_and_bounded_per_stream():
    cfg = config(sampling={"spatial_block_hw": [2, 3], "clean_blocks_per_stream": 3})
    banks = [CleanBank(cfg), CleanBank(cfg)]
    tensor = torch.arange(4 * 6 * 2).reshape(4, 6, 2).float()
    state = torch.get_rng_state().clone()
    for bank in banks:
        for frame in range(10):
            add(bank, frame, tensor + frame, tensor - frame, seed=7)
        assert len(bank.blocks[(0, 0, 0)]) == 3
        assert len(bank.trajectories[(0, 0, 0, "k")]) == 8
    assert torch.equal(state, torch.get_rng_state())
    expected = sorted((stable_seed(7, "synthetic", 0, 0, 0, frame, block), (frame, block))
                      for frame in range(10) for block in range(4))[:3]
    for bank in banks:
        assert sorted(bank.blocks[(0, 0, 0)]) == sorted(key for _, key in expected)
    for key, entry in banks[0].blocks[(0, 0, 0)].items():
        other = banks[1].blocks[(0, 0, 0)][key]
        assert entry.priority == other.priority and torch.equal(entry.k_pre, other.k_pre) and torch.equal(entry.v, other.v)


def test_raw_capture_byte_limit_applies_across_streams():
    # A full 2x3 block with D=2 FP32 uses 96 bytes for K+V.
    budget_bytes = 2000
    cfg = config(sampling={"spatial_block_hw": [2, 3], "clean_blocks_per_stream": 100,
                           "max_capture_mb": budget_bytes / 1024 ** 2})
    bank = CleanBank(cfg)
    tensor = torch.ones(4, 6, 2)
    for sample in (0, 1):
        for frame in range(3):
            add(bank, frame, tensor + frame, tensor, sample=sample)
    entries = [entry for stream in bank.blocks.values() for entry in stream.values()]
    assert bank.bytes == sum(entry.bytes for entry in entries) <= budget_bytes
    trajectory_bytes = sum(value.numel() * value.element_size() for history in bank.trajectories.values() for _, value in history)
    assert bank.capture_bytes == bank.bytes + trajectory_bytes <= budget_bytes
    assert entries and bank.dropped > 0


def temporal_values(frame, h=2, w=3):
    positions = torch.arange(1., h * w + 1).reshape(h, w, 1)
    return torch.cos(torch.tensor(2 * math.pi * frame / 8)) * positions.expand(h, w, 2)


def independent_temporal_ratio(values, cutoff):
    power = torch.fft.fft(values.float(), dim=0, norm="ortho").abs().square().sum((1, 2, 3))
    n = len(values)
    mask = torch.tensor([min(i, n-i) / n <= cutoff for i in range(n)])
    return float(power[mask].sum() / power.sum())


def test_temporal_same_position_windows_and_seeded_shuffle_match_independent_fft():
    cfg = config(sampling={"spatial_block_hw": [2, 3]}, spectral={"temporal_positions": 4})
    bank = CleanBank(cfg)
    state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    records = []
    for frame in range(8):
        k = temporal_values(frame)
        records = add(bank, frame, k, k * 2, seed=17)
    assert torch.equal(state, torch.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_state[0] and np.array_equal(numpy_after[1], numpy_state[1]) and numpy_after[2:] == numpy_state[2:]
    spatial_indices = np.linspace(0, 5, 4).round().astype(int)
    trajectories = bank.trajectories[(0, 0, 0, "k")]
    assert [frame for frame, _ in trajectories] == list(range(8))
    for frame, values in trajectories:
        expected = temporal_values(frame).reshape(6, 2)[spatial_indices, None]
        assert torch.equal(values, expected)
    temporal = [row for row in records if row["score_type"] == "temporal_low_energy"]
    assert len(temporal) == 2 * 2 * 2 * 3
    for row in temporal:
        length = row["temporal_window"]
        series = torch.stack([temporal_values(frame).reshape(6, 2)[spatial_indices, None]
                              for frame in range(8-length, 8)])
        if row["role"] == "v": series = series * 2
        if row["temporal_control"] == "shuffled":
            order = np.random.default_rng(stable_seed(17, "synthetic", 0, 0, 0, row["role"], 7, length)).permutation(length)
            series = series[order]
        assert row["score"] == pytest.approx(independent_temporal_ratio(series, row["cutoff"]), rel=3e-6, abs=2e-7)
        assert row["spatial_indices"] == spatial_indices.tolist()
        assert row["metric_scope"] == "clean_contiguous_temporal"
    at_eight = {(row["role"], row["temporal_control"]): row["score"] for row in temporal
                if row["temporal_window"] == 8 and row["cutoff"] == .15}
    assert at_eight["k", "ordered"] > .99
    assert at_eight["k", "shuffled"] < .9


def test_temporal_gap_resets_and_never_crosses_missing_frames():
    bank = CleanBank(config(spectral={"temporal_windows": [4]}))
    for frame in (0, 1, 4, 5, 6):
        rows = add(bank, frame, temporal_values(frame), temporal_values(frame) * 2)
        assert not any(row["score_type"] == "temporal_low_energy" for row in rows)
    rows = add(bank, 7, temporal_values(7), temporal_values(7) * 2)
    assert any(row["score_type"] == "temporal_low_energy" for row in rows)
    assert [frame for frame, _ in bank.trajectories[(0, 0, 0, "k")]] == [4, 5, 6, 7]


def test_temporal_grid_geometry_change_resets_same_position_trajectory():
    bank = CleanBank(config(spectral={"temporal_windows": [4], "temporal_positions": 4}))
    for frame in range(3):
        add(bank, frame, temporal_values(frame, 2, 3), temporal_values(frame, 2, 3))
    rows = add(bank, 3, temporal_values(3, 3, 2), temporal_values(3, 3, 2))
    assert not any(row["score_type"] == "temporal_low_energy" for row in rows)
    assert [frame for frame, _ in bank.trajectories[(0, 0, 0, "k")]] == [3]


def test_candidate_absolute_mapping_after_sink_rolling_and_protection():
    cfg = config(sampling={"spatial_block_hw": [2, 3], "clean_blocks_per_stream": 100,
                           "max_history_blocks_per_event": 100})
    bank = CleanBank(cfg)
    tensor = torch.arange(4*6*2).reshape(4, 6, 2).float()
    for frame in range(9):
        add(bank, frame, tensor + 1000*frame, tensor - 1000*frame)
    # Sink frame 0 remains while history frames 1..4 have rolled out.
    positions = list(range(24)) + list(range(5*24, 9*24))
    ctx = context(positions, sink_frames=1, recent_frames=1)
    candidates = bank.candidates(ctx)
    assert {entry.frame for entry, _ in candidates} == {5, 6}
    lookup = {position: i for i, position in enumerate(positions)}
    for entry, indices in candidates:
        absolute = [entry.frame * 24 + spatial for spatial in entry.spatial_indices]
        assert indices == tuple(lookup[position] for position in absolute)
        assert all(bool(ctx.eligible_mask[index]) and not bool(ctx.protected_mask[index]) for index in indices)
        expected_pre = (tensor + entry.frame*1000).reshape(-1, 2)[list(entry.spatial_indices)].reshape(1, -1, 1, 2)
        assert torch.equal(entry.k_pre, expected_pre)
    assert bank.candidates(replace(ctx, sample_id=1)) == []
    assert bank.candidates(replace(ctx, layer_id=1)) == []
    assert bank.candidates(replace(ctx, head_id=1)) == []
    partial = replace(ctx, key_abs_positions=torch.tensor([position for position in positions if position != 5*24]))
    assert not any(entry.frame == 5 and 0 in entry.spatial_indices for entry, _ in bank.candidates(partial))


def test_empty_history_and_zero_energy_are_explicit_unmeasured_cases():
    bank = CleanBank(config(spectral={"temporal_windows": [2]}))
    tensor = torch.zeros(4, 6, 2)
    initial_rows = add(bank, 0, tensor, tensor)
    assert not any(row["score_type"] == "temporal_low_energy" for row in initial_rows)
    assert bank.candidates(context(list(range(24)), current_frame=0)) == []
    assert bank.candidates(context([], current_frame=1)) == []
    rows = add(bank, 1, tensor, tensor)
    spectra = [row for row in rows if row["score_type"] != "uniform_pooling_nmse"]
    assert spectra and all(row["near_zero_energy"] and math.isnan(row["score"]) for row in spectra)
    assert all(row["score"] == 0 for row in rows if row["score_type"] == "uniform_pooling_nmse")


@pytest.mark.parametrize("sampling", [{"protect_current_chunk": False}, {"protect_recent_latent_frames": -1},
                                       {"sink_policy": "drop"}, {"clean_blocks_per_stream": 0},
                                       {"max_capture_mb": float("inf")}, {"max_capture_mb": 0}])
def test_config_enforces_capture_bounds_and_protection(sampling):
    with pytest.raises(ValueError):
        config(sampling=sampling)


@pytest.mark.parametrize("spectral", [{"temporal_windows": []}, {"temporal_positions": 0},
                                       {"temporal_positions": -1}, {"temporal_windows": [1]},
                                       {"cutoffs": []}, {"cutoffs": [-.1]}, {"cutoffs": [float("nan")]},
                                       {"cutoffs": [.8]}])
def test_config_rejects_invalid_spectral_sampling(spectral):
    with pytest.raises(ValueError):
        config(spectral=spectral)


def test_config_round_trip_signature_and_max_clean_frame_limit():
    cfg = config(sampling={"max_clean_frames": 2})
    restored = ResearchConfig.from_dict(cfg.to_dict())
    assert restored == cfg and restored.signature == cfg.signature
    assert replace(cfg, output=replace(cfg.output, root="other-local-output")).signature == cfg.signature
    assert replace(cfg, sampling=replace(cfg.sampling, clean_blocks_per_stream=6)).signature != cfg.signature
    bank = CleanBank(cfg); tensor = torch.ones(4, 6, 2)
    add(bank, 1, tensor, tensor)
    snapshot_bytes, snapshot_keys = bank.bytes, tuple(bank.blocks[(0, 0, 0)])
    assert add(bank, 2, tensor, tensor) == []
    assert bank.bytes == snapshot_bytes and tuple(bank.blocks[(0, 0, 0)]) == snapshot_keys
