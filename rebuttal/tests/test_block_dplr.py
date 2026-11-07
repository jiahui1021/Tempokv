"""Numerical and real-codec checks for the CORAL-style CrossKV matrix study."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from rebuttal.experiments.block_dplr.model import BlockDPLR, RegressionStatistics, replay_layer
from rebuttal.experiments.block_dplr.experiment import run, load_trace
from tempokv.config import CodecConfig
from tempokv.parameters import CrossKVParameters
from tempokv.fitting import _reconstruct_cross_kv_trace


def test_global_factors_mix_heads_and_match_dense_formula():
    g = torch.Generator().manual_seed(4)
    block = torch.randn(3, 4, 4, generator=g)
    a, b = torch.randn(12, 2, generator=g), torch.randn(2, 12, generator=g)
    model = BlockDPLR(block, a, b)
    x = torch.randn(10, 3, 4, generator=g)
    dense = torch.block_diag(*block)+a@b
    torch.testing.assert_close(model(x).flatten(1), x.flatten(1)@dense, rtol=1e-5, atol=2e-6)
    assert model.storage_bytes() == (3*4*4+2*12*2)*4
    a.zero_(); b.zero_(); a[0, 0] = 1; b[0, 5] = 2
    model = BlockDPLR(torch.zeros_like(block), a, b)
    probe = torch.zeros(1, 3, 4); probe[0, 0, 0] = 3
    assert model(probe)[0, 1, 1] == 6


def test_alternating_fit_learns_cross_head_signal_and_decreases_objective():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(200, 2, 2, generator=g)
    y = torch.zeros_like(x); y[:, 1, 0] = 2*x[:, 0, 0]
    stats = RegressionStatistics(x, y, ridge=1e-6)
    block, _ = stats.fit(0)
    model, log = stats.fit(1, rounds=3)
    assert (model(x)-y).square().sum() < (block(x)-y).square().sum()*1e-4
    objectives = [row["objective"] for row in log]
    assert all(b <= a+1e-8 for a, b in zip(objectives, objectives[1:]))


def test_statistics_objective_matches_direct_penalized_regression():
    torch.manual_seed(8)
    x, y = torch.randn(50, 2, 3), torch.randn(50, 2, 3)
    stats = RegressionStatistics(x, y)
    d = torch.randn(2, 3, 3, dtype=torch.float64)
    a, b = torch.randn(6, 2, dtype=torch.float64), torch.randn(2, 6, dtype=torch.float64)
    w = torch.block_diag(*d)+a@b
    expected = (x.double().flatten(1)@w-y.double().flatten(1)).square().sum()
    expected += (stats.block_penalty[:, None, None]*d.square()).sum()+stats.penalty*(a@b).square().sum()
    assert stats.objective(d, a, b) == pytest.approx(expected.item(), rel=1e-12)


def test_rank_zero_replay_is_bit_exact_with_production_dense():
    torch.manual_seed(2)
    keys, values = torch.randn(1, 4, 6, 2, 4), torch.randn(1, 4, 6, 2, 4)
    weights = torch.randn(1, 2, 4, 4)*.1
    bias, history = torch.randn(1, 2, 4)*.1, torch.full((1, 1, 2, 4), .5)
    codec = CodecConfig(k_bits=4, v_bits=4, group_size=4, fit_mode="native", v_history_length=1,
                        scale_dtype="bfloat16", reconstruction_dtype="bfloat16", use_triton=False)
    params = CrossKVParameters(weights, bias, history_weights=history)
    expected = _reconstruct_cross_kv_trace(keys, values, params, codec)[0]
    model = BlockDPLR(weights[0], torch.empty(8, 0), torch.empty(0, 8))
    metrics, rebuilt = replay_layer(keys[0], values[0], model, bias[0], history[0], codec, return_reconstruction=True)
    assert torch.equal(expected, rebuilt)
    assert metrics["error_sse"] == pytest.approx((values[0, :, 1:].double()-expected[:, 1:].double()).square().sum().item(), rel=1e-12)


def small_config(tmp_path):
    g = torch.Generator().manual_seed(5)
    k = torch.randn(2, 12, 6, 2, 4, generator=g)
    v = .5*k+.2*torch.randn(k.shape, generator=g)
    path = tmp_path/"synthetic.npz"
    np.savez(path, keys=k.numpy(), values=v.numpy(), coordinate_system=np.asarray("pre_rope_normalized"))
    return {"trace": str(path), "ranks": [1, 2], "rounds": 3, "ridge": .001, "holdout_step": 3,
            "codec": {"k_bits": 4, "v_bits": 4, "anchor_bits": 4, "group_size": 4, "fit_mode": "native",
                      "k_history_length": 2, "v_history_length": 1, "cross_kv_weight_mode": "layer_head",
                      "scale_dtype": "bfloat16", "reconstruction_dtype": "bfloat16", "use_triton": False}}


def test_pipeline_freezes_artifacts_and_replays_factors(tmp_path):
    cfg = small_config(tmp_path)
    result = run(cfg, "pipeline", tmp_path/"run")
    assert result["status"] == "complete"
    assert [row["name"] for row in result["results"]] == ["dense_source", "block_diagonal_matched_ridge", "block_dplr_r1", "block_dplr_r2"]
    assert set(result["source"]["calibration_record_ids"]).isdisjoint(result["source"]["evaluation_record_ids"])
    baseline, _, rank1, rank2 = result["results"]
    assert rank1["matrix_bytes"] > baseline["matrix_bytes"]
    assert rank2["matrix_bytes"] > rank1["matrix_bytes"]
    assert all(np.isfinite(r["v_reconstruction_nmse"]) for r in result["results"])
    path = tmp_path/"run/block_dplr_predictors.pt"
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        run(cfg, "replay", tmp_path/"run")


def test_trace_hash_cannot_relabel_previous_experiment(tmp_path):
    cfg = small_config(tmp_path)
    cfg["expected_trace_sha256"] = "different"
    with pytest.raises(ValueError, match="expected SHA256"):
        load_trace(cfg)


def test_invalid_rank_and_geometry():
    x = torch.randn(20, 2, 3)
    stats = RegressionStatistics(x, x)
    with pytest.raises(ValueError):
        stats.fit(7)
    with pytest.raises(ValueError):
        BlockDPLR(torch.zeros(2, 3, 3), torch.zeros(3, 1), torch.zeros(1, 3))
