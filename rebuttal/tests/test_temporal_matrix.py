"""Numerical controls for temporal-only channel/dense fitting and replay."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from rebuttal.adapters.experimental_codec import ResearchTemporal
from rebuttal.experiments.architecture.temporal import identity_predictor
from rebuttal.experiments.temporal_matrix.model import fit_round, fit, replay, parameter_bytes
from rebuttal.experiments.temporal_matrix.experiment import run
from tempokv.config import CodecConfig
from tempokv.parameters import TemporalParameters
from tempokv.fitting import _reconstruct_temporal_trace


def config():
    return CodecConfig(k_bits=4, v_bits=4, group_size=4, k_history_length=2, v_history_length=1,
                       v_predictor_mode="temporal", fit_mode="native", use_triton=False)


def test_dense_diagonal_special_case_and_production_replay_match():
    torch.manual_seed(3)
    target = torch.randn(6, 7, 2, 4)
    channel, _ = fit(target, config(), kind="channel_linear", history=2, bits=4)
    dense = ResearchTemporal("dense_linear", channel.sharing, torch.diag_embed(channel.weights),
                              channel.bias.clone(), 1, 2)
    a, metrics = replay(target, channel, config(), bits=4)
    b, _ = replay(target, dense, config(), bits=4)
    assert torch.equal(a, b)
    production = TemporalParameters(channel.weights.permute(1, 2, 0, 3)[None],
                                    channel.bias.permute(1, 0, 2)[None])
    reference = _reconstruct_temporal_trace(target[None], production, config())[0]
    assert torch.equal(reference, a)
    assert len(metrics) == target.shape[1]-1


def test_dense_temporal_learns_off_diagonal_history():
    g = torch.Generator().manual_seed(5)
    first = torch.randn(150, 1, 4, generator=g)
    rotation = torch.tensor([[0., .7, 0., 0.], [-.7, 0., 0., 0.], [0., 0., 0., .7], [0., 0., -.7, 0.]])
    second = first@rotation
    target = torch.stack([first, second], dim=1)
    models = {}
    for kind in ("channel_linear", "dense_linear"):
        initial = identity_predictor(1, 1, 4, 1, "layer_head", kind=kind)
        models[kind], _ = fit_round(target, target, initial, ridge=1e-6, max_gain=1)
    history = [target[:, 0].unsqueeze(-2)]
    errors = {kind: (model.predict(history, layer=0).squeeze(-2)-second).square().sum()
              for kind, model in models.items()}
    assert errors["dense_linear"] < errors["channel_linear"]*1e-5


def test_ridge_gain_warmup_and_storage_controls_are_matched():
    torch.manual_seed(7)
    target = torch.randn(12, 6, 2, 4)
    logs, models = {}, {}
    for kind in ("channel_linear", "dense_linear"):
        initial = identity_predictor(1, 2, 4, 2, "layer_head", kind=kind)
        model, log = fit_round(target, target, initial, ridge=1.0, max_gain=.8)
        models[kind], logs[kind] = model, log
        if kind == "channel_linear":
            gain = model.weights.abs().sum(dim=2)
        else:
            gain = model.weights.abs().sum(dim=(2, 3))
        assert gain.max() <= .800001
        assert torch.count_nonzero(model.weights[:, 0, 1]) == 0
    assert logs["channel_linear"] == logs["dense_linear"]
    assert [r["tokens_per_head"] for r in logs["dense_linear"]] == [12, 48]
    assert parameter_bytes(models["channel_linear"]) == (2*2*2*4+2*2*4)*4
    assert parameter_bytes(models["dense_linear"]) == (2*2*2*4*4+2*2*4)*4


def test_dense_contains_diagonal_under_same_unclipped_objective():
    torch.manual_seed(11)
    target = torch.randn(30, 6, 2, 4)
    logs, scores = {}, {}
    for kind in ("channel_linear", "dense_linear"):
        initial = identity_predictor(1, 2, 4, 2, "layer_head", kind=kind)
        model, logs[kind] = fit_round(target, target, initial, ridge=.1, max_gain=100)
        score = 0.
        for order in (1, 2):
            stop = order+1 if order < 2 else 6
            x = torch.stack([target[:, order-lag:stop-lag] for lag in range(1, order+1)], dim=-2)
            x = x.reshape(-1, 2, order, 4).permute(1, 0, 2, 3).double()
            y = target[:, order:stop].reshape(-1, 2, 4).permute(1, 0, 2).double()
            w = model.weights[:, order-1, :order].double()
            w = torch.diag_embed(w) if kind == "channel_linear" else w
            w = w.reshape(2, order*4, 4)
            prior = torch.zeros(order*4, 4, dtype=torch.float64); prior[:4] = torch.eye(4)
            prediction = x.flatten(2)@w+model.bias[:, order-1, None].double()
            penalty = torch.tensor(logs[kind][order-1]["ridge_penalty_per_head"], dtype=torch.float64)
            score += float((prediction-y).square().sum()+(penalty[:, None, None]*(w-prior).square()).sum())
        scores[kind] = score
    assert logs["channel_linear"] == logs["dense_linear"]
    assert scores["dense_linear"] <= scores["channel_linear"]+1e-5


def test_future_frames_do_not_change_prior_reconstruction():
    torch.manual_seed(4)
    target = torch.randn(8, 8, 2, 4)
    for kind in ("channel_linear", "dense_linear"):
        model, _ = fit(target, config(), kind=kind, history=2, bits=4)
        before, _ = replay(target, model, config(), bits=4)
        changed = target.clone(); changed[:, 5:] += 100
        after, _ = replay(changed, model, config(), bits=4)
        assert torch.equal(before[:, :5], after[:, :5])


def test_pipeline_uses_no_crosskv_and_rejects_mutated_artifact(tmp_path, monkeypatch):
    import tempokv.fitting as fitting
    def forbidden(*args, **kwargs):
        raise AssertionError("CrossKV must not be used in a temporal-only comparison")
    monkeypatch.setattr(fitting, "fit_cross_kv", forbidden)
    monkeypatch.setattr(fitting, "_reconstruct_cross_kv_trace", forbidden)
    g = torch.Generator().manual_seed(10)
    k = torch.randn(2, 12, 6, 2, 4, generator=g)
    v = torch.randn(k.shape, generator=g)
    trace = tmp_path/"synthetic.npz"
    np.savez(trace, keys=k.numpy(), values=v.numpy(), coordinate_system=np.asarray("pre_rope_normalized"))
    cfg = {"trace": str(trace), "holdout_step": 3, "ridge": 1., "fit_rounds": 2,
           "codec": {"k_bits": 4, "v_bits": 4, "group_size": 4, "k_history_length": 2,
                     "v_history_length": 1, "v_predictor_mode": "temporal", "fit_mode": "native", "use_triton": False}}
    out = tmp_path/"out"
    result = run(cfg, "pipeline", out)
    assert result["status"] == "complete" and result["cross_kv_enabled"] is False
    assert [row["name"] for row in result["results"]] == ["k_per_channel", "k_dense", "v_per_channel", "v_dense"]
    assert {row["history_length"] for row in result["results"]} == {1, 2}
    assert all(np.isfinite(row["reconstruction_nmse"]) for row in result["results"])
    path = out/"temporal_predictors.pt"
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        run(cfg, "replay", out)
