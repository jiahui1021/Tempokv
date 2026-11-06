from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from tempokv.codec import TempoKVCodec
from tempokv.config import CodecConfig
from tempokv.fitting import fit_predictors
from rebuttal.adapters.experimental_codec import (
    ExperimentalCodec, ResearchTemporal, ResearchCross, encode_tensor,
)
from rebuttal.experiments.architecture.temporal import identity_predictor, fit_temporal_architecture
from rebuttal.experiments.architecture.cross_low_rank import fit_low_rank_regression
from rebuttal.experiments.architecture.fit import run, replay_codec

@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)

def trace(seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(2, 3, 9, 2, 64, generator=g)
    for t in range(1, 9):
        keys[:, :, t] = .7 * keys[:, :, t-1] + .3 * keys[:, :, t]
    values = .8 * keys + .1 * torch.randn(keys.shape, generator=g)
    return keys, values


def config(stride=1):
    return CodecConfig(k_bits=2, v_bits=2, group_size=64, stride=stride, k_history_length=2,
                       v_history_length=1, fit_mode="legacy", use_triton=False)


def baseline():
    keys, values = trace()
    cfg = config()
    kp, vp = fit_predictors(keys, values, config=cfg, temporal_ridge=1.0, cross_ridge=.1,
                            v_history_mode="diagonal")
    k = ResearchTemporal.from_production(kp, 2, 2)
    v = ResearchCross.from_production(vp, cfg.cross_kv_weight_mode, 2, 2)
    return cfg, kp, vp, ExperimentalCodec(k, v, cfg)


def test_reference_reproduces_shipped_channel_dense_codec():
    cfg, kp, vp, research = baseline()
    production = TempoKVCodec(kp, vp, cfg)
    keys, values = trace(10)
    for layer in range(2):
        inputs = [x[layer].permute(0, 2, 1, 3) for x in (keys, values)]
        expected = production.decode(production.encode(*inputs, layer=layer))
        packed = research.encode(*inputs, layer=layer)
        actual = research.decode(packed)
        for a, b in zip(expected, actual):
            assert torch.equal(a, b)
        assert packed.key.storage_bytes() + packed.value.storage_bytes() > 0


def test_dense_temporal_diagonal_special_case():
    cfg, _, _, codec = baseline()
    channel = codec.temporal
    dense = ResearchTemporal("dense_linear", channel.sharing, torch.diag_embed(channel.weights),
                             channel.bias.clone(), channel.layers, channel.heads)
    inputs = trace(3)[0][0].permute(0, 2, 1, 3)
    _, a = encode_tensor(inputs, channel, cfg, bits=2)
    packed, b = encode_tensor(inputs, dense, cfg, bits=2)
    assert torch.equal(a, b)
    assert torch.equal(b, packed.decode())


def test_factorization_uses_calibration_covariance_and_refines_svd():
    torch.manual_seed(2)
    x = torch.randn(500, 8, dtype=torch.float64) * torch.arange(1, 9, dtype=torch.float64)
    y = x @ torch.randn(8, 8, dtype=torch.float64)
    a, b, history = fit_low_rank_regression(x.T @ x, x.T @ y, rank=2, ridge=0, steps=12)
    assert a.shape == (8, 2) and b.shape == (2, 8)
    assert history[-1]["objective_without_target_constant"] <= history[0]["objective_without_target_constant"] + 1e-7
    # Full rank can recover the unconstrained map.
    a, b, _ = fit_low_rank_regression(x.T @ x, x.T @ y, rank=8, ridge=0, steps=2)
    assert torch.allclose((x @ (a.double() @ b.double())), y, atol=2e-4, rtol=2e-5)


def test_scalar_warmup_bias_and_causal_history():
    keys, _ = trace()
    cfg = config()
    initial = identity_predictor(2, 2, 64, 2, "shared")
    history = [torch.randn(3, 2, 1, 64), torch.randn(3, 2, 1, 64)]
    assert torch.equal(initial.predict(history, layer=0), history[0])
    for kind in ("scalar_linear", "dense_linear"):
        initial = identity_predictor(2, 2, 64, 2, "shared", kind=kind)
        fitted, log = fit_temporal_architecture(keys, initial, cfg, rounds=2)
        assert len(log) == 2 and fitted.history_length == 2
        if kind == "dense_linear":
            assert float(fitted.weights.abs().sum(dim=(-3, -2)).max()) <= 1.000001
        packed, rebuilt = encode_tensor(keys[0].permute(0, 2, 1, 3), fitted, cfg, bits=2)
        assert torch.equal(packed.decode(), rebuilt)
        assert torch.isfinite(rebuilt).all()


def test_research_archive_and_output_buffer_roundtrip(tmp_path):
    _, _, _, codec = baseline()
    keys, values = trace(5)
    codec.metadata = {"source": "unit test", "calibration": {"sha256": "abc"}}
    path = tmp_path / "params.npz"; codec.save(path)
    loaded = ExperimentalCodec.load(path)
    assert loaded.parameter_bytes() == codec.parameter_bytes()
    assert loaded.metadata == codec.metadata
    packed = loaded.encode(keys[0].permute(0, 2, 1, 3), values[0].permute(0, 2, 1, 3))
    expected_k, _ = loaded.decode(packed)
    buffer = torch.empty(expected_k.shape[0], expected_k.shape[2], expected_k.shape[1], expected_k.shape[3], dtype=torch.bfloat16)
    decoded = packed.key.decode(output_tensor=buffer, output_layout="BSHD")
    assert decoded.data_ptr() == buffer.data_ptr() and torch.equal(decoded, expected_k)


@pytest.mark.parametrize("axis,variants", [("temporal", ["identity", "scalar_linear", "channel_linear", "dense_linear"]),
                                            ("cross", ["diagonal", "low_rank", "dense"])])
def test_full_fit_replay_preserves_independent_axis_and_split(tmp_path, axis, variants):
    calk, calv = trace(0); evk, evv = trace(9)
    cal = tmp_path / "cal.npz"; evaluation = tmp_path / "test.npz"
    np.savez(cal, keys=calk.numpy(), values=calv.numpy(), prompt_ids=np.asarray(["cal-a", "cal-b", "cal-c"]))
    np.savez(evaluation, keys=evk.numpy(), values=evv.numpy(), prompt_ids=np.asarray(["test-a", "test-b", "test-c"]))
    cp = tmp_path / "codec.json"
    from dataclasses import asdict
    cp.write_text(json.dumps(asdict(config())))
    settings = {"axis": axis, "variants": variants, "calibration_trace": str(cal), "evaluation_trace": str(evaluation),
                "codec_config": str(cp), "ranks": [2], "als_steps": 3, "require_prompt_identity": True}
    out = tmp_path / "out"
    fit_report = run(settings, "fit", out)
    report = run(settings, "replay", out)
    assert len(report["variants"]) == len(variants)
    models = [ExperimentalCodec.load(x["parameters"]) for x in fit_report["variants"]]
    if axis == "cross":
        for item in models[1:]:
            assert torch.equal(item.temporal.weights, models[0].temporal.weights)
            assert torch.equal(item.v_parameters.history_weights, models[0].v_parameters.history_weights)
        lowrank = next(x for x in models if x.v_parameters.kind == "low_rank")
        assert lowrank.v_parameters.weight is None
        assert lowrank.v_parameters.factor_a.shape[-1] == 2
    for row in report["variants"]:
        assert row["split_validation"]["status"] == "verified_prompt_disjoint"
        assert row["k"]["prediction_nmse"] is not None and row["v"]["quantization_nmse"] is not None
    with pytest.raises(ValueError, match="identical"):
        run({**settings, "evaluation_trace": str(cal)}, "replay", out)


def test_generation_plan_keeps_reference_control_without_model_downloads(tmp_path):
    report = run({"axis": "cross", "variants": ["low_rank"], "ranks": [16]}, "plan", tmp_path)
    assert "dense" in report["variants"]
    assert report["generation"]["status"] == "needs_inputs"
    assert any("checkpoint" in x for x in report["generation"]["missing"])
    assert (tmp_path / "generation_plan.json").is_file()


def test_generation_builds_custom_worker_job_and_forwards_archive(tmp_path):
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"not-loaded")
    model = tmp_path / "model_root"; model.mkdir()
    prompts = tmp_path / "prompts.txt"; prompts.write_text("test prompt\n")
    archive = tmp_path / "architecture.npz"; archive.write_bytes(b"not-loaded-in-plan")
    rows = [{"variant": name, "parameters": str(archive)} for name in ("low_rank_r16", "dense")]
    (tmp_path / "fit.json").write_text(json.dumps({"axis": "cross", "variants": rows}))
    result = run({"axis": "cross", "checkpoint": str(checkpoint), "model_root": str(model),
                  "prompts": str(prompts), "frames": 717}, "plan", tmp_path)
    generation = result["generation"]
    assert generation["status"] == "planned" and len(generation["jobs"]) == 2
    job = generation["jobs"][0]
    assert job["command"][-2] == "--config"
    worker_config = json.loads(open(job["command"][-1]).read())
    assert worker_config["research_params"] == str(archive)
    assert worker_config["torch_decoder"] and not worker_config["incremental"]
