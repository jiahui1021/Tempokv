"""Regression checks for experiment validity, independent of model weights."""
from dataclasses import asdict
import ast
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tempokv.config import CodecConfig
from tempokv.fitting import fit_legacy_cross_kv
from rebuttal.adapters.experimental_codec import ExperimentalCodec, ResearchCross
from rebuttal.adapters.worker import research_span_frames, require_codec_activity
from rebuttal.experiments.architecture.cross_low_rank import fit_cross_architecture
from rebuttal.experiments.architecture.fit import load_trace, validate_split, run, benchmark_decode


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_trace(seed=0):
    rng = torch.Generator().manual_seed(seed)
    keys = torch.randn(1, 8, 6, 1, 64, generator=rng)
    values = .7 * keys + .2 * torch.randn(keys.shape, generator=rng)
    return keys, values


def config():
    return CodecConfig(k_bits=4, v_bits=4, group_size=64, stride=1, fit_mode="legacy", use_triton=False,
                       k_history_length=2, v_history_length=0)


def test_full_rank_cross_uses_same_ridge_and_fitting_protocol(tmp_path):
    keys, values = fixture_trace()
    trace = tmp_path / "cal.npz"
    np.savez(trace, keys=keys.numpy(), values=values.numpy(), prompt_ids=np.asarray(["cal"] * keys.shape[1]))
    codec = tmp_path / "codec.json"
    codec.write_text(json.dumps(asdict(config())))
    report = run({"axis": "cross", "variants": ["low_rank", "dense"], "ranks": [64], "als_steps": 3,
                  "calibration_trace": str(trace), "codec_config": str(codec)}, "fit", tmp_path / "out")
    models = {row["variant"]: ExperimentalCodec.load(row["parameters"]) for row in report["variants"]}
    assert set(models) == {"low_rank_r64", "dense", "production"}
    low = models["low_rank_r64"].v_parameters
    dense = models["dense"].v_parameters
    assert torch.allclose(low.factor_a @ low.factor_b, dense.weight, atol=2e-6, rtol=2e-5)
    assert torch.allclose(low.bias, dense.bias, atol=2e-6, rtol=2e-5)
    assert next(row for row in report["variants"] if row["variant"] == "production")["production_reference_control"]


def test_zero_target_does_not_inject_identity_prior():
    keys, _ = fixture_trace()
    values = torch.zeros_like(keys)
    production = fit_legacy_cross_kv(keys, values, ridge=.1, stride=1, weight_mode="layer_shared")
    baseline = ResearchCross.from_production(production, "layer_shared", 1, 1)
    for kind in ("dense", "diagonal", "low_rank"):
        result, _ = fit_cross_architecture(keys, values, baseline, config(), kind=kind,
                                           rank=64 if kind == "low_rank" else None, rounds=1, als_steps=2)
        weights = result.factor_a @ result.factor_b if kind == "low_rank" else result.weight
        assert torch.count_nonzero(weights) == 0


def write_identified(path, value, prompt, prompt_hash, *, partial=False):
    tensor = np.full((1, 2, 4, 1, 64), value, dtype=np.float32)
    np.savez(path, keys=tensor, values=tensor, prompt_id=np.asarray(str(prompt)), seed=np.asarray(0))
    path.with_suffix(".identity.json").write_text(json.dumps({"records": [
        {"record_start": 0, "record_stop": 1 if partial else 2,
         "prompt_id": prompt, "prompt_sha256": prompt_hash, "seed": 0}]}))


def test_converter_sidecars_verify_split_and_detect_hash_overlap(tmp_path):
    first, second = tmp_path / "cal.npz", tmp_path / "eval.npz"
    write_identified(first, 1, 0, "cal-hash")
    write_identified(second, 2, 0, "eval-hash")
    # Each input file may use local prompt index 0; content hashes disambiguate.
    result = validate_split(load_trace(first)[2], load_trace(second)[2], require_prompt_identity=True)
    assert result["status"] == "verified_prompt_disjoint"
    write_identified(second, 2, 99, "cal-hash")
    with pytest.raises(ValueError, match="overlap"):
        validate_split(load_trace(first)[2], load_trace(second)[2], require_prompt_identity=True)


def test_partial_sidecar_is_not_verified_from_scalar_fallback(tmp_path):
    first, second = tmp_path / "cal.npz", tmp_path / "eval.npz"
    write_identified(first, 1, 0, "cal-hash")
    write_identified(second, 2, 1, "eval-hash", partial=True)
    with pytest.raises(ValueError, match="identity is unavailable"):
        validate_split(load_trace(first)[2], load_trace(second)[2], require_prompt_identity=True)


@pytest.mark.parametrize("frames", [717, 1401])
def test_default_architecture_span_is_consumed_before_final_block(frames):
    # Execute the actual production scheduler without importing model dependencies.
    path = Path(__file__).resolve().parents[2] / "experiments/Self-Forcing/pipeline/causal_inference.py"
    tree = ast.parse(path.read_text())
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "ready_quantization_spans")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    horizon = (frames + 3) // 4
    span = research_span_frames({"frames": frames})
    assert span == 24
    assert namespace["ready_quantization_spans"](0, horizon - 3, span)
    with pytest.raises(ValueError, match="no compressed span"):
        research_span_frames({"frames": frames, "span_frames": horizon})


def test_activity_requires_runtime_encode_and_decode_for_both_roles():
    before = dict(encode_k=3, encode_v=3, decode_k=10, decode_v=10)
    with pytest.raises(RuntimeError, match="did not encode and decode"):
        require_codec_activity(before, before)
    with pytest.raises(RuntimeError):
        require_codec_activity(before, {**before, "encode_k": 4, "encode_v": 4})
    assert require_codec_activity(before, {key: value + 1 for key, value in before.items()}) == {
        "encode_k": 1, "encode_v": 1, "decode_k": 1, "decode_v": 1}


def test_decode_benchmark_excludes_encode_and_warmup(monkeypatch):
    module = importlib.import_module("rebuttal.experiments.architecture.fit")
    from types import SimpleNamespace
    clock = [0.0]
    class Packed:
        def decode(self, **kwargs):
            clock[0] += 1
            return torch.zeros(1)
    def encode(value, *args, **kwargs):
        clock[0] += 100
        return Packed(), torch.zeros(1)
    monkeypatch.setattr(module, "encode_tensor", encode)
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    codec = SimpleNamespace(config=config(), temporal=None, v_parameters=None, backend="test")
    keys = torch.zeros(2, 1, 3, 1, 64)
    result = benchmark_decode(codec, keys, keys, warmup=2, repeats=3)
    assert result["decode_seconds_samples"] == [4.0, 4.0, 4.0]
    assert result["decode_warmup"] == 2 and result["decode_repeats"] == 3


def test_codec_activity_counts_real_packed_decoding():
    from rebuttal.experiments.architecture.temporal import identity_predictor
    temporal = identity_predictor(1, 1, 64, 2, "shared")
    cross = ResearchCross("dense", "layer_shared", torch.zeros(1, 64), torch.empty(1, 0, 1, 64),
                          1, 1, weight=torch.eye(64)[None])
    codec = ExperimentalCodec(temporal, cross, config())
    before = dict(codec.activity_counts)
    k, v = [x[0].permute(0, 2, 1, 3) for x in fixture_trace()]
    packed = codec.encode(k, v)
    assert codec.activity_counts["decode_k"] == codec.activity_counts["decode_v"] == 0
    codec.decode(packed)
    assert require_codec_activity(before, codec.activity_counts) == {
        "encode_k": 1, "encode_v": 1, "decode_k": 1, "decode_v": 1}


@pytest.mark.parametrize("active", [True, False])
def test_worker_verifies_each_inference_before_accepting_video(tmp_path, monkeypatch, active):
    import sys
    from types import SimpleNamespace
    from rebuttal.adapters import worker
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a test video\n")
    counts = dict(encode_k=0, encode_v=0, decode_k=0, decode_v=0)
    fake_codec = SimpleNamespace(activity_counts=counts)
    monkeypatch.setattr(ExperimentalCodec, "load", lambda *args, **kwargs: fake_codec)
    def install_bridge(runtime, codec_factory):
        class Base:
            def __init__(self):
                self.tempokv_codec = codec_factory()
            def inference(self):
                if active:
                    for key in counts:
                        counts[key] += 1
                return "video"
        return Base
    monkeypatch.setattr(worker, "_load_bridge", lambda name: SimpleNamespace(install=install_bridge))
    monkeypatch.setitem(sys.modules, "pipeline", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pipeline.causal_inference", SimpleNamespace())
    # Restore environment changes made by install after the test.
    for name in ("TEMPOKV_HORIZON_FRAMES", "TEMPOKV_NUM_FRAMES", "TEMPOKV_SPAN_FRAMES",
                 "TEMPOKV_BF16_LAG_FRAMES", "TEMPOKV_INCREMENTAL", "RANK"):
        monkeypatch.setenv(name, "0")
    cls = worker.install({"frames": 717, "research_params": "mock.npz", "prompts": str(prompts),
                          "output_dir": str(tmp_path)}, tmp_path)
    pipeline = cls()
    pipeline.set_research_context(prompt_index=0, seed=0)
    if active:
        assert pipeline.inference() == "video"
        report = json.loads((tmp_path / "codec_activity.rank-0.jsonl").read_text())
        assert report["status"] == "verified" and report["span_frames"] == 24
        assert all(x == 1 for x in report["counts"].values())
    else:
        with pytest.raises(RuntimeError, match="did not encode and decode"):
            pipeline.inference()
        assert not (tmp_path / "codec_activity.rank-0.jsonl").exists()
