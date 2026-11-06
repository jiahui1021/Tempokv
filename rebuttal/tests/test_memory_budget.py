from __future__ import annotations

from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import weakref

import pytest
import torch

from tempokv.codec import TempoKVCodec
from tempokv.config import CodecConfig
from tempokv.fitting import build_fixed_temporal_parameters
from rebuttal.experiments.memory.accounting import allocation_bytes
from rebuttal.experiments.memory.budget_manager import BudgetManager, ReleasableKVCache
from rebuttal.experiments.memory.capacity import ModelShape, capacity_report, span_bytes


def cache(frames=20):
    return ReleasableKVCache(1, 2, 2, 4, frames, torch.bfloat16, torch.device("cpu"))


def make_codec(tmp_path):
    config = CodecConfig(k_bits=2, v_bits=2, group_size=2, stride=2, anchor_bits=4,
        k_predictor_mode="identity", v_predictor_mode="identity", k_history_length=1,
        v_history_length=1, fit_mode="native", use_triton=False)
    parameters = build_fixed_temporal_parameters("identity", history_length=1, dim=4, stride=2)
    codec_path = tmp_path / "codec.json"
    codec_path.write_text(json.dumps(asdict(config)))
    k_path, v_path = tmp_path / "k.npz", tmp_path / "v.npz"
    parameters.save(k_path)
    parameters.save(v_path)
    return TempoKVCodec(parameters, parameters, config), dict(codec_config=str(codec_path),
        k_params=str(k_path), v_params=str(v_path))


def test_allocation_count_includes_view_capacity_and_deduplicates_aliases():
    base = torch.arange(100)
    assert allocation_bytes({"a": base[:1], "b": base[40:50], "alias": base[:1]}) == 800
    assert allocation_bytes([base, base.clone()]) == 1600


def test_whole_span_release_removes_every_owner_and_preserves_positions():
    c = cache()
    c.write(0, 12, torch.ones(1, 12, 2, 4, dtype=torch.bfloat16))
    payload = torch.ones(10)
    pointer = weakref.ref(payload)
    c.store_quantized(0, 8, {"payload": payload})
    del payload
    assert c.release_before(3) == 0  # [0,4) remains required decode support
    assert pointer() is not None
    c.release_before(4)
    gc.collect()
    assert pointer() is None
    assert all(value is None for value in c._span_by_chunk[:4])
    assert c.global_frames[4:6] == [4, 5]
    assert not c.quantized_spans


def test_budget_reserves_whole_incoming_block_before_write_and_releases_prefix():
    caches = [{"k": cache(), "v": cache()}]
    manager = BudgetManager(caches, budget_bytes=256, history_frames=4, tokens_per_frame=2)
    manager.prepare(4, 0)
    for role in ("k", "v"):
        caches[0][role].write(0, 8, torch.ones(1, 8, 2, 4, dtype=torch.bfloat16))
    manager.snapshot("full")
    event = manager.prepare(6, 4)
    assert event["checked_bytes"] == 256
    assert event["reserved_incoming_bytes"] == 128
    assert caches[0]["k"].global_frames[:2] == [None, None]
    assert caches[0]["k"].global_frames[2:4] == [2, 3]
    for role in ("k", "v"):
        caches[0][role].write(8, 12, torch.ones(1, 4, 2, 4, dtype=torch.bfloat16))
    assert manager.snapshot("written")["kv_owned_bytes"] == 256
    too_small = BudgetManager(caches, budget_bytes=255, history_frames=4, tokens_per_frame=2)
    with pytest.raises(MemoryError, match="budget exceeded"):
        too_small.prepare(6, 4)


def test_shape_count_matches_real_packed_tempokv_and_excludes_parameters(tmp_path):
    codec, _ = make_codec(tmp_path)
    shape = ModelShape(layers=1, heads=2, head_dim=4, tokens_per_frame=2)
    pair = codec.encode(torch.randn(1, 2, 8, 4).bfloat16(), torch.randn(1, 2, 8, 4).bfloat16())
    assert allocation_bytes(pair) == span_bytes("tempokv", 4, shape, codec.config)


def test_capacity_enumerates_non_linear_overheads_without_claiming_measurement():
    config = {"model_shape": {"layers": 1, "heads": 2, "head_dim": 4, "tokens_per_frame": 2},
        "codec_overrides": {"group_size": 2, "stride": 2, "use_triton": False},
        "span_frames": 2, "block_frames": 1, "bf16_lag_frames": 0,
        "horizon_latent_frames": 12, "histories": [2, 4, 6, 12],
        "budgets_bytes": [256, 512], "methods": ["bf16", "tempokv"],
        "qvg": {"block_size": 2, "k_centroids": 2, "v_centroids": 2}}
    report = capacity_report(config)
    assert len(report["rows"]) == 4
    assert report["rows"][0]["history_frames"] == 4
    assert all("shape" in row["measurement"] for row in report["rows"])
    assert any(row["method"] == "tempokv" for row in report["unsupported"])


def stub_pipeline(monkeypatch, runtime):
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(blocks=[SimpleNamespace(self_attn=SimpleNamespace(sink_size=0)) for _ in range(2)])

        def forward(self, noisy_image_or_video, current_start, kv_cache):
            end = current_start + noisy_image_or_video.shape[1] * 2
            for layer in kv_cache:
                data = torch.full((1, end - current_start, 2, 4), float(current_start + 1), dtype=torch.bfloat16)
                layer["k"].write(current_start, end, data)
                layer["v"].write(current_start, end, data)
                first = max(0, end - 8)
                first = layer["k"].complete_span_read_start(first)
                k = layer["k"].read(first, end)
                v = layer["v"].read(first, end, source_tensor=k)
                assert torch.isfinite(v).all()
                layer["global_end_index"] = layer["local_end_index"] = end
            return v

    class Pipeline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.generator = Generator()
            self.num_transformer_blocks = 2
            self.num_heads = 2
            self.head_dim = 4
            self.frame_seq_length = 2
            self.local_attn_size = 4
            self.num_frame_per_block = 1
            self.kv_cache1 = None
            self.summary = {}

        def inference(self, noise):
            self._initialize_kv_cache(1, torch.bfloat16, torch.device("cpu"), noise.shape[1])
            self.reset_quantization_state()
            for frame in range(noise.shape[1]):
                result = self.generator(noisy_image_or_video=noise[:, frame:frame + 1],
                                        current_start=frame * 2, kv_cache=self.kv_cache1)
                self.quantize_kv_cache(frame * 2, (frame + 1) * 2, (frame + 1) * 2)
            return result

    package, module = ModuleType("pipeline"), ModuleType("pipeline.causal_inference")
    module.__file__ = str(runtime / "pipeline/causal_inference.py")
    module.CausalInferencePipeline = package.CausalInferencePipeline = Pipeline
    monkeypatch.setitem(sys.modules, "pipeline", package)
    monkeypatch.setitem(sys.modules, "pipeline.causal_inference", module)


@pytest.mark.parametrize("method", ["bf16", "tempokv"])
def test_actual_worker_binds_pipeline_and_enforces_budget_during_rollout(monkeypatch, tmp_path, method):
    from rebuttal.experiments.memory.worker import install
    runtime = Path(__file__).resolve().parents[2] / "experiments/Self-Forcing"
    stub_pipeline(monkeypatch, runtime)
    codec, paths = make_codec(tmp_path)
    budget = 640  # H=4 can retain one extra partial-span decode prefix.
    cls = install(runtime, {**paths, "method": method, "history_frames": 4,
        "span_frames": 2, "bf16_lag_frames": 0, "budget_bytes": budget,
        "memory_report": str(tmp_path / "report.json")})
    pipeline = cls()
    pipeline.inference(torch.ones(1, 12, 1, 1, 1))
    events = pipeline._budget_manager.events
    assert events and all(event["checked_bytes"] <= budget for event in events)
    assert any(event["event"] == "before_write" for event in events)
    assert pipeline.kv_cache1[0]["k"].global_frames[:6] == [None] * 6
    assert pipeline.kv_cache1[0]["k"].global_frames[10:12] == [10, 11]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["scope"] == "measured_tensor_allocations"
    assert report["history_frames"] == 4
    from rebuttal.experiments.memory.capacity import live_profile
    expected = live_profile(method, 4, 12,
        ModelShape(layers=2, heads=2, head_dim=4, tokens_per_frame=2), codec.config,
        span_frames=2, lag_frames=0, block_frames=1)
    assert report["peak_checked_bytes"] == max(row["kv_owned_bytes"] for row in expected)


def test_qvg_has_no_cpu_simulation_fallback(tmp_path):
    from rebuttal.adapters.qvg import QVGBackend
    (tmp_path / "quant_videogen/real").mkdir(parents=True)
    (tmp_path / "quant_videogen/real/prq.py").write_text("")
    backend = QVGBackend(tmp_path)
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        backend.encode_tensor(torch.zeros(1, 2, 4, 64), role="k")


def test_generation_plan_has_both_controls_and_real_worker(tmp_path):
    from rebuttal.experiments.memory.generation import build_generation_jobs
    plan = build_generation_jobs({"frames": 141, "horizon_latent_frames": 36,
        "model_shape": {"layers": 1, "heads": 1, "head_dim": 4, "tokens_per_frame": 2},
        "codec_overrides": {"group_size": 2, "stride": 2, "use_triton": False},
        "span_frames": 2, "block_frames": 1, "bf16_lag_frames": 0,
        "budget_basis_latent_frames": [6, 12], "matched_context_frames": 6,
        "methods": ["bf16", "tempokv"]}, tmp_path)
    scenarios = {j["comparison"] for j in plan["jobs"]}
    assert scenarios == {"matched_memory", "matched_context", "tempokv_short"}
    assert all(Path(j["command"][1]).is_file() for j in plan["jobs"])
    assert "checkpoint" in plan["missing_configuration"]
