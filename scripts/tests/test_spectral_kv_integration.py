from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from tempokv.spectral_kv.cache import SpectralKVCache
from tempokv.spectral_kv.config import SpectralKVConfig, load_config
from tempokv.spectral_kv.method_spec import parse_method
from tempokv.spectral_kv.model_spec import ModelSpec
from tempokv.spectral_kv.policy import FullPolicy, OracleTopKPolicy
from tempokv.spectral_kv.rope3d import full_positions, rope_apply
from tempokv.spectral_kv.attention import spectral_attention_ref
from tempokv.spectral.sensitivity import reference_attention

ROOT = Path(__file__).resolve().parents[2]


def load_script(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["BF16", "DROP_win21", "SPECSUM_dct_R4_B1", "SPECSUM_avg_R16_B0", "SPECSUM_dct_R64_B1_top8", "PYR_dct_b0.5", "KVSKIP_raw", "KVSKIP_lin"])
def test_method_names_are_explicit(name):
    assert parse_method(name).name == name


@pytest.mark.parametrize("name", ["SPECSUM_avg_R4_B1_hann", "SPECSUM_dct_R4_B1+TEMPOKV_k4v2", "unknown", "DROP_win0"])
def test_unsupported_combinations_never_silently_fallback(name):
    with pytest.raises(ValueError): parse_method(name)


def test_cache_full_roundtrip_and_ownership():
    spec = ModelSpec(1, 2, 8, (3, 5), 1, (4, 2, 2))
    rng = torch.Generator().manual_seed(3)
    k, v, q = [torch.randn(1, 30, 2, 8, generator=rng) for _ in range(3)]
    originals = [x.clone() for x in (k, v)]
    cache = SpectralKVCache(spec, 0, FullPolicy())
    cache.write_chunk(k, v, 0)
    cache.age(0)
    view = cache.materialize()
    post = rope_apply(k, *full_positions(0, 2, (3, 5)), spec)
    torch.testing.assert_close(spectral_attention_ref(q, view), reference_attention(q, post, v))
    assert view.token_count == 30
    for actual, original in zip((k, v), originals): torch.testing.assert_close(actual, original, rtol=0, atol=0)
    k.zero_()
    assert not torch.equal(cache.entries[0].k_pre, k[:, :15])


def test_cache_protection_iteration_and_original_oracle():
    spec = ModelSpec(1, 1, 8, (3, 5), 1, (4, 2, 2))
    k = torch.randn(1, 60, 1, 8)
    cache = SpectralKVCache(spec, 0, OracleTopKPolicy(0, ratio=4, recent=0), sink_frames=1)
    cache.write_chunk(k, k, 0)
    cache.current_frames = {3}
    cache.age(3)
    assert cache.entries[0].grid == cache.entries[3].grid == (3, 5)
    assert cache.entries[1].grid == cache.entries[2].grid == (2, 2)
    cache.policy = OracleTopKPolicy(0, ratio=16, recent=0)
    cache.age(3)
    assert cache.entries[1].grid == (1, 1)
    cache.policy = FullPolicy()
    with pytest.raises(ValueError, match="FULL"): cache.age(3)
    oracle = SpectralKVCache(spec, 0, OracleTopKPolicy(0, ratio=16, recent=0), keep_originals=True)
    oracle.write_chunk(k, k, 0); oracle.current_frames = {3}; oracle.age(3)
    oracle.policy = FullPolicy(); oracle.age(3)
    assert oracle.materialize().token_count == 60
    oracle.evict_outside({0, 2, 3})
    assert set(oracle.entries) == set(oracle.originals) == {0, 2, 3}


def test_suite_separation_config_and_guards():
    root = ROOT / "scripts/paper_experiments/spectral/configs"
    for path in root.glob("*.yaml"):
        config = load_config(path)
        assert config.suite_id == "B_SPECTRAL_KV"
        assert all(e.startswith("B_") for e in config.experiments)
    for payload in ({"suite_id": "A_FREQUENCY_ORACLE"}, {"method": "SPECSUM_dct_R4_B1"},
                    {"online": True, "experiments": ["B_E1_SPECTRUM"]},
                    {"online": True, "experiments": ["B_E2B_REFRESH"], "method": "KVSKIP_lin"}):
        with pytest.raises(ValueError): SpectralKVConfig.from_dict(payload)


@pytest.mark.parametrize("online,method", [(False, "BF16"), (True, "SPECSUM_dct_R4_B1"), (True, "SPECSUM_avg_R4_B1"), (True, "DROP_win1")])
def test_probe_synthetic_readonly_and_online(tmp_path, online, method):
    smoke = load_script("b_smoke", ROOT / "scripts/paper_experiments/spectral/smoke.py")
    path = smoke.run_smoke(tmp_path, method=method, online=online)
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["suite_id"] == "B_SPECTRAL_KV" and metadata["trajectory_changed"] == online
    assert metadata["status"] == "completed"
    assert metadata["measurement"]
    if not online:
        records = [json.loads(line) for line in (path / "B_E3_OFF_SHADOW/records.jsonl").read_text().splitlines()]
        assert all(r["relative_attention_output_error"] == 0 for r in records if r["method_spec"] == "BF16")
        analyzer = load_script("b_analyze", ROOT / "scripts/paper_experiments/spectral/analyze.py")
        summary = analyzer.analyze(tmp_path, tmp_path / "summary")
        assert (summary / "B_E1_SPECTRUM/results.png").stat().st_size > 1000
        assert (summary / "B_E5_RESIDUAL/summary.md").read_text().find("not measured") >= 0
        assert "B_SPECTRAL_KV" in (summary / "summary.md").read_text()


@pytest.mark.parametrize("runtime", ["self_forcing", "causal_forcing"])
def test_separate_launcher_dry_run(tmp_path, runtime):
    checkpoint, prompts, model = tmp_path / "checkpoint.pt", tmp_path / "prompts.txt", tmp_path / "model"
    checkpoint.touch(); prompts.write_text("fixture\n"); model.mkdir()
    result = subprocess.run([sys.executable, str(ROOT / "scripts/paper_experiments/spectral/run.py"),
        "--config", str(ROOT / "scripts/paper_experiments/spectral/configs/B_E1_SPECTRUM.yaml"),
        "--runtime", runtime, "--checkpoint", str(checkpoint), "--model-root", str(model), "--prompts", str(prompts),
        "--output", str(tmp_path / "b"), "--dry-run"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "B_SPECTRAL_KV" in result.stdout and '"spectral_kv"' in result.stdout
    assert not (tmp_path / "b").exists()


@pytest.mark.parametrize("runtime", ["self_forcing", "causal_forcing"])
@pytest.mark.parametrize("skip", [False, True])
def test_pipeline_context_and_explicit_skip_mode(tmp_path, monkeypatch, runtime, skip):
    import types
    from tempokv.spectral_kv.bridge import install
    events = []
    class Block(torch.nn.Identity):
        def __init__(self):
            super().__init__(); self.self_attn = types.SimpleNamespace(num_heads=1, sink_size=0)
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.model = types.SimpleNamespace(blocks=torch.nn.ModuleList([Block()]))
        def forward(self, noisy_image_or_video, conditional_dict, timestep, kv_cache=None, crossattn_cache=None, current_start=0):
            probe = self.model.blocks[0].self_attn.spectral_probe
            events.append((probe.phase, probe.context.step_idx, probe.context.is_last_denoise))
            return noisy_image_or_video
    class Base:
        def __init__(self):
            self.generator = Generator(); self.denoising_step_list = torch.tensor([999, 0])
            self.denoising_step_list_first_chunk = None
            self.vae = types.SimpleNamespace(decode_to_pixel=lambda x: x)
        def note_clean_refresh(self): pass
        def inference(self, noise, text_prompts, initial_latent=None):
            for step in (0, 1): self.generator(noise, {}, torch.tensor(step), kv_cache=[], current_start=0)
            self.note_clean_refresh(); self.generator(noise, {}, torch.tensor(444), kv_cache=[], current_start=0)
            return noise
    prefix = "pipeline" if runtime == "self_forcing" else "causal_forcing.pipeline"
    package, module = types.ModuleType(prefix), types.ModuleType(prefix + ".causal_inference")
    package.CausalInferencePipeline = module.CausalInferencePipeline = Base
    monkeypatch.setitem(sys.modules, prefix, package); monkeypatch.setitem(sys.modules, prefix + ".causal_inference", module)
    cfg = {"suite_id": "B_SPECTRAL_KV", "profile": "context_test", "layers": [0], "heads": [0],
           "experiments": ["B_E2B_REFRESH" if skip else "B_E1_SPECTRUM"], "online": skip, "method": "KVSKIP_raw" if skip else "BF16"}
    path = tmp_path / "config.json"; path.write_text(json.dumps(cfg))
    monkeypatch.setenv("SPECTRAL_KV_CONFIG", str(path)); monkeypatch.setenv("SPECTRAL_KV_OUTPUT_DIR", str(tmp_path / "out"))
    pipe = install(ROOT / "experiments" / ("Self-Forcing" if runtime == "self_forcing" else "Causal-Forcing"), runtime)()
    pipe.set_spectral_context(prompt_id=77, generation_seed=12)
    noise = torch.zeros(1, 1, 1, 1, 1)
    assert pipe.inference(noise, ["x"]) is noise
    assert events[:2] == [("denoise", 0, False), ("denoise", 1, True)]
    assert len(events) == (2 if skip else 3)
    if not skip: assert events[-1] == ("clean", -1, False)
    meta = json.loads(next((tmp_path / "out").rglob("metadata.json")).read_text())
    assert meta["prompt_id"] == 77 and meta["generation_seed"] == 12
    if skip: assert meta["skipped"]["clean_passes_skipped"] == 1


@pytest.mark.parametrize("runtime", ["Self-Forcing", "Causal-Forcing/causal_forcing"])
def test_native_bf16_module_with_b_probe_matches_disabled_including_sink_roll(tmp_path, runtime):
    import ast
    source = ast.parse((ROOT / "experiments" / runtime / "wan/modules/causal_model.py").read_text())
    names = {"CausalWanSelfAttention", "_cache_index", "_set_cache_index", "causal_rope_apply"}
    nodes = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    attention_file = ROOT / "experiments" / runtime / "wan/modules/attention.py"
    attention = load_script("b_native_attention", attention_file).attention
    from tempokv.runtime.rope import causal_rope_apply_long_input
    scope = {"torch": torch, "nn": torch.nn, "math": __import__("math"), "attention": attention,
             "causal_rope_apply_long_input": causal_rope_apply_long_input}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "native_b_module", "exec"), scope)
    layer_type = scope["CausalWanSelfAttention"]
    torch.manual_seed(4)
    disabled = layer_type(16, 2, local_attn_size=3, sink_size=1, qk_norm=False).to(torch.bfloat16)
    enabled = layer_type(16, 2, local_attn_size=3, sink_size=1, qk_norm=False).to(torch.bfloat16)
    enabled.load_state_dict(disabled.state_dict())
    from tempokv.spectral_kv.probe import SpectralKVProbe
    from tempokv.spectral_kv.context import RunContext
    probe = SpectralKVProbe(SpectralKVConfig(experiments=("B_E1_SPECTRUM", "B_E3_OFF_SHADOW"),
        profile="native", layers=(0,), heads=(0,), chunks=(), denoise_steps=(), spatial_bins=3, query_tokens=3,
        keep_recent_frames=0, shadow_methods=("BF16", "SPECSUM_avg_R4_B1")), tmp_path)
    probe.reset(); enabled.spectral_probe = probe; enabled.spectral_layer_idx = 0
    def cache(): return {"k": torch.zeros(1,45,2,8,dtype=torch.bfloat16), "v": torch.zeros(1,45,2,8,dtype=torch.bfloat16),
                         "global_end_index":0,"local_end_index":0,"cache_backend":"bf16"}
    caches = [cache(), cache()]
    freq = torch.polar(torch.ones(64,4,dtype=torch.float64), torch.arange(64,dtype=torch.float64)[:,None]*torch.tensor([1,.01,1,1],dtype=torch.float64))
    with torch.no_grad():
        for frame in range(4):
            for clean in (False, True):
                probe.set_call(RunContext(frame, 0, clean, frame, not clean))
                x = torch.randn(1,15,16).to(torch.bfloat16)
                state = torch.get_rng_state().clone()
                outputs = [layer(x, seq_lens=None, grid_sizes=torch.tensor([[1,3,5]]), freqs=freq,
                    block_mask=None, kv_cache=cache, current_start=frame*15) for layer,cache in zip((disabled,enabled),caches)]
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                assert torch.equal(state, torch.get_rng_state())
                for role in ("k","v"): torch.testing.assert_close(caches[0][role],caches[1][role],rtol=0,atol=0)
    probe.flush()
    assert caches[0]["global_end_index"] == 60 and caches[0]["local_end_index"] == 45


def test_linear_skip_uses_only_covered_map_and_keeps_native_shape(tmp_path):
    from tempokv.spectral_kv.calibration import LinearCalibration
    from tempokv.spectral_kv.probe import SpectralKVProbe
    from tempokv.spectral_kv.context import RunContext
    rng = torch.Generator().manual_seed(7)
    calibration = LinearCalibration(1e-8)
    x = torch.randn(64,2,8,generator=rng)
    for role in ("k","v"): calibration.update(0,role,x,x*1.2+.1)
    path = tmp_path / "map.npz"; calibration.fit().save(path)
    cfg = SpectralKVConfig(experiments=("B_E2B_REFRESH",),profile="lin",layers=(0,),heads=(0,1),
        online=True,method="KVSKIP_lin",calibration_file=str(path))
    probe = SpectralKVProbe(cfg,tmp_path);probe.reset();probe.set_call(RunContext(0,1,False,0,True))
    k,v = [torch.randn(1,15,2,8,generator=rng) for _ in range(2)]
    probe.on_projection(0,k_pre=k,v=v,grid_sizes=[[1,3,5]],current_start=0)
    spec=probe.specs[0]; positions=full_positions(0,1,(3,5))
    native={"k":rope_apply(k,*positions,spec),"v":v.clone(),"local_end_index":15}
    probe.skip_clean([native])
    torch.testing.assert_close(native["k"],rope_apply(k*1.2+.1,*positions,spec),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(native["v"],v*1.2+.1,atol=1e-5,rtol=1e-5)
    probe.flush()
