from __future__ import annotations

import ast
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from tempokv.spectral import SpectralConfig, SpectralProbe
from tempokv.spectral.sensitivity import reference_attention

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "experiments/Self-Forcing"


def config(**extra):
    payload = {"sampling": {"layers": [0], "heads": [0, 1], "latent_frames": [3],
                            "query_tokens_per_head": 4, "max_candidate_blocks": 100},
               "frequency": {"spatial_block_size": [2, 2], "temporal_positions_per_frame": 3},
               "oracle": {"protect_recent_frames": 1, "key_chunk_size": 7}}
    payload.update(extra)
    return SpectralConfig.from_dict(payload)


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_capture_denoise_filter_and_absolute_history_scores(tmp_path):
    probe = SpectralProbe(config(), tmp_path, rank=2)
    probe.reset(prompt_id=17, batch_size=2)
    rng = torch.Generator().manual_seed(4)
    clean = torch.randn(2, 45, 2, 8, generator=rng)
    original = clean.clone()
    # Denoise projection must not populate clean statistics.
    probe.on_projection(0, k_pre=clean, v=clean, grid_sizes=[[3, 3, 5]] * 2, current_start=0, phase="denoise")
    assert not probe._features
    for frame in range(3):
        x = clean[:, frame * 15:(frame + 1) * 15]
        probe.on_projection(0, k_pre=x, v=x, grid_sizes=[[1, 3, 5]] * 2, current_start=frame * 15, phase="clean")
    keys = torch.cat([clean, torch.randn(2, 15, 2, 8, generator=rng)], dim=1)
    # The effective window has rolled: frames 1,2,3 remain, local slot 2 is current.
    window = keys[:, 15:]
    query = torch.randn(2, 15, 2, 8, generator=rng)
    inputs = [x.clone() for x in (query, window)]
    probe.on_attention(0, q_post=query, k_post=window, v=window,
                       metadata={"grid_sizes": [[1, 3, 5]] * 2, "current_start": 45,
                                 "local_start_index": 30, "attention_start": 0,
                                 "frame_seqlen": 15, "local_end_index": 45})
    paths = dict(probe.sample_dirs)
    probe.flush()
    torch.testing.assert_close(clean, original, rtol=0, atol=0)
    torch.testing.assert_close(query, inputs[0], rtol=0, atol=0)
    torch.testing.assert_close(window, inputs[1], rtol=0, atol=0)
    for sample, path in paths.items():
        spectrum = rows(path / "spectrum/spatial_energy.csv")
        scores = rows(path / "sensitivity/block_scores.csv")
        assert len(spectrum) == 3 * 2 * 2 * 3
        assert len(scores) == 6 * 2  # Every edge block from historical frame 1, two heads.
        assert {int(r["history_frame"]) for r in scores} == {1}
        assert {int(r["sample_id"]) for r in scores} == {sample}
        assert all(15 <= i < 30 for r in scores for i in json.loads(r["absolute_token_indices"]))
        assert all(int(r["effective_k_tokens"]) == 45 for r in scores)
        meta = json.loads((path / "metadata.json").read_text())
        assert meta["effective_windows"][0]["absolute_start"] == 15
        with np.load(path / "spectrum/temporal_energy.npz") as temporal:
            assert temporal["layer_0_k_segment_0_positions"].tolist() == [0, 7, 14]
            np.testing.assert_allclose(temporal["layer_0_k_segment_0_samples"],
                                       original[sample].reshape(3, 15, 2, 8)[:, [0, 7, 14]].numpy())
    first = paths[0]
    probe.reset(prompt_id=17, batch_size=1)
    second = probe.sample_dirs[0]
    probe.flush()
    assert second != first and first.exists()
    empty = json.loads((second / "metadata.json").read_text())
    assert empty["measurement"]["oracle"] == "not measured"


def test_temporal_gaps_caps_zero_energy_and_no_history(tmp_path):
    cfg = config(frequency={"spatial_block_size": [2, 2], "max_clean_frames": 2})
    probe = SpectralProbe(cfg, tmp_path)
    probe.reset()
    x = torch.zeros(1, 15, 2, 8)
    for frame in (0, 2, 3):
        probe.on_projection(0, k_pre=x, v=x, grid_sizes=[[1, 3, 5]], current_start=frame * 15, phase="clean")
    assert probe.counts["clean_frames_skipped_cap"] == 1
    probe.on_attention(0, q_post=x, k_post=x, v=x,
                       metadata={"grid_sizes": [[1, 3, 5]], "current_start": 45,
                                 "local_start_index": 0, "attention_start": 0, "frame_seqlen": 15})
    path = probe.sample_dirs[0]
    probe.flush()
    assert all(r["near_zero_energy"] == "True" and r["ratio"] == "nan" for r in rows(path / "spectrum/spatial_energy.csv"))
    with np.load(path / "spectrum/temporal_energy.npz") as data:
        assert not any(key.endswith("_low_ratio") for key in data.files)
        assert data["layer_0_k_segment_0_frames"].tolist() == [0]
        assert data["layer_0_k_segment_1_frames"].tolist() == [2]
    assert not rows(path / "sensitivity/block_scores.csv")


def test_live_hook_disabled_and_enabled_bf16_numerical_regression(tmp_path):
    # Execute the real attention class with its native attention implementation.
    # Removing only the optional hook statements reconstructs the prior BF16 path.
    source = ast.parse((RUNTIME / "wan/modules/causal_model.py").read_text())
    wanted = {"CausalWanSelfAttention", "_cache_index", "_set_cache_index"}
    nodes = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    attention_module = load_script("spectral_native_attention", RUNTIME / "wan/modules/attention.py")
    assert not attention_module.FLASH_ATTN_2_AVAILABLE and not attention_module.FLASH_ATTN_3_AVAILABLE
    def rope(x, *args, **kwargs):
        # A distinct deterministic rotation makes pre/post coordinates observable.
        signs = x.new_ones(x.shape[-1])
        signs[::2] = -1
        return x * signs
    namespace = {"torch": torch, "nn": torch.nn, "math": __import__("math"),
                 "attention": attention_module.attention, "causal_rope_apply": rope,
                 "causal_rope_apply_long_input": rope}
    def build(tree):
        scope = dict(namespace)
        exec(compile(ast.fix_missing_locations(ast.Module(body=tree, type_ignores=[])), "spectral_regression", "exec"), scope)
        return scope["CausalWanSelfAttention"]
    class RemoveProbe(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == "spectral_probe" for t in node.targets):
                return None
            return self.generic_visit(node)
        def visit_If(self, node):
            if isinstance(node.test, ast.BoolOp) and any(isinstance(n, ast.Name) and n.id == "spectral_probe" for n in ast.walk(node.test)):
                return [self.visit(child) for child in node.orelse]
            if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == "spectral_probe":
                return None
            return self.generic_visit(node)
    baseline_type = build([RemoveProbe().visit(copy.deepcopy(n)) for n in nodes])
    live_type = build(copy.deepcopy(nodes))
    torch.manual_seed(19)
    base = baseline_type(16, 2, local_attn_size=3, qk_norm=False).to(torch.bfloat16)
    disabled = live_type(16, 2, local_attn_size=3, qk_norm=False).to(torch.bfloat16)
    enabled = live_type(16, 2, local_attn_size=3, qk_norm=False).to(torch.bfloat16)
    disabled.load_state_dict(base.state_dict()); enabled.load_state_dict(base.state_dict())
    probe = SpectralProbe(config(), tmp_path)
    probe.reset()
    enabled.spectral_probe, enabled.spectral_layer_idx = probe, 0
    def cache():
        return {"k": torch.zeros(1, 45, 2, 8, dtype=torch.bfloat16),
                "v": torch.zeros(1, 45, 2, 8, dtype=torch.bfloat16),
                "cache_backend": "bf16", "global_end_index": 0, "local_end_index": 0}
    caches = [cache() for _ in range(3)]
    for frame in range(6):
        for phase in ("denoise", "clean"):
            x = torch.randn(1, 15, 16).to(torch.bfloat16)
            state = torch.get_rng_state().clone()
            probe.set_call(phase=phase)
            outputs = [layer(x, seq_lens=None, grid_sizes=torch.tensor([[1, 3, 5]]), freqs=None,
                             block_mask=None, kv_cache=cache, current_start=frame * 15)
                       for layer, cache in zip((base, disabled, enabled), caches)]
            assert torch.equal(torch.get_rng_state(), state)
            for actual in outputs[1:]:
                torch.testing.assert_close(actual, outputs[0], rtol=0, atol=0)
            for key in ("k", "v"):
                for cache in caches[1:]:
                    torch.testing.assert_close(cache[key], caches[0][key], rtol=0, atol=0)
    probe.flush()
    assert probe.counts["sensitivity/block_scores.csv"] > 0


def test_bridge_phase_announcement_and_prompt_rank_isolation(tmp_path, monkeypatch):
    cfg = config(sampling={"layers": [0], "heads": [0], "latent_frames": []})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg.to_dict()))
    monkeypatch.setenv("SPECTRAL_CONFIG", str(config_path))
    monkeypatch.setenv("SPECTRAL_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("RANK", "3")
    events = []
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = types.SimpleNamespace(blocks=[types.SimpleNamespace(self_attn=types.SimpleNamespace(sink_size=0, num_heads=1))])
        def forward(self, noisy_image_or_video, conditional_dict, timestep, kv_cache=None,
                    crossattn_cache=None, current_start=None):
            return noisy_image_or_video
    class Base:
        def __init__(self):
            self.generator = Generator()
            self._uses_pre_rope_cache = False
        def note_clean_refresh(self):
            pass
        def inference(self, noise, text_prompts, initial_latent=None):
            self.generator(noise, {}, torch.tensor(0), None, None, 60)
            events.append((self.spectral_probe.phase, self.spectral_probe.denoise_call))
            self.note_clean_refresh()
            self.generator(noise, {}, torch.tensor(999), current_start=60)
            events.append((self.spectral_probe.phase, self.spectral_probe.denoise_call))
            self.generator(noise, {}, torch.tensor(0), current_start=60)
            events.append((self.spectral_probe.phase, self.spectral_probe.denoise_call))
            return noise
    package, module = types.ModuleType("pipeline"), types.ModuleType("pipeline.causal_inference")
    package.CausalInferencePipeline = module.CausalInferencePipeline = Base
    monkeypatch.setitem(sys.modules, "pipeline", package)
    monkeypatch.setitem(sys.modules, "pipeline.causal_inference", module)
    bridge = load_script("spectral_bridge_test", RUNTIME / "tempokv_adapter/spectral_bridge.py")
    pipeline = bridge.install(RUNTIME)()
    pipeline.set_spectral_context(prompt_id=99, generation_seed=7)
    noise = torch.zeros(2, 1, 1, 1, 1)
    assert pipeline.inference(noise, ["a", "a"]) is noise
    assert events == [("denoise", 0), ("clean", 1), ("denoise", 1)]
    assert len(list((tmp_path / "output").rglob("metadata.json"))) == 2
    for path in (tmp_path / "output").rglob("metadata.json"):
        meta = json.loads(path.read_text())
        assert meta["prompt_id"] == 99 and meta["rank"] == 3 and meta["generation_seed"] == 7
    with pytest.raises(ValueError, match="text-to-video"):
        pipeline.inference(noise, ["a", "a"], initial_latent=noise)


def test_generation_spectral_dry_run_and_missing_environment(tmp_path, monkeypatch):
    checkpoint, prompts = tmp_path / "checkpoint.pt", tmp_path / "prompts.txt"
    checkpoint.touch(); prompts.write_text("test\n")
    script = RUNTIME / "tempokv_adapter/generate.py"
    command = [sys.executable, str(script), "--mode", "spectral", "--checkpoint", str(checkpoint),
               "--prompts", str(prompts), "--output-dir", str(tmp_path / "videos"), "--frames", "189", "--dry-run"]
    env = {**os.environ, "SPECTRAL_CONFIG": str(ROOT / "scripts/spectral/configs/self_forcing.yaml"),
           "SPECTRAL_OUTPUT_DIR": str(tmp_path / "spectral")}
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == "spectral" and "parameters" not in report
    assert report["generation_command"][report["generation_command"].index("--num_output_frames") + 1] == "48"
    assert not (tmp_path / "videos").exists() and not (tmp_path / "spectral").exists()
    env.pop("SPECTRAL_CONFIG")
    failed = subprocess.run(command, env=env, text=True, capture_output=True)
    assert failed.returncode != 0 and "SPECTRAL_CONFIG" in failed.stderr


def test_smoke_analysis_and_offline_replay(tmp_path):
    smoke = load_script("spectral_smoke", ROOT / "scripts/spectral/smoke.py")
    output = tmp_path / "reports"
    assert smoke.main(["--output", str(output)]) == 0
    analyze = load_script("spectral_analyze", ROOT / "scripts/spectral/analyze_results.py")
    summary = analyze.analyze(output, tmp_path / "summary")
    assert "Synthetic CPU smoke" in (summary / "summary.md").read_text()
    for name in ("frequency_by_layer.png", "frequency_vs_importance.png", "error_vs_retention.png"):
        assert (summary / name).stat().st_size > 1000
    snapshot = next(output.rglob("window-0.npz"))
    replay = load_script("spectral_replay", ROOT / "scripts/spectral/replay_compression.py")
    target = tmp_path / "replay.csv"
    assert replay.main(["--input", str(snapshot), "--output", str(target)]) == 0
    replayed = rows(target)
    assert all(row["status"] == "measured" for row in replayed)
    original = [r for r in rows(snapshot.parent / "error_budget.csv") if r["head"] == "0"]
    assert len(original) == len(replayed)
    for online, offline in zip(original, replayed):
        assert online["random_seed"] == offline["random_seed"]
        assert online["actual_retained_tokens"] == offline["actual_retained_tokens"]
        assert float(online["relative_attention_output_error"]) == float(offline["relative_attention_output_error"])
    again = analyze.analyze(output, summary)
    assert again != summary and (summary / "summary.md").exists()


def test_empty_results_remain_unmeasured_in_report(tmp_path):
    probe = SpectralProbe(config(), tmp_path / "empty")
    probe.reset()
    probe.flush()
    analyze = load_script("spectral_empty_analyze", ROOT / "scripts/spectral/analyze_results.py")
    summary = analyze.analyze(tmp_path / "empty", tmp_path / "empty_summary")
    assert "not measured" in (summary / "summary.md").read_text()
    assert not rows(summary / "budget_summary.csv")


def test_oracle_caps_and_invalid_window_metadata(tmp_path):
    cfg = config(sampling={"layers": [0], "heads": [0], "latent_frames": [1],
                           "max_oracle_calls_per_layer": 1}, oracle={"protect_recent_frames": 0})
    probe = SpectralProbe(cfg, tmp_path)
    probe.reset()
    x = torch.randn(1, 15, 1, 4)
    probe.on_projection(0, k_pre=x, v=x, grid_sizes=[[1, 3, 5]], current_start=0, phase="clean")
    meta = {"grid_sizes": [[1, 3, 5]], "current_start": 15,
            "local_start_index": 15, "attention_start": 0, "frame_seqlen": 15}
    with pytest.raises(ValueError, match="explicit"):
        probe.on_attention(0, q_post=x, k_post=torch.cat([x, x], 1), v=torch.cat([x, x], 1),
                           metadata={**meta, "sink_frames": 1})
    with pytest.raises(ValueError, match="length"):
        probe.on_attention(0, q_post=x, k_post=torch.cat([x, x], 1), v=torch.cat([x, x], 1),
                           metadata={**meta, "local_end_index": 31})
    for _ in range(2):
        probe.on_attention(0, q_post=x, k_post=torch.cat([x, x], 1), v=torch.cat([x, x], 1), metadata=meta)
    assert probe.counts["oracle_calls_skipped_cap"] == 1
    path = probe.sample_dirs[0]
    probe.flush()
    # First historical frame has no similarity: comparison uses a common empty
    # finite subset instead of comparing different populations by score.
    assert {int(row["finite_candidates"]) for row in rows(path / "sensitivity/correlation.csv")} == {0}
