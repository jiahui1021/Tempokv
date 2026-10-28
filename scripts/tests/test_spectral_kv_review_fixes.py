"""Regression checks for review R5/R6/R7, using synthetic CPU inputs only."""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from tempokv.spectral_kv.bridge import install
from tempokv.spectral_kv.stats import OnlineSpectrum
from tempokv.spectral_kv.temporal import OnlineTemporalSpectrum


ROOT = Path(__file__).resolve().parents[2]


def _analyzer():
    spec = importlib.util.spec_from_file_location("b_review_fixed_analyzer", ROOT / "scripts/paper_experiments/spectral/analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(root, rows, **extra):
    experiment = root / "B_E1_SPECTRUM"
    experiment.mkdir(parents=True)
    meta = {"suite_id": "B_SPECTRAL_KV", "profile": "review_fixture", "runtime": "self_forcing",
            "local_attn_size": 21, "sink_size": 0, "model": "fixture-model",
            "source": "synthetic CPU fixture; not research evidence", "config": {}, **extra}
    (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (experiment / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _point(value, frame=0):
    return {"record_kind": "spatial", "layer": 0, "head": 0, "role": "k", "coordinate": "pre",
            "frame_idx": frame, "sample_id": 0, "r95": value, "dc_ratio": value, "valid": True, "status": "ok"}


def test_spatial_and_joint_dc_keep_separate_definitions_and_observation_counts(tmp_path):
    x = torch.ones(2, 4, 4, 1, 1)
    x[1].neg_()
    spatial = [dict(row, record_kind="spatial") for row in OnlineSpectrum(bins=2).update(0, x)]
    joint = OnlineTemporalSpectrum(window=2, spatial_bins=2, temporal_bins=2).update(0, x)
    _run(tmp_path / "input/run", spatial + joint)
    out = _analyzer().analyze(tmp_path / "input", tmp_path / "output", "B_E1_SPECTRUM") / "B_E1_SPECTRUM"
    combined = _csv(out / "dc_summary.csv")
    assert len(combined) == 2
    by_kind = {row["record_kind"]: row for row in combined}
    assert float(by_kind["spatial"]["dc_ratio"]) == pytest.approx(1)
    assert by_kind["spatial"]["dc_definition"] == "spatial_h0_w0"
    assert int(by_kind["spatial"]["observation_count"]) == 2
    assert float(by_kind["temporal"]["dc_ratio"]) == pytest.approx(0)
    assert by_kind["temporal"]["dc_definition"] == "joint_t0_h0_w0"
    assert int(by_kind["temporal"]["observation_count"]) == 1
    assert len(_csv(out / "spatial_dc_summary.csv")) == len(_csv(out / "joint_dc_summary.csv")) == 1
    assert len(_csv(out / "spectrum_observations.csv")) == 3
    assert len(_csv(out / "records.csv")) == len(spatial) + len(joint)


def test_common_metrics_are_not_weighted_by_repeated_band_rows(tmp_path):
    rows = [dict(_point(.2, 0), band=0)] + [dict(_point(.8, 1), band=i) for i in range(8)]
    _run(tmp_path / "input/run", rows)
    analyzer = _analyzer()
    out = analyzer.analyze(tmp_path / "input", tmp_path / "output", "B_E1_SPECTRUM") / "B_E1_SPECTRUM"
    summary = _csv(out / "spatial_dc_summary.csv")
    assert len(summary) == 1 and float(summary[0]["dc_ratio"]) == pytest.approx(.5)
    assert int(summary[0]["observation_count"]) == 2
    observations = _csv(out / "spectrum_observations.csv")
    assert np.mean([float(row["r95"]) for row in observations]) == pytest.approx(.5)


def test_joint_warmup_window_lengths_are_not_averaged_together(tmp_path):
    joint = OnlineTemporalSpectrum(window=3, spatial_bins=2, temporal_bins=2).update(0, torch.ones(3, 4, 4, 1, 1))
    _run(tmp_path / "input/run", joint)
    out = _analyzer().analyze(tmp_path / "input", tmp_path / "output", "B_E1_SPECTRUM") / "B_E1_SPECTRUM"
    summary = _csv(out / "joint_dc_summary.csv")
    assert {int(row["window_frames"]) for row in summary} == {2, 3}
    assert all(int(row["observation_count"]) == 1 for row in summary)


@pytest.mark.parametrize("identity", ["model", "checkpoint"])
def test_model_and_checkpoint_cohorts_do_not_average_and_keep_sources(tmp_path, identity):
    for name, value in (("A", .1), ("B", .9)):
        _run(tmp_path / "input" / name, [_point(value)], **{identity: "checkpoint-" + name})
    out = _analyzer().analyze(tmp_path / "input", tmp_path / "output", "B_E1_SPECTRUM")
    settings = sorted(out.glob("setting-*"))
    assert len(settings) == 2
    observed = []
    preserved = []
    for setting in settings:
        rows = _csv(setting / "B_E1_SPECTRUM/records.csv")
        assert len(rows) == 1 and rows[0]["source_id"] and rows[0]["cohort_id"]
        assert rows[0]["source"].startswith("synthetic")
        preserved.append(rows[0][identity])
        observed.append(float(_csv(setting / "B_E1_SPECTRUM/spatial_dc_summary.csv")[0]["dc_ratio"]))
    assert sorted(observed) == [.1, .9]
    assert sorted(preserved) == ["checkpoint-A", "checkpoint-B"]
    manifest = json.loads((out / "sources.json").read_text())
    assert {source["metadata"][identity] for source in manifest} == {"checkpoint-A", "checkpoint-B"}


@pytest.mark.parametrize("key,first,second", [
    ("spatial_bins", 4, 8), ("feature_channels", 8, 16), ("rope_precision", "float32", "float64"),
])
def test_measurement_defining_configuration_splits_cohorts(tmp_path, key, first, second):
    for name, value, config in (("A", .1, first), ("B", .9, second)):
        _run(tmp_path / "input" / name, [_point(value)], config={key: config})
    out = _analyzer().analyze(tmp_path / "input", tmp_path / "output", "B_E1_SPECTRUM")
    settings = list(out.glob("setting-*"))
    assert len(settings) == 2
    observed = [float(_csv(path / "B_E1_SPECTRUM/spatial_dc_summary.csv")[0]["dc_ratio"]) for path in settings]
    assert sorted(observed) == [.1, .9]


def test_replica_seed_does_not_change_definition_but_is_preserved(tmp_path):
    for name, value, seed in (("A", .1, 41), ("B", .9, 42)):
        _run(tmp_path / "input" / name, [_point(value)], config={"seed": seed, "spatial_bins": 8})
    out = _analyzer().analyze(tmp_path / "input", tmp_path / "output", "B_E1_SPECTRUM")
    assert not list(out.glob("setting-*"))
    assert float(_csv(out / "B_E1_SPECTRUM/spatial_dc_summary.csv")[0]["dc_ratio"]) == pytest.approx(.5)
    assert {row["metadata"]["config"]["seed"] for row in json.loads((out / "sources.json").read_text())} == {41, 42}


def _native_cf_fixture(monkeypatch, tmp_path, config):
    # Run the actual native terminal-cache branch without loading model weights.
    tree = ast.parse((ROOT / "experiments/Causal-Forcing/causal_forcing/pipeline/causal_inference.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CausalInferencePipeline")
    inference = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "inference")
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), inference], type_ignores=[])
    scope = {"torch": torch, "os": os, "tqdm": types.SimpleNamespace(tqdm=lambda values: values)}
    exec(compile(ast.fix_missing_locations(code), "native_cf_review_fixture", "exec"), scope)
    phases = []

    class Block(torch.nn.Identity):
        def __init__(self):
            super().__init__()
            self.self_attn = types.SimpleNamespace(num_heads=1, sink_size=0)

    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = types.SimpleNamespace(blocks=torch.nn.ModuleList([Block()]), local_attn_size=-1)

        def forward(self, noisy_image_or_video, conditional_dict, timestep, kv_cache=None, crossattn_cache=None, current_start=0):
            probe = self.model.blocks[0].self_attn.spectral_probe
            phases.append((current_start, probe.phase))
            probe.on_projection(0, k_pre=torch.ones(1, 1, 1, 8), v=torch.ones(1, 1, 1, 8),
                                grid_sizes=[[1, 1, 1]], current_start=current_start)
            return None, noisy_image_or_video

    class Base:
        inference = scope["inference"]

        def __init__(self):
            self.generator = Generator()
            self.vae = types.SimpleNamespace(decode_to_pixel=lambda values, **kwargs: values)
            self.denoising_step_list = torch.tensor([999])
            self.denoising_step_list_first_chunk = None
            self.independent_first_frame = False
            self.num_frame_per_block = self.num_transformer_blocks = self.frame_seq_length = 1
            self.kv_cache1 = self.crossattn_cache = None
            self.summary = {}
            self.args = types.SimpleNamespace(context_noise=0)
            self.text_encoder = lambda **kwargs: {}
            self.local_attn_size = -1

        def _initialize_kv_cache(self, **kwargs):
            self.kv_cache1 = []

        def _initialize_crossattn_cache(self, **kwargs):
            self.crossattn_cache = []

        def note_ephemeral_block(self, count): pass
        def note_clean_refresh(self): pass
        def reset_kv_cache_state(self): pass
        def discard_terminal_kv_cache(self, *args): pass
        def finalize_kv_cache(self, *args): pass
        def flush_kv_cache(self): pass

    prefix = "causal_forcing.pipeline"
    package, module = types.ModuleType(prefix), types.ModuleType(prefix + ".causal_inference")
    package.CausalInferencePipeline = module.CausalInferencePipeline = Base
    monkeypatch.setitem(sys.modules, prefix, package)
    monkeypatch.setitem(sys.modules, prefix + ".causal_inference", module)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"suite_id": "B_SPECTRAL_KV", "profile": "terminal_review", "layers": [0], "heads": [0], **config}))
    monkeypatch.setenv("SPECTRAL_KV_CONFIG", str(cfg))
    monkeypatch.setenv("SPECTRAL_KV_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CAUSAL_FORCING_BENCHMARK", "0")
    return install(ROOT / "experiments/Causal-Forcing/causal_forcing", "causal_forcing")(), phases


def _finish(pipe, frames=1, *, positional=False, **kwargs):
    noise = torch.ones(1, frames, 1, 1, 1)
    if positional:
        result = pipe.inference(noise, ["synthetic fixture"], None, False, False, False, False, False)
    else:
        result = pipe.inference(noise, ["synthetic fixture"], **kwargs)
    metadata = json.loads((pipe.spectral_probe.sink.root / "metadata.json").read_text())
    return result, metadata


@pytest.mark.parametrize("positional", [False, True])
def test_cf_single_chunk_readonly_collects_terminal_clean_and_records_added_work(tmp_path, monkeypatch, positional):
    pipe, phases = _native_cf_fixture(monkeypatch, tmp_path, {"experiments": ["B_E1_SPECTRUM"]})
    result, metadata = _finish(pipe, positional=positional)
    assert result.shape == (1, 1, 1, 1, 1)
    assert phases == [(0, "denoise"), (0, "clean")]
    assert pipe.summary["terminal_cache_event"]["action"] == "retained"
    assert metadata["row_counts"]["B_E1_SPECTRUM"] == 16
    terminal = metadata["terminal_clean_collection"]
    assert terminal["required_by_readonly_collection"] and terminal["retain_final_cache_effective"]
    assert terminal["additional_clean_requested"] and not terminal["requested_by_caller"]
    assert terminal["additional_clean_forward_calls"] == terminal["additional_clean_forwards_completed"] == 1
    assert metadata["host_section_ms"]["terminal_clean_collection"] > 0
    assert metadata["host_section_ms"]["generator_clean"] >= metadata["host_section_ms"]["terminal_clean_collection"]


@pytest.mark.parametrize("experiment,collect", [("B_E2B_REFRESH", False), ("B_E4_DISTANCE", False), ("B_E2B_REFRESH", True)])
def test_cf_other_clean_collectors_request_terminal_cache(tmp_path, monkeypatch, experiment, collect):
    pipe, phases = _native_cf_fixture(monkeypatch, tmp_path, {"experiments": [experiment], "collect_calibration": collect})
    _, metadata = _finish(pipe)
    assert phases[-1] == (0, "clean")
    assert metadata["terminal_clean_collection"]["additional_clean_forwards_completed"] == 1


@pytest.mark.parametrize("config", [
    {"experiments": ["B_E2_CHUNK_AXIS"]},
    {"experiments": ["B_E3_OFF_SHADOW"]},
    {"experiments": ["B_E3_ON_ROLLOUT"], "online": True, "method": "BF16"},
    {"experiments": ["B_E3_ON_ROLLOUT"], "online": True, "method": "SPECSUM_dct_R4_B1"},
    {"experiments": ["B_E2B_REFRESH"], "online": True, "method": "KVSKIP_raw"},
])
def test_cf_unneeded_or_online_runs_keep_native_terminal_choice(tmp_path, monkeypatch, config):
    pipe, phases = _native_cf_fixture(monkeypatch, tmp_path, config)
    _, metadata = _finish(pipe)
    assert phases == [(0, "denoise")]
    terminal = metadata["terminal_clean_collection"]
    assert not terminal["additional_clean_requested"] and not terminal["retain_final_cache_effective"]
    assert terminal["additional_clean_forward_calls"] == 0
    assert "terminal_clean_collection" not in metadata["host_section_ms"]


def test_cf_caller_retention_is_preserved_without_marking_it_additional(tmp_path, monkeypatch):
    pipe, phases = _native_cf_fixture(monkeypatch, tmp_path, {"experiments": ["B_E1_SPECTRUM"]})
    _, metadata = _finish(pipe, retain_final_cache=True)
    assert phases[-1] == (0, "clean")
    terminal = metadata["terminal_clean_collection"]
    assert terminal["requested_by_caller"] and terminal["retain_final_cache_effective"]
    assert not terminal["additional_clean_requested"] and terminal["additional_clean_forward_calls"] == 0


@pytest.mark.parametrize("limit", [{"max_clean_frames": 1}, {"max_chunks": 1}])
def test_cf_terminal_beyond_collection_limit_does_not_add_work(tmp_path, monkeypatch, limit):
    pipe, phases = _native_cf_fixture(monkeypatch, tmp_path, {"experiments": ["B_E1_SPECTRUM"], **limit})
    _, metadata = _finish(pipe, frames=2)
    assert phases == [(0, "denoise"), (0, "clean"), (1, "denoise")]
    assert not metadata["terminal_clean_collection"]["additional_clean_requested"]
    assert metadata["terminal_clean_collection"]["additional_clean_forward_calls"] == 0


def test_failed_run_does_not_change_completed_dc_summary(tmp_path):
    row={'record_kind':'spatial','layer':0,'head':0,'role':'v','coordinate':'raw','frame_idx':0,'dc_ratio':.25}
    _run(tmp_path/'input/good',[row],status='completed')
    _run(tmp_path/'input/failed',[{**row,'dc_ratio':1.}],status='failed')
    out=_analyzer().analyze(tmp_path/'input',tmp_path/'out',experiment='B_E1_SPECTRUM')
    dc=_csv(out/'B_E1_SPECTRUM/dc_summary.csv')
    assert len(dc)==1 and float(dc[0]['dc_ratio'])==.25
    selection=json.loads((out/'analysis_selection.json').read_text())
    assert selection['retained_runs']==1 and len(selection['excluded_incomplete_or_failed'])==1
