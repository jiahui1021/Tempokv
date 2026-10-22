"""Saved-record reconstruction checks; no generation or GPU is involved."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tempokv.research.analysis import analyze, load_rows, summarize_rows
from tempokv.research.io import SCHEMA


def write_run(root, name, rows, *, signature="config-a", origin="synthetic_fixture", prompt=0, seed=0, schema="research.v1"):
    folder = root / name
    folder.mkdir(parents=True)
    manifest = {"schema_version": schema, "config_signature": signature, "data_origin": origin,
                "prompt_index": prompt, "seed": seed, "model": "fixture-model", "checkpoint": "fixture-checkpoint",
                "config": {"source": "bf16", "local_attn_size": 24, "sink_size": 0}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    by_experiment = {}
    for row in rows:
        experiment = row["experiment_id"]
        by_experiment.setdefault(experiment, []).append(row)
    for experiment, entries in by_experiment.items():
        target = folder / experiment
        target.mkdir()
        with (target / "events-fixture.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCHEMA)
            writer.writeheader()
            for row in entries:
                data = {"schema_version": "research.v1", "data_origin": origin, "status": "measured", **row}
                extras = {key: value for key, value in data.items() if key not in SCHEMA}
                data["details_json"] = json.dumps(extras)
                writer.writerow({key: data.get(key) for key in SCHEMA})
    return folder


@pytest.fixture
def mixed_records(tmp_path):
    root = tmp_path / "records"
    records = [
        {"experiment_id": "S1", "event_id": "spatial-k", "role": "k", "layer": 0, "head": 0,
         "score_type": "spatial_low_energy", "score": .25, "cutoff": .25, "metric_scope": "clean_dense_spatial"},
        {"experiment_id": "S1", "event_id": "spatial-v", "role": "v", "layer": 0, "head": 0,
         "score_type": "spatial_low_energy", "score": .75, "cutoff": .25, "metric_scope": "clean_dense_spatial"},
        {"experiment_id": "S1", "event_id": "temporal", "role": "k", "score_type": "temporal_low_energy",
         "score": .5, "cutoff": .25, "temporal_window": 8, "temporal_control": "ordered"},
        {"experiment_id": "S2", "event_id": "candidate", "score_type": "frequency", "score": .1,
         "local_attention_relative_error": .2, "metric_scope": "attention_local"},
        {"experiment_id": "S2", "event_id": "correlation", "score_type": "frequency",
         "spearman": -.5, "topk_recall_sampled": 0., "statistic": "candidate_rank_correlation"},
        {"experiment_id": "E1", "event_id": "e1-a", "head": 0, "strategy": "drop", "local_attention_relative_error": 0.},
        {"experiment_id": "E1", "event_id": "e1-b", "head": 1, "strategy": "drop", "local_attention_relative_error": 2.},
        {"experiment_id": "E3", "event_id": "e3-drop", "strategy": "drop", "local_attention_relative_error": .2},
        {"experiment_id": "E3", "event_id": "e3-merge", "strategy": "weighted_merge", "local_attention_relative_error": .3},
        {"experiment_id": "E3", "event_id": "e3-condition", "strategy": "conditional_drop",
         "conditioning_set": "[1]", "local_attention_relative_error": .4},
        {"experiment_id": "E4", "event_id": "e4-noise", "role": "k", "strength": 0.,
         "strategy": "matched_relative_kv_rmse", "local_attention_relative_error": 0.},
        {"experiment_id": "E4", "event_id": "e4-quant", "k_bits": 4, "v_bits": 2, "logical_bytes": 128,
         "strategy": "fake_quant_grid", "local_attention_relative_error": .04},
        {"experiment_id": "B_E1_SPECTRUM", "event_id": "legacy-row", "score": 999.},
    ]
    first = write_run(root, "synthetic-p0", records)
    # A resumed part repeats a saved event, not another independent observation.
    original = first / "E1" / "events-fixture.csv"
    (original.parent / "events-resumed.csv").write_bytes(original.read_bytes())
    write_run(root, "synthetic-p1", [{"experiment_id": "E1", "event_id": "e1-c", "strategy": "drop", "local_attention_relative_error": 5.}], prompt=1, seed=1)
    write_run(root, "other-config", [{"experiment_id": "E1", "event_id": "e1-a", "strategy": "drop", "local_attention_relative_error": 100.}], signature="config-b")
    write_run(root, "real-fixture", [{"experiment_id": "E1", "event_id": "e1-a", "strategy": "drop", "local_attention_relative_error": 8.}], origin="runtime_bf16")
    write_run(root, "legacy-manifest", [{"experiment_id": "E1", "event_id": "legacy-manifest", "local_attention_relative_error": 999.}], schema="spectral.v1")
    return root


@pytest.fixture
def missing_clean_records(tmp_path):
    root = tmp_path / "missing-clean"
    write_run(root, "synthetic-empty-history", [
        {"experiment_id": "E1", "event_id": "identity", "strategy": "null", "local_attention_relative_error": 0.,
         "model_output_relative_error": None, "clean_latent_relative_error": None},
        {"experiment_id": "S1", "event_id": "no-clean", "status": "not_measured", "reason": "no_clean_data"},
    ])
    return root


def test_ingestion_cohorts_clustered_statistics_and_m1_figures(mixed_records, tmp_path):
    rows = load_rows(mixed_records)
    assert len(rows) == 15
    assert {row["experiment_id"] for row in rows} == {"S1", "S2", "E1", "E3", "E4"}
    assert len({row["cohort_id"] for row in rows}) == 3
    assert sum(row["event_id"] == "e1-a" for row in rows) == 3
    correlation = next(row for row in rows if row["event_id"] == "correlation")
    assert correlation["spearman"] == -.5 and correlation["topk_recall_sampled"] == 0
    assert isinstance(correlation["details_json"], dict)
    assert next(row for row in rows if row["event_id"] == "spatial-k")["cutoff"] == .25
    summary = summarize_rows(rows, bootstrap=100)
    cohort = next(entry for entry in summary["cohorts"] if entry["data_origin"] == "synthetic_fixture" and entry["config_signature"] == "config-a")
    stat = next(entry for entry in summary["statistics"] if entry["cohort_id"] == cohort["cohort_id"] and entry["experiment_id"] == "E1")
    assert stat["metric"] == "local_attention_relative_error"
    assert stat["mean"] == 3 and stat["median"] == 3 and stat["std"] == 2
    assert stat["independent_prompt_seed_units"] == 2 and stat["raw_observations"] == 3
    assert summary["completed_experiments"] == ["E1"]
    assert summary["synthetic_completed_experiments"] == ["E1", "E3", "E4", "S1", "S2"]
    assert all(value is None for value in summary["quality_metrics"].values())
    output = analyze(mixed_records, tmp_path / "artifacts")
    rebuilt = json.loads((output / "summary.json").read_text())
    assert rebuilt["ingestion"]["duplicate_event_ids"] == 2
    assert rebuilt["ingestion"]["excluded_non_research_manifests"] == 1
    with (output / "merged.csv").open() as handle:
        merged = list(csv.DictReader(handle))
    assert len(merged) == 15 and "spearman" in merged[0]
    assert all((output / f"{experiment}.csv").exists() for experiment in ("S1", "S2", "S3", "E1", "E2", "E3", "E4", "E5", "E6", "E7"))
    names = {item["name"] for item in rebuilt["figures"] if item["cohort_id"] == cohort["cohort_id"] and item["status"] == "measured"}
    assert names == {"frequency_energy_by_layer_head", "spatial_vs_temporal", "score_vs_attention_oracle",
                     "local_vs_generation_importance", "conditional_substitutability", "kv_sensitivity_and_bit_budget"}
    for figure in rebuilt["figures"]:
        for path in figure.get("paths", []):
            assert (output / path).stat().st_size > 100
    report = (output / "research_summary.md").read_text()
    assert "support no scientific conclusion" in report and "not lossless" in report


def test_missing_clean_generation_and_repeat_rebuild_do_not_overwrite(missing_clean_records, tmp_path, monkeypatch):
    from tempokv.research import attention_oracle
    def no_generation(*args, **kwargs):
        raise AssertionError("saved-record analysis must not evaluate attention or generate")
    monkeypatch.setattr(attention_oracle, "reference", no_generation)
    base = tmp_path / "existing-output"
    base.mkdir()
    marker = base / "user-content.txt"
    marker.write_text("preserve")
    first = analyze(missing_clean_records, base)
    original = (first / "summary.json").read_bytes()
    second = analyze(missing_clean_records, base)
    assert first != second and first.parent == second.parent == base
    assert marker.read_text() == "preserve" and (first / "summary.json").read_bytes() == original
    summary = json.loads((second / "summary.json").read_text())
    assert summary["completed_experiments"] == [] and summary["synthetic_completed_experiments"] == ["E1"]
    assert "S1" in summary["unrun_experiments"] and summary["measured_fields"]["S1"] == []
    assert "model_output_relative_error" not in summary["measured_fields"]["E1"]
    assert summary["quality_metrics"] == {name: None for name in ("PSNR", "SSIM", "LPIPS", "FID", "VBench")}
    s1 = [entry for entry in summary["figures"] if entry["experiment_id"] == "S1"]
    assert s1 and all(entry["status"] == "not_measured" and not entry.get("paths") for entry in s1)
    stat = next(entry for entry in summary["statistics"] if entry["metric"] == "local_attention_relative_error")
    assert stat["mean"] == 0 and stat["independent_prompt_seed_units"] == 1 and stat["ci_low"] is None
    assert "generation outcomes and quality scores remain unmeasured" in (second / "research_summary.md").read_text()


def test_propagation_plots_keep_dose_step_head_separate_and_exclude_unmatched():
    from tempokv.research.analysis import propagation_partitions
    base={'intervention':'gaussian','role':'v','requested_strength':.1,'calibration_protocol':'local_d_att',
          'denoise_step_index':0,'head':0,'calibration_matched':True,'comparison_cohort':'matched'}
    rows=[{**base,'layer':0},{**base,'layer':1},{**base,'requested_strength':.2},
          {**base,'denoise_step_index':1},{**base,'head':1},
          {**base,'calibration_matched':False},{**base,'comparison_cohort':'unmatched_dtype_floor'}]
    groups=propagation_partitions(rows)
    assert sorted(len(part) for _,part in groups)==[1,1,1,2]
    assert len({identity for identity,_ in groups})==4
    assert all(row['calibration_matched'] and row['comparison_cohort']=='matched' for _,part in groups for row in part)
