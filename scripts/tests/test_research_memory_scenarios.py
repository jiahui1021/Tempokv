"""Small E5 CPU fixtures and an explicit, evidence-backed exploration gate."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from tempokv.research.memory_scenarios import (
    SCENARIO_NAMES, heavy_benchmark_gate, load_evidence_json, make_scenario,
)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_small_deterministic_reappearance_fixture_and_matched_static_control(name):
    data = make_scenario(name, seed=7)
    again = make_scenario(name, seed=7)
    features = data["reference_features"]
    control = data["control"]["reference_features"]
    assert features.shape == control.shape == (24, 12) and features.dtype == np.float32
    np.testing.assert_array_equal(features, again["reference_features"])
    assert not np.array_equal(features, make_scenario(name, seed=8)["reference_features"])
    assert np.isfinite(features).all() and np.isfinite(control).all()
    assert data["data_origin"] == data["control"]["data_origin"] == "synthetic_e5_cpu"
    assert data["feature_fixture_only"] and not data["generated_video"] and data["quality_metrics"] is None
    assert data["prompt"] != data["control"]["prompt"]
    assert isinstance(data["annotations"], list) and len(data["annotations"]) == 1
    annotation = data["annotations"][0]
    assert {"entity_id", "first_frame", "disappearance_frame", "reappearance_frame", "feature_indices", "sequence_id"} <= annotation.keys()
    assert (annotation["first_frame"], annotation["disappearance_frame"], annotation["reappearance_frame"]) == (0, 8, 16)
    assert annotation["actual_disappearance"] and annotation["present_mask"] == [True] * 8 + [False] * 8 + [True] * 8
    identity = annotation["feature_indices"]
    np.testing.assert_array_equal(features[0, identity], features[16, identity])
    assert not features[8:16, identity].any()
    np.testing.assert_array_equal(control, np.repeat(control[:1], 24, axis=0))
    control_annotation = data["control"]["annotations"][0]
    assert control_annotation["event_kind"] == "static_control" and not control_annotation["actual_disappearance"]
    assert "present_mask" not in control_annotation and control_annotation["sequence_id"] != annotation["sequence_id"]
    assert not np.shares_memory(features, control)
    if name == "interaction_persistence":
        assert features[0, 9] == 0 and features[4, 9] == features[16, 9] == 1
        assert annotation["state_feature_indices"] == [9] and annotation["state_reference_frame"] == 7


def test_scenario_validation_and_outputs_do_not_share_across_calls():
    a, b = make_scenario("object_exit_return"), make_scenario("object_exit_return")
    a["reference_features"][:] = 100
    assert not np.array_equal(a["reference_features"], b["reference_features"])
    assert make_scenario("character_exit_return")["scenario_kind"] == "object_exit_return"
    with pytest.raises(ValueError): make_scenario("unknown")
    with pytest.raises(ValueError): make_scenario("camera_return", seed=-1)
    with pytest.raises(ValueError): make_scenario("camera_return", seed=True)


@pytest.fixture
def evidence():
    shared = {"schema_version": "research.v1", "data_origin": "runtime_bf16", "source": "bf16",
              "config_signature": "config-a", "model": "declared-runtime-model", "checkpoint": "checkpoint-a",
              "window": {"local_attn_size": 24, "sink_size": 0}}
    rows = []
    for prompt in range(2):
        base = {"prompt_index": prompt, "seed": 3, "paired_run_id": f"pair-{prompt}", "paired_rng": True,
                "status": "measured", "layer": prompt}
        rows += [{**base, "experiment_id": "E1", "event_id": f"e1-{prompt}", "model_output_relative_error": 0.,
                  "local_attention_relative_error": prompt / 10},
                 {**base, "experiment_id": "E2", "event_id": f"e2-{prompt}", "clean_latent_relative_error": 0., "horizon_chunks": 2}]
    rows.append({"experiment_id": "S2", "event_id": "negative-spectrum", "prompt_index": 0, "seed": 3,
                 "status": "measured", "score_type": "frequency", "spearman": -.8})
    return {"evidence_metadata": shared, "records": rows,
            "decision": {"decision": "explore_memory",
                         "rationale": "Paired future-horizon measurements motivate an explicit return-event check, even with zero effects in these measurements.",
                         "supporting_event_ids": [row["event_id"] for row in rows],
                         "criteria": [{"kind": "reappearance_rationale", "min_horizon_chunks": 2,
                                       "scenario_names": ["camera_return", "interaction_persistence"]}]}}


def test_real_declared_evidenced_decision_accepts_zero_effects_and_negative_frequency(evidence):
    result = heavy_benchmark_gate(evidence)
    assert result["allowed"] and result["reasons"] == []
    assert result["independent_prompt_seed_pairs"] == 2
    assert result["criteria_results"][0]["requires_positive_effect"] is False
    assert result["positive_spectral_finding_required"] is False
    assert result["quality_conclusion"] is None
    assert result["scope"] == "eligibility_for_exploratory_E5_only"
    assert result["decision"] == evidence["decision"]["decision"]
    assert result["declared_rationale"] == evidence["decision"]["rationale"]


def test_coverage_or_measured_heterogeneity_are_declared_criteria_not_frequency_requirements(evidence):
    evidence["decision"]["criteria"] = [{"kind": "paired_horizon_coverage", "min_horizon_chunks": 1},
                                        {"kind": "measured_heterogeneity", "metric": "local_attention_relative_error",
                                         "group_by": "layer", "min_range": .05}]
    result = heavy_benchmark_gate(evidence)
    assert result["allowed"] and result["criteria_results"][1]["observed_range"] == pytest.approx(.1)
    evidence["decision"]["criteria"][1]["min_range"] = .2
    result = heavy_benchmark_gate(evidence)
    assert not result["allowed"] and "declared_measurable_criteria_not_met" in result["reasons"]


@pytest.mark.parametrize("origin", ["synthetic_e5_cpu", "synthetic_m3", "unknown", "real", ""])
def test_synthetic_or_unknown_provenance_never_unlocks_heavy_run(evidence, origin):
    evidence["evidence_metadata"]["data_origin"] = origin
    result = heavy_benchmark_gate(evidence)
    assert not result["allowed"] and result["accepted_record_count"] == 0
    assert result["quality_conclusion"] is None


def test_empty_fake_quality_only_and_missing_decision_are_rejected(evidence):
    assert not heavy_benchmark_gate([])["allowed"]
    assert not heavy_benchmark_gate({})["allowed"]
    assert not heavy_benchmark_gate(evidence["records"])["allowed"]
    no_decision = copy.deepcopy(evidence); no_decision.pop("decision")
    assert "explicit_explore_memory_decision_required" in heavy_benchmark_gate(no_decision)["reasons"]
    fake = copy.deepcopy(evidence)
    fake["evidence_metadata"] = {"data_origin": "runtime_bf16", "source": "bf16", "is_real": True}
    assert not heavy_benchmark_gate(fake)["allowed"]
    quality = copy.deepcopy(evidence)
    for row in quality["records"]:
        for name in ("model_output_relative_error", "local_attention_relative_error", "clean_latent_relative_error", "spearman"):
            row.pop(name, None)
        row["PSNR"] = 100
    result = heavy_benchmark_gate(quality)
    assert not result["allowed"] and result["quality_conclusion"] is None


def test_support_ids_pairing_settings_and_independent_units_are_required(evidence):
    wrong = copy.deepcopy(evidence); wrong["decision"]["supporting_event_ids"].append("nonexistent")
    assert not heavy_benchmark_gate(wrong)["allowed"]
    one_unit = copy.deepcopy(evidence)
    for row in one_unit["records"]: row["prompt_index"] = 0
    assert heavy_benchmark_gate(one_unit)["independent_prompt_seed_pairs"] == 1
    assert not heavy_benchmark_gate(one_unit)["allowed"]
    mixed = copy.deepcopy(evidence)
    for row in mixed["records"]:
        if row["prompt_index"] == 1: row["config_signature"] = "config-b"
    assert "supporting_evidence_must_belong_to_one_explicit_cohort" in heavy_benchmark_gate(mixed)["reasons"]
    unpaired = copy.deepcopy(evidence); unpaired["records"][3]["paired_rng"] = False
    assert not heavy_benchmark_gate(unpaired)["allowed"]
    unmatched = copy.deepcopy(evidence); unmatched["records"][3]["paired_run_id"] = "different-pair"
    assert not heavy_benchmark_gate(unmatched)["allowed"]
    no_criteria = copy.deepcopy(evidence); no_criteria["decision"]["criteria"] = []
    assert "declared_measurable_criteria_required" in heavy_benchmark_gate(no_criteria)["reasons"]


def test_safe_local_json_loader_rejects_nonfinite_duplicate_and_oversized_inputs(tmp_path, evidence):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    assert load_evidence_json(path) == evidence and heavy_benchmark_gate(path)["allowed"]
    with pytest.raises(ValueError, match="max_bytes"): load_evidence_json(path, max_bytes=5)
    bad = tmp_path / "duplicate.json"; bad.write_text('{"records":[],"records":[1]}')
    with pytest.raises(ValueError, match="duplicate JSON key"): load_evidence_json(bad)
    assert not heavy_benchmark_gate(bad)["allowed"]
    bad.write_text('{"records":[],"score":NaN}')
    with pytest.raises(ValueError, match="nonfinite"): load_evidence_json(bad)
    assert not heavy_benchmark_gate(bad)["allowed"]
    bad.write_text('"__import__(\\"os\\").system(\\"false\\")"')
    assert not heavy_benchmark_gate(bad)["allowed"]
