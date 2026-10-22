"""Synthetic declared entity/state features; not real video-quality findings."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from tempokv.research.memory_metrics import evaluate_reappearance, load_reappearance_bundle


SOURCE = {"extractor": "synthetic_declared_one_hot", "feature_space_id": "e5-fixture-v1",
          "feature_scope": "declared_entity_and_state", "data_origin": "synthetic_ground_truth"}


def fixture(*, identity_flip=False, stale_state=False):
    reference = np.zeros((8, 8), dtype=np.float64)
    reference[:, 0] = 1.; reference[:, 2] = 1.; reference[:, 4:] = 1e6
    reference[5:, 2] = 0.; reference[5:, 3] = 1.
    candidate = reference.copy()
    if identity_flip:
        candidate[5:, 0] = 0.; candidate[5:, 1] = 1.
    if stale_state:
        candidate[5:, 2] = 1.; candidate[5:, 3] = 0.
    annotations = [{"entity_id": "object-a", "first_frame": 0, "disappearance_frame": 2,
                    "reappearance_frame": 5, "feature_indices": [0, 1], "state_feature_indices": [2, 3],
                    "present_mask": [True, True, False, False, False, True, True, True]}]
    return reference, candidate, annotations


def evaluate(reference, candidate, annotations, **kwargs):
    return evaluate_reappearance(reference, candidate, annotations, source_metadata=SOURCE, **kwargs)


def test_identity_flip_and_state_change_are_separate_objective_consistency_metrics():
    ref, cand, annotations = fixture(identity_flip=True, stale_state=True)
    result = evaluate(ref, cand, annotations)
    row = result["rows"][0]
    assert row["reference_identity_cosine"] == 1. and row["reference_identity_relative_l2"] == 0.
    assert row["candidate_identity_cosine"] == 0.
    assert row["candidate_identity_relative_l2"] == pytest.approx(math.sqrt(2))
    assert row["matched_reappearance_identity_cosine"] == 0.
    assert row["matched_reappearance_identity_relative_l2"] == pytest.approx(math.sqrt(2))
    assert row["matched_first_identity_cosine"] == 1.
    # Higher first-to-reappearance state consistency can mean stale state.
    assert row["reference_state_cosine"] == 0. and row["candidate_state_cosine"] == 1.
    assert row["matched_reappearance_state_cosine"] == 0.
    assert row["matched_reappearance_state_relative_l2"] == pytest.approx(math.sqrt(2))
    assert row["status"] == "measured" and row["quality_status"] == "unknown"
    assert result["summary"]["quality_status"] == "unknown"
    assert result["summary"]["exploratory"] and result["summary"]["sequence_count"] == 1
    assert row["source_metadata_status"] == "declared"


def test_large_global_nuisance_cannot_hide_entity_flip_or_invalidate_target_features():
    ref, cand, annotations = fixture(identity_flip=True)
    global_cosine = float(ref[5] @ cand[5] / np.linalg.norm(ref[5]) / np.linalg.norm(cand[5]))
    assert global_cosine > .99999
    result = evaluate(ref, cand, annotations)
    assert result["rows"][0]["matched_reappearance_identity_cosine"] == 0.
    changed = cand.copy(); changed[:, 4:] = np.nan
    repeated = evaluate(ref, changed, annotations)
    assert repeated["rows"][0]["matched_reappearance_identity_cosine"] == 0.
    assert repeated["rows"][0]["status"] == "measured"
    # Absent-frame feature values are not substituted for a valid reappearance.
    ref[2:5, :4] = np.nan; cand[2:5, :4] = np.inf
    assert evaluate(ref, cand, annotations)["rows"][0]["status"] == "measured"


def test_state_anchor_measures_persistent_state_established_before_disappearance():
    ref = np.zeros((14, 4)); ref[:, 0] = 1.; ref[:4, 2] = 1.; ref[4:, 3] = 1.
    annotations = [{"entity_id": "interacted-object", "first_frame": 0, "disappearance_frame": 8,
                    "reappearance_frame": 11, "state_reference_frame": 7,
                    "feature_indices": [0, 1], "state_feature_indices": [2, 3]}]
    row = evaluate(ref, ref.copy(), annotations)["rows"][0]
    assert row["state_reference_frame"] == 7 and row["reference_state_cosine"] == 1.
    assert row["candidate_state_relative_l2"] == row["matched_state_anchor_relative_l2"] == 0.
    assert row["matched_reappearance_state_cosine"] == 1.
    stale = ref.copy(); stale[11, 2:4] = [1., 0.]
    stale_row = evaluate(ref, stale, annotations)["rows"][0]
    assert stale_row["candidate_state_cosine"] == stale_row["matched_reappearance_state_cosine"] == 0.
    assert stale_row["matched_reappearance_identity_cosine"] == 1.
    without_anchor = [{key: value for key, value in annotations[0].items() if key != "state_reference_frame"}]
    assert evaluate(ref, ref, without_anchor)["rows"][0]["reference_state_cosine"] == 0.


@pytest.mark.parametrize("with_mask", [False, True])
def test_static_control_uses_aligned_anchors_without_claiming_disappearance(with_mask):
    ref, cand, annotations = fixture()
    event = {**annotations[0], "event_kind": "static_control", "actual_disappearance": False}
    event.pop("present_mask")
    if with_mask: event["present_mask"] = [True] * len(ref)
    row = evaluate(ref, cand, [event])["rows"][0]
    assert row["event_kind"] == "static_control" and row["actual_disappearance"] is False
    assert row["candidate_absent_through_gap"] is None
    assert row["matched_reappearance_identity_cosine"] == 1.


def test_candidate_absent_at_reappearance_is_not_a_zero_feature_identity_measurement():
    ref, cand, annotations = fixture()
    reference_mask = annotations[0]["present_mask"]
    candidate_mask = list(reference_mask); candidate_mask[5] = False
    event = {**annotations[0], "present_mask": {"reference": reference_mask, "candidate": candidate_mask}}
    cand[5, :4] = 0
    result = evaluate(ref, cand, [event]); row = result["rows"][0]
    assert row["candidate_identity_status"] == row["matched_reappearance_identity_status"] == "entity_absent"
    assert row["candidate_identity_cosine"] is row["matched_reappearance_identity_relative_l2"] is None
    assert row["reference_identity_status"] == "measured" and row["status"] == "partially_measured"
    assert row["candidate_present_at_reappearance"] is False
    assert result["summary"]["measured_sequence_count"] == 0
    assert result["summary"]["status"] == "not_measured" and result["summary"]["quality_status"] == "unknown"


def test_candidate_failure_to_disappear_is_separately_recorded_from_identity():
    ref, cand, annotations = fixture()
    event = {**annotations[0], "present_mask": {"reference": annotations[0]["present_mask"], "candidate": [True] * 8}}
    row = evaluate(ref, cand, [event])["rows"][0]
    assert row["candidate_absent_through_gap"] is False
    assert row["matched_reappearance_identity_cosine"] == 1.
    assert row["quality_status"] == "unknown"


@pytest.mark.parametrize("origin,destination,status,relative", [([0., 0.], [1., 0.], "zero_origin_feature", None),
    ([1., 0.], [0., 0.], "zero_destination_feature", 1.), ([0., 0.], [0., 0.], "zero_both_features", None),
    ([float("nan"), 0.], [1., 0.], "nonfinite_feature", None),
    ([1., 0.], [float("inf"), 0.], "nonfinite_feature", None)])
def test_zero_and_nonfinite_selected_features_have_explicit_status(origin, destination, status, relative):
    ref, cand, annotations = fixture()
    cand[0, :2] = origin; cand[5, :2] = destination
    row = evaluate(ref, cand, annotations)["rows"][0]
    assert row["candidate_identity_cosine"] is None
    assert row["candidate_identity_relative_l2"] == relative
    assert row["candidate_identity_status"] == status
    assert row["status"] == "partially_measured"
    json.dumps(row, allow_nan=False)


def test_finite_large_features_avoid_spurious_cosine_overflow():
    ref, cand, annotations = fixture()
    ref[:, :4] *= 1e300; cand[:, :4] *= 1e300
    row = evaluate(ref, cand, annotations)["rows"][0]
    assert row["reference_identity_cosine"] == row["matched_reappearance_identity_cosine"] == 1.
    assert row["matched_reappearance_identity_relative_l2"] == 0.


def test_distinct_sequence_threshold_never_counts_many_entities_as_sequences():
    ref, cand, annotations = fixture()
    same_sequence = [{**annotations[0], "entity_id": f"object-{i}"} for i in range(20)]
    summary = evaluate(ref, cand, same_sequence)["summary"]
    assert summary["event_count"] == 20 and summary["sequence_count"] == summary["measured_sequence_count"] == 1
    assert summary["exploratory"]
    distinct = [{**annotations[0], "sequence_id": f"sequence-{i}"} for i in range(20)]
    summary = evaluate(ref, cand, distinct)["summary"]
    assert summary["sequence_count"] == summary["measured_sequence_count"] == 20
    assert not summary["exploratory"] and summary["evidence_level"] == "consistency_only"
    assert summary["quality_status"] == "unknown"


def test_summary_weights_sequences_equally_instead_of_repeated_events():
    ref = np.tile(np.array([1., 0.]), (20, 1)); cand = ref.copy(); cand[11] = [-1., 0.]
    annotations = [{"sequence_id": sequence, "entity_id": "object", "first_frame": first,
                    "disappearance_frame": first + 2, "reappearance_frame": first + 5, "feature_indices": [0, 1]}
                   for sequence, first in (("a", 0), ("a", 6), ("b", 12))]
    result = evaluate(ref, cand, annotations, min_sequences=2)
    metric = result["summary"]["metrics"]["matched_reappearance_identity_relative_l2"]
    assert metric == {"mean": .5, "median": .5, "valid_events": 3, "valid_sequences": 2}
    assert not result["summary"]["exploratory"]


def test_no_state_features_no_annotations_and_unknown_source_remain_explicit():
    ref, cand, annotations = fixture()
    event = {key: value for key, value in annotations[0].items() if key != "state_feature_indices"}
    result = evaluate_reappearance(ref, cand, [event])
    row = result["rows"][0]
    assert row["state_status"] == "not_supplied" and "reference_state_cosine" not in row
    assert row["source_metadata_status"] == "unknown" and row["quality_status"] == "unknown"
    empty = evaluate(ref, cand, [])
    assert empty["rows"] == [] and empty["summary"]["status"] == "not_measured"
    assert empty["summary"]["sequence_count"] == 0 and empty["summary"]["metrics"] == {}


def test_declared_feature_source_wrappers_and_rng_input_immutability():
    ref, cand, annotations = fixture()
    original_ref, original_cand = ref.copy(), cand.copy()
    torch_state = torch.get_rng_state().clone(); numpy_state = np.random.get_state()
    result = evaluate_reappearance({"features": torch.from_numpy(ref), "source_metadata": SOURCE},
        {"features": torch.from_numpy(cand), "source_metadata": SOURCE}, annotations)
    assert result["summary"]["source_metadata"] == {"reference": SOURCE, "candidate": SOURCE}
    assert result["summary"]["source_metadata_status"] == "declared"
    assert np.array_equal(original_ref, ref) and np.array_equal(original_cand, cand)
    assert torch.equal(torch_state, torch.get_rng_state())
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and np.array_equal(numpy_state[1], after[1]) and numpy_state[2:] == after[2:]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("updates", [{"first_frame": -1}, {"first_frame": 2}, {"reappearance_frame": 2},
    {"reappearance_frame": 8}, {"disappearance_frame": 2.0}, {"first_frame": True}, {"entity_id": ""},
    {"feature_indices": []}, {"feature_indices": [0, 0]}, {"feature_indices": [8]}, {"feature_indices": [0.]},
    {"state_feature_indices": [1, 2]}, {"state_reference_frame": 2}, {"state_reference_frame": -1},
    {"state_reference_frame": 1.0}, {"actual_disappearance": 0}, {"present_mask": [1] * 8},
    {"present_mask": [True] * 8}, {"present_mask": [True] * 7}])
def test_invalid_events_dimensions_absence_order_and_anchor_rejected(updates):
    ref, cand, annotations = fixture()
    with pytest.raises(ValueError):
        evaluate(ref, cand, [{**annotations[0], **updates}])


def test_same_entity_events_cannot_overlap_but_other_entities_may_share_frames():
    ref, cand, annotations = fixture()
    with pytest.raises(ValueError, match="overlap"):
        evaluate(ref, cand, annotations * 2)
    result = evaluate(ref, cand, [annotations[0], {**annotations[0], "entity_id": "object-b"}])
    assert len(result["rows"]) == 2


def test_invalid_features_source_space_and_threshold_rejected():
    ref, cand, annotations = fixture()
    for bad in (np.empty((0, 8)), np.empty((8, 0)), np.zeros((8, 8, 1)), np.ones((8, 8), dtype=bool),
                np.ones((8, 8), dtype=complex), np.array([["feature"]])):
        with pytest.raises(ValueError):
            evaluate(bad, cand, annotations)
    with pytest.raises(ValueError, match="matching"):
        evaluate(ref, cand[:-1], annotations)
    with pytest.raises(ValueError, match="feature_space_id"):
        evaluate_reappearance({"features": ref, "source_metadata": {"feature_space_id": "a"}},
                              {"features": cand, "source_metadata": {"feature_space_id": "b"}}, annotations)
    for minimum in (0, True, 1.0):
        with pytest.raises(ValueError, match="min_sequences"):
            evaluate(ref, cand, annotations, min_sequences=minimum)


@pytest.mark.parametrize("suffix", ["json", "npz"])
def test_load_json_or_npz_feature_annotation_bundle_without_pickle(tmp_path, suffix):
    ref, cand, annotations = fixture()
    path = tmp_path / f"features.{suffix}"
    if suffix == "json":
        path.write_text(json.dumps({"reference": ref.tolist(), "candidate": cand.tolist(),
                                   "annotations": annotations, "source_metadata": SOURCE}))
    else:
        np.savez(path, reference=ref, candidate=cand, annotations_json=np.array(json.dumps(annotations)),
                 source_metadata_json=np.array(json.dumps(SOURCE)))
    loaded = load_reappearance_bundle(path)
    result = evaluate_reappearance(**loaded)
    assert result["rows"][0]["matched_reappearance_identity_cosine"] == 1.
    assert result["summary"]["source_metadata_status"] == "declared"


def test_npz_rejects_object_payload_and_malformed_bundles(tmp_path):
    ref, cand, _ = fixture()
    path = tmp_path / "unsafe.npz"
    np.savez(path, reference=ref, candidate=cand, annotations_json=np.array([{"entity_id": "x"}], dtype=object))
    with pytest.raises(ValueError):
        load_reappearance_bundle(path)
    malformed = tmp_path / "missing.json"; malformed.write_text("{}")
    with pytest.raises(ValueError, match="requires"):
        load_reappearance_bundle(malformed)
