"""Objective E5 consistency on caller-supplied entity/state feature sequences.

No visual encoder, entity detector, or video-quality estimator is implemented
here. Feature dimensions and absence/reappearance events are declared by the
caller. A state may legitimately change while an entity is absent; therefore
first-to-reappearance consistency and matched-reference consistency are always
reported separately, without a quality/pass/fail interpretation.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import torch


def _array(value):
    if isinstance(value, torch.Tensor):
        if value.is_complex() or value.dtype == torch.bool:
            raise ValueError("features require real numeric values")
        value = value.detach().double().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.ndim != 2 or min(array.shape) < 1:
        raise ValueError("features must be a nonempty real numeric [frames, feature_dim] array")
    # Nonfinite selected features are handled per comparison. Unused nuisance
    # dimensions and absent frames must not invalidate an otherwise valid event.
    return array.astype(np.float64, copy=False)


def _metadata(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("source_metadata must be a mapping describing the supplied features")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("source_metadata must contain finite JSON values") from error


def _features(value):
    metadata = {}
    if isinstance(value, Mapping):
        if "features" not in value:
            raise ValueError("feature mappings need a features array")
        metadata = _metadata(value.get("source_metadata"))
        value = value["features"]
    return _array(value), metadata


def _identifier(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)) or isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} must be a nonempty string or integer")
    return int(value) if isinstance(value, Integral) else value


def _indices(value, width, name):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    try:
        values = list(value)
    except TypeError as error:
        raise ValueError(f"{name} must contain feature indices") from error
    if not values or any(isinstance(index, bool) or not isinstance(index, Integral) for index in values):
        raise ValueError(f"{name} must contain nonempty integer feature indices")
    values = [int(index) for index in values]
    if len(values) != len(set(values)) or any(index < 0 or index >= width for index in values):
        raise ValueError(f"{name} must contain unique in-range feature indices")
    return values


def _mask(value, frames):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.dtype.kind != "b" or value.shape != (frames,):
        raise ValueError("present_mask must contain one boolean per frame")
    return value


def _events(annotations, frames, width):
    if isinstance(annotations, (Mapping, str, bytes)):
        raise ValueError("annotations must be a sequence of event mappings")
    result = []; intervals = defaultdict(list)
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError("each annotation must be an event mapping")
        entity = _identifier(annotation.get("entity_id"), "entity_id")
        sequence = _identifier(annotation.get("sequence_id", "sequence-0"), "sequence_id")
        times = [annotation.get(name) for name in ("first_frame", "disappearance_frame", "reappearance_frame")]
        if any(isinstance(frame, bool) or not isinstance(frame, Integral) for frame in times):
            raise ValueError("event frames must be integers")
        first, disappearance, reappearance = (int(frame) for frame in times)
        if not 0 <= first < disappearance < reappearance < frames:
            raise ValueError("require 0 <= first_frame < disappearance_frame < reappearance_frame < frames")
        identity = _indices(annotation.get("feature_indices"), width, "feature_indices")
        state = annotation.get("state_feature_indices")
        state = None if state is None else _indices(state, width, "state_feature_indices")
        if state is not None and set(identity) & set(state):
            raise ValueError("identity and state feature indices must not overlap")
        state_anchor = annotation.get("state_reference_frame", first)
        if isinstance(state_anchor, bool) or not isinstance(state_anchor, Integral) or not 0 <= state_anchor < disappearance:
            raise ValueError("state_reference_frame must be an integer before disappearance_frame")
        actual_disappearance = annotation.get("actual_disappearance", True)
        if not isinstance(actual_disappearance, bool):
            raise ValueError("actual_disappearance must be boolean")
        event_kind = annotation.get("event_kind", "disappearance_reappearance" if actual_disappearance else "static_control")
        if not isinstance(event_kind, str) or not event_kind.strip():
            raise ValueError("event_kind must be a nonempty string")
        presence = annotation.get("present_mask")
        if isinstance(presence, Mapping):
            if set(presence) - {"reference", "candidate"}:
                raise ValueError("present_mask mapping only accepts reference and candidate")
            ref_mask = _mask(presence.get("reference"), frames)
            cand_mask = _mask(presence.get("candidate"), frames)
        else:
            ref_mask = cand_mask = _mask(presence, frames)
        if ref_mask is not None:
            valid_presence = (bool(ref_mask[first:disappearance].all()) and not bool(ref_mask[disappearance:reappearance].any())
                              and bool(ref_mask[reappearance])) if actual_disappearance else bool(ref_mask[first:reappearance+1].all())
            if not valid_presence:
                raise ValueError("reference present_mask must confirm the declared disappearance or static-control event order")
        event = {"sequence_id": sequence, "entity_id": entity, "first_frame": first,
                 "disappearance_frame": disappearance, "reappearance_frame": reappearance,
                 "state_reference_frame": int(state_anchor), "actual_disappearance": actual_disappearance, "event_kind": event_kind,
                 "feature_indices": identity, "state_feature_indices": state,
                 "reference_mask": ref_mask, "candidate_mask": cand_mask}
        result.append(event)
        intervals[(type(sequence).__name__, sequence, type(entity).__name__, entity)].append((first, reappearance))
    for events in intervals.values():
        ordered = sorted(events)
        if any(second[0] <= first[1] for first, second in zip(ordered, ordered[1:])):
            raise ValueError("reappearance events for the same sequence/entity must not overlap")
    return result


def _comparison(origin, destination, *, present=True):
    if not present:
        return {"cosine": None, "relative_l2": None, "status": "entity_absent"}
    if not np.isfinite(origin).all() or not np.isfinite(destination).all():
        return {"cosine": None, "relative_l2": None, "status": "nonfinite_feature"}
    # Scaling first avoids overflow when valid finite feature magnitudes are
    # large, while preserving cosine and the explicitly directed relative L2.
    scale = max(float(np.max(np.abs(origin))), float(np.max(np.abs(destination))))
    if scale == 0:
        return {"cosine": None, "relative_l2": None, "status": "zero_both_features"}
    a, b = origin / scale, destination / scale
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cosine = max(-1., min(1., float(np.dot(a, b) / norm_a / norm_b))) if norm_a > 0 and norm_b > 0 else None
    relative = float(np.linalg.norm(b - a) / norm_a) if norm_a > 0 else None
    status = "zero_origin_feature" if norm_a == 0 else "zero_destination_feature" if norm_b == 0 else "measured"
    if relative is not None and not math.isfinite(relative):
        relative = None; status = "relative_l2_overflow"
    return {"cosine": cosine, "relative_l2": relative, "status": status}


def _source_metadata(common, ref_metadata, cand_metadata):
    common = _metadata(common)
    if "reference" in common or "candidate" in common:
        ref_metadata = {**_metadata(common.get("reference")), **ref_metadata}
        cand_metadata = {**_metadata(common.get("candidate")), **cand_metadata}
    else:
        ref_metadata = {**common, **ref_metadata}; cand_metadata = {**common, **cand_metadata}
    a, b = ref_metadata.get("feature_space_id"), cand_metadata.get("feature_space_id")
    if a is not None and b is not None and a != b:
        raise ValueError("reference and candidate must use the same declared feature_space_id")
    return ref_metadata, cand_metadata


def evaluate_reappearance(reference, candidate, annotations, *, min_sequences=20, source_metadata=None) -> dict:
    """Measure annotated first/absence/reappearance entity and state features.

    Features are ``[frames, feature_dim]`` arrays, or mappings containing
    ``features`` and optional ``source_metadata``. Shared source_metadata may be
    supplied explicitly; missing provenance is labelled unknown. This function
    neither extracts features nor assigns a video-quality label.

    Identity and optional state dimensions must be disjoint. An annotation's
    shared boolean present_mask validates event order. A mapping with reference
    and candidate masks also allows an actually absent candidate endpoint to be
    reported without interpreting its feature vector as an identity failure.
    Events for one sequence/entity may not overlap, including shared endpoints.
    ``sequence_id`` defaults to one sequence for the entire call.
    """
    if type(min_sequences) is not int or min_sequences < 1:
        raise ValueError("min_sequences must be a positive integer")
    ref, ref_metadata = _features(reference); cand, cand_metadata = _features(candidate)
    if ref.shape != cand.shape:
        raise ValueError("reference/candidate must have matching frame and feature dimensions")
    ref_metadata, cand_metadata = _source_metadata(source_metadata, ref_metadata, cand_metadata)
    source_status = "declared" if ref_metadata and cand_metadata else "unknown"
    events = _events(annotations, *ref.shape)
    rows = []; metric_names = set()
    for event in events:
        first, end = event["first_frame"], event["reappearance_frame"]
        ref_mask, cand_mask = event["reference_mask"], event["candidate_mask"]
        visible_first = cand_mask is None or bool(cand_mask[first])
        visible_end = cand_mask is None or bool(cand_mask[end])
        row = {key: value for key, value in event.items() if not key.endswith("_mask")}
        row.update({"metric_scope": "declared_entity_state_feature_consistency", "quality_status": "unknown",
                    "source_metadata_status": source_status, "state_status": "not_supplied" if event["state_feature_indices"] is None else "supplied",
                    "reference_presence_validated": ref_mask is not None,
                    "candidate_present_at_first": bool(cand_mask[first]) if cand_mask is not None else None,
                    "candidate_present_at_reappearance": bool(cand_mask[end]) if cand_mask is not None else None,
                    "candidate_absent_through_gap": bool((~cand_mask[event["disappearance_frame"]:end]).all())
                    if cand_mask is not None and event["actual_disappearance"] else None})
        comparisons = []
        for role, indices in (("identity", event["feature_indices"]), ("state", event["state_feature_indices"])):
            if indices is None:
                continue
            anchor = event["state_reference_frame"] if role == "state" else first
            visible_anchor = cand_mask is None or bool(cand_mask[anchor])
            ref_anchor_visible = ref_mask is None or bool(ref_mask[anchor])
            matched_anchor = "matched_state_anchor" if role == "state" else "matched_first_identity"
            pairs = ((f"reference_{role}", ref[anchor, indices], ref[end, indices], ref_anchor_visible),
                     (f"candidate_{role}", cand[anchor, indices], cand[end, indices], visible_anchor and visible_end),
                     (matched_anchor, ref[anchor, indices], cand[anchor, indices], ref_anchor_visible and visible_anchor),
                     (f"matched_reappearance_{role}", ref[end, indices], cand[end, indices], visible_end))
            for prefix, a, b, present in pairs:
                measured = _comparison(a, b, present=present)
                comparisons.append(measured)
                for name, value in measured.items():
                    row[f"{prefix}_{name}"] = value
                    if name != "status": metric_names.add(f"{prefix}_{name}")
        valid = sum(value[name] is not None for value in comparisons for name in ("cosine", "relative_l2"))
        row["status"] = "measured" if all(value["status"] == "measured" for value in comparisons) else "partially_measured" if valid else "not_measured"
        rows.append(row)
    sequence_ids = {row["sequence_id"] for row in rows}
    measured_ids = {row["sequence_id"] for row in rows if row["matched_reappearance_identity_cosine"] is not None
                    and row["matched_reappearance_identity_relative_l2"] is not None}
    exploratory = len(measured_ids) < min_sequences
    metrics = {}
    for name in sorted(metric_names):
        by_sequence = defaultdict(list)
        for row in rows:
            value = row.get(name)
            if value is not None:
                by_sequence[row["sequence_id"]].append(value)
        sequence_means = [float(np.mean(values)) for values in by_sequence.values()]
        metrics[name] = {"mean": float(np.mean(sequence_means)) if sequence_means else None,
                         "median": float(np.median(sequence_means)) if sequence_means else None,
                         "valid_events": sum(len(values) for values in by_sequence.values()), "valid_sequences": len(sequence_means)}
    for row in rows:
        row["exploratory"] = exploratory
    summary = {"event_count": len(rows), "sequence_count": len(sequence_ids), "measured_sequence_count": len(measured_ids),
               "minimum_sequences": min_sequences, "exploratory": exploratory,
               "evidence_level": "exploratory" if exploratory else "consistency_only",
               "status": "measured" if measured_ids else "not_measured", "quality_status": "unknown",
               "quality_reason": "declared entity/state consistency is not an unconditional video-quality assessment",
               "metric_scope": "declared_entity_state_feature_consistency", "metrics": metrics,
               "source_metadata_status": source_status, "source_metadata": {"reference": ref_metadata, "candidate": cand_metadata},
               "aggregation": "event means within sequence, then equal-weight sequence means",
               "relative_l2_definition": "norm(destination-origin)/norm(origin); zero origin is undefined",
               "state_interpretation": "first-to-reappearance state changes may be legitimate; compare the matched reference separately"}
    return {"rows": rows, "summary": summary}


def load_reappearance_bundle(path) -> dict:
    """Load local JSON or non-pickled NPZ features and declared event metadata.

    JSON requires reference, candidate and annotations. NPZ requires reference,
    candidate and a scalar Unicode annotations_json; source_metadata_json is an
    optional scalar Unicode JSON mapping. No feature extraction is performed.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or not {"reference", "candidate", "annotations"}.issubset(payload):
            raise ValueError("JSON bundle requires reference, candidate and annotations")
        result = {"reference": payload["reference"], "candidate": payload["candidate"], "annotations": payload["annotations"],
                  "source_metadata": payload.get("source_metadata")}
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as bundle:
            if not {"reference", "candidate", "annotations_json"}.issubset(bundle.files):
                raise ValueError("NPZ bundle requires reference, candidate and annotations_json")
            def text(name):
                value = bundle[name]
                if value.shape != () or value.dtype.kind != "U":
                    raise ValueError(f"{name} must be a scalar Unicode JSON string")
                return json.loads(value.item())
            result = {"reference": bundle["reference"].copy(), "candidate": bundle["candidate"].copy(),
                      "annotations": text("annotations_json"),
                      "source_metadata": text("source_metadata_json") if "source_metadata_json" in bundle.files else None}
    else:
        raise ValueError("feature bundles must use .json or .npz")
    # Validate feature/metadata types immediately; event bounds are checked when
    # the caller evaluates the bundle with its chosen sequence threshold.
    _features(result["reference"]); _features(result["candidate"]); _metadata(result["source_metadata"])
    return result
