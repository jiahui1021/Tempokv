"""Small, explicit E5 fixtures and a real-evidence exploration gate.

Feature fixtures are deterministic CPU arrays, not generated videos. Their
annotations describe an intended event and a continuously visible static
control. The evidence gate authorizes an exploratory benchmark only; it never
creates a scientific decision, infers a positive spectral result, or reports
video-quality conclusions.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCENARIO_NAMES = ("object_exit_return", "camera_return", "background_revisit", "interaction_persistence")
_ALIASES = {"character_exit_return": "object_exit_return", "object_return": "object_exit_return",
            "camera_scene_return": "camera_return", "background_state_revisit": "background_revisit"}
_EXPERIMENTS = {"S2", "E1", "E2", "E4", "S3", "E7"}
_METRICS = ("local_attention_relative_error", "model_output_relative_error", "clean_latent_relative_error",
            "spearman", "topk_recall_sampled", "interaction", "interaction_error")
_PROMPTS = {
    "object_exit_return": (
        "A small blue robot wearing a yellow scarf stands beside a red door. The robot walks completely out of view. "
        "After an empty interval, the same blue robot returns wearing the same yellow scarf.",
        "A small blue robot wearing a yellow scarf remains beside a red door throughout the clip. The camera stays fixed and the robot stays visible."),
    "camera_return": (
        "A fixed arrangement of a green armchair, a brass lamp and a striped rug fills an old reading room. The camera turns away into a plain corridor. "
        "After an interval, it returns to the original reading room with the same arrangement.",
        "The camera remains fixed on a green armchair, a brass lamp and a striped rug in an old reading room. The arrangement stays visible and unchanged."),
    "background_revisit": (
        "A courtyard wall has a blue chalk spiral beside a red shutter. A large opaque screen hides the wall for an interval. "
        "The screen moves away, revealing the same chalk spiral and shutter in the same state.",
        "A courtyard wall with a blue chalk spiral beside a red shutter remains fully visible. The camera stays fixed and the background remains unchanged."),
    "interaction_persistence": (
        "A person moves a brass latch on a wooden box from closed to open. The box is carried completely out of view. "
        "After an interval, the same box returns with its brass latch still open.",
        "A wooden box with its brass latch already open remains fully visible on a table. The camera stays fixed and the latch remains open throughout."),
}


def make_scenario(name, seed=0):
    """Return independent float32 [24,12] reference/control feature fixtures.

    Channels 0..3 encode target identity, 4..7 the visible setting, 8 visibility,
    9 persistent state, 10 the away-view indicator and 11 small visible motion.
    The annotations use absolute, zero-based frame indices, with half-open
    absence [disappearance_frame, reappearance_frame).
    """
    if name not in SCENARIO_NAMES and name not in _ALIASES:
        raise ValueError(f"unknown scenario; choose one of {SCENARIO_NAMES}")
    if type(seed) is not int or seed < 0: raise ValueError("seed must be a nonnegative integer")
    requested = name
    name = _ALIASES.get(name, name)
    offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    rng = np.random.default_rng(np.random.SeedSequence([seed, offset]))
    identity, background, away = rng.normal(size=(3, 4))
    identity /= np.linalg.norm(identity)
    background /= np.linalg.norm(background)
    away /= np.linalg.norm(away)
    base = np.zeros(12, dtype=np.float32)
    base[:4], base[4:8], base[8] = identity, background, 1.
    if name in {"background_revisit", "interaction_persistence"}: base[9] = 1.
    control = np.repeat(base[None], 24, axis=0)
    reference = control.copy()
    reference[8:16, :4] = 0.
    reference[8:16, 8:10] = 0.
    reference[8:16, 10] = 1.
    if name in {"camera_return", "background_revisit"}: reference[8:16, 4:8] = away
    if name == "interaction_persistence": reference[:4, 9] = 0.
    reference[:, 11] = (.01 * np.sin(np.arange(24) / 3)).astype(np.float32)
    reference[8:16, 11] = 0.
    sequence = f"{requested}:seed-{seed}"
    annotation = {"entity_id": f"{name}:target", "first_frame": 0, "disappearance_frame": 8,
                  "reappearance_frame": 16, "feature_indices": [0, 1, 2, 3], "sequence_id": sequence,
                  "first_appearance": 0, "event_kind": name, "actual_disappearance": True,
                  "present_mask": [True] * 8 + [False] * 8 + [True] * 8}
    if name in {"background_revisit", "interaction_persistence"}:
        annotation.update(state_feature_indices=[9], state_reference_frame=7)
    if name == "interaction_persistence": annotation["state_change_frame"] = 4
    control_annotation = {key: value for key, value in annotation.items() if key != "present_mask"}
    control_annotation.update(sequence_id=f"{sequence}:static-control", event_kind="static_control", actual_disappearance=False,
                              temporal_anchor_note="Matched time anchors only; the control does not leave or become hidden.")
    control_annotation.pop("state_change_frame", None)
    prompt, control_prompt = _PROMPTS[name]
    return {"scenario_id": requested, "scenario_kind": name, "seed": seed, "data_origin": "synthetic_e5_cpu",
            "reference_features": reference, "reference": reference, "annotations": [annotation], "prompt": prompt,
            "prompts": {"reference": prompt, "static_control": control_prompt},
            "control": {"reference_features": control, "reference": control, "annotations": [control_annotation],
                        "prompt": control_prompt, "event_kind": "static_control", "data_origin": "synthetic_e5_cpu"},
            "feature_fixture_only": True, "generated_video": False, "quality_metrics": None,
            "feature_channels": {"identity": [0, 1, 2, 3], "setting": [4, 5, 6, 7], "visibility": [8],
                                 "persistent_state": [9], "away_view": [10], "motion": [11]}}


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant {value} is unsupported")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads(text):
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_pairs)


def load_evidence_json(path, *, max_bytes=4 * 1024 ** 2):
    """Read local JSON only, with a byte limit, no pickle/eval and no NaN values."""
    if type(max_bytes) is not int or max_bytes < 1: raise ValueError("max_bytes must be positive")
    source = Path(path).expanduser()
    if source.stat().st_size > max_bytes: raise ValueError("evidence JSON exceeds max_bytes")
    with source.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes: raise ValueError("evidence JSON exceeds max_bytes")
    value = _loads(data.decode("utf-8"))
    if not isinstance(value, (dict, list)): raise ValueError("evidence JSON must contain records or an evidence mapping")
    return value


def _number(value):
    if isinstance(value, (bool, np.bool_)): return None
    try: value = float(value)
    except (TypeError, ValueError): return None
    return value if math.isfinite(value) else None


def _integer(value):
    number = _number(value)
    return int(number) if number is not None and number >= 0 and number.is_integer() else None


def _known(value):
    return value is not None and str(value).strip().lower() not in {"", "unknown", "none", "null", "not_available"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _record(row, common):
    if not isinstance(row, dict): raise ValueError("evidence rows must be mappings")
    extra = row.get("details_json") or {}
    if isinstance(extra, str): extra = _loads(extra)
    if not isinstance(extra, dict): raise ValueError("details_json must contain a mapping")
    embedded = row.get("evidence_metadata") or {}
    if not isinstance(embedded, dict): raise ValueError("evidence_metadata must be a mapping")
    result = {**common, **embedded, **extra, **{key: value for key, value in row.items() if value not in (None, "")}}
    # A contradictory provenance declaration cannot be promoted to real data by
    # another metadata layer. Direct, detail and shared origins must agree.
    origins = {item.get("data_origin") for item in (common, embedded, extra, row) if item.get("data_origin")}
    if len(origins) > 1: raise ValueError("conflicting evidence data origins")
    if result.get("schema_version") != "research.v1" or result.get("data_origin") != "runtime_bf16" or result.get("source") != "bf16":
        raise ValueError("requires research.v1 runtime_bf16 evidence with source=bf16")
    if result.get("status") != "measured" or result.get("experiment_id") not in _EXPERIMENTS:
        raise ValueError("requires a measured S2/E1/E2/E4/S3/E7 record")
    for name in ("event_id", "config_signature", "model", "checkpoint", "window", "prompt_index"):
        if not _known(result.get(name)): raise ValueError(f"concrete evidence metadata missing: {name}")
    if not isinstance(result["event_id"], str): raise ValueError("event_id must be an explicit string")
    if _integer(result.get("seed")) is None: raise ValueError("seed must identify an independent prompt/seed unit")
    if not any(_number(result.get(name)) is not None for name in _METRICS):
        raise ValueError("record has no finite measured local/paired metric")
    result["seed"] = _integer(result["seed"])
    result["_unit"] = (str(result["prompt_index"]), result["seed"])
    result["_cohort"] = tuple(_canonical(result[name]) for name in ("config_signature", "model", "checkpoint", "window", "source"))
    return result


def _paired(row):
    return _known(row.get("paired_run_id")) and row.get("paired_rng") in (True, "true", "matched", "per_call_and_final_matched")


def _paired_units(rows, horizon=1):
    pairs = defaultdict(lambda: {"E1": False, "E2": False})
    for row in rows:
        if not _paired(row): continue
        key = row["_unit"], str(row["paired_run_id"])
        if row["experiment_id"] == "E1":
            value = _number(row.get("model_output_relative_error"))
            if value is not None and value >= 0: pairs[key]["E1"] = True
        elif row["experiment_id"] == "E2":
            value = _number(row.get("clean_latent_relative_error"))
            actual_horizon = _integer(row.get("horizon_chunks"))
            if value is not None and value >= 0 and actual_horizon is not None and actual_horizon >= horizon:
                pairs[key]["E2"] = True
    return {unit for (unit, _), present in pairs.items() if all(present.values())}


def _criterion(criterion, rows, minimum):
    if not isinstance(criterion, dict): return {"passed": False, "reason": "criteria must be explicit mappings"}
    kind = criterion.get("kind")
    if kind in {"paired_horizon_coverage", "reappearance_rationale"}:
        horizon = _integer(criterion.get("min_horizon_chunks", 1))
        if horizon is None or horizon < 1: return {"kind": kind, "passed": False, "reason": "a positive horizon criterion is required"}
        units = _paired_units(rows, horizon)
        scenarios = criterion.get("scenario_names", [])
        if kind == "reappearance_rationale" and (not isinstance(scenarios, list) or not scenarios or any(name not in SCENARIO_NAMES for name in scenarios)):
            return {"kind": kind, "passed": False, "reason": "reappearance rationale must name fixed scenarios"}
        return {"kind": kind, "passed": len(units) >= minimum, "min_horizon_chunks": horizon,
                "observed_prompt_seed_pairs": len(units), "required_prompt_seed_pairs": minimum,
                "scenario_names": scenarios, "requires_positive_effect": False}
    if kind == "measured_heterogeneity":
        metric, group_by = criterion.get("metric"), criterion.get("group_by")
        threshold = _number(criterion.get("min_range", 0.))
        if metric not in _METRICS or group_by not in {"block_id", "layer", "head", "strategy", "horizon_chunks", "role"} or threshold is None or threshold < 0:
            return {"kind": kind, "passed": False, "reason": "heterogeneity requires a measured metric, grouping field and nonnegative min_range"}
        groups = defaultdict(lambda: defaultdict(list))
        for row in rows:
            value = _number(row.get(metric))
            if value is not None and row.get(group_by) is not None:
                groups[_canonical(row[group_by])][row["_unit"]].append(value)
        means = [float(np.mean([np.mean(values) for values in units.values()])) for units in groups.values()]
        observed = max(means) - min(means) if len(means) >= 2 else 0.
        return {"kind": kind, "passed": len(means) >= 2 and observed > 0 and observed >= threshold,
                "metric": metric, "group_by": group_by, "observed_range": observed,
                "required_range": threshold, "groups": len(means), "aggregation": "within_prompt_seed_then_group_mean"}
    return {"kind": kind, "passed": False, "reason": "unknown declared criterion"}


def heavy_benchmark_gate(records, *, min_prompt_seed_pairs=2):
    """Require an explicit evidenced decision before exploring a heavier E5 run.

    Input is an evidence mapping (or local JSON path) with ``records``, optional
    shared ``evidence_metadata``, and ``decision`` containing decision=
    ``explore_memory``, rationale, supporting_event_ids and criteria. Accepted
    criteria are paired_horizon_coverage, reappearance_rationale and
    measured_heterogeneity. Paired E1 step errors and E2 future clean errors must
    cover the requested independent prompt/seed units in one cohort, even when
    all errors are zero and spectral correlations are negative.
    """
    if type(min_prompt_seed_pairs) is not int or min_prompt_seed_pairs < 1:
        raise ValueError("min_prompt_seed_pairs must be a positive integer")
    reasons, invalid, accepted = [], [], []
    if isinstance(records, (str, Path)):
        try: records = load_evidence_json(records)
        except (OSError, ValueError, UnicodeError) as exc:
            return {"allowed": False, "reasons": [f"invalid_evidence_json: {exc}"], "quality_conclusion": None}
    if isinstance(records, list): envelope, raw = {}, records
    elif isinstance(records, dict): envelope, raw = records, records.get("records")
    else: envelope, raw = {}, None
    if not isinstance(raw, list) or not raw: reasons.append("nonempty_evidence_records_required")
    common = envelope.get("evidence_metadata") or {}
    if not isinstance(common, dict): reasons.append("shared_evidence_metadata_must_be_a_mapping"); common = {}
    for index, row in enumerate(raw if isinstance(raw, list) else []):
        try: accepted.append(_record(row, common))
        except (ValueError, TypeError) as exc: invalid.append({"row_index": index, "reason": str(exc)})
    decision = envelope.get("decision")
    if not isinstance(decision, dict): reasons.append("explicit_explore_memory_decision_required"); decision = {}
    if decision.get("decision") != "explore_memory": reasons.append("decision_must_explicitly_be_explore_memory")
    if not isinstance(decision.get("rationale"), str) or not decision.get("rationale", "").strip(): reasons.append("declared_memory_rationale_required")
    support = decision.get("supporting_event_ids")
    if not isinstance(support, list) or not support or not all(isinstance(value, str) and value for value in support) or len(set(support)) != len(support):
        reasons.append("unique_supporting_event_ids_required"); support = []
    by_id = defaultdict(dict)
    for row in accepted:
        by_id[row["event_id"]][(row["_cohort"], row["_unit"])] = row
    selected = []
    for identifier in support:
        matches = by_id.get(identifier, {})
        if not matches: reasons.append(f"supporting_event_not_real_measured: {identifier}")
        elif len(matches) != 1: reasons.append(f"supporting_event_id_ambiguous: {identifier}")
        else: selected.append(next(iter(matches.values())))
    cohorts = {row["_cohort"] for row in selected}
    if len(cohorts) != 1: reasons.append("supporting_evidence_must_belong_to_one_explicit_cohort")
    units = _paired_units(selected) if len(cohorts) == 1 else set()
    if len(units) < min_prompt_seed_pairs: reasons.append("insufficient_real_paired_E1_E2_prompt_seed_units")
    criteria = decision.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        reasons.append("declared_measurable_criteria_required"); criteria = []
    results = [_criterion(item, selected, min_prompt_seed_pairs) for item in criteria]
    if any(not result["passed"] for result in results): reasons.append("declared_measurable_criteria_not_met")
    return {"allowed": not reasons, "reasons": list(dict.fromkeys(reasons)), "decision": decision.get("decision"),
            "declared_rationale": decision.get("rationale"), "criteria_results": results,
            "supporting_event_ids": support, "accepted_record_count": len(accepted), "excluded_records": invalid,
            "independent_prompt_seed_pairs": len(units), "required_prompt_seed_pairs": min_prompt_seed_pairs,
            "scope": "eligibility_for_exploratory_E5_only", "quality_conclusion": None,
            "evidence_provenance": "explicit_runtime_metadata; authenticity is not independently inferred",
            "positive_spectral_finding_required": False}
