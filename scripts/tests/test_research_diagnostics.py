"""CPU integration evidence for the local M1 diagnostic record contract."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import math

import pytest
import torch

from tempokv.research import diagnostics
from tempokv.research.capture import CleanBlock
from tempokv.research.config import ResearchConfig
from tempokv.research.context import FrozenAttention, ProbeContext
from tempokv.research.io import RunStore, SCHEMA


def native_attention(q, k, v):
    logits = torch.einsum("bqhd,bkhd->bhqk", q, k) * q.shape[-1] ** -.5
    return torch.einsum("bhqk,bkhd->bqhd", logits.softmax(-1), v)


def local_error(reference, changed):
    return float(torch.linalg.vector_norm(reference.float() - changed.float()) /
                 (torch.linalg.vector_norm(reference.float()) + 1e-12))


def fixture_window():
    q = torch.tensor([[[[.4, -1.2, .8, .3, -.6]], [[-.3, .7, 1.1, -.8, .1]]]])
    axis = torch.arange(60, dtype=torch.float32).reshape(1, 12, 1, 5)
    k = 1.3 * torch.sin(axis * .43) + .2 * torch.cos(axis * .17)
    v = .7 * torch.cos(axis * .19) + .03 * axis + .3
    context = ProbeContext(
        prompt_index="cpu-prompt", seed=137, sample_id=0, layer_id=0, head_id=0,
        chunk_id=5, phase="denoise", denoise_step_index=0, denoise_timestep=999,
        current_chunk_start=10, grid_t=6, grid_h=1, grid_w=2,
        query_abs_positions=torch.arange(10, 12), key_abs_positions=torch.arange(12),
        sink_frames=1, protect_recent_frames=0,
    )
    window = FrozenAttention(context, q, k, v, native_attention, k.clone(),
                             lambda pre, coordinates: pre.clone(), torch.ones(12, dtype=torch.bool))
    candidates = []
    for frame in range(1, 5):
        indices = (frame * 2, frame * 2 + 1)
        entry = CleanBlock(frame, 0, (0, 1), k[:, indices].clone(), v[:, indices].clone(),
                           .07 * frame, .11 * frame, .9 - .04 * frame, frame)
        candidates.append((entry, indices))
    assert context.eligible_mask.nonzero().flatten().tolist() == list(range(2, 10))
    assert context.protected_mask.nonzero().flatten().tolist() == [0, 1, 10, 11]
    return window, candidates


def small_config(root, experiments=("S2", "E1", "E3", "E4")):
    return ResearchConfig.from_dict({
        "seed": 137, "experiments": list(experiments),
        "selection": {"layers": [0], "heads": [0], "chunk_indices": [5], "denoise_step_indices": [0]},
        "sampling": {"protect_recent_latent_frames": 0, "spatial_block_hw": [1, 2], "max_pairs": 2,
                     "max_pair_frame_distance": 5, "max_history_blocks_per_event": 4},
        "oracle": {"query_batch_size": 1, "block_batch_size": 2, "key_chunk_size": 3},
        "intervention": {"strengths": [.15]},
        "compression": {"bits": [2, 4], "group_size": 3, "allocation_max_states": 64},
        "output": {"root": str(root)},
    })


def read_rows(root):
    result = {}
    for path in sorted(root.glob("*/events-*.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames) == SCHEMA
            for row in reader:
                details = json.loads(row["details_json"])
                result.setdefault(row["experiment_id"], []).append({**row, **details})
    return result


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    window, candidates = fixture_window()
    cfg = small_config(tmp_path_factory.mktemp("local-diagnostics"))
    original = tuple(value.clone() for value in (window.q_post, window.k_post, window.v, window.k_pre))
    rng = torch.get_rng_state().clone()
    variants = []
    original_evaluate = diagnostics.evaluate_variant

    def observed_evaluate(current_window, variant, config):
        # Every actual diagnostic intervention preserves each sink/current pair.
        for protected in (0, 1, 10, 11):
            locations = [i for i, source in enumerate(variant.provenance) if source == (protected,)]
            assert len(locations) == 1
            token = locations[0]
            torch.testing.assert_close(variant.k_post[:, token], window.k_post[:, protected], atol=0, rtol=0)
            torch.testing.assert_close(variant.v[:, token], window.v[:, protected], atol=0, rtol=0)
        variants.append(variant)
        return original_evaluate(current_window, variant, config)

    patch = pytest.MonkeyPatch()
    patch.setattr(diagnostics, "evaluate_variant", observed_evaluate)
    store = RunStore(cfg, origin="synthetic_m1_cpu", seed=window.context.seed)
    try:
        result = diagnostics.write_local(window, candidates, store, cfg)
        root = store.close()
    finally:
        patch.undo()
    assert torch.equal(rng, torch.get_rng_state())
    for unchanged, value in zip(original, (window.q_post, window.k_post, window.v, window.k_pre)):
        torch.testing.assert_close(unchanged, value, atol=0, rtol=0)
    return {"window": window, "candidates": candidates, "cfg": cfg, "rows": read_rows(root),
            "result": result, "variants": variants, "root": root}


def test_records_are_measured_with_exact_null_controls_and_scope(measured):
    assert set(measured["rows"]) == {"S2", "E1", "E3", "E4"}
    for experiment, rows in measured["rows"].items():
        assert rows and all(row["status"] == "measured" for row in rows)
        assert all(row["experiment_id"] == experiment and row["data_origin"] == "synthetic_m1_cpu" for row in rows)
        null = [row for row in rows if row["strategy"] == "null"]
        assert len(null) == 1 and float(null[0]["local_attention_relative_error"]) == 0
        assert int(null[0]["actual_retained_tokens"]) == 12 and float(null[0]["actual_retention"]) == 1
        assert all(row["metric_scope"] == "attention_local" for row in rows)
        assert all(row["model_output_relative_error"] == row["clean_latent_relative_error"] == "" for row in rows)
    summary = json.loads((measured["root"] / "summary.json").read_text())
    assert summary["completed_experiments"] == []
    assert summary["synthetic_completed_experiments"] == ["E1", "E3", "E4", "S2"]
    assert set(summary["quality_metrics"].values()) == {"not_run"}


def test_e1_is_independent_single_drop_against_full_native_window(measured):
    window = measured["window"]
    baseline = native_attention(window.q_post, window.k_post, window.v)
    rows = [row for row in measured["rows"]["E1"] if row["intervention"] == "drop"]
    assert len(rows) == 4
    for row in rows:
        frame = int(row["history_frame"])
        removed = (frame * 2, frame * 2 + 1)
        keep = [i for i in range(12) if i not in removed]
        changed = native_attention(window.q_post, window.k_post[:, keep], window.v[:, keep])
        assert float(row["local_attention_relative_error"]) == pytest.approx(local_error(baseline, changed), rel=1e-6, abs=1e-8)
        assert float(row["local_attention_relative_error"]) > 0
        assert row["query_scope"] == "sampled_queries_full_actual_KV_window"
        assert row["candidate_scope"] == "sampled_clean_block_reservoir"
        positions = row["key_abs_positions"]
        if isinstance(positions, str):
            positions = json.loads(positions)
        assert positions == list(removed)


def test_s2_all_scores_share_the_same_sampled_candidate_set(measured):
    rows = measured["rows"]["S2"]
    scored = [row for row in rows if row["intervention"] == "drop"]
    expected = {f"frame{frame}_spatialblock0" for frame in range(1, 5)}
    for score in ("frequency", "similarity", "residual", "attention", "coarse_qk", "frequency_qk", "random"):
        selected = [row for row in scored if row["score_type"] == score]
        assert {row["block_id"] for row in selected} == expected
        assert all(int(row["candidate_count"]) == 4 and math.isfinite(float(row["score"])) for row in selected)
    correlations = [row for row in rows if row.get("statistic") == "candidate_rank_correlation"]
    assert len(correlations) == 7
    assert all(row["finite_candidates"] == row["sampled_candidates"] == 4 for row in correlations)
    assert all(row["excluded_missing_candidates"] == 0 and row["candidate_scope"] == "sampled_candidate_set" for row in correlations)


def test_s2_missing_score_uses_common_finite_candidates_for_every_ranking(tmp_path):
    window, candidates = fixture_window()
    candidates[0] = (replace(candidates[0][0], freq_k_high=float("nan")), candidates[0][1])
    cfg = small_config(tmp_path, ("S2",))
    store = RunStore(cfg, origin="synthetic_m1_cpu")
    diagnostics.write_local(window, candidates, store, cfg)
    rows = read_rows(store.close())["S2"]
    correlations = [row for row in rows if row.get("statistic") == "candidate_rank_correlation"]
    assert len(correlations) == 7
    assert all(row["finite_candidates"] == 3 and row["sampled_candidates"] == 4 for row in correlations)
    assert all(row["excluded_missing_candidates"] == 1 for row in correlations)


def test_e3_merge_drop_costs_and_replace_full_retention_are_separate(measured):
    pair_rows = [row for row in measured["rows"]["E3"] if row["strategy"] in {"drop", "replace", "average_merge", "weighted_merge"}]
    pair_ids = {row["block_id"] for row in pair_rows}
    assert len(pair_ids) == measured["cfg"].sampling.max_pairs == 2
    for pair_id in pair_ids:
        rows = {row["strategy"]: row for row in pair_rows if row["block_id"] == pair_id}
        assert set(rows) == {"drop", "replace", "average_merge", "weighted_merge"}
        for strategy in ("drop", "average_merge", "weighted_merge"):
            assert int(rows[strategy]["actual_retained_tokens"]) == 10
            assert rows[strategy]["budget_cohort"] == "tokens:10"
            assert float(rows[strategy]["actual_retention"]) == pytest.approx(10 / 12)
        assert int(rows["replace"]["actual_retained_tokens"]) == 12
        assert rows["replace"]["budget_cohort"] == "tokens:12"
        assert float(rows["replace"]["actual_retention"]) == 1
        assert rows["weighted_merge"]["backend_protocol"] == "native_group_attention_fp32_lse"
        assert rows["average_merge"]["backend_protocol"] == "native_attention"


def test_e3_conditioned_reference_is_measured_instead_of_adding_loo_errors(measured):
    window = measured["window"]
    original = native_attention(window.q_post, window.k_post, window.v)
    conditional = [row for row in measured["rows"]["E3"] if row["strategy"] == "conditional_drop"]
    assert len(conditional) == 4
    for row in conditional:
        frame = int(row["block_id"].split("|")[0].split(":")[0])
        target = [frame * 2, frame * 2 + 1]
        condition = json.loads(row["conditioning_set"])
        conditioned_keep = [i for i in range(12) if i not in condition]
        changed_keep = [i for i in conditioned_keep if i not in target]
        reference = native_attention(window.q_post, window.k_post[:, conditioned_keep], window.v[:, conditioned_keep])
        changed = native_attention(window.q_post, window.k_post[:, changed_keep], window.v[:, changed_keep])
        measured_error = float(row["local_attention_relative_error"])
        assert measured_error == pytest.approx(local_error(reference, changed), rel=1e-6, abs=1e-8)
        assert row["comparison_reference"] == "conditioned_key_set"
        if condition:
            target_keep = [i for i in range(12) if i not in target]
            target_loo = native_attention(window.q_post, window.k_post[:, target_keep], window.v[:, target_keep])
            additive = local_error(original, target_loo) + local_error(original, reference)
            assert abs(measured_error - additive) > 1e-3


def test_e3_diversity_merge_has_an_equal_cost_drop_control(measured):
    rows = {row["strategy"]: row for row in measured["rows"]["E3"]
            if row["strategy"] in {"diversity_based_group_merge", "diversity_matched_drop"}}
    assert set(rows) == {"diversity_based_group_merge", "diversity_matched_drop"}
    merge, drop = rows["diversity_based_group_merge"], rows["diversity_matched_drop"]
    assert int(merge["actual_retained_tokens"]) == int(drop["actual_retained_tokens"]) == 8
    assert merge["budget_cohort"] == drop["budget_cohort"] == "tokens:8"
    assert merge["grouping_protocol"] == drop["grouping_protocol"] == "farthest_point_joint_KV_representatives_then_nearest_assignment"
    assert all(math.isfinite(float(row["local_attention_relative_error"])) for row in rows.values())


def test_e4_role_noise_is_normalized_and_preserves_actual_model_seed(measured):
    rows = [row for row in measured["rows"]["E4"] if row["strategy"] == "matched_relative_kv_rmse"]
    assert len(rows) == 6
    assert {row["role"] for row in rows} == {"k", "v", "kv"}
    for row in rows:
        strength = float(row["strength"])
        assert int(row["seed"]) == measured["window"].context.seed == 137
        assert row["noise_seed"] != 137
        assert row["nmse_scope"] == "intervened_history_tokens"
        assert row["calibration_protocol"] == "relative_kv_rmse"
        for role in ("k", "v"):
            assert float(row[f"nmse_{role}"]) == pytest.approx(strength ** 2 if role in row["role"] else 0, rel=2e-5, abs=1e-10)
        if strength == 0:
            assert float(row["local_attention_relative_error"]) == 0
    for rows in measured["rows"].values():
        assert all(int(row["seed"]) == 137 for row in rows)
    noise = [variant for variant in measured["variants"] if variant.details.get("intervention") == "perturb"]
    assert len(noise) == 6
    assert all(variant.details["indices"] == list(range(2, 10)) for variant in noise)


def expected_role_cost(bits):
    # Eight candidate tokens, one head, D=5. FP32 source; groups of three.
    if bits == 16:
        return {"payload_bytes": 8 * 5 * 4, "scale_bytes": 0, "metadata_bytes": 0}
    return {"payload_bytes": math.ceil(8 * 5 * bits / 8), "scale_bytes": 8 * math.ceil(5 / 3) * 4,
            "metadata_bytes": 16 + 4 * 8 + 16}


def test_e4_grid_exact_logical_bytes_includes_scales_metadata_and_native_protected(measured):
    rows = [row for row in measured["rows"]["E4"] if row["strategy"] == "fake_quant_grid"]
    assert len(rows) == 9
    native_protected_bytes = 4 * 5 * 4 * 2
    for row in rows:
        kb, vb = int(row["k_bits"]), int(row["v_bits"])
        k, v = expected_role_cost(kb), expected_role_cost(vb)
        payload = k["payload_bytes"] + v["payload_bytes"] + native_protected_bytes
        scales = k["scale_bytes"] + v["scale_bytes"]
        metadata = k["metadata_bytes"] + v["metadata_bytes"]
        assert int(row["payload_bytes"]) == payload
        assert int(row["scale_bytes"]) == scales
        assert int(row["metadata_bytes"]) == metadata
        assert int(row["logical_bytes"]) == payload + scales + metadata
        assert int(row["physical_kv_bytes"]) == 12 * 5 * 4 * 2
        assert row["precision_scope"] == "sampled_candidate_tokens; other/protected tokens retain native dtype"
        assert int(row["actual_retained_tokens"]) == 12
        if kb == vb == 16:
            assert float(row["local_attention_relative_error"]) == 0
            assert float(row["nmse_k"]) == float(row["nmse_v"]) == 0


def test_e4_joint_oracle_uses_actual_grid_attention_error_not_additive_proxy(measured):
    rows = measured["rows"]["E4"]
    grid = {(int(row["k_bits"]), int(row["v_bits"])): row for row in rows if row["strategy"] == "fake_quant_grid"}
    joint = [row for row in rows if row["strategy"] == "joint_attention_precision_oracle"]
    assert joint
    native_protected_bytes = 4 * 5 * 4 * 2
    expected_frontier = set()
    for bits, row in grid.items():
        cost, error = int(row["logical_bytes"]), float(row["local_attention_relative_error"])
        dominated = any(int(other["logical_bytes"]) <= cost and float(other["local_attention_relative_error"]) <= error
                        and (int(other["logical_bytes"]) < cost or float(other["local_attention_relative_error"]) < error)
                        for other in grid.values())
        if not dominated:
            expected_frontier.add(bits)
    assert {(int(row["k_bits"]), int(row["v_bits"])) for row in joint} == expected_frontier
    for row in joint:
        bits = (int(row["k_bits"]), int(row["v_bits"]))
        assert row["allocation_objective"] == "callback_evaluated"
        assert float(row["local_attention_relative_error"]) == float(grid[bits]["local_attention_relative_error"])
        assert int(row["allocated_candidate_bytes"]) + native_protected_bytes == int(row["logical_bytes"])
        assert row["budget_cohort"] == f"bytes:{row['allocated_candidate_bytes']}"
    assert any(abs(float(grid[kb, vb]["local_attention_relative_error"])
                   - float(grid[kb, 16]["local_attention_relative_error"])
                   - float(grid[16, vb]["local_attention_relative_error"])) > 1e-4 for kb in (2, 4) for vb in (2, 4))
    controls = [row for row in rows if row["strategy"] in {"uniform", "k_higher"}]
    assert controls and {row["strategy"] for row in controls} == {"uniform", "k_higher"}
    for row in controls:
        budget = int(row["budget_cohort"].split(":")[1])
        assert int(row["allocated_candidate_bytes"]) <= budget


def test_no_history_writes_only_null_and_explicit_not_measured_reason(tmp_path):
    window, _ = fixture_window()
    cfg = small_config(tmp_path)
    store = RunStore(cfg, origin="synthetic_m1_cpu")
    result = diagnostics.write_local(window, [], store, cfg)
    rows = read_rows(store.close())
    assert result == {"candidates": [], "oracle": [], "scores": {}, "nearest": {}}
    for experiment in cfg.experiments:
        assert len(rows[experiment]) == 2
        null = next(row for row in rows[experiment] if row["status"] == "measured")
        skipped = next(row for row in rows[experiment] if row["status"] == "not_measured")
        assert null["strategy"] == "null" and float(null["local_attention_relative_error"]) == 0
        assert skipped["reason"] == "no_eligible_clean_history_blocks"
        assert skipped["local_attention_relative_error"] == skipped["score"] == ""
        assert skipped["model_output_relative_error"] == skipped["clean_latent_relative_error"] == ""
