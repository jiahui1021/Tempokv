from __future__ import annotations

from copy import deepcopy
import itertools
import math

import pytest
import torch

from tempokv.research.step_policy import build_schedule, enumerate_step_policies, evaluate_step_policies


def base_policies():
    return enumerate_step_policies([5, 5], [.5, 1], protected_counts=[1, 1])


def replay(policy):
    """A two-step CPU denoising recurrence with genuinely non-additive effects."""
    schedule = build_schedule(policy, 2)
    initial = torch.tensor([1., .5], dtype=torch.float64)
    missing0, missing1 = 4 - schedule[0], 4 - schedule[1]
    first = initial + missing0 * torch.tensor([.8, .1], dtype=torch.float64)
    final = (1 + missing1 * .2) * first + missing1 * torch.tensor([.1, .3], dtype=torch.float64)
    return {"clean_latent_relative_error": float(torch.linalg.vector_norm(final - initial) / initial.norm()),
            "model_output_relative_error": float(torch.linalg.vector_norm(first - initial) / initial.norm()),
            "horizon_metrics": {"one_extra_contraction_absolute_error": float(((final - initial) * .5).norm())}}


def test_exact_half_up_rounding_and_mandatory_counts():
    policies = enumerate_step_policies([7, 10], [0, .1, .3, .5, 1], protected_counts=[2, 2])
    assert {policy["kept_history_tokens"][0] for policy in policies} == {0, 1, 2, 3, 5}
    assert {policy["kept_history_tokens"][1] for policy in policies} == {0, 1, 2, 4, 8}
    for policy in policies:
        assert policy["actual_tokens_per_step"] == tuple(k + 2 for k in policy["kept_history_tokens"])
        assert policy["total_token_accesses"] == sum(policy["actual_tokens_per_step"])
        assert policy["protected_counts"] == (2, 2)
        assert policy["phase"] == "denoise" and policy["step_indices"] == (0, 1)


def test_duplicate_actual_vectors_are_deduplicated_but_equal_total_schedules_survive():
    policies = enumerate_step_policies([2, 2], [.25, .5, .75, 1], protected_counts=[1, 1], max_states=4)
    assert len(policies) == 4
    assert {policy["kept_history_tokens"] for policy in policies} == set(itertools.product([0, 1], repeat=2))
    same_total = [policy for policy in policies if policy["total_token_accesses"] == 3]
    assert len(same_total) == 2 and {policy["kept_history_tokens"] for policy in same_total} == {(0, 1), (1, 0)}
    null = next(policy for policy in policies if policy["is_null"])
    assert null["retentions"] == (1., 1.)
    assert null["uniform_retention_levels"] == (.5, .75, 1.)


def test_ids_and_canonical_labels_are_stable_across_input_order_duplicates_and_budget():
    first = enumerate_step_policies([5, 5], [.5, 1], protected_counts=[1, 1])
    second = enumerate_step_policies((5, 5), (1, .5, .5), protected_counts=(1, 1), total_budget=8)
    first_by_kept = {policy["kept_history_tokens"]: policy for policy in first}
    for policy in second:
        original = first_by_kept[policy["kept_history_tokens"]]
        assert policy["policy_id"] == original["policy_id"]
        assert policy["retentions"] == original["retentions"]
        assert policy["requested_budget"] == 8
    assert len({policy["policy_id"] for policy in first}) == len(first)


def test_budget_boundary_is_integer_and_protected_tokens_cannot_be_saved():
    policies = enumerate_step_policies([5, 5], [0, .5, 1], protected_counts=[1, 1], total_budget=2)
    assert len(policies) == 1
    assert policies[0]["kept_history_tokens"] == (0, 0)
    assert policies[0]["actual_tokens_per_step"] == (1, 1)
    assert policies[0]["total_token_accesses"] == 2
    with pytest.raises(ValueError, match="no feasible"):
        enumerate_step_policies([5, 5], [0, 1], protected_counts=[1, 1], total_budget=1)


def test_all_protected_and_zero_count_steps_have_one_actual_policy():
    policies = enumerate_step_policies([3, 0, 5], [0, .25, .5, .75, 1], protected_counts=[3, 0, 5], max_states=1)
    assert len(policies) == 1
    assert policies[0]["kept_history_tokens"] == (0, 0, 0)
    assert policies[0]["actual_tokens_per_step"] == (3, 0, 5)
    assert policies[0]["is_null"] is True
    assert policies[0]["retentions"] == (1., 1., 1.)
    assert build_schedule(policies[0], 3) == {0: 0, 1: 0, 2: 0}


def test_exhaustive_callback_measures_real_nonadditive_final_clean_error():
    policies = base_policies()
    calls = []
    def measure(policy):
        calls.append(policy["policy_id"])
        return replay(policy)
    result = evaluate_step_policies(policies, measure)
    assert len(calls) == len(policies) == result["states_evaluated"] == 4
    rows = {row["kept_history_tokens"]: row for row in result["rows"]}
    assert rows[4, 4]["clean_latent_relative_error"] == 0
    assert rows[4, 4]["is_null"] is True
    assert rows[2, 4]["total_token_accesses"] == rows[4, 2]["total_token_accesses"] == 8
    assert rows[4, 2]["clean_latent_relative_error"] < rows[2, 4]["clean_latent_relative_error"]
    assert rows[2, 2]["clean_latent_relative_error"] != pytest.approx(rows[2, 4]["clean_latent_relative_error"] + rows[4, 2]["clean_latent_relative_error"])
    assert all(row["objective_kind"] == "measured_clean_latent_relative_error" for row in rows.values())
    assert all(row["measurement_protocol"] == "complete_schedule_callback" for row in rows.values())
    assert all(row["oracle_scope"] == "supplied_feasible_schedules" for row in rows.values())
    assert "horizon_metrics" in rows[4, 2] and "model_output_relative_error" in rows[4, 2]
    assert result["evidence_kind"] == "measured_clean_latent_differences"


def test_budget_oracle_and_uniform_control_labels_come_from_measured_replays():
    called = []
    def measure(policy):
        called.append(policy["total_token_accesses"])
        return replay(policy)
    result = evaluate_step_policies(base_policies(), measure, budget=8)
    assert called == [6, 8, 8]
    assert result["oracle"]["kept_history_tokens"] == (4, 2)
    assert result["oracle"]["is_budget_oracle"] is True
    assert result["oracle"]["comparison_group"] == "mixed_step"
    assert sum(row["is_budget_oracle"] for row in result["rows"]) == 1
    assert [row["kept_history_tokens"] for row in result["uniform_rows"]] == [(2, 2)]
    assert all(row["comparison_group"] == "uniform_control" for row in result["uniform_rows"])
    assert [row["kept_history_tokens"] for row in result["frontier"]] == [(2, 2), (4, 2)]


def test_exact_costs_are_preserved_and_callback_gets_private_copy():
    policies = base_policies()
    before = deepcopy(policies)
    def measure(policy):
        measurement = replay(policy)
        policy["total_token_accesses"] = -999
        policy["kept_history_tokens"] = (999, 999)
        return measurement
    result = evaluate_step_policies(policies, measure)
    assert policies == before
    original = {policy["policy_id"]: policy for policy in policies}
    for row in result["rows"]:
        for key, value in original[row["policy_id"]].items():
            assert row[key] == value
    with pytest.raises(ValueError, match="cannot replace exact policy field"):
        evaluate_step_policies(policies, lambda _: {"clean_latent_relative_error": 0., "total_token_accesses": 0})


def test_constant_zero_evidence_cannot_invent_quality_separation():
    result = evaluate_step_policies(base_policies(), lambda _: {"clean_latent_relative_error": 0.})
    assert result["evidence_kind"] == "constant_objective_no_quality_separation"
    assert {row["clean_latent_relative_error"] for row in result["rows"]} == {0.}
    assert result["oracle"]["total_token_accesses"] == 6
    assert [row["total_token_accesses"] for row in result["frontier"]] == [6]
    assert len(result["uniform_rows"]) == 2
    assert "predicted_clean_error" not in result and "additive_proxy" not in str(result)


def test_identical_cost_error_points_are_all_nondominated():
    policies = [policy for policy in base_policies() if policy["total_token_accesses"] == 8]
    result = evaluate_step_policies(policies, lambda _: {"clean_latent_relative_error": .2})
    assert len(result["frontier"]) == 2
    assert result["evidence_kind"] == "constant_objective_no_quality_separation"


def test_schedule_is_local_fresh_and_rejects_clean_refresh_or_wrong_indices():
    policy = base_policies()[0]
    assert build_schedule(policy, 2) == {0: 2, 1: 2}
    mutated = build_schedule(policy, 2)
    mutated[0] = 999
    assert build_schedule(policy, 2) == {0: 2, 1: 2}
    assert "active" not in policy and "persistent" not in policy
    for indices in ((0, 2), (-1, 0), (1, 0), (0, 0), (0, "clean_refresh")):
        with pytest.raises(ValueError, match="in-schedule"):
            build_schedule({**policy, "step_indices": indices}, 2)
    with pytest.raises(ValueError, match="clean_refresh"):
        build_schedule({**policy, "phase": "clean_refresh"}, 2)
    for count in (1, 3, 0, -1, True):
        with pytest.raises(ValueError):
            build_schedule(policy, count)


def test_search_limit_checks_unique_cartesian_product_before_budget_filter():
    with pytest.raises(ValueError, match="state space.*no approximate policy"):
        enumerate_step_policies([10, 10, 10], [.5, 1], total_budget=0, max_states=7)
    assert len(enumerate_step_policies([10, 10, 10], [.5, 1], max_states=8)) == 8


@pytest.mark.parametrize("counts", [[], [-1], [True], [1.], "1", None])
def test_invalid_token_counts(counts):
    with pytest.raises(ValueError):
        enumerate_step_policies(counts)


@pytest.mark.parametrize("protected", [[], [-1, 0], [1], [6, 0], [True, 0], [1., 0]])
def test_invalid_protected_counts(protected):
    with pytest.raises(ValueError):
        enumerate_step_policies([5, 5], protected_counts=protected)


@pytest.mark.parametrize("levels", [[], [-.1], [1.1], [float("nan")], [float("inf")], [True], [".5"], None])
def test_invalid_retention_levels(levels):
    with pytest.raises(ValueError):
        enumerate_step_policies([5], levels)


@pytest.mark.parametrize("kwargs", [{"total_budget": -1}, {"total_budget": True}, {"total_budget": 1.5},
                                         {"max_states": 0}, {"max_states": True}, {"max_states": 1.5}])
def test_invalid_enumerator_budgets_or_limits(kwargs):
    with pytest.raises(ValueError):
        enumerate_step_policies([5], **kwargs)


@pytest.mark.parametrize("value", [None, {}, {"model_output_relative_error": .1},
                                     {"clean_latent_relative_error": -1}, {"clean_latent_relative_error": math.inf},
                                     {"clean_latent_relative_error": math.nan}, {"clean_latent_relative_error": True},
                                     {"clean_latent_relative_error": "0"}])
def test_callback_requires_actual_finite_nonnegative_clean_measurement(value):
    with pytest.raises(ValueError):
        evaluate_step_policies(base_policies(), lambda _: value)


def test_evaluator_invalid_structure_infeasible_budget_and_no_silent_proxy():
    for policies in ([], None, "policies"):
        with pytest.raises(ValueError):
            evaluate_step_policies(policies, replay)
    with pytest.raises(TypeError, match="callable"):
        evaluate_step_policies(base_policies(), None)
    with pytest.raises(ValueError, match="unique"):
        evaluate_step_policies([base_policies()[0]] * 2, replay)
    with pytest.raises(ValueError, match="no feasible"):
        evaluate_step_policies(base_policies(), lambda _: pytest.fail("infeasible callback was invoked"), budget=5)
    for budget in (-1, True, 2.5):
        with pytest.raises(ValueError):
            evaluate_step_policies(base_policies(), replay, budget=budget)


def test_tampered_policy_costs_or_retention_are_rejected():
    policy = base_policies()[0]
    for bad in ({"actual_tokens_per_step": (0, 0)}, {"kept_history_tokens": (9, 9)},
                {"protected_counts": (8, 8)}, {"total_token_accesses": 999}, {"retentions": (1, 1)},
                {"retentions": None}, {"policy_id": ""}):
        with pytest.raises(ValueError):
            build_schedule({**policy, **bad}, 2)
    with pytest.raises(ValueError, match="missing"):
        build_schedule({}, 2)
    with pytest.raises(TypeError, match="mapping"):
        build_schedule(None, 2)
