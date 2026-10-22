from __future__ import annotations

from dataclasses import replace
import math
import random

import numpy as np
import pytest
import torch

from tempokv.research.archive import POLICIES, select_archive
from tempokv.research.attention_oracle import drop_window, reference
from tempokv.research.context import FrozenAttention, ProbeContext


def native(q, k, v):
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * q.shape[-1] ** -.5
    return torch.einsum("bhqk,bkhd->bqhd", logits.softmax(-1), v.float()).to(q.dtype)


def fixture(dtype=torch.float32):
    positions = torch.tensor([0, 1, 8, 9, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
    context = ProbeContext("archive-prompt", 71, 0, 2, 0, 12, "denoise", 0, 999,
                           24, 2, 1, 2, torch.arange(24, 28), positions, 1, 1)
    axis = torch.arange(32, dtype=torch.float32).reshape(1, 16, 1, 2)
    k = (.5 * torch.sin(axis * .31)).to(dtype)
    v = (.8 * torch.cos(axis * .17)).to(dtype)
    v[:, 2:10, 0] = torch.tensor([[0, 0], [.1, 0], [3, 0], [.2, 0], [0, 4], [0, .3], [-2, 0], [0, 0]], dtype=dtype)
    q = torch.tensor([[[[.1, .2]], [[.3, -.1]], [[-.3, .4]], [[.5, .2]]]], dtype=dtype)
    return FrozenAttention(context, q, k, v, native)


def scores():
    return [100., 100., 1., 3., 2., 0., 8., 4., 6., 5., 999., 999., 999., 999., 999., 999.]


def select(window, policy, retention=.3125, **kwargs):
    return select_archive(window, policy=policy, retention=retention,
                          attention_scores=scores(), oracle_scores=list(reversed(scores())),
                          online=policy != "oracle_archive", **kwargs)


@pytest.mark.parametrize("policy", POLICIES)
def test_all_policies_use_complete_eligible_history_with_exact_same_half_up_budget(policy):
    window = fixture()
    result = select(window, policy)
    assert result["eligible_tokens"] == 8 and result["eligible_kept_tokens"] == 3  # half-up(8 * .3125)
    assert result["protected_tokens"] == 8 and result["actual_retained_tokens"] == 11
    assert len(result["kept_indices"]) == 11 and len(result["removed_indices"]) == 5
    assert result["budget_cohort"] == "tokens:11"
    assert result["selection_scope"] == "complete_eligible_history_tokens"
    protected = set(window.context.protected_mask.nonzero().flatten().tolist())
    assert protected == {0, 1, 10, 11, 12, 13, 14, 15}
    assert protected.issubset(result["kept_indices"])
    assert not protected.intersection(result["removed_indices"])
    assert set(result["kept_indices"]).isdisjoint(result["removed_indices"])
    assert sorted((*result["kept_indices"], *result["removed_indices"])) == list(range(16))


def test_policy_priorities_use_original_absolute_mapping_with_unequal_gaps():
    window = fixture()
    recent = select(window, "recent_only")
    uniform = select(window, "uniform_archive")
    attention = select(window, "attention_archive")
    diversity = select(window, "diversity_archive")
    protected = set(window.context.protected_mask.nonzero().flatten().tolist())
    def historical(result):
        return tuple(index for index in result["kept_indices"] if index not in protected)
    assert historical(recent) == (7, 8, 9)
    assert window.context.key_abs_positions[list(historical(recent))].tolist() == [19, 20, 21]
    assert historical(uniform) == (2, 5, 9)
    assert window.context.key_abs_positions[list(historical(uniform))].tolist() == [8, 14, 21]
    assert historical(attention) == (6, 8, 9)
    assert historical(diversity) == (2, 4, 6)


@pytest.mark.parametrize("policy", POLICIES)
def test_zero_and_full_retention_need_no_score_or_reencoding(policy):
    window = fixture()
    for retention, expected, eligible_kept in ((0, (0, 1, 10, 11, 12, 13, 14, 15), 0), (1, tuple(range(16)), 8)):
        result = select_archive(window, policy=policy, retention=retention, online=policy != "oracle_archive")
        assert result["kept_indices"] == expected
        assert result["eligible_kept_tokens"] == eligible_kept
        assert result["reencoding"] is False
        assert result["actual_retained_tokens"] == len(expected)
        if retention == 1:
            assert result["removed_indices"] == ()


@pytest.mark.parametrize("policy", POLICIES)
def test_all_protected_window_is_identity_without_scores(policy):
    window = fixture()
    context = replace(window.context, protect_recent_frames=100)
    result = select_archive(replace(window, context=context), policy=policy, retention=.2,
                            online=policy != "oracle_archive")
    assert result["eligible_tokens"] == result["eligible_kept_tokens"] == 0
    assert result["protected_tokens"] == result["actual_retained_tokens"] == 16
    assert result["kept_indices"] == tuple(range(16)) and result["removed_indices"] == ()


@pytest.mark.parametrize("retention", [0, .5, 1])
def test_future_aware_oracle_is_explicitly_diagnostic_and_denied_online(retention):
    window = fixture()
    with pytest.raises(ValueError, match="future-aware.*online=False"):
        select_archive(window, policy="oracle_archive", retention=retention, oracle_scores=scores())
    result = select_archive(window, policy="oracle_archive", retention=retention, oracle_scores=scores(), online=False)
    assert result["diagnostic_only"] is True and result["online"] is False
    assert result["causal_scope"] == "future_aware_scores_on_current_available_history"
    assert result["score_provenance"] == "future_aware_diagnostic"
    assert "not a joint subset optimum" in result["oracle_interpretation"]
    for policy in set(POLICIES) - {"oracle_archive"}:
        causal = select(window, policy, retention)
        assert causal["diagnostic_only"] is False
        assert causal["causal_scope"] == "current_available_history_only"


def test_attention_scores_are_causal_input_contract_and_protected_values_do_not_rank():
    window = fixture()
    initial = select(window, "attention_archive")
    changed = scores()
    for index in (0, 1, 10, 11, 12, 13, 14, 15):
        changed[index] = math.nan
    result = select_archive(window, policy="attention_archive", retention=.3125, attention_scores=changed)
    assert result["kept_indices"] == initial["kept_indices"]
    assert result["score_provenance"] == "current_or_past_attention"
    assert result["available_chunk_end_exclusive"] == 28


def test_diversity_uses_only_available_history_v_and_deterministic_distance_ties():
    window = fixture()
    baseline = select(window, "diversity_archive")
    altered_v = window.v.clone()
    altered_v[:, window.context.protected_mask] = torch.nan
    changed = replace(window, q_post=torch.full_like(window.q_post, torch.nan),
                      k_post=torch.full_like(window.k_post, torch.inf), v=altered_v)
    assert select(changed, "diversity_archive")["kept_indices"] == baseline["kept_indices"]
    identical_v = torch.zeros_like(window.v)
    tied = select(replace(window, v=identical_v), "diversity_archive")
    assert set(tied["kept_indices"]).intersection(range(2, 10)) == {2, 3, 4}
    assert tied == select(replace(window, v=identical_v), "diversity_archive")
    invalid_v = window.v.clone()
    invalid_v[:, 3] = torch.nan
    with pytest.raises(ValueError, match="finite available historical V"):
        select(replace(window, v=invalid_v), "diversity_archive")


def test_extreme_finite_v_keeps_farthest_point_distances_finite():
    window = fixture(torch.float64)
    values = window.v.clone()
    values[:, 2:10] = 0
    values[:, 3] = 1e308
    values[:, 4] = -1e308
    result = select(replace(window, v=values), "diversity_archive")
    assert set(result["kept_indices"]).intersection(range(2, 10)) == {2, 3, 4}


def test_score_ties_use_absolute_positions_and_all_eligible_entries_must_be_finite():
    window = fixture()
    for policy, argument in (("attention_archive", "attention_scores"), ("oracle_archive", "oracle_scores")):
        options = {"policy": policy, "retention": .3125, "online": policy != "oracle_archive"}
        tied = select_archive(window, **options, **{argument: [1.] * 16})
        assert set(tied["kept_indices"]).intersection(range(2, 10)) == {2, 3, 4}
        for invalid in (None, [0.] * 15, [math.nan] * 16, [math.inf] * 16, [[1.]] * 16, [True] * 16):
            with pytest.raises(ValueError, match="original token window|finite scalar"):
                select_archive(window, **options, **{argument: invalid})
        partial_missing = [1.] * 16
        partial_missing[4] = math.nan
        with pytest.raises(ValueError, match="finite scalar"):
            select_archive(window, **options, **{argument: partial_missing})


def test_input_and_global_rng_states_are_unchanged():
    window = fixture()
    originals = [value.clone() for value in (window.q_post, window.k_post, window.v)]
    attention = np.asarray(scores())
    copied = attention.copy()
    state_torch, state_numpy, state_python = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    for policy in POLICIES:
        select_archive(window, policy=policy, retention=.3125, attention_scores=attention,
                       oracle_scores=attention, online=policy != "oracle_archive")
    assert torch.equal(state_torch, torch.get_rng_state())
    state_after = np.random.get_state()
    assert state_numpy[0] == state_after[0] and np.array_equal(state_numpy[1], state_after[1]) and state_numpy[2:] == state_after[2:]
    assert random.getstate() == state_python and np.array_equal(attention, copied)
    for original, value in zip(originals, (window.q_post, window.k_post, window.v)):
        torch.testing.assert_close(original, value, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_logical_payload_and_absolute_index_costs_match_native_selected_values(dtype):
    window = fixture(dtype)
    result = select(window, "uniform_archive")
    payload = result["actual_retained_tokens"] * 2 * 2 * window.k_post.element_size()
    assert result["payload_bytes"] == result["logical_payload_bytes"] == payload
    assert result["index_bytes"] == result["logical_index_bytes"] == result["actual_retained_tokens"] * 8
    assert result["logical_bytes"] == payload + result["index_bytes"]
    assert result["compute_only_full_backing_bytes"] == result["physical_kv_bytes"] == 16 * 2 * 2 * window.k_post.element_size()
    assert result["physical_storage_kind"] == "unchanged_full_input_backing_compute_only"
    assert result["packed_or_gathered_storage_allocated"] is False
    assert result["reencoding"] is False


def test_shared_large_backing_is_counted_once_and_does_not_shrink_on_selection():
    window = fixture()
    base = torch.zeros(1, 64, 1, 2)
    window = replace(window, k_post=base[:, ::4], v=base[:, 1::4])
    result = select(window, "recent_only", 0)
    assert result["actual_retained_tokens"] == 8
    assert result["payload_bytes"] == 8 * 2 * 2 * 4
    assert result["compute_only_full_backing_bytes"] == base.untyped_storage().nbytes() == 64 * 2 * 4
    assert result["compute_only_full_backing_bytes"] > result["logical_bytes"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_attention_uses_matching_selected_kv_without_reencoding(dtype):
    window = fixture(dtype)
    calls = []
    def observed(q, k, v):
        assert q.dtype == k.dtype == v.dtype == dtype
        calls.append((k.clone(), v.clone()))
        return native(q, k, v)
    window = replace(window, attention_fn=observed)
    result = select(window, "recent_only")
    assert calls == []  # Selection itself performs no attention.
    reduced = drop_window(window, result["removed_indices"])
    output = reference(reduced, query_batch_size=2)
    assert output.dtype == dtype
    assert reduced.context.key_abs_positions.tolist() == window.context.key_abs_positions[list(result["kept_indices"])].tolist()
    for k, v in calls:
        torch.testing.assert_close(k, window.k_post[:, result["kept_indices"]], atol=0, rtol=0)
        torch.testing.assert_close(v, window.v[:, result["kept_indices"]], atol=0, rtol=0)
    full = select(window, "recent_only", 1)
    assert drop_window(window, full["removed_indices"]) is window


def test_future_key_or_query_positions_are_denied_even_for_offline_oracle():
    window = fixture()
    future_context = replace(window.context, key_abs_positions=torch.tensor([*window.context.key_abs_positions.tolist(), 28]))
    future_key = replace(window, context=future_context,
                         k_post=torch.cat((window.k_post, window.k_post[:, -1:]), 1),
                         v=torch.cat((window.v, window.v[:, -1:]), 1))
    future_query = replace(window, context=replace(window.context, query_abs_positions=torch.tensor([24, 25, 26, 28])))
    for invalid in (future_key, future_query):
        for policy in ("uniform_archive", "oracle_archive"):
            with pytest.raises(ValueError, match="future positions"):
                select_archive(invalid, policy=policy, retention=1, online=False)


def test_single_head_scores_never_share_another_head_array():
    window = fixture()
    other = replace(window, context=replace(window.context, head_id=1))
    first_scores, other_scores = [0.] * 16, [0.] * 16
    first_scores[2], other_scores[9] = 1., 1.
    first = select_archive(window, policy="attention_archive", retention=.125, attention_scores=first_scores)
    second = select_archive(other, policy="attention_archive", retention=.125, attention_scores=other_scores)
    assert set(first["kept_indices"]).intersection(range(2, 10)) == {2}
    assert set(second["kept_indices"]).intersection(range(2, 10)) == {9}
    multi = replace(window, q_post=window.q_post.expand(-1, -1, 2, -1),
                    k_post=window.k_post.expand(-1, -1, 2, -1), v=window.v.expand(-1, -1, 2, -1))
    with pytest.raises(ValueError, match="single frozen head"):
        select(multi, "uniform_archive")


@pytest.mark.parametrize("retention", [-.1, 1.1, math.nan, math.inf, None, ".5", True])
def test_invalid_retention(retention):
    with pytest.raises(ValueError, match="retention"):
        select(fixture(), "recent_only", retention)


def test_invalid_window_policy_or_online_flag():
    with pytest.raises(TypeError, match="FrozenAttention"):
        select_archive(None, policy="recent_only", retention=.5)
    for policy in ("unknown", None, ["recent_only"]):
        with pytest.raises(ValueError, match="policy"):
            select_archive(fixture(), policy=policy, retention=.5)
    with pytest.raises(ValueError, match="online"):
        select_archive(fixture(), policy="recent_only", retention=.5, online=1)
