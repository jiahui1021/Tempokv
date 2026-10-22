from __future__ import annotations

from dataclasses import replace
import math
import random

import numpy as np
import pytest
import torch

from tempokv.research.attention_oracle import drop_window, reference
from tempokv.research.context import FrozenAttention, ProbeContext
from tempokv.research.selection import POLICIES, select_history


def native_attention(q, k, v):
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * q.shape[-1] ** -.5
    return torch.einsum("bhqk,bkhd->bqhd", logits.softmax(-1), v.float()).to(q.dtype)


def fixture(dtype=torch.float32):
    # A rolled window has an absolute-coordinate gap after the retained sink.
    positions = torch.tensor([0, 1, *range(8, 28)])
    context = ProbeContext("selection-prompt", 37, 0, 2, 0, 12, "denoise", 0, 999,
                           24, 2, 1, 2, torch.arange(24, 28), positions,
                           sink_frames=1, protect_recent_frames=1)
    axis = torch.arange(positions.numel() * 3, dtype=torch.float32).reshape(1, -1, 1, 3)
    k = (.1 * torch.sin(axis * .27)).to(dtype)
    v = (.2 + torch.cos(axis * .13)).to(dtype)
    q = torch.tensor([[[[.1, -.3, .7]], [[.2, .4, -.1]], [[-.3, .2, .5]], [[.5, -.2, .3]]]], dtype=dtype)
    window = FrozenAttention(context, q, k, v, native_attention)
    candidates = [(2, 3, 4), (5, 6), (8, 9, 10, 11), (13,)]
    scores = {"similarity": [.8, .3, .9, .1], "attention": [5., 1., 2., 6.],
              "frequency": [2., 4., 1., 3.], "frequency_qk": [1., 2., 5., 3.]}
    return window, candidates, scores


def select(window, candidates, scores, policy, retention=.35, **kwargs):
    return select_history(window, candidates, policy=policy, retention=retention, scores=scores,
                          oracle_errors=[.1, .5, .3, .2], **kwargs)


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_has_identical_exact_token_budget_on_unequal_blocks(policy):
    window, candidates, scores = fixture()
    result = select(window, candidates, scores, policy)
    assert result["candidate_tokens"] == 10
    assert result["kept_candidate_tokens"] == 4  # half-up(10 * .35)
    assert result["actual_retained_tokens"] == 16
    assert len(result["removed_indices"]) == 6 and len(result["kept_indices"]) == 16
    assert result["budget_cohort"] == "tokens:16"
    assert result["untouched_tokens"] == 12 and result["protected_tokens"] == 8
    assert result["actual_candidate_retention"] == .4
    assert result["actual_retention"] == pytest.approx(16 / 22)
    assert result["score_scope"] == "current_head_sampled_candidate_blocks"
    assert result["diagnostic_only"] is (policy == "oracle")


def test_boundary_block_partial_tokens_and_decimal_half_up_are_exact():
    window, candidates, scores = fixture()
    result = select(window, candidates, scores, "frequency")
    assert result["kept_candidate_indices"] == (2, 5, 6, 13)
    assert result["partial_block_tokens"] == 1
    assert select(window, candidates, scores, "frequency", .05)["kept_candidate_tokens"] == 1
    assert select(window, candidates, scores, "frequency", .15)["kept_candidate_tokens"] == 2
    assert select(window, candidates, scores, "frequency", .25)["kept_candidate_tokens"] == 3
    for result in (select(window, candidates, scores, policy) for policy in POLICIES):
        kept = set(result["kept_candidate_indices"])
        partial = [len(kept.intersection(block)) for block in candidates if 0 < len(kept.intersection(block)) < len(block)]
        assert len(partial) <= 1
        assert result["partial_block_tokens"] == (partial[0] if partial else 0)


def test_similarity_keeps_less_redundant_frequency_and_attention_keep_high_scores():
    window, candidates, scores = fixture()
    low = select(window, candidates, scores, "similarity", .1)
    high = select(window, candidates, scores, "frequency", .1)
    attention = select(window, candidates, scores, "attention", .1)
    oracle = select(window, candidates, scores, "oracle", .1)
    assert low["kept_candidate_indices"] == (13,)
    assert high["kept_candidate_indices"] == (5,)
    assert attention["kept_candidate_indices"] == (13,)
    assert oracle["kept_candidate_indices"] == (5,)
    assert "not a joint subset optimum" in oracle["oracle_interpretation"]


def test_uniform_spreads_absolute_position_order_and_is_input_order_independent():
    window, candidates, scores = fixture()
    first = select(window, candidates, scores, "uniform")
    second = select(window, list(reversed(candidates)), {}, "uniform")
    assert first["kept_candidate_indices"] == second["kept_candidate_indices"] == (2, 3, 4, 13)
    assert first["partial_block_tokens"] == 0
    # Frame/spatial coordinates use the original absolute positions after rolling.
    assert window.context.key_abs_positions[list(first["kept_candidate_indices"])].tolist() == [8, 9, 10, 19]


@pytest.mark.parametrize("policy", POLICIES)
def test_sink_recent_current_and_nonselected_history_are_untouched(policy):
    window, candidates, scores = fixture()
    result = select(window, candidates, scores, policy, 0)
    candidate = {token for block in candidates for token in block}
    assert result["removed_indices"] == tuple(sorted(candidate))
    assert result["kept_indices"] == tuple(i for i in range(22) if i not in candidate)
    protected = set(window.context.protected_mask.nonzero().flatten().tolist())
    assert protected == {0, 1, 16, 17, 18, 19, 20, 21}
    assert protected.issubset(result["kept_indices"])
    assert {7, 12, 14, 15}.issubset(result["kept_indices"])  # unlisted eligible history
    assert result["protected_tokens"] == 8 and result["kept_candidate_tokens"] == 0


@pytest.mark.parametrize("retention", [0, .25, 1])
@pytest.mark.parametrize("policy", POLICIES)
def test_no_candidates_is_full_identity_without_scores(policy, retention):
    window, _, _ = fixture()
    result = select_history(window, [], policy=policy, retention=retention, scores={})
    assert result["kept_indices"] == tuple(range(22)) and result["removed_indices"] == ()
    assert result["candidate_tokens"] == result["kept_candidate_tokens"] == result["partial_block_tokens"] == 0
    assert result["actual_retained_tokens"] == 22 and result["actual_retention"] == 1


@pytest.mark.parametrize("policy", POLICIES)
def test_full_retention_identity_needs_no_scores(policy):
    window, candidates, _ = fixture()
    result = select_history(window, candidates, policy=policy, retention=1, scores={})
    assert result["kept_indices"] == tuple(range(22)) and result["removed_indices"] == ()
    assert result["kept_candidate_tokens"] == 10


def test_global_rng_and_input_tensors_scores_and_candidate_lists_are_unchanged():
    window, candidates, scores = fixture()
    original = tuple(value.clone() for value in (window.q_post, window.k_post, window.v))
    blocks_before = list(candidates)
    score_before = {name: values.copy() for name, values in scores.items()}
    state_torch, state_numpy, state_python = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    first = select(window, candidates, scores, "random", seed=83)
    second = select(window, candidates, scores, "random", seed=83)
    assert first == second
    assert torch.equal(state_torch, torch.get_rng_state())
    after_numpy = np.random.get_state()
    assert state_numpy[0] == after_numpy[0] and np.array_equal(state_numpy[1], after_numpy[1]) and state_numpy[2:] == after_numpy[2:]
    assert state_python == random.getstate()
    assert candidates == blocks_before and scores == score_before
    for before, current in zip(original, (window.q_post, window.k_post, window.v)):
        torch.testing.assert_close(before, current, atol=0, rtol=0)
    sets = {select(window, candidates, scores, "random", seed=seed)["kept_candidate_indices"] for seed in range(8)}
    assert len(sets) > 1


def test_nan_and_missing_scores_are_last_with_absolute_position_ties():
    window, candidates, _ = fixture()
    result = select_history(window, candidates, policy="attention", retention=.7,
                            scores={"attention": [math.nan, .5, .5, None]})
    assert result["kept_candidate_indices"] == (2, 5, 6, 8, 9, 10, 11)
    assert result["partial_block_tokens"] == 1 and result["finite_score_blocks"] == 2
    reversed_result = select_history(window, list(reversed(candidates)), policy="attention", retention=.7,
                                     scores={"attention": [None, .5, .5, math.nan]})
    assert reversed_result["kept_indices"] == result["kept_indices"]


@pytest.mark.parametrize("policy", ["similarity", "attention", "frequency", "frequency_qk", "oracle"])
def test_all_missing_rank_scores_explicitly_fail_without_random_fallback(policy):
    window, candidates, _ = fixture()
    with pytest.raises(ValueError, match="not measured: no finite"):
        select_history(window, candidates, policy=policy, retention=.5,
                       scores={policy: [None, math.nan, math.inf, -math.inf]},
                       oracle_errors=[None, math.nan, math.inf, -math.inf])
    with pytest.raises(ValueError, match="not measured: no finite"):
        select_history(window, candidates, policy=policy, retention=.5, scores={})


def test_score_arrays_are_current_head_only_and_no_result_cache_is_shared():
    window, candidates, scores = fixture()
    other = replace(window, context=replace(window.context, head_id=1))
    first = select_history(window, candidates, policy="attention", retention=.2, scores={"attention": [9, 0, 0, 0]})
    second = select_history(other, candidates, policy="attention", retention=.2, scores={"attention": [0, 9, 0, 0]})
    assert first["kept_candidate_indices"] == (2, 3)
    assert second["kept_candidate_indices"] == (5, 6)
    assert first == select_history(window, candidates, policy="attention", retention=.2, scores={"attention": [9, 0, 0, 0]})
    for invalid in (np.ones((2, 4)), torch.ones(2, 4), np.ones((4, 2))):
        with pytest.raises(ValueError, match="current head|head arrays"):
            select_history(window, candidates, policy="attention", retention=.2, scores={"attention": invalid})
    multi = replace(window, q_post=window.q_post.expand(-1, -1, 2, -1),
                    k_post=window.k_post.expand(-1, -1, 2, -1), v=window.v.expand(-1, -1, 2, -1))
    with pytest.raises(ValueError, match="single-head"):
        select(multi, candidates, scores, "attention")


@pytest.mark.parametrize("retention", [0, .5, 1])
def test_oracle_is_never_permitted_online_even_for_null_controls(retention):
    window, candidates, scores = fixture()
    for blocks in (candidates, []):
        with pytest.raises(ValueError, match="diagnostic only.*online"):
            select(window, blocks, scores, "oracle", retention, online=True)
    for policy in set(POLICIES) - {"oracle"}:
        assert select(window, candidates, scores, policy, retention, online=True) == select(window, candidates, scores, policy, retention, online=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_selected_original_sets_gather_k_and_v_together_in_native_dtype(dtype):
    window, candidates, scores = fixture(dtype)
    result = select(window, candidates, scores, "frequency")
    calls = []
    def observed(q, k, v):
        assert q.dtype == k.dtype == v.dtype == dtype
        calls.append((k.clone(), v.clone()))
        return native_attention(q, k, v)
    window = replace(window, attention_fn=observed)
    full = reference(window, query_batch_size=2)
    changed = drop_window(window, result["removed_indices"])
    output = reference(changed, query_batch_size=2)
    assert output.dtype == full.dtype == dtype
    assert changed.context.key_abs_positions.tolist() == window.context.key_abs_positions[list(result["kept_indices"])].tolist()
    torch.testing.assert_close(changed.k_post, window.k_post[:, result["kept_indices"]], atol=0, rtol=0)
    torch.testing.assert_close(changed.v, window.v[:, result["kept_indices"]], atol=0, rtol=0)
    for gathered_k, gathered_v in calls[2:]:
        torch.testing.assert_close(gathered_k, changed.k_post, atol=0, rtol=0)
        torch.testing.assert_close(gathered_v, changed.v, atol=0, rtol=0)
    null = select(window, candidates, scores, "frequency", 1)
    assert drop_window(window, null["removed_indices"]) is window


@pytest.mark.parametrize("candidates", [[(0,)], [(16,)], [(18,)], [(24,)], [(-1,)], [()], [(2, 2)], [(2, 3), (3, 4)], [(2.,)], [(True,)], ["2"], [torch.tensor([[2]])]])
def test_invalid_overlapping_outside_or_protected_candidate_blocks(candidates):
    window, _, scores = fixture()
    with pytest.raises(ValueError):
        select(window, candidates, scores, "random")


@pytest.mark.parametrize("retention", [-.1, 1.1, math.nan, math.inf, True, ".5", None])
def test_invalid_retention(retention):
    window, candidates, scores = fixture()
    with pytest.raises(ValueError, match="retention"):
        select(window, candidates, scores, "random", retention)


def test_invalid_score_shape_scalar_type_policy_seed_and_context():
    window, candidates, scores = fixture()
    for values in ([1], "1234", 1, [[1], [2], [3], [4]], [True] * 4):
        with pytest.raises(ValueError):
            select_history(window, candidates, policy="attention", retention=.5, scores={"attention": values})
    for seed in (True, 1.5, "0"):
        with pytest.raises(ValueError, match="seed"):
            select(window, candidates, scores, "random", seed=seed)
    with pytest.raises(ValueError, match="policy"):
        select(window, candidates, scores, "unknown")
    with pytest.raises(ValueError, match="online"):
        select(window, candidates, scores, "random", online=1)
    with pytest.raises(TypeError, match="scores"):
        select_history(window, candidates, policy="random", retention=.5, scores=None)
    with pytest.raises(TypeError, match="FrozenAttention"):
        select_history(None, candidates, policy="random", retention=.5, scores={})


def test_numpy_and_torch_scalar_score_sequences_and_integer_block_tensors():
    window, candidates, _ = fixture()
    array = select_history(window, [torch.tensor(block) for block in candidates], policy="attention", retention=.2,
                           scores={"attention": np.array([0., 9., 0., 0.])})
    tensor = select_history(window, candidates, policy="attention", retention=.2,
                            scores={"attention": torch.tensor([0., 9., 0., 0.])})
    assert array["kept_candidate_indices"] == tensor["kept_candidate_indices"] == (5, 6)
