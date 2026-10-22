"""CPU mathematical checks for the independent research.v1 attention oracle."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from tempokv.research import attention_oracle as oracle
from tempokv.research.context import FrozenAttention, ProbeContext
from tempokv.spectral.metrics import relative_error


def dense_attention(q, k, v, log_count=None):
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(q.shape[-1])
    if log_count is not None:
        scores = scores + log_count.float()[None, None, None, :]
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float()).to(q.dtype)


def window(*, dtype=torch.float32, batch=1, heads=2, queries=5, tokens=13,
           current_start=9, positions=None, sink_frames=0, protect_recent_frames=0,
           attention_fn=dense_attention):
    rng = torch.Generator().manual_seed(71)
    q = torch.randn(batch, queries, heads, 4, generator=rng).to(dtype)
    k, v = [torch.randn(batch, tokens, heads, 4, generator=rng).to(dtype) for _ in range(2)]
    ctx = ProbeContext(0, 0, 0, 0, 0, 1, "denoise", 0, 0, current_start, 1, 1, 1,
                       torch.arange(current_start, current_start + queries),
                       torch.arange(tokens) if positions is None else torch.tensor(positions),
                       sink_frames=sink_frames, protect_recent_frames=protect_recent_frames)
    return FrozenAttention(ctx, q, k, v, attention_fn)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_reference_batches_queries_and_preserves_full_window_dtype_heads(dtype):
    calls = []
    def native(q, k, v):
        calls.append((q.shape, k.shape, v.shape, q.dtype, k.dtype, v.dtype))
        return dense_attention(q, k, v)
    w = window(dtype=dtype, batch=2, heads=3, attention_fn=native)
    actual = oracle.reference(w, query_batch_size=2)
    torch.testing.assert_close(actual, dense_attention(w.q_post, w.k_post, w.v), rtol=0, atol=0)
    assert actual.dtype == dtype
    assert [call[0][1] for call in calls] == [2, 2, 1]
    assert all(call[1] == w.k_post.shape and call[2] == w.v.shape for call in calls)
    assert all(call[0][2] == 3 and call[3:] == (dtype, dtype, dtype) for call in calls)


def test_no_removal_and_uniform_log_counts_use_original_callback():
    calls = []
    w = window()
    def native(q, k, v):
        assert k is w.k_post and v is w.v
        calls.append(q.shape[1])
        return dense_attention(q, k, v)
    w = replace(w, attention_fn=native)
    assert oracle.drop_window(w) is w
    for bias in (None, torch.zeros(13), torch.full((13,), 7.0)):
        result = oracle.evaluate(w, log_count=bias, query_batch_size=3)
        assert result["relative_error"] == 0
        assert result["backend_protocol"] == oracle.NATIVE_PROTOCOL
        assert result["output"] is result["reference_output"]
    assert calls == [3, 2] * 3
    row = oracle.leave_one_out(w, [()], query_batch_size=3)[0]
    assert row == {"candidate_index": 0, "attention_mass": 0.0, "local_attention_relative_error": 0.0}


def test_leave_one_out_is_independent_and_does_not_mutate_original():
    w = window()
    before = [t.clone() for t in (w.q_post, w.k_post, w.v)]
    blocks = [(0, 1), (2, 3), (0, 2, 4)]
    baseline = dense_attention(w.q_post, w.k_post, w.v)
    rows = oracle.leave_one_out(w, blocks, query_batch_size=2, block_batch_size=2, key_chunk_size=3)
    for index, (block, row) in enumerate(zip(blocks, rows)):
        keep = [i for i in range(13) if i not in block]
        expected = dense_attention(w.q_post, w.k_post[:, keep], w.v[:, keep])
        assert row["candidate_index"] == index
        assert row["local_attention_relative_error"] == pytest.approx(relative_error(baseline, expected))
    cumulative = dense_attention(w.q_post, w.k_post[:, 4:], w.v[:, 4:])
    assert rows[1]["local_attention_relative_error"] != pytest.approx(relative_error(baseline, cumulative))
    for actual, expected in zip((w.q_post, w.k_post, w.v), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_chunked_mass_matches_all_key_denominator_across_batches_heads():
    w = window(batch=2, heads=3)
    blocks = [(), (0,), (0, 2, 4, 6, 8), tuple(range(9))]
    probs = (torch.einsum("bqhd,bkhd->bhqk", w.q_post, w.k_post) / 2).softmax(-1)
    rows = oracle.leave_one_out(w, blocks, query_batch_size=2, block_batch_size=2, key_chunk_size=2)
    for block, row in zip(blocks, rows):
        expected = 0.0 if not block else float(probs[..., list(block)].sum(-1).mean())
        assert row["attention_mass"] == pytest.approx(expected, abs=2e-7)
    # The last four keys remain in the denominator although candidates are history only.
    assert rows[-1]["attention_mass"] < 1.0


@pytest.mark.parametrize("query_batch_size,block_batch_size,key_chunk_size", [(1, 1, 1), (2, 2, 3), (64, 4, 1024)])
def test_batch_sizes_do_not_change_oracle_definition(query_batch_size, block_batch_size, key_chunk_size):
    w = window()
    blocks = [(0, 1), (3,), (5, 6, 7)]
    rows = oracle.leave_one_out(w, blocks, query_batch_size, block_batch_size, key_chunk_size)
    canonical = oracle.leave_one_out(w, blocks)
    for actual, expected in zip(rows, canonical):
        assert actual["candidate_index"] == expected["candidate_index"]
        assert actual["attention_mass"] == pytest.approx(expected["attention_mass"], abs=2e-7)
        assert actual["local_attention_relative_error"] == pytest.approx(expected["local_attention_relative_error"], abs=2e-7)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 3e-7), (torch.bfloat16, 0.02)])
def test_nonuniform_log_counts_match_dense_weighted_softmax(dtype, tolerance, monkeypatch):
    monkeypatch.setattr(oracle, "_KEY_CHUNK_SIZE", 2)
    w = window(dtype=dtype, batch=2, heads=3)
    bias = torch.tensor([0., math.log(2), math.log(2), -1., 0., 0., 1., -1., 0., 1., 0., 0., 0.])
    result = oracle.evaluate(w, log_count=bias, query_batch_size=2)
    expected = dense_attention(w.q_post, w.k_post, w.v, bias)
    torch.testing.assert_close(result["output"], expected, rtol=tolerance, atol=tolerance)
    assert result["output"].dtype == dtype
    assert result["backend_protocol"] == oracle.COUNT_PROTOCOL
    assert result["relative_error"] == pytest.approx(relative_error(result["reference_output"], result["output"]))


def test_duplicate_key_value_count_merge_is_exact_special_case():
    w = window(tokens=4, current_start=3, heads=2)
    k, v = w.k_post.clone(), w.v.clone()
    k[:, 1], v[:, 1] = k[:, 0], v[:, 0]
    w = replace(w, k_post=k, v=v)
    result = oracle.evaluate(w, k=k[:, [0, 2, 3]], v=v[:, [0, 2, 3]],
                             log_count=torch.tensor([math.log(2), 0., 0.]), query_batch_size=2)
    torch.testing.assert_close(result["output"], result["reference_output"], rtol=2e-6, atol=2e-7)
    assert result["relative_error"] < 5e-7


def test_averaging_nonidentical_keys_and_values_with_count_is_only_approximate():
    w = window(tokens=3, current_start=2, queries=1, heads=1)
    q = torch.ones(1, 1, 1, 4)
    k = torch.tensor([-2., 2., 0.]).reshape(1, 3, 1, 1).expand(-1, -1, -1, 4).clone()
    v = torch.tensor([0., 2., 0.]).reshape(1, 3, 1, 1).expand_as(k).clone()
    w = replace(w, q_post=q, k_post=k, v=v)
    merged_k = torch.cat((k[:, :2].mean(1, keepdim=True), k[:, 2:]), dim=1)
    merged_v = torch.cat((v[:, :2].mean(1, keepdim=True), v[:, 2:]), dim=1)
    result = oracle.evaluate(w, k=merged_k, v=merged_v, log_count=torch.tensor([math.log(2), 0.]))
    assert result["relative_error"] > 0.5
    torch.testing.assert_close(result["output"], dense_attention(q, merged_k, merged_v,
                                                                torch.tensor([math.log(2), 0.])))


def test_conditioned_drop_compares_conditioned_reference_and_removes_kv_together():
    w = window()
    result = oracle.conditional_drop(w, (0, 3), conditioning_removed=(1, 5),
                                     query_batch_size=2, key_chunk_size=2)
    base_keep = [i for i in range(13) if i not in (1, 5)]
    final_keep = [i for i in range(13) if i not in (0, 1, 3, 5)]
    base = dense_attention(w.q_post, w.k_post[:, base_keep], w.v[:, base_keep])
    expected = dense_attention(w.q_post, w.k_post[:, final_keep], w.v[:, final_keep])
    torch.testing.assert_close(result["reference_output"], base)
    torch.testing.assert_close(result["output"], expected)
    assert result["local_attention_relative_error"] == pytest.approx(relative_error(base, expected))
    probs = (torch.einsum("bqhd,bkhd->bhqk", w.q_post, w.k_post[:, base_keep]) / 2).softmax(-1)
    assert result["attention_mass"] == pytest.approx(float(probs[..., [base_keep.index(0), base_keep.index(3)]].sum(-1).mean()), abs=2e-7)
    empty = oracle.conditional_drop(w, (), conditioning_removed=(1, 5))
    assert empty["relative_error"] == 0 and empty["attention_mass"] == 0
    with pytest.raises(ValueError, match="disjoint"):
        oracle.conditional_drop(w, (0,), conditioning_removed=(0,))


def test_sink_recent_current_protection_uses_original_gapped_positions():
    w = window(tokens=8, current_start=10, positions=[0, 1, 5, 6, 8, 9, 10, 11],
               sink_frames=2, protect_recent_frames=2)
    assert w.context.eligible_mask.nonzero().flatten().tolist() == [2, 3]
    w = replace(w, k_pre=w.k_post.clone())
    dropped = oracle.drop_window(w, [2])
    assert dropped.context.key_abs_positions.tolist() == [0, 1, 6, 8, 9, 10, 11]
    torch.testing.assert_close(dropped.k_post, w.k_post[:, [0, 1, 3, 4, 5, 6, 7]])
    torch.testing.assert_close(dropped.v, w.v[:, [0, 1, 3, 4, 5, 6, 7]])
    torch.testing.assert_close(dropped.k_pre, w.k_pre[:, [0, 1, 3, 4, 5, 6, 7]])
    for protected in (0, 1, 4, 5, 6, 7):
        with pytest.raises(ValueError, match="protected"):
            oracle.drop_window(w, [protected])
    with pytest.raises(ValueError, match="protected"):
        oracle.conditional_drop(w, (2,), conditioning_removed=(6,))
    # Original slot 3 must still identify absolute position 6 after conditioning on slot 2.
    result = oracle.conditional_drop(w, (3,), conditioning_removed=(2,))
    assert result["relative_error"] > 0


def test_empty_history_and_zero_reference_norm_are_defined():
    w = window(tokens=4, current_start=0)
    assert not bool(w.context.eligible_mask.any())
    assert oracle.leave_one_out(w, []) == []
    assert oracle.evaluate(w)["relative_error"] == 0
    zeros = replace(w, v=torch.zeros_like(w.v))
    assert oracle.evaluate(zeros)["relative_error"] == 0
    result = oracle.evaluate(zeros, v=torch.ones_like(zeros.v))
    assert math.isfinite(result["relative_error"]) and result["relative_error"] > 1e12


def test_sparse_pre_valid_mask_tracks_history_and_conditioned_drops():
    w = window()
    mask = torch.tensor([True, False, True, False, False, True, False,
                         True, False, True, True, True, True])
    w = replace(w, k_pre=w.k_post.clone(), pre_valid_mask=mask)
    dropped = oracle.drop_window(w, (1, 5))
    keep = [i for i in range(13) if i not in (1, 5)]
    assert dropped.pre_valid_mask.tolist() == mask[keep].tolist()
    assert dropped.pre_valid_mask.numel() == dropped.k_pre.shape[1] == dropped.k_post.shape[1]
    result = oracle.conditional_drop(w, (0, 3), conditioning_removed=(1, 5))
    final_keep = [i for i in range(13) if i not in (0, 1, 3, 5)]
    torch.testing.assert_close(result["output"], dense_attention(w.q_post, w.k_post[:, final_keep], w.v[:, final_keep]))
    assert w.pre_valid_mask.tolist() == mask.tolist()


def test_extreme_logits_merge_stably_without_exponential_overflow():
    w = window(tokens=3, current_start=2, heads=1)
    w = replace(w, q_post=torch.full_like(w.q_post, 1000.), k_post=torch.full_like(w.k_post, 1000.))
    bias = torch.tensor([0., 1., 2.])
    result = oracle.evaluate(w, log_count=bias, query_batch_size=2)
    assert bool(torch.isfinite(result["output"]).all())
    torch.testing.assert_close(result["output"], dense_attention(w.q_post, w.k_post, w.v, bias), rtol=1e-6, atol=1e-6)


def test_score_workspaces_are_bounded_by_query_and_key_chunks(monkeypatch):
    seen = []
    native_matmul = torch.matmul
    def checked_matmul(query, key):
        seen.append((query.shape[-2], key.shape[-1]))
        assert query.shape[-2] <= 2 and key.shape[-1] <= 2
        return native_matmul(query, key)
    monkeypatch.setattr(torch, "matmul", checked_matmul)
    monkeypatch.setattr(oracle, "_KEY_CHUNK_SIZE", 2)
    w = window()
    oracle.leave_one_out(w, [(0, 1, 2, 3, 4), (5,)], query_batch_size=2, key_chunk_size=2)
    oracle.evaluate(w, log_count=torch.arange(13).remainder(2).float(), query_batch_size=2)
    assert seen and len(seen) > 10


def test_removing_every_key_is_rejected_even_without_protected_keys():
    w = window(tokens=4, current_start=10)
    assert not bool(w.context.protected_mask.any())
    with pytest.raises(ValueError, match="every attention token"):
        oracle.drop_window(w, range(4))


@pytest.mark.parametrize("bad", [(-1,), (13,), (0, 0), (0.5,), (True,)])
def test_bad_original_sequence_indices_rejected(bad):
    with pytest.raises(ValueError):
        oracle.drop_window(window(), bad)


def test_invalid_shapes_counts_and_batch_sizes_are_rejected():
    w = window()
    for bias in (torch.zeros(12), torch.zeros(13, 1), torch.ones(13, dtype=torch.long),
                 torch.full((13,), float("nan"))):
        with pytest.raises(ValueError, match="log_count"):
            oracle.evaluate(w, log_count=bias)
    with pytest.raises(ValueError, match="dtype"):
        oracle.evaluate(w, k=w.k_post.double())
    with pytest.raises(ValueError, match="aligned"):
        oracle.evaluate(w, k=w.k_post[:, :-1])
    with pytest.raises(ValueError, match="at least one"):
        oracle.evaluate(w, k=w.k_post[:, :0], v=w.v[:, :0])
    for name in ("query_batch_size", "block_batch_size", "key_chunk_size"):
        with pytest.raises(ValueError, match=name):
            oracle.leave_one_out(w, [(0,)], **{name: 0})
