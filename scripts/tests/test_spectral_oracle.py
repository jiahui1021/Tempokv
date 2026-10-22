from __future__ import annotations

import numpy as np
import pytest
import torch

from tempokv.spectral.compression import evaluate_budgets, load_snapshot, save_snapshot, select_tokens
from tempokv.spectral.config import Compression
from tempokv.spectral.metrics import relative_error
from tempokv.spectral.sensitivity import (
    Block, attention_mass, effective_positions, leave_one_block_out, reference_attention, remove_attention,
)


def tensors(dtype=torch.float32):
    rng = torch.Generator().manual_seed(100)
    q = torch.randn(1, 5, 1, 8, generator=rng).to(dtype)
    k, v = [torch.randn(1, 13, 1, 8, generator=rng).to(dtype) for _ in range(2)]
    return q, k, v


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_oracle_empty_removal_reproducibility_and_dtype(dtype):
    q, k, v = tensors(dtype)
    before = [x.clone() for x in (q, k, v)]
    observed = []
    def attention(q, k, v):
        observed.append((q.dtype, k.dtype, v.dtype, q.shape[2], k.shape[2], v.shape[2]))
        return reference_attention(q, k, v)
    blocks = [Block(0, 0, (0, 1, 2)), Block(0, 1, (3, 4))]
    ref, scores = leave_one_block_out(q, k, v, blocks, attention_fn=attention, chunk_size=3)
    assert relative_error(ref, remove_attention(q, k, v, [], attention_fn=attention)) == 0
    _, repeated = leave_one_block_out(q, k, v, blocks, attention_fn=attention, chunk_size=7)
    np.testing.assert_allclose([s["oracle_error"] for s in scores], [s["oracle_error"] for s in repeated], rtol=0, atol=0)
    assert scores[0]["oracle_error"] != scores[1]["oracle_error"]
    assert all(entry == (dtype, dtype, dtype, 1, 1, 1) for entry in observed)
    for actual, expected in zip((q, k, v), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_chunked_mass_uses_all_kv_in_denominator():
    q, k, v = tensors()
    blocks = [Block(0, 0, (0, 2, 4)), Block(0, 1, tuple(range(13)))]
    scores = torch.softmax(torch.einsum("bqhd,bkhd->bhqk", q, k) / 8**0.5, dim=-1)
    actual = attention_mass(q, k, blocks, chunk_size=2)
    assert actual[0] == pytest.approx(float(scores[..., [0, 2, 4]].sum(-1).mean()), rel=1e-6)
    assert actual[1] == pytest.approx(1.0, abs=1e-6)


def test_rolling_absolute_mapping_and_explicit_gaps():
    # Current frame 7 occupies slots 3..4 after rolling a five-frame buffer.
    positions = effective_positions(tokens=20, current_start=28, local_start_index=12,
                                    attention_start=0, frame_seqlen=4)
    np.testing.assert_array_equal(positions, np.arange(16, 36))
    positions = effective_positions(tokens=12, current_start=28, local_start_index=12,
                                    attention_start=8, frame_seqlen=4)
    np.testing.assert_array_equal(positions, np.arange(24, 36))
    with pytest.raises(ValueError, match="explicit"):
        effective_positions(tokens=12, current_start=28, local_start_index=12,
                            attention_start=8, frame_seqlen=4, sink_frames=1)
    explicit = np.r_[np.arange(4), np.arange(28, 36)]
    np.testing.assert_array_equal(effective_positions(tokens=12, current_start=28,
                                                     local_start_index=4, attention_start=0,
                                                     frame_seqlen=4, sink_frames=1,
                                                     absolute_positions=explicit), explicit)
    with pytest.raises(ValueError):
        effective_positions(tokens=4, current_start=0, local_start_index=8, attention_start=0, frame_seqlen=4)
    with pytest.raises(ValueError):
        effective_positions(tokens=2, current_start=0, local_start_index=0, attention_start=0,
                            frame_seqlen=4, absolute_positions=[0.0, 1.5])


def blocks_and_scores():
    blocks = [Block(0, 0, (0, 1, 2, 3), .3, .5, .8),
              Block(0, 1, (4, 5), .7, .4, .2),
              Block(1, 0, (6, 7, 8), .2, .1, .6)]
    scores = [{"attention_mass": x, "oracle_error": x / 2} for x in (.1, .5, .3)]
    return blocks, scores


@pytest.mark.parametrize("policy", ["random", "uniform", "similarity", "attention", "frequency", "frequency_attention", "oracle_greedy"])
def test_every_policy_exact_token_budget_with_unequal_blocks(policy):
    blocks, scores = blocks_and_scores()
    for budget in range(10):
        keep, partial = select_tokens(blocks, budget, policy=policy, rows=scores, seed=7)
        assert len(keep) == budget == len(set(keep))
        assert set(keep) <= set(range(9))
        assert (keep, partial) == select_tokens(blocks, budget, policy=policy, rows=scores, seed=7)


def test_joint_deletion_keeps_current_and_protected_tokens():
    q, k, v = tensors()
    blocks, scores = blocks_and_scores()
    config = Compression(retention_ratios=(.25, .5, .75, 1), policies=("random", "uniform", "frequency"))
    observed = []
    original = k.clone()
    def attention(q, key, value):
        # Last four tokens are protected (recent + current) and always present.
        for token in range(9, 13):
            assert any(torch.equal(original[:, token], key[:, i]) for i in range(key.shape[1]))
        observed.append(key.shape[1])
        return reference_attention(q, key, value)
    rows = evaluate_budgets(q, k, v, blocks, scores, config, attention_fn=attention, protected_indices=range(9, 13))
    for row in rows:
        assert row["actual_retained_tokens"] == row["kept_candidate_tokens"] + 4
        if row["retention_ratio"] == 1:
            assert row["relative_attention_output_error"] == 0
    for ratio in config.retention_ratios:
        assert len({r["actual_retained_tokens"] for r in rows if r["retention_ratio"] == ratio}) == 1
    with pytest.raises(ValueError, match="protected"):
        evaluate_budgets(q, k, v, blocks, scores, config, protected_indices=[8])


def test_nan_score_missingness_and_snapshot_roundtrip(tmp_path):
    q, k, v = tensors(torch.bfloat16)
    blocks = [Block(0, 0, (0, 1))]
    scores = [{"attention_mass": .1, "oracle_error": .2}]
    rows = evaluate_budgets(q, k, v, blocks, scores, Compression(policies=("similarity",)))
    assert all(row["status"].startswith("not measured") for row in rows)
    path = tmp_path / "snapshot.npz"
    save_snapshot(path, q, k, v, blocks, scores, {"test": True})
    qq, kk, vv, bb, ss, mm = load_snapshot(path)
    for a, b in zip((q, k, v), (qq, kk, vv)):
        assert a.dtype == b.dtype
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert bb[0].indices == blocks[0].indices and ss == scores and mm == {"test": True}
    with pytest.raises(FileExistsError):
        save_snapshot(path, q, k, v, blocks, scores, {})
    with pytest.raises(ValueError, match="every attention token"):
        remove_attention(q, k, v, list(range(k.shape[1])))
