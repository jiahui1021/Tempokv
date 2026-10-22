"""Synthetic CPU 2x2 controls; no model-level interaction claim is made."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch
import torch.nn.functional as F

from tempokv.research import attention_oracle as oracle
from tempokv.research import interaction
from tempokv.research.context import FrozenAttention, ProbeContext


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def window(dtype=torch.float32, *, positions=None, current_start=6, sink=1, recent=1):
    generator = torch.Generator(device="cpu").manual_seed(914)
    q = torch.randn(2, 3, 2, 5, generator=generator).to(dtype)
    k, v = [torch.randn(2, 8, 2, 5, generator=generator).to(dtype) for _ in range(2)]
    ctx = ProbeContext(0, 2, 0, 0, 0, 6, "denoise", 0, 0, current_start, 1, 1, 1,
                       torch.arange(current_start, current_start + 3),
                       torch.arange(8) if positions is None else torch.tensor(positions), sink, recent)
    return FrozenAttention(ctx, q, k, v, sdpa)


def manual_quantize(source, eligible, bits, group_size):
    if bits == 16:
        return source
    result = source.clone()
    selected = source[:, eligible].float()
    qmax = 2 ** (bits - 1) - 1
    for start in range(0, selected.shape[-1], group_size):
        value = selected[..., start:start + group_size]
        scale = value.abs().amax(-1, keepdim=True) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        selected[..., start:start + group_size] = (value / scale).round().clamp(-qmax, qmax) * scale
    result[:, eligible] = selected.to(source.dtype)
    return result


def manual_error(reference, changed):
    return float((reference.float() - changed.float()).norm() / (reference.float().norm() + 1e-12))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bits", [(3, 2), (16, 2), (2, 16)])
def test_all_four_cells_match_independent_sdpa_with_common_mask(dtype, bits):
    w = window(dtype)
    kept = [0, 1, 3, 5, 6, 7]
    eligible = [1, 2, 3, 4]
    qk = manual_quantize(w.k_post, eligible, bits[0], 3)
    qv = manual_quantize(w.v, eligible, bits[1], 3)
    expected = {"A": sdpa(w.q_post, w.k_post, w.v),
                "B": sdpa(w.q_post, w.k_post[:, kept], w.v[:, kept]),
                "C": sdpa(w.q_post, qk, qv),
                "D": sdpa(w.q_post, qk[:, kept], qv[:, kept])}
    result = interaction.quant_prune_2x2(w, kept_indices=kept, k_bits=bits[0], v_bits=bits[1], group_size=3, query_batch_size=2)
    assert [row["strategy"] for row in result["rows"]] == ["A_exact", "B_pruning_only", "C_quant_only", "D_joint"]
    for row in result["rows"]:
        cell = row["cell"]
        torch.testing.assert_close(result["outputs"][cell], expected[cell], rtol=1e-6, atol=1e-7)
        assert row["local_attention_relative_error"] == pytest.approx(manual_error(expected["A"], expected[cell]), abs=2e-7)
        assert row["comparison_reference"] == "A_exact_same_snapshot"
        assert row["actual_retained_tokens"] == (8 if cell in ("A", "C") else len(kept))
        assert row["actual_retention"] == (1. if cell in ("A", "C") else len(kept) / 8)
    errors = {row["cell"]: row["local_attention_relative_error"] for row in result["rows"]}
    assert errors["A"] == 0.
    assert result["interaction"] == pytest.approx(errors["D"] - errors["C"] - errors["B"])
    assert result["D_minus_C"] == pytest.approx(errors["D"] - errors["C"])
    assert result["B_minus_A"] == errors["B"]
    assert result["joint_mask_frozen"] and result["frozen_kept_indices"] == kept


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_protected_values_roles_and_original_tensors_unchanged(dtype, monkeypatch):
    w = window(dtype)
    original = [value.clone() for value in (w.q_post, w.k_post, w.v)]
    state = torch.get_rng_state().clone()
    calls = []
    evaluate = oracle.evaluate
    def audited(window_arg, **kwargs):
        assert window_arg is w and window_arg.q_post is w.q_post
        calls.append(kwargs)
        return evaluate(window_arg, **kwargs)
    monkeypatch.setattr(oracle, "evaluate", audited)
    for kb, vb in ((16, 2), (2, 16)):
        calls.clear()
        interaction.quant_prune_2x2(w, kept_indices=[0, 1, 5, 6, 7], k_bits=kb, v_bits=vb, group_size=3)
        quant = calls[2]
        protected = [0, 5, 6, 7]
        assert torch.equal(quant["k"][:, protected], w.k_post[:, protected])
        assert torch.equal(quant["v"][:, protected], w.v[:, protected])
        if kb == 16: assert quant["k"] is w.k_post
        if vb == 16: assert quant["v"] is w.v
        assert calls[1]["k"].shape == calls[1]["v"].shape == calls[3]["k"].shape == calls[3]["v"].shape
    assert all(torch.equal(value, before) for value, before in zip((w.q_post, w.k_post, w.v), original))
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bits,kept", [((16, 16), list(range(8))), ((16, 16), [0, 2, 5, 6, 7]),
                                       ((3, 2), list(range(8)))])
def test_identity_precision_and_keep_all_null_controls(dtype, bits, kept):
    result = interaction.quant_prune_2x2(window(dtype), kept_indices=kept, k_bits=bits[0], v_bits=bits[1])
    assert result["interaction"] == 0.
    outputs = result["outputs"]
    if bits == (16, 16):
        assert torch.equal(outputs["A"], outputs["C"])
        assert torch.equal(outputs["B"], outputs["D"])
    if len(kept) == 8:
        assert torch.equal(outputs["A"], outputs["B"])
        assert torch.equal(outputs["C"], outputs["D"])
    assert result["error_correlation"]["flattened_signed"] is None
    assert result["error_correlation"]["status"] == "zero_variance"
    assert all(value is None for value in result["error_correlation"]["per_token_signed"])


def test_nonadditive_contrast_when_quantized_tokens_are_all_pruned():
    w = window()
    result = interaction.quant_prune_2x2(w, kept_indices=[0, 5, 6, 7], k_bits=2, v_bits=2)
    assert torch.equal(result["outputs"]["B"], result["outputs"]["D"])
    c_error = result["rows"][2]["local_attention_relative_error"]
    assert c_error > 0 and result["interaction"] == pytest.approx(-c_error)
    assert "nonlinear" in result["interpretation"] and "synergy conclusion" in result["interpretation"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kept", [[0, 5, 6, 7], [0, 1, 5, 6, 7], [0, 1, 2, 4, 5, 6, 7]])
def test_exact_byte_accounting_uses_full_snapshot_while_sparse_changes_compute_only(dtype, kept):
    w = window(dtype)
    result = interaction.quant_prune_2x2(w, kept_indices=kept, k_bits=3, v_bits=2, group_size=3)
    a, b, c, d = result["rows"]
    eligible_elements = 2 * 4 * 2 * 5
    protected_native = 2 * 4 * 2 * 5 * w.k_post.element_size() * 2
    quant_payload = math.ceil(eligible_elements * 3 / 8) + math.ceil(eligible_elements * 2 / 8)
    scales = 2 * (2 * 4 * 2) * math.ceil(5 / 3) * 4
    metadata = 2 * (16 + 4 * 8 + 16)
    native_payload = 2 * 8 * 2 * 5 * w.k_post.element_size() * 2
    assert c["payload_bytes"] == d["payload_bytes"] == quant_payload + protected_native
    assert c["quant_payload_bytes"] == quant_payload and c["native_protected_bytes"] == protected_native
    assert c["scale_bytes"] == d["scale_bytes"] == scales
    assert c["metadata_bytes"] == d["metadata_bytes"] == metadata
    assert c["logical_bytes"] == d["logical_bytes"] == quant_payload + protected_native + scales + metadata
    assert a["logical_bytes"] == b["logical_bytes"] == native_payload
    assert all(row["physical_kv_bytes"] == native_payload for row in result["rows"])
    assert b["attention_input_value_bytes"] == d["attention_input_value_bytes"] == native_payload * len(kept) // 8
    assert all(not row["cache_memory_reduction_claimed"] and not row["packed_storage_allocated"] for row in result["rows"])
    assert b["pruning_kind"] == d["pruning_kind"] == "compute_only_frozen_token_gather"


def test_signed_error_correlations_match_independent_pearson():
    result = interaction.quant_prune_2x2(window(), kept_indices=[0, 1, 3, 5, 6, 7], k_bits=2, v_bits=3)
    a = result["outputs"]["C"].double() - result["outputs"]["A"].double()
    b = result["outputs"]["B"].double() - result["outputs"]["A"].double()
    expected = torch.corrcoef(torch.stack((a.flatten(), b.flatten())))[0, 1]
    actual = result["error_correlation"]
    assert actual["flattened_signed"] == pytest.approx(float(expected), abs=2e-7)
    assert actual["per_token_shape"] == [2, 3]
    for index, (error_a, error_b) in enumerate(zip(a.flatten(0, 1), b.flatten(0, 1))):
        expected = torch.corrcoef(torch.stack((error_a.flatten(), error_b.flatten())))[0, 1]
        assert actual["per_token_signed"][index] == pytest.approx(float(expected), abs=2e-7)


def supplied(w, k=None, v=None, **metadata):
    return {"k_post": w.k_post.clone() if k is None else k, "v": w.v.clone() if v is None else v,
            "key_abs_positions": w.context.key_abs_positions.clone(), "snapshot_identity": w.context.identity, **metadata}


@pytest.mark.parametrize("known_storage", [False, True])
def test_supplied_decoded_values_used_exactly_with_explicit_snapshot_contract(known_storage, monkeypatch):
    w = window()
    k, v = w.k_post.clone(), w.v.clone()
    k[:, [1, 2, 3, 4]] += .125; v[:, [1, 2, 3, 4]] *= .8
    meta = {"storage_metadata": {"payload_bytes": 100, "scale_bytes": 16, "metadata_bytes": 24}} if known_storage else {}
    pair = supplied(w, k, v, **meta)
    def forbidden(*args, **kwargs):
        raise AssertionError("supplied decoded data must not use the independent quantizer")
    monkeypatch.setattr(interaction, "fake_quantize", forbidden)
    kept = [0, 2, 5, 6, 7]
    result = interaction.quant_prune_2x2(w, kept_indices=kept, quantized_pair=pair)
    assert torch.equal(result["outputs"]["C"], sdpa(w.q_post, k, v))
    assert torch.equal(result["outputs"]["D"], sdpa(w.q_post, k[:, kept], v[:, kept]))
    assert result["quantization_source"] == "supplied_decoded_snapshot"
    assert result["rows"][2]["source"] == result["rows"][3]["source"] == "supplied_decoded_snapshot"
    assert result["rows"][2]["k_bits"] is result["rows"][2]["v_bits"] is None
    assert result["rows"][2]["logical_bytes"] == (140 if known_storage else None)
    assert torch.equal(pair["k_post"], k) and torch.equal(pair["v"], v)


def test_optional_selection_diagnostic_never_replaces_main_joint_mask():
    w = window()
    frozen = [0, 1, 2, 5, 6, 7]
    calls = []
    def selector(candidate):
        calls.append(candidate)
        assert candidate.q_post is w.q_post and candidate.context is w.context
        if len(calls) == 1:
            return {"kept_indices": frozen, "scores": torch.arange(8).float()}
        return {"kept_indices": [0, 3, 4, 5, 6, 7], "scores": -torch.arange(8).float()}
    result = interaction.quant_prune_2x2(w, kept_indices=frozen, k_bits=2, v_bits=2, selection_fn=selector)
    consistency = result["selection_consistency"]
    assert len(calls) == 2 and not consistency["same_selection"]
    assert consistency["eligible_jaccard"] == 0. and consistency["rank_spearman"] == pytest.approx(-1.)
    assert consistency["frozen_mask_matches_baseline_selection"] and consistency["joint_mask_frozen"]
    assert result["rows"][1]["kept_indices"] == result["rows"][3]["kept_indices"] == frozen
    qk = manual_quantize(w.k_post, [1, 2, 3, 4], 2, 64)
    qv = manual_quantize(w.v, [1, 2, 3, 4], 2, 64)
    assert torch.equal(result["outputs"]["D"], sdpa(w.q_post, qk[:, frozen], qv[:, frozen]))


def test_original_absolute_mapping_with_rolling_gap_and_mask_order():
    w = window(positions=[0, 2, 5, 6, 8, 9, 10, 11], current_start=10, sink=1, recent=2)
    kept = [0, 2, 4, 5, 6, 7]
    result = interaction.quant_prune_2x2(w, kept_indices=torch.tensor(kept[::-1]), k_bits=3, v_bits=2)
    assert result["frozen_kept_indices"] == kept
    assert torch.equal(result["outputs"]["B"], sdpa(w.q_post, w.k_post[:, kept], w.v[:, kept]))
    assert result["rows"][2]["eligible_quantized_tokens"] == 3


def test_no_eligible_history_does_not_charge_phantom_quantization_metadata():
    w = window(sink=6)
    result = interaction.quant_prune_2x2(w, kept_indices=list(range(8)), k_bits=2, v_bits=2)
    a, _, c, _ = result["rows"]
    assert c["eligible_quantized_tokens"] == 0 and c["scale_bytes"] == c["metadata_bytes"] == 0
    assert c["logical_bytes"] == a["logical_bytes"]
    assert all(row["local_attention_relative_error"] == 0. for row in result["rows"])


@pytest.mark.parametrize("kept", [[], [0, 1, 5, 6], [1, 2, 5, 6, 7], [0, 0, 5, 6, 7],
                                  [-1, 0, 5, 6, 7], [0, 5, 6, 7, 8], [0., 5, 6, 7],
                                  [True, 0, 5, 6, 7], torch.tensor([0., 5., 6., 7.])])
def test_invalid_or_protected_omitting_masks_are_rejected(kept):
    with pytest.raises(ValueError):
        interaction.quant_prune_2x2(window(), kept_indices=kept)


def test_supplied_pair_rejects_missing_or_wrong_snapshot_and_modified_protection():
    w = window()
    bad = [ (w.k_post, w.v), {"k_post": w.k_post, "v": w.v},
            supplied(w, snapshot_identity="another_snapshot"),
            supplied(w, key_abs_positions=torch.arange(1, 9)),
            supplied(w, k=w.k_post.double()),
            supplied(w, storage_metadata={"payload_bytes": -1}),
            supplied(w, storage_metadata={"payload_bytes": 10, "scale_bytes": 1, "metadata_bytes": 1, "logical_bytes": 99})]
    protected_k = w.k_post.clone(); protected_k[:, 6] += .1
    protected_v = w.v.clone(); protected_v[:, 0] = 0
    bad.extend([supplied(w, protected_k), supplied(w, v=protected_v)])
    for pair in bad:
        with pytest.raises(ValueError):
            interaction.quant_prune_2x2(w, kept_indices=[0, 5, 6, 7], quantized_pair=pair)


def test_tempokv_only_snapshot_cannot_supply_missing_bf16_counterfactual():
    w = window(); w = replace(w, context=replace(w.context, source="tempokv"))
    with pytest.raises(ValueError, match="BF16 reference snapshot"):
        interaction.quant_prune_2x2(w, kept_indices=list(range(8)), quantized_pair=supplied(w))


def test_nonfinite_data_invalid_precision_and_query_batch_rejected():
    w = window()
    nonfinite = w.k_post.clone(); nonfinite[:, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        interaction.quant_prune_2x2(replace(w, k_post=nonfinite), kept_indices=list(range(8)))
    for options in ({"k_bits": 1}, {"v_bits": 15}, {"group_size": 0}, {"query_batch_size": 0}):
        with pytest.raises(ValueError):
            interaction.quant_prune_2x2(w, kept_indices=list(range(8)), **options)


def test_nondeterministic_reference_callback_cannot_mix_snapshots():
    w = window(); count = 0
    def changing(q, k, v):
        nonlocal count
        count += 1
        return sdpa(q, k, v) + count
    with pytest.raises(ValueError, match="reference changed"):
        interaction.quant_prune_2x2(replace(w, attention_fn=changing), kept_indices=[0, 5, 6, 7])
