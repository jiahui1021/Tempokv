"""CPU synthetic calibration math, with no model-level research conclusions."""
from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest
import torch

from tempokv.research import attention_oracle as oracle
from tempokv.research import interventions as ops
from tempokv.research import calibration
from tempokv.research.context import FrozenAttention, ProbeContext


def attention(q, k, v):
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(q.shape[-1])
    return torch.einsum("bhqk,bkhd->bqhd", logits.softmax(-1), v.float()).to(q.dtype)


def window(dtype=torch.float32, *, linear=True):
    gen = torch.Generator(device="cpu").manual_seed(818)
    q = torch.randn(1, 3, 2, 4, generator=gen).to(dtype)
    k = torch.randn(1, 6, 2, 4, generator=gen).to(dtype)
    v = torch.randn(1, 6, 2, 4, generator=gen).to(dtype)
    if linear:
        q.zero_(); v.fill_(1.)
    ctx = ProbeContext(0, 0, 0, 0, 0, 4, "denoise", 0, 0, 4, 1, 1, 1,
                       torch.arange(4, 7), torch.arange(6))
    return FrozenAttention(ctx, q, k, v, attention)


def role_nmse(source, changed, indices):
    before = source[:, indices].double()
    return float((changed[:, indices].double() - before).square().sum() / before.square().sum())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["gaussian", "low_rank", "temporal_noise", "quantization_like", "zero"])
@pytest.mark.parametrize("role", ["k", "v", "kv"])
def test_relative_rmse_protocol_reports_actual_per_role_and_protects_window(dtype, kind, role):
    w = window(dtype, linear=False)
    before = [tensor.clone() for tensor in (w.q_post, w.k_post, w.v)]
    rng = torch.get_rng_state().clone()
    result = calibration.relative_rmse_intervention(w, [0, 1, 3], .2, kind, role, 17)
    variant = result["variant"]
    assert result["protocol"] == result["matched_quantity"] == "relative_kv_rmse"
    assert result["requested_local_error"] is result["achieved_local_error"] is None
    assert result["target_nmse"] == pytest.approx(.04) and result["evaluations"] == 0
    error_sum = energy_sum = 0.
    for name, source, changed in (("k", w.k_post, variant.k_post), ("v", w.v, variant.v)):
        assert torch.equal(changed[:, [2, 4, 5]], source[:, [2, 4, 5]])
        if name not in role:
            assert torch.equal(source, changed)
            continue
        actual = role_nmse(source, changed, [0, 1, 3])
        assert result[f"nmse_{name}"] == pytest.approx(actual, abs=1e-14)
        assert actual == pytest.approx(1. if kind == "zero" else .04, rel=.02 if dtype == torch.bfloat16 else 2e-6)
        error_sum += float((changed[:, [0, 1, 3]].double() - source[:, [0, 1, 3]].double()).square().sum())
        energy_sum += float(source[:, [0, 1, 3]].double().square().sum())
    assert result["relative_kv_rmse"] == pytest.approx(math.sqrt(error_sum / energy_sum), abs=1e-14)
    assert result["matched"] == (kind != "zero")
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(original, snapshot) for original, snapshot in zip((w.q_post, w.k_post, w.v), before))


@pytest.mark.parametrize("kind", ["gaussian", "low_rank", "temporal_noise", "quantization_like"])
def test_local_attention_target_is_measured_and_stored_seed_scale_reproduce_reference(kind):
    w = window()
    state = torch.get_rng_state().clone()
    before = [tensor.clone() for tensor in (w.q_post, w.k_post, w.v)]
    result = calibration.calibrate_local_error(w, [0, 1, 2], kind=kind, role="v", target_error=.17,
                                             seed=29, query_batch_size=2, tolerance=.01)
    assert result["protocol"] == result["matched_quantity"] == "local_d_att"
    assert result["requested_relative_kv_rmse"] is None
    assert result["matched"] and result["status"] == "matched"
    assert result["achieved_local_error"] == pytest.approx(.17, rel=.01)
    assert result["relative_kv_rmse"] != pytest.approx(.17, rel=.01)
    assert result["target_nmse"] == pytest.approx(result["calibration_scale"] ** 2)
    recreated = ops.perturb(w, result["indices"], kind=result["kind"], role=result["role"],
                            target_nmse=result["target_nmse"], seed=result["seed"])
    assert torch.equal(recreated.k_post, result["variant"].k_post)
    assert torch.equal(recreated.v, result["variant"].v)
    measured = oracle.evaluate(w, k=result["variant"].k_post, v=result["variant"].v, query_batch_size=2)
    assert result["achieved_local_error"] == measured["relative_error"]
    assert result["evaluations"] <= 16
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(original, snapshot) for original, snapshot in zip((w.q_post, w.k_post, w.v), before))


def test_achievable_linear_attention_target_has_known_strength():
    w = window()
    unit = ops.perturb(w, [0, 1], role="v", kind="gaussian", target_nmse=1., seed=38)
    slope = oracle.evaluate(w, v=unit.v)["relative_error"]
    result = calibration.calibrate_local_error(w, [0, 1], role="v", target_error=.13, seed=38, tolerance=.001)
    assert result["matched"]
    assert result["calibration_scale"] == pytest.approx(.13 / slope, rel=.001)


def test_full_window_oracle_evaluation_cap_and_seed_direction_are_fixed(monkeypatch):
    w = window(linear=False)
    calls = []
    original_evaluate = oracle.evaluate
    original_perturb = ops.perturb
    perturb_calls = []
    def audited(window_arg, **kwargs):
        assert window_arg is w and kwargs["k"].shape == w.k_post.shape and kwargs["v"].shape == w.v.shape
        assert kwargs["query_batch_size"] == 2
        calls.append(kwargs)
        return original_evaluate(window_arg, **kwargs)
    def audited_perturb(window_arg, indices, **kwargs):
        perturb_calls.append(kwargs.copy())
        return original_perturb(window_arg, indices, **kwargs)
    monkeypatch.setattr(oracle, "evaluate", audited)
    monkeypatch.setattr(ops, "perturb", audited_perturb)
    result = calibration.calibrate_local_error(w, torch.tensor([0, 2]), role="k", target_error=1e6, seed=22,
                                             query_batch_size=2, max_evaluations=7)
    assert not result["matched"] and result["evaluations"] == len(calls) <= 7
    assert result["attempts"] <= 7 and len(perturb_calls) <= 7
    assert all(call["seed"] == 22 and call["role"] == "k" for call in perturb_calls)
    valid = [point for point in result["evaluated_points"] if point["error"] is not None]
    closest = min(valid, key=lambda point: (abs(point["error"] - 1e6), point["scale"]))
    assert result["achieved_local_error"] == closest["error"]
    assert result["calibration_scale"] == closest["scale"]


def test_impossible_unresponsive_attention_target_remains_explicitly_unmatched():
    w = window()
    result = calibration.calibrate_local_error(w, [0, 1], kind="gaussian", role="k", target_error=.2, seed=2)
    assert not result["matched"] and result["achieved_local_error"] == 0.
    assert result["status"] == "unmatched_unresponsive"
    assert result["calibration_scale"] == 0. and result["target_nmse"] == 0.


def test_bf16_floor_is_unmatched_and_never_fabricated_as_target_error():
    w = window(torch.bfloat16)
    result = calibration.calibrate_local_error(w, [0, 1], role="v", target_error=1e-7, seed=11)
    assert result["status"] == "unmatched_dtype_floor" and not result["matched"]
    assert result["achieved_local_error"] == 0.
    assert result["target_nmse"] == 0.
    assert any(point["error"] is not None and point["error"] > 1e-7 for point in result["evaluated_points"])
    relative = calibration.relative_rmse_intervention(w, [0, 1], 1e-10, "gaussian", "v", 11)
    assert not relative["matched"] and relative["relative_kv_rmse"] == 0.
    assert relative["status"] == "unmatched_dtype_rounding"


@pytest.mark.parametrize("target", [0., .1])
def test_zero_reference_output_is_not_a_successful_relative_calibration(target):
    w = window(); w = replace(w, v=torch.zeros_like(w.v))
    result = calibration.calibrate_local_error(w, [0, 1], role="k", target_error=target, seed=1)
    assert result["status"] == "zero_reference" and not result["matched"]
    assert result["achieved_local_error"] is None and result["evaluations"] == 1
    assert result["variant"].k_post is w.k_post and result["variant"].v is w.v


def test_zero_input_role_energy_and_empty_history_do_not_trigger_search():
    w = window(); w = replace(w, k_post=torch.zeros_like(w.k_post))
    local = calibration.calibrate_local_error(w, [0, 1], role="k", target_error=.1, seed=0)
    assert local["status"] == "zero_input_energy" and not local["matched"] and local["evaluations"] == 1
    relative = calibration.relative_rmse_intervention(w, [0, 1], .1, "gaussian", "k", 0)
    assert relative["status"] == "zero_input_energy" and not relative["matched"]
    assert relative["nmse_k"] is None and relative["relative_kv_rmse"] is None
    for indices in ([], torch.empty(0, dtype=torch.int64)):
        empty = calibration.calibrate_local_error(w, indices, target_error=.1, seed=0)
        assert empty["status"] == "no_targets" and not empty["matched"] and empty["evaluations"] == 1


@pytest.mark.parametrize("kind", ["gaussian", "low_rank", "temporal_noise", "quantization_like", "zero"])
def test_target_zero_is_exact_null_without_global_random_effect(kind):
    w = window(torch.bfloat16)
    state = torch.get_rng_state().clone()
    result = calibration.calibrate_local_error(w, [0, 1], kind=kind, target_error=0., seed=10)
    assert result["variant"].k_post is w.k_post and result["variant"].v is w.v
    assert result["matched"] and result["status"] == "exact_null"
    assert result["achieved_local_error"] == result["target_nmse"] == result["calibration_scale"] == 0.
    assert result["evaluations"] == 1 and torch.equal(torch.get_rng_state(), state)
    relative = calibration.relative_rmse_intervention(w, [0, 1], 0., kind, "kv", 10)
    assert relative["matched"] and relative["status"] == "exact_null"


def test_zero_ablation_is_discrete_and_does_not_bisect_strength():
    w = window()
    matched = calibration.calibrate_local_error(w, [0, 1, 2], kind="zero", role="v", target_error=.5, seed=2)
    assert matched["matched"] and matched["evaluations"] == 2 and matched["relative_kv_rmse"] == 1.
    unmatched = calibration.calibrate_local_error(w, [0, 1, 2], kind="zero", role="v", target_error=.2, seed=2)
    assert not unmatched["matched"] and unmatched["status"] == "unmatched_discrete_ablation"
    assert unmatched["evaluations"] == 2


def test_nonmonotonic_bracket_refinement_considers_later_crossings():
    # Observed error rises, then falls. No monotonic trend may discard the
    # later adjacent crossing, which is closest to the requested target.
    points = [{"scale": 0., "error": 0.}, {"scale": 1., "error": .8},
              {"scale": 2., "error": .1}, {"scale": 3., "error": .7}]
    strength = calibration._next_strength(points, .2)
    assert 1. < strength < 3.
    assert strength != pytest.approx(.25)


def test_metadata_is_strict_json_and_contains_branch_reproduction_inputs():
    result = calibration.calibrate_local_error(window(), [0, 1], role="v", target_error=.1, seed=4)
    metadata = calibration.calibration_metadata(result)
    encoded = json.dumps(metadata, allow_nan=False)
    decoded = json.loads(encoded)
    assert "variant" not in decoded
    for field in ("target_nmse", "seed", "kind", "role", "indices", "protocol", "matched", "status"):
        assert decoded[field] == result[field]
    assert decoded["intervention_details"]["seed"] == 4


def test_protected_tokens_and_invalid_requests_are_rejected():
    w = window()
    ctx = replace(w.context, sink_frames=1, protect_recent_frames=1)
    w = replace(w, context=ctx)
    for protected in (0, 3, 4, 5):
        with pytest.raises(ValueError, match="unprotected historical"):
            calibration.calibrate_local_error(w, [protected], target_error=.1, seed=0)
        with pytest.raises(ValueError, match="unprotected historical"):
            calibration.relative_rmse_intervention(w, [protected], .1, "gaussian", "kv", 0)
    for options in ({"target_error": -1}, {"target_error": float("nan")}, {"max_evaluations": 0},
                    {"query_batch_size": 0}, {"tolerance": -.1}, {"kind": "drop"}):
        with pytest.raises(ValueError):
            calibration.calibrate_local_error(w, [1], **{"target_error": .1, "seed": 0, **options})
    for strength in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            calibration.relative_rmse_intervention(w, [1], strength)


def test_one_evaluation_budget_is_respected_without_claiming_target_match():
    result = calibration.calibrate_local_error(window(), [0], role="v", target_error=.3, seed=3, max_evaluations=1)
    assert result["status"] == "unmatched_evaluation_limit" and result["evaluations"] == 1
    assert not result["matched"] and result["achieved_local_error"] == 0.
