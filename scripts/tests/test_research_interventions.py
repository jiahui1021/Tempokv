from __future__ import annotations

from dataclasses import replace as dataclass_replace
import math

import pytest
import torch
import torch.nn.functional as F

from tempokv.research.context import FrozenAttention, ProbeContext
from tempokv.research.interventions import identity, drop, replace, merge, perturb


def rotate(pre, coordinates):
    angle = (coordinates[:, 0] * .2 + coordinates[:, 1] * .3 + coordinates[:, 2] * .7)[None, :, None, None]
    pairs = pre.float().reshape(*pre.shape[:-1], -1, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    return torch.stack((real * angle.cos() - imag * angle.sin(),
                        real * angle.sin() + imag * angle.cos()), -1).flatten(-2).to(pre.dtype)


def make_window(dtype=torch.float32, *, positions=None, sink_frames=0, recent_frames=0, current_start=12):
    positions = torch.arange(16) if positions is None else torch.tensor(positions, dtype=torch.int64)
    ctx = ProbeContext(0, 7, 0, 0, 0, 3, "denoise", 0, 0, current_start, 1, 2, 2,
                       torch.arange(current_start, current_start + 2), positions,
                       sink_frames, recent_frames)
    gen = torch.Generator(device="cpu").manual_seed(55)
    pre = torch.randn(2, len(positions), 2, 4, generator=gen).to(dtype)
    values = torch.randn(2, len(positions), 2, 4, generator=gen).to(dtype)
    queries = torch.randn(2, 2, 2, 4, generator=gen).to(dtype)
    coords = torch.stack((positions // 4, (positions // 2) % 2, positions % 2), 1).float()
    return FrozenAttention(ctx, queries, rotate(pre, coords), values, None, pre, rotate)


def independent_attention(window, variant):
    return F.scaled_dot_product_attention(window.q_post.transpose(1, 2), variant.k_post.transpose(1, 2),
        variant.v.transpose(1, 2), attn_mask=variant.log_count[None, None, None, :]).transpose(1, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_identity_drop_pairing_and_null(dtype):
    window = make_window(dtype)
    original_k, original_v = window.k_post.clone(), window.v.clone()
    base = identity(window)
    assert base.k_post is window.k_post and base.v is window.v
    assert base.provenance == tuple((i,) for i in range(16))
    assert base.token_count == 16 and torch.count_nonzero(base.log_count) == 0
    assert torch.equal(base.coordinates, torch.tensor([[i // 4, i // 2 % 2, i % 2] for i in range(16)]).float())
    null = drop(window, [])
    assert null.k_post is window.k_post
    result = drop(window, [1, 9])
    keep = [i for i in range(16) if i not in {1, 9}]
    assert torch.equal(result.k_post, original_k[:, keep])
    assert torch.equal(result.v, original_v[:, keep])
    assert result.provenance == tuple((i,) for i in keep)
    assert torch.equal(window.k_post, original_k) and torch.equal(window.v, original_v)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_replacement_uses_donor_pre_at_target_absolute_coordinate(dtype):
    window = make_window(dtype, positions=[0, 1, 8, 9, 12, 13], sink_frames=1)
    result = replace(window, [2, 3], [0, 0])
    target_coords = torch.tensor([[2., 0., 0.], [2., 0., 1.]])
    expected = rotate(window.k_pre[:, [0, 0]], target_coords)
    assert torch.equal(result.k_post[:, [2, 3]], expected)
    assert torch.equal(result.v[:, [2, 3]], window.v[:, [0, 0]])
    assert not torch.equal(result.k_post[:, 2], window.k_post[:, 0])
    assert torch.equal(result.k_post[:, [0, 1, 4, 5]], window.k_post[:, [0, 1, 4, 5]])
    assert result.provenance[2:4] == ((0,), (0,))
    assert replace(window, [], []).k_post is window.k_post
    with pytest.raises(ValueError, match="donors must be historical"):
        replace(window, [2], [4])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_merge_pre_keys_values_and_coordinates_share_weights(dtype):
    window = make_window(dtype, positions=[0, 1, 8, 9, 12, 13], sink_frames=1)
    before = (window.k_pre.clone(), window.k_post.clone(), window.v.clone())
    weighted = merge(window, [(2, 3)], weights=[[1., 3.]])
    coordinate = torch.tensor([[2., 0., .75]])
    pre = (window.k_pre[:, 2:3].float() * .25 + window.k_pre[:, 3:4].float() * .75).to(dtype)
    value = (window.v[:, 2:3].float() * .25 + window.v[:, 3:4].float() * .75).to(dtype)
    assert weighted.token_count == 5 and weighted.provenance[2] == (2, 3)
    assert torch.equal(weighted.coordinates[2:3], coordinate)
    torch.testing.assert_close(weighted.k_post[:, 2:3], rotate(pre, coordinate), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(weighted.v[:, 2:3], value, rtol=2e-6, atol=2e-7)
    # Averaging already rotated keys would change the representative geometry.
    wrong_post_mean = (window.k_post[:, 2:3].float() * .25 + window.k_post[:, 3:4].float() * .75).to(dtype)
    assert not torch.allclose(weighted.k_post[:, 2:3].float(), wrong_post_mean.float(), atol=.001, rtol=.001)
    assert weighted.log_count[2].item() == pytest.approx(math.log(2))
    assert weighted.details["representative_counts"] == [1, 1, 2, 1, 1]
    assert weighted.details["approximation"] == "approximate"
    assert torch.equal(weighted.k_post[:, [0, 1, 3, 4]], window.k_post[:, [0, 1, 4, 5]])
    averaged = merge(window, [(2, 3)])
    assert torch.equal(averaged.coordinates[2], torch.tensor([2., 0., .5]))
    no_bias = merge(window, [(2, 3)], multiplicity=False, weights=[1., 3.])
    assert torch.equal(no_bias.k_post, weighted.k_post) and torch.equal(no_bias.v, weighted.v)
    assert torch.count_nonzero(no_bias.log_count) == 0
    for tensor, snapshot in zip((window.k_pre, window.k_post, window.v), before):
        assert torch.equal(tensor, snapshot)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_duplicate_merge_log_count_independent_sdpa(dtype):
    window = make_window(dtype)
    pre, v = window.k_pre.clone(), window.v.clone()
    pre[:, 1:3] = pre[:, :1]
    v[:, 1:3] = v[:, :1]
    window = dataclass_replace(window, k_pre=pre, k_post=pre.clone(), v=v, rotate_pre=lambda k, c: k)
    merged = merge(window, [(0, 1, 2)])
    original = independent_attention(window, identity(window))
    combined = independent_attention(window, merged)
    tolerance = .02 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(original, combined, atol=tolerance, rtol=tolerance)
    assert merged.details["approximation"] == "exact_duplicate_key_value"
    without_bias = independent_attention(window, merge(window, [(0, 1, 2)], multiplicity=False))
    assert not torch.allclose(original.float(), without_bias.float(), atol=1e-3, rtol=1e-3)


def test_nonidentical_merge_is_not_an_exact_attention_identity():
    window = make_window()
    k, v, q = torch.zeros_like(window.k_post), torch.zeros_like(window.v), torch.ones_like(window.q_post)
    k[:, 0] = -2; k[:, 1] = 2; v[:, 1] = 4
    window = dataclass_replace(window, k_pre=k, k_post=k.clone(), v=v, q_post=q, rotate_pre=lambda x, c: x)
    merged = merge(window, [(0, 1)])
    error = (independent_attention(window, merged) - independent_attention(window, identity(window))).abs().max()
    assert error > .1 and merged.details["approximation"] == "approximate"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["gaussian", "low_rank", "temporal_noise", "quantization_like", "zero"])
@pytest.mark.parametrize("role", ["k", "v", "kv"])
def test_perturbation_roles_actual_nmse_and_protection(dtype, kind, role):
    window = make_window(dtype, sink_frames=1, recent_frames=1)
    indices = [4, 5, 6, 7]
    before_k, before_v = window.k_post.clone(), window.v.clone()
    rng_before = torch.get_rng_state().clone()
    result = perturb(window, indices, kind=kind, role=role, target_nmse=.08, seed=123, rank=2)
    again = perturb(window, indices, kind=kind, role=role, target_nmse=.08, seed=123, rank=2)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(result.k_post, again.k_post) and torch.equal(result.v, again.v)
    for name, original in (("k", before_k), ("v", before_v)):
        output = result.k_post if name == "k" else result.v
        outside = [i for i in range(16) if i not in indices]
        assert torch.equal(output[:, outside], original[:, outside])
        if name not in role:
            assert torch.equal(output, original) and result.details[f"nmse_{name}"] == 0
            continue
        actual = float((output[:, indices].double() - original[:, indices].double()).square().sum()
                       / original[:, indices].double().square().sum())
        assert result.details[f"nmse_{name}"] == pytest.approx(actual, abs=1e-12)
        assert actual == pytest.approx(1. if kind == "zero" else .08, rel=.015 if dtype == torch.bfloat16 else 1e-5)
    assert torch.equal(window.k_post, before_k) and torch.equal(window.v, before_v)
    assert torch.equal(result.coordinates, identity(window).coordinates)
    if kind == "quantization_like":
        assert "not a production codec" in result.details["noise_model"]


@pytest.mark.parametrize("kind", ["gaussian", "low_rank", "temporal_noise", "quantization_like", "zero"])
def test_exact_null_and_empty_targets_do_not_need_pre_or_rng(kind):
    window = dataclass_replace(make_window(torch.bfloat16), k_pre=None, rotate_pre=None)
    for indices, nmse in (([1, 2], 0.), ([], .1)):
        state = torch.get_rng_state().clone()
        result = perturb(window, indices, kind=kind, role="kv", target_nmse=nmse, seed=7)
        assert result.k_post is window.k_post and result.v is window.v
        assert result.details["nmse_k"] == result.details["nmse_v"] == 0.
        assert torch.equal(state, torch.get_rng_state())


def test_temporal_noise_reuses_spatial_pattern_and_low_rank_direction():
    window = make_window()
    temporal = perturb(window, [0, 1, 4, 5, 8, 9], kind="temporal_noise", role="v", target_nmse=.2, seed=5)
    delta = temporal.v - window.v
    torch.testing.assert_close(delta[:, 0], delta[:, 4], atol=3e-7, rtol=2e-6)
    torch.testing.assert_close(delta[:, 0], delta[:, 8], atol=3e-7, rtol=2e-6)
    low_rank = perturb(window, list(range(12)), kind="low_rank", role="v", target_nmse=.2, seed=5, rank=1)
    singular = torch.linalg.svdvals((low_rank.v[:, :12] - window.v[:, :12]).reshape(2, 12, -1))
    assert torch.all(singular[:, 1:] < 2e-6)
    other_seed = perturb(window, [0, 1, 4, 5, 8, 9], kind="temporal_noise", role="v", target_nmse=.2, seed=6)
    assert not torch.equal(other_seed.v, temporal.v)


@pytest.mark.parametrize("kind", ["gaussian", "low_rank", "temporal_noise", "quantization_like", "zero"])
def test_zero_energy_is_reported_without_fabricated_relative_error(kind):
    window = make_window()
    window = dataclass_replace(window, k_post=torch.zeros_like(window.k_post))
    result = perturb(window, [0, 1], kind=kind, role="kv", target_nmse=.1, seed=1)
    assert torch.count_nonzero(result.k_post) == 0
    assert result.details["nmse_k"] is None
    assert "zero_energy" in result.details["status_k"] and result.details["reason_k"]
    assert result.details["nmse_v"] == pytest.approx(1. if kind == "zero" else .1, rel=1e-5)


def test_protected_targets_and_invalid_indices_are_rejected():
    window = make_window(sink_frames=1, recent_frames=1)
    for protected in (0, 8, 12):
        for operation in (lambda: drop(window, [protected]), lambda: replace(window, [protected], [4]),
                          lambda: merge(window, [(4, protected)]),
                          lambda: perturb(window, [protected], target_nmse=.1, seed=1)):
            with pytest.raises(ValueError, match="unprotected historical"):
                operation()
    for indices in ([4, 4], [-1], [16], [4.0], [True], torch.tensor([4.])):
        with pytest.raises(ValueError):
            drop(window, indices)
    with pytest.raises(ValueError, match="equal lengths"):
        replace(window, [4, 5], [0])
    with pytest.raises(ValueError, match="disjoint"):
        merge(window, [(4, 5), (5, 6)])
    with pytest.raises(ValueError, match="empty"):
        merge(window, [()])
    for weights in ([[0, 1]], [[float("nan"), 1]], [[1]], [[-1, 2]]):
        with pytest.raises(ValueError, match="weights"):
            merge(window, [(4, 5)], weights=weights)
    all_history = make_window(positions=list(range(12)))
    with pytest.raises(ValueError, match="every key"):
        drop(all_history, list(range(12)))


def test_pre_availability_and_rotation_return_validation():
    window = make_window()
    unavailable = dataclass_replace(window, k_pre=None)
    for operation in (lambda: merge(unavailable, [(0, 1)]), lambda: replace(unavailable, [0], [1])):
        with pytest.raises(ValueError, match="pre-RoPE"):
            operation()
    # pre_valid_mask=None means the complete provided pre-K tensor is valid.
    valid = torch.ones(16, dtype=torch.bool); valid[1] = False
    unavailable = dataclass_replace(window, pre_valid_mask=valid)
    with pytest.raises(ValueError, match="unavailable"):
        merge(unavailable, [(0, 1)])
    with pytest.raises(ValueError, match="unavailable"):
        replace(unavailable, [0], [1])
    assert merge(unavailable, [(2, 3)]).token_count == 15
    bad_rotation = dataclass_replace(window, rotate_pre=lambda k, c: k.double())
    with pytest.raises(ValueError, match="rotate_pre"):
        replace(bad_rotation, [0], [1])


def test_invalid_strength_rank_finite_data_and_empty_shapes():
    window = make_window()
    for strength in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="target_nmse"):
            perturb(window, [0], target_nmse=strength, seed=0)
    for options in ({"rank": 0}, {"role": "q"}, {"kind": "codec"}, {"seed": 1.5}):
        with pytest.raises(ValueError):
            perturb(window, [0], target_nmse=.1, **{"seed": 0, **options})
    nonfinite = window.k_post.clone(); nonfinite[:, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        identity(dataclass_replace(window, k_post=nonfinite))
    ctx = dataclass_replace(window.context, key_abs_positions=torch.empty(0, dtype=torch.int64))
    empty = dataclass_replace(window, context=ctx, k_post=window.k_post[:, :0], v=window.v[:, :0], k_pre=window.k_pre[:, :0])
    with pytest.raises(ValueError, match="nonempty"):
        identity(empty)
