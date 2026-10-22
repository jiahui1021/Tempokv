"""Suite B attention correctness, multiplicity and bounded-memory paths."""

import math

import pytest
import torch

from tempokv.spectral_kv.attention import (
    AttentionGroup, KVView, spectral_attention_fast, spectral_attention_ref,
)


def _fixture(dtype=torch.float32):
    gen = torch.Generator().manual_seed(31)
    q = torch.randn(2, 7, 3, 8, dtype=dtype, generator=gen)
    groups = [
        AttentionGroup(torch.randn(2, s, 3, 8, dtype=dtype, generator=gen),
                       torch.randn(2, s, 3, 8, dtype=dtype, generator=gen), bias)
        for s, bias in [(5, 0.0), (9, math.log(4)), (0, 0.0), (3, math.log(16))]
    ]
    return q, KVView(groups)


def _backend(q, k, v):
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    qt, kt, vt = (t.transpose(1, 2).to(dtype) for t in (q, k, v))
    logits = qt @ kt.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return (logits.softmax(-1) @ vt).transpose(1, 2), logits.logsumexp(-1)


@pytest.mark.parametrize("qchunk,kchunk", [(1, 1), (3, 4), (32, 1024)])
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 7e-7), (torch.float64, 2e-14), (torch.bfloat16, 0.02)])
def test_tiled_matches_reference(dtype, atol, qchunk, kchunk):
    q, view = _fixture(dtype)
    actual = spectral_attention_fast(q, view, query_chunk_size=qchunk, key_chunk_size=kchunk)
    expected = spectral_attention_ref(q, view)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)
    assert actual.dtype == q.dtype and actual.device == q.device
    assert view.token_count == 17


def test_injected_group_backend_matches_reference_and_skips_empty():
    q, view = _fixture()
    calls = []
    def backend(q, k, v):
        calls.append(k.shape[1])
        return _backend(q, k, v)
    actual = spectral_attention_fast(q, view, group_attention=backend)
    torch.testing.assert_close(actual, spectral_attention_ref(q, view), atol=7e-7, rtol=7e-7)
    assert calls == [5, 9, 3]


def test_scalar_multiplicity_equals_repeated_identical_tokens():
    q, view = _fixture()
    first, second = view.groups[:2]
    compact = KVView([AttentionGroup(first.k, first.v, math.log(3)), AttentionGroup(second.k, second.v, math.log(2))])
    expanded = KVView([AttentionGroup(first.k.repeat_interleave(3, 1), first.v.repeat_interleave(3, 1)),
                       AttentionGroup(second.k.repeat_interleave(2, 1), second.v.repeat_interleave(2, 1))])
    expected = spectral_attention_ref(q, expanded)
    torch.testing.assert_close(spectral_attention_ref(q, compact), expected, atol=5e-7, rtol=5e-7)
    torch.testing.assert_close(spectral_attention_fast(q, compact, key_chunk_size=2), expected, atol=5e-7, rtol=5e-7)


def test_per_token_multiplicity_equals_repetition_and_uses_tiled_fallback():
    q, view = _fixture()
    group = view.groups[0]
    counts = torch.tensor([1, 3, 2, 4, 1])
    compact = KVView([AttentionGroup(group.k, group.v, counts.float().log())])
    expanded = KVView([AttentionGroup(group.k.repeat_interleave(counts, 1), group.v.repeat_interleave(counts, 1))])
    def unused_backend(q, k, v):
        raise AssertionError("per-token bias requires the tiled implementation")
    actual = spectral_attention_fast(q, compact, group_attention=unused_backend, key_chunk_size=2, query_chunk_size=3)
    expected = spectral_attention_ref(q, expanded)
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=5e-7)
    torch.testing.assert_close(spectral_attention_ref(q, compact), expected, atol=5e-7, rtol=5e-7)


@pytest.mark.parametrize("shape", [(5,), (2, 5), (2, 3, 5), (2, 3, 1, 5)])
def test_supported_count_broadcast_shapes(shape):
    q, view = _fixture()
    g = view.groups[0]
    bias = torch.linspace(-2, 2, math.prod(shape)).reshape(shape)
    local = KVView([AttentionGroup(g.k, g.v, bias)])
    torch.testing.assert_close(spectral_attention_fast(q, local, key_chunk_size=2), spectral_attention_ref(q, local), atol=7e-7, rtol=7e-7)


def test_zero_dimensional_tensor_scalar_uses_injected_backend():
    q, view = _fixture()
    g = view.groups[0]
    called = []
    def backend(q, k, v):
        called.append(True)
        return _backend(q, k, v)
    local = KVView([AttentionGroup(g.k, g.v, torch.tensor(2.0))])
    torch.testing.assert_close(spectral_attention_fast(q, local, group_attention=backend), spectral_attention_ref(q, local), atol=6e-7, rtol=6e-7)
    assert called == [True]


def test_extreme_equal_logits_remain_finite_and_weight_all_groups():
    q = torch.full((1, 2, 1, 4), 1e4)
    k = torch.full((1, 1, 1, 4), 1e4)
    groups = [AttentionGroup(k, torch.full_like(k, value)) for value in (1, 2, 6)]
    actual = spectral_attention_fast(q, KVView(groups), key_chunk_size=1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.full_like(q, 3.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(actual, spectral_attention_ref(q, KVView(groups)), atol=1e-6, rtol=0)


def test_inputs_are_not_mutated_and_empty_queries_work():
    q, view = _fixture()
    saved = q.clone(), [(g.k.clone(), g.v.clone()) for g in view.groups]
    spectral_attention_fast(q, view, key_chunk_size=2)
    assert torch.equal(q, saved[0])
    assert all(torch.equal(g.k, k) and torch.equal(g.v, v) for g, (k, v) in zip(view.groups, saved[1]))
    assert spectral_attention_fast(q[:, :0], view).shape == q[:, :0].shape
    assert spectral_attention_ref(q[:, :0], view).shape == q[:, :0].shape


@pytest.mark.parametrize("bias", [torch.tensor([True] * 5), torch.tensor([0., 0., 0., 0., -torch.inf]),
                                   torch.zeros(3), torch.zeros(2, 2, 5), float("nan"), True])
def test_invalid_multiplicity_masks_and_shapes_fail(bias):
    q, view = _fixture()
    g = view.groups[0]
    local = KVView([AttentionGroup(g.k, g.v, bias)])
    for fn in (spectral_attention_ref, spectral_attention_fast):
        with pytest.raises(ValueError, match="log_count"):
            fn(q, local)


@pytest.mark.parametrize("case", ["none", "all_empty", "bad_v", "heads", "dtype", "q_rank", "integer_k"])
def test_invalid_views_and_tensor_shapes_fail(case):
    q, view = _fixture()
    g = view.groups[0]
    if case == "none":
        local = KVView([])
    elif case == "all_empty":
        local = KVView([AttentionGroup(g.k[:, :0], g.v[:, :0])])
    elif case == "bad_v":
        local = KVView([AttentionGroup(g.k, g.v[:, :-1])])
    elif case == "heads":
        local = KVView([AttentionGroup(g.k[:, :, :2], g.v[:, :, :2])])
    elif case == "dtype":
        local = KVView([AttentionGroup(g.k.double(), g.v.double())])
    elif case == "integer_k":
        local = KVView([AttentionGroup(g.k.long(), g.v.long())])
    else:
        local, q = view, q[0]
    for fn in (spectral_attention_ref, spectral_attention_fast):
        with pytest.raises(ValueError):
            fn(q, local)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_tile_sizes_fail(bad):
    q, view = _fixture()
    with pytest.raises(ValueError, match="positive integer"):
        spectral_attention_fast(q, view, key_chunk_size=bad)


@pytest.mark.parametrize("case", ["tuple", "shape", "lse", "nan"])
def test_invalid_backend_results_fail(case):
    q, view = _fixture()
    def backend(q, k, v):
        out, lse = _backend(q, k, v)
        if case == "tuple":
            return out
        if case == "shape":
            return out[:, :-1], lse
        if case == "lse":
            return out, lse.transpose(1, 2)
        return out * float("nan"), lse
    with pytest.raises(ValueError, match="group_attention"):
        spectral_attention_fast(q, view, group_attention=backend)
