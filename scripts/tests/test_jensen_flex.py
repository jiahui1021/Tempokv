"""Fused score terms, production routing, and optional CUDA compilation.

Run from repository root with pytest scripts/tests/test_jensen_flex.py.
"""

import ast
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest
import torch

from tempokv.runtime.jensen import query_channels
from tempokv.runtime.jensen_flex import (
    corrected_attention, jensen_backend, make_score_mod, score_terms,
)


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_terms_match_legacy_channel_rounding(group_size, dtype):
    torch.manual_seed(19)
    q = torch.randn(2, 5, 3, 128, dtype=dtype)
    delta = torch.rand(2, 9, 3, 128 // group_size)
    steps = delta.to(dtype).pow(2)  # production cast BEFORE square
    steps[:, -2:] = 0  # unquantized current chunk
    qt, kt = score_terms(q, steps, group_size)
    assert qt.is_contiguous() and kt.is_contiguous()
    assert qt.shape == (128 // group_size, 2, 3, 5)
    assert kt.shape == (128 // group_size, 2, 3, 9)
    assert qt.dtype == kt.dtype == torch.float32
    b, h, qi, ki = torch.meshgrid(
        torch.arange(2), torch.arange(3), torch.arange(5), torch.arange(9),
        indexing="ij",
    )
    modifier = make_score_mod(qt, kt, 128 ** -0.5)
    actual = modifier(torch.zeros(2, 3, 5, 9), b, h, qi, ki)
    legacy_q = query_channels(q, group_size).to(dtype).float().permute(0, 2, 1, 3)
    legacy_k = steps.float().permute(0, 2, 1, 3)
    expected = torch.einsum("bhqg,bhkg->bhqk", legacy_q, legacy_k) * 128 ** -0.5
    torch.testing.assert_close(actual, expected)
    assert not actual[..., -2:].any()


def test_backend_default_and_invalid(monkeypatch):
    monkeypatch.delenv("TEMPOKV_JENSEN_BACKEND", raising=False)
    assert jensen_backend() == "augmented"
    monkeypatch.setenv("TEMPOKV_JENSEN_BACKEND", "flex")
    assert jensen_backend() == "flex"
    monkeypatch.setenv("TEMPOKV_JENSEN_BACKEND", "triton")
    assert jensen_backend() == "triton"
    monkeypatch.setenv("TEMPOKV_JENSEN_BACKEND", "invalid")
    with pytest.raises(ValueError, match="BACKEND"):
        jensen_backend()


def test_score_modifier_captures_as_one_graph_with_fresh_terms():
    """CPU graph capture only; CUDA tests verify actual fused kernel lowering."""
    for groups in (2, 4):
        for multiplier in (1.0, 2.0):
            qt = torch.ones(groups, 1, 1, 3) * multiplier
            kt = torch.ones(groups, 1, 1, 5)
            modifier = torch.compile(
                make_score_mod(qt, kt, 0.5), backend="eager", fullgraph=True, dynamic=True
            )
            qi, ki = torch.meshgrid(torch.arange(3), torch.arange(5), indexing="ij")
            result = modifier(torch.zeros(3, 5), 0, 0, qi, ki)
            torch.testing.assert_close(result, torch.full((3, 5), groups * multiplier * 0.5))


def test_input_guards_do_not_silently_use_dense_cpu_attention():
    q = torch.zeros(1, 2, 2, 128, dtype=torch.bfloat16)
    k = torch.zeros(1, 4, 2, 128, dtype=torch.bfloat16)
    steps = torch.zeros(1, 4, 2, 2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="group_size"):
        corrected_attention(q, k, k, steps, group_size=0)
    with pytest.raises(ValueError, match="token-for-token"):
        corrected_attention(q, k, k, steps[:, :2], group_size=64)
    with pytest.raises(RuntimeError, match="CUDA"):
        corrected_attention(q, k, k, steps, group_size=64)


@pytest.mark.parametrize("direction", ["k_to_v", "v_to_k"])
@pytest.mark.parametrize("backend", ["augmented", "flex", "triton"])
def test_production_reads_and_sink_window_alignment(monkeypatch, direction, backend):
    """Execute actual cache-read/attention source without importing model weights."""
    monkeypatch.setenv("TEMPOKV_JENSEN_BACKEND", backend)
    path = Path("experiments/Self-Forcing/wan/modules/causal_model.py")
    source = path.read_text()
    tree = ast.parse(source)
    helper_names = {"_attention_window_ranges", "_take_window", "_jensen_padded_width"}
    helpers = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in helper_names],
        type_ignores=[],
    )
    namespace = {"torch": torch, "jensen_backend": jensen_backend}
    exec(compile(helpers, str(path), "exec"), namespace)
    calls = []
    raw_steps = torch.rand(1, 6, 2, 2).to(torch.bfloat16)
    raw_steps[:, -2:] = 0

    class Cache:
        head_dim = 128
        tempokv_group_size = 64
        tempokv_jensen_correction = True
        tempokv_jensen_cache_mean = False
        tempokv_predictor_direction = direction

        def __init__(self, role):
            self.role = role
            self.tensor = torch.randn(1, 6, 2, 128, dtype=torch.bfloat16)

        def read(self, start, end, *, extra_channels=0, source_tensor=None):
            calls.append((self.role, extra_channels))
            if source_tensor is not None:
                assert source_tensor.shape[-1] == 128
            return torch.cat((self.tensor, self.tensor.new_zeros(1, 6, 2, extra_channels)), -1)

        def read_step_sizes(self, start, end, *, out, squared, group_size):
            assert squared and group_size == 64
            out.copy_(raw_steps)
            out.pow_(2)

    cache = {"k": Cache("k"), "v": Cache("v")}

    def fused(q, k, v, steps, *, group_size):
        assert backend in {"flex", "triton"}
        assert k.shape == v.shape == (1, 3, 2, 128)
        torch.testing.assert_close(steps, raw_steps[:, [0, 4, 5]].pow(2))
        torch.testing.assert_close(k, cache["k"].tensor[:, [0, 4, 5]])
        return q

    def flex(*args, **kwargs):
        assert backend == "flex"
        return fused(*args, **kwargs)

    def triton(*args, **kwargs):
        assert backend == "triton"
        return fused(*args, **kwargs)

    def augmented(q, k, v, **kwargs):
        assert backend == "augmented"
        assert k.shape == v.shape == (1, 3, 2, 136)
        torch.testing.assert_close(k[..., 128:130], raw_steps[:, [0, 4, 5]].pow(2))
        return q

    namespace.update(
        kv_cache=cache, self=SimpleNamespace(local_attn_size=3, sink_size=1),
        local_end_index=6, frame_seqlen=1, num_new_tokens=2,
        grid_sizes=None, freqs=None,
        roped_query=torch.randn(1, 2, 2, 128, dtype=torch.bfloat16),
        causal_rope_apply_long_input=lambda *args, **kwargs: None,
        jensen_flex_attention=flex, jensen_triton_attention=triton,
        _jensen_corrected_attention=augmented,
    )
    start = source.index('            direction = getattr(\n')
    end = source.index('            _set_cache_index(kv_cache, "global_end_index", current_end)', start)
    exec(compile(textwrap.dedent(source[start:end]), str(path), "exec"), namespace)
    assert calls == [("k" if direction == "k_to_v" else "v", 8 if backend == "augmented" else 0),
                     ("v" if direction == "k_to_v" else "k", 8 if backend == "augmented" else 0)]
    assert namespace["x"].shape[-1] == 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Inductor")
@pytest.mark.parametrize("group_size", [32, 64])
def test_cuda_compiled_attention_matches_corrected_reference(group_size):
    torch.manual_seed(51)
    q = torch.randn(2, 65, 2, 128, dtype=torch.bfloat16, device="cuda")
    # Changed lengths AND changed scale contents check dynamic-cache reuse.
    for length in (257, 321, 257):
        k = torch.randn(2, length, 2, 128, dtype=q.dtype, device=q.device)
        v = torch.randn_like(k)
        steps = (torch.rand(2, length, 2, 128 // group_size, device=q.device) * 2).to(q.dtype).pow(2)
        steps[:, -17:] = 0
        qt, kt = score_terms(q, steps, group_size)
        scores = q.float().permute(0, 2, 1, 3) @ k.float().permute(0, 2, 3, 1)
        scores = scores * 128 ** -0.5
        scores += torch.einsum("gbhq,gbhk->bhqk", qt, kt) * 128 ** -0.5
        reference = (scores.softmax(-1) @ v.float().permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        out = corrected_attention(q, k, v, steps, group_size=group_size)
        assert out.shape == q.shape and out.is_contiguous()
        torch.testing.assert_close(out.float(), reference, rtol=0.02, atol=0.02)
