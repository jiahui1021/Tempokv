"""Suite B only: runtime-shaped, fractional-position three-axis RoPE."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tempokv.spectral_kv.model_spec import ModelSpec, from_runtime
from tempokv.spectral_kv.rope3d import full_positions, rope_apply, rope_invert


def _spec(head_dim=14):
    c = head_dim // 2
    return ModelSpec(5, 3, head_dim, (3, 5), 2, (2 * (c - 2 * (c // 3)), 2 * (c // 3), 2 * (c // 3)))


def _table(spec, count=64, dtype=torch.float64, theta=None):
    axes = []
    for size in spec.rope_split:
        omega = 1.0 / torch.pow(
            spec.rope_theta if theta is None else theta,
            torch.arange(0, size, 2, dtype=dtype).div(size),
        )
        phase = torch.outer(torch.arange(count, dtype=dtype), omega)
        axes.append(torch.polar(torch.ones_like(phase), phase))
    return torch.cat(axes, dim=-1)


def _official_causal_rope():
    # Extract only the original function: importing the model would require
    # diffusers, CUDA attention packages and unrelated checkpoint dependencies.
    path = Path(__file__).resolve().parents[2] / "experiments/Self-Forcing/wan/modules/causal_model.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "causal_rope_apply")
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["causal_rope_apply"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("head_dim", [14, 128])
def test_integer_positions_exactly_match_official_source(dtype, head_dim):
    spec = _spec(head_dim)
    x = torch.randn(2, 30, spec.heads, head_dim, dtype=dtype, generator=torch.Generator().manual_seed(13))
    original = x.clone()
    freqs = _table(spec, theta=37.0)  # Runtime table must override spec.rope_theta.
    saved_freqs = freqs.clone()
    positions = full_positions(7, 2, (3, 5))
    result = rope_apply(x, *positions, spec, freqs=freqs)
    expected = _official_causal_rope()(x, torch.tensor([[2, 3, 5], [2, 3, 5]]), freqs, start_frame=7)
    assert torch.equal(result, expected)
    assert result.dtype == dtype and result.device == x.device
    assert torch.equal(x, original) and torch.equal(freqs, saved_freqs)


def test_float32_path_matches_exact_complex64_arithmetic():
    spec = _spec()
    x = torch.randn(1, 30, spec.heads, spec.head_dim, generator=torch.Generator().manual_seed(2))
    pos = full_positions(4, 2, spec.frame_hw)
    freqs = _table(spec, dtype=torch.float32, theta=31.0)
    t, h, w = freqs.split([v // 2 for v in spec.rope_split], dim=1)
    phase = torch.cat([axis.index_select(0, p.long()) for axis, p in zip((t, h, w), pos)], dim=-1)
    complex_x = torch.view_as_complex(x.reshape(1, 30, spec.heads, -1, 2))
    expected = torch.view_as_real(complex_x * phase[None, :, None]).flatten(-2)
    actual = rope_apply(x, *pos, spec, freqs=freqs, precision="float32")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_theta_path_agrees_with_runtime_table(precision):
    spec = _spec()
    dtype = torch.float64 if precision == "float64" else torch.float32
    x = torch.randn(1, 30, spec.heads, spec.head_dim, dtype=dtype)
    pos = full_positions(5, 2, spec.frame_hw)
    expected = rope_apply(x, *pos, spec, freqs=_table(spec, dtype=dtype), precision=precision)
    actual = rope_apply(x, *pos, spec, precision=precision)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype,atol", [(torch.float64, 2e-14), (torch.float32, 5e-7), (torch.bfloat16, 0.016)])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_fractional_inverse_and_no_mutation(dtype, atol, precision):
    spec = _spec()
    x = torch.randn(2, 4, spec.heads, spec.head_dim, dtype=dtype, generator=torch.Generator().manual_seed(8))
    saved = x.clone()
    pos = tuple(torch.tensor([0.5, 1.25, 2.75, 3.5], dtype=torch.float64) + offset for offset in (2, 0, 1))
    freqs = _table(spec, theta=11.0)
    rotated = rope_apply(x, *pos, spec, freqs=freqs, precision=precision)
    inverse = rope_invert(rotated, *pos, spec, freqs=freqs, precision=precision)
    effective_atol = max(atol, 8e-7) if precision == "float32" else atol
    torch.testing.assert_close(inverse, x, atol=effective_atol, rtol=effective_atol)
    assert torch.equal(x, saved)
    assert inverse.dtype == dtype and inverse.device == x.device


def test_fractional_continuity_through_wrapped_integer_phases():
    spec = _spec()
    # At position 41 the fast channel has crossed the angle branch cut many
    # times; interpolation must use relative phase, not absolute wrapped angles.
    p = torch.tensor([41 - 1e-9, 41.0, 41 + 1e-9], dtype=torch.float64)
    x = torch.randn(1, 1, spec.heads, spec.head_dim, dtype=torch.float64).expand(1, 3, -1, -1)
    result = rope_apply(x, p, p, p, spec, freqs=_table(spec, theta=11.0))
    torch.testing.assert_close(result[:, 0], result[:, 1], atol=5e-9, rtol=5e-9)
    torch.testing.assert_close(result[:, 2], result[:, 1], atol=5e-9, rtol=5e-9)


def test_fractional_runtime_phase_and_endpoint_extrapolation():
    spec = _spec()
    p = torch.tensor([-3.25, 2.5, 12.125, 42.0], dtype=torch.float64)
    x = torch.randn(1, 4, spec.heads, spec.head_dim, dtype=torch.float64)
    custom = ModelSpec(spec.layers, spec.heads, spec.head_dim, spec.frame_hw, spec.chunk_frames, spec.rope_split, 13.0)
    actual = rope_apply(x, p, p / 3, p / 2, spec, freqs=_table(spec, count=8, theta=13.0))
    expected = rope_apply(x, p, p / 3, p / 2, custom)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)


def test_full_positions_are_non_square_row_major():
    t, h, w = full_positions(7, 2, (2, 3))
    assert t.tolist() == [7] * 6 + [8] * 6
    assert h.tolist() == [0, 0, 0, 1, 1, 1] * 2
    assert w.tolist() == [0, 1, 2, 0, 1, 2] * 2
    assert all(p.dtype == torch.float64 for p in (t, h, w))


def test_spec_uses_runtime_dimensions_and_grid():
    attention = SimpleNamespace(num_heads=7, head_dim=14)
    model = SimpleNamespace(blocks=[SimpleNamespace(self_attn=attention)] * 9, num_frame_per_block=3)
    spec = ModelSpec.from_runtime(model, torch.tensor([[2, 7, 11], [2, 7, 11]]))
    assert spec == ModelSpec(9, 7, 14, (7, 11), 2, (6, 4, 4))
    assert from_runtime(model, frame_hw=(5, 9)).chunk_frames == 3
    assert from_runtime(model, [[1, 7, 11]], chunk_frames=3).chunk_frames == 3
    model = SimpleNamespace(num_layers=4, num_heads=5, dim=640)
    assert from_runtime(model, [[3, 13, 17]]).rope_split == (44, 42, 42)


@pytest.mark.parametrize("changes", [
    {"layers": 0}, {"heads": True}, {"head_dim": 15}, {"frame_hw": (0, 3)},
    {"chunk_frames": 1.5}, {"rope_split": (4, 5, 5)}, {"rope_split": (4, 4, 4)},
    {"rope_split": (0, 6, 8)}, {"rope_theta": float("nan")},
])
def test_invalid_model_specs_fail(changes):
    args = dict(layers=5, heads=3, head_dim=14, frame_hw=(3, 5), chunk_frames=2, rope_split=(6, 4, 4))
    args.update(changes)
    with pytest.raises(ValueError):
        ModelSpec(**args)


def test_runtime_grid_validation():
    model = SimpleNamespace(num_layers=4, num_heads=5, dim=70)
    for grid in ([[1, 3, 5], [1, 5, 3]], [[1, 3.5, 5]], [], [1, 3, 5]):
        with pytest.raises(ValueError):
            from_runtime(model, grid)
    with pytest.raises(ValueError, match="disagrees"):
        from_runtime(model, [[1, 3, 5]], frame_hw=(3, 4))
    with pytest.raises(ValueError, match="required"):
        from_runtime(model)


@pytest.mark.parametrize("kind", ["real", "shape", "empty", "nan", "magnitude", "rank"])
def test_invalid_runtime_frequencies_fail(kind):
    spec = _spec()
    table = _table(spec)
    if kind == "real":
        table = table.real
    elif kind == "shape":
        table = table[:, :-1]
    elif kind == "empty":
        table = table[:0]
    elif kind == "nan":
        table[0, 0] = complex(float("nan"), 0)
    elif kind == "magnitude":
        table[1, 0] = 0.5
    else:
        table = table[None]
    x = torch.ones(1, 1, spec.heads, spec.head_dim)
    p = torch.zeros(1)
    with pytest.raises(ValueError, match="freqs"):
        rope_apply(x, p, p, p, spec, freqs=table)


def test_invalid_positions_precision_shapes_and_single_row_table():
    spec = _spec()
    x = torch.ones(1, 2, spec.heads, spec.head_dim)
    p = torch.zeros(2)
    for bad in (torch.zeros(2, 1), torch.zeros(3), torch.tensor([0.0, float("nan")]), torch.zeros(2, dtype=torch.bool)):
        with pytest.raises(ValueError, match="position"):
            rope_apply(x, bad, p, p, spec)
    with pytest.raises(ValueError, match="precision"):
        rope_apply(x, p, p, p, spec, precision="fp16")
    with pytest.raises(ValueError, match="match ModelSpec"):
        rope_apply(x[..., :-1], p, p, p, spec)
    with pytest.raises(ValueError, match="two frequency rows"):
        rope_apply(x, p + 0.5, p, p, spec, freqs=_table(spec, count=1))
    assert torch.equal(rope_apply(x, p, p, p, spec, freqs=_table(spec, count=1)), x)


@pytest.mark.parametrize("start,frames,hw", [(-1, 1, (2, 3)), (0, 0, (2, 3)), (0, 1, (2, 0)), (0.5, 1, (2, 3))])
def test_invalid_full_positions_fail(start, frames, hw):
    with pytest.raises(ValueError):
        full_positions(start, frames, hw)
