"""Mathematical checks for suite B; no model weights or GPU are required."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


# Keep this component's tests independent of the suite B package initializer.
_path = Path(__file__).resolve().parents[2] / "src/tempokv/spectral_kv/projection.py"
_spec = importlib.util.spec_from_file_location("spectral_kv_projection_under_test", _path)
assert _spec is not None and _spec.loader is not None
_projection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_projection)
dct_matrix = _projection.dct_matrix
make_projection = _projection.make_projection
compress_grid = _projection.compress_grid
reconstruct_grid = _projection.reconstruct_grid
cell_centers = _projection.cell_centers


@pytest.mark.parametrize("n", [1, 2, 3, 7, 32, 97])
def test_dct_orthogonal_parseval_and_dc(n):
    d = dct_matrix(n)
    assert d.dtype == torch.float32 and d.device.type == "cpu"
    torch.testing.assert_close(d @ d.T, torch.eye(n), atol=8e-7, rtol=0)
    x = torch.randn(n, 4, generator=torch.Generator().manual_seed(31))
    torch.testing.assert_close((d @ x).square().sum(), x.square().sum(), atol=3e-5, rtol=2e-6)
    torch.testing.assert_close(d[0], torch.full((n,), 1 / math.sqrt(n)), atol=1e-7, rtol=0)


@pytest.mark.parametrize("n,c", [(1, 1), (5, 2), (8, 3), (15, 7)])
def test_dct_formula_and_scaled_row_orthogonality(n, c):
    p = make_projection("dct", n, c)
    expected = math.sqrt(c / n) * dct_matrix(c).T @ dct_matrix(n)[:c]
    torch.testing.assert_close(p, expected, atol=2e-7, rtol=1e-6)
    torch.testing.assert_close(p @ p.T, torch.eye(c) * (c / n), atol=3e-7, rtol=1e-6)


@pytest.mark.parametrize("kind,taper", [("dct", None), ("dct", "hann"), ("avg", None)])
def test_full_resolution_is_exact_identity(kind, taper):
    x = torch.randn(2, 5, 7, 2, 4, generator=torch.Generator().manual_seed(5))
    ph, pw = make_projection(kind, 5, 5, taper), make_projection(kind, 7, 7, taper)
    assert torch.equal(ph, torch.eye(5)) and torch.equal(pw, torch.eye(7))
    assert torch.equal(compress_grid(x, ph, pw), x)
    assert torch.equal(reconstruct_grid(x, (5, 7), kind, taper), x)


@pytest.mark.parametrize("kind,taper", [("dct", None), ("dct", "hann"), ("avg", None)])
def test_nondivisible_constant_preservation(kind, taper):
    x = torch.full((2, 7, 11, 3, 2), 2.75)
    y = compress_grid(x, make_projection(kind, 7, 3, taper), make_projection(kind, 11, 4, taper))
    assert y.shape == (2, 3, 4, 3, 2)
    torch.testing.assert_close(y, torch.full_like(y, 2.75), atol=1e-6, rtol=0)
    torch.testing.assert_close(reconstruct_grid(y, (7, 11), kind, taper), x, atol=1e-5, rtol=0)


def test_average_fractional_cell_overlap_and_global_mean():
    p = make_projection("avg", 5, 2)
    torch.testing.assert_close(p, torch.tensor([[0.4, 0.4, 0.2, 0., 0.], [0., 0., 0.2, 0.4, 0.4]]))
    x = torch.arange(5, dtype=torch.float32)
    torch.testing.assert_close(p @ x, torch.tensor([0.8, 3.2]))
    torch.testing.assert_close(p.sum(1), torch.ones(2))
    torch.testing.assert_close((p @ x).mean(), x.mean())
    torch.testing.assert_close(make_projection("avg", 6, 2), torch.tensor([
        [1 / 3, 1 / 3, 1 / 3, 0., 0., 0.], [0., 0., 0., 1 / 3, 1 / 3, 1 / 3],
    ]))


def test_hann_weights_dc_and_cutoff_convention():
    n, c = 9, 4
    p = make_projection("dct", n, c, "hann")
    coefficient_map = math.sqrt(n / c) * dct_matrix(c) @ p @ dct_matrix(n).T
    expected = 0.5 * (1 + torch.cos(math.pi * torch.arange(c) / c))
    torch.testing.assert_close(coefficient_map[:, :c], torch.diag(expected), atol=2e-7, rtol=1e-6)
    torch.testing.assert_close(coefficient_map[:, c:], torch.zeros(c, n - c), atol=2e-7, rtol=0)
    torch.testing.assert_close(make_projection("dct", 7, 1, "hann"), torch.full((1, 7), 1 / 7))


@pytest.mark.parametrize("taper", [None, "hann"])
def test_bandlimited_grid_reconstruction(taper):
    h, w, ch, cw = 9, 13, 4, 5
    coefficients = torch.randn(2, ch, cw, 2, 3, generator=torch.Generator().manual_seed(19))
    x = torch.einsum("ah,fabnd,bw->fhwnd", dct_matrix(h)[:ch], coefficients, dct_matrix(w)[:cw])
    y = compress_grid(x, make_projection("dct", h, ch, taper), make_projection("dct", w, cw, taper))
    torch.testing.assert_close(reconstruct_grid(y, (h, w), "dct", taper), x, atol=3e-6, rtol=1e-5)
    # Removing high frequencies preserves the spatial mean under either taper.
    torch.testing.assert_close(y.mean((1, 2)), x.mean((1, 2)), atol=2e-7, rtol=1e-5)


def test_dct_projection_removes_unretained_frequencies():
    x = dct_matrix(8)[5].reshape(1, 8, 1, 1, 1)
    y = compress_grid(x, make_projection("dct", 8, 3), torch.ones(1, 1))
    torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-7, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_grid_dtype_device_and_explicit_fp32_math(dtype):
    x = torch.randn(2, 7, 11, 2, 3, generator=torch.Generator().manual_seed(43)).to(dtype)
    ph, pw = make_projection("dct", 7, 3), make_projection("avg", 11, 4)
    expected = torch.einsum("ah,fhwnd,bw->fabnd", ph, x.float(), pw).to(dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = compress_grid(x, ph, pw)
    assert y.dtype == dtype and y.device == x.device
    # Einsum contraction order can differ, so float32 results use a tolerance.
    torch.testing.assert_close(y, expected, atol=2e-7 if dtype in (torch.float32, torch.float64) else 1e-3, rtol=1e-5)


def test_noncontiguous_input_and_gradients():
    x = torch.randn(2, 7, 5, 2, 3, requires_grad=True).transpose(1, 2)
    y = compress_grid(x, make_projection("avg", 5, 2), make_projection("dct", 7, 3))
    y.square().sum().backward()
    assert x._base.grad is not None and torch.isfinite(x._base.grad).all()


def test_public_cache_results_are_not_mutable_aliases():
    dct_matrix(7).zero_()
    make_projection("dct", 7, 3).zero_()
    make_projection("avg", 7, 3).zero_()
    torch.testing.assert_close(dct_matrix(7) @ dct_matrix(7).T, torch.eye(7), atol=3e-7, rtol=0)
    for kind in ("dct", "avg"):
        torch.testing.assert_close(make_projection(kind, 7, 3).sum(1), torch.ones(3), atol=2e-7, rtol=0)


def test_fractional_cell_centers():
    torch.testing.assert_close(cell_centers(5, 2), torch.tensor([0.75, 3.25]))
    assert torch.equal(cell_centers(5, 5), torch.arange(5, dtype=torch.float32))
    torch.testing.assert_close(cell_centers(7, 1), torch.tensor([3.]))


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "3", None])
def test_invalid_axis_sizes(bad):
    with pytest.raises(ValueError):
        dct_matrix(bad)
    with pytest.raises(ValueError):
        make_projection("dct", 7, bad)
    with pytest.raises(ValueError):
        cell_centers(bad, 1)


@pytest.mark.parametrize("kind,n,c,taper", [
    ("fft", 5, 2, None), ("DCT", 5, 2, None), ("dct", 5, 6, None),
    ("dct", 5, 2, "hamming"), ("avg", 5, 2, "hann"), ("avg", 5, 5, "hann"),
])
def test_invalid_projection_methods(kind, n, c, taper):
    with pytest.raises(ValueError):
        make_projection(kind, n, c, taper)


@pytest.mark.parametrize("x,ph,pw", [
    (torch.zeros(3, 4), torch.eye(3), torch.eye(4)),
    (torch.zeros(1, 3, 4, 1, 1, dtype=torch.int32), torch.eye(3), torch.eye(4)),
    (torch.zeros(1, 3, 4, 1, 1, dtype=torch.complex64), torch.eye(3), torch.eye(4)),
    (torch.zeros(0, 3, 4, 1, 1), torch.eye(3), torch.eye(4)),
    (torch.zeros(1, 3, 4, 1, 1), torch.eye(4), torch.eye(4)),
    (torch.zeros(1, 3, 4, 1, 1), torch.zeros(4, 3), torch.eye(4)),
    (torch.zeros(1, 3, 4, 1, 1), torch.zeros(0, 3), torch.eye(4)),
    (torch.zeros(1, 3, 4, 1, 1), torch.ones(2, 3, dtype=torch.int32), torch.eye(4)),
])
def test_invalid_grid_or_projection_shape(x, ph, pw):
    with pytest.raises(ValueError):
        compress_grid(x, ph, pw)


@pytest.mark.parametrize("shape", [(2,), (2, 3, 4), (1, 2), (0, 4), "34"])
def test_invalid_reconstruction_shape(shape):
    with pytest.raises(ValueError):
        reconstruct_grid(torch.zeros(1, 2, 3, 1, 1), shape)


def test_optional_scipy_dct_equivalence():
    scipy_fft = pytest.importorskip("scipy.fft")
    x = torch.randn(13, generator=torch.Generator().manual_seed(21))
    expected = torch.as_tensor(scipy_fft.dct(x.numpy(), type=2, norm="ortho"))
    torch.testing.assert_close(dct_matrix(13) @ x, expected, atol=3e-7, rtol=2e-6)
