"""CPU synthetic checks of transform geometry; these are not model findings."""
from __future__ import annotations

import math

import pytest
import torch

from tempokv.research.transforms import (
    dct2, idct2, grid_shape, spatial_layout, spatial_blocks, spatial_fft,
    temporal_fft, radial_mask, spatial_energy, temporal_energy,
)


def synthetic(shape, dtype=torch.float32):
    generator = torch.Generator(device="cpu").manual_seed(173)
    return torch.randn(shape, generator=generator).to(dtype)


@pytest.mark.parametrize("hw", [(3, 5), (30, 52)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_spatial_fft_parseval_inverse_and_full_nonsquare_layout(hw, dtype):
    h, w = hw
    x = synthetic((2, 2, h, w, 2, 3), dtype)
    coeff = spatial_fft(x)
    restored = torch.fft.ifft2(coeff, dim=(2, 3), norm="ortho")
    torch.testing.assert_close(restored.real, x.float(), atol=8e-7, rtol=3e-6)
    assert restored.imag.abs().max() < 8e-7
    torch.testing.assert_close(coeff.abs().square().sum((2, 3)), x.float().square().sum((2, 3)), rtol=2e-6, atol=2e-5)
    flat = x.reshape(2, 2 * h * w, 2, 3)
    assert grid_shape(flat, [[2, h, w], [2, h, w]]) == (2, h, w)
    assert torch.equal(spatial_layout(flat, [[2, h, w], [2, h, w]]), x)


@pytest.mark.parametrize("hw", [(3, 5), (30, 52)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dct_parseval_inverse_and_constant_dc(hw, dtype):
    h, w = hw
    x = synthetic((2, h, w, 2, 3), dtype)
    coeff = dct2(x)
    torch.testing.assert_close(idct2(coeff), x.float(), rtol=4e-6, atol=2e-6)
    torch.testing.assert_close(coeff.square().sum((1, 2)), x.float().square().sum((1, 2)), rtol=2e-6, atol=2e-5)
    constant = torch.full((1, h, w, 1, 2), 2.5, dtype=dtype)
    constant_coeff = dct2(constant)
    torch.testing.assert_close(constant_coeff[:, 0, 0], torch.full((1, 1, 2), 2.5 * math.sqrt(h * w)), rtol=2e-6, atol=1e-6)
    constant_coeff = constant_coeff.clone(); constant_coeff[:, 0, 0] = 0
    assert constant_coeff.abs().max() < 2e-5


def test_fft_constant_dc_and_zero_energy_status():
    x = torch.full((1, 2, 30, 52, 2, 3), 2.)
    coeff = spatial_fft(x)
    torch.testing.assert_close(coeff[:, :, 0, 0].real, torch.full((1, 2, 2, 3), 2. * math.sqrt(30 * 52)), atol=1e-5, rtol=1e-6)
    coeff[:, :, 0, 0] = 0
    assert coeff.abs().max() < 1e-5
    ratios, total, zero = spatial_energy(x, (0., .1, .4))
    assert not zero.any()
    for value in ratios.values():
        torch.testing.assert_close(value, torch.ones_like(value))
    ratios, total, zero = spatial_energy(torch.zeros_like(x), (0., .1))
    assert zero.all() and torch.count_nonzero(total) == 0
    assert all(torch.isnan(value).all() for value in ratios.values())


@pytest.mark.parametrize("hw", [(5, 7), (30, 52)])
def test_radial_mask_uses_centered_signed_fftfreq_geometry(hw):
    h, w = hw
    cutoff = .173
    expected = torch.tensor([[(min(y, h-y) / h) ** 2 + (min(x, w-x) / w) ** 2 <= cutoff ** 2
                              for x in range(w)] for y in range(h)])
    mask = radial_mask(h, w, cutoff)
    assert torch.equal(mask, expected)
    assert mask[0, 0]
    if h == 30:
        assert mask[1, 0] and mask[-1, 0] and mask[0, 1] and mask[0, -1]
        assert not mask[h // 2, w // 2]
        centered_y = torch.arange(-h // 2, h // 2).double() / h
        centered_x = torch.arange(-w // 2, w // 2).double() / w
        centered_expected = centered_y[:, None].square() + centered_x[None, :].square() <= cutoff ** 2
        assert torch.equal(torch.fft.fftshift(mask), centered_expected)


def test_fft_low_energy_includes_negative_frequency_partner():
    h, w = 30, 52
    column = torch.arange(w)
    low = torch.cos(2 * math.pi * column * 2 / w)[None, :].expand(h, w)
    high = torch.cos(2 * math.pi * column * 20 / w)[None, :].expand(h, w)
    for field, expected in ((low, 1.), (high, 0.)):
        ratios, _, _ = spatial_energy(field[None, None, :, :, None, None], (.1,))
        assert ratios[.1].item() == pytest.approx(expected, abs=2e-6)
    # A crop from the upper left would retain only half of this real sinusoid.
    spectrum = spatial_fft(low[None, None, :, :, None, None]).abs().square()
    assert spectrum[0, 0, 0, 2].sum() == pytest.approx(float(spectrum[0, 0, 0, -2].sum()), rel=2e-6)


def test_radial_cutoff_includes_exact_boundary_on_nonsquare_grid():
    mask = radial_mask(30, 52, .4)
    assert mask[12, 0] and mask[-12, 0]
    quarter = radial_mask(30, 52, .25)
    assert quarter[0, 13] and quarter[0, -13]


@pytest.mark.parametrize("length", [8, 16, 32])
def test_temporal_fft_preserves_fixed_spatial_positions_and_energy(length):
    x = synthetic((length, 5, 2, 4))
    coeff = temporal_fft(x)
    torch.testing.assert_close(torch.fft.ifft(coeff, dim=0, norm="ortho").real, x, atol=6e-7, rtol=3e-6)
    torch.testing.assert_close(coeff.abs().square().sum(0), x.square().sum(0), rtol=2e-6, atol=2e-5)
    time = torch.arange(length).float()
    signal = torch.cos(2 * math.pi * time / length)[:, None, None, None]
    distinct_spatial_scale = torch.arange(1., 6.)[None, :, None, None]
    aligned = (signal * distinct_spatial_scale).expand(length, 5, 2, 4)
    frequency, power, ratios, zero = temporal_energy(aligned, (0., 1.01 / length, .49))
    assert not zero.any()
    torch.testing.assert_close(ratios[0], torch.zeros(2), atol=1e-10, rtol=0)
    torch.testing.assert_close(ratios[1], torch.ones(2), atol=2e-6, rtol=2e-6)
    assert frequency[1].item() == pytest.approx(1 / length)
    assert power[1].sum() == pytest.approx(float(power[-1].sum()), rel=2e-6)


def test_spatial_block_partition_keeps_every_nonsquare_grid_token():
    blocks = list(spatial_blocks(30, 52, (4, 4)))
    assert len(blocks) == 8 * 13
    flattened = [index for _, _, _, indices in blocks for index in indices]
    assert sorted(flattened) == list(range(30 * 52)) and len(set(flattened)) == len(flattened)
    for block_id, rows, cols, indices in blocks:
        assert indices == tuple(y * 52 + x for y in range(rows.start, rows.stop) for x in range(cols.start, cols.stop))
        assert len(indices) == (rows.stop - rows.start) * (cols.stop - cols.start)
    assert len(blocks[-1][3]) == 2 * 4


def test_reject_incompatible_layouts_and_transform_rank():
    x = torch.zeros(2, 12, 1, 2)
    for grids in ([[1, 3, 4]], [[1, 3, 4], [1, 2, 6]], [[1, 3, 3], [1, 3, 3]], [[1, 3.5, 4], [1, 3.5, 4]]):
        with pytest.raises(ValueError):
            spatial_layout(x, grids)
    for operation in (lambda: dct2(torch.zeros(2, 3)), lambda: idct2(torch.zeros(2, 3)),
                      lambda: spatial_fft(torch.zeros(2, 3)), lambda: temporal_fft(torch.zeros(1, 3, 1, 2))):
        with pytest.raises(ValueError):
            operation()
