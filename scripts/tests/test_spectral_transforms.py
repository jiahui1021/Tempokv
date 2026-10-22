from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from tempokv.spectral.config import SpectralConfig, load_config
from tempokv.spectral.metrics import nmse, relative_error, score_metrics
from tempokv.spectral.spectrum import spatial_energy, temporal_energy
from tempokv.spectral.transforms import spatial_blocks, spatial_fft, spatial_layout, temporal_fft


@pytest.mark.parametrize("h,w", [(3, 5), (1, 7), (6, 4)])
def test_non_square_layout_parseval_and_edge_blocks(h, w):
    x = torch.randn(2, 2 * h * w, 3, 8, generator=torch.Generator().manual_seed(9))
    dense = spatial_layout(x, [[2, h, w], [2, h, w]])
    assert dense.shape == (2, 2, h, w, 3, 8)
    torch.testing.assert_close(dense.flatten(1, 3), x)
    torch.testing.assert_close(spatial_fft(dense).abs().square().sum(), dense.square().sum())
    blocks = list(spatial_blocks(h, w, (2, 3)))
    tokens = [t for _, _, _, indices in blocks for t in indices]
    assert sorted(tokens) == list(range(h * w))
    assert len(tokens) == len(set(tokens))
    assert any(len(indices) < 6 for _, _, _, indices in blocks) if h * w % 6 else True


def test_constant_and_zero_spatial_features():
    constant = torch.full((1, 2, 3, 5, 2, 4), 2.0)
    ratios, energy, zero = spatial_energy(constant, [0, 0.1, 0.25, 0.4])
    for value in ratios.values():
        torch.testing.assert_close(value, torch.ones_like(value), atol=1e-6, rtol=0)
    assert energy.min() > 0 and not zero.any()
    ratios, _, zero = spatial_energy(torch.zeros_like(constant), [0.25])
    assert zero.all() and ratios[0.25].isnan().all()


def test_temporal_parseval_fixed_positions_and_sinusoid():
    time = torch.arange(16, dtype=torch.float32)
    x = torch.cos(2 * math.pi * time / 4)[:, None, None, None].expand(-1, 3, 2, 4)
    torch.testing.assert_close(temporal_fft(x).abs().square().sum(), x.square().sum())
    freq, power, ratios, zero = temporal_energy(x, [0.1, 0.25])
    assert ratios[0].max() < 1e-10
    torch.testing.assert_close(ratios[1], torch.ones(2), atol=1e-6, rtol=0)
    assert abs(float(freq[power[:, 0].argmax()])) == 0.25
    assert not zero.any()
    with pytest.raises(ValueError, match="at least two"):
        temporal_fft(x[:1])


@pytest.mark.parametrize("grid", [[[1, 3, 4]], [[1, 3.5, 5]], [[0, 3, 5]], [[1, 3, 5], [1, 5, 3]]])
def test_invalid_grids_fail(grid):
    batch = len(grid)
    with pytest.raises(ValueError):
        spatial_layout(torch.zeros(batch, 15, 1, 4), grid)


def test_config_json_yaml_subset_and_validation(tmp_path):
    import json
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps({"sampling": {"layers": [0], "heads": [1]}}))
    config = load_config(path)
    assert config.sampling.layers == (0,) and config.sampling.heads == (1,)
    assert SpectralConfig.from_dict(config.to_dict()) == config
    for payload in ({"source": "tempokv"}, {"oracle": {"sink_frames": 1}},
                    {"sampling": {"only_phase": "clean"}}, {"sampling": {"heads": []}},
                    {"frequency": {"cutoffs": [float("nan")]}},
                    {"compression": {"retention_ratios": [1.1]}},
                    {"sampling": {"query_tokens_per_head": 0}},
                    {"frequency": {"spatial_block_size": [0, 2]}}):
        with pytest.raises(ValueError):
            SpectralConfig.from_dict(payload)
    with pytest.raises(TypeError):
        SpectralConfig.from_dict({"sampling": {"typo": 1}})


def test_metrics_nan_ties_and_zero_denominator():
    assert score_metrics([1, 2, 3, np.nan], [1, 2, 3, 4])["spearman"] == 1
    assert math.isnan(score_metrics([1, 1], [1, 2])["spearman"])
    assert math.isnan(score_metrics([1, 1], [1, 2])["topk_recall_sampled"])
    assert score_metrics([np.nan], [0])["finite_candidates"] == 0
    x = torch.zeros(1, 2)
    assert relative_error(x, x) == 0 and nmse(x, x) == 0
    assert math.isfinite(relative_error(x, torch.ones_like(x)))
