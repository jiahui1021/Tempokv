from __future__ import annotations

import pytest
import torch

import tempokv.fitting as fitting
from tempokv.config import CodecConfig
from tempokv.fitting import (
    fit_cross_kv,
    fit_predictors,
    fit_temporal,
    fit_temporal_layers,
)
from tempokv.parameters import CrossKVParameters, TemporalParameters


def _temporal_targets(
    reconstructed: torch.Tensor,
    coefficients: list[torch.Tensor],
    biases: list[torch.Tensor],
) -> torch.Tensor:
    """Apply one warm-up row per order and reuse the final row thereafter."""
    target = torch.zeros_like(reconstructed)
    target[..., 0, :, :] = reconstructed[..., 0, :, :]
    maximum_order = len(coefficients)
    for frame in range(1, reconstructed.shape[-3]):
        order = min(frame, maximum_order)
        prediction = biases[order - 1].expand_as(target[..., frame, :, :]).clone()
        for lag in range(1, order + 1):
            prediction += reconstructed[..., frame - lag, :, :] * coefficients[order - 1][lag - 1]
        target[..., frame, :, :] = prediction
    return target


def test_temporal_shared_formula_shape_warmup_and_precision() -> None:
    generator = torch.Generator().manual_seed(7)
    reconstructed = torch.randn(96, 5, 3, 2, generator=generator, dtype=torch.float64)
    coefficients = [
        torch.tensor([[0.7, -0.2]], dtype=torch.float64),
        torch.tensor([[0.4, 0.8], [-0.1, 0.15]], dtype=torch.float64),
        torch.tensor(
            [[0.25, 0.5], [0.1, -0.2], [-0.05, 0.1]], dtype=torch.float64
        ),
    ]
    biases = [
        torch.tensor([[0.1, -0.3]], dtype=torch.float64),
        torch.tensor([[-0.2, 0.05]], dtype=torch.float64),
        torch.tensor([[0.4, -0.1]], dtype=torch.float64),
    ]
    target = _temporal_targets(reconstructed, coefficients, biases)

    fitted = fit_temporal(
        reconstructed,
        target,
        history_length=3,
        shared=True,
        ridge=0.0,
        max_gain=2.0,
        stride=16,
    )

    assert isinstance(fitted, TemporalParameters)
    assert fitted.history_weights.shape == (3, 3, 2)
    assert fitted.history_bias.shape == (3, 2)
    assert fitted.stride == 16
    for order in range(1, 4):
        torch.testing.assert_close(
            fitted.history_weights[order - 1, :order],
            coefficients[order - 1].float(),
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            fitted.history_bias[order - 1],
            biases[order - 1][0].float(),
            atol=2e-5,
            rtol=2e-5,
        )
    for row in range(3):
        assert torch.count_nonzero(fitted.history_weights[row, row + 1 :]) == 0


def test_temporal_per_head_fits_independent_parameters() -> None:
    generator = torch.Generator().manual_seed(11)
    reconstructed = torch.randn(128, 4, 2, 2, generator=generator)
    # Each entry is [lag, head, channel].
    coefficients = [
        torch.tensor([[[0.2, 0.4], [0.7, -0.3]]]),
        torch.tensor(
            [
                [[0.3, 0.1], [0.6, -0.2]],
                [[-0.15, 0.25], [0.1, 0.4]],
            ]
        ),
    ]
    biases = [
        torch.tensor([[0.1, -0.2], [0.3, 0.05]]),
        torch.tensor([[-0.1, 0.2], [0.25, -0.3]]),
    ]
    target = _temporal_targets(reconstructed, coefficients, biases)

    fitted = fit_temporal(
        reconstructed,
        target,
        history_length=2,
        shared=False,
        max_gain=2.0,
    )

    assert fitted.per_head
    assert fitted.history_weights.shape == (2, 2, 2, 2)
    assert fitted.history_bias.shape == (2, 2, 2)
    for order in range(1, 3):
        torch.testing.assert_close(
            fitted.history_weights[order - 1, :order],
            coefficients[order - 1],
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            fitted.history_bias[order - 1], biases[order - 1], atol=2e-5, rtol=2e-5
        )


def test_temporal_layered_fits_each_layer_and_head() -> None:
    generator = torch.Generator().manual_seed(13)
    layers, observations, frames, heads, dim = 2, 96, 4, 2, 2
    reconstructed = torch.randn(
        layers, observations, frames, heads, dim, generator=generator
    )
    expected_weights = torch.tensor(
        [
            [
                [[[0.2, 0.4], [0.6, -0.1]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.3, 0.1], [0.5, -0.2]], [[-0.1, 0.2], [0.15, 0.3]]],
            ],
            [
                [[[-0.2, 0.1], [0.25, 0.5]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.4, -0.1], [0.2, 0.35]], [[0.2, 0.3], [-0.05, 0.1]]],
            ],
        ]
    )
    expected_bias = torch.tensor(
        [
            [[[0.1, -0.1], [0.2, 0.05]], [[-0.2, 0.1], [0.05, -0.3]]],
            [[[-0.1, 0.2], [0.3, -0.2]], [[0.15, -0.05], [-0.1, 0.25]]],
        ]
    )
    target = torch.empty_like(reconstructed)
    for layer in range(layers):
        coefficients = [
            expected_weights[layer, 0, :1],
            expected_weights[layer, 1, :2],
        ]
        biases = [expected_bias[layer, 0], expected_bias[layer, 1]]
        target[layer] = _temporal_targets(reconstructed[layer], coefficients, biases)

    fitted = fit_temporal_layers(
        reconstructed,
        target,
        history_length=2,
        max_gain=2.0,
        stride=8,
    )

    assert fitted.layered
    assert fitted.history_weights.shape == (layers, 2, 2, heads, dim)
    assert fitted.history_bias.shape == (layers, 2, heads, dim)
    assert fitted.stride == 8
    torch.testing.assert_close(fitted.history_weights, expected_weights, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(fitted.history_bias, expected_bias, atol=2e-5, rtol=2e-5)


def test_temporal_ridge_shrinks_to_persistence_and_caps_gain() -> None:
    x = torch.linspace(-2.0, 2.0, 101).reshape(101, 1, 1, 1)
    reconstructed = torch.cat([x, torch.zeros_like(x)], dim=1)
    target = reconstructed.clone()
    target[:, 1] = 3.0 * reconstructed[:, 0] + 0.4

    shrunk = fit_temporal(
        reconstructed,
        target,
        history_length=1,
        ridge=1.0,
        max_gain=10.0,
    )
    # For AR(1), relative ridge=1 gives the midpoint between OLS gain 3 and
    # the persistence prior gain 1.  Bias is recomputed after shrinkage.
    torch.testing.assert_close(shrunk.history_weights[0, 0], torch.tensor([2.0]))
    torch.testing.assert_close(shrunk.history_bias[0], torch.tensor([0.4]))

    capped = fit_temporal(
        reconstructed,
        target,
        history_length=1,
        ridge=0.0,
        max_gain=0.75,
    )
    torch.testing.assert_close(capped.history_weights[0, 0], torch.tensor([0.75]))
    assert float(capped.history_weights.abs().sum(dim=1).max()) <= 0.75


def test_cross_kv_formula_shapes_and_precision() -> None:
    generator = torch.Generator().manual_seed(19)
    layers, observations, heads, dim = 3, 160, 2, 3
    keys = torch.randn(layers, observations, heads, dim, generator=generator)
    expected_weight = torch.randn(layers, heads, dim, dim, generator=generator) * 0.4
    expected_bias = torch.randn(layers, heads, dim, generator=generator) * 0.2
    values = torch.einsum("lnhi,lhij->lnhj", keys, expected_weight)
    values += expected_bias[:, None, :, :]

    fitted = fit_cross_kv(keys, values, ridge=0.0, stride=32)

    assert isinstance(fitted, CrossKVParameters)
    assert fitted.weight.shape == (layers, heads, dim, dim)
    assert fitted.bias.shape == (layers, heads, dim)
    assert fitted.stride == 32
    torch.testing.assert_close(fitted.weight, expected_weight, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(fitted.bias, expected_bias, atol=2e-5, rtol=2e-5)
    prediction = torch.einsum("lnhi,lhij->lnhj", keys, fitted.weight)
    prediction += fitted.bias[:, None, :, :]
    torch.testing.assert_close(prediction, values, atol=3e-5, rtol=3e-5)


def test_cross_kv_ridge_uses_centered_bias_regression() -> None:
    keys = torch.arange(6, dtype=torch.float64).reshape(1, 6, 1, 1)
    values = 4.0 * keys + 3.0
    fitted = fit_cross_kv(keys, values, ridge=1.0)

    # In one dimension, penalty equals the centered Gram matrix, halving the
    # slope.  The unpenalized intercept still passes through both means.
    torch.testing.assert_close(fitted.weight, torch.tensor([[[[2.0]]]]))
    torch.testing.assert_close(fitted.bias, torch.tensor([[[8.0]]]))


def test_closed_loop_fitting_uses_reconstructed_keys(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(23)
    keys = torch.randn(2, 48, 4, 2, 4, generator=generator) * 2.5
    values = torch.sin(keys) + 0.25 * keys.square()
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=2,
        group_size=2,
        stride=7,
        residual_mode="asymmetric",
        scale_dtype="float32",
        use_triton=False,
    )
    temporal_sources: list[torch.Tensor] = []
    cross_sources: list[torch.Tensor] = []
    original_temporal = fitting.fit_temporal_layers
    original_cross = fitting.fit_cross_kv

    def capture_temporal(source, target, **kwargs):
        temporal_sources.append(source.detach().clone())
        return original_temporal(source, target, **kwargs)

    def capture_cross(source, target, **kwargs):
        cross_sources.append(source.detach().clone())
        return original_cross(source, target, **kwargs)

    monkeypatch.setattr(fitting, "fit_temporal_layers", capture_temporal)
    monkeypatch.setattr(fitting, "fit_cross_kv", capture_cross)

    temporal, cross = fit_predictors(
        keys,
        values,
        config=config,
        history_length=2,
        iterations=2,
        temporal_ridge=1e-3,
        cross_ridge=1e-3,
        max_gain=1.0,
    )

    assert temporal.history_weights.shape == (2, 2, 2, 2, 4)
    assert temporal.history_bias.shape == (2, 2, 2, 4)
    assert cross.weight.shape == (2, 2, 4, 4)
    assert cross.bias.shape == (2, 2, 4)
    assert temporal.stride == cross.stride == config.stride
    assert len(temporal_sources) == 2
    assert len(cross_sources) == 1
    assert all(not torch.equal(source, keys) for source in temporal_sources)
    assert not torch.equal(cross_sources[0], keys)
    assert all(source.dtype == torch.bfloat16 for source in temporal_sources)
    assert cross_sources[0].dtype == torch.bfloat16
    # The final Cross fit receives a fresh rollout from the selected temporal
    # parameters, rather than a stale reconstruction from the previous round.
    expected_final = fitting._reconstruct_temporal_trace(keys, temporal, config)
    torch.testing.assert_close(cross_sources[0], expected_final)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: fit_temporal(torch.ones(2, 2), torch.ones(2, 2), history_length=1), ValueError),
        (
            lambda: fit_temporal(
                torch.ones(2, 3, 1, 1), torch.ones(2, 4, 1, 1), history_length=1
            ),
            ValueError,
        ),
        (
            lambda: fit_temporal(
                torch.ones(2, 2, 1, 1), torch.ones(2, 2, 1, 1), history_length=2
            ),
            ValueError,
        ),
        (
            lambda: fit_temporal(
                torch.ones(2, 2, 1, 1), torch.ones(2, 2, 1, 1), history_length=1, ridge=-1
            ),
            ValueError,
        ),
        (
            lambda: fit_temporal(
                torch.ones(2, 2, 1, 1),
                torch.ones(2, 2, 1, 1),
                history_length=1,
                max_gain=0,
            ),
            ValueError,
        ),
        (
            lambda: fit_cross_kv(
                torch.ones(2, 4, 1, 2), torch.full((2, 4, 1, 2), float("nan"))
            ),
            ValueError,
        ),
        (
            lambda: fit_cross_kv(
                torch.ones(2, 4, 1, 2, dtype=torch.int64),
                torch.ones(2, 4, 1, 2, dtype=torch.int64),
            ),
            TypeError,
        ),
        (lambda: fit_cross_kv(torch.ones(4, 2, 3), torch.ones(4, 2, 3)), ValueError),
        (
            lambda: fit_cross_kv(
                torch.ones(2, 4, 1, 2), torch.ones(2, 4, 1, 2), stride=0
            ),
            ValueError,
        ),
    ],
)
def test_invalid_inputs(call, error) -> None:
    with pytest.raises(error):
        call()
