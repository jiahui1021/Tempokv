from __future__ import annotations

import pytest
import torch

from tempokv.config import CodecConfig
from tempokv.fitting import fit_legacy_cross_kv, fit_predictors, refine_attention
from tempokv.parameters import CrossKVParameters, TemporalParameters


def _legacy_config(**overrides: object) -> CodecConfig:
    values: dict[str, object] = {
        "anchor_bits": 4,
        "group_size": 64,
        "k_history_length": 2,
        "v_history_length": 1,
        "k_predictor_mode": "shared",
        "v_predictor_mode": "cross_kv",
        "cross_matmul_precision": "ieee",
        "fit_mode": "legacy",
    }
    values.update(overrides)
    return CodecConfig(**values)


@pytest.mark.parametrize("mode", ("layer_shared", "head_shared", "shared"))
def test_nondefault_modes_are_signature_separated_and_reject_wrong_families(
    mode: str,
) -> None:
    baseline = _legacy_config()
    grouped = _legacy_config(cross_kv_weight_mode=mode)

    assert baseline.cross_kv_weight_mode == "layer_head"
    assert '"cross_kv_weight_mode"' not in baseline.format_signature()
    assert f'"cross_kv_weight_mode":"{mode}"' in grouped.format_signature()
    with pytest.raises(ValueError, match="requires fit_mode=legacy"):
        CodecConfig(cross_kv_weight_mode=mode)
    with pytest.raises(ValueError, match="requires v_predictor_mode=cross_kv"):
        _legacy_config(
            cross_kv_weight_mode=mode,
            v_predictor_mode="temporal",
        )


@pytest.mark.parametrize("mode", ("layer_shared", "head_shared", "shared"))
def test_grouped_cross_kv_fits_and_broadcasts_the_requested_axes(
    mode: str,
) -> None:
    generator = torch.Generator().manual_seed(20260825)
    layers, heads, dim = 3, 2, 4
    keys = torch.randn(layers, 4, 7, heads, dim, generator=generator)
    if mode == "layer_shared":
        grouped_weight = torch.randn(heads, dim, dim, generator=generator)
        grouped_bias = torch.randn(heads, dim, generator=generator)
        true_weight = grouped_weight.unsqueeze(0).expand(
            layers, heads, dim, dim
        )
        true_bias = grouped_bias.unsqueeze(0).expand(layers, heads, dim)
    elif mode == "head_shared":
        grouped_weight = torch.randn(layers, dim, dim, generator=generator)
        grouped_bias = torch.randn(layers, dim, generator=generator)
        true_weight = grouped_weight.unsqueeze(1).expand(
            layers, heads, dim, dim
        )
        true_bias = grouped_bias.unsqueeze(1).expand(layers, heads, dim)
    else:
        grouped_weight = torch.randn(dim, dim, generator=generator)
        grouped_bias = torch.randn(dim, generator=generator)
        true_weight = grouped_weight.view(1, 1, dim, dim).expand(
            layers, heads, dim, dim
        )
        true_bias = grouped_bias.view(1, 1, dim).expand(layers, heads, dim)
    values = torch.einsum("lbfhi,lhij->lbfhj", keys, true_weight)
    values += true_bias[:, None, None]

    fitted = fit_legacy_cross_kv(
        keys,
        values,
        ridge=0.0,
        weight_mode=mode,
        layer_ids=(2, 5, 9),
    )

    torch.testing.assert_close(fitted.weight, true_weight, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(fitted.bias, true_bias, atol=2e-5, rtol=2e-5)
    assert fitted.layer_ids == (2, 5, 9)


@pytest.mark.parametrize("mode", ("layer_shared", "head_shared", "shared"))
def test_grouped_cross_kv_broadcast_copies_are_exact(mode: str) -> None:
    generator = torch.Generator().manual_seed(20260828)
    keys = torch.randn(3, 4, 7, 2, 4, generator=generator)
    values = torch.randn(keys.shape, generator=generator)

    fitted = fit_legacy_cross_kv(keys, values, ridge=0.1, weight_mode=mode)

    if mode == "layer_shared":
        for head in range(2):
            torch.testing.assert_close(
                fitted.weight[:, head],
                fitted.weight[0, head].expand_as(fitted.weight[:, head]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                fitted.bias[:, head],
                fitted.bias[0, head].expand_as(fitted.bias[:, head]),
                rtol=0,
                atol=0,
            )
    elif mode == "head_shared":
        for layer in range(3):
            torch.testing.assert_close(
                fitted.weight[layer],
                fitted.weight[layer, 0].expand_as(fitted.weight[layer]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                fitted.bias[layer],
                fitted.bias[layer, 0].expand_as(fitted.bias[layer]),
                rtol=0,
                atol=0,
            )
    else:
        torch.testing.assert_close(
            fitted.weight,
            fitted.weight[0, 0].expand_as(fitted.weight),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            fitted.bias,
            fitted.bias[0, 0].expand_as(fitted.bias),
            rtol=0,
            atol=0,
        )


def test_explicit_layer_head_mode_is_identical_to_the_default() -> None:
    generator = torch.Generator().manual_seed(20260826)
    keys = torch.randn(2, 3, 6, 2, 4, generator=generator)
    values = torch.randn(keys.shape, generator=generator)

    default = fit_legacy_cross_kv(keys, values, ridge=0.1)
    explicit = fit_legacy_cross_kv(
        keys, values, ridge=0.1, weight_mode="layer_head"
    )

    torch.testing.assert_close(default.weight, explicit.weight, rtol=0, atol=0)
    torch.testing.assert_close(default.bias, explicit.bias, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ("layer_shared", "head_shared", "shared"))
def test_value_refinement_rejects_every_grouped_cross_kv_mode(mode: str) -> None:
    key_parameters = TemporalParameters(
        torch.ones(2, 2, 64), torch.zeros(2, 64), stride=1560
    )
    value_parameters = CrossKVParameters(
        torch.zeros(1, 1, 64, 64),
        torch.zeros(1, 1, 64),
        stride=1560,
        history_weights=torch.zeros(1, 1, 1, 64),
    )
    trace = torch.zeros(1, 1, 3, 1, 64)

    with pytest.raises(ValueError, match=f"cross_kv_weight_mode={mode}"):
        refine_attention(
            trace,
            trace,
            trace,
            trace,
            key_parameters,
            value_parameters,
            _legacy_config(cross_kv_weight_mode=mode),
            refine_tensors=("value",),
        )


@pytest.mark.parametrize("mode", ("layer_shared", "head_shared", "shared"))
def test_legacy_pipeline_preserves_grouping_after_v_history_fit(mode: str) -> None:
    generator = torch.Generator().manual_seed(20260829)
    keys = torch.randn(2, 1, 5, 2, 64, generator=generator)
    values = torch.randn(keys.shape, generator=generator)

    _, cross = fit_predictors(
        keys,
        values,
        config=_legacy_config(cross_kv_weight_mode=mode),
        iterations=2,
        temporal_ridge=1.0,
        cross_ridge=0.1,
        v_history_ridge=0.0,
        max_gain=1.0,
        v_history_mode="diagonal",
    )

    if mode == "layer_shared":
        for head in range(2):
            torch.testing.assert_close(
                cross.weight[:, head],
                cross.weight[0, head].expand_as(cross.weight[:, head]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                cross.bias[:, head],
                cross.bias[0, head].expand_as(cross.bias[:, head]),
                rtol=0,
                atol=0,
            )
    elif mode == "head_shared":
        for layer in range(2):
            torch.testing.assert_close(
                cross.weight[layer],
                cross.weight[layer, 0].expand_as(cross.weight[layer]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                cross.bias[layer],
                cross.bias[layer, 0].expand_as(cross.bias[layer]),
                rtol=0,
                atol=0,
            )
    else:
        torch.testing.assert_close(
            cross.weight,
            cross.weight[0, 0].expand_as(cross.weight),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cross.bias,
            cross.bias[0, 0].expand_as(cross.bias),
            rtol=0,
            atol=0,
        )
    assert cross.history_weights.shape == (2, 1, 2, 64)
