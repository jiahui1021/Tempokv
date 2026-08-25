from __future__ import annotations

import pytest
import torch

from tempokv.config import CodecConfig
from tempokv.fitting import fit_cross_kv, fit_predictors, fit_temporal_layers


def _trace() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    keys = 4.0 * torch.randn(2, 3, 5, 2, 4, generator=generator)
    values = 0.7 * keys + torch.randn(keys.shape, generator=generator)
    return keys, values


def _native_config(**overrides: object) -> CodecConfig:
    values: dict[str, object] = {
        "k_bits": 2,
        "v_bits": 2,
        "anchor_bits": 2,
        "group_size": 4,
        "stride": 2,
        "k_history_length": 2,
        "v_history_length": 1,
        "v_predictor_mode": "cross_kv",
        "fit_mode": "native",
        "k_predictor_mode": "layer_head",
        "cross_matmul_precision": "tf32",
        "residual_mode": "asymmetric",
        "scale_dtype": "float32",
        "reconstruction_dtype": "bfloat16",
        "use_triton": False,
    }
    values.update(overrides)
    return CodecConfig(**values)


def _fit(
    config: CodecConfig, fit_history_source: str | None = None
) -> tuple[object, object]:
    keys, values = _trace()
    kwargs = {} if fit_history_source is None else {
        "fit_history_source": fit_history_source
    }
    return fit_predictors(
        keys,
        values,
        config=config,
        iterations=2,
        temporal_ridge=0.1,
        cross_ridge=0.1,
        v_history_mode="diagonal",
        **kwargs,
    )


def _parameter_tensors(parameters: object) -> tuple[torch.Tensor, ...]:
    if hasattr(parameters, "weight"):
        tensors = [
            parameters.weight,
            parameters.history_weights,
            parameters.bias,
        ]
    else:
        tensors = [parameters.history_weights, parameters.history_bias]
    return tuple(tensors)


def test_reconstructed_is_the_byte_identical_default() -> None:
    config = _native_config()
    default_k, default_v = _fit(config)
    explicit_k, explicit_v = _fit(config, "reconstructed")

    for default, explicit in zip(
        _parameter_tensors(default_k) + _parameter_tensors(default_v),
        _parameter_tensors(explicit_k) + _parameter_tensors(explicit_v),
        strict=True,
    ):
        assert torch.equal(default, explicit)
    assert default_k.parameter_set_id == explicit_k.parameter_set_id


def test_clean_cross_kv_fit_uses_clean_k_and_v_history() -> None:
    keys, values = _trace()
    config = _native_config()

    actual_k, actual_v = _fit(config, "clean")
    expected_k = fit_temporal_layers(
        keys,
        keys,
        history_length=2,
        ridge=0.1,
        stride=2,
    )
    expected_v = fit_cross_kv(
        keys,
        values,
        reconstructed_values=values,
        history_length=1,
        history_mode="diagonal",
        ridge=0.1,
        stride=2,
    )

    assert torch.equal(actual_k.history_weights, expected_k.history_weights)
    assert torch.equal(actual_k.history_bias, expected_k.history_bias)
    assert torch.equal(actual_v.weight, expected_v.weight)
    assert torch.equal(actual_v.history_weights, expected_v.history_weights)
    assert torch.equal(actual_v.bias, expected_v.bias)


def test_clean_temporal_fit_is_independent_of_residual_bit_width() -> None:
    low_k, low_v = _fit(_native_config(v_predictor_mode="temporal"), "clean")
    high_k, high_v = _fit(
        _native_config(
            k_bits=8,
            v_bits=8,
            anchor_bits=8,
            v_predictor_mode="temporal",
        ),
        "clean",
    )

    for low, high in zip(
        _parameter_tensors(low_k) + _parameter_tensors(low_v),
        _parameter_tensors(high_k) + _parameter_tensors(high_v),
        strict=True,
    ):
        assert torch.equal(low, high)


def test_closed_loop_and_clean_fits_diverge_under_coarse_quantization() -> None:
    closed_k, closed_v = _fit(_native_config(), "reconstructed")
    clean_k, clean_v = _fit(_native_config(), "clean")

    assert not torch.equal(closed_k.history_weights, clean_k.history_weights)
    assert not torch.equal(closed_v.history_weights, clean_v.history_weights)


def test_legacy_cross_kv_accepts_clean_fitting_state() -> None:
    generator = torch.Generator().manual_seed(5)
    keys = torch.randn(2, 2, 4, 1, 64, generator=generator)
    values = 0.3 * keys + torch.randn(keys.shape, generator=generator)
    config = CodecConfig(
        k_bits=4,
        v_bits=4,
        anchor_bits=4,
        group_size=64,
        stride=2,
        k_history_length=2,
        v_history_length=1,
        k_predictor_mode="shared",
        v_predictor_mode="cross_kv",
        fit_mode="legacy",
        cross_matmul_precision="ieee",
        strict_triton=True,
        use_triton=False,
    )

    temporal, cross = fit_predictors(
        keys,
        values,
        config=config,
        iterations=2,
        temporal_ridge=1.0,
        cross_ridge=0.1,
        v_history_ridge=0.0,
        max_gain=1.0,
        v_history_mode="diagonal",
        fit_history_source="clean",
    )

    assert temporal.history_weights.shape == (2, 2, 64)
    assert cross.history_weights.shape == (2, 1, 1, 64)
    assert temporal.parameter_set_id == cross.parameter_set_id


def test_legacy_reconstructed_default_is_unchanged() -> None:
    generator = torch.Generator().manual_seed(13)
    keys = torch.randn(1, 2, 4, 1, 64, generator=generator)
    values = torch.randn(keys.shape, generator=generator)
    config = CodecConfig(
        k_bits=4,
        v_bits=4,
        anchor_bits=4,
        group_size=64,
        stride=2,
        k_history_length=2,
        v_history_length=1,
        k_predictor_mode="shared",
        fit_mode="legacy",
        cross_matmul_precision="ieee",
        use_triton=False,
    )
    controls = {
        "config": config,
        "iterations": 2,
        "temporal_ridge": 1.0,
        "cross_ridge": 0.1,
        "v_history_ridge": 0.0,
        "max_gain": 1.0,
        "v_history_mode": "diagonal",
    }

    default = fit_predictors(keys, values, **controls)
    explicit = fit_predictors(
        keys, values, fit_history_source="reconstructed", **controls
    )

    for default_tensor, explicit_tensor in zip(
        _parameter_tensors(default[0]) + _parameter_tensors(default[1]),
        _parameter_tensors(explicit[0]) + _parameter_tensors(explicit[1]),
        strict=True,
    ):
        assert torch.equal(default_tensor, explicit_tensor)


def test_legacy_temporal_v_accepts_clean_fitting_state() -> None:
    generator = torch.Generator().manual_seed(19)
    keys = torch.randn(1, 2, 4, 1, 64, generator=generator)
    values = torch.randn(keys.shape, generator=generator)
    config = CodecConfig(
        k_bits=4,
        v_bits=4,
        anchor_bits=4,
        group_size=64,
        stride=2,
        k_history_length=2,
        v_history_length=1,
        k_predictor_mode="shared",
        v_predictor_mode="temporal",
        fit_mode="legacy",
        cross_matmul_precision="ieee",
        use_triton=False,
    )

    temporal_k, temporal_v = fit_predictors(
        keys,
        values,
        config=config,
        iterations=2,
        temporal_ridge=1.0,
        cross_ridge=0.1,
        v_history_ridge=0.0,
        max_gain=1.0,
        fit_history_source="clean",
    )

    assert temporal_k.history_weights.shape == (2, 2, 64)
    assert temporal_v.history_weights.shape == (1, 1, 64)
    assert temporal_k.parameter_set_id == temporal_v.parameter_set_id


def test_invalid_fitting_history_source_fails_before_fitting() -> None:
    with pytest.raises(ValueError, match="fit_history_source"):
        _fit(_native_config(), "runtime")
