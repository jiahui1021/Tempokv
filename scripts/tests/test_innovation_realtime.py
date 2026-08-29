from __future__ import annotations

import pytest
import torch

from tempokv import CodecConfig, CrossKVParameters, TempoKVCodec, TemporalParameters
from tempokv.codec import (
    decode_tensor,
    encode_innovation,
    reconstructed_key_innovation,
)
from tempokv.fitting import fit_predictors, refine_attention


def _config(**updates: object) -> CodecConfig:
    values: dict[str, object] = {
        "k_bits": 4,
        "v_bits": 4,
        "anchor_bits": 4,
        "group_size": 64,
        "stride": 4,
        "residual_mode": "asymmetric",
        "scale_dtype": "bfloat16",
        "reconstruction_dtype": "bfloat16",
        "use_triton": False,
        "strict_triton": False,
        "k_history_length": 2,
        "v_history_length": 1,
        "cross_matmul_precision": "ieee",
        "k_predictor_mode": "shared",
        "v_predictor_mode": "innovation",
        "fit_mode": "legacy",
        "cross_kv_weight_mode": "layer_shared",
    }
    values.update(updates)
    return CodecConfig(**values)


def _key_parameters() -> TemporalParameters:
    weights = torch.zeros(2, 2, 64)
    weights[0, 0] = 0.5
    weights[1, 0] = 0.25
    weights[1, 1] = 0.1
    bias = torch.zeros(2, 64)
    bias[0] = 0.125
    bias[1] = -0.25
    return TemporalParameters(weights, bias, stride=4)


def _innovation_parameters(*, projection_source: str = "innovation") -> CrossKVParameters:
    weight = torch.eye(64).unsqueeze(0) * 0.2
    bias = torch.linspace(-0.05, 0.05, 64).unsqueeze(0)
    history = torch.full((1, 1, 64), 0.4)
    return CrossKVParameters(
        weight,
        bias,
        stride=4,
        history_weights=history,
        projection_source=projection_source,
    )


def test_reconstructed_key_innovation_uses_warmup_rows_and_partial_tail() -> None:
    reconstructed = torch.empty(1, 1, 11, 64, dtype=torch.bfloat16)
    reconstructed[:, :, :4] = 1.0
    reconstructed[:, :, 4:8] = 2.0
    reconstructed[:, :, 8:] = 4.0

    innovation = reconstructed_key_innovation(
        reconstructed,
        _key_parameters(),
        (4, 4, 3),
        legacy=True,
    )

    assert torch.count_nonzero(innovation[:, :, :4]) == 0
    expected_second = torch.tensor(1.375, dtype=torch.bfloat16)
    expected_tail = torch.tensor(3.65, dtype=torch.bfloat16)
    assert torch.equal(
        innovation[:, :, 4:8], torch.full_like(innovation[:, :, 4:8], expected_second)
    )
    assert torch.equal(
        innovation[:, :, 8:], torch.full_like(innovation[:, :, 8:], expected_tail)
    )


def test_innovation_codec_encoder_and_decoder_are_bit_exact() -> None:
    generator = torch.Generator().manual_seed(20260829)
    key = torch.randn(1, 1, 11, 64, generator=generator)
    value = torch.randn(1, 1, 11, 64, generator=generator)
    config = _config()
    key_parameters = _key_parameters()
    value_parameters = _innovation_parameters()
    codec = TempoKVCodec(key_parameters, value_parameters, config)

    _, reconstructed_key = codec.encode_key(key)
    expected_packed, encoder_reconstruction = encode_innovation(
        value,
        reconstructed_key,
        key_parameters,
        value_parameters,
        config,
    )
    packed = codec.encode_value(value, reconstructed_key)
    direct_output = torch.empty(1, 11, 1, 64, dtype=torch.bfloat16)
    decoded = decode_tensor(
        packed,
        source=reconstructed_key,
        output_tensor=direct_output,
        output_layout="BSHD",
        use_triton=False,
    )

    assert packed.predictor_kind == "innovation"
    assert packed.temporal is not None
    assert packed.cross_kv is not None
    assert packed.cross_kv.projection_source == "innovation"
    assert torch.equal(packed.residual_quant, expected_packed.residual_quant)
    assert torch.equal(packed.residual_scale, expected_packed.residual_scale)
    assert torch.equal(decoded, encoder_reconstruction)
    assert torch.equal(direct_output.permute(0, 2, 1, 3), encoder_reconstruction)


def test_projection_source_roundtrips_and_prevents_cross_innovation_mixup(
    tmp_path,
) -> None:
    path = tmp_path / "v_innovation.npz"
    parameters = _innovation_parameters()
    parameters.save(path)
    loaded = CrossKVParameters.load(path)

    assert loaded.projection_source == "innovation"
    with pytest.raises(ValueError, match="projection_source=cross_kv"):
        TempoKVCodec(
            _key_parameters(),
            loaded,
            _config(v_predictor_mode="cross_kv"),
        )
    with pytest.raises(ValueError, match="projection_source=innovation"):
        TempoKVCodec(
            _key_parameters(),
            _innovation_parameters(projection_source="cross_kv"),
            _config(),
        )


def test_legacy_innovation_fit_keeps_layer_shared_map_and_diagonal_v_history() -> None:
    generator = torch.Generator().manual_seed(81)
    keys = torch.randn(2, 6, 1, 64, generator=generator)
    values = torch.randn(2, 6, 1, 64, generator=generator)

    key_parameters, value_parameters = fit_predictors(
        keys,
        values,
        config=_config(),
        iterations=2,
        temporal_ridge=1.0,
        cross_ridge=0.1,
        v_history_ridge=0.0,
        max_gain=1.0,
        v_history_mode="diagonal",
    )

    assert key_parameters.history_length == 2
    assert isinstance(value_parameters, CrossKVParameters)
    assert value_parameters.projection_source == "innovation"
    assert tuple(value_parameters.weight.shape) == (2, 1, 64, 64)
    assert tuple(value_parameters.history_weights.shape) == (2, 1, 1, 64)
    assert torch.equal(value_parameters.weight[0], value_parameters.weight[1])
    assert torch.equal(value_parameters.bias[0], value_parameters.bias[1])


def test_attention_refinement_rejects_innovation_before_optimization() -> None:
    trace = torch.zeros(1, 4, 1, 64)
    with pytest.raises(ValueError, match="does not support innovation"):
        refine_attention(
            trace,
            trace,
            trace,
            trace,
            _key_parameters(),
            _innovation_parameters(),
            _config(),
        )
