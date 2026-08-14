from __future__ import annotations

import torch
import pytest

from tempokv.attention_objective import (
    cross_kv_rollout,
    objective_terms,
    temporal_rollout,
)
from tempokv.config import CodecConfig
from tempokv.fitting import _objective_sums, refine_attention
from tempokv.parameters import CrossKVParameters, TemporalParameters


def _config() -> CodecConfig:
    return CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=2,
        group_size=4,
        stride=2,
        k_history_length=1,
        v_history_length=1,
        v_predictor_mode="cross_kv",
        residual_mode="asymmetric",
        scale_dtype="float32",
        reconstruction_dtype="bfloat16",
        use_triton=False,
    )


def test_cross_kv_rollout_uses_reconstructed_history_and_is_differentiable() -> None:
    source = torch.tensor(
        [[[[[1.0, 2.0, 0.5, -0.5]], [[2.0, 1.0, -1.0, 0.5]], [[3.0, -1.0, 0.25, 1.0]]]]]
    )
    target = torch.tensor(
        [[[[[0.5, -0.5, 0.25, 1.0]], [[1.0, 0.25, -0.5, 0.5]], [[1.5, 0.75, 1.0, -1.0]]]]]
    )
    weight = torch.eye(4).view(1, 1, 4, 4).requires_grad_(True)
    bias = torch.zeros(1, 1, 4, requires_grad=True)
    history = torch.full((1, 1, 1, 4), 0.25, requires_grad=True)

    rebuilt = cross_kv_rollout(
        source,
        target,
        weight,
        bias,
        history,
        bits=2,
        config=_config(),
    )

    assert rebuilt.shape == target.shape
    rebuilt[..., 1:, :, :].square().sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert bias.grad is not None and torch.isfinite(bias.grad).all()
    assert history.grad is not None and torch.isfinite(history.grad).all()


@pytest.mark.parametrize("phase", ["key", "value"])
def test_attention_adam_gradient_matches_normalized_half_objective(
    phase: str,
) -> None:
    generator = torch.Generator().manual_seed(11)
    keys = torch.randn(1, 2, 4, 1, 4, generator=generator)
    values = torch.randn(keys.shape, generator=generator)
    query = torch.randn(keys.shape, generator=generator)
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=2,
        group_size=4,
        stride=2,
        k_history_length=1,
        v_history_length=1,
        v_predictor_mode="temporal",
        scale_dtype="float32",
        use_triton=False,
    )
    key_weight = torch.full((1, 1, 4), 0.7)
    key_bias = torch.zeros(1, 4)
    value_weight = torch.full((1, 1, 4), 0.6)
    value_bias = torch.zeros(1, 4)
    baseline = _objective_sums(
        keys,
        values,
        query,
        keys,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        config=config,
        layer_chunk=1,
    )

    actual_weight = (
        key_weight if phase == "value" else key_weight.clone().requires_grad_(True)
    )
    actual_bias = (
        key_bias if phase == "value" else key_bias.clone().requires_grad_(True)
    )
    actual_value_weight = (
        value_weight
        if phase == "key"
        else value_weight.clone().requires_grad_(True)
    )
    actual_value_bias = (
        value_bias if phase == "key" else value_bias.clone().requires_grad_(True)
    )
    _objective_sums(
        keys,
        values,
        query,
        keys,
        actual_weight,
        actual_bias,
        actual_value_weight,
        actual_value_bias,
        config=config,
        layer_chunk=1,
        backward_tensor=phase,
        reference_sums=baseline,
    )

    expected_key_weight = key_weight.clone().requires_grad_(phase == "key")
    expected_key_bias = key_bias.clone().requires_grad_(phase == "key")
    expected_value_weight = value_weight.clone().requires_grad_(phase == "value")
    expected_value_bias = value_bias.clone().requires_grad_(phase == "value")
    rebuilt_key = temporal_rollout(
        keys,
        expected_key_weight,
        expected_key_bias,
        bits=config.k_bits,
        config=config,
    )
    rebuilt_value = temporal_rollout(
        values,
        expected_value_weight,
        expected_value_bias,
        bits=config.v_bits,
        config=config,
    )
    terms = objective_terms(
        clean_key=keys,
        clean_value=values,
        rebuilt_key=rebuilt_key,
        rebuilt_value=rebuilt_value,
        query_post_rope=query,
        key_post_rope=keys,
    )
    own_error = terms.key_error if phase == "key" else terms.value_error
    own_reference = baseline[1] if phase == "key" else baseline[3]
    expected_loss = 0.5 * own_error / own_reference + 0.5 * (
        terms.attention_error / baseline[5]
    )
    expected_loss.backward()

    if phase == "key":
        assert torch.allclose(actual_weight.grad, expected_key_weight.grad)
        assert torch.allclose(actual_bias.grad, expected_key_bias.grad)
    else:
        assert torch.allclose(actual_value_weight.grad, expected_value_weight.grad)
        assert torch.allclose(actual_value_bias.grad, expected_value_bias.grad)


def test_refine_attention_accepts_cross_kv_with_v_history() -> None:
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn(1, 3, 4, 1, 4, generator=generator)
    values = 0.6 * keys + 0.2 * torch.randn(
        keys.shape, generator=generator
    )
    query_post_rope = torch.randn(keys.shape, generator=generator)
    key_post_rope = keys.clone()
    key_parameters = TemporalParameters(
        history_weights=torch.full((1, 1, 4), 0.8),
        history_bias=torch.zeros(1, 4),
        stride=2,
    )
    value_parameters = CrossKVParameters(
        weight=torch.eye(4).view(1, 1, 4, 4) * 0.5,
        bias=torch.zeros(1, 1, 4),
        stride=2,
        layer_ids=(0,),
        codec_signature="test-signature",
        parameter_set_id="test-set",
        coordinate_system="pre_rope_normalized",
        history_weights=torch.full((1, 1, 1, 4), 0.2),
    )

    refined_key, refined_value, report = refine_attention(
        keys,
        values,
        query_post_rope,
        key_post_rope,
        key_parameters,
        value_parameters,
        _config(),
        steps=3,
        learning_rate=1e-4,
        patience=2,
        layer_chunk=1,
    )

    assert isinstance(refined_key, TemporalParameters)
    assert isinstance(refined_value, CrossKVParameters)
    assert refined_key.parameter_set_id == refined_value.parameter_set_id
    assert refined_value.parameter_set_id != "test-set"
    assert refined_value.coordinate_system == "pre_rope_normalized"
    assert report["optimizer"] == "adam"
    assert report["objective_weights"] == {
        "reconstruction": 0.5,
        "attention": 0.5,
    }
    assert [phase["tensor"] for phase in report["phases"]] == ["key", "value"]
    assert all(1 <= phase["steps_run"] <= 3 for phase in report["phases"])
    assert all(
        torch.isfinite(torch.tensor(list(report[name].values()))).all()
        for name in ("baseline", "final")
    )


def test_refine_attention_keeps_temporal_v_path() -> None:
    generator = torch.Generator().manual_seed(23)
    keys = torch.randn(1, 2, 4, 1, 4, generator=generator)
    values = torch.randn(keys.shape, generator=generator)
    query_post_rope = torch.randn(keys.shape, generator=generator)
    key_parameters = TemporalParameters(
        history_weights=torch.full((1, 1, 4), 0.7),
        history_bias=torch.zeros(1, 4),
        stride=2,
    )
    value_parameters = TemporalParameters(
        history_weights=torch.full((1, 1, 4), 0.6),
        history_bias=torch.zeros(1, 4),
        stride=2,
    )
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=2,
        group_size=4,
        stride=2,
        k_history_length=1,
        v_history_length=1,
        v_predictor_mode="temporal",
        scale_dtype="float32",
        use_triton=False,
    )

    _, refined_value, report = refine_attention(
        keys,
        values,
        query_post_rope,
        keys,
        key_parameters,
        value_parameters,
        config,
        steps=3,
        learning_rate=1e-4,
        patience=2,
        layer_chunk=1,
    )

    assert isinstance(refined_value, TemporalParameters)
    assert report["optimizer"] == "adam"
    assert [phase["tensor"] for phase in report["phases"]] == ["key", "value"]


def test_refine_attention_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(29)
    keys = torch.randn(1, 2, 4, 1, 4, generator=generator)
    values = torch.randn(keys.shape, generator=generator)
    query = torch.randn(keys.shape, generator=generator)
    key_parameters = TemporalParameters(
        history_weights=torch.full((1, 1, 4), 0.7),
        history_bias=torch.zeros(1, 4),
        stride=2,
    )
    value_parameters = CrossKVParameters(
        weight=torch.eye(4).view(1, 1, 4, 4) * 0.4,
        bias=torch.zeros(1, 1, 4),
        stride=2,
        layer_ids=(0,),
        history_weights=torch.full((1, 1, 1, 4), 0.1),
    )

    first = refine_attention(
        keys,
        values,
        query,
        keys,
        key_parameters,
        value_parameters,
        _config(),
        steps=4,
        learning_rate=1e-4,
        patience=2,
        layer_chunk=1,
    )
    second = refine_attention(
        keys,
        values,
        query,
        keys,
        key_parameters,
        value_parameters,
        _config(),
        steps=4,
        learning_rate=1e-4,
        patience=2,
        layer_chunk=1,
    )

    assert torch.equal(first[0].history_weights, second[0].history_weights)
    assert torch.equal(first[1].weight, second[1].weight)
    assert torch.equal(first[1].history_weights, second[1].history_weights)
    assert first[2] == second[2]


def test_native_temporal_rollout_matches_independent_layers() -> None:
    generator = torch.Generator().manual_seed(31)
    target = torch.randn(2, 3, 4, 2, 4, generator=generator)
    weight = torch.randn(2, 1, 1, 2, 4, generator=generator).requires_grad_(True)
    bias = torch.randn(2, 1, 2, 4, generator=generator).requires_grad_(True)
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=2,
        group_size=4,
        stride=2,
        k_history_length=1,
        v_history_length=1,
        v_predictor_mode="temporal",
        scale_dtype="float32",
        use_triton=False,
    )

    layered = temporal_rollout(
        target, weight, bias, bits=config.k_bits, config=config
    )
    independent = torch.stack(
        [
            temporal_rollout(
                target[layer],
                weight[layer],
                bias[layer],
                bits=config.k_bits,
                config=config,
            )
            for layer in range(2)
        ]
    )

    assert torch.equal(layered, independent)
    layered[..., 1:, :, :].square().sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert bias.grad is not None and torch.isfinite(bias.grad).all()


def test_native_layer_chunks_preserve_the_global_objective_gradient() -> None:
    generator = torch.Generator().manual_seed(33)
    keys = torch.randn(2, 2, 4, 1, 4, generator=generator)
    values = torch.randn(keys.shape, generator=generator)
    query = torch.randn(keys.shape, generator=generator)
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=2,
        group_size=4,
        stride=2,
        k_history_length=1,
        v_history_length=1,
        v_predictor_mode="temporal",
        scale_dtype="float32",
        use_triton=False,
    )
    key_weight = torch.full((2, 1, 1, 1, 4), 0.7)
    key_bias = torch.zeros(2, 1, 1, 4)
    value_weight = torch.full((2, 1, 1, 1, 4), 0.6)
    value_bias = torch.zeros(2, 1, 1, 4)
    reference = _objective_sums(
        keys,
        values,
        query,
        keys,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        config=config,
        layer_chunk=2,
    )

    gradients = []
    for layer_chunk in (1, 2):
        candidate = key_weight.clone().requires_grad_(True)
        _objective_sums(
            keys,
            values,
            query,
            keys,
            candidate,
            key_bias,
            value_weight,
            value_bias,
            config=config,
            layer_chunk=layer_chunk,
            backward_tensor="key",
            reference_sums=reference,
        )
        assert candidate.grad is not None
        gradients.append(candidate.grad)

    assert torch.allclose(gradients[0], gradients[1])


@pytest.mark.parametrize("v_predictor_mode", ["temporal", "cross_kv"])
def test_refine_attention_accepts_native_layered_parameters(
    v_predictor_mode: str,
) -> None:
    generator = torch.Generator().manual_seed(37)
    keys = torch.randn(2, 2, 4, 1, 4, generator=generator)
    values = 0.4 * keys + torch.randn(keys.shape, generator=generator) * 0.1
    query = torch.randn(keys.shape, generator=generator)
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=2,
        group_size=4,
        stride=2,
        k_history_length=1,
        v_history_length=1,
        v_predictor_mode=v_predictor_mode,
        scale_dtype="float32",
        use_triton=False,
    )
    key_parameters = TemporalParameters(
        history_weights=torch.full((2, 1, 1, 1, 4), 0.7),
        history_bias=torch.zeros(2, 1, 1, 4),
        stride=2,
        layer_ids=(0, 1),
    )
    value_parameters: TemporalParameters | CrossKVParameters
    if v_predictor_mode == "temporal":
        value_parameters = TemporalParameters(
            history_weights=torch.full((2, 1, 1, 1, 4), 0.6),
            history_bias=torch.zeros(2, 1, 1, 4),
            stride=2,
            layer_ids=(0, 1),
        )
    else:
        value_parameters = CrossKVParameters(
            weight=torch.eye(4).view(1, 1, 4, 4).repeat(2, 1, 1, 1) * 0.4,
            bias=torch.zeros(2, 1, 4),
            stride=2,
            layer_ids=(0, 1),
            history_weights=torch.full((2, 1, 1, 4), 0.1),
        )

    refined_key, refined_value, report = refine_attention(
        keys,
        values,
        query,
        keys,
        key_parameters,
        value_parameters,
        config,
        steps=2,
        learning_rate=1e-4,
        patience=1,
        layer_chunk=1,
    )

    assert refined_key.history_weights.shape == (2, 1, 1, 1, 4)
    assert refined_value.layer_ids == (0, 1)
    assert report["optimizer"] == "adam"
    assert report["refine_tensors"] == ["key", "value"]


def test_reconstruction_only_value_phase_and_sign_scan_are_explicit() -> None:
    generator = torch.Generator().manual_seed(41)
    keys = torch.randn(1, 2, 4, 1, 4, generator=generator)
    values = torch.randn(keys.shape, generator=generator)
    query = torch.randn(keys.shape, generator=generator)
    key_parameters = TemporalParameters(
        history_weights=torch.full((1, 1, 4), 0.7),
        history_bias=torch.zeros(1, 4),
        stride=2,
    )
    value_parameters = TemporalParameters(
        history_weights=torch.full((1, 1, 4), 0.6),
        history_bias=torch.zeros(1, 4),
        stride=2,
    )
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=2,
        group_size=4,
        stride=2,
        k_history_length=1,
        v_history_length=1,
        v_predictor_mode="temporal",
        scale_dtype="float32",
        use_triton=False,
    )

    refined_key, _, report = refine_attention(
        keys,
        values,
        query,
        keys,
        key_parameters,
        value_parameters,
        config,
        optimizer_name="sign_gradient",
        reconstruction_weight=1.0,
        attention_weight=0.0,
        refine_tensors=("value",),
        sign_step_sizes=(1e-4, 1e-3),
        layer_chunk=1,
    )

    assert torch.equal(refined_key.history_weights, key_parameters.history_weights)
    assert report["optimizer"] == "sign_gradient"
    assert report["objective_weights"] == {
        "reconstruction": 1.0,
        "attention": 0.0,
    }
    assert report["refine_tensors"] == ["value"]
    assert report["sign_step_sizes"] == [1e-4, 1e-3]
    assert [phase["tensor"] for phase in report["phases"]] == ["value"]


def test_adam_reconstruction_only_is_a_distinct_objective() -> None:
    generator = torch.Generator().manual_seed(43)
    keys = torch.randn(1, 2, 4, 1, 4, generator=generator)
    values = torch.randn(keys.shape, generator=generator)
    query = torch.randn(keys.shape, generator=generator)
    key_parameters = TemporalParameters(
        history_weights=torch.full((1, 1, 4), 0.7),
        history_bias=torch.zeros(1, 4),
        stride=2,
    )
    value_parameters = TemporalParameters(
        history_weights=torch.full((1, 1, 4), 0.6),
        history_bias=torch.zeros(1, 4),
        stride=2,
    )
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=2,
        group_size=4,
        stride=2,
        k_history_length=1,
        v_history_length=1,
        v_predictor_mode="temporal",
        scale_dtype="float32",
        use_triton=False,
    )

    _, _, report = refine_attention(
        keys,
        values,
        query,
        keys,
        key_parameters,
        value_parameters,
        config,
        steps=2,
        learning_rate=1e-4,
        patience=1,
        reconstruction_weight=1.0,
        attention_weight=0.0,
        layer_chunk=1,
    )

    assert report["optimizer"] == "adam"
    assert report["objective_weights"] == {
        "reconstruction": 1.0,
        "attention": 0.0,
    }


def test_cross_kv_only_refinement_skips_the_empty_history_parameter() -> None:
    generator = torch.Generator().manual_seed(47)
    keys = torch.randn(1, 2, 4, 1, 4, generator=generator)
    values = torch.randn(keys.shape, generator=generator)
    query = torch.randn(keys.shape, generator=generator)
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        anchor_bits=2,
        group_size=4,
        stride=2,
        k_history_length=1,
        v_history_length=0,
        v_predictor_mode="cross_kv",
        scale_dtype="float32",
        use_triton=False,
    )
    key_parameters = TemporalParameters(
        history_weights=torch.full((1, 1, 1, 1, 4), 0.7),
        history_bias=torch.zeros(1, 1, 1, 4),
        stride=2,
        layer_ids=(0,),
    )
    value_parameters = CrossKVParameters(
        weight=torch.eye(4).view(1, 1, 4, 4) * 0.4,
        bias=torch.zeros(1, 1, 4),
        stride=2,
        layer_ids=(0,),
        history_weights=torch.empty(1, 0, 1, 4),
    )

    _, refined_value, report = refine_attention(
        keys,
        values,
        query,
        keys,
        key_parameters,
        value_parameters,
        config,
        steps=2,
        learning_rate=1e-4,
        patience=1,
        refine_tensors=("value",),
        layer_chunk=1,
    )

    assert isinstance(refined_value, CrossKVParameters)
    assert refined_value.history_weights.shape == (1, 0, 1, 4)
    assert report["refine_tensors"] == ["value"]
