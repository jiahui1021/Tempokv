from __future__ import annotations

import torch

from tempokv.parameters import (
    CrossKVParameters,
    TemporalParameters,
    load_parameters,
)


def test_temporal_parameter_roundtrip(tmp_path) -> None:
    value = TemporalParameters(
        torch.eye(3)[:, :, None].expand(3, 3, 4),
        torch.zeros(3, 4),
        1560,
    )
    path = tmp_path / "temporal.npz"
    value.save(path)
    loaded = load_parameters(path)
    assert isinstance(loaded, TemporalParameters)
    assert loaded.stride == 1560
    torch.testing.assert_close(loaded.history_weights, value.history_weights)


def test_layered_cross_kv_selection_and_roundtrip(tmp_path) -> None:
    weight = torch.randn(3, 2, 4, 4)
    bias = torch.randn(3, 2, 4)
    value = CrossKVParameters(weight, bias, 1560)
    path = tmp_path / "cross_kv.npz"
    value.save(path)
    loaded = load_parameters(path)
    assert isinstance(loaded, CrossKVParameters)
    selected = loaded.for_layer(1)
    assert selected.weight.shape == (2, 4, 4)
    torch.testing.assert_close(selected.weight, weight[1])


def test_explicit_layer_ids_are_used_instead_of_array_position() -> None:
    weight = torch.stack([torch.full((2, 4, 4), 7.0), torch.full((2, 4, 4), 19.0)])
    bias = torch.zeros(2, 2, 4)
    parameters = CrossKVParameters(weight, bias, layer_ids=(7, 19))
    torch.testing.assert_close(parameters.for_layer(19).weight, weight[1])


def test_parameter_codec_signature_roundtrip(tmp_path) -> None:
    signature = "tempokv-codec-v1:test"
    value = TemporalParameters(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 2),
        stride=4,
        codec_signature=signature,
        parameter_set_id="paired",
        coordinate_system="pre_rope_normalized",
    )
    path = tmp_path / "signed.npz"
    value.save(path)
    loaded = TemporalParameters.load(path)
    assert loaded.codec_signature == signature
    assert loaded.parameter_set_id == "paired"
    assert loaded.coordinate_system == "pre_rope_normalized"
