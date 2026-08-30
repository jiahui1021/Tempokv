"""Regressions for the defects found in the repository review.

Each test names the failure it prevents rather than the code it touches, so a
future change that reintroduces one of these fails with the reason attached.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from tempokv.attention_objective import temporal_rollout
from tempokv.config import CodecConfig, config_from_mapping
from tempokv.quantization import quantize_asymmetric, quantize_symmetric


# --------------------------------------------------------------------------
# Integral codec settings
# --------------------------------------------------------------------------


def test_whole_floats_are_coerced_so_signatures_cannot_diverge() -> None:
    """``4`` and ``4.0`` described the same codec but signed differently."""
    integral = config_from_mapping(
        {"k_bits": 4, "v_bits": 4, "group_size": 64, "stride": 1560}
    )
    floating = config_from_mapping(
        {"k_bits": 4.0, "v_bits": 4, "group_size": 64.0, "stride": 1560.0}
    )
    assert isinstance(floating.k_bits, int)
    assert isinstance(floating.group_size, int)
    assert isinstance(floating.stride, int)
    assert floating.format_signature() == integral.format_signature()


@pytest.mark.parametrize(
    "payload",
    [
        {"k_bits": 4.5},
        {"group_size": 64.5},
        {"stride": 1560.5},
    ],
)
def test_non_integral_settings_are_refused_at_construction(payload: dict) -> None:
    """Failing here beats failing later inside a reshape or a slice."""
    with pytest.raises(ValueError, match="whole number"):
        config_from_mapping({"k_bits": 4, "v_bits": 4, **payload})


def test_booleans_are_not_accepted_as_bit_widths() -> None:
    with pytest.raises(ValueError, match="whole number"):
        config_from_mapping({"k_bits": True, "v_bits": 4})


# --------------------------------------------------------------------------
# Scale storage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("quantize", [quantize_symmetric, quantize_asymmetric])
def test_a_scale_that_overflows_its_storage_dtype_is_refused(quantize) -> None:
    """It used to store ``inf`` and dequantize every value to NaN."""
    value = torch.full((1, 1, 4, 64), 1e7)
    with pytest.raises(ValueError, match="overflows"):
        quantize(value, bits=4, group_size=64, scale_dtype=torch.float16)


@pytest.mark.parametrize("quantize", [quantize_symmetric, quantize_asymmetric])
def test_ordinary_inputs_still_quantize_into_float16_scales(quantize) -> None:
    """The overflow guard must not reject the range the codec actually uses."""
    torch.manual_seed(0)
    result = quantize(
        torch.randn(1, 1, 4, 64), bits=4, group_size=64, scale_dtype=torch.float16
    )
    scale = result[1]
    assert torch.isfinite(scale).all()


# --------------------------------------------------------------------------
# Refinement rollout precision
# --------------------------------------------------------------------------


def _rollout_config(reconstruction_dtype: str) -> CodecConfig:
    return CodecConfig(
        k_bits=4,
        v_bits=4,
        anchor_bits=4,
        group_size=4,
        stride=2,
        scale_dtype="float32",
        reconstruction_dtype=reconstruction_dtype,
        k_history_length=1,
        v_history_length=0,
        use_triton=False,
    )


def test_the_rollout_follows_the_configured_reconstruction_dtype() -> None:
    """It hardcoded bfloat16, so a float32 codec was refined against another."""
    torch.manual_seed(0)
    target = torch.randn(1, 2, 12, 4)
    weight = torch.full((1, 1, 4), 0.8)
    bias = torch.zeros(1, 4)
    outputs = {
        name: temporal_rollout(
            target, weight, bias, bits=4, config=_rollout_config(name)
        )
        for name in ("float32", "bfloat16")
    }
    assert not torch.equal(outputs["float32"], outputs["bfloat16"])


# --------------------------------------------------------------------------
# Profiling attribution
# --------------------------------------------------------------------------


PROFILING = Path(__file__).parents[2] / "src" / "tempokv" / "profiling.py"


def test_decode_sections_are_attributed_by_cache_role_not_predictor_kind() -> None:
    """A temporal V, and a mirrored K, both defeat the predictor-kind shortcut."""
    source = PROFILING.read_text(encoding="utf-8")
    assert "_ACTIVE_CACHE_ROLE" in source
    # The old shortcut decided K purely from the predictor kind.
    assert 'DECODE_K if kind == "temporal" else DECODE_V' not in source


# --------------------------------------------------------------------------
# LongCat mirrored decode
# --------------------------------------------------------------------------


LONGCAT = Path(__file__).parents[2] / "experiments" / "LongCat"


def _longcat_bridge():
    if str(LONGCAT) not in sys.path:
        sys.path.insert(0, str(LONGCAT))
    return importlib.import_module("longcat.bridge")


def _fitted_codec(direction: str):
    from tempokv.codec import TempoKVCodec
    from tempokv.fitting import fit_predictors

    torch.manual_seed(0)
    layers, records, frames, heads, dim = 2, 6, 5, 2, 128
    keys = torch.zeros(layers, records, frames, heads, dim)
    values = torch.zeros_like(keys)
    shape = (layers, records, heads, dim)
    keys[:, :, 0] = torch.randn(shape)
    values[:, :, 0] = torch.randn(shape)
    for frame in range(1, frames):
        keys[:, :, frame] = 0.9 * keys[:, :, frame - 1] + 0.1 * torch.randn(shape)
        values[:, :, frame] = (
            0.6 * keys[:, :, frame]
            + 0.3 * values[:, :, frame - 1]
            + 0.1 * torch.randn(shape)
        )
    config = CodecConfig(
        k_bits=2,
        v_bits=2,
        stride=4,
        group_size=64,
        k_history_length=2,
        v_history_length=1,
        use_triton=False,
        strict_triton=False,
        predictor_direction=direction,
    )
    temporal, secondary = fit_predictors(
        keys,
        values,
        config=config,
        iterations=2,
        temporal_ridge=1.0,
        cross_ridge=0.1,
        v_history_mode="diagonal",
        coordinate_system="pre_rope_normalized",
    )
    codec = TempoKVCodec(temporal, secondary, config)
    span_keys = keys[0, 0].permute(1, 0, 2).unsqueeze(0).contiguous()
    span_values = values[0, 0].permute(1, 0, 2).unsqueeze(0).contiguous()
    return codec, span_keys, span_values


@pytest.mark.parametrize("direction", ["k_to_v", "v_to_k"])
def test_longcat_decodes_both_predictor_directions(direction: str) -> None:
    """The mirrored path used to be rejected before it decoded anything."""
    bridge = _longcat_bridge()
    codec, span_keys, span_values = _fitted_codec(direction)
    packed = codec.encode(span_keys, span_values, layer=0)
    rebuilt_keys, rebuilt_values = bridge.decode_kv_cache(
        packed.key, packed.value, codec
    )
    assert rebuilt_keys.shape == span_keys.shape
    assert rebuilt_values.shape == span_values.shape

    def error(rebuilt, reference):
        return float((rebuilt.double() - reference.double()).square().sum())

    assert error(rebuilt_keys, span_keys) < error(rebuilt_keys, span_values)
    assert error(rebuilt_values, span_values) < error(rebuilt_values, span_keys)


@pytest.mark.parametrize("direction", ["k_to_v", "v_to_k"])
def test_longcat_span_sets_are_labelled_by_the_spans_they_hold(
    direction: str,
) -> None:
    """The encoder used to label K temporal and V cross regardless of direction.

    Building the span sets by hand would not catch that, so this drives the
    real encoder and then decodes what it produced.
    """
    bridge = _longcat_bridge()
    codec, span_keys, span_values = _fitted_codec(direction)
    # The LongCat cache is addressed in tokens: one temporal unit is `stride`
    # tokens wide, so expand each frame before handing it to the encoder.
    stride = codec.config.stride
    frames = span_keys.shape[2]

    def _expand(tensor):
        batch, heads, _, dim = tensor.shape
        return (
            tensor.permute(0, 2, 1, 3)
            .unsqueeze(2)
            .expand(batch, frames, stride, heads, dim)
            .reshape(batch, frames * stride, heads, dim)
            .permute(0, 2, 1, 3)
            # The encoder reconstructs in place, so it needs the cache dtype.
            .to(torch.bfloat16)
            .contiguous()
        )

    cache_keys, cache_values = _expand(span_keys), _expand(span_values)
    spec = bridge.LongCatRuntimeSpec(
        num_layers=1,
        num_heads=span_keys.shape[1],
        head_dim=span_keys.shape[3],
        stride=codec.config.stride,
    )
    combined, _ = bridge.encode_kv_cache_spans(
        {0: (cache_keys, cache_values)},
        codec=codec,
        span_temporal_units=(frames,),
        spec=spec,
    )
    key_set, value_set = combined[0]
    expected = (
        ("cross_kv", "temporal") if direction == "v_to_k" else ("temporal", "cross_kv")
    )
    assert (key_set.predictor_kind, value_set.predictor_kind) == expected
    # A label that contradicts the packed state would be rejected here.
    rebuilt_keys, rebuilt_values = bridge.decode_kv_cache(key_set, value_set, codec)

    def error(rebuilt, reference):
        return float((rebuilt.double() - reference.double()).square().sum())

    assert error(rebuilt_keys, cache_keys) < error(rebuilt_keys, cache_values)
    assert error(rebuilt_values, cache_values) < error(rebuilt_values, cache_keys)


def test_history_lengths_are_coerced_like_the_other_integral_settings() -> None:
    config = config_from_mapping(
        {"k_bits": 4, "v_bits": 4, "k_history_length": 2.0, "v_history_length": 1.0}
    )
    assert isinstance(config.k_history_length, int)
    assert isinstance(config.v_history_length, int)
