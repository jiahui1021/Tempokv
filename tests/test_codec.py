from __future__ import annotations

import pytest
import torch

from tempokv import (
    CodecConfig,
    CrossKVParameters,
    TempoKVCodec,
    TemporalParameters,
    decode_tensor,
    encode_cross_kv,
    encode_temporal,
)
from tempokv.codec import predict_temporal


def temporal_parameters(history: int = 3, *, stride: int = 4) -> TemporalParameters:
    weights = torch.zeros(history, history, 8)
    weights[0, 0] = 0.8
    if history > 1:
        weights[1, 0] = 0.6
        weights[1, 1] = 0.25
    if history > 2:
        weights[2, 0] = 0.55
        weights[2, 1] = 0.25
        weights[2, 2] = 0.1
    return TemporalParameters(weights, torch.zeros(history, 8), stride)


def cross_parameters(heads: int = 2, width: int = 8, *, stride: int = 4) -> CrossKVParameters:
    generator = torch.Generator().manual_seed(9)
    weight = torch.randn(heads, width, width, generator=generator) * 0.1
    bias = torch.randn(heads, width, generator=generator) * 0.05
    return CrossKVParameters(weight, bias, stride)


def config(**updates) -> CodecConfig:
    values = dict(
        k_bits=2,
        v_bits=4,
        anchor_bits=4,
        group_size=4,
        stride=4,
        residual_mode="asymmetric",
        scale_dtype="float32",
        reconstruction_dtype="float32",
        use_triton=False,
    )
    values.update(updates)
    return CodecConfig(**values)


def test_temporal_formula_uses_separate_warmup_rows() -> None:
    weights = torch.zeros(3, 3, 4)
    weights[0, 0] = 2.0
    weights[1, 0] = 0.5
    weights[1, 1] = 0.25
    weights[2, 0] = 0.6
    weights[2, 1] = 0.3
    weights[2, 2] = 0.1
    bias = torch.tensor([[1.0] * 4, [2.0] * 4, [3.0] * 4])
    parameters = TemporalParameters(weights, bias)
    newest = torch.full((1, 2, 3, 4), 3.0)
    middle = torch.full_like(newest, 2.0)
    oldest = torch.full_like(newest, 1.0)

    torch.testing.assert_close(
        predict_temporal([newest], parameters), torch.full_like(newest, 7.0)
    )
    torch.testing.assert_close(
        predict_temporal([newest, middle], parameters), torch.full_like(newest, 4.0)
    )
    torch.testing.assert_close(
        predict_temporal([newest, middle, oldest], parameters),
        torch.full_like(newest, 5.5),
    )


@pytest.mark.parametrize("mode", ["symmetric", "asymmetric"])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_k_closed_loop_encode_matches_decode(mode: str, bits: int) -> None:
    generator = torch.Generator().manual_seed(2026 + bits)
    key = torch.randn(1, 2, 14, 8, generator=generator, dtype=torch.float32)
    settings = config(k_bits=bits, residual_mode=mode)
    packed, encoder_reconstruction = encode_temporal(
        key, temporal_parameters(), settings
    )
    decoded = decode_tensor(packed, output_dtype=torch.float32, use_triton=False)

    assert packed.unit_lengths == (4, 4, 4, 2)
    torch.testing.assert_close(decoded, encoder_reconstruction, rtol=0, atol=0)


def test_cross_kv_closed_loop_encode_matches_decode() -> None:
    generator = torch.Generator().manual_seed(31)
    key = torch.randn(1, 2, 13, 8, generator=generator)
    value = torch.randn(1, 2, 13, 8, generator=generator)
    settings = config()
    _, key_reconstruction = encode_temporal(key, temporal_parameters(), settings)
    packed, encoder_reconstruction = encode_cross_kv(
        value, key_reconstruction, cross_parameters(), settings
    )
    decoded = decode_tensor(
        packed,
        source=key_reconstruction,
        output_dtype=torch.float32,
        use_triton=False,
    )

    torch.testing.assert_close(decoded, encoder_reconstruction, rtol=0, atol=0)
    with pytest.raises(ValueError, match="requires reconstructed K"):
        decode_tensor(packed, output_dtype=torch.float32, use_triton=False)


def test_cross_kv_rejects_source_with_different_closed_loop_dtype() -> None:
    value = torch.randn(1, 2, 8, 8)
    parameters = cross_parameters()
    settings = config(reconstruction_dtype="bfloat16")
    with pytest.raises(ValueError, match="reconstructed K dtype must match"):
        encode_cross_kv(value, value.float(), parameters, settings)
    source = value.to(torch.bfloat16)
    packed, _ = encode_cross_kv(value, source, parameters, settings)
    with pytest.raises(ValueError, match="reconstructed K dtype must match"):
        decode_tensor(packed, source=value.float(), use_triton=False)


def test_end_to_end_codec_uses_reconstructed_k_for_v() -> None:
    generator = torch.Generator().manual_seed(77)
    key = torch.randn(1, 2, 15, 8, generator=generator, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    codec = TempoKVCodec(temporal_parameters(), cross_parameters(), config())
    packed = codec.encode(key, value)
    rebuilt_key, rebuilt_value = codec.decode(packed)

    assert rebuilt_key.shape == key.shape
    assert rebuilt_value.shape == value.shape
    assert torch.isfinite(rebuilt_key).all()
    assert torch.isfinite(rebuilt_value).all()


def test_runtime_rejects_parameter_stride_mismatch() -> None:
    key = torch.zeros(1, 2, 8, 8)
    with pytest.raises(ValueError, match="different stride"):
        encode_temporal(key, temporal_parameters(stride=8), config(stride=4))


def test_decode_rejects_different_closed_loop_dtype() -> None:
    key = torch.randn(1, 2, 8, 8)
    packed, _ = encode_temporal(key, temporal_parameters(), config())
    with pytest.raises(ValueError, match="decode dtype must match"):
        decode_tensor(packed, output_dtype=torch.bfloat16, use_triton=False)


@pytest.mark.parametrize(
    ("bits", "width", "group_size", "divisor"),
    [(2, 6, 3, 4), (4, 3, 3, 2)],
)
def test_low_bit_packing_rejects_unaligned_head_dimension(
    bits: int, width: int, group_size: int, divisor: int
) -> None:
    key = torch.randn(1, 2, 8, width)
    weights = torch.ones(1, 1, width)
    parameters = TemporalParameters(weights, torch.zeros(1, width), 4)
    with pytest.raises(ValueError, match=f"divisible by {divisor}"):
        encode_temporal(
            key,
            parameters,
            config(k_bits=bits, group_size=group_size),
        )


def test_per_head_temporal_parameters() -> None:
    weights = torch.zeros(2, 2, 2, 8)
    weights[0, 0, 0] = 0.9
    weights[0, 0, 1] = 0.5
    weights[1, 0] = 0.6
    weights[1, 1] = 0.2
    parameters = TemporalParameters(weights, torch.zeros(2, 2, 8), 4)
    key = torch.randn(1, 2, 11, 8)
    packed, reference = encode_temporal(key, parameters, config())
    decoded = decode_tensor(packed, output_dtype=torch.float32, use_triton=False)
    torch.testing.assert_close(decoded, reference, rtol=0, atol=0)
