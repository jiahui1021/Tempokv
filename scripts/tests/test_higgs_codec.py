"""Regression checks for HIGGS state and the Self-Forcing cache layout."""
from dataclasses import replace

import pytest
import torch

from tempokv import CodecConfig, TemporalParameters, encode_temporal
from tempokv.codec import IncrementalTensorEncoder, decode_tensor
from tempokv.fitting import _reconstruct_temporal_trace


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("higgs_anchor", [False, True])
def test_higgs_fitting_replay_and_incremental_decode_agree(bits, higgs_anchor):
    generator = torch.Generator().manual_seed(944)
    settings = CodecConfig(
        k_bits=bits, v_bits=bits, k_history_length=1, stride=2,
        fit_mode="legacy", residual_quantizer="higgs",
        higgs_anchor=higgs_anchor, higgs_group_size=128, anchor_bits=bits if higgs_anchor else 4,
    )
    predictor = TemporalParameters(
        torch.full((1, 1, 64), 0.5), torch.zeros(1, 64), stride=2
    )
    # Three heads exercise padding, and the zero row exercises scale == 0.
    key = torch.randn(2, 3, 8, 64, generator=generator).to(torch.bfloat16)
    key[:, :, :2] = 0
    packed, expected = encode_temporal(key, predictor, settings)
    trace = key.transpose(1, 2).reshape(1, 2, 4, 2, 3, 64).transpose(2, 3)
    replay = _reconstruct_temporal_trace(trace, predictor, settings).to(expected.dtype)
    replay = replay.transpose(2, 3).reshape(2, 8, 3, 64).transpose(1, 2)
    assert torch.equal(replay, expected)
    assert not expected[:, :, :2].any()
    prefix, reconstructed = encode_temporal(key[:, :, :2], predictor, settings)
    encoder = IncrementalTensorEncoder(prefix, reconstructed, settings, capacity_units=5)
    for start in (2, 4, 6):
        encoder.append(key[:, :, start:start + 2])
        # The final append can leave the reserved capacity larger than the prefix.
        output = torch.empty(2, start + 2, 3, 64, dtype=key.dtype)
        rebuilt = decode_tensor(encoder.packed, output_tensor=output)
        assert torch.equal(rebuilt, expected[:, :, :start + 2])
    assert torch.equal(packed.residual_quant, encoder.packed.residual_quant[:, :, :6])
    assert encoder.packed.logical_storage_bytes() == packed.storage_bytes()
    assert packed.residual_zero_point is None
    assert packed.residual_scale.dtype == torch.float32

    scalar = replace(settings, residual_quantizer="scalar", higgs_anchor=False)
    assert settings.format_signature() != scalar.format_signature()
    assert scalar.format_signature() == replace(scalar, higgs_group_size=256).format_signature()
