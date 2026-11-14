"""Regression checks for HIGGS state and the Self-Forcing cache layout."""
from dataclasses import replace

import pytest
import torch

from tempokv import CodecConfig, TemporalParameters, encode_temporal
from tempokv import higgs
from tempokv.codec import IncrementalTensorEncoder, decode_tensor, encode_cross_kv
from tempokv.fitting import _reconstruct_temporal_trace
from tempokv.parameters import CrossKVParameters


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_fused_higgs_packed_state_matches_reference(bits):
    if higgs.for_device(torch.device("cuda"), 512) is None:
        pytest.skip("HIGGS fusion requires Triton 3")
    generator = torch.Generator().manual_seed(78)
    value = torch.randn(2, 3, 5, 128, generator=generator).to(torch.bfloat16).cuda()
    value[:, :, 0] = 0
    value = value.transpose(1, 2)
    reference, ref_scales = higgs.encode(value, bits, 512, use_triton=False)
    codes, scales = higgs.encode(value, bits, 512)
    torch.testing.assert_close(scales, ref_scales)
    # Norm reductions can round differently near a centroid boundary.
    assert (codes != reference).float().mean().item() < 0.001
    expected = higgs.decode(reference, ref_scales, bits, 512, 3, 128, use_triton=False)
    torch.testing.assert_close(
        higgs.decode(reference, ref_scales, bits, 512, 3, 128), expected,
        rtol=1e-5, atol=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("quantizer,anchor", [("higgs", False), ("higgs", True), ("scalar", True)])
def test_higgs_cache_uses_persistent_prediction(quantizer, anchor, monkeypatch):
    if higgs.for_device(torch.device("cuda"), 256) is None:
        pytest.skip("HIGGS fusion requires Triton 3")
    from tempokv.triton_ops import route_report

    monkeypatch.setenv("TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH", "2")
    config = CodecConfig(
        fit_mode="native", stride=3, k_history_length=2, v_history_length=1,
        residual_quantizer=quantizer, k_bits=3, v_bits=5, anchor_bits=3,
        higgs_anchor=anchor, higgs_group_size=256, strict_triton=True,
    )
    assert config.use_triton
    generator = torch.Generator().manual_seed(45)
    value = torch.randn(2, 3, 20, 128, generator=generator).to(torch.bfloat16).cuda()
    key = TemporalParameters(torch.full((2, 2, 128), 0.25), torch.zeros(2, 128))
    cross = CrossKVParameters(
        torch.eye(128).expand(3, -1, -1) * 0.5, torch.zeros(3, 128),
        history_weights=torch.full((1, 3, 128), 0.25),
    )
    packed_k, expected_k = encode_temporal(value, key, config)
    packed_v, expected_v = encode_cross_kv(value * 0.5, expected_k, cross, config)
    # Use an offset and gaps in the output storage, as in the attention cache.
    target = torch.empty((2, 22, 3, 132), device=value.device, dtype=value.dtype)[:, 1:-1, :, :128]
    before = route_report()
    actual = decode_tensor(packed_k, output_tensor=target, strict_triton=True)
    assert actual.data_ptr() == target.data_ptr()
    torch.testing.assert_close(actual, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(
        decode_tensor(packed_v, source=actual, strict_triton=True), expected_v, rtol=0, atol=0,
    )
    after = route_report()
    for route in ("temporal_persistent_direct", "cross_kv_persistent"):
        assert after[route] == before.get(route, 0) + 1
