"""The previous-frame ablation performs no parameter IO or predictor calls."""
from dataclasses import asdict, replace
import json

import pytest
import torch

import tempokv.codec as implementation
from tempokv.codec import TempoKVCodec, decode_tensor
from tempokv.config import CodecConfig
from tempokv.fitting import build_fixed_temporal_parameters
from tempokv.integration import load_codec, load_codec_from_env, previous_frame_codec
from tempokv.parameters import PreviousFrameParameters, TemporalParameters


def config():
    return CodecConfig(k_bits=2, v_bits=2, anchor_bits=4, group_size=4,
                       stride=3, k_predictor_mode="identity", v_predictor_mode="identity",
                       k_history_length=1, v_history_length=1, use_triton=False)


def forbidden(*args, **kwargs):
    raise AssertionError("previous-frame copy must not read or evaluate predictor parameters")


def test_previous_frame_loads_without_parameter_files(tmp_path, monkeypatch):
    path = tmp_path / "codec.json"
    path.write_text(json.dumps(asdict(config())))
    monkeypatch.setattr(TemporalParameters, "load", forbidden)
    monkeypatch.setattr("tempokv.integration.load_parameters", forbidden)
    monkeypatch.setenv("TEMPOKV_CONFIG_PATH", str(path))
    monkeypatch.delenv("TEMPOKV_K_PARAMS_PATH", raising=False)
    monkeypatch.delenv("TEMPOKV_V_PARAMS_PATH", raising=False)
    for codec in [load_codec(path), load_codec_from_env()]:
        assert isinstance(codec.temporal, PreviousFrameParameters)
        assert isinstance(codec.v_parameters, PreviousFrameParameters)
        for rule in [codec.temporal, codec.v_parameters]:
            assert rule.history_length == 1
            assert rule.history_weights.numel() == rule.history_bias.numel() == 0
            assert rule.to("cpu") is rule
            with pytest.raises(ValueError, match="no parameter file"):
                rule.save(tmp_path / "not-created.npz")


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_previous_frame_batch_and_incremental_never_call_predictor(monkeypatch, dtype):
    torch.manual_seed(19)
    cfg = replace(config(), reconstruction_dtype=dtype)
    key = torch.randn(1, 2, 21, 8).bfloat16()
    value = torch.randn_like(key)
    fixed = build_fixed_temporal_parameters("identity", history_length=1, dim=8, stride=3)
    old = TempoKVCodec(fixed, fixed, cfg)
    old_packed = old.encode(key.clone(), value.clone())
    expected_key = decode_tensor(old_packed.key, use_triton=False)
    expected_value = decode_tensor(old_packed.value, use_triton=False)
    monkeypatch.setattr(implementation, "predict_temporal", forbidden)
    monkeypatch.setattr(implementation, "predict_cross_kv", forbidden)
    codec = previous_frame_codec(cfg)
    packed = codec.encode(key.clone(), value.clone())
    assert torch.equal(decode_tensor(packed.key, use_triton=False), expected_key)
    assert torch.equal(decode_tensor(packed.value, use_triton=False), expected_value)
    key_encoder, rebuilt_key = codec.begin_incremental_primary(key[:, :, :3].clone(), capacity_units=7)
    value_encoder = codec.begin_incremental_secondary(value[:, :, :3].clone(), rebuilt_key, capacity_units=7)
    for start, stop in [(3, 12), (12, 21)]:
        rebuilt_key = key_encoder.append(key[:, :, start:stop].clone())
        value_encoder.append(value[:, :, start:stop].clone(), source=rebuilt_key)
    assert torch.equal(decode_tensor(key_encoder.packed, use_triton=False), expected_key)
    assert torch.equal(decode_tensor(value_encoder.packed, use_triton=False), expected_value)


def test_previous_frame_still_rejects_wrong_codec_signature():
    codec = previous_frame_codec(config())
    with pytest.raises(ValueError, match="fitted for codec"):
        TempoKVCodec(codec.temporal, codec.v_parameters, replace(config(), k_bits=4))
