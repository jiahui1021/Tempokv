from __future__ import annotations

import json

import pytest
import torch

from tempokv import (
    CrossKVParameters,
    TemporalParameters,
    load_codec,
    load_codec_from_env,
)


def _files(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "k_bits": 2,
                "v_bits": 2,
                "anchor_bits": 4,
                "group_size": 4,
                "stride": 2,
                "residual_mode": "asymmetric",
                "scale_dtype": "float32",
                "reconstruction_dtype": "float32",
                "use_triton": False,
                "strict_triton": False,
            }
        ),
        encoding="utf-8",
    )
    k_path = tmp_path / "k.npz"
    TemporalParameters(
        torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 4), stride=2
    ).save(k_path)
    v_path = tmp_path / "v.npz"
    CrossKVParameters(
        torch.eye(4).reshape(1, 4, 4), torch.zeros(1, 4), stride=2
    ).save(v_path)
    return config, k_path, v_path


def test_load_codec_from_files(tmp_path) -> None:
    config, k_path, v_path = _files(tmp_path)
    codec = load_codec(config, k_path, v_path)
    keys = torch.randn(1, 1, 4, 4)
    values = torch.randn_like(keys)
    rebuilt = codec.decode(codec.encode(keys, values, layer=0))
    assert rebuilt[0].shape == keys.shape
    assert rebuilt[1].shape == values.shape


def test_load_codec_from_environment(tmp_path, monkeypatch) -> None:
    config, k_path, v_path = _files(tmp_path)
    monkeypatch.setenv("TEMPOKV_CONFIG_PATH", str(config))
    monkeypatch.setenv("TEMPOKV_K_PARAMS_PATH", str(k_path))
    monkeypatch.setenv("TEMPOKV_V_PARAMS_PATH", str(v_path))
    assert load_codec_from_env().config.stride == 2


def test_load_codec_from_environment_reports_missing(monkeypatch) -> None:
    for name in ("CONFIG_PATH", "K_PARAMS_PATH", "V_PARAMS_PATH"):
        monkeypatch.delenv(f"TEMPOKV_{name}", raising=False)
    with pytest.raises(RuntimeError, match="TEMPOKV_CONFIG_PATH"):
        load_codec_from_env()
