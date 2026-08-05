from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from tempokv import CrossKVParameters, TemporalParameters


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("tempokv_generate_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate = _load_script()


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    generator = np.random.default_rng(8)
    input_path = tmp_path / "cache.npz"
    keys = generator.normal(size=(2, 1, 2, 10, 4)).astype(np.float32)
    values = generator.normal(size=keys.shape).astype(np.float32)
    np.savez(input_path, keys=keys, values=values)
    k_path = tmp_path / "k.npz"
    weights = torch.zeros(2, 1, 1, 2, 4)
    weights[:, 0, 0] = 0.8
    TemporalParameters(
        weights, torch.zeros(2, 1, 2, 4), stride=4, layer_ids=(3, 7)
    ).save(k_path)
    v_path = tmp_path / "v.npz"
    CrossKVParameters(
        torch.eye(4).repeat(2, 2, 1, 1),
        torch.zeros(2, 2, 4),
        stride=4,
        layer_ids=(3, 7),
    ).save(v_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "k_bits": 2,
                "v_bits": 2,
                "anchor_bits": 4,
                "group_size": 4,
                "stride": 4,
                "residual_mode": "asymmetric",
                "scale_dtype": "float32",
                "reconstruction_dtype": "float32",
                "use_triton": False,
                "strict_triton": False,
            }
        ),
        encoding="utf-8",
    )
    return input_path, k_path, v_path, config_path


def test_generate_script_roundtrip_and_cli_override(tmp_path: Path) -> None:
    input_path, k_path, v_path, config_path = _artifacts(tmp_path)
    output = tmp_path / "output"
    report = generate.main(
        [
            str(input_path),
            str(output),
            "--config",
            str(config_path),
            "--k-params",
            str(k_path),
            "--v-params",
            str(v_path),
            "--k-bits",
            "4",
            "--device",
            "cpu",
        ]
    )

    assert report["status"] == "complete"
    assert report["config"]["k_bits"] == 4
    assert report["layer_ids"] == [3, 7]
    assert report["compression_ratio"] > 1.0
    with np.load(output / "reconstructed.npz") as rebuilt:
        assert rebuilt["keys"].shape == (2, 1, 2, 10, 4)
        assert rebuilt["values"].shape == (2, 1, 2, 10, 4)
        np.testing.assert_array_equal(rebuilt["layer_ids"], [3, 7])
    saved_report = json.loads((output / "report.json").read_text())
    assert saved_report["packed_cache_bytes"] == report["packed_cache_bytes"]


def test_generate_script_rejects_missing_cache_key(tmp_path: Path) -> None:
    _, k_path, v_path, config_path = _artifacts(tmp_path)
    bad_input = tmp_path / "bad.npz"
    np.savez(bad_input, keys=np.zeros((1, 1, 4, 4), dtype=np.float32))
    with pytest.raises(KeyError, match="values"):
        generate.main(
            [
                str(bad_input),
                str(tmp_path / "output"),
                "--config",
                str(config_path),
                "--k-params",
                str(k_path),
                "--v-params",
                str(v_path),
            ]
        )


def test_generate_help(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        generate.main(["--help"])
    assert error.value.code == 0
    assert "reconstructed caches" in capsys.readouterr().out
