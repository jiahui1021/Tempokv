from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tempokv.parameters import CrossKVParameters, TemporalParameters, load_parameters


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fit_params.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_runs_from_checkout() -> None:
    completed = _run("--help")
    assert completed.returncode == 0
    assert "k_temporal.npz" not in completed.stderr
    assert "clean .npz trace" in completed.stdout
    assert "--history-length" in completed.stdout
    assert "--reconstruction-dtype" in completed.stdout


def test_small_trace_fit_writes_reloadable_artifacts(tmp_path) -> None:
    generator = np.random.default_rng(31)
    keys = generator.normal(size=(2, 40, 4, 2, 4)).astype(np.float32)
    matrix = np.asarray(
        [[0.6, 0.1, 0.0, -0.1], [0.0, 0.5, 0.2, 0.0], [0.1, 0.0, 0.7, 0.1], [0.0, -0.1, 0.2, 0.5]],
        dtype=np.float32,
    )
    values = np.einsum("lnthi,ij->lnthj", keys, matrix) + 0.15
    trace_path = tmp_path / "trace.npz"
    np.savez(trace_path, keys=keys, values=values, layer_ids=np.asarray([4, 9]))
    config_path = tmp_path / "codec.json"
    config_path.write_text(
        json.dumps(
            {
                "codec": {
                    "k_bits": 2,
                    "v_bits": 2,
                    "anchor_bits": 2,
                    "group_size": 2,
                    "stride": 5,
                    "residual_mode": "asymmetric",
                    "scale_dtype": "float32",
                    "reconstruction_dtype": "float32",
                    "use_triton": False,
                    "strict_triton": False,
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "fitted"

    completed = _run(
        str(trace_path),
        "--output-dir",
        str(output_dir),
        "--config",
        str(config_path),
        "--history-length",
        "2",
        "--iterations",
        "1",
        "--temporal-ridge",
        "0.001",
        "--cross-ridge",
        "0.001",
        "--max-gain",
        "1.0",
        "--k-bits",
        "4",
    )

    assert completed.returncode == 0, completed.stderr
    temporal = load_parameters(output_dir / "k_temporal.npz")
    cross = load_parameters(output_dir / "v_cross_kv.npz")
    assert isinstance(temporal, TemporalParameters)
    assert isinstance(cross, CrossKVParameters)
    assert temporal.history_weights.shape == (2, 2, 2, 2, 4)
    assert cross.weight.shape == (2, 2, 4, 4)
    assert temporal.layer_ids == cross.layer_ids == (4, 9)
    assert temporal.stride == cross.stride == 5

    report = json.loads((output_dir / "fit_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "fit_complete"
    assert report["input"]["shape"] == list(keys.shape)
    assert report["input"]["layer_ids"] == [4, 9]
    assert report["codec_config"]["k_bits"] == 4
    assert report["codec_config"]["anchor_bits"] == 2
    assert report["fit"]["history_length"] == 2
    assert report["artifacts"] == {
        "k_temporal": "k_temporal.npz",
        "v_cross_kv": "v_cross_kv.npz",
    }


def test_missing_trace_array_fails_without_outputs(tmp_path) -> None:
    trace_path = tmp_path / "missing_values.npz"
    np.savez(trace_path, keys=np.ones((1, 5, 1, 2), dtype=np.float32))
    output_dir = tmp_path / "output"

    completed = _run(
        str(trace_path),
        "--output-dir",
        str(output_dir),
        "--history-length",
        "2",
        "--group-size",
        "1",
    )

    assert completed.returncode != 0
    assert "missing arrays: values" in completed.stderr
    assert not output_dir.exists()


def test_invalid_config_field_is_rejected(tmp_path) -> None:
    trace_path = tmp_path / "trace.npz"
    arrays = np.ones((1, 5, 1, 2), dtype=np.float32)
    np.savez(trace_path, keys=arrays, values=arrays)
    config_path = tmp_path / "bad.json"
    config_path.write_text('{"unknown_setting": 1}', encoding="utf-8")

    completed = _run(str(trace_path), "--config", str(config_path))

    assert completed.returncode != 0
    assert "unknown CodecConfig fields: unknown_setting" in completed.stderr
