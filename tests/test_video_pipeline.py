from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_video_pipeline.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    generator = np.random.default_rng(17)
    keys = generator.normal(size=(1, 12, 3, 1, 4)).astype(np.float32)
    values = (keys * 0.75 + 0.1).astype(np.float32)
    trace = tmp_path / "trace.npz"
    np.savez(trace, keys=keys, values=values)
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
    model = tmp_path / "model_generate.py"
    model.write_text(
        """from pathlib import Path
import os
import torch
from tempokv import load_codec_from_env

codec = load_codec_from_env()
k = torch.randn(1, 1, 4, 4)
v = torch.randn_like(k)
codec.decode(codec.encode(k, v, layer=0))
output = Path(os.environ[\"TEMPOKV_VIDEO_OUTPUT_DIR\"])
output.mkdir(parents=True, exist_ok=True)
(output / \"sample.mp4\").write_bytes(b\"video-placeholder\")
""",
        encoding="utf-8",
    )
    return trace, config, model


def test_pipeline_fits_and_launches_integrated_generator(tmp_path) -> None:
    trace, config, model = _inputs(tmp_path)
    output = tmp_path / "run"
    completed = _run(
        "--trace",
        str(trace),
        "--config",
        str(config),
        "--output-dir",
        str(output),
        "--history-length",
        "1",
        "--iterations",
        "1",
        "--",
        sys.executable,
        str(model),
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "pipeline_report.json").read_text())
    assert report["status"] == "complete"
    assert len(report["videos"]) == 1
    assert (output / "parameters" / "k_temporal.npz").is_file()
    assert (output / "parameters" / "v_cross_kv.npz").is_file()


def test_pipeline_dry_run_does_not_create_output(tmp_path) -> None:
    trace, config, model = _inputs(tmp_path)
    output = tmp_path / "run"
    completed = _run(
        "--trace",
        str(trace),
        "--config",
        str(config),
        "--output-dir",
        str(output),
        "--dry-run",
        "--",
        sys.executable,
        str(model),
        "--output",
        "{video_dir}",
    )
    assert completed.returncode == 0, completed.stderr
    assert '"status": "dry_run"' in completed.stdout
    assert str(output / "videos") in completed.stdout
    assert not output.exists()


def test_pipeline_requires_generator_command(tmp_path) -> None:
    trace, config, _ = _inputs(tmp_path)
    completed = _run(
        "--trace",
        str(trace),
        "--config",
        str(config),
        "--output-dir",
        str(tmp_path / "run"),
    )
    assert completed.returncode != 0
    assert "generator command is required" in completed.stderr
