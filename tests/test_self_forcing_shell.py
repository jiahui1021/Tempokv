from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "self-forcing"


def _run(script: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(DIRECTORY / script)],
        cwd=ROOT,
        env={**os.environ, **environment},
        text=True,
        capture_output=True,
        check=False,
    )


def test_shell_scripts_have_valid_syntax() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(DIRECTORY / "prepare.sh"), str(DIRECTORY / "generate.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_prepare_shell_dry_run_orders_capture_merge_and_fit(tmp_path: Path) -> None:
    completed = _run(
        "prepare.sh",
        {
            "TRACE_RUNNER": str(tmp_path / "capture.sh"),
            "CHECKPOINT": str(tmp_path / "checkpoint.pt"),
            "WAN_MODEL_DIR": str(tmp_path / "wan"),
            "PROMPTS": str(tmp_path / "prompts.txt"),
            "WORK_DIR": str(tmp_path / "prepared"),
            "CONFIG": str(ROOT / "configs" / "int2.json"),
            "DRY_RUN": "1",
        },
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 3
    assert "TRACE_QUANT_TYPE=none" in lines[0]
    assert "prepare_trace.py" in lines[1]
    assert "fit_params.py" in lines[2]
    assert "generate.py" not in completed.stdout


def test_generate_shell_only_passes_existing_parameters(tmp_path: Path) -> None:
    completed = _run(
        "generate.sh",
        {
            "PYTHON_BIN": "/bin/echo",
            "MODEL_REPO": str(tmp_path / "model"),
            "CHECKPOINT": str(tmp_path / "checkpoint.pt"),
            "WAN_MODEL_DIR": str(tmp_path / "wan"),
            "PROMPTS": str(tmp_path / "prompts.txt"),
            "PARAM_DIR": str(tmp_path / "parameters"),
            "OUTPUT_DIR": str(tmp_path / "videos"),
            "CONFIG": str(ROOT / "configs" / "int2.json"),
            "DRY_RUN": "1",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "generate.py" in completed.stdout
    assert "k_temporal.npz" in completed.stdout
    assert "v_cross_kv.npz" in completed.stdout
    assert "--dry-run" in completed.stdout
    assert "fit_params.py" not in completed.stdout
    assert "prepare_trace.py" not in completed.stdout
