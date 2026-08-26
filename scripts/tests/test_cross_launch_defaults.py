from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
UNITS_ENV = "TEMPOKV_PERSISTENT_UNITS_PER_LAUNCH"


def _environment_probe(tmp_path: Path) -> Path:
    probe = tmp_path / "python-probe"
    probe.write_text(
        f"""#!{sys.executable}
import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == "-":
    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
print("{UNITS_ENV}=" + os.environ.get("{UNITS_ENV}", "<unset>"))
""",
        encoding="utf-8",
    )
    probe.chmod(0o755)
    return probe


def _run(script: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    inherited = {key: value for key, value in os.environ.items() if key != UNITS_ENV}
    return subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env={**inherited, **environment},
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("model", ("Self-Forcing", "Causal-Forcing", "LongCat"))
def test_cross_generation_defaults_to_eight_persistent_units(
    model: str, tmp_path: Path
) -> None:
    probe = _environment_probe(tmp_path)
    common = {
        "MODE": "tempokv",
        "CONFIG": str(ROOT / "configs" / "legacy.json"),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "PYTHON_BIN": str(probe),
    }
    if model == "Self-Forcing":
        environment = {
            **common,
            "CHECKPOINT": str(tmp_path / "checkpoint.pt"),
            "WAN_MODEL_DIR": str(tmp_path / "wan"),
            "PROMPTS": str(tmp_path / "prompts.txt"),
            "DRY_RUN": "1",
        }
    elif model == "Causal-Forcing":
        environment = {
            **common,
            "CHECKPOINT": str(tmp_path / "checkpoint.pt"),
            "MODEL_ROOT": str(tmp_path / "model"),
            "PROMPTS": str(tmp_path / "prompts.txt"),
            "DRY_RUN": "1",
        }
    else:
        environment = {
            **common,
            "CHECKPOINT_DIR": str(tmp_path),
            "PREFLIGHT": "0",
            "TORCHRUN_BIN": str(probe),
        }

    completed = _run(ROOT / "scripts" / model / "generate.sh", environment)

    assert completed.returncode == 0, completed.stderr
    assert f"{UNITS_ENV}=8" in completed.stdout


def test_explicit_cross_launch_size_is_preserved(tmp_path: Path) -> None:
    probe = _environment_probe(tmp_path)
    completed = _run(
        ROOT / "scripts" / "Self-Forcing" / "generate.sh",
        {
            "MODE": "tempokv",
            "CONFIG": str(ROOT / "configs" / "legacy.json"),
            "OUTPUT_DIR": str(tmp_path / "output"),
            "PYTHON_BIN": str(probe),
            "CHECKPOINT": str(tmp_path / "checkpoint.pt"),
            "WAN_MODEL_DIR": str(tmp_path / "wan"),
            "PROMPTS": str(tmp_path / "prompts.txt"),
            "DRY_RUN": "1",
            UNITS_ENV: "4",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert f"{UNITS_ENV}=4" in completed.stdout
