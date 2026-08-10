from __future__ import annotations

import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_all.sh"


def _environment(tmp_path: Path, *, fail_name: str = "") -> tuple[dict[str, str], Path]:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "0_0.mp4").write_bytes(b"baseline")
    (candidate / "0_0.mp4").write_bytes(b"candidate")
    log = tmp_path / "calls.log"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$(basename "$1") $*" >> "$CALL_LOG"
if [[ -n "${FAIL_NAME:-}" && "$*" == *"--name $FAIL_NAME"* ]]; then
  exit 9
fi
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output|--config-out)
      destination="$2"
      mkdir -p "$(dirname "$destination")"
      printf '{"status":"complete"}\\n' > "$destination"
      shift 2
      ;;
    *) shift ;;
  esac
done
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(fake_python),
            "CALL_LOG": str(log),
            "BASELINE_DIR": str(baseline),
            "CANDIDATE_DIR": str(candidate),
            "CANDIDATE_NAME": "int2",
            "OUTPUT_DIR": str(tmp_path / "evaluation"),
            "DEVICE": "cuda:1",
            "FRAME_STRIDE": "3",
            "FID_BATCH_SIZE": "8",
            "FAIL_NAME": fail_name,
        }
    )
    return env, log


def test_shell_calls_quality_then_baseline_and_candidate_vbench(tmp_path) -> None:
    env, log = _environment(tmp_path)

    subprocess.run(["bash", str(SCRIPT)], env=env, check=True, capture_output=True)

    calls = log.read_text(encoding="utf-8").splitlines()
    assert [line.split()[0] for line in calls] == [
        "evaluate_quality.py",
        "evaluate_vbench.py",
        "evaluate_vbench.py",
    ]
    assert "--frame-stride 3" in calls[0]
    assert "--fid-batch-size 8" in calls[0]
    assert "--name bf16" in calls[1]
    assert "--name int2" in calls[2]
    assert "--reuse-complete" in calls[1]


def test_force_disables_reuse_and_vbench_failure_stops_the_shell(tmp_path) -> None:
    env, log = _environment(tmp_path, fail_name="int2")
    env["FORCE"] = "1"

    completed = subprocess.run(
        ["bash", str(SCRIPT)], env=env, check=False, capture_output=True, text=True
    )

    assert completed.returncode == 9
    assert "[evaluate_all] 完成" not in completed.stdout
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    assert all("--reuse-complete" not in line for line in calls)
