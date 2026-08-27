#!/usr/bin/env python3
"""Self-Forcing latency adapter.

Runs generate.py in --benchmark --profile mode for the BF16 / TempoKV_INT2 /
TempoKV_INT4 methods, then converts the pipeline_report + profile into the
harness's canonical report via report_adapter.py --format self-forcing-profile.

The harness always supplies one runtime_instance_id per warmup/measure pair;
this adapter reuses that id for both phases (the pilot contract accepts that
the loaded instance is not proven to be the same process).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ADAPTER = PROJECT_ROOT / "scripts/paper_experiments/latency/report_adapter.py"
GENERATE = PROJECT_ROOT / "experiments/Self-Forcing/tempokv_adapter/generate.py"

# Default parameter directories per method (relative to project root).
PARAMS = {
    "BF16": None,
    "TempoKV_INT2": "data/self-forcing/params/legacy-kh2-vh1-int2-layershared",
    "TempoKV_INT4": "data/self-forcing/params/legacy-kh2-vh1-int4-layershared",
}
CONFIG = PROJECT_ROOT / "configs/legacy.json"


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing harness env {name}")
    return value


def _compilation_report(context: dict[str, str]) -> None:
    report = {
        "schema_version": "tempokv.compilation-run.v1",
        "method": context["method"],
        "one_time_compilation_s": 0.0,
        "session_id": context["session_id"],
        "session_scope": context["session_scope"],
        "frozen": json.loads(context["frozen"]),
    }
    out = Path(_env("PAPER_EXPERIMENT_REPORT"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _run_generation(context: dict[str, str], frames: int, output: Path) -> None:
    method = context["method"]
    env = os.environ.copy()
    wan_dir = os.environ.get("WAN_MODEL_DIR")
    if not wan_dir:
        raise RuntimeError("missing env WAN_MODEL_DIR")
    env["FORCING_WAN_MODEL_DIR"] = str(Path(wan_dir).expanduser())
    command = [
        sys.executable,
        str(GENERATE),
        "--mode", "bf16" if method == "BF16" else "tempokv",
        "--checkpoint", str(Path(_env("CHECKPOINT")).expanduser()),
        "--prompts", str(Path(_env("PROMPTS")).expanduser()),
        "--output-dir", str(output),
        "--frames", str(frames),
        "--samples", "1",
        "--seed", context["seed"],
        "--start-index", "0",
        "--end-index", "30",
        "--num-gpus", "1",
        "--benchmark",
        "--profile",
        "--triton-cache", str(output / "triton_cache"),
    ]
    if method != "BF16":
        params = PARAMS[method]
        if params is None:
            raise RuntimeError(f"no param dir for {method}")
        command += [
            "--config", str(CONFIG),
            "--k-params", str(PROJECT_ROOT / params / "k_temporal.npz"),
            "--v-params", str(PROJECT_ROOT / params / "v_cross_kv.npz"),
        ]
    print("[adapter] " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"generate.py failed for {method} frames={frames}: exit {result.returncode}")


def _timing_report(context: dict[str, str], frames: int, run_dir: Path) -> None:
    pipeline = run_dir / "pipeline_report.json"
    profile = run_dir / "profile" / "rank-0.json"
    if not pipeline.is_file() or not profile.is_file():
        raise RuntimeError(f"missing pipeline_report or profile for {context['method']} frames={frames}")
    command = [
        sys.executable,
        str(ADAPTER),
        "--format", "self-forcing-profile",
        "--pipeline-report", str(pipeline),
        "--profile", str(profile),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode:
        raise RuntimeError(f"report_adapter failed: exit {result.returncode}")


def main() -> int:
    phase = _env("PAPER_EXPERIMENT_PHASE")
    frames = int(_env("PAPER_EXPERIMENT_FRAMES"))
    context = {
        "method": _env("PAPER_EXPERIMENT_METHOD"),
        "phase": phase,
        "frames": str(frames),
        "repeat": _env("PAPER_EXPERIMENT_REPEAT"),
        "seed": _env("PAPER_EXPERIMENT_SEED"),
        "session_id": _env("PAPER_EXPERIMENT_SESSION_ID"),
        "session_scope": _env("PAPER_EXPERIMENT_SESSION_SCOPE"),
        "frozen": _env("PAPER_EXPERIMENT_FROZEN_JSON"),
    }
    if phase == "compilation":
        _compilation_report(context)
        return 0
    output = Path(_env("PAPER_EXPERIMENT_OUTPUT"))
    run_dir = output / "runs" / context["method"] / f"frames-{frames}" / context["repeat"]
    run_dir.mkdir(parents=True, exist_ok=True)
    _run_generation(context, frames, run_dir)
    _timing_report(context, frames, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
