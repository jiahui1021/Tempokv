from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_vbench.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("tempokv_evaluate_vbench_cache", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vbench = _load_script()


def test_completed_fingerprint_is_reused_and_video_change_invalidates(
    tmp_path, monkeypatch
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    video = videos / "0_0.mp4"
    video.write_bytes(b"video")
    report = tmp_path / "evaluation.json"
    calls = []
    monkeypatch.setattr(vbench, "run", lambda config, prompts: calls.append(config.copy()))
    argv = [
        str(videos),
        "--output-dir",
        str(tmp_path / "results"),
        "--config-out",
        str(report),
        "--reuse-complete",
    ]

    first = vbench.main(argv)
    second = vbench.main(argv)
    video.write_bytes(b"changed-video")
    third = vbench.main(argv)

    assert first["status"] == "complete"
    assert second["status"] == "reused"
    assert third["status"] == "complete"
    assert len(calls) == 2
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["status"] == "complete"
    assert saved["input_fingerprint"] == third["input_fingerprint"]
