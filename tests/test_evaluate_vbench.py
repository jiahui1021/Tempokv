from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_vbench.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("tempokv_evaluate_vbench", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vbench = _load_script()


def _videos(tmp_path: Path) -> Path:
    directory = tmp_path / "sample-videos"
    directory.mkdir()
    (directory / "first.mp4").write_bytes(b"video-one")
    (directory / "second.gif").write_bytes(b"video-two")
    return directory


def test_dry_run_validates_and_writes_config_without_importing_vbench(
    tmp_path, monkeypatch
) -> None:
    videos = _videos(tmp_path)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(
        json.dumps({"first.mp4": "first prompt", "second.gif": "second prompt"}),
        encoding="utf-8",
    )
    config_path = tmp_path / "prepared" / "vbench.json"
    monkeypatch.setattr(
        vbench, "run", lambda *_: pytest.fail("dry-run must not invoke VBench")
    )

    report = vbench.main(
        [
            str(videos),
            "--output-dir",
            str(tmp_path / "results"),
            "--dimensions",
            "motion_smoothness,imaging_quality",
            "--prompt-file",
            str(prompts),
            "--config-out",
            str(config_path),
            "--dry-run",
        ]
    )

    assert report["status"] == "dry_run"
    assert report["video_count"] == 2
    assert report["name"] == "sample-videos"
    assert report["dimensions"] == ["motion_smoothness", "imaging_quality"]
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["videos_path"] == str(videos.resolve())
    assert saved["prompt_file"] == str(prompts.resolve())


def test_run_calls_official_package_interface(tmp_path, monkeypatch) -> None:
    videos = _videos(tmp_path)
    args = vbench._parser().parse_args(
        [str(videos), "--output-dir", str(tmp_path / "results"), "--device", "cpu"]
    )
    config, prompts = vbench.prepare(args)
    calls = {}

    class FakeVBench:
        def __init__(self, device, full_info, output):
            calls["init"] = (device, full_info, output)

        def evaluate(self, **kwargs):
            calls["evaluate"] = kwargs
            return {"ok": True}

    real_import = vbench.importlib.import_module

    def fake_import(name):
        if name == "vbench":
            return SimpleNamespace(VBench=FakeVBench)
        return real_import(name)

    monkeypatch.setattr(vbench.importlib, "import_module", fake_import)
    monkeypatch.setattr(vbench, "_device", lambda name: "cpu-device")

    result = vbench.run(config, prompts)

    assert result == {"ok": True}
    assert calls["init"][0] == "cpu-device"
    assert calls["evaluate"]["videos_path"] == str(videos.resolve())
    assert calls["evaluate"]["dimension_list"] == list(vbench.DEFAULT_DIMENSIONS)
    assert calls["evaluate"]["mode"] == "custom_input"


def test_prompt_file_must_cover_each_video(tmp_path) -> None:
    videos = _videos(tmp_path)
    prompts = tmp_path / "prompts.json"
    prompts.write_text('{"first.mp4": "only one"}', encoding="utf-8")
    args = vbench._parser().parse_args(
        [
            str(videos),
            "--output-dir",
            str(tmp_path / "results"),
            "--prompt-file",
            str(prompts),
        ]
    )

    with pytest.raises(ValueError, match="missing video filename.*second.gif"):
        vbench.prepare(args)


def test_standard_mode_requires_full_info(tmp_path) -> None:
    videos = _videos(tmp_path)
    args = vbench._parser().parse_args(
        [
            str(videos),
            "--output-dir",
            str(tmp_path / "results"),
            "--mode",
            "vbench_standard",
        ]
    )

    with pytest.raises(ValueError, match="--full-info is required"):
        vbench.prepare(args)


def test_missing_official_package_has_dry_run_hint(tmp_path, monkeypatch) -> None:
    videos = _videos(tmp_path)
    args = vbench._parser().parse_args(
        [str(videos), "--output-dir", str(tmp_path / "results")]
    )
    config, prompts = vbench.prepare(args)
    monkeypatch.setattr(
        vbench.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )

    with pytest.raises(RuntimeError, match="use --dry-run"):
        vbench.run(config, prompts)
