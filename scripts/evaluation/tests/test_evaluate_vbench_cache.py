from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_vbench.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("tempokv_evaluate_vbench_cache", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vbench = _load_script()


def _inputs(tmp_path: Path, names: tuple[str, ...] = ("0_0.mp4",)) -> tuple[Path, Path]:
    videos = tmp_path / "videos"
    videos.mkdir()
    prompts = {}
    for index, name in enumerate(names):
        (videos / name).write_bytes(f"video-{index}".encode())
        prompts[name] = f"prompt {index}"
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps(prompts), encoding="utf-8")
    return videos, prompt_file


def test_completed_fingerprint_is_reused_and_video_change_invalidates(
    tmp_path, monkeypatch
) -> None:
    videos, prompt_file = _inputs(tmp_path)
    video = videos / "0_0.mp4"
    report = tmp_path / "evaluation.json"
    calls = []

    def fake_run(config, prompts):
        calls.append((config.copy(), prompts.copy()))
        return list(config["dimensions"])

    monkeypatch.setattr(vbench, "run", fake_run)
    argv = [
        str(videos),
        "--output-dir",
        str(tmp_path / "results"),
        "--prompt-file",
        str(prompt_file),
        "--config-out",
        str(report),
        "--reuse-complete",
    ]

    first = vbench.main(argv)
    second = vbench.main(argv)
    video.write_bytes(b"changed-video")
    third = vbench.main(argv)

    assert first["status"] == "complete"
    assert first["evaluator"] == "vbench_long"
    assert second["status"] == "reused"
    assert third["status"] == "complete"
    assert len(calls) == 2
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["status"] == "complete"
    assert saved["input_fingerprint"] == third["input_fingerprint"]


def test_custom_mode_requires_complete_prompt_mapping_and_uses_eight_metrics(tmp_path) -> None:
    videos, prompt_file = _inputs(tmp_path)
    common = [str(videos), "--output-dir", str(tmp_path / "results")]

    with pytest.raises(ValueError, match="--prompt-file is required"):
        vbench.prepare(vbench._parser().parse_args(common))

    prompt_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing video filename"):
        vbench.prepare(
            vbench._parser().parse_args(common + ["--prompt-file", str(prompt_file)])
        )

    prompt_file.write_text('{"0_0.mp4": "prompt"}', encoding="utf-8")
    config, _ = vbench.prepare(
        vbench._parser().parse_args(common + ["--prompt-file", str(prompt_file)])
    )
    assert config["dimensions"] == list(vbench.CUSTOM_INPUT_DIMENSIONS)
    assert config["dimensions"][-2:] == ["overall_consistency", "clip_score"]
    assert config["display_columns"][-2:] == ["visual_quality", "text_alignment"]

    with pytest.raises(ValueError, match="does not support temporal_flickering"):
        vbench.prepare(
            vbench._parser().parse_args(
                common
                + [
                    "--prompt-file",
                    str(prompt_file),
                    "--dimensions",
                    "subject_consistency,temporal_flickering",
                ]
            )
        )

    prompt_file.write_text(
        '{"0_0.mp4":"first","0_0.mp4":"second"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate filename"):
        vbench.prepare(
            vbench._parser().parse_args(common + ["--prompt-file", str(prompt_file)])
        )


def test_legacy_modes_are_rejected(tmp_path) -> None:
    videos, prompt_file = _inputs(tmp_path)
    with pytest.raises(SystemExit):
        vbench._parser().parse_args(
            [
                str(videos),
                "--output-dir",
                str(tmp_path / "results"),
                "--prompt-file",
                str(prompt_file),
                "--mode",
                "custom_input",
            ]
        )


def test_long_standard_mode_requires_full_info_and_remains_available(tmp_path) -> None:
    videos, _ = _inputs(tmp_path)
    common = [
        str(videos),
        "--output-dir",
        str(tmp_path / "results"),
        "--mode",
        "long_vbench_standard",
    ]
    with pytest.raises(ValueError, match="--full-info is required"):
        vbench.prepare(vbench._parser().parse_args(common))

    full_info = tmp_path / "VBench_full_info.json"
    full_info.write_text("[]", encoding="utf-8")
    config, prompts = vbench.prepare(
        vbench._parser().parse_args(common + ["--full-info", str(full_info)])
    )

    assert config["mode"] == "long_vbench_standard"
    assert config["dimensions"] == list(vbench.DEFAULT_STANDARD_DIMENSIONS)
    assert prompts == {}


def test_vbench_long_stages_links_and_runs_each_dimension(tmp_path, monkeypatch) -> None:
    videos, prompt_file = _inputs(tmp_path, ("0_0.mp4", "10_0.mp4"))
    output = tmp_path / "results"
    cache = tmp_path / "cache"
    cache.mkdir()
    package = tmp_path / "vbench2_beta_long"
    configs = package / "configs"
    configs.mkdir(parents=True)
    for filename in (
        "clip_length_mix.yaml",
        "slow_fast_params.yaml",
        "subject_mapping_table.yaml",
        "background_mapping_table.yaml",
    ):
        (configs / filename).write_text("{}", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "VBench_full_info.json").write_text("[]", encoding="utf-8")
    calls = []
    competition_calls = []

    class FakeVBenchLong:
        def __init__(self, device, full_info, output_path):
            self.output_path = Path(output_path)
            assert device == "fake-device"
            assert Path(full_info).name == "VBench_full_info.json"

        def evaluate(self, **kwargs):
            calls.append(kwargs)
            dimension = kwargs["dimension_list"][0]
            result = self.output_path / f"{kwargs['name']}_eval_results.json"
            result.write_text(
                json.dumps({dimension: [0.75, {"video": 0.5}]}),
                encoding="utf-8",
            )

    fake_module = SimpleNamespace(
        VBenchLong=FakeVBenchLong, __file__=str(package / "__init__.py")
    )
    competition_package = tmp_path / "competitions"
    competition_configs = competition_package / "configs"
    competition_configs.mkdir(parents=True)
    (competition_package / "__init__.py").write_text("", encoding="utf-8")

    class FakeVBenchCompetition:
        def __init__(self, device, _full_info, output_path):
            self.output_path = Path(output_path)
            assert device == "fake-device"

        def evaluate_long(self, **kwargs):
            competition_calls.append(kwargs)
            result = self.output_path / f"{kwargs['name']}_eval_results.json"
            result.write_text(
                json.dumps(
                    {
                        "text_alignment": [
                            0.75,
                            {
                                "overall_consistency": [0.75, {"video": 0.5}],
                                "clip_score": [0.75, {"video": 0.5}],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

    fake_competitions = SimpleNamespace(
        VBenchCompetition=FakeVBenchCompetition,
        __file__=str(competition_package / "__init__.py"),
    )
    monkeypatch.setenv("VBENCH_CACHE_DIR", str(cache))
    monkeypatch.setattr(vbench, "_device", lambda name: "fake-device")
    monkeypatch.setattr(
        vbench.importlib,
        "import_module",
        lambda name: fake_competitions if name == "competitions" else fake_module,
    )
    before = sorted(path.name for path in videos.iterdir())
    config, prompts = vbench.prepare(
        vbench._parser().parse_args(
            [
                str(videos),
                "--output-dir",
                str(output),
                "--prompt-file",
                str(prompt_file),
            ]
        )
    )

    completed = vbench.run(config, prompts)

    assert completed == list(vbench.CUSTOM_INPUT_DIMENSIONS)
    assert sorted(path.name for path in videos.iterdir()) == before
    staging = output / vbench.STAGING_DIR_NAME
    assert sorted(path.name for path in staging.iterdir()) == [
        "0001_input-0.mp4",
        "0002_input-0.mp4",
    ]
    assert all(path.is_symlink() for path in staging.iterdir())
    assert [call["dimension_list"] for call in calls] == [
        [dimension] for dimension in vbench.VISUAL_DIMENSIONS
    ]
    assert [call["name"] for call in calls] == [
        f"videos_{dimension}" for dimension in vbench.VISUAL_DIMENSIONS
    ]
    assert [call["dimension_list"] for call in competition_calls] == [["text_alignment"]]
    assert all(call["local"] is False for call in calls)
    assert all(call["mode"] == "long_custom_input" for call in calls)
    assert all(call["prompt_list"] == ["prompt 0", "prompt 1"] for call in calls)
    assert all(call["use_semantic_splitting"] is False for call in calls)
    assert all(call["imaging_quality_preprocessing_mode"] == "longer" for call in calls)
    summary = json.loads(
        (output / "videos_vbench_long_summary.json").read_text(encoding="utf-8")
    )
    assert summary["protocol"] == "vbench_long_custom8_derived"
    assert summary["display_columns"] == list(vbench.DISPLAY_COLUMNS)
    assert summary["display_scores"]["visual_quality"] == pytest.approx(0.75)
    assert summary["display_scores"]["text_alignment"] == pytest.approx(0.75)
    assert summary["official_vbench_total"] is False


def test_local_checkpoints_are_fingerprinted_and_forwarded(tmp_path, monkeypatch) -> None:
    videos, prompt_file = _inputs(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    package = tmp_path / "vbench2_beta_long"
    configs = package / "configs"
    configs.mkdir(parents=True)
    for filename in (
        "clip_length_mix.yaml",
        "slow_fast_params.yaml",
        "subject_mapping_table.yaml",
        "background_mapping_table.yaml",
    ):
        (configs / filename).write_text("{}", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "VBench_full_info.json").write_text("[]", encoding="utf-8")
    calls = []

    class FakeVBenchLong:
        def __init__(self, _device, _full_info, output_path):
            self.output_path = Path(output_path)

        def evaluate(self, **kwargs):
            calls.append(kwargs)
            result = self.output_path / f"{kwargs['name']}_eval_results.json"
            dimension = kwargs["dimension_list"][0]
            result.write_text(json.dumps({dimension: [0.5]}), encoding="utf-8")

    fake_module = SimpleNamespace(
        VBenchLong=FakeVBenchLong, __file__=str(package / "__init__.py")
    )
    monkeypatch.setenv("VBENCH_CACHE_DIR", str(cache))
    monkeypatch.setattr(vbench, "_device", lambda _name: "fake-device")
    monkeypatch.setattr(vbench, "_install_torchvision_write_video_compat", lambda: None)
    monkeypatch.setattr(vbench.importlib, "import_module", lambda _name: fake_module)
    config, prompts = vbench.prepare(
        vbench._parser().parse_args(
            [
                str(videos),
                "--output-dir",
                str(tmp_path / "results"),
                "--prompt-file",
                str(prompt_file),
                "--dimensions",
                "subject_consistency",
                "--local-checkpoints",
            ]
        )
    )

    assert config["local_checkpoints"] is True
    vbench.run(config, prompts)
    assert calls[0]["local"] is True


def test_failure_is_recorded_and_not_reusable(tmp_path, monkeypatch) -> None:
    videos, prompt_file = _inputs(tmp_path)
    report = tmp_path / "evaluation.json"

    def fail(config, prompts):
        raise RuntimeError("dimension failed")

    monkeypatch.setattr(vbench, "run", fail)
    with pytest.raises(RuntimeError, match="dimension failed"):
        vbench.main(
            [
                str(videos),
                "--output-dir",
                str(tmp_path / "results"),
                "--prompt-file",
                str(prompt_file),
                "--config-out",
                str(report),
            ]
        )

    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["error"]["type"] == "RuntimeError"


def test_non_finite_dimension_result_is_rejected(tmp_path) -> None:
    result = tmp_path / "result.json"
    result.write_text('{"subject_consistency":[NaN]}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing or non-finite"):
        vbench._validate_result(result, "subject_consistency")


def test_summary_matches_eight_column_paper_protocol(tmp_path) -> None:
    scores = {
        dimension: float(index)
        for index, dimension in enumerate(vbench.CUSTOM_INPUT_DIMENSIONS, start=1)
    }

    path = vbench._write_summary(tmp_path, "candidate", scores)
    summary = json.loads(path.read_text(encoding="utf-8"))

    assert summary["display_scores"]["visual_quality"] == pytest.approx(3.5)
    assert summary["display_scores"]["text_alignment"] == pytest.approx(7.5)
    assert len(summary["display_scores"]) == 8
