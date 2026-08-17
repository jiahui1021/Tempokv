from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "run_vbench_long_matrix.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("tempokv_vbench_long_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


matrix = _load_script()


def test_inventory_accepts_both_cf_filename_schemes(tmp_path: Path) -> None:
    padded = tmp_path / "padded"
    padded.mkdir()
    for index in range(3):
        (padded / f"{index:06d}.mp4").write_bytes(b"video")
    tempo = tmp_path / "tempo"
    tempo.mkdir()
    for index in range(3):
        (tempo / f"{index}_0.mp4").write_bytes(b"video")

    assert [path.name for path in matrix._inventory(padded, 3)] == [
        "000000.mp4",
        "000001.mp4",
        "000002.mp4",
    ]
    assert [path.name for path in matrix._inventory(tempo, 3)] == [
        "0_0.mp4",
        "1_0.mp4",
        "2_0.mp4",
    ]


def test_inventory_rejects_gaps_and_prompt_mapping_uses_global_index(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "000000.mp4").write_bytes(b"video")
    (videos / "000002.mp4").write_bytes(b"video")
    with pytest.raises(ValueError, match="does not cover indices"):
        matrix._inventory(videos, 2)

    paths = [videos / "000002.mp4", videos / "000000.mp4"]
    assert matrix._prompt_mapping(paths, ["zero", "one", "two"]) == {
        "000002.mp4": "two",
        "000000.mp4": "zero",
    }


def test_manifest_allows_per_entry_wait_markers(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "name": "BF16",
                        "videos": str(videos),
                        "lane": 0,
                    },
                    {
                        "name": "QVG_INT2",
                        "videos": str(videos),
                        "lane": 1,
                        "wait_for": str(tmp_path / "generation.accepted"),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = matrix._load_manifest(manifest)
    assert payload["entries"][1]["wait_for"].endswith("generation.accepted")


def test_acceptance_requires_eight_metrics_and_derived_summary(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    name = "BF16"
    report = {
        "status": "complete",
        "video_count": 128,
        "protocol": matrix.PROTOCOL,
        "dimensions": list(matrix.DIMENSIONS),
        "completed_dimensions": list(matrix.DIMENSIONS),
        "input_fingerprint": "fingerprint",
    }
    (output / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    for dimension in matrix.DIMENSIONS:
        (output / f"{name}_{dimension}_eval_results.json").write_text(
            json.dumps({dimension: [0.5]}), encoding="utf-8"
        )
    display_scores = {
        dimension: 0.5 for dimension in matrix.DIMENSIONS[:6]
    }
    display_scores.update({"visual_quality": 0.5, "text_alignment": 0.5})
    (output / f"{name}_vbench_long_summary.json").write_text(
        json.dumps(
            {
                "protocol": matrix.PROTOCOL,
                "display_scores": display_scores,
            }
        ),
        encoding="utf-8",
    )

    accepted = matrix._validate_evaluation(output, name, 128)

    assert len(accepted["result_files"]) == 8
    assert accepted["summary_file"].endswith("BF16_vbench_long_summary.json")
