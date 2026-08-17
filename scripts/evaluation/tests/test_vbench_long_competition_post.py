import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "run_vbench_long_text_alignment_matrix.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("vbench_long_competition_post", SCRIPT)
post = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(post)


def test_nvidia_memory_parses_uuid_usage(monkeypatch) -> None:
    monkeypatch.setattr(
        post.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="GPU-aaa, 512\nGPU-bbb, 0\n"
        ),
    )

    assert post._nvidia_memory() == {"GPU-aaa": 512, "GPU-bbb": 0}


def test_collect_scores_and_competition_aggregation(tmp_path: Path) -> None:
    name = "BF16_REFERENCE"
    values = {
        dimension: float(index)
        for index, dimension in enumerate(
            post.evaluate_vbench.CUSTOM_INPUT_DIMENSIONS, start=1
        )
    }
    for dimension, value in values.items():
        (tmp_path / f"{name}_{dimension}_eval_results.json").write_text(
            json.dumps({dimension: [value, []]}), encoding="utf-8"
        )

    scores = post._collect_scores(tmp_path, name)
    summary_path = post.evaluate_vbench._write_summary(tmp_path, name, scores)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    temporal = (1 + 2 + 3 + 0.5 * 4) / 3.5
    frame = (5 + 6) / 2
    assert summary["official_competition_aspects"]["temporal_quality"] == pytest.approx(temporal)
    assert summary["official_competition_aspects"]["frame_wise_quality"] == pytest.approx(frame)
    assert summary["display_scores"]["visual_quality"] == pytest.approx(3.5)
    assert summary["official_competition_aspects"]["text_alignment"] == pytest.approx(7.5)
    assert summary["official_vbench_total"] is False


def test_scalar_rejects_non_finite(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"clip_score": [NaN]}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-finite"):
        post._scalar(path, "clip_score")


def test_table_uses_six_dimension_visual_quality(tmp_path: Path) -> None:
    table = tmp_path / "scores.md"
    post._write_table(
        table,
        [
            {
                "name": "candidate",
                "scores": {
                    "visual_quality": 0.35,
                    "temporal_quality": 0.8,
                    "frame_wise_quality": 0.6,
                    "text_alignment": 0.4,
                },
            }
        ],
    )

    assert "| candidate | 0.350000 | 0.800000 | 0.600000 | 0.400000 |" in table.read_text(
        encoding="utf-8"
    )
