from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_quality.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("tempokv_evaluate_psnr", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


quality = _load_script()


class FakeBackend:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def ssim(self, reference, candidate) -> float:
        return 1.0 - float(np.abs(reference.astype(float) - candidate).mean()) / 255.0

    def lpips(self, reference, candidate) -> float:
        return float(np.abs(reference.astype(float) - candidate).mean()) / 255.0


@pytest.fixture(autouse=True)
def _quality_backend(monkeypatch):
    monkeypatch.setattr(quality, "QualityBackend", FakeBackend)


def _pair(root: Path, relative: str) -> tuple[Path, Path]:
    reference = root / "reference" / relative
    candidate = root / "candidate" / relative
    reference.parent.mkdir(parents=True, exist_ok=True)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(b"reference")
    candidate.write_bytes(b"candidate")
    return reference.resolve(), candidate.resolve()


def test_evaluate_matches_relative_paths_and_pools_pixels(tmp_path, monkeypatch) -> None:
    first = _pair(tmp_path, "one.mp4")
    second = _pair(tmp_path, "nested/two.mp4")
    frames = {
        first[0]: [np.zeros((1, 1, 3), dtype=np.uint8)],
        first[1]: [np.ones((1, 1, 3), dtype=np.uint8)],
        second[0]: [np.zeros((2, 1, 3), dtype=np.uint8)],
        second[1]: [np.full((2, 1, 3), 2, dtype=np.uint8)],
    }
    monkeypatch.setattr(quality, "_iter_frames", lambda path: iter(frames[path.resolve()]))

    report = quality.evaluate(tmp_path / "reference", tmp_path / "candidate")

    assert report["videos"] == 2
    assert report["frames"] == 2
    assert report["missing"] == []
    assert report["mse"] == pytest.approx(3.0)
    assert report["psnr_db"] == pytest.approx(10 * math.log10(255**2 / 3))
    assert report["ssim"] == pytest.approx(1.0 - 1.5 / 255.0)
    assert report["lpips"] == pytest.approx(1.5 / 255.0)
    assert [item["video"] for item in report["per_video"]] == [
        "nested/two.mp4",
        "one.mp4",
    ]


def test_identical_video_reports_unbounded_psnr(tmp_path, monkeypatch) -> None:
    reference, candidate = _pair(tmp_path, "same.mp4")
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    monkeypatch.setattr(
        quality,
        "_iter_frames",
        lambda path: iter([frame.copy()]) if path in {reference, candidate} else iter(()),
    )

    report = quality.evaluate(tmp_path / "reference", tmp_path / "candidate")

    assert report["mse"] == 0
    assert report["psnr_db"] is None
    assert report["ssim"] == 1.0
    assert report["lpips"] == 0.0


def test_missing_candidate_is_rejected_before_decoding(tmp_path) -> None:
    reference = tmp_path / "reference" / "missing.mp4"
    reference.parent.mkdir()
    reference.write_bytes(b"reference")
    (tmp_path / "candidate").mkdir()

    with pytest.raises(FileNotFoundError, match="missing 1 matching video"):
        quality.evaluate(tmp_path / "reference", tmp_path / "candidate")


def test_frame_count_mismatch_has_clear_error(tmp_path, monkeypatch) -> None:
    reference, candidate = _pair(tmp_path, "different.mp4")
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(
        quality,
        "_iter_frames",
        lambda path: iter([frame, frame]) if path == reference else iter([frame]),
    )

    with pytest.raises(ValueError, match="frame count differs"):
        quality.compare_video(reference, candidate)


def test_help_is_available_without_video_decoder(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        quality.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "pooled" in output
    assert "PSNR" in output
    assert "SSIM" in output
    assert "LPIPS" in output
