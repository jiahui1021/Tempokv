from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_quality.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("tempokv_evaluate_quality", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


quality = _load_script()


class FakeQualityBackend:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def ssim(self, reference, candidate) -> float:
        return 1.0 - float(np.abs(reference.astype(float) - candidate).mean()) / 255.0

    def lpips(self, reference, candidate) -> float:
        return float(np.abs(reference.astype(float) - candidate).mean()) / 255.0


class FakeFidBackend:
    instances = []

    def __init__(self, *args, **kwargs) -> None:
        self.frames = 0
        self.pairs = []
        self.batch_size = kwargs.get("batch_size")
        self.instances.append(self)

    def add(self, reference, candidate) -> None:
        self.frames += 1
        self.pairs.append((reference.copy(), candidate.copy()))

    def compute(self) -> float:
        if self.frames < 2:
            raise ValueError("FID requires at least two matched frames")
        return 7.5


@pytest.fixture(autouse=True)
def _backends(monkeypatch):
    FakeFidBackend.instances.clear()
    monkeypatch.setattr(quality, "QualityBackend", FakeQualityBackend)
    monkeypatch.setattr(quality, "FidBackend", FakeFidBackend)


def _pair(root: Path, relative: str = "sample.mp4") -> tuple[Path, Path]:
    reference = root / "reference" / relative
    candidate = root / "candidate" / relative
    reference.parent.mkdir(parents=True, exist_ok=True)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(b"reference")
    candidate.write_bytes(b"candidate")
    return reference.resolve(), candidate.resolve()


def test_evaluate_adds_directory_fid_without_per_video_fid(tmp_path, monkeypatch) -> None:
    reference, candidate = _pair(tmp_path)
    frames = {
        reference: [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)],
        candidate: [np.ones((2, 2, 3), dtype=np.uint8) for _ in range(3)],
    }
    monkeypatch.setattr(quality, "_iter_frames", lambda path: iter(frames[path.resolve()]))

    report = quality.evaluate(tmp_path / "reference", tmp_path / "candidate")

    assert report["fid"] == 7.5
    assert report["fid_feature"] == 2048
    assert report["fid_reference"] == "bf16"
    assert report["fid_frames"] == 3
    assert report["fid_batch_size"] == 32
    assert FakeFidBackend.instances[0].batch_size == 32
    assert report["frame_stride"] == 1
    assert "fid" not in report["per_video"][0]


def test_stride_and_max_frames_feed_only_selected_frames(tmp_path, monkeypatch) -> None:
    reference, candidate = _pair(tmp_path)
    frames = {
        reference: [np.full((1, 1, 3), value, dtype=np.uint8) for value in range(6)],
        candidate: [np.full((1, 1, 3), value + 1, dtype=np.uint8) for value in range(6)],
    }
    monkeypatch.setattr(quality, "_iter_frames", lambda path: iter(frames[path.resolve()]))

    report = quality.evaluate(
        tmp_path / "reference",
        tmp_path / "candidate",
        frame_stride=2,
        max_frames=2,
    )

    assert report["frames"] == 2
    assert report["fid_frames"] == 2
    selected = FakeFidBackend.instances[0].pairs
    assert [int(pair[0][0, 0, 0]) for pair in selected] == [0, 2]


def test_missing_candidate_and_empty_reference_are_rejected(tmp_path) -> None:
    (tmp_path / "reference").mkdir()
    (tmp_path / "candidate").mkdir()
    with pytest.raises(ValueError, match="no videos"):
        quality.evaluate(tmp_path / "reference", tmp_path / "candidate")

    (tmp_path / "reference" / "missing.mp4").write_bytes(b"video")
    with pytest.raises(FileNotFoundError, match="missing 1 matching video"):
        quality.evaluate(tmp_path / "reference", tmp_path / "candidate")


def test_invalid_frame_stride_is_rejected(tmp_path, monkeypatch) -> None:
    reference, candidate = _pair(tmp_path)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    monkeypatch.setattr(quality, "_iter_frames", lambda path: iter([frame, frame]))

    with pytest.raises(ValueError, match="frame-stride"):
        quality.compare_video(reference, candidate, frame_stride=0)
