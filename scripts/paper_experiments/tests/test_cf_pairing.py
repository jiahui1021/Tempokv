from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    script = ROOT / "scripts/cf_compare_quality.py"
    spec = importlib.util.spec_from_file_location("cf_compare_quality", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cf_pairing_matches_numeric_index_only(tmp_path):
    compare = _load_module()
    reference = tmp_path / "reference"
    target = tmp_path / "target"
    reference.mkdir()
    target.mkdir()

    for name in ("10_0.mp4", "2_0.mp4", "prompt text.mp4"):
        (reference / name).write_bytes(b"reference")
    for name in ("2_0.mp4", "10_0.mp4", "99_0.mp4", "prompt text.mp4"):
        (target / name).write_bytes(b"target")

    pairs = compare.match_pairs(reference, target)

    assert [name for name, _, _ in pairs] == ["2_0.mp4", "10_0.mp4"]
    assert compare.match_pairs(reference, target, mode="index") == pairs
