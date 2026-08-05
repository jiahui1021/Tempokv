from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "self-forcing" / "prepare_trace.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_self_forcing_trace", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare = _load_script()


def test_prepare_trace_places_frames_on_temporal_axis(tmp_path) -> None:
    event = tmp_path / "source" / "prompt_00000" / "event_000"
    event.mkdir(parents=True)
    for layer in range(30):
        key = np.zeros((1, 4, 2, 12, 128), dtype=np.float16)
        value = np.zeros_like(key)
        for frame in range(4):
            key[:, frame] = layer * 10 + frame
            value[:, frame] = layer * 10 + frame + 1
        np.savez(
            event / f"layer_{layer:02d}.npz",
            k_pre_rope=key,
            v=value,
            layer_index=np.asarray(layer),
        )
    output = tmp_path / "trace.npz"
    report = prepare.convert_trace_root(tmp_path / "source", output)
    assert report["shape"] == [30, 2, 4, 12, 128]
    with np.load(output, allow_pickle=False) as trace:
        assert trace["keys"].shape == (30, 2, 4, 12, 128)
        np.testing.assert_array_equal(trace["keys"][3, 0, :, 0, 0], [30, 31, 32, 33])
        assert trace["coordinate_system"].item() == "pre_rope_normalized"
        assert trace["stride"].item() == 1560
