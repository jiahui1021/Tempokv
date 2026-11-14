from __future__ import annotations

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tempokv_adapter.bridge import RollingTraceCollector


def test_trace_collector_writes_fitter_layout(tmp_path) -> None:
    collector = RollingTraceCollector(tmp_path / "clean_trace.npz", tokens_per_frame=2)
    for layer in range(2):
        keys = torch.arange(1 * 6 * 1 * 2, dtype=torch.float32).reshape(1, 6, 1, 2)
        values = keys + 100
        collector.capture(
            layer,
            keys,
            values,
            frame_stride=2,
            absolute_start_frame=0,
            num_frames=3,
        )
    path = collector.write(num_layers=2)
    with np.load(path, allow_pickle=False) as trace:
        assert trace["keys"].shape == (2, 2, 3, 1, 2)
        assert trace["values"].shape == trace["keys"].shape
        assert trace["trace_layout"].item() == "tempokv_flat_samples_v1"

    collector.start_run()
    for layer in range(2):
        collector.capture(
            layer,
            torch.zeros(1, 6, 1, 2),
            torch.zeros(1, 6, 1, 2),
            frame_stride=2,
            absolute_start_frame=0,
            num_frames=3,
        )
    collector.write(num_layers=2)
    with np.load(path, allow_pickle=False) as trace:
        assert trace["keys"].shape == (2, 4, 3, 1, 2)
