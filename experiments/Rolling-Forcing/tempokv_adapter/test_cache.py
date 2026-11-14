from __future__ import annotations

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tempokv_adapter.cache import RollingChunkedKVCache


def test_evict_middle_preserves_sink_and_absolute_frames() -> None:
    cache = RollingChunkedKVCache(
        batch_size=1,
        frame_seq_length=2,
        num_heads=1,
        head_dim=4,
        max_num_chunks=6,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        layout="BHSD",
    )
    values = torch.arange(1 * 1 * 10 * 4, dtype=torch.float32).reshape(1, 1, 10, 4)
    cache.write(0, 10, values.to(torch.bfloat16), global_start_frame=0)

    local_end = cache.evict_middle(4, sink_tokens=2, resident_end=10)

    assert local_end == 6
    assert cache.global_frames[:3] == [0, 3, 4]
    rebuilt = cache.read_bhsd(0, 6)
    expected = torch.cat((values[:, :, :2], values[:, :, 6:]), dim=2).to(torch.bfloat16)
    torch.testing.assert_close(rebuilt, expected)
    assert [(span.global_start_frame, span.num_frames) for span in cache.spans(0, 6)] == [
        (0, 1),
        (3, 2),
    ]
