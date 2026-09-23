from __future__ import annotations

import pytest
import torch

from causal_forcing.tempokv_adapter import bridge, generate
from tempokv.runtime.cache import ChunkedKVCache


def test_multi_frame_write_uses_one_foreach_copy(monkeypatch) -> None:
    cache = ChunkedKVCache(
        batch_size=2,
        frame_seq_length=3,
        num_heads=2,
        head_dim=4,
        max_num_chunks=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        layout="BSHD",
    )
    source = torch.randn(2, 12, 2, 4).bfloat16()
    calls: list[int] = []
    original = torch._foreach_copy_

    def recording_foreach(destinations, sources):
        calls.append(len(sources))
        return original(destinations, sources)

    monkeypatch.setattr(torch, "_foreach_copy_", recording_foreach)
    cache.write(0, 12, source)

    assert calls == [4]
    for index, chunk in enumerate(cache.chunks):
        assert chunk is not None
        assert torch.equal(chunk, source[:, index * 3 : (index + 1) * 3])


@pytest.mark.parametrize("output_layout", ("BSHD", "BHSD"))
def test_dense_tail_chunks_use_one_foreach_copy(monkeypatch, output_layout: str) -> None:
    cache = ChunkedKVCache(
        batch_size=2,
        frame_seq_length=3,
        num_heads=2,
        head_dim=4,
        max_num_chunks=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        layout="BSHD",
    )
    source = torch.randn(2, 12, 2, 4).bfloat16()
    cache.write(0, 12, source)

    calls: list[int] = []
    original = torch._foreach_copy_

    def recording_foreach(destinations, sources):
        calls.append(len(sources))
        return original(destinations, sources)

    monkeypatch.setattr(torch, "_foreach_copy_", recording_foreach)
    actual = cache.read(0, 12) if output_layout == "BSHD" else cache.read_bhsd(0, 12)
    expected = source if output_layout == "BSHD" else source.permute(0, 2, 1, 3)

    assert torch.equal(actual, expected)
    assert calls == [4]


def test_decode_timing_is_explicit_and_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv(bridge.DECODE_TIMING_ENV, raising=False)
    assert bridge._decode_timing_enabled() is False

    monkeypatch.setenv(bridge.DECODE_TIMING_ENV, "1")
    assert bridge._decode_timing_enabled() is True

    monkeypatch.setenv(bridge.DECODE_TIMING_ENV, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        bridge._decode_timing_enabled()


def test_runtime_report_marks_untimed_decode_as_missing() -> None:
    report = bridge._CausalTempoKVRuntimeReport(
        model="causal_forcing_chunkwise",
        kv_mode="tempokv",
        num_layers=30,
        num_heads=12,
        head_dim=128,
        stride=1560,
    )
    report.decode_ms = 7.5

    untimed = report.to_dict()
    report.decode_timing_enabled = True
    timed = report.to_dict()

    assert untimed["decode_timing_enabled"] is False
    assert untimed["decode_ms"] is None
    assert timed["decode_timing_enabled"] is True
    assert timed["decode_ms"] == 7.5


def _runtime_report(*, timing_enabled: bool, decode_ms: float | None) -> dict:
    return {
        "model": "causal_forcing_chunkwise",
        "kv_mode": "tempokv",
        "num_layers": 30,
        "num_heads": 12,
        "head_dim": 128,
        "stride": 1560,
        "decode_timing_enabled": timing_enabled,
        "decode_ms": decode_ms,
    }


def test_merged_report_does_not_turn_missing_decode_timing_into_zero() -> None:
    ranks = [
        {"tempokv_report": _runtime_report(timing_enabled=False, decode_ms=None)},
        {"tempokv_report": _runtime_report(timing_enabled=False, decode_ms=None)},
    ]

    merged = generate._merge_tempokv_reports(ranks)

    assert merged["decode_timing_enabled"] is False
    assert merged["decode_ms"] is None


def test_merged_report_accepts_pre_flag_timed_reports() -> None:
    report = _runtime_report(timing_enabled=True, decode_ms=7.5)
    report.pop("decode_timing_enabled")

    merged = generate._merge_tempokv_reports([{"tempokv_report": report}])

    assert merged["decode_timing_enabled"] is True
    assert merged["decode_ms"] == 7.5
