"""Whole-span release and hard checks over the actual KV-owned allocations."""
from __future__ import annotations

from typing import Any, Callable

import torch

from tempokv.runtime.cache import ChunkedKVCache, ChunkState

from .accounting import cache_snapshot


class ReleasableKVCache(ChunkedKVCache):
    def __init__(self, *args, after_write: Callable[[], Any] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_write = after_write

    def release_before(self, first_visible_frame: int) -> int:
        """Release only whole spans ending before the visible range; retain decode support."""
        cut = max(0, min(int(first_visible_frame), self.max_num_chunks))
        for span in self.quantized_spans:
            if span["start_chunk"] < cut < span["end_chunk"]:
                cut = span["start_chunk"]
        removed = sum(self.chunk_state[i] != ChunkState.EMPTY for i in range(cut))
        self.quantized_spans = [s for s in self.quantized_spans if s["end_chunk"] > cut]
        for i in range(cut):
            self.chunks[i] = None
            self.chunk_state[i] = ChunkState.EMPTY
            self._span_by_chunk[i] = None
            self.global_frames[i] = None
        return removed

    def write(self, *args, **kwargs):
        super().write(*args, **kwargs)
        if self.after_write is not None:
            self.after_write()


class BudgetManager:
    def __init__(self, caches: list[dict], *, budget_bytes: int | None,
                 history_frames: int, tokens_per_frame: int, parameters=None,
                 event_sink: Callable[[dict], Any] | None = None) -> None:
        if history_frames < 1 or (budget_bytes is not None and budget_bytes < 1):
            raise ValueError("history_frames and budget_bytes must be positive")
        self.caches = caches
        self.budget_bytes = budget_bytes
        self.history_frames = history_frames
        self.tokens_per_frame = tokens_per_frame
        self.parameters = parameters
        self.event_sink = event_sink
        self.events: list[dict] = []
        self.peak_bytes = 0
        self.end_frame = 0

    def snapshot(self, event: str, *, reserved_bytes: int = 0) -> dict:
        report = cache_snapshot(self.caches, parameters=self.parameters)
        usage = report["kv_owned_bytes"] + reserved_bytes
        self.peak_bytes = max(self.peak_bytes, usage)
        report.update(event=event, end_frame=self.end_frame, reserved_incoming_bytes=reserved_bytes,
                      checked_bytes=usage, budget_bytes=self.budget_bytes,
                      peak_checked_bytes=self.peak_bytes,
                      first_visible_frame=max(0, self.end_frame - self.history_frames))
        if self.budget_bytes is not None and usage > self.budget_bytes:
            raise MemoryError(f"KV budget exceeded at {event}: {usage} > {self.budget_bytes} bytes")
        self.events.append(report)
        if self.event_sink is not None:
            self.event_sink(report)
        return report

    def prepare(self, end_frame: int, start_frame: int) -> dict:
        """Evict before a generator pass and reserve all not-yet-allocated write chunks."""
        if not 0 <= start_frame < end_frame:
            raise ValueError("invalid incoming frame range")
        self.end_frame = int(end_frame)
        visible = max(0, end_frame - self.history_frames)
        for layer in self.caches:
            for role in ("k", "v"):
                layer[role].release_before(visible)
        reserved = 0
        for layer in self.caches:
            for role in ("k", "v"):
                cache = layer[role]
                for frame in range(start_frame, end_frame):
                    if cache.chunk_state[frame] == ChunkState.QUANTIZED:
                        raise RuntimeError("generator is rewriting an already packed clean span")
                    if cache.chunks[frame] is None:
                        reserved += (cache.batch_size * cache.frame_seq_length * cache.num_heads *
                                     cache.head_dim * torch.empty((), dtype=cache.dtype).element_size())
        return self.snapshot("before_write", reserved_bytes=reserved)

    def close(self):
        """Drop observer references as well as cache payloads at the end of an owner lifecycle."""
        for layer in self.caches:
            for role in ("k", "v"):
                layer[role].after_write = None
        self.parameters = None
        self.caches = []
