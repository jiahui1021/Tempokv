"""Runtime accounting shared by the TempoKV model integrations.

Two reporting rules matter for a rolling causal cache and are enforced here.

Memory is reported as *resident* cache state, not as the running total of every
encode event: a cache that packs a span, then evicts it, never held both at
once.  ``current_*`` tracks what is resident now, ``peak_*`` the high-water
mark, and ``total_*`` the cumulative encoded volume — the three are kept apart
so a compression ratio is never computed by dividing one kind by another.

Compression ratio always compares the compressed and uncompressed bytes *of the
same cache state*, which is why every span records both together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import torch


BF16_BYTES_PER_VALUE = 2

# One drain costs one ``torch.cuda.synchronize``.  ``decode_ms`` is bracketed
# once per packed span per cache read per layer, so the pool has to hold a
# comfortable multiple of that before a drain becomes necessary; profiling.py
# uses the same bound for the same reason.
TIMER_EVENT_POOL_SIZE = 8192


@dataclass
class _SpanBytes:
    compressed: int
    uncompressed: int


@dataclass
class TempoKVRuntimeReport:
    """Counters, byte accounting, and latency split for one generation run."""

    model: str
    kv_mode: str
    num_layers: int
    num_heads: int
    head_dim: int
    stride: int

    encode_key_calls: int = 0
    encode_value_calls: int = 0
    decode_tensor_calls: int = 0
    finalized_frames: int = 0
    ephemeral_writes: int = 0
    cache_evictions: int = 0
    cache_resets: int = 0

    predictor_parameter_bytes: int = 0

    # Only latencies this runtime actually measures live here. Attention time
    # is deliberately absent: timing it per layer needs device syncs that
    # distort the thing being measured, and it is recoverable as
    # `generation_ms` minus `encode_ms` and `decode_ms`. Per-prompt wall time
    # lives in the runtime's own `summary.json`.
    #
    # All three are GPU intervals measured with CUDA events, not host clocks:
    # kernel launches are asynchronous, so wall time around an encode or a
    # decode measures how long it took to *queue* the work. They are only
    # correct once `drain_timers` has read the events back, which `to_dict`
    # does; read them through `to_dict` rather than as attributes.
    encode_ms: float = 0.0
    decode_ms: float = 0.0
    generation_ms: float = 0.0

    total_encoded_cache_bytes: int = 0
    total_uncompressed_cache_bytes: int = 0
    peak_compressed_cache_bytes: int = 0
    peak_uncompressed_cache_bytes: int = 0

    # Dense frames that are finalized but not packed yet, plus the block being
    # denoised. Tracked apart from the packed spans because both are resident
    # at the same time and a peak-memory figure needs their sum.
    resident_bf16_bytes: int = 0
    peak_resident_total_bytes: int = 0

    _resident: dict[Any, _SpanBytes] = field(default_factory=dict, repr=False)
    # Recorded-but-unread CUDA event pairs, and the pool they are recycled
    # through.  ``None`` means the CUDA check has not run yet.
    _pending_intervals: list[tuple[str, Any, Any]] = field(
        default_factory=list, repr=False
    )
    _free_events: list[tuple[Any, Any]] = field(default_factory=list, repr=False)
    _uses_cuda_events: bool | None = field(default=None, repr=False)
    _timer_drains: int = field(default=0, repr=False)

    # -- cache state -----------------------------------------------------

    def record_span(
        self, key: Any, *, compressed_bytes: int, uncompressed_bytes: int
    ) -> None:
        """Install one packed span, replacing any earlier span with this key."""
        if compressed_bytes < 0 or uncompressed_bytes < 0:
            raise ValueError("span byte counts must not be negative")
        self._resident[key] = _SpanBytes(compressed_bytes, uncompressed_bytes)
        self.total_encoded_cache_bytes += compressed_bytes
        self.total_uncompressed_cache_bytes += uncompressed_bytes
        self._refresh_span_peaks()

    def update_span(
        self, key: Any, *, compressed_bytes: int, uncompressed_bytes: int
    ) -> None:
        """Grow a span installed earlier by :meth:`record_span`.

        An incrementally encoded span is recorded once when it opens and again
        after every append.  Only the growth is added to the cumulative totals,
        so a span that is appended to eight times is not counted eight times.
        """
        previous = self._resident.get(key)
        if previous is None:
            self.record_span(
                key,
                compressed_bytes=compressed_bytes,
                uncompressed_bytes=uncompressed_bytes,
            )
            return
        if compressed_bytes < 0 or uncompressed_bytes < 0:
            raise ValueError("span byte counts must not be negative")
        self._resident[key] = _SpanBytes(compressed_bytes, uncompressed_bytes)
        self.total_encoded_cache_bytes += max(
            0, compressed_bytes - previous.compressed
        )
        self.total_uncompressed_cache_bytes += max(
            0, uncompressed_bytes - previous.uncompressed
        )
        self._refresh_span_peaks()

    def drop_span(self, key: Any) -> None:
        if self._resident.pop(key, None) is not None:
            self.cache_evictions += 1

    def observe_resident_bf16_bytes(self, resident_bytes: int) -> None:
        """Record the dense part of the cache that is currently allocated.

        This is the only thing that moves the resident peak.  Packing a span
        frees dense chunks and installs compressed bytes in their place, so
        raising the peak from inside :meth:`record_span` would add the new
        compressed span on top of a dense figure that the same operation had
        just invalidated — and a peak, once raised, cannot be corrected down.
        """
        if resident_bytes < 0:
            raise ValueError("resident byte counts must not be negative")
        self.resident_bf16_bytes = int(resident_bytes)
        self.peak_resident_total_bytes = max(
            self.peak_resident_total_bytes, self.resident_total_bytes
        )

    def reset_cache(self) -> None:
        self._resident.clear()
        self.resident_bf16_bytes = 0
        self.cache_resets += 1

    def _refresh_span_peaks(self) -> None:
        self.peak_compressed_cache_bytes = max(
            self.peak_compressed_cache_bytes, self.current_compressed_cache_bytes
        )
        self.peak_uncompressed_cache_bytes = max(
            self.peak_uncompressed_cache_bytes,
            self.current_uncompressed_cache_bytes,
        )

    @property
    def current_compressed_cache_bytes(self) -> int:
        return sum(span.compressed for span in self._resident.values())

    @property
    def current_uncompressed_cache_bytes(self) -> int:
        return sum(span.uncompressed for span in self._resident.values())

    @property
    def resident_total_bytes(self) -> int:
        """Everything the KV cache actually holds: packed spans plus BF16."""
        return self.current_compressed_cache_bytes + self.resident_bf16_bytes

    @property
    def compression_ratio(self) -> float | None:
        """Uncompressed over compressed bytes for the currently resident cache."""
        compressed = self.current_compressed_cache_bytes
        if not compressed:
            return None
        return self.current_uncompressed_cache_bytes / compressed

    # -- latency ---------------------------------------------------------

    def timer(self, field_name: str) -> "_Timer":
        return _Timer(self, field_name)

    def uses_cuda_events(self) -> bool:
        """Whether timing brackets record CUDA events instead of a host clock."""
        if self._uses_cuda_events is None:
            self._uses_cuda_events = torch.cuda.is_available()
        return self._uses_cuda_events

    def _acquire_events(self) -> tuple[Any, Any]:
        if self._free_events:
            return self._free_events.pop()
        if len(self._pending_intervals) >= TIMER_EVENT_POOL_SIZE:
            self.drain_timers()
            if self._free_events:
                return self._free_events.pop()
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def drain_timers(self) -> None:
        """Fold every finished GPU interval into its field, synchronizing once.

        Recording an event does not synchronize, so an instrumented run keeps
        the launch pattern of an uninstrumented one.  The cost is that the
        latency fields are stale until this runs; ``to_dict`` calls it, and a
        caller that reads the attributes directly has to call it itself.
        """
        if not self._pending_intervals:
            return
        torch.cuda.synchronize()
        self._timer_drains += 1
        for field_name, start, end in self._pending_intervals:
            setattr(
                self,
                field_name,
                getattr(self, field_name) + start.elapsed_time(end),
            )
            self._free_events.append((start, end))
        self._pending_intervals.clear()

    # -- output ----------------------------------------------------------

    def require_tempokv_activity(self) -> None:
        """Fail a TempoKV run that never actually used the codec.

        Without this a misconfigured run produces a perfectly good video from
        an untouched BF16 cache and silently reports it as a TempoKV result.
        """
        if self.kv_mode != "tempokv":
            return
        empty = [
            name
            for name, value in (
                ("encode_key_calls", self.encode_key_calls),
                ("encode_value_calls", self.encode_value_calls),
                ("decode_tensor_calls", self.decode_tensor_calls),
                ("finalized_frames", self.finalized_frames),
            )
            if value <= 0
        ]
        if empty:
            raise RuntimeError(
                "TempoKV generation did not exercise the codec: "
                + ", ".join(f"{name}=0" for name in empty)
            )

    def to_dict(self) -> dict[str, Any]:
        # The latency fields are event intervals that have not been read back
        # yet, so this has to happen before any of them is serialized.
        self.drain_timers()
        return {
            "model": self.model,
            "kv_mode": self.kv_mode,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "stride": self.stride,
            "encode_key_calls": self.encode_key_calls,
            "encode_value_calls": self.encode_value_calls,
            "decode_tensor_calls": self.decode_tensor_calls,
            "finalized_frames": self.finalized_frames,
            "ephemeral_writes": self.ephemeral_writes,
            "cache_evictions": self.cache_evictions,
            "cache_resets": self.cache_resets,
            "current_compressed_cache_bytes": self.current_compressed_cache_bytes,
            "peak_compressed_cache_bytes": self.peak_compressed_cache_bytes,
            "resident_bf16_bytes": self.resident_bf16_bytes,
            "resident_total_bytes": self.resident_total_bytes,
            "peak_resident_total_bytes": self.peak_resident_total_bytes,
            "total_encoded_cache_bytes": self.total_encoded_cache_bytes,
            "current_uncompressed_cache_bytes": (
                self.current_uncompressed_cache_bytes
            ),
            "peak_uncompressed_cache_bytes": self.peak_uncompressed_cache_bytes,
            "total_uncompressed_cache_bytes": self.total_uncompressed_cache_bytes,
            "compression_ratio": self.compression_ratio,
            "predictor_parameter_bytes": self.predictor_parameter_bytes,
            "encode_ms": self.encode_ms,
            "decode_ms": self.decode_ms,
            "generation_ms": self.generation_ms,
        }


class _Timer:
    """Accumulate one section's GPU time into a report field.

    On CUDA the bracket is a pair of events rather than a host clock.  The
    encode and decode paths issue their whole kernel sequence without ever
    synchronizing, so ``perf_counter`` around them measures launch time, not
    device time -- and the amount it understates by depends on how much of the
    launch queue happened to be full, which makes it not merely small but
    unpredictable.  Off CUDA the host clock stays, so the accounting logic is
    exercisable without a device.
    """

    def __init__(self, report: TempoKVRuntimeReport, field_name: str) -> None:
        if not hasattr(report, field_name):
            raise AttributeError(f"report has no timing field {field_name!r}")
        self._report = report
        self._field = field_name
        self._started = 0.0
        self._events: tuple[Any, Any] | None = None

    def __enter__(self) -> "_Timer":
        if self._report.uses_cuda_events():
            self._events = self._report._acquire_events()
            self._events[0].record()
            return self
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._events is not None:
            start, end = self._events
            end.record()
            # Read back later: one synchronize per drain instead of one per
            # bracket, which would serialize the pipeline being measured.
            self._report._pending_intervals.append((self._field, start, end))
            self._events = None
            return
        elapsed_ms = (time.perf_counter() - self._started) * 1000.0
        setattr(self._report, self._field, getattr(self._report, self._field) + elapsed_ms)


def bf16_bytes(shape: tuple[int, ...]) -> int:
    """BF16 footprint of a cache tensor, used as the uncompressed baseline."""
    total = BF16_BYTES_PER_VALUE
    for extent in shape:
        total *= int(extent)
    return total


__all__ = ["BF16_BYTES_PER_VALUE", "TempoKVRuntimeReport", "bf16_bytes"]
