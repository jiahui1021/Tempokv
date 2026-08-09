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


BF16_BYTES_PER_VALUE = 2


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

    encode_ms: float = 0.0
    decode_ms: float = 0.0
    attention_ms: float = 0.0
    generation_ms: float = 0.0
    first_chunk_ms: float | None = None

    total_encoded_cache_bytes: int = 0
    total_uncompressed_cache_bytes: int = 0
    peak_compressed_cache_bytes: int = 0
    peak_uncompressed_cache_bytes: int = 0

    _resident: dict[Any, _SpanBytes] = field(default_factory=dict, repr=False)

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
        self._refresh_peaks()

    def drop_span(self, key: Any) -> None:
        if self._resident.pop(key, None) is not None:
            self.cache_evictions += 1

    def reset_cache(self) -> None:
        self._resident.clear()
        self.cache_resets += 1

    def _refresh_peaks(self) -> None:
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
    def compression_ratio(self) -> float | None:
        """Uncompressed over compressed bytes for the currently resident cache."""
        compressed = self.current_compressed_cache_bytes
        if not compressed:
            return None
        return self.current_uncompressed_cache_bytes / compressed

    # -- latency ---------------------------------------------------------

    def timer(self, field_name: str) -> "_Timer":
        return _Timer(self, field_name)

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
            "attention_ms": self.attention_ms,
            "generation_ms": self.generation_ms,
            "first_chunk_ms": self.first_chunk_ms,
        }


class _Timer:
    """Accumulate wall time into one report field."""

    def __init__(self, report: TempoKVRuntimeReport, field_name: str) -> None:
        if not hasattr(report, field_name):
            raise AttributeError(f"report has no timing field {field_name!r}")
        self._report = report
        self._field = field_name
        self._started = 0.0

    def __enter__(self) -> "_Timer":
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        elapsed_ms = (time.perf_counter() - self._started) * 1000.0
        setattr(self._report, self._field, getattr(self._report, self._field) + elapsed_ms)


def bf16_bytes(shape: tuple[int, ...]) -> int:
    """BF16 footprint of a cache tensor, used as the uncompressed baseline."""
    total = BF16_BYTES_PER_VALUE
    for extent in shape:
        total *= int(extent)
    return total


__all__ = ["BF16_BYTES_PER_VALUE", "TempoKVRuntimeReport", "bf16_bytes"]
