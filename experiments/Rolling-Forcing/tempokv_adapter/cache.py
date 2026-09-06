"""Rolling-window cache primitives for the TempoKV Rolling Forcing bridge.

The upstream Rolling Forcing runtime stores post-RoPE tensors in a mutable
dense buffer.  TempoKV needs the opposite boundary: normalized pre-RoPE K/V
are kept in frame-aligned :class:`ChunkedKVCache` objects and positions are
restored only when a read is made.  This small helper implements the one
operation that the generic cache intentionally does not guess: removing an
interior range while retaining an attention sink.
"""

from __future__ import annotations

from typing import Any

import torch

from tempokv.runtime.cache import ChunkedKVCache


class RollingChunkedKVCache(ChunkedKVCache):
    """A frame-aligned cache with explicit sink/recent compaction.

    Compaction materializes the resident K/V pair before moving it.  That is
    deliberate: a rolling eviction can cut through a packed span, while a
    packed TempoKV payload cannot be sliced safely.  The next clean refresh
    repacks the newly visible block, so correctness does not depend on span
    boundaries coinciding with the rolling window.
    """

    def evict_middle(
        self,
        num_tokens: int,
        *,
        sink_tokens: int,
        resident_end: int,
        source_tensor: torch.Tensor | None = None,
    ) -> int:
        if num_tokens <= 0:
            return resident_end
        unit = self.frame_seq_length
        if num_tokens % unit or sink_tokens % unit or resident_end % unit:
            raise ValueError("rolling eviction must be frame aligned")
        if not 0 <= sink_tokens <= resident_end:
            raise ValueError("sink_tokens must fit inside the resident cache")
        if sink_tokens + num_tokens > resident_end:
            raise ValueError("rolling eviction removes more than the working cache")

        # ``source_tensor`` is the reconstructed K for a V cache.  The K cache
        # passes None; a V cache passes the matching K read so CrossKV decoding
        # stays on the same closed loop as production attention.
        resident = self.read(0, resident_end, source_tensor=source_tensor)
        sink_end = sink_tokens
        recent_start = sink_tokens + num_tokens
        sink_slice = [slice(None)] * resident.ndim
        sink_slice[self.seq_dim] = slice(0, sink_end)
        sink = resident[tuple(sink_slice)].clone() if sink_end else None
        recent_slice = [slice(None)] * resident.ndim
        recent_slice[self.seq_dim] = slice(recent_start, resident_end)
        recent = resident[tuple(recent_slice)].clone()
        sink_global = self.global_frames[0]
        recent_global = self.global_frames[recent_start // unit]
        self.clear()
        if sink is not None and sink.numel():
            self.write(0, sink_end, sink, global_start_frame=sink_global)
        recent_end = sink_end + recent.shape[self.seq_dim]
        if recent.numel():
            self.write(
                sink_end,
                recent_end,
                recent,
                global_start_frame=recent_global,
            )
        return recent_end


def cache_layer_metadata(cache: RollingChunkedKVCache) -> dict[str, Any]:
    """Return a compact state snapshot useful in reports and tests."""
    resident_chunks = sum(state.name != "EMPTY" for state in cache.chunk_state)
    return {
        "resident_chunks": int(resident_chunks),
        "resident_frames": int(resident_chunks),
        "global_frames": [frame for frame in cache.global_frames if frame is not None],
        "capacity_frames": int(cache.max_num_chunks),
    }


__all__ = ["RollingChunkedKVCache", "cache_layer_metadata"]
