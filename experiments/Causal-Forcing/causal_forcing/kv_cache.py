"""Causal Forcing view of the shared TempoKV causal KV cache.

The implementation lives in :mod:`tempokv.runtime.cache` and is shared with
Self-Forcing: both run the same Wan2.1-1.3B causal backbone, so both need the
same frame-aligned pre-RoPE cache with BF16, ephemeral, and packed TempoKV
chunk states.
"""

from __future__ import annotations

from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if _SOURCE_ROOT.is_dir() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from tempokv.runtime.cache import (  # noqa: E402
    CacheSpan,
    ChunkedKVCache,
    ChunkState,
    set_quantized_decoder,
    uncompress_single_kv_cache,
)

__all__ = [
    "CacheSpan",
    "ChunkState",
    "ChunkedKVCache",
    "set_quantized_decoder",
    "uncompress_single_kv_cache",
]
