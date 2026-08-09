"""Self-Forcing view of the shared TempoKV causal KV cache.

The implementation lives in :mod:`tempokv.runtime.cache` because Self-Forcing
and Causal Forcing run the same Wan2.1-1.3B causal backbone and therefore need
the same frame-aligned pre-RoPE cache.  This module only re-exports it so the
bundled runtime keeps its flat ``from kv_cache import ChunkedKVCache`` import.
"""

from __future__ import annotations

from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
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
