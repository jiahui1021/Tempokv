"""Runtime pieces shared by the TempoKV model integrations.

``launch`` holds process-level kernel configuration.  ``cache``, ``rope``,
``streaming`` and ``metrics`` hold the causal video KV runtime that
Self-Forcing and Causal Forcing share: both run Wan2.1-1.3B as a causal
autoregressive video diffusion model, so both need the same frame-aligned
pre-RoPE cache, the same read-time absolute-position RoPE, the same streaming
span encoder, and the same memory/latency accounting.  ``jensen`` holds the
quantized-key attention-score correction those integrations can opt into.
"""

from .cache import (
    CacheSpan,
    ChunkedKVCache,
    ChunkState,
    set_quantized_decoder,
    uncompress_single_kv_cache,
)
from .jensen import augment as augment_for_jensen
from .jensen import bias as jensen_bias
from .launch import (
    PERSISTENT_UNITS_ENV,
    configure_triton_cache,
    persistent_units_per_launch,
)
from .metrics import BF16_BYTES_PER_VALUE, TempoKVRuntimeReport, bf16_bytes
from .rope import apply_rope_by_span, causal_rope_apply_long_input
from .streaming import StreamingTempoKVEncoder

__all__ = [
    "BF16_BYTES_PER_VALUE",
    "CacheSpan",
    "ChunkState",
    "ChunkedKVCache",
    "PERSISTENT_UNITS_ENV",
    "StreamingTempoKVEncoder",
    "TempoKVRuntimeReport",
    "apply_rope_by_span",
    "augment_for_jensen",
    "bf16_bytes",
    "causal_rope_apply_long_input",
    "configure_triton_cache",
    "jensen_bias",
    "persistent_units_per_launch",
    "set_quantized_decoder",
    "uncompress_single_kv_cache",
]
