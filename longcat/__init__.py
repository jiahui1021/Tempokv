"""LongCat integration for the model-independent TempoKV codec."""

from .bridge import (
    LONGCAT_COORDINATE_SYSTEM,
    LONGCAT_HEAD_DIM,
    LONGCAT_NUM_HEADS,
    LONGCAT_NUM_LAYERS,
    LONGCAT_STRIDE,
    PackedTensorSpanSet,
    LongCatRuntimeSpec,
    LongCatTempoKVBridge,
    decode_kv_cache,
    encode_kv_cache,
    encode_kv_cache_spans,
    install_longcat_bridge,
    move_kv_cache,
    move_packed_tensor,
    validate_cache_layout,
)
from .trace import LongCatTraceCollector, dump_clean_kv

__all__ = [
    "LONGCAT_COORDINATE_SYSTEM",
    "LONGCAT_HEAD_DIM",
    "LONGCAT_NUM_HEADS",
    "LONGCAT_NUM_LAYERS",
    "LONGCAT_STRIDE",
    "PackedTensorSpanSet",
    "LongCatRuntimeSpec",
    "LongCatTempoKVBridge",
    "LongCatTraceCollector",
    "decode_kv_cache",
    "dump_clean_kv",
    "encode_kv_cache",
    "encode_kv_cache_spans",
    "install_longcat_bridge",
    "move_kv_cache",
    "move_packed_tensor",
    "validate_cache_layout",
]
