"""TempoKV implementation of Rolling Forcing attention.

This function intentionally keeps the call signature of the upstream
``wan.modules.rollingforcing.rollingforcing`` function.  The bridge replaces
that symbol after importing the upstream runtime, so the model and sampling
loop remain untouched.
"""

from __future__ import annotations

from typing import Any

import torch

from tempokv.runtime.rope import apply_rope_by_span

from .cache import RollingChunkedKVCache


def _index(cache: dict[str, Any], name: str) -> int:
    value = cache[name]
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _set_index(cache: dict[str, Any], name: str, value: int) -> None:
    current = cache[name]
    if isinstance(current, torch.Tensor):
        current.fill_(int(value))
    else:
        cache[name] = int(value)


def _rotate(
    values: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    cache: RollingChunkedKVCache,
    start: int,
    end: int,
) -> torch.Tensor:
    return apply_rope_by_span(
        values,
        grid_sizes,
        freqs,
        cache.spans(start, end),
    ).type_as(values)


def _decode_cache_pair(
    layer: dict[str, Any],
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Rolling Forcing tensors are [B, S, H, D], matching native BSHD storage.
    keys = layer["k"].read(start, end)
    values = layer["v"].read(start, end, source_tensor=keys)
    return keys, values


def _encode_clean_block(
    layer: dict[str, Any],
    *,
    local_start: int,
    local_end: int,
    absolute_start_frame: int,
) -> None:
    """Pack one clean block after all transformer layers have refreshed it."""
    encoder = layer.get("_tempokv_encoder")
    if encoder is None:
        return
    encoder.encode_block(
        layer["_tempokv_layers"],
        local_start=local_start,
        local_end=local_end,
        absolute_start_frame=absolute_start_frame,
    )


def rollingforcing(
    kv_cache: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    current_start: int,
    cache_start: int,
    updating_cache: bool,
    meta: dict[str, Any],
) -> torch.Tensor:
    """Run one Rolling Forcing attention call with a TempoKV cache."""
    from wan.modules.attention import attention
    from wan.modules.causal_model import rope_apply

    frame_seq_length = int(meta["frame_seqlen"])
    num_frame_per_block = int(meta["num_frame_per_block"])
    block_length = num_frame_per_block * frame_seq_length
    if q.shape[1] < block_length or k.shape[1] < block_length:
        raise ValueError("Rolling Forcing input is shorter than one cache block")
    current_start_frame = int(current_start) // frame_seq_length
    cache_start_frame = int(cache_start) // frame_seq_length
    cache_end = int(cache_start) + block_length

    cache_k: RollingChunkedKVCache = kv_cache["k"]
    cache_v: RollingChunkedKVCache = kv_cache["v"]
    if cache_k.frame_seq_length != frame_seq_length:
        raise ValueError("Rolling Forcing frame stride does not match TempoKV")
    cached_local_end = _index(kv_cache, "local_end_index")
    global_end = _index(kv_cache, "global_end_index")
    num_new_tokens = max(0, cache_end - global_end)
    capacity = cache_k.kv_cache_size
    sink_frames = int(meta.get("sink_size", 0))
    sink_tokens = sink_frames * frame_seq_length
    if sink_tokens > capacity:
        raise ValueError("Rolling Forcing sink is larger than the TempoKV cache")
    if sink_tokens and sink_tokens < block_length:
        # The original runtime keeps a complete first block as its anchor. A
        # smaller configured sink would make the anchor partial and ambiguous.
        sink_tokens = block_length

    if num_new_tokens + cached_local_end > capacity:
        evicted = num_new_tokens + cached_local_end - capacity
        evicted = ((evicted + frame_seq_length - 1) // frame_seq_length) * frame_seq_length
        if evicted + sink_tokens > cached_local_end:
            raise ValueError("Rolling Forcing cache cannot retain its sink block")
        unit = frame_seq_length
        dropped_frames = [
            int(frame)
            for frame in cache_k.global_frames[
                sink_tokens // unit : (sink_tokens + evicted) // unit
            ]
            if frame is not None
        ]
        # Materialize both caches before compaction; V may be CrossKV packed and
        # therefore must be decoded with reconstructed K as its source.
        resident_k = cache_k.read(0, cached_local_end)
        cache_k.evict_middle(
            evicted,
            sink_tokens=sink_tokens,
            resident_end=cached_local_end,
            source_tensor=None,
        )
        cache_v.evict_middle(
            evicted,
            sink_tokens=sink_tokens,
            resident_end=cached_local_end,
            source_tensor=resident_k,
        )
        del resident_k
        cached_local_end = cached_local_end - evicted
        encoder = kv_cache.get("_tempokv_encoder")
        if encoder is not None:
            encoder.drop_frames(dropped_frames)
            encoder.reset_after_eviction()
        report = kv_cache.get("_tempokv_report")
        layers = kv_cache.get("_tempokv_layers")
        if report is not None and layers is not None:
            report.observe_resident_bf16_bytes(
                sum(
                    layer[role].bf16_resident_bytes()
                    for layer in layers
                    for role in ("k", "v")
                )
            )

    local_end = cached_local_end + num_new_tokens
    local_start = local_end - block_length
    if local_start < 0 or local_end > capacity:
        raise ValueError("Rolling Forcing TempoKV cache capacity is too small")

    # Only the first block of each rolling window enters the persistent cache;
    # the remaining query blocks stay ephemeral, exactly as in the upstream
    # implementation. Rewrites are allowed because they cover the full block.
    cache_k.write(
        local_start,
        local_end,
        k[:, :block_length],
        global_start_frame=cache_start_frame,
    )
    cache_v.write(
        local_start,
        local_end,
        v[:, :block_length],
        global_start_frame=cache_start_frame,
    )
    trace_collector = kv_cache.get("_tempokv_trace")
    if updating_cache and trace_collector is not None:
        trace_collector.capture(
            int(meta.get("block_index", -1)),
            k[:, :block_length],
            v[:, :block_length],
            frame_stride=frame_seq_length,
            absolute_start_frame=cache_start_frame,
            num_frames=num_frame_per_block,
        )
    if num_new_tokens:
        _set_index(kv_cache, "global_end_index", cache_end)
        _set_index(kv_cache, "local_end_index", local_end)
    else:
        # The cache still contains the rewritten latest block.
        _set_index(kv_cache, "local_end_index", max(cached_local_end, local_end))

    kv_cache["_tempokv_last_local_start"] = local_start
    kv_cache["_tempokv_last_local_end"] = local_end
    kv_cache["_tempokv_last_new"] = bool(num_new_tokens)
    kv_cache["_tempokv_last_absolute_start"] = cache_start_frame

    # Upstream Rolling receives per-block tensors and its RoPE table is already
    # aligned to that block; absolute positions are handled only for cache reads.
    roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
    roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

    is_last_layer = int(meta.get("block_index", -1)) == int(
        kv_cache.get("_tempokv_num_layers", -2)
    ) - 1
    if local_start == 0:
        output = attention(roped_query, roped_key, v)
        if updating_cache and is_last_layer:
            _encode_clean_block(
                kv_cache,
                local_start=local_start,
                local_end=local_end,
                absolute_start_frame=cache_start_frame,
            )
        return output

    if updating_cache:
        cached_k, cached_v = _decode_cache_pair(kv_cache, 0, local_end)
        absolute_k = _rotate(cached_k, grid_sizes, freqs, cache_k, 0, local_end)
        output = attention(roped_query[:, :block_length], absolute_k, cached_v)
        if is_last_layer:
            _encode_clean_block(
                kv_cache,
                local_start=local_start,
                local_end=local_end,
                absolute_start_frame=cache_start_frame,
            )
        return output

    query_length = int(roped_query.shape[1])
    local_attn_size = int(meta.get("local_attn_size", -1))
    max_attention_size = (
        local_attn_size * frame_seq_length
        if local_attn_size != -1
        else local_end + query_length
    )
    working_budget = max(0, max_attention_size - query_length - block_length)
    anchor_end = min(sink_tokens, local_start)
    working_start = max(anchor_end, local_start - working_budget)

    pieces_k: list[torch.Tensor] = []
    pieces_v: list[torch.Tensor] = []
    pieces_spans = []
    if anchor_end:
        anchor_k, anchor_v = _decode_cache_pair(kv_cache, 0, anchor_end)
        pieces_k.append(anchor_k)
        pieces_v.append(anchor_v)
        pieces_spans.extend(cache_k.spans(0, anchor_end))
    if working_start < local_start:
        working_k, working_v = _decode_cache_pair(kv_cache, working_start, local_start)
        pieces_k.append(working_k)
        pieces_v.append(working_v)
        pieces_spans.extend(cache_k.spans(working_start, local_start))
    if pieces_k:
        history_k = torch.cat(pieces_k, dim=1)
        history_v = torch.cat(pieces_v, dim=1)
        history_k = apply_rope_by_span(
            history_k,
            grid_sizes,
            freqs,
            pieces_spans,
        ).type_as(v)
        input_k = torch.cat((history_k, roped_key), dim=1)
        input_v = torch.cat((history_v, v), dim=1)
    else:
        input_k, input_v = roped_key, v
    return attention(roped_query, input_k, input_v)


__all__ = ["rollingforcing"]
