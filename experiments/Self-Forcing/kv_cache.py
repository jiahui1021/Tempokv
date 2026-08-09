"""Frame-aligned Self-Forcing KV cache used by BF16 and TempoKV.

The runtime stores clean, normalized pre-RoPE K and matching V in BSHD
chunks.  A span may later be replaced by an opaque TempoKV state.  This file
only owns cache storage; the TempoKV bridge registers the decoder.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable

import torch


class ChunkState(IntEnum):
    EMPTY = 0
    BF16 = 1
    QUANTIZED = 2


_quantized_decoder: Callable[..., torch.Tensor] | None = None


def set_quantized_decoder(decoder: Callable[..., torch.Tensor] | None) -> None:
    global _quantized_decoder
    _quantized_decoder = decoder


def uncompress_single_kv_cache(
    state: Any,
    *,
    source_tensor: torch.Tensor | None = None,
    output_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    if _quantized_decoder is None:
        raise RuntimeError("no TempoKV decoder is installed for this cache span")
    return _quantized_decoder(
        state,
        source_tensor=source_tensor,
        output_tensor=output_tensor,
    )


class ChunkedKVCache:
    """One K or V cache with lazy BF16 chunks and packed TempoKV spans."""

    def __init__(
        self,
        batch_size: int,
        frame_seq_length: int,
        num_heads: int,
        head_dim: int,
        max_num_chunks: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        layout: str = "BSHD",
        cache_role: str | None = None,
    ) -> None:
        if layout not in {"BSHD", "BHSD"}:
            raise ValueError(f"unsupported cache layout: {layout}")
        if min(
            batch_size,
            frame_seq_length,
            num_heads,
            head_dim,
            max_num_chunks,
        ) <= 0:
            raise ValueError("cache dimensions must be positive")
        self.batch_size = batch_size
        self.frame_seq_length = frame_seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_num_chunks = max_num_chunks
        self.dtype = dtype
        self.device = device
        self.layout = layout
        self.cache_role = cache_role
        self.seq_dim = 1 if layout == "BSHD" else 2
        self.chunks: list[torch.Tensor | None] = [None] * max_num_chunks
        self.chunk_state: list[ChunkState] = [ChunkState.EMPTY] * max_num_chunks
        self.quantized_spans: list[dict[str, Any]] = []
        self._span_by_chunk: list[dict[str, Any] | None] = [None] * max_num_chunks

    @property
    def kv_cache_size(self) -> int:
        return self.max_num_chunks * self.frame_seq_length

    def _token_to_chunk(self, token_index: int) -> int:
        if token_index % self.frame_seq_length:
            raise ValueError(
                f"token index {token_index} is not aligned to "
                f"{self.frame_seq_length}"
            )
        chunk = token_index // self.frame_seq_length
        if not 0 <= chunk <= self.max_num_chunks:
            raise ValueError(f"cache index exceeds capacity: {token_index}")
        return chunk

    def _shape(self) -> tuple[int, int, int, int]:
        if self.layout == "BSHD":
            return (
                self.batch_size,
                self.frame_seq_length,
                self.num_heads,
                self.head_dim,
            )
        return (
            self.batch_size,
            self.num_heads,
            self.frame_seq_length,
            self.head_dim,
        )

    def _remove_covered_spans(self, start_chunk: int, end_chunk: int) -> None:
        kept: list[dict[str, Any]] = []
        for span in self.quantized_spans:
            overlap = not (
                span["end_chunk"] <= start_chunk
                or span["start_chunk"] >= end_chunk
            )
            covered = (
                start_chunk <= span["start_chunk"]
                and end_chunk >= span["end_chunk"]
            )
            if overlap and not covered:
                raise ValueError("a packed TempoKV span cannot be partially overwritten")
            if not overlap:
                kept.append(span)
        self.quantized_spans = kept
        self._span_by_chunk = [None] * self.max_num_chunks
        for span in kept:
            for chunk in range(span["start_chunk"], span["end_chunk"]):
                self._span_by_chunk[chunk] = span

    def write(self, start_index: int, end_index: int, data: torch.Tensor) -> None:
        start_chunk = self._token_to_chunk(start_index)
        end_chunk = self._token_to_chunk(end_index)
        if data.ndim != 4 or data.shape[self.seq_dim] != end_index - start_index:
            raise ValueError("cache write tensor does not match the token range")
        expected = list(self._shape())
        expected[self.seq_dim] = end_index - start_index
        if tuple(data.shape) != tuple(expected):
            raise ValueError(
                f"cache write shape mismatch: expected {tuple(expected)}, "
                f"got {tuple(data.shape)}"
            )
        self._remove_covered_spans(start_chunk, end_chunk)
        for offset, chunk_index in enumerate(range(start_chunk, end_chunk)):
            token_start = offset * self.frame_seq_length
            token_end = token_start + self.frame_seq_length
            chunk = self.chunks[chunk_index]
            if chunk is None:
                chunk = torch.empty(self._shape(), dtype=self.dtype, device=self.device)
                self.chunks[chunk_index] = chunk
            if self.layout == "BSHD":
                chunk.copy_(data[:, token_start:token_end])
            else:
                chunk.copy_(data[:, :, token_start:token_end])
            self.chunk_state[chunk_index] = ChunkState.BF16

    def read(
        self,
        start_index: int,
        end_index: int,
        source_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._read_layout(
            start_index,
            end_index,
            output_layout=self.layout,
            source_tensor=source_tensor,
        )

    def read_bhsd(
        self,
        start_index: int,
        end_index: int,
        source_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._read_layout(
            start_index,
            end_index,
            output_layout="BHSD",
            source_tensor=source_tensor,
        )

    def _source_for_span(
        self,
        source_tensor: torch.Tensor | None,
        *,
        read_start_chunk: int,
        span: dict[str, Any],
        source_layout: str,
    ) -> torch.Tensor | None:
        if source_tensor is None:
            return None
        offset = (
            span["start_chunk"] - read_start_chunk
        ) * self.frame_seq_length
        length = (
            span["end_chunk"] - span["start_chunk"]
        ) * self.frame_seq_length
        seq_dim = 1 if source_layout == "BSHD" else 2
        if offset < 0 or offset + length > source_tensor.shape[seq_dim]:
            raise ValueError("cache read does not contain the complete K source span")
        if source_layout == "BSHD":
            return source_tensor[:, offset:offset + length].permute(0, 2, 1, 3)
        return source_tensor[:, :, offset:offset + length].contiguous()

    def _read_layout(
        self,
        start_index: int,
        end_index: int,
        *,
        output_layout: str,
        source_tensor: torch.Tensor | None,
    ) -> torch.Tensor:
        start_chunk = self._token_to_chunk(start_index)
        end_chunk = self._token_to_chunk(end_index)
        if start_chunk >= end_chunk:
            raise ValueError("cache read range must be non-empty")
        if (
            end_chunk - start_chunk == 1
            and self.chunk_state[start_chunk] == ChunkState.BF16
            and self.layout == output_layout
        ):
            chunk = self.chunks[start_chunk]
            if chunk is None:
                raise RuntimeError("BF16 cache chunk has no tensor")
            return chunk
        total_tokens = end_index - start_index
        if output_layout == "BSHD":
            result = torch.empty(
                (
                    self.batch_size,
                    total_tokens,
                    self.num_heads,
                    self.head_dim,
                ),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            result = torch.empty(
                (
                    self.batch_size,
                    self.num_heads,
                    total_tokens,
                    self.head_dim,
                ),
                device=self.device,
                dtype=self.dtype,
            )
        chunk_index = start_chunk
        while chunk_index < end_chunk:
            output_start = (
                chunk_index - start_chunk
            ) * self.frame_seq_length
            state = self.chunk_state[chunk_index]
            if state == ChunkState.BF16:
                part = self.chunks[chunk_index]
                if part is None:
                    raise RuntimeError("BF16 cache chunk has no tensor")
                if self.layout != output_layout:
                    part = part.permute(0, 2, 1, 3)
                output_end = output_start + self.frame_seq_length
                if output_layout == "BSHD":
                    result[:, output_start:output_end].copy_(part)
                else:
                    result[:, :, output_start:output_end].copy_(part)
                chunk_index += 1
                continue
            if state != ChunkState.QUANTIZED:
                raise ValueError(f"cache chunk {chunk_index} has not been written")
            span = self._span_by_chunk[chunk_index]
            if span is None:
                raise RuntimeError("packed cache span index is inconsistent")
            if span["start_chunk"] < start_chunk or span["end_chunk"] > end_chunk:
                raise ValueError("packed TempoKV spans must be read in full")
            source = self._source_for_span(
                source_tensor,
                read_start_chunk=start_chunk,
                span=span,
                source_layout=output_layout,
            )
            expected_tokens = (
                span["end_chunk"] - span["start_chunk"]
            ) * self.frame_seq_length
            output_end = output_start + expected_tokens
            # The normal attention path reads BSHD. For batch size one its
            # sequence slice is contiguous, so Triton can decode straight
            # into the final cache-read buffer without keeping per-span
            # outputs alive until a later torch.cat.
            direct_output = None
            if output_layout == "BSHD":
                candidate = result[:, output_start:output_end]
                if candidate.is_contiguous():
                    direct_output = candidate
            decoded = uncompress_single_kv_cache(
                span["quant_data"],
                source_tensor=source,
                output_tensor=direct_output,
            )
            expected_shape = (
                self.batch_size,
                self.num_heads,
                expected_tokens,
                self.head_dim,
            )
            if tuple(decoded.shape) != expected_shape:
                raise ValueError(
                    f"decoded TempoKV shape mismatch: expected {expected_shape}, "
                    f"got {tuple(decoded.shape)}"
                )
            if direct_output is None:
                if output_layout == "BSHD":
                    result[:, output_start:output_end].copy_(
                        decoded.permute(0, 2, 1, 3)
                    )
                else:
                    result[:, :, output_start:output_end].copy_(decoded)
            chunk_index = span["end_chunk"]
        return result

    def store_quantized(
        self, start_index: int, end_index: int, quant_data: Any
    ) -> None:
        start_chunk = self._token_to_chunk(start_index)
        end_chunk = self._token_to_chunk(end_index)
        if start_chunk >= end_chunk:
            raise ValueError("packed cache span must be non-empty")
        self._remove_covered_spans(start_chunk, end_chunk)
        span = {
            "start_chunk": start_chunk,
            "end_chunk": end_chunk,
            "quant_data": quant_data,
        }
        self.quantized_spans.append(span)
        for chunk_index in range(start_chunk, end_chunk):
            self.chunks[chunk_index] = None
            self.chunk_state[chunk_index] = ChunkState.QUANTIZED
            self._span_by_chunk[chunk_index] = span

    def clear(self) -> None:
        self.chunks = [None] * self.max_num_chunks
        self.chunk_state = [ChunkState.EMPTY] * self.max_num_chunks
        self.quantized_spans.clear()
        self._span_by_chunk = [None] * self.max_num_chunks


__all__ = [
    "ChunkState",
    "ChunkedKVCache",
    "set_quantized_decoder",
    "uncompress_single_kv_cache",
]
