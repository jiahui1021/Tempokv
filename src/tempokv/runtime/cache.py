"""Frame-aligned causal KV cache shared by the Wan-based experiments.

Each layer keeps K (normalized, pre-RoPE) or V in frame-sized chunks that are
``EMPTY``, dense ``BF16``, or ``QUANTIZED`` inside a packed TempoKV span.  RoPE
is applied at read time from each chunk's absolute frame position.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

import torch


class ChunkState(IntEnum):
    EMPTY = 0
    BF16 = 1
    QUANTIZED = 2


@dataclass(frozen=True)
class CacheSpan:
    """A run of resident chunks with consecutive absolute frames (one RoPE span)."""

    local_start_chunk: int
    global_start_frame: int
    num_frames: int

    @property
    def local_end_chunk(self) -> int:
        return self.local_start_chunk + self.num_frames

    @property
    def global_end_frame(self) -> int:
        return self.global_start_frame + self.num_frames


_quantized_decoder: Callable[..., torch.Tensor] | None = None


def set_quantized_decoder(decoder: Callable[..., torch.Tensor] | None) -> None:
    global _quantized_decoder
    _quantized_decoder = decoder


def uncompress_single_kv_cache(
    state: Any,
    *,
    source_tensor: torch.Tensor | None = None,
    output_tensor: torch.Tensor | None = None,
    output_layout: str = "BSHD",
    decoder: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    resolved_decoder = decoder if decoder is not None else _quantized_decoder
    if resolved_decoder is None:
        raise RuntimeError("no TempoKV decoder is installed for this cache span")
    kwargs: dict[str, Any] = {
        "source_tensor": source_tensor,
        "output_tensor": output_tensor,
    }
    # Keep the historical decoder callback contract unchanged for BSHD.  A
    # decoder only needs the new keyword when a BHSD target is requested.
    if output_layout != "BSHD":
        kwargs["output_layout"] = output_layout
    return resolved_decoder(state, **kwargs)


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
        quantized_decoder: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.frame_seq_length = frame_seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_num_chunks = max_num_chunks
        self.dtype = dtype
        self.device = device
        self.layout = layout
        self.cache_role = cache_role
        # Per-instance decoder, so a later pipeline cannot redirect old spans.
        self._quantized_decoder = quantized_decoder
        # The TempoKV bridge switches reconstructed K to the legacy complex64
        # cached-key RoPE without changing native/BF16 cache behavior.
        self.rope_precision = "native"
        self.seq_dim = 1 if layout == "BSHD" else 2
        self.chunks: list[torch.Tensor | None] = [None] * max_num_chunks
        self.chunk_state: list[ChunkState] = [ChunkState.EMPTY] * max_num_chunks
        self.quantized_spans: list[dict[str, Any]] = []
        self._span_by_chunk: list[dict[str, Any] | None] = [None] * max_num_chunks
        # Absolute frame of every occupied chunk (None = free).
        self.global_frames: list[int | None] = [None] * max_num_chunks

    def set_quantized_decoder(
        self, decoder: Callable[..., torch.Tensor] | None
    ) -> None:
        """Bind packed-span decoding to this cache instance."""
        self._quantized_decoder = decoder

    @property
    def kv_cache_size(self) -> int:
        return self.max_num_chunks * self.frame_seq_length

    def _token_to_chunk(self, token_index: int) -> int:
        if token_index % self.frame_seq_length:
            raise ValueError(
                f"token index {token_index} is not aligned to "
                f"{self.frame_seq_length}"
            )
        return token_index // self.frame_seq_length

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

    def write(
        self,
        start_index: int,
        end_index: int,
        data: torch.Tensor,
        *,
        global_start_frame: int | None = None,
    ) -> None:
        """Store BF16 chunks for one token range at ``global_start_frame``."""
        start_chunk = self._token_to_chunk(start_index)
        end_chunk = self._token_to_chunk(end_index)
        expected = list(self._shape())
        expected[self.seq_dim] = end_index - start_index
        if tuple(data.shape) != tuple(expected):
            raise ValueError(
                f"cache write shape mismatch: expected {tuple(expected)}, "
                f"got {tuple(data.shape)}"
            )
        if global_start_frame is None:
            global_start_frame = start_chunk
        self._remove_covered_spans(start_chunk, end_chunk)
        destinations: list[torch.Tensor] = []
        sources: list[torch.Tensor] = []
        for offset, chunk_index in enumerate(range(start_chunk, end_chunk)):
            token_start = offset * self.frame_seq_length
            token_end = token_start + self.frame_seq_length
            chunk = self.chunks[chunk_index]
            if chunk is None:
                chunk = torch.empty(self._shape(), dtype=self.dtype, device=self.device)
                self.chunks[chunk_index] = chunk
            if self.layout == "BSHD":
                source = data[:, token_start:token_end]
            else:
                source = data[:, :, token_start:token_end]
            destinations.append(chunk)
            sources.append(source)
            self.chunk_state[chunk_index] = ChunkState.BF16
            self.global_frames[chunk_index] = global_start_frame + offset
        if len(sources) == 1:
            destinations[0].copy_(sources[0])
        elif sources:
            torch._foreach_copy_(destinations, sources)

    def spans(self, start_index: int, end_index: int) -> list[CacheSpan]:
        """Group a token range into runs of consecutive absolute frames."""
        start_chunk = self._token_to_chunk(start_index)
        end_chunk = self._token_to_chunk(end_index)
        result: list[CacheSpan] = []
        for chunk_index in range(start_chunk, end_chunk):
            if self.chunk_state[chunk_index] == ChunkState.EMPTY:
                raise ValueError(f"cache chunk {chunk_index} has not been written")
            frame = self.global_frames[chunk_index]
            if frame is None:
                raise RuntimeError(
                    f"cache chunk {chunk_index} has no absolute frame position"
                )
            if result:
                last = result[-1]
                if (
                    last.local_end_chunk == chunk_index
                    and last.global_end_frame == frame
                ):
                    result[-1] = CacheSpan(
                        last.local_start_chunk,
                        last.global_start_frame,
                        last.num_frames + 1,
                    )
                    continue
            result.append(CacheSpan(chunk_index, frame, 1))
        return result

    def read(
        self,
        start_index: int,
        end_index: int,
        source_tensor: torch.Tensor | None = None,
        *,
        extra_channels: int = 0,
    ) -> torch.Tensor:
        """Decode a token range.

        ``extra_channels`` zero-pads the head dimension (for the Jensen correction)
        without a second cache-sized copy.
        """
        return self._read_layout(
            start_index,
            end_index,
            output_layout=self.layout,
            source_tensor=source_tensor,
            extra_channels=extra_channels,
        )

    def read_bhsd(
        self,
        start_index: int,
        end_index: int,
        source_tensor: torch.Tensor | None = None,
        *,
        extra_channels: int = 0,
    ) -> torch.Tensor:
        return self._read_layout(
            start_index,
            end_index,
            output_layout="BHSD",
            source_tensor=source_tensor,
            extra_channels=extra_channels,
        )

    def read_bhsd_independent(
        self,
        start_index: int,
        end_index: int,
        source_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return BHSD data that never aliases resident cache storage."""
        result = self.read_bhsd(
            start_index,
            end_index,
            source_tensor=source_tensor,
        )
        result_storage = result.untyped_storage()
        identity = (int(result_storage.data_ptr()), int(result_storage.nbytes()))
        start_chunk = self._token_to_chunk(start_index)
        end_chunk = self._token_to_chunk(end_index)
        for chunk in self.chunks[start_chunk:end_chunk]:
            if chunk is None:
                continue
            storage = chunk.untyped_storage()
            if identity == (int(storage.data_ptr()), int(storage.nbytes())):
                return result.clone(memory_format=torch.contiguous_format)
        return result

    def complete_span_read_start(self, start_index: int) -> int:
        """Move a requested start backward to the owning packed-span boundary."""
        start_chunk = self._token_to_chunk(start_index)
        if start_chunk == self.max_num_chunks:
            return start_index
        if self.chunk_state[start_chunk] != ChunkState.QUANTIZED:
            return start_index
        span = self._span_by_chunk[start_chunk]
        return span["start_chunk"] * self.frame_seq_length

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
        # Triton CrossKV takes explicit strides, so the source can stay a view.
        return source_tensor[:, :, offset:offset + length]

    def _read_layout(
        self,
        start_index: int,
        end_index: int,
        *,
        output_layout: str,
        source_tensor: torch.Tensor | None,
        extra_channels: int = 0,
    ) -> torch.Tensor:
        start_chunk = self._token_to_chunk(start_index)
        end_chunk = self._token_to_chunk(end_index)
        if (
            end_chunk - start_chunk == 1
            and self.chunk_state[start_chunk] == ChunkState.BF16
            and self.layout == output_layout
            and not extra_channels
        ):
            return self.chunks[start_chunk]
        total_tokens = end_index - start_index
        width = self.head_dim + extra_channels
        shape = (
            (self.batch_size, total_tokens, self.num_heads, width)
            if output_layout == "BSHD"
            else (self.batch_size, self.num_heads, total_tokens, width)
        )
        # Only the padding has to be cleared; every leading channel is written
        # by the decode below.
        if extra_channels:
            wide = torch.empty(shape, device=self.device, dtype=self.dtype)
            wide[..., self.head_dim:].zero_()
            result = wide[..., : self.head_dim]
        else:
            wide = result = torch.empty(shape, device=self.device, dtype=self.dtype)
        # Batch the dense-chunk copies into one foreach call.
        bf16_destinations: list[torch.Tensor] = []
        bf16_sources: list[torch.Tensor] = []
        chunk_index = start_chunk
        while chunk_index < end_chunk:
            output_start = (
                chunk_index - start_chunk
            ) * self.frame_seq_length
            state = self.chunk_state[chunk_index]
            if state == ChunkState.BF16:
                part = self.chunks[chunk_index]
                if self.layout != output_layout:
                    part = part.permute(0, 2, 1, 3)
                output_end = output_start + self.frame_seq_length
                if output_layout == "BSHD":
                    destination = result[:, output_start:output_end]
                else:
                    destination = result[:, :, output_start:output_end]
                bf16_destinations.append(destination)
                bf16_sources.append(part)
                chunk_index += 1
                continue
            if state != ChunkState.QUANTIZED:
                raise ValueError(f"cache chunk {chunk_index} has not been written")
            span = self._span_by_chunk[chunk_index]
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
            if output_layout == "BSHD":
                direct_output = result[:, output_start:output_end]
            else:
                direct_output = result[:, :, output_start:output_end]
            decoded = uncompress_single_kv_cache(
                span["quant_data"],
                source_tensor=source,
                output_tensor=direct_output,
                output_layout=output_layout,
                decoder=self._quantized_decoder,
            )
            if decoded.data_ptr() != direct_output.data_ptr():
                if output_layout == "BSHD":
                    direct_output.copy_(decoded.permute(0, 2, 1, 3))
                else:
                    direct_output.copy_(decoded)
            chunk_index = span["end_chunk"]
        if len(bf16_sources) == 1:
            bf16_destinations[0].copy_(bf16_sources[0])
        elif bf16_sources:
            torch._foreach_copy_(bf16_destinations, bf16_sources)
        return wide

    def read_step_sizes(
        self,
        start_index: int,
        end_index: int,
        *,
        group_size: int,
        output_layout: str | None = None,
        out: torch.Tensor | None = None,
        squared: bool = False,
    ) -> torch.Tensor:
        """Quantizer step sizes for a read range, aligned token-for-token with :meth:`read`.

        BF16 chunks report zero.  ``out`` fills a caller buffer and ``squared``
        returns the squared step the Jensen correction uses.
        """
        layout = output_layout or self.layout
        start_chunk = self._token_to_chunk(start_index)
        end_chunk = self._token_to_chunk(end_index)
        groups = self.head_dim // group_size
        total_tokens = end_index - start_index
        shape = (
            (self.batch_size, total_tokens, self.num_heads, groups)
            if layout == "BSHD"
            else (self.batch_size, self.num_heads, total_tokens, groups)
        )
        if out is None:
            result = torch.zeros(shape, device=self.device, dtype=torch.float32)
        else:
            out.zero_()
            result = out
        chunk_index = start_chunk
        while chunk_index < end_chunk:
            output_start = (chunk_index - start_chunk) * self.frame_seq_length
            if self.chunk_state[chunk_index] != ChunkState.QUANTIZED:
                chunk_index += 1
                continue
            span = self._span_by_chunk[chunk_index]
            if span["start_chunk"] < start_chunk or span["end_chunk"] > end_chunk:
                raise ValueError("packed TempoKV spans must be read in full")
            packed = span["quant_data"]
            # Self-Forcing's bridge wraps the PackedTensor in a dict.
            if isinstance(packed, dict):
                packed = next(v for v in packed.values() if hasattr(v, "step_sizes"))
            steps = packed.step_sizes()            # [B, H, S, G]
            output_end = output_start + steps.shape[2]
            if layout == "BSHD":
                result[:, output_start:output_end] = steps.permute(0, 2, 1, 3)
            else:
                result[:, :, output_start:output_end] = steps
            if squared:
                target = (
                    result[:, output_start:output_end]
                    if layout == "BSHD"
                    else result[:, :, output_start:output_end]
                )
                target.pow_(2)
            chunk_index = span["end_chunk"]
        return result

    def store_quantized(
        self, start_index: int, end_index: int, quant_data: Any
    ) -> None:
        start_chunk = self._token_to_chunk(start_index)
        end_chunk = self._token_to_chunk(end_index)
        self._remove_covered_spans(start_chunk, end_chunk)
        span = {
            "start_chunk": start_chunk,
            "end_chunk": end_chunk,
            "quant_data": quant_data,
        }
        self.quantized_spans.append(span)
        for chunk_index in range(start_chunk, end_chunk):
            # Absolute frame positions survive packing: only the payload
            # changes state, never the coordinate the chunk represents.
            self.chunks[chunk_index] = None
            self.chunk_state[chunk_index] = ChunkState.QUANTIZED
            self._span_by_chunk[chunk_index] = span

    def extend_quantized(self, start_index: int, end_index: int) -> None:
        """Extend an open packed span over newly finalized BF16 chunks (no new anchor)."""
        start_chunk = self._token_to_chunk(start_index)
        end_chunk = self._token_to_chunk(end_index)
        span = self._span_by_chunk[start_chunk]
        old_end = span["end_chunk"]
        for chunk_index in range(old_end, end_chunk):
            if self.chunk_state[chunk_index] != ChunkState.BF16:
                raise ValueError(f"cache chunk {chunk_index} is not finalized BF16")
        span["end_chunk"] = end_chunk
        for chunk_index in range(old_end, end_chunk):
            self.chunks[chunk_index] = None
            self.chunk_state[chunk_index] = ChunkState.QUANTIZED
            self._span_by_chunk[chunk_index] = span

    def bf16_resident_bytes(self) -> int:
        """Bytes held by dense chunks not covered by a packed span."""
        total = 0
        for chunk in self.chunks:
            if chunk is not None:
                total += chunk.numel() * chunk.element_size()
        return total

    def clear(self) -> None:
        self.chunks = [None] * self.max_num_chunks
        self.chunk_state = [ChunkState.EMPTY] * self.max_num_chunks
        self.quantized_spans.clear()
        self._span_by_chunk = [None] * self.max_num_chunks
        self.global_frames = [None] * self.max_num_chunks


__all__ = [
    "CacheSpan",
    "ChunkState",
    "ChunkedKVCache",
    "set_quantized_decoder",
    "uncompress_single_kv_cache",
]
