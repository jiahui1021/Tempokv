"""Streaming TempoKV encoder for causal autoregressive video caches.

A causal video model produces one clean block at a time, but the TempoKV codec
is a batch API: :meth:`TempoKVCodec.encode_key` starts a fresh anchor at the
front of whatever tensor it is given.  Encoding every finalized block on its own
would therefore re-anchor every few frames — with ``num_frame_per_block=3`` that
is one anchor per two residual frames, which both wastes bits and throws away
the temporal correlation TempoKV exists to exploit.

This encoder keeps finalized clean blocks in BF16 until a whole **span** of
frames is available, then packs the span in one call.  Inside a span the
predictor chain is continuous (``frame 0 -> 1 -> 2 -> 3 -> ...``) and every
prediction is made from decoder-visible reconstructed history, because the
codec's own closed loop produces it.  The most recent frames stay BF16, so the
frames the model attends to most heavily are never approximated.

Anchor policy
-------------
A new anchor is emitted only at a span boundary, and a span boundary is only
created by an explicit event:

* the encoder is constructed or :meth:`reset` is called (cache reset);
* :meth:`reset_span` is called (scene change, eviction gap, sink/recent
  discontinuity);
* ``span_frames`` finalized frames have accumulated.

Nothing is inferred from timesteps or block indices.  The pipeline states what
happened; the encoder never guesses.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch

from ..codec import PackedTensor, TempoKVCodec
from .metrics import TempoKVRuntimeReport, bf16_bytes


class StreamingTempoKVEncoder:
    """Pack finalized clean K/V into TempoKV spans across a streaming rollout."""

    def __init__(
        self,
        codec: TempoKVCodec,
        *,
        num_layers: int,
        frame_seq_length: int,
        span_frames: int,
        report: TempoKVRuntimeReport,
        wrap_state: Callable[[PackedTensor], Any] | None = None,
    ) -> None:
        if num_layers < 1:
            raise ValueError("streaming encoder needs at least one layer")
        if frame_seq_length < 1:
            raise ValueError("frame_seq_length must be positive")
        if span_frames < 2:
            raise ValueError(
                "a TempoKV span needs at least two frames: one anchor and one "
                "predicted residual"
            )
        self.codec = codec
        self.num_layers = num_layers
        self.frame_seq_length = frame_seq_length
        self.span_frames = span_frames
        self.report = report
        self._wrap_state = wrap_state or (lambda state: state)
        # First frame of the span currently being accumulated, and the first
        # frame not yet finalized. Both are absolute model-timeline positions.
        self._span_start_frame = 0
        self._finalized_frame = 0

    # -- state -----------------------------------------------------------

    @property
    def span_start_frame(self) -> int:
        return self._span_start_frame

    @property
    def finalized_frame(self) -> int:
        return self._finalized_frame

    @property
    def pending_frames(self) -> int:
        """Finalized frames still held in BF16, waiting for a full span."""
        return self._finalized_frame - self._span_start_frame

    def reset(self) -> None:
        """Forget all span state, as when the whole cache is cleared."""
        self._span_start_frame = 0
        self._finalized_frame = 0
        self.report.reset_cache()

    def reset_span(self, *, at_frame: int | None = None) -> None:
        """Force the next packed span to start a new anchor.

        Call this whenever the temporal chain is genuinely broken — an evicted
        interior range, a sink/recent boundary, a scene reset.  Any frames
        already accumulated for the current span stay BF16 and are dropped from
        the predictor chain rather than being predicted across the break.
        """
        self._span_start_frame = (
            self._finalized_frame if at_frame is None else int(at_frame)
        )

    # -- driving ---------------------------------------------------------

    def note_ephemeral_write(self, count: int = 1) -> None:
        """Record noisy denoising writes, which never enter a packed span."""
        self.report.ephemeral_writes += int(count)

    def finalize(
        self,
        caches: Sequence[Mapping[str, Any]],
        *,
        num_frames: int,
        flush: bool = True,
    ) -> int:
        """Accept ``num_frames`` of clean cache and pack any complete spans.

        ``caches`` is the per-layer list of ``{"k": ..., "v": ...}`` cache
        objects.  Returns the number of frames packed by this call.
        """
        if num_frames < 1:
            raise ValueError("finalize needs at least one frame")
        self._finalized_frame += int(num_frames)
        self.report.finalized_frames += int(num_frames)
        if not flush:
            return 0
        return self.flush(caches)

    def flush(self, caches: Sequence[Mapping[str, Any]], *, force: bool = False) -> int:
        """Pack every complete span; with ``force`` also pack a partial one."""
        packed_frames = 0
        while True:
            available = self.pending_frames
            if available >= self.span_frames:
                length = self.span_frames
            elif force and available >= 2:
                length = available
            else:
                break
            self._encode_span(caches, self._span_start_frame, length)
            self._span_start_frame += length
            packed_frames += length
        return packed_frames

    # -- encoding --------------------------------------------------------

    def _encode_span(
        self,
        caches: Sequence[Mapping[str, Any]],
        start_frame: int,
        num_frames: int,
    ) -> None:
        if len(caches) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} cache layers, got {len(caches)}"
            )
        token_start = start_frame * self.frame_seq_length
        token_end = token_start + num_frames * self.frame_seq_length
        with self.report.timer("encode_ms"), torch.no_grad():
            for layer_index, layer in enumerate(caches):
                keys = layer["k"].read_bhsd(token_start, token_end)
                packed_key, reconstructed_key = self.codec.encode_key(
                    keys,
                    layer=layer_index,
                    reuse_input=True,
                )
                layer["k"].store_quantized(
                    token_start, token_end, self._wrap_state(packed_key)
                )
                self.report.encode_key_calls += 1
                self.report.record_span(
                    ("k", layer_index, token_start),
                    compressed_bytes=packed_key.storage_bytes(),
                    uncompressed_bytes=bf16_bytes(tuple(packed_key.shape)),
                )
                del keys, packed_key

                # V is predicted from the *reconstructed* K the decoder will
                # see, never from the clean K, so encoder and decoder stay on
                # the same closed loop.
                values = layer["v"].read_bhsd(token_start, token_end)
                packed_value = self.codec.encode_value(
                    values,
                    reconstructed_key,
                    layer=layer_index,
                    reuse_input=True,
                )
                layer["v"].store_quantized(
                    token_start, token_end, self._wrap_state(packed_value)
                )
                self.report.encode_value_calls += 1
                self.report.record_span(
                    ("v", layer_index, token_start),
                    compressed_bytes=packed_value.storage_bytes(),
                    uncompressed_bytes=bf16_bytes(tuple(packed_value.shape)),
                )
                del values, packed_value, reconstructed_key


__all__ = ["StreamingTempoKVEncoder"]
