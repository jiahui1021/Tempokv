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

Incremental mode
----------------
With ``incremental=True`` the span is not held back until it is complete.
Each finalized frame is appended to the open span once it is
``bf16_lag_frames`` behind the newest finalized frame, through the same
closed-loop arithmetic the batch codec uses, so the packed result is
bit-identical to encoding the span at once.  Only the lag stays dense, instead
of up to a whole span.  The lag is a quality knob: with no lag every frame the
model attends to is a reconstruction from the next block on, and in an
autoregressive rollout that error compounds.  :meth:`flush` with ``force``
packs the remaining lag tail once no later block will read it.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

import torch

from ..codec import IncrementalTensorEncoder, PackedTensor, TempoKVCodec
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
        incremental: bool = False,
        bf16_lag_frames: int = 0,
        horizon_frames: int | None = None,
        event_hook: Callable[[dict[str, Any]], None] | None = None,
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
        if bf16_lag_frames < 0:
            raise ValueError("bf16_lag_frames must not be negative")
        if horizon_frames is not None and horizon_frames < 1:
            raise ValueError("horizon_frames must be positive")
        self.codec = codec
        self.incremental = bool(incremental)
        self.bf16_lag_frames = int(bf16_lag_frames)
        # The rollout length, when the caller knows it.  The last span is then
        # sized to the frames that remain instead of a whole span, so its
        # residual buffers never grow past what will actually be written.
        self.horizon_frames = horizon_frames
        # Called after every incremental begin/append with what was packed.
        self._event_hook = event_hook
        self.num_layers = num_layers
        self.frame_seq_length = frame_seq_length
        self.span_frames = span_frames
        self.report = report
        self._wrap_state = wrap_state or (lambda state: state)
        # First frame of the span currently being accumulated, and the first
        # frame not yet finalized. Both are absolute model-timeline positions.
        self._span_start_frame = 0
        self._finalized_frame = 0
        # Incremental mode only: the first frame not yet packed, and the open
        # span's per-layer encoders.  In batch mode every packed frame belongs
        # to a closed span, so ``_span_start_frame`` already marks both.
        self._encoded_frame = 0
        self._open_units = 0
        self._open_capacity = 0
        self._open_encoders: list[dict[str, IncrementalTensorEncoder]] | None = None

    # -- state -----------------------------------------------------------

    @property
    def span_start_frame(self) -> int:
        return self._span_start_frame

    @property
    def finalized_frame(self) -> int:
        return self._finalized_frame

    @property
    def pending_frames(self) -> int:
        """Finalized frames still held in BF16, waiting to be packed."""
        packed_until = (
            self._encoded_frame if self.incremental else self._span_start_frame
        )
        return self._finalized_frame - packed_until

    @property
    def encoded_frame(self) -> int:
        """First frame not yet packed (incremental mode)."""
        return self._encoded_frame

    @property
    def open_units(self) -> int:
        """Frames packed into the span that is still open."""
        return self._open_units

    @property
    def open_capacity(self) -> int:
        """Frames the open span will hold before it closes."""
        return self._open_capacity

    def layer_history_bytes(self, layer_index: int) -> int:
        """Dense predictor history one layer's open span keeps alive."""
        if self._open_encoders is None:
            return 0
        return sum(
            int(encoder.history_bytes())
            for encoder in self._open_encoders[layer_index].values()
        )

    def history_bytes(self) -> int:
        """Dense predictor history the open incremental span keeps alive."""
        if self._open_encoders is None:
            return 0
        return sum(
            self.layer_history_bytes(index)
            for index in range(len(self._open_encoders))
        )

    def _close_open_span(self) -> None:
        self._open_encoders = None
        self._open_units = 0
        self._open_capacity = 0

    def reset(self) -> None:
        """Forget all span state, as when the whole cache is cleared."""
        self._span_start_frame = 0
        self._finalized_frame = 0
        self._encoded_frame = 0
        self._close_open_span()
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
        self._encoded_frame = self._span_start_frame
        self._close_open_span()

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
        self._validate_caches(caches)
        self._finalized_frame += int(num_frames)
        self.report.finalized_frames += int(num_frames)
        # Measure before packing as well: the moment just before a span is
        # compressed is when the cache holds the most dense frames, and that is
        # the peak a memory claim has to survive.
        self._observe_resident(caches)
        if not flush:
            return 0
        return self.flush(caches)

    def _validate_caches(self, caches: Sequence[Mapping[str, Any]]) -> None:
        if len(caches) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} cache layers, got {len(caches)}"
            )
        for layer_index, layer in enumerate(caches):
            for role in ("k", "v"):
                if role not in layer:
                    raise ValueError(
                        f"cache layer {layer_index} is missing {role.upper()}"
                    )
                cache = layer[role]
                methods = ["bf16_resident_bytes", "read_bhsd", "store_quantized"]
                if self.incremental:
                    methods.append("extend_quantized")
                for method in methods:
                    if not callable(getattr(cache, method, None)):
                        raise TypeError(
                            f"cache layer {layer_index} {role.upper()} does not "
                            f"support {method}"
                        )

    def flush(self, caches: Sequence[Mapping[str, Any]], *, force: bool = False) -> int:
        """Pack every complete span; with ``force`` also pack a partial one.

        In incremental mode this packs every frame older than the BF16 lag, and
        ``force`` packs the lag tail as well.
        """
        self._validate_caches(caches)
        if self.incremental:
            return self._flush_incremental(caches, force=force)
        packed_frames = 0
        while True:
            available = self.pending_frames
            if available >= self.span_frames:
                length = self.span_frames
            elif force and available >= 2:
                length = available
            else:
                break
            self.encode_span(caches, self._span_start_frame, length)
            self._span_start_frame += length
            packed_frames += length
        if packed_frames:
            self._observe_resident(caches)
        return packed_frames

    def _flush_incremental(
        self, caches: Sequence[Mapping[str, Any]], *, force: bool
    ) -> int:
        if force:
            # Nothing reads the cache after a forced flush, so a finalized
            # frame past the horizon is simply left dense.
            target = self._finalized_frame
            if self.horizon_frames is not None:
                target = min(target, self.horizon_frames)
        else:
            target = max(
                self._encoded_frame,
                self._finalized_frame - self.bf16_lag_frames,
            )
        packed_frames = 0
        while self._encoded_frame < target:
            opening = self._open_encoders is None
            if opening:
                start = self._encoded_frame
                capacity = self.span_frames
                if self.horizon_frames is not None:
                    if start >= self.horizon_frames:
                        raise ValueError(
                            "incremental TempoKV write exceeds the configured "
                            f"{self.horizon_frames}-frame horizon"
                        )
                    capacity = min(capacity, self.horizon_frames - start)
                self._span_start_frame = start
                self._open_units = 0
                self._open_capacity = capacity
            count = min(
                self._open_capacity - self._open_units,
                target - self._encoded_frame,
            )
            started = time.perf_counter()
            if opening:
                self._begin_span(caches, self._encoded_frame, count)
            else:
                self._append_span(caches, self._encoded_frame, count)
            self._open_units += count
            span_closed = self._open_units == self._open_capacity
            if self._event_hook is not None:
                token_start = self._encoded_frame * self.frame_seq_length
                self._event_hook(
                    {
                        "tokens_start": int(token_start),
                        "tokens_end": int(
                            token_start + count * self.frame_seq_length
                        ),
                        "num_tokens": int(count * self.frame_seq_length),
                        "duration_ms": (time.perf_counter() - started) * 1000.0,
                        "codec": "TempoKV-incremental",
                        "span_start_frame": int(self._span_start_frame),
                        "span_units": int(self._open_units),
                        "span_capacity": int(self._open_capacity),
                        "anchor": bool(opening),
                        "span_closed": bool(span_closed),
                        "finalized_frames": int(self._finalized_frame),
                        "forced": bool(force),
                    }
                )
            self._encoded_frame += count
            packed_frames += count
            if span_closed:
                self._close_open_span()
                self._span_start_frame = self._encoded_frame
        if packed_frames:
            self._observe_resident(caches)
        return packed_frames

    def _observe_resident(self, caches: Sequence[Mapping[str, Any]]) -> None:
        # The open span's predictor history is dense BF16 too, and it is
        # allocated for as long as the span stays open.
        self.report.observe_resident_bf16_bytes(
            sum(
                layer[role].bf16_resident_bytes()
                for layer in caches
                for role in ("k", "v")
            )
            + self.history_bytes()
        )

    # -- encoding --------------------------------------------------------

    def encode_span(
        self,
        caches: Sequence[Mapping[str, Any]],
        start_frame: int,
        num_frames: int,
    ) -> None:
        """Pack one complete span in a single batch call (batch schedule)."""
        token_start = start_frame * self.frame_seq_length
        token_end = token_start + num_frames * self.frame_seq_length
        with self.report.timer("encode_ms"), torch.no_grad():
            # Slots are addressed by codec role rather than by name.  The
            # primary tensor is the one predicted from its own temporal
            # history; the secondary tensor is projected from the primary's
            # reconstruction.  Forward runs K then V, and the mirrored
            # direction runs V then K through this same loop.
            primary_role = self.codec.primary_role
            secondary_role = self.codec.secondary_role
            direction = self.codec.config.predictor_direction
            for layer_index, layer in enumerate(caches):
                # The attention layer decides its read order from the cache
                # entries alone, so the direction has to travel with them.
                for role in ("k", "v"):
                    layer[role].tempokv_predictor_direction = direction
                primary = self._read_for_encode(
                    layer[primary_role], token_start, token_end
                )
                packed_primary, reconstructed_primary = self.codec.encode_primary(
                    primary,
                    layer=layer_index,
                    reuse_input=True,
                )
                layer[primary_role].store_quantized(
                    token_start, token_end, self._wrap_state(packed_primary)
                )
                self.report.encode_key_calls += 1
                self.report.record_span(
                    (primary_role, layer_index, token_start),
                    compressed_bytes=packed_primary.storage_bytes(),
                    uncompressed_bytes=bf16_bytes(tuple(packed_primary.shape)),
                )
                del primary, packed_primary

                # The secondary tensor is predicted from the *reconstructed*
                # primary the decoder will see, never from the clean one, so
                # encoder and decoder stay on the same closed loop.
                secondary = self._read_for_encode(
                    layer[secondary_role], token_start, token_end
                )
                packed_secondary = self.codec.encode_secondary(
                    secondary,
                    reconstructed_primary,
                    layer=layer_index,
                    reuse_input=True,
                )
                layer[secondary_role].store_quantized(
                    token_start, token_end, self._wrap_state(packed_secondary)
                )
                self.report.encode_value_calls += 1
                self.report.record_span(
                    (secondary_role, layer_index, token_start),
                    compressed_bytes=packed_secondary.storage_bytes(),
                    uncompressed_bytes=bf16_bytes(tuple(packed_secondary.shape)),
                )
                del secondary, packed_secondary, reconstructed_primary


    @staticmethod
    def _read_for_encode(cache: Any, token_start: int, token_end: int) -> torch.Tensor:
        # Incremental encoding reconstructs its input in place, so it must
        # never be handed a view of a dense chunk that is still resident.
        reader = getattr(cache, "read_bhsd_independent", None)
        if callable(reader):
            return reader(token_start, token_end)
        return cache.read_bhsd(token_start, token_end)

    def _record_open_span(
        self, role: str, layer_index: int, span_token_start: int, packed: PackedTensor
    ) -> None:
        self.report.update_span(
            (role, layer_index, span_token_start),
            compressed_bytes=packed.storage_bytes(),
            uncompressed_bytes=bf16_bytes(tuple(packed.shape)),
        )

    def _begin_span(
        self,
        caches: Sequence[Mapping[str, Any]],
        start_frame: int,
        num_frames: int,
    ) -> None:
        token_start = start_frame * self.frame_seq_length
        token_end = token_start + num_frames * self.frame_seq_length
        primary_role = self.codec.primary_role
        secondary_role = self.codec.secondary_role
        direction = self.codec.config.predictor_direction
        encoders: list[dict[str, IncrementalTensorEncoder]] = []
        with self.report.timer("encode_ms"), torch.no_grad():
            for layer_index, layer in enumerate(caches):
                for role in ("k", "v"):
                    layer[role].tempokv_predictor_direction = direction
                primary = self._read_for_encode(
                    layer[primary_role], token_start, token_end
                )
                primary_encoder, reconstructed_primary = (
                    self.codec.begin_incremental_primary(
                        primary,
                        capacity_units=self._open_capacity,
                        layer=layer_index,
                        reuse_input=True,
                    )
                )
                layer[primary_role].store_quantized(
                    token_start, token_end, self._wrap_state(primary_encoder.packed)
                )
                self.report.encode_key_calls += 1
                self._record_open_span(
                    primary_role, layer_index, token_start, primary_encoder.packed
                )
                del primary

                secondary = self._read_for_encode(
                    layer[secondary_role], token_start, token_end
                )
                secondary_encoder = self.codec.begin_incremental_secondary(
                    secondary,
                    reconstructed_primary,
                    capacity_units=self._open_capacity,
                    layer=layer_index,
                    reuse_input=True,
                )
                layer[secondary_role].store_quantized(
                    token_start, token_end, self._wrap_state(secondary_encoder.packed)
                )
                self.report.encode_value_calls += 1
                self._record_open_span(
                    secondary_role, layer_index, token_start, secondary_encoder.packed
                )
                encoders.append(
                    {primary_role: primary_encoder, secondary_role: secondary_encoder}
                )
                del secondary, reconstructed_primary
        self._open_encoders = encoders

    def _append_span(
        self,
        caches: Sequence[Mapping[str, Any]],
        start_frame: int,
        num_frames: int,
    ) -> None:
        assert self._open_encoders is not None
        token_start = start_frame * self.frame_seq_length
        token_end = token_start + num_frames * self.frame_seq_length
        span_token_start = self._span_start_frame * self.frame_seq_length
        primary_role = self.codec.primary_role
        secondary_role = self.codec.secondary_role
        with self.report.timer("encode_ms"), torch.no_grad():
            for layer_index, layer in enumerate(caches):
                encoders = self._open_encoders[layer_index]
                primary = self._read_for_encode(
                    layer[primary_role], token_start, token_end
                )
                reconstructed_primary = encoders[primary_role].append(
                    primary, reuse_input_for_reconstruction=True
                )
                layer[primary_role].extend_quantized(span_token_start, token_end)
                self.report.encode_key_calls += 1
                self._record_open_span(
                    primary_role,
                    layer_index,
                    span_token_start,
                    encoders[primary_role].packed,
                )
                del primary

                secondary = self._read_for_encode(
                    layer[secondary_role], token_start, token_end
                )
                secondary_encoder = encoders[secondary_role]
                secondary_encoder.append(
                    secondary,
                    source=(
                        None
                        if secondary_encoder.packed.predictor_kind == "temporal"
                        else reconstructed_primary
                    ),
                    reuse_input_for_reconstruction=True,
                )
                layer[secondary_role].extend_quantized(span_token_start, token_end)
                self.report.encode_value_calls += 1
                self._record_open_span(
                    secondary_role,
                    layer_index,
                    span_token_start,
                    secondary_encoder.packed,
                )
                del secondary, reconstructed_primary


__all__ = ["StreamingTempoKVEncoder"]
