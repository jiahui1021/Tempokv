"""Streaming TempoKV encoder for causal autoregressive video caches.

The codec is a batch API: every call starts a fresh anchor at the front of the
tensor it is given.  Encoding each finalized block on its own would therefore
re-anchor every few frames, wasting bits and throwing away the temporal
correlation TempoKV exists to exploit.  This encoder accumulates finalized
blocks into one span so a span spends one anchor.

A new anchor is emitted only at a span boundary, and only an explicit event
creates one: construction, :meth:`reset`, or a full span.  Nothing is inferred
from timesteps or block indices.

Batch mode holds a span in BF16 until it is complete, then packs it in one
call.  Incremental mode appends each finalized frame to the open span once it
is ``bf16_lag_frames`` behind the newest one, through the same closed-loop
arithmetic, so the packed bytes are identical and only the lag stays dense.
The lag is a quality knob: with no lag every frame the model attends to is a
reconstruction, and in an autoregressive rollout that error compounds.
:meth:`flush` with ``force`` packs the lag tail once no later block reads it.
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
        if span_frames < 2:
            raise ValueError("a span needs an anchor and at least one residual")
        self.codec = codec
        self.incremental = incremental
        self.bf16_lag_frames = bf16_lag_frames
        # The rollout length, when the caller knows it: the last span is then
        # sized to the frames that remain rather than to a whole span.
        self.horizon_frames = horizon_frames
        self._event_hook = event_hook
        self.num_layers = num_layers
        self.frame_seq_length = frame_seq_length
        self.span_frames = span_frames
        self.report = report
        self._wrap_state = wrap_state or (lambda state: state)
        # Absolute model-timeline positions: the span being accumulated, the
        # first unfinalized frame, and (incremental only) the first unpacked
        # one.  Batch mode packs whole spans, so the last two coincide.
        self._span_start_frame = 0
        self._finalized_frame = 0
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

    def layer_history_bytes(self, layer_index: int) -> int:
        """Dense predictor history one layer's open span keeps alive."""
        if self._open_encoders is None:
            return 0
        return sum(
            encoder.history_bytes()
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

    # -- driving ---------------------------------------------------------

    def note_ephemeral_write(self, count: int = 1) -> None:
        """Record noisy denoising writes, which never enter a packed span."""
        self.report.ephemeral_writes += count

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
        self._finalized_frame += num_frames
        self.report.finalized_frames += num_frames
        # Measure before packing too: the moment before a span is compressed is
        # when the cache holds the most dense frames, and that is the peak.
        self._observe_resident(caches)
        if not flush:
            return 0
        return self.flush(caches)

    def flush(self, caches: Sequence[Mapping[str, Any]], *, force: bool = False) -> int:
        """Pack every complete span; with ``force`` also pack a partial one.

        In incremental mode this packs every frame older than the BF16 lag, and
        ``force`` packs the lag tail as well.
        """
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
            self._pack_units(caches, self._encoded_frame, count, opening=opening)
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

    @staticmethod
    def _read_for_encode(cache: Any, token_start: int, token_end: int) -> torch.Tensor:
        # Incremental encoding reconstructs in place, so it must never be handed
        # a view of a dense chunk that is still resident.
        reader = getattr(cache, "read_bhsd_independent", None)
        return (reader or cache.read_bhsd)(token_start, token_end)

    def _record(
        self, role: str, layer_index: int, token_start: int, packed: PackedTensor, *, grow: bool
    ) -> None:
        # An open span is re-recorded after every append, so only its growth may
        # reach the cumulative totals.
        install = self.report.update_span if grow else self.report.record_span
        install(
            (role, layer_index, token_start),
            compressed_bytes=packed.storage_bytes(),
            uncompressed_bytes=bf16_bytes(tuple(packed.shape)),
        )

    def encode_span(
        self,
        caches: Sequence[Mapping[str, Any]],
        start_frame: int,
        num_frames: int,
    ) -> None:
        """Pack one complete span in a single batch call."""
        token_start = start_frame * self.frame_seq_length
        token_end = token_start + num_frames * self.frame_seq_length
        # Slots are addressed by codec role, not by name: the primary tensor is
        # predicted from its own temporal history and the secondary one from the
        # primary's reconstruction, so the mirrored direction runs V then K
        # through this same loop.
        primary_role = self.codec.primary_role
        secondary_role = self.codec.secondary_role
        with self.report.timer("encode_ms"), torch.no_grad():
            for layer_index, layer in enumerate(caches):
                self._stamp_direction(layer)
                primary = self._read_for_encode(
                    layer[primary_role], token_start, token_end
                )
                packed_primary, reconstructed_primary = self.codec.encode_primary(
                    primary, layer=layer_index, reuse_input=True
                )
                layer[primary_role].store_quantized(
                    token_start, token_end, self._wrap_state(packed_primary)
                )
                self.report.encode_key_calls += 1
                self._record(
                    primary_role, layer_index, token_start, packed_primary, grow=False
                )
                del primary

                # The secondary tensor is predicted from the *reconstructed*
                # primary the decoder will see, never from the clean one.
                secondary = self._read_for_encode(
                    layer[secondary_role], token_start, token_end
                )
                packed_secondary = self.codec.encode_secondary(
                    secondary, reconstructed_primary, layer=layer_index, reuse_input=True
                )
                layer[secondary_role].store_quantized(
                    token_start, token_end, self._wrap_state(packed_secondary)
                )
                self.report.encode_value_calls += 1
                self._record(
                    secondary_role,
                    layer_index,
                    token_start,
                    packed_secondary,
                    grow=False,
                )
                del secondary, packed_secondary, reconstructed_primary

    def _stamp_direction(self, layer: Mapping[str, Any]) -> None:
        # The attention layer picks its read order from the cache entries alone,
        # so the codec direction has to travel with them.
        for role in ("k", "v"):
            layer[role].tempokv_predictor_direction = self.codec.config.predictor_direction

    def _pack_units(
        self,
        caches: Sequence[Mapping[str, Any]],
        start_frame: int,
        num_frames: int,
        *,
        opening: bool,
    ) -> None:
        """Open the span on these frames, or append them to the open one."""
        token_start = start_frame * self.frame_seq_length
        token_end = token_start + num_frames * self.frame_seq_length
        # The span is keyed and extended from where it started, which is this
        # call's own start only while the span is being opened.
        span_token_start = self._span_start_frame * self.frame_seq_length
        primary_role = self.codec.primary_role
        secondary_role = self.codec.secondary_role
        opened: list[dict[str, IncrementalTensorEncoder]] = []
        with self.report.timer("encode_ms"), torch.no_grad():
            for layer_index, layer in enumerate(caches):
                encoders = {} if opening else self._open_encoders[layer_index]
                primary = self._read_for_encode(
                    layer[primary_role], token_start, token_end
                )
                if opening:
                    self._stamp_direction(layer)
                    encoders[primary_role], reconstructed_primary = (
                        self.codec.begin_incremental_primary(
                            primary,
                            capacity_units=self._open_capacity,
                            layer=layer_index,
                            reuse_input=True,
                        )
                    )
                    layer[primary_role].store_quantized(
                        token_start,
                        token_end,
                        self._wrap_state(encoders[primary_role].packed),
                    )
                else:
                    reconstructed_primary = encoders[primary_role].append(
                        primary, reuse_input_for_reconstruction=True
                    )
                    layer[primary_role].extend_quantized(span_token_start, token_end)
                self.report.encode_key_calls += 1
                self._record(
                    primary_role,
                    layer_index,
                    span_token_start,
                    encoders[primary_role].packed,
                    grow=not opening,
                )
                del primary

                secondary = self._read_for_encode(
                    layer[secondary_role], token_start, token_end
                )
                if opening:
                    encoders[secondary_role] = self.codec.begin_incremental_secondary(
                        secondary,
                        reconstructed_primary,
                        capacity_units=self._open_capacity,
                        layer=layer_index,
                        reuse_input=True,
                    )
                    layer[secondary_role].store_quantized(
                        token_start,
                        token_end,
                        self._wrap_state(encoders[secondary_role].packed),
                    )
                    opened.append(encoders)
                else:
                    encoder = encoders[secondary_role]
                    encoder.append(
                        secondary,
                        source=(
                            None
                            if encoder.packed.predictor_kind == "temporal"
                            else reconstructed_primary
                        ),
                        reuse_input_for_reconstruction=True,
                    )
                    layer[secondary_role].extend_quantized(span_token_start, token_end)
                self.report.encode_value_calls += 1
                self._record(
                    secondary_role,
                    layer_index,
                    span_token_start,
                    encoders[secondary_role].packed,
                    grow=not opening,
                )
                del secondary, reconstructed_primary
        if opening:
            self._open_encoders = opened


__all__ = ["StreamingTempoKVEncoder"]
