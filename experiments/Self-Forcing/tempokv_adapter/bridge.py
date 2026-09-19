from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any, Callable
import torch

from tempokv.codec import PackedTensor, TempoKVCodec, decode_tensor
from tempokv.integration import load_codec_from_env

# This module is loaded both as ``bridge`` off the adapter directory and by
# file path from the test suite, so it puts its own directory on the path
# rather than relying on whoever imported it having done so.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import efficiency_record as efficiency


_TEMPOKV_STATE_KEY = "_tempokv_packed"


def wrap_cache_state(state: PackedTensor) -> dict[str, Any]:
    """Store one TempoKV state in the bundled Self-Forcing cache."""
    if not isinstance(state, PackedTensor):
        raise TypeError(f"expected PackedTensor, got {type(state).__name__}")
    return {
        _TEMPOKV_STATE_KEY: state,
        "predictor_kind": state.predictor_kind,
        "method": "tempokv",
    }


def _unwrap_cache_state(state: Any) -> PackedTensor | None:
    if isinstance(state, PackedTensor):
        return state
    if not isinstance(state, dict):
        return None
    packed = state.get(_TEMPOKV_STATE_KEY)
    return packed if isinstance(packed, PackedTensor) else None


def decode_cache_state(
    state: Any,
    *,
    source_tensor: torch.Tensor | None = None,
    output_tensor: torch.Tensor | None = None,
    output_layout: str = "BSHD",
    codec: TempoKVCodec,
    fallback: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Decode a TempoKV span or delegate an unrecognized cache state."""
    packed = _unwrap_cache_state(state)
    if packed is None:
        if fallback is None:
            raise TypeError(f"unsupported cache state: {type(state).__name__}")
        fallback_kwargs = {
            "source_tensor": source_tensor,
            "output_tensor": output_tensor,
        }
        if output_layout != "BSHD":
            fallback_kwargs["output_layout"] = output_layout
        return fallback(state, **fallback_kwargs)
    # Every route decodes straight into the caller's cache buffer.  Forcing a
    # standalone BHSD result here made the cache copy the whole span back, and
    # forcing a contiguous K source copied it forward, on top of the decode.
    decode_kwargs = {
        "source": source_tensor,
        "output_tensor": output_tensor,
        "use_triton": codec.config.use_triton,
        "strict_triton": codec.config.strict_triton,
    }
    if output_layout != "BSHD":
        decode_kwargs["output_layout"] = output_layout
    return decode_tensor(packed, **decode_kwargs)


def install(runtime_root: str | Path) -> type:
    """Install TempoKV into the Self-Forcing runtime bundled in this repo."""
    runtime_root = Path(runtime_root).resolve()
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))

    pipeline_package = importlib.import_module("pipeline")
    pipeline_module = importlib.import_module("pipeline.causal_inference")
    base = pipeline_module.CausalInferencePipeline
    cache_module = sys.modules[pipeline_module.ChunkedKVCache.__module__]
    fallback_decode = getattr(cache_module, "_quantized_decoder", None)

    class TempoKVCausalInferencePipeline(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.codec_shape = SimpleNamespace(
                frame_seq_length=self.frame_seq_length,
                num_heads=getattr(self, "num_heads", 12),
                head_dim=getattr(self, "head_dim", 128),
            )
            # TempoKV intentionally keeps the pre-RoPE, frame-addressable
            # cache so finalized spans can be predicted and re-rotated at
            # absolute positions during reads.  The BF16 baseline gets the
            # dense window-sized cache instead.
            self._uses_pre_rope_cache = True
            self.tempokv_codec = load_codec_from_env()
            config = self.tempokv_codec.config
            self._tempokv_rope_precision = "complex64"
            self._tempokv_span_frames = int(
                os.environ.get("TEMPOKV_SPAN_FRAMES", "24")
            )
            # The rollout length, not the attention window.  These coincide by
            # default, but the TEMPOKV_ATTENTION_FRAMES ablation narrows
            # TEMPOKV_NUM_FRAMES to make a run comparable against a baseline
            # with a smaller window, and the rollout still runs to the full
            # horizon.  Reading the window here would size the spans against
            # the wrong number and reject every write past it.
            self._tempokv_total_frames = int(
                os.environ.get(
                    "TEMPOKV_HORIZON_FRAMES",
                    os.environ.get("TEMPOKV_NUM_FRAMES", "0"),
                )
            )
            if self._tempokv_span_frames < 2:
                raise ValueError("TEMPOKV_SPAN_FRAMES must be at least 2")
            if self._tempokv_total_frames < 1:
                raise ValueError("TEMPOKV_HORIZON_FRAMES must be positive")
            # Span-mean K needs the final mean before the first residual can be
            # encoded, and innovation V needs a continuation-aware K innovation
            # source.  Preserve their existing batch behavior until those
            # experimental paths gain dedicated incremental state.
            self._incremental_tempokv = bool(
                os.environ.get("TEMPOKV_INCREMENTAL", "1") != "0"
                and hasattr(self.tempokv_codec, "begin_incremental_primary")
                and hasattr(self.tempokv_codec, "begin_incremental_secondary")
                and not bool(getattr(self.tempokv_codec, "k_channel_mean", False))
                and getattr(config, "v_predictor_mode", "cross_kv") != "innovation"
            )
            # Incremental encoding decides *when* a finalized frame stops being
            # readable as exact BF16, and that is a quality knob, not an
            # implementation detail.  The batch schedule left every frame BF16
            # until a whole span closed, so the newest finalized block was
            # always exact and short rollouts were never compressed at all.
            # Encoding each block the moment it is finalized removes that
            # window entirely: from the second block on, every key the model
            # attends to is a reconstruction, and in a self-forcing rollout
            # that error compounds.  The lag restores a bounded exact tail --
            # a frame is encoded only once it is this many frames behind the
            # newest finalized frame -- and costs exactly that many frames of
            # dense BF16.  One model block is the cheapest setting that keeps
            # the immediately preceding block exact.
            self._tempokv_bf16_lag_frames = int(
                os.environ.get(
                    "TEMPOKV_BF16_LAG_FRAMES",
                    str(getattr(self, "num_frame_per_block", 1)),
                )
            )
            if self._tempokv_bf16_lag_frames < 0:
                raise ValueError("TEMPOKV_BF16_LAG_FRAMES must not be negative")
            self._tempokv_finalized_frames = 0
            self._tempokv_encoded_frames = 0
            print(
                "[TempoKV] span encoding mode: "
                + (
                    "INCREMENTAL"
                    if self._incremental_tempokv
                    else "BATCHED"
                )
                + f", span_frames={self._tempokv_span_frames}"
                + f", bf16_lag_frames={self._tempokv_bf16_lag_frames}"
                + f", horizon_frames={self._tempokv_total_frames}",
                flush=True,
            )
            self._tempokv_open_start_frame: int | None = None
            self._tempokv_open_units = 0
            self._tempokv_open_capacity = 0
            self._tempokv_open_encoders: list[dict[str, Any]] | None = None
            self._apply_tempokv_cache_settings()
            if config.stride != self.frame_seq_length:
                raise ValueError(
                    f"TempoKV stride {config.stride} does not match Self-Forcing "
                    f"frame size {self.frame_seq_length}"
                )
            if config.reconstruction_dtype != "bfloat16":
                raise ValueError("Self-Forcing requires bfloat16 reconstruction")
            if (
                getattr(config, "v_predictor_mode", "cross_kv")
                not in {"temporal", "identity", "taylor", "taylor2", "direct"}
                and getattr(config, "v_history_length", 0)
                and not self.tempokv_codec.v_parameters.history_diagonal
            ):
                raise ValueError(
                    "Self-Forcing requires fast diagonal V history parameters; "
                    "refit the selected V parameter artifact with the current fitter"
                )
            if getattr(self.quant_config, "kv_cache_cpu_offload", False):
                raise ValueError("TempoKV Self-Forcing bridge does not support cache offload")
            self._patch_attention_windows()

            def decode_with_codec(
                state: Any,
                source_tensor: torch.Tensor | None = None,
                output_tensor: torch.Tensor | None = None,
                output_layout: str = "BSHD",
            ) -> torch.Tensor:
                return decode_cache_state(
                    state,
                    source_tensor=source_tensor,
                    output_tensor=output_tensor,
                    output_layout=output_layout,
                    codec=self.tempokv_codec,
                    fallback=fallback_decode,
                )

            cache_module.set_quantized_decoder(decode_with_codec)

        def _apply_tempokv_cache_settings(self) -> None:
            """Stamp the settings the attention layer has to read back.

            The attention code decides which cache slot to decode first, and it
            only has the cache entries to go on, so the direction travels with
            them.  It is stamped on both slots because either one can be the
            tensor that leads.
            """
            if self.kv_cache1 is None:
                return
            # Read defensively, as this bridge already does for the other
            # codec settings, so a config predating the field still installs.
            direction = getattr(
                self.tempokv_codec.config, "predictor_direction", "k_to_v"
            )
            # Same defensive read: a config predating the correction installs
            # with it off, which is also the shipped default.
            config = self.tempokv_codec.config
            jensen = bool(getattr(config, "jensen_correction", False))
            group_size = int(getattr(config, "group_size", 64))
            for layer in self.kv_cache1:
                layer["k"].rope_precision = self._tempokv_rope_precision
                # The correction reads the K cache's step sizes, so only the K
                # slot needs to carry it.
                layer["k"].tempokv_jensen_correction = jensen
                layer["k"].tempokv_group_size = group_size
                # Ablation switch, read from the environment rather than the
                # codec config: it changes no stored value and exists to tell
                # the exact correction apart from the cheap approximation.
                layer["k"].tempokv_jensen_cache_mean = (
                    os.environ.get("TEMPOKV_JENSEN_CACHE_MEAN") == "1"
                )
                for role in ("k", "v"):
                    layer[role].tempokv_predictor_direction = direction

        def _initialize_kv_cache(self, *args: Any, **kwargs: Any) -> Any:
            result = super()._initialize_kv_cache(*args, **kwargs)
            self._apply_tempokv_cache_settings()
            return result

        def reset_quantization_state(self) -> None:
            self._tempokv_finalized_frames = 0
            self._tempokv_encoded_frames = 0
            self._tempokv_open_start_frame = None
            self._tempokv_open_units = 0
            self._tempokv_open_capacity = 0
            self._tempokv_open_encoders = None

        def _patch_attention_windows(self) -> None:
            # TempoKV takes over cache compression, so the model must keep the
            # full sequence span readable instead of capping attention at the
            # 21-frame default window. The target frame count arrives through
            # TEMPOKV_NUM_FRAMES because omegaconf args lack num_output_frames.
            # A bounded local attention window (with or without a sink) is
            # legitimate: the causal model reads the full resident cache and
            # slices its attention window afterward, so TempoKV spans are never
            # cut.
            frames = int(os.environ.get("TEMPOKV_NUM_FRAMES", "0"))
            if frames <= 0:
                raise ValueError("TEMPOKV_NUM_FRAMES must be set for TempoKV runs")
            target = frames * self.frame_seq_length
            for block in self.generator.model.blocks:
                if hasattr(block.self_attn, "max_attention_size"):
                    block.self_attn.max_attention_size = target

        def _incremental_history_bytes(self) -> int:
            if self._tempokv_open_encoders is None:
                return 0
            return sum(
                int(encoder.history_bytes())
                for layer in self._tempokv_open_encoders
                for encoder in layer.values()
            )

        @staticmethod
        def _read_bhsd_for_encode(
            cache: Any, token_start: int, token_end: int
        ) -> torch.Tensor:
            reader = getattr(cache, "read_bhsd_independent", None)
            if callable(reader):
                return reader(token_start, token_end)
            return cache.read_bhsd(token_start, token_end)

        def _begin_incremental_segment(
            self,
            token_start: int,
            token_end: int,
        ) -> None:
            codec = self.tempokv_codec
            primary_role = codec.primary_role
            secondary_role = codec.secondary_role
            encoders: list[dict[str, Any]] = []
            for layer_index, layer in enumerate(self.kv_cache1):
                primary = self._read_bhsd_for_encode(
                    layer[primary_role], token_start, token_end
                )
                primary_encoder, reconstructed_primary = (
                    codec.begin_incremental_primary(
                        primary,
                        capacity_units=self._tempokv_open_capacity,
                        layer=layer_index,
                        reuse_input=True,
                    )
                )
                layer[primary_role].store_quantized(
                    token_start,
                    token_end,
                    wrap_cache_state(primary_encoder.packed),
                )
                del primary

                secondary = self._read_bhsd_for_encode(
                    layer[secondary_role], token_start, token_end
                )
                secondary_encoder = codec.begin_incremental_secondary(
                    secondary,
                    reconstructed_primary,
                    capacity_units=self._tempokv_open_capacity,
                    layer=layer_index,
                    reuse_input=True,
                )
                layer[secondary_role].store_quantized(
                    token_start,
                    token_end,
                    wrap_cache_state(secondary_encoder.packed),
                )
                encoders.append(
                    {
                        primary_role: primary_encoder,
                        secondary_role: secondary_encoder,
                    }
                )
                del secondary, reconstructed_primary
            self._tempokv_open_encoders = encoders

        def _append_incremental_segment(
            self,
            token_start: int,
            token_end: int,
        ) -> None:
            if (
                self._tempokv_open_encoders is None
                or self._tempokv_open_start_frame is None
            ):
                raise RuntimeError("incremental TempoKV span is not open")
            codec = self.tempokv_codec
            primary_role = codec.primary_role
            secondary_role = codec.secondary_role
            span_token_start = (
                self._tempokv_open_start_frame * self.frame_seq_length
            )
            for layer_index, layer in enumerate(self.kv_cache1):
                encoders = self._tempokv_open_encoders[layer_index]
                primary = self._read_bhsd_for_encode(
                    layer[primary_role], token_start, token_end
                )
                reconstructed_primary = encoders[primary_role].append(
                    primary,
                    reuse_input_for_reconstruction=True,
                )
                layer[primary_role].extend_quantized(
                    span_token_start, token_end
                )
                del primary

                secondary = self._read_bhsd_for_encode(
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
                layer[secondary_role].extend_quantized(
                    span_token_start, token_end
                )
                del secondary, reconstructed_primary

        def _quantize_incrementally(
            self,
            tokens_to_quantize_start: int,
            tokens_to_quantize_end: int,
            flush: bool = False,
        ) -> None:
            frame_start = tokens_to_quantize_start // self.frame_seq_length
            finalized_frames = tokens_to_quantize_end // self.frame_seq_length
            if frame_start != self._tempokv_finalized_frames:
                raise ValueError(
                    "incremental TempoKV blocks must be finalized in order: "
                    f"expected frame {self._tempokv_finalized_frames}, "
                    f"got {frame_start}"
                )
            self._tempokv_finalized_frames = finalized_frames
            # Everything newer than the lag stays dense BF16 and is encoded on
            # a later call, once the rollout has moved past it.  A flush is
            # the last call of all: with generation over, the tail can no
            # longer serve attention, so pack through the finalized end.
            if flush:
                frame_end = min(
                    self._tempokv_total_frames, finalized_frames
                )
            else:
                frame_end = max(
                    self._tempokv_encoded_frames,
                    finalized_frames - self._tempokv_bf16_lag_frames,
                )
            cursor = self._tempokv_encoded_frames
            if cursor >= frame_end:
                return
            while cursor < frame_end:
                if self._tempokv_open_start_frame is None:
                    if cursor >= self._tempokv_total_frames:
                        raise ValueError(
                            "incremental TempoKV write exceeds the configured "
                            f"{self._tempokv_total_frames}-frame horizon"
                        )
                    self._tempokv_open_start_frame = cursor
                    self._tempokv_open_units = 0
                    self._tempokv_open_capacity = min(
                        self._tempokv_span_frames,
                        self._tempokv_total_frames - cursor,
                    )
                expected = (
                    self._tempokv_open_start_frame + self._tempokv_open_units
                )
                if cursor != expected:
                    raise ValueError(
                        "incremental TempoKV blocks must be finalized in order: "
                        f"expected frame {expected}, got {cursor}"
                    )
                remaining = self._tempokv_open_capacity - self._tempokv_open_units
                count = min(remaining, frame_end - cursor)
                token_start = cursor * self.frame_seq_length
                token_end = (cursor + count) * self.frame_seq_length
                started = time.perf_counter()
                is_anchor_block = self._tempokv_open_units == 0
                with torch.no_grad():
                    if is_anchor_block:
                        self._begin_incremental_segment(token_start, token_end)
                    else:
                        self._append_incremental_segment(token_start, token_end)
                self._tempokv_open_units += count
                duration_ms = (time.perf_counter() - started) * 1000.0
                span_closed = (
                    self._tempokv_open_units == self._tempokv_open_capacity
                )
                print(
                    "[TempoKV-incremental] encoded frames "
                    f"[{cursor},{cursor + count}) "
                    + ("as anchor" if is_anchor_block else "as residual")
                    + f", open span {self._tempokv_open_units}/"
                    f"{self._tempokv_open_capacity}, finalized="
                    f"{finalized_frames}, BF16 tail="
                    f"{self._tempokv_bf16_lag_frames}, {duration_ms:.1f} ms"
                    + (", span closed -> re-anchor next" if span_closed else ""),
                    flush=True,
                )
                self.summary["quantization_events"].append(
                    {
                        "tokens_start": int(token_start),
                        "tokens_end": int(token_end),
                        "num_tokens": int(token_end - token_start),
                        "duration_ms": duration_ms,
                        "codec": "TempoKV-incremental",
                        "span_start_frame": int(self._tempokv_open_start_frame),
                        "span_units": int(self._tempokv_open_units),
                        "span_capacity": int(self._tempokv_open_capacity),
                    }
                )
                cursor += count
                self._tempokv_encoded_frames = cursor
                if span_closed:
                    self._tempokv_open_start_frame = None
                    self._tempokv_open_units = 0
                    self._tempokv_open_capacity = 0
                    self._tempokv_open_encoders = None
            self._record_kv_cache_memory()

        def flush_incremental_tail(self) -> None:
            """Pack the BF16 tail the incremental schedule still owes.

            During generation the newest ``lag`` frames must stay exact for
            attention; once the rollout is over that tail is dead weight, and
            leaving it unpacked makes the finished cache larger than the
            batched schedule's, which closes its last span before returning.
            ``TEMPOKV_FLUSH_TAIL=0`` restores the unpacked-tail behavior.
            """
            if not getattr(self, "_incremental_tempokv", False):
                return
            if os.environ.get("TEMPOKV_FLUSH_TAIL", "1") == "0":
                return
            if (
                self._tempokv_encoded_frames
                >= self._tempokv_finalized_frames
            ):
                return
            start = self._tempokv_finalized_frames * self.frame_seq_length
            self._quantize_incrementally(start, start, flush=True)
            print(
                "[TempoKV-incremental] final flush: packed tail through "
                f"frame {self._tempokv_finalized_frames}",
                flush=True,
            )

        def quantize_kv_cache(
            self,
            tokens_to_quantize_start: int,
            tokens_to_quantize_end: int,
            max_tokens_to_quantize: int,
        ) -> None:
            if tokens_to_quantize_end > max_tokens_to_quantize:
                raise ValueError("quantization range exceeds the available cache")
            if self._incremental_tempokv:
                self._quantize_incrementally(
                    tokens_to_quantize_start,
                    tokens_to_quantize_end,
                )
                return
            span = tokens_to_quantize_end - tokens_to_quantize_start
            if span < 2 * self.frame_seq_length:
                raise ValueError("TempoKV needs at least two cache units per span")
            started = time.perf_counter()
            with torch.no_grad():
                # The cache slots are addressed by codec role, not by name:
                # the primary tensor is the one predicted from its own temporal
                # history, and its reconstruction is what the secondary tensor
                # projects from.  Forward runs K then V; the mirrored direction
                # runs V then K through the identical code path.
                codec = self.tempokv_codec
                primary_role = codec.primary_role
                secondary_role = codec.secondary_role
                for layer_index, layer in enumerate(self.kv_cache1):
                    primary = self._read_bhsd_for_encode(
                        layer[primary_role],
                        tokens_to_quantize_start,
                        tokens_to_quantize_end,
                    )
                    if (
                        primary.shape[1] != self.codec_shape.num_heads
                        or primary.shape[-1] != self.codec_shape.head_dim
                    ):
                        raise ValueError("Self-Forcing cache shape changed unexpectedly")
                    packed_primary, reconstructed_primary = codec.encode_primary(
                        primary,
                        layer=layer_index,
                        reuse_input=True,
                    )
                    layer[primary_role].store_quantized(
                        tokens_to_quantize_start,
                        tokens_to_quantize_end,
                        wrap_cache_state(packed_primary),
                    )
                    del primary, packed_primary

                    secondary = self._read_bhsd_for_encode(
                        layer[secondary_role],
                        tokens_to_quantize_start,
                        tokens_to_quantize_end,
                    )
                    packed_secondary = codec.encode_secondary(
                        secondary,
                        reconstructed_primary,
                        layer=layer_index,
                        reuse_input=True,
                    )
                    layer[secondary_role].store_quantized(
                        tokens_to_quantize_start,
                        tokens_to_quantize_end,
                        wrap_cache_state(packed_secondary),
                    )
                    del secondary, packed_secondary, reconstructed_primary
            self.summary["quantization_events"].append(
                {
                    "tokens_start": int(tokens_to_quantize_start),
                    "tokens_end": int(tokens_to_quantize_end),
                    "num_tokens": int(span),
                    "duration_ms": (time.perf_counter() - started) * 1000.0,
                    "codec": "TempoKV",
                }
            )
            self._record_kv_cache_memory()

        def _resident_kv_bytes(self) -> tuple[int, int]:
            """Packed spans plus dense chunks, per ``resident_analytic.v1``.

            Predictor parameters live inside every ``PackedTensor`` but are
            shared across spans, so they are skipped here and reported once as
            ``predictor_parameter_bytes``.
            """
            if not self.kv_cache1:
                return 0, 0
            resident = 0
            for layer in self.kv_cache1:
                for role in ("k", "v"):
                    cache = layer[role]
                    for span in cache.quantized_spans:
                        packed = _unwrap_cache_state(span.get("quant_data"))
                        if packed is None:
                            continue
                        resident += efficiency.tensor_bytes(
                            packed, skip_fields=frozenset({"temporal", "cross_kv"})
                        )
                    resident += efficiency.tensor_bytes(cache.chunks)
            resident += self._incremental_history_bytes()
            tokens = int(self.kv_cache1[0]["local_end_index"])
            equivalent = efficiency.bf16_equivalent_bytes(
                batch=self.kv_cache1[0]["k"].batch_size,
                tokens=tokens,
                heads=self.num_heads,
                head_dim=self.head_dim,
            ) * len(self.kv_cache1)
            return resident, equivalent

        def _record_kv_cache_memory(self) -> None:
            """Aggregate resident K/V bytes across every layer.

            Fills ``self.summary["last_kv_cache_memory"]`` (consumed by the
            Self-Forcing runtime as ``summary.json -> memory.kv_cache``) with a
            per-layer breakdown, totals, and the achieved compression rate
            versus the equivalent BF16 cache.  Predictor parameters are shared
            across spans and already excluded from ``storage_bytes``.

            Frames that have not been packed yet are still dense on the device,
            so they are counted separately rather than omitted: a peak-memory
            figure built only from packed spans understates what the run
            actually needed.  ``compression_rate`` deliberately compares only
            the packed part against its own BF16 equivalent, so it stays a
            property of one cache state.
            """
            per_layer: dict[str, dict[str, object]] = {}
            total_kv_bytes = 0
            total_logical_kv_bytes = 0
            total_bf16_bytes = 0
            resident_bf16_bytes = 0
            bf16_per_value = 2  # bytes for a bfloat16 element

            for layer_index, layer in enumerate(self.kv_cache1):
                k_bytes = v_bytes = 0
                k_logical_bytes = v_logical_bytes = 0
                layer_dense_bytes = 0
                for cache, name in ((layer["k"], "k"), (layer["v"], "v")):
                    bytes_per_cache = 0
                    logical_bytes_per_cache = 0
                    for span in cache.quantized_spans:
                        packed = _unwrap_cache_state(span.get("quant_data"))
                        if packed is None:
                            continue
                        bytes_per_cache += packed.storage_bytes()
                        logical_bytes_per_cache += packed.logical_storage_bytes()
                        # Plain Python arithmetic: this runs once per span per
                        # role per layer, and the incremental schedule reaches
                        # it once per model block instead of once per span.
                        total_bf16_bytes += (
                            math.prod(packed.shape) * bf16_per_value
                        )
                    layer_dense_bytes += cache.bf16_resident_bytes()
                    if name == "k":
                        k_bytes = bytes_per_cache
                        k_logical_bytes = logical_bytes_per_cache
                    else:
                        v_bytes = bytes_per_cache
                        v_logical_bytes = logical_bytes_per_cache
                if self._tempokv_open_encoders is not None:
                    layer_dense_bytes += sum(
                        int(encoder.history_bytes())
                        for encoder in self._tempokv_open_encoders[
                            layer_index
                        ].values()
                    )
                total_kv_bytes += k_bytes + v_bytes
                total_logical_kv_bytes += k_logical_bytes + v_logical_bytes
                resident_bf16_bytes += layer_dense_bytes
                per_layer[str(layer_index)] = {
                    "k_bytes": k_bytes,
                    "v_bytes": v_bytes,
                    "kv_bytes": k_bytes + v_bytes,
                    "logical_kv_bytes": k_logical_bytes + v_logical_bytes,
                    "resident_bf16_bytes": layer_dense_bytes,
                }

            self.summary["last_kv_cache_memory"] = {
                "num_layers": len(self.kv_cache1),
                "per_layer_kv_bytes": per_layer,
                "total_kv_cache_bytes": total_kv_bytes,
                "total_logical_kv_cache_bytes": total_logical_kv_bytes,
                "total_bf16_kv_bytes": total_bf16_bytes,
                "resident_bf16_bytes": resident_bf16_bytes,
                "resident_total_bytes": total_kv_bytes + resident_bf16_bytes,
                "compression_rate": (
                    total_bf16_bytes / total_kv_bytes if total_kv_bytes else None
                ),
                "logical_compression_rate": (
                    total_bf16_bytes / total_logical_kv_bytes
                    if total_logical_kv_bytes
                    else None
                ),
            }

    TempoKVCausalInferencePipeline.__name__ = "TempoKVCausalInferencePipeline"
    pipeline_module.CausalInferencePipeline = TempoKVCausalInferencePipeline
    pipeline_package.CausalInferencePipeline = TempoKVCausalInferencePipeline
    return TempoKVCausalInferencePipeline
