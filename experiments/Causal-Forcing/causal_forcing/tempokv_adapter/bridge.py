"""Install TempoKV into the bundled Causal Forcing inference pipeline.

The pipeline announces denoising writes, one finalize per block after the clean
refresh, and resets; this bridge feeds finalized blocks to the streaming
encoder and decodes packed spans on cache reads.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any, Callable

import torch

from tempokv.codec import PackedTensor, TempoKVCodec, decode_tensor
from tempokv.integration import load_codec_from_env
from tempokv.runtime.metrics import TempoKVRuntimeReport
from tempokv.runtime.streaming import StreamingTempoKVEncoder


_TEMPOKV_STATE_KEY = "_tempokv_packed"

# Fallback span length; generate.py sets TEMPOKV_SPAN_FRAMES to the rollout.
DEFAULT_SPAN_FRAMES = 12
# Finalized frames kept exact before packing (sized for 180 latent frames).
DEFAULT_BF16_LAG_FRAMES = 18
DECODE_TIMING_ENV = "TEMPOKV_DECODE_TIMING"
# Diagnostic: print the NMSE between layer-0 cache reads and the clean K/V
# they replaced, to check that bit widths reach the cache.
READBACK_PROBE_ENV = "TEMPOKV_READBACK_PROBE"
_PROBE_TOKENS_PER_FRAME = 16


def _decode_timing_enabled() -> bool:
    raw = os.environ.get(DECODE_TIMING_ENV, "0")
    if raw not in {"0", "1"}:
        raise ValueError(f"{DECODE_TIMING_ENV} must be 0 or 1")
    return raw == "1"


class _CausalTempoKVRuntimeReport(TempoKVRuntimeReport):
    """TempoKV counters plus explicit terminal-cache discard accounting."""

    decode_timing_enabled: bool = False
    terminal_discard_events: int = 0
    # Frames in the final block whose clean refresh/finalize was skipped.
    terminal_discarded_frames: int = 0
    # Finalized dense frames plus the skipped final block that the old
    # terminal flush would have encoded despite having no future reader.
    terminal_encode_frames_avoided: int = 0
    terminal_discarded_bf16_bytes: int = 0
    terminal_discarded_compressed_bytes: int = 0

    def record_terminal_discard(
        self,
        *,
        frames: int,
        encode_frames_avoided: int,
        bf16_bytes: int,
        compressed_bytes: int,
    ) -> None:
        self.terminal_discard_events += 1
        self.terminal_discarded_frames += frames
        self.terminal_encode_frames_avoided += encode_frames_avoided
        self.terminal_discarded_bf16_bytes += bf16_bytes
        self.terminal_discarded_compressed_bytes += compressed_bytes

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["decode_timing_enabled"] = self.decode_timing_enabled
        if not self.decode_timing_enabled:
            payload["decode_ms"] = None
        payload.update(
            {
                "terminal_discard_events": self.terminal_discard_events,
                "terminal_discarded_frames": self.terminal_discarded_frames,
                "terminal_encode_frames_avoided": (
                    self.terminal_encode_frames_avoided
                ),
                "terminal_discarded_bf16_bytes": (
                    self.terminal_discarded_bf16_bytes
                ),
                "terminal_discarded_compressed_bytes": (
                    self.terminal_discarded_compressed_bytes
                ),
            }
        )
        return payload


def wrap_cache_state(state: PackedTensor) -> dict[str, Any]:
    """Store one TempoKV state in the bundled Causal Forcing cache."""
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
        kwargs = {
            "source_tensor": source_tensor,
            "output_tensor": output_tensor,
        }
        if output_layout != "BSHD":
            kwargs["output_layout"] = output_layout
        return fallback(state, **kwargs)
    return decode_tensor(
        packed,
        source=source_tensor,
        output_tensor=output_tensor,
        output_layout=output_layout,
        use_triton=codec.config.use_triton,
        strict_triton=codec.config.strict_triton,
    )


def _span_frames() -> int:
    raw = os.environ.get("TEMPOKV_SPAN_FRAMES")
    if raw is None:
        return DEFAULT_SPAN_FRAMES
    value = int(raw)
    if value < 2:
        raise ValueError("TEMPOKV_SPAN_FRAMES must be at least 2")
    return value


def install(runtime_root: str | Path | None = None) -> type:
    """Install TempoKV into the Causal Forcing runtime bundled in this repo."""
    if runtime_root is not None:
        runtime_root = Path(runtime_root).resolve()
        if str(runtime_root) not in sys.path:
            sys.path.insert(0, str(runtime_root))

    pipeline_package = importlib.import_module("causal_forcing.pipeline")
    pipeline_module = importlib.import_module(
        "causal_forcing.pipeline.causal_inference"
    )
    base = pipeline_module.CausalInferencePipeline
    cache_module = sys.modules[pipeline_module.ChunkedKVCache.__module__]
    fallback_decode = getattr(cache_module, "_quantized_decoder", None)

    class TempoKVCausalInferencePipeline(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # TempoKV needs the pre-RoPE frame-addressable cache.
            self._uses_pre_rope_cache = True
            self.tempokv_codec = load_codec_from_env()
            config = self.tempokv_codec.config
            if config.stride != self.frame_seq_length:
                raise ValueError(
                    f"TempoKV stride {config.stride} does not match Causal "
                    f"Forcing frame size {self.frame_seq_length}"
                )
            if config.reconstruction_dtype != "bfloat16":
                raise ValueError("Causal Forcing requires bfloat16 reconstruction")
            # The temporal V predictor has no Cross(K) map, so the diagonal
            # requirement applies only to the CrossKV route.
            if (
                config.v_predictor_mode
                not in {"temporal", "identity", "taylor", "taylor2", "direct"}
                and config.v_history_length
                and not self.tempokv_codec.cross_kv.history_diagonal
            ):
                raise ValueError(
                    "Causal Forcing requires fast diagonal V history parameters; "
                    "refit v_cross_kv.npz with the current fitter"
                )
            self.tempokv_report = _CausalTempoKVRuntimeReport(
                model=os.environ.get("TEMPOKV_MODEL_NAME", "causal_forcing"),
                kv_mode="tempokv",
                num_layers=self.num_transformer_blocks,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                stride=config.stride,
                predictor_parameter_bytes=_predictor_parameter_bytes(
                    self.tempokv_codec
                ),
            )
            self.tempokv_report.decode_timing_enabled = _decode_timing_enabled()
            self._readback_probe = os.environ.get(READBACK_PROBE_ENV, "0") == "1"
            self._probe_clean: dict[str, list[torch.Tensor]] = {"k": [], "v": []}
            # Incremental schedule as on Self-Forcing; span-mean K and innovation V
            # have no incremental state.
            incremental = (
                os.environ.get("TEMPOKV_INCREMENTAL", "1") != "0"
                and not self.tempokv_codec.k_channel_mean
                and config.v_predictor_mode != "innovation"
            )
            bf16_lag_frames = int(
                os.environ.get(
                    "TEMPOKV_BF16_LAG_FRAMES", str(DEFAULT_BF16_LAG_FRAMES)
                )
            )
            self.tempokv_stream = StreamingTempoKVEncoder(
                self.tempokv_codec,
                num_layers=self.num_transformer_blocks,
                frame_seq_length=self.frame_seq_length,
                span_frames=_span_frames(),
                report=self.tempokv_report,
                wrap_state=wrap_cache_state,
                incremental=incremental,
                bf16_lag_frames=bf16_lag_frames,
            )
            print(
                "[TempoKV] span encoding mode: "
                + ("INCREMENTAL" if incremental else "BATCHED")
                + f", span_frames={self.tempokv_stream.span_frames}"
                + f", bf16_lag_frames={bf16_lag_frames}",
                flush=True,
            )

            def decode_with_codec(
                state: Any,
                source_tensor: torch.Tensor | None = None,
                output_tensor: torch.Tensor | None = None,
                output_layout: str = "BSHD",
            ) -> torch.Tensor:
                self.tempokv_report.decode_tensor_calls += 1
                if self.tempokv_report.decode_timing_enabled:
                    with self.tempokv_report.timer("decode_ms"):
                        return decode_cache_state(
                            state,
                            source_tensor=source_tensor,
                            output_tensor=output_tensor,
                            output_layout=output_layout,
                            codec=self.tempokv_codec,
                            fallback=fallback_decode,
                        )
                return decode_cache_state(
                    state,
                    source_tensor=source_tensor,
                    output_tensor=output_tensor,
                    output_layout=output_layout,
                    codec=self.tempokv_codec,
                    fallback=fallback_decode,
                )

            cache_module.set_quantized_decoder(decode_with_codec)

        # -- write-mode hooks --------------------------------------------

        def reset_kv_cache_state(self) -> None:
            self.tempokv_stream.reset()

        def note_ephemeral_block(self, num_writes: int) -> None:
            self.tempokv_stream.note_ephemeral_write(num_writes)

        def finalize_kv_cache(self, start_frame: int, num_frames: int) -> None:
            expected = self.tempokv_stream.finalized_frame
            if start_frame != expected:
                raise RuntimeError(
                    "Causal Forcing finalized frame "
                    f"{start_frame} out of order; expected {expected}"
                )
            previous_span_start = self.tempokv_stream.span_start_frame
            if self._readback_probe:
                self._probe_record(start_frame, num_frames)
            self.tempokv_stream.finalize(self.kv_cache1, num_frames=num_frames)
            if self._readback_probe:
                self._probe_compare()
            if os.environ.get("CAUSAL_FORCING_BENCHMARK") == "1":
                packed_until = self.tempokv_stream.span_start_frame
                while previous_span_start < packed_until:
                    span_end = min(
                        previous_span_start + self.tempokv_stream.span_frames,
                        packed_until,
                    )
                    print(
                        "[TempoKVSchedule] "
                        f"start_frame={previous_span_start} "
                        f"end_frame={span_end} "
                        f"num_frames={span_end - previous_span_start}",
                        flush=True,
                    )
                    previous_span_start = span_end
            self.summary["finalize_events"].append(
                {"start_frame": int(start_frame), "num_frames": int(num_frames)}
            )

        def _probe_tokens(self, frames: int) -> torch.Tensor:
            step = max(1, self.frame_seq_length // _PROBE_TOKENS_PER_FRAME)
            offsets = torch.arange(0, self.frame_seq_length, step)
            return (
                torch.arange(frames)[:, None] * self.frame_seq_length + offsets
            ).reshape(-1)

        def _probe_record(self, start_frame: int, num_frames: int) -> None:
            if start_frame == 0:
                self._probe_clean = {"k": [], "v": []}
            layer = self.kv_cache1[0]
            start = start_frame * self.frame_seq_length
            end = start + num_frames * self.frame_seq_length
            index = self._probe_tokens(num_frames).to(layer["k"].device)
            for role in ("k", "v"):
                dense = layer[role].read(start, end)
                self._probe_clean[role].append(
                    dense.index_select(1, index).float().cpu()
                )

        def _probe_compare(self) -> None:
            encoded = self.tempokv_stream.encoded_frame
            if not encoded:
                return
            layer = self.kv_cache1[0]
            end = encoded * self.frame_seq_length
            role_primary = self.tempokv_codec.primary_role
            role_secondary = self.tempokv_codec.secondary_role
            primary = layer[role_primary].read(0, end)
            decoded = {
                role_primary: primary,
                role_secondary: layer[role_secondary].read(
                    0, end, source_tensor=primary
                ),
            }
            index = self._probe_tokens(encoded).to(primary.device)
            errors = {}
            for role in ("k", "v"):
                clean = torch.cat(self._probe_clean[role], dim=1)[:, : index.numel()]
                read = decoded[role].index_select(1, index).float().cpu()
                errors[role] = float(
                    (read - clean).square().sum() / clean.square().sum().clamp_min(1e-30)
                )
            config = self.tempokv_codec.config
            print(
                f"[TempoKVReadback] k{config.k_bits}v{config.v_bits} "
                f"packed_frames={encoded} "
                f"k_nmse={errors['k']:.3e} v_nmse={errors['v']:.3e}",
                flush=True,
            )

        def flush_kv_cache(self) -> None:
            # Benchmarks may keep the final partial span as a BF16 tail.
            if self.tempokv_stream.incremental:
                # Pack the lag tail unless TEMPOKV_FLUSH_TAIL=0.
                force = os.environ.get("TEMPOKV_FLUSH_TAIL", "1") != "0"
            else:
                force_partial = os.environ.get("TEMPOKV_FORCE_PARTIAL_SPAN", "1")
                force = force_partial not in {"0", "false", "no"}
            self.tempokv_stream.flush(self.kv_cache1, force=force)
            self.summary["last_kv_cache_memory"] = self.tempokv_report.to_dict()

        def discard_terminal_kv_cache(
            self, start_frame: int, num_frames: int
        ) -> None:
            encode_frames_avoided = (
                self.tempokv_stream.pending_frames + num_frames
            )
            resident_bf16_bytes = sum(
                layer[role].bf16_resident_bytes()
                for layer in self.kv_cache1
                for role in ("k", "v")
            )
            # The skipped finalize no longer observes this dense terminal
            # state, so account for its real peak before clearing the cache.
            self.tempokv_report.observe_resident_bf16_bytes(
                resident_bf16_bytes
            )
            compressed_bytes = (
                self.tempokv_report.current_compressed_cache_bytes
            )
            super().discard_terminal_kv_cache(start_frame, num_frames)
            self.tempokv_report.record_terminal_discard(
                frames=num_frames,
                encode_frames_avoided=encode_frames_avoided,
                bf16_bytes=resident_bf16_bytes,
                compressed_bytes=compressed_bytes,
            )
            # The physical cache has been released, so current-resident byte
            # counters must reach zero while cumulative and peak counters stay.
            self.tempokv_stream.reset()
            self.summary["last_kv_cache_memory"] = self.tempokv_report.to_dict()

        def inference(self, *args: Any, **kwargs: Any) -> Any:
            with self.tempokv_report.timer("generation_ms"):
                result = super().inference(*args, **kwargs)
            # A run that produced a video without touching the codec is a
            # configuration failure, not a success.
            self.tempokv_report.require_tempokv_activity()
            return result

    TempoKVCausalInferencePipeline.__name__ = "TempoKVCausalInferencePipeline"
    pipeline_module.CausalInferencePipeline = TempoKVCausalInferencePipeline
    pipeline_package.CausalInferencePipeline = TempoKVCausalInferencePipeline
    return TempoKVCausalInferencePipeline


def _predictor_parameter_bytes(codec: TempoKVCodec) -> int:
    temporal = codec.temporal
    v = codec.v_parameters
    if codec.predicts_v_from_key:
        # CrossKV V: a Cross(K) map plus optional V history.
        tensors = (
            temporal.history_weights,
            temporal.history_bias,
            v.weight,
            v.bias,
            v.history_weights,
        )
    else:
        # Temporal V: the V predictor reads its own history, so it carries the
        # same temporal history tensors as K and no Cross(K) map.
        tensors = (
            temporal.history_weights,
            temporal.history_bias,
            v.history_weights,
            v.history_bias,
        )
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in tensors
        if tensor is not None
    )


__all__ = [
    "DEFAULT_SPAN_FRAMES",
    "decode_cache_state",
    "install",
    "wrap_cache_state",
]
