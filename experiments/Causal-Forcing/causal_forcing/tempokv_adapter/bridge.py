"""Install TempoKV into the bundled Causal Forcing inference pipeline.

The pipeline already owns the cache write-mode state machine: it announces
ephemeral denoising writes, announces exactly one finalize per block after the
clean/context refresh, and announces resets.  This bridge subclasses it and
gives those announcements their TempoKV meaning — the finalize hook feeds the
streaming span encoder, and every packed span is decoded back through the
codec when attention reads the cache.

Nothing here inspects a timestep.  If this file is not installed, the pipeline
runs BF16 and the hooks are inert.
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

# Frames packed into one TempoKV span. One anchor is emitted per span and every
# other frame in it is predicted from reconstructed history, so a larger span
# means better compression; the most recent (< span) frames stay BF16.
#
# The default has to leave room for the packed span to actually be *read*: the
# 81-frame default rollout is 21 latent frames, so a 24-frame span would never
# complete, and even a 21-frame span would only pack after the last forward
# pass — producing a video that never decoded a single compressed key.
DEFAULT_SPAN_FRAMES = 12


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
    codec: TempoKVCodec,
    fallback: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Decode a TempoKV span or delegate an unrecognized cache state."""
    packed = _unwrap_cache_state(state)
    if packed is None:
        if fallback is None:
            raise TypeError(f"unsupported cache state: {type(state).__name__}")
        return fallback(
            state, source_tensor=source_tensor, output_tensor=output_tensor
        )
    return decode_tensor(
        packed,
        source=source_tensor,
        output_tensor=output_tensor,
        use_triton=codec.config.use_triton,
        strict_triton=codec.config.strict_triton,
    )


def _span_frames() -> int:
    raw = os.environ.get("TEMPOKV_SPAN_FRAMES")
    if raw is None:
        return DEFAULT_SPAN_FRAMES
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"TEMPOKV_SPAN_FRAMES must be an integer, got {raw!r}"
        ) from error
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
            self.tempokv_codec = load_codec_from_env()
            config = self.tempokv_codec.config
            if config.stride != self.frame_seq_length:
                raise ValueError(
                    f"TempoKV stride {config.stride} does not match Causal "
                    f"Forcing frame size {self.frame_seq_length}"
                )
            if config.reconstruction_dtype != "bfloat16":
                raise ValueError("Causal Forcing requires bfloat16 reconstruction")
            if (
                getattr(config, "v_history_length", 0)
                and not self.tempokv_codec.cross_kv.history_diagonal
            ):
                raise ValueError(
                    "Causal Forcing requires fast diagonal V history parameters; "
                    "refit v_cross_kv.npz with the current fitter"
                )
            self.tempokv_report = TempoKVRuntimeReport(
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
            self.tempokv_stream = StreamingTempoKVEncoder(
                self.tempokv_codec,
                num_layers=self.num_transformer_blocks,
                frame_seq_length=self.frame_seq_length,
                span_frames=_span_frames(),
                report=self.tempokv_report,
                wrap_state=wrap_cache_state,
            )

            def decode_with_codec(
                state: Any,
                source_tensor: torch.Tensor | None = None,
                output_tensor: torch.Tensor | None = None,
            ) -> torch.Tensor:
                self.tempokv_report.decode_tensor_calls += 1
                with self.tempokv_report.timer("decode_ms"):
                    return decode_cache_state(
                        state,
                        source_tensor=source_tensor,
                        output_tensor=output_tensor,
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
            self.tempokv_stream.finalize(self.kv_cache1, num_frames=num_frames)
            self.summary["finalize_events"].append(
                {"start_frame": int(start_frame), "num_frames": int(num_frames)}
            )

        def flush_kv_cache(self) -> None:
            self.tempokv_stream.flush(self.kv_cache1, force=True)
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
    tensors = (
        codec.temporal.history_weights,
        codec.temporal.history_bias,
        codec.cross_kv.weight,
        codec.cross_kv.bias,
        codec.cross_kv.history_weights,
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
