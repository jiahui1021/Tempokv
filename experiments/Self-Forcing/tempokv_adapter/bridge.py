from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
import torch

from tempokv.codec import PackedTensor, TempoKVCodec, decode_tensor
from tempokv.integration import load_codec_from_env
from tempokv.runtime.metrics import TempoKVRuntimeReport
from tempokv.runtime.streaming import StreamingTempoKVEncoder

# Loaded both as ``bridge`` and by file path, so add this directory itself.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import efficiency_record as efficiency
import pruning


_TEMPOKV_STATE_KEY = "_tempokv_packed"
# Finalized frames kept exact before they are packed.
DEFAULT_BF16_LAG_FRAMES = 18


def wrap_cache_state(state: PackedTensor) -> dict[str, Any]:
    """Store one TempoKV state in the bundled Self-Forcing cache."""
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
    custom_decode = getattr(packed, "decode", None)
    if callable(custom_decode):
        return custom_decode(
            source=source_tensor,
            output_tensor=output_tensor,
            output_layout=output_layout,
        )
    # Decode straight into the caller's cache buffer.
    decode_kwargs = {
        "source": source_tensor,
        "output_tensor": output_tensor,
        "use_triton": codec.config.use_triton,
        "strict_triton": codec.config.strict_triton,
    }
    if output_layout != "BSHD":
        decode_kwargs["output_layout"] = output_layout
    return decode_tensor(packed, **decode_kwargs)


def install(
    runtime_root: str | Path,
    *,
    codec_factory: Callable[[], TempoKVCodec] | None = None,
) -> type:
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
            # TempoKV needs the pre-RoPE frame-addressable cache; BF16 keeps the dense one.
            self._uses_pre_rope_cache = True
            self.tempokv_codec = (
                load_codec_from_env() if codec_factory is None else codec_factory()
            )
            config = self.tempokv_codec.config
            self._tempokv_rope_precision = "complex64"
            # Rollout length (the attention-window ablation narrows TEMPOKV_NUM_FRAMES).
            self._tempokv_total_frames = int(
                os.environ.get(
                    "TEMPOKV_HORIZON_FRAMES",
                    os.environ.get("TEMPOKV_NUM_FRAMES", "0"),
                )
            )
            if self._tempokv_total_frames < 1:
                raise ValueError("TEMPOKV_HORIZON_FRAMES must be positive")
            # Default: one anchor for the whole rollout.
            self._tempokv_span_frames = int(
                os.environ.get(
                    "TEMPOKV_SPAN_FRAMES", str(self._tempokv_total_frames)
                )
            )
            if self._tempokv_span_frames < 2:
                raise ValueError("TEMPOKV_SPAN_FRAMES must be at least 2")
            # Span-mean K and innovation V have no incremental state.
            self._incremental_tempokv = (
                os.environ.get("TEMPOKV_INCREMENTAL", "1") != "0"
                and not self.tempokv_codec.k_channel_mean
                and config.v_predictor_mode != "innovation"
            )
            # Frames stay exact until this far behind the newest finalized one; 18
            # matched the batch schedule's quality at 24-frame spans.
            self._tempokv_bf16_lag_frames = int(
                os.environ.get(
                    "TEMPOKV_BF16_LAG_FRAMES", str(DEFAULT_BF16_LAG_FRAMES)
                )
            )
            num_layers = self.num_transformer_blocks
            self.tempokv_stream = StreamingTempoKVEncoder(
                self.tempokv_codec,
                num_layers=num_layers,
                frame_seq_length=self.frame_seq_length,
                span_frames=self._tempokv_span_frames,
                report=TempoKVRuntimeReport(
                    model="self_forcing",
                    kv_mode="tempokv",
                    num_layers=num_layers,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    stride=config.stride,
                ),
                wrap_state=wrap_cache_state,
                incremental=self._incremental_tempokv,
                bf16_lag_frames=self._tempokv_bf16_lag_frames,
                horizon_frames=self._tempokv_total_frames,
                event_hook=self._on_incremental_encode,
            )
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
            self.tempokv_pruning = pruning.from_env(
                self.tempokv_codec,
                num_layers=num_layers,
                frame_seqlen=self.frame_seq_length,
                max_frames=self._tempokv_total_frames,
            )
            if self.tempokv_pruning is not None:
                for block, pruner in zip(
                    self.generator.model.blocks, self.tempokv_pruning.layers
                ):
                    block.self_attn.tempokv_pruner = pruner

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
            """Stamp the codec direction and Jensen settings on both cache slots."""
            if self.kv_cache1 is None:
                return
            config = self.tempokv_codec.config
            for layer in self.kv_cache1:
                layer["k"].rope_precision = self._tempokv_rope_precision
                # The correction reads the K cache's step sizes, so only the K
                # slot needs to carry it.
                layer["k"].tempokv_jensen_correction = config.jensen_correction
                layer["k"].tempokv_group_size = config.group_size
                # Ablation switch; changes no stored value.
                layer["k"].tempokv_jensen_cache_mean = (
                    os.environ.get("TEMPOKV_JENSEN_CACHE_MEAN") == "1"
                )
                for role in ("k", "v"):
                    layer[role].tempokv_predictor_direction = (
                        config.predictor_direction
                    )

        def _initialize_kv_cache(self, *args: Any, **kwargs: Any) -> Any:
            result = super()._initialize_kv_cache(*args, **kwargs)
            self._apply_tempokv_cache_settings()
            return result

        def reset_quantization_state(self) -> None:
            self.tempokv_stream.reset()
            if self.tempokv_pruning is not None:
                self.tempokv_pruning.reset()

        def inference(self, *args: Any, **kwargs: Any) -> Any:
            result = super().inference(*args, **kwargs)
            if self.tempokv_pruning is not None:
                self.tempokv_pruning.finish(self._tempokv_total_frames)
            return result

        def _patch_attention_windows(self) -> None:
            # Keep the whole compressed sequence readable instead of the 21-frame default.
            frames = int(os.environ.get("TEMPOKV_NUM_FRAMES", "0"))
            if frames <= 0:
                raise ValueError("TEMPOKV_NUM_FRAMES must be set for TempoKV runs")
            for block in self.generator.model.blocks:
                block.self_attn.max_attention_size = frames * self.frame_seq_length

        def _on_incremental_encode(self, event: dict[str, Any]) -> None:
            """Log one incremental begin/append the way the run logs expect."""
            start = event["tokens_start"] // self.frame_seq_length
            end = event["tokens_end"] // self.frame_seq_length
            print(
                f"[TempoKV-incremental] encoded frames [{start},{end}) "
                + ("as anchor" if event["anchor"] else "as residual")
                + f", open span {event['span_units']}/{event['span_capacity']}"
                + f", finalized={event['finalized_frames']}"
                + f", BF16 tail={self._tempokv_bf16_lag_frames}"
                + f", {event['duration_ms']:.1f} ms"
                + (", span closed -> re-anchor next" if event["span_closed"] else ""),
                flush=True,
            )
            self.summary["quantization_events"].append(
                {
                    key: event[key]
                    for key in (
                        "tokens_start",
                        "tokens_end",
                        "num_tokens",
                        "duration_ms",
                        "codec",
                        "span_start_frame",
                        "span_units",
                        "span_capacity",
                    )
                }
            )

        def flush_incremental_tail(self) -> None:
            """Pack the BF16 lag tail once no later block reads it (``TEMPOKV_FLUSH_TAIL=0`` keeps it)."""
            if not getattr(self, "_incremental_tempokv", False):
                return
            if os.environ.get("TEMPOKV_FLUSH_TAIL", "1") == "0":
                return
            stream = self.tempokv_stream
            if stream.encoded_frame >= stream.finalized_frame:
                return
            with torch.no_grad():
                stream.flush(self.kv_cache1, force=True)
            self._record_kv_cache_memory()
            print(
                "[TempoKV-incremental] final flush: packed tail through "
                f"frame {stream.encoded_frame}",
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
                stream = self.tempokv_stream
                frame_start = tokens_to_quantize_start // self.frame_seq_length
                if frame_start != stream.finalized_frame:
                    raise ValueError(
                        "incremental TempoKV blocks must be finalized in order: "
                        f"expected frame {stream.finalized_frame}, "
                        f"got {frame_start}"
                    )
                with torch.no_grad():
                    packed = stream.finalize(
                        self.kv_cache1,
                        num_frames=(
                            tokens_to_quantize_end - tokens_to_quantize_start
                        )
                        // self.frame_seq_length,
                    )
                if packed:
                    self._record_kv_cache_memory()
                return
            span = tokens_to_quantize_end - tokens_to_quantize_start
            if span < 2 * self.frame_seq_length:
                raise ValueError("TempoKV needs at least two cache units per span")
            started = time.perf_counter()
            with torch.no_grad():
                self.tempokv_stream.encode_span(
                    self.kv_cache1,
                    tokens_to_quantize_start // self.frame_seq_length,
                    span // self.frame_seq_length,
                )
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
            """Packed spans plus dense chunks per ``resident_analytic.v1`` (parameters excluded)."""
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
            resident += self.tempokv_stream.history_bytes()
            tokens = int(self.kv_cache1[0]["local_end_index"])
            equivalent = efficiency.bf16_equivalent_bytes(
                batch=self.kv_cache1[0]["k"].batch_size,
                tokens=tokens,
                heads=self.num_heads,
                head_dim=self.head_dim,
            ) * len(self.kv_cache1)
            return resident, equivalent

        def _record_kv_cache_memory(self) -> None:
            """Aggregate resident K/V bytes into ``summary["last_kv_cache_memory"]``.

            Unpacked dense frames are counted separately; ``compression_rate`` compares
            only the packed part with its BF16 equivalent.
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
                layer_dense_bytes += self.tempokv_stream.layer_history_bytes(
                    layer_index
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
