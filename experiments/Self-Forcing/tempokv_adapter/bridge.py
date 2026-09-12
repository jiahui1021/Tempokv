from __future__ import annotations

import importlib
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

        def quantize_kv_cache(
            self,
            tokens_to_quantize_start: int,
            tokens_to_quantize_end: int,
            max_tokens_to_quantize: int,
        ) -> None:
            if tokens_to_quantize_end > max_tokens_to_quantize:
                raise ValueError("quantization range exceeds the available cache")
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
                    primary = layer[primary_role].read_bhsd(
                        tokens_to_quantize_start, tokens_to_quantize_end
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

                    secondary = layer[secondary_role].read_bhsd(
                        tokens_to_quantize_start, tokens_to_quantize_end
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
            total_bf16_bytes = 0
            resident_bf16_bytes = 0
            bf16_per_value = 2  # bytes for a bfloat16 element

            for layer_index, layer in enumerate(self.kv_cache1):
                k_bytes = v_bytes = 0
                layer_dense_bytes = 0
                for cache, name in ((layer["k"], "k"), (layer["v"], "v")):
                    bytes_per_cache = 0
                    for span in cache.quantized_spans:
                        packed = _unwrap_cache_state(span.get("quant_data"))
                        if packed is None:
                            continue
                        bytes_per_cache += packed.storage_bytes()
                        total_bf16_bytes += (
                            int(torch.tensor(packed.shape).prod().item())
                            * bf16_per_value
                        )
                    layer_dense_bytes += cache.bf16_resident_bytes()
                    if name == "k":
                        k_bytes = bytes_per_cache
                    else:
                        v_bytes = bytes_per_cache
                total_kv_bytes += k_bytes + v_bytes
                resident_bf16_bytes += layer_dense_bytes
                per_layer[str(layer_index)] = {
                    "k_bytes": k_bytes,
                    "v_bytes": v_bytes,
                    "kv_bytes": k_bytes + v_bytes,
                    "resident_bf16_bytes": layer_dense_bytes,
                }

            self.summary["last_kv_cache_memory"] = {
                "num_layers": len(self.kv_cache1),
                "per_layer_kv_bytes": per_layer,
                "total_kv_cache_bytes": total_kv_bytes,
                "total_bf16_kv_bytes": total_bf16_bytes,
                "resident_bf16_bytes": resident_bf16_bytes,
                "resident_total_bytes": total_kv_bytes + resident_bf16_bytes,
                "compression_rate": (
                    total_bf16_bytes / total_kv_bytes if total_kv_bytes else None
                ),
            }

    TempoKVCausalInferencePipeline.__name__ = "TempoKVCausalInferencePipeline"
    pipeline_module.CausalInferencePipeline = TempoKVCausalInferencePipeline
    pipeline_package.CausalInferencePipeline = TempoKVCausalInferencePipeline
    return TempoKVCausalInferencePipeline
