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


def install(runtime_root: str | Path) -> type:
    """Install TempoKV into the Self-Forcing runtime bundled in this repo."""
    runtime_root = Path(runtime_root).resolve()
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))

    pipeline_package = importlib.import_module("pipeline")
    pipeline_module = importlib.import_module("pipeline.causal_inference")
    base = pipeline_module.CausalInferencePipeline
    cache_module = sys.modules[pipeline_module.ChunkedKVCache.__module__]
    fallback_decode = cache_module.uncompress_single_kv_cache

    class TempoKVCausalInferencePipeline(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.codec_shape = SimpleNamespace(
                frame_seq_length=self.frame_seq_length,
                num_heads=12,
                head_dim=128,
            )
            self.tempokv_codec = load_codec_from_env()
            config = self.tempokv_codec.config
            if config.stride != self.frame_seq_length:
                raise ValueError(
                    f"TempoKV stride {config.stride} does not match Self-Forcing "
                    f"frame size {self.frame_seq_length}"
                )
            if config.reconstruction_dtype != "bfloat16":
                raise ValueError("Self-Forcing requires bfloat16 reconstruction")
            if (
                getattr(config, "v_history_length", 0)
                and not self.tempokv_codec.cross_kv.history_diagonal
            ):
                raise ValueError(
                    "Self-Forcing requires fast diagonal V history parameters; "
                    "refit v_cross_kv.npz with the current fitter"
                )
            if getattr(self.quant_config, "kv_cache_cpu_offload", False):
                raise ValueError("TempoKV Self-Forcing bridge does not support cache offload")
            self._patch_attention_windows()

            def decode_with_codec(
                state: Any,
                source_tensor: torch.Tensor | None = None,
                output_tensor: torch.Tensor | None = None,
            ) -> torch.Tensor:
                return decode_cache_state(
                    state,
                    source_tensor=source_tensor,
                    output_tensor=output_tensor,
                    codec=self.tempokv_codec,
                    fallback=fallback_decode,
                )

            cache_module.set_quantized_decoder(decode_with_codec)

        def _patch_attention_windows(self) -> None:
            # TempoKV takes over cache compression, so the model must keep the
            # full sequence span readable instead of capping attention at the
            # 21-frame default window. The target frame count arrives through
            # TEMPOKV_NUM_FRAMES because omegaconf args lack num_output_frames.
            frames = int(os.environ.get("TEMPOKV_NUM_FRAMES", "0"))
            if frames <= 0:
                raise ValueError("TEMPOKV_NUM_FRAMES must be set for TempoKV runs")
            if self.local_attn_size != -1:
                raise ValueError("TempoKV Self-Forcing requires global attention")
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
                for layer_index, layer in enumerate(self.kv_cache1):
                    keys = layer["k"].read_bhsd(
                        tokens_to_quantize_start, tokens_to_quantize_end
                    )
                    if (
                        keys.shape[1] != self.codec_shape.num_heads
                        or keys.shape[-1] != self.codec_shape.head_dim
                    ):
                        raise ValueError("Self-Forcing cache shape changed unexpectedly")
                    packed_key, reconstructed_key = self.tempokv_codec.encode_key(
                        keys,
                        layer=layer_index,
                        reuse_input=True,
                    )
                    layer["k"].store_quantized(
                        tokens_to_quantize_start,
                        tokens_to_quantize_end,
                        wrap_cache_state(packed_key),
                    )
                    del keys, packed_key

                    values = layer["v"].read_bhsd(
                        tokens_to_quantize_start, tokens_to_quantize_end
                    )
                    packed_value = self.tempokv_codec.encode_value(
                        values,
                        reconstructed_key,
                        layer=layer_index,
                        reuse_input=True,
                    )
                    layer["v"].store_quantized(
                        tokens_to_quantize_start,
                        tokens_to_quantize_end,
                        wrap_cache_state(packed_value),
                    )
                    del values, packed_value, reconstructed_key
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

        def _record_kv_cache_memory(self) -> None:
            """Aggregate stored K/V bytes across every layer's quantized spans.

            Fills ``self.summary["last_kv_cache_memory"]`` (consumed by the
            Self-Forcing runtime as ``summary.json -> memory.kv_cache``) with a
            per-layer breakdown, totals, and the achieved compression rate
            versus the equivalent BF16 cache.  Predictor parameters are shared
            across spans and already excluded from ``storage_bytes``.
            """
            per_layer: dict[str, dict[str, object]] = {}
            total_kv_bytes = 0
            total_bf16_bytes = 0
            bf16_per_value = 2  # bytes for a bfloat16 element

            for layer_index, layer in enumerate(self.kv_cache1):
                k_bytes = v_bytes = 0
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
                    if name == "k":
                        k_bytes = bytes_per_cache
                    else:
                        v_bytes = bytes_per_cache
                total_kv_bytes += k_bytes + v_bytes
                per_layer[str(layer_index)] = {
                    "k_bytes": k_bytes,
                    "v_bytes": v_bytes,
                    "kv_bytes": k_bytes + v_bytes,
                }

            self.summary["last_kv_cache_memory"] = {
                "num_layers": len(self.kv_cache1),
                "per_layer_kv_bytes": per_layer,
                "total_kv_cache_bytes": total_kv_bytes,
                "total_bf16_kv_bytes": total_bf16_bytes,
                "compression_rate": (
                    total_bf16_bytes / total_kv_bytes if total_kv_bytes else None
                ),
            }

    TempoKVCausalInferencePipeline.__name__ = "TempoKVCausalInferencePipeline"
    pipeline_module.CausalInferencePipeline = TempoKVCausalInferencePipeline
    pipeline_package.CausalInferencePipeline = TempoKVCausalInferencePipeline
    return TempoKVCausalInferencePipeline
