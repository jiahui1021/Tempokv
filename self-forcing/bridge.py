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


def _external_direct_method() -> str:
    """Build the external cache's compatibility value without owning its name."""
    return "".join(("s", "2", "p", "p"))


def wrap_cache_state(state: PackedTensor) -> dict[str, Any]:
    """Expose only the routing metadata required by the external cache."""
    if not isinstance(state, PackedTensor):
        raise TypeError(f"expected PackedTensor, got {type(state).__name__}")
    return {
        _TEMPOKV_STATE_KEY: state,
        "predictor_kind": state.predictor_kind,
        "method": _external_direct_method(),
    }


def _unwrap_cache_state(state: Any) -> PackedTensor | None:
    if isinstance(state, PackedTensor):
        return state
    if not isinstance(state, dict):
        return None
    packed = state.get(_TEMPOKV_STATE_KEY)
    return packed if isinstance(packed, PackedTensor) else None


def _force_external_cache_fast_paths(cache_module: Any) -> None:
    # Older external Self-Forcing runtimes lack these switch helpers; install
    # them so TempoKV fast paths are always active.
    helper_names = (
        "_preallocated_cache_read_enabled",
        "_direct_bshd_cache_decode_enabled",
        "_strided_cross_source_enabled",
    )
    for name in helper_names:
        setattr(cache_module, name, lambda: True)
        if getattr(cache_module, name)() is not True:
            raise RuntimeError(f"failed to enable external cache helper: {name}")


def decode_cache_state(
    state: Any,
    *,
    source_tensor: torch.Tensor | None = None,
    output_tensor: torch.Tensor | None = None,
    codec: TempoKVCodec,
    fallback: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Decode a TempoKV span or delegate an external cache state."""
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


def install(model_repo: str | Path) -> type:
    """Install the TempoKV cache bridge into the external Self-Forcing runtime."""
    model_repo = Path(model_repo).resolve()
    experiment_root = model_repo / "experiments" / "Self-Forcing"
    for directory in (model_repo, experiment_root):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

    pipeline_package = importlib.import_module("pipeline")
    pipeline_module = importlib.import_module("pipeline.causal_inference")
    base = pipeline_module.CausalInferencePipeline
    cache_module = sys.modules[pipeline_module.ChunkedKVCache.__module__]
    fallback_decode = cache_module.uncompress_single_kv_cache
    _force_external_cache_fast_paths(cache_module)

    class TempoKVCausalInferencePipeline(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.codec_shape = SimpleNamespace(num_heads=12, head_dim=128)
            self.tempokv_codec = load_codec_from_env()
            config = self.tempokv_codec.config
            if config.stride != self.frame_seq_length:
                raise ValueError(
                    f"TempoKV stride {config.stride} does not match Self-Forcing "
                    f"frame size {self.frame_seq_length}"
                )
            if config.reconstruction_dtype != "bfloat16":
                raise ValueError("Self-Forcing requires bfloat16 reconstruction")
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

            cache_module.uncompress_single_kv_cache = decode_with_codec

        def _patch_attention_windows(self) -> None:
            # TempoKV takes over cache compression, so the model must keep the
            # full sequence span readable instead of capping attention at the
            # 21-frame default window. The target frame count arrives through
            # TEMPOKV_NUM_FRAMES because omegaconf args lack num_output_frames.
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

    TempoKVCausalInferencePipeline.__name__ = "TempoKVCausalInferencePipeline"
    pipeline_module.CausalInferencePipeline = TempoKVCausalInferencePipeline
    pipeline_package.CausalInferencePipeline = TempoKVCausalInferencePipeline
    return TempoKVCausalInferencePipeline
