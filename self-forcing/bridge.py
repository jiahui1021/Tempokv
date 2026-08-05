from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any, Callable

import torch

from tempokv.codec import PackedTensor, TempoKVCodec, decode_tensor
from tempokv.integration import load_codec_from_env


def decode_cache_state(
    state: Any,
    *,
    source_tensor: torch.Tensor | None = None,
    output_tensor: torch.Tensor | None = None,
    codec: TempoKVCodec,
    fallback: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Decode a TempoKV span or delegate an external cache state."""
    if not isinstance(state, PackedTensor):
        if fallback is None:
            raise TypeError(f"unsupported cache state: {type(state).__name__}")
        return fallback(
            state, source_tensor=source_tensor, output_tensor=output_tensor
        )
    rebuilt = decode_tensor(
        state,
        source=source_tensor,
        use_triton=codec.config.use_triton,
        strict_triton=codec.config.strict_triton,
    )
    if output_tensor is not None:
        expected = rebuilt.permute(0, 2, 1, 3)
        if output_tensor.shape != expected.shape:
            raise ValueError("preallocated cache output has the wrong shape")
        output_tensor.copy_(expected)
    return rebuilt


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
                    values = layer["v"].read_bhsd(
                        tokens_to_quantize_start, tokens_to_quantize_end
                    )
                    if (
                        keys.shape[1] != self.codec_shape.num_heads
                        or keys.shape[-1] != self.codec_shape.head_dim
                    ):
                        raise ValueError("Self-Forcing cache shape changed unexpectedly")
                    packed = self.tempokv_codec.encode(
                        keys, values, layer=layer_index
                    )
                    layer["k"].store_quantized(
                        tokens_to_quantize_start,
                        tokens_to_quantize_end,
                        packed.key,
                    )
                    layer["v"].store_quantized(
                        tokens_to_quantize_start,
                        tokens_to_quantize_end,
                        packed.value,
                    )
                    del keys, values, packed
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
