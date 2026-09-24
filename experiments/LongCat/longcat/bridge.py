"""Runtime adapter between LongCat's KV cache and TempoKV.

Clean pre-RoPE K/V are encoded at the cache boundary; the primary tensor is
decoded first so the cross predictor sees the same reconstruction as the encoder.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
import importlib
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from tempokv.codec import PackedTensor, TempoKVCodec, decode_tensor
from tempokv.integration import load_codec
from tempokv.parameters import CrossKVParameters

from .trace import DEFAULT_TRACE_SPATIAL_POSITIONS, LongCatTraceCollector


LONGCAT_NUM_LAYERS = 48
LONGCAT_NUM_HEADS = 32
LONGCAT_HEAD_DIM = 128
LONGCAT_STRIDE = 1560
LONGCAT_COORDINATE_SYSTEM = "pre_rope_normalized"
_RECONSTRUCTION_ELEMENT_SIZE = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}
_RECONSTRUCTION_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True, slots=True)
class LongCatRuntimeSpec:
    """Shape and cache constraints for a LongCat runtime."""

    num_layers: int = LONGCAT_NUM_LAYERS
    num_heads: int = LONGCAT_NUM_HEADS
    head_dim: int = LONGCAT_HEAD_DIM
    stride: int | None = LONGCAT_STRIDE
    coordinate_system: str = LONGCAT_COORDINATE_SYSTEM


@dataclass(frozen=True, slots=True)
class PackedTensorSpanSet:
    """Independent TempoKV spans (e.g. sink + recent) concatenated at the cache boundary."""

    shape: tuple[int, int, int, int]
    predictor_kind: str
    spans: tuple[PackedTensor, ...]
    # Newest conditioning latents kept exact, after the packed spans.
    tail: torch.Tensor | None = None

    def storage_bytes(self) -> int:
        tail_bytes = (
            0 if self.tail is None else self.tail.numel() * self.tail.element_size()
        )
        return sum(span.storage_bytes() for span in self.spans) + tail_bytes


PackedState = PackedTensor | PackedTensorSpanSet


def _move_tensor(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    return None if value is None else value.to(device)


def move_packed_tensor(packed: PackedState, device: torch.device | str) -> PackedState:
    """Move PackedTensor storage between devices; predictor parameters stay with the codec."""
    if isinstance(packed, PackedTensorSpanSet):
        return replace(
            packed,
            spans=tuple(move_packed_tensor(span, device) for span in packed.spans),
            tail=_move_tensor(packed.tail, torch.device(device)),
        )
    target = torch.device(device)
    components = (
        packed.anchor_quant,
        packed.anchor_scale,
        packed.residual_quant,
        packed.residual_scale,
        packed.residual_zero_point,
    )
    if all(component is None or component.device == target for component in components):
        return packed
    return replace(
        packed,
        anchor_quant=packed.anchor_quant.to(target),
        anchor_scale=packed.anchor_scale.to(target),
        residual_quant=_move_tensor(packed.residual_quant, target),
        residual_scale=_move_tensor(packed.residual_scale, target),
        residual_zero_point=_move_tensor(packed.residual_zero_point, target),
    )


def move_kv_cache(
    kv_cache: tuple[Any, Any], device: torch.device | str
) -> tuple[Any, Any]:
    """Move a LongCat cache pair, supporting both tensors and PackedTensor."""
    target = torch.device(device)
    key, value = (
        item.to(target)
        if isinstance(item, torch.Tensor)
        else move_packed_tensor(item, target)
        for item in kv_cache
    )
    return key, value


def decode_kv_cache(
    k_cache: Any,
    v_cache: Any,
    codec: TempoKVCodec,
    *,
    key_output: torch.Tensor | None = None,
    value_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode LongCat K/V in the required K-first closed-loop order."""
    if isinstance(k_cache, torch.Tensor):
        if key_output is not None:
            key_output.copy_(k_cache)
            k_cache = key_output
        if value_output is not None:
            value_output.copy_(v_cache)
            v_cache = value_output
        return k_cache, v_cache
    # Forward: K temporal / V cross; mirrored: V temporal / K cross.
    if isinstance(k_cache, PackedTensorSpanSet):
        output_dtype = _RECONSTRUCTION_DTYPE[
            k_cache.spans[0].reconstruction_dtype
        ]
        device = k_cache.spans[0].anchor_quant.device
        key = _prepare_decode_output(
            key_output,
            shape=k_cache.shape,
            dtype=output_dtype,
            device=device,
        )
        value = _prepare_decode_output(
            value_output,
            shape=v_cache.shape,
            dtype=output_dtype,
            device=device,
        )
        start = 0
        mirrored = codec.config.mirrored
        for packed_key, packed_value in zip(k_cache.spans, v_cache.spans):
            end = start + int(packed_key.shape[2])
            key_target = key[:, :, start:end, :]
            value_target = value[:, :, start:end, :]
            packed_primary, packed_secondary = (
                (packed_value, packed_key) if mirrored else (packed_key, packed_value)
            )
            primary_target, secondary_target = (
                (value_target, key_target) if mirrored else (key_target, value_target)
            )
            primary_part = decode_tensor(
                packed_primary,
                output_tensor=primary_target,
                output_layout="BHSD",
                use_triton=codec.config.use_triton,
                strict_triton=codec.config.strict_triton,
            )
            decode_tensor(
                packed_secondary,
                source=primary_part,
                output_tensor=secondary_target,
                output_layout="BHSD",
                use_triton=codec.config.use_triton,
                strict_triton=codec.config.strict_triton,
            )
            start = end
        if k_cache.tail is not None:
            key[:, :, start:, :].copy_(k_cache.tail)
            value[:, :, start:, :].copy_(v_cache.tail)
        return key, value
    output_dtype = _RECONSTRUCTION_DTYPE[k_cache.reconstruction_dtype]
    key = _prepare_decode_output(
        key_output,
        shape=k_cache.shape,
        dtype=output_dtype,
        device=k_cache.anchor_quant.device,
    )
    value = _prepare_decode_output(
        value_output,
        shape=v_cache.shape,
        dtype=output_dtype,
        device=v_cache.anchor_quant.device,
    )
    mirrored = codec.config.mirrored
    packed_primary, packed_secondary = (
        (v_cache, k_cache) if mirrored else (k_cache, v_cache)
    )
    primary_output, secondary_output = (
        (value, key) if mirrored else (key, value)
    )
    decoded_primary = decode_tensor(
        packed_primary,
        output_tensor=primary_output,
        output_layout="BHSD",
        use_triton=codec.config.use_triton,
        strict_triton=codec.config.strict_triton,
    )
    decode_tensor(
        packed_secondary,
        source=decoded_primary,
        output_tensor=secondary_output,
        output_layout="BHSD",
        use_triton=codec.config.use_triton,
        strict_triton=codec.config.strict_triton,
    )
    return key, value


def _prepare_decode_output(
    output: torch.Tensor | None,
    *,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device) if output is None else output


def validate_cache_layout(
    kv_cache_dict: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    temporal_units: int,
    codec: TempoKVCodec,
    spec: LongCatRuntimeSpec | None = None,
) -> dict[str, int | str]:
    """Validate a clean LongCat cache and calculate its runtime stride."""
    resolved_spec = spec or LongCatRuntimeSpec()
    layer_ids = sorted(int(layer) for layer in kv_cache_dict)
    if layer_ids != list(range(resolved_spec.num_layers)):
        raise ValueError(
            f"LongCat checkpoint must expose {resolved_spec.num_layers} layers, "
            f"got {len(layer_ids)}"
        )

    first_key, _ = kv_cache_dict[layer_ids[0]]
    batch, heads, sequence, width = (int(item) for item in first_key.shape)
    if heads != resolved_spec.num_heads or width != resolved_spec.head_dim:
        raise ValueError(
            "LongCat cache shape mismatch: expected "
            f"H={resolved_spec.num_heads}, D={resolved_spec.head_dim}; "
            f"got H={heads}, D={width}"
        )
    if sequence % temporal_units:
        raise ValueError(
            f"LongCat cache sequence {sequence} is not divisible by "
            f"temporal_units {temporal_units}"
        )
    runtime_stride = sequence // temporal_units
    if resolved_spec.stride is not None and runtime_stride != resolved_spec.stride:
        raise ValueError(
            f"LongCat runtime stride is {runtime_stride}, expected {resolved_spec.stride}"
        )
    if codec.config.stride != runtime_stride:
        raise ValueError(
            f"TempoKV stride {codec.config.stride} does not match LongCat runtime "
            f"stride {runtime_stride}"
        )

    return {
        "batch": batch,
        "num_layers": len(layer_ids),
        "num_heads": heads,
        "head_dim": width,
        "sequence": sequence,
        "temporal_units": temporal_units,
        "stride": runtime_stride,
        "coordinate_system": resolved_spec.coordinate_system,
    }


def encode_kv_cache(
    kv_cache_dict: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    codec: TempoKVCodec,
    temporal_units: int,
    spec: LongCatRuntimeSpec | None = None,
    encode_device: torch.device | str | None = None,
    storage_device: torch.device | str | None = None,
    profile: bool = False,
) -> tuple[dict[int, tuple[PackedState, PackedState]], dict[str, object]]:
    """Encode all clean LongCat layers with K-first closed-loop semantics."""
    layout = validate_cache_layout(
        kv_cache_dict,
        temporal_units=temporal_units,
        codec=codec,
        spec=spec,
    )
    uncompressed_cache_bytes = sum(
        int(key.numel() * key.element_size() + value.numel() * value.element_size())
        for key, value in kv_cache_dict.values()
    )
    encoded: dict[int, tuple[PackedState, PackedState]] = {}
    target_device = None if encode_device is None else torch.device(encode_device)
    started = time.perf_counter() if profile else None
    with torch.no_grad():
        for layer_id in sorted(kv_cache_dict):
            key, value = kv_cache_dict[layer_id]
            if target_device is not None:
                key = key.to(target_device)
                value = value.to(target_device)
            key = key.contiguous()
            value = value.contiguous()
            # Encode by role: primary first, then secondary from its reconstruction.
            mirrored = codec.config.mirrored
            primary, secondary = (value, key) if mirrored else (key, value)
            packed_primary, reconstructed_primary = codec.encode_primary(
                primary, layer=int(layer_id), reuse_input=True
            )
            packed_secondary = codec.encode_secondary(
                secondary,
                reconstructed_primary,
                layer=int(layer_id),
                reuse_input=True,
            )
            packed_key, packed_value = (
                (packed_secondary, packed_primary)
                if mirrored
                else (packed_primary, packed_secondary)
            )
            if storage_device is not None:
                packed_key = move_packed_tensor(packed_key, storage_device)
                packed_value = move_packed_tensor(packed_value, storage_device)
            encoded[int(layer_id)] = (packed_key, packed_value)
            del key, value, reconstructed_primary
    storage_bytes = sum(
        packed.storage_bytes()
        for pair in encoded.values()
        for packed in pair
    )
    reconstruction_element_size = _RECONSTRUCTION_ELEMENT_SIZE[
        codec.config.reconstruction_dtype
    ]
    temporary_reconstruction_bytes = (
        int(layout["batch"])
        * int(layout["num_heads"])
        * int(layout["sequence"])
        * int(layout["head_dim"])
        * reconstruction_element_size
        * 2
    )
    return encoded, {
        "codec": "TempoKV",
        "encode_key_calls": int(len(encoded)),
        "encode_value_calls": int(len(encoded)),
        "layers": int(layout["num_layers"]),
        "num_heads": int(layout["num_heads"]),
        "head_dim": int(layout["head_dim"]),
        "sequence": int(layout["sequence"]),
        "temporal_units": int(layout["temporal_units"]),
        "stride": int(layout["stride"]),
        "coordinate_system": str(layout["coordinate_system"]),
        "storage_bytes": int(storage_bytes),
        "uncompressed_cache_bytes": int(uncompressed_cache_bytes),
        "compression_ratio": (
            float(uncompressed_cache_bytes / storage_bytes)
            if storage_bytes
            else None
        ),
        "temporary_reconstruction_bytes": temporary_reconstruction_bytes,
        "enqueue_duration_ms": (
            (time.perf_counter() - started) * 1000.0
            if started is not None
            else None
        ),
    }


def _env_span_splits(temporal_units: int) -> tuple[int, ...] | None:
    """Parse ``TEMPOKV_SPAN_SPLITS="first+step"`` (e.g. ``"19+5"``) into span sizes.

    Returns ``None`` when unset or too short, keeping one anchor for the video.
    """
    raw = os.environ.get("TEMPOKV_SPAN_SPLITS", "").strip()
    if not raw:
        return None
    try:
        first, step = (int(part) for part in raw.split("+", 1))
    except ValueError as error:
        raise ValueError(
            f"TEMPOKV_SPAN_SPLITS must look like '19+5', got {raw!r}"
        ) from error
    if first < 2 or step < 1:
        raise ValueError("TEMPOKV_SPAN_SPLITS parts must be >= 2 and >= 1")
    if temporal_units <= first:
        return None
    units = [first]
    remaining = temporal_units - first
    while remaining > step:
        units.append(step)
        remaining -= step
    if remaining == 1:
        # A one-unit span carries only an anchor; merge it into the last span.
        units[-1] += 1
    else:
        units.append(remaining)
    assert sum(units) == temporal_units
    return tuple(units)


def encode_kv_cache_spans(
    kv_cache_dict: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    codec: TempoKVCodec,
    span_temporal_units: Sequence[int],
    spec: LongCatRuntimeSpec | None = None,
    encode_device: torch.device | str | None = None,
    storage_device: torch.device | str | None = None,
    profile: bool = False,
) -> tuple[dict[int, tuple[PackedTensorSpanSet, PackedTensorSpanSet]], dict[str, object]]:
    """Encode independent temporal spans and concatenate them for attention."""
    units = tuple(int(item) for item in span_temporal_units)
    layout = validate_cache_layout(
        kv_cache_dict,
        temporal_units=sum(units),
        codec=codec,
        spec=spec,
    )
    stride = int(layout["stride"])
    span_cache: list[Mapping[int, tuple[torch.Tensor, torch.Tensor]]] = []
    start = 0
    for temporal_units in units:
        end = start + temporal_units * stride
        span_cache.append(
            {
                int(layer): (
                    pair[0][:, :, start:end, :],
                    pair[1][:, :, start:end, :],
                )
                for layer, pair in kv_cache_dict.items()
            }
        )
        start = end
    if start != int(layout["sequence"]):
        raise ValueError("temporal spans do not cover the LongCat cache")

    per_layer: dict[int, list[list[PackedState]]] = {
        int(layer): [[], []] for layer in kv_cache_dict
    }
    reports = []
    for cache, temporal_units in zip(span_cache, units):
        encoded, report = encode_kv_cache(
            cache,
            codec=codec,
            temporal_units=temporal_units,
            spec=spec,
            encode_device=encode_device,
            storage_device=storage_device,
            profile=profile,
        )
        reports.append(report)
        for layer, (packed_key, packed_value) in encoded.items():
            per_layer[layer][0].append(packed_key)
            per_layer[layer][1].append(packed_value)

    combined: dict[int, tuple[PackedTensorSpanSet, PackedTensorSpanSet]] = {}
    for layer, (key_parts, value_parts) in per_layer.items():
        key_spans = tuple(key_parts)
        value_spans = tuple(value_parts)
        combined[layer] = (
            PackedTensorSpanSet(
                shape=(
                    key_spans[0].shape[0],
                    key_spans[0].shape[1],
                    sum(item.shape[2] for item in key_spans),
                    key_spans[0].shape[3],
                ),
                # Take the kind from the spans, which the mirrored direction reverses.
                predictor_kind=key_spans[0].predictor_kind,
                spans=key_spans,
            ),
            PackedTensorSpanSet(
                shape=(
                    value_spans[0].shape[0],
                    value_spans[0].shape[1],
                    sum(item.shape[2] for item in value_spans),
                    value_spans[0].shape[3],
                ),
                predictor_kind=value_spans[0].predictor_kind,
                spans=value_spans,
            ),
        )
    storage_bytes = sum(
        packed.storage_bytes()
        for pair in combined.values()
        for packed in pair
    )
    uncompressed_cache_bytes = sum(
        int(report["uncompressed_cache_bytes"]) for report in reports
    )
    return combined, {
        "codec": "TempoKV",
        "encode_key_calls": sum(
            int(report["encode_key_calls"]) for report in reports
        ),
        "encode_value_calls": sum(
            int(report["encode_value_calls"]) for report in reports
        ),
        "layers": int(layout["num_layers"]),
        "num_heads": int(layout["num_heads"]),
        "head_dim": int(layout["head_dim"]),
        "sequence": int(layout["sequence"]),
        "temporal_units": list(units),
        "stride": stride,
        "coordinate_system": str(layout["coordinate_system"]),
        "spans": len(units),
        "storage_bytes": storage_bytes,
        "uncompressed_cache_bytes": uncompressed_cache_bytes,
        "compression_ratio": (
            float(uncompressed_cache_bytes / storage_bytes)
            if storage_bytes
            else None
        ),
        "temporary_reconstruction_bytes": max(
            int(report["temporary_reconstruction_bytes"]) for report in reports
        ),
        "enqueue_duration_ms": (
            sum(float(report["enqueue_duration_ms"]) for report in reports)
            if profile
            else None
        ),
    }


_ACTIVE_BRIDGE: ContextVar["LongCatTempoKVBridge | None"] = ContextVar(
    "tempokv_longcat_active_bridge", default=None
)


def _unwrap_compiled_module(value: Any) -> Any:
    """Return the original module behind torch.compile wrappers."""
    current = value
    seen: set[int] = set()
    for _ in range(8):
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        original = getattr(current, "_orig_mod", None)
        if original is None or original is current:
            break
        current = original
    return current


@dataclass(slots=True)
class LongCatTempoKVBridge:
    """Install and own a TempoKV bridge for one LongCat runtime."""

    codec: TempoKVCodec | None
    mode: str = "tempokv"
    trace_dir: Path | None = None
    max_trace_spatial_positions: int = DEFAULT_TRACE_SPATIAL_POSITIONS
    spec: LongCatRuntimeSpec = field(default_factory=LongCatRuntimeSpec)
    context_parallel_size: int = 1
    context_parallel_rank: int = 0
    attention_sink_frames: int = 0
    auto_context_parallel_stride: bool = True
    profile_encode: bool = False
    profile_decode: bool = False
    trace_collector: LongCatTraceCollector | None = field(init=False, default=None)
    local_trace_spatial_positions: int = field(init=False, default=0)
    _installed: bool = field(init=False, default=False)
    trace_events: list[dict[str, object]] = field(init=False, default_factory=list)
    quantization_events: list[dict[str, object]] = field(init=False, default_factory=list)
    decode_events: list[dict[str, object]] = field(init=False, default_factory=list)
    decode_calls: int = field(init=False, default=0)
    decode_tensor_calls: int = field(init=False, default=0)
    decoded_sequence_tokens: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.mode not in {"tempokv", "bf16", "trace"}:
            raise ValueError("LongCat bridge mode must be tempokv, bf16, or trace")
        if self.mode == "tempokv" and self.codec is None:
            raise ValueError("tempokv mode requires a loaded TempoKV codec")
        if (
            self.context_parallel_size > 1
            and self.auto_context_parallel_stride
            and self.spec.stride == LONGCAT_STRIDE
        ):
            self.spec = replace(self.spec, stride=None)
        if self.trace_dir is not None:
            if self.max_trace_spatial_positions < self.context_parallel_size:
                raise ValueError(
                    "global trace spatial budget must be at least context_parallel_size"
                )
            local_budget, remainder = divmod(
                self.max_trace_spatial_positions, self.context_parallel_size
            )
            if self.context_parallel_rank < remainder:
                local_budget += 1
            self.local_trace_spatial_positions = local_budget
            self.trace_collector = LongCatTraceCollector(
                Path(self.trace_dir),
                max_spatial_positions=self.local_trace_spatial_positions,
            )

    def validate_model(self, dit: Any) -> None:
        model = _unwrap_compiled_module(dit)
        blocks = model.blocks
        if len(blocks) != self.spec.num_layers:
            raise ValueError(
                f"LongCat checkpoint must expose {self.spec.num_layers} blocks, got {len(blocks)}"
            )
        if (
            self.context_parallel_size == 1
            and self.codec is not None
            and self.spec.stride is not None
            and self.codec.config.stride != self.spec.stride
        ):
            raise ValueError(
                f"TempoKV config stride {self.codec.config.stride} does not match "
                f"LongCat CP=1 stride {self.spec.stride}"
            )
        cp_split_hw = getattr(dit, "cp_split_hw", None)
        actual_context_parallel_size = 1
        if cp_split_hw is not None:
            actual_context_parallel_size = int(cp_split_hw[0]) * int(cp_split_hw[1])
        if actual_context_parallel_size != self.context_parallel_size:
            raise ValueError(
                "LongCat context parallel mismatch: bridge expects "
                f"{self.context_parallel_size}, model uses {actual_context_parallel_size}"
            )
        if self.codec is not None:
            # The codec signature already pins bits, modes, history lengths and
            # stride; what it cannot see is which model the trace came from.
            expected_layers = tuple(range(self.spec.num_layers))
            for name, parameters in (
                ("K", self.codec.temporal),
                ("V", self.codec.v_parameters),
            ):
                if parameters.coordinate_system != self.spec.coordinate_system:
                    raise ValueError(
                        f"{name} parameters use coordinate system "
                        f"{parameters.coordinate_system!r}; LongCat requires "
                        f"{self.spec.coordinate_system!r}"
                    )
                if parameters.layered and parameters.layer_ids != expected_layers:
                    raise ValueError(
                        f"{name} parameters must contain layer ids 0 through "
                        f"{self.spec.num_layers - 1}, got {parameters.layer_ids}"
                    )
                if isinstance(parameters, CrossKVParameters):
                    geometry = tuple(parameters.weight.shape[-3:-1])
                elif parameters.layered or parameters.per_head:
                    geometry = tuple(parameters.history_weights.shape[-2:])
                else:
                    geometry = (self.spec.num_heads, int(parameters.history_weights.shape[-1]))
                # A one-head map would otherwise broadcast silently over every head.
                if geometry != (self.spec.num_heads, self.spec.head_dim):
                    raise ValueError(
                        f"{name} parameters have [heads, head_dim]={list(geometry)}; "
                        f"LongCat has [{self.spec.num_heads}, {self.spec.head_dim}]"
                    )
        for index, block_wrapper in enumerate(blocks):
            attention = _unwrap_compiled_module(block_wrapper).attn
            heads = int(getattr(attention, "num_heads", -1))
            head_dim = int(getattr(attention, "head_dim", -1))
            if heads != self.spec.num_heads or head_dim != self.spec.head_dim:
                raise ValueError(
                    f"LongCat block {index} has H={heads}, D={head_dim}; expected "
                    f"H={self.spec.num_heads}, D={self.spec.head_dim}"
                )
            layer_index = getattr(attention, "layer_idx", None)
            if layer_index is not None and int(layer_index) != index:
                raise ValueError(
                    f"LongCat block {index} attention layer_idx is {layer_index}"
                )

    def attach(self, dit: Any, pipeline: Any | None = None) -> None:
        """Attach bridge state to model instances after they are loaded."""
        self.validate_model(dit)
        setattr(dit, "_tempokv_bridge", self)
        setattr(dit, "_tempokv_codec", self.codec)
        model = _unwrap_compiled_module(dit)
        setattr(model, "_tempokv_bridge", self)
        setattr(model, "_tempokv_codec", self.codec)
        for block_wrapper in model.blocks:
            block = _unwrap_compiled_module(block_wrapper)
            setattr(block.attn, "_tempokv_bridge", self)
        if pipeline is not None:
            setattr(pipeline, "_tempokv_bridge", self)
            self._patch_vae_decode(pipeline)

    @staticmethod
    def _patch_vae_decode(pipeline: Any) -> None:
        """Drop the shared RoPE table immediately before VAE allocation peaks."""
        vae = getattr(pipeline, "vae", None)
        if vae is None or getattr(vae, "_tempokv_rope_release_patch", False):
            return
        original_decode = getattr(vae, "decode", None)
        if not callable(original_decode):
            return

        def decode(*args: Any, **kwargs: Any) -> Any:
            from longcat_video.modules.rope_3d import (
                clear_shared_rope_device_cache,
            )

            clear_shared_rope_device_cache()
            return original_decode(*args, **kwargs)

        setattr(vae, "decode", decode)
        setattr(vae, "_tempokv_rope_release_patch", True)

    def on_clean_cache(self, pipeline: Any, cond_latents: torch.Tensor) -> None:
        temporal_units = int(cond_latents.shape[2])
        dit = pipeline.dit
        setattr(dit, "_tempokv_num_units", temporal_units)
        sink_units = 0
        if self.attention_sink_frames:
            temporal_factor = int(getattr(pipeline, "vae_scale_factor_temporal", 4))
            sink_units = 1 + (self.attention_sink_frames - 1) // temporal_factor
            if sink_units >= temporal_units:
                raise ValueError(
                    "attention sink must leave at least one latest conditioning latent"
                )
            setattr(
                dit,
                "_tempokv_span_temporal_units",
                (sink_units, temporal_units - sink_units),
            )
        else:
            setattr(dit, "_tempokv_span_temporal_units", None)
        if self.trace_collector is not None:
            cache = pipeline._get_kv_cache_dict()
            if sink_units:
                runtime_stride = int(cache[0][0].shape[2]) // temporal_units
                cache = {
                    int(layer): (
                        pair[0][:, :, sink_units * runtime_stride :, :],
                        pair[1][:, :, sink_units * runtime_stride :, :],
                    )
                    for layer, pair in cache.items()
                }
                event = self.trace_collector.record(
                    cache, temporal_units=temporal_units - sink_units
                )
            else:
                event = self.trace_collector.record(cache, temporal_units=temporal_units)
            self.trace_events.append(event)

    def quantize(
        self,
        dit: Any,
        kv_cache_dict: Mapping[int, tuple[Any, Any]],
        *,
        offload: bool = False,
    ) -> Any:
        if self.mode in {"bf16", "trace"}:
            # BF16 runs the original uncompressed LongCat attention path.  Trace
            # collection uses the same clean path after saving the raw cache.
            return kv_cache_dict
        if kv_cache_dict and all(
            isinstance(pair[0], (PackedTensor, PackedTensorSpanSet))
            and isinstance(pair[1], (PackedTensor, PackedTensorSpanSet))
            for pair in kv_cache_dict.values()
        ):
            if not offload:
                return kv_cache_dict
            moved = {
                int(layer): move_kv_cache(pair, torch.device("cpu"))
                for layer, pair in kv_cache_dict.items()
            }
            if isinstance(kv_cache_dict, MutableMapping):
                kv_cache_dict.clear()
                kv_cache_dict.update(moved)
                return kv_cache_dict
            return moved
        temporal_units = int(getattr(dit, "_tempokv_num_units", 0))
        if temporal_units <= 0:
            raise ValueError(
                "LongCat temporal units are unknown; clean KV must be generated "
                "through LongCatVideoPipeline._cache_clean_latents first"
            )
        encode_device = None
        storage_device = None
        if offload:
            storage_device = torch.device("cpu")
            try:
                encode_device = next(dit.parameters()).device
            except (AttributeError, StopIteration):
                encode_device = next(iter(kv_cache_dict.values()))[0].device
        span_units = getattr(dit, "_tempokv_span_temporal_units", None)
        if span_units is None:
            span_units = _env_span_splits(temporal_units)
        lag_units = int(os.environ.get("TEMPOKV_BF16_LAG_FRAMES", "0"))
        if lag_units:
            return self._quantize_with_bf16_tail(
                dit,
                kv_cache_dict,
                temporal_units=temporal_units,
                span_units=span_units,
                lag_units=lag_units,
                encode_device=encode_device,
                storage_device=storage_device,
            )
        if span_units is not None:
            encoded, report = encode_kv_cache_spans(
                kv_cache_dict,
                codec=self.codec,
                span_temporal_units=span_units,
                spec=self.spec,
                encode_device=encode_device,
                storage_device=storage_device,
                profile=self.profile_encode,
            )
        else:
            encoded, report = encode_kv_cache(
                kv_cache_dict,
                codec=self.codec,
                temporal_units=temporal_units,
                spec=self.spec,
                encode_device=encode_device,
                storage_device=storage_device,
                profile=self.profile_encode,
            )
        events = getattr(dit, "_tempokv_quantization_events", None)
        if events is None:
            events = []
            setattr(dit, "_tempokv_quantization_events", events)
        events.append(report)
        self.quantization_events.append(report)
        if isinstance(kv_cache_dict, MutableMapping):
            kv_cache_dict.clear()
            kv_cache_dict.update(encoded)
            return kv_cache_dict
        return encoded

    def _quantize_with_bf16_tail(
        self,
        dit: Any,
        kv_cache_dict: Mapping[int, tuple[Any, Any]],
        *,
        temporal_units: int,
        span_units: Sequence[int] | None,
        lag_units: int,
        encode_device: torch.device | None,
        storage_device: torch.device | None,
    ) -> Any:
        """Pack all but the newest ``lag_units`` conditioning latents (taken from the last span)."""
        assert self.codec is not None
        spans = (
            tuple(int(item) for item in span_units)
            if span_units is not None
            else (int(temporal_units),)
        )
        kept = min(lag_units, spans[-1])
        packed_spans = spans[:-1] + ((spans[-1] - kept,) if spans[-1] > kept else ())
        first_key = next(iter(kv_cache_dict.values()))[0]
        stride = int(first_key.shape[2]) // int(temporal_units)
        head_tokens = sum(packed_spans) * stride
        # Copy the tail before encoding: the codec reconstructs in place, and a
        # contiguous head slice shares storage with the clean cache.
        tails = {
            int(layer): (
                pair[0][:, :, head_tokens:, :].clone(),
                pair[1][:, :, head_tokens:, :].clone(),
            )
            for layer, pair in kv_cache_dict.items()
        }
        if not packed_spans:
            # The lag covers the whole conditioning window: nothing is packed.
            return kv_cache_dict
        head = {
            int(layer): (
                pair[0][:, :, :head_tokens, :],
                pair[1][:, :, :head_tokens, :],
            )
            for layer, pair in kv_cache_dict.items()
        }
        encoded, report = encode_kv_cache_spans(
            head,
            codec=self.codec,
            span_temporal_units=packed_spans,
            spec=self.spec,
            encode_device=encode_device,
            storage_device=storage_device,
            profile=self.profile_encode,
        )
        combined: dict[int, tuple[PackedTensorSpanSet, PackedTensorSpanSet]] = {}
        tail_bytes = 0
        for layer, (packed_key, packed_value) in encoded.items():
            key_tail, value_tail = tails[layer]
            if storage_device is not None:
                key_tail = key_tail.to(storage_device)
                value_tail = value_tail.to(storage_device)
            tail_bytes += (
                key_tail.numel() * key_tail.element_size()
                + value_tail.numel() * value_tail.element_size()
            )
            combined[layer] = (
                replace(
                    packed_key,
                    shape=(*packed_key.shape[:2], packed_key.shape[2] + key_tail.shape[2], packed_key.shape[3]),
                    tail=key_tail,
                ),
                replace(
                    packed_value,
                    shape=(*packed_value.shape[:2], packed_value.shape[2] + value_tail.shape[2], packed_value.shape[3]),
                    tail=value_tail,
                ),
            )
        storage_bytes = int(report["storage_bytes"]) + tail_bytes
        uncompressed = int(report["uncompressed_cache_bytes"]) + tail_bytes
        report = {
            **report,
            "temporal_units": int(temporal_units),
            "sequence": int(first_key.shape[2]),
            "bf16_lag_units": int(kept),
            "bf16_tail_bytes": int(tail_bytes),
            "storage_bytes": storage_bytes,
            "uncompressed_cache_bytes": uncompressed,
            "compression_ratio": (
                float(uncompressed / storage_bytes) if storage_bytes else None
            ),
        }
        events = getattr(dit, "_tempokv_quantization_events", None)
        if events is None:
            events = []
            setattr(dit, "_tempokv_quantization_events", events)
        events.append(report)
        self.quantization_events.append(report)
        if isinstance(kv_cache_dict, MutableMapping):
            kv_cache_dict.clear()
            kv_cache_dict.update(combined)
            return kv_cache_dict
        return combined

    def decode(
        self,
        k_cache: Any,
        v_cache: Any,
        *,
        key_output: torch.Tensor | None = None,
        value_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # No per-layer synchronization; profiling measures host enqueue time only.
        started = time.perf_counter() if self.profile_decode else None
        result = decode_kv_cache(
            k_cache,
            v_cache,
            self.codec,
            key_output=key_output,
            value_output=value_output,
        )
        span_count = (
            len(k_cache.spans) if isinstance(k_cache, PackedTensorSpanSet) else 1
        )
        decode_tensor_calls = (
            2 * span_count
            if isinstance(k_cache, (PackedTensor, PackedTensorSpanSet))
            else 0
        )
        sequence = int(result[0].shape[2])
        self.decode_calls += 1
        self.decode_tensor_calls += decode_tensor_calls
        self.decoded_sequence_tokens += sequence
        if started is not None:
            self.decode_events.append(
                {
                    "spans": span_count,
                    "decode_tensor_calls": decode_tensor_calls,
                    "sequence": sequence,
                    "enqueue_duration_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
        return result

    def report(self) -> dict[str, object]:
        """Return a serializable summary of bridge activity for one rank."""
        predictor_parameter_bytes = 0
        codec_config = None
        if self.codec is not None:
            # This is the post-override runtime config, rather than the source
            # JSON.  Recording it makes mixed-precision results self-describing.
            codec_config = asdict(self.codec.config)
            v_parameters = self.codec.v_parameters
            if self.codec.predicts_v_from_key:
                parameters = (
                    self.codec.temporal.history_weights,
                    self.codec.temporal.history_bias,
                    v_parameters.weight,
                    v_parameters.history_weights,
                    v_parameters.bias,
                )
            else:
                parameters = (
                    self.codec.temporal.history_weights,
                    self.codec.temporal.history_bias,
                    v_parameters.history_weights,
                    v_parameters.history_bias,
                )
            predictor_parameter_bytes = sum(
                int(parameter.numel() * parameter.element_size())
                for parameter in parameters
            )
        compressed_event_bytes = [
            int(event.get("storage_bytes", 0)) for event in self.quantization_events
        ]
        uncompressed_event_bytes = [
            int(event.get("uncompressed_cache_bytes", 0))
            for event in self.quantization_events
        ]
        current_compressed_cache_bytes = (
            compressed_event_bytes[-1] if compressed_event_bytes else 0
        )
        peak_compressed_cache_bytes = max(compressed_event_bytes, default=0)
        total_encoded_cache_bytes = sum(compressed_event_bytes)
        current_uncompressed_cache_bytes = (
            uncompressed_event_bytes[-1] if uncompressed_event_bytes else 0
        )
        peak_uncompressed_cache_bytes = max(uncompressed_event_bytes, default=0)
        total_uncompressed_cache_bytes = sum(uncompressed_event_bytes)
        encode_key_calls = sum(
            int(event.get("encode_key_calls", 0))
            for event in self.quantization_events
        )
        encode_value_calls = sum(
            int(event.get("encode_value_calls", 0))
            for event in self.quantization_events
        )
        decode_routes: dict[str, int] = {}
        try:
            from tempokv.triton_ops import route_report
        except ImportError:
            # Triton is optional for CPU-only development and tests.
            pass
        else:
            decode_routes = route_report()
        temporary_reconstruction_bytes = max(
            (
                int(event.get("temporary_reconstruction_bytes", 0))
                for event in self.quantization_events
            ),
            default=0,
        )
        observed_runtime_strides = sorted(
            {
                int(event["stride"])
                for event in self.quantization_events
                if event.get("stride") is not None
            }
        )
        observed_trace_strides = sorted(
            {
                int(event.get("trace_stride", event.get("stride")))
                for event in self.trace_events
                if event.get("trace_stride", event.get("stride")) is not None
            }
        )
        reported_stride = (
            observed_runtime_strides[0]
            if len(observed_runtime_strides) == 1
            else (self.spec.stride if self.mode == "tempokv" else None)
        )
        return {
            "mode": self.mode,
            "k_bits": codec_config["k_bits"] if codec_config is not None else None,
            "v_bits": codec_config["v_bits"] if codec_config is not None else None,
            "anchor_bits": (
                codec_config["anchor_bits"] if codec_config is not None else None
            ),
            "codec_config": codec_config,
            "context_parallel_size": self.context_parallel_size,
            "context_parallel_rank": self.context_parallel_rank,
            "attention_sink_frames": self.attention_sink_frames,
            "profile_encode": self.profile_encode,
            "profile_decode": self.profile_decode,
            "num_layers": self.spec.num_layers,
            "num_heads": self.spec.num_heads,
            "head_dim": self.spec.head_dim,
            "stride": reported_stride,
            "configured_stride": self.spec.stride,
            "runtime_strides": observed_runtime_strides,
            "trace_strides": observed_trace_strides,
            "coordinate_system": self.spec.coordinate_system,
            "global_trace_spatial_positions": self.max_trace_spatial_positions,
            "local_trace_spatial_positions": self.local_trace_spatial_positions,
            "predictor_parameter_bytes": predictor_parameter_bytes,
            "encode_key_calls": encode_key_calls,
            "encode_value_calls": encode_value_calls,
            "decode_calls": self.decode_calls,
            "decode_tensor_calls": self.decode_tensor_calls,
            "decoded_sequence_tokens": self.decoded_sequence_tokens,
            "decode_routes": decode_routes,
            "current_compressed_cache_bytes": current_compressed_cache_bytes,
            "peak_compressed_cache_bytes": peak_compressed_cache_bytes,
            "total_encoded_cache_bytes": total_encoded_cache_bytes,
            "current_uncompressed_cache_bytes": current_uncompressed_cache_bytes,
            "peak_uncompressed_cache_bytes": peak_uncompressed_cache_bytes,
            "total_uncompressed_cache_bytes": total_uncompressed_cache_bytes,
            # Compatibility aliases now describe peak resident cache memory,
            # never cumulative encode traffic across continuation segments.
            "compressed_cache_bytes": peak_compressed_cache_bytes,
            "uncompressed_cache_bytes": peak_uncompressed_cache_bytes,
            "compression_ratio": (
                float(
                    peak_uncompressed_cache_bytes / peak_compressed_cache_bytes
                )
                if peak_compressed_cache_bytes
                else None
            ),
            "current_compression_ratio": (
                float(
                    current_uncompressed_cache_bytes
                    / current_compressed_cache_bytes
                )
                if current_compressed_cache_bytes
                else None
            ),
            "temporary_reconstruction_bytes": temporary_reconstruction_bytes,
            "trace_events": list(self.trace_events),
            "quantization_events": list(self.quantization_events),
            "decode_events": list(self.decode_events),
        }

    def install(
        self,
        *,
        dit_module: Any | None = None,
        attention_module: Any | None = None,
        pipeline_module: Any | None = None,
    ) -> "LongCatTempoKVBridge":
        """Patch the repository-local LongCat runtime modules."""
        if dit_module is None:
            dit_module = importlib.import_module(
                "longcat_video.modules.longcat_video_dit"
            )
        if attention_module is None:
            attention_module = importlib.import_module("longcat_video.modules.attention")
        if pipeline_module is None:
            pipeline_module = importlib.import_module(
                "longcat_video.pipeline_longcat_video"
            )

        self._patch_dit_module(dit_module)
        self._patch_attention_module(attention_module)
        self._patch_pipeline_module(pipeline_module)
        self._installed = True
        return self

    def _patch_dit_module(self, module: Any) -> None:
        model_class = module.LongCatVideoTransformer3DModel
        if not getattr(model_class, "_tempokv_quantize_patch", False):
            original_quantize = model_class.quantize_kv_cache

            def quantize_kv_cache(
                instance: Any,
                kv_cache_dict: Any,
                offload_kv_cache: bool = False,
            ) -> Any:
                bridge = getattr(instance, "_tempokv_bridge", None)
                if bridge is None:
                    return original_quantize(instance, kv_cache_dict, offload_kv_cache)
                return bridge.quantize(
                    instance, kv_cache_dict, offload=bool(offload_kv_cache)
                )

            model_class.quantize_kv_cache = quantize_kv_cache
            model_class._tempokv_quantize_patch = True

        original_onload = getattr(module, "_onload_kv_cache", None)
        if original_onload is not None and not getattr(module, "_tempokv_onload_patch", False):

            def onload_kv_cache(kv_cache: Any, device: torch.device | str) -> Any:
                if isinstance(kv_cache, (tuple, list)) and any(
                    isinstance(item, (PackedTensor, PackedTensorSpanSet)) for item in kv_cache
                ):
                    return move_kv_cache(kv_cache, device)
                return original_onload(kv_cache, device)

            module._onload_kv_cache = onload_kv_cache
            module._tempokv_onload_patch = True

    def _patch_attention_module(self, module: Any) -> None:
        original_uncompress = module.uncompress_kv_cache
        if not getattr(module, "_tempokv_uncompress_patch", False):

            def uncompress_kv_cache(
                k_cache: Any,
                v_cache: Any,
                *,
                key_output: torch.Tensor | None = None,
                value_output: torch.Tensor | None = None,
            ) -> Any:
                bridge = _ACTIVE_BRIDGE.get()
                if bridge is None or bridge.mode != "tempokv":
                    if key_output is not None or value_output is not None:
                        raise RuntimeError(
                            "direct LongCat cache output requires an active TempoKV bridge"
                        )
                    return original_uncompress(k_cache, v_cache)
                if isinstance(k_cache, (PackedTensor, PackedTensorSpanSet)):
                    return bridge.decode(
                        k_cache,
                        v_cache,
                        key_output=key_output,
                        value_output=value_output,
                    )
                if key_output is not None or value_output is not None:
                    raise TypeError("direct LongCat cache output requires packed TempoKV K/V")
                return original_uncompress(k_cache, v_cache)

            module.uncompress_kv_cache = uncompress_kv_cache
            module._tempokv_uncompress_patch = True

        attention_class = module.Attention
        if not getattr(attention_class, "_tempokv_attention_patch", False):
            original_forward = attention_class.forward_with_kv_cache

            def forward_with_kv_cache(instance: Any, *args: Any, **kwargs: Any) -> Any:
                bridge = getattr(instance, "_tempokv_bridge", None)
                if bridge is None:
                    return original_forward(instance, *args, **kwargs)
                token = _ACTIVE_BRIDGE.set(bridge)
                try:
                    return original_forward(instance, *args, **kwargs)
                finally:
                    _ACTIVE_BRIDGE.reset(token)

            attention_class.forward_with_kv_cache = forward_with_kv_cache
            attention_class._tempokv_attention_patch = True

    def _patch_pipeline_module(self, module: Any) -> None:
        pipeline_class = module.LongCatVideoPipeline
        if getattr(pipeline_class, "_tempokv_pipeline_patch", False):
            return
        original_clean_cache = pipeline_class._cache_clean_latents

        def cache_clean_latents(
            instance: Any,
            cond_latents: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            bridge = getattr(instance, "_tempokv_bridge", None)
            result = original_clean_cache(instance, cond_latents, *args, **kwargs)
            if bridge is not None:
                # Trace the original normalized, pre-RoPE BF16 cache before
                # ownership is transferred to the packed TempoKV state.
                bridge.on_clean_cache(instance, cond_latents)
                if bridge.mode == "tempokv":
                    offload = kwargs.get("offload_kv_cache")
                    if offload is None and len(args) >= 2:
                        offload = args[1]
                    cache = instance._get_kv_cache_dict()
                    cache = instance.dit.quantize_kv_cache(
                        cache,
                        offload_kv_cache=bool(offload),
                    )
                    update_cache = getattr(instance, "_update_kv_cache_dict", None)
                    if callable(update_cache):
                        update_cache(cache)
                    else:
                        instance.kv_cache_dict = cache
            return result

        pipeline_class._cache_clean_latents = cache_clean_latents
        pipeline_class._tempokv_pipeline_patch = True


def install_longcat_bridge(
    *,
    config_path: str | Path | None = None,
    k_parameters_path: str | Path | None = None,
    v_parameters_path: str | Path | None = None,
    triton_cache: str | Path | None = None,
    codec: TempoKVCodec | None = None,
    codec_overrides: Mapping[str, Any] | None = None,
    mode: str = "tempokv",
    trace_dir: str | Path | None = None,
    max_trace_spatial_positions: int = DEFAULT_TRACE_SPATIAL_POSITIONS,
    spec: LongCatRuntimeSpec | None = None,
    context_parallel_size: int = 1,
    context_parallel_rank: int = 0,
    attention_sink_frames: int = 0,
    profile_encode: bool = False,
    profile_decode: bool = False,
    **modules: Any,
) -> LongCatTempoKVBridge:
    """Create and install a LongCat bridge from a codec or config/parameter paths."""
    if codec is None and mode == "tempokv":
        paths = (config_path, k_parameters_path, v_parameters_path)
        if any(path is None for path in paths):
            raise ValueError(
                "tempokv mode requires config_path, k_parameters_path, and "
                "v_parameters_path when codec is not supplied"
            )
        codec = load_codec(
            paths[0],
            paths[1],
            paths[2],
            triton_cache=triton_cache,
            overrides=codec_overrides,
        )
    resolved_spec = spec or LongCatRuntimeSpec()
    bridge = LongCatTempoKVBridge(
        codec=codec,
        mode=mode,
        trace_dir=None if trace_dir is None else Path(trace_dir),
        max_trace_spatial_positions=max_trace_spatial_positions,
        spec=resolved_spec,
        context_parallel_size=context_parallel_size,
        context_parallel_rank=context_parallel_rank,
        attention_sink_frames=attention_sink_frames,
        auto_context_parallel_stride=spec is None,
        profile_encode=profile_encode,
        profile_decode=profile_decode,
    )
    bridge.install(**modules)
    return bridge
