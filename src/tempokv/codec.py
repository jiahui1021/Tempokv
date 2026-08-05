from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal
import warnings

import torch

from .config import CodecConfig
from .parameters import CrossKVParameters, TemporalParameters
from .quantization import (
    dequantize_asymmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    quantize_symmetric,
)


PredictorKind = Literal["temporal", "cross_kv"]


@dataclass(slots=True)
class PackedTensor:
    """Packed anchor and predictive residuals for one cache tensor."""

    shape: tuple[int, int, int, int]
    predictor_kind: PredictorKind
    bits: int
    anchor_bits: int
    group_size: int
    stride: int
    residual_mode: str
    reconstruction_dtype: str
    unit_lengths: tuple[int, ...]
    anchor_quant: torch.Tensor
    anchor_scale: torch.Tensor
    residual_quant: torch.Tensor | None
    residual_scale: torch.Tensor | None
    residual_zero_point: torch.Tensor | None
    temporal: TemporalParameters | None = None
    cross_kv: CrossKVParameters | None = None

    def __post_init__(self) -> None:
        batch, heads, sequence, width = self.shape
        if min(batch, heads, sequence, width) <= 0:
            raise ValueError("packed shape must be positive")
        if width % self.group_size:
            raise ValueError("group_size must divide the head dimension")
        if sum(self.unit_lengths) != sequence or not self.unit_lengths:
            raise ValueError("unit_lengths must exactly cover the sequence")
        if self.predictor_kind == "temporal" and self.temporal is None:
            raise ValueError("temporal packed state is missing predictor parameters")
        if self.predictor_kind == "cross_kv" and self.cross_kv is None:
            raise ValueError("CrossKV packed state is missing predictor parameters")
        if self.temporal is not None and self.temporal.layered:
            raise ValueError("packed temporal parameters must contain one selected layer")
        if self.cross_kv is not None and self.cross_kv.layered:
            raise ValueError("packed CrossKV parameters must contain one selected layer")
        has_residual = len(self.unit_lengths) > 1
        if has_residual != (self.residual_quant is not None):
            raise ValueError("residual tensors do not match unit layout")
        if self.residual_mode == "asymmetric" and has_residual:
            if self.residual_zero_point is None:
                raise ValueError("asymmetric residuals require zero points")
        _dtype(self.reconstruction_dtype)


@dataclass(slots=True)
class PackedKV:
    key: PackedTensor
    value: PackedTensor

    def __post_init__(self) -> None:
        if self.key.predictor_kind != "temporal":
            raise ValueError("PackedKV key must use temporal prediction")
        if self.value.predictor_kind != "cross_kv":
            raise ValueError("PackedKV value must use CrossKV prediction")
        if self.key.shape != self.value.shape:
            raise ValueError("packed K and V must have the same shape")
        if self.key.reconstruction_dtype != self.value.reconstruction_dtype:
            raise ValueError("packed K and V must use the same reconstruction dtype")


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _validate_input(value: torch.Tensor, config: CodecConfig) -> None:
    if value.ndim != 4:
        raise ValueError(f"expected [B,H,S,D], got {tuple(value.shape)}")
    if value.shape[-1] % config.group_size:
        raise ValueError("group_size must divide the head dimension")


def predict_temporal(
    history: list[torch.Tensor], parameters: TemporalParameters
) -> torch.Tensor:
    """Predict one unit from newest-to-oldest reconstructed K units."""
    if not history:
        raise ValueError("temporal prediction requires reconstructed history")
    if parameters.layered:
        raise ValueError("select one layer before using temporal parameters")
    order = min(len(history), parameters.history_length)
    weights = parameters.history_weights[order - 1, :order].to(history[0].device)
    bias = parameters.history_bias[order - 1].to(history[0].device)
    prediction = torch.zeros_like(history[0], dtype=torch.float32)
    if parameters.per_head:
        if weights.shape[1] != history[0].shape[1]:
            raise ValueError("per-head temporal parameters do not match the cache")
        for index in range(order):
            prediction.add_(history[index].float() * weights[index][None, :, None, :])
        return prediction + bias[None, :, None, :]
    for index in range(order):
        prediction.add_(history[index].float() * weights[index][None, None, None, :])
    return prediction + bias[None, None, None, :]


def predict_cross_kv(source: torch.Tensor, parameters: CrossKVParameters) -> torch.Tensor:
    """Apply the per-head K-to-V affine projection."""
    if parameters.layered:
        raise ValueError("select one layer before using CrossKV parameters")
    weight = parameters.weight.to(source.device)
    bias = parameters.bias.to(source.device)
    if source.shape[1] != weight.shape[0] or source.shape[-1] != weight.shape[-1]:
        raise ValueError("CrossKV parameters do not match cache heads/head dimension")
    with _ieee_matmul(source.device):
        prediction = torch.matmul(source.float(), weight)
    return prediction + bias[None, :, None, :]


@contextmanager
def _ieee_matmul(device: torch.device):
    """Match the fixed IEEE policy used by the accelerated decoder."""
    if device.type != "cuda":
        yield
        return
    backend = torch.backends.cuda.matmul
    previous = backend.allow_tf32
    backend.allow_tf32 = False
    try:
        yield
    finally:
        backend.allow_tf32 = previous


def _quantize(
    value: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    mode: str,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if mode == "asymmetric":
        quant, scale, zero_point = quantize_asymmetric(
            value, bits, group_size, scale_dtype
        )
        rebuilt = dequantize_asymmetric(
            quant, scale, zero_point, bits, group_size, value.shape[-1], torch.float32
        )
        return quant, scale, zero_point, rebuilt
    quant, scale = quantize_symmetric(value, bits, group_size, scale_dtype)
    rebuilt = dequantize_symmetric(
        quant, scale, bits, group_size, value.shape[-1], torch.float32
    )
    return quant, scale, None, rebuilt


def _dequantize(
    quant: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    *,
    bits: int,
    group_size: int,
    width: int,
    mode: str,
) -> torch.Tensor:
    if mode == "asymmetric":
        if zero_point is None:
            raise ValueError("asymmetric residual is missing zero points")
        return dequantize_asymmetric(
            quant, scale, zero_point, bits, group_size, width, torch.float32
        )
    return dequantize_symmetric(
        quant, scale, bits, group_size, width, torch.float32
    )


def _encode(
    value: torch.Tensor,
    *,
    predictor_kind: PredictorKind,
    temporal: TemporalParameters | None,
    cross_kv: CrossKVParameters | None,
    source: torch.Tensor | None,
    bits: int,
    config: CodecConfig,
    reuse_input_for_reconstruction: bool = False,
) -> tuple[PackedTensor, torch.Tensor]:
    _validate_input(value, config)
    batch, heads, sequence, width = map(int, value.shape)
    stride = min(config.stride, sequence)
    scale_dtype = _dtype(config.scale_dtype)
    reconstruction_dtype = _dtype(config.reconstruction_dtype)
    if reuse_input_for_reconstruction and value.dtype != reconstruction_dtype:
        raise ValueError(
            "in-place reconstruction requires input dtype to match "
            f"{reconstruction_dtype}, got {value.dtype}"
        )
    if predictor_kind == "temporal":
        if temporal is None:
            raise ValueError("K encoding requires temporal parameters")
        if temporal.stride is not None and temporal.stride != config.stride:
            raise ValueError("temporal parameters were fitted for a different stride")
        temporal = temporal.to(value.device)
    else:
        if cross_kv is None or source is None:
            raise ValueError("V encoding requires CrossKV parameters and reconstructed K")
        if tuple(source.shape) != tuple(value.shape):
            raise ValueError("reconstructed K and V must have the same shape")
        if source.device != value.device:
            raise ValueError("reconstructed K and V must be on the same device")
        if source.dtype != reconstruction_dtype:
            raise ValueError(
                "reconstructed K dtype must match config.reconstruction_dtype "
                f"{reconstruction_dtype}, got {source.dtype}"
            )
        if cross_kv.stride is not None and cross_kv.stride != config.stride:
            raise ValueError("CrossKV parameters were fitted for a different stride")
        cross_kv = cross_kv.to(value.device)

    anchor = value[:, :, :stride]
    anchor_quant, anchor_scale = quantize_symmetric(
        anchor, config.anchor_bits, config.group_size, scale_dtype
    )
    anchor_rebuilt = dequantize_symmetric(
        anchor_quant,
        anchor_scale,
        config.anchor_bits,
        config.group_size,
        width,
        torch.float32,
    ).to(reconstruction_dtype)
    reconstruction = (
        value
        if reuse_input_for_reconstruction
        else torch.empty(value.shape, device=value.device, dtype=reconstruction_dtype)
    )
    reconstruction[:, :, :stride] = anchor_rebuilt
    history = [anchor_rebuilt]
    residual_sequence = sequence - stride
    packed_width = width if bits == 8 else width // (8 // bits)
    scale_width = width // config.group_size
    residual_quant = (
        torch.empty(
            batch,
            heads,
            residual_sequence,
            packed_width,
            device=value.device,
            dtype=torch.uint8,
        )
        if residual_sequence
        else None
    )
    residual_scale = (
        torch.empty(
            batch,
            heads,
            residual_sequence,
            scale_width,
            device=value.device,
            dtype=scale_dtype,
        )
        if residual_sequence
        else None
    )
    residual_zero_point = (
        torch.empty(
            batch,
            heads,
            residual_sequence,
            scale_width,
            device=value.device,
            dtype=torch.uint8,
        )
        if residual_sequence and config.residual_mode == "asymmetric"
        else None
    )
    residual_offset = 0
    unit_lengths = [stride]

    offset = stride
    while offset < sequence:
        unit_length = min(stride, sequence - offset)
        current = value[:, :, offset : offset + unit_length]
        if predictor_kind == "temporal":
            assert temporal is not None
            prediction = predict_temporal(
                [item[:, :, :unit_length] for item in history], temporal
            )
        else:
            assert cross_kv is not None and source is not None
            prediction = predict_cross_kv(
                source[:, :, offset : offset + unit_length], cross_kv
            )
        quant, scale, zero_point, residual_rebuilt = _quantize(
            current.float() - prediction,
            bits=bits,
            group_size=config.group_size,
            mode=config.residual_mode,
            scale_dtype=scale_dtype,
        )
        rebuilt = (prediction + residual_rebuilt).to(reconstruction_dtype)
        reconstruction[:, :, offset : offset + unit_length] = rebuilt
        if predictor_kind == "temporal":
            assert temporal is not None
            history = [rebuilt, *history[: temporal.history_length - 1]]
        residual_end = residual_offset + unit_length
        assert residual_quant is not None and residual_scale is not None
        residual_quant[:, :, residual_offset:residual_end].copy_(quant)
        residual_scale[:, :, residual_offset:residual_end].copy_(scale)
        if zero_point is not None:
            assert residual_zero_point is not None
            residual_zero_point[:, :, residual_offset:residual_end].copy_(zero_point)
        residual_offset = residual_end
        unit_lengths.append(unit_length)
        offset += unit_length

    packed = PackedTensor(
        shape=(batch, heads, sequence, width),
        predictor_kind=predictor_kind,
        bits=bits,
        anchor_bits=config.anchor_bits,
        group_size=config.group_size,
        stride=stride,
        residual_mode=config.residual_mode,
        reconstruction_dtype=config.reconstruction_dtype,
        unit_lengths=tuple(unit_lengths),
        anchor_quant=anchor_quant,
        anchor_scale=anchor_scale,
        residual_quant=residual_quant,
        residual_scale=residual_scale,
        residual_zero_point=residual_zero_point,
        temporal=temporal,
        cross_kv=cross_kv,
    )
    return packed, reconstruction


def encode_temporal(
    key: torch.Tensor,
    parameters: TemporalParameters,
    config: CodecConfig | None = None,
    *,
    reuse_input_for_reconstruction: bool = False,
) -> tuple[PackedTensor, torch.Tensor]:
    """Encode K and return both packed data and its closed-loop reconstruction."""
    resolved = config or CodecConfig()
    return _encode(
        key,
        predictor_kind="temporal",
        temporal=parameters,
        cross_kv=None,
        source=None,
        bits=resolved.k_bits,
        config=resolved,
        reuse_input_for_reconstruction=reuse_input_for_reconstruction,
    )


def encode_cross_kv(
    value: torch.Tensor,
    reconstructed_key: torch.Tensor,
    parameters: CrossKVParameters,
    config: CodecConfig | None = None,
    *,
    reuse_input_for_reconstruction: bool = False,
) -> tuple[PackedTensor, torch.Tensor]:
    """Encode V from reconstructed K and return its closed-loop reconstruction."""
    resolved = config or CodecConfig()
    return _encode(
        value,
        predictor_kind="cross_kv",
        temporal=None,
        cross_kv=parameters,
        source=reconstructed_key,
        bits=resolved.v_bits,
        config=resolved,
        reuse_input_for_reconstruction=reuse_input_for_reconstruction,
    )


def decode_tensor(
    packed: PackedTensor,
    *,
    source: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    output_tensor: torch.Tensor | None = None,
    use_triton: bool = True,
    strict_triton: bool = False,
) -> torch.Tensor:
    """Decode a packed cache, using Triton when its supported CUDA path is available."""
    stored_dtype = _dtype(packed.reconstruction_dtype)
    if output_dtype is None:
        output_dtype = stored_dtype
    elif output_dtype != stored_dtype:
        raise ValueError(
            "decode dtype must match the closed-loop reconstruction dtype "
            f"{stored_dtype}, got {output_dtype}"
        )
    if output_tensor is not None:
        batch, heads, sequence, width = packed.shape
        expected_shape = (batch, sequence, heads, width)
        if tuple(output_tensor.shape) != expected_shape:
            raise ValueError(
                "direct decode output must use [B,S,H,D] shape "
                f"{expected_shape}, got {tuple(output_tensor.shape)}"
            )
        if output_tensor.device != packed.anchor_quant.device:
            raise ValueError("direct decode output must be on the packed tensor device")
        if output_tensor.dtype != output_dtype:
            raise ValueError(
                f"direct decode output must use {output_dtype}, got {output_tensor.dtype}"
            )
        if not output_tensor.is_contiguous():
            raise ValueError("direct decode output must be contiguous BSHD")
    if packed.predictor_kind == "cross_kv":
        if source is None:
            raise ValueError("CrossKV decoding requires reconstructed K")
        if tuple(source.shape) != packed.shape:
            raise ValueError("reconstructed K does not match the packed V shape")
        if source.device != packed.anchor_quant.device:
            raise ValueError("reconstructed K and packed V must be on the same device")
        if source.dtype != stored_dtype:
            raise ValueError(
                "reconstructed K dtype must match the packed V reconstruction dtype "
                f"{stored_dtype}, got {source.dtype}"
            )
    if use_triton and packed.anchor_quant.is_cuda:
        try:
            from .triton_ops import decode as decode_triton

            accelerated = decode_triton(
                packed,
                source=source,
                output_dtype=output_dtype,
                output_tensor=output_tensor,
            )
            if accelerated is not None:
                return accelerated
            if strict_triton:
                raise RuntimeError("this packed layout is not supported by Triton")
        except Exception as error:
            if strict_triton:
                raise
            warnings.warn(
                f"Triton decode failed; using the PyTorch decoder: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    _, _, sequence, width = packed.shape
    anchor_length = packed.unit_lengths[0]
    output = (
        output_tensor.permute(0, 2, 1, 3)
        if output_tensor is not None
        else torch.empty(
            packed.shape,
            device=packed.anchor_quant.device,
            dtype=output_dtype,
        )
    )
    anchor = dequantize_symmetric(
        packed.anchor_quant,
        packed.anchor_scale,
        packed.anchor_bits,
        packed.group_size,
        width,
        output_dtype,
    )
    output[:, :, :anchor_length] = anchor
    if packed.residual_quant is None:
        return output
    assert packed.residual_scale is not None
    history = [anchor]
    offset = anchor_length
    residual_offset = 0
    for unit_length in packed.unit_lengths[1:]:
        if packed.predictor_kind == "temporal":
            assert packed.temporal is not None
            prediction = predict_temporal(
                [item[:, :, :unit_length] for item in history], packed.temporal
            )
        else:
            if source is None or packed.cross_kv is None:
                raise ValueError("CrossKV decoding requires reconstructed K")
            prediction = predict_cross_kv(
                source[:, :, offset : offset + unit_length], packed.cross_kv
            )
        end = residual_offset + unit_length
        zero_point = (
            packed.residual_zero_point[:, :, residual_offset:end]
            if packed.residual_zero_point is not None
            else None
        )
        residual = _dequantize(
            packed.residual_quant[:, :, residual_offset:end],
            packed.residual_scale[:, :, residual_offset:end],
            zero_point,
            bits=packed.bits,
            group_size=packed.group_size,
            width=width,
            mode=packed.residual_mode,
        )
        rebuilt = (prediction + residual).to(output_dtype)
        output[:, :, offset : offset + unit_length] = rebuilt
        if packed.predictor_kind == "temporal":
            assert packed.temporal is not None
            history = [rebuilt, *history[: packed.temporal.history_length - 1]]
        offset += unit_length
        residual_offset = end
    return output


class TempoKVCodec:
    """End-to-end K temporal + V CrossKV predictive residual codec."""

    def __init__(
        self,
        temporal: TemporalParameters,
        cross_kv: CrossKVParameters,
        config: CodecConfig | None = None,
    ) -> None:
        self.temporal = temporal
        self.cross_kv = cross_kv
        self.config = config or CodecConfig()
        expected_signature = self.config.format_signature()
        for name, parameters in (("K", temporal), ("V", cross_kv)):
            signature = parameters.codec_signature
            if signature is not None and signature != expected_signature:
                raise ValueError(
                    f"{name} parameters were fitted with a different codec config"
                )
        if (
            temporal.codec_signature is not None
            and cross_kv.codec_signature is not None
            and temporal.codec_signature != cross_kv.codec_signature
        ):
            raise ValueError("K and V parameters use different codec configs")
        if (
            temporal.parameter_set_id is not None
            and cross_kv.parameter_set_id is not None
            and temporal.parameter_set_id != cross_kv.parameter_set_id
        ):
            raise ValueError("K and V parameters were not fitted together")
        if (
            temporal.coordinate_system is not None
            and cross_kv.coordinate_system is not None
            and temporal.coordinate_system != cross_kv.coordinate_system
        ):
            raise ValueError("K and V parameters use different coordinate systems")
        self._temporal_by_device: dict[str, TemporalParameters] = {}
        self._cross_by_device: dict[str, CrossKVParameters] = {}

    def _temporal_for_layer(
        self, device: torch.device, layer: int
    ) -> TemporalParameters:
        device_key = str(device)
        parameters = self._temporal_by_device.get(device_key)
        if parameters is None:
            parameters = self.temporal.to(device)
            self._temporal_by_device[device_key] = parameters
        return parameters.for_layer(layer) if parameters.layered else parameters

    def _cross_for_layer(
        self, device: torch.device, layer: int
    ) -> CrossKVParameters:
        device_key = str(device)
        parameters = self._cross_by_device.get(device_key)
        if parameters is None:
            parameters = self.cross_kv.to(device)
            self._cross_by_device[device_key] = parameters
        return parameters.for_layer(layer) if parameters.layered else parameters

    def encode_key(
        self,
        key: torch.Tensor,
        *,
        layer: int = 0,
        reuse_input: bool = False,
    ) -> tuple[PackedTensor, torch.Tensor]:
        """Encode one K cache, optionally reusing its input for reconstruction."""
        temporal = self._temporal_for_layer(key.device, layer)
        return encode_temporal(
            key,
            temporal,
            self.config,
            reuse_input_for_reconstruction=reuse_input,
        )

    def encode_value(
        self,
        value: torch.Tensor,
        reconstructed_key: torch.Tensor,
        *,
        layer: int = 0,
        reuse_input: bool = False,
    ) -> PackedTensor:
        """Encode one V cache from reconstructed K without retaining V output."""
        cross = self._cross_for_layer(value.device, layer)
        packed, _ = encode_cross_kv(
            value,
            reconstructed_key,
            cross,
            self.config,
            reuse_input_for_reconstruction=reuse_input,
        )
        return packed

    def encode(
        self, key: torch.Tensor, value: torch.Tensor, *, layer: int = 0
    ) -> PackedKV:
        if tuple(key.shape) != tuple(value.shape):
            raise ValueError("K and V must have the same shape")
        if key.device != value.device:
            raise ValueError("K and V must be on the same device")
        key_state, key_rebuilt = self.encode_key(key, layer=layer)
        value_state = self.encode_value(value, key_rebuilt, layer=layer)
        return PackedKV(key_state, value_state)

    def decode(
        self, packed: PackedKV, *, output_dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = decode_tensor(
            packed.key,
            output_dtype=output_dtype,
            use_triton=self.config.use_triton,
            strict_triton=self.config.strict_triton,
        )
        value = decode_tensor(
            packed.value,
            source=key,
            output_dtype=output_dtype,
            use_triton=self.config.use_triton,
            strict_triton=self.config.strict_triton,
        )
        return key, value
