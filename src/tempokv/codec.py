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
    cross_matmul_precision: str = "tf32"
    compatibility_mode: str = "native"
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
        if any(length <= 0 or length > self.stride for length in self.unit_lengths):
            raise ValueError("unit lengths must be in [1, stride]")
        if any(length != self.stride for length in self.unit_lengths[:-1]):
            raise ValueError("only the final packed unit may be shorter than stride")
        if self.predictor_kind == "temporal" and self.temporal is None:
            raise ValueError("temporal packed state is missing predictor parameters")
        if self.predictor_kind == "cross_kv" and self.cross_kv is None:
            raise ValueError("CrossKV packed state is missing predictor parameters")
        if self.temporal is not None and self.temporal.layered:
            raise ValueError("packed temporal parameters must contain one selected layer")
        if self.cross_kv is not None and self.cross_kv.layered:
            raise ValueError("packed CrossKV parameters must contain one selected layer")
        if self.cross_matmul_precision not in {"ieee", "tf32"}:
            raise ValueError("cross_matmul_precision must be ieee or tf32")
        if self.compatibility_mode not in {"native", "legacy"}:
            raise ValueError("compatibility_mode must be native or legacy")
        has_residual = len(self.unit_lengths) > 1
        if has_residual != (self.residual_quant is not None):
            raise ValueError("residual tensors do not match unit layout")
        if self.residual_mode == "asymmetric" and has_residual:
            if self.residual_zero_point is None:
                raise ValueError("asymmetric residuals require zero points")
        _dtype(self.reconstruction_dtype)

    def storage_bytes(self) -> int:
        """On-device bytes actually stored for this packed tensor.

        Counts every quantized component (anchor quant/scale and, when present,
        residual quant/scale/zero-point).  Predictor parameters are shared and
        not replicated per span, so they are excluded.
        """
        total = self.anchor_quant.numel() * self.anchor_quant.element_size()
        total += self.anchor_scale.numel() * self.anchor_scale.element_size()
        for component in (
            self.residual_quant,
            self.residual_scale,
            self.residual_zero_point,
        ):
            if component is not None:
                total += component.numel() * component.element_size()
        return total


@dataclass(slots=True)
class PackedKV:
    key: PackedTensor
    value: PackedTensor

    def __post_init__(self) -> None:
        if self.key.predictor_kind != "temporal":
            raise ValueError("PackedKV key must use temporal prediction")
        if self.value.predictor_kind not in {"cross_kv", "temporal"}:
            raise ValueError("PackedKV value must use CrossKV or temporal prediction")
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
    history: list[torch.Tensor],
    parameters: TemporalParameters,
    *,
    legacy: bool = False,
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
    if legacy:
        if parameters.per_head:
            raise ValueError("legacy temporal parameters must share heads")
        for index in range(order):
            coefficient = weights[index].view(
                *([1] * (history[index].ndim - 1)), history[index].shape[-1]
            )
            prediction += history[index].float() * coefficient
        return prediction + bias.view(
            *([1] * (history[0].ndim - 1)), history[0].shape[-1]
        )
    if parameters.per_head:
        if weights.shape[1] != history[0].shape[1]:
            raise ValueError("per-head temporal parameters do not match the cache")
        for index in range(order):
            prediction.addcmul_(
                history[index], weights[index][None, :, None, :]
            )
        prediction.add_(bias[None, :, None, :])
        return prediction
    for index in range(order):
        prediction.addcmul_(
            history[index], weights[index][None, None, None, :]
        )
    prediction.add_(bias[None, None, None, :])
    return prediction


def cross_projection(
    source: torch.Tensor,
    parameters: CrossKVParameters,
    *,
    matmul_precision: str = "tf32",
) -> torch.Tensor:
    """Project a whole reconstructed-K span through Cross(K) in one GEMM.

    The projection depends only on K, so every unit's prediction is known
    before the V recurrence starts.  Computing them per unit issues one small
    matmul per unit inside the sequential loop and recomputes the projection of
    a unit that the history term needs again; one batched call over the span
    replaces both.  Encoder and decoder must call this the same way, otherwise
    the residual is quantized against a prediction the decoder cannot rebuild.
    """
    if parameters.layered:
        raise ValueError("select one layer before using CrossKV parameters")
    weight = parameters.weight.to(source.device)
    bias = parameters.bias.to(source.device)
    if source.shape[1] != weight.shape[0] or source.shape[-1] != weight.shape[-1]:
        raise ValueError("CrossKV parameters do not match cache heads/head dimension")
    batch, heads, tokens, dim = source.shape
    with _cuda_matmul_precision(source.device, matmul_precision):
        # Head-major and contiguous: the cache read is usually a strided
        # view, and matmul would otherwise materialize its own copy.
        head_major = source.permute(1, 0, 2, 3).to(
            dtype=torch.float32, memory_format=torch.contiguous_format
        )
        prediction = torch.matmul(head_major, weight[:, None, :, :])
        # A span-sized FP32 buffer is large; release the source copy before
        # the bias pass so two of them never coexist.
        del head_major
        prediction = prediction.permute(1, 0, 2, 3)
    prediction = prediction.contiguous()
    prediction.add_(bias.view(1, heads, 1, dim))
    return prediction


def predict_cross_kv(
    source: torch.Tensor,
    parameters: CrossKVParameters,
    history: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    *,
    # Defaults follow CodecConfig so an implicit call keeps native semantics.
    matmul_precision: str = "tf32",
    legacy: bool = False,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    """Predict V from current reconstructed K and newest-to-oldest V history.

    ``projection`` supplies ``Cross(K)+bias`` computed for this unit ahead of
    time by :func:`cross_projection`; the recurrence then costs no GEMM at all.
    """
    if parameters.layered:
        raise ValueError("select one layer before using CrossKV parameters")
    resolved_history = () if history is None else tuple(history)
    order = min(len(resolved_history), parameters.history_length)
    for previous in resolved_history[:order]:
        if tuple(previous.shape) != tuple(source.shape):
            raise ValueError("reconstructed K and V history must have the same shape")
        if previous.device != source.device:
            raise ValueError("reconstructed K and V history must be on the same device")
    weight = parameters.weight.to(source.device)
    bias = parameters.bias.to(source.device)
    if source.shape[1] != weight.shape[0] or source.shape[-1] != weight.shape[-1]:
        raise ValueError("CrossKV parameters do not match cache heads/head dimension")
    with _cuda_matmul_precision(source.device, matmul_precision):
        if projection is not None:
            if tuple(projection.shape) != tuple(source.shape):
                raise ValueError("projection must cover exactly this unit")
            prediction = projection if legacy else projection.clone()
        elif legacy:
            prediction = torch.matmul(
                source.float().permute(1, 0, 2, 3),
                weight[:, None, :, :],
            ).permute(1, 0, 2, 3)
            prediction = prediction + bias.view(
                1, source.shape[1], 1, source.shape[-1]
            )
        else:
            prediction = torch.matmul(source.float(), weight)
            prediction.add_(bias[None, :, None, :])
        for lag, previous in enumerate(resolved_history[:order]):
                history_weight = parameters.history_weights[lag].to(source.device)
                if parameters.history_diagonal:
                    if legacy:
                        prediction = prediction + history_weight.view(
                            1, source.shape[1], 1, source.shape[-1]
                        ) * previous.float()
                    else:
                        prediction.addcmul_(
                            previous, history_weight[None, :, None, :]
                        )
                else:
                    prediction.add_(torch.matmul(previous.float(), history_weight))
    return prediction


@contextmanager
def _cuda_matmul_precision(device: torch.device, precision: str):
    """Apply one explicit FP32 GEMM policy around a CrossKV projection."""
    if precision not in {"ieee", "tf32"}:
        raise ValueError("matmul precision must be ieee or tf32")
    if device.type != "cuda":
        yield
        return
    backend = torch.backends.cuda.matmul
    previous = backend.allow_tf32
    backend.allow_tf32 = precision == "tf32"
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
    history_length: int | None = None,
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
            raise ValueError("temporal encoding requires temporal parameters")
        temporal = temporal.with_history_length(
            config.k_history_length if history_length is None else history_length
        )
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
        cross_kv = cross_kv.with_history_length(config.v_history_length)
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
    # The legacy predictor keeps every encoder history unit as the standalone
    # result returned by QDQ.  Native TempoKV keeps cheaper views into the
    # reconstruction buffer.  The distinction affects CUDA dispatch and exact
    # accumulation.
    history = (
        [anchor_rebuilt]
        if config.fit_mode == "legacy"
        else [reconstruction[:, :, :stride]]
    )
    del anchor_rebuilt
    # Cross(K) depends only on K, so the whole span is projected once here
    # instead of once per unit inside the sequential residual loop.
    projection = (
        cross_projection(
            source,
            cross_kv,
            matmul_precision=config.cross_matmul_precision,
        )
        if predictor_kind == "cross_kv" and cross_kv is not None and source is not None
        else None
    )
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
            if temporal.history_length == 0:
                # Zero-order K is the direct-quantization ablation. Keep the
                # temporal packed-state contract while spending all residual
                # bits on the current unit.
                prediction = torch.zeros_like(current, dtype=torch.float32)
            else:
                prediction = predict_temporal(
                    [item[:, :, :unit_length] for item in history],
                    temporal,
                    legacy=config.fit_mode == "legacy",
                )
        else:
            assert cross_kv is not None and source is not None
            prediction = predict_cross_kv(
                source[:, :, offset : offset + unit_length],
                cross_kv,
                [item[:, :, :unit_length] for item in history],
                matmul_precision=config.cross_matmul_precision,
                legacy=config.fit_mode == "legacy",
                projection=projection[:, :, offset : offset + unit_length],
            )
        if config.fit_mode == "legacy":
            residual_input = current.float() - prediction
        else:
            residual_input = torch.empty_like(current, dtype=torch.float32)
            torch.sub(current, prediction, out=residual_input)
        quant, scale, zero_point, residual_rebuilt = _quantize(
            residual_input,
            bits=bits,
            group_size=config.group_size,
            mode=config.residual_mode,
            scale_dtype=scale_dtype,
        )
        del residual_input
        rebuilt = (
            (prediction + residual_rebuilt).to(reconstruction_dtype)
            if config.fit_mode == "legacy"
            else prediction.add_(residual_rebuilt).to(reconstruction_dtype)
        )
        reconstruction[:, :, offset : offset + unit_length] = rebuilt
        rebuilt_history = (
            rebuilt
            if config.fit_mode == "legacy"
            else reconstruction[:, :, offset : offset + unit_length]
        )
        if predictor_kind == "temporal":
            if temporal is not None and temporal.history_length:
                history = [
                    rebuilt_history,
                    *history[: temporal.history_length - 1],
                ]
        else:
            assert cross_kv is not None
            if cross_kv.history_length:
                history = [
                    rebuilt_history,
                    *history[: cross_kv.history_length - 1],
                ]
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
        del current, prediction, quant, scale, zero_point, residual_rebuilt, rebuilt

    packed = PackedTensor(
        shape=(batch, heads, sequence, width),
        predictor_kind=predictor_kind,
        bits=bits,
        anchor_bits=config.anchor_bits,
        group_size=config.group_size,
        stride=stride,
        residual_mode=config.residual_mode,
        reconstruction_dtype=config.reconstruction_dtype,
        cross_matmul_precision=config.cross_matmul_precision,
        compatibility_mode=config.fit_mode,
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
    bits: int | None = None,
    history_length: int | None = None,
) -> tuple[PackedTensor, torch.Tensor]:
    """Encode a temporally predicted tensor with its closed-loop reconstruction.

    ``bits`` and ``history_length`` default to the K settings.  A temporal V
    predictor passes the V settings instead; the predictor itself is identical.
    """
    resolved = config or CodecConfig()
    return _encode(
        key,
        predictor_kind="temporal",
        temporal=parameters,
        cross_kv=None,
        source=None,
        bits=resolved.k_bits if bits is None else bits,
        config=resolved,
        reuse_input_for_reconstruction=reuse_input_for_reconstruction,
        history_length=history_length,
    )


def encode_cross_kv(
    value: torch.Tensor,
    reconstructed_key: torch.Tensor,
    parameters: CrossKVParameters,
    config: CodecConfig | None = None,
    *,
    reuse_input_for_reconstruction: bool = False,
) -> tuple[PackedTensor, torch.Tensor]:
    """Encode V from reconstructed K/V history and return its reconstruction."""
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


def _direct_output_shape(
    packed_shape: tuple[int, int, int, int],
    output_layout: str,
) -> tuple[int, int, int, int]:
    batch, heads, sequence, width = packed_shape
    if output_layout == "BHSD":
        return batch, heads, sequence, width
    if output_layout == "BSHD":
        return batch, sequence, heads, width
    raise ValueError(
        f"direct decode output_layout must be 'BHSD' or 'BSHD', got {output_layout!r}"
    )


def _has_non_overlapping_positive_strides(tensor: torch.Tensor) -> bool:
    """Return whether a strided output is safe for independent kernel stores.

    A sequence slice of a larger BSHD/BHSD allocation is commonly not dense:
    its batch or head stride still includes the omitted suffix.  Sorting the
    dimensions by stride and requiring every dimension to start beyond the
    complete span of the faster dimensions accepts those slices while
    conservatively rejecting overlapping ``as_strided`` views.
    """
    dimensions = sorted(
        (stride, size)
        for size, stride in zip(tensor.shape, tensor.stride())
        if size > 1
    )
    covered_span = 1
    for stride, size in dimensions:
        if stride <= 0 or stride < covered_span:
            return False
        covered_span += (size - 1) * stride
    return all(stride > 0 for stride in tensor.stride())


def decode_tensor(
    packed: PackedTensor,
    *,
    source: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    output_tensor: torch.Tensor | None = None,
    output_layout: str = "BSHD",
    use_triton: bool = True,
    strict_triton: bool = False,
) -> torch.Tensor:
    """Decode a packed cache, using Triton when its CUDA path is available.

    ``output_tensor`` may be a non-overlapping positive-stride BHSD or BSHD
    view.  ``output_layout`` describes that physical target and defaults to the
    historical BSHD direct-output API.  The returned tensor remains BHSD so a
    reconstructed K can always be passed directly as the CrossKV ``source``.
    """
    stored_dtype = _dtype(packed.reconstruction_dtype)
    if output_dtype is None:
        output_dtype = stored_dtype
    elif output_dtype != stored_dtype:
        raise ValueError(
            "decode dtype must match the closed-loop reconstruction dtype "
            f"{stored_dtype}, got {output_dtype}"
        )
    expected_shape = _direct_output_shape(packed.shape, output_layout)
    if output_tensor is not None:
        if tuple(output_tensor.shape) != expected_shape:
            layout_shape = "[B,S,H,D]" if output_layout == "BSHD" else "[B,H,S,D]"
            raise ValueError(
                f"direct decode output must use {layout_shape} shape "
                f"{expected_shape}, got {tuple(output_tensor.shape)}"
            )
        if output_tensor.device != packed.anchor_quant.device:
            raise ValueError("direct decode output must be on the packed tensor device")
        if output_tensor.dtype != output_dtype:
            raise ValueError(
                f"direct decode output must use {output_dtype}, got {output_tensor.dtype}"
            )
        if not _has_non_overlapping_positive_strides(output_tensor):
            raise ValueError(
                "direct decode output must have non-overlapping positive strides"
            )
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
    # Legacy mode picks its decode route deliberately, so a silent PyTorch
    # fallback would hide both a correctness and a throughput regression.
    effective_strict_triton = (
        strict_triton or packed.compatibility_mode == "legacy"
    )
    # Every Triton route projects Cross(K) inside its own kernel, so the
    # batched projection is built only if the Torch decoder is reached.
    needs_projection = (
        packed.predictor_kind == "cross_kv"
        and packed.cross_kv is not None
        and source is not None
        and len(packed.unit_lengths) > 1
    )
    projection = None
    if use_triton and packed.anchor_quant.is_cuda:
        try:
            from .triton_ops import decode as decode_triton

            accelerated = decode_triton(
                packed,
                source=source,
                output_dtype=output_dtype,
                output_tensor=output_tensor,
                output_layout=output_layout,
            )
            if accelerated is not None:
                return accelerated
            if effective_strict_triton:
                raise RuntimeError("this packed layout is not supported by Triton")
        except Exception as error:
            if effective_strict_triton:
                raise
            warnings.warn(
                f"Triton decode failed; using the PyTorch decoder: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    _, _, sequence, width = packed.shape
    anchor_length = packed.unit_lengths[0]
    if output_tensor is None:
        output = torch.empty(
            packed.shape,
            device=packed.anchor_quant.device,
            dtype=output_dtype,
        )
    elif output_layout == "BSHD":
        output = output_tensor.permute(0, 2, 1, 3)
    else:
        output = output_tensor
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
    history = (
        [anchor]
        if packed.compatibility_mode == "legacy"
        else [output[:, :, :anchor_length]]
    )
    del anchor
    if needs_projection and projection is None:
        projection = cross_projection(
            source,
            packed.cross_kv,
            matmul_precision=packed.cross_matmul_precision,
        )
    offset = anchor_length
    residual_offset = 0
    for unit_length in packed.unit_lengths[1:]:
        if packed.predictor_kind == "temporal":
            assert packed.temporal is not None
            if packed.temporal.history_length == 0:
                prediction = torch.zeros(
                    output.shape[0],
                    output.shape[1],
                    unit_length,
                    width,
                    device=output.device,
                    dtype=torch.float32,
                )
            else:
                prediction = predict_temporal(
                    [item[:, :, :unit_length] for item in history],
                    packed.temporal,
                    legacy=packed.compatibility_mode == "legacy",
                )
        else:
            if source is None or packed.cross_kv is None:
                raise ValueError("CrossKV decoding requires reconstructed K")
            prediction = predict_cross_kv(
                source[:, :, offset : offset + unit_length],
                packed.cross_kv,
                [item[:, :, :unit_length] for item in history],
                matmul_precision=packed.cross_matmul_precision,
                legacy=packed.compatibility_mode == "legacy",
                projection=projection[:, :, offset : offset + unit_length],
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
        rebuilt = (
            (prediction + residual).to(output_dtype)
            if packed.compatibility_mode == "legacy"
            else prediction.add_(residual).to(output_dtype)
        )
        output[:, :, offset : offset + unit_length] = rebuilt
        rebuilt_history = (
            rebuilt
            if packed.compatibility_mode == "legacy"
            else output[:, :, offset : offset + unit_length]
        )
        if packed.predictor_kind == "temporal":
            if packed.temporal is not None and packed.temporal.history_length:
                history = [
                    rebuilt_history,
                    *history[: packed.temporal.history_length - 1],
                ]
        else:
            assert packed.cross_kv is not None
            if packed.cross_kv.history_length:
                history = [
                    rebuilt_history,
                    *history[: packed.cross_kv.history_length - 1],
                ]
        offset += unit_length
        residual_offset = end
        del prediction, zero_point, residual, rebuilt
    return output


class TempoKVCodec:
    """End-to-end temporal K and joint K/V-history predictive residual codec."""

    def __init__(
        self,
        temporal: TemporalParameters,
        cross_kv: CrossKVParameters | TemporalParameters,
        config: CodecConfig | None = None,
    ) -> None:
        self.config = config or CodecConfig()
        expects_temporal_v = self.config.v_predictor_mode == "temporal"
        if expects_temporal_v != isinstance(cross_kv, TemporalParameters):
            expected = "TemporalParameters" if expects_temporal_v else "CrossKVParameters"
            raise ValueError(
                f"v_predictor_mode={self.config.v_predictor_mode} requires V "
                f"parameters of type {expected}, got {type(cross_kv).__name__}"
            )
        temporal = temporal.with_history_length(self.config.k_history_length)
        cross_kv = cross_kv.with_history_length(self.config.v_history_length)
        self.temporal = temporal
        self.v_parameters = cross_kv
        expected_signature = self.config.format_signature()
        for name, parameters in (("K", self.temporal), ("V", self.v_parameters)):
            signature = parameters.codec_signature
            if signature is not None and signature != expected_signature:
                raise ValueError(
                    f"{name} parameters were fitted with a different codec config"
                )
        if (
            self.temporal.codec_signature is not None
            and self.v_parameters.codec_signature is not None
            and self.temporal.codec_signature != self.v_parameters.codec_signature
        ):
            raise ValueError("K and V parameters use different codec configs")
        if (
            self.temporal.parameter_set_id is not None
            and self.v_parameters.parameter_set_id is not None
            and self.temporal.parameter_set_id != self.v_parameters.parameter_set_id
        ):
            raise ValueError("K and V parameters were not fitted together")
        if (
            self.temporal.coordinate_system is not None
            and self.v_parameters.coordinate_system is not None
            and self.temporal.coordinate_system != self.v_parameters.coordinate_system
        ):
            raise ValueError("K and V parameters use different coordinate systems")
        self._temporal_by_device: dict[str, TemporalParameters] = {}
        self._cross_by_device: dict[str, CrossKVParameters] = {}
        # Per (device, layer) caches so every packed span shares the same
        # single-layer parameter object instead of building a fresh copy.
        self._temporal_layer_cache: dict[tuple[str, int], TemporalParameters] = {}
        self._cross_layer_cache: dict[tuple[str, int], CrossKVParameters] = {}

    def _temporal_for_layer(
        self, device: torch.device, layer: int
    ) -> TemporalParameters:
        device_key = str(device)
        cache_key = (device_key, layer)
        cached = self._temporal_layer_cache.get(cache_key)
        if cached is not None:
            return cached
        parameters = self._temporal_by_device.get(device_key)
        if parameters is None:
            parameters = self.temporal.to(device)
            self._temporal_by_device[device_key] = parameters
        selected = parameters.for_layer(layer) if parameters.layered else parameters
        self._temporal_layer_cache[cache_key] = selected
        return selected

    def _cross_for_layer(
        self, device: torch.device, layer: int
    ) -> CrossKVParameters:
        device_key = str(device)
        cache_key = (device_key, layer)
        cached = self._cross_layer_cache.get(cache_key)
        if cached is not None:
            return cached
        parameters = self._cross_by_device.get(device_key)
        if parameters is None:
            parameters = self.v_parameters.to(device)
            self._cross_by_device[device_key] = parameters
        selected = parameters.for_layer(layer) if parameters.layered else parameters
        self._cross_layer_cache[cache_key] = selected
        return selected

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

    @property
    def cross_kv(self) -> CrossKVParameters:
        """The CrossKV parameters, when this codec predicts V from K."""
        parameters = self.v_parameters
        if not isinstance(parameters, CrossKVParameters):
            raise AttributeError(
                "this codec predicts V with the temporal predictor; read "
                "codec.v_parameters instead of codec.cross_kv"
            )
        return parameters

    @property
    def predicts_v_from_key(self) -> bool:
        """Whether encoding V requires the reconstructed K of the same span."""
        return self.config.v_predictor_mode == "cross_kv"

    def encode_value(
        self,
        value: torch.Tensor,
        reconstructed_key: torch.Tensor | None = None,
        *,
        layer: int = 0,
        reuse_input: bool = False,
    ) -> PackedTensor:
        """Encode one V cache without retaining the decoded V output.

        ``reconstructed_key`` is required only when V is predicted from K; the
        temporal V predictor reads its own reconstructed history instead.
        """
        parameters = self._cross_for_layer(value.device, layer)
        if not self.predicts_v_from_key:
            assert isinstance(parameters, TemporalParameters)
            packed, _ = encode_temporal(
                value,
                parameters,
                self.config,
                reuse_input_for_reconstruction=reuse_input,
                bits=self.config.v_bits,
                history_length=self.config.v_history_length,
            )
            return packed
        if reconstructed_key is None:
            raise ValueError("CrossKV V encoding requires the reconstructed K")
        assert isinstance(parameters, CrossKVParameters)
        packed, _ = encode_cross_kv(
            value,
            reconstructed_key,
            parameters,
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
            source=key if self.predicts_v_from_key else None,
            output_dtype=output_dtype,
            use_triton=self.config.use_triton,
            strict_triton=self.config.strict_triton,
        )
        return key, value
