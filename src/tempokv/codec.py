from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal
import warnings

import torch

from .config import (
    RAW_ANCHOR_BITS,
    CodecConfig,
    is_cross_kv_v_predictor,
    is_temporal_v_predictor,
)
from .parameters import CrossKVParameters, TemporalParameters
from .quantization import (
    code_dtype,
    dequantize_asymmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    quantize_symmetric,
    roundtrip,
)


PredictorKind = Literal["temporal", "cross_kv", "innovation"]


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
    cross_matmul_precision: str = "ieee"
    compatibility_mode: str = "legacy"
    temporal: TemporalParameters | None = None
    cross_kv: CrossKVParameters | None = None
    # Mean-split K spans store the per-channel mean [B, H, D] subtracted
    # before prediction; decode adds it back after reconstruction.
    channel_mean: torch.Tensor | None = None

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
        if self.predictor_kind == "innovation" and (
            self.temporal is None or self.cross_kv is None
        ):
            raise ValueError(
                "innovation packed state needs K temporal and V projection parameters"
            )
        if self.cross_kv is not None and self.predictor_kind in {
            "cross_kv",
            "innovation",
        }:
            expected_source = (
                "innovation" if self.predictor_kind == "innovation" else "cross_kv"
            )
            if self.cross_kv.projection_source != expected_source:
                raise ValueError(
                    f"{self.predictor_kind} packed state requires projection_source="
                    f"{expected_source}"
                )
        if self.temporal is not None and self.temporal.layered:
            raise ValueError("packed temporal parameters must contain one selected layer")
        if self.cross_kv is not None and self.cross_kv.layered:
            raise ValueError("packed CrossKV parameters must contain one selected layer")
        if self.cross_matmul_precision not in {"ieee", "tf32"}:
            raise ValueError("cross_matmul_precision must be ieee or tf32")
        if self.compatibility_mode not in {"native", "legacy"}:
            raise ValueError("compatibility_mode must be native or legacy")
        if self.anchor_bits == RAW_ANCHOR_BITS:
            # The width alone cannot tell codes from values, and a packed state
            # built under the older INT16 reading has the same anchor_bits.
            # Both decoders read this tensor as values, so a mismatch here
            # would reinterpret code bytes as floats and decode the anchor span
            # to garbage without raising.
            if self.anchor_quant.dtype != _dtype(self.reconstruction_dtype):
                raise ValueError(
                    "an unquantized anchor must be stored in "
                    f"{self.reconstruction_dtype}, got {self.anchor_quant.dtype}"
                )
            if int(self.anchor_quant.shape[-1]) != width:
                raise ValueError(
                    "an unquantized anchor stores one value per channel, "
                    f"expected width {width}, got {self.anchor_quant.shape[-1]}"
                )
            if self.anchor_scale.numel():
                raise ValueError("an unquantized anchor has no group scale")
        has_residual = len(self.unit_lengths) > 1
        if has_residual != (self.residual_quant is not None):
            raise ValueError("residual tensors do not match unit layout")
        if self.residual_mode == "asymmetric" and has_residual:
            if self.residual_zero_point is None:
                raise ValueError("asymmetric residuals require zero points")
        _dtype(self.reconstruction_dtype)

    def step_sizes(self) -> torch.Tensor:
        """Per-token per-group quantizer step size, shaped ``[B, H, S, D/g]``.

        This is the ``Delta`` a Jensen-bias correction needs (arXiv 2605.26266):
        the reconstruction error of a cached key is bounded by ``Delta/2``,
        because the residual loop is closed and every unit after the anchor
        stores only the quantization error of its own residual.

        The step size is the stored scale itself at every width but one, so
        reading it costs nothing and adds nothing to the cache.  The exception
        is 1-bit symmetric quantization, which has no zero code and places its
        two levels at -scale and +scale: the distance between adjacent levels
        is then 2*scale, not scale.  Reporting the scale there would understate
        the step by 2x and the bias, which is quadratic in it, by 4x.

        An unquantized anchor reports zero, which is correct rather than merely
        convenient: a key stored at bf16 carries no quantization noise and must
        receive no correction.
        """
        batch, heads, sequence, width = self.shape
        groups = width // self.group_size
        if self.raw_anchor:
            anchor = self.anchor_quant.new_zeros(
                (batch, heads, self.stride, groups), dtype=torch.float32
            )
        else:
            # The anchor is always quantized symmetrically.
            anchor = self.anchor_scale.float() * (2.0 if self.anchor_bits == 1 else 1.0)
        if self.residual_scale is None:
            return anchor
        residual_binary = self.residual_mode == "symmetric" and self.bits == 1
        residual_length = self.shape[2] - self.unit_lengths[0]
        residual = self.residual_scale[:, :, :residual_length].float() * (
            2.0 if residual_binary else 1.0
        )
        return torch.cat([anchor, residual], dim=2)

    @property
    def raw_anchor(self) -> bool:
        """Whether ``anchor_quant`` holds values rather than packed codes."""
        return self.anchor_bits == RAW_ANCHOR_BITS

    def storage_bytes(self) -> int:
        """On-device bytes actually stored for this packed tensor.

        Counts every quantized component (anchor quant/scale and, when present,
        residual quant/scale/zero-point).  Predictor parameters are shared and
        not replicated per span, so they are excluded.
        """
        total = self.anchor_quant.numel() * self.anchor_quant.element_size()
        total += self.anchor_scale.numel() * self.anchor_scale.element_size()
        if self.channel_mean is not None:
            total += self.channel_mean.numel() * self.channel_mean.element_size()
        for component in (
            self.residual_quant,
            self.residual_scale,
            self.residual_zero_point,
        ):
            if component is not None:
                total += component.numel() * component.element_size()
        return total

    def logical_storage_bytes(self) -> int:
        """Bytes initialized for the logical prefix of an open packed span.

        ``storage_bytes`` is the allocator-facing figure and includes unused
        geometric-growth capacity.  This method exists only for codec-format
        analysis; resident-memory reports must continue to use allocated bytes.
        """
        total = self.anchor_quant.numel() * self.anchor_quant.element_size()
        total += self.anchor_scale.numel() * self.anchor_scale.element_size()
        if self.channel_mean is not None:
            total += self.channel_mean.numel() * self.channel_mean.element_size()
        residual_length = self.shape[2] - self.unit_lengths[0]
        for component in (
            self.residual_quant,
            self.residual_scale,
            self.residual_zero_point,
        ):
            if component is not None:
                logical = component[:, :, :residual_length]
                total += logical.numel() * logical.element_size()
        return total


@dataclass(slots=True)
class PackedKV:
    key: PackedTensor
    value: PackedTensor

    # Exactly one side carries the cross dependency, and which side that is
    # depends on ``predictor_direction``: forward runs K temporal / V cross,
    # mirrored runs V temporal / K cross.  Both orientations are accepted here
    # and the pair is rejected only when neither side predicts temporally,
    # which would leave the cross source undefined.
    _SOURCE_KINDS = {"cross_kv", "innovation"}

    def __post_init__(self) -> None:
        kinds = (self.key.predictor_kind, self.value.predictor_kind)
        if self.key.predictor_kind in self._SOURCE_KINDS:
            if self.value.predictor_kind != "temporal":
                raise ValueError(
                    "a CrossKV-predicted K needs a temporally predicted V to "
                    f"project from, got {kinds}"
                )
        elif self.value.predictor_kind in self._SOURCE_KINDS:
            if self.key.predictor_kind != "temporal":
                raise ValueError(
                    "a CrossKV-predicted V needs a temporally predicted K to "
                    f"project from, got {kinds}"
                )
        elif not set(kinds) <= {"temporal", "direct"}:
            raise ValueError(f"unsupported PackedKV predictor pair {kinds}")
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
            # Legacy layer_head K: the same out-of-place FP32 accumulation as
            # the shared predictor, with one coefficient row per head.
            if weights.shape[1] != history[0].shape[1]:
                raise ValueError("per-head temporal parameters do not match the cache")
            head_shape = (1, weights.shape[1], 1, weights.shape[-1])
            for index in range(order):
                prediction += history[index].float() * weights[index].view(head_shape)
            return prediction + bias.view(head_shape)
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


def reconstructed_key_innovation(
    reconstructed_key: torch.Tensor,
    parameters: TemporalParameters,
    unit_lengths: tuple[int, ...],
    *,
    legacy: bool = True,
) -> torch.Tensor:
    """Return the decoder-visible K innovation for one reconstructed span.

    The anchor has no preceding temporal prediction and is defined as zero.
    Every later unit subtracts the same prediction the K decoder can reproduce,
    then rounds the innovation to reconstructed K's dtype before V projects it.
    """
    if reconstructed_key.ndim != 4:
        raise ValueError("reconstructed K must have shape [B,H,S,D]")
    if not unit_lengths or sum(unit_lengths) != reconstructed_key.shape[2]:
        raise ValueError("unit_lengths must exactly cover reconstructed K")
    if any(length <= 0 for length in unit_lengths):
        raise ValueError("unit_lengths must be positive")
    if parameters.layered:
        raise ValueError("select one K temporal layer before building innovation")
    if parameters.history_length < 1:
        raise ValueError("K innovation requires a temporal K predictor")
    if parameters.stride is not None and any(
        length != parameters.stride for length in unit_lengths[:-1]
    ):
        raise ValueError("K temporal parameters do not match innovation unit lengths")

    innovation = torch.zeros_like(reconstructed_key)
    anchor_length = unit_lengths[0]
    history = [reconstructed_key[:, :, :anchor_length]]
    offset = anchor_length
    for unit_length in unit_lengths[1:]:
        prediction = predict_temporal(
            [item[:, :, :unit_length] for item in history],
            parameters,
            legacy=legacy,
        )
        current = reconstructed_key[:, :, offset : offset + unit_length]
        innovation[:, :, offset : offset + unit_length].copy_(
            (current.float() - prediction).to(reconstructed_key.dtype)
        )
        history = [
            current,
            *history[: parameters.history_length - 1],
        ]
        offset += unit_length
    return innovation


def cross_projection(
    source: torch.Tensor,
    parameters: CrossKVParameters,
    *,
    matmul_precision: str = "ieee",
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
    # Defaults follow the repository-wide legacy codec semantics.
    matmul_precision: str = "ieee",
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


def anchor_roundtrip(value: torch.Tensor, config: CodecConfig) -> torch.Tensor:
    """Replay what the encoder stores for one anchor unit.

    Fitting, attention refinement and the analysis tools all rebuild the anchor
    outside the codec, and each of them has to round exactly where
    :func:`_encode` rounds.  The unquantized anchor keeps its value, so handing
    ``config.anchor_bits`` to a quantizer there would optimize and report
    against an anchor the codec never produces.

    Every out-of-codec anchor replay goes through this function instead of
    calling a quantizer with ``config.anchor_bits``; the quantized branch
    returns float32 exactly as ``roundtrip`` does, so existing callers keep
    their own reconstruction-dtype rounding.
    """
    if config.raw_anchor:
        return value.to(_dtype(config.reconstruction_dtype))
    return roundtrip(
        value,
        bits=config.anchor_bits,
        group_size=config.group_size,
        scale_dtype=_dtype(config.scale_dtype),
        symmetric=True,
    )


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
    channel_mean: bool = False,
) -> tuple[PackedTensor, torch.Tensor]:
    _validate_input(value, config)
    batch, heads, sequence, width = map(int, value.shape)
    stride = min(config.stride, sequence)
    scale_dtype = _dtype(config.scale_dtype)
    reconstruction_dtype = _dtype(config.reconstruction_dtype)
    span_mean: torch.Tensor | None = None
    if channel_mean:
        if predictor_kind != "temporal":
            raise ValueError("channel-mean encoding is only supported for K")
        # The mean is stored at reconstruction dtype so encoder and decoder
        # add back bit-identical values.
        span_mean = (
            value.float().mean(dim=2, keepdim=True).to(reconstruction_dtype)
        )
        value = (value.float() - span_mean).to(reconstruction_dtype)
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
            raise ValueError("V encoding requires projection parameters and reconstructed K")
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
        expected_source = "innovation" if predictor_kind == "innovation" else "cross_kv"
        if cross_kv.projection_source != expected_source:
            raise ValueError(
                f"{predictor_kind} encoding requires projection_source="
                f"{expected_source}, got {cross_kv.projection_source}"
            )
        if predictor_kind == "innovation":
            if temporal is None:
                raise ValueError("innovation encoding requires K temporal parameters")
            temporal = temporal.with_history_length(config.k_history_length)
            if temporal.stride is not None and temporal.stride != config.stride:
                raise ValueError("K temporal parameters were fitted for a different stride")
            temporal = temporal.to(value.device)
            source = reconstructed_key_innovation(
                source,
                temporal,
                tuple(
                    min(stride, sequence - offset)
                    for offset in range(0, sequence, stride)
                ),
                legacy=config.fit_mode == "legacy",
            )

    anchor = value[:, :, :stride]
    if config.raw_anchor:
        # No quantizer: the anchor is the stored value.  ``copy=True`` matters
        # because ``to`` returns the input itself when the dtype already
        # matches, and an in-place reconstruction would then write through the
        # tensor the packed state is supposed to own.  It also keeps this to a
        # single copy when the dtype does differ.
        anchor_quant = anchor.to(reconstruction_dtype, copy=True)
        # No codes means no group scale.  Keeping the field a real tensor
        # rather than None spares every reader an optional, and an empty one
        # contributes nothing to storage_bytes().
        anchor_scale = anchor_quant.new_empty((0,), dtype=scale_dtype)
        anchor_rebuilt = anchor_quant
    else:
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
        if predictor_kind in {"cross_kv", "innovation"}
        and cross_kv is not None
        and source is not None
        else None
    )
    residual_sequence = sequence - stride
    packed_width = (width * bits + 7) // 8
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
            # A zero point is a code, so it needs the same room a code does.
            # Residual widths stop at 8 today, which keeps this uint8; asking
            # the packer means a wider residual could never truncate here.
            dtype=code_dtype(bits),
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

    if span_mean is not None:
        reconstruction.add_(span_mean)
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
        channel_mean=(span_mean.squeeze(2) if span_mean is not None else None),
    )
    return packed, reconstruction


class IncrementalTensorEncoder:
    """Append finalized units to one predictor-continuous packed span.

    Starts from a batch-encoded prefix and applies the same closed-loop
    predictor/quantizer arithmetic to newly finalized units, so the packed
    bytes match a batch encode of the whole span.  ``shape`` and
    ``unit_lengths`` describe the logical prefix; the residual buffers behind
    them may be longer.  Only reconstructed predictor history is kept in BF16,
    bounded by the history order and released when the span closes.
    """

    def __init__(
        self,
        packed: PackedTensor,
        reconstruction: torch.Tensor,
        config: CodecConfig,
        *,
        capacity_units: int,
    ) -> None:
        if capacity_units < len(packed.unit_lengths):
            raise ValueError("incremental capacity is smaller than the prefix")
        # Span-mean K only knows its mean once the span is complete, and K
        # innovation needs a continuation-aware source; neither can be appended.
        if packed.channel_mean is not None or packed.predictor_kind == "innovation":
            raise ValueError("this predictor has no incremental form")
        self.packed = packed
        self.cache_role: str | None = None
        self.capacity_units = capacity_units
        self._allocated_units = len(packed.unit_lengths)
        self._scale_dtype = (
            packed.residual_scale.dtype
            if packed.residual_scale is not None
            else _dtype(config.scale_dtype)
        )

        history_length = self._history_length
        units: list[torch.Tensor] = []
        offset = 0
        for length in packed.unit_lengths:
            units.append(reconstruction[:, :, offset : offset + length])
            offset += length
        self._history = (
            [
                unit.clone(memory_format=torch.contiguous_format)
                for unit in reversed(units[-history_length:])
            ]
            if history_length
            else []
        )

    @property
    def _history_length(self) -> int:
        if self.packed.predictor_kind == "temporal":
            assert self.packed.temporal is not None
            return int(self.packed.temporal.history_length)
        assert self.packed.cross_kv is not None
        return int(self.packed.cross_kv.history_length)

    @property
    def units(self) -> int:
        return len(self.packed.unit_lengths)

    @property
    def full(self) -> bool:
        return self.units == self.capacity_units

    @property
    def allocated_units(self) -> int:
        """Compressed capacity currently allocated for this open span."""
        return self._allocated_units

    def _ensure_capacity(self, required_units: int) -> None:
        if required_units <= self._allocated_units:
            return
        grown_units = max(required_units, max(2, self._allocated_units * 2))
        grown_units = min(self.capacity_units, grown_units)
        batch, heads, _, width = self.packed.shape
        residual_capacity = (grown_units - 1) * self.packed.stride
        packed_width = (width * self.packed.bits + 7) // 8
        scale_width = width // self.packed.group_size
        device = self.packed.anchor_quant.device
        quant_storage = torch.empty(
            batch,
            heads,
            residual_capacity,
            packed_width,
            device=device,
            dtype=torch.uint8,
        )
        scale_storage = torch.empty(
            batch,
            heads,
            residual_capacity,
            scale_width,
            device=device,
            dtype=self._scale_dtype,
        )
        zero_storage = (
            torch.empty(
                batch,
                heads,
                residual_capacity,
                scale_width,
                device=device,
                dtype=code_dtype(self.packed.bits),
            )
            if self.packed.residual_mode == "asymmetric"
            else None
        )
        used = self.packed.shape[2] - self.packed.stride
        if used:
            assert self.packed.residual_quant is not None
            assert self.packed.residual_scale is not None
            quant_storage[:, :, :used].copy_(
                self.packed.residual_quant[:, :, :used]
            )
            scale_storage[:, :, :used].copy_(
                self.packed.residual_scale[:, :, :used]
            )
            if zero_storage is not None:
                assert self.packed.residual_zero_point is not None
                zero_storage[:, :, :used].copy_(
                    self.packed.residual_zero_point[:, :, :used]
                )
        self.packed.residual_quant = quant_storage
        self.packed.residual_scale = scale_storage
        self.packed.residual_zero_point = zero_storage
        self._allocated_units = grown_units

    def history_bytes(self) -> int:
        seen: set[tuple[int, int]] = set()
        total = 0
        for tensor in self._history:
            storage = tensor.untyped_storage()
            identity = (int(storage.data_ptr()), int(storage.nbytes()))
            if identity in seen:
                continue
            seen.add(identity)
            total += identity[1]
        return total

    def append(
        self,
        value: torch.Tensor,
        *,
        source: torch.Tensor | None = None,
        reuse_input_for_reconstruction: bool = False,
    ) -> torch.Tensor:
        """Append one or more complete units and return their reconstruction."""
        if value.shape[2] % self.packed.stride:
            raise ValueError("incremental append must contain complete units")
        appended_units = value.shape[2] // self.packed.stride
        if self.units + appended_units > self.capacity_units:
            raise ValueError("incremental append exceeds the span capacity")
        self._ensure_capacity(self.units + appended_units)
        reconstruction_dtype = _dtype(self.packed.reconstruction_dtype)
        reconstruction = (
            value
            if reuse_input_for_reconstruction
            else torch.empty_like(value, dtype=reconstruction_dtype)
        )

        if self.packed.predictor_kind == "temporal":
            if source is not None:
                raise ValueError("temporal incremental encoding does not use a source")
            projection = None
        else:
            if source is None or tuple(source.shape) != tuple(value.shape):
                raise ValueError("CrossKV incremental source must match the append")
            if source.dtype != reconstruction_dtype or source.device != value.device:
                raise ValueError("CrossKV incremental source dtype/device mismatch")
            assert self.packed.cross_kv is not None
            projection = cross_projection(
                source,
                self.packed.cross_kv,
                matmul_precision=self.packed.cross_matmul_precision,
            )

        assert self.packed.residual_quant is not None
        assert self.packed.residual_scale is not None
        residual_offset = self.packed.shape[2] - self.packed.stride
        output_offset = 0
        new_lengths = list(self.packed.unit_lengths)
        for _ in range(appended_units):
            unit_length = self.packed.stride
            current = value[:, :, output_offset : output_offset + unit_length]
            if self.packed.predictor_kind == "temporal":
                assert self.packed.temporal is not None
                if self.packed.temporal.history_length == 0:
                    prediction = torch.zeros_like(current, dtype=torch.float32)
                else:
                    prediction = predict_temporal(
                        self._history,
                        self.packed.temporal,
                        legacy=self.packed.compatibility_mode == "legacy",
                    )
            else:
                assert source is not None
                assert projection is not None
                assert self.packed.cross_kv is not None
                source_unit = source[
                    :, :, output_offset : output_offset + unit_length
                ]
                projection_unit = projection[
                    :, :, output_offset : output_offset + unit_length
                ]
                prediction = predict_cross_kv(
                    source_unit,
                    self.packed.cross_kv,
                    self._history,
                    matmul_precision=self.packed.cross_matmul_precision,
                    legacy=self.packed.compatibility_mode == "legacy",
                    projection=projection_unit,
                )
            if self.packed.compatibility_mode == "legacy":
                residual_input = current.float() - prediction
            else:
                residual_input = torch.empty_like(current, dtype=torch.float32)
                torch.sub(current, prediction, out=residual_input)
            quant, scale, zero_point, residual_rebuilt = _quantize(
                residual_input,
                bits=self.packed.bits,
                group_size=self.packed.group_size,
                mode=self.packed.residual_mode,
                scale_dtype=self.packed.residual_scale.dtype,
            )
            rebuilt = (
                (prediction + residual_rebuilt).to(reconstruction_dtype)
                if self.packed.compatibility_mode == "legacy"
                else prediction.add_(residual_rebuilt).to(reconstruction_dtype)
            )
            reconstruction[
                :, :, output_offset : output_offset + unit_length
            ] = rebuilt
            residual_end = residual_offset + unit_length
            self.packed.residual_quant[
                :, :, residual_offset:residual_end
            ].copy_(quant)
            self.packed.residual_scale[
                :, :, residual_offset:residual_end
            ].copy_(scale)
            if zero_point is not None:
                assert self.packed.residual_zero_point is not None
                self.packed.residual_zero_point[
                    :, :, residual_offset:residual_end
                ].copy_(zero_point)
            if self._history_length:
                self._history = [
                    rebuilt,
                    *self._history[: self._history_length - 1],
                ]
            residual_offset = residual_end
            output_offset += unit_length
            new_lengths.append(unit_length)

        batch, heads, sequence, width = self.packed.shape
        self.packed.shape = (
            batch,
            heads,
            sequence + int(value.shape[2]),
            width,
        )
        self.packed.unit_lengths = tuple(new_lengths)
        return reconstruction


def encode_temporal(
    key: torch.Tensor,
    parameters: TemporalParameters,
    config: CodecConfig | None = None,
    *,
    reuse_input_for_reconstruction: bool = False,
    bits: int | None = None,
    history_length: int | None = None,
    channel_mean: bool = False,
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
        channel_mean=channel_mean,
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


def encode_innovation(
    value: torch.Tensor,
    reconstructed_key: torch.Tensor,
    key_parameters: TemporalParameters,
    parameters: CrossKVParameters,
    config: CodecConfig | None = None,
    *,
    reuse_input_for_reconstruction: bool = False,
) -> tuple[PackedTensor, torch.Tensor]:
    """Encode V from decoder-visible K innovation and reconstructed V history."""
    resolved = config or CodecConfig(v_predictor_mode="innovation")
    return _encode(
        value,
        predictor_kind="innovation",
        temporal=key_parameters,
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
    if packed.predictor_kind in {"cross_kv", "innovation"}:
        if source is None:
            raise ValueError("V decoding requires reconstructed K")
        if tuple(source.shape) != packed.shape:
            raise ValueError("reconstructed K does not match the packed V shape")
        if source.device != packed.anchor_quant.device:
            raise ValueError("reconstructed K and packed V must be on the same device")
        if source.dtype != stored_dtype:
            raise ValueError(
                "reconstructed K dtype must match the packed V reconstruction dtype "
                f"{stored_dtype}, got {source.dtype}"
            )
        if packed.predictor_kind == "innovation":
            if packed.temporal is None:
                raise ValueError("innovation decoding is missing K temporal parameters")
            source = reconstructed_key_innovation(
                source,
                packed.temporal,
                packed.unit_lengths,
                legacy=packed.compatibility_mode == "legacy",
            )
    # Legacy mode picks its decode route deliberately, so a silent PyTorch
    # fallback would hide both a correctness and a throughput regression.
    effective_strict_triton = (
        strict_triton or packed.compatibility_mode == "legacy"
    )
    # Every Triton route projects Cross(K) inside its own kernel, so the
    # batched projection is built only if the Torch decoder is reached.
    needs_projection = (
        packed.predictor_kind in {"cross_kv", "innovation"}
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
                if packed.channel_mean is not None:
                    accelerated.add_(packed.channel_mean.unsqueeze(2))
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
    anchor = (
        packed.anchor_quant.to(output_dtype)
        if packed.raw_anchor
        else dequantize_symmetric(
            packed.anchor_quant,
            packed.anchor_scale,
            packed.anchor_bits,
            packed.group_size,
            width,
            output_dtype,
        )
    )
    output[:, :, :anchor_length] = anchor
    if packed.residual_quant is None:
        if packed.channel_mean is not None:
            output.add_(packed.channel_mean.unsqueeze(2))
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
    if packed.channel_mean is not None:
        output.add_(packed.channel_mean.unsqueeze(2))
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
        # Experimental mean-split K path: subtract each span's per-channel
        # mean before temporal prediction and store it as side information.
        self.k_channel_mean = os.environ.get("TEMPOKV_K_MEAN_SPLIT") == "1"
        expects_temporal_v = is_temporal_v_predictor(self.config.v_predictor_mode)
        if expects_temporal_v != isinstance(cross_kv, TemporalParameters):
            expected = "TemporalParameters" if expects_temporal_v else "CrossKVParameters"
            raise ValueError(
                f"v_predictor_mode={self.config.v_predictor_mode} requires V "
                f"parameters of type {expected}, got {type(cross_kv).__name__}"
            )
        if isinstance(cross_kv, CrossKVParameters):
            expected_source = (
                "innovation"
                if self.config.v_predictor_mode == "innovation"
                else "cross_kv"
            )
            if cross_kv.projection_source != expected_source:
                raise ValueError(
                    f"v_predictor_mode={self.config.v_predictor_mode} requires V "
                    f"projection_source={expected_source}, got "
                    f"{cross_kv.projection_source}"
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
        self._cross_by_device: dict[
            str, CrossKVParameters | TemporalParameters
        ] = {}
        # Per (device, layer) caches so every packed span shares the same
        # single-layer parameter object instead of building a fresh copy.
        self._temporal_layer_cache: dict[tuple[str, int], TemporalParameters] = {}
        self._cross_layer_cache: dict[
            tuple[str, int], CrossKVParameters | TemporalParameters
        ] = {}

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
    ) -> CrossKVParameters | TemporalParameters:
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
            channel_mean=self.k_channel_mean,
        )

    @property
    def primary_role(self) -> str:
        """Cache slot encoded first, whose reconstruction feeds the other."""
        return self.config.primary_tensor

    @property
    def secondary_role(self) -> str:
        """Cache slot encoded second, from the reconstructed primary tensor."""
        return self.config.secondary_tensor

    def encode_primary(
        self,
        tensor: torch.Tensor,
        *,
        layer: int = 0,
        reuse_input: bool = False,
    ) -> tuple[PackedTensor, torch.Tensor]:
        """Encode the temporally predicted tensor and return its reconstruction.

        Callers address the cache by role rather than by name so a runtime does
        not have to know which direction it is running; ``primary_role`` says
        which slot to read.
        """
        return self.encode_key(tensor, layer=layer, reuse_input=reuse_input)

    def begin_incremental_primary(
        self,
        tensor: torch.Tensor,
        *,
        capacity_units: int,
        layer: int = 0,
        reuse_input: bool = False,
    ) -> tuple[IncrementalTensorEncoder, torch.Tensor]:
        """Encode a primary prefix and keep bounded state for later appends."""
        packed, reconstruction = self.encode_primary(
            tensor, layer=layer, reuse_input=reuse_input
        )
        encoder = IncrementalTensorEncoder(
            packed,
            reconstruction,
            self.config,
            capacity_units=capacity_units,
        )
        encoder.cache_role = self.primary_role
        return encoder, reconstruction

    def encode_secondary(
        self,
        tensor: torch.Tensor,
        reconstructed_primary: torch.Tensor | None,
        *,
        layer: int = 0,
        reuse_input: bool = False,
    ) -> PackedTensor:
        """Encode the tensor that is predicted from the reconstructed primary."""
        return self.encode_value(
            tensor, reconstructed_primary, layer=layer, reuse_input=reuse_input
        )

    def begin_incremental_secondary(
        self,
        tensor: torch.Tensor,
        reconstructed_primary: torch.Tensor | None,
        *,
        capacity_units: int,
        layer: int = 0,
        reuse_input: bool = False,
    ) -> IncrementalTensorEncoder:
        """Encode a secondary prefix and keep bounded state for later appends."""
        if reuse_input:
            packed = self.encode_secondary(
                tensor,
                reconstructed_primary,
                layer=layer,
                reuse_input=True,
            )
            reconstruction = tensor
        else:
            packed, reconstruction = self._encode_value_with_reconstruction(
                tensor,
                reconstructed_primary,
                layer=layer,
                reuse_input=False,
            )
        encoder = IncrementalTensorEncoder(
            packed,
            reconstruction,
            self.config,
            capacity_units=capacity_units,
        )
        encoder.cache_role = self.secondary_role
        return encoder

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
        return is_cross_kv_v_predictor(self.config.v_predictor_mode)

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
        packed, _ = self._encode_value_with_reconstruction(
            value,
            reconstructed_key,
            layer=layer,
            reuse_input=reuse_input,
        )
        return packed

    def _encode_value_with_reconstruction(
        self,
        value: torch.Tensor,
        reconstructed_key: torch.Tensor | None = None,
        *,
        layer: int = 0,
        reuse_input: bool = False,
    ) -> tuple[PackedTensor, torch.Tensor]:
        """Encode V and retain the exact decoder-visible reconstruction."""
        parameters = self._cross_for_layer(value.device, layer)
        if not self.predicts_v_from_key:
            assert isinstance(parameters, TemporalParameters)
            return encode_temporal(
                value,
                parameters,
                self.config,
                reuse_input_for_reconstruction=reuse_input,
                bits=self.config.v_bits,
                history_length=self.config.v_history_length,
            )
        if reconstructed_key is None:
            raise ValueError("key-conditioned V encoding requires the reconstructed K")
        assert isinstance(parameters, CrossKVParameters)
        if self.config.v_predictor_mode == "innovation":
            key_parameters = self._temporal_for_layer(value.device, layer)
            return encode_innovation(
                value,
                reconstructed_key,
                key_parameters,
                parameters,
                self.config,
                reuse_input_for_reconstruction=reuse_input,
            )
        else:
            return encode_cross_kv(
                value,
                reconstructed_key,
                parameters,
                self.config,
                reuse_input_for_reconstruction=reuse_input,
            )

    def encode(
        self, key: torch.Tensor, value: torch.Tensor, *, layer: int = 0
    ) -> PackedKV:
        if tuple(key.shape) != tuple(value.shape):
            raise ValueError("K and V must have the same shape")
        if key.device != value.device:
            raise ValueError("K and V must be on the same device")
        # Under the mirrored direction V leads and K is projected from it, so
        # address the pair by role and put the results back under their own
        # names.
        primary, secondary = (
            (value, key) if self.config.mirrored else (key, value)
        )
        primary_state, primary_rebuilt = self.encode_primary(primary, layer=layer)
        secondary_state = self.encode_secondary(
            secondary, primary_rebuilt, layer=layer
        )
        if self.config.mirrored:
            return PackedKV(secondary_state, primary_state)
        return PackedKV(primary_state, secondary_state)

    def decode(
        self, packed: PackedKV, *, output_dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a packed pair, mirroring the order ``encode`` used.

        The dependent tensor is decoded from the reconstruction of the tensor
        it was projected from, so the order follows the codec direction rather
        than the K/V names: forward decodes K then V, and the mirrored
        direction decodes V then K.
        """
        mirrored = self.config.mirrored
        packed_primary, packed_secondary = (
            (packed.value, packed.key) if mirrored else (packed.key, packed.value)
        )
        primary = decode_tensor(
            packed_primary,
            output_dtype=output_dtype,
            use_triton=self.config.use_triton,
            strict_triton=self.config.strict_triton,
        )
        secondary = decode_tensor(
            packed_secondary,
            source=primary if self.predicts_v_from_key else None,
            output_dtype=output_dtype,
            use_triton=self.config.use_triton,
            strict_triton=self.config.strict_triton,
        )
        if mirrored:
            return secondary, primary
        return primary, secondary
