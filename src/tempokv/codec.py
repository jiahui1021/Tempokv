from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal
import warnings

import torch

from . import higgs
from .config import (
    RAW_ANCHOR_BITS,
    CodecConfig,
    is_cross_kv_v_predictor,
    is_temporal_v_predictor,
)
from .parameters import CrossKVParameters, PreviousFrameParameters, TemporalParameters
from .quantization import (
    code_dtype,
    cuda_matmul_precision as _cuda_matmul_precision,
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
    residual_higgs_group: int | None = None
    anchor_higgs_group: int | None = None

    def step_sizes(self) -> torch.Tensor:
        """Per-token per-group quantizer step ``[B, H, S, D/g]`` for the Jensen correction.

        This is the stored scale, except 1-bit symmetric (step ``2*scale``); a raw
        anchor reports zero.
        """
        if self.residual_higgs_group is not None or self.anchor_higgs_group is not None:
            raise ValueError("HIGGS does not have scalar quantization step sizes")
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
        """On-device bytes of the quantized components (predictor parameters excluded)."""
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
        """Bytes of the logical prefix of an open span, excluding growth capacity."""
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


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def predict_temporal(
    history: list[torch.Tensor],
    parameters: TemporalParameters,
    *,
    legacy: bool = False,
) -> torch.Tensor:
    """Predict one unit from newest-to-oldest reconstructed K units."""
    order = min(len(history), parameters.history_length)
    weights = parameters.history_weights[order - 1, :order].to(history[0].device)
    bias = parameters.history_bias[order - 1].to(history[0].device)
    prediction = torch.zeros_like(history[0], dtype=torch.float32)
    if legacy:
        if parameters.per_head:
            # Legacy layer_head K: the same out-of-place FP32 accumulation as
            # the shared predictor, with one coefficient row per head.
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
    """Return the decoder-visible K innovation of a span (zero at the anchor)."""
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
    """Project a reconstructed-K span through Cross(K) in one GEMM.

    Encoder and decoder must both use it so their predictions match.
    """
    weight = parameters.weight.to(source.device)
    bias = parameters.bias.to(source.device)
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
    """Predict V from a precomputed ``Cross(K)+bias`` and newest-to-oldest V history."""
    resolved_history = () if history is None else tuple(history)
    order = min(len(resolved_history), parameters.history_length)
    weight = parameters.weight.to(source.device)
    bias = parameters.bias.to(source.device)
    with _cuda_matmul_precision(source.device, matmul_precision):
        if projection is not None:
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
            if not parameters.history_diagonal:
                prediction.add_(torch.matmul(previous.float(), history_weight))
            elif legacy:
                prediction = prediction + history_weight.view(
                    1, source.shape[1], 1, source.shape[-1]
                ) * previous.float()
            else:
                prediction.addcmul_(previous, history_weight[None, :, None, :])
    return prediction


def anchor_roundtrip(value: torch.Tensor, config: CodecConfig) -> torch.Tensor:
    """Replay what the encoder stores for one anchor unit.

    Out-of-codec anchor replays (fitting, refinement, analysis) go through here so
    the raw anchor is handled exactly as the codec handles it.
    """
    if config.raw_anchor:
        return value.to(_dtype(config.reconstruction_dtype))
    if config.higgs_anchor:
        return higgs.roundtrip(
            value, config.anchor_bits, config.higgs_group_size, use_triton=config.use_triton,
        )
    return roundtrip(
        value,
        bits=config.anchor_bits,
        group_size=config.group_size,
        scale_dtype=_dtype(config.scale_dtype),
        symmetric=True,
    )


def residual_roundtrip(
    value: torch.Tensor, *, bits: int, config: CodecConfig, symmetric: bool
) -> torch.Tensor:
    """Replay the residual format used by the cache on a [..., H, D] slice."""
    if config.residual_quantizer == "higgs":
        return higgs.roundtrip(value, bits, config.higgs_group_size, use_triton=config.use_triton)
    return roundtrip(
        value,
        bits=bits,
        group_size=config.group_size,
        scale_dtype=_dtype(config.scale_dtype),
        symmetric=symmetric,
    )


def _encode_higgs(
    value: torch.Tensor, bits: int, group: int, *, use_triton: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the token axis in the same place as the scalar packed tensors.
    codes, scales = higgs.encode(value.transpose(1, 2), bits, group, use_triton=use_triton)
    return codes.unsqueeze(1), scales.unsqueeze(1)


def _decode_higgs(
    codes: torch.Tensor, scales: torch.Tensor, bits: int, group: int, heads: int, dim: int,
    *, use_triton: bool,
) -> torch.Tensor:
    return higgs.decode(
        codes.squeeze(1), scales.squeeze(1), bits, group, heads, dim, use_triton=use_triton,
    ).transpose(1, 2)


def _residual_storage_layout(
    heads: int, dim: int, bits: int, group_size: int,
    higgs_group: int | None, scale_dtype: torch.dtype,
) -> tuple[int, int, int, torch.dtype]:
    if higgs_group is not None:
        codes, scales = higgs.packed_widths(heads, dim, bits, higgs_group)
        return 1, codes, scales, torch.float32
    return heads, (dim * bits + 7) // 8, dim // group_size, scale_dtype


def _quantize(
    value: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    mode: str,
    scale_dtype: torch.dtype,
    higgs_group: int | None = None,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if higgs_group is not None:
        codes, scales = _encode_higgs(value, bits, higgs_group, use_triton=use_triton)
        rebuilt = _decode_higgs(
            codes, scales, bits, higgs_group, value.shape[1], value.shape[-1], use_triton=use_triton,
        )
        return codes, scales, None, rebuilt
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
    batch, heads, sequence, width = map(int, value.shape)
    residual_higgs_group = (
        config.higgs_group_size if config.residual_quantizer == "higgs" else None
    )
    anchor_higgs_group = config.higgs_group_size if config.higgs_anchor and not config.raw_anchor else None
    stride = min(config.stride, sequence)
    scale_dtype = _dtype(config.scale_dtype)
    reconstruction_dtype = _dtype(config.reconstruction_dtype)
    span_mean: torch.Tensor | None = None
    if channel_mean:
        # The mean is stored at reconstruction dtype so encoder and decoder
        # add back bit-identical values.
        span_mean = (
            value.float().mean(dim=2, keepdim=True).to(reconstruction_dtype)
        )
        value = (value.float() - span_mean).to(reconstruction_dtype)
    if predictor_kind == "temporal":
        temporal = temporal.with_history_length(
            config.k_history_length if history_length is None else history_length
        ).to(value.device)
    else:
        cross_kv = cross_kv.with_history_length(config.v_history_length).to(value.device)
        if predictor_kind == "innovation":
            temporal = temporal.with_history_length(config.k_history_length).to(
                value.device
            )
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
        # Raw anchor: copy so the packed state owns its storage.
        anchor_quant = anchor.to(reconstruction_dtype, copy=True)
        # An empty scale keeps the field a tensor.
        anchor_scale = anchor_quant.new_empty((0,), dtype=scale_dtype)
        anchor_rebuilt = anchor_quant
    elif anchor_higgs_group is not None:
        anchor_quant, anchor_scale = _encode_higgs(
            anchor, config.anchor_bits, anchor_higgs_group, use_triton=config.use_triton,
        )
        anchor_rebuilt = _decode_higgs(
            anchor_quant, anchor_scale, config.anchor_bits, anchor_higgs_group, heads, width,
            use_triton=config.use_triton,
        ).to(reconstruction_dtype)
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
    # Legacy keeps standalone QDQ results as history; native keeps views.
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
    packed_heads, packed_width, scale_width, residual_scale_dtype = _residual_storage_layout(
        heads, width, bits, config.group_size, residual_higgs_group, scale_dtype
    )
    residual_quant = (
        torch.empty(
            batch,
            packed_heads,
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
            packed_heads,
            residual_sequence,
            scale_width,
            device=value.device,
            dtype=residual_scale_dtype,
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
            # A zero point is a code, so it takes the code dtype.
            dtype=code_dtype(bits),
        )
        if (
            residual_sequence
            and residual_higgs_group is None
            and config.residual_mode == "asymmetric"
        )
        else None
    )
    residual_offset = 0
    unit_lengths = [stride]

    offset = stride
    while offset < sequence:
        unit_length = min(stride, sequence - offset)
        current = value[:, :, offset : offset + unit_length]
        if predictor_kind == "temporal":
            if isinstance(temporal, PreviousFrameParameters):
                prediction = history[0][:, :, :unit_length].to(dtype=torch.float32, copy=True)
            elif temporal.history_length == 0:
                # Zero-order K: direct quantization within the temporal state layout.
                prediction = torch.zeros_like(current, dtype=torch.float32)
            else:
                prediction = predict_temporal(
                    [item[:, :, :unit_length] for item in history],
                    temporal,
                    legacy=config.fit_mode == "legacy",
                )
        else:
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
            higgs_group=residual_higgs_group,
            use_triton=config.use_triton,
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
            if cross_kv.history_length:
                history = [
                    rebuilt_history,
                    *history[: cross_kv.history_length - 1],
                ]
        residual_end = residual_offset + unit_length
        residual_quant[:, :, residual_offset:residual_end].copy_(quant)
        residual_scale[:, :, residual_offset:residual_end].copy_(scale)
        if zero_point is not None:
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
        residual_higgs_group=residual_higgs_group,
        anchor_higgs_group=anchor_higgs_group,
    )
    return packed, reconstruction


class IncrementalTensorEncoder:
    """Append finalized units to an open packed span.

    The arithmetic matches a batch encode of the whole span; only the predictor
    history is kept in BF16 until the span closes.
    """

    def __init__(
        self,
        packed: PackedTensor,
        reconstruction: torch.Tensor,
        config: CodecConfig,
        *,
        capacity_units: int,
    ) -> None:
        # Span-mean K and innovation V have no incremental form; the streaming
        # encoder keeps them on the batch schedule.
        self.packed = packed
        self.cache_role: str | None = None
        self.capacity_units = capacity_units
        self._use_triton = config.use_triton
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
            return int(self.packed.temporal.history_length)
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
        packed_heads, packed_width, scale_width, scale_dtype = _residual_storage_layout(
            heads, width, self.packed.bits, self.packed.group_size,
            self.packed.residual_higgs_group, self._scale_dtype,
        )
        device = self.packed.anchor_quant.device
        quant_storage = torch.empty(
            batch,
            packed_heads,
            residual_capacity,
            packed_width,
            device=device,
            dtype=torch.uint8,
        )
        scale_storage = torch.empty(
            batch,
            packed_heads,
            residual_capacity,
            scale_width,
            device=device,
            dtype=scale_dtype,
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
            if (
                self.packed.residual_higgs_group is None
                and self.packed.residual_mode == "asymmetric"
            )
            else None
        )
        used = self.packed.shape[2] - self.packed.stride
        if used:
            quant_storage[:, :, :used].copy_(
                self.packed.residual_quant[:, :, :used]
            )
            scale_storage[:, :, :used].copy_(
                self.packed.residual_scale[:, :, :used]
            )
            if zero_storage is not None:
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
        appended_units = value.shape[2] // self.packed.stride
        self._ensure_capacity(self.units + appended_units)
        reconstruction_dtype = _dtype(self.packed.reconstruction_dtype)
        reconstruction = (
            value
            if reuse_input_for_reconstruction
            else torch.empty_like(value, dtype=reconstruction_dtype)
        )

        if self.packed.predictor_kind == "temporal":
            projection = None
        else:
            projection = cross_projection(
                source,
                self.packed.cross_kv,
                matmul_precision=self.packed.cross_matmul_precision,
            )

        residual_offset = self.packed.shape[2] - self.packed.stride
        output_offset = 0
        new_lengths = list(self.packed.unit_lengths)
        for _ in range(appended_units):
            unit_length = self.packed.stride
            current = value[:, :, output_offset : output_offset + unit_length]
            if self.packed.predictor_kind == "temporal":
                if isinstance(self.packed.temporal, PreviousFrameParameters):
                    prediction = self._history[0].to(dtype=torch.float32, copy=True)
                elif self.packed.temporal.history_length == 0:
                    prediction = torch.zeros_like(current, dtype=torch.float32)
                else:
                    prediction = predict_temporal(
                        self._history,
                        self.packed.temporal,
                        legacy=self.packed.compatibility_mode == "legacy",
                    )
            else:
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
                higgs_group=self.packed.residual_higgs_group,
                use_triton=self._use_triton,
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
    """Encode a temporally predicted tensor and return its reconstruction."""
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
    """Decode a packed cache, using Triton when available.

    ``output_tensor`` may be a strided BHSD/BSHD view; the result is always BHSD.
    """
    # Decoding replays the closed loop, so it always produces the dtype the
    # encoder reconstructed in.
    output_dtype = _dtype(packed.reconstruction_dtype)
    if packed.predictor_kind == "innovation":
        source = reconstructed_key_innovation(
            source,
            packed.temporal,
            packed.unit_lengths,
            legacy=packed.compatibility_mode == "legacy",
        )
    # Legacy mode picks its decode route deliberately, so a silent PyTorch
    # fallback would hide both a correctness and a throughput regression.
    uses_higgs = packed.residual_higgs_group is not None or packed.anchor_higgs_group is not None
    effective_strict_triton = strict_triton or (
        packed.compatibility_mode == "legacy" and not uses_higgs
    )
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
                if uses_higgs:
                    higgs.record_cache_read()
                return accelerated
            if effective_strict_triton:
                raise RuntimeError("this packed layout is not supported by Triton")
        except Exception as error:
            if effective_strict_triton:
                raise
            use_triton = False
            warnings.warn(
                f"Triton decode failed; using the PyTorch decoder: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    _, heads, sequence, width = packed.shape
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
    if packed.raw_anchor:
        anchor = packed.anchor_quant.to(output_dtype)
    elif packed.anchor_higgs_group is not None:
        anchor = _decode_higgs(
            packed.anchor_quant, packed.anchor_scale, packed.anchor_bits,
            packed.anchor_higgs_group, heads, width, use_triton=use_triton,
        ).to(output_dtype)
    else:
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
        if packed.channel_mean is not None:
            output.add_(packed.channel_mean.unsqueeze(2))
        if packed.anchor_higgs_group is not None:
            higgs.record_cache_read()
        return output
    history = (
        [anchor]
        if packed.compatibility_mode == "legacy"
        else [output[:, :, :anchor_length]]
    )
    del anchor
    # Every Triton route projects Cross(K) inside its own kernel, so the
    # batched projection is built only when the Torch decoder is reached.
    projection = (
        cross_projection(
            source, packed.cross_kv, matmul_precision=packed.cross_matmul_precision
        )
        if packed.predictor_kind != "temporal"
        else None
    )
    offset = anchor_length
    residual_offset = 0
    residual_group = packed.residual_higgs_group
    kernels = (
        higgs.for_device(output.device, residual_group)
        if use_triton and residual_group is not None else None
    )
    for unit_length in packed.unit_lengths[1:]:
        if packed.predictor_kind == "temporal":
            if isinstance(packed.temporal, PreviousFrameParameters):
                prediction = history[0][:, :, :unit_length].to(dtype=torch.float32, copy=True)
            elif packed.temporal.history_length == 0:
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
        codes = packed.residual_quant[:, :, residual_offset:end]
        scales = packed.residual_scale[:, :, residual_offset:end]
        target = output[:, :, offset : offset + unit_length]
        if kernels is not None:
            higgs.decode_into(
                codes.squeeze(1), scales.squeeze(1), packed.bits, residual_group,
                target.transpose(1, 2), kernels=kernels, predictor=prediction.transpose(1, 2),
            )
        elif residual_group is not None:
            residual = _decode_higgs(
                codes, scales, packed.bits, residual_group, heads, width, use_triton=False,
            )
        else:
            residual = _dequantize(
                codes, scales, zero_point, bits=packed.bits,
                group_size=packed.group_size, width=width, mode=packed.residual_mode,
            )
        rebuilt_history = target
        if kernels is None:
            rebuilt = (
                (prediction + residual).to(output_dtype)
                if packed.compatibility_mode == "legacy"
                else prediction.add_(residual).to(output_dtype)
            )
            target.copy_(rebuilt)
            if packed.compatibility_mode == "legacy":
                rebuilt_history = rebuilt
            del residual, rebuilt
        if packed.predictor_kind == "temporal":
            if packed.temporal is not None and packed.temporal.history_length:
                history = [
                    rebuilt_history,
                    *history[: packed.temporal.history_length - 1],
                ]
        else:
            if packed.cross_kv.history_length:
                history = [
                    rebuilt_history,
                    *history[: packed.cross_kv.history_length - 1],
                ]
        offset += unit_length
        residual_offset = end
        del prediction, zero_point
    if packed.channel_mean is not None:
        output.add_(packed.channel_mean.unsqueeze(2))
    if uses_higgs:
        higgs.record_cache_read()
    return output


def check_parameters_match_config(
    temporal: TemporalParameters,
    cross_kv: CrossKVParameters | TemporalParameters,
    config: CodecConfig,
) -> None:
    """Refuse stamped parameters fitted for a different codec configuration."""
    expected = config.format_signature()
    for name, parameters in (("K", temporal), ("V", cross_kv)):
        signature = parameters.codec_signature
        if signature is not None and signature != expected:
            raise ValueError(
                f"{name} parameters were fitted for codec {signature!r}, but this "
                f"run is configured as {expected!r}; refit them or point at the "
                "parameter directory that matches this config"
            )
    # One stamped and one unstamped file means a fresh fit was paired with a
    # leftover from an older run; two unstamped files compare equal.
    if temporal.parameter_set_id != cross_kv.parameter_set_id:
        raise ValueError(
            "K and V parameters come from different fitting runs "
            f"({temporal.parameter_set_id!r} vs {cross_kv.parameter_set_id!r})"
        )
    if (
        temporal.coordinate_system is not None
        and cross_kv.coordinate_system is not None
        and temporal.coordinate_system != cross_kv.coordinate_system
    ):
        raise ValueError(
            "K and V parameters use different coordinate systems: "
            f"{temporal.coordinate_system!r} vs {cross_kv.coordinate_system!r}"
        )
    if isinstance(cross_kv, CrossKVParameters):
        expected_source = (
            "innovation" if config.v_predictor_mode == "innovation" else "cross_kv"
        )
        if cross_kv.projection_source != expected_source:
            raise ValueError(
                f"v_predictor_mode={config.v_predictor_mode} requires V "
                f"projection_source={expected_source}, got "
                f"{cross_kv.projection_source}"
            )


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
            raise ValueError(
                f"v_predictor_mode={self.config.v_predictor_mode} does not match "
                f"V parameters of type {type(cross_kv).__name__}"
            )
        check_parameters_match_config(temporal, cross_kv, self.config)
        self.temporal = temporal.with_history_length(self.config.k_history_length)
        self.v_parameters = cross_kv.with_history_length(self.config.v_history_length)
        # One single-layer parameter object per (device, layer), shared by
        # every packed span instead of rebuilt per encode.
        self._layer_cache: dict[tuple[str, str, int], Any] = {}

    def _for_layer(self, role: str, device: torch.device, layer: int) -> Any:
        key = (role, str(device), layer)
        cached = self._layer_cache.get(key)
        if cached is None:
            parameters = (self.temporal if role == "k" else self.v_parameters).to(device)
            cached = parameters.for_layer(layer) if parameters.layered else parameters
            self._layer_cache[key] = cached
        return cached

    def _temporal_for_layer(self, device: torch.device, layer: int) -> TemporalParameters:
        return self._for_layer("k", device, layer)

    def _cross_for_layer(
        self, device: torch.device, layer: int
    ) -> CrossKVParameters | TemporalParameters:
        return self._for_layer("v", device, layer)

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
        """Encode the temporally predicted tensor and return its reconstruction."""
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
        """Encode one V cache; ``reconstructed_key`` is needed only for Cross(K) V."""
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
            return encode_temporal(
                value,
                parameters,
                self.config,
                reuse_input_for_reconstruction=reuse_input,
                bits=self.config.v_bits,
                history_length=self.config.v_history_length,
            )
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
        # Address the pair by role so the mirrored direction works too.
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
        """Decode a packed pair in encode order (K then V, or V then K when mirrored)."""
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
