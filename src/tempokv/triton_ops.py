from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from .codec import PackedTensor


_ROUTE_COUNTS: dict[str, int] = {}


def route_report() -> dict[str, int]:
    return dict(_ROUTE_COUNTS)


def reset_route_report() -> None:
    _ROUTE_COUNTS.clear()


def _record(route: str) -> None:
    _ROUTE_COUNTS[route] = _ROUTE_COUNTS.get(route, 0) + 1


@triton.jit
def _anchor_decode_kernel(
    quant_ptr,
    scale_ptr,
    output_ptr,
    sequence: tl.constexpr,
    unit_length: tl.constexpr,
    width: tl.constexpr,
    packed_width: tl.constexpr,
    scale_width: tl.constexpr,
    bits: tl.constexpr,
    group_size: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
):
    batch_head = tl.program_id(0)
    token = tl.program_id(1) * block_s + tl.arange(0, block_s)
    channel = tl.program_id(2) * block_d + tl.arange(0, block_d)
    valid = (token[:, None] < unit_length) & (channel[None, :] < width)
    if bits == 8:
        quant_offset = (
            batch_head * unit_length * width
            + token[:, None] * width
            + channel[None, :]
        )
        unsigned = tl.load(quant_ptr + quant_offset, mask=valid, other=0).to(tl.int32)
    else:
        lanes = 8 // bits
        packed_channel = channel // lanes
        lane = channel - packed_channel * lanes
        shift = bits * (lanes - lane - 1)
        quant_offset = (
            batch_head * unit_length * packed_width
            + token[:, None] * packed_width
            + packed_channel[None, :]
        )
        byte = tl.load(quant_ptr + quant_offset, mask=valid, other=0).to(tl.int32)
        unsigned = (byte >> shift[None, :]) & ((1 << bits) - 1)
    group = channel // group_size
    scale_offset = (
        batch_head * unit_length * scale_width
        + token[:, None] * scale_width
        + group[None, :]
    )
    scale = tl.load(scale_ptr + scale_offset, mask=valid, other=1.0).to(tl.float32)
    value = (unsigned - ((1 << (bits - 1)) - 1)).to(tl.float32) * scale
    output_offset = (
        batch_head * sequence * width
        + token[:, None] * width
        + channel[None, :]
    )
    tl.store(output_ptr + output_offset, value, mask=valid)


@triton.jit
def _temporal_residual_decode_kernel(
    quant_ptr,
    scale_ptr,
    zero_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    sequence: tl.constexpr,
    residual_sequence: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    packed_width: tl.constexpr,
    scale_width: tl.constexpr,
    stride: tl.constexpr,
    current_start: tl.constexpr,
    residual_start: tl.constexpr,
    unit_length: tl.constexpr,
    order: tl.constexpr,
    bits: tl.constexpr,
    group_size: tl.constexpr,
    per_head: tl.constexpr,
    asymmetric: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
):
    batch_head = tl.program_id(0)
    head = batch_head % heads
    token = tl.program_id(1) * block_s + tl.arange(0, block_s)
    channel = tl.program_id(2) * block_d + tl.arange(0, block_d)
    valid = (token[:, None] < unit_length) & (channel[None, :] < width)
    prediction = tl.zeros((block_s, block_d), tl.float32)
    for lag in tl.static_range(0, order):
        previous_token = current_start - (lag + 1) * stride + token
        previous_offset = (
            batch_head * sequence * width
            + previous_token[:, None] * width
            + channel[None, :]
        )
        previous = tl.load(output_ptr + previous_offset, mask=valid, other=0.0).to(tl.float32)
        if per_head:
            weight_offset = (lag * heads + head) * width + channel
        else:
            weight_offset = lag * width + channel
        weight = tl.load(weight_ptr + weight_offset, mask=channel < width, other=0.0)
        prediction += previous * weight[None, :]
    if per_head:
        bias_offset = head * width + channel
    else:
        bias_offset = channel
    bias = tl.load(bias_ptr + bias_offset, mask=channel < width, other=0.0)
    prediction += bias[None, :]

    residual_token = residual_start + token
    if bits == 8:
        quant_offset = (
            batch_head * residual_sequence * width
            + residual_token[:, None] * width
            + channel[None, :]
        )
        unsigned = tl.load(quant_ptr + quant_offset, mask=valid, other=0).to(tl.int32)
    else:
        lanes = 8 // bits
        packed_channel = channel // lanes
        lane = channel - packed_channel * lanes
        shift = bits * (lanes - lane - 1)
        quant_offset = (
            batch_head * residual_sequence * packed_width
            + residual_token[:, None] * packed_width
            + packed_channel[None, :]
        )
        byte = tl.load(quant_ptr + quant_offset, mask=valid, other=0).to(tl.int32)
        unsigned = (byte >> shift[None, :]) & ((1 << bits) - 1)
    group = channel // group_size
    scale_offset = (
        batch_head * residual_sequence * scale_width
        + residual_token[:, None] * scale_width
        + group[None, :]
    )
    scale = tl.load(scale_ptr + scale_offset, mask=valid, other=1.0).to(tl.float32)
    if asymmetric:
        zero = tl.load(zero_ptr + scale_offset, mask=valid, other=0).to(tl.float32)
        residual = (unsigned.to(tl.float32) - zero) * scale
    else:
        residual = (unsigned - ((1 << (bits - 1)) - 1)).to(tl.float32) * scale
    output_token = current_start + token
    output_offset = (
        batch_head * sequence * width
        + output_token[:, None] * width
        + channel[None, :]
    )
    tl.store(output_ptr + output_offset, prediction + residual, mask=valid)


@triton.jit
def _temporal_persistent_decode_kernel(
    quant_ptr,
    scale_ptr,
    zero_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    sequence: tl.constexpr,
    residual_sequence: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    packed_width: tl.constexpr,
    scale_width: tl.constexpr,
    stride: tl.constexpr,
    history_length: tl.constexpr,
    start_unit: tl.constexpr,
    chunk_units: tl.constexpr,
    bits: tl.constexpr,
    group_size: tl.constexpr,
    per_head: tl.constexpr,
    asymmetric: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
):
    """Decode several recurrent temporal units in one launch."""
    batch_head = tl.program_id(0)
    head = batch_head % heads
    token = tl.program_id(1) * block_s + tl.arange(0, block_s)
    channel = tl.program_id(2) * block_d + tl.arange(0, block_d)
    for chunk_index in tl.static_range(0, chunk_units):
        unit_index = start_unit + chunk_index
        current_token = unit_index * stride + token
        residual_token = (unit_index - 1) * stride + token
        valid = (
            (current_token[:, None] < sequence)
            & (residual_token[:, None] < residual_sequence)
            & (channel[None, :] < width)
        )
        if bits == 8:
            quant_offset = (
                batch_head * residual_sequence * width
                + residual_token[:, None] * width
                + channel[None, :]
            )
            unsigned = tl.load(quant_ptr + quant_offset, mask=valid, other=0).to(tl.int32)
        else:
            lanes = 8 // bits
            packed_channel = channel // lanes
            lane = channel - packed_channel * lanes
            shift = bits * (lanes - lane - 1)
            quant_offset = (
                batch_head * residual_sequence * packed_width
                + residual_token[:, None] * packed_width
                + packed_channel[None, :]
            )
            byte = tl.load(quant_ptr + quant_offset, mask=valid, other=0).to(tl.int32)
            unsigned = (byte >> shift[None, :]) & ((1 << bits) - 1)
        group = channel // group_size
        scale_offset = (
            batch_head * residual_sequence * scale_width
            + residual_token[:, None] * scale_width
            + group[None, :]
        )
        scale = tl.load(scale_ptr + scale_offset, mask=valid, other=1.0).to(tl.float32)
        if asymmetric:
            zero = tl.load(zero_ptr + scale_offset, mask=valid, other=0).to(tl.float32)
            residual = (unsigned.to(tl.float32) - zero) * scale
        else:
            residual = (unsigned - ((1 << (bits - 1)) - 1)).to(tl.float32) * scale

        order = unit_index
        if order > history_length:
            order = history_length
        if per_head:
            bias_offset = ((order - 1) * heads + head) * width + channel
        else:
            bias_offset = (order - 1) * width + channel
        bias = tl.load(bias_ptr + bias_offset, mask=channel < width, other=0.0)
        prediction = bias[None, :] + tl.zeros((block_s, block_d), tl.float32)
        for lag in tl.static_range(0, history_length):
            if lag < order:
                previous_token = (unit_index - lag - 1) * stride + token
                previous_offset = (
                    batch_head * sequence * width
                    + previous_token[:, None] * width
                    + channel[None, :]
                )
                previous = tl.load(
                    output_ptr + previous_offset, mask=valid, other=0.0
                ).to(tl.float32)
                if per_head:
                    weight_offset = (
                        (((order - 1) * history_length + lag) * heads + head) * width
                        + channel
                    )
                else:
                    weight_offset = (
                        ((order - 1) * history_length + lag) * width + channel
                    )
                coefficient = tl.load(
                    weight_ptr + weight_offset,
                    mask=channel < width,
                    other=0.0,
                )
                prediction += previous * coefficient[None, :]
        current = (prediction + residual).to(tl.bfloat16).to(tl.float32)
        output_offset = (
            batch_head * sequence * width
            + current_token[:, None] * width
            + channel[None, :]
        )
        tl.store(output_ptr + output_offset, current, mask=valid)


@triton.jit
def _cross_kv_residual_decode_kernel(
    quant_ptr,
    scale_ptr,
    zero_ptr,
    source_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    sequence: tl.constexpr,
    residual_sequence: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    packed_width: tl.constexpr,
    scale_width: tl.constexpr,
    current_start: tl.constexpr,
    residual_start: tl.constexpr,
    unit_length: tl.constexpr,
    bits: tl.constexpr,
    group_size: tl.constexpr,
    asymmetric: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
    block_k: tl.constexpr,
):
    batch_head = tl.program_id(0)
    head = batch_head % heads
    token = tl.program_id(1) * block_s + tl.arange(0, block_s)
    channel = tl.program_id(2) * block_d + tl.arange(0, block_d)
    input_channel = tl.arange(0, block_k)
    valid = (token[:, None] < unit_length) & (channel[None, :] < width)
    source_token = current_start + token
    source_offset = (
        batch_head * sequence * width
        + source_token[:, None] * width
        + input_channel[None, :]
    )
    source = tl.load(
        source_ptr + source_offset,
        mask=(token[:, None] < unit_length) & (input_channel[None, :] < width),
        other=0.0,
    ).to(tl.float32)
    weight_offset = (
        head * width * width
        + input_channel[:, None] * width
        + channel[None, :]
    )
    weight = tl.load(
        weight_ptr + weight_offset,
        mask=(input_channel[:, None] < width) & (channel[None, :] < width),
        other=0.0,
    ).to(tl.float32)
    prediction = tl.dot(
        source, weight, out_dtype=tl.float32, input_precision="ieee"
    )
    bias = tl.load(
        bias_ptr + head * width + channel, mask=channel < width, other=0.0
    )
    prediction += bias[None, :]

    residual_token = residual_start + token
    if bits == 8:
        quant_offset = (
            batch_head * residual_sequence * width
            + residual_token[:, None] * width
            + channel[None, :]
        )
        unsigned = tl.load(quant_ptr + quant_offset, mask=valid, other=0).to(tl.int32)
    else:
        lanes = 8 // bits
        packed_channel = channel // lanes
        lane = channel - packed_channel * lanes
        shift = bits * (lanes - lane - 1)
        quant_offset = (
            batch_head * residual_sequence * packed_width
            + residual_token[:, None] * packed_width
            + packed_channel[None, :]
        )
        byte = tl.load(quant_ptr + quant_offset, mask=valid, other=0).to(tl.int32)
        unsigned = (byte >> shift[None, :]) & ((1 << bits) - 1)
    group = channel // group_size
    scale_offset = (
        batch_head * residual_sequence * scale_width
        + residual_token[:, None] * scale_width
        + group[None, :]
    )
    scale = tl.load(scale_ptr + scale_offset, mask=valid, other=1.0).to(tl.float32)
    if asymmetric:
        zero = tl.load(zero_ptr + scale_offset, mask=valid, other=0).to(tl.float32)
        residual = (unsigned.to(tl.float32) - zero) * scale
    else:
        residual = (unsigned - ((1 << (bits - 1)) - 1)).to(tl.float32) * scale
    output_offset = (
        batch_head * sequence * width
        + source_token[:, None] * width
        + channel[None, :]
    )
    tl.store(output_ptr + output_offset, prediction + residual, mask=valid)


@triton.jit
def _cross_kv_persistent_decode_kernel(
    quant_ptr,
    scale_ptr,
    zero_ptr,
    source_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    sequence: tl.constexpr,
    residual_sequence: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    packed_width: tl.constexpr,
    scale_width: tl.constexpr,
    stride: tl.constexpr,
    start_unit: tl.constexpr,
    chunk_units: tl.constexpr,
    bits: tl.constexpr,
    group_size: tl.constexpr,
    asymmetric: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
    block_k: tl.constexpr,
):
    """Decode several independent CrossKV units in one launch."""
    batch_head = tl.program_id(0)
    head = batch_head % heads
    token = tl.program_id(1) * block_s + tl.arange(0, block_s)
    channel = tl.program_id(2) * block_d + tl.arange(0, block_d)
    input_channel = tl.arange(0, block_k)
    bias = tl.load(
        bias_ptr + head * width + channel,
        mask=channel < width,
        other=0.0,
    ).to(tl.float32)
    for chunk_index in tl.static_range(0, chunk_units):
        unit_index = start_unit + chunk_index
        current_token = unit_index * stride + token
        residual_token = (unit_index - 1) * stride + token
        valid = (
            (current_token[:, None] < sequence)
            & (residual_token[:, None] < residual_sequence)
            & (channel[None, :] < width)
        )
        if bits == 8:
            quant_offset = (
                batch_head * residual_sequence * width
                + residual_token[:, None] * width
                + channel[None, :]
            )
            unsigned = tl.load(quant_ptr + quant_offset, mask=valid, other=0).to(tl.int32)
        else:
            lanes = 8 // bits
            packed_channel = channel // lanes
            lane = channel - packed_channel * lanes
            shift = bits * (lanes - lane - 1)
            quant_offset = (
                batch_head * residual_sequence * packed_width
                + residual_token[:, None] * packed_width
                + packed_channel[None, :]
            )
            byte = tl.load(quant_ptr + quant_offset, mask=valid, other=0).to(tl.int32)
            unsigned = (byte >> shift[None, :]) & ((1 << bits) - 1)
        group = channel // group_size
        scale_offset = (
            batch_head * residual_sequence * scale_width
            + residual_token[:, None] * scale_width
            + group[None, :]
        )
        scale = tl.load(scale_ptr + scale_offset, mask=valid, other=1.0).to(tl.float32)
        if asymmetric:
            zero = tl.load(zero_ptr + scale_offset, mask=valid, other=0).to(tl.float32)
            residual = (unsigned.to(tl.float32) - zero) * scale
        else:
            residual = (unsigned - ((1 << (bits - 1)) - 1)).to(tl.float32) * scale
        source_offset = (
            batch_head * sequence * width
            + current_token[:, None] * width
            + input_channel[None, :]
        )
        source = tl.load(
            source_ptr + source_offset,
            mask=(current_token[:, None] < sequence) & (input_channel[None, :] < width),
            other=0.0,
        ).to(tl.float32)
        weight_offset = (
            head * width * width
            + input_channel[:, None] * width
            + channel[None, :]
        )
        weight = tl.load(
            weight_ptr + weight_offset,
            mask=(input_channel[:, None] < width) & (channel[None, :] < width),
            other=0.0,
        ).to(tl.float32)
        prediction = tl.dot(
            source, weight, out_dtype=tl.float32, input_precision="ieee"
        ) + bias[None, :]
        output_offset = (
            batch_head * sequence * width
            + current_token[:, None] * width
            + channel[None, :]
        )
        tl.store(output_ptr + output_offset, prediction + residual, mask=valid)


def _ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right


def _uniform_layout(packed: "PackedTensor") -> bool:
    return all(
        length == packed.stride
        for length in packed.unit_lengths[1:-1]
    ) and packed.unit_lengths[-1] <= packed.stride


def _supported(packed: "PackedTensor", source: torch.Tensor | None) -> bool:
    if not packed.anchor_quant.is_cuda:
        return False
    if packed.bits not in (2, 4, 8) or packed.anchor_bits not in (2, 4, 8):
        return False
    if packed.predictor_kind == "cross_kv":
        return source is not None and source.is_cuda and packed.cross_kv is not None
    return packed.temporal is not None


def decode(
    packed: "PackedTensor",
    *,
    source: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Fused GPU prediction and residual decode for TempoKV's two runtime routes."""
    if not _supported(packed, source):
        return None
    batch, heads, sequence, width = packed.shape
    batch_heads = batch * heads
    block_s = 16
    block_d = max(16, min(128, triton.next_power_of_2(width)))
    scale_width = width // packed.group_size
    output = torch.empty(packed.shape, device=packed.anchor_quant.device, dtype=output_dtype)
    output_flat = output.view(batch_heads, sequence, width)
    anchor_length = packed.unit_lengths[0]
    anchor_packed_width = width if packed.anchor_bits == 8 else width // (8 // packed.anchor_bits)
    anchor_grid = (
        batch_heads,
        _ceil_div(anchor_length, block_s),
        _ceil_div(width, block_d),
    )
    _anchor_decode_kernel[anchor_grid](
        packed.anchor_quant.contiguous().view(batch_heads, anchor_length, anchor_packed_width),
        packed.anchor_scale.contiguous().view(batch_heads, anchor_length, scale_width),
        output_flat,
        sequence,
        anchor_length,
        width,
        anchor_packed_width,
        scale_width,
        packed.anchor_bits,
        packed.group_size,
        block_s,
        block_d,
    )
    if packed.residual_quant is None:
        _record("anchor")
        return output
    assert packed.residual_scale is not None
    residual_sequence = int(packed.residual_quant.shape[2])
    residual_packed_width = width if packed.bits == 8 else width // (8 // packed.bits)
    quant = packed.residual_quant.contiguous().view(
        batch_heads, residual_sequence, residual_packed_width
    )
    scale = packed.residual_scale.contiguous().view(
        batch_heads, residual_sequence, scale_width
    )
    asymmetric = packed.residual_mode == "asymmetric"
    if asymmetric:
        assert packed.residual_zero_point is not None
        zero = packed.residual_zero_point.contiguous().view(
            batch_heads, residual_sequence, scale_width
        )
    else:
        zero = scale

    current_start = anchor_length
    residual_start = 0
    if packed.predictor_kind == "temporal":
        assert packed.temporal is not None
        parameters = packed.temporal.to(output.device)
        if output_dtype == torch.bfloat16 and _uniform_layout(packed):
            grid = (
                batch_heads,
                _ceil_div(packed.stride, block_s),
                _ceil_div(width, block_d),
            )
            for start_unit in range(1, len(packed.unit_lengths), 8):
                chunk_units = min(8, len(packed.unit_lengths) - start_unit)
                _temporal_persistent_decode_kernel[grid](
                    quant,
                    scale,
                    zero,
                    parameters.history_weights.contiguous(),
                    parameters.history_bias.contiguous(),
                    output_flat,
                    sequence,
                    residual_sequence,
                    heads,
                    width,
                    residual_packed_width,
                    scale_width,
                    packed.stride,
                    parameters.history_length,
                    start_unit,
                    chunk_units,
                    packed.bits,
                    packed.group_size,
                    parameters.per_head,
                    asymmetric,
                    block_s,
                    block_d,
                )
            _record("temporal_persistent")
            return output
        for unit_index, unit_length in enumerate(packed.unit_lengths[1:], start=1):
            order = min(unit_index, parameters.history_length)
            weights = parameters.history_weights[order - 1, :order].contiguous()
            bias = parameters.history_bias[order - 1].contiguous()
            grid = (
                batch_heads,
                _ceil_div(unit_length, block_s),
                _ceil_div(width, block_d),
            )
            _temporal_residual_decode_kernel[grid](
                quant,
                scale,
                zero,
                weights,
                bias,
                output_flat,
                sequence,
                residual_sequence,
                heads,
                width,
                residual_packed_width,
                scale_width,
                packed.stride,
                current_start,
                residual_start,
                unit_length,
                order,
                packed.bits,
                packed.group_size,
                parameters.per_head,
                asymmetric,
                block_s,
                block_d,
            )
            current_start += unit_length
            residual_start += unit_length
        _record("temporal")
        return output

    assert source is not None and packed.cross_kv is not None
    parameters = packed.cross_kv.to(output.device)
    if parameters.layered:
        return None
    source_flat = source.contiguous().view(batch_heads, sequence, width)
    weight = parameters.weight.contiguous()
    bias = parameters.bias.contiguous()
    block_k = max(16, triton.next_power_of_2(width))
    if (
        output_dtype == torch.bfloat16
        and source.dtype == torch.bfloat16
        and width == 128
        and _uniform_layout(packed)
    ):
        grid = (
            batch_heads,
            _ceil_div(packed.stride, block_s),
            _ceil_div(width, block_d),
        )
        for start_unit in range(1, len(packed.unit_lengths), 8):
            chunk_units = min(8, len(packed.unit_lengths) - start_unit)
            _cross_kv_persistent_decode_kernel[grid](
                quant,
                scale,
                zero,
                source_flat,
                weight,
                bias,
                output_flat,
                sequence,
                residual_sequence,
                heads,
                width,
                residual_packed_width,
                scale_width,
                packed.stride,
                start_unit,
                chunk_units,
                packed.bits,
                packed.group_size,
                asymmetric,
                block_s,
                block_d,
                block_k,
            )
        _record("cross_kv_persistent")
        return output
    for unit_length in packed.unit_lengths[1:]:
        grid = (
            batch_heads,
            _ceil_div(unit_length, block_s),
            _ceil_div(width, block_d),
        )
        _cross_kv_residual_decode_kernel[grid](
            quant,
            scale,
            zero,
            source_flat,
            weight,
            bias,
            output_flat,
            sequence,
            residual_sequence,
            heads,
            width,
            residual_packed_width,
            scale_width,
            current_start,
            residual_start,
            unit_length,
            packed.bits,
            packed.group_size,
            asymmetric,
            block_s,
            block_d,
            block_k,
        )
        current_start += unit_length
        residual_start += unit_length
    _record("cross_kv")
    return output
