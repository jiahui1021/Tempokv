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


def _route_name(route: str, direct_output: bool) -> str:
    return f"{route}_direct" if direct_output else route


@triton.jit
def _strided_bhsd_offsets(
    batch_head,
    token,
    channel,
    heads: tl.constexpr,
    stride_b,
    stride_h,
    stride_s,
    stride_d,
):
    batch = batch_head // heads
    head = batch_head - batch * heads
    return (
        batch.to(tl.int64) * stride_b
        + head.to(tl.int64) * stride_h
        + token[:, None].to(tl.int64) * stride_s
        + channel[None, :].to(tl.int64) * stride_d
    )


@triton.jit
def _anchor_decode_kernel(
    quant_ptr,
    scale_ptr,
    output_ptr,
    output_stride_b,
    output_stride_h,
    output_stride_s,
    output_stride_d,
    sequence: tl.constexpr,
    unit_length: tl.constexpr,
    heads: tl.constexpr,
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
    output_offset = _strided_bhsd_offsets(
        batch_head,
        token,
        channel,
        heads,
        output_stride_b,
        output_stride_h,
        output_stride_s,
        output_stride_d,
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
    output_stride_b,
    output_stride_h,
    output_stride_s,
    output_stride_d,
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
        previous_offset = _strided_bhsd_offsets(
            batch_head,
            previous_token,
            channel,
            heads,
            output_stride_b,
            output_stride_h,
            output_stride_s,
            output_stride_d,
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
    output_offset = _strided_bhsd_offsets(
        batch_head,
        output_token,
        channel,
        heads,
        output_stride_b,
        output_stride_h,
        output_stride_s,
        output_stride_d,
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
    output_stride_b,
    output_stride_h,
    output_stride_s,
    output_stride_d,
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
            (token[:, None] < stride)
            & (current_token[:, None] < sequence)
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
                previous_offset = _strided_bhsd_offsets(
                    batch_head,
                    previous_token,
                    channel,
                    heads,
                    output_stride_b,
                    output_stride_h,
                    output_stride_s,
                    output_stride_d,
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
        output_offset = _strided_bhsd_offsets(
            batch_head,
            current_token,
            channel,
            heads,
            output_stride_b,
            output_stride_h,
            output_stride_s,
            output_stride_d,
        )
        tl.store(output_ptr + output_offset, current, mask=valid)


@triton.jit
def _cross_kv_residual_decode_kernel(
    quant_ptr,
    scale_ptr,
    zero_ptr,
    source_ptr,
    weight_ptr,
    history_weight_ptr,
    bias_ptr,
    output_ptr,
    source_stride_b,
    source_stride_h,
    source_stride_s,
    source_stride_d,
    output_stride_b,
    output_stride_h,
    output_stride_s,
    output_stride_d,
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
    history_length: tl.constexpr,
    history_order: tl.constexpr,
    history_diagonal: tl.constexpr,
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
    source_offset = _strided_bhsd_offsets(
        batch_head,
        source_token,
        input_channel,
        heads,
        source_stride_b,
        source_stride_h,
        source_stride_s,
        source_stride_d,
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
        source, weight, out_dtype=tl.float32, input_precision="tf32"
    )
    for lag in tl.static_range(0, history_length):
        if lag < history_order:
            previous_token = current_start - (lag + 1) * stride + token
            if history_diagonal:
                previous_offset = _strided_bhsd_offsets(
                    batch_head,
                    previous_token,
                    channel,
                    heads,
                    output_stride_b,
                    output_stride_h,
                    output_stride_s,
                    output_stride_d,
                )
                previous = tl.load(
                    output_ptr + previous_offset,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                history_weight_offset = (
                    (lag * heads + head) * width + channel
                )
                history_weight = tl.load(
                    history_weight_ptr + history_weight_offset,
                    mask=channel < width,
                    other=0.0,
                ).to(tl.float32)
                prediction += previous * history_weight[None, :]
            else:
                previous_offset = _strided_bhsd_offsets(
                    batch_head,
                    previous_token,
                    input_channel,
                    heads,
                    output_stride_b,
                    output_stride_h,
                    output_stride_s,
                    output_stride_d,
                )
                previous = tl.load(
                    output_ptr + previous_offset,
                    mask=(token[:, None] < unit_length)
                    & (input_channel[None, :] < width),
                    other=0.0,
                ).to(tl.float32)
                history_weight_offset = (
                    (lag * heads + head) * width * width
                    + input_channel[:, None] * width
                    + channel[None, :]
                )
                history_weight = tl.load(
                    history_weight_ptr + history_weight_offset,
                    mask=(input_channel[:, None] < width)
                    & (channel[None, :] < width),
                    other=0.0,
                ).to(tl.float32)
                prediction += tl.dot(
                    previous,
                    history_weight,
                    out_dtype=tl.float32,
                    input_precision="tf32",
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
    output_offset = _strided_bhsd_offsets(
        batch_head,
        source_token,
        channel,
        heads,
        output_stride_b,
        output_stride_h,
        output_stride_s,
        output_stride_d,
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
    source_stride_b,
    source_stride_h,
    source_stride_s,
    source_stride_d,
    output_stride_b,
    output_stride_h,
    output_stride_s,
    output_stride_d,
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
            (token[:, None] < stride)
            & (current_token[:, None] < sequence)
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
        source_offset = _strided_bhsd_offsets(
            batch_head,
            current_token,
            input_channel,
            heads,
            source_stride_b,
            source_stride_h,
            source_stride_s,
            source_stride_d,
        )
        source = tl.load(
            source_ptr + source_offset,
            mask=(token[:, None] < stride)
            & (current_token[:, None] < sequence)
            & (input_channel[None, :] < width),
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
            source, weight, out_dtype=tl.float32, input_precision="tf32"
        ) + bias[None, :]
        output_offset = _strided_bhsd_offsets(
            batch_head,
            current_token,
            channel,
            heads,
            output_stride_b,
            output_stride_h,
            output_stride_s,
            output_stride_d,
        )
        tl.store(output_ptr + output_offset, prediction + residual, mask=valid)


@triton.jit
def _cross_kv_history_persistent_decode_kernel(
    quant_ptr,
    scale_ptr,
    zero_ptr,
    source_ptr,
    weight_ptr,
    history_weight_ptr,
    bias_ptr,
    output_ptr,
    source_stride_b,
    source_stride_h,
    source_stride_s,
    source_stride_d,
    output_stride_b,
    output_stride_h,
    output_stride_s,
    output_stride_d,
    sequence: tl.constexpr,
    residual_sequence: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    packed_width: tl.constexpr,
    scale_width: tl.constexpr,
    stride: tl.constexpr,
    history_length: tl.constexpr,
    history_diagonal: tl.constexpr,
    start_unit: tl.constexpr,
    chunk_units: tl.constexpr,
    bits: tl.constexpr,
    group_size: tl.constexpr,
    asymmetric: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
    block_k: tl.constexpr,
):
    """Decode recurrent joint K/V-history units without cross-program races."""
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

    previous_token = (start_unit - 1) * stride + token
    if history_diagonal:
        history_channel = channel
    else:
        history_channel = input_channel
    previous_offset = _strided_bhsd_offsets(
        batch_head,
        previous_token,
        history_channel,
        heads,
        output_stride_b,
        output_stride_h,
        output_stride_s,
        output_stride_d,
    )
    previous_one = tl.load(
        output_ptr + previous_offset,
        mask=(token[:, None] < stride)
        & (previous_token[:, None] < sequence)
        & (history_channel[None, :] < width),
        other=0.0,
    ).to(tl.float32)
    if history_length > 1:
        if start_unit > 1:
            previous_two_token = (start_unit - 2) * stride + token
            previous_two_offset = _strided_bhsd_offsets(
                batch_head,
                previous_two_token,
                history_channel,
                heads,
                output_stride_b,
                output_stride_h,
                output_stride_s,
                output_stride_d,
            )
            previous_two = tl.load(
                output_ptr + previous_two_offset,
                mask=(token[:, None] < stride)
                & (previous_two_token[:, None] < sequence)
                & (history_channel[None, :] < width),
                other=0.0,
            ).to(tl.float32)
        else:
            previous_two = tl.zeros((block_s, block_d), tl.float32)

    for chunk_index in tl.static_range(0, chunk_units):
        unit_index = start_unit + chunk_index
        current_token = unit_index * stride + token
        residual_token = (unit_index - 1) * stride + token
        valid = (
            (token[:, None] < stride)
            & (current_token[:, None] < sequence)
            & (residual_token[:, None] < residual_sequence)
            & (channel[None, :] < width)
        )
        if bits == 8:
            quant_offset = (
                batch_head * residual_sequence * width
                + residual_token[:, None] * width
                + channel[None, :]
            )
            unsigned = tl.load(quant_ptr + quant_offset, mask=valid, other=0).to(
                tl.int32
            )
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
            byte = tl.load(quant_ptr + quant_offset, mask=valid, other=0).to(
                tl.int32
            )
            unsigned = (byte >> shift[None, :]) & ((1 << bits) - 1)
        group = channel // group_size
        scale_offset = (
            batch_head * residual_sequence * scale_width
            + residual_token[:, None] * scale_width
            + group[None, :]
        )
        scale = tl.load(scale_ptr + scale_offset, mask=valid, other=1.0).to(
            tl.float32
        )
        if asymmetric:
            zero = tl.load(zero_ptr + scale_offset, mask=valid, other=0).to(
                tl.float32
            )
            residual = (unsigned.to(tl.float32) - zero) * scale
        else:
            residual = (
                unsigned - ((1 << (bits - 1)) - 1)
            ).to(tl.float32) * scale

        source_offset = _strided_bhsd_offsets(
            batch_head,
            current_token,
            input_channel,
            heads,
            source_stride_b,
            source_stride_h,
            source_stride_s,
            source_stride_d,
        )
        source = tl.load(
            source_ptr + source_offset,
            mask=(token[:, None] < stride)
            & (current_token[:, None] < sequence)
            & (input_channel[None, :] < width),
            other=0.0,
        ).to(tl.float32)
        weight_offset = (
            head * width * width
            + input_channel[:, None] * width
            + channel[None, :]
        )
        weight = tl.load(
            weight_ptr + weight_offset,
            mask=(input_channel[:, None] < width)
            & (channel[None, :] < width),
            other=0.0,
        ).to(tl.float32)
        prediction = tl.dot(
            source, weight, out_dtype=tl.float32, input_precision="tf32"
        )

        if history_diagonal:
            history_weight_one_offset = head * width + channel
            history_weight_one = tl.load(
                history_weight_ptr + history_weight_one_offset,
                mask=channel < width,
                other=0.0,
            ).to(tl.float32)
            prediction += previous_one * history_weight_one[None, :]
        else:
            history_weight_one_offset = (
                head * width * width
                + input_channel[:, None] * width
                + channel[None, :]
            )
            history_weight_one = tl.load(
                history_weight_ptr + history_weight_one_offset,
                mask=(input_channel[:, None] < width)
                & (channel[None, :] < width),
                other=0.0,
            ).to(tl.float32)
            prediction += tl.dot(
                previous_one,
                history_weight_one,
                out_dtype=tl.float32,
                input_precision="tf32",
            )
        if history_length > 1:
            if unit_index > 1:
                if history_diagonal:
                    history_weight_two_offset = (heads + head) * width + channel
                    history_weight_two = tl.load(
                        history_weight_ptr + history_weight_two_offset,
                        mask=channel < width,
                        other=0.0,
                    ).to(tl.float32)
                    prediction += previous_two * history_weight_two[None, :]
                else:
                    history_weight_two_offset = (
                        (heads + head) * width * width
                        + input_channel[:, None] * width
                        + channel[None, :]
                    )
                    history_weight_two = tl.load(
                        history_weight_ptr + history_weight_two_offset,
                        mask=(input_channel[:, None] < width)
                        & (channel[None, :] < width),
                        other=0.0,
                    ).to(tl.float32)
                    prediction += tl.dot(
                        previous_two,
                        history_weight_two,
                        out_dtype=tl.float32,
                        input_precision="tf32",
                    )
        current = (prediction + bias[None, :] + residual).to(tl.bfloat16).to(
            tl.float32
        )
        output_offset = _strided_bhsd_offsets(
            batch_head,
            current_token,
            channel,
            heads,
            output_stride_b,
            output_stride_h,
            output_stride_s,
            output_stride_d,
        )
        tl.store(output_ptr + output_offset, current, mask=valid)
        if history_length > 1:
            previous_two = previous_one
        previous_one = current


def _ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right


def _uniform_layout(packed: "PackedTensor") -> bool:
    return all(
        length == packed.stride
        for length in packed.unit_lengths[:-1]
    ) and packed.unit_lengths[-1] <= packed.stride


def _supported(packed: "PackedTensor", source: torch.Tensor | None) -> bool:
    if not packed.anchor_quant.is_cuda:
        return False
    if packed.bits not in (2, 4, 8) or packed.anchor_bits not in (2, 4, 8):
        return False
    if packed.predictor_kind == "cross_kv":
        return source is not None and source.is_cuda and packed.cross_kv is not None
    return packed.temporal is not None


def _prepare_output(
    packed: "PackedTensor",
    output_dtype: torch.dtype,
    output_tensor: torch.Tensor | None,
) -> tuple[torch.Tensor, bool] | None:
    batch, heads, sequence, width = packed.shape
    device = packed.anchor_quant.device
    if output_tensor is None:
        return (
            torch.empty(packed.shape, device=device, dtype=output_dtype),
            False,
        )
    if (
        not output_tensor.is_cuda
        or output_tensor.device != device
        or output_tensor.dtype != output_dtype
        or tuple(output_tensor.shape) != (batch, sequence, heads, width)
        or not output_tensor.is_contiguous()
    ):
        return None
    return output_tensor, True


def _logical_bhsd_output(
    output: torch.Tensor,
    output_bshd: bool,
) -> torch.Tensor:
    return output.permute(0, 2, 1, 3) if output_bshd else output


def decode(
    packed: "PackedTensor",
    *,
    source: torch.Tensor | None,
    output_dtype: torch.dtype,
    output_tensor: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Fused GPU prediction and residual decode for TempoKV's two runtime routes."""
    if not _supported(packed, source):
        return None
    batch, heads, sequence, width = packed.shape
    batch_heads = batch * heads
    block_s = 16
    block_d = max(16, min(128, triton.next_power_of_2(width)))
    scale_width = width // packed.group_size
    prepared_output = _prepare_output(packed, output_dtype, output_tensor)
    if prepared_output is None:
        return None
    output, output_bshd = prepared_output
    if output_bshd:
        output_stride_b, output_stride_s, output_stride_h, output_stride_d = (
            output.stride()
        )
    else:
        output_stride_b, output_stride_h, output_stride_s, output_stride_d = (
            output.stride()
        )
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
        output,
        output_stride_b,
        output_stride_h,
        output_stride_s,
        output_stride_d,
        sequence,
        anchor_length,
        heads,
        width,
        anchor_packed_width,
        scale_width,
        packed.anchor_bits,
        packed.group_size,
        block_s,
        block_d,
    )
    if packed.residual_quant is None:
        _record(_route_name("anchor", output_bshd))
        return _logical_bhsd_output(output, output_bshd)
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
        if (
            output_dtype == torch.bfloat16
            and parameters.history_length <= 8
            and _uniform_layout(packed)
        ):
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
                    output,
                    output_stride_b,
                    output_stride_h,
                    output_stride_s,
                    output_stride_d,
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
            _record(_route_name("temporal_persistent", output_bshd))
            return _logical_bhsd_output(output, output_bshd)
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
                output,
                output_stride_b,
                output_stride_h,
                output_stride_s,
                output_stride_d,
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
        _record(_route_name("temporal", output_bshd))
        return _logical_bhsd_output(output, output_bshd)

    assert source is not None and packed.cross_kv is not None
    parameters = packed.cross_kv.to(output.device)
    if parameters.layered:
        return None
    source_stride_b, source_stride_h, source_stride_s, source_stride_d = (
        source.stride()
    )
    weight = parameters.weight.contiguous()
    history_weight = (
        parameters.history_weights.contiguous()
        if parameters.history_length
        else weight
    )
    bias = parameters.bias.contiguous()
    block_k = max(16, triton.next_power_of_2(width))
    cross_block_d = 32 if width >= 128 else block_d
    if (
        output_dtype == torch.bfloat16
        and source.dtype == torch.bfloat16
        and width == 128
        and _uniform_layout(packed)
    ):
        grid = (
            batch_heads,
            _ceil_div(packed.stride, block_s),
            _ceil_div(width, cross_block_d),
        )
        for start_unit in range(1, len(packed.unit_lengths), 8):
            chunk_units = min(8, len(packed.unit_lengths) - start_unit)
            if parameters.history_length:
                # Diagonal history only depends on the same output channel, so
                # it keeps the legacy 32-channel tiling and one dense K dot.
                # Dense legacy history still needs one program to own all D
                # channels because the recurrent matmul mixes channels.
                recurrent_block_d = (
                    cross_block_d if parameters.history_diagonal else block_k
                )
                recurrent_grid = (
                    batch_heads,
                    _ceil_div(packed.stride, block_s),
                    _ceil_div(width, recurrent_block_d),
                )
                _cross_kv_history_persistent_decode_kernel[recurrent_grid](
                    quant,
                    scale,
                    zero,
                    source,
                    weight,
                    history_weight,
                    bias,
                    output,
                    source_stride_b,
                    source_stride_h,
                    source_stride_s,
                    source_stride_d,
                    output_stride_b,
                    output_stride_h,
                    output_stride_s,
                    output_stride_d,
                    sequence,
                    residual_sequence,
                    heads,
                    width,
                    residual_packed_width,
                    scale_width,
                    packed.stride,
                    parameters.history_length,
                    parameters.history_diagonal,
                    start_unit,
                    chunk_units,
                    packed.bits,
                    packed.group_size,
                    asymmetric,
                    block_s,
                    recurrent_block_d,
                    block_k,
                )
            else:
                _cross_kv_persistent_decode_kernel[grid](
                    quant,
                    scale,
                    zero,
                    source,
                    weight,
                    bias,
                    output,
                    source_stride_b,
                    source_stride_h,
                    source_stride_s,
                    source_stride_d,
                    output_stride_b,
                    output_stride_h,
                    output_stride_s,
                    output_stride_d,
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
                    cross_block_d,
                    block_k,
                )
        _record(_route_name("cross_kv_persistent", output_bshd))
        return _logical_bhsd_output(output, output_bshd)
    for unit_index, unit_length in enumerate(packed.unit_lengths[1:], start=1):
        history_order = min(unit_index, parameters.history_length)
        grid = (
            batch_heads,
            _ceil_div(unit_length, block_s),
            _ceil_div(width, cross_block_d),
        )
        _cross_kv_residual_decode_kernel[grid](
            quant,
            scale,
            zero,
            source,
            weight,
            history_weight,
            bias,
            output,
            source_stride_b,
            source_stride_h,
            source_stride_s,
            source_stride_d,
            output_stride_b,
            output_stride_h,
            output_stride_s,
            output_stride_d,
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
            parameters.history_length,
            history_order,
            parameters.history_diagonal,
            packed.bits,
            packed.group_size,
            asymmetric,
            block_s,
            cross_block_d,
            block_k,
        )
        current_start += unit_length
        residual_start += unit_length
    _record(_route_name("cross_kv", output_bshd))
    return _logical_bhsd_output(output, output_bshd)
